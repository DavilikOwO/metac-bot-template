"""
bot_pro.py - Bot de pronósticos para el torneo de bots de Metaculus (FutureEval) y MiniBench.

Se apoya en la plantilla oficial (main.py / FallTemplateBot2026) y añade lo que, según los
análisis publicados de temporadas anteriores, separa a los bots de cabeza del resto:

  1. Conjunto de modelos punteros de varias casas (OpenAI, Anthropic, Google, xAI...).
     Cada pregunta la responde cada modelo por separado y se combina con una mediana
     ponderada en escala log-odds.
  2. Investigación de varias fuentes en paralelo (AskNews + modelos con búsqueda web).
  3. Prompt con tasa base explícita, lectura estricta de los criterios de resolución
     y comprobación de si la pregunta ya está prácticamente resuelta.
  4. Calibración: recalibración tipo Platt (aprendida de nuestros propios resultados con
     learn.py) y recorte de extremos.
  5. Registro de cada predicción individual (data/forecast_log.jsonl) para que learn.py
     aprenda qué modelos aciertan más y cómo recalibrar.

Todo es automático: ninguna persona mira ni toca los pronósticos antes de enviarlos
(norma del torneo).

Uso:
  python bot_pro.py --mode tournament       # torneo grande + MiniBench
  python bot_pro.py --mode test_questions   # prueba en el área de pruebas de Metaculus
  python bot_pro.py --mode tournament --dry-run   # sin publicar
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import dotenv
import numpy as np
import requests

from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    AskNewsSearcher,
    BinaryQuestion,
    DateQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
)

import main as template_main  # noqa: E402
from forecasting_tools import BinaryPrediction  # noqa: E402
from main import FallTemplateBot2026  # noqa: E402  (plantilla oficial de Metaculus)
import market_data  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger("bot_pro")

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "bot_config.json"
LOG_PATH = ROOT / "data" / "forecast_log.jsonl"

CURRENT_MODEL: contextvars.ContextVar[str | None] = contextvars.ContextVar("CURRENT_MODEL", default=None)


# --------------------------------------------------------------------------- utilidades
def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def weighted_median(values: list[float], weights: list[float]) -> float:
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc >= total / 2 - 1e-12:
            # media de los dos centrales si cae justo en la mitad
            if abs(acc - total / 2) < 1e-9 and i + 1 < len(pairs):
                return (v + pairs[i + 1][0]) / 2
            return v
    return pairs[-1][0]


def lessons_block(n: int = 8) -> str:
    """Lecciones aprendidas de los bots que mejor lo hicieron (las escribe learn_from_bots.py cada día)."""
    path = ROOT / "data" / "lessons.json"
    if n <= 0 or not path.exists():
        return ""
    try:
        items = [str(x.get("lesson", "")).strip() for x in json.loads(path.read_text(encoding="utf-8"))]
    except Exception:
        return ""
    items = [x for x in items if x][-n:]
    if not items:
        return ""
    return ("### Lessons learned from stronger bots on past questions (general habits, not facts about this question)\n"
            + "\n".join(f"- {x}" for x in items))


def openrouter_available_models() -> set[str] | None:
    """Lista pública de modelos de OpenRouter (sin clave). None si no se puede consultar."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
        r.raise_for_status()
        return {m["id"] for m in r.json().get("data", [])}
    except Exception as e:  # pragma: no cover
        logger.warning(f"No se pudo consultar la lista de modelos de OpenRouter: {e}")
        return None


def strip_prefix(model: str) -> str:
    return model.removeprefix("openrouter/")


GEMINI_FALLBACK = "openrouter/google/gemini-3.8-flash"
GEMINI_FREE_FLASH: str | None = None          # se rellena en resolve_gemini_models
GEMINI_SEM = asyncio.Semaphore(int(os.getenv("GEMINI_CONCURRENCY", "3")))  # el plan gratis limita peticiones/minuto


def has_asknews() -> bool:
    """AskNews acepta la clave nueva (ASKNEWS_API_KEY) o la antigua (ASKNEWS_CLIENT_ID + ASKNEWS_SECRET)."""
    return bool(os.getenv("ASKNEWS_API_KEY") or (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")))


ASKNEWS_USAGE = Path(__file__).resolve().parent / "data" / "asknews_usage.json"
_AN_LOCK = asyncio.Lock()


async def asknews_news(query: str, cfg: dict[str, Any]) -> str:
    """Noticias de AskNews. Con 'asknews_budget' en la configuración (saldo pequeño de pago por uso) hace UNA sola
    búsqueda de noticias recientes (1 crédito) y respeta un tope diario y mensual para que el saldo dure toda la
    temporada; sin él, usa la búsqueda completa de la plantilla (noticias recientes + archivo, ~6 créditos)."""
    budget = cfg.get("asknews_budget")
    if not budget:
        return await AskNewsSearcher().call_preconfigured_version("asknews/news-summaries", query)
    now = datetime.now(timezone.utc)
    month, day = now.strftime("%Y-%m"), now.strftime("%Y-%m-%d")
    async with _AN_LOCK:
        try:
            u = json.loads(ASKNEWS_USAGE.read_text(encoding="utf-8"))
        except Exception:
            u = {}
        if u.get("month") != month:
            u = {"month": month, "credits": 0, "day": day, "day_credits": 0}
        if u.get("day") != day:
            u.update(day=day, day_credits=0)
        if (u["credits"] >= int(budget.get("monthly_credits", 80))
                or u["day_credits"] >= int(budget.get("daily_credits", 4))):
            logger.info(f"AskNews: tope de créditos alcanzado ({u}); se omite")
            return ""
        u["credits"] += 1
        u["day_credits"] += 1
        ASKNEWS_USAGE.parent.mkdir(parents=True, exist_ok=True)
        ASKNEWS_USAGE.write_text(json.dumps(u), encoding="utf-8")
    from asknews_sdk import AsyncAskNewsSDK
    s = AskNewsSearcher()
    async with AsyncAskNewsSDK(client_id=s.client_id, client_secret=s.client_secret, api_key=s.api_key,
                               scopes={"news"}) as ask:
        resp = await ask.news.search_news(query=query[:400], n_articles=int(budget.get("n_articles", 10)),
                                          return_type="both", strategy="latest news")
    arts = resp.as_dicts or []
    if not arts:
        return ""
    return "Recent news articles (AskNews):\n\n" + s._format_articles(arts)


def free_mode() -> bool:
    """Sin clave de OpenRouter pero con la de Gemini: el bot funciona solo con Gemini gratis."""
    return not os.getenv("OPENROUTER_API_KEY") and bool(os.getenv("GEMINI_API_KEY"))


def _is_rate_limit(e: BaseException) -> bool:
    t = str(e).lower()
    return "429" in t or "resource_exhausted" in t or "rate limit" in t or "quota" in t


def _is_daily_quota(e: BaseException) -> bool:
    """Límite diario agotado (no vale la pena esperar): 'PerDay' o 'retry in XhYm'."""
    t = str(e)
    return "PerDay" in t or "per_day" in t.lower() or bool(re.search(r"retry in \d+h", t))


def _is_overloaded(e: BaseException) -> bool:
    t = str(e).lower()
    return "503" in t or "unavailable" in t or "overloaded" in t or "high demand" in t


GEMINI_POOL: list[str] = []          # modelos gratis de Google, de mejor a peor (cada uno tiene su propio cupo diario)
GEMINI_EXHAUSTED: set[str] = set()   # modelos que ya agotaron el cupo de hoy en esta ejecución


# --------------------------------------------------------------------------- OpenAI gratis (compartir datos)
# OpenAI regala cada día 250.000 tokens de sus modelos grandes y 2,5 millones de los mini a las cuentas que
# comparten datos. Pasarse de ahí se cobra del saldo, así que el bot lleva la cuenta y para antes del límite.
OPENAI_USAGE = Path(__file__).resolve().parent / "data" / "openai_usage.json"
OPENAI_DAILY = {"big": int(os.getenv("OPENAI_BIG_DAILY", "200000")), "small": int(os.getenv("OPENAI_SMALL_DAILY", "2100000"))}
OPENAI_MAX_OUT = {"big": 12000, "small": 6000}
OPENAI_FREE = {
    "big": {"gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o", "o1", "o3"},
    "small": {"gpt-5.4-mini", "gpt-5.4-nano", "gpt-5-mini", "gpt-5-nano", "gpt-4.1-mini", "gpt-4.1-nano",
              "gpt-4o-mini", "o3-mini", "o4-mini"},
}
_OA_LOCK = asyncio.Lock()


class OpenAIQuotaExhausted(RuntimeError):
    pass


def openai_pool(model: str) -> str | None:
    """'big', 'small' o None si el modelo no entra en los tokens gratis (entonces no se usa nunca)."""
    name = model.split("/", 1)[-1]
    for pool, names in OPENAI_FREE.items():
        if name in names:
            return pool
    return None


def _oa_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def openai_usage() -> dict[str, Any]:
    try:
        data = json.loads(OPENAI_USAGE.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    if data.get("date") != _oa_today():
        data = {"date": _oa_today(), "big": 0, "small": 0, "calls": 0}
    return data


def _oa_save(data: dict[str, Any]) -> None:
    OPENAI_USAGE.parent.mkdir(parents=True, exist_ok=True)
    OPENAI_USAGE.write_text(json.dumps(data), encoding="utf-8")


def _oa_post(body: dict[str, Any], timeout: int) -> dict[str, Any]:
    r = requests.post("https://api.openai.com/v1/chat/completions", json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"})
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
    return r.json()


async def openai_complete(model: str, prompt: str, reasoning: str | None, timeout: int) -> str:
    pool = openai_pool(model)
    if pool is None:
        raise OpenAIQuotaExhausted(f"{model} no está en los tokens gratis de OpenAI; no se usa")
    name = model.split("/", 1)[-1]
    max_out = OPENAI_MAX_OUT[pool]
    reserve = len(prompt) // 3 + max_out          # estimación prudente antes de llamar
    async with _OA_LOCK:
        data = openai_usage()
        if data[pool] + reserve > OPENAI_DAILY[pool]:
            raise OpenAIQuotaExhausted(f"cupo gratis de OpenAI ({pool}) de hoy casi agotado: {data[pool]} tokens")
        data[pool] += reserve
        _oa_save(data)
    body: dict[str, Any] = {"model": name, "messages": [{"role": "user", "content": prompt}],
                            "max_completion_tokens": max_out}
    if reasoning and (name.startswith(("gpt-5", "o1", "o3", "o4"))):
        body["reasoning_effort"] = {"high": "medium"}.get(reasoning, reasoning)   # 'high' gasta demasiados tokens
    used: int | None = None
    try:
        js: dict[str, Any] = {}
        for attempt in range(3):
            try:
                js = await asyncio.to_thread(_oa_post, body, timeout)
                break
            except RuntimeError as e:
                t = str(e)
                if "reasoning_effort" in t and "reasoning_effort" in body:
                    body.pop("reasoning_effort")
                    continue
                if attempt < 2 and (" 429" in t or " 50" in t):
                    await asyncio.sleep(20 * (attempt + 1))
                    continue
                raise
        usage = js.get("usage") or {}
        used = int(usage.get("total_tokens") or reserve)
        out = ((js.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if not out.strip():
            raise RuntimeError(f"{model}: respuesta vacía (¿se le acabaron los tokens de salida?)")
        return out
    finally:
        if used is None:
            used = len(prompt) // 4      # llamada fallida: se cuenta la entrada, por prudencia
        async with _OA_LOCK:
            data = openai_usage()
            data[pool] = max(0, data[pool] - reserve + used)
            data["calls"] = int(data.get("calls", 0)) + 1
            _oa_save(data)
            logger.info(f"OpenAI {name}: {used} tokens · hoy {data[pool]}/{OPENAI_DAILY[pool]} ({pool})")


# --------------------------------------------------------------------------- Mistral (plan gratis, sin tarjeta)
# Sin tarjeta no se puede cobrar nada: si se agota el cupo gratis, Mistral simplemente responde con error.
MISTRAL_DEFAULT = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
_MISTRAL_LOCK = asyncio.Lock()
_MISTRAL_LAST = [0.0]


def _mistral_post(body: dict[str, Any], timeout: int) -> dict[str, Any]:
    r = requests.post("https://api.mistral.ai/v1/chat/completions", json=body, timeout=timeout,
                      headers={"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY', '')}"})
    if r.status_code >= 400:
        raise RuntimeError(f"Mistral {r.status_code}: {r.text[:300]}")
    return r.json()


async def _mistral_once(name: str, prompt: str, reasoning: str | None, timeout: int) -> str:
    body: dict[str, Any] = {"model": name, "messages": [{"role": "user", "content": prompt}]}
    if reasoning:
        body["reasoning_effort"] = "high" if reasoning == "high" else "medium" if reasoning == "medium" else "low"
    for attempt in range(3):
        async with _MISTRAL_LOCK:   # el plan gratis admite ~1 petición por segundo
            wait = 1.2 - (asyncio.get_event_loop().time() - _MISTRAL_LAST[0])
            if wait > 0:
                await asyncio.sleep(wait)
            _MISTRAL_LAST[0] = asyncio.get_event_loop().time()
        try:
            js = await asyncio.to_thread(_mistral_post, body, timeout)
        except RuntimeError as e:
            t = str(e)
            if ("reasoning" in t or " 400" in t or " 422" in t) and "reasoning_effort" in body:
                body.pop("reasoning_effort")
                continue
            if attempt < 1 and (" 429" in t or " 50" in t):
                await asyncio.sleep(10)
                continue
            raise
        content = ((js.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if isinstance(content, list):   # los modelos con razonamiento devuelven trozos
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")
        if content.strip():
            return content
        raise RuntimeError(f"Mistral {name}: respuesta vacía")
    raise RuntimeError(f"Mistral {name}: sin respuesta")


MISTRAL_CHAIN = [MISTRAL_DEFAULT] + [m for m in ("mistral-small-latest", "mistral-large-latest", "open-mistral-nemo")
                                     if m != MISTRAL_DEFAULT]


async def mistral_complete(model: str | None, prompt: str, reasoning: str | None, timeout: int) -> str:
    """Prueba el modelo pedido y, si el plan gratis no lo permite (429/403), los demás modelos de Mistral."""
    chain = [model.split("/", 1)[-1]] if model else []
    chain += [m for m in MISTRAL_CHAIN if m not in chain]
    last: Exception | None = None
    for name in chain:
        try:
            out = await _mistral_once(name, prompt, reasoning, timeout)
            if name != chain[0]:
                logger.info(f"Mistral: {chain[0]} no disponible; respondió {name}")
                MISTRAL_CHAIN.remove(name)
                MISTRAL_CHAIN.insert(0, name)   # la próxima vez, directamente el que funciona
            return out
        except RuntimeError as e:
            last = e
            if not any(c in str(e) for c in (" 429", " 403", " 404", " 401")):
                raise
    raise last or RuntimeError("Mistral sin modelos disponibles")

class RobustLlm:
    """Llama al modelo con razonamiento alto; si el proveedor no lo admite, repite sin él.
    Los modelos 'gemini/...' van directos a Google (clave gratuita de AI Studio). En el plan gratis
    cada modelo tiene un cupo pequeño por minuto y por día: si se agota, se pasa al siguiente modelo
    gratuito de la lista; con OpenRouter, se usa el mismo tipo de modelo por OpenRouter."""

    def __init__(self, model: str, reasoning: str | None = "high", timeout: int = 420, search: bool = False):
        self.model = model
        self.reasoning = reasoning
        self.timeout = timeout
        self.search = search
        if model.startswith("openai/"):
            self._fallback = None
        elif model.startswith("gemini/"):
            self._fallback: RobustLlm | None = None if (search or free_mode()) else RobustLlm(GEMINI_FALLBACK, reasoning, timeout)
        else:
            self._with_reasoning = (
                GeneralLlm(model=model, timeout=timeout, allowed_tries=2,
                           extra_body={"reasoning": {"effort": reasoning}})
                if reasoning else None
            )
            self._plain = GeneralLlm(model=model, timeout=timeout, allowed_tries=3)
            self._fallback = None

    def _gemini_llm(self, model: str, reasoning: bool) -> GeneralLlm:
        extra: dict[str, Any] = {"tools": [{"googleSearch": {}}]} if self.search else {}
        if reasoning and self.reasoning and "gemma" not in model:
            extra["reasoning_effort"] = self.reasoning
        return GeneralLlm(model=model, timeout=self.timeout, allowed_tries=1, **extra)

    async def _gemini_once(self, model: str, prompt: str) -> str:
        try:
            out = await self._gemini_llm(model, True).invoke(prompt)
            if out and out.strip():
                return out
        except Exception as e:
            if _is_rate_limit(e):
                raise
            logger.warning(f"{model}: fallo con razonamiento ({str(e)[:150]}); repito sin él")
        return await self._gemini_llm(model, False).invoke(prompt)

    async def invoke(self, prompt: str) -> str:
        if self.model.startswith("mistral/"):
            try:
                return await mistral_complete(self.model, prompt, self.reasoning, self.timeout)
            except Exception as e:
                backup = f"gemini/{GEMINI_FREE_FLASH}" if GEMINI_FREE_FLASH else None
                if not backup:
                    raise
                logger.warning(f"{self.model} no disponible ({str(e)[:160]}); uso {backup}")
                return await RobustLlm(backup, self.reasoning, self.timeout).invoke(prompt)
        if self.model.startswith("openai/"):
            try:
                return await openai_complete(self.model, prompt, self.reasoning, self.timeout)
            except Exception as e:
                backup = f"gemini/{GEMINI_FREE_FLASH}" if GEMINI_FREE_FLASH else None
                if not backup:
                    raise
                logger.warning(f"{self.model} no disponible ({str(e)[:160]}); uso {backup}")
                return await RobustLlm(backup, self.reasoning, self.timeout).invoke(prompt)
        if not self.model.startswith("gemini/"):
            return await self._invoke(prompt)
        candidates = [self.model]
        if free_mode():
            candidates += [m for m in GEMINI_POOL if m != self.model and not (self.search and "gemma" in m)]
        last: BaseException | None = None
        for model in candidates:
            # El cupo de búsqueda en Google es aparte: agotarlo no debe bloquear el modelo para pronosticar
            key = f"{model}|search" if self.search else model
            if model in GEMINI_EXHAUSTED or key in GEMINI_EXHAUSTED:
                continue
            for i in range(3):
                try:
                    async with GEMINI_SEM:
                        return await self._gemini_once(model, prompt)
                except Exception as e:
                    last = e
                    if _is_daily_quota(e):
                        GEMINI_EXHAUSTED.add(key)
                        logger.warning(f"{model}: cupo gratis de hoy agotado ({'búsqueda' if self.search else 'modelo'}); paso al siguiente")
                        break
                    if _is_overloaded(e) and i == 0:
                        logger.warning(f"{model}: Google saturado (503), reintento en 15s")
                        await asyncio.sleep(15)
                        continue
                    if not _is_rate_limit(e):
                        break
                    wait = 25 * (i + 1)
                    logger.warning(f"{model}: límite por minuto del plan gratis, espero {wait}s")
                    await asyncio.sleep(wait)
            if not free_mode():
                break
        if self._fallback is not None:
            logger.warning(f"{self.model}: Google no responde ({str(last)[:150]}); uso {self._fallback.model}")
            return await self._fallback.invoke(prompt)
        if not self.search and os.getenv("OPENAI_API_KEY"):
            # Google saturado o sin cupo: primero GPT-5.4-mini (tokens gratis diarios), luego Mistral
            logger.warning(f"{self.model}: Google no responde ({str(last)[:150]}); uso openai/gpt-5.4-mini")
            try:
                return await openai_complete("openai/gpt-5.4-mini", prompt, self.reasoning, self.timeout)
            except Exception as e:
                logger.warning(f"GPT-5.4-mini tampoco responde: {str(e)[:150]}")
                last = e
        if not self.search and os.getenv("MISTRAL_API_KEY"):
            # último recurso: Mistral (gratis)
            logger.warning(f"{self.model}: Google no responde ({str(last)[:150]}); uso Mistral ({MISTRAL_DEFAULT})")
            try:
                return await mistral_complete(None, prompt, self.reasoning, self.timeout)
            except Exception as e:
                logger.warning(f"Mistral tampoco responde: {str(e)[:150]}")
                last = e
        raise last or RuntimeError("sin modelos Gemini disponibles")

    async def _invoke(self, prompt: str) -> str:
        if self._with_reasoning is not None:
            try:
                out = await self._with_reasoning.invoke(prompt)
                if out and out.strip():
                    return out
            except Exception as e:
                logger.warning(f"{self.model}: fallo con razonamiento alto ({e}); repito sin él")
        return await self._plain.invoke(prompt)


def resolve_gemini_models(cfg: dict[str, Any]) -> None:
    """Sustituye 'gemini/auto-flash' por el mejor modelo Flash gratuito disponible con GEMINI_API_KEY.
    Sin clave (o si falla la consulta) se usa el equivalente por OpenRouter."""
    global GEMINI_FREE_FLASH
    key = os.getenv("GEMINI_API_KEY")
    best = best_pro = best_gemma = None
    lite: list[str] = []
    if key:
        try:
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                             params={"key": key, "pageSize": 200}, timeout=30)
            r.raise_for_status()
            names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                     if "generateContent" in (m.get("supportedGenerationMethods") or [])]
            bad = ("lite", "tts", "image", "audio", "live", "embed", "transcribe", "exp", "thinking")
            flash = [n for n in names if n.startswith("gemini-") and "flash" in n and not any(b in n for b in bad)]

            def ver(n: str) -> tuple:
                m = re.match(r"gemini-(\d+)(?:\.(\d+))?", n)
                v = (int(m.group(1)), int(m.group(2) or 0)) if m else (0, 0)
                return v + (0 if "preview" in n else 1, -len(n))
            if flash:
                best = sorted(flash, key=ver)[-1]
            pro = [n for n in names if n.startswith("gemini-") and "pro" in n and not any(b in n for b in bad)]
            best_pro = sorted(pro, key=ver)[-1] if pro else None
            lite = [n for n in names if n.startswith("gemini-") and "flash-lite" in n
                    and not any(b in n for b in bad if b != "lite")]
            gemma = [n for n in names if n.startswith("gemma-") and n.endswith("-it")]

            def gsize(n: str) -> tuple:
                m = re.match(r"gemma-(\d+)[^-]*-(\d+)b", n)
                return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
            best_gemma = sorted(gemma, key=gsize)[-1] if gemma else None
            GEMINI_POOL[:] = (["gemini/" + n for n in sorted(flash, key=ver, reverse=True)[:3]]
                              + ["gemini/" + n for n in sorted(lite, key=ver, reverse=True)[:2]]
                              + (["gemini/" + best_gemma] if best_gemma else []))
        except Exception as e:
            logger.warning(f"No se pudo consultar la lista de modelos de Gemini: {e}")
    GEMINI_FREE_FLASH = best
    logger.info(f"Gemini gratis: {'gemini/' + best if best else 'no disponible, se usa OpenRouter'}"
                f"{' · pro: gemini/' + best_pro if best_pro else ''}")
    if GEMINI_POOL:
        logger.info(f"Modelos gratis en reserva (cada uno con su cupo diario): {GEMINI_POOL}")

    def sub(m: str) -> str | None:
        if m == "gemini/auto-flash":
            return f"gemini/{best}" if best else GEMINI_FALLBACK
        if m == "gemini-search/auto-flash":
            return f"gemini-search/{best}" if best else None
        if m == "gemini/auto-lite":   # para leer/convertir respuestas: el modelo con más cupo gratis
            if best_gemma:
                return f"gemini/{best_gemma}"
            return f"gemini/{sorted(lite)[-1]}" if lite else (f"gemini/{best}" if best else GEMINI_FALLBACK)
        if m == "gemini/auto-pro":
            return f"gemini/{best_pro}" if best_pro else (f"gemini/{best}" if best else None)
        return m

    def fix(lst: list[str]) -> list[str]:
        out = []
        for m in lst:
            v = sub(m)
            if v and v not in out:
                out.append(v)
        return out

    for k in ("forecasters", "fallback_forecasters", "researchers"):
        if k in cfg:
            cfg[k] = fix(cfg[k])
    for k in ("research_planner", "supervisor", "followup_researcher", "parser"):
        if cfg.get(k):
            cfg[k] = sub(cfg[k]) or cfg[k]


# --------------------------------------------------------------------------- el bot
class ProBot(FallTemplateBot2026):
    _max_concurrent_questions = 5
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, cfg: dict[str, Any], forecasters: list[str], researchers: list[str], **kwargs):
        self.cfg = cfg
        self.forecasters = forecasters
        self.researchers = researchers
        self.weights: dict[str, float] = cfg.get("weights", {})
        self._robust: dict[str, RobustLlm] = {}
        self._rr: dict[str, int] = {}
        self._individual: dict[str, list[dict[str, Any]]] = {}
        self._reasons: dict[str, list[tuple[str, float, str]]] = {}
        self._research: dict[str, str] = {}
        self._extra: dict[str, dict[str, Any]] = {}
        self._quant: dict[str, dict[str, Any]] = {}
        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------- fecha "de hoy"
    def _today(self, question: MetaculusQuestion) -> str:
        """En las pruebas con preguntas pasadas, 'hoy' es el día en que la pregunta cerró."""
        if self.cfg.get("_as_of_close") and question.close_time:
            return question.close_time.strftime("%Y-%m-%d")
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ---------------------------------------------------------------- modelos
    def _qkey(self, question: MetaculusQuestion) -> str:
        return str(question.id_of_question or question.id_of_post or question.page_url)

    def _llm_for(self, model: str) -> RobustLlm:
        if model not in self._robust:
            self._robust[model] = RobustLlm(model, reasoning=self.cfg.get("reasoning_effort", "high"))
        return self._robust[model]

    def get_llm(self, purpose: str = "default", guarantee_type: Any = None):  # type: ignore[override]
        model = CURRENT_MODEL.get()
        if purpose == "default" and model is not None:
            return self._llm_for(model)
        return super().get_llm(purpose, guarantee_type)

    async def _make_prediction(self, question: MetaculusQuestion, research: str):  # type: ignore[override]
        key = self._qkey(question)
        idx = self._rr.get(key, -1) + 1
        self._rr[key] = idx
        model = self.forecasters[idx % len(self.forecasters)]
        token = CURRENT_MODEL.set(model)
        try:
            prediction = await super()._make_prediction(question, research)
        finally:
            CURRENT_MODEL.reset(token)
        value = prediction.prediction_value
        rec: dict[str, Any] = {"model": model}
        if isinstance(question, BinaryQuestion):
            rec["p"] = float(value)  # type: ignore[arg-type]
            self._reasons.setdefault(key, []).append((model, float(value), str(prediction.reasoning or "")[-4000:]))  # type: ignore[arg-type]
        elif isinstance(question, MultipleChoiceQuestion):
            rec["p"] = {o.option_name: o.probability for o in value.predicted_options}  # type: ignore[union-attr]
        self._individual.setdefault(key, []).append(rec)
        return prediction

    # ---------------------------------------------------------------- investigación
    async def run_research(self, question: MetaculusQuestion) -> str:  # type: ignore[override]
        async with self._concurrency_limiter:
            prompt = self._pro_research_prompt(question)
            tasks: list[tuple[str, Any]] = []
            for r in self.researchers:
                if r.startswith("asknews/"):
                    if has_asknews():
                        if r == "asknews/news-summaries":
                            tasks.append(("AskNews (noticias)", asknews_news(question.question_text, self.cfg)))
                        else:
                            tasks.append((r, AskNewsSearcher().call_preconfigured_version(r, prompt)))
                elif r.startswith("gemini-search/"):
                    tasks.append((f"Google Search ({r.split('/', 1)[1]})",
                                  RobustLlm("gemini/" + r.split("/", 1)[1], reasoning=None, timeout=300, search=True).invoke(prompt)))
                else:
                    tasks.append((r, RobustLlm(r, reasoning=None, timeout=int(self.cfg.get("research_timeout", 600))).invoke(prompt)))
            if self.cfg.get("read_resolution_sources", True):
                for u in market_data.urls_in(question.resolution_criteria, question.fine_print):
                    tasks.append((f"Fuente de resolución {u}", self._source_page(u)))
            if self.cfg.get("market_lookup", True):
                tasks.append(("Mercados de predicción (Polymarket/Kalshi/Manifold)", self._market_lookup(question)))
            if self.cfg.get("related_metaculus", True):
                tasks.append(("Preguntas relacionadas en Metaculus (predicción de la comunidad)",
                              asyncio.to_thread(market_data.related_metaculus, question.question_text, question.id_of_post)))
            if self.cfg.get("free_sources", True):
                tasks.append(("Buscadores gratuitos (Tavily, Google News, GDELT, Wikipedia…)",
                              asyncio.to_thread(market_data.free_search, question.question_text)))
            parts = await self._run_research_tasks(tasks)

            # Segunda ronda: un modelo lee lo encontrado y pide lo que falta (búsqueda "agéntica")
            if parts and int(self.cfg.get("research_rounds", 2)) >= 2:
                try:
                    followups = await self._plan_followups(question, "\n\n".join(parts))
                except Exception as e:
                    logger.warning(f"No se pudieron planear búsquedas de seguimiento: {e}")
                    followups = []
                tasks2: list[tuple[str, Any]] = []
                web = [r for r in self.researchers if not r.startswith(("asknews/", "gemini-search/"))]
                fr = self.cfg.get("followup_researcher")
                web_f = [fr] if fr in web else web[:1]
                gsearch = [r for r in self.researchers if r.startswith("gemini-search/")]
                for fq in followups:
                    for r in web_f:
                        tasks2.append((f"{r} · seguimiento: {fq[:80]}",
                                       RobustLlm(r, reasoning=None, timeout=300).invoke(self._followup_prompt(question, fq))))
                    if self.cfg.get("free_sources", True) and (os.getenv("SERPER_API_KEY") or os.getenv("TAVILY_API_KEY") or os.getenv("BRAVE_API_KEY") or os.getenv("EXA_API_KEY")):
                        tasks2.append((f"Buscadores gratuitos · seguimiento: {fq[:80]}",
                                       asyncio.to_thread(market_data.free_search, question.question_text, fq)))
                    if not web_f and gsearch:
                        g = "gemini/" + gsearch[0].split("/", 1)[1]
                        tasks2.append((f"Google Search · seguimiento: {fq[:80]}",
                                       RobustLlm(g, reasoning=None, timeout=300, search=True).invoke(self._followup_prompt(question, fq))))
                    if has_asknews() and not self.cfg.get("asknews_budget"):
                        tasks2.append((f"asknews · seguimiento: {fq[:80]}",
                                       AskNewsSearcher().call_preconfigured_version("asknews/news-summaries", fq)))
                parts += await self._run_research_tasks(tasks2)

            if isinstance(question, NumericQuestion) and self.cfg.get("quant_baseline", True):
                try:
                    qb = await self._quant_baseline(question)
                    if qb:
                        parts.insert(0, "### Línea base cuantitativa (datos de mercado reales)\n" + qb["text"])
                except Exception as e:
                    logger.warning(f"Línea base cuantitativa falló en {question.page_url}: {e}")
            research = "\n\n".join(parts) if parts else "(No se pudo obtener investigación; razona con lo que sepas.)"
            les = lessons_block(int(self.cfg.get("lessons_in_prompt", 8)))
            if les:
                research = les + "\n\n" + research
            self._research[self._qkey(question)] = research
            logger.info(f"Investigación para {question.page_url}: {len(parts)} bloques, {len(research)} caracteres")
            return research

    async def _quant_baseline(self, question: NumericQuestion) -> dict[str, Any] | None:
        planner = RobustLlm(self.cfg["research_planner"], reasoning="low", timeout=180)
        out = await planner.invoke(clean_indents(
            f"""
            Does this forecasting question ask for the LEVEL of a public market or economic time series on a specific date
            (e.g. a stock index close, a stock price, an exchange rate, a commodity future, a Treasury yield, VIX)?
            Question: {question.question_text}
            Sub-question option (if any): {question.group_question_option}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}
            Units: {question.unit_of_measure}

            If yes, give the best free data source: Yahoo Finance ticker (e.g. ^GSPC, ^NDX, ^VIX, NVDA, EURUSD=X, GC=F, CL=F, BTC-USD)
            or a FRED series id (e.g. DGS10, DGS2, BAMLH0A0HYM2, DFF), and the target date.
            The value from the source must be in the SAME units as the question (e.g. percent yields as 4.25, not 0.0425).
            Answer ONLY with JSON: {{"applicable": true/false, "source": "yahoo" or "fred", "symbol": "...", "target_date": "YYYY-MM-DD", "measure": "level" or "other"}}
            """
        ))
        m = re.search(r"\{.*\}", out or "", re.S)
        if not m:
            return None
        spec = json.loads(m.group(0))
        if not spec.get("applicable") or spec.get("measure") != "level" or spec.get("source") not in ("yahoo", "fred"):
            return None
        target = datetime.fromisoformat(str(spec["target_date"])[:10]).date()
        today = datetime.fromisoformat(self._today(question)).date()
        b = await asyncio.to_thread(market_data.baseline, spec["source"], str(spec["symbol"]), target, today)
        if not b:
            return None
        lo, hi = question.lower_bound, question.upper_bound
        if lo is not None and hi is not None and not (lo <= b["current"] <= hi):
            logger.warning(f"Línea base descartada: valor actual {b['current']} fuera de [{lo}, {hi}] en {question.page_url}")
            return None
        b["spec"] = spec
        self._quant[self._qkey(question)] = b
        self._extra.setdefault(self._qkey(question), {})["quant"] = {
            "source": spec["source"], "symbol": spec["symbol"], "target": str(target), "current": b["current"]}
        return b

    @staticmethod
    async def _source_page(url: str) -> str:
        txt = await asyncio.to_thread(market_data.page_text, url)
        return f"(Text of the page named in the resolution criteria, fetched today; may be partial)\n{txt}" if txt else ""

    async def _market_lookup(self, question: MetaculusQuestion) -> str:
        """Busca la misma pregunta en mercados de predicción y devuelve sus precios como evidencia."""
        if free_mode():   # sin gastar cupo: busca con el propio título
            words = re.findall(r"[A-Za-z0-9][\w'.-]*", question.question_text or "")
            stop = {"will", "the", "a", "an", "of", "in", "on", "by", "before", "after", "be", "to", "for", "what", "how", "many",
                    "which", "who", "is", "are", "and", "or", "than", "more", "less", "at", "least", "with", "end", "between"}
            terms = [" ".join([w for w in words if w.lower() not in stop][:5])]
            found = [mk for t in terms for mk in await asyncio.to_thread(market_data.prediction_markets, t)]
            if found:
                self._extra.setdefault(self._qkey(question), {})["markets"] = found[:8]
            return market_data.markets_text(found[:8])
        llm = RobustLlm(self.cfg.get("parser", self.cfg["research_planner"]), reasoning="low", timeout=120)
        out = await llm.invoke(clean_indents(
            f"""
            Give 2 short keyword searches (2-5 words each) to find prediction markets (Polymarket, Manifold)
            about this question: {question.question_text}
            Answer ONLY with a JSON array of strings.
            """
        ))
        m = re.search(r"\[.*\]", out or "", re.S)
        terms = [str(x) for x in json.loads(m.group(0))][:2] if m else []
        found: list[dict] = []
        for t in terms:
            for mk in await asyncio.to_thread(market_data.prediction_markets, t):
                if mk.get("url") not in {f.get("url") for f in found}:
                    found.append(mk)
        if found:
            self._extra.setdefault(self._qkey(question), {})["markets"] = found[:8]
        return market_data.markets_text(found[:8])

    @staticmethod
    async def _run_research_tasks(tasks: list[tuple[str, Any]]) -> list[str]:
        if not tasks:
            return []
        results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
        parts = []
        for (name, _), res in zip(tasks, results):
            if isinstance(res, Exception):
                logger.warning(f"Investigación con {name} falló: {res}")
                continue
            if res and str(res).strip():
                parts.append(f"### Fuente: {name}\n{str(res).strip()}")
        return parts

    async def _plan_followups(self, question: MetaculusQuestion, research: str) -> list[str]:
        planner = RobustLlm(self.cfg["research_planner"], reasoning="medium", timeout=240)
        out = await planner.invoke(clean_indents(
            f"""
            A superforecaster must forecast this question:
            {question.question_text}

            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}

            This is the research gathered so far:
            {research[:15000]}

            List up to {int(self.cfg.get("followups", 3))} specific web-search questions that would fill the most important GAPS
            (e.g. the latest value of the exact metric in the criteria, an upcoming scheduled event, a missing base rate,
            whether the question may already be resolved, contradictions between sources).
            If nothing important is missing, return an empty list.
            Answer ONLY with a JSON array of strings.
            """
        ))
        m = re.search(r"\[.*\]", out or "", re.S)
        items: list[str] = []
        if m:
            try:
                items = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
            except Exception:
                items = []
        if not items:
            items = [ln.strip("-*• ").strip() for ln in (out or "").splitlines() if ln.strip().startswith(("-", "*", "•"))]
        return items[: int(self.cfg.get("followups", 3))]

    @staticmethod
    def _followup_prompt(question: MetaculusQuestion, follow: str) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return clean_indents(
            f"""
            Today is {today}. Search the web and answer this research question precisely, with dates and sources.
            Do NOT give a forecast.
            Research question: {follow}
            (Context: it is needed to forecast "{question.question_text}")
            """
        )

    @staticmethod
    def _pro_research_prompt(question: MetaculusQuestion) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return clean_indents(
            f"""
            You are the research assistant of a professional superforecaster. Today is {today}.
            Search the web for the most recent and relevant information about the question below. Do NOT give a forecast.

            Question: {question.question_text}

            Resolution criteria: {question.resolution_criteria}

            Fine print: {question.fine_print}

            Report, with dates and sources for every fact:
            1. Current status: the latest data points and news directly relevant to the resolution criteria (exact numbers, dates).
            2. Whether the question may ALREADY be effectively resolved or about to be, according to the exact criteria.
            3. Scheduled events before the resolution date that could decide it (votes, releases, data publications, deadlines).
            4. Base rates: list at least 3 resolved analogous past cases (what, when, outcome) and the resulting frequency.
            5. What prediction markets (Polymarket, Kalshi, Manifold), polls or expert forecasts say, with prices/numbers and dates.
            6. Read the resolution source named in the criteria (if any) and report its latest relevant figure.
            Be concise and factual.
            """
        )

    # ---------------------------------------------------------------- preguntas sí/no
    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str):  # type: ignore[override]
        today = self._today(question)
        close = question.close_time.strftime("%Y-%m-%d") if question.close_time else "unknown"
        resolve = question.scheduled_resolution_time.strftime("%Y-%m-%d") if question.scheduled_resolution_time else "unknown"
        prompt = clean_indents(
            f"""
            You are an elite superforecaster competing in a forecasting tournament scored with log scores:
            confident forecasts that turn out wrong are punished very hard, but timid forecasts near 50% also lose points
            when the evidence is clear.

            Question: {question.question_text}

            Background: {question.background_info}

            Resolution criteria (these have NOT yet been satisfied when the question was written):
            {question.resolution_criteria}

            Fine print: {question.fine_print}

            Today: {today}. Question closes: {close}. Scheduled resolution: {resolve}.

            Research from several independent sources (may contain errors or contradictions; weigh them):
            {research}

            Work through these steps in writing:
            1. Exactly what must happen, by when, and according to which source, for the question to resolve YES. List any traps
               (dates and time zones, thresholds, "at least" vs "more than", which data release counts, ambiguity rules).
            2. Is it already effectively decided by the information available? If so, say so clearly.
            3. Base rate: name one or two reference classes and estimate how often YES happens in them.
            4. Time left and the status quo outcome if nothing changes (the world usually changes slowly).
            5. The strongest specific arguments for YES and for NO from the research, and how much each should move you from the base rate.
            6. Final calibrated probability. Be decisive when the evidence is strong, humble when it is not.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(self, question: BinaryQuestion, prompt: str):  # type: ignore[override]
        """Lee 'Probability: ZZ%' directamente del texto (ahorra una llamada al modelo por predicción)."""
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        m = re.findall(r"Probability:\s*\**\s*(\d{1,3}(?:\.\d+)?)\s*%", reasoning or "")
        if m:
            p = max(0.01, min(0.99, float(m[-1]) / 100))
            return ReasonedPrediction(prediction_value=p, reasoning=reasoning)
        binary_prediction: BinaryPrediction = await template_main.structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples)
        return ReasonedPrediction(prediction_value=max(0.01, min(0.99, binary_prediction.prediction_in_decimal)),
                                  reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str):  # type: ignore[override]
        today = self._today(question)
        close = question.close_time.strftime("%Y-%m-%d") if question.close_time else "unknown"
        prompt = clean_indents(
            f"""
            You are an elite superforecaster in a tournament scored with log scores: putting a tiny probability on the option
            that happens is punished very hard, but spreading probability evenly when the evidence is clear also loses points.

            Question: {question.question_text}
            Options: {question.options}

            Background: {question.background_info}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}

            Today: {today}. Question closes: {close}.

            Research from several independent sources (may contain errors; weigh them):
            {research}

            Work through these steps in writing:
            1. Exactly how the question resolves and any traps (which source counts, dates, ties, "or more" buckets, edge cases).
            2. Is the answer already effectively decided by the research? If so, concentrate probability accordingly.
            3. Base rates: for comparable past cases, how often did each kind of option happen?
            4. Status quo and time left: what happens if nothing changes?
            5. If the options are ORDERED (numbers, ranges, dates), your probabilities should form a smooth, usually single-peaked
               shape around your best estimate, with the tails decaying gradually (no gaps between neighbouring options).
            6. Never put less than 1% on an option that is not ruled out by facts; leave real probability for surprises.

            {self._get_conditional_disclaimer_if_necessary(question)}
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    # ---------------------------------------------------------------- agregación y calibración
    def _calibrate_binary(self, p_raw: float) -> float:
        platt = self.cfg.get("platt", {"a": 0.0, "b": 1.0})
        lo, hi = self.cfg.get("clip", [0.02, 0.98])
        p = sigmoid(platt.get("a", 0.0) + platt.get("b", 1.0) * logit(p_raw))
        return min(max(p, lo), hi)

    async def _aggregate_predictions(self, predictions, question):  # type: ignore[override]
        key = self._qkey(question)
        recs = self._individual.get(key, [])
        if isinstance(question, BinaryQuestion) and recs:
            vals = [logit(r["p"]) for r in recs]
            ws = [float(self.weights.get(r["model"], 1.0)) for r in recs]
            raw = sigmoid(weighted_median(vals, ws))
            ps = [r["p"] for r in recs]
            if len(ps) >= 2 and max(ps) - min(ps) >= float(self.cfg.get("supervisor_spread", 0.30)):
                try:
                    sup = await self._supervise_binary(question)
                    if sup is not None:
                        self._extra.setdefault(key, {})["supervisor"] = {"p": sup, "ensemble": raw}
                        raw = sigmoid((logit(raw) + logit(sup)) / 2)
                except Exception as e:
                    logger.warning(f"El juez falló en {question.page_url}: {e}")
            if self.cfg.get("sanity_check", True) and not free_mode():
                try:
                    chk = await self._sanity_check_binary(question, raw)
                    if chk:
                        self._extra.setdefault(key, {})["check"] = chk
                        if chk.get("verdict") == "adjust" and chk.get("p") is not None:
                            raw = sigmoid((logit(raw) + logit(chk["p"])) / 2)
                except Exception as e:
                    logger.warning(f"La revisión final falló en {question.page_url}: {e}")
            final = self._calibrate_binary(raw)
            self._write_log(question, recs, raw=raw, final=final)
            logger.info(f"{question.page_url}: modelos {[round(r['p'], 3) for r in recs]} -> bruto {raw:.3f} -> final {final:.3f}")
            return final
        if isinstance(question, (NumericQuestion, DateQuestion)) and self.cfg.get("numeric_mixture", True) and len(predictions) > 1:
            tails: dict[str, float] = {}
            if self.cfg.get("ask_tails", True) and not free_mode():
                try:
                    tails = await self._tail_probs(question)
                    if tails:
                        self._extra.setdefault(key, {})["tails"] = tails
                except Exception as e:
                    logger.warning(f"No se pudieron estimar las colas en {question.page_url}: {e}")
            try:
                aggregate = self._mixture_numeric(predictions, question, self._quant.get(key),
                                                  float(self.cfg.get("quant_weight", 2.0)),
                                                  float(self.cfg.get("numeric_tail_min", 0.02)),
                                                  float(self.cfg.get("cdf_max_jump", 0.18)), tails)
                self._write_log(question, recs, final=None)
                return aggregate
            except Exception as e:
                logger.warning(f"Mezcla numérica falló ({e}); uso la mediana de la plantilla")
        aggregate = await super()._aggregate_predictions(predictions, question)
        if isinstance(question, MultipleChoiceQuestion) and isinstance(aggregate, PredictedOptionList):
            floor = float(self.cfg.get("mc_floor", 0.01))
            opts = aggregate.predicted_options
            temp = float((self.cfg.get("mc_calibration") or {}).get("temp", 1.0))   # lo aprende learn.py
            raw = [max(o.probability, 1e-6) ** (1.0 / temp) for o in opts]
            tot = sum(raw)
            probs = [max(r / tot, floor) for r in raw]
            s = sum(probs)
            for o, p in zip(opts, probs):
                o.probability = p / s
            # redondeo para que sume exactamente 1
            diff = 1.0 - sum(o.probability for o in opts)
            opts[max(range(len(opts)), key=lambda i: opts[i].probability)].probability += diff
            self._write_log(question, recs, final={o.option_name: o.probability for o in opts})
        else:
            self._write_log(question, recs, final=None)
        return aggregate

    @staticmethod
    def _safe_cdf(h, question, tail_min: float = 0.02, max_jump: float = 0.18, tails: dict[str, float] | None = None):
        """Protege contra los dos errores numéricos más caros:
        1) colas aplastadas en límites abiertos (si resuelve fuera de rango, se pierde muchísimo);
        2) saltos de la CDF mayores que el máximo que permite Metaculus (0,2 entre puntos)."""
        h = np.clip(np.maximum.accumulate(np.asarray(h, dtype=float)), 0.0, 1.0)
        tails = tails or {}
        # si se preguntaron las colas, se mezcla a medias lo que dicen los modelos con lo que se estimó aparte
        lo_min = max(tail_min, 0.5 * h[0] + 0.5 * tails["below"]) if "below" in tails else tail_min
        hi_min = max(tail_min, 0.5 * (1 - h[-1]) + 0.5 * tails["above"]) if "above" in tails else tail_min
        if question.open_lower_bound and h[0] < lo_min:
            t = (lo_min - h[0]) / (1 - h[0])
            h = (1 - t) * h + t
        if question.open_upper_bound and 1 - h[-1] < hi_min:
            t = (hi_min - (1 - h[-1])) / h[-1]
            h = (1 - t) * h
        for _ in range(200):
            if np.max(np.diff(h)) <= max_jump:
                break
            sm = np.convolve(np.pad(h, 2, mode="edge"), np.ones(5) / 5, mode="valid")
            sm[0], sm[-1] = h[0], h[-1]
            h = np.maximum.accumulate(0.6 * h + 0.4 * sm)
        return h

    @staticmethod
    def _mixture_numeric(predictions, question, quant: dict[str, Any] | None = None, quant_weight: float = 2.0,
                         tail_min: float = 0.02, max_jump: float = 0.18, tails: dict[str, float] | None = None):
        """Mezcla de distribuciones: media de las CDF de cada modelo (+ la línea base cuantitativa si la hay)."""
        cdfs = [p.get_cdf() for p in predictions]
        weights = [1.0] * len(cdfs)
        if quant:
            try:
                lo, hi = question.lower_bound, question.upper_bound
                pts = []
                for pr, v in quant["percentiles"]:
                    if lo is not None and not question.open_lower_bound:
                        v = max(v, lo)
                    if hi is not None and not question.open_upper_bound:
                        v = min(v, hi)
                    if pts and v <= pts[-1].value:
                        continue
                    pts.append(Percentile(value=v, percentile=pr))
                qd = NumericDistribution.from_question(pts, question)
                cdfs.append(qd.get_cdf())
                weights.append(quant_weight)
            except Exception as e:
                logger.warning(f"No se pudo añadir la línea base a la mezcla: {e}")
        xs = [pt.value for pt in cdfs[0]]
        for c in cdfs:
            if [pt.value for pt in c] != xs:
                raise ValueError("ejes distintos")
        heights = np.average(np.array([[pt.percentile for pt in c] for c in cdfs]), axis=0, weights=weights)
        heights = ProBot._safe_cdf(heights, question, tail_min, max_jump, tails)
        mixed = [Percentile(value=x, percentile=h) for x, h in zip(xs, heights.tolist())]
        return NumericDistribution.from_question(mixed, question)

    async def _sanity_check_binary(self, question: BinaryQuestion, p: float) -> dict[str, Any] | None:
        """Última revisión antes de enviar: busca los errores que más puntos cuestan
        (pregunta ya decidida, fechas/zonas horarias, umbrales, unidades, criterios mal leídos)."""
        key = self._qkey(question)
        research = self._research.get(key, "")
        close = question.close_time.strftime("%Y-%m-%d") if question.close_time else "unknown"
        resolve = question.scheduled_resolution_time.strftime("%Y-%m-%d") if question.scheduled_resolution_time else "unknown"
        checker = RobustLlm(self.cfg.get("checker", self.cfg["supervisor"]), reasoning="medium", timeout=300)
        out = await checker.invoke(clean_indents(
            f"""
            You are the final reviewer of a forecasting team. Before submission, check this forecast for costly mistakes ONLY.
            Today: {self._today(question)}. Closes: {close}. Scheduled resolution: {resolve}.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}

            Research:
            {research[:15000]}

            Proposed probability of YES: {p:.1%}

            Check, citing the research:
            1. Is the outcome ALREADY determined (or practically certain) by facts in the research? If so, is the forecast extreme enough?
            2. Dates and time zones: is the deadline/measurement date read correctly? Is there enough time left for the event?
            3. Thresholds and units: "more than" vs "at least", which data release or source counts, units/scales.
            4. Any other misreading of the resolution criteria or fine print.
            Do NOT re-forecast from scratch and do NOT adjust for mere judgement differences.
            Answer ONLY with JSON: {{"verdict": "ok" or "adjust", "issue": "<one sentence>", "probability": <0-100, only if adjust>}}
            """
        ))
        m = re.search(r"\{.*\}", out or "", re.S)
        if not m:
            return None
        j = json.loads(m.group(0))
        res: dict[str, Any] = {"verdict": str(j.get("verdict", "ok")).lower(), "issue": str(j.get("issue", ""))[:300]}
        if res["verdict"] == "adjust" and isinstance(j.get("probability"), (int, float)):
            res["p"] = min(max(float(j["probability"]) / 100, 0.005), 0.995)
            logger.info(f"Revisión final en {question.page_url}: {res['issue']} -> {res['p']:.0%} (antes {p:.0%})")
        else:
            res["verdict"] = "ok"
        return res

    async def _tail_probs(self, question: MetaculusQuestion) -> dict[str, float]:
        """Pregunta directamente la probabilidad de quedar fuera del rango en los límites abiertos."""
        if not (question.open_lower_bound or question.open_upper_bound):
            return {}
        lo = getattr(question, "lower_bound", None)
        hi = getattr(question, "upper_bound", None)
        research = self._research.get(self._qkey(question), "")
        llm = RobustLlm(self.cfg.get("checker", self.cfg["supervisor"]), reasoning="medium", timeout=300)
        out = await llm.invoke(clean_indents(
            f"""
            Today: {self._today(question)}. Forecasting question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Units: {getattr(question, "unit_of_measure", "")}
            The answer range shown is from {lo} to {hi}{" (values below are possible)" if question.open_lower_bound else ""}{" (values above are possible)" if question.open_upper_bound else ""}.

            Research:
            {research[:12000]}

            Estimate the probability that the final value is BELOW {lo} and the probability that it is ABOVE {hi}.
            Think about tail risks and surprises; do not say 0.
            Answer ONLY with JSON: {{"below": <0-100>, "above": <0-100>}}
            """
        ))
        m = re.search(r"\{.*\}", out or "", re.S)
        if not m:
            return {}
        j = json.loads(m.group(0))
        res = {}
        if question.open_lower_bound and isinstance(j.get("below"), (int, float)):
            res["below"] = min(max(float(j["below"]) / 100, 0.0), 0.3)
        if question.open_upper_bound and isinstance(j.get("above"), (int, float)):
            res["above"] = min(max(float(j["above"]) / 100, 0.0), 0.3)
        return res

    async def _crux_research(self, question: MetaculusQuestion, blocks: str) -> str:
        """Cuando los modelos discrepan, busca justo el dato que explica el desacuerdo (lo hacía el mejor bot de código abierto)."""
        planner = RobustLlm(self.cfg.get("research_planner", self.cfg["supervisor"]), reasoning="low", timeout=180)
        out = await planner.invoke(clean_indents(
            f"""
            Several forecasters disagree on: {question.question_text}
            Their reasoning (truncated):
            {blocks[:12000]}
            Identify the single factual point that most explains the disagreement and write ONE short web-search query
            (max 10 words) that would settle it. Answer ONLY with the query.
            """
        ))
        q = (out or "").strip().strip('"').splitlines()[0][:160] if out else ""
        if not q:
            return ""
        parts = []
        txt = await asyncio.to_thread(market_data.free_search, question.question_text, q)
        if txt:
            parts.append(txt)
        if has_asknews():
            try:
                parts.append(str(await asknews_news(q, self.cfg))[:6000])
            except Exception as e:
                logger.info(f"AskNews en el desacuerdo falló: {e}")
        self._extra.setdefault(self._qkey(question), {})["crux_query"] = q
        return "\n\n".join(parts)

    async def _supervise_binary(self, question: BinaryQuestion) -> float | None:
        key = self._qkey(question)
        views = self._reasons.get(key, [])
        research = self._research.get(key, "")
        blocks = "\n\n".join(f"--- Forecaster {i + 1} said {p:.0%} ---\n{txt}" for i, (_, p, txt) in enumerate(views))
        today = self._today(question)
        crux_research = ""
        if self.cfg.get("crux_research", True):
            try:
                crux_research = await self._crux_research(question, blocks)
            except Exception as e:
                logger.warning(f"Búsqueda sobre el desacuerdo falló en {question.page_url}: {e}")
        if crux_research:
            research = research + "\n\n### Targeted research on the point of disagreement\n" + crux_research
        judge = RobustLlm(self.cfg["supervisor"], reasoning=self.cfg.get("reasoning_effort", "high"), timeout=420)
        out = await judge.invoke(clean_indents(
            f"""
            You are the lead superforecaster of a team. Your forecasters disagree strongly on this question.
            Today: {today}.

            Question: {question.question_text}
            Resolution criteria: {question.resolution_criteria}
            Fine print: {question.fine_print}

            Research:
            {research[:20000]}

            The forecasters' reasoning:
            {blocks[:30000]}

            Find the source of the disagreement: a misread of the resolution criteria, a fact one of them missed,
            an outdated piece of information, or a genuine judgement call. Check each claim against the research.
            Then give your own calibrated probability. The last thing you write is: "Probability: ZZ%"
            """
        ))
        m = re.findall(r"Probability:\s*([0-9]+(?:\.[0-9]+)?)\s*%", out or "")
        if not m:
            return None
        return min(max(float(m[-1]) / 100, 0.001), 0.999)

    def _write_log(self, question: MetaculusQuestion, recs, raw: float | None = None, final: Any = None) -> None:
        if self.cfg.get("_no_log"):
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "post_id": question.id_of_post,
            "question_id": question.id_of_question,
            "url": question.page_url,
            "type": type(question).__name__,
            "title": (question.question_text or "")[:200],
            "close_time": question.close_time.isoformat() if question.close_time else None,
            "models": recs,
            "raw": raw,
            "final": final,
            **self._extra.get(self._qkey(question), {}),
            "platt": self.cfg.get("platt"),
            "mode": self.cfg.get("_mode"),
            "profile": self.cfg.get("_profile"),
            "config_version": self.cfg.get("version"),
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- Market Pulse
def last_forecast_times() -> dict[int, datetime]:
    out: dict[int, datetime] = {}
    if not LOG_PATH.exists():
        return out
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
            qid, ts = e.get("question_id"), datetime.fromisoformat(e["ts"])
        except Exception:
            continue
        if qid is not None and (qid not in out or ts > out[qid]):
            out[qid] = ts
    return out


def spot_time(q: MetaculusQuestion) -> datetime | None:
    for src in (q.api_json.get("question", {}) if isinstance(q.api_json, dict) else {}, q.api_json or {}):
        v = src.get("spot_scoring_time") if isinstance(src, dict) else None
        if v:
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except ValueError:
                pass
    return q.close_time


def preclose_due(q: MetaculusQuestion, last: datetime | None, now: datetime, cfg: dict[str, Any]) -> bool:
    """Torneo principal: volver a pronosticar en los últimos minutos antes del cierre, con las noticias más frescas."""
    close = q.close_time
    if close is None or last is None or now >= close:
        return False
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    mins_left = (close - now).total_seconds() / 60
    mins_since = (now - last).total_seconds() / 60
    return mins_left <= float(cfg.get("minutes_before", 45)) and mins_since >= float(cfg.get("min_gap_minutes", 60))


def market_pulse_due(q: MetaculusQuestion, last: datetime | None, now: datetime, cfg: dict[str, Any]) -> bool:
    """Cuenta solo el pronóstico vigente en el momento de puntuar: actualizar a diario y más a menudo al final."""
    spot = spot_time(q)
    if spot is not None and now >= spot:
        return False
    if last is None:
        return True
    hours_since = (now - last).total_seconds() / 3600
    hours_left = (spot - now).total_seconds() / 3600 if spot else 1e9
    if hours_left <= float(cfg.get("final_window_hours", 12)):
        return hours_since >= float(cfg.get("final_every_hours", 3))
    return hours_since >= float(cfg.get("every_hours", 20))


# --------------------------------------------------------------------------- arranque
def pick_models(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    available = openrouter_available_models()

    def ok(m: str) -> bool:
        if m.startswith(("gemini/", "gemini-search/")):
            return bool(os.getenv("GEMINI_API_KEY"))
        if m.startswith("openrouter/") and not os.getenv("OPENROUTER_API_KEY"):
            return False
        if m.startswith("openai/"):
            return bool(os.getenv("OPENAI_API_KEY")) and openai_pool(m) is not None
        if m.startswith("mistral/"):
            return bool(os.getenv("MISTRAL_API_KEY"))
        if not m.startswith("openrouter/"):
            return True
        base = strip_prefix(m).split(":online")[0]
        return available is None or base in available

    forecasters = [m for m in cfg["forecasters"] if ok(m)]
    for m in cfg.get("fallback_forecasters", []):
        if len(forecasters) >= cfg.get("min_forecasters", 4):
            break
        if ok(m) and m not in forecasters:
            forecasters.append(m)
    researchers = [r for r in cfg["researchers"] if ok(r)]
    for role in ("research_planner", "supervisor", "checker"):
        if not cfg.get(role) or not ok(cfg[role]):
            cfg[role] = forecasters[0] if forecasters else cfg.get(role)
    missing = [m for m in cfg["forecasters"] if m not in forecasters]
    if missing:
        logger.warning(f"Modelos no disponibles hoy en OpenRouter (se omiten): {missing}")
    if not forecasters or len(forecasters) * int(cfg.get("predictions_per_model", 1)) < 2:
        raise RuntimeError(f"Muy pocos modelos disponibles: {forecasters}")
    return forecasters, researchers


# --------------------------------------------------------------------------- presupuesto
USAGE_LOG = ROOT / "data" / "usage_log.jsonl"


def openrouter_usage() -> float | None:
    """Gasto total de la clave de OpenRouter (dólares), leído de la propia API de OpenRouter."""
    try:
        r = requests.get("https://openrouter.ai/api/v1/key",
                         headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}"}, timeout=30)
        r.raise_for_status()
        return float(r.json()["data"].get("usage") or 0.0)
    except Exception as e:
        logger.warning(f"No se pudo leer el gasto de OpenRouter: {e}")
        return None


def budget_state(cfg: dict[str, Any]) -> dict[str, Any]:
    """Cuánto se ha gastado, cuánto queda y cuánto se puede gastar hoy para que el presupuesto dure toda la temporada."""
    b = cfg.get("budget", {})
    used = openrouter_usage()
    now = datetime.now(timezone.utc)
    st: dict[str, Any] = {"used": used, "total": b.get("total_usd")}
    if used is None or not b.get("total_usd"):
        return st
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    start_of_day = used
    if USAGE_LOG.exists():
        for line in USAGE_LOG.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e["ts"][:10] == now.strftime("%Y-%m-%d"):
                start_of_day = min(start_of_day, float(e["used"]))
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now.isoformat(timespec="seconds"), "used": round(used, 4)}) + "\n")
    end = datetime.fromisoformat(b.get("season_end", "2027-01-20")).replace(tzinfo=timezone.utc)
    days_left = max(1.0, (end - now).total_seconds() / 86400)
    remaining = float(b["total_usd"]) - used
    st.update(remaining=remaining, days_left=days_left, daily=max(0.0, remaining) / days_left,
              spent_today=used - start_of_day)
    return st


def profile_for(cfg: dict[str, Any], mode: str, st: dict[str, Any]) -> str | None:
    if free_mode():
        if os.getenv("OPENAI_API_KEY") and "gratis_openai" in cfg.get("profiles", {}):
            return "gratis_openai"
        return "gratis"
    name = cfg.get("active_profiles", {}).get(mode, mode)
    if st.get("remaining") is not None:
        if st["remaining"] <= float(cfg.get("budget", {}).get("reserve_usd", 5)):
            return None
        slack = float(cfg.get("budget", {}).get("daily_slack", 1.5))
        if st["spent_today"] > st["daily"] * slack:
            logger.warning(f"Hoy ya se han gastado {st['spent_today']:.2f} $ (límite diario {st['daily']:.2f} $): modo ahorro")
            return "ahorro"
    return name


def build_bot(base_cfg: dict[str, Any], profile: str, publish: bool) -> ProBot:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.update(base_cfg.get("profiles", {}).get(profile, {}))
    cfg["_profile"] = profile
    resolve_gemini_models(cfg)
    forecasters, researchers = pick_models(cfg)
    logger.info(f"[{profile}] modelos: {forecasters} · investigación: {researchers}")
    bot = ProBot(
        cfg=cfg,
        forecasters=forecasters,
        researchers=researchers,
        research_reports_per_question=1,
        predictions_per_research_report=len(forecasters) * int(cfg.get("predictions_per_model", 1)),
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        enable_summarize_research=False,
        llms={
            "default": GeneralLlm(model=forecasters[0], timeout=420, allowed_tries=2),
            "summarizer": cfg["parser"],
            "researcher": "no_research",
            "parser": cfg["parser"],
        },
    )
    # al menos la mitad de los modelos tiene que responder para enviar el pronóstico
    bot.required_successful_predictions = 0.5
    if free_mode():
        bot._structure_output_validation_samples = 1
    return bot


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tournament", "minibench", "test_questions", "market_pulse"], default="tournament")
    parser.add_argument("--dry-run", action="store_true", help="No publica en Metaculus ni escribe el registro")
    args = parser.parse_args()

    if not os.getenv("METACULUS_TOKEN") or not (os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")):
        # Aún faltan claves: salir sin error
        print("Faltan METACULUS_TOKEN y alguna clave de IA (OPENROUTER_API_KEY o GEMINI_API_KEY). No se hace nada en esta pasada.")
        raise SystemExit(0)
    if free_mode():
        logger.info("Modo GRATIS: solo Gemini (clave de Google AI Studio) hasta que llegue la clave de OpenRouter")
    check_environment(strict=True)
    cfg = load_config()
    cfg["_no_log"] = args.dry_run or args.mode == "test_questions"   # el área de pruebas no cuenta para aprender
    cfg["_mode"] = args.mode
    publish = not args.dry_run
    print_startup_banner(args.mode, will_publish=publish)

    st = budget_state(cfg)
    if st.get("remaining") is not None:
        logger.info(f"Presupuesto: gastado {st['used']:.2f} $ de {st['total']} $ · quedan {st['remaining']:.2f} $ "
                    f"para {st['days_left']:.0f} días ({st['daily']:.2f} $/día) · hoy {st['spent_today']:.2f} $")

    client = MetaculusClient()
    reports: list = []
    url = "https://www.metaculus.com/tournament/fall-futureeval-2026/"
    if args.mode == "market_pulse" and free_mode():
        print("Modo gratis: Market Pulse se salta para no gastar el cupo diario de Google."); raise SystemExit(0)
    if args.mode == "market_pulse":
        prof = profile_for(cfg, "market_pulse", st)
        if prof is None:
            print("Presupuesto agotado: no se hace nada."); raise SystemExit(0)
        bot = build_bot(cfg, prof, publish)
        now = datetime.now(timezone.utc)
        qs = client.get_all_open_questions_from_tournament(cfg['tournaments']['market_pulse'])
        last = last_forecast_times()
        due = [q for q in qs if market_pulse_due(q, last.get(q.id_of_question), now, cfg.get("market_pulse", {}))]
        logger.info(f"Market Pulse: {len(qs)} preguntas abiertas, {len(due)} toca actualizar ahora")
        bot.skip_previously_forecasted_questions = False
        reports += asyncio.run(bot.forecast_questions(due, return_exceptions=True)) if due else []
        url = f"https://www.metaculus.com/tournament/{cfg['tournaments']['market_pulse']}/"
    elif args.mode in ("tournament", "minibench"):
        plan = ([("tournament", cfg['tournaments']['main'])] if args.mode == "tournament" else []) + \
               [("minibench", cfg['tournaments']['minibench'])]
        if not cfg.get("minibench_enabled", True):  # decisión: todos los créditos al torneo principal
            plan = [p for p in plan if p[0] != "minibench"]
            if not plan:
                print("MiniBench desactivado en la configuración."); raise SystemExit(0)
        if free_mode():  # el cupo gratis de Google da para pocas preguntas al día: todo al torneo principal
            plan = [p for p in plan if p[0] == "tournament"]
            if not plan:
                print("Modo gratis: MiniBench se salta para no gastar el cupo diario de Google."); raise SystemExit(0)
        for mode_name, tid in plan:
            prof = profile_for(cfg, mode_name, st)
            if prof is None:
                print("Presupuesto agotado: no se hace nada."); break
            bot = build_bot(cfg, prof, publish)
            reports += asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
            pc = cfg.get("preclose_update") or {}
            if mode_name == "tournament" and pc.get("enabled") and not free_mode():
                try:
                    now = datetime.now(timezone.utc)
                    last = last_forecast_times()
                    qs = client.get_all_open_questions_from_tournament(tid)
                    due = [q for q in qs if preclose_due(q, last.get(q.id_of_question), now, pc)]
                    if due:
                        logger.info(f"Actualización antes del cierre: {len(due)} preguntas")
                        bot.skip_previously_forecasted_questions = False
                        reports += asyncio.run(bot.forecast_questions(due, return_exceptions=True))
                except Exception as e:
                    logger.warning(f"No se pudo hacer la actualización antes del cierre: {e}")
    else:
        bot = build_bot(cfg, "gratis" if free_mode() else cfg.get("active_profiles", {}).get("tournament", "tournament"), publish)
        bot.skip_previously_forecasted_questions = False
        reports += asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        url = "https://www.metaculus.com/tournament/bot-testing-area/"

    if reports:
        bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=url)
