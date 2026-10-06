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


class RobustLlm:
    """Llama al modelo con razonamiento alto; si el proveedor no lo admite, repite sin él.
    Los modelos 'gemini/...' van directos a Google (clave gratuita de AI Studio); si Google
    no responde (p. ej. límite de uso gratis), se usa el mismo tipo de modelo por OpenRouter."""

    def __init__(self, model: str, reasoning: str | None = "high", timeout: int = 420, search: bool = False):
        self.model = model
        extra: dict[str, Any] = {"tools": [{"googleSearch": {}}]} if search else {}
        if model.startswith("gemini/"):
            self._with_reasoning = (GeneralLlm(model=model, timeout=timeout, allowed_tries=2, reasoning_effort=reasoning, **extra)
                                    if reasoning else None)
            self._plain = GeneralLlm(model=model, timeout=timeout, allowed_tries=2, **extra)
            self._fallback: RobustLlm | None = None if search else RobustLlm(GEMINI_FALLBACK, reasoning, timeout)
        else:
            self._with_reasoning = (
                GeneralLlm(model=model, timeout=timeout, allowed_tries=2,
                           extra_body={"reasoning": {"effort": reasoning}})
                if reasoning else None
            )
            self._plain = GeneralLlm(model=model, timeout=timeout, allowed_tries=3)
            self._fallback = None

    async def invoke(self, prompt: str) -> str:
        if self._with_reasoning is not None:
            try:
                out = await self._with_reasoning.invoke(prompt)
                if out and out.strip():
                    return out
            except Exception as e:
                logger.warning(f"{self.model}: fallo con razonamiento alto ({e}); repito sin él")
        try:
            return await self._plain.invoke(prompt)
        except Exception as e:
            if self._fallback is None:
                raise
            logger.warning(f"{self.model}: Google no responde ({e}); uso {self._fallback.model}")
            return await self._fallback.invoke(prompt)


def resolve_gemini_models(cfg: dict[str, Any]) -> None:
    """Sustituye 'gemini/auto-flash' por el mejor modelo Flash gratuito disponible con GEMINI_API_KEY.
    Sin clave (o si falla la consulta) se usa el equivalente por OpenRouter."""
    key = os.getenv("GEMINI_API_KEY")
    best = None
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
        except Exception as e:
            logger.warning(f"No se pudo consultar la lista de modelos de Gemini: {e}")
    logger.info(f"Gemini gratis: {'gemini/' + best if best else 'no disponible, se usa OpenRouter'}")

    def sub(m: str) -> str | None:
        if m == "gemini/auto-flash":
            return f"gemini/{best}" if best else GEMINI_FALLBACK
        if m == "gemini-search/auto-flash":
            return f"gemini-search/{best}" if best else None
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
    for k in ("research_planner", "supervisor", "followup_researcher"):
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
                    if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                        q = question.question_text if r == "asknews/news-summaries" else prompt
                        tasks.append((r, AskNewsSearcher().call_preconfigured_version(r, q)))
                elif r.startswith("gemini-search/"):
                    tasks.append((f"Google Search ({r.split('/', 1)[1]})",
                                  RobustLlm("gemini/" + r.split("/", 1)[1], reasoning=None, timeout=300, search=True).invoke(prompt)))
                else:
                    tasks.append((r, RobustLlm(r, reasoning=None, timeout=int(self.cfg.get("research_timeout", 600))).invoke(prompt)))
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
                for fq in followups:
                    for r in web_f:
                        tasks2.append((f"{r} · seguimiento: {fq[:80]}",
                                       RobustLlm(r, reasoning=None, timeout=300).invoke(self._followup_prompt(question, fq))))
                    if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
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
            4. Base rates: how often similar events happened historically (reference classes with numbers).
            5. What prediction markets, polls or expert forecasts say, if anything.
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
            final = self._calibrate_binary(raw)
            self._write_log(question, recs, raw=raw, final=final)
            logger.info(f"{question.page_url}: modelos {[round(r['p'], 3) for r in recs]} -> bruto {raw:.3f} -> final {final:.3f}")
            return final
        if isinstance(question, (NumericQuestion, DateQuestion)) and self.cfg.get("numeric_mixture", True) and len(predictions) > 1:
            try:
                aggregate = self._mixture_numeric(predictions, question, self._quant.get(key),
                                                  float(self.cfg.get("quant_weight", 2.0)))
                self._write_log(question, recs, final=None)
                return aggregate
            except Exception as e:
                logger.warning(f"Mezcla numérica falló ({e}); uso la mediana de la plantilla")
        aggregate = await super()._aggregate_predictions(predictions, question)
        if isinstance(question, MultipleChoiceQuestion) and isinstance(aggregate, PredictedOptionList):
            floor = float(self.cfg.get("mc_floor", 0.01))
            opts = aggregate.predicted_options
            probs = [max(o.probability, floor) for o in opts]
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
    def _mixture_numeric(predictions, question, quant: dict[str, Any] | None = None, quant_weight: float = 2.0):
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
        heights = np.average(np.array([[pt.percentile for pt in c] for c in cdfs]), axis=0, weights=weights).tolist()
        mixed = [Percentile(value=x, percentile=h) for x, h in zip(xs, heights)]
        return NumericDistribution.from_question(mixed, question)

    async def _supervise_binary(self, question: BinaryQuestion) -> float | None:
        key = self._qkey(question)
        views = self._reasons.get(key, [])
        research = self._research.get(key, "")
        blocks = "\n\n".join(f"--- Forecaster {i + 1} said {p:.0%} ---\n{txt}" for i, (_, p, txt) in enumerate(views))
        today = self._today(question)
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
    for role in ("research_planner", "supervisor"):
        if not cfg.get(role) or not ok(cfg[role]):
            cfg[role] = forecasters[0] if forecasters else cfg.get(role)
    missing = [m for m in cfg["forecasters"] if m not in forecasters]
    if missing:
        logger.warning(f"Modelos no disponibles hoy en OpenRouter (se omiten): {missing}")
    if len(forecasters) < 2:
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
    return bot


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tournament", "minibench", "test_questions", "market_pulse"], default="tournament")
    parser.add_argument("--dry-run", action="store_true", help="No publica en Metaculus ni escribe el registro")
    args = parser.parse_args()

    if not os.getenv("METACULUS_TOKEN") or not os.getenv("OPENROUTER_API_KEY"):
        # Aún faltan claves (p. ej. los créditos de OpenRouter no han llegado): salir sin error
        print("Faltan METACULUS_TOKEN u OPENROUTER_API_KEY en los Secrets de GitHub. No se hace nada en esta pasada.")
        raise SystemExit(0)
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
    if args.mode == "market_pulse":
        prof = profile_for(cfg, "market_pulse", st)
        if prof is None:
            print("Presupuesto agotado: no se hace nada."); raise SystemExit(0)
        bot = build_bot(cfg, prof, publish)
        now = datetime.now(timezone.utc)
        qs = client.get_all_open_questions_from_tournament(client.CURRENT_MARKET_PULSE_ID)
        last = last_forecast_times()
        due = [q for q in qs if market_pulse_due(q, last.get(q.id_of_question), now, cfg.get("market_pulse", {}))]
        logger.info(f"Market Pulse: {len(qs)} preguntas abiertas, {len(due)} toca actualizar ahora")
        bot.skip_previously_forecasted_questions = False
        reports += asyncio.run(bot.forecast_questions(due, return_exceptions=True)) if due else []
        url = f"https://www.metaculus.com/tournament/{client.CURRENT_MARKET_PULSE_ID}/"
    elif args.mode in ("tournament", "minibench"):
        plan = ([("tournament", client.CURRENT_AI_COMPETITION_ID)] if args.mode == "tournament" else []) + \
               [("minibench", client.CURRENT_MINIBENCH_ID)]
        for mode_name, tid in plan:
            prof = profile_for(cfg, mode_name, st)
            if prof is None:
                print("Presupuesto agotado: no se hace nada."); break
            bot = build_bot(cfg, prof, publish)
            reports += asyncio.run(bot.forecast_on_tournament(tid, return_exceptions=True))
    else:
        bot = build_bot(cfg, cfg.get("active_profiles", {}).get("tournament", "tournament"), publish)
        bot.skip_previously_forecasted_questions = False
        reports += asyncio.run(bot.forecast_on_tournament("bot-testing-area", return_exceptions=True))
        url = "https://www.metaculus.com/tournament/bot-testing-area/"

    if reports:
        bot.log_report_summary(reports)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=url)
