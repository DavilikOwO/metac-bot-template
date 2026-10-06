"""
learn_from_bots.py - El bot aprende de los demás bots del torneo (se ejecuta cada día, después de learn.py).

Cuando una pregunta CIERRA, Metaculus muestra:
  - la predicción agregada de todos los bots (la "comunidad" del torneo de bots), y
  - los comentarios con el razonamiento de cada bot.
Este script, sin tocar ningún pronóstico ya enviado:
  1. Compara nuestro pronóstico con el de la comunidad de bots en cada pregunta sí/no cerrada
     (y, si ya se resolvió, quién acertó más) -> data/crowd.json y data/crowd_report.md
  2. En las preguntas donde más nos separamos de los demás o donde ellos acertaron y nosotros no,
     lee los comentarios de los bots que más acertaron y saca lecciones GENERALES (qué fuentes usaron,
     qué razonamiento nos faltó) -> data/lessons.json
  3. bot_pro.py añade las lecciones más recientes a la investigación de cada pregunta nueva.
Así el bot mejora solo con lo que hacen los mejores, sin intervención humana.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "data" / "forecast_log.jsonl"
CROWD_PATH = ROOT / "data" / "crowd.json"
LESSONS_PATH = ROOT / "data" / "lessons.json"
REPORT_PATH = ROOT / "data" / "crowd_report.md"
API = os.getenv("METACULUS_API_BASE_URL", "https://www.metaculus.com/api")
MAX_LESSON_QUESTIONS = 6      # preguntas analizadas por día (para gastar poco)
MAX_LESSONS = 14              # lecciones que se guardan (las más útiles/recientes)

logger = logging.getLogger("learn_from_bots")


def clip(p: float, e: float = 0.01) -> float:
    return min(max(p, e), 1 - e)


def logit(p: float) -> float:
    p = clip(p, 1e-4)
    return math.log(p / (1 - p))


def logloss(p: float, y: int) -> float:
    p = clip(p)
    return -math.log(p if y else 1 - p)


CATS = [("elecciones", r"election|vote|president|primary|parliament|minister|governor|senate|congress|party|poll|seats"),
        ("economía", r"gdp|inflation|cpi|interest rate|fed\b|treasury|yield|unemployment|oil|tariff|jobs|s&p|nasdaq|vix|gold|price|stock|market"),
        ("ia_tecnología", r"\bai\b|openai|anthropic|model|llm|nvidia|google|apple|microsoft|tesla|spacex|chip"),
        ("geopolítica", r"russia|ukraine|israel|iran|china|taiwan|gaza|war|ceasefire|nato|missile|strike|sanction"),
        ("salud_clima", r"virus|ebola|covid|measles|outbreak|hurricane|temperature|earthquake|weather|cases")]


def cat_of(t: str) -> str:
    for k, rx in CATS:
        if re.search(rx, t or "", re.I):
            return k
    return "otros"


def headers() -> dict:
    return {"Authorization": f"Token {os.getenv('METACULUS_TOKEN', '')}", "Accept-Language": "en"}


def api_get(path: str, params: dict | None = None):
    r = requests.get(f"{API}{path}", params=params, headers=headers(), timeout=40)
    r.raise_for_status()
    return r.json()


def read_log() -> dict[int, dict]:
    latest: dict[int, dict] = {}
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("post_id") is not None:
                latest[int(e["post_id"])] = e
    return latest


def crowd_of(post: dict) -> float | None:
    """Predicción agregada (comunidad de bots) de una pregunta sí/no."""
    try:
        agg = post["question"]["aggregations"]
        for k in ("recency_weighted", "unweighted"):
            latest = (agg.get(k) or {}).get("latest") or {}
            c = latest.get("centers")
            if c:
                return float(c[0])
    except Exception:
        return None
    return None


def resolution_of(post: dict) -> int | None:
    r = str((post.get("question") or {}).get("resolution") or "").lower()
    return 1 if r == "yes" else 0 if r == "no" else None


def my_user() -> tuple[int | None, str | None]:
    try:
        j = api_get("/users/me")
        return j.get("id"), j.get("username")
    except Exception as e:
        logger.warning(f"No se pudo leer el usuario del bot: {e}")
        return None, None


def comments_of(post_id: int) -> list[dict]:
    for key in ("post", "on_post"):
        try:
            j = api_get("/comments/", {key: post_id, "limit": 100, "sort": "-created_at", "is_private": "false"})
            res = j.get("results", j) if isinstance(j, dict) else j
            if isinstance(res, list):
                return res
        except Exception as e:
            logger.info(f"Comentarios de {post_id} con '{key}': {e}")
    return []


def last_pct(text: str) -> float | None:
    m = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text or "")
    try:
        return float(m[-1]) / 100 if m else None
    except ValueError:
        return None


def best_comments(cs: list[dict], me_id, me_name, target: float, k: int = 4) -> list[tuple[str, float | None, str]]:
    """Comentarios de otros bots, primero los que más se acercaron al resultado (o a la comunidad si aún no se sabe)."""
    seen, rows = set(), []
    for c in cs:
        a = c.get("author") or {}
        who = a.get("username") or str(a.get("id") or c.get("author_id") or "?")
        if (me_id and a.get("id") == me_id) or (me_name and who == me_name) or who in seen:
            continue
        text = str(c.get("text") or "")
        if len(text) < 200:
            continue
        seen.add(who)
        p = last_pct(text[-1500:])
        rows.append((abs(p - target) if p is not None else 0.5, who, p, text))
    rows.sort(key=lambda r: r[0])
    return [(w, p, t) for _, w, p, t in rows[:k]]


def llm():
    import bot_pro  # import tardío: trae la configuración y los modelos

    cfg = bot_pro.load_config()
    small = {"forecasters": [cfg.get("research_planner", "gemini/auto-flash")], "researchers": []}
    if bot_pro.free_mode():
        small["forecasters"] = ["gemini/auto-flash"]
    bot_pro.resolve_gemini_models(small)
    return bot_pro.RobustLlm(small["forecasters"][0], reasoning="medium", timeout=300)


LESSON_PROMPT = """You are improving an AI forecasting bot that competes against ~220 other bots on Metaculus.
On this question our bot's forecast differed from what the other bots did, and the other bots were {who_was_right}.

Question: {title}
Our probability: {ours:.0%}. Aggregate of all bots: {crowd:.0%}. Outcome: {outcome}.

Reasoning written by the bots that did best on this question:
{comments}

Extract what the better bots did that our bot should do on FUTURE, DIFFERENT questions:
reasoning habits, base rates, how they read the resolution criteria, which kinds of sources they checked, calibration.
Rules: each lesson must be GENERAL (no facts specific to this question), actionable, at most 30 words, in English.
If there is nothing useful to learn, return empty lists.
Answer ONLY with JSON: {{"lessons": ["...", "..."], "sources": ["kinds of sources worth checking"]}}"""


MERGE_PROMPT = """These are lessons an AI forecasting bot learned from stronger bots. Merge duplicates and keep the {n} most useful,
general and actionable ones (at most 30 words each, English). Return ONLY a JSON list of strings.
{items}"""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not os.getenv("METACULUS_TOKEN"):
        print("Sin METACULUS_TOKEN: no se hace nada.")
        return
    now = datetime.now(timezone.utc)
    latest = read_log()
    crowd: dict[str, dict] = json.loads(CROWD_PATH.read_text(encoding="utf-8")) if CROWD_PATH.exists() else {}
    lessons: list[dict] = json.loads(LESSONS_PATH.read_text(encoding="utf-8")) if LESSONS_PATH.exists() else []
    me_id, me_name = my_user()

    # 1) comunidad de bots en las preguntas sí/no ya cerradas
    for pid, e in latest.items():
        if e.get("type") != "BinaryQuestion" or e.get("final") is None:
            continue
        rec = crowd.get(str(pid), {})
        if rec.get("resolution") is not None:
            continue
        ct = e.get("close_time")
        try:
            if ct and datetime.fromisoformat(ct.replace("Z", "+00:00")) > now:
                continue          # aún abierta: la media de los bots no se ve
        except Exception:
            pass
        try:
            post = api_get(f"/posts/{pid}/")
        except Exception as ex:
            logger.warning(f"No se pudo leer el post {pid}: {ex}")
            continue
        c = crowd_of(post)
        if c is None:
            continue
        rec.update(title=e.get("title"), url=e.get("url"), ours=float(e["final"]), crowd=c,
                   resolution=resolution_of(post), cat=cat_of(e.get("title", "")), checked=now.isoformat(timespec="seconds"))
        crowd[str(pid)] = rec

    # 2) lecciones a partir de los comentarios de los bots que mejor lo hicieron
    def gap(r: dict) -> float:
        g = abs(logit(r["ours"]) - logit(r["crowd"]))
        if r.get("resolution") is not None:
            g += max(0.0, logloss(r["ours"], r["resolution"]) - logloss(r["crowd"], r["resolution"])) * 2
        return g

    todo = [(pid, r) for pid, r in crowd.items() if not r.get("lesson_done") and gap(r) >= 0.8]
    todo.sort(key=lambda kv: -gap(kv[1]))
    model = None
    n_comments = 0
    for pid, r in todo[:MAX_LESSON_QUESTIONS]:
        y = r.get("resolution")
        target = float(y) if y is not None else r["crowd"]
        if y is not None and logloss(r["ours"], y) <= logloss(r["crowd"], y):
            r["lesson_done"] = True          # acertamos más que la media: no hay nada que copiar
            continue
        cs = best_comments(comments_of(int(pid)), me_id, me_name, target)
        n_comments += len(cs)
        if not cs:
            r["lesson_done"] = y is not None   # si aún no hay comentarios, se reintenta otro día
            continue
        model = model or llm()
        block = "\n\n".join(f"--- {w} (said {p:.0%}) ---\n{t[:2000]}" if p is not None else f"--- {w} ---\n{t[:2000]}"
                            for w, p, t in cs)
        try:
            out = await model.invoke(LESSON_PROMPT.format(
                who_was_right="closer to the actual outcome" if y is not None else "in strong disagreement with us",
                title=r.get("title", ""), ours=r["ours"], crowd=r["crowd"],
                outcome={1: "YES", 0: "NO"}.get(y, "not resolved yet"), comments=block))
            m = re.search(r"\{.*\}", out or "", re.S)
            j = json.loads(m.group(0)) if m else {}
        except Exception as ex:
            logger.warning(f"No se pudieron sacar lecciones de {pid}: {ex}")
            continue
        for les in (j.get("lessons") or [])[:2]:
            les = str(les).strip()
            if 15 <= len(les) <= 260:
                lessons.append({"ts": now.isoformat(timespec="seconds"), "lesson": les, "from": r.get("url"), "cat": r.get("cat")})
        for src in (j.get("sources") or [])[:2]:
            src = str(src).strip()
            if 5 <= len(src) <= 160:
                lessons.append({"ts": now.isoformat(timespec="seconds"), "lesson": f"Useful kind of source: {src}",
                                "from": r.get("url"), "cat": r.get("cat")})
        r["lesson_done"] = True

    # 3) mantener la lista corta y sin repeticiones
    if len(lessons) > MAX_LESSONS + 6:
        try:
            model = model or llm()
            items = "\n".join(f"- {x['lesson']}" for x in lessons[-40:])
            out = await model.invoke(MERGE_PROMPT.format(n=MAX_LESSONS, items=items))
            m = re.search(r"\[.*\]", out or "", re.S)
            merged = [str(s).strip() for s in json.loads(m.group(0))] if m else []
            if merged:
                lessons = [{"ts": now.isoformat(timespec="seconds"), "lesson": s, "from": "merge", "cat": None}
                           for s in merged[:MAX_LESSONS] if 15 <= len(s) <= 260]
        except Exception as ex:
            logger.warning(f"No se pudieron fusionar las lecciones: {ex}")
            lessons = lessons[-MAX_LESSONS:]

    CROWD_PATH.parent.mkdir(exist_ok=True)
    CROWD_PATH.write_text(json.dumps(crowd, indent=1, ensure_ascii=False), encoding="utf-8")
    LESSONS_PATH.write_text(json.dumps(lessons, indent=1, ensure_ascii=False), encoding="utf-8")

    # 4) informe
    rows = list(crowd.values())
    res = [r for r in rows if r.get("resolution") is not None]
    lines = [f"# Comparación con los demás bots ({now:%d/%m/%Y %H:%M} UTC)", "",
             f"- Preguntas sí/no cerradas comparadas: {len(rows)} · ya resueltas: {len(res)}"]
    if rows:
        lines.append(f"- Separación media respecto a la media de los bots: {sum(abs(r['ours'] - r['crowd']) for r in rows) / len(rows):.0%}")
    if res:
        ours = sum(logloss(r["ours"], r["resolution"]) for r in res) / len(res)
        them = sum(logloss(r["crowd"], r["resolution"]) for r in res) / len(res)
        lines.append(f"- Pérdida logarítmica: nosotros {ours:.3f} · media de los bots {them:.3f} "
                     f"({'mejor que la media' if ours < them else 'peor que la media'})")
        by = defaultdict(list)
        for r in res:
            by[r.get("cat", "otros")].append(logloss(r["ours"], r["resolution"]) - logloss(r["crowd"], r["resolution"]))
        lines += ["", "| Tema | Resueltas | Diferencia con la media (negativo = mejor) |", "|---|---|---|"]
        for k, v in sorted(by.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            lines.append(f"| {k} | {len(v)} | {sum(v) / len(v):+.3f} |")
    lines += ["", f"- Comentarios de otros bots leídos hoy: {n_comments}", "", "## Lecciones que el bot usa ahora", ""]
    lines += [f"- {x['lesson']}" for x in lessons[-MAX_LESSONS:]] or ["- (todavía ninguna)"]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
