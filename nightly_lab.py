"""
nightly_lab.py - "Laboratorio" nocturno: usa los tokens gratis de OpenAI que han sobrado del día (se pierden a
medianoche UTC) para pronosticar preguntas YA RESUELTAS de torneos de bots anteriores, sin publicar nada.

- Sin investigación (las búsquedas de hoy revelarían el resultado) y con "hoy" = día de cierre de cada pregunta.
- Cada noche toma preguntas que aún no ha probado, así que la muestra crece día a día (data/lab.jsonl).
- Compara GPT-5.4 con GPT-5.4-mini y el conjunto, y calcula la calibración que sugieren los datos.
- No cambia la configuración del bot: el informe (data/lab_report.md) se revisa cada lunes antes de tocar nada.
- Deja siempre un margen de tokens para el torneo y nunca pasa de la parte gratis (el contador de bot_pro manda).

Uso: python nightly_lab.py --max 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

# Sin modelos de reserva: si un modelo no responde, se apunta como fallo en vez de mezclar resultados de otro modelo
os.environ["LAB_NO_FALLBACK"] = "1"
# Margen para el torneo: el laboratorio se para a los 160.000 tokens de GPT-5.4 (el torneo puede llegar a 200.000)
os.environ.setdefault("OPENAI_BIG_DAILY", "160000")
os.environ.setdefault("OPENAI_SMALL_DAILY", "1900000")

from bot_helpers import silence_noisy_dependencies  # noqa: E402

silence_noisy_dependencies()
from forecasting_tools import ApiFilter, BinaryQuestion, GeneralLlm, MetaculusClient  # noqa: E402

import bot_pro  # noqa: E402
from learn import brier, fit_platt, logit, logloss  # noqa: E402

ROOT = Path(__file__).resolve().parent
LAB = ROOT / "data" / "lab.jsonl"
REPORT = ROOT / "data" / "lab_report.md"
TOKENS_PER_BIG_CALL = 20000      # estimación prudente para GPT-5.4 con razonamiento alto y sin investigación
# La API de Metaculus no enseña la resolución (ni los criterios) de las preguntas ya resueltas a las cuentas de bot,
# así que la fuente principal son mercados YA RESUELTOS de Manifold con bastante participación (sí/no, con su
# descripción como criterio). Metaculus queda como reserva por si algún día vuelve a mostrar las resoluciones.
TOURNAMENTS: list = []
MAX_LOOKUPS = 60
MIN_BETTORS = 25
MIN_VOLUME = 3000


def manifold_questions(since: datetime, n: int, seen: set[int]) -> list:
    import hashlib
    import re
    import time
    import requests
    try:
        r = requests.get("https://api.manifold.markets/v0/search-markets",
                         params={"term": "", "filter": "resolved", "contractType": "BINARY", "sort": "resolve-date",
                                 "limit": 1000}, timeout=60)
        r.raise_for_status()
        markets = r.json()
    except Exception as e:
        print(f"No se pudieron leer los mercados de Manifold: {e}")
        return []
    personal = re.compile(r"(?i)^\s*(will|do|did|am|should|can|would|have)\s+i\b|\bmy\b")
    out = []
    for m in markets:
        if len(out) >= n:
            break
        qid = int(hashlib.md5(m["id"].encode()).hexdigest()[:8], 16)
        if (m.get("resolution") not in ("YES", "NO") or qid in seen
                or (m.get("uniqueBettorCount") or 0) < MIN_BETTORS or (m.get("volume") or 0) < MIN_VOLUME
                or (m.get("resolutionTime") or 0) / 1000 < since.timestamp() or personal.search(m.get("question", ""))):
            continue
        desc = ""
        try:
            time.sleep(0.5)
            d = requests.get(f"https://api.manifold.markets/v0/market/{m['id']}", timeout=30).json()
            desc = (d.get("textDescription") or "").strip()
        except Exception:
            pass
        close_ms = min(m.get("closeTime") or m["resolutionTime"], m["resolutionTime"])
        q = BinaryQuestion(question_text=m["question"], id_of_post=qid, id_of_question=qid, page_url=m.get("url", ""),
                           close_time=datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc),
                           resolution_criteria=(desc[:3000] or m["question"]), fine_print="", background_info="")
        q.resolution_string = "yes" if m["resolution"] == "YES" else "no"
        out.append(q)
    print(f"Mercados resueltos de Manifold elegidos: {len(out)} (de {len(markets)} revisados)")
    return out


_DEBUG_LEFT = [3]


def resolution_of(q, debug: bool = False) -> int | None:
    """1 = sí, 0 = no, None = sin resolver/anulada. Si forecasting-tools no trae la resolución, la pide a la API."""
    r = str(q.resolution_string or "").strip().lower()
    if r in ("yes", "no"):
        return 1 if r == "yes" else 0
    import time
    import requests
    qq = None
    for attempt in range(3):
        time.sleep(1.5)   # la API corta si se le pregunta demasiado deprisa
        try:
            resp = requests.get(f"https://www.metaculus.com/api/posts/{q.id_of_post}/",
                                headers={"Authorization": f"Token {os.getenv('METACULUS_TOKEN', '')}"}, timeout=30)
            if resp.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            qq = resp.json().get("question") or {}
            break
        except Exception as e:
            print(f"No se pudo leer la pregunta {q.id_of_post}: {e}")
            return None
    if qq is None:
        return None
    if _DEBUG_LEFT[0] > 0:
        _DEBUG_LEFT[0] -= 1
        print("DEBUG", q.id_of_post, {k: qq.get(k) for k in qq if "resol" in k.lower() or k in ("status", "outcome")})
    for k in ("resolution", "resolution_value", "resolved_value", "outcome"):
        v = qq.get(k)
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, str) and v.strip().lower() in ("yes", "no"):
            return 1 if v.strip().lower() == "yes" else 0
    return None


def done_ids() -> set[int]:
    if not LAB.exists():
        return set()
    out = set()
    for line in LAB.read_text(encoding="utf-8").splitlines():
        try:
            out.add(int(json.loads(line)["qid"]))
        except Exception:
            continue
    return out


def write_report() -> None:
    rows = []
    if LAB.exists():
        for line in LAB.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    n = len(rows)
    lines = [f"# Laboratorio nocturno ({datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC)", "",
             f"- Preguntas resueltas probadas en total: **{n}** (sin investigación, como si fuera el día de cierre)"]
    if n:
        lines.append(f"- Proporción de SÍ: {sum(r['y'] for r in rows) / n:.0%}")
        ll = sum(logloss(r["final"], r["y"]) for r in rows) / n
        br = sum(brier(r["final"], r["y"]) for r in rows) / n
        lines.append(f"- Conjunto final: pérdida log **{ll:.3f}** · Brier {br:.3f} (decir siempre 50 % = 0,693 / 0,250)")
        per: dict[str, list[float]] = {}
        for r in rows:
            for m in r.get("models", []):
                if isinstance(m.get("p"), (int, float)):
                    per.setdefault(m["model"], []).append(logloss(m["p"], r["y"]))
        lines += ["", "| Modelo | n | Pérdida log (menos es mejor) |", "|---|---|---|"]
        for m, v in sorted(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            lines.append(f"| {m} | {len(v)} | {sum(v) / len(v):.3f} |")
        if n >= 20:
            a, b = fit_platt([logit(r["final"]) for r in rows], [r["y"] for r in rows])
            lines.append(f"\n- Calibración que sugieren los datos: a = {a:+.3f}, b = {b:.3f} "
                         f"(b > 1: el bot es demasiado prudente; b < 1: demasiado extremo). Solo orientativa.")
        else:
            lines.append(f"\n- Calibración: hacen falta 20 preguntas (hay {n}).")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10, help="Máximo de preguntas por noche")
    ap.add_argument("--since", default="2026-06-01")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY") or not os.getenv("METACULUS_TOKEN"):
        print("Faltan OPENAI_API_KEY o METACULUS_TOKEN: no se hace nada.")
        return

    usage = bot_pro.openai_usage()
    left = bot_pro.OPENAI_DAILY["big"] - int(usage.get("big", 0))
    n_q = min(args.max, max(0, left // TOKENS_PER_BIG_CALL))
    print(f"Tokens grandes usados hoy: {usage.get('big', 0)} · margen del laboratorio: {left} · preguntas esta noche: {n_q}")
    if n_q < 1:
        write_report()
        return

    cfg = bot_pro.load_config()
    cfg.update(cfg.get("profiles", {}).get("gratis_openai", {}))
    cfg.update({"_no_log": True, "_as_of_close": True, "research_rounds": 0, "followups": 0,
                "read_resolution_sources": False, "market_lookup": False, "free_sources": False,
                "related_metaculus": False, "crux_research": False, "supervisor_spread": 99,
                "research_planner": "openai/gpt-5.4-mini", "parser": "gemini/auto-lite",
                "predictions_per_model": 1})
    bot_pro.resolve_gemini_models(cfg)   # el lector de respuestas usa Gemma (gratis), no tokens de OpenAI
    forecasters = ["openai/gpt-5.4", "openai/gpt-5.4-mini"]

    seen = done_ids()
    pool = []
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    lookups = 0
    pool += manifold_questions(since, n_q, seen)
    for t in TOURNAMENTS:
        if len(pool) >= n_q or lookups >= MAX_LOOKUPS:
            break
        tour = int(t) if str(t).isdigit() else t
        flt = ApiFilter(allowed_statuses=["resolved"], allowed_types=["binary"],
                        allowed_tournaments=[tour] if tour else None, scheduled_resolve_time_gt=since)
        try:
            qs = asyncio.run(MetaculusClient().get_questions_matching_filter(
                flt, num_questions=200, randomly_sample=True, error_if_question_target_missed=False))
        except Exception as e:
            print(f"No se pudieron leer las preguntas del torneo {t}: {e}")
            continue
        from collections import Counter
        print("Resoluciones según forecasting-tools:", Counter(str(q.resolution_string) for q in qs).most_common(6))
        for q in qs:
            if len(pool) >= n_q or lookups >= MAX_LOOKUPS:
                break
            if not isinstance(q, BinaryQuestion) or q.id_of_question in seen:
                continue
            if str(q.resolution_string or "").lower() not in ("yes", "no"):
                lookups += 1
            y = resolution_of(q, debug=len(pool) == 0)
            if y is not None:
                q.resolution_string = "yes" if y else "no"
                pool.append(q)
    qs = pool[:n_q]
    if not qs:
        print("No quedan preguntas resueltas nuevas para probar.")
        write_report()
        return
    answers = {q.id_of_question: (1 if q.resolution_string == "yes" else 0) for q in qs}
    for q in qs:  # que el bot no vea la resolución
        q.resolution_string = None
    print(f"Preguntas para el laboratorio: {len(qs)}")

    bot = bot_pro.ProBot(
        cfg=cfg, forecasters=forecasters, researchers=[],
        research_reports_per_question=1, predictions_per_research_report=len(forecasters),
        publish_reports_to_metaculus=False, enable_summarize_research=False,
        llms={"default": GeneralLlm(model=cfg["parser"]), "summarizer": cfg["parser"],
              "researcher": "no_research", "parser": cfg["parser"]},
    )
    reports = asyncio.run(bot.forecast_questions(qs, return_exceptions=True))

    LAB.parent.mkdir(exist_ok=True)
    added = 0
    with open(LAB, "a", encoding="utf-8") as f:
        for q, rep in zip(qs, reports):
            if isinstance(rep, Exception):
                continue
            key = bot._qkey(q)
            f.write(json.dumps({"qid": q.id_of_question, "title": q.question_text[:150], "y": answers[q.id_of_question],
                                "close": q.close_time.isoformat() if q.close_time else None,
                                "models": bot._individual.get(key, []), "final": float(rep.prediction),
                                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                               ensure_ascii=False) + "\n")
            added += 1
    print(f"Añadidas {added} preguntas al laboratorio.")
    write_report()


if __name__ == "__main__":
    main()
