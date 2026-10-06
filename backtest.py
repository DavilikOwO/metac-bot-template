"""
backtest.py - Prueba el bot con preguntas YA RESUELTAS de torneos de bots anteriores, sin publicar nada.

- Por defecto, SIN investigación (las búsquedas de hoy revelarían el resultado) y con "hoy" = día de cierre
  de cada pregunta. Mide sobre todo la calibración de los modelos y cuál acierta más.
- Escribe data/backtest.jsonl y data/backtest_report.md. NO cambia la configuración: los números
  se revisan antes de tocar nada (sin investigación los modelos son más prudentes que en directo).

Uso: python backtest.py --n 60 --tournament 33022 --since 2026-08-15
Coste aproximado: 5 modelos x n preguntas (~0,15-0,30 $ por pregunta).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

from bot_helpers import silence_noisy_dependencies

silence_noisy_dependencies()
from forecasting_tools import ApiFilter, BinaryQuestion, GeneralLlm, MetaculusClient  # noqa: E402

import bot_pro  # noqa: E402
from learn import fit_platt, logit, logloss, brier  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tournament", default="33022")  # FutureEval verano 2026
    ap.add_argument("--since", default="2026-08-15")
    ap.add_argument("--with-research", action="store_true", help="Usa investigación (¡riesgo de ver el resultado!)")
    args = ap.parse_args()

    cfg = bot_pro.load_config()
    cfg["_no_log"] = True
    cfg["_as_of_close"] = True
    if not args.with_research:
        cfg["research_rounds"] = 0
    forecasters, researchers = bot_pro.pick_models(cfg)
    if not args.with_research:
        researchers = []

    tour = int(args.tournament) if str(args.tournament).isdigit() else args.tournament
    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    flt = ApiFilter(allowed_statuses=["resolved"], allowed_types=["binary"], allowed_tournaments=[tour],
                    scheduled_resolve_time_gt=since)
    qs = asyncio.run(MetaculusClient().get_questions_matching_filter(
        flt, num_questions=args.n, randomly_sample=True, error_if_question_target_missed=False))
    qs = [q for q in qs if isinstance(q, BinaryQuestion) and q.resolution_string in ("yes", "no")]
    answers = {q.id_of_question: (1 if q.resolution_string == "yes" else 0) for q in qs}
    for q in qs:  # que el bot no vea la resolución
        q.resolution_string = None
    print(f"Preguntas resueltas para la prueba: {len(qs)}")

    bot = bot_pro.ProBot(
        cfg=cfg, forecasters=forecasters, researchers=researchers,
        research_reports_per_question=1, predictions_per_research_report=len(forecasters),
        publish_reports_to_metaculus=False, enable_summarize_research=False,
        llms={"default": GeneralLlm(model=forecasters[0]), "summarizer": cfg["parser"],
              "researcher": "no_research", "parser": cfg["parser"]},
    )
    reports = asyncio.run(bot.forecast_questions(qs, return_exceptions=True))

    rows = []
    for q, rep in zip(qs, reports):
        if isinstance(rep, Exception):
            continue
        key = bot._qkey(q)
        rows.append({"qid": q.id_of_question, "title": q.question_text[:150], "y": answers[q.id_of_question],
                     "models": bot._individual.get(key, []), "final": float(rep.prediction),
                     **bot._extra.get(key, {})})
    out = ROOT / "data" / "backtest.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    n = len(rows)
    lines = [f"# Prueba con preguntas resueltas ({datetime.now(timezone.utc):%d/%m/%Y})", "",
             f"- Torneo {args.tournament}, resueltas desde {args.since}, {'con' if args.with_research else 'sin'} investigación.",
             f"- Preguntas: {n} · proporción de SÍ: {sum(r['y'] for r in rows) / max(n, 1):.0%}"]
    if n:
        ll = sum(logloss(r["final"], r["y"]) for r in rows) / n
        br = sum(brier(r["final"], r["y"]) for r in rows) / n
        lines.append(f"- Conjunto: pérdida log {ll:.3f} · Brier {br:.3f} (decir siempre 50 % = 0,693 / 0,250)")
        per: dict[str, list[float]] = {}
        for r in rows:
            for m in r["models"]:
                if isinstance(m.get("p"), (int, float)):
                    per.setdefault(m["model"], []).append(logloss(m["p"], r["y"]))
        lines += ["", "| Modelo | n | Pérdida log |", "|---|---|---|"]
        for m, v in sorted(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            lines.append(f"| {m} | {len(v)} | {sum(v) / len(v):.3f} |")
        if n >= 20:
            a, b = fit_platt([logit(r["final"]) for r in rows], [r["y"] for r in rows])
            lines.append(f"\n- Calibración sugerida (solo orientativa): a = {a:+.3f}, b = {b:.3f}")
        sup = [r for r in rows if "supervisor" in r]
        if sup:
            lj = sum(logloss(r["supervisor"]["p"], r["y"]) for r in sup) / len(sup)
            le = sum(logloss(r["supervisor"]["ensemble"], r["y"]) for r in sup) / len(sup)
            lines.append(f"- Juez activado en {len(sup)} preguntas: pérdida juez {lj:.3f} vs conjunto sin juez {le:.3f}")
    rep_path = ROOT / "data" / "backtest_report.md"
    rep_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
