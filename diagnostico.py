"""
diagnostico.py - Comprueba qué funciona ya, sin publicar nada y gastando lo mínimo.

1. Token de Metaculus: lee las preguntas abiertas de los torneos.
2. Proxy de IA de Metaculus (metaculus/<modelo>, se paga con los créditos de Metaculus):
   prueba varios nombres de modelo con una pregunta de una línea.
3. Gemini gratis (GEMINI_API_KEY): modelo elegido + búsqueda en Google.
4. OpenRouter y AskNews, si hay claves.
5. Si algún modelo funciona: pronóstico de prueba (sin publicar) de 2 preguntas abiertas.
Escribe data/diagnostico.md y lo añade al resumen de la ejecución de GitHub.
"""
from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from bot_helpers import silence_noisy_dependencies

silence_noisy_dependencies()
from forecasting_tools import GeneralLlm, MetaculusClient  # noqa: E402

import bot_pro  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "diagnostico.md"
lines: list[str] = [f"# Diagnóstico ({datetime.now(timezone.utc):%d/%m/%Y %H:%M} UTC)", ""]


def say(s: str = "") -> None:
    print(s)
    lines.append(s)


def short(e: BaseException) -> str:
    return (type(e).__name__ + ": " + str(e)).replace("\n", " ")[:220]


async def ask(model: str, prompt: str = "Reply with the single word OK.", **kw) -> str:
    return (await GeneralLlm(model=model, timeout=90, allowed_tries=1, **kw).invoke(prompt)).strip()


async def main() -> None:
    have = {k: bool(os.getenv(k)) for k in ("METACULUS_TOKEN", "OPENROUTER_API_KEY", "GEMINI_API_KEY", "ASKNEWS_API_KEY", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "TAVILY_API_KEY", "EXA_API_KEY", "SERPER_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY", "LINKUP_API_KEY")}
    say("## Claves presentes")
    say(", ".join(f"{k}: {'sí' if v else 'no'}" for k, v in have.items()))

    # 1. Metaculus
    say("\n## 1. Token de Metaculus")
    client = MetaculusClient()
    T = bot_pro.load_config()["tournaments"]
    import forecasting_tools
    say(f"- Versión de forecasting-tools: {getattr(forecasting_tools, '__version__', '?')}")
    open_qs = {}
    for name, tid in (("Torneo otoño", T["main"]), ("MiniBench", T["minibench"]), ("Market Pulse", T["market_pulse"])):
        try:
            qs = await asyncio.to_thread(client.get_all_open_questions_from_tournament, tid)
            open_qs[name] = qs
            say(f"- {name} ({tid}): {len(qs)} preguntas abiertas ✅")
        except Exception as e:
            say(f"- {name} ({tid}): ERROR {short(e)}")

    # 2. Proxy de Metaculus
    say("\n## 2. Proxy de IA de Metaculus (créditos de Metaculus)")
    proxy_ok = []
    for m in ["metaculus/gpt-4o-mini", "metaculus/gpt-4o", "metaculus/gpt-5", "metaculus/gpt-5.6-sol", "metaculus/gpt-6.1-sol",
              "metaculus/o3", "metaculus/claude-sonnet-4-20250514", "metaculus/claude-sonnet-5-5", "metaculus/claude-sonnet-5.5",
              "metaculus/claude-opus-5-5", "metaculus/claude-opus-5.5"]:
        try:
            out = await ask(m)
            proxy_ok.append(m)
            say(f"- {m}: ✅ ({out[:30]!r})")
        except Exception as e:
            say(f"- {m}: no ({short(e)})")

    # 3. Gemini
    say("\n## 3. Gemini gratis")
    cfg = bot_pro.load_config()
    gcfg = {"forecasters": ["gemini/auto-flash"], "researchers": ["gemini-search/auto-flash"]}
    bot_pro.resolve_gemini_models(gcfg)
    gem = gcfg["forecasters"][0]
    gem_ok = False
    if gem.startswith("gemini/"):
        try:
            out = await ask(gem)
            gem_ok = True
            say(f"- {gem}: ✅ ({out[:30]!r})")
        except Exception as e:
            say(f"- {gem}: ERROR {short(e)}")
        try:
            sg = gcfg["researchers"][0].replace("gemini-search/", "gemini/") if gcfg.get("researchers") else gem
            out = await ask(sg, "Search the web: what is today's date and one top news headline? Answer in one line.",
                            tools=[{"googleSearch": {}}])
            say(f"- Búsqueda en Google con {sg}: ✅ ({out[:120]!r})")
        except Exception as e:
            say(f"- Búsqueda en Google: ERROR {short(e)}")
    else:
        say(f"- Sin modelo Gemini gratuito disponible (se usaría {gem})")

    # 4. OpenRouter / AskNews
    say("\n## 4. OpenRouter y AskNews")
    if have["OPENROUTER_API_KEY"]:
        used = bot_pro.openrouter_usage()
        say(f"- OpenRouter: gasto acumulado {used} $")
    else:
        say("- OpenRouter: sin clave todavía")
    if have["OPENAI_API_KEY"]:
        for m in ("openai/gpt-5.4-mini", "openai/gpt-5.4"):
            try:
                out = await bot_pro.openai_complete(m, "Reply with the single word OK.", "low", 120)
                say(f"- OpenAI {m}: ✅ ({out.strip()[:20]!r})")
            except Exception as e:
                say(f"- OpenAI {m}: ERROR {short(e)}")
        u = bot_pro.openai_usage()
        say(f"- Tokens gratis de OpenAI usados hoy: grandes {u['big']}/{bot_pro.OPENAI_DAILY['big']}, "
            f"mini {u['small']}/{bot_pro.OPENAI_DAILY['small']}")
    else:
        say("- OpenAI: sin clave")
    if have["MISTRAL_API_KEY"]:
        try:
            out = await bot_pro.mistral_complete(None, "Reply with the single word OK.", None, 120)
            say(f"- Mistral ({bot_pro.MISTRAL_CHAIN[0]}): ✅ ({out.strip()[:20]!r})")
        except Exception as e:
            say(f"- Mistral: ERROR {short(e)}")
    if have["ASKNEWS_API_KEY"] or (have["ASKNEWS_CLIENT_ID"] and have["ASKNEWS_SECRET"]):
        try:
            out = await bot_pro.asknews_news("US Federal Reserve interest rate decision", bot_pro.load_config())
            say(f"- AskNews: ✅ ({len(out)} caracteres)")
        except Exception as e:
            say(f"- AskNews: ERROR {short(e)}")
    else:
        say("- AskNews: sin claves")

    # 4b. Buscadores gratuitos y preguntas relacionadas
    say("\n## 4b. Buscadores gratuitos")
    import market_data
    for name, fn in (("Serper", market_data.serper), ("Linkup", market_data.linkup), ("Tavily", market_data.tavily), ("Exa", market_data.exa), ("Google News", market_data.google_news),
                     ("GDELT", market_data.gdelt), ("Wikipedia", market_data.wikipedia)):
        try:
            rows = await asyncio.to_thread(fn, "Federal Reserve interest rates")
            say(f"- {name}: {'✅ ' + str(len(rows)) + ' resultados' if rows else 'sin resultados (¿falta la clave?)'}")
        except Exception as e:
            say(f"- {name}: ERROR {short(e)}")
    try:
        k = await asyncio.to_thread(market_data.kalshi, "Federal Reserve interest rate decision")
        say(f"- Kalshi: {'✅ ' + str(len(k)) + ' mercados' if k else 'sin coincidencias'}")
    except Exception as e:
        say(f"- Kalshi: ERROR {short(e)}")
    try:
        rel = await asyncio.to_thread(market_data.related_metaculus, "Will the Federal Reserve cut interest rates")
        say(f"- Preguntas relacionadas de Metaculus: {'✅' if rel else 'ninguna encontrada'}")
    except Exception as e:
        say(f"- Preguntas relacionadas de Metaculus: ERROR {short(e)}")
    try:
        fs = await asyncio.to_thread(market_data.free_search, "Will the Federal Reserve cut interest rates at its next meeting?")
        i = fs.find("Full text of the top results")
        n_pages = fs[i:].count("\n--- ") if i >= 0 else 0
        say(f"- Lectura de páginas completas: {'✅ ' + str(n_pages) + ' páginas leídas (' + str(len(fs) - i) + ' caracteres)' if n_pages else 'ninguna página leída'}")
    except Exception as e:
        say(f"- Lectura de páginas completas: ERROR {short(e)}")

    # 5. Pronóstico de prueba sin publicar
    say("\n## 5. Pronóstico de prueba (no se publica)")
    model = gem if gem_ok else (proxy_ok[-1] if proxy_ok else None)
    if not model:
        say("- Ningún modelo funciona todavía: no se puede hacer la prueba.")
    else:
        sample = (open_qs.get("Torneo otoño") or [])[:1] + (open_qs.get("MiniBench") or [])[:1]
        if not sample:
            say("- No hay preguntas abiertas ahora mismo para probar.")
        tcfg = json.loads(json.dumps(cfg))
        tcfg.update({"forecasters": [model], "fallback_forecasters": [], "min_forecasters": 1, "min_forecasters_required": 1,
                     "researchers": ["gemini-search/auto-flash"] if gem_ok else [], "research_planner": model,
                     "supervisor": model, "parser": model, "predictions_per_model": 3, "_no_log": True,
                     "quant_baseline": True})
        if gem_ok:
            tcfg["researchers"] = [gcfg["researchers"][0]] if gcfg["researchers"] else []
        bot = bot_pro.ProBot(cfg=tcfg, forecasters=[model], researchers=tcfg["researchers"],
                             research_reports_per_question=1, predictions_per_research_report=3,
                             publish_reports_to_metaculus=False, enable_summarize_research=False,
                             llms={"default": GeneralLlm(model=model), "summarizer": model, "researcher": "no_research", "parser": model})
        reps = await bot.forecast_questions(sample, return_exceptions=True)
        for q, r in zip(sample, reps):
            if isinstance(r, BaseException):
                say(f"- {q.page_url}: ERROR {short(r)}")
                print(traceback.format_exception(r))
                continue
            key = bot._qkey(q)
            pred = r.prediction
            if hasattr(pred, "predicted_options"):
                pred = {o.option_name: round(o.probability, 3) for o in pred.predicted_options}
            elif hasattr(pred, "declared_percentiles"):
                pred = [(round(p.percentile, 2), round(p.value, 3)) for p in pred.declared_percentiles][:6]
            say(f"- [{(q.question_text or '')[:90]}]({q.page_url}) → {pred} · investigación {len(bot._research.get(key, ''))} caracteres"
                + (f" · línea base {bot._quant[key]['spec']['symbol']}" if key in bot._quant else ""))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summ = os.getenv("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    asyncio.run(main())
