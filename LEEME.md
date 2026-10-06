# Bot de pronósticos (Metaculus FutureEval)

Basado en la plantilla oficial de Metaculus, con mejoras:

- **Varios modelos punteros** (OpenAI, Anthropic, Google, xAI) responden cada pregunta por separado; se combinan con una mediana ponderada.
- **Investigación en dos rondas**: varias fuentes en paralelo (AskNews + búsqueda web) y luego un modelo detecta lo que falta y lanza búsquedas de seguimiento.
- **Juez**: si los modelos discrepan mucho (más de 30 puntos), un modelo jefe revisa los argumentos de todos contra la investigación y su opinión cuenta la mitad.
- **Preguntas numéricas**: mezcla de las distribuciones de todos los modelos.
- **Prompt de superpronosticador**: criterios de resolución al pie de la letra, tasa base, statu quo, argumentos a favor y en contra.
- **Calibración** y recorte de extremos (2 %–98 %).
- **Aprende solo**: `learn.py` se ejecuta cada día, mira qué preguntas se han resuelto, sube el peso de los modelos que aciertan y recalibra. El informe queda en `data/learn_report.md`.

## Archivos
- `bot_pro.py` — el bot.
- `config/bot_config.json` — modelos, pesos y calibración (learn.py actualiza pesos y calibración).
- `learn.py` — aprendizaje diario.
- `backtest.py` — prueba con preguntas ya resueltas (Actions → "Prueba con preguntas resueltas" → Run workflow).
- `data/forecast_log.jsonl` — cada predicción de cada modelo.
- `.github/workflows/` — pronostica cada 20 min, aprende cada día a las 05:13 UTC, y "Test Bot" para probar.

## Claves (Settings → Secrets and variables → Actions)
`METACULUS_TOKEN`, `OPENROUTER_API_KEY` y, si los tienes, `ASKNEWS_CLIENT_ID` y `ASKNEWS_SECRET`.
Sin `METACULUS_TOKEN` u `OPENROUTER_API_KEY` el bot no hace nada (y no da error).
