# Laboratorio nocturno (09/11/2026 01:58 UTC)

- Preguntas resueltas probadas en total: **98** (sin investigación, como si fuera el día de cierre)
- Proporción de SÍ: 40%
- Conjunto final: pérdida log **0.714** · Brier 0.222 (decir siempre 50 % = 0,693 / 0,250)

| Modelo | n | Pérdida log (menos es mejor) |
|---|---|---|
| openai/gpt-5.4 | 26 | 0.323 |
| openai/gpt-5.4-mini | 96 | 0.803 |

- Calibración de GPT-5.4 (26 preguntas): a = +0.053, b = 1.029 (b > 1: demasiado prudente; b < 1: demasiado extremo). Solo orientativa.
