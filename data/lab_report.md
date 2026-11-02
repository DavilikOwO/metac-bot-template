# Laboratorio nocturno (02/11/2026 23:20 UTC)

- Preguntas resueltas probadas en total: **78** (sin investigación, como si fuera el día de cierre)
- Proporción de SÍ: 41%
- Conjunto final: pérdida log **0.677** · Brier 0.212 (decir siempre 50 % = 0,693 / 0,250)

| Modelo | n | Pérdida log (menos es mejor) |
|---|---|---|
| openai/gpt-5.4 | 21 | 0.362 |
| openai/gpt-5.4-mini | 76 | 0.778 |

- Calibración de GPT-5.4 (21 preguntas): a = +0.145, b = 1.021 (b > 1: demasiado prudente; b < 1: demasiado extremo). Solo orientativa.
