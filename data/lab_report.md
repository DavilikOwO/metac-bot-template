# Laboratorio nocturno (01/11/2026 23:14 UTC)

- Preguntas resueltas probadas en total: **77** (sin investigación, como si fuera el día de cierre)
- Proporción de SÍ: 40%
- Conjunto final: pérdida log **0.685** · Brier 0.215 (decir siempre 50 % = 0,693 / 0,250)

| Modelo | n | Pérdida log (menos es mejor) |
|---|---|---|
| openai/gpt-5.4 | 20 | 0.380 |
| openai/gpt-5.4-mini | 75 | 0.789 |

- Calibración de GPT-5.4 (20 preguntas): a = +0.143, b = 1.020 (b > 1: demasiado prudente; b < 1: demasiado extremo). Solo orientativa.
