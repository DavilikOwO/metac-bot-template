# Laboratorio nocturno (28/10/2026 23:16 UTC)

- Preguntas resueltas probadas en total: **34** (sin investigación, como si fuera el día de cierre)
- Proporción de SÍ: 32%
- Conjunto final: pérdida log **0.690** · Brier 0.211 (decir siempre 50 % = 0,693 / 0,250)

| Modelo | n | Pérdida log (menos es mejor) |
|---|---|---|
| openai/gpt-5.4 | 10 | 0.341 |
| openai/gpt-5.4-mini | 32 | 0.733 |

- Calibración con todas las filas (34, muchas solo con GPT-5.4-mini): a = +0.187, b = 0.812 (b > 1: demasiado prudente; b < 1: demasiado extremo). Solo orientativa.
