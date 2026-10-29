# Laboratorio nocturno (29/10/2026 23:16 UTC)

- Preguntas resueltas probadas en total: **49** (sin investigación, como si fuera el día de cierre)
- Proporción de SÍ: 33%
- Conjunto final: pérdida log **0.650** · Brier 0.201 (decir siempre 50 % = 0,693 / 0,250)

| Modelo | n | Pérdida log (menos es mejor) |
|---|---|---|
| openai/gpt-5.4 | 10 | 0.341 |
| openai/gpt-5.4-mini | 47 | 0.678 |

- Calibración con todas las filas (49, muchas solo con GPT-5.4-mini): a = +0.265, b = 0.785 (b > 1: demasiado prudente; b < 1: demasiado extremo). Solo orientativa.
