# Laboratorio nocturno (31/10/2026 23:12 UTC)

- Preguntas resueltas probadas en total: **71** (sin investigación, como si fuera el día de cierre)
- Proporción de SÍ: 42%
- Conjunto final: pérdida log **0.676** · Brier 0.217 (decir siempre 50 % = 0,693 / 0,250)

| Modelo | n | Pérdida log (menos es mejor) |
|---|---|---|
| openai/gpt-5.4 | 16 | 0.458 |
| openai/gpt-5.4-mini | 69 | 0.700 |

- Calibración con todas las filas (71, muchas solo con GPT-5.4-mini): a = +0.564, b = 0.792 (b > 1: demasiado prudente; b < 1: demasiado extremo). Solo orientativa.
