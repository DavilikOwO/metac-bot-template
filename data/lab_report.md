# Laboratorio nocturno (30/10/2026 23:16 UTC)

- Preguntas resueltas probadas en total: **64** (sin investigación, como si fuera el día de cierre)
- Proporción de SÍ: 42%
- Conjunto final: pérdida log **0.698** · Brier 0.223 (decir siempre 50 % = 0,693 / 0,250)

| Modelo | n | Pérdida log (menos es mejor) |
|---|---|---|
| openai/gpt-5.4 | 14 | 0.432 |
| openai/gpt-5.4-mini | 62 | 0.720 |

- Calibración con todas las filas (64, muchas solo con GPT-5.4-mini): a = +0.534, b = 0.773 (b > 1: demasiado prudente; b < 1: demasiado extremo). Solo orientativa.
