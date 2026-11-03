# Laboratorio nocturno (03/11/2026 23:23 UTC)

- Preguntas resueltas probadas en total: **84** (sin investigación, como si fuera el día de cierre)
- Proporción de SÍ: 38%
- Conjunto final: pérdida log **0.634** · Brier 0.198 (decir siempre 50 % = 0,693 / 0,250)

| Modelo | n | Pérdida log (menos es mejor) |
|---|---|---|
| openai/gpt-5.4 | 22 | 0.348 |
| openai/gpt-5.4-mini | 82 | 0.728 |

- Calibración de GPT-5.4 (22 preguntas): a = +0.139, b = 1.024 (b > 1: demasiado prudente; b < 1: demasiado extremo). Solo orientativa.
