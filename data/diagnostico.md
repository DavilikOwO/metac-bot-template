# Diagnóstico (06/10/2026 08:28 UTC)

## Claves presentes
METACULUS_TOKEN: sí, OPENROUTER_API_KEY: no, GEMINI_API_KEY: sí, ASKNEWS_API_KEY: no, ASKNEWS_CLIENT_ID: no, ASKNEWS_SECRET: no, TAVILY_API_KEY: sí, EXA_API_KEY: sí, SERPER_API_KEY: sí, OPENAI_API_KEY: sí

## 1. Token de Metaculus
- Versión de forecasting-tools: ?
- Torneo otoño (33121): 0 preguntas abiertas ✅
- MiniBench (minibench): 1 preguntas abiertas ✅
- Market Pulse (market-pulse-26q4): 0 preguntas abiertas ✅

## 2. Proxy de IA de Metaculus (créditos de Metaculus)
- metaculus/gpt-4o-mini: no (BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'error': "You don't have an allowance for model <gpt-4o-mini> on <Openai>  ."})
- metaculus/gpt-4o: no (BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'error': "You don't have an allowance for model <gpt-4o> on <Openai>  ."})
- metaculus/gpt-5: no (BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'error': "You don't have an allowance for model <gpt-5> on <Openai>  ."})
- metaculus/gpt-5.6-sol: no (BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'error': "You don't have an allowance for model <gpt-5.6-sol> on <Openai>  ."})
- metaculus/gpt-6.1-sol: no (BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'error': "You don't have an allowance for model <gpt-6.1-sol> on <Openai>  ."})
- metaculus/o3: no (BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'error': "You don't have an allowance for model <o3> on <Openai>  ."})
- metaculus/claude-sonnet-4-20250514: no (BadRequestError: litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=claude-sonnet-4-20250514  Pass model as E.g. For 'Huggingface' inference endpoints pa)
- metaculus/claude-sonnet-5-5: no (BadRequestError: litellm.BadRequestError: AnthropicException - {"error": "You don't have an allowance for model <claude-sonnet-5-5> on <Anthropic>  ."})
- metaculus/claude-sonnet-5.5: no (BadRequestError: litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=claude-sonnet-5.5  Pass model as E.g. For 'Huggingface' inference endpoints pass in `)
- metaculus/claude-opus-5-5: no (BadRequestError: litellm.BadRequestError: AnthropicException - {"error": "You don't have an allowance for model <claude-opus-5-5> on <Anthropic>  ."})
- metaculus/claude-opus-5.5: no (BadRequestError: litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=claude-opus-5.5  Pass model as E.g. For 'Huggingface' inference endpoints pass in `co)

## 3. Gemini gratis
- gemini/gemini-3.8-flash: ERROR ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {   "error": {     "code": 503,     "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please
- Búsqueda en Google: ERROR NotFoundError: litellm.NotFoundError: GeminiException - {   "error": {     "code": 404,     "message": "This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini

## 4. OpenRouter y AskNews
- OpenRouter: sin clave todavía
- OpenAI openai/gpt-5.4-mini: ✅ ('OK')
- OpenAI openai/gpt-5.4: ✅ ('OK')
- Tokens gratis de OpenAI usados hoy: grandes 29/200000, mini 30/2100000
- AskNews: sin claves

## 4b. Buscadores gratuitos
- Serper: ✅ 10 resultados
- Tavily: ✅ 7 resultados
- Exa: ✅ 5 resultados
- Google News: ✅ 8 resultados
- GDELT: ✅ 8 resultados
- Wikipedia: ✅ 2 resultados
- Kalshi: ✅ 9 mercados
- Preguntas relacionadas de Metaculus: ninguna encontrada

## 5. Pronóstico de prueba (no se publica)
- Ningún modelo funciona todavía: no se puede hacer la prueba.
