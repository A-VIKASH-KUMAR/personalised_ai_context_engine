# MyNaksh Personalized AI Context Engine

Intelligence layer between MyNaksh astrological backend services (User, Kundli, Horoscope, Panchang) and the LLM.

## What it does

1. **Gathers** user context from multiple upstream services **concurrently**
2. **Detects** user intent (career, relationship, health, finance, general)
3. **Selects** only relevant context via a **configuration-driven** Personalization Engine
4. **Personalizes** language, tone, and response length from the user profile
5. **Builds** an optimized prompt with only selected context
6. **Generates** a grounded AI response (mock by default; OpenAI when keyed)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/personalize` | Full pipeline → answer + confidence + sourcesUsed |
| `POST` | `/debug/personalization` | Personalization decisions only (no LLM) |
| `GET`  | `/health` | Liveness |

### Example: `/personalize`

```bash
curl -s -X POST http://127.0.0.1:8000/personalize \
  -H 'Content-Type: application/json' \
  -d '{"userId":"user_101","question":"Should I consider changing my job in the next few months?"}'
```

### Example: `/debug/personalization`

```bash
curl -s -X POST http://127.0.0.1:8000/debug/personalization \
  -H 'Content-Type: application/json' \
  -d '{"userId":"user_101","question":"How does this month look for my relationship?"}'
```

## Project layout

```
app/
  main.py                 # FastAPI app & routes
  config.py               # Settings + PERSONALIZATION_RULES
  logging_config.py       # Request-ID middleware & structured logs
  models/schemas.py       # Pydantic models
  cache/memory.py         # In-memory TTL cache
  services/
    upstream.py           # Concurrent mocks with retry/timeout/cache
    intent.py             # Keyword-scored intent detection
    personalization.py    # Config-driven engine
    prompt_builder.py     # Optimized prompt construction
    llm.py                # Swappable LLM (Mock / OpenAI)
```

## Personalization rules (extensible)

Defined in `app/config.py` as `PERSONALIZATION_RULES`. Example:

| Intent       | Primary                         | Secondary              | Exclude                  |
|--------------|---------------------------------|------------------------|--------------------------|
| Career       | 10th House, Career Horoscope, Dasha | Panchang, Lagna, Moon | Relationship Horoscope   |
| Relationship | 7th House, Rel. Horoscope, Moon | Dasha, Panchang        | Career / Finance         |
| Health       | 6th House, Health Horoscope, Moon | Panchang, Dasha      | Finance / Career         |
| Finance      | 2nd House, Finance Horoscope, Dasha | 10th, Panchang       | Relationship / Health    |
| General      | All major sources               | Houses                 | —                        |

Add a new intent by extending `INTENT_KEYWORDS` and `PERSONALIZATION_RULES` — no large if/else trees required.

## Run locally

```bash
cd personalized_ai_context
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

OpenAPI docs: http://127.0.0.1:8000/docs

### Optional real LLM

```bash
export MYNAKSH_LLM_PROVIDER=openai
export MYNAKSH_OPENAI_API_KEY=sk-...
export MYNAKSH_LLM_MODEL=gpt-4o-mini
```

## Design notes

- **Partial failure tolerant**: upstream `asyncio.gather(..., return_exceptions=True)`; missing services are logged and skipped.
- **Retries + timeouts**: Tenacity exponential backoff; per-call `asyncio.wait_for`.
- **Caching**: TTL in-memory cache per upstream key.
- **Logging**: request id, latency, prompt size, sources used.
- **Qdrant**: not required for the core path. Intent is keyword-scored for determinism and zero infra. To add semantic intent, embed example questions into Qdrant and rank against the query; the rest of the pipeline stays unchanged.

## Sample users

| userId   | Name         | Tone          | Subscription |
|----------|--------------|---------------|--------------|
| user_101 | Aarav Sharma | motivational  | premium      |
| user_102 | Priya Patel  | empathetic    | free         |
| user_103 | Rohan Mehta  | practical     | premium      |
# personalised_ai_context_engine
