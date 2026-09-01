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

### set the below api keys

```bash
PANCHANG_API_KEY=hindutithi_live_
OPENAI_API_KEY=sk-proj-
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

## Assumptions

- **User identity is trusted**: the `userId` is taken at face value from the request. There is no auth/JWT layer; the caller (e.g. the MyNaksh backend) is assumed to have already authenticated the user.
- **Upstream services are reachable**: `upstream.py` is wired with mock data shaped like the real MyNaksh services. Swap the mock functions for real HTTP clients to productionize — the orchestration (gather, retry, timeout, cache) is unchanged.
- **Deterministic intent is acceptable**: intent is classified with keyword scoring, not an embedding model. Multi-language or paraphrase-heavy queries will fall back to `general` and pull all sources.
- **Profiles are stable per request**: tone, language, and subscription are read once at request time. Profile edits made mid-request are not reflected.
- **Single-region, single-process**: in-memory cache assumes one app instance. No Redis/external store.
- **English-first**: tone instructions and prompt scaffolding are written in English. Hindi/regional-language output relies on the LLM's own ability.

## Trade-offs

- **In-memory TTL cache over Redis**: simpler, no extra infra, but doesn't survive restarts and doesn't share state across workers/replicas. Each replica would re-fetch the same upstream data.
- **Keyword intent over embeddings/ML classifier**: deterministic, zero-dependency, easy to debug and extend in `config.py`. Trades semantic accuracy for transparency and speed.
- **Mock LLM by default**: keeps the pipeline runnable without API keys. Real LLM (OpenAI) is opt-in via `OPENAI_API_KEY`. No streaming, no token budgeting, no cost controls.
- **Prompt built inline, not templated via Jinja**: fewer moving parts, but harder to A/B test tone/length variants at runtime.
- **No persistence layer**: conversation history, feedback, and confidence scores are not stored. The `confidence` field is computed per-request only.
- **Synchronous response**: no Server-Sent Events or websockets. Long-running generation blocks the request thread.

## Production concerns left out

- **Authentication & authorization**: no JWT/API-key validation, no per-user rate limiting, no tenant isolation.
- **Observability beyond logs**: no Prometheus metrics, no OpenTelemetry tracing, no structured log shipping (Datadog/Loki/etc.). Request-ID exists but isn't propagated to upstreams.
- **Resilience hardening**: no circuit breaker (only retry+timeout), no bulkhead between upstream calls, no graceful degradation policy beyond `return_exceptions=True`.
- **Caching**: no distributed cache, no cache stampede protection, no per-user response cache, no negative caching for upstream 404s.
- **LLM safety & cost**: no content moderation guardrails, no PII redaction before sending to OpenAI, no token/cost budgets, no model fallback chain, no prompt-injection mitigation beyond omitting unselected fields.
- **Data protection**: no encryption-at-rest for cached data, no audit log of what context was sent to the LLM per user (relevant for astrology + personal data).
- **Testing**: no automated test suite, no contract tests against real upstream services, no load/perf testing.
- **Deployment**: no Dockerfile, no CI/CD pipeline, no health/readiness split (only `/health`), no graceful shutdown config, no multi-worker guidance.
- **Schema versioning**: `PERSONALIZATION_RULES` and upstream payload shapes have no version field — breaking changes in MyNaksh services would silently degrade.
- **Internationalization**: tone is a single enum string; no locale-aware formatting for dates/numbers in horoscope content.
