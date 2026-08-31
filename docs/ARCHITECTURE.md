# Personalized AI Context Engine — Architecture Flow

This document describes the end-to-end flow used to **detect user intent** and **deliver the right context** to the LLM for generating a grounded response.

## High-level architecture

```mermaid
flowchart TD
    %% ===== Client =====
    U[User / Frontend<br/>app/static/index.html] -->|HTTP POST /personalize<br/>or WS /ws/chat| R

    %% ===== Edge =====
    subgraph EDGE["FastAPI Layer — app/main.py"]
        R[/personalize<br/>websocket_chat/]
        AUTH[Auth dependency<br/>JWT via get_current_user]
        LOG[RequestLoggingMiddleware]
    end

    R --> AUTH
    R --> LOG

    %% ===== Stage 1: Upstream context fetch =====
    subgraph S1["Stage 1 — Concurrent Context Fetch (app/services/upstream.py)"]
        FETCH[fetch_all_context]
        CACHE[upstream_cache<br/>TTL: user 5m · kundli 10m<br/>horoscope 3m · panchang 2m]
        RETRY[_fetch_with_retry<br/>tenacity exp-backoff + asyncio.wait_for]

        SVC_USR[UserService]
        SVC_KUN[KundliService]
        SVC_HOR[HoroscopeService]
        SVC_PAN[PanchangService<br/>httpx → /panchang]

        PAN_EXT[(External: hindutithi.in)]
    end

    AUTH --> FETCH
    FETCH -->|asyncio.gather return_exceptions=True| RETRY
    RETRY --> CACHE
    RETRY --> SVC_USR
    RETRY --> SVC_KUN
    RETRY --> SVC_HOR
    RETRY --> SVC_PAN
    SVC_PAN --> PAN_EXT

    SVC_USR & SVC_KUN & SVC_HOR & SVC_PAN --> BUNDLE[UpstreamBundle<br/>user · kundli · horoscope · panchang · errors]

    %% ===== Stage 2: Intent detection =====
    subgraph S2["Stage 2 — Hybrid Intent Detection (app/services/intent.py)"]
        Q[user question]
        KW[_keyword_detect<br/>re word-boundary · weighted scoring<br/>→ confidence 0.40-0.95]
        SE[_semantic_detect<br/>Qdrant + FastEmbed BAAI/bge-small-en-v1.5<br/>top-k=5 nearest neighbours]
        MODE{MYNAKSH_INTENT_MODE<br/>keyword · semantic · hybrid}
        RES[(INTENT_KEYWORDS<br/>app/config.py)]
        EXAMPLES[(Qdrant intent_examples<br/>seeded at startup)]

        Q --> MODE
        MODE -->|keyword| KW
        MODE -->|semantic| SE
        MODE -->|hybrid| KW
        MODE -->|hybrid| SE
        KW --> RES
        SE --> EXAMPLES
    end

    Q --> S2
    BUNDLE --> S2
    S2 --> INTENT[intent + confidence + reasoning]

    %% ===== Stage 3: Personalization =====
    subgraph S3["Stage 3 — Personalization Engine (app/services/personalization.py)"]
        BUILD[build_personalization]
        AVAIL[_available_sources<br/>filter by what actually exists]
        RULES[(PERSONALIZATION_RULES<br/>app/config.py<br/>primary · secondary · exclude · max_words)]
        RESOLVE_L[_resolve_language<br/>user.language → LANGUAGE_MAP]
        RESOLVE_T[_resolve_tone<br/>user.tonePreference → TONE_INSTRUCTIONS]
        RESOLVE_W[_resolve_max_words<br/>premium: +80 cap 350]
        EXTRACT[extract_context_payload]

        BUILD --> AVAIL
        BUILD --> RULES
        BUILD --> RESOLVE_L
        BUILD --> RESOLVE_T
        BUILD --> RESOLVE_W
        AVAIL --> EXTRACT
    end

    INTENT --> BUILD
    BUNDLE --> BUILD
    BUILD --> PCFG[PersonalizationConfig<br/>intent · language · tone<br/>maxWords · selectedContext · reasoning]
    PCFG --> EXTRACT
    BUNDLE --> EXTRACT
    EXTRACT --> CTX[context_payload<br/>compact JSON dict<br/>only selected keys]

    %% ===== Stage 4: Prompt construction =====
    subgraph S4["Stage 4 — Prompt Builder (app/services/prompt_builder.py)"]
        PB[build_messages]
        SYS[SYSTEM_TEMPLATE<br/>persona · tone · max_words]
        USR[USER_TEMPLATE<br/>question · intent · context JSON]
        TRUNC{> max_prompt_chars?<br/>truncate context JSON}
        PB --> SYS
        PB --> USR
        USR --> TRUNC
        TRUNC -->|yes| USR
    end

    Q --> PB
    PCFG --> PB
    CTX --> PB
    PB --> MSGS[messages: system + user<br/>prompt_char_count]

    %% ===== Stage 5: LLM =====
    subgraph S5["Stage 5 — LLM Provider (app/services/llm.py)"]
        PICK{settings.llm_provider}
        MOCK[MockLLMProvider<br/>intent-specific template synthesis]
        OAI[OpenAILLMProvider<br/>gpt-4o-mini · chat completions]
        MSGS --> PICK
        PICK -->|mock or no key| MOCK
        PICK -->|openai + key| OAI
    end

    PCFG --> MOCK
    CTX --> MOCK
    PCFG --> OAI
    CTX --> OAI

    MOCK --> RES_LLM[LLMResult<br/>answer · confidence HIGH/MED/LOW<br/>sourcesUsed · latency_ms · prompt_chars]
    OAI --> RES_LLM

    %% ===== Response =====
    RES_LLM --> RESP[PersonalizeResponse]
    RESP --> U
```

## Stage-by-stage breakdown

### 1. Request entry & auth — `app/main.py`
- `POST /personalize` and `WS /ws/chat` both authenticate via JWT (`get_current_user` / `get_current_user_ws`).
- `RequestLoggingMiddleware` attaches request-id and structured logs.

### 2. Concurrent upstream fetch — `app/services/upstream.py`
- `fetch_all_context(user_id)` runs **4 calls in parallel** with `asyncio.gather(return_exceptions=True)`:
  - `get_user` → profile (name, language, tonePreference, birthDetails)
  - `get_kundli` → lagna, moon sign, current dasha, houses (2/6/7/10)
  - `get_horoscope` → career/finance/health/relationship daily lines
  - `get_panchang` → today's tithi, nakshatra, yoga, karana
- Each call: TTL cache → `_fetch_with_retry` (tenacity exp-backoff + `asyncio.wait_for`).
- Partial failures are captured in `bundle.errors`; the pipeline continues.

### 3. Hybrid intent detection — `app/services/intent.py`
| Mode | Behaviour |
|------|-----------|
| `keyword` | Regex word-boundary scan over `INTENT_KEYWORDS`; weighted by phrase length; `general` fallback. |
| `semantic` | Embed the question (FastEmbed `BAAI/bge-small-en-v1.5`), Qdrant top-k nearest neighbours over `INTENT_EXAMPLES`, weighted vote. |
| `hybrid` (default) | Run both. If semantic confidence ≥ `semantic_min_confidence` (0.55), use semantic (boost when both agree, override when keywords are weak or semantic is clearly stronger). |

Output → `(intent, confidence, reasoning)` where intent ∈ `career | relationship | health | finance | general`.

### 4. Config-driven personalization — `app/services/personalization.py`
- Loads `PERSONALIZATION_RULES[intent]` from `app/config.py` (e.g. career → primary `10th House, Career Horoscope, Current Dasha`, exclude `Relationship Horoscope, 7th House`).
- `_available_sources(bundle)` filters rules by data that actually came back upstream.
- Resolves `language` (`LANGUAGE_MAP`), `tone` (`TONE_INSTRUCTIONS`), `max_words` (premium users get +80 up to `premium_max_words=350`).
- `extract_context_payload` materialises only the selected keys into a compact dict (e.g. `house_10`, `careerHoroscope`, `panchang`).

### 5. Prompt construction — `app/services/prompt_builder.py`
- `SYSTEM_TEMPLATE` injects persona + tone instruction + max-words cap.
- `USER_TEMPLATE` carries `question`, `intent`, and the **JSON-serialised selected context only**.
- Hard cap `max_prompt_chars=4000` triggers aggressive JSON truncation if exceeded.

### 6. LLM generation — `app/services/llm.py`
- `get_llm_provider()` picks `OpenAILLMProvider` (when `OPENAI_API_KEY` is set) or the deterministic `MockLLMProvider` (which synthesises grounded answers from `context_payload`).
- Returns `LLMResult { answer, confidence, sourcesUsed, latency_ms, prompt_chars }` → wrapped in `PersonalizeResponse` and sent back to the client.

## Personalization rules (from `app/config.py`)

| Intent | Primary | Excludes |
|--------|---------|----------|
| career | 10th House · Career Horoscope · Current Dasha | Relationship Horoscope · 7th House |
| relationship | 7th House · Relationship Horoscope · Moon Sign | Career · Finance |
| health | 6th House · Health Horoscope · Moon Sign | Finance · Career |
| finance | 2nd House · Finance Horoscope · Current Dasha | Relationship · Health |
| general | All major sources | — |

## Failure / edge handling

- **Upstream timeout / error** → caught by tenacity + `asyncio.gather(return_exceptions=True)`; partial context still usable, rest is logged.
- **Empty context payload** → 503 from `/personalize`, `error` frame on `/ws/chat`.
- **Prompt overflow** → context JSON truncated; header preserved.
- **LLM error** → 502; WS sends `{type:"error"}` and keeps the connection alive.

## File map

| Concern | File |
|---------|------|
| HTTP/WS routes | `app/main.py` |
| Upstream fetch (parallel, retry, cache) | `app/services/upstream.py` |
| Intent detection | `app/services/intent.py`, `app/services/semantic_intent.py` |
| Personalization rules + intent keys + tone map | `app/config.py` |
| Source selection & payload materialisation | `app/services/personalization.py` |
| Prompt assembly | `app/services/prompt_builder.py` |
| LLM adapters | `app/services/llm.py` |
| Pydantic schemas | `app/models/schemas.py` |