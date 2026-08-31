"""Application configuration and personalization rules."""

from __future__ import annotations

from typing import Any

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime settings. Override via environment variables."""

    app_name: str = "Personalized AI Context Engine for Kundli"
    app_version: str = "1.0.0"
    debug: bool = True

    # Upstream service timeouts / retries
    upstream_timeout_seconds: float = 2.5
    upstream_max_retries: int = 2
    upstream_retry_backoff: float = 0.3

    # Cache TTL (seconds)
    cache_ttl_user: int = 300
    cache_ttl_kundli: int = 600
    cache_ttl_horoscope: int = 180
    cache_ttl_panchang: int = 120

    # LLM
    llm_provider: str = "gemini"  # mock | gemini
    llm_model: str = "gemini-3.5-flash"
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    panchang_api_key: str | None = None
    llm_timeout_seconds: float = 15.0
    max_prompt_chars: int = 4000

    # Response defaults
    default_max_words: int = 200
    premium_max_words: int = 350

    # Intent detection
    # keyword | semantic | hybrid (keyword + Qdrant semantic, recommended)
    intent_mode: str = "hybrid"

    # Qdrant vector store
    # memory  → in-process (default, zero infra)
    # local   → on-disk path (qdrant_path)
    # remote  → Qdrant server / Cloud (qdrant_url + optional api_key)
    qdrant_mode: str = "memory"
    qdrant_path: str = "./data/qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    semantic_top_k: int = 5
    # Semantic search
    semantic_min_confidence: float = 0.55

    # Auth / JWT
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017/mynaksh"
    mongodb_db_name: str = "mynaksh"
    mongodb_users_collection: str = "users"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


# ---------------------------------------------------------------------------
# Configuration-driven Personalization Engine rules
# Extensible: add intents / sources without large if/else blocks.
# ---------------------------------------------------------------------------

INTENT_KEYWORDS: dict[str, list[str]] = {
    "career": [
        "job", "jobs", "career", "work", "profession", "promotion", "office",
        "business", "interview", "resignation", "switch", "employment",
        "boss", "colleague", "workplace", "salary raise", "startup",
    ],
    "relationship": [
        "relationship", "partner", "marriage", "love", "spouse", "dating",
        "romance", "breakup", "compatibility", "husband", "wife", "boyfriend",
        "girlfriend", "family conflict", "in-laws",
    ],
    "health": [
        "health", "illness", "sick", "disease", "fitness", "energy",
        "sleep", "stress", "anxiety", "body", "wellness", "recovery",
        "mental health", "diet", "exercise",
    ],
    "finance": [
        "money", "finance", "investment", "wealth", "income", "expense",
        "loan", "debt", "savings", "stock", "market", "budget", "property",
        "real estate", "profit", "loss",
    ],
    "general": [
        "today", "week", "month", "year", "guidance", "advice", "focus",
        "prioritize", "overview", "summary", "how is", "what's next",
    ],
}

# Source identifiers used across the system
SOURCE_CAREER_HOROSCOPE = "Career Horoscope"
SOURCE_FINANCE_HOROSCOPE = "Finance Horoscope"
SOURCE_HEALTH_HOROSCOPE = "Health Horoscope"
SOURCE_RELATIONSHIP_HOROSCOPE = "Relationship Horoscope"
SOURCE_10TH_HOUSE = "10th House"
SOURCE_7TH_HOUSE = "7th House"
SOURCE_6TH_HOUSE = "6th House"
SOURCE_2ND_HOUSE = "2nd House"
SOURCE_CURRENT_DASHA = "Current Dasha"
SOURCE_MOON_SIGN = "Moon Sign"
SOURCE_LAGNA = "Lagna"
SOURCE_PANCHANG = "Today's Panchang"
SOURCE_BIRTH_DETAILS = "Birth Details"
SOURCE_USER_PROFILE = "User Profile"

PERSONALIZATION_RULES: dict[str, dict[str, Any]] = {
    "career": {
        "primary": [SOURCE_10TH_HOUSE, SOURCE_CAREER_HOROSCOPE, SOURCE_CURRENT_DASHA],
        "secondary": [SOURCE_PANCHANG, SOURCE_LAGNA, SOURCE_MOON_SIGN],
        "exclude": [SOURCE_RELATIONSHIP_HOROSCOPE, SOURCE_7TH_HOUSE],
        "max_words_base": 250,
    },
    "relationship": {
        "primary": [SOURCE_7TH_HOUSE, SOURCE_RELATIONSHIP_HOROSCOPE, SOURCE_MOON_SIGN],
        "secondary": [SOURCE_CURRENT_DASHA, SOURCE_PANCHANG],
        "exclude": [SOURCE_CAREER_HOROSCOPE, SOURCE_10TH_HOUSE, SOURCE_FINANCE_HOROSCOPE],
        "max_words_base": 220,
    },
    "health": {
        "primary": [SOURCE_6TH_HOUSE, SOURCE_HEALTH_HOROSCOPE, SOURCE_MOON_SIGN],
        "secondary": [SOURCE_PANCHANG, SOURCE_CURRENT_DASHA],
        "exclude": [SOURCE_FINANCE_HOROSCOPE, SOURCE_CAREER_HOROSCOPE],
        "max_words_base": 200,
    },
    "finance": {
        "primary": [SOURCE_2ND_HOUSE, SOURCE_FINANCE_HOROSCOPE, SOURCE_CURRENT_DASHA],
        "secondary": [SOURCE_10TH_HOUSE, SOURCE_PANCHANG],
        "exclude": [SOURCE_RELATIONSHIP_HOROSCOPE, SOURCE_HEALTH_HOROSCOPE],
        "max_words_base": 220,
    },
    "general": {
        "primary": [
            SOURCE_CAREER_HOROSCOPE,
            SOURCE_RELATIONSHIP_HOROSCOPE,
            SOURCE_HEALTH_HOROSCOPE,
            SOURCE_FINANCE_HOROSCOPE,
            SOURCE_CURRENT_DASHA,
            SOURCE_PANCHANG,
            SOURCE_MOON_SIGN,
            SOURCE_LAGNA,
        ],
        "secondary": [SOURCE_10TH_HOUSE, SOURCE_7TH_HOUSE, SOURCE_6TH_HOUSE],
        "exclude": [],
        "max_words_base": 280,
    },
}

# Language code → display name
LANGUAGE_MAP: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
}

# Tone preference → prompt instruction
TONE_INSTRUCTIONS: dict[str, str] = {
    "motivational": (
        "Use an encouraging, uplifting, and motivational tone. "
        "Emphasize growth, opportunity, and positive action."
    ),
    "practical": (
        "Use a clear, practical, and actionable tone. "
        "Focus on concrete steps and realistic expectations."
    ),
    "empathetic": (
        "Use a warm, empathetic, and supportive tone. "
        "Acknowledge emotions and offer gentle guidance."
    ),
    "direct": (
        "Use a concise, direct, and straightforward tone. "
        "Avoid fluff; state insights clearly."
    ),
    "spiritual": (
        "Use a calm, reflective, and spiritually grounded tone. "
        "Connect planetary influences to inner growth."
    ),
}
