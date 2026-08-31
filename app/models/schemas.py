"""Pydantic request / response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PersonalizeRequest(BaseModel):
    userId: str | None = Field(None, description="Optional legacy user identifier")
    question: str = Field(..., min_length=3, max_length=1000, description="User question")


class PersonalizeResponse(BaseModel):
    answer: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    sourcesUsed: list[str]


class SemanticHitSchema(BaseModel):
    intent: str
    score: float
    example: str


class DebugPersonalizationResponse(BaseModel):
    intent: str
    selectedContext: list[str]
    excludedContext: list[str]
    language: str
    tone: str
    maxWords: int
    reasoning: str | None = None
    intentMode: str | None = None
    semanticHits: list[SemanticHitSchema] | None = None
    semanticAvailable: bool | None = None


class SemanticSearchResponse(BaseModel):
    query: str
    intent: str
    confidence: float
    method: str
    hits: list[SemanticHitSchema]
    available: bool



# --- Upstream / internal models ------------------------------------------------

class BirthDetails(BaseModel):
    date: str
    time: str
    place: str


class UserProfile(BaseModel):
    id: str
    name: str
    language: str = "en"
    subscription: str = "free"
    tonePreference: str = "motivational"
    birthDetails: BirthDetails | None = None


class DashaInfo(BaseModel):
    mahadasha: str
    antardasha: str


class HouseInfo(BaseModel):
    lord: str
    strength: str


class KundliData(BaseModel):
    lagna: str
    moonSign: str
    currentDasha: DashaInfo
    houses: dict[str, HouseInfo]


class HoroscopeData(BaseModel):
    career: str
    finance: str
    health: str
    relationship: str


class PanchangData(BaseModel):
    date: str
    tithi: str
    nakshatra: str
    yoga: str
    karana: str


class UpstreamBundle(BaseModel):
    """Aggregated result of concurrent upstream fetches."""

    user: UserProfile | None = None
    kundli: KundliData | None = None
    horoscope: HoroscopeData | None = None
    panchang: PanchangData | None = None
    errors: dict[str, str] = Field(default_factory=dict)


class PersonalizationConfig(BaseModel):
    """Internal output of the Personalization Engine (not returned to client)."""

    intent: str
    language: str
    language_code: str
    tone: str
    maxWords: int
    selectedContext: list[str]
    excludedContext: list[str]
    reasoning: str = ""


class LLMResult(BaseModel):
    answer: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    sourcesUsed: list[str]
    prompt_chars: int = 0
    latency_ms: float = 0.0
