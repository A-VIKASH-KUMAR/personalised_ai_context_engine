"""Configuration-driven Personalization Engine.

Given a question + upstream data, decides:
  - intent
  - which context sources to include / exclude
  - language, tone, response length
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import (
    LANGUAGE_MAP,
    PERSONALIZATION_RULES,
    SOURCE_10TH_HOUSE,
    SOURCE_2ND_HOUSE,
    SOURCE_6TH_HOUSE,
    SOURCE_7TH_HOUSE,
    SOURCE_BIRTH_DETAILS,
    SOURCE_CAREER_HOROSCOPE,
    SOURCE_CURRENT_DASHA,
    SOURCE_FINANCE_HOROSCOPE,
    SOURCE_HEALTH_HOROSCOPE,
    SOURCE_LAGNA,
    SOURCE_MOON_SIGN,
    SOURCE_PANCHANG,
    SOURCE_RELATIONSHIP_HOROSCOPE,
    SOURCE_USER_PROFILE,
    TONE_INSTRUCTIONS,
    settings,
)
from app.models.schemas import PersonalizationConfig, UpstreamBundle, UserProfile
from app.services.intent import detect_intent

logger = logging.getLogger(__name__)


def _available_sources(bundle: UpstreamBundle) -> set[str]:
    """Return the set of context sources that actually have data."""
    available: set[str] = set()

    if bundle.user:
        available.add(SOURCE_USER_PROFILE)
        if bundle.user.birthDetails:
            available.add(SOURCE_BIRTH_DETAILS)

    if bundle.kundli:
        available.add(SOURCE_LAGNA)
        available.add(SOURCE_MOON_SIGN)
        available.add(SOURCE_CURRENT_DASHA)
        houses = bundle.kundli.houses or {}
        if "10" in houses:
            available.add(SOURCE_10TH_HOUSE)
        if "7" in houses:
            available.add(SOURCE_7TH_HOUSE)
        if "6" in houses:
            available.add(SOURCE_6TH_HOUSE)
        if "2" in houses:
            available.add(SOURCE_2ND_HOUSE)

    if bundle.horoscope:
        available.add(SOURCE_CAREER_HOROSCOPE)
        available.add(SOURCE_FINANCE_HOROSCOPE)
        available.add(SOURCE_HEALTH_HOROSCOPE)
        available.add(SOURCE_RELATIONSHIP_HOROSCOPE)

    if bundle.panchang:
        available.add(SOURCE_PANCHANG)

    return available


def _resolve_language(user: UserProfile | None) -> tuple[str, str]:
    code = (user.language if user else "en") or "en"
    display = LANGUAGE_MAP.get(code, "English")
    return display, code


def _resolve_tone(user: UserProfile | None) -> str:
    if not user:
        return "motivational"
    tone = (user.tonePreference or "motivational").lower().strip()
    if tone not in TONE_INSTRUCTIONS:
        return "motivational"
    return tone


def _resolve_max_words(user: UserProfile | None, intent: str) -> int:
    rules = PERSONALIZATION_RULES.get(intent, PERSONALIZATION_RULES["general"])
    base = int(rules.get("max_words_base", settings.default_max_words))
    if user and user.subscription == "premium":
        return min(base + 80, settings.premium_max_words)
    return min(base, settings.default_max_words)


def build_personalization(
    question: str,
    bundle: UpstreamBundle,
) -> PersonalizationConfig:
    """
    Core Personalization Engine entry point.

    Configuration-driven: intent → primary/secondary/exclude lists from
    PERSONALIZATION_RULES, filtered by what is actually available upstream.
    """
    intent, intent_conf, intent_reason = detect_intent(question)
    rules = PERSONALIZATION_RULES.get(intent, PERSONALIZATION_RULES["general"])

    available = _available_sources(bundle)
    primary: list[str] = list(rules.get("primary", []))
    secondary: list[str] = list(rules.get("secondary", []))
    exclude_cfg: list[str] = list(rules.get("exclude", []))

    # Only keep sources that exist in upstream data
    selected: list[str] = []
    for src in primary + secondary:
        if src in available and src not in selected and src not in exclude_cfg:
            selected.append(src)

    # Always include user profile metadata when present (for name / language cues)
    if SOURCE_USER_PROFILE in available and SOURCE_USER_PROFILE not in selected:
        selected.insert(0, SOURCE_USER_PROFILE)

    excluded = [s for s in exclude_cfg if s in available]

    language, language_code = _resolve_language(bundle.user)
    tone = _resolve_tone(bundle.user)
    max_words = _resolve_max_words(bundle.user, intent)

    reasoning = (
        f"Intent={intent} (conf≈{intent_conf:.2f}). {intent_reason}. "
        f"Primary sources requested={primary}; secondary={secondary}; "
        f"available={sorted(available)}; selected={selected}."
    )

    config = PersonalizationConfig(
        intent=intent,
        language=language,
        language_code=language_code,
        tone=tone,
        maxWords=max_words,
        selectedContext=selected,
        excludedContext=excluded,
        reasoning=reasoning,
    )
    logger.info(
        "Personalization ready: intent=%s tone=%s lang=%s sources=%s",
        config.intent,
        config.tone,
        config.language,
        config.selectedContext,
    )
    return config


def extract_context_payload(
    selected: list[str],
    bundle: UpstreamBundle,
) -> dict[str, Any]:
    """
    Materialize the selected context into a compact dict for the prompt builder.
    Only includes keys corresponding to selected sources.
    """
    payload: dict[str, Any] = {}
    user = bundle.user
    kundli = bundle.kundli
    horoscope = bundle.horoscope
    panchang = bundle.panchang

    for src in selected:
        if src == SOURCE_USER_PROFILE and user:
            payload["user"] = {
                "name": user.name,
                "subscription": user.subscription,
            }
        elif src == SOURCE_BIRTH_DETAILS and user and user.birthDetails:
            payload["birthDetails"] = user.birthDetails.model_dump()
        elif src == SOURCE_LAGNA and kundli:
            payload["lagna"] = kundli.lagna
        elif src == SOURCE_MOON_SIGN and kundli:
            payload["moonSign"] = kundli.moonSign
        elif src == SOURCE_CURRENT_DASHA and kundli:
            payload["currentDasha"] = kundli.currentDasha.model_dump()
        elif src == SOURCE_10TH_HOUSE and kundli and "10" in kundli.houses:
            payload["house_10"] = kundli.houses["10"].model_dump()
        elif src == SOURCE_7TH_HOUSE and kundli and "7" in kundli.houses:
            payload["house_7"] = kundli.houses["7"].model_dump()
        elif src == SOURCE_6TH_HOUSE and kundli and "6" in kundli.houses:
            payload["house_6"] = kundli.houses["6"].model_dump()
        elif src == SOURCE_2ND_HOUSE and kundli and "2" in kundli.houses:
            payload["house_2"] = kundli.houses["2"].model_dump()
        elif src == SOURCE_CAREER_HOROSCOPE and horoscope:
            payload["careerHoroscope"] = horoscope.career
        elif src == SOURCE_FINANCE_HOROSCOPE and horoscope:
            payload["financeHoroscope"] = horoscope.finance
        elif src == SOURCE_HEALTH_HOROSCOPE and horoscope:
            payload["healthHoroscope"] = horoscope.health
        elif src == SOURCE_RELATIONSHIP_HOROSCOPE and horoscope:
            payload["relationshipHoroscope"] = horoscope.relationship
        elif src == SOURCE_PANCHANG and panchang:
            payload["panchang"] = panchang.model_dump()

    return payload
