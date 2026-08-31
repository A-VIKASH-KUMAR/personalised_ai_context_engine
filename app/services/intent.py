"""Hybrid intent detection: keyword scoring + optional Qdrant semantic search.

Modes (MYNAKSH_INTENT_MODE):
  keyword  — pure keyword rules (fast, deterministic)
  semantic — pure Qdrant nearest-neighbour over seeded examples
  hybrid   — keyword first; promote/override with semantic when confident
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from app.config import INTENT_KEYWORDS, settings

logger = logging.getLogger(__name__)


def _keyword_detect(question: str) -> tuple[str, float, str]:
    text = question.lower().strip()
    scores: dict[str, float] = defaultdict(float)
    matched_terms: dict[str, list[str]] = defaultdict(list)

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text):
                weight = 1.0 + 0.15 * (len(kw.split()) - 1)
                scores[intent] += weight
                matched_terms[intent].append(kw)

    if not scores:
        return "general", 0.40, "No keyword signals"

    ranked = sorted(
        scores.items(),
        key=lambda x: (-x[1], 0 if x[0] != "general" else 1),
    )
    best_intent, best_score = ranked[0]
    total = sum(scores.values())
    confidence = min(0.95, 0.5 + (best_score / max(total, 1.0)) * 0.45)
    reasoning = (
        f"Keyword matched '{best_intent}': {matched_terms[best_intent]}; "
        f"scores={dict(scores)}"
    )
    return best_intent, confidence, reasoning


def _semantic_detect(question: str) -> tuple[str, float, str]:
    from app.services.semantic_intent import semantic_detect_intent

    return semantic_detect_intent(question, top_k=settings.semantic_top_k)


def detect_intent(question: str) -> tuple[str, float, str]:
    """
    Detect primary intent.

    Returns:
        (intent_label, confidence 0-1, reasoning_string)
    """
    mode = (settings.intent_mode or "hybrid").lower()

    if mode == "keyword":
        intent, conf, reason = _keyword_detect(question)
        logger.info("Intent[%s]: %s (%.2f) — %s", mode, intent, conf, reason)
        return intent, conf, reason

    if mode == "semantic":
        intent, conf, reason = _semantic_detect(question)
        if conf <= 0.0:
            # Semantic unavailable → soft fallback
            intent, conf, reason = _keyword_detect(question)
            reason = f"Semantic unavailable; keyword fallback. {reason}"
        logger.info("Intent[%s]: %s (%.2f) — %s", mode, intent, conf, reason)
        return intent, conf, reason

    # hybrid (default)
    kw_intent, kw_conf, kw_reason = _keyword_detect(question)
    sem_intent, sem_conf, sem_reason = _semantic_detect(question)

    if sem_conf >= settings.semantic_min_confidence:
        # Semantic is confident enough to lead
        if kw_intent == sem_intent:
            # Agreement → boost confidence
            conf = min(0.98, max(kw_conf, sem_conf) + 0.05)
            reason = f"Hybrid agree. {kw_reason} | {sem_reason}"
            intent = sem_intent
        elif kw_conf < 0.55:
            # Weak keywords, trust semantic
            intent, conf, reason = sem_intent, sem_conf, f"Hybrid semantic-led. {sem_reason}"
        else:
            # Conflict: prefer semantic if clearly stronger, else keyword
            if sem_conf >= kw_conf + 0.08:
                intent, conf = sem_intent, sem_conf
                reason = f"Hybrid conflict→semantic. {kw_reason} | {sem_reason}"
            else:
                intent, conf = kw_intent, kw_conf
                reason = f"Hybrid conflict→keyword. {kw_reason} | {sem_reason}"
    else:
        intent, conf, reason = kw_intent, kw_conf, f"Hybrid keyword-led (weak semantic). {kw_reason}"

    logger.info("Intent[hybrid]: %s (%.2f) — %s", intent, conf, reason)
    return intent, conf, reason
