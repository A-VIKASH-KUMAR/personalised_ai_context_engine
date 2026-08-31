"""Swappable LLM provider layer.

Default is a grounded mock that synthesizes answers from selected context
so the service runs without external API keys. OpenAI / Anthropic adapters
are available when keys are configured.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from app.config import settings
from app.models.schemas import LLMResult, PersonalizationConfig

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        config: PersonalizationConfig,
        context_payload: dict[str, Any],
    ) -> LLMResult:
        ...


class MockLLMProvider(BaseLLMProvider):
    """Deterministic, context-grounded mock responses for local / CI use."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: PersonalizationConfig,
        context_payload: dict[str, Any],
    ) -> LLMResult:
        start = time.perf_counter()
        answer = self._compose_answer(config, context_payload)
        latency_ms = (time.perf_counter() - start) * 1000
        confidence = self._confidence(config, context_payload)
        sources = list(config.selectedContext)
        # Prefer human-readable domain sources in the response
        domain_sources = [
            s
            for s in sources
            if s
            not in (
                "User Profile",
                "Birth Details",
            )
        ]
        if not domain_sources:
            domain_sources = sources

        return LLMResult(
            answer=answer,
            confidence=confidence,
            sourcesUsed=domain_sources,
            prompt_chars=sum(len(m.get("content", "")) for m in messages),
            latency_ms=latency_ms,
        )

    def _confidence(
        self,
        config: PersonalizationConfig,
        context_payload: dict[str, Any],
    ) -> str:
        n = len(config.selectedContext)
        has_core = any(
            k in context_payload
            for k in (
                "careerHoroscope",
                "relationshipHoroscope",
                "healthHoroscope",
                "financeHoroscope",
                "currentDasha",
                "house_10",
                "house_7",
                "house_6",
            )
        )
        if n >= 4 and has_core:
            return "HIGH"
        if n >= 2 and has_core:
            return "MEDIUM"
        return "LOW"

    def _compose_answer(
        self,
        config: PersonalizationConfig,
        ctx: dict[str, Any],
    ) -> str:
        name = (ctx.get("user") or {}).get("name", "friend")
        intent = config.intent
        tone = config.tone
        parts: list[str] = []

        greeting = {
            "motivational": f"{name}, the stars are offering a constructive window for you right now.",
            "practical": f"{name}, here is a clear read based on your current chart indicators.",
            "empathetic": f"{name}, I hear the weight behind this question—let’s look at what your chart highlights.",
            "direct": f"{name}, based on your current indicators:",
            "spiritual": f"{name}, the planetary currents around you suggest a meaningful phase of growth.",
        }.get(tone, f"{name}, here is guidance tailored to your chart.")

        parts.append(greeting)

        # Intent-specific synthesis from available context
        if intent == "career":
            parts.extend(self._career_lines(ctx))
        elif intent == "relationship":
            parts.extend(self._relationship_lines(ctx))
        elif intent == "health":
            parts.extend(self._health_lines(ctx))
        elif intent == "finance":
            parts.extend(self._finance_lines(ctx))
        else:
            parts.extend(self._general_lines(ctx))

        # Closing action aligned with tone
        closing = {
            "motivational": "Trust your preparation and take one bold, aligned step this period.",
            "practical": "Prioritize the highest-impact action in the next 2–4 weeks and reassess.",
            "empathetic": "Be kind to yourself as you navigate this; small steady steps matter.",
            "direct": "Act on the strongest signal and avoid over-complicating the decision.",
            "spiritual": "Align outer action with inner clarity; the timing supports mindful progress.",
        }.get(tone, "Move forward with awareness and intention.")
        parts.append(closing)

        text = " ".join(p.strip() for p in parts if p and p.strip())
        # Soft word limit
        words = text.split()
        if len(words) > config.maxWords:
            text = " ".join(words[: config.maxWords]) + "…"
        return text

    def _career_lines(self, ctx: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if "careerHoroscope" in ctx:
            lines.append(f"Career outlook: {ctx['careerHoroscope']}")
        if "house_10" in ctx:
            h = ctx["house_10"]
            lines.append(
                f"Your 10th house is governed by {h.get('lord')} with {h.get('strength', '').lower()} strength, "
                "which supports visibility and professional momentum when you lean into structured effort."
            )
        if "currentDasha" in ctx:
            d = ctx["currentDasha"]
            lines.append(
                f"The current {d.get('mahadasha')}–{d.get('antardasha')} dasha period tends to activate "
                "ambition and decisive career moves—useful if you are weighing a job change."
            )
        if "panchang" in ctx:
            p = ctx["panchang"]
            lines.append(
                f"Today’s Panchang ({p.get('tithi')}, Nakshatra {p.get('nakshatra')}, Yoga {p.get('yoga')}) "
                "favours constructive beginnings when paired with careful planning."
            )
        if not lines:
            lines.append(
                "Career indicators are limited in the available context; focus on skills growth "
                "and network quality while fuller chart data is refreshed."
            )
        return lines

    def _relationship_lines(self, ctx: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if "relationshipHoroscope" in ctx:
            lines.append(f"Relationship outlook: {ctx['relationshipHoroscope']}")
        if "house_7" in ctx:
            h = ctx["house_7"]
            lines.append(
                f"The 7th house lord {h.get('lord')} is currently {h.get('strength', '').lower()}, "
                "so partnership themes benefit from patience and clear communication."
            )
        if "moonSign" in ctx:
            lines.append(
                f"With Moon in {ctx['moonSign']}, emotional needs centre on security and honest expression."
            )
        if "currentDasha" in ctx:
            d = ctx["currentDasha"]
            lines.append(
                f"Under {d.get('mahadasha')}–{d.get('antardasha')}, relationship dynamics may feel more intense—"
                "channel that into constructive dialogue."
            )
        if not lines:
            lines.append("Relationship context is partial; prioritize empathy and listening this period.")
        return lines

    def _health_lines(self, ctx: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if "healthHoroscope" in ctx:
            lines.append(f"Health outlook: {ctx['healthHoroscope']}")
        if "house_6" in ctx:
            h = ctx["house_6"]
            lines.append(
                f"The 6th house (daily routine & wellness) is led by {h.get('lord')} "
                f"with {h.get('strength', '').lower()} strength—consistent habits matter more than intensity."
            )
        if "moonSign" in ctx:
            lines.append(
                f"Moon in {ctx['moonSign']} suggests sensitivity to stress; protect sleep and recovery windows."
            )
        if "panchang" in ctx:
            lines.append("Align heavier tasks with supportive Panchang days and keep rest non-negotiable.")
        if not lines:
            lines.append("Health signals are limited; default to sleep, hydration, and moderate activity.")
        return lines

    def _finance_lines(self, ctx: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        if "financeHoroscope" in ctx:
            lines.append(f"Finance outlook: {ctx['financeHoroscope']}")
        if "house_2" in ctx:
            h = ctx["house_2"]
            lines.append(
                f"2nd-house indicators (resources) show {h.get('lord')} as lord with "
                f"{h.get('strength', '').lower()} strength—favour steady accumulation over speculation."
            )
        if "currentDasha" in ctx:
            d = ctx["currentDasha"]
            lines.append(
                f"During {d.get('mahadasha')}–{d.get('antardasha')}, review budgets and avoid impulsive outflows."
            )
        if not lines:
            lines.append("Financial context is sparse; conserve capital and defer high-risk moves.")
        return lines

    def _general_lines(self, ctx: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        for key, label in (
            ("careerHoroscope", "Career"),
            ("relationshipHoroscope", "Relationships"),
            ("healthHoroscope", "Health"),
            ("financeHoroscope", "Finance"),
        ):
            if key in ctx:
                lines.append(f"{label}: {ctx[key]}")
        if "currentDasha" in ctx:
            d = ctx["currentDasha"]
            lines.append(
                f"Current dasha {d.get('mahadasha')}–{d.get('antardasha')} colours the overall theme of this phase."
            )
        if "panchang" in ctx:
            p = ctx["panchang"]
            lines.append(
                f"Today’s guidance rests on {p.get('tithi')} / {p.get('nakshatra')} with Yoga {p.get('yoga')}."
            )
        if "moonSign" in ctx:
            lines.append(f"Moon sign {ctx['moonSign']} frames emotional priorities for the period.")
        if not lines:
            lines.append("General guidance: stay consistent, protect energy, and act on the clearest opportunity.")
        return lines


class OpenAILLMProvider(BaseLLMProvider):
    """Optional OpenAI chat completions adapter."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def generate(
        self,
        messages: list[dict[str, str]],
        config: PersonalizationConfig,
        context_payload: dict[str, Any],
    ) -> LLMResult:
        import httpx

        start = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": max(256, config.maxWords * 2),
        }
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()

        answer = data["choices"][0]["message"]["content"].strip()
        latency_ms = (time.perf_counter() - start) * 1000
        n = len(config.selectedContext)
        confidence: str = "HIGH" if n >= 4 else ("MEDIUM" if n >= 2 else "LOW")
        return LLMResult(
            answer=answer,
            confidence=confidence,  # type: ignore[arg-type]
            sourcesUsed=[s for s in config.selectedContext if s != "User Profile"],
            prompt_chars=sum(len(m.get("content", "")) for m in messages),
            latency_ms=latency_ms,
        )


def get_llm_provider() -> BaseLLMProvider:
    provider = (settings.llm_provider or "mock").lower()
    if provider == "openai" and settings.openai_api_key:
        logger.info("Using OpenAI LLM provider model=%s", settings.llm_model)
        return OpenAILLMProvider(settings.openai_api_key, settings.llm_model)
    if provider != "mock":
        logger.warning("LLM provider '%s' unavailable or missing key; falling back to mock", provider)
    return MockLLMProvider()
