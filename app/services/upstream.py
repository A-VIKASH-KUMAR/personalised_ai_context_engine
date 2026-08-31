"""Concurrent upstream service clients with retry, timeout, caching, and partial failure handling."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable, Coroutine

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.cache.memory import upstream_cache
from app.config import settings
from app.models.schemas import (
    BirthDetails,
    DashaInfo,
    HoroscopeData,
    HouseInfo,
    KundliData,
    PanchangData,
    UpstreamBundle,
    UserProfile,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock data stores (stand-in for real HTTP microservices)
# ---------------------------------------------------------------------------

_MOCK_USERS: dict[str, dict[str, Any]] = {
    "user_101": {
        "id": "user_101",
        "name": "Aarav Sharma",
        "language": "en",
        "subscription": "premium",
        "tonePreference": "motivational",
        "birthDetails": {
            "date": "1997-08-15",
            "time": "09:35",
            "place": "Delhi",
        },
    },
    "user_102": {
        "id": "user_102",
        "name": "Priya Patel",
        "language": "hi",
        "subscription": "free",
        "tonePreference": "empathetic",
        "birthDetails": {
            "date": "1992-03-22",
            "time": "14:10",
            "place": "Mumbai",
        },
    },
    "user_103": {
        "id": "user_103",
        "name": "Rohan Mehta",
        "language": "en",
        "subscription": "premium",
        "tonePreference": "practical",
        "birthDetails": {
            "date": "1988-11-05",
            "time": "06:45",
            "place": "Bangalore",
        },
    },
    "user_104": {
        "id": "user_104",
        "name": "vikash kumar",
        "language": "en",
        "subscription": "premium",
        "tonePreference": "practical",
        "birthDetails": {
            "date": "1995-04-04",
            "time": "20:00",
            "place": "vizainagaram",
        },
    },
}

_MOCK_KUNDLI: dict[str, dict[str, Any]] = {
    "user_101": {
        "lagna": "Libra",
        "moonSign": "Scorpio",
        "currentDasha": {"mahadasha": "Rahu", "antardasha": "Mars"},
        "houses": {
            "2": {"lord": "Mars", "strength": "Average"},
            "6": {"lord": "Jupiter", "strength": "Average"},
            "7": {"lord": "Mars", "strength": "Weak"},
            "10": {"lord": "Moon", "strength": "Strong"},
        },
    },
    "user_102": {
        "lagna": "Cancer",
        "moonSign": "Taurus",
        "currentDasha": {"mahadasha": "Venus", "antardasha": "Mercury"},
        "houses": {
            "2": {"lord": "Sun", "strength": "Strong"},
            "6": {"lord": "Jupiter", "strength": "Weak"},
            "7": {"lord": "Saturn", "strength": "Average"},
            "10": {"lord": "Mars", "strength": "Strong"},
        },
    },
    "user_103": {
        "lagna": "Capricorn",
        "moonSign": "Leo",
        "currentDasha": {"mahadasha": "Saturn", "antardasha": "Jupiter"},
        "houses": {
            "2": {"lord": "Uranus", "strength": "Average"},
            "6": {"lord": "Mercury", "strength": "Strong"},
            "7": {"lord": "Moon", "strength": "Strong"},
            "10": {"lord": "Venus", "strength": "Average"},
        },
    },
    "user_104": {
        "lagna": "scorpio",
        "moonsign": "taurus",
        "currentDasha": {"mahadasha": "rahu"},
        "houses": {
            "2": {"lord": "jupiter", "strength": "strong"},
            "9": {"lord": "moon", "strength": "strong"},
        },
    },
}

_MOCK_HOROSCOPE: dict[str, dict[str, str]] = {
    "user_101": {
        "career": "Networking may bring new opportunities. A senior contact could open doors this period.",
        "finance": "Avoid risky investments. Steady savings and reviewing existing portfolios are favoured.",
        "health": "Prioritize proper sleep. Mild stress-related fatigue is possible; rest and light exercise help.",
        "relationship": "Communication with your partner improves. Honest conversations deepen bonds.",
    },
    "user_102": {
        "career": "A collaborative project may highlight your skills. Stay open to lateral moves.",
        "finance": "Unexpected expenses possible mid-month. Keep an emergency buffer ready.",
        "health": "Digestive sensitivity noted. Favour light meals and hydration.",
        "relationship": "Emotional closeness grows. Plan quality time without digital distractions.",
    },
    "user_103": {
        "career": "Leadership responsibilities increase. Document achievements for upcoming reviews.",
        "finance": "Long-term investments look favourable. Avoid speculative short trades.",
        "health": "Energy levels rise after mid-week. Maintain consistent sleep schedule.",
        "relationship": "Harmony at home supports overall well-being. Express appreciation openly.",
    },
}


def _ensure_default_astrology_data(user_id: str) -> None:
    if user_id in _MOCK_KUNDLI and user_id in _MOCK_HOROSCOPE:
        return
    lagnas = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
    ]
    lords = ["Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn"]
    strengths = ["Strong", "Average", "Weak"]
    dashas = [
        {"mahadasha": "Jupiter", "antardasha": "Saturn"},
        {"mahadasha": "Venus", "antardasha": "Mercury"},
        {"mahadasha": "Saturn", "antardasha": "Jupiter"},
        {"mahadasha": "Mercury", "antardasha": "Ketu"},
    ]
    _MOCK_KUNDLI[user_id] = {
        "lagna": random.choice(lagnas),
        "moonSign": random.choice(lagnas),
        "currentDasha": random.choice(dashas),
        "houses": {
            "2": {"lord": random.choice(lords), "strength": random.choice(strengths)},
            "6": {"lord": random.choice(lords), "strength": random.choice(strengths)},
            "7": {"lord": random.choice(lords), "strength": random.choice(strengths)},
            "10": {"lord": random.choice(lords), "strength": random.choice(strengths)},
        },
    }
    _MOCK_HOROSCOPE[user_id] = {
        "career": "Career outlook: new opportunities may arise. Stay aligned with your long-term goals.",
        "finance": "Financial outlook: focus on steady growth and avoid speculative risks.",
        "health": "Health outlook: maintain consistent routines and prioritize rest.",
        "relationship": "Relationship outlook: open communication fosters deeper connections.",
    }
    logger.info("Generated default astrology data for user_id=%s", user_id)


class UpstreamError(Exception):
    """Raised when a mocked upstream call fails (used for retry testing)."""


async def _simulate_latency(min_ms: float = 40, max_ms: float = 180) -> None:
    await asyncio.sleep(random.uniform(min_ms, max_ms) / 1000.0)


async def _fetch_with_retry(
    name: str,
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    cache_key: str | None = None,
    cache_ttl: int | None = None,
) -> Any:
    """Execute an upstream call with optional cache, timeout, and retries."""
    if cache_key:
        cached = upstream_cache.get(cache_key)
        if cached is not None:
            return cached

    timeout = settings.upstream_timeout_seconds
    max_attempts = settings.upstream_max_retries + 1

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(
            multiplier=settings.upstream_retry_backoff,
            min=0.1,
            max=2.0,
        ),
        retry=retry_if_exception_type((UpstreamError, asyncio.TimeoutError, OSError)),
        reraise=True,
    ):
        with attempt:
            try:
                result = await asyncio.wait_for(coro_factory(), timeout=timeout)
                if cache_key and result is not None:
                    upstream_cache.set(cache_key, result, ttl=cache_ttl)
                return result
            except asyncio.TimeoutError:
                logger.warning("Upstream %s timed out (attempt %s)", name, attempt.retry_state.attempt_number)
                raise
            except UpstreamError as exc:
                logger.warning("Upstream %s failed: %s (attempt %s)", name, exc, attempt.retry_state.attempt_number)
                raise


# --- Individual service mocks ------------------------------------------------


async def get_user(user_id: str) -> UserProfile:
    async def _call() -> UserProfile:
        await _simulate_latency()
        raw = _MOCK_USERS.get(user_id)
        if raw is None:
            from app.services.auth import get_user_by_id
            auth_user = await get_user_by_id(user_id)
            if auth_user:
                birth_details = None
                if auth_user.get("date_of_birth") and auth_user.get("time_of_birth"):
                    birth_details = BirthDetails(
                        date=auth_user["date_of_birth"],
                        time=auth_user["time_of_birth"],
                        place=auth_user.get("place_of_birth", "Unknown"),
                    )
                return UserProfile(
                    id=auth_user["id"],
                    name=auth_user["name"],
                    language="en",
                    subscription="free",
                    tonePreference="motivational",
                    birthDetails=birth_details,
                )
            logger.warning("User %s not found; using minimal fallback profile", user_id)
            return UserProfile(
                id=user_id,
                name="Guest",
                language="en",
                subscription="free",
                tonePreference="motivational",
            )
        return UserProfile(
            id=raw["id"],
            name=raw["name"],
            language=raw.get("language", "en"),
            subscription=raw.get("subscription", "free"),
            tonePreference=raw.get("tonePreference", "motivational"),
            birthDetails=BirthDetails(**raw["birthDetails"]) if raw.get("birthDetails") else None,
        )

    return await _fetch_with_retry(
        "UserService",
        _call,
        cache_key=f"user:{user_id}",
        cache_ttl=settings.cache_ttl_user,
    )


async def get_kundli(user_id: str) -> KundliData | None:
    async def _call() -> KundliData | None:
        await _simulate_latency()
        _ensure_default_astrology_data(user_id)
        raw = _MOCK_KUNDLI.get(user_id)
        if raw is None:
            return None
        houses = {k: HouseInfo(**v) for k, v in raw["houses"].items()}
        return KundliData(
            lagna=raw["lagna"],
            moonSign=raw["moonSign"],
            currentDasha=DashaInfo(**raw["currentDasha"]),
            houses=houses,
        )

    return await _fetch_with_retry(
        "KundliService",
        _call,
        cache_key=f"kundli:{user_id}",
        cache_ttl=settings.cache_ttl_kundli,
    )


async def get_horoscope(user_id: str) -> HoroscopeData | None:
    async def _call() -> HoroscopeData | None:
        await _simulate_latency()
        _ensure_default_astrology_data(user_id)
        raw = _MOCK_HOROSCOPE.get(user_id)
        if raw is None:
            return None
        return HoroscopeData(**raw)

    return await _fetch_with_retry(
        "HoroscopeService",
        _call,
        cache_key=f"horoscope:{user_id}",
        cache_ttl=settings.cache_ttl_horoscope,
    )


async def get_panchang() -> PanchangData:
    async def _call() -> PanchangData:
        async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
            try:
                response = await client.get("/panchang")
                response.raise_for_status()
                data = response.json()

                date_str = data.get("date", "")
                if isinstance(date_str, str) and "T" in date_str:
                    date_str = date_str.split("T")[0]

                return PanchangData(
                    date=date_str,
                    tithi=data.get("tithiAtSunrise", ""),
                    nakshatra=data.get("nakshatraAtSunrise", ""),
                    yoga=data.get("yogaAtSunrise", ""),
                    karana=data.get("karanaAtSunrise", "")
                )
            except Exception as e:
                logger.error("Failed to fetch panchang from /panchang endpoint: %s", e)
                raise UpstreamError(f"Panchang fetch failed: {e}")

    return await _fetch_with_retry(
        "PanchangService",
        _call,
        cache_key="panchang:today",
        cache_ttl=settings.cache_ttl_panchang,
    )


async def fetch_all_context(user_id: str) -> UpstreamBundle:
    """
    Fetch all upstream services concurrently.
    Partial failures are captured in `errors`; successful data is still returned.
    """
    logger.info("Fetching upstream context for user_id=%s", user_id)
    start = asyncio.get_event_loop().time()

    results = await asyncio.gather(
        get_user(user_id),
        get_kundli(user_id),
        get_horoscope(user_id),
        get_panchang(),
        return_exceptions=True,
    )

    bundle = UpstreamBundle()
    names = ["user", "kundli", "horoscope", "panchang"]

    for name, result in zip(names, results):
        if isinstance(result, Exception):
            logger.error("Upstream %s failed permanently: %s", name, result)
            bundle.errors[name] = str(result)
        else:
            setattr(bundle, name, result)

    elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000
    logger.info(
        "Upstream fetch complete for %s in %.1fms (errors=%s)",
        user_id,
        elapsed_ms,
        list(bundle.errors.keys()) or "none",
    )
    return bundle
