"""Simple in-memory TTL cache for upstream service responses."""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any, Hashable

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe in-memory cache with per-key TTL."""

    def __init__(self, default_ttl: int = 300, maxsize: int = 1024) -> None:
        self._store: dict[Hashable, tuple[Any, float]] = {}
        self._default_ttl = default_ttl
        self._maxsize = maxsize
        self._lock = Lock()

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            value, expires_at = item
            if time.monotonic() > expires_at:
                del self._store[key]
                logger.debug("Cache miss (expired): %s", key)
                return None
            logger.debug("Cache hit: %s", key)
            return value

    def set(self, key: Hashable, value: Any, ttl: int | None = None) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if len(self._store) >= self._maxsize and key not in self._store:
                # Evict oldest by expiry
                oldest = min(self._store.items(), key=lambda kv: kv[1][1])
                del self._store[oldest[0]]
            self._store[key] = (value, time.monotonic() + ttl)

    def invalidate(self, key: Hashable) -> None:
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"size": len(self._store), "maxsize": self._maxsize}


# Shared singleton used by upstream clients
upstream_cache = TTLCache(default_ttl=300, maxsize=2048)
