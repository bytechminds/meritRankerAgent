"""Bounded TTL cache for retrieval candidates, bundles, and rerank order."""

from __future__ import annotations

import hashlib
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class RetrievalCache(Generic[T]):
    """Small instance-scoped cache; callers own its lifecycle."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl_seconds = max(ttl_seconds, 1)
        self._values: dict[str, tuple[float, T]] = {}

    @staticmethod
    def key(*parts: str) -> str:
        joined = "\x1f".join(parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()

    def get(self, key: str) -> T | None:
        entry = self._values.get(key)
        if entry is None:
            return None
        created_at, value = entry
        if time.monotonic() - created_at > self._ttl_seconds:
            self._values.pop(key, None)
            return None
        return value

    def put(self, key: str, value: T) -> None:
        self._values[key] = (time.monotonic(), value)
