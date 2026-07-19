"""Bounded process-local cache and single-flight coordination."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from schemas.image_question_classification import ImageClassificationResult


@dataclass(frozen=True)
class _CacheEntry:
    expires_at: float
    value: ImageClassificationResult


class ImageClassificationCache:
    def __init__(self, *, ttl_seconds: int, max_entries: int = 256) -> None:
        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> ImageClassificationResult | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value.model_copy(update={"cache_hit": True}, deep=True)

    def put(self, key: str, value: ImageClassificationResult) -> None:
        if self._ttl_seconds <= 0:
            return
        with self._lock:
            self._entries[key] = _CacheEntry(
                expires_at=time.monotonic() + self._ttl_seconds,
                value=value.model_copy(update={"cache_hit": False}, deep=True),
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


class SingleFlight:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._flights: dict[str, _Flight] = {}

    def enter(self, key: str) -> tuple[bool, _Flight]:
        with self._lock:
            flight = self._flights.get(key)
            if flight is not None:
                flight.waiters += 1
                return False, flight
            flight = _Flight(event=threading.Event())
            self._flights[key] = flight
            return True, flight

    def wait(
        self,
        key: str,
        flight: _Flight,
        *,
        timeout_seconds: float,
    ) -> ImageClassificationResult | None:
        completed = flight.event.wait(timeout=timeout_seconds)
        with self._lock:
            result = flight.result.model_copy(deep=True) if completed and flight.result else None
            flight.waiters -= 1
            if flight.completed and flight.waiters == 0:
                self._flights.pop(key, None)
            return result

    def finish(
        self,
        key: str,
        flight: _Flight,
        result: ImageClassificationResult,
    ) -> None:
        with self._lock:
            flight.result = result.model_copy(deep=True)
            flight.completed = True
            if flight.waiters == 0:
                self._flights.pop(key, None)
            flight.event.set()


@dataclass
class _Flight:
    event: threading.Event
    result: ImageClassificationResult | None = None
    waiters: int = 0
    completed: bool = False
