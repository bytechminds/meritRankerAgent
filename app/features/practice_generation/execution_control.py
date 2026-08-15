"""Process-local cancellation handles and execution-scoped fencing identity."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


class PracticeExecutionStopped(RuntimeError):
    """The current execution no longer owns permission to start expensive work."""


@dataclass
class ActivePracticeExecution:
    execution_id: str
    test_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)


class ActivePracticeExecutionRegistry:
    """Tiny best-effort registry bounded by this process's active executions."""

    def __init__(self) -> None:
        self._items: dict[str, ActivePracticeExecution] = {}
        self._lock = threading.Lock()

    def register(self, execution_id: str, test_id: str) -> ActivePracticeExecution:
        handle = ActivePracticeExecution(execution_id=execution_id, test_id=test_id)
        with self._lock:
            self._items[execution_id] = handle
        return handle

    def request_cancel(self, test_id: str) -> bool:
        signalled = False
        with self._lock:
            for handle in self._items.values():
                if handle.test_id == test_id:
                    handle.cancel_event.set()
                    signalled = True
        return signalled

    def is_cancelled(self, execution_id: str) -> bool:
        with self._lock:
            handle = self._items.get(execution_id)
            return handle.cancel_event.is_set() if handle is not None else False

    def unregister(self, execution_id: str) -> None:
        with self._lock:
            self._items.pop(execution_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


_execution_id: ContextVar[str | None] = ContextVar(
    "practice_execution_id",
    default=None,
)
_registry: ContextVar[ActivePracticeExecutionRegistry | None] = ContextVar(
    "practice_execution_registry",
    default=None,
)
_expensive_attempt_guard: ContextVar[Callable[[], None] | None] = ContextVar(
    "practice_expensive_attempt_guard",
    default=None,
)


@contextmanager
def bind_practice_execution(
    execution_id: str,
    registry: ActivePracticeExecutionRegistry,
    *,
    expensive_attempt_guard: Callable[[], None] | None = None,
) -> Iterator[None]:
    execution_token = _execution_id.set(execution_id)
    registry_token = _registry.set(registry)
    guard_token = _expensive_attempt_guard.set(expensive_attempt_guard)
    try:
        yield
    finally:
        _expensive_attempt_guard.reset(guard_token)
        _registry.reset(registry_token)
        _execution_id.reset(execution_token)


def current_practice_execution_id() -> str | None:
    return _execution_id.get()


def local_cancellation_requested() -> bool:
    execution_id = _execution_id.get()
    registry = _registry.get()
    return bool(
        execution_id is not None
        and registry is not None
        and registry.is_cancelled(execution_id)
    )


def current_expensive_attempt_guard() -> Callable[[], None] | None:
    """Return the optional operation-scoped guard copied into Practice worker threads."""
    return _expensive_attempt_guard.get()
