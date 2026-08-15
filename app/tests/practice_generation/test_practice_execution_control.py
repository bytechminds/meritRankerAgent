from __future__ import annotations

import pytest

from features.practice_generation.execution_control import (
    ActivePracticeExecutionRegistry,
    bind_practice_execution,
    current_expensive_attempt_guard,
    current_practice_execution_id,
    local_cancellation_requested,
)
from features.practice_generation.progress import AppSyncAssessmentProgressRepository


class _Assessments:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.renewals = 0

    def renew_execution_lease(
        self,
        _test_id: str,
        _execution_id: str,
        _lease_expires_at: str,
    ) -> bool:
        self.renewals += 1
        return self.allowed


def test_registry_is_bounded_by_active_executions_and_cleans_every_path() -> None:
    registry = ActivePracticeExecutionRegistry()

    for index in range(500):
        execution_id = f"execution-{index}"
        registry.register(execution_id, f"test-{index}")
        registry.unregister(execution_id)

    assert len(registry) == 0


def test_local_cancel_signal_is_execution_scoped_and_disposable() -> None:
    registry = ActivePracticeExecutionRegistry()
    registry.register("execution-1", "test-1")

    with bind_practice_execution("execution-1", registry):
        assert current_practice_execution_id() == "execution-1"
        assert local_cancellation_requested() is False
        assert registry.request_cancel("test-1") is True
        assert local_cancellation_requested() is True

    registry.unregister("execution-1")
    assert current_practice_execution_id() is None
    assert len(registry) == 0


def test_optional_expensive_attempt_guard_is_execution_scoped_and_disposable() -> None:
    registry = ActivePracticeExecutionRegistry()

    def guard() -> None:
        return None

    assert current_expensive_attempt_guard() is None
    with bind_practice_execution(
        "execution-1",
        registry,
        expensive_attempt_guard=guard,
    ):
        assert current_expensive_attempt_guard() is guard
    assert current_expensive_attempt_guard() is None


def test_cancelled_execution_stops_before_the_next_expensive_boundary() -> None:
    assessments = _Assessments()
    progress = AppSyncAssessmentProgressRepository(
        assessments=assessments,  # type: ignore[arg-type]
        questions=object(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
    )
    registry = ActivePracticeExecutionRegistry()
    registry.register("execution-1", "test-1")

    with bind_practice_execution("execution-1", registry):
        registry.request_cancel("test-1")
        with pytest.raises(RuntimeError, match="PRACTICE_CANCEL_OBSERVED"):
            progress.require_expensive_work_allowed("test-1")

    assert assessments.renewals == 0


def test_stale_execution_is_fenced_when_lease_renewal_loses_ownership() -> None:
    assessments = _Assessments(allowed=False)
    progress = AppSyncAssessmentProgressRepository(
        assessments=assessments,  # type: ignore[arg-type]
        questions=object(),  # type: ignore[arg-type]
        client=object(),  # type: ignore[arg-type]
    )
    registry = ActivePracticeExecutionRegistry()
    registry.register("old-execution", "test-1")

    with bind_practice_execution("old-execution", registry):
        with pytest.raises(RuntimeError, match="PRACTICE_EXECUTION_FENCED"):
            progress.require_expensive_work_allowed("test-1")

    assert assessments.renewals == 1
