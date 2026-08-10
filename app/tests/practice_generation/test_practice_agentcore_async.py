"""AgentCore tracked-task lifecycle tests for practice generation."""

from __future__ import annotations

import copy
import threading
from concurrent.futures import Future

import pytest
from bedrock_agentcore import BedrockAgentCoreApp

from features.practice_generation import agentcore_async
from features.practice_generation.agentcore_async import (
    AgentCorePracticeAsyncLauncher,
    PracticeLaunchError,
)
from features.practice_generation.planning import (
    deterministic_test_id,
    request_idempotency_key,
)
from features.practice_generation.schemas import PracticeGenerationRequest


def request(request_id: str = "request-1") -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id=request_id,
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        original_query="Create one algebra question",
        practice_type="QUICK_PRACTICE",
        requested_count=1,
        accepted_count=1,
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        assessment_title="Algebra Practice",
    )


class Tracker:
    def __init__(self, events: list[str], *, fail_registration: bool = False) -> None:
        self.events = events
        self.fail_registration = fail_registration
        self.added = 0
        self.completed = 0

    def add_async_task(self, name: str, metadata: dict | None = None) -> int:
        self.events.append("registered")
        if self.fail_registration:
            raise RuntimeError("registration failed")
        self.added += 1
        assert name == "practice_generation"
        assert metadata and metadata["testId"].startswith("practice-")
        return 17

    def complete_async_task(self, task_id: int) -> bool:
        assert task_id == 17
        self.events.append("completed")
        self.completed += 1
        return True


class BlockingTracker(Tracker):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.entered = threading.Event()
        self.release = threading.Event()

    def add_async_task(self, name: str, metadata: dict | None = None) -> int:
        self.entered.set()
        assert self.release.wait(timeout=1)
        return super().add_async_task(name, metadata)


class Assessments:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.item: dict = {}
        self.failed_code: str | None = None

    def create_or_get(self, test_id: str, value: PracticeGenerationRequest):
        if self.item:
            return self.item, True
        self.events.append("initialized")
        self.item = {
            "testId": test_id,
            "userId": value.user_id,
            "totalQuestions": value.accepted_count,
            "status": "GENERATING",
            "meta": {
                "playable": False,
                "readyCount": 0,
                "progressPercent": 0,
                "generationGroups": {},
            },
        }
        return self.item, False

    def get(self, _test_id: str):
        return self.item

    def mark_failed(self, _test_id: str, code: str) -> None:
        self.events.append("failed")
        self.failed_code = code
        self.item["status"] = "FAILED"
        self.item["meta"]["playable"] = False


class RecoveryProgress:
    def __init__(self, assessments: Assessments) -> None:
        self.assessments = assessments
        self.claims = 0

    def claim_recovery(self, _test_id: str) -> bool:
        meta = self.assessments.item["meta"]
        if int(meta.get("recoveryAttemptCount") or 0) >= 1:
            return False
        self.claims += 1
        meta["recoveryAttemptCount"] = 1
        meta["lastProgressAt"] = "2026-08-06T00:00:00+00:00"
        return True

    def mark_failed(self, test_id: str, code: str) -> None:
        self.assessments.mark_failed(test_id, code)


class BlockingFailureAssessments(Assessments):
    def __init__(self, events: list[str]) -> None:
        super().__init__(events)
        self.failure_entered = threading.Event()
        self.release_failure = threading.Event()

    def create_or_get(self, test_id: str, value: PracticeGenerationRequest):
        if self.item:
            return copy.deepcopy(self.item), True
        return super().create_or_get(test_id, value)

    def mark_failed(self, test_id: str, code: str) -> None:
        self.failure_entered.set()
        assert self.release_failure.wait(timeout=1)
        super().mark_failed(test_id, code)


class ReadyGraph:
    def __init__(self, assessments: Assessments, events: list[str]) -> None:
        self.assessments = assessments
        self.events = events

    def process(self, _command) -> None:
        self.events.append("graph")
        self.assessments.item["status"] = "READY"
        self.assessments.item["meta"].update(
            {"playable": True, "readyCount": 1, "progressPercent": 100}
        )


class FailingGraph:
    def process(self, _command) -> None:
        raise RuntimeError("graph failed")


class ImmediateExecutor:
    def submit(self, function, *args):
        function(*args)
        return Future()


class QueuedExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def submit(self, function, *args):
        self.calls.append((function, args))
        return Future()


class FailingExecutor:
    def submit(self, _function, *_args):
        raise RuntimeError("submit failed")


def test_stale_duplicate_gets_one_recovery_claim_then_second_stale_fails() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    value = request("request-a")
    test_id = deterministic_test_id(request_idempotency_key(value))
    assessments.item = {
        "testId": test_id,
        "userId": value.user_id,
        "totalQuestions": 1,
        "status": "GENERATING",
        "updatedAt": "2026-08-06T00:00:00+00:00",
        "meta": {
            "playable": False,
            "readyCount": 0,
            "progressPercent": 0,
            "generationGroups": {"g1": {"state": "RUNNING"}},
            "lastProgressAt": "2026-08-06T00:00:00+00:00",
            "recoveryAttemptCount": 0,
        },
    }
    progress = RecoveryProgress(assessments)
    first_tracker = Tracker(events)
    first = AgentCorePracticeAsyncLauncher(
        task_tracker=first_tracker,
        assessments=assessments,
        progress=progress,
        graph_runner=ReadyGraph(assessments, events),
        executor=QueuedExecutor(),
        recovery_stale_seconds=30,
    )

    recovered = first.launch(value)

    assert recovered.status == "GENERATING"
    assert recovered.duplicate_request is True
    assert progress.claims == 1
    assert first_tracker.added == 1

    assessments.item["meta"]["lastProgressAt"] = "2026-08-06T00:00:00+00:00"
    second_tracker = Tracker(events)
    second = AgentCorePracticeAsyncLauncher(
        task_tracker=second_tracker,
        assessments=assessments,
        progress=progress,
        graph_runner=ReadyGraph(assessments, events),
        executor=QueuedExecutor(),
        recovery_stale_seconds=30,
    )

    exhausted = second.launch(value)

    assert exhausted.status == "FAILED"
    assert assessments.failed_code == "PRACTICE_RECOVERY_EXHAUSTED"
    assert second_tracker.added == 0


def test_success_registers_once_and_persists_ready_before_completion() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=ImmediateExecutor(),
    )

    result = launcher.launch(request())
    launcher.start(result.test_id)

    assert result.test_id == deterministic_test_id(request_idempotency_key(request()))
    assert tracker.added == tracker.completed == 1
    assert assessments.item["status"] == "READY"
    assert events.index("graph") < events.index("completed")


def test_background_failure_persists_failed_before_completion() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=FailingGraph(),
        executor=ImmediateExecutor(),
    )

    result = launcher.launch(request())
    launcher.start(result.test_id)

    assert assessments.failed_code == "PRACTICE_GENERATION_FAILED"
    assert tracker.completed == 1
    assert events.index("failed") < events.index("completed")


def test_task_completion_reports_business_failure_not_execution_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    emitted: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(
        agentcore_async,
        "emit_practice_event",
        lambda event_name, **kwargs: emitted.append((event_name, kwargs.get("details"))),
    )
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=FailingGraph(),
        executor=ImmediateExecutor(),
    )

    launch = launcher.launch(request())
    launcher.start(launch.test_id)

    terminal = next(details for name, details in emitted if name == "practice_async_task_completed")
    assert terminal == {
        "executionStatus": "completed",
        "businessStatus": "FAILED",
        "acceptedCount": 0,
        "requestedCount": 1,
        "remainingCount": 1,
    }


def test_background_failure_emits_safe_exception_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    emitted: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(
        agentcore_async,
        "emit_practice_event",
        lambda event_name, **kwargs: emitted.append((event_name, kwargs.get("details"))),
    )
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=FailingGraph(),
        executor=ImmediateExecutor(),
    )

    launch = launcher.launch(request())
    launcher.start(launch.test_id)

    failure = next(details for name, details in emitted if name == "practice_async_task_failed")
    assert failure == {
        "reasonCode": "PRACTICE_GENERATION_FAILED",
        "errorClass": "RuntimeError",
    }


def test_background_repository_failure_preserves_only_its_safe_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    emitted: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(
        agentcore_async,
        "emit_practice_event",
        lambda event_name, **kwargs: emitted.append((event_name, kwargs.get("details"))),
    )

    class RepositoryFailingGraph:
        def process(self, _command) -> None:
            raise agentcore_async.PracticeRepositoryError(
                "PRACTICE_PROGRESS_GRAPHQL_REJECTED"
            )

    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=RepositoryFailingGraph(),
        executor=ImmediateExecutor(),
    )

    launch = launcher.launch(request())
    launcher.start(launch.test_id)

    failure = next(details for name, details in emitted if name == "practice_async_task_failed")
    assert failure == {
        "reasonCode": "PRACTICE_PROGRESS_GRAPHQL_REJECTED",
        "errorClass": "PracticeRepositoryError",
    }
    assert assessments.failed_code == "PRACTICE_PROGRESS_GRAPHQL_REJECTED"


def test_non_terminal_background_outcome_fails_closed_before_completion() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=ImmediateExecutor(),
    )
    launcher._run_to_terminal = lambda _test_id: "DELEGATED"  # type: ignore[method-assign]

    launcher._run_tracked(17, "test-1")

    assert assessments.failed_code == "PRACTICE_GENERATION_STALLED"
    assert tracker.completed == 1
    assert events.index("failed") < events.index("completed")


def test_duplicate_invocation_schedules_one_active_background_task() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    executor = QueuedExecutor()
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=executor,
    )

    first = launcher.launch(request())
    duplicate = launcher.launch(request())
    launcher.start(first.test_id)

    assert first.test_id == duplicate.test_id
    assert duplicate.duplicate_request is True
    assert tracker.added == 1
    assert len(executor.calls) == 1


def test_committed_delivery_protects_running_task_from_duplicate_abandon() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    executor = QueuedExecutor()
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=executor,
    )

    first = launcher.launch(request("request-a"))
    launcher.launch(request("request-b"))

    assert launcher.start(first.test_id, "request-a") is True
    launcher.abort(first.test_id, "DELIVERY_FAILED", "request-b")

    assert len(executor.calls) == 1
    assert assessments.failed_code is None
    assert tracker.completed == 0


def test_abandoned_duplicate_does_not_block_other_delivery_commit() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    executor = QueuedExecutor()
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=executor,
    )

    first = launcher.launch(request("request-a"))
    launcher.launch(request("request-b"))

    launcher.abort(first.test_id, "DELIVERY_FAILED", "request-a")
    assert launcher.start(first.test_id, "request-b") is True

    assert len(executor.calls) == 1
    assert assessments.failed_code is None


def test_final_abandoned_delivery_fails_pending_task_once() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=QueuedExecutor(),
    )

    first = launcher.launch(request("request-a"))
    launcher.launch(request("request-b"))

    launcher.abort(first.test_id, "DELIVERY_FAILED", "request-a")
    assert assessments.failed_code is None
    launcher.abort(first.test_id, "DELIVERY_FAILED", "request-b")
    launcher.abort(first.test_id, "DELIVERY_FAILED", "request-b")

    assert assessments.failed_code == "DELIVERY_FAILED"
    assert tracker.completed == 1
    assert events.count("failed") == 1


def test_unknown_delivery_and_post_start_duplicate_abort_are_noops() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    executor = QueuedExecutor()
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=executor,
    )

    launch = launcher.launch(request("request-a"))
    launcher.abort(launch.test_id, "DELIVERY_FAILED", "unknown")
    assert launcher.start(launch.test_id, "request-a") is True
    launcher.launch(request("request-b"))
    launcher.abort(launch.test_id, "DELIVERY_FAILED", "request-b")

    assert len(executor.calls) == 1
    assert assessments.failed_code is None


def test_ownerless_compatibility_requires_exactly_one_pending_delivery() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    executor = QueuedExecutor()
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=executor,
    )

    launch = launcher.launch(request("request-a"))
    assert launcher.start(launch.test_id) is True

    second_events: list[str] = []
    second_assessments = Assessments(second_events)
    second_launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=Tracker(second_events),
        assessments=second_assessments,
        progress=second_assessments,
        graph_runner=ReadyGraph(second_assessments, second_events),
        executor=QueuedExecutor(),
    )
    second = second_launcher.launch(request("request-a"))
    second_launcher.launch(request("request-b"))

    assert second_launcher.start(second.test_id) is False
    second_launcher.abort(second.test_id, "DELIVERY_FAILED")
    assert second_assessments.failed_code is None


def test_duplicate_registration_waits_for_consistent_task_visibility() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = BlockingTracker(events)
    executor = QueuedExecutor()
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=executor,
    )
    errors: list[BaseException] = []

    def launch(delivery_id: str) -> None:
        try:
            launcher.launch(request(delivery_id))
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=launch, args=("request-a",))
    duplicate = threading.Thread(target=launch, args=("request-b",))
    first.start()
    assert tracker.entered.wait(timeout=1)
    duplicate.start()
    assert duplicate.is_alive()
    tracker.release.set()
    first.join(timeout=1)
    duplicate.join(timeout=1)

    assert errors == []
    test_id = deterministic_test_id(request_idempotency_key(request("request-a")))
    assert launcher.start(test_id, "request-b") is True
    assert tracker.added == 1
    assert len(executor.calls) == 1


def test_final_abort_publishes_failure_before_stale_duplicate_can_register() -> None:
    events: list[str] = []
    assessments = BlockingFailureAssessments(events)
    tracker = Tracker(events)
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=QueuedExecutor(),
    )
    launch = launcher.launch(request("request-a"))
    duplicate_errors: list[BaseException] = []

    abort_thread = threading.Thread(
        target=launcher.abort,
        args=(launch.test_id, "DELIVERY_FAILED", "request-a"),
    )

    def launch_stale_duplicate() -> None:
        try:
            launcher.launch(request("request-b"))
        except BaseException as exc:
            duplicate_errors.append(exc)

    abort_thread.start()
    assert assessments.failure_entered.wait(timeout=1)
    duplicate_thread = threading.Thread(target=launch_stale_duplicate)
    duplicate_thread.start()
    assert duplicate_thread.is_alive()
    assessments.release_failure.set()
    abort_thread.join(timeout=1)
    duplicate_thread.join(timeout=1)

    assert len(duplicate_errors) == 1
    assert isinstance(duplicate_errors[0], PracticeLaunchError)
    assert tracker.added == tracker.completed == 1
    assert assessments.failed_code == "DELIVERY_FAILED"


def test_registration_failure_publishes_before_stale_duplicate_can_register() -> None:
    events: list[str] = []
    assessments = BlockingFailureAssessments(events)
    tracker = Tracker(events, fail_registration=True)
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=QueuedExecutor(),
    )
    launch_errors: list[BaseException] = []

    def launch(delivery_id: str) -> None:
        try:
            launcher.launch(request(delivery_id))
        except BaseException as exc:
            launch_errors.append(exc)

    first = threading.Thread(target=launch, args=("request-a",))
    duplicate = threading.Thread(target=launch, args=("request-b",))
    first.start()
    assert assessments.failure_entered.wait(timeout=1)
    duplicate.start()
    assert duplicate.is_alive()
    assessments.release_failure.set()
    first.join(timeout=1)
    duplicate.join(timeout=1)

    assert len(launch_errors) == 2
    assert all(isinstance(error, PracticeLaunchError) for error in launch_errors)
    assert tracker.added == tracker.completed == 0
    assert assessments.failed_code == "PRACTICE_ASYNC_TASK_REGISTRATION_FAILED"
    assert events.count("failed") == 1


def test_launch_registers_task_without_executing_until_start() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    executor = QueuedExecutor()
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=executor,
    )

    result = launcher.launch(request())

    assert tracker.added == 1
    assert executor.calls == []
    assert assessments.item["status"] == "GENERATING"
    assert launcher.start(result.test_id) is True
    assert len(executor.calls) == 1


def test_registration_failure_marks_assessment_failed_without_completion() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events, fail_registration=True)
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=FailingGraph(),
        executor=ImmediateExecutor(),
    )

    with pytest.raises(PracticeLaunchError) as raised:
        launcher.launch(request())

    assert raised.value.code == "PRACTICE_INITIALIZATION_FAILED"
    assert assessments.failed_code == "PRACTICE_ASYNC_TASK_REGISTRATION_FAILED"
    assert tracker.completed == 0


def test_submit_failure_marks_failed_then_completes_registered_task() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    tracker = Tracker(events)
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=tracker,
        assessments=assessments,
        progress=assessments,
        graph_runner=FailingGraph(),
        executor=FailingExecutor(),
    )

    launch = launcher.launch(request())
    with pytest.raises(PracticeLaunchError):
        launcher.start(launch.test_id)

    assert assessments.failed_code == "PRACTICE_ASYNC_TASK_START_FAILED"
    assert tracker.added == tracker.completed == 1
    assert events.index("failed") < events.index("completed")


def test_actual_agentcore_health_is_busy_until_tracked_task_completes() -> None:
    events: list[str] = []
    assessments = Assessments(events)
    executor = QueuedExecutor()
    app = BedrockAgentCoreApp()
    launcher = AgentCorePracticeAsyncLauncher(
        task_tracker=app,
        assessments=assessments,
        progress=assessments,
        graph_runner=ReadyGraph(assessments, events),
        executor=executor,
    )

    launch = launcher.launch(request())
    launcher.start(launch.test_id)
    assert app.get_current_ping_status().value == "HealthyBusy"

    function, args = executor.calls[0]
    function(*args)

    assert app.get_current_ping_status().value == "Healthy"
