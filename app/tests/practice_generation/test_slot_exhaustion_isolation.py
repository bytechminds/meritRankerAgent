"""One content-exhausted slot no longer aborts the other independent groups.

A group whose slot exhausted initial + repair + fresh regeneration is marked FAILED;
every other group still runs through the real scheduler, verified work is committed
and promoted, and the parent fails exactly once, after every executable group has
settled. Infrastructure/terminal stops keep their immediate behaviour.
"""

from __future__ import annotations

import threading

from features.practice_generation.agentcore_async import AgentCorePracticeAsyncLauncher
from features.practice_generation.execution_control import (
    ActivePracticeExecutionRegistry,
    bind_practice_execution,
)
from features.practice_generation.graph import PracticeGraphRunner
from features.practice_generation.planning import deterministic_blueprint
from features.practice_generation.schemas import VerificationResult
from tests.practice_generation.test_parallel_worker_ownership import (
    _CasProgress,
    _OverlapGenerator,
)
from tests.practice_generation.test_planner_first_generation import (
    build_orchestrator,
    request,
)

_TOPICS = [f"topic_{name}" for name in "abcdefghij"]


class _Generator(_OverlapGenerator):
    """Records every generation call; no barrier, since group counts vary here."""

    def __init__(self, progress: _CasProgress) -> None:
        super().__init__(progress)
        self.initial_barrier = type("NoBarrier", (), {"wait": staticmethod(lambda: None)})()


class _Verifier:
    """Rejects the listed slots NO_VALID_OPTION on every attempt; approves the rest."""

    def __init__(self, *, exhaust: set[str], terminal: set[str] = frozenset()) -> None:
        self._exhaust = exhaust
        self._terminal = terminal
        self._lock = threading.Lock()
        self.calls: dict[str, int] = {}

    def verify_slot(self, *, slot, question, **_kwargs):  # noqa: ANN001
        with self._lock:
            self.calls[slot.slot_id] = self.calls.get(slot.slot_id, 0) + 1
        common = {
            "schema_version": "2",
            "generation_item_id": question.generation_item_id,
            "slot_id": slot.slot_id,
        }
        if slot.slot_id in self._terminal:
            return VerificationResult(
                **common, decision="TERMINAL_REJECTION", valid_option_ids=[],
                reason_codes=["UNSAFE_CONTENT"],
            )
        if slot.slot_id in self._exhaust:
            return VerificationResult(
                **common, decision="REGENERATE", valid_option_ids=[],
                reason_codes=["NO_VALID_OPTION"],
            )
        return VerificationResult(
            **common, decision="ACCEPT", valid_option_ids=[question.correct_option_id],
            answer_explanation="The independently selected option is correct.",
            reason_codes=["SINGLE_VALID_OPTION"],
        )


def _assessment(assessments, *, groups: list[list[str]]) -> None:  # noqa: ANN001
    count = sum(len(slot_ids) for slot_ids in groups)
    topics = _TOPICS[:count]
    shaped = request().model_copy(
        update={"topics": topics, "accepted_count": count, "requested_count": count}
    )
    blueprint = deterministic_blueprint(shaped)
    bucket_by_slot = {
        slot.slot_id: bucket.bucket_id
        for slot, bucket in zip(blueprint.slots, blueprint.buckets, strict=True)
    }
    meta = assessments.item["meta"]
    meta["blueprint"] = blueprint.model_dump(mode="json")
    meta["generationGroups"] = {
        f"g{index}": {
            "groupId": f"g{index}",
            "bucketId": bucket_by_slot[slot_ids[0]],
            "requiredCount": len(slot_ids),
            "slotIds": slot_ids,
            "state": "PENDING",
        }
        for index, slot_ids in enumerate(groups, start=1)
    }
    meta["slotReadyCounts"] = {slot.slot_id: 0 for slot in blueprint.slots}
    meta["practiceRequest"].update(
        {"acceptedCount": count, "requestedCount": count, "topics": topics}
    )
    assessments.item["totalQuestions"] = count


def _drive(*, groups: list[list[str]], exhaust: set[str], terminal: set[str] = frozenset()):
    orchestrator, assessments = build_orchestrator(generator=None, verifier=None)
    _assessment(assessments, groups=groups)
    progress = _CasProgress(assessments, orchestrator._questions)
    failures: list[str] = []
    inner_mark_failed = progress.mark_failed

    def mark_failed(test_id, error_code, **kwargs):  # noqa: ANN001
        failures.append(error_code)
        return inner_mark_failed(test_id, error_code, **kwargs)

    progress.mark_failed = mark_failed
    generator = _Generator(progress)
    verifier = _Verifier(exhaust=exhaust, terminal=terminal)
    orchestrator._progress_updates = progress
    orchestrator._generator = generator
    orchestrator._verifier = verifier
    ready_finalizations: list[str] = []
    real_finalize = orchestrator._finalize

    def finalize(test_id, request, blueprint, *, attempt):  # noqa: ANN001
        groups_now = assessments.item["meta"]["generationGroups"].values()
        if all(group["state"] == "COMPLETED" for group in groups_now):
            # The full READY finalization is covered elsewhere; record it here.
            ready_finalizations.append(test_id)
            assessments.item["status"] = "READY"
            return None
        return real_finalize(test_id, request, blueprint, attempt=attempt)

    orchestrator._finalize = finalize
    launcher = object.__new__(AgentCorePracticeAsyncLauncher)
    launcher._assessments = assessments
    launcher._progress = progress
    launcher._graph_runner = PracticeGraphRunner(orchestrator)
    launcher._max_wall_time_seconds = 3_600
    with bind_practice_execution("execution-1", ActivePracticeExecutionRegistry()):
        status = launcher._run_to_terminal("test-v2")
    orchestrator.ready_finalizations = ready_finalizations
    return status, assessments, orchestrator, generator, verifier, progress, failures


def _groups_by_state(assessments) -> dict[str, list[str]]:  # noqa: ANN001
    result: dict[str, list[str]] = {}
    for group_id, group in assessments.item["meta"]["generationGroups"].items():
        result.setdefault(group["state"], []).append(group_id)
    return result


def test_case1_exhausted_group_does_not_stop_its_peer() -> None:
    status, assessments, orchestrator, generator, verifier, progress, failures = _drive(
        groups=[["slot-001"], ["slot-002"]], exhaust={"slot-001"}
    )

    assert status == "FAILED"
    assert failures == ["GENERATION_DEFICIT_EXHAUSTED"]
    assert set(orchestrator._questions.linked) == {"question-slot-002"}
    assert _groups_by_state(assessments) == {"FAILED": ["g1"], "COMPLETED": ["g2"]}
    assert verifier.calls == {"slot-001": 3, "slot-002": 1}
    assert progress.worker_writes == []


def test_case2_early_exhaustion_in_ten_groups_still_runs_the_other_nine() -> None:
    groups = [[f"slot-{index:03d}"] for index in range(1, 11)]

    status, assessments, orchestrator, generator, verifier, progress, failures = _drive(
        groups=groups, exhaust={"slot-001"}
    )

    assert failures == ["GENERATION_DEFICIT_EXHAUSTED"]
    assert {slot for slot, _wave in generator.waves} == {slot for [slot] in groups}
    assert len(orchestrator._questions.linked) == 9
    assert _groups_by_state(assessments)["FAILED"] == ["g1"]
    assert status == "FAILED"


def test_case3_multiple_exhaustions_fail_the_parent_once_with_the_aggregate_deficit() -> None:
    groups = [[f"slot-{index:03d}"] for index in range(1, 11)]
    exhausted = {"slot-002", "slot-005", "slot-009"}

    status, assessments, orchestrator, _generator, verifier, progress, failures = _drive(
        groups=groups, exhaust=exhausted
    )

    assert failures == ["GENERATION_DEFICIT_EXHAUSTED"]
    assert status == "FAILED"
    meta = assessments.item["meta"]
    unresolved = sorted(slot for slot, ready in meta["slotReadyCounts"].items() if not ready)
    assert unresolved == sorted(exhausted)
    assert len(orchestrator._questions.linked) == 7
    assert meta["readyQuestionIds"] == sorted(orchestrator._questions.linked)
    assert all(verifier.calls[slot] == 3 for slot in exhausted)


def test_case4_partial_group_keeps_its_accepted_question() -> None:
    status, assessments, orchestrator, _generator, _verifier, _progress, failures = _drive(
        groups=[["slot-001", "slot-002"], ["slot-003"]], exhaust={"slot-002"}
    )

    assert failures == ["GENERATION_DEFICIT_EXHAUSTED"]
    assert set(orchestrator._questions.linked) == {"question-slot-001", "question-slot-003"}
    group = assessments.item["meta"]["generationGroups"]["g1"]
    assert (group["state"], group["errorCode"]) == ("FAILED", "GENERATION_DEFICIT_EXHAUSTED")
    assert status == "FAILED"


def test_case5_terminal_rejection_still_stops_immediately() -> None:
    groups = [[f"slot-{index:03d}"] for index in range(1, 7)]

    status, _assessments, orchestrator, generator, _verifier, _progress, failures = _drive(
        groups=groups, exhaust=set(), terminal={"slot-001"}
    )

    assert status == "FAILED"
    assert failures == ["VERIFIER_TERMINAL_REJECTION"]
    assert len(generator.waves) <= 2, "no further group may start after a terminal stop"


def test_all_groups_succeeding_still_reach_normal_finalization() -> None:
    groups = [[f"slot-{index:03d}"] for index in range(1, 5)]

    _status, assessments, orchestrator, _generator, _verifier, progress, failures = _drive(
        groups=groups, exhaust=set()
    )

    assert failures == []
    assert _status == "READY"
    assert orchestrator.ready_finalizations == ["test-v2"]
    assert set(_groups_by_state(assessments)) == {"COMPLETED"}
    assert len(orchestrator._questions.linked) == 4
    assert progress.worker_writes == []
