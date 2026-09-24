"""Parallel slot workers never write the shared parent assessment.

Reproduces the 2026-09-24 incident: a worker entering semantic repair published parent
progress while its peer renewed the execution lease, so the publish carried a stale
``updatedAt`` and optimistic concurrency failed the whole Practice with
``ASSESSMENT_CONCURRENT_UPDATE``. Only the coordinator may publish parent state.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace

import pytest

from features.practice_generation.execution_control import (
    ActivePracticeExecutionRegistry,
    bind_practice_execution,
)
from features.practice_generation.planning import deterministic_blueprint
from features.practice_generation.repositories import PracticeRepositoryError
from features.practice_generation.schemas import GeneratedBatch, VerificationResult
from tests.practice_generation.test_planner_first_generation import (
    Assessments,
    Progress,
    build_orchestrator,
    generated_question,
    request,
)

_WORKER_THREAD_PREFIX = "practice-slot-wave"
_OVERLAP_TIMEOUT = 1.0


def _in_worker() -> bool:
    return threading.current_thread().name.startswith(_WORKER_THREAD_PREFIX)


def _two_bucket_assessment(assessments: Assessments) -> None:
    two_topic = request().model_copy(
        update={"topics": ["algebra", "geometry"], "accepted_count": 2, "requested_count": 2}
    )
    blueprint = deterministic_blueprint(two_topic)
    assert len(blueprint.buckets) == 2
    meta = assessments.item["meta"]
    meta["blueprint"] = blueprint.model_dump(mode="json")
    meta["generationGroups"] = {
        f"g{index}": {
            "groupId": f"g{index}",
            "bucketId": blueprint.buckets[index - 1].bucket_id,
            "requiredCount": 1,
            "slotIds": [f"slot-{index:03d}"],
            "state": "PENDING",
        }
        for index in (1, 2)
    }
    meta["slotReadyCounts"] = {"slot-001": 0, "slot-002": 0}
    meta["practiceRequest"].update(
        {"acceptedCount": 2, "requestedCount": 2, "topics": ["algebra", "geometry"]}
    )
    assessments.item["totalQuestions"] = 2


class _CasProgress(Progress):
    """Mirrors the real contract: lease renewal bumps ``updatedAt`` and a publish
    fails when ``updatedAt`` moved between its read and its write."""

    def __init__(self, assessments, questions) -> None:  # noqa: ANN001
        super().__init__(assessments, questions)
        self.lock = threading.Lock()
        self.writer_in_publish = threading.Event()
        self.peer_renewed = threading.Event()
        self.worker_writes: list[str] = []

    def require_expensive_work_allowed(self, test_id: str) -> None:
        super().require_expensive_work_allowed(test_id)
        with self.lock:
            self.assessments.item["updatedAt"] = str(int(self.assessments.item["updatedAt"]) + 1)
        if self.writer_in_publish.is_set():
            self.peer_renewed.set()

    def update(self, test_id: str, *, meta_updates, **kwargs):  # noqa: ANN001
        read_version = self.assessments.item["updatedAt"]
        if _in_worker():
            self.worker_writes.append(str(meta_updates.get("progressMessageKey")))
            self.writer_in_publish.set()
            self.peer_renewed.wait(_OVERLAP_TIMEOUT)
        with self.lock:
            if self.assessments.item["updatedAt"] != read_version:
                raise PracticeRepositoryError("ASSESSMENT_CONCURRENT_UPDATE")
            result = super().update(test_id, meta_updates=meta_updates, **kwargs)
            self.assessments.item["updatedAt"] = str(int(self.assessments.item["updatedAt"]) + 1)
            return result


class _OverlapGenerator:
    """slot-002 waits until slot-001's worker is inside a parent publish (or times out)."""

    def __init__(self, progress: _CasProgress) -> None:
        self._progress = progress
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.waves: list[tuple[str, int]] = []
        # Both initial generations must be in flight at once to pass this barrier,
        # which proves the two workers really run concurrently.
        self.initial_barrier = threading.Barrier(2, timeout=5)

    def generate_slots(self, *, bucket, slots, replacement_wave, **_kwargs):  # noqa: ANN001
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.waves.append((slots[0].slot_id, replacement_wave))
        try:
            if replacement_wave == 0:
                self.initial_barrier.wait()
            if slots[0].slot_id == "slot-002":
                self._progress.writer_in_publish.wait(_OVERLAP_TIMEOUT)
            questions = []
            for slot in slots:
                question = generated_question(bucket=bucket, slot=slot)
                question["question"] = f"Wave {replacement_wave} item for {slot.slot_id}?"
                question["answer_explanation"] = ""
                question["solution"] = ""
                questions.append(question)
            return GeneratedBatch(
                content=json.dumps({"questions": questions}),
                route_id=slots[0].generator_route_hint,
                model="test-model",
            )
        finally:
            with self._lock:
                self.active -= 1


class _SlotOneRecoversVerifier:
    """Each slot is rejected ``NO_VALID_OPTION`` its configured number of times."""

    def __init__(self, *, reject: dict[str, int]) -> None:
        self._reject = reject
        self._lock = threading.Lock()
        self.calls: dict[str, int] = {}

    def verify_slot(self, *, slot, question, **_kwargs):  # noqa: ANN001
        with self._lock:
            calls = self.calls[slot.slot_id] = self.calls.get(slot.slot_id, 0) + 1
        if calls <= self._reject.get(slot.slot_id, 0):
            return VerificationResult(
                schema_version="2",
                generation_item_id=question.generation_item_id,
                slot_id=slot.slot_id,
                decision="REGENERATE",
                valid_option_ids=[],
                reason_codes=["NO_VALID_OPTION"],
            )
        return VerificationResult(
            schema_version="2",
            generation_item_id=question.generation_item_id,
            slot_id=slot.slot_id,
            decision="ACCEPT",
            valid_option_ids=[question.correct_option_id],
            answer_explanation="The independently selected option is correct.",
            reason_codes=["SINGLE_VALID_OPTION"],
        )


def _run(*, reject: dict[str, int]):
    orchestrator, assessments = build_orchestrator(generator=None, verifier=None)
    _two_bucket_assessment(assessments)
    progress = _CasProgress(assessments, orchestrator._questions)
    generator = _OverlapGenerator(progress)
    verifier = _SlotOneRecoversVerifier(reject=reject)
    orchestrator._progress_updates = progress
    orchestrator._generator = generator
    orchestrator._verifier = verifier
    orchestrator._config = replace(orchestrator._config, question_bank_promotion_enabled=True)
    orchestrator.promoted = []

    def promote(*, slot, question, **_kwargs):  # noqa: ANN001
        orchestrator.promoted.append((slot.slot_id, question.question))
        return f"qb-{slot.slot_id}"

    orchestrator._questions.persist_verified_question = promote
    orchestrator.finalized = []
    orchestrator._maybe_finalize = lambda test_id, *_a, **_k: orchestrator.finalized.append(test_id)
    with bind_practice_execution("execution-1", ActivePracticeExecutionRegistry()):
        orchestrator.generate_wave("test-v2", ["g1", "g2"])
    return orchestrator, assessments, progress, generator, verifier


@pytest.mark.parametrize(
    ("reject", "recovered_wave"),
    [pytest.param(1, 1, id="semantic-repair"), pytest.param(2, 2, id="fresh-replacement")],
)
def test_recovering_worker_and_passing_peer_both_commit_without_a_parent_race(
    reject: int, recovered_wave: int
) -> None:
    orchestrator, assessments, progress, generator, verifier = _run(reject={"slot-001": reject})

    assert progress.worker_writes == []
    assert generator.max_active == 2
    assert ("slot-001", recovered_wave) in generator.waves
    assert verifier.calls == {"slot-001": reject + 1, "slot-002": 1}
    linked = orchestrator._questions.linked
    assert set(linked) == {"question-slot-001", "question-slot-002"}
    assert linked["question-slot-001"]["question"].startswith(f"Wave {recovered_wave}")
    assert linked["question-slot-002"]["question"].startswith("Wave 0")
    groups = assessments.item["meta"]["generationGroups"]
    assert {groups["g1"]["state"], groups["g2"]["state"]} == {"COMPLETED"}
    assert sorted(assessments.item["meta"]["readyQuestionIds"]) == [
        "question-slot-001",
        "question-slot-002",
    ]
    assert assessments.item["status"] == "GENERATING"
    assert orchestrator.finalized == ["test-v2"]
    # Only the final, independently accepted candidate of each slot is promoted.
    assert sorted(orchestrator.promoted) == [
        ("slot-001", f"Wave {recovered_wave} item for slot-001?"),
        ("slot-002", "Wave 0 item for slot-002?"),
    ]


@pytest.mark.parametrize(
    "reject",
    [
        pytest.param({}, id="both-first-pass"),
        pytest.param({"slot-001": 1, "slot-002": 2}, id="both-recover"),
    ],
)
def test_concurrent_workers_only_return_outcomes_for_the_coordinator(reject) -> None:  # noqa: ANN001
    orchestrator, assessments, progress, generator, verifier = _run(reject=reject)

    assert progress.worker_writes == []
    assert generator.max_active == 2
    assert verifier.calls == {slot: count + 1 for slot, count in
                              {"slot-001": 0, "slot-002": 0, **reject}.items()}
    assert set(orchestrator._questions.linked) == {"question-slot-001", "question-slot-002"}
    groups = assessments.item["meta"]["generationGroups"]
    assert {groups["g1"]["state"], groups["g2"]["state"]} == {"COMPLETED"}
