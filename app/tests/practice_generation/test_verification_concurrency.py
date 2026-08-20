"""Stage 1: bounded parallel execution of the existing per-question verifier.

Transport only. The verifier route, model, prompt, response schema, one-question
isolation, and rejection taxonomy are unchanged — these tests pin the concurrency
mechanics and the identity contract, not verification semantics.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from features.practice_generation.orchestration import PracticeGenerationOrchestrator
from features.practice_generation.schemas import (
    Complexity,
    DemandBucket,
    Difficulty,
    GeneratedQuestion,
    PlannerSlot,
    PracticeGenerationRequest,
    QuestionType,
    VerificationDecision,
    VerificationPolicy,
    VerificationResult,
)
from services.llm.orchestration.errors import ProviderExecutionError


class _RecordingVerifier:
    """Records overlap and completion order without changing verifier semantics."""

    def __init__(
        self,
        *,
        delay: float = 0.05,
        reject_slots: frozenset[str] = frozenset(),
        raise_for: dict[str, BaseException] | None = None,
        completion_delays: dict[str, float] | None = None,
    ) -> None:
        self._delay = delay
        self._reject_slots = reject_slots
        self._raise_for = raise_for or {}
        self._completion_delays = completion_delays or {}
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self.completion_order: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0

    def verify_slot(
        self,
        *,
        request: Any,
        bucket: Any,
        slot: PlannerSlot,
        question: GeneratedQuestion,
    ) -> VerificationResult:
        del request, bucket
        with self._lock:
            self.calls.append(slot.slot_id)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self._completion_delays.get(slot.slot_id, self._delay))
            if slot.slot_id in self._raise_for:
                raise self._raise_for[slot.slot_id]
            if slot.slot_id in self._reject_slots:
                return VerificationResult(
                    schema_version="2",
                    generation_item_id=question.generation_item_id,
                    slot_id=slot.slot_id,
                    decision=VerificationDecision.REGENERATE,
                    independently_solved_option_id="1",
                    reason_codes=["INCORRECT_KEY"],
                )
            return VerificationResult(
                schema_version="2",
                generation_item_id=question.generation_item_id,
                slot_id=slot.slot_id,
                decision=VerificationDecision.ACCEPT,
                independently_solved_option_id=question.correct_option_id,
                reason_codes=["MATCH"],
            )
        finally:
            with self._lock:
                self.in_flight -= 1
                self.completion_order.append(slot.slot_id)


def _slot(index: int) -> PlannerSlot:
    return PlannerSlot(
        slot_id=f"slot-{index:03d}",
        subject_id="math",
        topic_id="algebra",
        category_id="algebra",
        difficulty=Difficulty.INTERMEDIATE,
        complexity=Complexity.MEDIUM,
        exam_ids=["CAT"],
        question_type=QuestionType.MCQ,
        target_skill="solve_linear_equation",
        variation_hint=f"v{index}",
        generator_route_hint="quant.generator",
        reasoning_target="isolate_variable",
    )


def _question(index: int) -> GeneratedQuestion:
    return GeneratedQuestion(
        schema_version="2",
        generation_item_id=f"item-{index:03d}",
        bucket_id="bucket-1",
        slot_id=f"slot-{index:03d}",
        question=f"Solve equation variant {index}.",
        question_type=QuestionType.MCQ,
        options=[
            {"option_id": "0", "value": "1"},
            {"option_id": "1", "value": "2"},
            {"option_id": "2", "value": "3"},
            {"option_id": "3", "value": "4"},
        ],
        correct_option_id="1",
        correct_answer="2",
        answer_explanation="Isolate the variable.",
        solution="Isolate the variable.",
        subject="math",
        topic="algebra",
        difficulty=Difficulty.INTERMEDIATE,
    )


def _bucket() -> DemandBucket:
    return DemandBucket(
        bucket_id="bucket-1",
        subject="math",
        topic="algebra",
        difficulty=Difficulty.INTERMEDIATE,
        question_type=QuestionType.MCQ,
        required_count=1,
        question_intent="Solve a linear equation for the unknown.",
        verification_policy=VerificationPolicy.MANDATORY,
    )


def _request() -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        original_query="Create algebra questions",
        practice_type="QUIZ",
        requested_count=5,
        accepted_count=5,
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        assessment_title="Algebra practice",
    )


def _orchestrator(verifier: Any, *, max_concurrency: int = 3) -> Any:
    """Build only what _verify_slots_bounded touches, with no cancellation binding.

    _require_expensive_work_allowed short-circuits when no practice execution is
    bound (orchestration.py:241), so _progress_updates is never reached here.
    """
    from features.practice_generation.config import PracticeGenerationConfig

    orchestrator = object.__new__(PracticeGenerationOrchestrator)
    orchestrator._config = PracticeGenerationConfig(
        enabled=True,
        pattern_context_enabled=False,
        pattern_reuse_enabled=False,
        assessment_table="assessment",
        question_table="question",
        question_bank_table="bank",
        question_bank_category_index="category-index",
        question_test_index="test-index",
        aws_region="ap-south-1",
        verification_max_concurrency=max_concurrency,
    )
    orchestrator._verifier = verifier
    orchestrator._progress_updates = None
    return orchestrator


def _context(orchestrator: Any, count: int) -> Any:
    """Minimal stand-in: _verify_slots_bounded reads only test_id, request, bucket."""
    del orchestrator, count

    class _Context:
        test_id = "practice-verify-1"
        request = _request()
        bucket = _bucket()

    return _Context()


def _units(count: int) -> list[tuple[PlannerSlot, GeneratedQuestion]]:
    return [(_slot(i), _question(i)) for i in range(1, count + 1)]


def test_single_question_produces_exactly_one_verifier_call() -> None:
    verifier = _RecordingVerifier()
    orchestrator = _orchestrator(verifier)

    outcomes = orchestrator._verify_slots_bounded(
        context=_context(orchestrator, 1), units=_units(1)
    )

    assert verifier.calls == ["slot-001"]
    assert verifier.max_in_flight == 1
    assert set(outcomes) == {"slot-001"}


def test_two_questions_produce_two_calls_that_overlap() -> None:
    verifier = _RecordingVerifier(delay=0.08)
    orchestrator = _orchestrator(verifier, max_concurrency=2)

    started = time.perf_counter()
    outcomes = orchestrator._verify_slots_bounded(
        context=_context(orchestrator, 2), units=_units(2)
    )
    elapsed = time.perf_counter() - started

    assert len(verifier.calls) == 2
    assert verifier.max_in_flight == 2, "the two verifier calls did not overlap"
    assert elapsed < 0.16, "parallel calls should not sum sequentially"
    assert set(outcomes) == {"slot-001", "slot-002"}


def test_five_questions_respect_the_configured_concurrency_bound() -> None:
    verifier = _RecordingVerifier(delay=0.04)
    orchestrator = _orchestrator(verifier, max_concurrency=3)

    outcomes = orchestrator._verify_slots_bounded(
        context=_context(orchestrator, 5), units=_units(5)
    )

    assert len(verifier.calls) == 5
    assert verifier.max_in_flight <= 3, f"bound exceeded: {verifier.max_in_flight}"
    assert verifier.max_in_flight > 1, "no parallelism observed"
    assert len(outcomes) == 5


@pytest.mark.parametrize("bound", [1, 2, 3, 4, 5])
def test_concurrency_never_exceeds_the_configured_bound(bound: int) -> None:
    verifier = _RecordingVerifier(delay=0.03)
    orchestrator = _orchestrator(verifier, max_concurrency=bound)

    orchestrator._verify_slots_bounded(context=_context(orchestrator, 5), units=_units(5))

    assert verifier.max_in_flight <= bound


def test_completion_order_does_not_determine_identity() -> None:
    """Slot 5 finishes first, slot 1 last; every result must still map correctly."""
    verifier = _RecordingVerifier(
        completion_delays={
            "slot-001": 0.12,
            "slot-002": 0.09,
            "slot-003": 0.06,
            "slot-004": 0.03,
            "slot-005": 0.01,
        }
    )
    orchestrator = _orchestrator(verifier, max_concurrency=5)

    outcomes = orchestrator._verify_slots_bounded(
        context=_context(orchestrator, 5), units=_units(5)
    )

    assert verifier.completion_order[0] == "slot-005"
    assert verifier.completion_order[-1] == "slot-001"
    for index in range(1, 6):
        slot_id = f"slot-{index:03d}"
        result = outcomes[slot_id]
        assert isinstance(result, VerificationResult)
        assert result.slot_id == slot_id
        assert result.generation_item_id == f"item-{index:03d}"


def test_verifier_exception_is_isolated_to_its_own_question() -> None:
    verifier = _RecordingVerifier(raise_for={"slot-002": RuntimeError("boom")})
    orchestrator = _orchestrator(verifier, max_concurrency=3)

    outcomes = orchestrator._verify_slots_bounded(
        context=_context(orchestrator, 3), units=_units(3)
    )

    assert isinstance(outcomes["slot-002"], RuntimeError)
    assert isinstance(outcomes["slot-001"], VerificationResult)
    assert isinstance(outcomes["slot-003"], VerificationResult)


def test_provider_execution_error_is_returned_not_raised() -> None:
    failure = ProviderExecutionError("verifier failed", failure_kind="unknown_provider_error")
    verifier = _RecordingVerifier(raise_for={"slot-001": failure})
    orchestrator = _orchestrator(verifier, max_concurrency=2)

    outcomes = orchestrator._verify_slots_bounded(
        context=_context(orchestrator, 2), units=_units(2)
    )

    assert outcomes["slot-001"] is failure
    assert isinstance(outcomes["slot-002"], VerificationResult)


def test_rejection_and_approval_coexist_without_cross_contamination() -> None:
    verifier = _RecordingVerifier(reject_slots=frozenset({"slot-002"}))
    orchestrator = _orchestrator(verifier, max_concurrency=3)

    outcomes = orchestrator._verify_slots_bounded(
        context=_context(orchestrator, 3), units=_units(3)
    )

    assert outcomes["slot-002"].decision is VerificationDecision.REGENERATE
    assert outcomes["slot-001"].decision is VerificationDecision.ACCEPT
    assert outcomes["slot-003"].decision is VerificationDecision.ACCEPT


def test_empty_unit_list_performs_no_verifier_call() -> None:
    verifier = _RecordingVerifier()
    orchestrator = _orchestrator(verifier)

    assert orchestrator._verify_slots_bounded(context=_context(orchestrator, 1), units=[]) == {}
    assert verifier.calls == []


def test_every_question_receives_exactly_one_verifier_call() -> None:
    """No question may be skipped, duplicated, or implicitly approved."""
    verifier = _RecordingVerifier(delay=0.01)
    orchestrator = _orchestrator(verifier, max_concurrency=4)

    outcomes = orchestrator._verify_slots_bounded(
        context=_context(orchestrator, 5), units=_units(5)
    )

    assert sorted(verifier.calls) == [f"slot-{i:03d}" for i in range(1, 6)]
    assert len(verifier.calls) == len(set(verifier.calls)) == 5
    assert sorted(outcomes) == [f"slot-{i:03d}" for i in range(1, 6)]


def test_all_workers_are_joined_before_returning() -> None:
    """No fire-and-forget: nothing may still be in flight when the call returns."""
    verifier = _RecordingVerifier(delay=0.05)
    orchestrator = _orchestrator(verifier, max_concurrency=5)

    orchestrator._verify_slots_bounded(context=_context(orchestrator, 5), units=_units(5))

    assert verifier.in_flight == 0
    assert len(verifier.completion_order) == 5
