"""Focused regressions for classification, correction, quality, and call bounds."""

from __future__ import annotations

import pytest

from schemas.conversation import ConversationCandidateCard
from schemas.doubt_solver import QueryClassification
from schemas.llm_routing import RouteRequest
from services.conversation.context_need_gate import ContextNeedGate
from services.doubt_solver.answer_correctness import deterministic_verify
from services.doubt_solver.answer_quality import validate_answer_quality
from services.llm.orchestration.orchestrator import (
    MockModelExecutor,
    create_mock_orchestrator_for_tests,
)
from services.query_classifier_service import (
    ConversationClassificationConflict,
    _apply_deterministic_conversation_contract,
    _classify_deterministic,
    _validate_conversation_contract,
    evaluate_primary_classification,
)


def _classification(**updates: object) -> QueryClassification:
    values: dict[str, object] = {
        "intent": "solve_question",
        "subject": "math",
        "confidence": 0.97,
        "difficulty": "basic",
    }
    values.update(updates)
    return QueryClassification.model_validate(values)


def _card(turn_id: str, question: str, *, topic: str) -> ConversationCandidateCard:
    return ConversationCandidateCard(
        turn_id=turn_id,
        question_preview=question,
        answer_clue="A completed substantive answer.",
        subject="math",
        topic=topic,
    )


def test_english_narration_practice_is_standalone_english_practice() -> None:
    query = "give me 5 questions on english grammar narrations"

    gate = ContextNeedGate().evaluate(query)
    classification = _classify_deterministic(query)

    assert gate.decision == "CONTEXT_NOT_NEEDED"
    assert classification.subject == "english"
    assert classification.intent == "practice_question"
    assert classification.relation == "NEW_QUESTION"


@pytest.mark.parametrize(
    ("query", "wrong_subject", "expected_conflict"),
    [
        ("Create narration practice questions", "math", True),
        ("Solve this seating arrangement", "math", True),
        ("Explain Indian polity", "math", True),
    ],
)
def test_obvious_subject_conflict_rejects_high_confidence_primary(
    query: str,
    wrong_subject: str,
    expected_conflict: bool,
) -> None:
    decision = evaluate_primary_classification(
        query,
        _classification(subject=wrong_subject),
    )

    assert decision.strong_required is expected_conflict
    assert "subject,intent" in decision.material_conflicts


@pytest.mark.parametrize(
    "query",
    [
        "give me five English narration questions",
        "simple-interest problem",
        "complete algebra problem: solve 2x + 3 = 11",
        "seating arrangement problem with five people",
    ],
)
def test_explicit_complete_academic_requests_skip_recent_context(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


def test_complete_problem_with_local_demonstrative_skips_recent_context() -> None:
    assessment = ContextNeedGate().evaluate(
        "Solve this advanced reasoning problem: Riya is 18th from the left "
        "and 25th from the right. How many students are there?"
    )

    assert assessment.decision == "CONTEXT_NOT_NEEDED"


@pytest.mark.parametrize(
    "query",
    (
        "200 का 15% कितना है? गणना दिखाइए।",
        "300 ka 20 percent kitna hoga? Short calculation dikhao.",
    ),
)
def test_complete_multilingual_percentage_skips_recent_context(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


@pytest.mark.parametrize(
    "query",
    ["it is wrong answer", "last question answer is wrong", "previous answer is wrong"],
)
def test_generic_correction_selects_newest_substantive_turn(query: str) -> None:
    cards = (
        _card("older", "Find simple interest on Rs 1000.", topic="SIMPLE_INTEREST"),
        _card("newest", "Solve x squared minus 5x plus 6.", topic="QUADRATIC_FUNCTIONS"),
    )
    result = _apply_deterministic_conversation_contract(
        _classification(),
        query=query,
        candidate_cards=cards,
        candidate_turn_ids=("older", "newest"),
        context_gate="CONTEXT_REQUIRED",
    )

    assert result.relation == "CORRECTION"
    assert result.requested_action == "VERIFY_AND_CORRECT"
    assert result.selected_turn_id == "newest"


def test_primary_cannot_override_generic_latest_correction_with_older_turn() -> None:
    cards = (
        _card("older", "Find simple interest on Rs 1000.", topic="SIMPLE_INTEREST"),
        _card("newest", "Solve x squared minus 5x plus 6.", topic="QUADRATIC_FUNCTIONS"),
    )
    classification = _classification(
        relation="CORRECTION",
        requested_action="VERIFY_AND_CORRECT",
        selected_turn_id="older",
    )

    with pytest.raises(ConversationClassificationConflict) as error:
        _validate_conversation_contract(
            classification,
            query="last question answer is wrong",
            candidate_cards=cards,
            candidate_turn_ids=("older", "newest"),
            context_gate="CONTEXT_REQUIRED",
        )

    assert error.value.reason == "generic_latest_reference_selected_older_turn"


def test_primary_latest_correction_is_normalized_before_contract_validation() -> None:
    cards = (
        _card("older", "Solve x plus 4 equals 10.", topic="ALGEBRA"),
        _card("newest", "What is 30 percent of 300?", topic="PERCENTAGE"),
    )
    normalized = _apply_deterministic_conversation_contract(
        _classification(
            relation="CORRECTION",
            requested_action="VERIFY_AND_CORRECT",
            selected_turn_id="older",
        ),
        query="last question answer is wrong",
        candidate_cards=cards,
        candidate_turn_ids=("older", "newest"),
        context_gate="CONTEXT_REQUIRED",
    )

    validated = _validate_conversation_contract(
        normalized,
        query="last question answer is wrong",
        candidate_cards=cards,
        candidate_turn_ids=("older", "newest"),
        context_gate="CONTEXT_REQUIRED",
    )

    assert validated.selected_turn_id == "newest"
    assert validated.topic == "PERCENTAGE"


def test_advanced_answer_only_fails_without_explicit_answer_only_request() -> None:
    result = validate_answer_quality(
        "**Answer:** 6",
        subject="reasoning",
        difficulty="advanced",
        intent="solve_question",
        query="Solve this advanced seating arrangement.",
    )

    assert result.is_valid is False
    assert "insufficient_solve_working" in result.reason_codes


def test_explicit_answer_only_request_allows_compact_answer() -> None:
    result = validate_answer_quality(
        "**Answer:** 6",
        subject="reasoning",
        difficulty="advanced",
        intent="solve_question",
        query="Solve this advanced seating arrangement. Only answer.",
    )

    assert result.is_valid is True


def test_simple_interest_regression_is_deterministically_recomputed() -> None:
    query = (
        "At a certain simple rate of interest, a given sum amounts to Rs 13920 in "
        "3 years, and to Rs 18960 in 6 years and 6 months. If the same given sum "
        "had been invested for 2 years at the same rate as before but with interest "
        "compounded every 6 months, find the nearest total interest. "
        "A. 3096 B. 3150 C. 3180 D. 3221"
    )

    approved = deterministic_verify(
        query=query,
        candidate_answer="**Answer:** 3221\n\nThe half-yearly calculation gives this option.",
    )
    rejected = deterministic_verify(
        query=query,
        candidate_answer="**Answer:** 3180\n\nThe half-yearly calculation gives this option.",
    )
    formatted = deterministic_verify(
        query=query.replace("6 years and 6 months", "6 years 6 months"),
        candidate_answer=(
            "**Answer:** Option D (₹3,221)\n\n"
            "The half-yearly compound-interest calculation gives this option."
        ),
    )

    assert approved is not None and approved.approved
    assert rejected is not None and rejected.status == "mismatch"
    assert formatted is not None and formatted.approved


def test_percentage_verifier_uses_final_value_from_answer_expression() -> None:
    result = deterministic_verify(
        query="What is 30% of 300?",
        candidate_answer=(
            "**Answer:** 30% of 300 = "
            "\\(\\frac{30}{100} \\times 300 = 90\\)"
        ),
    )

    assert result is not None and result.approved


def test_practice_verifier_accepts_bounded_multi_item_independent_answer() -> None:
    from services.doubt_solver.answer_correctness import _VerifierOutput

    result = _VerifierOutput.model_validate(
        {
            "status": "MATCH",
            "independent_answer": " ".join(["checked-item"] * 100),
            "single_defensible_answer": True,
            "reason": "All requested practice items are independently answerable.",
        }
    )

    assert result.status == "MATCH"


def test_continuation_and_rewrite_share_two_generator_call_cap() -> None:
    orchestrator, executor = create_mock_orchestrator_for_tests(
        content="Incomplete calculation",
    )
    assert isinstance(executor, MockModelExecutor)
    executor._finish_reason = "length"

    orchestrator.generate(
        route_request=RouteRequest(
            request_id="call-cap",
            subject="math",
            task_role="generator",
            difficulty="advanced",
            intent="solve",
        ),
        query="Solve the advanced problem.",
    )

    assert executor.call_count == 2
