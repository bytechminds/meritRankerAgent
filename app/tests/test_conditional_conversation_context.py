"""Deterministic gate, compact candidate, and selected-context coverage."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from schemas.conversation import (
    ContextNeedAssessment,
    ConversationPreparation,
    RecentContextLoadResult,
    RecentConversationTurn,
)
from schemas.doubt_solver import QueryClassification
from services.conversation.candidate_builder import build_candidate_cards
from services.conversation.context_need_gate import ContextNeedGate
from services.conversation.selected_context_builder import (
    build_context_aware_clarification,
    build_selected_generation_context,
)


@pytest.mark.parametrize(
    "query",
    (
        "Solve 2x + 5 = 15.",
        "Explain photosynthesis.",
        "Who wrote the Indian Constitution?",
        "There is a number k such that the sum is 57. Find k.",
        "Find the value that satisfies x + 3 = 9.",
        "In the set {1,3,5,...,57}, find k.",
        (
            "A shopkeeper marks goods 40% above cost price and gives a 20% "
            "discount. Find the profit percentage."
        ),
        (
            "A container holds 200 litres of an acid-water solution. It contains "
            "25% acid. Ten percent of the mixture is removed and replaced with "
            "acid. The final acid percentage is required."
        ),
    ),
)
def test_gate_skips_context_for_complete_academic_requests(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


@pytest.mark.parametrize(
    "query",
    (
        "How did you calculate 75%?",
        "Why did you divide by 100?",
        "Explain the last step.",
        "Give me five similar questions.",
        "Create another question like this.",
        "Same method ke 3 questions do.",
        "Your answer is wrong.",
        "The answer 57 is incorrect.",
        "Solve it again from scratch.",
        "Again wrong, resolve it.",
        "Answer galat hai, dobara solve karo.",
        "how did u calculated",
        "ur answer is worng",
        "agane solve it",
        "last soluton",
        "How you replaced 10% of the new mix with acid?",
        "Why did you subtract 7 from both sides?",
        "Highlight the water-acid solution.",
        "isko explain karo",
        "isko highlight karo",
        "ye kaise aaya",
        "answer galat hai",
        "same type ke questions do",
        "pichla solution",
    ),
)
def test_gate_requires_context_for_explicit_references(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_REQUIRED"


@pytest.mark.parametrize(
    "query",
    (
        "What about 57?",
        "Can you do another?",
        "Why is that?",
        "Is that correct?",
        "water acid solution",
    ),
)
def test_gate_marks_elliptical_requests_uncertain(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "UNCERTAIN"


@pytest.mark.parametrize(
    "query",
    (
        "A price of ₹570 is reduced by 10%. What is the new price?",
        "Find x when 3x + 10 = 57.",
    ),
)
def test_numeric_overlap_does_not_make_complete_question_contextual(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


def _turn(turn_id: str, question: str, answer: str) -> RecentConversationTurn:
    return RecentConversationTurn(
        turn_id=turn_id,
        original_query=question,
        final_answer=answer,
        created_at=datetime.now(UTC),
    )


def _classification(**updates: object) -> QueryClassification:
    values: dict[str, object] = {
        "intent": "general_doubt",
        "subject": "math",
        "confidence": 0.94,
        "relation": "FOLLOW_UP",
        "selected_turn_id": "turn-1",
        "requested_action": "EXPLAIN_PREVIOUS",
    }
    values.update(updates)
    return QueryClassification.model_validate(values)


def _preparation(*turns: RecentConversationTurn) -> ConversationPreparation:
    cards = build_candidate_cards(
        tuple(turns),
        current_query="Why did you divide by 100?",
    )
    return ConversationPreparation(
        gate=ContextNeedAssessment(decision="CONTEXT_REQUIRED"),
        context_load=RecentContextLoadResult(turns=tuple(turns)),
        eligible_turns=tuple(turns),
        candidates=cards,
        candidate_characters=sum(len(card.model_dump_json()) for card in cards),
    )


def test_candidate_cards_are_bounded_and_do_not_include_full_answers() -> None:
    answer = "Long introduction. " * 300 + "\nFinal Answer: 75%"
    cards = build_candidate_cards(
        (_turn("turn-1", "Calculate 75% of 200.", answer),),
        current_query="How did you calculate 75%?",
    )

    assert len(cards) == 1
    assert len(cards[0].question_preview) <= 700
    assert len(cards[0].answer_clue) <= 500
    assert cards[0].answer_clue != answer


def test_candidate_clue_prioritizes_referenced_percentage_and_nearby_step() -> None:
    cards = build_candidate_cards(
        (
            _turn(
                "turn-1",
                "A container has an acid-water mixture.",
                (
                    "Start with 50 litres of acid. The remaining liquid is water.\n"
                    "Remove 10% of the new mixture, which removes both components.\n"
                    "Replace that 10% of the new mix with pure acid.\n"
                    "Final Answer: the acid concentration becomes 32.5%."
                ),
            ),
        ),
        current_query="How you replaced 10% of the new mix with acid?",
    )

    assert "Replace that 10% of the new mix with pure acid." in cards[0].answer_clue
    assert "numeric_overlap" in cards[0].query_match_indicators
    assert "term_overlap" in cards[0].query_match_indicators


def test_explanation_context_contains_only_selected_turn() -> None:
    first = _turn("turn-1", "Calculate 75% of 200.", "Final Answer: 150")
    second = _turn("turn-2", "Solve x + 2 = 5.", "Final Answer: x = 3")
    result = build_selected_generation_context(
        current_query="Why did you divide by 100?",
        classification=_classification(),
        preparation=_preparation(first, second),
    )

    assert result.selected_turn_id == "turn-1"
    assert "Calculate 75% of 200." in result.conversation_context
    assert "Solve x + 2 = 5." not in result.conversation_context


def test_similar_context_omits_previous_answer_by_default() -> None:
    turn = _turn("turn-1", "A shop earns 20% profit. Find SP.", "Final Answer: ₹120")
    result = build_selected_generation_context(
        current_query="Give me five similar questions.",
        classification=_classification(requested_action="GENERATE_SIMILAR"),
        preparation=_preparation(turn),
    )

    assert result.context_policy == "generate_similar"
    assert "₹120" not in result.conversation_context


def test_current_affairs_similar_context_requires_fresh_source_cards() -> None:
    turn = _turn(
        "turn-1",
        "Provide current affairs questions for July 2026.",
        "Previously grounded questions.",
    )
    result = build_selected_generation_context(
        current_query="Give me five more.",
        classification=_classification(
            subject="general",
            intent="practice_question",
            requested_action="GENERATE_SIMILAR",
            need_web_search=True,
            web_search_reason="current_affairs",
            web_search_query="current affairs July 2026",
        ),
        preparation=_preparation(turn),
    )

    assert "exclusively from the fresh numbered [Web Context]" in result.resolved_query
    assert "at most one question per source card" in result.resolved_query
    assert "Previously grounded questions." not in result.conversation_context


def test_correction_marks_previous_conclusion_untrusted() -> None:
    turn = _turn("turn-1", "Solve 2x = 10.", "Final Answer: x = 7")
    result = build_selected_generation_context(
        current_query="Your answer is wrong.",
        classification=_classification(
            relation="CORRECTION",
            requested_action="VERIFY_AND_CORRECT",
        ),
        preparation=_preparation(turn),
    )

    assert "UNTRUSTED" in result.conversation_context
    assert "Independently solve" in result.resolved_query


def test_resolve_from_scratch_omits_prior_reasoning() -> None:
    turn = _turn("turn-1", "Solve 2x = 10.", "Bad reasoning. Final Answer: x = 7")
    result = build_selected_generation_context(
        current_query="Again wrong. Solve it from scratch.",
        classification=_classification(
            relation="RESOLVE_AGAIN",
            requested_action="RESOLVE_FROM_SCRATCH",
        ),
        preparation=_preparation(turn),
    )

    assert "Bad reasoning" not in result.conversation_context
    assert "OMITTED" in result.conversation_context


def test_ambiguous_candidate_clarification_is_context_aware() -> None:
    preparation = _preparation(
        _turn(
            "turn-1",
            "A container contains an acid-water solution.",
            "Final Answer: 32.5% acid",
        )
    )

    clarification = build_context_aware_clarification(
        preparation,
        language="english",
    )

    assert "previous" in clarification
    assert "explain" in clarification
    assert "highlight its steps" in clarification
    assert "similar question" in clarification
