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


@pytest.mark.parametrize(
    "query",
    [
        # The reported incident: an anaphoric word with its source in the same turn.
        "A right triangle has legs of length 6 cm and 8 cm. Find the hypotenuse."
        " Create 2 similar questions.",
        # The literal word "practice" must not be what decides this.
        "A right triangle has legs of length 6 cm and 8 cm. Find the hypotenuse."
        " Create 2 similar practice questions.",
        "A triangle has angles 25 and 45 degrees. Find the third angle."
        " Create 3 similar questions.",
        "Solve 2x + 5 = 15. Give me 5 more questions like this.",
        "Solve x squared - 5x + 6 = 0 and give me 5 more questions like this.",
        "Create 10 questions similar to this: If 5 workers complete a job in 12 days,"
        " how long would 10 workers take?",
        "Here is a grammar sentence: The boy run fast every morning."
        " Create 4 similar questions.",
        "A right triangle has legs of 6 and 8. Find the hypotenuse."
        " Create 100 similar questions.",
    ],
)
def test_local_source_resolves_similar_without_history(query: str) -> None:
    assessment = ContextNeedGate().evaluate(query)

    assert assessment.decision == "CONTEXT_NOT_NEEDED", assessment.reason_codes


@pytest.mark.parametrize(
    "query",
    [
        "Create 2 similar questions.",
        "Give me 5 more like the previous one.",
        "Give me 5 more like that.",
        "Make questions similar to what we just did.",
        "Create 3 questions similar to the previous question.",
    ],
)
def test_similar_without_a_local_source_still_requires_history(query: str) -> None:
    assessment = ContextNeedGate().evaluate(query)

    assert assessment.decision == "CONTEXT_REQUIRED", assessment.reason_codes


def test_local_antecedent_is_reported_as_its_own_reason_code() -> None:
    assessment = ContextNeedGate().evaluate(
        "A triangle has angles 25 and 45 degrees. Find the third angle."
        " Create 3 similar questions."
    )

    assert assessment.reason_codes == ("local_antecedent_resolved",)
    assert assessment.matched_signals == ("similar_reference",)


def test_local_precedence_never_overrides_an_ordinal_history_reference() -> None:
    # Self-contained wording plus an explicitly prior pointer: history still wins.
    assessment = ContextNeedGate().evaluate(
        "Solve 2x + 5 = 15. Now give me 5 more like the previous one."
    )

    assert assessment.decision == "CONTEXT_REQUIRED"


def test_a_bare_local_source_introducer_is_not_treated_as_an_anchor() -> None:
    # The introducer must actually introduce something substantial.
    assert ContextNeedGate().evaluate("Here is a question.").decision != "CONTEXT_NOT_NEEDED"


@pytest.mark.parametrize(
    "query",
    (
        "explain simpler",
        "explain again",
        "explain in Hindi",
        "explain another way",
        "explain differently",
        "explain step 2",
        "explain more",
        "explain briefly",
        "explain once more",
        "explain in detail",
        "aur simple samjhao",
        "dobara samjhao",
        "hindi me samjhao",
    ),
)
def test_modifier_only_imperative_does_not_suppress_retrieval(query: str) -> None:
    """"explain <modifier>" names nothing to work on, so history stays eligible."""
    assert ContextNeedGate().evaluate(query).decision != "CONTEXT_NOT_NEEDED"


@pytest.mark.parametrize(
    "query",
    (
        "explain photosynthesis",
        "explain compound interest",
        "explain profit and loss",
        "explain Article 21",
        "explain Newton's first law",
        "explain blood relation",
        "explain computer memory",
        "explain power",
        "explain number system",
        "explain photosynthesis in detail",
        "explain the difference between speed and velocity",
        "explain step 2 of the water cycle",
        "define integer",
        "describe the water cycle",
    ),
)
def test_named_object_stays_self_contained(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


@pytest.mark.parametrize(
    "query",
    (
        "One of my friend is coming - which part is wrong and why?",
        "I have went there yesterday - what is incorrect here?",
        "Which option is wrong here? A, B, C, D",
        "What is wrong in 2x + 3 = 9?",
        "Tell me why this sentence is incorrect: She go to school",
    ),
)
def test_correction_word_alone_is_not_a_conversation_reference(query: str) -> None:
    """A student asking what is wrong with material they supplied needs no history."""
    assessment = ContextNeedGate().evaluate(query)
    assert assessment.decision != "CONTEXT_REQUIRED"
    assert "correction_reference" not in assessment.matched_signals


@pytest.mark.parametrize(
    "query",
    (
        "your answer is wrong",
        "your solution is incorrect",
        "why was your answer wrong?",
        "why is your answer wrong?",
        "you are wrong",
    ),
)
def test_correction_of_a_named_assistant_answer_still_requires_context(
    query: str,
) -> None:
    assessment = ContextNeedGate().evaluate(query)
    assert assessment.decision == "CONTEXT_REQUIRED"
    assert "correction_reference" in assessment.matched_signals


@pytest.mark.parametrize(
    "query",
    (
        "the previous answer is wrong",
        "the last solution is incorrect",
        "check your previous solution",
    ),
)
def test_previous_reference_wording_still_requires_context(query: str) -> None:
    """These stay contextual through previous_reference, not correction_reference."""
    assessment = ContextNeedGate().evaluate(query)
    assert assessment.decision == "CONTEXT_REQUIRED"


@pytest.mark.parametrize(
    "query",
    (
        "explain this grammar rule: subject verb agreement",
        "explain this sentence - He does not go",
        "check this equation: 2x + 3 = 9",
        "verify this calculation: 20% of 500 = 100",
        "explain this concept: photosynthesis",
    ),
)
def test_demonstrative_with_supplied_target_is_locally_resolved(query: str) -> None:
    """The turn supplies what "this" points at, so recent history adds nothing."""
    assessment = ContextNeedGate().evaluate(query)
    assert assessment.decision != "CONTEXT_REQUIRED"
    assert "selected_object_reference" not in assessment.matched_signals


@pytest.mark.parametrize(
    "query",
    (
        "explain this",
        "check this",
        "verify this",
        "explain this answer",
        "check this solution",
        "explain this again",
        "explain this in hindi",
        "check this properly",
    ),
)
def test_demonstrative_without_a_target_still_requires_context(query: str) -> None:
    assessment = ContextNeedGate().evaluate(query)
    assert assessment.decision == "CONTEXT_REQUIRED"
    assert "selected_object_reference" in assessment.matched_signals


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("ye wala kaise hoga", "CONTEXT_REQUIRED"),
        ("isko samjhao", "CONTEXT_REQUIRED"),
        ("ye kaise", "CONTEXT_REQUIRED"),
        ("iska answer kya hai", "CONTEXT_REQUIRED"),
        ("What is the capital of Australia?", "CONTEXT_NOT_NEEDED"),
        ("explain simpler", "UNCERTAIN"),
        ("aur simple samjhao", "UNCERTAIN"),
        ("are you sure?", "UNCERTAIN"),
    ),
)
def test_reference_and_follow_up_gate_decisions_are_preserved(
    query: str, expected: str
) -> None:
    assert ContextNeedGate().evaluate(query).decision == expected


@pytest.mark.parametrize(
    "query",
    (
        # supplied sentence, pronoun first
        "He don't like tea. What is wrong in this sentence?",
        "She go to school daily correct this sentence",
        "what is wrong in this sentence: she have two books",
        # multi-part: complete first proposition, pronoun in the second half
        "what is GDP and how is it calculated",
        "What is repo rate and how does RBI use it to control inflation",
        "largest gland in human body kaun si hai aur iska function kya hai",
        "Who appoints the governor and what is his tenure",
        # supplied equation, statement and option
        "this calculation is incorrect: 20% of 500 = 50",
        "which statement is wrong among these: A) Earth is flat B) Earth is round",
        "check this option: B. New Delhi",
    ),
)
def test_turn_that_supplies_its_own_referent_needs_no_history(query: str) -> None:
    assert ContextNeedGate().evaluate(query).decision != "CONTEXT_REQUIRED"


@pytest.mark.parametrize(
    "query",
    (
        "ye wala kaise hoga",
        "iska answer kya hai",
        "ye wrong kyu hai",
        "wo wala question phir se",
        "dusra option kyu sahi hai",
        "isko samjhao",
    ),
)
def test_reference_naming_nothing_still_requires_context(query: str) -> None:
    """The collision case: a bare demonstrative plus a placeholder resolves nothing."""
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_REQUIRED"


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        # a deictic followed by a conversational placeholder is not a local antecedent
        ("iska answer kya hai", False),
        ("uska solution batao", False),
        # a bare possessive question is still a follow-up: nothing here names the subject
        ("iska function kya hai", False),
        # the subject is supplied in the first half of the same turn
        ("what is GDP and how is it calculated", True),
        ("Who appoints the governor and what is his tenure", True),
    ),
)
def test_local_antecedent_distinguishes_placeholder_from_content(
    query: str, expected: bool
) -> None:
    from services.conversation.reference_resolution import analyze_reference

    assert analyze_reference(query).local_reference is expected


def test_first_reference_at_index_zero_without_local_content_stays_unresolved() -> None:
    """'ye wala kaise hoga' must not become answerable via the new whole-turn scan."""
    from services.conversation.reference_resolution import analyze_reference

    analysis = analyze_reference("ye wala kaise hoga")
    assert analysis.external_reference_detected is True
    assert analysis.local_reference is False
