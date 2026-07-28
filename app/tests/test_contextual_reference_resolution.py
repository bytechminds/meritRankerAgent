"""Regression coverage for bounded pronoun and contextual-reference resolution."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

import graphs.doubt_solver_graph as doubt_solver_graph
import services.query_classifier_service as classifier_service
from schemas.conversation import (
    ContextNeedAssessment,
    ConversationCandidateCard,
    ConversationPreparation,
    RecentContextLoadResult,
    RecentConversationTurn,
)
from schemas.doubt_solver import QueryClassification
from services.conversation.candidate_builder import build_candidate_cards
from services.conversation.context_need_gate import ContextNeedGate
from services.conversation.conversation_understanding import (
    ConversationUnderstandingService,
)
from services.conversation.memory_hygiene import (
    classify_non_substantive_response,
    classify_stored_turn,
    filter_substantive_turns,
)
from services.conversation.reference_resolution import (
    analyze_reference,
    assess_candidate_compatibility,
    grounded_entity_labels,
    group_compatible_candidates,
    validate_entity_label,
)
from services.conversation.selected_context_builder import (
    build_context_aware_clarification,
    build_selected_generation_context,
)


def _turn(turn_id: str, question: str, answer: str) -> RecentConversationTurn:
    return RecentConversationTurn(
        turn_id=turn_id,
        original_query=question,
        final_answer=answer,
        created_at=datetime.now(UTC),
    )


def _cards(
    query: str,
    *turns: RecentConversationTurn,
) -> tuple[ConversationCandidateCard, ...]:
    return build_candidate_cards(tuple(turns), current_query=query)


def _preparation(
    query: str,
    *turns: RecentConversationTurn,
) -> ConversationPreparation:
    cards = _cards(query, *turns)
    return ConversationPreparation(
        gate=ContextNeedAssessment(decision="UNCERTAIN"),
        context_load=RecentContextLoadResult(turns=tuple(turns)),
        eligible_turns=tuple(turns),
        candidates=cards,
        candidate_characters=sum(len(card.model_dump_json()) for card in cards),
    )


@pytest.mark.parametrize(
    "query,reference_type",
    (
        ("Does he ruled longest?", "person"),
        ("What happened after his death?", "event"),
        ("Did they win the battle?", "person"),
        ("Can this method work here?", "concept"),
        ("Why is the second one correct?", "ordinal"),
        ("usne kitne sal rule kiya", "person"),
        ("yeh kaise hua", "object"),
        ("pichla wala samjhao", "ordinal"),
    ),
)
def test_external_reference_families_are_detected(
    query: str,
    reference_type: str,
) -> None:
    analysis = analyze_reference(query)

    assert analysis.external_reference_detected is True
    assert reference_type in analysis.reference_types
    assert ContextNeedGate().evaluate(query).decision in {
        "UNCERTAIN",
        "CONTEXT_REQUIRED",
    }


def test_existing_classifier_prompt_owns_coreference_and_action_contract() -> None:
    prompt = classifier_service._load_classifier_prompt()

    assert "Pronoun and contextual reference resolution" in prompt
    assert "semantically compatible" in prompt
    assert "recency alone is insufficient" in prompt
    assert "ANSWER_WITH_CONTEXT" in prompt
    assert "Grammar, spelling, and malformed phrasing" in prompt


def test_graph_classifier_wrapper_forwards_typed_candidate_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "Does he ruled longest?"
    cards = _cards(
        query,
        _turn(
            "akbar-turn",
            "Explain Akbar's reign.",
            "Akbar was a Mughal emperor who ruled India.",
        ),
    )
    received: dict[str, object] = {}

    def classify(
        current_query: str,
        request_id: str | None,
        **kwargs: object,
    ) -> QueryClassification:
        received.update(kwargs)
        return QueryClassification(
            intent="general_doubt",
            subject="general",
            confidence=0.96,
            relation="FOLLOW_UP",
            selected_turn_id="akbar-turn",
            requested_action="ANSWER_WITH_CONTEXT",
        )

    monkeypatch.setattr(doubt_solver_graph, "classify_query", classify)

    result = doubt_solver_graph._run_graph_classifier(
        query,
        "graph-reference",
        conversation_candidates="bounded",
        candidate_cards=cards,
        candidate_turn_ids=("akbar-turn",),
        context_gate="CONTEXT_REQUIRED",
    )

    assert received["candidate_cards"] == cards
    assert result.requested_action == "ANSWER_WITH_CONTEXT"


@pytest.mark.parametrize(
    "query",
    (
        "Akbar ruled from 1556 to 1605. Did he introduce reforms?",
        "A number is divided by 3. It leaves remainder 2.",
        "If a number is odd, it cannot be divisible by 2. Explain why.",
    ),
)
def test_local_antecedent_remains_standalone(query: str) -> None:
    analysis = analyze_reference(query)

    assert analysis.local_reference is True
    assert analysis.external_reference_detected is False
    assert ContextNeedGate().evaluate(query).decision == "CONTEXT_NOT_NEEDED"


def test_person_reference_prefers_compatible_history_over_newer_math() -> None:
    query = "Does he ruled longest?"
    acid = _turn(
        "acid-turn",
        "A solution contains acid and water.",
        "The acid concentration is 30%.",
    )
    akbar = _turn(
        "akbar-turn",
        "Explain Akbar's rule and history in India.",
        "Akbar was a Mughal emperor who ruled from 1556 to 1605.",
    )
    percentage = _turn(
        "percentage-turn",
        "Find 25% of 400.",
        "The answer is 100.",
    )

    compatibility = assess_candidate_compatibility(
        query,
        _cards(query, acid, akbar, percentage),
    )

    assert tuple(item.turn_id for item in compatibility if item.compatible) == (
        "akbar-turn",
    )


def test_exact_akbar_fallback_uses_real_classifier_contract_without_model_mock() -> None:
    query = "Does he ruled longest?"
    akbar = _turn(
        "akbar-turn",
        "Explain Akbar's rule and history in India.",
        "Akbar was a Mughal emperor who ruled from 1556 to 1605.",
    )
    cards = _cards(query, akbar)

    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate="UNCERTAIN",
        candidate_cards=cards,
        candidate_turn_ids=("akbar-turn",),
    )

    assert result.classification.relation == "FOLLOW_UP"
    assert result.classification.requested_action == "ANSWER_WITH_CONTEXT"
    assert result.classification.selected_turn_id == "akbar-turn"


@pytest.mark.parametrize(
    "query,question,answer",
    (
        ("Was he a good ruler?", "Explain Ashoka's reign.", "Ashoka was an emperor."),
        ("Who defeated him?", "Explain Napoleon's career.", "Napoleon was a leader."),
        (
            "How long was her reign?",
            "Explain Rani Lakshmibai's life.",
            "Rani Lakshmibai was a queen.",
        ),
        (
            "Why were they defeated?",
            "Describe the Maratha leaders in the battle.",
            "The leaders lost the battle.",
        ),
        (
            "Can this method work here?",
            "Solve x + 4 = 9 by transposition.",
            "The same method subtracts 4 from both sides.",
        ),
        (
            "Is that formula correct?",
            "Explain the simple-interest formula.",
            "The formula is SI = PRT/100.",
        ),
        (
            "Why is the second one correct?",
            "Choose from options A, B, C, and D.",
            "Option B is correct.",
        ),
        (
            "usne kitne sal rule kiya",
            "Explain Ashoka's reign.",
            "Ashoka was an emperor.",
        ),
        (
            "woh acha ruler tha",
            "Explain Ashoka's reign.",
            "Ashoka was an emperor.",
        ),
        (
            "uska beta kaun tha",
            "Explain Ashoka's family.",
            "Ashoka was an emperor and ruler.",
        ),
        (
            "isko kyu use kiya",
            "Solve x + 4 = 9 using transposition.",
            "The method subtracts 4 from both sides.",
        ),
        (
            "dusra option kyu correct hai",
            "Choose from option 1, option 2, and option 3.",
            "Option 2 is correct.",
        ),
        (
            "wat happened after him",
            "Explain Napoleon's career.",
            "Napoleon was a leader.",
        ),
        (
            "why its used",
            "Explain the percentage formula.",
            "The formula divides by 100.",
        ),
    ),
)
def test_reference_families_use_single_compatible_candidate(
    query: str,
    question: str,
    answer: str,
) -> None:
    turn = _turn("reference-turn", question, answer)
    cards = _cards(query, turn)

    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate="UNCERTAIN",
        candidate_cards=cards,
        candidate_turn_ids=("reference-turn",),
    )

    assert result.classification.relation == "FOLLOW_UP"
    assert result.classification.requested_action == "ANSWER_WITH_CONTEXT"
    assert result.classification.selected_turn_id == "reference-turn"


def test_unresolved_new_question_invokes_existing_strong_classifier_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = "Does he ruled longest?"
    akbar = _turn(
        "akbar-turn",
        "Explain Akbar's rule and history in India.",
        "Akbar was a Mughal emperor who ruled from 1556 to 1605.",
    )
    cards = _cards(query, akbar)
    calls: list[str] = []
    primary = QueryClassification(
        intent="general_doubt",
        subject="general",
        confidence=0.96,
        relation="NEW_QUESTION",
        requested_action="ANSWER_CURRENT",
    )
    strong = primary.model_copy(
        update={
            "relation": "FOLLOW_UP",
            "selected_turn_id": "akbar-turn",
            "requested_action": "ANSWER_WITH_CONTEXT",
        }
    )

    def classify_once(
        current_query: str,
        _request_id: str | None = None,
        *,
        task_role: str,
        candidate_cards: tuple[ConversationCandidateCard, ...] = (),
        candidate_turn_ids: tuple[str, ...] = (),
        context_gate: str,
        **_: object,
    ) -> QueryClassification:
        calls.append(task_role)
        classification = primary if task_role == "classifier" else strong
        return classifier_service._validate_conversation_contract(
            classification,
            query=current_query,
            candidate_cards=candidate_cards,
            candidate_turn_ids=candidate_turn_ids,
            context_gate=context_gate,
        )

    monkeypatch.setattr(
        classifier_service,
        "_classify_with_llm_orchestrated",
        classify_once,
    )

    result = classifier_service._classify_with_llm_orchestrated_or_fallback(
        query,
        request_id="akbar-reference",
        conversation_candidates="bounded candidates",
        candidate_cards=cards,
        candidate_turn_ids=("akbar-turn",),
        context_gate="UNCERTAIN",
    )

    assert calls == ["classifier", "classifier_strong"]
    assert result.classification.requested_action == "ANSWER_WITH_CONTEXT"


def test_incompatible_selected_candidate_is_rejected() -> None:
    query = "Does he rule the longest?"
    acid = _turn(
        "acid-turn",
        "A solution contains acid and water.",
        "The acid concentration is 30%.",
    )
    akbar = _turn(
        "akbar-turn",
        "Explain Akbar's reign.",
        "Akbar was an emperor.",
    )
    cards = _cards(query, acid, akbar)
    classification = QueryClassification(
        intent="general_doubt",
        subject="general",
        confidence=0.96,
        relation="FOLLOW_UP",
        selected_turn_id="acid-turn",
        requested_action="ANSWER_WITH_CONTEXT",
    )

    with pytest.raises(
        classifier_service.ConversationClassificationConflict,
        match="selected_turn_semantically_incompatible",
    ):
        classifier_service._validate_conversation_contract(
            classification,
            query=query,
            candidate_cards=cards,
            candidate_turn_ids=("acid-turn", "akbar-turn"),
            context_gate="UNCERTAIN",
        )


def test_multiple_compatible_people_produce_contextual_clarification() -> None:
    query = "Did he rule longer?"
    preparation = _preparation(
        query,
        _turn("akbar-turn", "Explain Akbar's reign.", "Akbar was an emperor."),
        _turn(
            "shah-jahan-turn",
            "Explain Shah Jahan's reign.",
            "Shah Jahan was a Mughal emperor.",
        ),
    )
    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate="UNCERTAIN",
        candidate_cards=preparation.candidates,
        candidate_turn_ids=("akbar-turn", "shah-jahan-turn"),
    )

    clarification = build_context_aware_clarification(
        preparation,
        language="english",
        current_query=query,
    )

    assert result.classification.relation == "AMBIGUOUS"
    assert result.classification.selected_turn_id is None
    assert clarification == "Do you mean Akbar or Shah Jahan?"


def test_multiple_compatible_objects_prefer_actual_academic_topics() -> None:
    query = "Can you explain that again?"
    preparation = ConversationPreparation(
        gate=ContextNeedAssessment(decision="UNCERTAIN"),
        context_load=RecentContextLoadResult(),
        candidates=(
            ConversationCandidateCard(
                turn_id="interest-turn",
                question_preview="Explain simple interest briefly.",
                answer_clue="Rate times principal times time gives the interest.",
                subject="math",
            ),
            ConversationCandidateCard(
                turn_id="quadratic-turn",
                question_preview="Explain quadratic equations briefly.",
                answer_clue="Solutions are the roots of the equation.",
                subject="math",
            ),
        ),
    )

    clarification = build_context_aware_clarification(
        preparation,
        language="english",
        current_query=query,
    )

    assert clarification == "Do you mean Simple Interest or Quadratic Equations?"


def test_classifier_cannot_choose_by_recency_between_distinct_people() -> None:
    query = "Did he rule longer?"
    cards = _cards(
        query,
        _turn("akbar-turn", "Explain Akbar's reign.", "Akbar was an emperor."),
        _turn(
            "shah-jahan-turn",
            "Explain Shah Jahan's reign.",
            "Shah Jahan was a Mughal emperor.",
        ),
    )
    selected_latest = QueryClassification(
        intent="general_doubt",
        subject="general",
        confidence=0.96,
        relation="FOLLOW_UP",
        selected_turn_id="shah-jahan-turn",
        requested_action="ANSWER_WITH_CONTEXT",
    )

    with pytest.raises(
        classifier_service.ConversationClassificationConflict,
        match="multiple_compatible_reference_entities",
    ):
        classifier_service._validate_conversation_contract(
            selected_latest,
            query=query,
            candidate_cards=cards,
            candidate_turn_ids=("akbar-turn", "shah-jahan-turn"),
            context_gate="CONTEXT_REQUIRED",
        )


def test_repeated_cards_for_same_entity_select_latest_chain_turn() -> None:
    query = "What happened after his death?"
    cards = _cards(
        query,
        _turn(
            "akbar-turn",
            "Explain Akbar's reign.",
            "Akbar was a Mughal emperor who ruled India.",
        ),
        _turn(
            "akbar-follow-up",
            "Does he ruled longest?",
            "Akbar ruled for about 49 years during the Mughal period.",
        ),
    )

    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate="CONTEXT_REQUIRED",
        candidate_cards=cards,
        candidate_turn_ids=("akbar-turn", "akbar-follow-up"),
    )

    assert result.classification.relation == "FOLLOW_UP"
    assert result.classification.selected_turn_id == "akbar-follow-up"
    assert result.classification.requested_action == "ANSWER_WITH_CONTEXT"


def test_explicit_comparison_selects_other_latest_person() -> None:
    query = "Did he rule longer than Akbar?"
    cards = _cards(
        query,
        _turn("akbar-turn", "Explain Akbar's reign.", "Akbar was an emperor."),
        _turn(
            "shah-jahan-turn",
            "Explain Shah Jahan's reign.",
            "Shah Jahan was a Mughal emperor.",
        ),
    )

    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate="UNCERTAIN",
        candidate_cards=cards,
        candidate_turn_ids=("akbar-turn", "shah-jahan-turn"),
    )

    assert result.classification.selected_turn_id == "shah-jahan-turn"


def test_no_compatible_person_candidate_returns_clarification() -> None:
    query = "Did he rule the longest?"
    cards = _cards(
        query,
        _turn("math-turn", "Solve 3x + 2 = 11.", "The answer is x = 3."),
    )

    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate="UNCERTAIN",
        candidate_cards=cards,
        candidate_turn_ids=("math-turn",),
    )

    assert result.classification.relation == "AMBIGUOUS"
    assert result.classification.requested_action == "ASK_CLARIFICATION"
    preparation = ConversationPreparation(
        gate=ContextNeedAssessment(decision="UNCERTAIN"),
        context_load=RecentContextLoadResult(),
        eligible_turns=(),
        candidates=cards,
    )
    clarification = build_context_aware_clarification(
        preparation,
        language="english",
        current_query=query,
    )
    assert clarification == "Who or what does “he” refer to?"


def test_selected_context_resolves_he_to_akbar_and_passes_one_turn() -> None:
    query = "Does he ruled longest?"
    akbar = _turn(
        "akbar-turn",
        "Explain Akbar's rule and history in India.",
        "Akbar was a Mughal emperor who ruled from 1556 to 1605.",
    )
    unrelated = _turn(
        "math-turn",
        "Solve x + 2 = 5.",
        "The answer is x = 3.",
    )
    preparation = _preparation(query, akbar, unrelated)
    classification = QueryClassification(
        intent="general_doubt",
        subject="general",
        confidence=0.96,
        relation="FOLLOW_UP",
        selected_turn_id="akbar-turn",
        requested_action="ANSWER_WITH_CONTEXT",
    )

    selected = build_selected_generation_context(
        current_query=query,
        classification=classification,
        preparation=preparation,
    )

    assert selected.resolved_reference == "he → Akbar"
    assert selected.context_policy == "answer_with_context"
    assert "Akbar" in selected.conversation_context
    assert "Solve x + 2 = 5." not in selected.conversation_context
    assert "claim the referent is missing" in selected.conversation_context


def test_exact_polluted_memory_turn_is_rejected_before_candidate_building() -> None:
    query = "Does he ruled longest?"
    akbar = _turn(
        "akbar-turn",
        "Explain Akbar’s rule and history in India.",
        "Akbar was a Mughal emperor who ruled India from 1556 to 1605.",
    )
    polluted = _turn(
        "2471adfc",
        "Does he ruled longest?",
        (
            '**Answer:** The question "Does he ruled longest?" is incomplete '
            'because it does not specify who "he" refers to. To answer it, '
            "please provide the person's name."
        ),
    )
    persistence = MagicMock()
    persistence.load_recent_context.return_value = RecentContextLoadResult(
        source="agentcore_memory",
        memory_attempted=True,
        memory_status="succeeded",
        memory_completed_pair_count=2,
        turns=(akbar, polluted),
        usable_turn_count=2,
    )

    preparation = ConversationUnderstandingService(
        persistence=persistence,
    ).understand(
        actor_id="student-1",
        conversation_id="polluted-memory",
        query=query,
        request_id="polluted-memory-regression",
    )
    compatibility = assess_candidate_compatibility(query, preparation.candidates)
    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate=preparation.gate.decision,
        candidate_cards=preparation.candidates,
        candidate_turn_ids=tuple(card.turn_id for card in preparation.candidates),
    )

    assert classify_stored_turn(polluted).reason == "unresolved_reference_response"
    assert preparation.context_load.turns == (akbar, polluted)
    assert preparation.rejected_turn_ids == ("2471adfc",)
    assert tuple(card.turn_id for card in preparation.candidates) == ("akbar-turn",)
    assert tuple(item.turn_id for item in compatibility if item.compatible) == (
        "akbar-turn",
    )
    assert result.classification.relation == "FOLLOW_UP"
    assert result.classification.requested_action == "ANSWER_WITH_CONTEXT"
    assert result.classification.selected_turn_id == "akbar-turn"


def test_named_person_rule_noun_is_grounded_without_answer_name_overlap() -> None:
    card = ConversationCandidateCard(
        turn_id="akbar-root",
        question_preview="Explain Akbar’s rule and history in India.",
        answer_clue="Administrative innovations and cultural integration were significant.",
    )

    compatibility = assess_candidate_compatibility(
        "Does he ruled longest?",
        (card,),
    )

    assert compatibility[0].compatible is True
    assert compatibility[0].grounded_antecedent is True
    assert compatibility[0].label == "Akbar"


@pytest.mark.parametrize(
    ("answer", "reason"),
    (
        (
            "The question is incomplete because it does not specify who he refers to.",
            "unresolved_reference_response",
        ),
        (
            "Sawal clear nahi hai ki woh kis person ko refer karta hai.",
            "unresolved_reference_response",
        ),
        (
            "प्रश्न स्पष्ट नहीं है कि वह किस व्यक्ति के संदर्भ में है।",
            "unresolved_reference_response",
        ),
        ("Please clarify which earlier question you mean.", "clarification_response"),
        ("Okay, I understand.", "generic_acknowledgement"),
        ("I could not generate the requested answer.", "technical_failure"),
        (
            "I could not verify the requested current information from reliable live sources.",
            "technical_failure",
        ),
    ),
)
def test_response_primary_function_has_typed_hygiene_reason(
    answer: str,
    reason: str,
) -> None:
    assert classify_non_substantive_response(answer) == reason


def test_multi_question_person_follow_up_resolves_to_grounded_akbar() -> None:
    query = "How many wives he had? Where did he stay?"
    cards = _cards(
        query,
        _turn(
            "akbar-turn",
            "Explain Akbar’s rule and history in India.",
            "Akbar was a Mughal emperor who ruled India from 1556 to 1605.",
        ),
    )

    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate="CONTEXT_REQUIRED",
        candidate_cards=cards,
        candidate_turn_ids=("akbar-turn",),
    )

    assert result.classification.selected_turn_id == "akbar-turn"
    assert result.classification.requested_action == "ANSWER_WITH_CONTEXT"


@pytest.mark.parametrize(
    "value",
    ("To", "The", "Who", "Question", "Answer", "Please", "He", "Does", "Why"),
)
def test_arbitrary_capitalized_tokens_are_not_entity_labels(value: str) -> None:
    card = ConversationCandidateCard(
        turn_id="invalid-label",
        question_preview=f"{value} ruled longest?",
        answer_clue=f"{value} answer requests clarification about who.",
    )

    assert validate_entity_label(value) is None
    assert grounded_entity_labels(card) == ()


def test_polluted_pronoun_candidate_is_not_grounded_person_compatible() -> None:
    query = "Does he ruled longest?"
    polluted = _turn(
        "polluted-turn",
        "Does he ruled longest?",
        "The question is incomplete because it does not specify who he refers to.",
    )
    cards = _cards(query, polluted)
    compatibility = assess_candidate_compatibility(query, cards)

    assert compatibility[0].compatible is False
    assert compatibility[0].grounded_antecedent is False
    assert compatibility[0].label is None
    assert compatibility[0].reason == "grounded_person_unproven"


def test_three_turn_same_entity_chain_collapses_to_newest_grounded_turn() -> None:
    query = "How many wives did he have?"
    cards = _cards(
        query,
        _turn(
            "akbar-root",
            "Explain Akbar’s reign.",
            "Akbar was a Mughal emperor.",
        ),
        _turn(
            "akbar-duration",
            "How long did he rule?",
            "Akbar ruled for approximately 49 years.",
        ),
        _turn(
            "akbar-death",
            "What happened after his death?",
            "After Akbar died, Jahangir succeeded him.",
        ),
    )

    groups = group_compatible_candidates(query, cards)
    result = classifier_service._deterministic_classifier_fallback(
        query,
        strong_classifier_used=True,
        context_gate="CONTEXT_REQUIRED",
        candidate_cards=cards,
        candidate_turn_ids=tuple(card.turn_id for card in cards),
    )

    assert len(groups) == 1
    assert groups[0].label == "Akbar"
    assert groups[0].turn_ids == ("akbar-root", "akbar-duration", "akbar-death")
    assert result.classification.selected_turn_id == "akbar-death"


@pytest.mark.parametrize(
    "query",
    (
        "usne kitne saal raj kiya",
        "uska beta kaun tha",
        "woh kaha rehta tha",
    ),
)
def test_hinglish_person_references_require_grounded_person(
    query: str,
) -> None:
    cards = _cards(
        query,
        _turn(
            "akbar-turn",
            "Explain Akbar’s reign.",
            "Akbar was a Mughal emperor and ruler.",
        ),
    )

    compatibility = assess_candidate_compatibility(query, cards)

    assert compatibility[0].compatible is True
    assert compatibility[0].grounded_antecedent is True
    assert compatibility[0].label == "Akbar"


def test_polluted_formula_clarification_does_not_displace_grounded_method() -> None:
    query = "yeh formula kyu use hua"
    grounded = _turn(
        "formula-turn",
        "Explain the simple-interest formula.",
        "The formula is SI = PRT/100 and calculates simple interest.",
    )
    polluted = _turn(
        "formula-clarification",
        "Why was it used?",
        "The question is incomplete because it does not specify what it refers to.",
    )

    substantive, decisions = filter_substantive_turns((grounded, polluted))
    cards = _cards(query, *substantive)
    compatibility = assess_candidate_compatibility(query, cards)

    assert tuple(turn.turn_id for turn in substantive) == ("formula-turn",)
    assert decisions[-1].reason == "unresolved_reference_response"
    assert compatibility[0].compatible is True
