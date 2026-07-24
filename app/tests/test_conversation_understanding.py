from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from schemas.conversation import RecentContextLoadResult, RecentConversationTurn
from services.conversation.conversation_understanding import (
    ConversationUnderstandingService,
)
from services.doubt_solver.answer_generation_adapter import AnswerGenerationAdapter
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)
from services.llm.orchestration.orchestrator import create_mock_orchestrator_for_tests
from services.query_classifier_service import requires_recent_conversation


def _turn(
    turn_id: str,
    question: str,
    answer: str,
    *,
    minutes_ago: int,
) -> RecentConversationTurn:
    return RecentConversationTurn(
        turn_id=turn_id,
        original_query=question,
        final_answer=answer,
        created_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
    )


def _service(*turns: RecentConversationTurn, source: str = "agentcore_memory"):
    persistence = Mock()
    persistence.load_recent_context.return_value = RecentContextLoadResult(
        source=source,
        memory_attempted=True,
        memory_status="succeeded",
        memory_event_count=len(turns),
        memory_completed_pair_count=len(turns),
        memory_returned_turn_ids=tuple(turn.turn_id for turn in turns),
        turns=turns,
        usable_turn_count=len(turns),
        formatted_reference="available" if turns else "",
    )
    return ConversationUnderstandingService(persistence=persistence), persistence


@pytest.mark.parametrize(
    "query",
    [
        "how did u calculated 75%",
        "how did u calculate 75%",
        "how did you calculated 75%",
        "how u got 75%",
        "why 75 percent",
        "75% kaise nikala",
        "aapne 75% kaise calculate kiya",
    ],
)
def test_percentage_reference_is_linked_to_previous_turn(query: str) -> None:
    previous = _turn(
        "percentage-turn",
        "What is the formula for calculating percentage and how is it used?",
        "Percentage = (Part / Whole) x 100. For example, 75% is 75/100 or 0.75.",
        minutes_ago=1,
    )
    service, persistence = _service(previous)

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query=query,
    )

    persistence.load_recent_context.assert_called_once_with(
        "student-1",
        "conversation-1",
        2,
    )
    assert result.context_load.source == "agentcore_memory"
    assert result.relation.relation == "follow_up"
    assert result.relation.requires_recent_conversation is True
    assert result.relation.referenced_turn_id == "percentage-turn"
    assert result.relation.referenced_turn_position == "latest_turn"
    assert result.selection_confidence >= 0.75
    assert any(
        signal == "numeric_reference=75%"
        for signal in result.relation.matched_signals
    )
    assert result.resolved_query is not None
    assert "percentage" in result.resolved_query.lower()
    assert "config" not in result.resolved_query.lower()
    assert requires_recent_conversation(result.resolved_query) is False
    assert "75%" in result.conversation_context


def test_exact_reported_typo_is_also_a_deterministic_safety_signal() -> None:
    assert requires_recent_conversation("how did u calculated 75%") is True


def test_resolved_query_keeps_academic_topic_without_generic_sentence_words() -> None:
    previous = _turn(
        "percentage-turn",
        "A student scored 150 marks out of 200. What percentage did the student score?",
        "75%",
        minutes_ago=1,
    )
    service, _ = _service(previous)

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query="how did u calculated 75%",
    )

    assert result.resolved_query == (
        "Explain how or why the referenced value or result 75% "
        "is obtained in: percentage."
    )


@pytest.mark.parametrize(
    "query",
    [
        "Explain blood relations.",
        "What is the capital of Rajasthan?",
        "Give me a Simple Interest example.",
        "What is 75% of 200?",
    ],
)
def test_unrelated_or_self_contained_query_remains_independent(query: str) -> None:
    previous = _turn(
        "percentage-turn",
        "Explain percentage.",
        "Percentage compares a part with a whole.",
        minutes_ago=1,
    )
    service, _ = _service(previous)

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query=query,
    )

    assert result.relation.relation == "independent"
    assert result.relation.requires_recent_conversation is False
    assert result.selected_turns == ()
    assert result.conversation_context == ""
    assert result.resolved_query == query


def test_numeric_reference_selects_previous_of_two_turns() -> None:
    percentage = _turn(
        "percentage-turn",
        "How do percentages work?",
        "75% means 75/100.",
        minutes_ago=2,
    )
    square_root = _turn(
        "root-turn",
        "Evaluate sqrt(16) + 2^3.",
        "The result is 12.",
        minutes_ago=1,
    )
    service, _ = _service(percentage, square_root)

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query="how did u calculated 75%?",
    )

    assert result.relation.referenced_turn_id == "percentage-turn"
    assert result.relation.referenced_turn_position == "previous_of_two"
    assert result.selected_turns == (percentage,)


def test_contextual_wording_with_new_self_contained_topic_remains_independent() -> None:
    previous = _turn(
        "percentage-turn",
        "Explain percentage.",
        "Percentage compares a part with a whole.",
        minutes_ago=1,
    )
    service, _ = _service(previous)

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query="How did you calculate acceleration from force and mass?",
    )

    assert result.relation.relation == "independent"
    assert result.selected_turns == ()
    assert result.resolved_query == (
        "How did you calculate acceleration from force and mass?"
    )


def test_generic_action_overlap_does_not_contaminate_new_topic() -> None:
    previous = _turn(
        "interest-turn",
        "Explain Simple Interest.",
        "Use SI = PRT/100.",
        minutes_ago=1,
    )
    service, _ = _service(previous)
    query = "Why did you use photosynthesis to explain sunlight?"

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query=query,
    )

    assert result.relation.relation == "independent"
    assert result.selected_turns == ()
    assert result.resolved_query == query


def test_pronoun_only_reference_selects_single_available_turn() -> None:
    previous = _turn(
        "percentage-turn",
        "Explain percentage.",
        "Percentage compares a part with a whole.",
        minutes_ago=1,
    )
    service, _ = _service(previous)

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query="Can you explain this?",
    )

    assert result.relation.relation == "follow_up"
    assert result.relation.referenced_turn_id == "percentage-turn"


def test_weak_reference_across_two_turns_requires_clarification() -> None:
    service, _ = _service(
        _turn("first-turn", "Explain 25%.", "25% is one fourth.", minutes_ago=2),
        _turn("second-turn", "Explain 75%.", "75% is three fourths.", minutes_ago=1),
    )

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query="how did you get that?",
    )

    assert result.relation.relation == "ambiguous"
    assert result.selected_turns == ()
    assert result.resolved_query is None


@pytest.mark.parametrize(
    ("question", "answer", "query", "signal"),
    [
        ("Explain simple interest.", "Use SI = PRT/100.", "Why SI?", "formula_reference=si"),
        (
            "Explain blood relations.",
            "A brother and sister are siblings.",
            "Why siblings?",
            "semantic_relevance",
        ),
    ],
)
def test_formula_and_named_concept_references_link_to_context(
    question: str,
    answer: str,
    query: str,
    signal: str,
) -> None:
    service, _ = _service(
        _turn("reference-turn", question, answer, minutes_ago=1)
    )

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query=query,
    )

    assert result.relation.relation == "follow_up"
    assert result.relation.referenced_turn_id == "reference-turn"
    assert signal in result.relation.matched_signals
    assert result.resolved_query is not None
    assert "formula:" not in result.resolved_query
    assert query.split()[-1].rstrip("?").lower() in result.resolved_query.lower()


@pytest.mark.parametrize(
    ("previous_answer", "query"),
    [
        ("sqrt(16) + 2^3 = 12.", "why 12?"),
        ("Use SI = PRT/100.", "why divide by 100?"),
        ("The correct answer is option B.", "why option B?"),
        ("First multiply, then subtract.", "which operation did you use?"),
        ("The sequence increases by 3.", "what was the pattern?"),
    ],
)
def test_other_contextual_variants_are_follow_ups(
    previous_answer: str,
    query: str,
) -> None:
    previous = _turn(
        "previous-turn",
        "Solve the previous problem.",
        previous_answer,
        minutes_ago=1,
    )
    service, _ = _service(previous)

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query=query,
    )

    assert result.relation.requires_recent_conversation is True
    assert result.relation.referenced_turn_id == "previous-turn"
    assert result.resolved_query


def test_contextual_query_without_history_requires_clarification() -> None:
    service, persistence = _service()

    result = service.understand(
        actor_id="student-1",
        conversation_id="conversation-1",
        query="how did you get that?",
    )

    persistence.load_recent_context.assert_called_once()
    assert result.relation.relation == "ambiguous"
    assert result.relation.requires_recent_conversation is True
    assert result.resolved_query is None
    assert result.conversation_context == ""


def test_full_streaming_flow_prefetches_resolves_generates_and_persists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import config as config_module

    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
    config_module._settings = None
    previous = _turn(
        "percentage-turn",
        "What is the formula for calculating percentage and how is it used?",
        "Percentage = (Part / Whole) x 100. For example, 75% is 75/100.",
        minutes_ago=1,
    )
    understanding, persistence = _service(previous)
    orchestrator, _ = create_mock_orchestrator_for_tests(
        content=(
            "**Solution**\n\n"
            "Percentage means a value out of 100.\n\n"
            "1. Write 75% as 75/100.\n"
            "2. Divide 75 by 100 to get 0.75.\n\n"
            "**Final Answer:** 75% = 75/100 = 0.75."
        )
    )
    adapter = AnswerGenerationAdapter(orchestrator=orchestrator)

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id="request-1",
                actor_id="student-1",
                conversation_id="conversation-1",
                turn_id="turn-2",
                query="how did u calculated 75%",
                original_query="how did u calculated 75%",
            ),
            adapter=adapter,
            conversation_persistence=persistence,
            conversation_understanding=understanding,
        )
    )

    assert persistence.load_recent_context.call_args.args == (
        "student-1",
        "conversation-1",
        2,
    )
    persistence.persist_completed_turn.assert_called_once()
    assert events[-1].type == "complete"
    assert events[-1].response is not None
    assert "75/100" in events[-1].response.answer
    config_module._settings = None
