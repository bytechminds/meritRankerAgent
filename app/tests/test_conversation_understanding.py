from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

from schemas.conversation import RecentContextLoadResult, RecentConversationTurn
from services.conversation.conversation_understanding import (
    ConversationUnderstandingService,
)
from services.conversation.memory_hygiene import classify_stored_turn


def _turn(
    turn_id: str,
    question: str,
    answer: str,
    *,
    turn_type: str = "academic_question_answer",
) -> RecentConversationTurn:
    return RecentConversationTurn(
        turn_id=turn_id,
        original_query=question,
        final_answer=answer,
        created_at=datetime.now(UTC),
        turn_type=turn_type,
    )


def _service(*turns: RecentConversationTurn) -> tuple[ConversationUnderstandingService, Mock]:
    persistence = Mock()

    def load(_actor: str, _conversation: str, limit: int) -> RecentContextLoadResult:
        bounded = tuple(turns[-limit:])
        return RecentContextLoadResult(
            source="agentcore_memory" if bounded else "none",
            memory_attempted=bool(bounded),
            memory_status="succeeded" if bounded else "empty",
            turns=bounded,
            usable_turn_count=len(bounded),
        )

    persistence.load_recent_context.side_effect = load
    return ConversationUnderstandingService(persistence=persistence), persistence


def test_complete_new_question_skips_memory() -> None:
    service, persistence = _service(
        _turn("old", "Calculate 75% of 200.", "Final Answer: 150")
    )

    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Solve 2x + 5 = 15.",
    )

    persistence.load_recent_context.assert_not_called()
    assert result.gate.decision == "CONTEXT_NOT_NEEDED"
    assert result.candidates == ()


def test_complete_word_problem_without_command_verb_skips_memory() -> None:
    service, persistence = _service(
        _turn("old", "Calculate 75% of 200.", "Final Answer: 150")
    )

    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query=(
            "A container holds 200 litres of an acid-water solution. It contains "
            "25% acid. Ten percent is removed and replaced with acid. The final "
            "acid percentage is required."
        ),
    )

    assert result.gate.decision == "CONTEXT_NOT_NEEDED"
    persistence.load_recent_context.assert_not_called()


def test_complete_numeric_collision_question_skips_old_context() -> None:
    service, persistence = _service(
        _turn(
            "old",
            "A mixture used 10% replacement and produced 57 litres.",
            "Final Answer: 3",
        )
    )

    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Find x when 3x + 10 = 57.",
    )

    assert result.gate.decision == "CONTEXT_NOT_NEEDED"
    assert result.candidates == ()
    persistence.load_recent_context.assert_not_called()


def test_informal_follow_up_reads_three_pairs() -> None:
    service, persistence = _service(
        _turn("percent", "Calculate 75% of 200.", "Final Answer: 150")
    )

    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="how did u calculated 75%",
    )

    persistence.load_recent_context.assert_called_once_with(
        "actor", "conversation", 3
    )
    assert result.candidates[0].turn_id == "percent"


def test_replacement_step_follow_up_reads_three_pairs() -> None:
    service, persistence = _service(
        _turn(
            "mixture",
            "A container has 200 litres of acid-water solution.",
            "Replace 10% of the new mixture with pure acid.",
        )
    )

    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="How you replaced 10% of the new mix with acid?",
    )

    assert result.gate.decision == "CONTEXT_REQUIRED"
    persistence.load_recent_context.assert_called_once_with(
        "actor", "conversation", 3
    )


def test_image_derived_equation_step_follow_up_reads_context() -> None:
    service, persistence = _service(
        _turn(
            "image-turn",
            "Solve: 2x + 7 = 19. Find x.",
            "Subtract 7 from both sides, then divide by 2. Final Answer: x = 6",
        )
    )

    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Why did you subtract 7 from both sides?",
    )

    assert result.gate.decision == "CONTEXT_REQUIRED"
    persistence.load_recent_context.assert_called_once_with(
        "actor", "conversation", 3
    )
    assert result.candidates[0].turn_id == "image-turn"


def test_correction_request_requires_context() -> None:
    service, _ = _service(_turn("task", "Solve 2x=10.", "Final Answer: x=7"))
    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Your answer is wrong.",
    )
    assert result.gate.decision == "CONTEXT_REQUIRED"


def test_resolve_from_scratch_requires_context() -> None:
    service, _ = _service(_turn("task", "Solve 2x=10.", "Final Answer: x=7"))
    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Solve it again from scratch.",
    )
    assert result.gate.decision == "CONTEXT_REQUIRED"


def test_uncertain_request_reads_context() -> None:
    service, persistence = _service(_turn("task", "What is 25% of 80?", "20"))
    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="What about 75?",
    )
    assert result.gate.decision == "UNCERTAIN"
    persistence.load_recent_context.assert_called_once()


def test_only_latest_three_candidates_are_built() -> None:
    turns = tuple(
        _turn(f"turn-{index}", f"Question {index}: solve x={index}.", f"Answer {index}")
        for index in range(5)
    )
    service, _ = _service(*turns)
    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Explain the last step.",
    )
    assert tuple(card.turn_id for card in result.candidates) == (
        "turn-2",
        "turn-3",
        "turn-4",
    )


def test_expansion_is_bounded_to_five() -> None:
    turns = tuple(
        _turn(f"turn-{index}", f"Question {index}: solve x={index}.", f"Answer {index}")
        for index in range(6)
    )
    service, persistence = _service(*turns)
    prepared = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Again wrong, resolve it.",
    )
    assert len(prepared.candidates) == 5
    assert persistence.load_recent_context.call_args_list[-1].args[-1] == 5


def test_generic_clarification_is_rejected() -> None:
    service, _ = _service(
        _turn(
            "clarification",
            "What about this?",
            "Please clarify which earlier question you mean.",
        ),
        _turn("task", "Solve 2x=10.", "Final Answer: x=5"),
    )
    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Your answer is wrong.",
    )
    assert result.rejected_turn_ids == ("clarification",)
    assert tuple(card.turn_id for card in result.candidates) == ("task",)


def test_typed_non_academic_turn_is_rejected() -> None:
    service, _ = _service(
        _turn("error", "Solve x.", "Unable to complete.", turn_type="error")
    )
    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Explain the last step.",
    )
    assert result.candidates == ()
    assert result.rejected_turn_ids == ("error",)


def test_candidate_budget_is_bounded() -> None:
    service, _ = _service(
        _turn("one", "Q " * 2400, "A " * 3900),
        _turn("two", "Q " * 2400, "A " * 3900),
        _turn("three", "Q " * 2400, "A " * 3900),
    )
    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Explain the last step.",
    )
    assert result.candidate_characters <= 6200


def test_actor_and_conversation_are_forwarded_unchanged() -> None:
    service, persistence = _service()
    service.understand(
        actor_id="actor-123",
        conversation_id="conversation-456",
        query="Explain the last step.",
    )
    persistence.load_recent_context.assert_called_once_with(
        "actor-123", "conversation-456", 3
    )


def test_memory_hygiene_classifies_generic_acknowledgement() -> None:
    decision = classify_stored_turn(
        _turn("ack", "Your answer is wrong.", "I understand that you think it is wrong.")
    )
    assert decision.usable is False
    assert decision.turn_type == "correction_only"


def test_memory_hygiene_rejects_correction_query_even_with_full_answer() -> None:
    decision = classify_stored_turn(
        _turn(
            "correction",
            "Your answer is wrong.",
            "Solve 4x + 5 = 21. Subtract 5, then divide by 4. Final Answer: x = 4",
        )
    )

    assert decision.usable is False
    assert decision.turn_type == "correction_only"
    assert decision.reason == "non_substantive_answer"


def test_memory_hygiene_rejects_resolve_meta_query_even_with_full_answer() -> None:
    decision = classify_stored_turn(
        _turn(
            "resolve",
            "Again wrong, solve it from scratch.",
            "Solve 4x + 5 = 21 independently. Final Answer: x = 4",
        )
    )

    assert decision.usable is False
    assert decision.turn_type == "correction_only"
    assert decision.reason == "non_substantive_answer"


def test_no_context_does_not_invent_candidate() -> None:
    service, _ = _service()
    result = service.understand(
        actor_id="actor",
        conversation_id="conversation",
        query="Explain the last step.",
    )
    assert result.candidates == ()
    assert result.context_load.source == "none"
