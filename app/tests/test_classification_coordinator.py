"""Characterization tests for the single academic-classification boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from schemas.conversation import (
    ContextNeedAssessment,
    ConversationCandidateCard,
    ConversationPreparation,
    RecentContextLoadResult,
    RecentConversationTurn,
)
from schemas.doubt_solver import QueryClassification
from schemas.image_question_classification import (
    ImageClassificationResult,
    ImageClassificationStatus,
    ImageParseMetadata,
    VisualContext,
)
from services.classification.coordinator import ClassificationCoordinator
from services.classification.image_classification_adapter import (
    adapt_image_classification,
)
from services.classification.web_search_demand import normalize_web_search_demand
from services.conversation.candidate_builder import build_candidate_cards


def _classification(**updates: Any) -> QueryClassification:
    values: dict[str, Any] = {
        "intent": "solve_question",
        "subject": "math",
        "confidence": 0.96,
        "difficulty": "intermediate",
        "retrieval_need": "none",
        "classification_source": "llm",
    }
    values.update(updates)
    return QueryClassification(**values)


def test_text_classification_calls_existing_classifier_once() -> None:
    calls: list[str] = []

    def classifier(query: str, **_: object) -> QueryClassification:
        calls.append(query)
        return _classification()

    result = ClassificationCoordinator(classifier=classifier).classify_text(
        query="Solve 2x + 5 = 15.",
        request_id="request-1",
    )

    assert calls == ["Solve 2x + 5 = 15."]
    assert result.status == "validated"
    assert result.classification["subject"] == "math"
    assert result.classification["intent"] == "solve"


def test_image_adapter_does_not_call_text_classifier() -> None:
    def unexpected_classifier(query: str, **_: object) -> QueryClassification:
        raise AssertionError(f"text classifier called for image: {query}")

    result = adapt_image_classification(
        ImageClassificationResult(
            status=ImageClassificationStatus.CLASSIFIED,
            normalized_query="Solve 2x + 5 = 15.",
            classification=_classification(),
        ),
        request_id="request-2",
        coordinator=ClassificationCoordinator(classifier=unexpected_classifier),
    )

    assert result.modality == "image"
    assert result.status == "validated"
    assert result.classification["subject"] == "math"


def test_image_routing_conflict_calls_strong_once_when_extracted_text_is_safe() -> None:
    calls: list[str] = []

    def strong(query: str, **_: object) -> QueryClassification:
        calls.append(query)
        return _classification(pattern_family_candidate="MATH")

    result = ClassificationCoordinator(
        strong_classifier=strong
    ).accept_image_classification(
        query="What is 20% of 500?",
        classification=_classification(pattern_family_candidate="REASONING"),
        request_id="image-safe-strong",
        image_metadata=ImageParseMetadata(
            has_question=True,
            extraction_confidence=0.96,
            classification_confidence=0.96,
        ),
    )

    assert calls == ["What is 20% of 500?"]
    assert result.status == "validated"
    assert result.raw.subject == "math"


def test_image_visual_conflict_never_calls_text_only_strong_model() -> None:
    def unexpected_strong(query: str, **_: object) -> QueryClassification:
        raise AssertionError(f"strong classifier called for visual image: {query}")

    result = ClassificationCoordinator(
        strong_classifier=unexpected_strong
    ).accept_image_classification(
        query="Choose the missing figure.",
        classification=_classification(pattern_family_candidate="REASONING"),
        request_id="image-unsafe-strong",
        image_metadata=ImageParseMetadata(
            has_question=True,
            has_visual=True,
            visual_context=VisualContext(type="reasoning_figure", confidence=0.75),
            extraction_confidence=0.96,
            classification_confidence=0.96,
            warnings=["visual relationship uncertain"],
        ),
    )

    assert result.status == "technical_fallback"
    assert result.raw.classification_source == "fallback"


def test_text_and_image_use_the_same_downstream_mapping() -> None:
    raw = _classification(
        topic="current office holder",
        need_web_search=True,
        web_search_reason="freshness_required",
        web_search_query="current office holder official",
    )
    text = ClassificationCoordinator(
        classifier=lambda query, **kwargs: raw
    ).classify_text(
        query="Solve 2x + 5 = 15.",
        request_id="request-3",
    )
    image = ClassificationCoordinator().accept_image_classification(
        query="Solve 2x + 5 = 15.",
        classification=raw,
        request_id="request-4",
    )

    assert text.classification == image.classification
    assert image.classification["need_web_search"] is True
    assert image.classification["web_search_reason"] == "freshness_required"
    assert image.classification["web_search_query"] == "current office holder official"


@pytest.mark.parametrize(
    "query",
    (
        "provide current affairs question july 2026",
        "provide curent affairs cuqestion july 2026",
    ),
)
def test_current_affairs_demand_is_normalized_when_model_omits_it(query: str) -> None:
    result = ClassificationCoordinator(
        classifier=lambda query, **kwargs: _classification(
            intent="practice_question",
            subject="general",
            need_web_search=False,
        )
    ).classify_text(query=query, request_id="current-affairs-demand")

    assert result.classification["need_web_search"] is True
    assert result.classification["web_search_reason"] == "current_affairs"
    assert result.classification["web_search_query"] == "current affairs july 2026"


def test_static_definition_of_current_affairs_does_not_search() -> None:
    result = normalize_web_search_demand(
        "Explain the meaning of current affairs.",
        _classification(
            intent="explain_concept",
            subject="general",
            need_web_search=False,
        ),
    )

    assert result.need_web_search is False


def test_model_required_current_affairs_query_is_canonicalized() -> None:
    result = normalize_web_search_demand(
        "provide current affairs questions July 2026",
        _classification(
            intent="practice_question",
            subject="general",
            need_web_search=True,
            web_search_reason="explicit_latest_request",
            web_search_query="current affairs questions in July 2026",
        ),
    )

    assert result.need_web_search is True
    assert result.web_search_reason == "explicit_latest_request"
    assert result.web_search_query == "current affairs July 2026"


def test_more_questions_reuses_format_but_refreshes_current_facts() -> None:
    result = normalize_web_search_demand(
        "Give me five more.",
        _classification(
            intent="practice_question",
            subject="general",
            need_web_search=False,
            relation="FOLLOW_UP",
            selected_turn_id="turn-current-affairs",
            requested_action="GENERATE_SIMILAR",
            requires_recent_conversation=True,
        ),
        candidate_cards=(
            ConversationCandidateCard(
                turn_id="turn-current-affairs",
                question_preview="Provide current affairs questions for July 2026.",
                answer_clue="Previously generated question format.",
                subject="general",
                topic="Current Affairs",
            ),
        ),
    )

    assert result.need_web_search is True
    assert result.web_search_reason == "current_affairs"
    assert result.web_search_query == "current affairs July 2026"


def test_malformed_required_field_becomes_typed_technical_fallback() -> None:
    result = ClassificationCoordinator(
        classifier=lambda query, **kwargs: _classification(subject="")
    ).classify_text(
        query="What is the capital of Rajasthan?",
        request_id="request-5",
    )

    assert result.status == "technical_fallback"
    assert result.raw.subject == "general"
    assert result.raw.requires_recent_conversation is False
    assert result.classification["requires_recent_conversation"] is False


def test_classifier_exception_is_not_reported_as_student_ambiguity() -> None:
    def failing_classifier(query: str, **_: object) -> QueryClassification:
        raise TimeoutError(query)

    result = ClassificationCoordinator(
        classifier=failing_classifier
    ).classify_text(
        query="Find the next number: 2, 4, 8, 16.",
        request_id="request-6",
    )

    assert result.status == "technical_fallback"
    assert result.raw.reasoning_summary == "technical_classifier_fallback"
    assert result.raw.requires_recent_conversation is False


def _conversation_preparation() -> ConversationPreparation:
    turn = RecentConversationTurn(
        turn_id="turn-1",
        original_query="Calculate 75% of 200.",
        final_answer="Final Answer: 150",
        created_at=datetime.now(UTC),
    )
    cards = build_candidate_cards(
        (turn,),
        current_query="Why did you divide by 100?",
    )
    return ConversationPreparation(
        gate=ContextNeedAssessment(decision="CONTEXT_REQUIRED"),
        context_load=RecentContextLoadResult(turns=(turn,)),
        eligible_turns=(turn,),
        candidates=cards,
    )


def test_context_candidates_reach_existing_classifier_once() -> None:
    calls: list[dict[str, object]] = []

    def classifier(query: str, **kwargs: object) -> QueryClassification:
        calls.append({"query": query, **kwargs})
        return _classification(
            intent="explain_concept",
            relation="FOLLOW_UP",
            selected_turn_id="turn-1",
            requested_action="EXPLAIN_PREVIOUS",
        )

    result = ClassificationCoordinator(classifier=classifier).classify_text(
        query="Why did you divide by 100?",
        request_id="request-context",
        conversation=_conversation_preparation(),
    )

    assert len(calls) == 1
    assert "turn-1" in str(calls[0]["conversation_candidates"])
    assert result.raw.selected_turn_id == "turn-1"


def test_unknown_selected_turn_becomes_typed_technical_fallback() -> None:
    result = ClassificationCoordinator(
        classifier=lambda query, **kwargs: _classification(
            relation="FOLLOW_UP",
            selected_turn_id="invented-turn",
            requested_action="EXPLAIN_PREVIOUS",
        )
    ).classify_text(
        query="Explain the last step.",
        request_id="request-invalid-turn",
        conversation=_conversation_preparation(),
    )

    assert result.status == "technical_fallback"
    assert result.raw.relation == "AMBIGUOUS"
    assert result.raw.selected_turn_id is None
