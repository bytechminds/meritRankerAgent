"""Adapter and mapping for the existing query classifier."""

from __future__ import annotations

from collections.abc import Callable

from schemas.conversation import ConversationCandidateCard
from schemas.doubt_solver import DoubtSolverClassification, QueryClassification
from services.query_classifier_service import (
    apply_classification_policy,
    apply_classification_sanity,
    classify_query,
)

ClassifierCallable = Callable[..., QueryClassification]

ACADEMIC_SUBJECT_MAP: dict[str, str] = {
    "math": "math",
    "reasoning": "reasoning",
    "english": "english",
    "general": "general",
    "science": "general",
    "unknown": "general",
}
ACADEMIC_INTENT_MAP: dict[str, str] = {
    "solve_question": "solve",
    "explain_concept": "explain",
    "explain_option": "explain",
    "general_doubt": "explain",
    "practice_question": "practice",
    "visualize_question": "visualize",
    "unknown": "explain",
}


def run_existing_classifier(
    query: str,
    *,
    request_id: str,
    on_before_strong_classifier: Callable[[], None] | None = None,
    conversation_candidates: str | None = None,
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
    candidate_turn_ids: tuple[str, ...] = (),
    context_gate: str = "CONTEXT_NOT_NEEDED",
    classifier: ClassifierCallable = classify_query,
) -> QueryClassification:
    """Call the existing primary/strong classifier coordinator once."""
    kwargs: dict[str, object] = {}
    if candidate_cards:
        kwargs["candidate_cards"] = candidate_cards
    return classifier(
        query,
        request_id=request_id or None,
        on_before_strong_classifier=on_before_strong_classifier,
        conversation_candidates=conversation_candidates,
        candidate_turn_ids=candidate_turn_ids,
        context_gate=context_gate,
        **kwargs,
    )


def map_academic_classification(
    raw: QueryClassification,
    *,
    query: str,
    request_id: str,
) -> dict[str, object]:
    """Map the existing public classifier schema to downstream graph fields."""
    classification = DoubtSolverClassification(
        subject=ACADEMIC_SUBJECT_MAP.get(raw.subject, "general"),
        intent=ACADEMIC_INTENT_MAP.get(raw.intent, "explain"),
        difficulty=raw.difficulty,
        retrieval_required=raw.retrieval_need != "none",
        requires_recent_conversation=raw.requires_recent_conversation,
        topic=raw.topic,
        topic_confidence=raw.topic_confidence,
        pattern_topic_candidate=raw.pattern_topic_candidate,
        pattern_family_candidate=raw.pattern_family_candidate,
        retrieval_tags=raw.retrieval_tags,
        need_web_search=raw.need_web_search,
        web_search_reason=raw.web_search_reason,
        web_search_query=raw.web_search_query,
    ).model_dump()
    classification = apply_classification_sanity(
        query,
        classification,
        request_id=request_id,
        classifier_confidence=raw.confidence,
    )
    return apply_classification_policy(
        query,
        classification,
        request_id=request_id,
        classifier_confidence=raw.confidence,
    )
