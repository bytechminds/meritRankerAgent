"""Validation rules for final academic-classification results."""

from __future__ import annotations

from pydantic import ValidationError

from schemas.conversation import ConversationCandidateCard
from schemas.doubt_solver import DoubtSolverClassification, QueryClassification
from services.conversation.context_need_gate import is_short_incomplete_topic_phrase
from services.conversation.reference_resolution import (
    analyze_reference,
    compatible_turn_ids,
    definitively_incompatible_turn_ids,
    has_multiple_compatible_entities,
    unambiguous_compatible_turn_id,
)


class ClassificationValidationError(ValueError):
    """A classifier result is structurally unusable."""


_RELATION_ACTIONS: dict[str, frozenset[str]] = {
    "NEW_QUESTION": frozenset({"ANSWER_CURRENT"}),
    "FOLLOW_UP": frozenset(
        {
            "ANSWER_WITH_CONTEXT",
            "EXPLAIN_PREVIOUS",
            "GENERATE_SIMILAR",
            "TRANSFORM_PREVIOUS",
        }
    ),
    "CONTINUATION": frozenset({"CONTINUE_PREVIOUS"}),
    "CORRECTION": frozenset({"VERIFY_AND_CORRECT"}),
    "RESOLVE_AGAIN": frozenset({"RESOLVE_FROM_SCRATCH"}),
    "AMBIGUOUS": frozenset({"ASK_CLARIFICATION"}),
}


def validate_raw_classification(
    classification: QueryClassification,
) -> QueryClassification:
    """Reject required values that the base schema permits as blank strings."""
    try:
        validated = QueryClassification.model_validate(classification)
    except ValidationError as exc:
        raise ClassificationValidationError("Classifier output is invalid.") from exc
    if not validated.subject.strip():
        raise ClassificationValidationError("Classifier subject is blank.")
    return validated


def validate_conversation_classification(
    classification: QueryClassification,
    *,
    query: str,
    candidate_turn_ids: tuple[str, ...],
    context_gate: str,
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
) -> QueryClassification:
    if (
        is_short_incomplete_topic_phrase(query)
        and classification.intent == "solve_question"
        and classification.relation == "NEW_QUESTION"
    ):
        raise ClassificationValidationError(
            "Incomplete topic phrase was classified as a numerical solve."
        )
    if classification.requested_action not in _RELATION_ACTIONS[classification.relation]:
        raise ClassificationValidationError(
            "Classifier relation and requested action are incompatible."
        )
    reference = analyze_reference(query)
    compatible_ids = compatible_turn_ids(query, candidate_cards)
    unambiguous_id = unambiguous_compatible_turn_id(query, candidate_cards)
    incompatible_ids = definitively_incompatible_turn_ids(query, candidate_cards)
    if classification.relation in {"NEW_QUESTION", "AMBIGUOUS"}:
        if classification.selected_turn_id is not None:
            raise ClassificationValidationError(
                "Classifier selected a turn for a non-selected relation."
            )
        if (
            (
                context_gate == "CONTEXT_REQUIRED"
                or (
                    reference.external_reference_detected
                    and bool(compatible_ids)
                )
            )
            and classification.relation == "NEW_QUESTION"
        ):
            raise ClassificationValidationError(
                "Explicit context reference was classified as a new question."
            )
        if (
            classification.relation == "AMBIGUOUS"
            and reference.external_reference_detected
            and unambiguous_id is not None
        ):
            raise ClassificationValidationError(
                "Unambiguous external reference was classified as ambiguous."
            )
        return classification
    if classification.selected_turn_id not in candidate_turn_ids:
        raise ClassificationValidationError("Classifier selected an unknown turn ID.")
    if (
        reference.external_reference_detected
        and has_multiple_compatible_entities(query, candidate_cards)
    ):
        raise ClassificationValidationError(
            "Multiple compatible reference entities require clarification."
        )
    if (
        reference.external_reference_detected
        and classification.selected_turn_id in incompatible_ids
    ):
        raise ClassificationValidationError(
            "Classifier selected a semantically incompatible turn."
        )
    return classification


def validate_mapped_classification(
    classification: dict[str, object],
) -> dict[str, object]:
    """Return a fully populated downstream classification mapping."""
    try:
        validated = DoubtSolverClassification.model_validate(classification)
    except ValidationError as exc:
        raise ClassificationValidationError(
            "Mapped classifier output is invalid."
        ) from exc
    if not validated.subject.strip() or not validated.intent.strip():
        raise ClassificationValidationError(
            "Mapped classifier required fields are blank."
        )
    if not validated.difficulty.strip():
        raise ClassificationValidationError(
            "Mapped classifier difficulty is blank."
        )
    return validated.model_dump()
