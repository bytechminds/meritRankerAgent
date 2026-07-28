"""One shared coordinator for text and image academic classification."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from schemas.conversation import ConversationPreparation
from schemas.doubt_solver import QueryClassification
from schemas.image_question_classification import ImageParseMetadata
from services.classification.academic_classifier import (
    ClassifierCallable,
    map_academic_classification,
    run_existing_classifier,
)
from services.classification.classification_validator import (
    ClassificationValidationError,
    validate_conversation_classification,
    validate_mapped_classification,
    validate_raw_classification,
)
from services.classification.contracts import (
    ClassificationModality,
    ClassificationStageResult,
    ClassificationStageStatus,
)
from services.classification.web_search_demand import normalize_web_search_demand
from services.conversation.candidate_builder import format_candidate_cards
from services.query_classifier_service import (
    StrongFallbackReason,
    classify_query,
    classify_query_with_strong_model,
    evaluate_primary_classification,
)

logger = logging.getLogger(__name__)


def _technical_fallback(
    query: str,
    *,
    context_gate: str,
) -> QueryClassification:
    """Continue complete requests safely without calling failure ambiguity."""
    contextual = context_gate != "CONTEXT_NOT_NEEDED"
    return QueryClassification(
        intent="general_doubt",
        subject="general",
        confidence=0.0,
        retrieval_need="none",
        classification_source="fallback",
        requires_recent_conversation=contextual,
        relation=(
            "AMBIGUOUS"
            if contextual
            else "NEW_QUESTION"
        ),
        requested_action=(
            "ASK_CLARIFICATION"
            if contextual
            else "ANSWER_CURRENT"
        ),
        reasoning_summary="technical_classifier_fallback",
    )


class ClassificationCoordinator:
    """Validate and map exactly one final academic classification result."""

    def __init__(
        self,
        *,
        classifier: ClassifierCallable = classify_query,
        strong_classifier: Callable[..., QueryClassification] = (
            classify_query_with_strong_model
        ),
    ) -> None:
        self._classifier = classifier
        self._strong_classifier = strong_classifier

    def classify_text(
        self,
        *,
        query: str,
        request_id: str,
        on_before_strong_classifier: Callable[[], None] | None = None,
        conversation: ConversationPreparation | None = None,
    ) -> ClassificationStageResult:
        started = time.monotonic()
        status: ClassificationStageStatus = "validated"
        context_gate = (
            conversation.gate.decision
            if conversation is not None
            else "CONTEXT_NOT_NEEDED"
        )
        cards = conversation.candidates if conversation is not None else ()
        try:
            raw = run_existing_classifier(
                query,
                request_id=request_id,
                on_before_strong_classifier=on_before_strong_classifier,
                conversation_candidates=format_candidate_cards(cards) or None,
                candidate_cards=cards,
                candidate_turn_ids=tuple(card.turn_id for card in cards),
                context_gate=context_gate,
                classifier=self._classifier,
            )
            raw = validate_raw_classification(raw)
            raw = validate_conversation_classification(
                raw,
                query=query,
                candidate_cards=cards,
                candidate_turn_ids=tuple(card.turn_id for card in cards),
                context_gate=context_gate,
            )
            raw = normalize_web_search_demand(
                query,
                raw,
                candidate_cards=cards,
            )
        except Exception as exc:  # noqa: BLE001
            status = "technical_fallback"
            raw = _technical_fallback(query, context_gate=context_gate)
            logger.warning(
                "classification_coordinator technical_fallback=true "
                "error_type=%s request_id=%s",
                type(exc).__name__,
                request_id,
            )
        return self._finalize(
            raw=raw,
            query=query,
            request_id=request_id,
            modality="text",
            status=status,
            started=started,
        )

    def accept_image_classification(
        self,
        *,
        query: str,
        classification: QueryClassification,
        request_id: str,
        image_metadata: ImageParseMetadata | None = None,
    ) -> ClassificationStageResult:
        """Validate image routing, using strong text routing only when image-safe."""
        started = time.monotonic()
        try:
            raw = validate_raw_classification(classification)
            decision = evaluate_primary_classification(query, raw)
            material_reasons = tuple(
                reason
                for reason in decision.reasons
                if reason
                not in {
                    StrongFallbackReason.PRIMARY_ACCEPTED,
                    StrongFallbackReason.LOW_MATERIAL_CONFIDENCE,
                }
            )
            safe_extracted_text = bool(
                image_metadata is not None
                and image_metadata.has_question
                and not image_metadata.has_visual
                and not image_metadata.warnings
            )
            if material_reasons and safe_extracted_text:
                reason = material_reasons[0].value
                logger.info(
                    "classifier_strong_decision triggered=true role=classifier_strong "
                    "attempt=strong fallback_reason=%s modality=image request_id=%s",
                    reason,
                    request_id,
                )
                raw = self._strong_classifier(
                    query,
                    request_id=request_id,
                    primary_result=raw,
                    conflict_reason=reason,
                )
            elif material_reasons:
                raise ClassificationValidationError(
                    "Image routing conflict cannot be verified from extracted text."
                )
            raw = normalize_web_search_demand(query, raw)
            raw = raw.model_copy(
                update={
                    "relation": "NEW_QUESTION",
                    "selected_turn_id": None,
                    "requested_action": "ANSWER_CURRENT",
                    "requires_recent_conversation": False,
                }
            )
            status = "validated"
        except Exception as exc:  # noqa: BLE001
            raw = _technical_fallback(
                query,
                context_gate="CONTEXT_NOT_NEEDED",
            )
            status = "technical_fallback"
            logger.warning(
                "classification_coordinator image_fallback=true error_type=%s "
                "request_id=%s",
                type(exc).__name__,
                request_id,
            )
        return self._finalize(
            raw=raw,
            query=query,
            request_id=request_id,
            modality="image",
            status=status,
            started=started,
        )

    @staticmethod
    def _finalize(
        *,
        raw: QueryClassification,
        query: str,
        request_id: str,
        modality: ClassificationModality,
        status: ClassificationStageStatus,
        started: float,
    ) -> ClassificationStageResult:
        mapped = validate_mapped_classification(
            map_academic_classification(
                raw,
                query=query,
                request_id=request_id,
            )
        )
        resolved_status: ClassificationStageStatus = (
            "validated_fallback"
            if status == "validated" and raw.classification_source == "fallback"
            else status
        )
        logger.info(
            "classification_stage_completed request_id=%s modality=%s status=%s "
            "source=%s latency_ms=%d",
            request_id,
            modality,
            resolved_status,
            raw.classification_source,
            int((time.monotonic() - started) * 1000),
        )
        return ClassificationStageResult(
            raw=raw,
            classification=mapped,
            modality=modality,
            status=resolved_status,
            classifier_confidence=raw.confidence,
            classifier_fallback=raw.classification_source == "fallback",
        )
