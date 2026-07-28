"""Adapt existing image classification into the shared academic stage."""

from __future__ import annotations

from schemas.image_question_classification import ImageClassificationResult
from services.classification.contracts import ClassificationStageResult
from services.classification.coordinator import ClassificationCoordinator


def adapt_image_classification(
    result: ImageClassificationResult,
    *,
    request_id: str,
    coordinator: ClassificationCoordinator,
) -> ClassificationStageResult:
    if result.classification is None or not result.normalized_query:
        raise ValueError("Classified image result is incomplete.")
    return coordinator.accept_image_classification(
        query=result.normalized_query,
        classification=result.classification,
        request_id=request_id,
        image_metadata=result.image_parse_metadata,
    )
