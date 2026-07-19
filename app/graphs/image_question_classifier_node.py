"""Graph-compatible entry node for image-question classification."""

from __future__ import annotations

from dataclasses import dataclass

from schemas.image_input import ImageInput
from schemas.image_question_classification import ImageClassificationResult
from services.image_question_classification.classifier import (
    ImageQuestionClassifierProtocol,
)


@dataclass(frozen=True)
class ImageClassifierNodeInput:
    request_id: str
    image: ImageInput
    instruction: str | None = None


def image_classifier_node(
    input: ImageClassifierNodeInput,
    *,
    classifier: ImageQuestionClassifierProtocol,
) -> ImageClassificationResult:
    """Delegate image classification without importing any provider implementation."""
    return classifier.classify(
        image=input.image,
        instruction=input.instruction,
        request_id=input.request_id,
    )
