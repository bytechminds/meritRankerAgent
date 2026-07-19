"""Provider-neutral image classification interface."""

from __future__ import annotations

from typing import Protocol

from schemas.image_question_classification import ImageProviderOutput, ImageProviderRequest


class ImageQuestionClassificationProvider(Protocol):
    def classify(self, request: ImageProviderRequest) -> ImageProviderOutput:
        """Extract and classify one image using one multimodal provider call."""

