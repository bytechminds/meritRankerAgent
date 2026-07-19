"""Startup factory for the configured image-question classifier."""

from __future__ import annotations

from config import Settings
from services.image_question_classification.cache import ImageClassificationCache
from services.image_question_classification.classifier import ImageQuestionClassifier
from services.image_question_classification.gemini_provider import (
    GeminiImageQuestionClassificationProvider,
)
from services.image_question_classification.image_loader import InlineImageLoader
from services.image_question_classification.image_validator import (
    ImageInputValidator,
    ImageValidationConfig,
)


def build_image_question_classifier(settings: Settings) -> ImageQuestionClassifier:
    """Build the provider-neutral application service once at startup."""
    if settings.image_classifier_provider != "gemini":
        raise ValueError("Unsupported image classifier provider.")
    provider = GeminiImageQuestionClassificationProvider(
        api_key=settings.image_classifier_api_key
    )
    validator = ImageInputValidator(
        loader=InlineImageLoader(),
        config=ImageValidationConfig(
            max_image_bytes=settings.image_classifier_max_image_bytes,
            max_dimension=settings.image_classifier_max_dimension,
        ),
    )
    return ImageQuestionClassifier(
        provider=provider,
        validator=validator,
        cache=ImageClassificationCache(
            ttl_seconds=settings.image_classifier_cache_ttl_seconds
        ),
        provider_name=settings.image_classifier_provider,
        model=settings.image_classifier_model,
        timeout_seconds=settings.image_classifier_timeout_ms / 1000.0,
        max_output_tokens=settings.image_classifier_max_output_tokens,
        min_confidence=settings.image_classifier_min_confidence,
    )
