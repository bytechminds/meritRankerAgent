"""Application service for validated, cached image-question classification."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Protocol

from observability import record_llm_call
from schemas.image_input import ImageInput
from schemas.image_question_classification import (
    ImageClassificationResult,
    ImageClassificationStatus,
    ImageProviderRequest,
)
from schemas.llm_usage import ProviderTokenUsage, UsageStatus
from services.image_question_classification.cache import ImageClassificationCache, SingleFlight
from services.image_question_classification.errors import (
    ImageProviderResponseError,
    ImageProviderTemporaryError,
    InvalidImageError,
)
from services.image_question_classification.image_validator import ImageInputValidator
from services.image_question_classification.prompt_builder import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    build_image_classifier_prompt,
)
from services.image_question_classification.provider import (
    ImageQuestionClassificationProvider,
)

logger = logging.getLogger(__name__)

_USER_MESSAGES = {
    ImageClassificationStatus.REJECTED_UNREADABLE_IMAGE: (
        "The question is not readable clearly. Please crop the question and upload a sharper image."
    ),
    ImageClassificationStatus.REJECTED_NO_QUESTION_FOUND: (
        "I could not identify a question in this image. Please upload the complete question."
    ),
    ImageClassificationStatus.REJECTED_AMBIGUOUS_QUESTION: (
        "Multiple questions were detected. Please crop and upload the question you want to ask."
    ),
    ImageClassificationStatus.REJECTED_INCOMPLETE_CONTEXT: (
        "I could not identify a complete question. Please include the full question "
        "and its options."
    ),
    ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE: (
        "This image could not be processed. Please upload a JPEG, PNG, or WebP image."
    ),
    ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE: (
        "Image processing is temporarily unavailable. Please try again."
    ),
}


class ImageQuestionClassifierProtocol(Protocol):
    def classify(
        self,
        *,
        image: ImageInput,
        instruction: str | None,
        request_id: str,
    ) -> ImageClassificationResult:
        """Validate, extract, and classify one image question."""


class ImageQuestionClassifier:
    def __init__(
        self,
        *,
        provider: ImageQuestionClassificationProvider,
        validator: ImageInputValidator,
        cache: ImageClassificationCache,
        provider_name: str,
        model: str,
        timeout_seconds: float,
        max_output_tokens: int,
        min_confidence: float,
        single_flight: SingleFlight | None = None,
    ) -> None:
        self._provider = provider
        self._validator = validator
        self._cache = cache
        self._provider_name = provider_name
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._min_confidence = min_confidence
        self._single_flight = single_flight or SingleFlight()
        self._prompt = build_image_classifier_prompt()
        self._prompt_hash = hashlib.sha256(self._prompt.encode("utf-8")).hexdigest()

    def classify(
        self,
        *,
        image: ImageInput,
        instruction: str | None,
        request_id: str,
    ) -> ImageClassificationResult:
        started_at = time.monotonic()
        try:
            validated = self._validator.validate(image, timeout_seconds=self._timeout_seconds)
        except InvalidImageError:
            result = self._result(ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE)
            self._log_outcome(
                request_id,
                result,
                started_at,
                provider_latency_ms=0,
                provider_call_count=0,
            )
            return result

        cache_key = self._cache_key(validated.content_hash, instruction)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._log_outcome(
                request_id,
                cached,
                started_at,
                provider_latency_ms=0,
                provider_call_count=0,
            )
            return cached

        is_leader, flight = self._single_flight.enter(cache_key)
        if not is_leader:
            result = self._single_flight.wait(
                cache_key,
                flight,
                timeout_seconds=self._timeout_seconds + 1.0,
            )
            result = result or self._result(
                ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE
            )
            self._log_outcome(
                request_id,
                result,
                started_at,
                provider_latency_ms=0,
                provider_call_count=0,
            )
            return result

        result = self._result(ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE)
        try:
            request = ImageProviderRequest(
                image=validated,
                instruction=instruction,
                prompt=self._prompt,
                model=self._model,
                timeout_seconds=self._timeout_seconds,
                max_output_tokens=self._max_output_tokens,
            )
            provider_started_at = time.monotonic()
            result, provider_call_count = self._execute_provider(
                request,
                request_id=request_id,
            )
            provider_latency_ms = int((time.monotonic() - provider_started_at) * 1000)
            if result.status != ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE:
                self._cache.put(cache_key, result)
            self._log_outcome(
                request_id,
                result,
                started_at,
                provider_latency_ms=provider_latency_ms,
                provider_call_count=provider_call_count,
            )
            return result
        finally:
            self._single_flight.finish(cache_key, flight, result)

    def _execute_provider(
        self,
        request: ImageProviderRequest,
        *,
        request_id: str,
    ) -> tuple[ImageClassificationResult, int]:
        for attempt in range(2):
            attempt_started_at = time.monotonic()
            try:
                output = self._provider.classify(request)
                self._record_provider_attempt(
                    request_id=request_id,
                    attempt=attempt,
                    started_at=attempt_started_at,
                    usage=output.provider_usage or ProviderTokenUsage(),
                    status="succeeded",
                )
                break
            except ImageProviderTemporaryError as exc:
                self._record_provider_attempt(
                    request_id=request_id,
                    attempt=attempt,
                    started_at=attempt_started_at,
                    usage=exc.provider_usage or ProviderTokenUsage(),
                    status="failed",
                    error_type=type(exc).__name__,
                )
                if attempt == 0:
                    continue
                return (
                    self._result(
                        ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE
                    ),
                    attempt + 1,
                )
            except ImageProviderResponseError as exc:
                self._record_provider_attempt(
                    request_id=request_id,
                    attempt=attempt,
                    started_at=attempt_started_at,
                    usage=exc.provider_usage or ProviderTokenUsage(),
                    status="failed",
                    error_type=type(exc).__name__,
                )
                return (
                    self._result(
                        ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE
                    ),
                    attempt + 1,
                )
        else:  # pragma: no cover - loop always returns or breaks
            return (
                self._result(
                    ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE
                ),
                2,
            )

        if output.status != ImageClassificationStatus.CLASSIFIED:
            return (
                ImageClassificationResult(
                    status=output.status,
                    image_parse_metadata=output.image_parse_metadata,
                    user_message=_USER_MESSAGES.get(output.status),
                ),
                attempt + 1,
            )

        classification = output.classification
        if classification is None or output.normalized_query is None:
            return (
                self._result(ImageClassificationStatus.REJECTED_UNREADABLE_IMAGE),
                attempt + 1,
            )
        confidence = min(
            classification.confidence,
            output.image_parse_metadata.extraction_confidence,
            output.image_parse_metadata.classification_confidence,
        )
        if confidence < self._min_confidence:
            return (
                ImageClassificationResult(
                    status=ImageClassificationStatus.REJECTED_AMBIGUOUS_QUESTION,
                    image_parse_metadata=output.image_parse_metadata,
                    user_message=_USER_MESSAGES[
                        ImageClassificationStatus.REJECTED_AMBIGUOUS_QUESTION
                    ],
                ),
                attempt + 1,
            )
        if classification.subject == "unknown":
            classification = classification.model_copy(update={"subject": "general"})
        classification = classification.model_copy(update={"classification_source": "llm"})
        return (
            ImageClassificationResult(
                status=ImageClassificationStatus.CLASSIFIED,
                normalized_query=output.normalized_query,
                classification=classification,
                image_parse_metadata=output.image_parse_metadata,
            ),
            attempt + 1,
        )

    def _record_provider_attempt(
        self,
        *,
        request_id: str,
        attempt: int,
        started_at: float,
        usage: ProviderTokenUsage,
        status: UsageStatus,
        error_type: str | None = None,
    ) -> None:
        record_llm_call(
            request_id=request_id,
            role="image_question_classifier",
            provider=self._provider_name,
            model=self._model,
            deployment=None,
            attempt_type="primary" if attempt == 0 else "retry",
            streaming=False,
            usage=usage,
            duration_ms=max(int((time.monotonic() - started_at) * 1000), 0),
            status=status,
            error_type=error_type,
        )

    def _cache_key(self, content_hash: str, instruction: str | None) -> str:
        instruction_hash = hashlib.sha256((instruction or "").encode("utf-8")).hexdigest()
        material = "|".join(
            (
                content_hash,
                instruction_hash,
                PROMPT_VERSION,
                SCHEMA_VERSION,
                self._prompt_hash,
                self._provider_name,
                self._model,
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _result(status: ImageClassificationStatus) -> ImageClassificationResult:
        return ImageClassificationResult(status=status, user_message=_USER_MESSAGES.get(status))

    def _log_outcome(
        self,
        request_id: str,
        result: ImageClassificationResult,
        started_at: float,
        *,
        provider_latency_ms: int,
        provider_call_count: int,
    ) -> None:
        if result.status == ImageClassificationStatus.CLASSIFIED:
            outcome_class = "success"
        elif result.status == ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE:
            outcome_class = "error"
        else:
            outcome_class = "rejection"
        logger.info(
            "image_question_classifier request_id=%s status=%s outcome_class=%s metric_count=1 "
            "cache_hit=%s latency_ms=%d provider_latency_ms=%d confidence=%.3f "
            "provider_call_count=%d text_classifier_bypassed=true "
            "prompt_version=%s schema_version=%s provider=gemini model=%s "
            "need_web_search=%s",
            request_id,
            result.status.value,
            outcome_class,
            result.cache_hit,
            int((time.monotonic() - started_at) * 1000),
            provider_latency_ms,
            result.image_parse_metadata.classification_confidence,
            provider_call_count,
            PROMPT_VERSION,
            SCHEMA_VERSION,
            self._model,
            str(
                bool(
                    result.classification
                    and result.classification.need_web_search
                )
            ).lower(),
        )
