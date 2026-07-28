"""Gemini implementation of the provider-neutral image classifier contract."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from schemas.doubt_solver import (
    QueryClassification,
    QueryDifficulty,
    QueryIntent,
    ResponseStyle,
    RetrievalNeed,
)
from schemas.image_question_classification import (
    ImageClassificationStatus,
    ImageParseMetadata,
    ImageProviderOutput,
    ImageProviderRequest,
    VisualContext,
    VisualType,
)
from schemas.llm_usage import ProviderTokenUsage
from services.image_question_classification.errors import (
    ImageProviderResponseError,
    ImageProviderTemporaryError,
)
from services.llm.providers.usage import extract_gemini_usage

logger = logging.getLogger(__name__)


class _GeminiVisualEntity(BaseModel):
    name: str
    details: list[str]

    model_config = {"extra": "forbid"}


class _GeminiVisualRelationship(BaseModel):
    source: str
    relation: str
    target: str

    model_config = {"extra": "forbid"}


class _GeminiVisualValue(BaseModel):
    label: str
    value: str
    unit: str

    model_config = {"extra": "forbid"}


class _GeminiVisualContext(BaseModel):
    type: VisualType
    labels: list[str]
    entities: list[_GeminiVisualEntity]
    relationships: list[_GeminiVisualRelationship]
    values: list[_GeminiVisualValue]
    description: str
    confidence: float

    model_config = {"extra": "forbid"}


class _GeminiImageParseMetadata(BaseModel):
    has_question: bool
    has_options: bool
    has_visual: bool
    visual_context: _GeminiVisualContext
    extraction_confidence: float
    classification_confidence: float
    ignored_content_detected: bool
    warnings: list[str]

    model_config = {"extra": "forbid"}


class _GeminiQueryClassification(BaseModel):
    intent: QueryIntent
    subject: str
    topic: str
    topic_confidence: float
    pattern_topic_candidate: str
    pattern_family_candidate: str
    retrieval_tags: list[str]
    response_style: ResponseStyle
    confidence: float
    difficulty: QueryDifficulty
    retrieval_need: RetrievalNeed
    reasoning_summary: str
    need_web_search: bool
    web_search_reason: Literal[
        "",
        "explicit_latest_request",
        "current_affairs",
        "current_economy",
        "latest_exam_update",
        "current_event",
        "user_requested_web",
        "freshness_required",
        "none",
    ]
    web_search_query: str

    model_config = {"extra": "forbid"}


class _GeminiImageProviderOutput(BaseModel):
    """Gemini-compatible fixed wire shape mapped into the domain contract."""

    status: ImageClassificationStatus
    normalized_query: str
    classification: _GeminiQueryClassification
    image_parse_metadata: _GeminiImageParseMetadata

    model_config = {"extra": "forbid"}


class GeminiImageQuestionClassificationProvider:
    """One-call Gemini extraction and classification adapter."""

    def __init__(
        self,
        *,
        api_key: str,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini image-classifier API key is required.")
        self._api_key = api_key
        self._client_factory = client_factory

    def classify(self, request: ImageProviderRequest) -> ImageProviderOutput:
        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        instruction = request.instruction or "Classify the question shown in the image."
        client_factory = self._client_factory or genai.Client
        client = client_factory(
            api_key=self._api_key,
            http_options=types.HttpOptions(
                timeout=int(request.timeout_seconds * 1000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

        try:
            response = client.models.generate_content(
                model=request.model,
                contents=[
                    types.Part.from_text(text=instruction),
                    types.Part.from_bytes(
                        data=request.image.content,
                        mime_type=request.image.mime_type,
                    ),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=request.prompt,
                    response_mime_type="application/json",
                    response_schema=_GeminiImageProviderOutput,
                    temperature=0.0,
                    max_output_tokens=request.max_output_tokens,
                ),
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            message = str(exc).casefold()
            is_temporary = (
                status_code == 429
                or (isinstance(status_code, int) and status_code >= 500)
                or "timeout" in message
                or "deadline" in message
                or "connection" in message
            )
            logger.warning(
                "gemini_image_classifier provider_error category=%s "
                "status_code=%s error_type=%s model=%s",
                "temporary" if is_temporary else "execution",
                status_code if status_code is not None else "none",
                type(exc).__name__,
                request.model,
            )
            if is_temporary:
                raise ImageProviderTemporaryError(
                    "Image provider is temporarily unavailable."
                ) from exc
            raise ImageProviderResponseError("Image provider rejected the request.") from exc

        provider_usage = extract_gemini_usage(response)
        try:
            parsed = response.parsed
            if parsed is not None:
                wire_output = _GeminiImageProviderOutput.model_validate(parsed)
                return self._to_domain_output(
                    wire_output,
                    provider_usage=provider_usage,
                )
            if response.text:
                wire_output = _GeminiImageProviderOutput.model_validate_json(response.text)
                return self._to_domain_output(
                    wire_output,
                    provider_usage=provider_usage,
                )
        except (AttributeError, TypeError, ValidationError, ValueError) as exc:
            raise ImageProviderResponseError(
                "Image provider returned invalid structured output.",
                provider_usage=provider_usage,
            ) from exc
        raise ImageProviderResponseError(
            "Image provider returned no structured output.",
            provider_usage=provider_usage,
        )

    @staticmethod
    def _to_domain_output(
        wire: _GeminiImageProviderOutput,
        *,
        provider_usage: ProviderTokenUsage,
    ) -> ImageProviderOutput:
        visual = wire.image_parse_metadata.visual_context
        metadata = ImageParseMetadata(
            has_question=wire.image_parse_metadata.has_question,
            has_options=wire.image_parse_metadata.has_options,
            has_visual=wire.image_parse_metadata.has_visual,
            visual_context=VisualContext(
                type=visual.type,
                labels=visual.labels,
                entities=[entity.model_dump() for entity in visual.entities],
                relationships=[
                    relationship.model_dump() for relationship in visual.relationships
                ],
                values=[value.model_dump() for value in visual.values],
                description=visual.description or None,
                confidence=visual.confidence,
            ),
            extraction_confidence=wire.image_parse_metadata.extraction_confidence,
            classification_confidence=(
                wire.image_parse_metadata.classification_confidence
            ),
            ignored_content_detected=(
                wire.image_parse_metadata.ignored_content_detected
            ),
            warnings=wire.image_parse_metadata.warnings,
        )
        if wire.status != ImageClassificationStatus.CLASSIFIED:
            return ImageProviderOutput(
                status=wire.status,
                image_parse_metadata=metadata,
                provider_usage=provider_usage,
            )

        classification = wire.classification
        return ImageProviderOutput(
            status=wire.status,
            normalized_query=wire.normalized_query,
            classification=QueryClassification(
                intent=classification.intent,
                subject=classification.subject,
                topic=classification.topic or None,
                topic_confidence=classification.topic_confidence,
                pattern_topic_candidate=classification.pattern_topic_candidate or None,
                pattern_family_candidate=classification.pattern_family_candidate or None,
                retrieval_tags=classification.retrieval_tags,
                response_style=classification.response_style,
                confidence=classification.confidence,
                difficulty=classification.difficulty,
                retrieval_need=classification.retrieval_need,
                classification_source="llm",
                reasoning_summary=classification.reasoning_summary or None,
                need_web_search=classification.need_web_search,
                web_search_reason=classification.web_search_reason or None,
                web_search_query=classification.web_search_query or None,
            ),
            image_parse_metadata=metadata,
            provider_usage=provider_usage,
        )
