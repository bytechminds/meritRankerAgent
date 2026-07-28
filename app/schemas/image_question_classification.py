"""Typed contracts for image-question extraction and classification."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from schemas.doubt_solver import QueryClassification
from schemas.llm_usage import ProviderTokenUsage


class ImageClassificationStatus(StrEnum):
    CLASSIFIED = "CLASSIFIED"
    REJECTED_UNREADABLE_IMAGE = "REJECTED_UNREADABLE_IMAGE"
    REJECTED_NO_QUESTION_FOUND = "REJECTED_NO_QUESTION_FOUND"
    REJECTED_AMBIGUOUS_QUESTION = "REJECTED_AMBIGUOUS_QUESTION"
    REJECTED_INCOMPLETE_CONTEXT = "REJECTED_INCOMPLETE_CONTEXT"
    REJECTED_UNSUPPORTED_IMAGE = "REJECTED_UNSUPPORTED_IMAGE"
    PROVIDER_TEMPORARILY_UNAVAILABLE = "PROVIDER_TEMPORARILY_UNAVAILABLE"


VisualType = Literal[
    "none",
    "table",
    "chart",
    "geometry",
    "reasoning_figure",
    "map",
    "science_diagram",
    "other",
]


class VisualContext(BaseModel):
    type: VisualType = "none"
    labels: list[str] = Field(default_factory=list, max_length=100)
    entities: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    relationships: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    values: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    description: str | None = Field(default=None, max_length=2000)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class ImageParseMetadata(BaseModel):
    has_question: bool = False
    has_options: bool = False
    has_visual: bool = False
    visual_context: VisualContext = Field(default_factory=VisualContext)
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    classification_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    ignored_content_detected: bool = False
    warnings: list[str] = Field(default_factory=list, max_length=20)

    model_config = {"extra": "forbid"}


class ImageProviderOutput(BaseModel):
    """Strict provider output. It is untrusted until locally validated."""

    status: ImageClassificationStatus
    normalized_query: str | None = Field(default=None, max_length=5000)
    classification: QueryClassification | None = None
    image_parse_metadata: ImageParseMetadata = Field(default_factory=ImageParseMetadata)
    provider_usage: ProviderTokenUsage | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_status_payload(self) -> ImageProviderOutput:
        if self.status == ImageClassificationStatus.CLASSIFIED:
            if not self.normalized_query or self.classification is None:
                raise ValueError(
                    "CLASSIFIED output requires normalized_query and classification."
                )
            if not self.image_parse_metadata.has_question:
                raise ValueError("CLASSIFIED output requires has_question=true.")
        return self


class ImageClassificationResult(BaseModel):
    status: ImageClassificationStatus
    normalized_query: str | None = Field(default=None, max_length=5000)
    classification: QueryClassification | None = None
    image_parse_metadata: ImageParseMetadata = Field(default_factory=ImageParseMetadata)
    user_message: str | None = Field(default=None, max_length=300)
    cache_hit: bool = False

    model_config = {"extra": "forbid"}


class ValidatedImage(BaseModel):
    content: bytes = Field(repr=False)
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    content_hash: str = Field(min_length=64, max_length=64)
    was_resized: bool = False

    model_config = {"extra": "forbid"}


class ImageProviderRequest(BaseModel):
    image: ValidatedImage
    instruction: str | None = Field(default=None, max_length=5000)
    prompt: str = Field(min_length=1)
    model: str = Field(min_length=1, max_length=128)
    timeout_seconds: float = Field(gt=0.0, le=120.0)
    max_output_tokens: int = Field(gt=0, le=8192)

    model_config = {"extra": "forbid"}
