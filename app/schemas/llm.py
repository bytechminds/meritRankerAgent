"""
app/schemas/llm.py
------------------
Provider-neutral Pydantic v2 schemas for the LLM layer.

All schema changes here are a public contract — do not rename or remove fields
without a migration plan and updated tests.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LlmMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class LlmRoleConfig(BaseModel):
    """Configuration for a named LLM role (e.g. 'classifier', 'solver').

    Loaded from LLM_ROLE_CONFIG_JSON env var at runtime.
    """

    provider: Literal["mock", "azure_openai", "openai"]
    model_label: str = Field(..., min_length=1, description="Human-readable label for logging.")
    deployment: str | None = Field(
        default=None,
        description="Azure deployment name — required for azure_openai provider",
    )
    model: str | None = Field(
        default=None,
        description="Model ID — required for openai provider",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, gt=0)
    supports_streaming: bool = False


class LlmRequest(BaseModel):
    """Internal request passed to a provider's generate/stream method."""

    role: str = Field(..., min_length=1, description="Named role that owns this request")
    messages: list[LlmMessage] = Field(..., min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)


class LlmResponse(BaseModel):
    """Provider-neutral response from a generate call."""

    role: str
    provider: str
    model_label: str
    content: str
    finish_reason: str | None = None
    input_tokens: int | None = Field(default=None, ge=0, exclude=True)
    output_tokens: int | None = Field(default=None, ge=0, exclude=True)
    total_tokens: int | None = Field(default=None, ge=0, exclude=True)
    cached_input_tokens: int | None = Field(default=None, ge=0, exclude=True)
    reasoning_tokens: int | None = Field(default=None, ge=0, exclude=True)


class LlmStreamChunk(BaseModel):
    """One chunk from a streaming generate call."""

    role: str
    provider: str
    model_label: str
    content_delta: str
    is_final: bool = False
