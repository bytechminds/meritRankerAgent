"""
app/schemas/llm_orchestration.py
---------------------------------
Pydantic v2 schemas for the LLM Orchestrator service layer (Parts 3–4).

These schemas capture resolved model metadata, provider execution input, the
result of executing a model via the ModelExecutor boundary, and the final
normalized output returned by LlmOrchestrator.

Security contract:
- Neither schema stores prompt content, user query, classification data,
  retrieved context, API keys, secrets, or credentials.
- metadata fields on both models reject unsafe keys at validation time.
- OrchestrationResult does NOT include a messages field; callers that need
  to inspect the composed messages must use MockModelExecutor.last_messages
  in tests.

Public types:
    ModelExecutionResult  — raw result returned by any ModelExecutor
    OrchestrationResult   — normalized result returned by LlmOrchestrator
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schemas.llm import LlmMessage
from schemas.llm_routing import ModelConfig, ProviderName, ProviderProfile, RouteDecision

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Metadata keys that must never appear in result metadata — they indicate
# that prompt/query/context or credentials were accidentally included.
_UNSAFE_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "system_prompt",
        "user_prompt",
        "messages",
        "query",
        "context",
        "api_key",
        "api_key_env",
        "endpoint_env",
        "api_version_env",
        "base_url_env",
        "credential_ref",
        "secret",
        "credential",
    }
)


def _validate_safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Raise ValueError if any unsafe key is present in a metadata dict."""
    bad_keys = _UNSAFE_METADATA_KEYS.intersection(value.keys())
    if bad_keys:
        raise ValueError(
            f"metadata must not contain sensitive keys: {sorted(bad_keys)}. "
            "Do not include prompt/query/context/credential data in result metadata."
        )
    return value


# ---------------------------------------------------------------------------
# ResolvedModelConfig
# ---------------------------------------------------------------------------


class ResolvedModelConfig(BaseModel):
    """Resolved model/provider metadata for execution-boundary use only."""

    model_config = ConfigDict(populate_by_name=True)

    model_alias: str = Field(..., min_length=1)
    llm_model_config: ModelConfig = Field(alias="model_config")
    provider_profile_name: str = Field(..., min_length=1)
    provider_profile: ProviderProfile
    provider: ProviderName
    supports_streaming: bool
    supports_thinking: bool
    supports_reasoning: bool = False
    reasoning_effort: str = "none"
    timeout_seconds: int = Field(ge=1, le=120)
    safe_metadata: dict[str, Any] = Field(default_factory=dict)

    def __getattribute__(self, name: str) -> Any:
        if name == "model_config":
            return object.__getattribute__(self, "llm_model_config")
        return super().__getattribute__(name)

    @model_validator(mode="after")
    def populate_safe_metadata(self) -> ResolvedModelConfig:
        self.safe_metadata = {
            "model_alias": self.model_alias,
            "provider": self.provider,
            "supports_streaming": self.supports_streaming,
            "supports_thinking": self.supports_thinking,
            "supports_reasoning": self.supports_reasoning,
            "reasoning_effort": self.reasoning_effort,
            "timeout_seconds": self.timeout_seconds,
        }
        return self

    @field_validator("safe_metadata")
    @classmethod
    def validate_safe_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_safe_metadata(v)


# ---------------------------------------------------------------------------
# ProviderExecutionRequest
# ---------------------------------------------------------------------------


class ProviderExecutionRequest(BaseModel):
    """Internal provider execution input.

    This schema may contain composed messages because it never leaves the
    execution boundary and must not be logged or copied into OrchestrationResult.
    """

    route_decision: RouteDecision
    model_resolution: ResolvedModelConfig
    messages: list[LlmMessage] = Field(..., min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0, le=8000)
    provider_options: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# ModelExecutionResult
# ---------------------------------------------------------------------------


class ModelExecutionResult(BaseModel):
    """Raw result returned by a ModelExecutor.execute() call.

    This schema normalises what any model backend (mock, real provider adapter)
    returns.  It intentionally omits prompt content, user query, classification
    data, and retrieved context.

    Fields:
        content:       The generated text from the model.
        model:         Model alias (not the actual provider model_id).
        provider:      Provider name for safe logging/audit (e.g. "mock",
                       "gemini").  None when unknown.
        finish_reason: Stop reason as returned by the provider ("stop",
                       "length", etc.).
        input_tokens:  Tokens consumed in the prompt, if available.
        output_tokens: Tokens generated, if available.
        latency_ms:    Wall-clock execution time in milliseconds, if measured.
        fallback_used: True when the result came from a fallback model.
        metadata:      Supplemental safe metadata.  Must not contain
                       prompt/query/context/credential keys.
    """

    content: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    provider: str | None = None
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    fallback_used: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

    @field_validator("input_tokens")
    @classmethod
    def validate_input_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("input_tokens must be >= 0")
        return v

    @field_validator("output_tokens")
    @classmethod
    def validate_output_tokens(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("output_tokens must be >= 0")
        return v

    @field_validator("latency_ms")
    @classmethod
    def validate_latency_ms(cls, v: int | None) -> int | None:
        if v is not None and v < 0:
            raise ValueError("latency_ms must be >= 0")
        return v

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_safe_metadata(v)


# ---------------------------------------------------------------------------
# OrchestrationResult
# ---------------------------------------------------------------------------


class OrchestrationResult(BaseModel):
    """Normalized result returned by LlmOrchestrator.generate().

    This is the public output of the orchestration layer.  It intentionally
    omits the composed messages list (system prompt + user message) to prevent
    leaking prompt content, user queries, classification data, or retrieved
    context to upstream callers.

    Callers that need to inspect composed messages for testing purposes should
    use MockModelExecutor.last_messages instead.

    Fields:
        content:        The generated text answer.
        route_decision: The RouteDecision used for this invocation (for audit
                        and logging).
        model:          Model alias used for generation.
        provider:       Provider name for logging.  None when unknown/mock.
        fallback_used:  True when a fallback model produced the result.
        finish_reason:  Stop reason from the executor ("stop", "length", etc.).
        input_tokens:   Tokens consumed in the prompt, if available.
        output_tokens:  Tokens generated, if available.
        latency_ms:     Wall-clock execution time in milliseconds, if measured.
        answer_source:  Provenance: "llm" = real provider, "mock" = mock
                        executor, "fallback" = fallback model invoked.
        metadata:       Supplemental safe metadata.  Must not contain
                        prompt/query/context/credential keys.
    """

    content: str = Field(..., min_length=1)
    route_decision: RouteDecision
    model: str = Field(..., min_length=1)
    provider: str | None = None
    fallback_used: bool = False
    finish_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    answer_source: Literal["llm", "mock", "fallback"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

    @model_validator(mode="after")
    def validate_answer_source_consistency(self) -> OrchestrationResult:
        """answer_source='fallback' must agree with fallback_used=True."""
        if self.answer_source == "fallback" and not self.fallback_used:
            raise ValueError(
                "answer_source='fallback' requires fallback_used=True"
            )
        return self

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_safe_metadata(v)
