"""
app/services/llm_providers/errors.py
-------------------------------------
Exception hierarchy for the LLM provider layer.

Legacy error classes (model_router / BaseLlmProvider path):
    LlmProviderError          — base class
    LlmConfigurationError     — config/env missing
    LlmGenerationError        — provider call failed

Part 6 adapter error classes (ProviderAdapter path):
    LlmProviderAdapterError             — base class
    LlmProviderConfigurationError       — config/credentials missing
    LlmProviderExecutionError           — provider SDK/network call failed (has failure_kind)
    LlmProviderResponseError            — response malformed or empty
    LlmProviderUnsupportedFeatureError  — feature not supported by adapter

ProviderFailureKind — allowed failure_kind values for LlmProviderExecutionError.

Security rules:
- Error messages must not include API keys, tokens, or credential values.
- Error messages may include safe identifiers: provider name, model alias,
  deployment name, model_label.
- Error messages must not include prompt content, messages, query, or context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from services.llm.providers.finish_reasons import normalize_completion_outcome

if TYPE_CHECKING:
    from schemas.llm_usage import ProviderTokenUsage

# ---------------------------------------------------------------------------
# Provider failure kinds
# ---------------------------------------------------------------------------

ProviderFailureKind = Literal[
    "insufficient_quota",
    "rate_limited",
    "authentication_failed",
    "provider_not_configured",
    "model_not_configured",
    "model_not_found",
    "timeout",
    "provider_unavailable",
    "invalid_request",
    "unsupported_parameter",
    "empty_stream",
    "empty_answer",
    "output_token_exhausted",
    "safety_blocked",
    "unknown_provider_error",
]

# Failure kinds that are eligible for model-level fallback in the executor.
# Config/programming errors (invalid_request) and safety blocks are NOT eligible.
FALLBACK_ELIGIBLE_FAILURE_KINDS: frozenset[str] = frozenset({
    "insufficient_quota",
    "rate_limited",
    "authentication_failed",
    "provider_not_configured",
    "model_not_configured",
    "model_not_found",
    "timeout",
    "provider_unavailable",
    "unsupported_parameter",
    "empty_stream",
    "empty_answer",
    "output_token_exhausted",
    "unknown_provider_error",
})

# ---------------------------------------------------------------------------
# Legacy error classes — used by model_router / BaseLlmProvider path
# ---------------------------------------------------------------------------


class LlmProviderError(Exception):
    """Base class for all LLM provider errors."""


class LlmConfigurationError(LlmProviderError):
    """Raised when required configuration (env vars, role config) is missing or invalid."""


class LlmGenerationError(LlmProviderError):
    """Raised when a provider call succeeds at the network level but generation fails."""


# ---------------------------------------------------------------------------
# Part 6 adapter error classes — used by ProviderAdapter path
# ---------------------------------------------------------------------------


class LlmProviderAdapterError(Exception):
    """Base class for all Part 6 provider adapter errors.

    Security: messages must not include credential values, prompt content,
    messages, query, or context.  Safe identifiers (provider name, model alias,
    deployment name) are allowed.
    """


class LlmProviderConfigurationError(LlmProviderAdapterError):
    """Raised when required configuration or credentials are missing or invalid.

    Examples:
    - api_key is None/blank
    - model_id is missing for OpenAI provider
    - deployment is missing for Azure OpenAI provider
    - unsupported provider name passed to factory
    """


class LlmProviderExecutionError(LlmProviderAdapterError):
    """Raised when the provider SDK call fails at the network or API level.

    Includes a ``failure_kind`` that categorises the failure so the model
    execution layer can decide whether to attempt a fallback alias.

    Attributes:
        failure_kind: One of the ProviderFailureKind literals.
        provider:     Safe provider identifier (e.g. "openai", "azure_openai").
        model_alias:  Internal model alias (not the provider model_id).

    Examples:
    - openai.APIConnectionError
    - authentication error (message must not include the key)
    - rate limit / quota error → failure_kind="insufficient_quota" or "rate_limited"
    - timeout → failure_kind="timeout"
    """

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str = "unknown_provider_error",
        provider: str | None = None,
        model_alias: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_kind: str = failure_kind
        self.provider: str | None = provider
        self.model_alias: str | None = model_alias


class LlmProviderResponseError(LlmProviderAdapterError):
    """Raised when the provider response is malformed or missing expected content.

    Examples:
    - completion.choices is empty
    - message.content is None or blank
    """

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str = "empty_answer",
        finish_reason: str | None = None,
        provider_usage: ProviderTokenUsage | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_usage = provider_usage
        self.failure_kind: str = failure_kind
        self.finish_reason = finish_reason
        self.normalized_finish_reason = normalize_completion_outcome(finish_reason)
        self.input_tokens = (
            provider_usage.input_tokens if provider_usage is not None else None
        )
        self.output_tokens = (
            provider_usage.output_tokens if provider_usage is not None else None
        )
        self.total_tokens = (
            provider_usage.total_tokens if provider_usage is not None else None
        )
        self.cached_input_tokens = (
            provider_usage.cached_input_tokens if provider_usage is not None else None
        )
        self.reasoning_tokens = (
            provider_usage.reasoning_tokens if provider_usage is not None else None
        )


class LlmProviderUnsupportedFeatureError(LlmProviderAdapterError):
    """Raised when a requested feature is not supported by the adapter.

    Examples:
    - streaming requested but not implemented by the adapter
    - provider_option key not in adapter's supported allowlist
    """
