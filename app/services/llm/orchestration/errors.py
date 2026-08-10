"""
app/services/llm_orchestration/errors.py
-----------------------------------------
Exception hierarchy for the LLM orchestration layer.

Callers should catch LlmOrchestrationError (base) or specific subclasses.

Rules:
- Config errors (load/validation) must fail loudly at startup / test time.
- Runtime route resolution failures degrade to the safe_mock path where possible.
- Do not expose internal stack traces or config structure to user-facing responses.
"""

from __future__ import annotations


class LlmOrchestrationError(Exception):
    """Base class for all LLM orchestration errors."""


class LlmConfigLoadError(LlmOrchestrationError):
    """Raised when the YAML config file cannot be found or read."""


class LlmConfigValidationError(LlmOrchestrationError):
    """Raised when the config fails Pydantic validation or cross-validation checks."""


class LlmRouteNotFoundError(LlmOrchestrationError):
    """Raised when no route can be resolved — including after all fallback attempts."""


class LlmRouteResolutionError(LlmOrchestrationError):
    """Raised for unexpected errors during route resolution (e.g. unsupported task role
    with no fallback route available)."""


# ---------------------------------------------------------------------------
# Prompt resolver errors
# ---------------------------------------------------------------------------


class PromptResolverError(LlmOrchestrationError):
    """Base class for all prompt resolver errors."""


class PromptPathError(PromptResolverError):
    """Raised when a prompt path is invalid or unsafe.

    Examples: absolute path, path containing '..', URL, non-.md extension,
    or a path that resolves outside the configured prompt root.
    """


class PromptNotFoundError(PromptResolverError):
    """Raised when the resolved prompt file does not exist on disk."""


class PromptValidationError(PromptResolverError):
    """Raised when a prompt file is empty, whitespace-only, or exceeds the
    maximum allowed file size (MAX_PROMPT_FILE_CHARS)."""


# ---------------------------------------------------------------------------
# Orchestrator errors (Part 3)
# ---------------------------------------------------------------------------


class LlmOrchestratorError(LlmOrchestrationError):
    """Base class for LlmOrchestrator coordination errors.

    Raised for invalid inputs (empty query, over-length query) and any
    orchestration-level failure that is not a route or prompt resolver error.
    Route resolver errors and prompt resolver errors bubble up unchanged.
    """


class LlmExecutionError(LlmOrchestratorError):
    """Raised when the injected ModelExecutor.execute() raises any exception.

    The original exception is available via ``__cause__``.
    The message is sanitised — it does not include prompt content, user
    query, classification data, or retrieved context.
    """


# ---------------------------------------------------------------------------
# Model execution boundary errors (Part 4)
# ---------------------------------------------------------------------------


class ModelConfigResolutionError(LlmOrchestrationError):
    """Raised when model/provider metadata cannot be resolved safely."""


class ModelExecutionConfigError(LlmOrchestrationError):
    """Raised when a route requests unsupported model execution options."""


class ProviderExecutionError(LlmOrchestrationError):
    """Raised when the injected ProviderExecutor fails."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str = "unknown_provider_error",
        attempted_aliases: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.attempted_aliases = attempted_aliases
