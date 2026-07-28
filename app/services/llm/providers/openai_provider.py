"""
app/services/llm_providers/openai_provider.py
-----------------------------------------------
OpenAI provider implementations.

Legacy class (model_router / BaseLlmProvider path):
    OpenAIProvider — reads credentials from os.environ at call time.

Part 6 adapter class (ProviderAdapterExecutor path):
    OpenAIProviderAdapter — receives ProviderCredentials explicitly; supports
                            injected fake client for unit testing.

Security rules:
- No API key may appear in logs, errors, or metadata.
- Error messages use only safe identifiers: model_alias, model_label.
- No env reads inside OpenAIProviderAdapter.
- No SDK client created at import or construction time.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from schemas.llm import LlmRequest, LlmResponse, LlmRoleConfig, LlmStreamChunk
from schemas.llm_usage import ProviderTokenUsage
from services.llm.providers.base import BaseLlmProvider
from services.llm.providers.errors import (
    LlmConfigurationError,
    LlmGenerationError,
    LlmProviderConfigurationError,
    LlmProviderError,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.llm.providers.usage import (
    clear_stream_usage,
    extract_openai_usage,
    set_stream_usage,
)

if TYPE_CHECKING:
    from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
    from services.secrets.provider_credentials import ProviderCredentials

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLlmProvider):
    """OpenAI native provider — reads credentials from environment at call time."""

    def _build_client(self, config: LlmRoleConfig) -> OpenAI:
        """Construct and return an OpenAI client.

        Credentials are read fresh from environment on every call so that
        runtime credential rotation is supported without restarting the process.

        Raises:
            LlmConfigurationError: If any required env var or config field is missing.
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "") or None

        if not api_key:
            raise LlmConfigurationError("OPENAI_API_KEY is not set")
        if not config.model:
            raise LlmConfigurationError(
                f"LlmRoleConfig.model is required for openai provider "
                f"(model_label={config.model_label!r})"
            )

        # api_key is intentionally not logged
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(self, request: LlmRequest, config: LlmRoleConfig) -> LlmResponse:
        client = self._build_client(config)
        temperature = request.temperature if request.temperature is not None else config.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else config.max_tokens
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        logger.info(
            "openai_provider.generate  role=%s  model_label=%s — start",
            request.role,
            config.model_label,
        )
        try:
            completion = client.chat.completions.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise LlmGenerationError(f"OpenAI generation failed: {exc}") from exc

        content = completion.choices[0].message.content or ""
        finish_reason = completion.choices[0].finish_reason
        usage = extract_openai_usage(completion)
        logger.info(
            "openai_provider.generate  role=%s  model_label=%s — done",
            request.role,
            config.model_label,
        )
        return LlmResponse(
            role=request.role,
            provider="openai",
            model_label=config.model_label,
            content=content,
            finish_reason=finish_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )

    def stream(self, request: LlmRequest, config: LlmRoleConfig) -> Iterator[LlmStreamChunk]:
        if not config.supports_streaming:
            raise LlmProviderError(
                f"Streaming is not enabled for model_label={config.model_label!r}. "
                "Set supports_streaming=true in the role config to enable it."
            )

        client = self._build_client(config)
        temperature = request.temperature if request.temperature is not None else config.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else config.max_tokens
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        logger.info(
            "openai_provider.stream  role=%s  model_label=%s — start",
            request.role,
            config.model_label,
        )
        usage = ProviderTokenUsage()
        try:
            stream = client.chat.completions.create(
                model=config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                chunk_usage = extract_openai_usage(chunk)
                if chunk_usage.available:
                    usage = chunk_usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                finish_reason = chunk.choices[0].finish_reason
                is_final = finish_reason is not None
                yield LlmStreamChunk(
                    role=request.role,
                    provider="openai",
                    model_label=config.model_label,
                    content_delta=delta,
                    is_final=is_final,
                )
        except LlmProviderError:
            raise
        except Exception as exc:
            raise LlmGenerationError(f"OpenAI streaming failed: {exc}") from exc
        finally:
            set_stream_usage(usage)


# ---------------------------------------------------------------------------
# Part 6 adapter — ProviderAdapterExecutor path
# ---------------------------------------------------------------------------


class OpenAIProviderAdapter:
    """Part 6 OpenAI provider adapter conforming to the ProviderAdapter protocol.

    Receives ProviderCredentials explicitly — does not read os.environ.
    Supports an injected ``client_factory`` for unit testing with fake clients.

    No SDK client is created at import or construction time.

    Credential rules:
    - ``credentials.api_key`` is required.
    - ``credentials.base_url`` is optional (used as custom base URL override).
    - Credential values must never appear in logs, errors, or metadata.

    Model selection:
    - Uses ``request.model_resolution.model_config.model_id`` as the provider
      model ID.  This is not the route alias.
    """

    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentials], Any] | None = None,
    ) -> None:
        """
        Args:
            client_factory: Optional callable(credentials) -> client.
                Used to inject a fake client in unit tests.
                When None, a real openai.OpenAI client is created at call time.
        """
        self._client_factory = client_factory
        self.last_stream_finish_reason: str | None = None

    def _build_client(self, credentials: ProviderCredentials) -> Any:
        """Create or return the injected OpenAI client."""
        if self._client_factory is not None:
            return self._client_factory(credentials)
        # Deferred real import — no client at import/construction time
        from openai import OpenAI as _OpenAI  # noqa: PLC0415

        return _OpenAI(
            api_key=credentials.api_key,
            base_url=credentials.base_url,
            max_retries=0,  # Disable SDK-level retries; fallback happens at model-execution level.
        )

    def generate(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
    ) -> ModelExecutionResult:
        """Execute the request against OpenAI and return a normalized result.

        Args:
            request:     The resolved provider execution request.
            credentials: Resolved OpenAI credentials.  Must have api_key set.

        Returns:
            ModelExecutionResult with provider="openai".

        Raises:
            LlmProviderConfigurationError: api_key missing or model_id missing.
            LlmProviderExecutionError:     SDK call failed.
            LlmProviderResponseError:      Response content is empty/missing.
        """
        from schemas.llm_orchestration import ModelExecutionResult  # noqa: PLC0415

        if not credentials.api_key:
            raise LlmProviderConfigurationError(
                "OpenAIProviderAdapter requires credentials.api_key."
            )

        model_id = request.model_resolution.model_config.model_id
        if not model_id:
            raise LlmProviderConfigurationError(
                f"model_id is required for the OpenAI provider adapter "
                f"(model_alias={request.model_resolution.model_alias!r})."
            )

        client = self._build_client(credentials)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        logger.info(
            "openai_provider_adapter.generate  model_alias=%s  model_label=%s — start",
            request.model_resolution.model_alias,
            request.model_resolution.model_config.model_label,
        )

        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as exc:
            failure_kind = _classify_openai_error(exc)
            raise LlmProviderExecutionError(
                f"OpenAI call failed for model_alias={request.model_resolution.model_alias!r}: "
                f"{type(exc).__name__}",
                failure_kind=failure_kind,
                provider="openai",
                model_alias=request.model_resolution.model_alias,
            ) from exc

        content = self._extract_content(completion, request)
        finish_reason = self._extract_finish_reason(completion)
        usage = extract_openai_usage(completion)

        logger.info(
            "openai_provider_adapter.generate  model_alias=%s — done  "
            "finish_reason=%s  input_tokens=%s  output_tokens=%s",
            request.model_resolution.model_alias,
            finish_reason,
            usage.input_tokens,
            usage.output_tokens,
        )

        return ModelExecutionResult(
            content=content,
            model=request.route_decision.model,
            provider="openai",
            finish_reason=finish_reason,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            usage_source=("provider_reported" if usage.available else "unavailable"),
            metadata={
                "model_label": request.model_resolution.model_config.model_label,
            },
        )

    def generate_stream(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
    ) -> Iterator[str]:
        """Stream text deltas from OpenAI. Yields answer text chunks only."""
        if not credentials.api_key:
            raise LlmProviderConfigurationError(
                "OpenAIProviderAdapter requires credentials.api_key."
            )

        model_id = request.model_resolution.model_config.model_id
        if not model_id:
            raise LlmProviderConfigurationError(
                f"model_id is required for the OpenAI provider adapter "
                f"(model_alias={request.model_resolution.model_alias!r})."
            )

        client = self._build_client(credentials)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        logger.info(
            "openai_provider_adapter.generate_stream  model_alias=%s  model_label=%s — start",
            request.model_resolution.model_alias,
            request.model_resolution.model_config.model_label,
        )

        clear_stream_usage()
        usage = extract_openai_usage(object())
        try:
            stream = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )
            finish_reason: str | None = None
            for chunk in stream:
                chunk_usage = extract_openai_usage(chunk)
                if chunk_usage.available:
                    usage = chunk_usage
                if not chunk.choices:
                    continue
                fr = getattr(chunk.choices[0], "finish_reason", None)
                if fr is not None:
                    finish_reason = fr
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
            self.last_stream_finish_reason = finish_reason or "stop"
        except Exception as exc:
            failure_kind = _classify_openai_error(exc)
            raise LlmProviderExecutionError(
                f"OpenAI stream failed for model_alias={request.model_resolution.model_alias!r}: "
                f"{type(exc).__name__}",
                failure_kind=failure_kind,
                provider="openai",
                model_alias=request.model_resolution.model_alias,
            ) from exc
        finally:
            set_stream_usage(usage)

    @staticmethod
    def _extract_content(completion: Any, request: ProviderExecutionRequest) -> str:
        try:
            content = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            content = None
        if not content:
            raise LlmProviderResponseError(
                f"OpenAI response is missing content "
                f"(model_alias={request.model_resolution.model_alias!r})."
            )
        return content

    @staticmethod
    def _extract_finish_reason(completion: Any) -> str | None:
        try:
            return completion.choices[0].finish_reason
        except (AttributeError, IndexError, TypeError):
            return None

    @staticmethod
    def _extract_usage(completion: Any) -> tuple[int | None, int | None]:
        try:
            return completion.usage.prompt_tokens, completion.usage.completion_tokens
        except AttributeError:
            return None, None


# ---------------------------------------------------------------------------
# OpenAI error classifier (Part 9.1 — provider failure kind mapping)
# ---------------------------------------------------------------------------


def _classify_openai_error(exc: BaseException) -> str:
    """Map an openai SDK exception to a safe ProviderFailureKind string.

    Security: this function must NOT include API keys, endpoint URLs, prompt
    content, messages, or raw response bodies in the returned string.

    Returns one of the ProviderFailureKind literals.
    """
    # Lazy import — avoids binding to the openai module at module load time
    # when the legacy OpenAIProvider path is used.
    try:
        import openai as _openai  # noqa: PLC0415
    except ImportError:
        return "unknown_provider_error"

    if isinstance(exc, _openai.RateLimitError):
        # 429 — check the error body for the insufficient_quota indicator.
        error_code = getattr(exc, "code", None) or ""
        error_body = str(getattr(exc, "body", "") or "")
        if error_code == "insufficient_quota" or "insufficient_quota" in error_body:
            return "insufficient_quota"
        return "rate_limited"

    if isinstance(exc, (_openai.AuthenticationError, _openai.PermissionDeniedError)):
        return "authentication_failed"

    if isinstance(exc, _openai.NotFoundError):
        return "model_not_found"

    if isinstance(exc, _openai.APITimeoutError):
        return "timeout"

    if isinstance(exc, _openai.APIConnectionError):
        return "provider_unavailable"

    if isinstance(exc, _openai.BadRequestError):
        return "invalid_request"

    # openai.APIStatusError is the base for all HTTP status errors
    if isinstance(exc, _openai.APIStatusError):
        status = getattr(exc, "status_code", 0) or 0
        if status in (401, 403):
            return "authentication_failed"
        if status == 429:
            return "rate_limited"
        if status == 404:
            return "model_not_found"
        if status >= 500:
            return "provider_unavailable"

    return "unknown_provider_error"
