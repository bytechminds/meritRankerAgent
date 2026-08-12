"""
app/services/llm/providers/openai_compatible_adapter.py
-------------------------------------------------------
Shared OpenAI-SDK adapter for OpenAI-compatible provider endpoints.

Used by DeepSeek (api.deepseek.com) and Gemini (generativelanguage OpenAI compat).
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from services.llm.providers.errors import (
    LlmProviderConfigurationError,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.llm.providers.finish_reasons import normalize_completion_outcome
from services.llm.providers.openai_provider import _classify_openai_error
from services.llm.providers.usage import (
    clear_stream_usage,
    extract_openai_usage,
    set_stream_usage,
)

if TYPE_CHECKING:
    from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
    from services.secrets.provider_credentials import ProviderCredentials

logger = logging.getLogger(__name__)

_DEFAULT_GEMINI_OPENAI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
_DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _raise_not_configured(*, provider: str, model_alias: str) -> None:
    raise LlmProviderExecutionError(
        f"{provider} provider is not configured "
        f"(model_alias={model_alias!r}). Set the provider API key env var.",
        failure_kind="provider_not_configured",
        provider=provider,
        model_alias=model_alias,
    )


class OpenAICompatibleProviderAdapter:
    """Execute chat completions against an OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        *,
        provider: str,
        default_base_url: str | None = None,
        error_classifier: Callable[[BaseException], str] | None = None,
        client_factory: Callable[[ProviderCredentials, float | None], Any] | None = None,
    ) -> None:
        self._provider = provider
        self._default_base_url = default_base_url
        self._error_classifier = error_classifier or _classify_openai_error
        self._client_factory = client_factory
        self.last_stream_finish_reason: str | None = None

    def _build_client(self, credentials: ProviderCredentials, timeout_seconds: int) -> Any:
        if self._client_factory is not None:
            return self._client_factory(credentials, float(timeout_seconds))
        from openai import OpenAI as _OpenAI  # noqa: PLC0415

        base_url = credentials.base_url or self._default_base_url
        return _OpenAI(
            api_key=credentials.api_key,
            base_url=base_url,
            max_retries=0,
            timeout=float(timeout_seconds),
        )

    def _require_api_key(
        self,
        credentials: ProviderCredentials,
        request: ProviderExecutionRequest,
    ) -> None:
        if not credentials.api_key:
            _raise_not_configured(
                provider=self._provider,
                model_alias=request.model_resolution.model_alias,
            )

    def generate(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
    ) -> ModelExecutionResult:
        from schemas.llm_orchestration import ModelExecutionResult  # noqa: PLC0415

        self._require_api_key(credentials, request)

        model_id = request.model_resolution.model_config.model_id
        if not model_id:
            raise LlmProviderConfigurationError(
                f"model_id is required for the {self._provider} provider adapter "
                f"(model_alias={request.model_resolution.model_alias!r})."
            )

        timeout_seconds = request.model_resolution.timeout_seconds
        client = self._build_client(credentials, timeout_seconds)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        logger.info(
            "%s_provider_adapter.generate  model_alias=%s  model_label=%s — start",
            self._provider,
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
            failure_kind = self._error_classifier(exc)
            raise LlmProviderExecutionError(
                f"{self._provider} call failed for "
                f"model_alias={request.model_resolution.model_alias!r}: "
                f"{type(exc).__name__}",
                failure_kind=failure_kind,
                provider=self._provider,
                model_alias=request.model_resolution.model_alias,
            ) from exc

        finish_reason = _extract_finish_reason(completion)
        usage = extract_openai_usage(completion)
        content = _extract_content(
            completion,
            request,
            finish_reason=finish_reason,
            provider_usage=usage,
        )

        logger.info(
            "%s_provider_adapter.generate  model_alias=%s — done  "
            "finish_reason=%s  input_tokens=%s  output_tokens=%s",
            self._provider,
            request.model_resolution.model_alias,
            finish_reason,
            usage.input_tokens,
            usage.output_tokens,
        )

        return ModelExecutionResult(
            content=content,
            model=request.route_decision.model,
            provider=self._provider,
            finish_reason=finish_reason,
            normalized_finish_reason=normalize_completion_outcome(finish_reason),
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
        self._require_api_key(credentials, request)

        model_id = request.model_resolution.model_config.model_id
        if not model_id:
            raise LlmProviderConfigurationError(
                f"model_id is required for the {self._provider} provider adapter "
                f"(model_alias={request.model_resolution.model_alias!r})."
            )

        timeout_seconds = request.model_resolution.timeout_seconds
        client = self._build_client(credentials, timeout_seconds)
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        logger.info(
            "%s_provider_adapter.generate_stream  model_alias=%s  model_label=%s — start",
            self._provider,
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
            failure_kind = self._error_classifier(exc)
            raise LlmProviderExecutionError(
                f"{self._provider} stream failed for "
                f"model_alias={request.model_resolution.model_alias!r}: "
                f"{type(exc).__name__}",
                failure_kind=failure_kind,
                provider=self._provider,
                model_alias=request.model_resolution.model_alias,
            ) from exc
        finally:
            set_stream_usage(usage)

    def generate_with_image(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
        image_bytes: bytes,
        mime_type: str = "image/jpeg",
    ) -> ModelExecutionResult:
        """Multimodal generation for future image question extraction (adapter-level only)."""
        from schemas.llm_orchestration import ModelExecutionResult  # noqa: PLC0415

        self._require_api_key(credentials, request)

        model_id = request.model_resolution.model_config.model_id
        if not model_id:
            raise LlmProviderConfigurationError(
                f"model_id is required for the {self._provider} provider adapter "
                f"(model_alias={request.model_resolution.model_alias!r})."
            )

        timeout_seconds = request.model_resolution.timeout_seconds
        client = self._build_client(credentials, timeout_seconds)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"

        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "user":
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg.content},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                )
            else:
                messages.append({"role": msg.role, "content": msg.content})

        logger.info(
            "%s_provider_adapter.generate_with_image  model_alias=%s — start",
            self._provider,
            request.model_resolution.model_alias,
        )

        try:
            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        except Exception as exc:
            failure_kind = self._error_classifier(exc)
            raise LlmProviderExecutionError(
                f"{self._provider} image call failed for "
                f"model_alias={request.model_resolution.model_alias!r}: "
                f"{type(exc).__name__}",
                failure_kind=failure_kind,
                provider=self._provider,
                model_alias=request.model_resolution.model_alias,
            ) from exc

        content = _extract_content(completion, request)
        finish_reason = _extract_finish_reason(completion)
        usage = extract_openai_usage(completion)

        return ModelExecutionResult(
            content=content,
            model=request.route_decision.model,
            provider=self._provider,
            finish_reason=finish_reason,
            normalized_finish_reason=normalize_completion_outcome(finish_reason),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            usage_source=("provider_reported" if usage.available else "unavailable"),
            metadata={
                "model_label": request.model_resolution.model_config.model_label,
                "multimodal": True,
            },
        )


def _extract_content(
    completion: Any,
    request: ProviderExecutionRequest,
    *,
    finish_reason: str | None = None,
    provider_usage: Any = None,
) -> str:
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        content = None
    if not content:
        raise LlmProviderResponseError(
            f"Response is missing content "
            f"(model_alias={request.model_resolution.model_alias!r}).",
            finish_reason=finish_reason,
            provider_usage=provider_usage,
        )
    return content


def _extract_finish_reason(completion: Any) -> str | None:
    try:
        return completion.choices[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return None


def _extract_usage(completion: Any) -> tuple[int | None, int | None]:
    try:
        return completion.usage.prompt_tokens, completion.usage.completion_tokens
    except AttributeError:
        return None, None


def classify_gemini_error(exc: BaseException) -> str:
    """Map Gemini OpenAI-compatible errors, including safety blocks."""
    message = str(exc).lower()
    safety_markers = (
        "safety",
        "blocked",
        "block_reason",
        "harm",
        "recitation",
        "content_filter",
    )
    if any(marker in message for marker in safety_markers):
        return "safety_blocked"
    return _classify_openai_error(exc)


def classify_deepseek_error(exc: BaseException) -> str:
    """Map DeepSeek OpenAI-compatible errors."""
    return _classify_openai_error(exc)


class DeepSeekProviderAdapter(OpenAICompatibleProviderAdapter):
    """DeepSeek provider via OpenAI-compatible chat completions API."""

    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentials, float | None], Any] | None = None,
    ) -> None:
        super().__init__(
            provider="deepseek",
            default_base_url=_DEFAULT_DEEPSEEK_BASE_URL,
            error_classifier=classify_deepseek_error,
            client_factory=client_factory,
        )


class GeminiProviderAdapter(OpenAICompatibleProviderAdapter):
    """Gemini provider via OpenAI-compatible endpoint (text + optional image)."""

    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentials, float | None], Any] | None = None,
    ) -> None:
        super().__init__(
            provider="gemini",
            default_base_url=_DEFAULT_GEMINI_OPENAI_BASE_URL,
            error_classifier=classify_gemini_error,
            client_factory=client_factory,
        )
