"""Native Google Gen AI adapter for Gemini text generation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from schemas.doubt_solver import QueryClassification
from services.llm.providers.errors import (
    LlmProviderConfigurationError,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.llm.providers.finish_reasons import normalize_completion_outcome
from services.llm.providers.usage import (
    clear_stream_usage,
    extract_gemini_usage,
    set_stream_usage,
)

if TYPE_CHECKING:
    from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
    from services.secrets.provider_credentials import ProviderCredentials

logger = logging.getLogger(__name__)


def _classify_gemini_error(exc: BaseException) -> str:
    message = str(exc).lower()
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if any(
        marker in message
        for marker in ("safety", "blocked", "block_reason", "harm", "recitation")
    ):
        return "safety_blocked"
    if status_code in {401, 403}:
        return "authentication_failed"
    if status_code == 404:
        return "model_not_found"
    if status_code == 429:
        return "rate_limited"
    if isinstance(status_code, int) and status_code >= 500:
        return "provider_unavailable"
    if "timeout" in message or "deadline" in message:
        return "timeout"
    if "quota" in message:
        return "insufficient_quota"
    if "invalid" in message or status_code == 400:
        return "invalid_request"
    return "unknown_provider_error"


def _classifier_response_schema() -> dict[str, Any]:
    """Return the public classifier wire schema without internal provenance fields."""
    schema = QueryClassification.model_json_schema()
    properties = dict(schema.get("properties", {}))
    properties.pop("classification_source", None)
    properties.pop("requires_recent_conversation", None)
    schema["properties"] = properties
    schema["required"] = list(properties)
    return schema


def _verifier_response_schema() -> dict[str, Any]:
    """Return the schema-v2 Answer Authority wire shape.

    Written out rather than derived from ``VerificationResult`` because that model
    still carries the legacy v1 fields, and their optionality compiles to ``anyOf``
    with ``$ref``/``$defs`` — constructs this API rejects. The fields below are exactly
    the ones the v2 verifier prompt asks for, and the canonical model remains the
    authority: it re-validates every response after parsing.
    """
    option_id = {"type": "string", "enum": ["0", "1", "2", "3"]}
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["2"]},
            "generation_item_id": {"type": "string"},
            "slot_id": {"type": "string"},
            "decision": {
                "type": "string",
                "enum": ["ACCEPT", "REPAIRABLE", "REGENERATE", "TERMINAL_REJECTION"],
            },
            "valid_option_ids": {"type": "array", "items": option_id, "maxItems": 4},
            "reason_codes": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        },
        "required": [
            "schema_version",
            "generation_item_id",
            "slot_id",
            "decision",
            "valid_option_ids",
            "reason_codes",
        ],
    }


# Native response schemas keyed by execution role. A role absent from this map is
# executed as an ordinary free-text call. Keyed by role, never by model, so no
# model-specific branch exists.
_NATIVE_RESPONSE_SCHEMAS: dict[str, Callable[[], dict[str, Any]]] = {
    "classifier": _classifier_response_schema,
    "verifier": _verifier_response_schema,
}


class GeminiProviderAdapter:
    """Execute Gemini through the native ``google-genai`` SDK."""

    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentials, int], Any] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self.last_stream_finish_reason: str | None = None

    def _build_client(
        self,
        credentials: ProviderCredentials,
        timeout_seconds: int,
    ) -> Any:
        if self._client_factory is not None:
            return self._client_factory(credentials, timeout_seconds)

        from google import genai  # noqa: PLC0415
        from google.genai import types  # noqa: PLC0415

        http_options: dict[str, Any] = {
            "timeout": timeout_seconds * 1000,
            "retry_options": types.HttpRetryOptions(attempts=1),
        }
        if credentials.base_url:
            if "/openai" in credentials.base_url.casefold():
                raise LlmProviderConfigurationError(
                    "GEMINI_BASE_URL must be unset or point to a native Gemini API endpoint."
                )
            http_options["base_url"] = credentials.base_url
        return genai.Client(
            api_key=credentials.api_key,
            http_options=types.HttpOptions(**http_options),
        )

    @staticmethod
    def _require_config(
        credentials: ProviderCredentials,
        request: ProviderExecutionRequest,
    ) -> str:
        if not credentials.api_key:
            raise LlmProviderExecutionError(
                "gemini provider is not configured "
                f"(model_alias={request.model_resolution.model_alias!r}). "
                "Set GEMINI_API_KEY.",
                failure_kind="provider_not_configured",
                provider="gemini",
                model_alias=request.model_resolution.model_alias,
            )
        model_id = request.model_resolution.model_config.model_id
        if not model_id:
            raise LlmProviderConfigurationError(
                "model_id is required for the gemini provider adapter "
                f"(model_alias={request.model_resolution.model_alias!r})."
            )
        return model_id

    @staticmethod
    def _request_parts(
        request: ProviderExecutionRequest,
    ) -> tuple[str | None, list[Any]]:
        from google.genai import types  # noqa: PLC0415

        system_instruction = "\n\n".join(
            message.content for message in request.messages if message.role == "system"
        )
        contents = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in request.messages
            if message.role != "system"
        ]
        if not contents:
            raise LlmProviderConfigurationError(
                "Gemini request requires at least one non-system message."
            )
        return system_instruction or None, contents

    @staticmethod
    def _generation_config(
        request: ProviderExecutionRequest,
        *,
        system_instruction: str | None,
    ) -> Any:
        from google.genai import types  # noqa: PLC0415

        values: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        schema_builder = _NATIVE_RESPONSE_SCHEMAS.get(
            request.route_decision.task_role
        )
        if schema_builder is not None:
            values.update(
                {
                    "response_mime_type": "application/json",
                    "response_json_schema": schema_builder(),
                }
            )
        return types.GenerateContentConfig(**values)

    def generate(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
    ) -> ModelExecutionResult:
        from schemas.llm_orchestration import ModelExecutionResult  # noqa: PLC0415

        model_id = self._require_config(credentials, request)
        client = self._build_client(
            credentials,
            request.model_resolution.timeout_seconds,
        )
        system_instruction, contents = self._request_parts(request)
        config = self._generation_config(
            request,
            system_instruction=system_instruction,
        )
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config,
            )
        except LlmProviderConfigurationError:
            raise
        except Exception as exc:
            raise LlmProviderExecutionError(
                "gemini call failed for "
                f"model_alias={request.model_resolution.model_alias!r}: "
                f"{type(exc).__name__}",
                failure_kind=_classify_gemini_error(exc),
                provider="gemini",
                model_alias=request.model_resolution.model_alias,
            ) from exc

        usage = extract_gemini_usage(response)
        finish_reason = _finish_reason(response)
        content = getattr(response, "text", None)
        if not content:
            parsed = getattr(response, "parsed", None)
            if hasattr(parsed, "model_dump_json"):
                content = parsed.model_dump_json()
            elif isinstance(parsed, (dict, list)):
                content = json.dumps(parsed, separators=(",", ":"))
        if not content:
            raise LlmProviderResponseError(
                "Gemini response is missing content "
                f"(model_alias={request.model_resolution.model_alias!r}).",
                finish_reason=finish_reason,
                provider_usage=usage,
            )
        return ModelExecutionResult(
            content=content,
            model=request.route_decision.model,
            provider="gemini",
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
                "native_structured_output": (
                    request.route_decision.task_role in _NATIVE_RESPONSE_SCHEMAS
                ),
            },
        )

    def generate_stream(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
    ) -> Iterator[str]:
        model_id = self._require_config(credentials, request)
        client = self._build_client(
            credentials,
            request.model_resolution.timeout_seconds,
        )
        system_instruction, contents = self._request_parts(request)
        config = self._generation_config(
            request,
            system_instruction=system_instruction,
        )
        clear_stream_usage()
        usage = extract_gemini_usage(object())
        try:
            stream = client.models.generate_content_stream(
                model=model_id,
                contents=contents,
                config=config,
            )
            finish_reason: str | None = None
            for chunk in stream:
                chunk_usage = extract_gemini_usage(chunk)
                if chunk_usage.available:
                    usage = chunk_usage
                finish_reason = _finish_reason(chunk) or finish_reason
                text = getattr(chunk, "text", None)
                if text:
                    yield text
            self.last_stream_finish_reason = finish_reason or "stop"
        except Exception as exc:
            raise LlmProviderExecutionError(
                "gemini stream failed for "
                f"model_alias={request.model_resolution.model_alias!r}: "
                f"{type(exc).__name__}",
                failure_kind=_classify_gemini_error(exc),
                provider="gemini",
                model_alias=request.model_resolution.model_alias,
            ) from exc
        finally:
            set_stream_usage(usage)


def _finish_reason(response: object) -> str | None:
    try:
        reason = response.candidates[0].finish_reason  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        return None
    value = getattr(reason, "value", reason)
    return str(value).lower() if value is not None else None
