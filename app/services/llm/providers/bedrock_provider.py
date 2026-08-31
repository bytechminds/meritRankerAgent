"""Amazon Bedrock Converse adapter with native schema-constrained output.

Authentication is the runtime's existing AWS identity — this adapter holds no API
key and reads no environment variable. Only the AWS region travels through the
provider profile, and it is not a secret.

Structured output uses the Converse ``outputConfig.textFormat`` json_schema field,
so no prompt-level "return JSON" instruction, markdown fence stripping, or JSON
repair retry exists anywhere on this path. Schemas are static and versioned:
Bedrock compiles a grammar per schema and caches it, so a per-request schema would
pay compilation repeatedly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from schemas.practice_request_intelligence import (
    PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME,
    practice_request_intelligence_schema_json,
)
from services.llm.providers.errors import (
    LlmProviderConfigurationError,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.llm.providers.finish_reasons import normalize_completion_outcome
from services.llm.providers.usage import extract_bedrock_usage

if TYPE_CHECKING:
    from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
    from services.secrets.provider_credentials import ProviderCredentials

logger = logging.getLogger(__name__)


# Static structured-output grammars, selected by execution role. A role absent
# from this map is executed as an ordinary free-text Converse call.
#
# ``generator`` is deliberately absent. The canonical GenerationEnvelope types
# ``options`` as an array of strings, while the schema-v2 author contract requires
# an array of {option_id, value} objects that GeneratedQuestion normalises before
# validation. Constraining Bedrock to the exported schema therefore forbids the exact
# shape the prompt asks for: GLM 4.7 Flash and Qwen3 Next 80B both emitted well-formed
# questions with an empty options array, and every item failed contract validation.
# Authoring stays a free-text JSON call on this provider until the canonical schema
# can express the v2 option shape.
_STATIC_OUTPUT_SCHEMAS: dict[str, tuple[str, Callable[[], str]]] = {
    "request_intelligence": (
        PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME,
        practice_request_intelligence_schema_json,
    ),
}


def _classify_bedrock_error(exc: BaseException) -> str:
    """Map a botocore failure onto the shared provider failure vocabulary."""
    response = getattr(exc, "response", None)
    error = response.get("Error", {}) if isinstance(response, dict) else {}
    code = str(error.get("Code") or type(exc).__name__)
    status = 0
    if isinstance(response, dict):
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, dict):
            status = int(metadata.get("HTTPStatusCode") or 0)
    message = str(exc).lower()
    if code in {"AccessDeniedException", "UnrecognizedClientException"} or status == 403:
        return "authentication_failed"
    if code in {"ThrottlingException", "TooManyRequestsException"} or status == 429:
        return "rate_limited"
    if code == "ServiceQuotaExceededException":
        return "insufficient_quota"
    if code in {"ResourceNotFoundException", "ModelNotReadyException"} or status == 404:
        return "model_not_found"
    if code in {"ModelTimeoutException", "ReadTimeoutError", "ConnectTimeoutError"}:
        return "timeout"
    if code in {"ModelErrorException", "InternalServerException", "ServiceUnavailableException"}:
        return "provider_unavailable"
    if code == "NoCredentialsError" or "unable to locate credentials" in message:
        return "provider_not_configured"
    if code == "ValidationException" or status == 400:
        return "invalid_request"
    if status >= 500:
        return "provider_unavailable"
    if "timeout" in message:
        return "timeout"
    return "unknown_provider_error"


class BedrockProviderAdapter:
    """Execute a Bedrock model through the Converse API."""

    def __init__(
        self,
        *,
        client_factory: Callable[[ProviderCredentials, int], Any] | None = None,
    ) -> None:
        self._client_factory = client_factory

    def _build_client(
        self,
        credentials: ProviderCredentials,
        timeout_seconds: int,
    ) -> Any:
        if self._client_factory is not None:
            return self._client_factory(credentials, timeout_seconds)

        import boto3  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415

        return boto3.client(
            "bedrock-runtime",
            region_name=credentials.region or None,
            config=Config(
                read_timeout=timeout_seconds,
                connect_timeout=min(timeout_seconds, 10),
                # Infrastructure retry policy is owned by the orchestrator, not the SDK.
                retries={"max_attempts": 1, "mode": "standard"},
            ),
        )

    @staticmethod
    def _require_model_id(request: ProviderExecutionRequest) -> str:
        model_id = request.model_resolution.model_config.model_id
        if not model_id:
            raise LlmProviderConfigurationError(
                "model_id is required for the bedrock provider adapter "
                f"(model_alias={request.model_resolution.model_alias!r})."
            )
        return model_id

    @staticmethod
    def _request_parts(
        request: ProviderExecutionRequest,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system = [
            {"text": message.content}
            for message in request.messages
            if message.role == "system"
        ]
        messages = [
            {
                "role": "assistant" if message.role == "assistant" else "user",
                "content": [{"text": message.content}],
            }
            for message in request.messages
            if message.role != "system"
        ]
        if not messages:
            raise LlmProviderConfigurationError(
                "Bedrock request requires at least one non-system message."
            )
        return system, messages

    @staticmethod
    def _output_config(request: ProviderExecutionRequest) -> dict[str, Any] | None:
        entry = _STATIC_OUTPUT_SCHEMAS.get(request.route_decision.task_role)
        if entry is None:
            return None
        schema_name, schema_json = entry
        return {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {"name": schema_name, "schema": schema_json()}
                },
            }
        }

    @staticmethod
    def _content_text(response: dict[str, Any]) -> str:
        output = response.get("output") or {}
        message = output.get("message") or {}
        blocks = message.get("content") or []
        return "".join(
            str(block["text"]) for block in blocks if isinstance(block, dict) and "text" in block
        ).strip()

    def generate(
        self,
        *,
        request: ProviderExecutionRequest,
        credentials: ProviderCredentials,
    ) -> ModelExecutionResult:
        from schemas.llm_orchestration import ModelExecutionResult  # noqa: PLC0415

        model_id = self._require_model_id(request)
        client = self._build_client(
            credentials,
            request.model_resolution.timeout_seconds,
        )
        system, messages = self._request_parts(request)
        inference_config: dict[str, Any] = {"maxTokens": request.max_tokens}
        if request.model_resolution.model_config.supports_temperature:
            inference_config["temperature"] = request.temperature
        payload: dict[str, Any] = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if system:
            payload["system"] = system
        output_config = self._output_config(request)
        if output_config is not None:
            payload["outputConfig"] = output_config

        try:
            response = client.converse(**payload)
        except LlmProviderConfigurationError:
            raise
        except Exception as exc:
            raise LlmProviderExecutionError(
                "bedrock converse failed for "
                f"model_alias={request.model_resolution.model_alias!r}: "
                f"{type(exc).__name__}",
                failure_kind=_classify_bedrock_error(exc),
                provider="bedrock",
                model_alias=request.model_resolution.model_alias,
            ) from exc

        usage = extract_bedrock_usage(response)
        finish_reason = response.get("stopReason")
        content = self._content_text(response)
        if not content:
            raise LlmProviderResponseError(
                "Bedrock response is missing content "
                f"(model_alias={request.model_resolution.model_alias!r}).",
                finish_reason=finish_reason,
                provider_usage=usage,
            )
        return ModelExecutionResult(
            content=content,
            model=request.route_decision.model,
            provider="bedrock",
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
                "native_structured_output": output_config is not None,
            },
        )
