"""Bedrock Converse adapter: request shape, deserialization, and failure mapping.

No AWS call is made. A fake boto3 client captures the exact Converse payload so the
native structured-output contract is asserted on the wire, not inferred.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from schemas.llm import LlmMessage
from schemas.llm_orchestration import ProviderExecutionRequest
from schemas.llm_routing import RouteDecision
from schemas.practice_request_intelligence import (
    PRACTICE_REQUEST_INTELLIGENCE_SCHEMA,
    PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME,
)
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.providers.bedrock_provider import BedrockProviderAdapter
from services.llm.providers.errors import (
    FALLBACK_ELIGIBLE_FAILURE_KINDS,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.llm.providers.provider_factory import ProviderAdapterFactory
from services.secrets.provider_credentials import ProviderCredentials

_BEDROCK_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        request_intelligence:
          default:
            model: practice_request_intelligence
            prompt: practice_generation/request_intelligence.md
            temperature: 0.0
            max_tokens: 700
        generator:
          default:
            model: practice_request_intelligence
            prompt: subjects/general_generator.md
            temperature: 0.0
            max_tokens: 700
    models:
      practice_request_intelligence:
        provider: bedrock
        provider_profile: bedrock_apsouth1
        model_id: zai.glm-4.7-flash
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 20
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
    provider_profiles:
      local_mock:
        provider: mock
      bedrock_apsouth1:
        provider: bedrock
        region_env: BEDROCK_LLM_REGION
        optional_region: true
""")

_VALID_RESPONSE = {
    "interpretationStatus": "RESOLVED",
    "requestedCount": 20,
    "topics": [{"sourceText": "percentge", "normalizedName": "Percentage"}],
    "difficulty": {"mode": "UNSPECIFIED"},
}


class _FakeBedrockClient:
    """Capture the Converse payload and return a canned response."""

    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None):
        self.captured: dict[str, Any] | None = None
        self._response = response
        self._error = error

    def converse(self, **payload: Any) -> dict[str, Any]:
        self.captured = payload
        if self._error is not None:
            raise self._error
        return self._response or {}


def _converse_response(text: str, *, stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 120, "outputTokens": 45, "totalTokens": 165},
        "metrics": {"latencyMs": 310},
    }


def _route_decision(*, task_role: str) -> RouteDecision:
    return RouteDecision(
        route_id=f"general.{task_role}.default",
        subject="general",
        task_role=task_role,
        difficulty="default",
        model="practice_request_intelligence",
        prompt="practice_generation/request_intelligence.md",
        temperature=0.0,
        max_tokens=700,
        provider_options={},
        fallback=[],
        fallback_attempts=[],
        route_source="exact",
    )


def _execution_request(tmp_path: Path, *, task_role: str) -> ProviderExecutionRequest:
    yaml_path = tmp_path / "llm.yaml"
    yaml_path.write_text(_BEDROCK_YAML, encoding="utf-8")
    route = _route_decision(task_role=task_role)
    resolution = ModelConfigResolver(registry=LlmConfigRegistry(yaml_path=yaml_path)).resolve(route)
    return ProviderExecutionRequest(
        route_decision=route,
        model_resolution=resolution,
        messages=[
            LlmMessage(role="system", content="Interpret the request."),
            LlmMessage(role="user", content='{"query":"Create 20 percentge questions"}'),
        ],
        temperature=route.temperature,
        max_tokens=route.max_tokens,
        provider_options={},
    )


def _credentials() -> ProviderCredentials:
    return ProviderCredentials(provider="bedrock", region="ap-south-1")


class TestBedrockStructuredOutputRequestShape:
    """Test 13 — the provider request carries the native schema, not a prompt hack."""

    def test_request_intelligence_role_sends_native_json_schema(self, tmp_path: Path) -> None:
        client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        adapter.generate(
            request=_execution_request(tmp_path, task_role="request_intelligence"),
            credentials=_credentials(),
        )

        assert client.captured is not None
        text_format = client.captured["outputConfig"]["textFormat"]
        assert text_format["type"] == "json_schema"
        json_schema = text_format["structure"]["jsonSchema"]
        assert json_schema["name"] == PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME
        # The API takes the schema as a string; it must round-trip to the static schema.
        assert json.loads(json_schema["schema"]) == PRACTICE_REQUEST_INTELLIGENCE_SCHEMA

    def test_schema_is_static_across_requests(self, tmp_path: Path) -> None:
        """Bedrock caches a compiled grammar per schema, so it must not vary."""
        schemas = []
        for _ in range(3):
            client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
            adapter = BedrockProviderAdapter(client_factory=lambda _c, _t, c=client: c)
            adapter.generate(
                request=_execution_request(tmp_path, task_role="request_intelligence"),
                credentials=_credentials(),
            )
            assert client.captured is not None
            schemas.append(client.captured["outputConfig"]["textFormat"]["structure"]["jsonSchema"])
        assert schemas[0] == schemas[1] == schemas[2]

    def test_system_and_user_messages_are_separated(self, tmp_path: Path) -> None:
        client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        adapter.generate(
            request=_execution_request(tmp_path, task_role="request_intelligence"),
            credentials=_credentials(),
        )

        assert client.captured is not None
        assert client.captured["modelId"] == "zai.glm-4.7-flash"
        assert client.captured["system"] == [{"text": "Interpret the request."}]
        assert client.captured["messages"][0]["role"] == "user"
        assert client.captured["inferenceConfig"]["maxTokens"] == 700

    def test_no_service_tier_or_performance_override_is_sent(self, tmp_path: Path) -> None:
        """Standard on-demand tier only — latency is measured before tuning it."""
        client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        adapter.generate(
            request=_execution_request(tmp_path, task_role="request_intelligence"),
            credentials=_credentials(),
        )

        assert client.captured is not None
        assert "serviceTier" not in client.captured
        assert "performanceConfig" not in client.captured

    def test_role_without_a_static_schema_sends_no_output_config(self, tmp_path: Path) -> None:
        """``planner`` has no registered grammar, so it stays a free-text call.

        The role here must be one genuinely absent from the schema map. Both
        ``request_intelligence`` and ``verifier`` are now intentionally constrained, so
        using either would assert the opposite of the contract rather than the invariant.
        """
        client = _FakeBedrockClient(_converse_response("plain text answer"))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        result = adapter.generate(
            request=_execution_request(tmp_path, task_role="planner"),
            credentials=_credentials(),
        )

        assert client.captured is not None
        assert "outputConfig" not in client.captured
        assert result.metadata["native_structured_output"] is False

    def test_verifier_role_is_grammar_constrained(self, tmp_path: Path) -> None:
        """The Answer Authority wire shape, so a Bedrock Authority cannot emit prose."""
        client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        adapter.generate(
            request=_execution_request(tmp_path, task_role="verifier"),
            credentials=_credentials(),
        )

        assert client.captured is not None
        json_schema = client.captured["outputConfig"]["textFormat"]["structure"][
            "jsonSchema"
        ]
        assert json_schema["name"] == "practice_verification_result"
        schema = json.loads(json_schema["schema"])
        assert set(schema["required"]) == {
            "schema_version",
            "generation_item_id",
            "slot_id",
            "decision",
            "valid_option_ids",
            "reason_codes",
        }
        assert schema["properties"]["valid_option_ids"]["items"]["enum"] == [
            "0",
            "1",
            "2",
            "3",
        ]

    def test_generator_role_is_not_grammar_constrained(self, tmp_path: Path) -> None:
        """Authoring must stay free-text JSON on this provider.

        The canonical GenerationEnvelope types ``options`` as an array of strings,
        but the schema-v2 author contract requires {option_id, value} objects. Sending
        that schema as a Bedrock grammar forbids the shape the prompt asks for — two
        model families returned an empty options array and every item was rejected.
        """
        client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        result = adapter.generate(
            request=_execution_request(tmp_path, task_role="generator"),
            credentials=_credentials(),
        )

        assert client.captured is not None
        assert "outputConfig" not in client.captured
        assert result.metadata["native_structured_output"] is False

    def test_request_intelligence_schema_is_unchanged(
        self, tmp_path: Path
    ) -> None:
        """The interpretation grammar is the only one this provider sends."""
        from schemas.practice_request_intelligence import (
            PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME,
            practice_request_intelligence_schema_json,
        )

        client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        adapter.generate(
            request=_execution_request(tmp_path, task_role="request_intelligence"),
            credentials=_credentials(),
        )

        assert client.captured is not None
        json_schema = client.captured["outputConfig"]["textFormat"]["structure"][
            "jsonSchema"
        ]
        assert json_schema["name"] == PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME
        assert json_schema["schema"] == practice_request_intelligence_schema_json()


class TestBedrockResponseDeserialization:
    """Test 14 — the Converse response maps onto the shared execution result."""

    def test_content_usage_and_finish_reason_are_normalized(self, tmp_path: Path) -> None:
        client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        result = adapter.generate(
            request=_execution_request(tmp_path, task_role="request_intelligence"),
            credentials=_credentials(),
        )

        assert json.loads(result.content) == _VALID_RESPONSE
        assert result.provider == "bedrock"
        assert result.model == "practice_request_intelligence"
        assert result.finish_reason == "end_turn"
        assert result.normalized_finish_reason == "completed"
        assert (result.input_tokens, result.output_tokens) == (120, 45)
        assert result.usage_source == "provider_reported"
        assert result.metadata["native_structured_output"] is True

    def test_empty_content_raises_a_response_error(self, tmp_path: Path) -> None:
        client = _FakeBedrockClient(_converse_response("", stop_reason="max_tokens"))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        with pytest.raises(LlmProviderResponseError) as exc_info:
            adapter.generate(
                request=_execution_request(tmp_path, task_role="request_intelligence"),
                credentials=_credentials(),
            )

        assert exc_info.value.normalized_finish_reason == "output_token_exhausted"
        assert exc_info.value.input_tokens == 120


class TestBedrockFailureMapping:
    """Test 15 — provider failures follow the existing bounded error vocabulary."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        (
            ("AccessDeniedException", "authentication_failed"),
            ("ThrottlingException", "rate_limited"),
            ("ResourceNotFoundException", "model_not_found"),
            ("ModelTimeoutException", "timeout"),
            ("InternalServerException", "provider_unavailable"),
            ("ValidationException", "invalid_request"),
            ("SomethingUnmapped", "unknown_provider_error"),
        ),
    )
    def test_botocore_error_codes_map_to_failure_kinds(
        self,
        tmp_path: Path,
        code: str,
        expected: str,
    ) -> None:
        error = Exception("bedrock failed")
        error.response = {"Error": {"Code": code}}
        client = _FakeBedrockClient(error=error)
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(
                request=_execution_request(tmp_path, task_role="request_intelligence"),
                credentials=_credentials(),
            )

        assert exc_info.value.failure_kind == expected
        assert exc_info.value.provider == "bedrock"

    def test_unmapped_5xx_status_falls_back_to_provider_unavailable(
        self,
        tmp_path: Path,
    ) -> None:
        error = Exception("bedrock failed")
        error.response = {
            "Error": {"Code": "SomethingUnmapped"},
            "ResponseMetadata": {"HTTPStatusCode": 503},
        }
        client = _FakeBedrockClient(error=error)
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(
                request=_execution_request(tmp_path, task_role="request_intelligence"),
                credentials=_credentials(),
            )

        assert exc_info.value.failure_kind == "provider_unavailable"

    def test_access_denied_is_not_silently_retried_as_a_different_model(
        self,
        tmp_path: Path,
    ) -> None:
        """A first-use Marketplace/IAM gap must surface, not hide behind a fallback."""
        error = Exception("access denied")
        error.response = {"Error": {"Code": "AccessDeniedException"}}
        client = _FakeBedrockClient(error=error)
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(
                request=_execution_request(tmp_path, task_role="request_intelligence"),
                credentials=_credentials(),
            )

        # The alias itself configures no fallback model, so eligibility is moot here —
        # what matters is that the reason stays diagnosable.
        assert exc_info.value.failure_kind in FALLBACK_ELIGIBLE_FAILURE_KINDS
        assert exc_info.value.model_alias == "practice_request_intelligence"

    def test_error_message_never_contains_prompt_content(self, tmp_path: Path) -> None:
        error = Exception("boom")
        error.response = {"Error": {"Code": "ValidationException"}}
        client = _FakeBedrockClient(error=error)
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(
                request=_execution_request(tmp_path, task_role="request_intelligence"),
                credentials=_credentials(),
            )

        assert "Create 20 percentge questions" not in str(exc_info.value)


class TestBedrockProviderRegistration:
    def test_factory_resolves_the_bedrock_adapter(self) -> None:
        assert isinstance(
            ProviderAdapterFactory().get_provider("bedrock"),
            BedrockProviderAdapter,
        )

    def test_adapter_requires_no_api_key(self, tmp_path: Path) -> None:
        """AWS IAM is the only credential; no key, secret, or endpoint is involved."""
        client = _FakeBedrockClient(_converse_response(json.dumps(_VALID_RESPONSE)))
        adapter = BedrockProviderAdapter(client_factory=lambda _c, _t: client)

        result = adapter.generate(
            request=_execution_request(tmp_path, task_role="request_intelligence"),
            credentials=ProviderCredentials(provider="bedrock"),
        )

        assert result.content
