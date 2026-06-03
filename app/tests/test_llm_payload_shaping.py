"""
app/tests/test_llm_payload_shaping.py
---------------------------------------
Unit tests for provider/model capability payload shaping (Azure reasoning fix).
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import pytest

from schemas.llm import LlmMessage
from schemas.llm_orchestration import ProviderExecutionRequest
from schemas.llm_routing import ModelConfig, RouteDecision
from services.llm.providers.azure_openai_provider import (
    AzureOpenAIProviderAdapter,
    _azure_error_diagnostics,
    _is_unsupported_parameter_error,
    _log_azure_call_failure,
)
from services.llm.providers.errors import FALLBACK_ELIGIBLE_FAILURE_KINDS
from services.llm.providers.payload_shaping import (
    build_azure_openai_chat_completion_kwargs,
    effective_supports_temperature,
    effective_token_budget_param,
)
from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.model_config_resolver import ModelConfigResolver
from services.secrets.provider_credentials import ProviderCredentials

_REASONING_YAML = textwrap.dedent("""\
    version: 1
    routes:
      reasoning:
        generator:
          advanced:
            model: reasoning_advanced_generator
            prompt: subjects/reasoning_generator.md
            temperature: 0.3
            max_tokens: 3200
            provider_options:
              thinking: true
    models:
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 1
      reasoning_advanced_generator:
        provider: azure_openai
        provider_profile: azure_primary
        deployment: o4-mini
        supports_streaming: true
        supports_thinking: false
        supports_reasoning: true
        reasoning_effort: medium
        timeout_seconds: 45
      openai_gpt_4_1_mini:
        provider: azure_openai
        provider_profile: azure_primary
        deployment: gpt-4.1-mini
        supports_streaming: true
        supports_thinking: false
        supports_reasoning: false
        timeout_seconds: 25
    provider_profiles:
      local_mock:
        provider: mock
      azure_primary:
        provider: azure_openai
        api_key_env: AZURE_OPENAI_API_KEY
        endpoint_env: AZURE_OPENAI_ENDPOINT
        api_version_env: AZURE_OPENAI_API_VERSION
""")


def _registry(tmp_path: Path) -> LlmConfigRegistry:
    yaml_path = tmp_path / "llm_orchestration.yaml"
    yaml_path.write_text(_REASONING_YAML, encoding="utf-8")
    return LlmConfigRegistry(yaml_path=yaml_path)


def _route_decision(model: str = "reasoning_advanced_generator") -> RouteDecision:
    return RouteDecision(
        route_id="reasoning.generator.advanced",
        subject="reasoning",
        task_role="generator",
        difficulty="advanced",
        model=model,
        prompt="subjects/reasoning_generator.md",
        temperature=0.3,
        max_tokens=3200,
        provider_options={"thinking": True},
        fallback_attempts=[],
        route_source="exact",
    )


def _make_request(
    tmp_path: Path,
    *,
    model: str = "reasoning_advanced_generator",
    provider_options: dict | None = None,
) -> ProviderExecutionRequest:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))
    route = _route_decision(model=model)
    opts = (
        {"thinking": True}
        if provider_options is None and model == "reasoning_advanced_generator"
        else (provider_options or {})
    )
    route = route.model_copy(update={"provider_options": opts})
    model_resolution = resolver.resolve(route)
    return ProviderExecutionRequest(
        route_decision=route,
        model_resolution=model_resolution,
        messages=[LlmMessage(role="user", content="Solve this puzzle.")],
        temperature=0.3,
        max_tokens=3200,
        provider_options=dict(opts),
    )


class TestModelConfigReasoningCapabilities:
    def test_azure_reasoning_model_gets_completion_token_budget(self) -> None:
        cfg = ModelConfig(
            provider="azure_openai",
            provider_profile="azure_foundry_v1",
            deployment="o4-mini",
            supports_streaming=True,
            supports_reasoning=True,
            timeout_seconds=45,
        )
        assert effective_token_budget_param(cfg) == "max_completion_tokens"
        assert effective_supports_temperature(cfg) is False

    def test_azure_standard_model_keeps_max_tokens_and_temperature(self) -> None:
        cfg = ModelConfig(
            provider="azure_openai",
            provider_profile="azure_foundry_v1",
            deployment="gpt-4.1-mini",
            supports_streaming=True,
            supports_reasoning=False,
            timeout_seconds=25,
        )
        assert effective_token_budget_param(cfg) == "max_tokens"
        assert effective_supports_temperature(cfg) is True

    def test_production_registry_o4_mini_capabilities(self) -> None:
        registry = LlmConfigRegistry()
        cfg = registry.get_model("openai_o4_mini")
        assert cfg is not None
        assert effective_token_budget_param(cfg) == "max_completion_tokens"
        assert effective_supports_temperature(cfg) is False
        assert cfg.supports_reasoning is True

    def test_production_registry_gpt_4_1_mini_unchanged(self) -> None:
        registry = LlmConfigRegistry()
        cfg = registry.get_model("openai_gpt_4_1_mini")
        assert cfg is not None
        assert effective_token_budget_param(cfg) == "max_tokens"
        assert effective_supports_temperature(cfg) is True


class TestAzureReasoningPayloadShaping:
    def test_o4_mini_uses_max_completion_tokens_not_max_tokens(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path)
        kwargs, meta = build_azure_openai_chat_completion_kwargs(
            request=request,
            deployment="o4-mini",
            stream=False,
        )
        assert kwargs["max_completion_tokens"] == 3200
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs
        assert meta.token_budget_param_used == "max_completion_tokens"
        assert "max_tokens" in meta.dropped_params
        assert "temperature" in meta.dropped_params

    def test_o4_mini_drops_thinking_and_reasoning_effort_by_default(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path)
        kwargs, meta = build_azure_openai_chat_completion_kwargs(
            request=request,
            deployment="o4-mini",
        )
        assert "thinking" not in kwargs
        assert "reasoning_effort" not in kwargs
        assert meta.reasoning_param_sent is False
        assert "reasoning_effort" in meta.dropped_params

    def test_gpt_4_1_mini_preserves_max_tokens_and_temperature(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path, model="openai_gpt_4_1_mini")
        kwargs, meta = build_azure_openai_chat_completion_kwargs(
            request=request,
            deployment="gpt-4.1-mini",
        )
        assert kwargs["max_tokens"] == 3200
        assert kwargs["temperature"] == 0.3
        assert "max_completion_tokens" not in kwargs
        assert meta.token_budget_param_used == "max_tokens"


class FakeAzureOpenAIClient:
    def __init__(self) -> None:
        self.received_kwargs: dict = {}

        def _create(**kwargs):  # noqa: ANN202
            self.received_kwargs = kwargs
            message = types.SimpleNamespace(content="Answer.")
            choice = types.SimpleNamespace(message=message, finish_reason="stop")
            usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=2)
            return types.SimpleNamespace(choices=[choice], usage=usage)

        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )


class TestAzureAdapterPayloadIntegration:
    def test_adapter_sends_reasoning_safe_payload_for_o4_mini(self, tmp_path: Path) -> None:
        fake = FakeAzureOpenAIClient()
        adapter = AzureOpenAIProviderAdapter(
            client_factory=lambda _creds: fake,
        )
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        adapter.generate(request=_make_request(tmp_path), credentials=creds)
        assert fake.received_kwargs["max_completion_tokens"] == 3200
        assert "max_tokens" not in fake.received_kwargs
        assert "temperature" not in fake.received_kwargs

    def test_adapter_preserves_standard_gpt_payload(self, tmp_path: Path) -> None:
        fake = FakeAzureOpenAIClient()
        adapter = AzureOpenAIProviderAdapter(
            client_factory=lambda _creds: fake,
        )
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        adapter.generate(
            request=_make_request(tmp_path, model="openai_gpt_4_1_mini"),
            credentials=creds,
        )
        assert fake.received_kwargs["max_tokens"] == 3200
        assert fake.received_kwargs["temperature"] == 0.3
        assert "max_completion_tokens" not in fake.received_kwargs


class TestAzureUnsupportedParameterErrors:
    def test_unsupported_parameter_classified_as_fallback_eligible(self) -> None:
        exc = types.SimpleNamespace(
            status_code=400,
            code="unsupported_parameter",
            body={
                "error": {
                    "code": "unsupported_parameter",
                    "message": "Unsupported parameter: max_tokens",
                    "param": "max_tokens",
                    "type": "invalid_request_error",
                }
            },
        )
        assert _is_unsupported_parameter_error(exc) is True
        assert "unsupported_parameter" in FALLBACK_ELIGIBLE_FAILURE_KINDS

    def test_error_diagnostics_includes_safe_param_field(self) -> None:
        exc = types.SimpleNamespace(
            status_code=400,
            code="unsupported_parameter",
            body={
                "error": {
                    "code": "unsupported_parameter",
                    "message": "Unsupported parameter: temperature",
                    "param": "temperature",
                    "type": "invalid_request_error",
                }
            },
        )
        diag = _azure_error_diagnostics(exc)
        assert diag["status_code"] == 400
        assert diag["provider_error_code"] == "unsupported_parameter"
        assert diag["provider_error_param"] == "temperature"
        assert diag["provider_error_type"] == "invalid_request_error"

    def test_failure_log_does_not_include_secrets(self, caplog: pytest.LogCaptureFixture) -> None:
        exc = types.SimpleNamespace(
            status_code=400,
            code="unsupported_parameter",
            body={
                "error": {
                    "code": "unsupported_parameter",
                    "message": "Unsupported parameter: max_tokens",
                    "param": "max_tokens",
                }
            },
        )
        with caplog.at_level("WARNING"):
            _log_azure_call_failure(
                exc=exc,
                operation="generate",
                model_alias="reasoning_advanced_generator",
                deployment="o4-mini",
                route_id="reasoning.generator.advanced",
                azure_api_mode="azure_openai_v1",
                failure_kind="unsupported_parameter",
            )
        combined = " ".join(caplog.messages)
        assert "sk-" not in combined
        assert "Solve this puzzle" not in combined
