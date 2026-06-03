"""
Unit tests for Azure-hosted DeepSeek profile and aliases (no real API calls).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import get_settings
from schemas.llm import LlmMessage
from schemas.llm_orchestration import ProviderExecutionRequest
from schemas.llm_routing import RouteDecision
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.orchestration.model_execution import ProviderAdapterExecutor
from services.llm.providers.azure_openai_provider import AzureOpenAIProviderAdapter
from services.llm.providers.errors import (
    FALLBACK_ELIGIBLE_FAILURE_KINDS,
    LlmProviderExecutionError,
)
from services.llm.providers.provider_factory import ProviderAdapterFactory
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.provider_credentials import ProviderCredentialResolver, ProviderCredentials


def _fake_completion(content: str = "Azure DeepSeek answer.", finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=24),
    )


def _route_decision(
    *,
    model: str,
    route_id: str = "math.generator.deepseek_azure_test",
) -> RouteDecision:
    return RouteDecision(
        route_id=route_id,
        subject="math",
        task_role="generator",
        difficulty="deepseek_azure_test",
        model=model,
        prompt="subjects/math_generator.md",
        temperature=0.15,
        max_tokens=2600,
        provider_options={},
        fallback=["advanced", "default", "safe_mock"],
        fallback_attempts=[],
        route_source="exact",
    )


def _make_request(tmp_path: Path, yaml_text: str, route: RouteDecision) -> ProviderExecutionRequest:
    yaml_path = tmp_path / "llm.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    registry = LlmConfigRegistry(yaml_path=yaml_path)
    resolver = ModelConfigResolver(registry=registry)
    resolution = resolver.resolve(route)
    return ProviderExecutionRequest(
        route_decision=route,
        model_resolution=resolution,
        messages=[LlmMessage(role="user", content="Solve x^2 = 4.")],
        temperature=route.temperature,
        max_tokens=route.max_tokens,
        provider_options={},
    )


_AZURE_DEEPSEEK_YAML = textwrap.dedent("""\
    version: 1
    routes:
      math:
        generator:
          default:
            model: math_advanced_generator
            prompt: subjects/math_generator.md
            temperature: 0.15
            max_tokens: 2600
          deepseek_azure_test:
            model: deepseek_azure_reasoning_generator
            prompt: subjects/math_generator.md
            temperature: 0.15
            max_tokens: 2600
            fallback:
              - default
              - safe_mock
    models:
      deepseek_azure_reasoning_generator:
        provider: azure_openai
        provider_profile: azure_deepseek
        deployment: deepseek-r1-deploy
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 90
        fallback_models:
          - math_advanced_generator
      math_advanced_generator:
        provider: azure_openai
        provider_profile: azure_foundry_v1
        deployment: gpt-4.1
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 30
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
    provider_profiles:
      azure_deepseek:
        provider: azure_openai
        azure_api_mode: azure_deployment_chat_completions
        endpoint_env: AZURE_DEEPSEEK_ENDPOINT
        api_key_env: AZURE_DEEPSEEK_API_KEY
        api_version_env: AZURE_DEEPSEEK_API_VERSION
        optional_api_key: true
        optional_endpoint: true
        optional_api_version: true
      azure_foundry_v1:
        provider: azure_openai
        azure_api_mode: azure_openai_v1
        endpoint_env: AZURE_OPENAI_ENDPOINT
        api_key_env: AZURE_OPENAI_API_KEY
      local_mock:
        provider: mock
""")


class TestAzureDeepSeekProfileResolution:
    def test_profile_resolves_optional_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AZURE_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_DEEPSEEK_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_DEEPSEEK_API_VERSION", raising=False)
        resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())
        profile = LlmConfigRegistry().provider_profile_map["azure_deepseek"]
        creds = resolver.resolve(profile)
        assert creds.provider == "azure_openai"
        assert creds.api_key is None
        assert creds.endpoint is None
        assert creds.api_version is None
        assert creds.azure_api_mode == "azure_deployment_chat_completions"

    def test_native_deepseek_profile_unaffected(self) -> None:
        reg = LlmConfigRegistry()
        profile = reg.provider_profile_map["deepseek_primary"]
        assert profile.provider == "deepseek"
        assert profile.api_key_env == "DEEPSEEK_API_KEY"

    def test_azure_gpt_profile_unaffected(self) -> None:
        reg = LlmConfigRegistry()
        profile = reg.provider_profile_map["azure_foundry_v1"]
        assert profile.api_key_env == "AZURE_OPENAI_API_KEY"
        assert profile.endpoint_env == "AZURE_OPENAI_ENDPOINT"


class TestAzureDeepSeekAdapterExecution:
    def test_fake_client_returns_normalized_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_DEEPSEEK_REASONER_DEPLOYMENT", "deepseek-r1-deploy")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion()
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _c: fake_client)
        request = _make_request(
            tmp_path,
            _AZURE_DEEPSEEK_YAML,
            _route_decision(model="deepseek_azure_reasoning_generator"),
        )
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="azure-deepseek-key",
            endpoint="https://fake.openai.azure.com",
            api_version="2024-10-21",
            azure_api_mode="azure_deployment_chat_completions",
        )
        result = adapter.generate(request=request, credentials=creds)
        assert result.content == "Azure DeepSeek answer."
        assert result.provider == "azure_openai"
        assert result.finish_reason == "stop"
        assert result.input_tokens == 12
        assert result.output_tokens == 24
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "deepseek-r1-deploy"

    def test_missing_api_key_raises_provider_not_configured(self, tmp_path: Path) -> None:
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _c: MagicMock())
        request = _make_request(
            tmp_path,
            _AZURE_DEEPSEEK_YAML,
            _route_decision(model="deepseek_azure_reasoning_generator"),
        )
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key=None,
            endpoint="https://fake.openai.azure.com",
            api_version="2024-10-21",
        )
        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(request=request, credentials=creds)
        assert exc_info.value.failure_kind == "provider_not_configured"

    def test_missing_deployment_raises_model_not_configured(self, tmp_path: Path) -> None:
        yaml = _AZURE_DEEPSEEK_YAML.replace(
            "deployment: deepseek-r1-deploy",
            "deployment: ''",
        )
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _c: MagicMock())
        request = _make_request(
            tmp_path,
            yaml,
            _route_decision(model="deepseek_azure_reasoning_generator"),
        )
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="azure-deepseek-key",
            endpoint="https://fake.openai.azure.com",
            api_version="2024-10-21",
        )
        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(request=request, credentials=creds)
        assert exc_info.value.failure_kind == "model_not_configured"


class TestAzureDeepSeekFailureKinds:
    def test_provider_not_configured_is_fallback_eligible(self) -> None:
        assert "provider_not_configured" in FALLBACK_ELIGIBLE_FAILURE_KINDS

    def test_model_not_configured_is_fallback_eligible(self) -> None:
        assert "model_not_configured" in FALLBACK_ELIGIBLE_FAILURE_KINDS


class TestAzureDeepSeekSettings:
    def test_env_defaults_do_not_break_startup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config as cfg_module

        monkeypatch.delenv("AZURE_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_DEEPSEEK_ENDPOINT", raising=False)
        cfg_module._settings = None
        settings = get_settings()
        assert settings.azure_deepseek_api_key == ""
        assert settings.azure_deepseek_reasoner_deployment == ""


class TestAzureDeepSeekFactory:
    def test_uses_existing_azure_openai_adapter(self) -> None:
        factory = ProviderAdapterFactory()
        adapter = factory.get_provider("azure_openai")
        assert isinstance(adapter, AzureOpenAIProviderAdapter)

    def test_executor_resolves_azure_deepseek_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion("Done.")
        adapter = AzureOpenAIProviderAdapter(client_factory=lambda _c: fake_client)
        factory = ProviderAdapterFactory(adapter_map={"azure_openai": adapter, "mock": MagicMock()})
        executor = ProviderAdapterExecutor(
            credential_resolver=ProviderCredentialResolver(
                secret_resolver=EnvSecretResolver()
            ),
            provider_factory=factory,
        )
        monkeypatch.setenv("AZURE_DEEPSEEK_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_DEEPSEEK_ENDPOINT", "https://fake.openai.azure.com")
        monkeypatch.setenv("AZURE_DEEPSEEK_API_VERSION", "2024-10-21")
        monkeypatch.setenv("AZURE_DEEPSEEK_REASONER_DEPLOYMENT", "deepseek-r1-deploy")
        request = _make_request(
            tmp_path,
            _AZURE_DEEPSEEK_YAML,
            _route_decision(model="deepseek_azure_reasoning_generator"),
        )
        result = executor.execute(request)
        assert result.content == "Done."
