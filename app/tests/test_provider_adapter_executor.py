"""
app/tests/test_provider_adapter_executor.py
--------------------------------------------
Unit tests for ProviderAdapterExecutor (Part 6).

Tests cover:
- Resolves credentials through ProviderCredentialResolver
- Gets adapter through ProviderAdapterFactory
- Calls selected adapter
- mock provider end-to-end works
- openai adapter path works with fake client and fake env
- azure adapter path works with fake client and fake env
- missing env secret raises controlled error
- adapter is not called if credential resolution fails
- no graph call
- no AWS call
- no .env.local reliance
- no importlib.reload()
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from observability.context import bind_request_context
from observability.llm_usage import (
    begin_llm_usage_collection,
    bind_llm_attempt_type,
    reset_llm_usage_collection,
    snapshot_llm_usage_records,
)
from schemas.llm import LlmMessage
from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
from schemas.llm_routing import RouteDecision
from schemas.llm_usage import ProviderTokenUsage
from services.llm.providers.usage import set_stream_usage
from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.model_config_resolver import ModelConfigResolver
from services.llm_orchestration.model_execution import ProviderAdapterExecutor
from services.llm_providers.azure_openai_provider import AzureOpenAIProviderAdapter
from services.llm_providers.errors import (
    LlmProviderConfigurationError,
    LlmProviderExecutionError,
)
from services.llm_providers.mock_provider import MockProviderAdapter
from services.llm_providers.openai_provider import OpenAIProviderAdapter
from services.llm_providers.provider_factory import ProviderAdapterFactory
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.errors import SecretNotFoundError
from services.secrets.provider_credentials import ProviderCredentialResolver

# ---------------------------------------------------------------------------
# YAML configs for tests
# ---------------------------------------------------------------------------

_MOCK_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: safe_mock
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 800
    models:
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 1
    provider_profiles:
      local_mock:
        provider: mock
""")

_OPENAI_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: gpt4o_default
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 800
    models:
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 1
      gpt4o_default:
        provider: openai
        provider_profile: openai_primary
        model_id: gpt-4o
        model_label: gpt-4o-default
        cost_tier: medium
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 30
    provider_profiles:
      local_mock:
        provider: mock
      openai_primary:
        provider: openai
        api_key_env: OPENAI_API_KEY
""")

_AZURE_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: azure_gpt4o
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 800
    models:
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 1
      azure_gpt4o:
        provider: azure_openai
        provider_profile: azure_primary
        deployment: gpt-4o-deployment
        model_label: azure-gpt4o
        cost_tier: medium
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 30
    provider_profiles:
      local_mock:
        provider: mock
      azure_primary:
        provider: azure_openai
        api_key_env: AZURE_OPENAI_API_KEY
        endpoint_env: AZURE_OPENAI_ENDPOINT
        api_version_env: AZURE_OPENAI_API_VERSION
""")


def _write_yaml(tmp_path: Path, content: str) -> LlmConfigRegistry:
    yaml_path = tmp_path / "llm_orchestration.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return LlmConfigRegistry(yaml_path=yaml_path)


def _mock_route_decision() -> RouteDecision:
    return RouteDecision(
        route_id="general.generator.default",
        subject="general",
        task_role="generator",
        difficulty="default",
        model="safe_mock",
        prompt="subjects/general_generator.md",
        temperature=0.3,
        max_tokens=800,
        provider_options={},
        fallback_attempts=[],
        route_source="exact",
    )


def _openai_route_decision() -> RouteDecision:
    return RouteDecision(
        route_id="general.generator.default",
        subject="general",
        task_role="generator",
        difficulty="default",
        model="gpt4o_default",
        prompt="subjects/general_generator.md",
        temperature=0.3,
        max_tokens=800,
        provider_options={},
        fallback_attempts=[],
        route_source="exact",
    )


def _azure_route_decision() -> RouteDecision:
    return RouteDecision(
        route_id="general.generator.default",
        subject="general",
        task_role="generator",
        difficulty="default",
        model="azure_gpt4o",
        prompt="subjects/general_generator.md",
        temperature=0.3,
        max_tokens=800,
        provider_options={},
        fallback_attempts=[],
        route_source="exact",
    )


def _messages() -> list[LlmMessage]:
    return [
        LlmMessage(role="system", content="You are a tutor."),
        LlmMessage(role="user", content="What is photosynthesis?"),
    ]


def _make_request(
    tmp_path: Path, yaml: str, route_decision: RouteDecision
) -> ProviderExecutionRequest:
    registry = _write_yaml(tmp_path, yaml)
    resolver = ModelConfigResolver(registry=registry)
    model_resolution = resolver.resolve(route_decision)
    return ProviderExecutionRequest(
        route_decision=route_decision,
        model_resolution=model_resolution,
        messages=_messages(),
        temperature=0.3,
        max_tokens=800,
    )


def _fake_completion(content: str = "Answer.") -> object:
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message, finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=usage)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestProviderAdapterExecutorConstructor:
    def test_requires_credential_resolver(self) -> None:
        factory = ProviderAdapterFactory(adapter_map={"mock": MockProviderAdapter()})
        with pytest.raises(TypeError):
            ProviderAdapterExecutor(credential_resolver=None, provider_factory=factory)  # type: ignore[arg-type]

    def test_requires_provider_factory(self) -> None:
        env_resolver = EnvSecretResolver()
        cred_resolver = ProviderCredentialResolver(secret_resolver=env_resolver)
        with pytest.raises(TypeError):
            ProviderAdapterExecutor(credential_resolver=cred_resolver, provider_factory=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Mock provider end-to-end
# ---------------------------------------------------------------------------


class TestProviderAdapterExecutorMockPath:
    def test_mock_end_to_end_returns_result(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path, _MOCK_YAML, _mock_route_decision())

        mock_adapter = MockProviderAdapter(content="Expected answer.")
        factory = ProviderAdapterFactory(adapter_map={"mock": mock_adapter})
        # Mock profile has no env var refs — credential resolver won't call env
        env_resolver = EnvSecretResolver()
        cred_resolver = ProviderCredentialResolver(secret_resolver=env_resolver)

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )
        result = executor.execute(request)

        assert isinstance(result, ModelExecutionResult)
        assert result.content == "Expected answer."
        assert result.provider == "mock"

    def test_mock_adapter_is_called_once(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path, _MOCK_YAML, _mock_route_decision())
        mock_adapter = MockProviderAdapter()
        factory = ProviderAdapterFactory(adapter_map={"mock": mock_adapter})
        cred_resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )
        executor.execute(request)

        assert mock_adapter.call_count == 1

    def test_stream_terminal_usage_is_recorded(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path, _MOCK_YAML, _mock_route_decision())

        class UsageStreamAdapter:
            last_stream_finish_reason = "stop"

            def generate_stream(self, *, request, credentials):
                del request, credentials
                yield "Answer"
                set_stream_usage(
                    ProviderTokenUsage(
                        input_tokens=12,
                        output_tokens=3,
                        total_tokens=15,
                    )
                )

        executor = ProviderAdapterExecutor(
            credential_resolver=ProviderCredentialResolver(
                secret_resolver=EnvSecretResolver()
            ),
            provider_factory=ProviderAdapterFactory(
                adapter_map={"mock": UsageStreamAdapter()}  # type: ignore[dict-item]
            ),
        )

        with bind_request_context(request_id="request-stream-usage"):
            token = begin_llm_usage_collection()
            try:
                assert list(executor.execute_stream(request)) == ["Answer"]
                records = snapshot_llm_usage_records()
            finally:
                reset_llm_usage_collection(token)

        assert len(records) == 1
        assert records[0].streaming is True
        assert records[0].total_tokens == 15
        assert records[0].status == "succeeded"

    def test_stream_close_records_controlled_cancellation(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path, _MOCK_YAML, _mock_route_decision())

        class CancellableStreamAdapter:
            last_stream_finish_reason = None

            def generate_stream(self, *, request, credentials):
                del request, credentials
                yield "first"
                yield "second"

        executor = ProviderAdapterExecutor(
            credential_resolver=ProviderCredentialResolver(
                secret_resolver=EnvSecretResolver()
            ),
            provider_factory=ProviderAdapterFactory(
                adapter_map={"mock": CancellableStreamAdapter()}  # type: ignore[dict-item]
            ),
        )

        with bind_request_context(request_id="request-stream-cancelled"):
            token = begin_llm_usage_collection()
            try:
                stream = executor.execute_stream(request)
                assert next(stream) == "first"
                stream.close()
                records = snapshot_llm_usage_records()
            finally:
                reset_llm_usage_collection(token)

        assert len(records) == 1
        assert records[0].status == "cancelled"
        assert records[0].usage_source == "unavailable"


# ---------------------------------------------------------------------------
# OpenAI adapter path with fake client + fake env
# ---------------------------------------------------------------------------


class TestProviderAdapterExecutorOpenAIPath:
    def test_openai_path_with_fake_client_and_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key-for-openai")

        request = _make_request(tmp_path, _OPENAI_YAML, _openai_route_decision())

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion("OpenAI answer.")

        openai_adapter = OpenAIProviderAdapter(client_factory=lambda _creds: fake_client)
        factory = ProviderAdapterFactory(adapter_map={"openai": openai_adapter})
        env_resolver = EnvSecretResolver()
        cred_resolver = ProviderCredentialResolver(secret_resolver=env_resolver)

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )
        result = executor.execute(request)

        assert result.content == "OpenAI answer."
        assert result.provider == "openai"

    def test_openai_call_is_recorded_at_central_executor_boundary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key-for-openai")
        request = _make_request(tmp_path, _OPENAI_YAML, _openai_route_decision())
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion()
        executor = ProviderAdapterExecutor(
            credential_resolver=ProviderCredentialResolver(
                secret_resolver=EnvSecretResolver()
            ),
            provider_factory=ProviderAdapterFactory(
                adapter_map={
                    "openai": OpenAIProviderAdapter(
                        client_factory=lambda _creds: fake_client
                    )
                }
            ),
        )

        with bind_request_context(request_id="request-executor"):
            token = begin_llm_usage_collection()
            try:
                executor.execute(request)
                records = snapshot_llm_usage_records()
            finally:
                reset_llm_usage_collection(token)

        assert len(records) == 1
        assert records[0].role == "general.generator.default"
        assert records[0].provider == "openai"
        assert records[0].model == "gpt-4o"
        assert records[0].input_tokens == 10
        assert records[0].output_tokens == 5
        assert records[0].total_tokens == 15
        assert records[0].status == "succeeded"

    def test_caller_attempt_type_is_preserved_for_repair(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key-for-openai")
        request = _make_request(tmp_path, _OPENAI_YAML, _openai_route_decision())
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion()
        executor = ProviderAdapterExecutor(
            credential_resolver=ProviderCredentialResolver(
                secret_resolver=EnvSecretResolver()
            ),
            provider_factory=ProviderAdapterFactory(
                adapter_map={
                    "openai": OpenAIProviderAdapter(
                        client_factory=lambda _creds: fake_client
                    )
                }
            ),
        )

        with bind_request_context(request_id="request-repair"):
            token = begin_llm_usage_collection()
            try:
                with bind_llm_attempt_type("repair"):
                    executor.execute(request)
                records = snapshot_llm_usage_records()
            finally:
                reset_llm_usage_collection(token)

        assert len(records) == 1
        assert records[0].attempt_type == "repair"

    def test_openai_missing_env_raises_secret_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        request = _make_request(tmp_path, _OPENAI_YAML, _openai_route_decision())

        fake_client = MagicMock()
        openai_adapter = OpenAIProviderAdapter(client_factory=lambda _creds: fake_client)
        factory = ProviderAdapterFactory(adapter_map={"openai": openai_adapter})
        env_resolver = EnvSecretResolver()
        cred_resolver = ProviderCredentialResolver(secret_resolver=env_resolver)

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )

        with pytest.raises(SecretNotFoundError):
            executor.execute(request)

    def test_adapter_not_called_if_credential_resolution_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        request = _make_request(tmp_path, _OPENAI_YAML, _openai_route_decision())

        openai_adapter = MagicMock(spec=OpenAIProviderAdapter)
        factory = ProviderAdapterFactory(adapter_map={"openai": openai_adapter})
        env_resolver = EnvSecretResolver()
        cred_resolver = ProviderCredentialResolver(secret_resolver=env_resolver)

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )

        with pytest.raises(SecretNotFoundError):
            executor.execute(request)

        # Adapter must NOT have been called
        openai_adapter.generate.assert_not_called()


# ---------------------------------------------------------------------------
# Azure adapter path with fake client + fake env
# ---------------------------------------------------------------------------


class TestProviderAdapterExecutorAzurePath:
    def test_azure_path_with_fake_client_and_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        request = _make_request(tmp_path, _AZURE_YAML, _azure_route_decision())

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion("Azure answer.")

        azure_adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: fake_client)
        factory = ProviderAdapterFactory(adapter_map={"azure_openai": azure_adapter})
        env_resolver = EnvSecretResolver()
        cred_resolver = ProviderCredentialResolver(secret_resolver=env_resolver)

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )
        with bind_request_context(request_id="request-azure-executor"):
            token = begin_llm_usage_collection()
            try:
                result = executor.execute(request)
                records = snapshot_llm_usage_records()
            finally:
                reset_llm_usage_collection(token)

        assert result.content == "Azure answer."
        assert result.provider == "azure_openai"
        assert len(records) == 1
        assert records[0].model == "gpt-4o-deployment"
        assert records[0].deployment == "gpt-4o-deployment"

    def test_azure_missing_endpoint_env_raises_secret_not_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-azure-key")
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

        request = _make_request(tmp_path, _AZURE_YAML, _azure_route_decision())

        azure_adapter = AzureOpenAIProviderAdapter(client_factory=lambda _creds: MagicMock())
        factory = ProviderAdapterFactory(adapter_map={"azure_openai": azure_adapter})
        env_resolver = EnvSecretResolver()
        cred_resolver = ProviderCredentialResolver(secret_resolver=env_resolver)

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )

        with pytest.raises(SecretNotFoundError):
            executor.execute(request)


# ---------------------------------------------------------------------------
# Adapter execution errors propagate
# ---------------------------------------------------------------------------


class TestProviderAdapterExecutorErrorPropagation:
    def test_provider_execution_error_propagates(self, tmp_path: Path) -> None:
        request = _make_request(tmp_path, _MOCK_YAML, _mock_route_decision())

        failing_adapter = MagicMock()
        failing_adapter.generate.side_effect = LlmProviderExecutionError("SDK failed.")

        factory = ProviderAdapterFactory(adapter_map={"mock": failing_adapter})
        cred_resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )

        with pytest.raises(LlmProviderExecutionError, match="SDK failed"):
            executor.execute(request)

    def test_unsupported_provider_in_factory_raises_config_error(
        self, tmp_path: Path
    ) -> None:
        request = _make_request(tmp_path, _MOCK_YAML, _mock_route_decision())

        # Factory has no "mock" adapter
        factory = ProviderAdapterFactory(adapter_map={})
        cred_resolver = ProviderCredentialResolver(secret_resolver=EnvSecretResolver())

        executor = ProviderAdapterExecutor(
            credential_resolver=cred_resolver,
            provider_factory=factory,
        )

        with pytest.raises(LlmProviderConfigurationError):
            executor.execute(request)


# ---------------------------------------------------------------------------
# Import safety (no importlib.reload)
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_import_model_execution_module(self) -> None:
        import services.llm_orchestration.model_execution  # noqa: F401
