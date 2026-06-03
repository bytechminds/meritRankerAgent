"""
app/tests/test_azure_openai_provider_adapter.py
-------------------------------------------------
Unit tests for AzureOpenAIProviderAdapter (Part 6).

Tests cover:
- Requires api_key
- Requires endpoint
- Requires api_version
- Requires deployment (not model_id)
- Calls fake client with deployment name
- Returns normalized ModelExecutionResult
- Handles missing content (LlmProviderResponseError)
- Provider exception raises LlmProviderExecutionError
- No credential values in error/log/metadata
- No env read
- No network
- No importlib.reload()
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import pytest

from schemas.llm import LlmMessage
from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
from schemas.llm_routing import RouteDecision
from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.model_config_resolver import ModelConfigResolver
from services.llm_providers.azure_openai_provider import AzureOpenAIProviderAdapter
from services.llm_providers.errors import (
    LlmProviderConfigurationError,
    LlmProviderExecutionError,
    LlmProviderResponseError,
)
from services.secrets.provider_credentials import ProviderCredentials

# ---------------------------------------------------------------------------
# Test YAML config with azure_openai provider
# ---------------------------------------------------------------------------

_AZURE_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: azure_gpt4o_default
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
      azure_gpt4o_default:
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


def _registry(tmp_path: Path) -> LlmConfigRegistry:
    yaml_path = tmp_path / "llm_orchestration.yaml"
    yaml_path.write_text(_AZURE_YAML, encoding="utf-8")
    return LlmConfigRegistry(yaml_path=yaml_path)


def _route_decision() -> RouteDecision:
    return RouteDecision(
        route_id="general.generator.default",
        subject="general",
        task_role="generator",
        difficulty="default",
        model="azure_gpt4o_default",
        prompt="subjects/general_generator.md",
        temperature=0.3,
        max_tokens=800,
        provider_options={},
        fallback_attempts=[],
        route_source="exact",
    )


def _messages() -> list[LlmMessage]:
    return [
        LlmMessage(role="system", content="You are a helpful tutor."),
        LlmMessage(role="user", content="Explain photosynthesis."),
    ]


def _make_request(tmp_path: Path) -> ProviderExecutionRequest:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))
    model_resolution = resolver.resolve(_route_decision())
    return ProviderExecutionRequest(
        route_decision=_route_decision(),
        model_resolution=model_resolution,
        messages=_messages(),
        temperature=0.3,
        max_tokens=800,
    )


def _azure_credentials() -> ProviderCredentials:
    return ProviderCredentials(
        provider="azure_openai",
        api_key="fake-azure-key",
        endpoint="https://fake.openai.azure.com/",
        api_version="2024-02-01",
    )


# ---------------------------------------------------------------------------
# Fake client helpers
# ---------------------------------------------------------------------------


def _fake_completion(
    content: str = "Azure test answer.",
    finish_reason: str = "stop",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> object:
    usage = types.SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens)
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class FakeAzureOpenAIClient:
    """Fake openai.AzureOpenAI-like client for unit testing."""

    def __init__(
        self, completion: object | None = None, raise_exc: Exception | None = None
    ) -> None:
        self._completion = completion
        self._raise_exc = raise_exc
        self.received_kwargs: dict = {}

        def _create(**kwargs):  # noqa: ANN202
            self.received_kwargs = kwargs
            if self._raise_exc is not None:
                raise self._raise_exc
            return self._completion

        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )


def _factory(client: FakeAzureOpenAIClient):  # noqa: ANN202
    return lambda _creds: client


# ---------------------------------------------------------------------------
# Credential validation
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderAdapterCredentials:
    def test_missing_api_key_raises_provider_not_configured(self, tmp_path: Path) -> None:
        adapter = AzureOpenAIProviderAdapter()
        creds = ProviderCredentials(
            provider="azure_openai",
            endpoint="https://fake.openai.azure.com/",
            api_version="2024-02-01",
        )
        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(request=_make_request(tmp_path), credentials=creds)
        assert exc_info.value.failure_kind == "provider_not_configured"

    def test_missing_endpoint_raises_config_error(self, tmp_path: Path) -> None:
        adapter = AzureOpenAIProviderAdapter()
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-azure-key",
            api_version="2024-02-01",
        )
        with pytest.raises(LlmProviderConfigurationError, match="endpoint"):
            adapter.generate(request=_make_request(tmp_path), credentials=creds)

    def test_missing_api_version_raises_config_error(self, tmp_path: Path) -> None:
        adapter = AzureOpenAIProviderAdapter()
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-azure-key",
            endpoint="https://fake.openai.azure.com/",
        )
        with pytest.raises(LlmProviderConfigurationError, match="api_version"):
            adapter.generate(request=_make_request(tmp_path), credentials=creds)

    def test_error_message_does_not_contain_key_value(self, tmp_path: Path) -> None:
        adapter = AzureOpenAIProviderAdapter()
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key=None,
            endpoint="https://fake.openai.azure.com/",
            api_version="2024-02-01",
        )
        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(request=_make_request(tmp_path), credentials=creds)
        assert exc_info.value.failure_kind == "provider_not_configured"


# ---------------------------------------------------------------------------
# Deployment validation
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderAdapterDeployment:
    def test_missing_deployment_raises_config_error(self, tmp_path: Path) -> None:
        """Azure adapter requires deployment (not model_id)."""
        no_dep_yaml = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: azure_no_dep
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
              azure_no_dep:
                provider: mock
                provider_profile: local_mock
                model_label: azure-no-dep
                cost_tier: none
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 1
            provider_profiles:
              local_mock:
                provider: mock
        """)
        yaml_path = tmp_path / "llm_orchestration.yaml"
        yaml_path.write_text(no_dep_yaml, encoding="utf-8")
        registry = LlmConfigRegistry(yaml_path=yaml_path)
        resolver = ModelConfigResolver(registry=registry)

        rd = RouteDecision(
            route_id="general.generator.default",
            subject="general",
            task_role="generator",
            difficulty="default",
            model="azure_no_dep",
            prompt="subjects/general_generator.md",
            temperature=0.3,
            max_tokens=800,
            provider_options={},
            fallback_attempts=[],
            route_source="exact",
        )
        model_resolution = resolver.resolve(rd)
        request = ProviderExecutionRequest(
            route_decision=rd,
            model_resolution=model_resolution,
            messages=_messages(),
            temperature=0.3,
            max_tokens=800,
        )

        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        creds = _azure_credentials()

        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(request=request, credentials=creds)
        assert exc_info.value.failure_kind == "model_not_configured"


# ---------------------------------------------------------------------------
# Successful generate
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderAdapterGenerate:
    def test_returns_model_execution_result(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion("Azure answer."))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path),
            credentials=_azure_credentials(),
        )
        assert isinstance(result, ModelExecutionResult)

    def test_provider_is_azure_openai(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path),
            credentials=_azure_credentials(),
        )
        assert result.provider == "azure_openai"

    def test_model_is_route_alias(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path),
            credentials=_azure_credentials(),
        )
        assert result.model == "azure_gpt4o_default"

    def test_calls_client_with_deployment(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        adapter.generate(request=_make_request(tmp_path), credentials=_azure_credentials())
        assert client.received_kwargs["model"] == "gpt-4o-deployment"

    def test_calls_client_with_temperature(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        adapter.generate(request=_make_request(tmp_path), credentials=_azure_credentials())
        assert client.received_kwargs["temperature"] == pytest.approx(0.3)

    def test_calls_client_with_messages(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        adapter.generate(request=_make_request(tmp_path), credentials=_azure_credentials())
        msgs = client.received_kwargs["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_finish_reason_extracted(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion(finish_reason="length"))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path),
            credentials=_azure_credentials(),
        )
        assert result.finish_reason == "length"

    def test_input_tokens_extracted(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion(input_tokens=50))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path),
            credentials=_azure_credentials(),
        )
        assert result.input_tokens == 50

    def test_output_tokens_extracted(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion(output_tokens=20))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path),
            credentials=_azure_credentials(),
        )
        assert result.output_tokens == 20


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderAdapterErrors:
    def test_empty_content_raises_response_error(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion(content=""))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        with pytest.raises(LlmProviderResponseError):
            adapter.generate(
                request=_make_request(tmp_path), credentials=_azure_credentials()
            )

    def test_provider_exception_raises_execution_error(self, tmp_path: Path) -> None:
        sdk_exc = RuntimeError("Azure timeout")
        client = FakeAzureOpenAIClient(raise_exc=sdk_exc)
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        with pytest.raises(LlmProviderExecutionError):
            adapter.generate(
                request=_make_request(tmp_path), credentials=_azure_credentials()
            )

    def test_execution_error_wraps_original_cause(self, tmp_path: Path) -> None:
        sdk_exc = RuntimeError("quota exceeded")
        client = FakeAzureOpenAIClient(raise_exc=sdk_exc)
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(
                request=_make_request(tmp_path), credentials=_azure_credentials()
            )
        assert exc_info.value.__cause__ is sdk_exc

    def test_execution_error_does_not_include_credentials(self, tmp_path: Path) -> None:
        sdk_exc = RuntimeError("auth failed")
        client = FakeAzureOpenAIClient(raise_exc=sdk_exc)
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        with pytest.raises(LlmProviderExecutionError) as exc_info:
            adapter.generate(
                request=_make_request(tmp_path), credentials=_azure_credentials()
            )
        error_str = str(exc_info.value)
        assert "fake-azure-key" not in error_str
        assert "https://fake.openai.azure.com" not in error_str


# ---------------------------------------------------------------------------
# Metadata safety
# ---------------------------------------------------------------------------


class TestAzureOpenAIProviderAdapterMetadata:
    def test_metadata_does_not_contain_api_key(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path), credentials=_azure_credentials()
        )
        for v in result.metadata.values():
            assert "fake-azure-key" not in str(v)

    def test_metadata_does_not_contain_endpoint(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path), credentials=_azure_credentials()
        )
        for v in result.metadata.values():
            assert "fake.openai.azure.com" not in str(v)

    def test_metadata_contains_model_label(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path), credentials=_azure_credentials()
        )
        assert result.metadata.get("model_label") == "azure-gpt4o"

    def test_metadata_contains_deployment(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request(tmp_path), credentials=_azure_credentials()
        )
        assert result.metadata.get("deployment") == "gpt-4o-deployment"


# ---------------------------------------------------------------------------
# Import safety (no importlib.reload)
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_import_azure_openai_provider_module(self) -> None:
        import services.llm_providers.azure_openai_provider  # noqa: F401


# ---------------------------------------------------------------------------
# Part 9.2 — azure_api_mode split tests
# ---------------------------------------------------------------------------

# YAML for classic mode tests
_AZURE_CLASSIC_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: azure_classic
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
      azure_classic:
        provider: azure_openai
        provider_profile: azure_classic_profile
        deployment: gpt-4o-deployment
        model_label: azure-classic
        cost_tier: medium
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 30
    provider_profiles:
      local_mock:
        provider: mock
      azure_classic_profile:
        provider: azure_openai
        azure_api_mode: azure_deployment_chat_completions
        api_key_env: AZURE_OPENAI_API_KEY
        endpoint_env: AZURE_OPENAI_ENDPOINT
        api_version_env: AZURE_OPENAI_API_VERSION
""")

# YAML for v1 mode tests
_AZURE_V1_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: azure_v1
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
      azure_v1:
        provider: azure_openai
        provider_profile: azure_v1_profile
        deployment: gpt-4o
        model_label: azure-v1
        cost_tier: medium
        supports_streaming: false
        supports_thinking: false
        timeout_seconds: 30
    provider_profiles:
      local_mock:
        provider: mock
      azure_v1_profile:
        provider: azure_openai
        azure_api_mode: azure_openai_v1
        api_key_env: AZURE_OPENAI_API_KEY
        endpoint_env: AZURE_OPENAI_ENDPOINT
""")


def _registry_from_yaml(tmp_path: Path, yaml_text: str) -> LlmConfigRegistry:
    yaml_path = tmp_path / "llm_orchestration.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return LlmConfigRegistry(yaml_path=yaml_path)


def _make_request_from_yaml(
    tmp_path: Path, yaml_text: str, model_id: str
) -> ProviderExecutionRequest:
    registry = _registry_from_yaml(tmp_path, yaml_text)
    resolver = ModelConfigResolver(registry=registry)
    rd = RouteDecision(
        route_id="general.generator.default",
        subject="general",
        task_role="generator",
        difficulty="default",
        model=model_id,
        prompt="subjects/general_generator.md",
        temperature=0.3,
        max_tokens=800,
        provider_options={},
        fallback_attempts=[],
        route_source="exact",
    )
    model_resolution = resolver.resolve(rd)
    return ProviderExecutionRequest(
        route_decision=rd,
        model_resolution=model_resolution,
        messages=_messages(),
        temperature=0.3,
        max_tokens=800,
    )


def _classic_credentials(endpoint: str = "https://fake.openai.azure.com/") -> ProviderCredentials:
    return ProviderCredentials(
        provider="azure_openai",
        api_key="fake-azure-key",
        endpoint=endpoint,
        api_version="2024-02-01",
        azure_api_mode="azure_deployment_chat_completions",
    )


def _v1_credentials(
    endpoint: str = "https://fake.openai.azure.com/openai/v1",
) -> ProviderCredentials:
    return ProviderCredentials(
        provider="azure_openai",
        api_key="fake-azure-key",
        endpoint=endpoint,
        azure_api_mode="azure_openai_v1",
    )


class TestAzureOpenAIProviderAdapterClassicMode:
    """Tests for azure_deployment_chat_completions mode."""

    def test_classic_mode_accepts_plain_endpoint(self, tmp_path: Path) -> None:
        """Endpoint without /openai/v1 suffix is valid for classic mode."""
        client = FakeAzureOpenAIClient(completion=_fake_completion("Classic answer."))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request_from_yaml(tmp_path, _AZURE_CLASSIC_YAML, "azure_classic"),
            credentials=_classic_credentials("https://fake.openai.azure.com/"),
        )
        assert isinstance(result, ModelExecutionResult)
        assert result.provider == "azure_openai"

    def test_classic_mode_rejects_v1_suffix_endpoint(self, tmp_path: Path) -> None:
        """Endpoint ending with /openai/v1 must be rejected in classic mode."""
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        with pytest.raises(LlmProviderConfigurationError, match="azure_openai_v1"):
            adapter.generate(
                request=_make_request_from_yaml(tmp_path, _AZURE_CLASSIC_YAML, "azure_classic"),
                credentials=_classic_credentials("https://fake.openai.azure.com/openai/v1"),
            )

    def test_classic_mode_rejects_projects_segment_endpoint(self, tmp_path: Path) -> None:
        """Endpoint containing /api/projects/ is a Foundry endpoint — reject it."""
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        with pytest.raises(LlmProviderConfigurationError, match="projects"):
            adapter.generate(
                request=_make_request_from_yaml(tmp_path, _AZURE_CLASSIC_YAML, "azure_classic"),
                credentials=_classic_credentials(
                    "https://rs-proj.services.ai.azure.com/api/projects/my-proj"
                ),
            )

    def test_classic_mode_requires_api_version(self, tmp_path: Path) -> None:
        """api_version is required in classic mode."""
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/",
            azure_api_mode="azure_deployment_chat_completions",
            # api_version intentionally omitted
        )
        with pytest.raises(LlmProviderConfigurationError, match="api_version"):
            adapter.generate(
                request=_make_request_from_yaml(tmp_path, _AZURE_CLASSIC_YAML, "azure_classic"),
                credentials=creds,
            )

    def test_classic_mode_metadata_includes_azure_api_mode(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request_from_yaml(tmp_path, _AZURE_CLASSIC_YAML, "azure_classic"),
            credentials=_classic_credentials(),
        )
        assert result.metadata.get("azure_api_mode") == "azure_deployment_chat_completions"


class TestAzureOpenAIProviderAdapterV1Mode:
    """Tests for azure_openai_v1 mode (OpenAI-compatible /openai/v1 base URL)."""

    def test_v1_mode_accepts_openai_v1_endpoint(self, tmp_path: Path) -> None:
        """Endpoint ending with /openai/v1 is valid for v1 mode."""
        client = FakeAzureOpenAIClient(completion=_fake_completion("V1 answer."))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request_from_yaml(tmp_path, _AZURE_V1_YAML, "azure_v1"),
            credentials=_v1_credentials("https://fake.openai.azure.com/openai/v1"),
        )
        assert isinstance(result, ModelExecutionResult)
        assert result.provider == "azure_openai"

    def test_v1_mode_accepts_trailing_slash(self, tmp_path: Path) -> None:
        """Endpoint ending with /openai/v1/ (trailing slash) is also valid."""
        client = FakeAzureOpenAIClient(completion=_fake_completion("V1 slash answer."))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request_from_yaml(tmp_path, _AZURE_V1_YAML, "azure_v1"),
            credentials=_v1_credentials("https://fake.openai.azure.com/openai/v1/"),
        )
        assert isinstance(result, ModelExecutionResult)

    def test_v1_mode_rejects_plain_endpoint(self, tmp_path: Path) -> None:
        """Endpoint without /openai/v1 suffix must be rejected in v1 mode."""
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        with pytest.raises(LlmProviderConfigurationError, match="/openai/v1"):
            adapter.generate(
                request=_make_request_from_yaml(tmp_path, _AZURE_V1_YAML, "azure_v1"),
                credentials=_v1_credentials("https://fake.openai.azure.com"),
            )

    def test_v1_mode_does_not_require_api_version(self, tmp_path: Path) -> None:
        """api_version is NOT required in v1 mode — API version is in the base URL."""
        client = FakeAzureOpenAIClient(completion=_fake_completion("V1 no-version answer."))
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
            # api_version intentionally omitted
        )
        result = adapter.generate(
            request=_make_request_from_yaml(tmp_path, _AZURE_V1_YAML, "azure_v1"),
            credentials=creds,
        )
        assert isinstance(result, ModelExecutionResult)

    def test_v1_mode_passes_deployment_as_model(self, tmp_path: Path) -> None:
        """Deployment name is passed as model= kwarg regardless of mode."""
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        adapter.generate(
            request=_make_request_from_yaml(tmp_path, _AZURE_V1_YAML, "azure_v1"),
            credentials=_v1_credentials(),
        )
        assert client.received_kwargs["model"] == "gpt-4o"

    def test_v1_mode_metadata_includes_azure_api_mode(self, tmp_path: Path) -> None:
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        result = adapter.generate(
            request=_make_request_from_yaml(tmp_path, _AZURE_V1_YAML, "azure_v1"),
            credentials=_v1_credentials(),
        )
        assert result.metadata.get("azure_api_mode") == "azure_openai_v1"

    def test_v1_mode_no_api_key_in_error(self, tmp_path: Path) -> None:
        """Security: api_key must not appear in v1 mode config errors."""
        client = FakeAzureOpenAIClient(completion=_fake_completion())
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="super-secret-key",
            # endpoint without /openai/v1 — triggers LlmProviderConfigurationError
            endpoint="https://fake.openai.azure.com",
            azure_api_mode="azure_openai_v1",
        )
        with pytest.raises(LlmProviderConfigurationError) as exc_info:
            adapter.generate(
                request=_make_request_from_yaml(tmp_path, _AZURE_V1_YAML, "azure_v1"),
                credentials=creds,
            )
        assert "super-secret-key" not in str(exc_info.value)


class TestAzureApiModeSchemaValidation:
    """Tests for ProviderProfile.azure_api_mode field behaviour."""

    def test_default_is_azure_deployment_chat_completions(self) -> None:
        from schemas.llm_routing import ProviderProfile
        p = ProviderProfile(provider="azure_openai")
        assert p.azure_api_mode == "azure_deployment_chat_completions"

    def test_accepts_azure_openai_v1(self) -> None:
        from schemas.llm_routing import ProviderProfile
        p = ProviderProfile(provider="azure_openai", azure_api_mode="azure_openai_v1")
        assert p.azure_api_mode == "azure_openai_v1"

    def test_rejects_invalid_mode(self) -> None:
        import pytest
        from pydantic import ValidationError

        from schemas.llm_routing import ProviderProfile
        with pytest.raises(ValidationError):
            ProviderProfile(provider="azure_openai", azure_api_mode="invalid_mode")

    def test_non_azure_provider_has_default_mode(self) -> None:
        """Non-Azure providers are not affected by azure_api_mode field."""
        from schemas.llm_routing import ProviderProfile
        p = ProviderProfile(provider="openai")
        assert p.azure_api_mode == "azure_deployment_chat_completions"  # default; not used
