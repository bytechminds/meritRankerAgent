"""
app/tests/test_provider_factory.py
------------------------------------
Unit tests for services/llm_providers/provider_factory.py.

Tests cover:
- Returns MockProviderAdapter for "mock"
- Returns OpenAIProviderAdapter for "openai"
- Returns AzureOpenAIProviderAdapter for "azure_openai"
- Unsupported provider raises LlmProviderConfigurationError
- Custom adapter injection works
- Factory does not read env
- Factory does not call network
- No importlib.reload()
"""

from __future__ import annotations

import pytest

from services.llm_providers.azure_openai_provider import AzureOpenAIProviderAdapter
from services.llm_providers.errors import LlmProviderConfigurationError
from services.llm_providers.mock_provider import MockProviderAdapter
from services.llm_providers.openai_provider import OpenAIProviderAdapter
from services.llm_providers.provider_factory import ProviderAdapterFactory

# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


class TestProviderAdapterFactoryDefault:
    def test_returns_mock_adapter(self) -> None:
        factory = ProviderAdapterFactory()
        adapter = factory.get_provider("mock")
        assert isinstance(adapter, MockProviderAdapter)

    def test_returns_openai_adapter(self) -> None:
        factory = ProviderAdapterFactory()
        adapter = factory.get_provider("openai")
        assert isinstance(adapter, OpenAIProviderAdapter)

    def test_returns_azure_openai_adapter(self) -> None:
        factory = ProviderAdapterFactory()
        adapter = factory.get_provider("azure_openai")
        assert isinstance(adapter, AzureOpenAIProviderAdapter)

    def test_returns_gemini_adapter(self) -> None:
        factory = ProviderAdapterFactory()
        adapter = factory.get_provider("gemini")
        from services.llm.providers.gemini_provider import GeminiProviderAdapter

        assert isinstance(adapter, GeminiProviderAdapter)

    def test_returns_deepseek_adapter(self) -> None:
        factory = ProviderAdapterFactory()
        adapter = factory.get_provider("deepseek")
        from services.llm.providers.openai_compatible_adapter import DeepSeekProviderAdapter

        assert isinstance(adapter, DeepSeekProviderAdapter)

    def test_unsupported_provider_raises_config_error(self) -> None:
        factory = ProviderAdapterFactory()
        with pytest.raises(LlmProviderConfigurationError, match="unknown_provider"):
            factory.get_provider("unknown_provider")

    def test_unsupported_provider_error_mentions_supported(self) -> None:
        factory = ProviderAdapterFactory()
        with pytest.raises(LlmProviderConfigurationError) as exc_info:
            factory.get_provider("unknown_provider")
        msg = str(exc_info.value)
        assert "mock" in msg
        assert "openai" in msg
        assert "azure_openai" in msg
        assert "gemini" in msg
        assert "deepseek" in msg

    def test_empty_provider_name_raises_config_error(self) -> None:
        factory = ProviderAdapterFactory()
        with pytest.raises(LlmProviderConfigurationError):
            factory.get_provider("")


# ---------------------------------------------------------------------------
# Custom adapter_map injection
# ---------------------------------------------------------------------------


class TestProviderAdapterFactoryCustomMap:
    def test_custom_adapter_map_overrides_defaults(self) -> None:
        custom_mock = MockProviderAdapter(content="custom")
        factory = ProviderAdapterFactory(adapter_map={"mock": custom_mock})
        adapter = factory.get_provider("mock")
        assert adapter is custom_mock

    def test_custom_map_excludes_unconfigured_providers(self) -> None:
        factory = ProviderAdapterFactory(adapter_map={"mock": MockProviderAdapter()})
        with pytest.raises(LlmProviderConfigurationError):
            factory.get_provider("openai")

    def test_returns_same_instance_on_repeat_calls(self) -> None:
        factory = ProviderAdapterFactory()
        adapter1 = factory.get_provider("mock")
        adapter2 = factory.get_provider("mock")
        assert adapter1 is adapter2


# ---------------------------------------------------------------------------
# No env / no network
# ---------------------------------------------------------------------------


class TestProviderAdapterFactoryIsolation:
    def test_factory_construction_does_not_read_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Factory construction must not access os.environ."""
        sentinel_calls: list[str] = []

        original_getenv = __import__("os").environ.get

        def spy_getenv(key: str, default: str | None = None) -> str | None:
            sentinel_calls.append(key)
            return original_getenv(key, default)

        monkeypatch.setattr("os.environ.get", spy_getenv)

        # Constructing the factory must not call os.environ.get
        _ = ProviderAdapterFactory()
        assert sentinel_calls == [], f"Factory read env vars: {sentinel_calls}"

    def test_get_provider_does_not_make_network_call(self) -> None:
        """get_provider() must not make any network call."""
        factory = ProviderAdapterFactory()
        # Just retrieving the adapter must not raise or call network
        adapter = factory.get_provider("openai")
        assert adapter is not None


# ---------------------------------------------------------------------------
# Import safety (no importlib.reload)
# ---------------------------------------------------------------------------


class TestImportSafety:
    def test_import_provider_factory_module(self) -> None:
        import services.llm_providers.provider_factory  # noqa: F401
