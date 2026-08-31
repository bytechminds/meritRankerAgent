"""
app/services/llm_providers/provider_factory.py
------------------------------------------------
ProviderAdapterFactory — maps provider names to ProviderAdapter instances.

Design:
- Factory is constructed with a default adapter map that covers mock, openai,
  and azure_openai providers.
- Tests can inject a custom adapter_map to avoid any real SDK dependency.
- No secret fetching occurs inside this module.
- No env reads occur inside this module.
- No SDK client is created at import time.

Supported providers (Part 6):
    mock         → MockProviderAdapter
    openai       → OpenAIProviderAdapter
    azure_openai → AzureOpenAIProviderAdapter
    gemini       → GeminiProviderAdapter
    deepseek     → DeepSeekProviderAdapter
    bedrock      → BedrockProviderAdapter

Unsupported provider raises LlmProviderConfigurationError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.llm.providers.errors import LlmProviderConfigurationError

if TYPE_CHECKING:
    from services.llm.providers.base import ProviderAdapter


class ProviderAdapterFactory:
    """Map provider names to ProviderAdapter instances.

    Usage (production):
        factory = ProviderAdapterFactory()

    Usage (tests — inject custom adapters):
        factory = ProviderAdapterFactory(adapter_map={"mock": my_mock_adapter})

    The default adapter_map is constructed lazily on first access so that
    importing this module never creates SDK clients.
    """

    def __init__(
        self,
        *,
        adapter_map: dict[str, ProviderAdapter] | None = None,
    ) -> None:
        """
        Args:
            adapter_map: Optional mapping of provider name → adapter instance.
                When provided, the factory uses it exactly (no defaults added).
                When None, the factory builds a default map from the standard adapters.
        """
        if adapter_map is not None:
            self._adapters: dict[str, ProviderAdapter] = dict(adapter_map)
        else:
            # Deferred imports — no SDK clients at module/import time
            from services.llm.providers.azure_openai_provider import (  # noqa: PLC0415
                AzureOpenAIProviderAdapter,
            )
            from services.llm.providers.bedrock_provider import (  # noqa: PLC0415
                BedrockProviderAdapter,
            )
            from services.llm.providers.gemini_provider import (  # noqa: PLC0415
                GeminiProviderAdapter,
            )
            from services.llm.providers.mock_provider import MockProviderAdapter  # noqa: PLC0415
            from services.llm.providers.openai_compatible_adapter import (  # noqa: PLC0415
                DeepSeekProviderAdapter,
            )
            from services.llm.providers.openai_provider import (  # noqa: PLC0415
                OpenAIProviderAdapter,
            )

            self._adapters = {
                "mock": MockProviderAdapter(),
                "openai": OpenAIProviderAdapter(),
                "azure_openai": AzureOpenAIProviderAdapter(),
                "gemini": GeminiProviderAdapter(),
                "deepseek": DeepSeekProviderAdapter(),
                "bedrock": BedrockProviderAdapter(),
            }

    def get_provider(self, provider: str) -> ProviderAdapter:
        """Return the adapter registered for *provider*.

        Args:
            provider: Provider name string (e.g. "mock", "openai", "azure_openai").

        Returns:
            The registered ProviderAdapter instance.

        Raises:
            LlmProviderConfigurationError: If the provider is not registered.
        """
        adapter = self._adapters.get(provider)
        if adapter is None:
            supported = sorted(self._adapters.keys())
            raise LlmProviderConfigurationError(
                f"Unsupported provider {provider!r}. "
                f"Supported providers: {supported}."
            )
        return adapter
