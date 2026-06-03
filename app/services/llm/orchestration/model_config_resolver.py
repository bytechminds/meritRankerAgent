"""
app/services/llm_orchestration/model_config_resolver.py
-------------------------------------------------------
Resolve model/provider metadata from the compiled LLM config registry.
"""

from __future__ import annotations

from schemas.llm_orchestration import ResolvedModelConfig
from schemas.llm_routing import ModelConfig, ProviderProfile, RouteDecision
from services.llm.orchestration.config_registry import LlmConfigRegistry, get_registry
from services.llm.orchestration.errors import (
    ModelConfigResolutionError,
    ModelExecutionConfigError,
)

_SUPPORTED_PROVIDER_OPTIONS: frozenset[str] = frozenset({"thinking", "stream"})


class ModelConfigResolver:
    """Resolve route model aliases to provider metadata without I/O per request."""

    def __init__(self, registry: LlmConfigRegistry | None = None) -> None:
        self._registry = registry if registry is not None else get_registry()

    def resolve(self, route_decision: RouteDecision) -> ResolvedModelConfig:
        """Resolve model and provider profile metadata for a route decision."""
        model_alias = route_decision.model
        model_config = self._resolve_model_config(model_alias)
        provider_profile_name = model_config.provider_profile
        provider_profile = self._resolve_provider_profile(
            model_alias=model_alias,
            provider_profile_name=provider_profile_name,
        )
        self._validate_provider_consistency(
            model_alias=model_alias,
            model_config=model_config,
            provider_profile=provider_profile,
            provider_profile_name=provider_profile_name,
        )
        self.validate_provider_options(
            provider_options=route_decision.provider_options,
            model_config=model_config,
            model_alias=model_alias,
        )

        return ResolvedModelConfig(
            model_alias=model_alias,
            model_config=model_config,
            provider_profile_name=provider_profile_name,
            provider_profile=provider_profile,
            provider=model_config.provider,
            supports_streaming=model_config.supports_streaming,
            supports_thinking=model_config.supports_thinking,
            supports_reasoning=model_config.supports_reasoning,
            reasoning_effort=model_config.reasoning_effort,
            timeout_seconds=model_config.timeout_seconds,
        )

    def validate_provider_options(
        self,
        *,
        provider_options: dict[str, object],
        model_config: ModelConfig,
        model_alias: str,
    ) -> None:
        """Validate route provider options against the resolved model metadata."""
        for option_name in provider_options:
            if option_name not in _SUPPORTED_PROVIDER_OPTIONS:
                raise ModelExecutionConfigError(
                    f"Unsupported provider option '{option_name}' for model '{model_alias}'."
                )

        thinking = provider_options.get("thinking")
        if thinking is not None and not isinstance(thinking, bool):
            raise ModelExecutionConfigError(
                f"Provider option 'thinking' must be a boolean for model '{model_alias}'."
            )
        supports_reasoning_mode = (
            model_config.supports_thinking or model_config.supports_reasoning
        )
        if thinking is True and not supports_reasoning_mode:
            raise ModelExecutionConfigError(
                f"Model '{model_alias}' does not support provider option 'thinking'."
            )

        stream = provider_options.get("stream")
        if stream is not None and not isinstance(stream, bool):
            raise ModelExecutionConfigError(
                f"Provider option 'stream' must be a boolean for model '{model_alias}'."
            )

    def _resolve_model_config(self, model_alias: str) -> ModelConfig:
        model_config = self._registry.get_model(model_alias)
        if model_config is None:
            raise ModelConfigResolutionError(
                f"Unknown model alias '{model_alias}'."
            )
        return model_config

    def _resolve_provider_profile(
        self,
        *,
        model_alias: str,
        provider_profile_name: str,
    ) -> ProviderProfile:
        provider_profile = self._registry.get_provider_profile(provider_profile_name)
        if provider_profile is None:
            raise ModelConfigResolutionError(
                f"Model '{model_alias}' references missing provider profile "
                f"'{provider_profile_name}'."
            )
        return provider_profile

    def _validate_provider_consistency(
        self,
        *,
        model_alias: str,
        model_config: ModelConfig,
        provider_profile: ProviderProfile,
        provider_profile_name: str,
    ) -> None:
        if model_config.provider != provider_profile.provider:
            raise ModelConfigResolutionError(
                f"Model '{model_alias}' provider '{model_config.provider}' does not match "
                f"provider profile '{provider_profile_name}' provider "
                f"'{provider_profile.provider}'."
            )

    def resolve_for_alias(self, alias: str) -> ResolvedModelConfig:
        """Resolve model/provider metadata for a specific alias (e.g. a fallback alias).

        Unlike ``resolve(route_decision)``, this method does NOT validate
        provider_options — the fallback model may have a different capability
        set than the primary.  The caller is responsible for ensuring the
        fallback alias was validated at registry build time.

        Args:
            alias: Internal model alias from the catalog.

        Returns:
            ResolvedModelConfig for the given alias.

        Raises:
            ModelConfigResolutionError: alias not found or profile mismatch.
        """
        model_config = self._resolve_model_config(alias)
        provider_profile_name = model_config.provider_profile
        provider_profile = self._resolve_provider_profile(
            model_alias=alias,
            provider_profile_name=provider_profile_name,
        )
        self._validate_provider_consistency(
            model_alias=alias,
            model_config=model_config,
            provider_profile=provider_profile,
            provider_profile_name=provider_profile_name,
        )
        return ResolvedModelConfig(
            model_alias=alias,
            model_config=model_config,
            provider_profile_name=provider_profile_name,
            provider_profile=provider_profile,
            provider=model_config.provider,
            supports_streaming=model_config.supports_streaming,
            supports_thinking=model_config.supports_thinking,
            supports_reasoning=model_config.supports_reasoning,
            reasoning_effort=model_config.reasoning_effort,
            timeout_seconds=model_config.timeout_seconds,
        )
