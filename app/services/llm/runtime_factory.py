"""Central real/mock model-executor bootstrap shared by runtime workflows."""

from __future__ import annotations

from config import ConfigurationError, Settings
from services.llm.orchestration.config_registry import get_registry
from services.llm.orchestration.errors import LlmConfigValidationError
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.orchestration.model_execution import (
    ProviderAdapterExecutor,
    RegistryBackedModelExecutor,
)
from services.llm.orchestration.orchestrator import MockModelExecutor, ModelExecutor
from services.llm.providers.provider_factory import ProviderAdapterFactory
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.provider_credentials import ProviderCredentialResolver
from services.secrets.secret_resolver import SecretResolver


def build_model_executor(
    settings: Settings,
    *,
    mock_content: str = "Mock response. <ANSWER_DONE>",
    secret_resolver: SecretResolver | None = None,
) -> ModelExecutor:
    if not settings.enable_real_llm:
        return MockModelExecutor(content=mock_content)
    try:
        get_registry().validate_real_mode_deployments()
    except LlmConfigValidationError as exc:
        raise ConfigurationError(str(exc)) from exc
    credential_resolver = ProviderCredentialResolver(
        secret_resolver=secret_resolver or EnvSecretResolver()
    )
    model_config_resolver = ModelConfigResolver()
    provider_executor = ProviderAdapterExecutor(
        credential_resolver=credential_resolver,
        provider_factory=ProviderAdapterFactory(),
    )
    return RegistryBackedModelExecutor(
        provider_executor=provider_executor,
        model_config_resolver=model_config_resolver,
    )
