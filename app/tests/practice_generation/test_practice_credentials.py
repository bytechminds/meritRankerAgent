"""Focused shared credential tests for practice model routes."""

from __future__ import annotations

import pytest

import config as config_module
from config import get_settings
from features.practice_generation.planning import resolve_practice_request
from features.practice_generation.providers import (
    RoutedPlannerProvider,
    RoutedQuestionGenerator,
    RoutedQuestionVerifier,
)
from services.llm.orchestration.errors import ProviderExecutionError
from services.llm.orchestration.orchestrator import LlmOrchestrator
from services.llm.runtime_factory import build_model_executor
from services.secrets.errors import SecretNotFoundError


class MappingResolver:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.requested: list[str] = []

    def get_secret(self, name: str) -> str:
        self.requested.append(name)
        value = self.values.get(name)
        if value is None:
            raise SecretNotFoundError(f"missing: {name}")
        return value


def test_real_runtime_does_not_eagerly_request_optional_route_credentials(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_REAL_LLM", "true")
    config_module._settings = None
    resolver = MappingResolver({})

    executor = build_model_executor(get_settings(), secret_resolver=resolver)

    assert executor.__class__.__name__ == "RegistryBackedModelExecutor"
    assert resolver.requested == []


def test_selected_practice_route_resolves_credentials_lazily_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_REAL_LLM", "true")
    config_module._settings = None
    resolver = MappingResolver({})
    executor = build_model_executor(get_settings(), secret_resolver=resolver)
    provider = RoutedPlannerProvider(LlmOrchestrator(model_executor=executor))
    request = resolve_practice_request(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        query="Create five algebra questions",
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        exam_stage=None,
    )

    with pytest.raises(ProviderExecutionError):
        provider.plan(request, tier="light")

    assert "AZURE_OPENAI_API_KEY" in resolver.requested


def test_practice_provider_composition_uses_shared_runtime_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_REAL_LLM", "true")
    config_module._settings = None
    executor = build_model_executor(
        get_settings(),
        secret_resolver=MappingResolver(
            {
                "AZURE_OPENAI_API_KEY": "azure-key",
                "AZURE_OPENAI_ENDPOINT": "https://example.invalid/openai/v1",
                "DEEPSEEK_API_KEY": "deepseek-key",
            }
        ),
    )
    orchestrator = LlmOrchestrator(model_executor=executor)

    providers = (
        RoutedPlannerProvider(orchestrator),
        RoutedQuestionGenerator(orchestrator),
        RoutedQuestionVerifier(orchestrator),
    )

    assert all(provider._orchestrator is orchestrator for provider in providers)
