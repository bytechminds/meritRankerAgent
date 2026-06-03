"""
Tests for ModelConfigResolver (LLM Orchestration Foundation — Part 4).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Any

import pytest

from schemas.llm_routing import ProviderProfile, RouteDecision
from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.errors import (
    ModelConfigResolutionError,
    ModelExecutionConfigError,
)
from services.llm_orchestration.model_config_resolver import ModelConfigResolver

_TEST_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: safe_mock
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 800
            fallback:
              - safe_mock
    models:
      gemini_flash_light:
        provider: gemini
        provider_profile: gemini_primary
        model_id: gemini-2.5-flash
        model_label: gemini-flash-light
        cost_tier: low
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 20
        capabilities:
          general: medium
      gemini_flash_reasoning_light:
        provider: gemini
        provider_profile: gemini_primary
        model_id: gemini-2.5-flash
        model_label: gemini-flash-reasoning-light
        cost_tier: low
        supports_streaming: true
        supports_thinking: true
        timeout_seconds: 25
        capabilities:
          general: high
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
        capabilities:
          general: low
    provider_profiles:
      gemini_primary:
        provider: gemini
        api_key_env: GEMINI_API_KEY
      local_mock:
        provider: mock
""")

_UNSAFE_METADATA_KEYS = {
    "api_key_env",
    "endpoint_env",
    "api_version_env",
    "base_url_env",
    "credential_ref",
    "prompt",
    "query",
    "context",
    "messages",
    "secret",
    "api_key",
}


def _registry(tmp_path: Path) -> LlmConfigRegistry:
    yaml_path = tmp_path / "llm_orchestration.yaml"
    yaml_path.write_text(_TEST_YAML, encoding="utf-8")
    return LlmConfigRegistry(yaml_path=yaml_path)


def _route_decision(
    *,
    model: str = "gemini_flash_light",
    provider_options: dict[str, Any] | None = None,
) -> RouteDecision:
    return RouteDecision(
        route_id="general.generator.default",
        subject="general",
        task_role="generator",
        difficulty="default",
        model=model,
        prompt="subjects/general_generator.md",
        temperature=0.3,
        max_tokens=800,
        provider_options=provider_options or {},
        route_source="exact",
    )


def test_resolves_route_model_alias_to_model_config(tmp_path: Path) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    resolved = resolver.resolve(_route_decision(model="gemini_flash_light"))

    assert resolved.model_alias == "gemini_flash_light"
    assert resolved.model_config.model_label == "gemini-flash-light"


def test_resolves_model_provider_profile(tmp_path: Path) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    resolved = resolver.resolve(_route_decision(model="gemini_flash_light"))

    assert resolved.provider_profile_name == "gemini_primary"
    assert resolved.provider_profile.provider == "gemini"


def test_resolved_config_copies_runtime_capabilities(tmp_path: Path) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    resolved = resolver.resolve(_route_decision(model="gemini_flash_reasoning_light"))

    assert resolved.provider == "gemini"
    assert resolved.supports_streaming is True
    assert resolved.supports_thinking is True
    assert resolved.timeout_seconds == 25


def test_safe_metadata_contains_only_allowed_keys(tmp_path: Path) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    resolved = resolver.resolve(_route_decision(model="gemini_flash_light"))

    assert resolved.safe_metadata == {
        "model_alias": "gemini_flash_light",
        "provider": "gemini",
        "supports_streaming": True,
        "supports_thinking": False,
        "supports_reasoning": False,
        "reasoning_effort": "none",
        "timeout_seconds": 20,
    }
    assert not _UNSAFE_METADATA_KEYS.intersection(resolved.safe_metadata)


def test_unknown_model_alias_raises_resolution_error(tmp_path: Path) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    with pytest.raises(ModelConfigResolutionError, match="Unknown model alias"):
        resolver.resolve(_route_decision(model="missing_model"))


def test_missing_provider_profile_raises_resolution_error(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.provider_profile_map.pop("gemini_primary")
    resolver = ModelConfigResolver(registry=registry)

    with pytest.raises(ModelConfigResolutionError, match="missing provider profile"):
        resolver.resolve(_route_decision(model="gemini_flash_light"))


def test_provider_profile_mismatch_raises_resolution_error(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.provider_profile_map["gemini_primary"] = ProviderProfile(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
    )
    resolver = ModelConfigResolver(registry=registry)

    with pytest.raises(ModelConfigResolutionError, match="does not match"):
        resolver.resolve(_route_decision(model="gemini_flash_light"))


def test_thinking_true_allowed_when_model_supports_thinking(tmp_path: Path) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    resolved = resolver.resolve(
        _route_decision(
            model="gemini_flash_reasoning_light",
            provider_options={"thinking": True},
        )
    )

    assert resolved.supports_thinking is True


def test_thinking_true_rejected_when_model_does_not_support_thinking(
    tmp_path: Path,
) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    with pytest.raises(ModelExecutionConfigError, match="does not support"):
        resolver.resolve(
            _route_decision(
                model="gemini_flash_light",
                provider_options={"thinking": True},
            )
        )


def test_unsupported_provider_option_raises_config_error(tmp_path: Path) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    with pytest.raises(ModelExecutionConfigError, match="Unsupported provider option"):
        resolver.resolve(
            _route_decision(
                model="gemini_flash_light",
                provider_options={"unknown_option": True},
            )
        )


def test_provider_option_values_must_be_boolean(tmp_path: Path) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    with pytest.raises(ModelExecutionConfigError, match="must be a boolean"):
        resolver.resolve(
            _route_decision(
                model="gemini_flash_light",
                provider_options={"thinking": "true"},
            )
        )


def test_stream_option_boolean_is_accepted_without_execution_logic(
    tmp_path: Path,
) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    resolved = resolver.resolve(
        _route_decision(
            model="gemini_flash_light",
            provider_options={"stream": True},
        )
    )

    assert resolved.supports_streaming is True


def test_resolver_does_not_read_environment_variables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = ModelConfigResolver(registry=_registry(tmp_path))

    def fail_getenv(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("os.getenv must not be called by ModelConfigResolver")

    monkeypatch.setattr(os, "getenv", fail_getenv)

    resolved = resolver.resolve(_route_decision(model="gemini_flash_light"))

    assert resolved.provider_profile.api_key_env == "GEMINI_API_KEY"


def test_resolver_does_not_parse_yaml_per_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)
    resolver = ModelConfigResolver(registry=registry)

    def fail_open(self: Path, *_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"YAML should not be opened per request: {self}")

    monkeypatch.setattr(Path, "open", fail_open)

    resolved = resolver.resolve(_route_decision(model="gemini_flash_light"))

    assert resolved.model_alias == "gemini_flash_light"
