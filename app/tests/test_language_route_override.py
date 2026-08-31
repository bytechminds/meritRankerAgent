"""
app/tests/test_language_route_override.py
-----------------------------------------
The optional language dimension in route resolution.

Language override is a centralized capability: it lives entirely in the routes
config and the resolver, so no feature module ever names a model. These tests pin
both halves of the contract — that an exact configured override is honoured, and
that every other resolution stays byte-for-byte what it was before language became
a selection input.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from schemas.llm_routing import RouteRequest
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.errors import LlmConfigValidationError
from services.llm.orchestration.route_resolver import resolve_route

_MODELS_YAML = """
version: 1
models:
  base_generator:
    provider: mock
    provider_profile: local_mock
    model_id: local-mock
    timeout_seconds: 1
  hindi_generator:
    provider: mock
    provider_profile: local_mock
    model_id: local-mock
    timeout_seconds: 1
  english_generator:
    provider: mock
    provider_profile: local_mock
    model_id: local-mock
    timeout_seconds: 1
  safe_mock:
    provider: mock
    provider_profile: local_mock
    model_id: local-mock
    timeout_seconds: 1
"""

_PROFILES_YAML = """
version: 1
provider_profiles:
  local_mock:
    provider: mock
"""

_ROUTES_BASE = """
version: 1
routes:
  reasoning:
    generator:
      default:
        model: base_generator
        prompt: subjects/reasoning_generator.md
        temperature: 0.2
        max_tokens: 900
      basic:
        inherits: default
      advanced:
        inherits: default
  general:
    generator:
      default:
        model: base_generator
        prompt: subjects/general_generator.md
        temperature: 0.2
        max_tokens: 900
"""

_OVERRIDES = """
language_model_overrides:
  reasoning:
    generator:
      basic:
        hindi: hindi_generator
        english: english_generator
"""


def _registry(tmp_path: Path, routes_yaml: str) -> LlmConfigRegistry:
    routes = tmp_path / "llm_routes.yaml"
    models = tmp_path / "model_registry.yaml"
    profiles = tmp_path / "provider_profiles.yaml"
    routes.write_text(textwrap.dedent(routes_yaml), encoding="utf-8")
    models.write_text(textwrap.dedent(_MODELS_YAML), encoding="utf-8")
    profiles.write_text(textwrap.dedent(_PROFILES_YAML), encoding="utf-8")
    return LlmConfigRegistry(
        routes_path=routes,
        model_registry_path=models,
        provider_profiles_path=profiles,
    )


def _request(
    *,
    subject: str = "reasoning",
    difficulty: str = "basic",
    language: str = "english",
) -> RouteRequest:
    return RouteRequest(
        request_id="lang-001",
        subject=subject,
        task_role="generator",
        difficulty=difficulty,
        language=language,
    )


class TestDefaultBehaviourIsUnchanged:
    """Without configured overrides, language must not influence anything."""

    @pytest.mark.parametrize("language", ["english", "hindi", "hinglish"])
    def test_no_override_block_leaves_route_resolution_untouched(
        self, tmp_path: Path, language: str
    ) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE)
        decision = resolve_route(_request(language=language), reg)

        assert decision.model == "base_generator"
        assert decision.route_id == "reasoning.generator.basic"
        assert decision.route_source == "exact"

    def test_all_languages_resolve_identically_without_overrides(
        self, tmp_path: Path
    ) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE)
        models = {
            resolve_route(_request(language=lang), reg).model
            for lang in ("english", "hindi", "hinglish")
        }
        assert models == {"base_generator"}

    def test_production_config_declares_no_language_overrides(self) -> None:
        """Qualification must not silently activate a real production override."""
        assert LlmConfigRegistry()._language_override_map == {}


class TestExactOverrideIsHonoured:
    def test_hindi_override_selects_the_configured_model(self, tmp_path: Path) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE + _OVERRIDES)
        decision = resolve_route(_request(language="hindi"), reg)

        assert decision.model == "hindi_generator"
        assert decision.language == "hindi"

    def test_english_override_selects_the_configured_model(
        self, tmp_path: Path
    ) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE + _OVERRIDES)
        assert resolve_route(_request(language="english"), reg).model == (
            "english_generator"
        )

    def test_hinglish_without_an_entry_keeps_the_route_model(
        self, tmp_path: Path
    ) -> None:
        """A configured route with no entry for this language is not an override."""
        reg = _registry(tmp_path, _ROUTES_BASE + _OVERRIDES)
        assert resolve_route(_request(language="hinglish"), reg).model == (
            "base_generator"
        )

    def test_override_changes_only_the_model(self, tmp_path: Path) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE + _OVERRIDES)
        base = resolve_route(_request(language="hinglish"), reg)
        overridden = resolve_route(_request(language="hindi"), reg)

        assert overridden.model != base.model
        assert overridden.prompt == base.prompt
        assert overridden.temperature == base.temperature
        assert overridden.max_tokens == base.max_tokens
        assert overridden.overlays == base.overlays
        assert overridden.provider_options == base.provider_options
        assert overridden.fallback_attempts == base.fallback_attempts
        assert overridden.route_id == base.route_id
        assert overridden.route_source == base.route_source


class TestOverrideIsScopedExactly:
    def test_other_difficulty_is_not_affected(self, tmp_path: Path) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE + _OVERRIDES)
        decision = resolve_route(
            _request(difficulty="advanced", language="hindi"), reg
        )
        assert decision.model == "base_generator"
        assert decision.difficulty == "advanced"

    def test_other_subject_is_not_affected(self, tmp_path: Path) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE + _OVERRIDES)
        decision = resolve_route(
            _request(subject="english", language="hindi"), reg
        )
        assert decision.model == "base_generator"
        assert decision.subject == "general"

    def test_other_task_role_is_not_affected(self, tmp_path: Path) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE + _OVERRIDES)
        assert reg.get_language_model_override(
            "reasoning", "verifier", "basic", "hindi"
        ) is None

    def test_fallback_resolved_route_keeps_its_own_key(self, tmp_path: Path) -> None:
        """An override on `basic` must not leak onto a route reached by fallback."""
        reg = _registry(tmp_path, _ROUTES_BASE + _OVERRIDES)
        decision = resolve_route(
            _request(subject="math", difficulty="basic", language="hindi"), reg
        )
        assert decision.route_source == "general_default"
        assert decision.model == "base_generator"


class TestLanguageInputSafety:
    def test_missing_language_uses_the_canonical_default(
        self, tmp_path: Path
    ) -> None:
        reg = _registry(tmp_path, _ROUTES_BASE)
        request = RouteRequest(
            request_id="lang-002",
            subject="reasoning",
            task_role="generator",
            difficulty="basic",
        )
        assert request.language == "english"
        assert resolve_route(request, reg).model == "base_generator"

    def test_unknown_language_is_rejected_at_the_boundary(self) -> None:
        """The resolver never sees an uncanonical language, so it never guesses."""
        with pytest.raises(ValueError):
            RouteRequest(
                request_id="lang-003",
                subject="reasoning",
                task_role="generator",
                difficulty="basic",
                language="klingon",
            )


class TestOverrideConfigFailsClosed:
    def test_override_naming_an_unknown_model_is_rejected(
        self, tmp_path: Path
    ) -> None:
        bad = _ROUTES_BASE + """
language_model_overrides:
  reasoning:
    generator:
      basic:
        hindi: model_that_does_not_exist
"""
        with pytest.raises(LlmConfigValidationError, match="does not exist"):
            _registry(tmp_path, bad)

    def test_override_naming_an_unknown_route_is_rejected(
        self, tmp_path: Path
    ) -> None:
        bad = _ROUTES_BASE + """
language_model_overrides:
  reasoning:
    verifier:
      basic:
        hindi: hindi_generator
"""
        with pytest.raises(LlmConfigValidationError, match="route that does not exist"):
            _registry(tmp_path, bad)
