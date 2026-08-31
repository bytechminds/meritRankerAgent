"""
app/tests/test_llm_config_split.py
-------------------------------------
Tests for Part 7.1: Split Simple Model Registry from Route Config.

Verifies:
1.  Split config files exist at the expected production paths.
2.  LlmConfigRegistry() (no args) loads from split files by default.
3.  config_version returns 1.
4.  route_map is populated from llm_routes.yaml.
5.  model_map is populated from model_registry.yaml.
6.  provider_profile_map is populated from provider_profiles.yaml.
7.  New capability-readable aliases are present in model_map.
8.  safe_mock exists in model_map with provider=mock.
9.  math.generator.default route references math_basic_generator.
10. math.generator.advanced route references math_reasoning_generator.
11. general.generator.default route references general_fast_generator.
12. Route entry with model_id field is rejected.
13. Route entry with provider field is rejected.
14. Route entry with deployment field is rejected.
15. Route entry with provider_profile field is rejected.
16. Route entry with api_key_env field is rejected.
17. Route entry with endpoint_env field is rejected.
18. Route entry with credential_ref field is rejected.
19. ModelConfig works without model_label (field is optional).
20. ModelConfig works without cost_tier (field is optional).
21. ModelConfig with description field is accepted.
22. Unknown model alias in routes raises LlmConfigValidationError.
23. Unknown provider_profile in model registry raises LlmConfigValidationError.
24. Provider mismatch between model and provider profile raises.
25. LlmConfigRegistry(yaml_path=...) still loads old combined format (backward compat).
26. dry-run path works: production registry has safe_mock with provider=mock.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from services.llm_orchestration.config_registry import (
    DEFAULT_MODEL_REGISTRY_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    DEFAULT_ROUTES_PATH,
    LlmConfigRegistry,
)
from services.llm_orchestration.errors import (
    LlmConfigValidationError,
)

# ---------------------------------------------------------------------------
# Shared helpers — write minimal split YAML fixtures to tmp_path
# ---------------------------------------------------------------------------

_MINIMAL_ROUTES_YAML = textwrap.dedent("""\
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
""")

_MINIMAL_MODELS_YAML = textwrap.dedent("""\
    version: 1
    models:
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
""")

_MINIMAL_PROFILES_YAML = textwrap.dedent("""\
    version: 1
    provider_profiles:
      local_mock:
        provider: mock
""")

_MINIMAL_COMBINED_YAML = textwrap.dedent("""\
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
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
        capabilities: {}
    provider_profiles:
      local_mock:
        provider: mock
""")


def _write_split_yaml(
    tmp_path: Path,
    *,
    routes_content: str = _MINIMAL_ROUTES_YAML,
    models_content: str = _MINIMAL_MODELS_YAML,
    profiles_content: str = _MINIMAL_PROFILES_YAML,
) -> tuple[Path, Path, Path]:
    routes_path = tmp_path / "llm_routes.yaml"
    models_path = tmp_path / "model_registry.yaml"
    profiles_path = tmp_path / "provider_profiles.yaml"
    routes_path.write_text(routes_content, encoding="utf-8")
    models_path.write_text(models_content, encoding="utf-8")
    profiles_path.write_text(profiles_content, encoding="utf-8")
    return routes_path, models_path, profiles_path


def _make_split_registry(
    tmp_path: Path,
    *,
    routes_content: str = _MINIMAL_ROUTES_YAML,
    models_content: str = _MINIMAL_MODELS_YAML,
    profiles_content: str = _MINIMAL_PROFILES_YAML,
) -> LlmConfigRegistry:
    rp, mp, pp = _write_split_yaml(
        tmp_path,
        routes_content=routes_content,
        models_content=models_content,
        profiles_content=profiles_content,
    )
    return LlmConfigRegistry(routes_path=rp, model_registry_path=mp, provider_profiles_path=pp)


# ---------------------------------------------------------------------------
# Test 1 — split files exist at production paths
# ---------------------------------------------------------------------------


class TestSplitFilesExist:
    def test_llm_routes_yaml_exists(self) -> None:
        assert DEFAULT_ROUTES_PATH.exists(), (
            f"Expected llm_routes.yaml at {DEFAULT_ROUTES_PATH}"
        )

    def test_model_registry_yaml_exists(self) -> None:
        assert DEFAULT_MODEL_REGISTRY_PATH.exists(), (
            f"Expected model_registry.yaml at {DEFAULT_MODEL_REGISTRY_PATH}"
        )

    def test_provider_profiles_yaml_exists(self) -> None:
        assert DEFAULT_PROVIDER_PROFILES_PATH.exists(), (
            f"Expected provider_profiles.yaml at {DEFAULT_PROVIDER_PROFILES_PATH}"
        )


# ---------------------------------------------------------------------------
# Test 2–6 — LlmConfigRegistry() loads from split files
# ---------------------------------------------------------------------------


class TestSplitRegistryLoads:
    def test_registry_loads_from_split_files_by_default(self) -> None:
        """LlmConfigRegistry() with no args must load from the 3 split files."""
        reg = LlmConfigRegistry()
        assert reg is not None

    def test_config_version_from_routes_file(self) -> None:
        reg = LlmConfigRegistry()
        assert reg.config_version == 1

    def test_route_map_populated(self) -> None:
        reg = LlmConfigRegistry()
        assert len(reg.route_map) > 0

    def test_model_map_populated(self) -> None:
        reg = LlmConfigRegistry()
        assert len(reg.model_map) > 0

    def test_provider_profile_map_populated(self) -> None:
        reg = LlmConfigRegistry()
        assert len(reg.provider_profile_map) > 0


# ---------------------------------------------------------------------------
# Test 7–11 — new aliases and route model references
# ---------------------------------------------------------------------------


class TestNewModelAliases:
    def test_math_basic_generator_in_model_map(self) -> None:
        reg = LlmConfigRegistry()
        assert "math_basic_generator" in reg.model_map

    def test_math_reasoning_generator_in_model_map(self) -> None:
        reg = LlmConfigRegistry()
        assert "math_reasoning_generator" in reg.model_map

    def test_general_fast_generator_in_model_map(self) -> None:
        reg = LlmConfigRegistry()
        assert "general_fast_generator" in reg.model_map

    def test_safe_mock_in_model_map_with_mock_provider(self) -> None:
        reg = LlmConfigRegistry()
        assert "safe_mock" in reg.model_map
        assert reg.model_map["safe_mock"].provider == "mock"

    def test_math_default_uses_math_basic_generator(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("math", "generator", "default")
        assert route is not None
        assert route.model == "math_basic_generator"

    def test_math_advanced_uses_math_advanced_generator(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("math", "generator", "advanced")
        assert route is not None
        # Fix A (50Q closure): the advanced-math structured generator moved off
        # deepseek-reasoner, whose hidden-reasoning variance exhausted the output
        # budget on single-slot calls, to the already-configured GPT-4.1 route.
        assert route.model == "openai_gpt_4_1"

    def test_math_intermediate_uses_math_intermediate_generator(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("math", "generator", "intermediate")
        assert route is not None
        assert route.model == "math_intermediate_generator"

    def test_reasoning_default_uses_reasoning_basic_generator(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("reasoning", "generator", "default")
        assert route is not None
        assert route.model == "reasoning_basic_generator"

    def test_reasoning_intermediate_uses_reasoning_intermediate_generator(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("reasoning", "generator", "intermediate")
        assert route is not None
        assert route.model == "reasoning_intermediate_generator"

    def test_reasoning_advanced_uses_reasoning_advanced_generator(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("reasoning", "generator", "advanced")
        assert route is not None
        assert route.model == "reasoning_advanced_generator"

    def test_no_active_route_uses_reasoning_advanced_complex_generator(self) -> None:
        reg = LlmConfigRegistry()
        for difficulty in ("default", "basic", "intermediate", "advanced"):
            route = reg.get_route("reasoning", "generator", difficulty)
            assert route is not None
            assert route.model != "reasoning_advanced_complex_generator"

    def test_general_default_uses_general_fast_generator(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("general", "generator", "default")
        assert route is not None
        assert route.model == "general_fast_generator"


# ---------------------------------------------------------------------------
# Test 12–18 — RouteEntry forbidden provider fields
# ---------------------------------------------------------------------------


class TestRouteEntryForbiddenFields:
    """Route entries MUST NOT contain provider-specific fields.

    These belong in model_registry.yaml, not in llm_routes.yaml.
    """

    def _routes_with_extra(self, extra_key: str, extra_value: str) -> str:
        return textwrap.dedent(f"""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: safe_mock
                    prompt: subjects/general_generator.md
                    temperature: 0.3
                    max_tokens: 800
                    {extra_key}: {extra_value}
                    fallback:
                      - safe_mock
        """)

    def test_route_with_model_id_rejected(self, tmp_path: Path) -> None:
        bad_routes = self._routes_with_extra("model_id", "gpt-4o-mini")
        with pytest.raises((LlmConfigValidationError, ValueError)):
            _make_split_registry(tmp_path, routes_content=bad_routes)

    def test_route_with_provider_rejected(self, tmp_path: Path) -> None:
        bad_routes = self._routes_with_extra("provider", "openai")
        with pytest.raises((LlmConfigValidationError, ValueError)):
            _make_split_registry(tmp_path, routes_content=bad_routes)

    def test_route_with_deployment_rejected(self, tmp_path: Path) -> None:
        bad_routes = self._routes_with_extra("deployment", "my-deployment")
        with pytest.raises((LlmConfigValidationError, ValueError)):
            _make_split_registry(tmp_path, routes_content=bad_routes)

    def test_route_with_provider_profile_rejected(self, tmp_path: Path) -> None:
        bad_routes = self._routes_with_extra("provider_profile", "openai_primary")
        with pytest.raises((LlmConfigValidationError, ValueError)):
            _make_split_registry(tmp_path, routes_content=bad_routes)

    def test_route_with_api_key_env_rejected(self, tmp_path: Path) -> None:
        bad_routes = self._routes_with_extra("api_key_env", "OPENAI_API_KEY")
        with pytest.raises((LlmConfigValidationError, ValueError)):
            _make_split_registry(tmp_path, routes_content=bad_routes)

    def test_route_with_endpoint_env_rejected(self, tmp_path: Path) -> None:
        bad_routes = self._routes_with_extra("endpoint_env", "AZURE_OPENAI_ENDPOINT")
        with pytest.raises((LlmConfigValidationError, ValueError)):
            _make_split_registry(tmp_path, routes_content=bad_routes)

    def test_route_with_credential_ref_rejected(self, tmp_path: Path) -> None:
        bad_routes = self._routes_with_extra("credential_ref", "my-secret-ref")
        with pytest.raises((LlmConfigValidationError, ValueError)):
            _make_split_registry(tmp_path, routes_content=bad_routes)


# ---------------------------------------------------------------------------
# Test 19–21 — ModelConfig optional fields
# ---------------------------------------------------------------------------


class TestModelConfigOptionalFields:
    def test_model_config_works_without_model_label(self, tmp_path: Path) -> None:
        """model_label is optional in the simplified model registry."""
        models_yaml = textwrap.dedent("""\
            version: 1
            models:
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
        """)
        reg = _make_split_registry(tmp_path, models_content=models_yaml)
        cfg = reg.model_map["safe_mock"]
        assert cfg.model_label is None

    def test_model_config_works_without_cost_tier(self, tmp_path: Path) -> None:
        """cost_tier is optional in the simplified model registry."""
        models_yaml = textwrap.dedent("""\
            version: 1
            models:
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
        """)
        reg = _make_split_registry(tmp_path, models_content=models_yaml)
        cfg = reg.model_map["safe_mock"]
        assert cfg.cost_tier is None

    def test_model_config_description_field_accepted(self, tmp_path: Path) -> None:
        """description is a valid optional field on ModelConfig."""
        models_yaml = textwrap.dedent("""\
            version: 1
            models:
              safe_mock:
                description: A safe local mock model.
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
        """)
        reg = _make_split_registry(tmp_path, models_content=models_yaml)
        cfg = reg.model_map["safe_mock"]
        assert cfg.description == "A safe local mock model."


# ---------------------------------------------------------------------------
# Test 22–24 — cross-validation with split files
# ---------------------------------------------------------------------------


class TestSplitCrossValidation:
    def test_unknown_model_alias_in_routes_raises(self, tmp_path: Path) -> None:
        """Cross-validation must catch a route referencing a non-existent model alias."""
        bad_routes = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: nonexistent_model_alias
                    prompt: subjects/general_generator.md
                    temperature: 0.3
                    max_tokens: 800
                    fallback:
                      - safe_mock
        """)
        with pytest.raises(LlmConfigValidationError, match="nonexistent_model_alias"):
            _make_split_registry(tmp_path, routes_content=bad_routes)

    def test_unknown_provider_profile_in_model_registry_raises(
        self, tmp_path: Path
    ) -> None:
        """Cross-validation must catch a model referencing a non-existent provider profile."""
        bad_models = textwrap.dedent("""\
            version: 1
            models:
              safe_mock:
                provider: mock
                provider_profile: nonexistent_profile
                model_id: local-mock
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
        """)
        with pytest.raises(LlmConfigValidationError, match="nonexistent_profile"):
            _make_split_registry(tmp_path, models_content=bad_models)

    def test_provider_mismatch_between_model_and_profile_raises(
        self, tmp_path: Path
    ) -> None:
        """Cross-validation must reject model.provider != provider_profile.provider."""
        bad_models = textwrap.dedent("""\
            version: 1
            models:
              safe_mock:
                provider: openai
                provider_profile: local_mock
                model_id: gpt-4o-mini
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 20
        """)
        # local_mock profile has provider: mock — mismatch with openai
        with pytest.raises(LlmConfigValidationError):
            _make_split_registry(tmp_path, models_content=bad_models)


# ---------------------------------------------------------------------------
# Test 25 — yaml_path backward compat
# ---------------------------------------------------------------------------


class TestYamlPathBackwardCompat:
    def test_yaml_path_combined_format_still_loads(self, tmp_path: Path) -> None:
        """LlmConfigRegistry(yaml_path=...) must still load the old combined YAML."""
        combined_path = tmp_path / "llm_orchestration.yaml"
        combined_path.write_text(_MINIMAL_COMBINED_YAML, encoding="utf-8")
        reg = LlmConfigRegistry(yaml_path=combined_path)
        assert reg.config_version == 1
        assert ("general", "generator", "default") in reg.route_map
        assert "safe_mock" in reg.model_map
        assert "local_mock" in reg.provider_profile_map


# ---------------------------------------------------------------------------
# Test 26 — dry-run path
# ---------------------------------------------------------------------------


class TestDryRunCompatibility:
    def test_production_registry_has_safe_mock_for_dry_run(self) -> None:
        """The production split registry must include safe_mock (mock provider).

        dry_run.py constructs a synthetic RouteDecision with model='safe_mock'.
        The production registry must have this alias for the dry-run to work.
        """
        reg = LlmConfigRegistry()
        assert "safe_mock" in reg.model_map
        model = reg.model_map["safe_mock"]
        assert model.provider == "mock"
        assert "local_mock" in reg.provider_profile_map
