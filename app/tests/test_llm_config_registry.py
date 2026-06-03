"""
app/tests/test_llm_config_registry.py
--------------------------------------
Unit tests for app/services/llm_orchestration/config_registry.py

Coverage:
- Valid YAML loads and builds all maps.
- route_map / model_map / provider_profile_map populated correctly.
- Inheritance resolution (basic inherits model/prompt/temperature from default).
- Overlays are concatenated (parent + child), not replaced.
- Fallback: child wins if non-empty; parent used otherwise.
- Invalid model alias raises LlmConfigValidationError.
- Invalid provider profile reference raises LlmConfigValidationError.
- Invalid fallback symbol raises (caught via Pydantic).
- Self-referencing fallback raises LlmConfigValidationError.
- Secret-like value in ProviderProfile rejected.
- Invalid env var format rejected.
- max_tokens > 8000 rejected.
- temperature > 2 rejected.
- Prompt path with '..' rejected.
- Absolute prompt path rejected.
- URL prompt path rejected.
- safe_mock exists and provider=mock.
- YAML missing raises LlmConfigLoadError.
- Non-mapping YAML raises LlmConfigLoadError.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.errors import (
    LlmConfigLoadError,
    LlmConfigValidationError,
)

# ---------------------------------------------------------------------------
# Helper: write a minimal valid YAML and return its Path
# ---------------------------------------------------------------------------

_MINIMAL_VALID_YAML = textwrap.dedent("""\
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
        capabilities:
          general: low
    provider_profiles:
      local_mock:
        provider: mock
""")


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "llm_orchestration.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _make_registry(tmp_path: Path, content: str) -> LlmConfigRegistry:
    return LlmConfigRegistry(yaml_path=_write_yaml(tmp_path, content))


# ---------------------------------------------------------------------------
# TestValidYamlLoads
# ---------------------------------------------------------------------------


class TestValidYamlLoads:
    def test_registry_builds_from_standard_yaml(self) -> None:
        """The real project YAML loads without error."""
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

    def test_minimal_yaml_loads(self, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, _MINIMAL_VALID_YAML)
        assert reg.config_version == 1
        assert ("general", "generator", "default") in reg.route_map


# ---------------------------------------------------------------------------
# TestInheritanceResolution
# ---------------------------------------------------------------------------


class TestInheritanceResolution:
    _YAML_WITH_INHERITANCE = textwrap.dedent("""\
        version: 1
        routes:
          math:
            generator:
              default:
                model: safe_mock
                prompt: subjects/math_generator.md
                temperature: 0.2
                max_tokens: 800
                fallback:
                  - safe_mock
              basic:
                inherits: default
                overlays:
                  - levels/basic.md
                max_tokens: 700
                fallback:
                  - default
                  - safe_mock
              advanced:
                inherits: default
                model: safe_mock
                overlays:
                  - levels/advanced.md
                temperature: 0.1
                max_tokens: 1000
                fallback:
                  - basic
                  - default
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

    def test_basic_inherits_model_from_default(self, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, self._YAML_WITH_INHERITANCE)
        basic = reg.get_route("math", "generator", "basic")
        assert basic is not None
        assert basic.model == "safe_mock"  # inherited from default

    def test_basic_inherits_prompt_from_default(self, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, self._YAML_WITH_INHERITANCE)
        basic = reg.get_route("math", "generator", "basic")
        assert basic is not None
        assert basic.prompt == "subjects/math_generator.md"

    def test_basic_inherits_temperature_from_default(self, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, self._YAML_WITH_INHERITANCE)
        basic = reg.get_route("math", "generator", "basic")
        assert basic is not None
        assert basic.temperature == 0.2

    def test_basic_child_max_tokens_overrides_parent(self, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, self._YAML_WITH_INHERITANCE)
        basic = reg.get_route("math", "generator", "basic")
        assert basic is not None
        assert basic.max_tokens == 700  # child overrides

    def test_overlays_concatenated_parent_plus_child(self, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, self._YAML_WITH_INHERITANCE)
        # default has no overlays, basic has [levels/basic.md]
        basic = reg.get_route("math", "generator", "basic")
        assert basic is not None
        assert basic.overlays == ["levels/basic.md"]

    def test_advanced_overlays_concatenated(self, tmp_path: Path) -> None:
        # default has no overlays, advanced has [levels/advanced.md]
        reg = _make_registry(tmp_path, self._YAML_WITH_INHERITANCE)
        advanced = reg.get_route("math", "generator", "advanced")
        assert advanced is not None
        assert advanced.overlays is not None
        assert "levels/advanced.md" in advanced.overlays

    def test_fallback_child_wins_when_non_empty(self, tmp_path: Path) -> None:
        reg = _make_registry(tmp_path, self._YAML_WITH_INHERITANCE)
        basic = reg.get_route("math", "generator", "basic")
        assert basic is not None
        # child provides fallback = [default, safe_mock] — should win
        assert basic.fallback == ["default", "safe_mock"]

    def test_fallback_parent_used_when_child_empty(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              math:
                generator:
                  default:
                    model: safe_mock
                    prompt: subjects/math_generator.md
                    temperature: 0.2
                    max_tokens: 800
                    fallback:
                      - safe_mock
                  basic:
                    inherits: default
                    max_tokens: 700
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
        reg = _make_registry(tmp_path, yaml_content)
        basic = reg.get_route("math", "generator", "basic")
        assert basic is not None
        # child has no fallback → inherits parent's [safe_mock]
        assert basic.fallback == ["safe_mock"]

    def test_provider_options_shallow_merge(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              math:
                generator:
                  default:
                    model: safe_mock
                    prompt: subjects/math_generator.md
                    temperature: 0.2
                    max_tokens: 800
                    provider_options:
                      thinking: false
                      stream: true
                    fallback:
                      - safe_mock
                  advanced:
                    inherits: default
                    provider_options:
                      thinking: true
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
        reg = _make_registry(tmp_path, yaml_content)
        advanced = reg.get_route("math", "generator", "advanced")
        assert advanced is not None
        # child overrides thinking; parent's stream is inherited
        assert advanced.provider_options["thinking"] is True
        assert advanced.provider_options["stream"] is True


# ---------------------------------------------------------------------------
# TestModelMapCompiled
# ---------------------------------------------------------------------------


class TestModelMapCompiled:
    def test_math_basic_generator_in_model_map(self) -> None:
        reg = LlmConfigRegistry()
        assert "math_basic_generator" in reg.model_map

    def test_math_reasoning_generator_in_model_map(self) -> None:
        reg = LlmConfigRegistry()
        assert "math_reasoning_generator" in reg.model_map

    def test_safe_mock_in_model_map(self) -> None:
        reg = LlmConfigRegistry()
        assert "safe_mock" in reg.model_map

    def test_safe_mock_provider_is_mock(self) -> None:
        reg = LlmConfigRegistry()
        assert reg.model_map["safe_mock"].provider == "mock"


# ---------------------------------------------------------------------------
# TestProviderProfileMapCompiled
# ---------------------------------------------------------------------------


class TestProviderProfileMapCompiled:
    def test_gemini_primary_exists(self) -> None:
        reg = LlmConfigRegistry()
        assert "gemini_primary" in reg.provider_profile_map

    def test_local_mock_exists(self) -> None:
        reg = LlmConfigRegistry()
        assert "local_mock" in reg.provider_profile_map

    def test_azure_primary_exists(self) -> None:
        reg = LlmConfigRegistry()
        assert "azure_primary" in reg.provider_profile_map

    def test_openai_primary_exists(self) -> None:
        reg = LlmConfigRegistry()
        assert "openai_primary" in reg.provider_profile_map


# ---------------------------------------------------------------------------
# TestCrossValidation
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def test_invalid_model_alias_raises(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: nonexistent_model
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
        with pytest.raises(LlmConfigValidationError, match="nonexistent_model"):
            _make_registry(tmp_path, yaml_content)

    def test_invalid_provider_profile_ref_raises(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
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
                provider_profile: nonexistent_profile
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
        with pytest.raises(LlmConfigValidationError, match="nonexistent_profile"):
            _make_registry(tmp_path, yaml_content)

    def test_missing_safe_mock_raises(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: my_model
                    prompt: subjects/general_generator.md
                    temperature: 0.3
                    max_tokens: 800
                    fallback: []
            models:
              my_model:
                provider: mock
                provider_profile: local_mock
                model_id: local-mock
                model_label: my-model
                cost_tier: none
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
                capabilities: {}
            provider_profiles:
              local_mock:
                provider: mock
        """)
        with pytest.raises(LlmConfigValidationError, match="safe_mock"):
            _make_registry(tmp_path, yaml_content)


# ---------------------------------------------------------------------------
# TestFallbackSymbolValidation
# ---------------------------------------------------------------------------


class TestFallbackSymbolValidation:
    def test_invalid_fallback_symbol_raises_at_yaml_load(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
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
                      - this_is_not_a_valid_symbol
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
        with pytest.raises((LlmConfigValidationError, Exception)):
            _make_registry(tmp_path, yaml_content)

    def test_self_referencing_fallback_raises(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
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
                      - default
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
        with pytest.raises(LlmConfigValidationError, match="self-referencing"):
            _make_registry(tmp_path, yaml_content)

    def test_basic_self_referencing_fallback_raises(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              math:
                generator:
                  default:
                    model: safe_mock
                    prompt: subjects/math_generator.md
                    temperature: 0.2
                    max_tokens: 800
                    fallback:
                      - safe_mock
                  basic:
                    inherits: default
                    fallback:
                      - basic
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
        with pytest.raises(LlmConfigValidationError, match="self-referencing"):
            _make_registry(tmp_path, yaml_content)


# ---------------------------------------------------------------------------
# TestSecretLeakRejection
# ---------------------------------------------------------------------------


class TestSecretLeakRejection:
    def test_sk_prefix_rejected(self, tmp_path: Path) -> None:
        """api_key_env starting with sk- looks like an OpenAI key — reject it."""
        yaml_content = textwrap.dedent("""\
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
                provider_profile: bad_profile
                model_id: local-mock
                model_label: safe-mock
                cost_tier: none
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
                capabilities: {}
            provider_profiles:
              bad_profile:
                provider: openai
                api_key_env: sk-REALKEY123
        """)
        with pytest.raises(Exception, match="sk-"):
            _make_registry(tmp_path, yaml_content)

    def test_aiza_prefix_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
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
                provider_profile: bad_profile
                model_id: local-mock
                model_label: safe-mock
                cost_tier: none
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
                capabilities: {}
            provider_profiles:
              bad_profile:
                provider: gemini
                api_key_env: AIzaSyABC123
        """)
        with pytest.raises(Exception, match="AIza"):
            _make_registry(tmp_path, yaml_content)

    def test_akia_prefix_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
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
                provider_profile: bad_profile
                model_id: local-mock
                model_label: safe-mock
                cost_tier: none
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
                capabilities: {}
            provider_profiles:
              bad_profile:
                provider: openai
                api_key_env: AKIAIOSFODNN7EXAMPLE
        """)
        with pytest.raises((LlmConfigValidationError, LlmConfigLoadError, ValueError)):
            _make_registry(tmp_path, yaml_content)

    def test_begin_prefix_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
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
                provider_profile: bad_profile
                model_id: local-mock
                model_label: safe-mock
                cost_tier: none
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
                capabilities: {}
            provider_profiles:
              bad_profile:
                provider: openai
                credential_ref: "-----BEGIN PRIVATE KEY-----"
        """)
        with pytest.raises(Exception, match="-----BEGIN"):
            _make_registry(tmp_path, yaml_content)

    def test_env_var_name_accepted(self, tmp_path: Path) -> None:
        """A valid SCREAMING_SNAKE_CASE env var name must be accepted."""
        _make_registry(tmp_path, _MINIMAL_VALID_YAML)
        # gemini_primary in the real YAML has api_key_env: GEMINI_API_KEY
        real_reg = LlmConfigRegistry()
        profile = real_reg.get_provider_profile("gemini_primary")
        assert profile is not None
        assert profile.api_key_env == "GEMINI_API_KEY"

    def test_lowercase_env_var_rejected(self, tmp_path: Path) -> None:
        """Lowercase or mixed-case strings are not valid env var names."""
        yaml_content = textwrap.dedent("""\
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
                provider_profile: bad_profile
                model_id: local-mock
                model_label: safe-mock
                cost_tier: none
                supports_streaming: true
                supports_thinking: false
                timeout_seconds: 1
                capabilities: {}
            provider_profiles:
              bad_profile:
                provider: openai
                api_key_env: my_api_key
        """)
        with pytest.raises((LlmConfigValidationError, LlmConfigLoadError, ValueError)):
            _make_registry(tmp_path, yaml_content)


# ---------------------------------------------------------------------------
# TestRouteEntryValidation
# ---------------------------------------------------------------------------


class TestRouteEntryValidation:
    def test_max_tokens_over_limit_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: safe_mock
                    prompt: subjects/general_generator.md
                    temperature: 0.3
                    max_tokens: 99999
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
        with pytest.raises((LlmConfigValidationError, LlmConfigLoadError, ValueError)):
            _make_registry(tmp_path, yaml_content)

    def test_temperature_over_limit_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: safe_mock
                    prompt: subjects/general_generator.md
                    temperature: 5.0
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
        with pytest.raises((LlmConfigValidationError, LlmConfigLoadError, ValueError)):
            _make_registry(tmp_path, yaml_content)

    def test_dotdot_in_prompt_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: safe_mock
                    prompt: ../etc/passwd.md
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
        with pytest.raises((LlmConfigValidationError, LlmConfigLoadError, ValueError)):
            _make_registry(tmp_path, yaml_content)

    def test_absolute_prompt_path_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: safe_mock
                    prompt: /etc/passwd.md
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
        with pytest.raises((LlmConfigValidationError, LlmConfigLoadError, ValueError)):
            _make_registry(tmp_path, yaml_content)

    def test_url_prompt_path_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: safe_mock
                    prompt: https://evil.com/prompt.md
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
        with pytest.raises((LlmConfigValidationError, LlmConfigLoadError, ValueError)):
            _make_registry(tmp_path, yaml_content)

    def test_non_md_prompt_path_rejected(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: safe_mock
                    prompt: subjects/prompt.txt
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
        with pytest.raises((LlmConfigValidationError, LlmConfigLoadError, ValueError)):
            _make_registry(tmp_path, yaml_content)


# ---------------------------------------------------------------------------
# TestSafeMockExists
# ---------------------------------------------------------------------------


class TestSafeMockExists:
    def test_safe_mock_in_model_map(self) -> None:
        reg = LlmConfigRegistry()
        assert "safe_mock" in reg.model_map

    def test_safe_mock_provider_is_mock(self) -> None:
        reg = LlmConfigRegistry()
        assert reg.model_map["safe_mock"].provider == "mock"

    def test_safe_mock_cost_tier_none(self) -> None:
        # cost_tier is optional in the simplified model_registry.yaml;
        # production safe_mock does not set it, so cost_tier is None.
        reg = LlmConfigRegistry()
        assert reg.model_map["safe_mock"].cost_tier is None


class TestSubjectDifficultyRouteAliases:
    def test_math_intermediate_route_alias(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("math", "generator", "intermediate")
        assert route is not None
        assert route.model == "math_intermediate_generator"

    def test_math_advanced_route_alias(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("math", "generator", "advanced")
        assert route is not None
        assert route.model == "math_advanced_generator"

    def test_reasoning_basic_route_alias(self) -> None:
        reg = LlmConfigRegistry()
        for difficulty in ("default", "basic"):
            route = reg.get_route("reasoning", "generator", difficulty)
            assert route is not None
            assert route.model == "reasoning_basic_generator"

    def test_reasoning_intermediate_route_alias(self) -> None:
        reg = LlmConfigRegistry()
        route = reg.get_route("reasoning", "generator", "intermediate")
        assert route is not None
        assert route.model == "reasoning_intermediate_generator"

    def test_math_intermediate_uses_gpt_41_mini(self) -> None:
        reg = LlmConfigRegistry()
        cfg = reg.model_map["math_intermediate_generator"]
        assert cfg.deployment == "gpt-4.1-mini"
        assert cfg.deployment != "gpt-4.1"

    def test_active_aliases_do_not_use_unprovisioned_placeholders(self) -> None:
        reg = LlmConfigRegistry()
        active_aliases = [
            "math_intermediate_generator",
            "math_advanced_generator",
            "reasoning_basic_generator",
            "reasoning_intermediate_generator",
            "reasoning_advanced_generator",
        ]
        forbidden = {"gpt-5.4-mini", "YOUR_", "placeholder"}
        for alias in active_aliases:
            cfg = reg.model_map[alias]
            deployment = cfg.deployment or cfg.model_id or ""
            assert deployment, f"{alias} must have deployment or model_id"
            for bad in forbidden:
                assert bad not in deployment, f"{alias} uses forbidden deployment fragment {bad!r}"
        route = reg.get_route("reasoning", "generator", "advanced")
        assert route is not None
        assert route.model == "reasoning_advanced_generator"
        model_cfg = reg.model_map["reasoning_advanced_generator"]
        assert model_cfg.provider == "azure_openai"
        assert model_cfg.deployment == "o4-mini"
        assert model_cfg.fallback_models == ["openai_o3", "deepseek_v4pro"]
        deepseek_cfg = reg.model_map.get("reasoning_advanced_generator_deepseek")
        assert deepseek_cfg is not None
        assert deepseek_cfg.provider == "deepseek"
        deprecated = reg.model_map.get("reasoning_advanced_complex_generator")
        assert deprecated is not None


# ---------------------------------------------------------------------------
# TestConfigLoadErrors
# ---------------------------------------------------------------------------


class TestConfigLoadErrors:
    def test_missing_yaml_raises_load_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with pytest.raises(LlmConfigLoadError):
            LlmConfigRegistry(yaml_path=missing)

    def test_non_mapping_yaml_raises_load_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list\n- not a mapping\n", encoding="utf-8")
        with pytest.raises(LlmConfigLoadError):
            LlmConfigRegistry(yaml_path=p)

    def test_invalid_yaml_syntax_raises_load_error(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("version: 1\nroutes: :\n  bad:::yaml\n", encoding="utf-8")
        with pytest.raises(LlmConfigLoadError):
            LlmConfigRegistry(yaml_path=p)
