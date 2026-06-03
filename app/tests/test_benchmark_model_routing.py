"""
Benchmark-based model routing tests (no real provider calls).
"""

from __future__ import annotations

import pytest

from services.llm.orchestration.config_registry import LlmConfigRegistry, reset_registry
from services.llm.orchestration.model_config_resolver import ModelConfigResolver


class TestBenchmarkModelAliases:
    def test_required_base_aliases_load(self):
        reg = LlmConfigRegistry()
        for alias in (
            "openai_gpt_4_1_mini",
            "openai_gpt_4_1",
            "openai_o4_mini",
            "openai_o3",
            "openai_gpt_5_4",
            "deepseek_v4pro",
        ):
            assert alias in reg.model_map, f"missing alias {alias}"

    def test_o4_mini_reasoning_metadata(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["openai_o4_mini"]
        assert cfg.supports_reasoning is True
        assert cfg.reasoning_effort == "medium"
        assert cfg.supports_streaming is True

    def test_o3_reasoning_metadata(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["openai_o3"]
        assert cfg.supports_reasoning is True
        assert cfg.reasoning_effort == "high"

    def test_gpt_41_mini_no_reasoning(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["openai_gpt_4_1_mini"]
        assert cfg.supports_reasoning is False
        assert cfg.reasoning_effort == "none"

    def test_deepseek_v4pro_reasoning_metadata(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["deepseek_v4pro"]
        assert cfg.supports_reasoning is True
        assert cfg.reasoning_effort == "high"
        assert cfg.fallback_models == ["openai_gpt_5_4"]

    def test_exam_specific_generators_exist(self):
        reg = LlmConfigRegistry()
        assert "math_cat_advanced_generator" in reg.model_map
        assert "reasoning_sbi_po_complex_generator" in reg.model_map
        assert "reasoning_cat_lrdi_generator" in reg.model_map


class TestBenchmarkGeneratorMappings:
    def test_math_basic_uses_gpt_41_mini(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["math_basic_generator"]
        assert cfg.deployment == "gpt-4.1-mini"
        assert cfg.fallback_models[0] == "openai_o4_mini"

    def test_math_intermediate_uses_gpt_41_mini(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["math_intermediate_generator"]
        assert cfg.deployment == "gpt-4.1-mini"
        assert "openai_o4_mini" in cfg.fallback_models

    def test_math_advanced_uses_deepseek_v4pro(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["math_advanced_generator"]
        assert cfg.provider == "deepseek"
        assert cfg.model_id == "deepseek-reasoner"
        assert cfg.fallback_models == ["openai_o3", "openai_gpt_5_4"]

    def test_reasoning_basic_uses_gpt_41_mini(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["reasoning_basic_generator"]
        assert cfg.deployment == "gpt-4.1-mini"

    def test_reasoning_intermediate_uses_o4_mini(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["reasoning_intermediate_generator"]
        assert cfg.deployment == "o4-mini"
        assert cfg.supports_reasoning is True

    def test_reasoning_advanced_uses_o4_mini(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["reasoning_advanced_generator"]
        assert cfg.deployment == "o4-mini"
        assert "openai_o3" in cfg.fallback_models

    def test_general_uses_gpt_41_mini(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["general_fast_generator"]
        assert cfg.deployment == "gpt-4.1-mini"
        assert cfg.fallback_models[0] == "openai_gpt_4_1"


class TestBenchmarkRouteResolution:
    def _route_model(self, subject: str, difficulty: str) -> str:
        reg = LlmConfigRegistry()
        route = reg.get_route(subject, "generator", difficulty)
        assert route is not None, f"missing route {subject}.generator.{difficulty}"
        return route.model

    def test_math_basic_route(self):
        assert self._route_model("math", "basic") == "math_basic_generator"

    def test_math_intermediate_route(self):
        assert self._route_model("math", "intermediate") == "math_intermediate_generator"

    def test_math_advanced_route(self):
        assert self._route_model("math", "advanced") == "math_advanced_generator"

    def test_reasoning_basic_route(self):
        assert self._route_model("reasoning", "basic") == "reasoning_basic_generator"

    def test_reasoning_intermediate_route(self):
        assert self._route_model("reasoning", "intermediate") == "reasoning_intermediate_generator"

    def test_reasoning_advanced_route(self):
        assert self._route_model("reasoning", "advanced") == "reasoning_advanced_generator"

    def test_general_default_route(self):
        assert self._route_model("general", "default") == "general_fast_generator"

    def test_classifier_routes_unchanged(self):
        reg = LlmConfigRegistry()
        primary = reg.get_route("general", "classifier", "default")
        strong = reg.get_route("general", "classifier_strong", "default")
        assert primary is not None and primary.model == "doubt_solver_classifier"
        assert strong is not None and strong.model == "doubt_solver_classifier_strong"


class TestReasoningProviderOptions:
    def test_gpt_41_mini_rejects_thinking_option(self):
        reg = LlmConfigRegistry()
        resolver = ModelConfigResolver(registry=reg)
        cfg = reg.model_map["openai_gpt_4_1_mini"]
        with pytest.raises(Exception, match="does not support provider option 'thinking'"):
            resolver.validate_provider_options(
                provider_options={"thinking": True},
                model_config=cfg,
                model_alias="openai_gpt_4_1_mini",
            )

    def test_o4_mini_allows_thinking_option(self):
        reg = LlmConfigRegistry()
        resolver = ModelConfigResolver(registry=reg)
        cfg = reg.model_map["openai_o4_mini"]
        resolver.validate_provider_options(
            provider_options={"thinking": True},
            model_config=cfg,
            model_alias="openai_o4_mini",
        )

    def test_o3_allows_thinking_option(self):
        reg = LlmConfigRegistry()
        resolver = ModelConfigResolver(registry=reg)
        cfg = reg.model_map["openai_o3"]
        resolver.validate_provider_options(
            provider_options={"thinking": True},
            model_config=cfg,
            model_alias="openai_o3",
        )


class TestBenchmarkEnvOverrides:
    def test_o4_mini_deployment_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_O4_MINI", "my-o4-mini-deploy")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["openai_o4_mini"].deployment == "my-o4-mini-deploy"
        assert reg.model_map["reasoning_intermediate_generator"].deployment == "my-o4-mini-deploy"

    def test_deepseek_v4pro_model_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEEPSEEK_V4PRO_MODEL", "deepseek-custom-v4")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["deepseek_v4pro"].model_id == "deepseek-custom-v4"
        assert reg.model_map["math_advanced_generator"].model_id == "deepseek-custom-v4"

    def test_production_preflight_passes_with_defaults(self):
        reg = LlmConfigRegistry()
        reg.validate_real_mode_deployments()
