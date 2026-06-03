"""
Tests for model_registry.yaml env overrides and active-route deployment safety.
"""

from __future__ import annotations

import pytest

from services.llm.orchestration.config_registry import LlmConfigRegistry, reset_registry
from services.llm.orchestration.errors import LlmConfigValidationError


class TestModelRegistryEnvOverrides:
    def test_gpt_41_aliases_load_with_defaults(self):
        reg = LlmConfigRegistry()
        assert reg.model_map["openai_gpt_4_1"].deployment == "gpt-4.1"
        assert reg.model_map["openai_gpt_4_1_mini"].deployment == "gpt-4.1-mini"

    def test_gpt_54_aliases_blank_when_env_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_GPT_5_4", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT_GPT_5_4_MINI", raising=False)
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["openai_gpt_5_4"].deployment == ""
        assert reg.model_map["openai_gpt_5_4_mini"].deployment == ""

    def test_gpt_54_aliases_use_env_when_set(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_GPT_5_4", "my-gpt54-deploy")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_GPT_5_4_MINI", "my-gpt54mini-deploy")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["openai_gpt_5_4"].deployment == "my-gpt54-deploy"
        assert reg.model_map["openai_gpt_5_4_mini"].deployment == "my-gpt54mini-deploy"

    def test_gemini_aliases_load(self):
        reg = LlmConfigRegistry()
        assert reg.model_map["gemini_flash_lite_text"].provider == "gemini"
        assert reg.model_map["gemini_flash_text"].model_id == "gemini-2.5-flash"
        assert reg.model_map["gemini_image_extractor"].supports_streaming is False

    def test_deepseek_aliases_load(self):
        reg = LlmConfigRegistry()
        assert reg.model_map["deepseek_standard_generator"].model_id == "deepseek-chat"
        assert reg.model_map["deepseek_reasoning_generator"].model_id == "deepseek-reasoner"
        assert reg.model_map["deepseek_advanced_generator"].model_id == "deepseek-reasoner"

    def test_azure_deepseek_aliases_load(self):
        reg = LlmConfigRegistry()
        reasoning = reg.model_map["deepseek_azure_reasoning_generator"]
        standard = reg.model_map["deepseek_azure_standard_generator"]
        advanced = reg.model_map["deepseek_azure_advanced_generator"]
        assert reasoning.provider == "azure_openai"
        assert reasoning.provider_profile == "azure_deepseek"
        assert standard.provider_profile == "azure_deepseek"
        assert advanced.provider_profile == "azure_deepseek"
        assert reasoning.deployment == ""
        assert reasoning.supports_streaming is True
        assert reasoning.supports_thinking is False

    def test_azure_deepseek_profile_loads(self):
        reg = LlmConfigRegistry()
        profile = reg.provider_profile_map["azure_deepseek"]
        assert profile.provider == "azure_openai"
        assert profile.api_key_env == "AZURE_DEEPSEEK_API_KEY"
        assert profile.endpoint_env == "AZURE_DEEPSEEK_ENDPOINT"
        assert profile.optional_api_key is True
        assert profile.optional_endpoint is True

    def test_azure_deepseek_deployments_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_DEEPSEEK_REASONER_DEPLOYMENT", "my-deepseek-r1")
        monkeypatch.setenv("AZURE_DEEPSEEK_CHAT_DEPLOYMENT", "my-deepseek-chat")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["deepseek_azure_reasoning_generator"].deployment == "my-deepseek-r1"
        assert reg.model_map["deepseek_azure_standard_generator"].deployment == "my-deepseek-chat"

    def test_production_routes_unchanged_no_azure_deepseek_default(self):
        reg = LlmConfigRegistry()
        active = reg._active_route_model_aliases()
        assert "math_intermediate_generator" in active
        assert "deepseek_azure_reasoning_generator" not in active
        assert "deepseek_azure_standard_generator" not in active
        reg.validate_real_mode_deployments()

    def test_active_azure_deepseek_blank_deployment_fails_validation(self):
        reg = LlmConfigRegistry()
        from schemas.llm_routing import ResolvedRouteEntry

        reg._route_map[("math", "generator", "default")] = ResolvedRouteEntry(
            model="deepseek_azure_reasoning_generator",
            prompt="subjects/math_generator.md",
            overlays=[],
            intent_overlays={},
            temperature=0.15,
            max_tokens=2600,
            provider_options={},
            fallback=["advanced"],
        )
        with pytest.raises(LlmConfigValidationError, match="empty Azure DeepSeek deployment"):
            reg.validate_real_mode_deployments()

    def test_classifier_uses_safe_azure_deployment(self):
        reg = LlmConfigRegistry()
        primary = reg.model_map["doubt_solver_classifier"]
        strong = reg.model_map["doubt_solver_classifier_strong"]
        assert primary.deployment == "gpt-4.1-mini"
        assert strong.deployment == "gpt-4.1"
        assert primary.supports_streaming is False
        assert strong.supports_streaming is False

    def test_math_intermediate_uses_gpt_41_mini_not_gpt_41(self):
        reg = LlmConfigRegistry()
        cfg = reg.model_map["math_intermediate_generator"]
        assert cfg.deployment == "gpt-4.1-mini"
        assert cfg.deployment != "gpt-4.1"

    def test_active_routes_point_to_available_aliases(self):
        reg = LlmConfigRegistry()
        active = reg._active_route_model_aliases()
        assert "doubt_solver_classifier" in active
        assert "math_intermediate_generator" in active
        reg.validate_real_mode_deployments()

    def test_active_blank_gpt54_route_would_fail_validation(self):
        """If a route pointed at openai_gpt_5_4 with blank deployment, preflight fails."""
        reg = LlmConfigRegistry()
        from schemas.llm_routing import ResolvedRouteEntry

        reg._route_map[("test", "classifier", "default")] = ResolvedRouteEntry(
            model="openai_gpt_5_4",
            prompt="query_classifier.md",
            overlays=[],
            intent_overlays={},
            temperature=0.0,
            max_tokens=500,
            provider_options={},
            fallback=[],
        )
        with pytest.raises(LlmConfigValidationError, match="empty Azure deployment"):
            reg.validate_real_mode_deployments()

    def test_gemini_model_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("GEMINI_TEXT_MODEL", "custom-gemini-model")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["gemini_flash_text"].model_id == "custom-gemini-model"

    def test_deepseek_advanced_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEEPSEEK_ADVANCED_MODEL", "deepseek-custom-advanced")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["deepseek_advanced_generator"].model_id == "deepseek-custom-advanced"


class TestSettingsEnvVars:
    def test_azure_deployment_settings_present(self, monkeypatch: pytest.MonkeyPatch):
        import config as cfg_module

        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_GPT_4_1", "deploy-41")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_GPT_5_4", "")
        monkeypatch.setenv("GEMINI_API_KEY", "")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "")
        cfg_module._settings = None
        s = cfg_module.get_settings()
        assert s.azure_openai_deployment_gpt_4_1 == "deploy-41"
        assert s.azure_openai_deployment_gpt_5_4 == ""
        assert s.gemini_api_key == ""
        assert s.deepseek_api_key == ""
        assert s.deepseek_advanced_model == "deepseek-reasoner"

    def test_azure_deepseek_settings_defaults(self, monkeypatch: pytest.MonkeyPatch):
        import config as cfg_module

        monkeypatch.delenv("AZURE_DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_DEEPSEEK_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_DEEPSEEK_REASONER_DEPLOYMENT", raising=False)
        cfg_module._settings = None
        s = cfg_module.get_settings()
        assert s.azure_deepseek_api_key == ""
        assert s.azure_deepseek_endpoint == ""
        assert s.azure_deepseek_timeout_seconds == 90
        assert s.azure_deepseek_reasoner_deployment == ""
