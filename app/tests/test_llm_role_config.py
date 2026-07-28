"""
Tests for LLM_ROLE_CONFIG_JSON parsing and alias resolution (legacy path).
"""

from __future__ import annotations

import json

import pytest

import config as cfg_module
from services.llm.orchestration.config_registry import reset_registry
from services.llm.orchestration.errors import LlmConfigValidationError
from services.llm.providers.errors import LlmConfigurationError


def _reset_settings() -> None:
    cfg_module._settings = None


class TestLlmRoleConfigJson:
    def test_orchestrated_path_ignores_role_config_json(self, monkeypatch: pytest.MonkeyPatch):
        """YAML is primary when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true."""
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "true")
        monkeypatch.setenv(
            "LLM_ROLE_CONFIG_JSON",
            json.dumps({"doubt_solver_classifier": "nonexistent_alias_xyz"}),
        )
        _reset_settings()
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        reg = LlmConfigRegistry()
        route = reg.get_route("general", "classifier", "default")
        assert route is not None
        assert route.model == "doubt_solver_classifier_gemini"

    def test_alias_format_resolves_from_registry(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv(
            "LLM_ROLE_CONFIG_JSON",
            json.dumps({"doubt_solver_classifier": "doubt_solver_classifier"}),
        )
        _reset_settings()
        reset_registry()

        cfg = cfg_module.get_llm_role_config("doubt_solver_classifier")
        assert cfg.provider == "azure_openai"
        assert cfg.deployment == "gpt-4.1-mini"
        assert cfg.model_label == "doubt_solver_classifier"

    def test_unknown_alias_fails_validation(self, monkeypatch: pytest.MonkeyPatch):
        from services.llm.llm_role_config import (
            parse_role_config_map,
            validate_role_config_aliases,
        )

        role_map = parse_role_config_map(
            json.dumps({"doubt_solver_classifier": "totally_unknown_alias"})
        )
        with pytest.raises(LlmConfigValidationError, match="unknown model alias"):
            validate_role_config_aliases(role_map)

    def test_legacy_inline_format_still_works(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv(
            "LLM_ROLE_CONFIG_JSON",
            json.dumps(
                {
                    "doubt_solver_classifier": {
                        "provider": "openai",
                        "model_label": "gpt-4o-mini",
                        "model": "gpt-4o-mini",
                        "temperature": 0,
                        "max_tokens": 500,
                        "supports_streaming": False,
                    }
                }
            ),
        )
        _reset_settings()

        cfg = cfg_module.get_llm_role_config("doubt_solver_classifier")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4o-mini"

    def test_malformed_json_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "NOT_JSON")
        _reset_settings()

        with pytest.raises(LlmConfigurationError, match="not valid JSON"):
            cfg_module.get_llm_role_config("doubt_solver_classifier")

    def test_canonical_role_alias_map_includes_classifiers(self):
        from services.llm.llm_role_config import _CANONICAL_ROLE_ALIASES

        assert _CANONICAL_ROLE_ALIASES["classifier.default"] == "doubt_solver_classifier"
        assert _CANONICAL_ROLE_ALIASES["classifier.strong"] == "doubt_solver_classifier_strong"
