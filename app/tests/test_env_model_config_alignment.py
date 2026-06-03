"""
Env + model config alignment tests for benchmark routing (no real API calls).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import config as cfg_module
from services.llm.orchestration.config_registry import LlmConfigRegistry, reset_registry
from services.llm.providers.azure_openai_provider import (
    _azure_error_diagnostics,
    _log_azure_call_failure,
)


def _reset_settings() -> None:
    cfg_module._settings = None


class TestEnvLocalExample:
    def test_example_documents_benchmark_deployment_vars(self):
        text = Path(__file__).resolve().parents[1].joinpath(".env.local.example").read_text()
        for var in (
            "AZURE_OPENAI_DEPLOYMENT_O4_MINI",
            "AZURE_OPENAI_DEPLOYMENT_O3",
            "DEEPSEEK_V4PRO_MODEL",
            "AZURE_OPENAI_DEPLOYMENT_GPT_4_1",
            "AZURE_OPENAI_DEPLOYMENT_GPT_4_1_MINI",
            "ENABLE_ORCHESTRATED_DOUBT_SOLVER",
            "LLM_ROLE_CONFIG_JSON",
        ):
            assert var in text, f"missing {var} in .env.local.example"
        assert (
            "IGNORED when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true" in text
            or "ignores LLM_ROLE_CONFIG_JSON" in text
        )
        assert "openai_o4_mini" in text or "O4_MINI" in text
        assert "sk-" not in text.split("OPENAI_API_KEY=")[-1][:20]


class TestSettingsEnvFields:
    def test_benchmark_env_fields_in_settings(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_O4_MINI", "my-o4")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_O3", "my-o3")
        monkeypatch.setenv("DEEPSEEK_V4PRO_MODEL", "deepseek-custom")
        _reset_settings()
        s = cfg_module.get_settings()
        assert s.azure_openai_deployment_o4_mini == "my-o4"
        assert s.azure_openai_deployment_o3 == "my-o3"
        assert s.deepseek_v4pro_model == "deepseek-custom"


class TestOrchestratedIgnoresRoleJson:
    def test_orchestrated_mode_uses_yaml_not_stale_role_json(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "true")
        monkeypatch.setenv(
            "LLM_ROLE_CONFIG_JSON",
            json.dumps(
                {
                    "doubt_solver_classifier": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "model_label": "stale",
                        "temperature": 0,
                        "max_tokens": 100,
                        "supports_streaming": False,
                    }
                }
            ),
        )
        _reset_settings()
        reset_registry()
        reg = LlmConfigRegistry()
        route = reg.get_route("reasoning", "generator", "advanced")
        assert route is not None
        assert route.model == "reasoning_advanced_generator"
        assert reg.model_map["reasoning_advanced_generator"].deployment == "o4-mini"


class TestRegistryEnvMapping:
    def test_o4_mini_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_O4_MINI", "deploy-o4-mini")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["openai_o4_mini"].deployment == "deploy-o4-mini"

    def test_o3_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_O3", "deploy-o3")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["openai_o3"].deployment == "deploy-o3"

    def test_deepseek_v4pro_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DEEPSEEK_V4PRO_MODEL", "deepseek-v4pro-custom")
        reset_registry()
        reg = LlmConfigRegistry()
        assert reg.model_map["deepseek_v4pro"].model_id == "deepseek-v4pro-custom"
        assert reg.model_map["math_advanced_generator"].model_id == "deepseek-v4pro-custom"


class TestAzureSafeErrorLogging:
    def test_azure_error_diagnostics_extracts_safe_fields(self):
        exc = MagicMock()
        exc.status_code = 400
        exc.code = "DeploymentNotFound"
        exc.body = {
            "error": {
                "code": "DeploymentNotFound",
                "message": "The API deployment for this resource does not exist.",
            }
        }

        diag = _azure_error_diagnostics(exc)
        assert diag["status_code"] == 400
        assert diag["provider_error_code"] == "DeploymentNotFound"
        assert "deployment" in str(diag["provider_error_message_short"]).lower()

    def test_log_azure_call_failure_no_secrets(
        self, caplog: pytest.LogCaptureFixture
    ):
        caplog.set_level(logging.WARNING)
        exc = MagicMock()
        exc.status_code = 400
        exc.code = "invalid_request_error"
        exc.body = {"error": {"code": "invalid_request_error", "message": "Model not found"}}

        _log_azure_call_failure(
            exc=exc,
            operation="generate",
            model_alias="reasoning_intermediate_generator",
            deployment="wrong-o4-mini",
            route_id="reasoning.generator.intermediate",
            azure_api_mode="azure_openai_v1",
            failure_kind="model_not_found",
        )
        record = caplog.records[-1]
        assert "wrong-o4-mini" in record.message
        assert "reasoning_intermediate_generator" in record.message
        assert "status_code=400" in record.message or "400" in record.message
        assert "sk-" not in record.message
        assert "api_key" not in record.message.lower()


class TestLegacyRoleJsonAliasFormat:
    def test_benchmark_alias_json_resolves_azure_aliases(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "false")
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv(
            "LLM_ROLE_CONFIG_JSON",
            json.dumps(
                {
                    "reasoning.intermediate": "openai_o4_mini",
                    "math.basic": "openai_gpt_4_1_mini",
                }
            ),
        )
        _reset_settings()
        reset_registry()

        intermediate = cfg_module.get_llm_role_config("reasoning.intermediate")
        basic = cfg_module.get_llm_role_config("math.basic")
        assert intermediate.deployment == "o4-mini"
        assert basic.deployment == "gpt-4.1-mini"

    def test_deepseek_alias_not_supported_on_legacy_router(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Legacy model_router supports Azure/OpenAI only — DeepSeek needs orchestrated path."""
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "false")
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv(
            "LLM_ROLE_CONFIG_JSON",
            json.dumps({"math.advanced": "deepseek_v4pro"}),
        )
        _reset_settings()
        reset_registry()
        from services.llm.providers.errors import LlmConfigurationError

        with pytest.raises(LlmConfigurationError, match="does not support provider 'deepseek'"):
            cfg_module.get_llm_role_config("math.advanced")
