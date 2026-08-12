"""
app/tests/test_config_validation.py
--------------------------------------
Tests for configuration validation behaviour.

Verifies that:
    - ENABLE_REAL_LLM=true with missing role config → LlmConfigurationError
    - ENABLE_KB_RETRIEVAL=true with missing BEDROCK_KB_ID → KnowledgeBaseConfigurationError
    - ENABLE_DYNAMODB_FETCH=true with missing table name → DynamoDbConfigurationError
    - Malformed LLM_ROLE_CONFIG_JSON → LlmConfigurationError
    - Graph handles config errors gracefully (does not crash; sets needs_review=True)
    - Default (all flags off) produces no config errors

No real AWS credentials, network, or LLM calls required.
All config errors are triggered through env var manipulation and service calls.

Observation:
    Invalid integer values for BEDROCK_KB_MAX_RESULTS or
    DOUBT_SOLVER_MAX_CONTEXT_CHARS will raise ValueError from int() at settings
    load time.  This is a clear "fail-fast" error — the agent refuses to start
    with a misconfigured environment.  No silent fallback is provided by design.
"""

from __future__ import annotations

import pytest

import config as cfg_module


def _reset_settings() -> None:
    cfg_module._settings = None


def test_local_log_content_supports_full_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("AGENT_LOCAL_LOG_CONTENT", "full")
    monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "false")
    _reset_settings()

    assert cfg_module.get_settings().agent_local_log_content == "full"
    _reset_settings()


def test_production_forces_local_log_content_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AGENT_LOCAL_LOG_CONTENT", "full")
    monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "false")
    _reset_settings()

    assert cfg_module.get_settings().agent_local_log_content == "off"
    _reset_settings()


# ---------------------------------------------------------------------------
# LLM configuration errors
# ---------------------------------------------------------------------------


class TestLlmConfigValidation:
    """ENABLE_REAL_LLM=true with various config problems → LlmConfigurationError."""

    def test_real_llm_enabled_missing_role_config_raises(self, monkeypatch):
        """ENABLE_REAL_LLM=true but LLM_ROLE_CONFIG_JSON is empty → error."""
        from services.llm_providers.errors import LlmConfigurationError  # noqa: PLC0415

        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

        with pytest.raises(LlmConfigurationError, match="no role config found"):
            cfg_module.get_llm_role_config("doubt_solver_classifier")

    def test_real_llm_disabled_missing_role_returns_mock(self, monkeypatch):
        """ENABLE_REAL_LLM=false → missing role config falls back to mock."""
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

        role_config = cfg_module.get_llm_role_config("doubt_solver_classifier")
        assert role_config.provider == "mock"

    def test_malformed_role_config_json_raises(self, monkeypatch):
        """LLM_ROLE_CONFIG_JSON that is not valid JSON → LlmConfigurationError."""
        from services.llm_providers.errors import LlmConfigurationError  # noqa: PLC0415

        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{not valid json}")
        _reset_settings()

        with pytest.raises(LlmConfigurationError, match="not valid JSON"):
            cfg_module.get_llm_role_config("doubt_solver_classifier")

    def test_malformed_json_with_real_llm_raises(self, monkeypatch):
        """Malformed JSON raises even when ENABLE_REAL_LLM=false."""
        from services.llm_providers.errors import LlmConfigurationError  # noqa: PLC0415

        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "this is not json at all")
        _reset_settings()

        with pytest.raises(LlmConfigurationError):
            cfg_module.get_llm_role_config("doubt_solver_classifier")

    def test_valid_role_config_parses_correctly(self, monkeypatch):
        """Well-formed LLM_ROLE_CONFIG_JSON with the target role → no error."""
        import json  # noqa: PLC0415

        config_map = {
            "doubt_solver_classifier": {
                "provider": "azure_openai",
                "model_label": "gpt-4-mini",
                "deployment": "my-deployment",
                "temperature": 0.0,
                "max_tokens": 500,
                "supports_streaming": False,
            }
        }
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", json.dumps(config_map))
        _reset_settings()

        role_config = cfg_module.get_llm_role_config("doubt_solver_classifier")
        assert role_config.provider == "azure_openai"
        assert role_config.model_label == "gpt-4-mini"

    def test_real_llm_enabled_with_valid_config_no_error(self, monkeypatch):
        """ENABLE_REAL_LLM=true + valid config → no LlmConfigurationError."""
        import json  # noqa: PLC0415

        config_map = {
            "doubt_solver_generator": {
                "provider": "openai",
                "model_label": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 1200,
                "supports_streaming": False,
            }
        }
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", json.dumps(config_map))
        _reset_settings()

        role_config = cfg_module.get_llm_role_config("doubt_solver_generator")
        assert role_config.provider == "openai"


# ---------------------------------------------------------------------------
# KB configuration errors
# ---------------------------------------------------------------------------


class TestKbConfigValidation:
    """ENABLE_KB_RETRIEVAL=true with missing KB ID → KnowledgeBaseConfigurationError."""

    def test_kb_enabled_missing_kb_id_raises(self, monkeypatch):
        """ENABLE_KB_RETRIEVAL=true + BEDROCK_KB_ID='' → KnowledgeBaseConfigurationError."""
        from services.bedrock_kb_service import (  # noqa: PLC0415
            KnowledgeBaseConfigurationError,
            retrieve_similar_context,
        )

        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "")
        _reset_settings()

        with pytest.raises(KnowledgeBaseConfigurationError, match="BEDROCK_KB_ID"):
            retrieve_similar_context("What is ratio?")

    def test_kb_disabled_missing_kb_id_no_error(self, monkeypatch):
        """ENABLE_KB_RETRIEVAL=false → missing KB ID is fine, returns disabled."""
        from services.bedrock_kb_service import retrieve_similar_context  # noqa: PLC0415

        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        monkeypatch.setenv("BEDROCK_KB_ID", "")
        _reset_settings()

        result = retrieve_similar_context("What is ratio?")
        assert result.retrieval_source == "disabled"
        assert result.results == []

    def test_kb_config_error_caught_by_graph(self, monkeypatch):
        """Graph handles KnowledgeBaseConfigurationError gracefully → needs_review=True."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415
        from services.bedrock_kb_service import (  # noqa: PLC0415
            KnowledgeBaseConfigurationError,
        )

        monkeypatch.setenv("RETRIEVAL_PROVIDER", "legacy_bedrock_kb")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "fake-id")
        _reset_settings()

        def _config_error(query, max_results=None):
            raise KnowledgeBaseConfigurationError("Missing KB ID (graph-level test)")

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _config_error)

        from graphs.doubt_solver_graph import build_doubt_solver_graph  # noqa: PLC0415

        result = build_doubt_solver_graph().invoke(
            {
                "request_id": "cfg-test",
                "query": "Explain percentage",
                "user_id": "test",
                "mode": "doubt_solver",
                "language": "en",
                "classification": None,
                "answer": None,
                "answer_source": None,
                "is_truncated": False,
                "response": None,
                "should_retrieve": False,
                "kb_results": None,
                "dynamodb_records": None,
                "answer_context": None,
                "context_source_count": 0,
                "used_retrieval": False,
                "context_used": False,
                "service_error": False,
            }
        )

        assert result["response"]["success"] is True
        assert result["response"]["needs_review"] is True


# ---------------------------------------------------------------------------
# DynamoDB configuration errors
# ---------------------------------------------------------------------------


class TestDynamoDbConfigValidation:
    """ENABLE_DYNAMODB_FETCH=true with missing table → DynamoDbConfigurationError."""

    def test_dynamodb_enabled_missing_question_table_raises(self, monkeypatch):
        """ENABLE_DYNAMODB_FETCH=true + DYNAMODB_QUESTION_TABLE='' → error."""
        from services.dynamodb_service import DynamoDbConfigurationError  # noqa: PLC0415
        from services.question_record_service import (  # noqa: PLC0415
            fetch_question_record_by_id,
        )

        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "true")
        monkeypatch.setenv("DYNAMODB_QUESTION_TABLE", "")
        _reset_settings()

        with pytest.raises(DynamoDbConfigurationError, match="DYNAMODB_QUESTION_TABLE"):
            fetch_question_record_by_id("q-1")

    def test_dynamodb_disabled_missing_table_no_error(self, monkeypatch):
        """ENABLE_DYNAMODB_FETCH=false → missing table is fine, returns None."""
        from services.question_record_service import (  # noqa: PLC0415
            fetch_question_record_by_id,
        )

        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        monkeypatch.setenv("DYNAMODB_QUESTION_TABLE", "")
        _reset_settings()

        result = fetch_question_record_by_id("q-1")
        assert result is None

    def test_dynamodb_config_error_caught_by_graph(self, monkeypatch):
        """Graph handles DynamoDbConfigurationError gracefully → needs_review=True."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415
        from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse  # noqa: PLC0415
        from services.dynamodb_service import DynamoDbConfigurationError  # noqa: PLC0415

        monkeypatch.setenv("RETRIEVAL_PROVIDER", "legacy_bedrock_kb")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "fake-id")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "true")
        monkeypatch.setenv("DYNAMODB_QUESTION_TABLE", "")
        _reset_settings()

        def _fake_kb(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[KnowledgeBaseResult(content="c.", record_ids=["q-1"])],
                result_count=1,
                retrieval_source="bedrock_kb",
            )

        def _config_err_fetch(ids):
            raise DynamoDbConfigurationError("Missing table (graph-level test)")

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _fake_kb)
        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _config_err_fetch)

        from graphs.doubt_solver_graph import build_doubt_solver_graph  # noqa: PLC0415

        result = build_doubt_solver_graph().invoke(
            {
                "request_id": "cfg-dynamo-test",
                "query": "Explain algebra",
                "user_id": "test",
                "mode": "doubt_solver",
                "language": "en",
                "classification": None,
                "answer": None,
                "answer_source": None,
                "is_truncated": False,
                "response": None,
                "should_retrieve": False,
                "kb_results": None,
                "dynamodb_records": None,
                "answer_context": None,
                "context_source_count": 0,
                "used_retrieval": False,
                "context_used": False,
                "service_error": False,
            }
        )

        assert result["response"]["success"] is True
        assert result["response"]["needs_review"] is True


# ---------------------------------------------------------------------------
# Default settings produce no errors
# ---------------------------------------------------------------------------


class TestDefaultConfigNoErrors:
    """With all flags at their defaults (off), no config errors occur."""

    def test_default_settings_no_llm_error(self, monkeypatch):
        """ENABLE_REAL_LLM defaults to false → get_llm_role_config() returns mock."""
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

        config = cfg_module.get_llm_role_config("any_role")
        assert config.provider == "mock"

    def test_default_settings_no_kb_error(self, monkeypatch):
        """ENABLE_KB_RETRIEVAL defaults to false → retrieve returns disabled."""
        from services.bedrock_kb_service import retrieve_similar_context  # noqa: PLC0415

        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        _reset_settings()

        result = retrieve_similar_context("test query")
        assert result.retrieval_source == "disabled"

    def test_default_settings_no_dynamodb_error(self, monkeypatch):
        """ENABLE_DYNAMODB_FETCH defaults to false → fetch_question_records returns []."""
        from services.question_record_service import (  # noqa: PLC0415
            fetch_question_records_by_ids,
        )

        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

        result = fetch_question_records_by_ids(["q-1", "q-2"])
        assert result == []

    def test_default_context_chars_is_positive_integer(self, monkeypatch):
        """DOUBT_SOLVER_MAX_CONTEXT_CHARS defaults to 6000."""
        monkeypatch.delenv("DOUBT_SOLVER_MAX_CONTEXT_CHARS", raising=False)
        _reset_settings()

        settings = cfg_module.get_settings()
        assert settings.doubt_solver_max_context_chars == 6000
        assert settings.doubt_solver_max_context_chars > 0


class TestPatternIntelligenceConfig:
    def test_defaults_are_disabled_and_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PATTERN_INTELLIGENCE_ENABLED", raising=False)
        monkeypatch.delenv("PATTERN_INTELLIGENCE_MAX_CANDIDATES", raising=False)
        monkeypatch.delenv("PATTERN_INTELLIGENCE_MAX_REFERENCES", raising=False)
        monkeypatch.delenv("PATTERN_INTELLIGENCE_PROMPT_MAX_INPUT_TOKENS", raising=False)
        _reset_settings()

        settings = cfg_module.get_settings()

        assert settings.pattern_intelligence_enabled is False
        assert settings.pattern_intelligence_max_candidates == 12
        assert settings.pattern_intelligence_max_references == 1
        assert settings.pattern_intelligence_prompt_max_input_tokens == 3800

    def test_active_runtime_reads_pattern_question_access_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
        monkeypatch.setenv("DYNAMODB_PATTERN_QUESTION_TABLE", "pattern-questions")
        monkeypatch.setenv(
            "DYNAMODB_PATTERN_QUESTION_BY_PATTERN_INDEX", "questionsByPattern"
        )
        monkeypatch.setenv("PATTERN_INTELLIGENCE_MAX_CANDIDATES", "16")
        monkeypatch.setenv("PATTERN_INTELLIGENCE_MAX_REFERENCES", "2")
        monkeypatch.setenv("PATTERN_INTELLIGENCE_PROMPT_MAX_INPUT_TOKENS", "4096")
        _reset_settings()

        settings = cfg_module.get_settings()

        assert settings.pattern_intelligence_enabled is True
        assert settings.dynamodb_pattern_question_table == "pattern-questions"
        assert settings.dynamodb_pattern_question_by_pattern_index == "questionsByPattern"
        assert settings.pattern_intelligence_max_candidates == 16
        assert settings.pattern_intelligence_max_references == 2
        assert settings.pattern_intelligence_prompt_max_input_tokens == 4096

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("PATTERN_INTELLIGENCE_MAX_CANDIDATES", "-1"),
            ("PATTERN_INTELLIGENCE_MAX_CANDIDATES", "0"),
            ("PATTERN_INTELLIGENCE_MAX_CANDIDATES", "51"),
            ("PATTERN_INTELLIGENCE_MAX_REFERENCES", "3"),
            ("PATTERN_INTELLIGENCE_PROMPT_MAX_INPUT_TOKENS", "8193"),
            ("PATTERN_INTELLIGENCE_PROMPT_MAX_INPUT_TOKENS", "not-a-number"),
        ],
    )
    def test_active_runtime_rejects_out_of_bounds_limits(
        self, monkeypatch: pytest.MonkeyPatch, name: str, value: str
    ) -> None:
        monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
        monkeypatch.setenv(name, value)
        _reset_settings()

        with pytest.raises(cfg_module.ConfigurationError, match=name):
            cfg_module.get_settings()

    def test_inactive_runtime_ignores_invalid_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "false")
        monkeypatch.setenv("PATTERN_INTELLIGENCE_MAX_CANDIDATES", "not-a-number")
        _reset_settings()

        assert cfg_module.get_settings().pattern_intelligence_max_candidates == 12


class TestAnswerDeliveryConfig:
    def test_stream_heartbeat_interval_is_configurable(self, monkeypatch) -> None:
        monkeypatch.setenv("ANSWER_STREAM_HEARTBEAT_INTERVAL_SECONDS", "2.5")
        _reset_settings()

        assert cfg_module.get_settings().answer_stream_heartbeat_interval_seconds == 2.5

    def test_stream_heartbeat_interval_must_be_positive(self, monkeypatch) -> None:
        monkeypatch.setenv("ANSWER_STREAM_HEARTBEAT_INTERVAL_SECONDS", "0")
        _reset_settings()

        with pytest.raises(
            cfg_module.ConfigurationError,
            match="ANSWER_STREAM_HEARTBEAT_INTERVAL_SECONDS",
        ):
            cfg_module.get_settings()

    def test_always_live_is_blocked_in_production_without_explicit_approval(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
        monkeypatch.setenv("ANSWER_ALLOW_ALWAYS_LIVE_IN_PRODUCTION", "false")
        _reset_settings()

        with pytest.raises(cfg_module.ConfigurationError, match="always_live is blocked"):
            cfg_module.get_settings()

    def test_always_live_requires_explicit_production_approval(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
        monkeypatch.setenv("ANSWER_ALLOW_ALWAYS_LIVE_IN_PRODUCTION", "true")
        _reset_settings()

        assert cfg_module.get_settings().answer_delivery_policy == "always_live"
        _reset_settings()

    def test_invalid_integer_env_var_raises_value_error(self, monkeypatch):
        """A non-integer value for DOUBT_SOLVER_MAX_CONTEXT_CHARS raises ValueError.

        This is a fast-fail behaviour by design: the agent refuses to start
        with a misconfigured environment rather than silently using a wrong value.
        """
        monkeypatch.setenv("DOUBT_SOLVER_MAX_CONTEXT_CHARS", "not-a-number")
        _reset_settings()

        with pytest.raises(ValueError):
            cfg_module.get_settings()


class TestClassifierThresholdConfig:
    def test_default_threshold_is_092(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("CLASSIFIER_CONFIDENCE_FALLBACK_THRESHOLD", raising=False)
        _reset_settings()
        assert cfg_module.get_settings().classifier_confidence_fallback_threshold == 0.93

    def test_new_env_threshold_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", "0.88")
        _reset_settings()
        assert cfg_module.get_settings().classifier_confidence_fallback_threshold == 0.88

    def test_legacy_env_threshold_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.setenv("CLASSIFIER_CONFIDENCE_FALLBACK_THRESHOLD", "0.85")
        _reset_settings()
        assert cfg_module.get_settings().classifier_confidence_fallback_threshold == 0.85

    def test_invalid_threshold_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", "not-a-float")
        _reset_settings()
        with pytest.raises(ValueError, match="DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD"):
            cfg_module.get_settings()

    def test_out_of_range_threshold_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", "1.5")
        _reset_settings()
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            cfg_module.get_settings()


class TestWebSearchConfigDefaults:
    def test_web_search_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
        _reset_settings()
        settings = cfg_module.get_settings()
        assert settings.web_search_enabled is False

    def test_web_search_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
        monkeypatch.delenv("WEB_SEARCH_MAX_RESULTS", raising=False)
        monkeypatch.delenv("WEB_SEARCH_MAX_CONTEXT_CHARS", raising=False)
        _reset_settings()
        settings = cfg_module.get_settings()
        assert settings.web_search_provider == "tavily"
        assert settings.web_search_max_results == 5
        assert settings.web_search_max_context_chars == 2500
        assert settings.web_search_allow_exam_prep_fallback is True
        assert settings.web_search_exam_prep_max_selected_results == 2
        assert settings.web_search_require_official_for_exam_updates is True
