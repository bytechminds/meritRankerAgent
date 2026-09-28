"""Dev stays verbose; APP_ENV=production (Prod and PreProd-local) is JSON, INFO, file-free."""

from __future__ import annotations

import json
import logging

import pytest

import config as cfg_module
from observability import log_event
from observability.logging import JsonEventFormatter, configure_logging, reset_logging_for_tests

QUERY = "SENTINEL-STUDENT-QUERY what is 17 x 23"
ANSWER = "SENTINEL-GENERATED-ANSWER 391"
API_KEY = "SENTINEL-PROVIDER-KEY-abc123"


@pytest.fixture(autouse=True)
def _fresh_logging():
    reset_logging_for_tests()
    yield
    reset_logging_for_tests()


def test_dev_logging_remains_verbose_with_local_event_file(tmp_path) -> None:
    configure_logging(
        "DEBUG",
        environment="local",
        file_enabled=True,
        file_path=str(tmp_path / ".logs" / "agent-runtime.log"),
    )
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    handler_types = {type(handler).__name__ for handler in root.handlers}
    assert "RichHandler" in handler_types
    assert "SafeRequestBlockHandler" in handler_types


def test_production_forces_json_info_and_no_local_file(tmp_path) -> None:
    configure_logging(
        "INFO",
        environment="production",
        log_format="pretty_and_json_file",
        file_enabled=True,
        file_path=str(tmp_path / ".logs" / "agent-runtime.log"),
        local_log_content="full",
    )
    handlers = logging.getLogger().handlers
    assert [type(h).__name__ for h in handlers] == ["StreamHandler"]
    assert isinstance(handlers[0].formatter, JsonEventFormatter)
    log_event("runtime_ready", component="test", stage="startup", status="ready")
    assert not (tmp_path / ".logs").exists()


def test_production_settings_cannot_run_debug_or_write_local_content(monkeypatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("AGENT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AGENT_LOCAL_LOG_CONTENT", "full")
    monkeypatch.delenv("AGENT_LOG_FILE_ENABLED", raising=False)
    cfg_module._settings = None
    settings = cfg_module.get_settings()
    assert settings.log_level == "INFO"
    assert settings.agent_local_log_content == "off"
    assert settings.agent_log_file_enabled is False


def test_production_events_keep_correlation_but_never_content_or_credentials(capsys) -> None:
    configure_logging("INFO", environment="production")
    details = {
        "query": QUERY,
        "normalized_query": QUERY,
        "answer": ANSWER,
        "final_answer_preview": ANSWER,
        "prompt": QUERY,
        "api_key": API_KEY,
        "authorization": f"Bearer {API_KEY}",
        "conversation_id": "conv-123",
        "turn_id": "turn-456",
        "test_id": "practice-789",
        "route": "math",
        "provider": "azure_openai",
    }
    log_event("llm_usage_summary", component="test", status="completed", details=details)
    log_event(
        "generator_model_attempt_failed",
        component="test",
        status="failed",
        error_code="PROVIDER_RATE_LIMITED",
        details=details,
        level=logging.WARNING,
    )
    log_event("generator_model_attempt_started", component="test", details=details, level=logging.DEBUG)

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    assert [event["event"] for event in events] == ["llm_usage_summary", "generator_model_attempt_failed"]
    rendered = "\n".join(lines)
    for sentinel in (QUERY, ANSWER, API_KEY, "SENTINEL"):
        assert sentinel not in rendered
    assert events[0]["details"]["conversation_id"] == "conv-123"
    assert events[0]["details"]["turn_id"] == "turn-456"
    assert events[0]["details"]["test_id"] == "practice-789"
    assert events[1]["error_code"] == "PROVIDER_RATE_LIMITED"
    assert events[1]["level"] == "WARNING"


def test_production_redacts_plain_logger_text_and_tracebacks(capsys) -> None:
    configure_logging("INFO", environment="production")
    try:
        raise RuntimeError(f"provider echoed {QUERY} with {API_KEY}")
    except RuntimeError as exc:
        logging.getLogger("services.example").warning("classifier failed: %s", exc, exc_info=True)
    out = capsys.readouterr().out
    assert json.loads(out.strip())["details"] == {"message_redacted": True}
    assert "SENTINEL" not in out
