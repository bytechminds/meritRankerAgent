"""M4: a production failure line must name its own root-cause class.

Details are suppressed outside DEBUG, which previously reduced a failure to its event
name. These tests pin that the classification survives while healthy lines stay small
and no student content is ever promoted.
"""

from __future__ import annotations

import logging

import pytest

from observability.events import log_event


@pytest.fixture(autouse=True)
def _production_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin production mode: details suppressed, so only promotion can surface them."""
    from observability import events

    monkeypatch.setattr(events, "_detailed_logs", False)
    monkeypatch.setattr(events._logger, "level", logging.INFO)


def _line(caplog: pytest.LogCaptureFixture) -> str:
    return caplog.records[-1].getMessage()


@pytest.mark.parametrize(
    ("event", "reason", "expected"),
    [
        ("planner_validation_failed", "PLANNER_SCHEMA_TYPE_MISMATCH",
         "PLANNER_SCHEMA_TYPE_MISMATCH"),
        ("planner_validation_failed", "PLANNER_TOPIC_COVERAGE_INVALID",
         "PLANNER_TOPIC_COVERAGE_INVALID"),
        ("planner_validation_failed", "PLANNER_TOPIC_EVIDENCE_UNGROUNDED",
         "PLANNER_TOPIC_EVIDENCE_UNGROUNDED"),
        ("planner_validation_failed", "PLANNER_SLOT_COUNT_MISMATCH",
         "PLANNER_SLOT_COUNT_MISMATCH"),
        ("practice_model_execution_failed", "PRACTICE_GENERATOR_FALLBACK_EXHAUSTED",
         "PRACTICE_GENERATOR_FALLBACK_EXHAUSTED"),
    ],
)
def test_each_failure_class_is_distinguishable_in_the_log_line(
    caplog: pytest.LogCaptureFixture, event: str, reason: str, expected: str
) -> None:
    with caplog.at_level(logging.WARNING, logger="agent.observability"):
        log_event(
            event, component="practice.planning", stage="validate", status="failed",
            details={"reasonCode": reason, "routeId": "quant_reasoning.planner.advanced",
                     "modelAlias": "openai_gpt_4_1", "attempt": 1},
            level=logging.WARNING,
        )

    assert expected in _line(caplog)


def test_provider_timeout_and_token_exhaustion_are_distinguishable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="agent.observability"):
        log_event(
            "practice_model_execution_failed", component="practice.generation",
            stage="generate", status="failed",
            details={"errorClass": "TimeoutError", "failureStage": "GENERATOR"},
            level=logging.WARNING,
        )
        timeout_line = _line(caplog)
        log_event(
            "practice_model_execution_failed", component="practice.generation",
            stage="generate", status="failed",
            details={"errorClass": "ProviderExecutionError",
                     "failureStage": "GENERATOR", "finishReason": "length"},
            level=logging.WARNING,
        )
        exhaustion_line = _line(caplog)

    assert "TimeoutError" in timeout_line
    assert "finishreason=length" in exhaustion_line
    assert timeout_line != exhaustion_line


def test_the_missing_cost_profile_is_named(caplog: pytest.LogCaptureFixture) -> None:
    """M3 is only actionable when the log says which profile is uncertified."""
    with caplog.at_level(logging.WARNING, logger="agent.observability"):
        log_event(
            "COST_PROFILE_MISSING", component="llm.billing", stage="complete",
            status="incomplete",
            details={"missing_cost_profiles": "azure_openai:gpt-4.1",
                     "billing_config_version": "v1"},
            level=logging.WARNING,
        )

    assert "azure_openai:gpt-4.1" in _line(caplog)


def test_healthy_info_lines_are_not_enlarged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="agent.observability"):
        log_event(
            "practice_planning_route_selected", component="practice.planning",
            stage="route", status="selected",
            details={"planningMode": "INTELLIGENCE", "requestedCount": 50},
        )

    line = _line(caplog)
    assert "INTELLIGENCE" not in line
    assert "requestedCount" not in line


def test_student_content_is_never_promoted(caplog: pytest.LogCaptureFixture) -> None:
    """Only the allowlisted classification keys may reach the log line."""
    with caplog.at_level(logging.WARNING, logger="agent.observability"):
        log_event(
            "planner_validation_failed", component="practice.planning",
            stage="validate", status="failed",
            details={
                "reasonCode": "PLANNER_SCHEMA_TYPE_MISMATCH",
                "question": "What is 15% of 200?",
                "prompt": "secret system prompt",
                "answer": "30",
                "originalQuery": "मुझे प्रतिशत के सवाल दो",
            },
            level=logging.WARNING,
        )

    line = _line(caplog)
    assert "PLANNER_SCHEMA_TYPE_MISMATCH" in line
    for leaked in ("15%", "secret system prompt", "मुझे", "originalQuery"):
        assert leaked not in line


def test_promotion_is_bounded(caplog: pytest.LogCaptureFixture) -> None:
    from observability.events import (
        _FAILURE_DIAGNOSTIC_KEYS,
        _MAX_PROMOTED_DIAGNOSTICS,
    )

    with caplog.at_level(logging.WARNING, logger="agent.observability"):
        log_event(
            "planner_validation_failed", component="practice.planning",
            stage="validate", status="failed",
            details={key: f"value-{key}" for key in _FAILURE_DIAGNOSTIC_KEYS},
            level=logging.WARNING,
        )

    line = _line(caplog)
    promoted = sum(1 for key in _FAILURE_DIAGNOSTIC_KEYS if f"{key}=" in line)
    assert promoted == _MAX_PROMOTED_DIAGNOSTICS
