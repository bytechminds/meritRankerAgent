"""Async Practice operation-scope observability tests."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from observability import log_event
from observability.context import bind_request_context
from observability.events import configure_event_metadata
from observability.logging import configure_logging
from observability.readable_log import (
    begin_operation_log,
    configure_readable_request_log,
    finalize_operation_log,
    reset_operation_log,
    reset_readable_log_for_tests,
)
from observability.summary import begin_request_summary, reset_request_summary
from tools.inspect_agent_logs import load_operation_lines, select_operation_lines


@pytest.fixture
def operation_log(tmp_path: Path):
    path = tmp_path / "agent-runtime.log"

    def _configure(*, detailed: bool = True, enabled: bool = True) -> Path:
        reset_readable_log_for_tests()
        configure_readable_request_log(
            enabled=enabled,
            file_path=str(path),
            detailed=detailed,
        )
        return path

    yield _configure
    reset_readable_log_for_tests()
    configure_event_metadata(environment="test", detailed_logs=False)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _run_operation(
    *,
    test_id: str,
    status: str = "READY",
    error_code: str | None = None,
    events: int = 2,
) -> None:
    token = begin_operation_log(
        operation_id=test_id,
        test_id=test_id,
        request_id=f"req-{test_id}",
        execution_id=f"exec-{test_id}",
    )
    try:
        for index in range(events):
            log_event(
                "practice_graph_started",
                component="practice_generation",
                stage="practice",
                status="started",
                details={"slotOrdinal": index, "testId": test_id},
            )
        finalize_operation_log(status=status, error_code=error_code)
    finally:
        reset_operation_log(token)


def test_background_events_reach_the_file_after_the_request_completed(
    operation_log,
) -> None:
    path = operation_log()
    with bind_request_context(request_id="req-1", request_type="standalone"):
        summary_token = begin_request_summary()
        reset_request_summary(summary_token)
    # The parent request scope is now fully closed and flushed.
    _run_operation(test_id="practice-after-request")

    content = _read(path)
    assert "[operation] " in content
    assert "practice-after-request" in content
    assert "[operation-summary] " in content


def test_request_completion_never_finalizes_an_active_operation(
    operation_log,
) -> None:
    path = operation_log()
    token = begin_operation_log(
        operation_id="practice-live",
        test_id="practice-live",
        request_id="req-live",
        execution_id="exec-live",
    )
    try:
        with bind_request_context(request_id="req-live", request_type="standalone"):
            summary_token = begin_request_summary()
            reset_request_summary(summary_token)
        log_event(
            "practice_graph_started",
            component="practice_generation",
            stage="practice",
            status="started",
        )
        assert "[operation-summary] " not in _read(path)
        finalize_operation_log(status="READY")
    finally:
        reset_operation_log(token)

    content = _read(path)
    assert content.count("[operation-summary] ") == 1
    assert "practice-live" in content


def test_full_operation_timeline_is_reconstructable(operation_log) -> None:
    path = operation_log()
    _run_operation(test_id="practice-timeline", events=50)

    timeline = select_operation_lines(
        load_operation_lines(path),
        operation_id="practice-timeline",
    )
    assert len(timeline) >= 50
    assert timeline[0].execution_id == "exec-practice-timeline"
    assert timeline[-1].summary is True


def test_concurrent_operations_never_leak_identity(operation_log) -> None:
    path = operation_log()
    errors: list[BaseException] = []

    def _worker(test_id: str) -> None:
        try:
            _run_operation(test_id=test_id, events=25)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=(f"practice-concurrent-{index}",))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    lines = load_operation_lines(path)
    for index in range(4):
        operation_id = f"practice-concurrent-{index}"
        selected = select_operation_lines(lines, operation_id=operation_id)
        assert all(line.operation_id == operation_id for line in selected)
        assert all(line.test_id == operation_id for line in selected)
        assert sum(1 for line in selected if line.summary) == 1


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        ("READY", None),
        ("FAILED", "AUTHORITATIVE_COUNTER_MISMATCH"),
        ("CANCELLED", "USER_CANCELLED"),
    ],
)
def test_terminal_summary_is_emitted_exactly_once(
    operation_log,
    status: str,
    error_code: str | None,
) -> None:
    path = operation_log()
    _run_operation(test_id="practice-terminal", status=status, error_code=error_code)
    # A second finalize for the same operation must be ignored.
    token = begin_operation_log(operation_id="practice-terminal", test_id="practice-terminal")
    try:
        finalize_operation_log(status=status, error_code=error_code)
        finalize_operation_log(status=status, error_code=error_code)
    finally:
        reset_operation_log(token)

    content = _read(path)
    summaries = [line for line in content.splitlines() if line.startswith("[operation-summary] ")]
    assert len(summaries) == 2
    assert f"status={status}" in summaries[0]
    if error_code:
        assert error_code in summaries[0]


def test_detailed_logs_disabled_suppresses_per_event_detail(operation_log) -> None:
    path = operation_log(detailed=False)
    _run_operation(test_id="practice-concise", events=30)

    content = _read(path)
    assert "[operation] " not in content
    assert content.count("[operation-summary] ") == 1


def test_disabled_local_file_writes_nothing(operation_log) -> None:
    path = operation_log(enabled=False)
    _run_operation(test_id="practice-production", events=10)

    assert _read(path) == ""


def test_operation_lines_never_contain_content_or_secrets(operation_log) -> None:
    path = operation_log()
    token = begin_operation_log(operation_id="practice-safe", test_id="practice-safe")
    try:
        log_event(
            "practice_graph_started",
            component="practice_generation",
            stage="practice",
            status="started",
            details={
                "question": "What is 2 + 2?",
                "answer": "4",
                "prompt": "SYSTEM PROMPT",
                "authorization": "Bearer secret-token",
                "slotId": "slot-001",
            },
        )
        finalize_operation_log(status="READY")
    finally:
        reset_operation_log(token)

    content = _read(path)
    assert "slot-001" in content
    for forbidden in ("What is 2 + 2?", "SYSTEM PROMPT", "Bearer", "secret-token"):
        assert forbidden not in content


def test_request_readable_block_behaviour_is_unchanged(operation_log) -> None:
    path = operation_log()
    with bind_request_context(request_id="req-doubt", request_type="standalone"):
        summary_token = begin_request_summary()
        log_event(
            "classification_completed",
            component="doubt_solver",
            stage="understanding",
            status="completed",
        )
        log_event(
            "practice_graph_started",
            component="practice_generation",
            stage="practice",
            status="started",
        )
        reset_request_summary(summary_token)

    content = _read(path)
    # No operation scope was open, so no operation lines may appear at all.
    assert "[operation] " not in content
    assert "[operation-summary] " not in content


def test_inspector_returns_only_the_requested_operation(operation_log) -> None:
    path = operation_log()
    _run_operation(test_id="practice-alpha", events=5)
    _run_operation(test_id="practice-beta", events=5)

    lines = load_operation_lines(path)
    alpha = select_operation_lines(lines, operation_id="practice-alpha")
    assert alpha
    assert all(line.operation_id == "practice-alpha" for line in alpha)
    assert not any(line.operation_id == "practice-beta" for line in alpha)

    by_execution = select_operation_lines(lines, execution_id="exec-practice-beta")
    assert by_execution
    assert all(line.test_id == "practice-beta" for line in by_execution)


def test_operation_buffer_memory_stays_bounded_for_long_operations(
    operation_log,
) -> None:
    operation_log(detailed=False)
    token = begin_operation_log(operation_id="practice-long", test_id="practice-long")
    try:
        from observability.readable_log import current_operation_log

        for index in range(5000):
            log_event(
                "practice_graph_started",
                component="practice_generation",
                stage="practice",
                status="started",
                details={"slotOrdinal": index},
            )
        buffer = current_operation_log()
        assert buffer is not None
        assert buffer.event_count == 5000
        # Only aggregates are retained, never the per-event payloads.
        assert len(buffer.aggregates) <= 24
        finalize_operation_log(status="READY")
    finally:
        reset_operation_log(token)


@pytest.mark.parametrize(
    ("key", "value", "retained"),
    [
        ("answer_tokens", 123, True),
        ("answerTokens", 123, True),
        ("reasoningTokens", 456, True),
        ("outputTokens", 789, True),
        ("input_tokens", 111, True),
        ("cached_input_tokens", 222, True),
        ("reasoning_tokens", 333, True),
        ("answer", "the answer is 4", False),
        ("answer_text", "leaked", False),
        ("correct_answer", "B", False),
        ("student_answer", "C", False),
        ("prompt", "SYSTEM", False),
        ("prompt_tokens", 999, False),
        ("arbitrary_token_payload", "sk-secret", False),
        ("answer_tokens", "not-a-number", False),
        ("answer_tokens", True, False),
    ],
)
def test_numeric_usage_allowlist_is_exact_and_numeric_only(
    key: str,
    value: object,
    retained: bool,
) -> None:
    from observability.events import sanitize_details

    result = sanitize_details({key: value})
    assert (key.strip().lower() in result) is retained
    if not retained:
        assert str(value) not in str(result)


def test_terminal_error_code_is_visible_without_detailed_logs(caplog) -> None:
    configure_event_metadata(environment="prod", detailed_logs=False)
    with caplog.at_level(logging.ERROR, logger="agent.observability"):
        log_event(
            "practice_failed",
            component="practice_generation",
            stage="practice",
            status="failed",
            error_code="VERIFIER_OUTPUT_EXHAUSTED",
            level=logging.ERROR,
        )
    assert "error_code=VERIFIER_OUTPUT_EXHAUSTED" in caplog.text


def test_terminal_console_line_hides_raw_exception_content(caplog) -> None:
    configure_event_metadata(environment="prod", detailed_logs=False)
    with caplog.at_level(logging.ERROR, logger="agent.observability"):
        log_event(
            "practice_failed",
            component="practice_generation",
            stage="practice",
            status="failed",
            error_code="PRACTICE_GENERATOR_PROVIDER_FAILED",
            details={
                "exceptionMessage": "boto3 ClientError: secret-arn-12345 denied",
                "question": "What is 2 + 2?",
            },
            level=logging.ERROR,
        )
    assert "error_code=PRACTICE_GENERATOR_PROVIDER_FAILED" in caplog.text
    for forbidden in ("secret-arn-12345", "What is 2 + 2?", "boto3 ClientError"):
        assert forbidden not in caplog.text


def test_success_events_never_gain_a_fabricated_error_code(caplog) -> None:
    configure_event_metadata(environment="prod", detailed_logs=False)
    with caplog.at_level(logging.INFO, logger="agent.observability"):
        log_event(
            "practice_ready",
            component="practice_generation",
            stage="practice",
            status="completed",
        )
    assert "error_code=" not in caplog.text


def test_info_level_events_do_not_gain_error_codes(caplog) -> None:
    configure_event_metadata(environment="prod", detailed_logs=False)
    with caplog.at_level(logging.INFO, logger="agent.observability"):
        log_event(
            "practice_graph_started",
            component="practice_generation",
            stage="practice",
            status="started",
            error_code="SOME_INFO_CODE",
        )
    assert "error_code=" not in caplog.text


@pytest.mark.parametrize(
    ("log_level", "detailed_env", "expect_detail"),
    [
        ("DEBUG", False, True),
        ("INFO", True, True),
        ("INFO", False, False),
        ("WARNING", False, False),
    ],
)
def test_debug_level_is_the_documented_operation_detail_control(
    tmp_path: Path,
    log_level: str,
    detailed_env: bool,
    expect_detail: bool,
) -> None:
    """docs/dev/backend-env.md + agent-observability.md make LEVEL the control.

    AGENT_DETAILED_LOGS remains a supported opt-in for the same detail, but must
    not be required in order for DEBUG to produce an operation timeline.
    """
    import logging as _logging

    from observability import readable_log as _readable_log
    from observability.logging import reset_logging_for_tests

    reset_logging_for_tests()
    reset_readable_log_for_tests()
    try:
        configure_logging(
            log_level,
            environment="local",
            log_format="pretty_and_json_file",
            file_enabled=True,
            file_path=str(tmp_path / "agent-runtime.log"),
            detailed_logs=detailed_env,
        )
        assert _readable_log._operation_detail_enabled is expect_detail
        del _logging
    finally:
        reset_logging_for_tests()
        reset_readable_log_for_tests()
        configure_event_metadata(environment="test", detailed_logs=False)


def test_production_debug_level_cannot_enable_the_operation_timeline(
    tmp_path: Path,
) -> None:
    """Production forcibly disables the local file, so DEBUG cannot leak a timeline."""
    from observability import readable_log as _readable_log
    from observability.logging import reset_logging_for_tests

    reset_logging_for_tests()
    reset_readable_log_for_tests()
    try:
        configure_logging(
            "DEBUG",
            environment="production",
            file_path=str(tmp_path / "agent-runtime.log"),
            detailed_logs=True,
        )
        assert _readable_log._handler is None
    finally:
        reset_logging_for_tests()
        reset_readable_log_for_tests()
        configure_event_metadata(environment="test", detailed_logs=False)
