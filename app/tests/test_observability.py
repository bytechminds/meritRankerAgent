from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from observability.context import (
    bind_request_context,
    current_request_context,
    update_request_type,
)
from observability.events import EVENT_NAMES, build_event, log_event, sanitize_details
from observability.lifecycle import observe_invocation
from observability.logging import JsonEventFormatter
from observability.summary import (
    begin_request_summary,
    current_request_summary,
    emit_request_summary,
    reset_request_summary,
    update_request_summary,
)
from observability.tracing import configure_tracing, stage_span, tracing_available
from tools.inspect_agent_logs import DIVIDER, load_blocks, main, select_blocks


def test_event_contract_contains_required_events() -> None:
    assert {
        "request_started",
        "request_execution_summary",
        "request_completed",
        "request_failed",
        "request_cancelled",
        "classification_completed",
        "follow_up_context_loaded",
        "retrieval_completed",
        "generation_completed",
        "quality_validation_completed",
        "conversation_persistence_completed",
    } <= EVENT_NAMES


def test_structured_event_has_stable_top_level_fields() -> None:
    with bind_request_context(
        request_id="request-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
    ):
        event = build_event(
            "request_completed",
            component="test",
            status="completed",
            duration_ms=4,
        )

    assert set(event) == {
        "timestamp",
        "level",
        "service",
        "environment",
        "event",
        "component",
        "request_id",
        "trace_id",
        "conversation_id",
        "turn_id",
        "stage",
        "status",
        "duration_ms",
        "error_code",
        "details",
    }
    assert event["request_id"] == "request-1"


def test_sensitive_detail_keys_are_removed() -> None:
    details = sanitize_details(
        {
            "prompt": "PRIVATE PROMPT",
            "query": "PRIVATE QUERY",
            "conversation_context": "PRIVATE CONTEXT",
            "api_key": "PRIVATE KEY",
            "authorization": "PRIVATE AUTH",
            "jwt": "PRIVATE JWT",
            "aws_secret_access_key": "PRIVATE AWS SECRET",
            "user_id": "PRIVATE USER",
            "subject": "math",
        }
    )

    serialized = json.dumps(details)
    assert details == {"subject": "math"}
    assert "PRIVATE" not in serialized


def test_non_structured_json_log_omits_message() -> None:
    record = logging.LogRecord(
        "unsafe.logger",
        logging.INFO,
        __file__,
        1,
        "PRIVATE STUDENT QUERY",
        (),
        None,
    )
    encoded = JsonEventFormatter(environment="local").format(record)

    assert "PRIVATE STUDENT QUERY" not in encoded
    assert json.loads(encoded)["details"] == {"message_redacted": True}


def test_json_formatter_has_safe_serialization_fallback() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "safe", (), None)
    record.observability_event = {"not_serializable": object()}

    encoded = JsonEventFormatter(environment="local").format(record)

    payload = json.loads(encoded)
    assert payload["error_code"] == "LOG_SERIALIZATION_FAILED"


def test_context_is_cleared_after_scope() -> None:
    with bind_request_context(request_id="request-1"):
        assert current_request_context() is not None

    assert current_request_context() is None


def test_request_type_can_be_refined_without_changing_identity() -> None:
    with bind_request_context(request_id="request-1", request_type="unknown"):
        before = current_request_context()
        after = update_request_type("follow_up")

    assert after is not None
    assert after.request_id == before.request_id
    assert after.trace_id == before.trace_id
    assert after.request_type == "follow_up"


def test_async_contexts_do_not_leak_between_requests() -> None:
    async def read_identity(request_id: str) -> tuple[str, str]:
        with bind_request_context(request_id=request_id):
            await asyncio.sleep(0)
            context = current_request_context()
            return context.request_id, context.trace_id

    async def run_concurrently() -> tuple[tuple[str, str], tuple[str, str]]:
        first, second = await asyncio.gather(
            read_identity("request-a"), read_identity("request-b")
        )
        return first, second

    first, second = asyncio.run(run_concurrently())

    assert first[0] == "request-a"
    assert second[0] == "request-b"
    assert first[1] != second[1]


def test_context_is_explicitly_copied_to_worker() -> None:
    with bind_request_context(request_id="request-worker"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(copy_context().run, current_request_context)
            worker_context = future.result()

    assert worker_context is not None
    assert worker_context.request_id == "request-worker"


def test_summary_updates_replace_frozen_value() -> None:
    with bind_request_context(request_id="request-1"):
        token = begin_request_summary()
        original = current_request_summary()
        updated = update_request_summary(subject="math", terminal_status="completed")
        try:
            assert updated is not original
            assert original.subject is None
            assert updated.subject == "math"
            with pytest.raises(FrozenInstanceError):
                updated.subject = "reasoning"
        finally:
            reset_request_summary(token)


def test_summary_rejects_unbounded_unknown_fields() -> None:
    with bind_request_context(request_id="request-1"):
        token = begin_request_summary()
        try:
            with pytest.raises(ValueError, match="Unsupported request summary fields"):
                update_request_summary(raw_prompt="private")
        finally:
            reset_request_summary(token)


@pytest.mark.parametrize(
    "changes",
    [
        {
            "request_type": "standalone",
            "terminal_status": "completed",
            "terminal_reason": "completed",
        },
        {
            "request_type": "follow_up",
            "context_source": "dynamodb_fallback",
            "usable_recent_turns": 2,
            "terminal_status": "completed",
            "terminal_reason": "completed",
        },
        {
            "request_type": "follow_up",
            "context_source": "none",
            "memory_write_status": "failed_transient",
            "terminal_status": "completed",
            "terminal_reason": "clarification",
        },
        {
            "quality_status": "failed_quality_gate",
            "terminal_status": "failed",
            "terminal_reason": "quality_failure",
        },
        {
            "terminal_status": "cancelled",
            "terminal_reason": "client_disconnected",
        },
        {
            "history_write_status": "succeeded",
            "session_write_status": "failed_transient",
            "memory_write_status": "succeeded",
            "terminal_status": "completed",
            "terminal_reason": "completed",
        },
        {
            "terminal_status": "failed",
            "terminal_reason": "unexpected_exception",
        },
    ],
)
def test_summary_represents_terminal_operational_scenarios(
    changes: dict[str, object],
) -> None:
    with bind_request_context(
        request_id="request-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
    ):
        token = begin_request_summary()
        try:
            summary = update_request_summary(**changes)
        finally:
            reset_request_summary(token)

    assert summary is not None
    for field, expected in changes.items():
        assert getattr(summary, field) == expected


def test_summary_emits_once_when_called_once(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="agent.observability"):
        with bind_request_context(request_id="request-1"):
            token = begin_request_summary()
            update_request_summary(
                subject="math",
                terminal_status="completed",
                total_duration_ms=12,
            )
            try:
                emit_request_summary()
            finally:
                reset_request_summary(token)

    records = [
        record
        for record in caplog.records
        if getattr(record, "observability_event", {}).get("event")
        == "request_execution_summary"
    ]
    assert len(records) == 1
    assert "prompt" not in json.dumps(records[0].observability_event)


def test_log_event_attaches_sanitized_payload(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="agent.observability"):
        with bind_request_context(request_id="request-1"):
            log_event(
                "generation_completed",
                component="test",
                details={"subject": "math", "query": "PRIVATE QUERY"},
            )

    payload = caplog.records[-1].observability_event
    assert payload["details"] == {"subject": "math"}
    assert "PRIVATE QUERY" not in json.dumps(payload)
    assert "request_id=request-" in caplog.records[-1].message
    assert "request_id=request-1" not in caplog.records[-1].message


def test_tracing_is_safe_noop_when_sdk_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import observability.tracing as tracing_module

    monkeypatch.setattr(tracing_module, "_otel_trace", None)
    configure_tracing(enabled=True)
    with stage_span("doubt_solver.generate", subject="math") as span:
        assert span is None
    assert tracing_available() is False


def test_tracing_is_noop_when_disabled() -> None:
    configure_tracing(enabled=False)
    with stage_span("doubt_solver.generate") as span:
        assert span is None


def test_tracing_creates_safe_attributes_and_suppresses_export_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import observability.tracing as tracing_module

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key: str, value: object) -> None:
            self.attributes[key] = value

        def get_span_context(self):
            return type(
                "SpanContext",
                (),
                {"is_valid": True, "trace_id": int("1" * 32, 16)},
            )()

    class FakeManager:
        def __init__(self) -> None:
            self.span = FakeSpan()

        def __enter__(self) -> FakeSpan:
            return self.span

        def __exit__(self, *args: object) -> None:
            raise RuntimeError("export failed")

    class FakeTracer:
        def __init__(self) -> None:
            self.name = ""
            self.manager = FakeManager()

        def start_as_current_span(self, name: str) -> FakeManager:
            self.name = name
            return self.manager

    fake_tracer = FakeTracer()

    class FakeTrace:
        @staticmethod
        def get_tracer(name: str) -> FakeTracer:
            assert name == "meritranker-tutor"
            return fake_tracer

    monkeypatch.setattr(tracing_module, "_otel_trace", FakeTrace())
    configure_tracing(enabled=True)
    with stage_span(
        "doubt_solver.generate",
        subject="math",
        query="PRIVATE QUERY",
        prompt="PRIVATE PROMPT",
    ):
        pass

    assert fake_tracer.name == "doubt_solver.generate"
    assert fake_tracer.manager.span.attributes == {"agent.subject": "math"}


def test_production_logging_is_json_fileless_idempotent_and_quiets_dependencies(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "must-not-exist.jsonl"
    script = f"""
import json
import logging
from observability.logging import configure_logging
configure_logging(
    "INFO",
    environment="production",
    log_format="pretty_and_json_file",
    file_enabled=True,
    file_path={str(log_path)!r},
)
configure_logging("DEBUG", environment="local")
root = logging.getLogger()
print(json.dumps({{
    "handler_count": len(root.handlers),
    "handler_types": [type(handler).__name__ for handler in root.handlers],
    "httpx_level": logging.getLogger("httpx").level,
    "botocore_level": logging.getLogger("botocore").level,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["handler_count"] == 1
    assert result["handler_types"] == ["StreamHandler"]
    assert result["httpx_level"] == logging.WARNING
    assert result["botocore_level"] == logging.WARNING
    assert not log_path.exists()


def test_observed_invocation_emits_summary_and_terminal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @observe_invocation
    def invoke(payload: dict) -> dict:
        assert current_request_context() is not None
        return {"success": True}

    with caplog.at_level(logging.INFO, logger="agent.observability"):
        result = invoke({"conversation_id": "conversation-1", "turn_id": "turn-1"})

    events = [
        record.observability_event["event"]
        for record in caplog.records
        if hasattr(record, "observability_event")
    ]
    assert result == {"success": True}
    assert events == [
        "request_started",
        "request_execution_summary",
        "request_completed",
    ]
    assert current_request_context() is None


def test_observed_stream_defers_terminal_events(caplog: pytest.LogCaptureFixture) -> None:
    class Stream:
        body_iterator = iter(())

    @observe_invocation
    def invoke(payload: dict) -> Stream:
        return Stream()

    with caplog.at_level(logging.INFO, logger="agent.observability"):
        invoke({})

    events = [
        record.observability_event["event"]
        for record in caplog.records
        if hasattr(record, "observability_event")
    ]
    assert events == ["request_started"]


def test_inspector_supports_required_block_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    log_path = tmp_path / "agent-runtime.log"
    log_path.write_text(
        "REQUEST request1 | 2026-07-23 10:00:00 | follow_up | clarification\n"
        "Conversation: conversa   Turn: turn0001   Duration: 0.20s\n"
        "Trace: 1234567890abcdef1234567890abcdef\n"
        "--------------------------------------------------------------------------------\n"
        "10:00:00  FAIL  Follow-up resolution failed: NO_COMPLETED_PAIRS\n"
        "10:00:00  WARN  Memory write unavailable\n"
        "--------------------------------------------------------------------------------\n"
        "RESULT: Follow-up failed because no previous completed turn was found.\n"
        f"{DIVIDER}\n",
        encoding="utf-8",
    )

    blocks = load_blocks(log_path)
    assert len(blocks) == 1
    assert blocks[0].trace_id == "1234567890abcdef1234567890abcdef"
    assert select_blocks(blocks, command="latest", value=None, limit=20) == blocks
    assert select_blocks(blocks, command="failures", value=None, limit=20) == blocks
    assert select_blocks(
        blocks, command="persistence-failures", value=None, limit=20
    ) == blocks
    assert main(["--path", str(log_path), "--request-id", "request1"]) == 0
    assert "Follow-up resolution failed" in capsys.readouterr().out
    assert main(
        ["--path", str(log_path), "--conversation-id", "conversation-full", "--json"]
    ) == 0
    assert '"request_id": "request1"' in capsys.readouterr().out
