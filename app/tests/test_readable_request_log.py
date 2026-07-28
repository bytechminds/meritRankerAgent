from __future__ import annotations

import stat
import threading
from pathlib import Path

import pytest

from observability.context import bind_request_context
from observability.events import log_event
from observability.lifecycle import observe_invocation
from observability.readable_log import (
    configure_readable_request_log,
    configure_runtime_identity,
    record_local_preview,
    reset_readable_log_for_tests,
)
from observability.summary import (
    begin_request_summary,
    emit_request_summary,
    reset_request_summary,
    update_request_summary,
)

DIVIDER = "=" * 80
PRIVATE_SENTINEL = "PRIVATE-STUDENT-CONTENT-DO-NOT-WRITE"


@pytest.fixture(autouse=True)
def reset_writer() -> None:
    reset_readable_log_for_tests()
    yield
    reset_readable_log_for_tests()


def _configure(path: Path, *, max_bytes: int = 20 * 1024 * 1024) -> None:
    configure_readable_request_log(
        enabled=True,
        file_path=str(path),
        max_bytes=max_bytes,
        backup_count=3,
    )


def _write_request(
    path: Path,
    *,
    request_id: str = "request-123456789",
    request_type: str = "standalone",
    terminal_status: str = "completed",
    terminal_reason: str = "completed",
    quality_status: str = "passed_quality_gate",
) -> str:
    _configure(path)
    with bind_request_context(
        request_id=request_id,
        conversation_id="conversation-123456",
        turn_id="turn-123456",
        request_type=request_type,
    ):
        token = begin_request_summary()
        try:
            log_event(
                "request_started",
                component="request.lifecycle",
                status="started",
            )
            log_event(
                "classification_completed",
                component="classifier",
                status="completed",
                details={
                    "subject": "math",
                    "intent": "solve",
                    "difficulty": "medium",
                },
            )
            log_event(
                "generation_completed",
                component="generator",
                status="completed",
                details={"route": "primary"},
            )
            log_event(
                "quality_decision",
                component="quality",
                status="passed",
                details={
                    "passed": True,
                    "reason_code": "none",
                    "repair_required": False,
                },
            )
            log_event(
                "quality_validation_completed",
                component="quality",
                status=quality_status,
            )
            update_request_summary(
                request_type=request_type,
                subject="math",
                generation_model="o4-mini",
                quality_status=quality_status,
                terminal_status=terminal_status,
                terminal_reason=terminal_reason,
                total_duration_ms=1234,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)
    return path.read_text(encoding="utf-8")


def test_successful_request_block(tmp_path: Path) -> None:
    content = _write_request(tmp_path / "agent-runtime.log")

    assert "REQUEST request- | " in content
    assert "| standalone | completed" in content
    assert "OK    Request received" in content
    assert "OK    Answer generated: o4-mini" in content
    assert (
        "QUALITY_DECISION passed=true reason_code=none repair_required=false"
        in content
    )
    assert "OK    Quality passed" in content
    assert "PERSISTENCE\nHistory: not_attempted" in content
    assert "RESULT: Completed successfully." in content
    assert content.count(DIVIDER) == 1


def test_runtime_models_web_search_and_grounding_are_readable(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(
        request_id="observable-request",
        conversation_id="observable-conversation",
        turn_id="observable-turn",
        request_type="standalone",
    ):
        token = begin_request_summary()
        try:
            log_event(
                "request_started",
                component="request.lifecycle",
                status="started",
            )
            log_event(
                "model_execution_completed",
                component="llm.model_execution",
                stage="classifier",
                status="completed",
                duration_ms=120,
                details={
                    "route": "general.classifier.default",
                    "role": "classifier",
                    "provider": "azure_openai",
                    "model": "gpt-4.1-mini",
                },
            )
            log_event(
                "web_search_decision",
                component="web",
                status="required",
                details={
                    "required": True,
                    "reason": "current_affairs",
                    "provider": "tavily",
                    "provider_enabled": True,
                    "provider_configured": True,
                    "will_call": True,
                },
            )
            log_event(
                "web_search_execution",
                component="web",
                status="succeeded",
                duration_ms=350,
                details={
                    "provider": "tavily",
                    "attempts": 1,
                    "candidate_count": 5,
                    "selected_count": 3,
                    "context_characters": 1200,
                },
            )
            log_event(
                "grounding_completed",
                component="grounding",
                status="grounded",
                details={
                    "web_required": True,
                    "web_executed": True,
                    "web_context_used": True,
                    "citation_count": 3,
                    "verification_status": "grounded",
                },
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=500,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert "provider=azure_openai" in content
    assert "model=gpt-4.1-mini" in content
    assert "provider=tavily" in content
    assert "selected=3" in content
    assert "status=grounded" in content


def test_llm_usage_and_primary_decision_are_visible_in_readable_log(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="llm-usage-visible", request_type="standalone"):
        token = begin_request_summary()
        try:
            log_event(
                "classifier_primary_decision",
                component="classifier",
                status="accepted",
                details={
                    "primary_confidence": 0.96,
                    "primary_accepted": True,
                    "strong_triggered": False,
                    "strong_reason": "PRIMARY_ACCEPTED",
                },
            )
            log_event(
                "llm_call_usage",
                component="llm.usage",
                status="succeeded",
                duration_ms=125,
                details={
                    "role": "general.classifier.default",
                    "provider": "gemini",
                    "model": "gemini-3.1-flash-lite",
                    "attempt_type": "primary",
                    "usage_source": "provider_reported",
                    "input_tokens": 420,
                    "output_tokens": 80,
                    "total_tokens": 500,
                    "estimated_cost_usd": 0.0001,
                },
            )
            log_event(
                "llm_usage_summary",
                component="llm.usage",
                status="completed",
                details={
                    "calls": 1,
                    "input_tokens": 420,
                    "output_tokens": 80,
                    "total_tokens": 500,
                    "estimated_cost_usd": 0.0001,
                    "cost_complete": True,
                    "generator_calls": 0,
                    "rewrite_count": 0,
                    "verification_calls": 0,
                    "role_breakdown": "classifier:calls=1,tokens=500,cost=0.0001",
                },
            )
            log_event(
                "retrieval_completed",
                component="retrieval",
                status="completed",
                details={"source": "fresh_solve"},
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
            )
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert "confidence=0.96, accepted=true, strong_triggered=false" in content
    assert "LLM CALL role=general.classifier.default provider=gemini" in content
    assert "tokens=in:420 out:80 total:500" in content
    assert "LLM USAGE SUMMARY calls=1" in content
    assert "Retrieval policy: fresh_solve" in content
    assert "External context selected: none" in content
    assert "Retrieval fallback used" not in content


def test_runtime_configuration_reports_memory_enabled(tmp_path: Path) -> None:
    configure_runtime_identity(
        history_configured=True,
        session_configured=True,
        memory_configured=True,
    )
    try:
        content = _write_request(tmp_path / "agent-runtime.log")
    finally:
        configure_runtime_identity(
            history_configured=False,
            session_configured=False,
            memory_configured=False,
        )

    assert "Configured: history=true session=true memory=true" in content


def test_generation_event_prefers_actual_model_alias_over_path_label(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="generator-provenance", request_type="standalone"):
        token = begin_request_summary()
        try:
            log_event(
                "generation_completed",
                component="generator",
                status="completed",
                details={
                    "route": "orchestrated_non_stream",
                    "route_id": "math.generator.basic",
                    "task_role": "generator",
                    "model_alias": "math_basic_generator",
                    "provider": "azure_openai",
                    "deployment": "o4-mini",
                },
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
            )
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert "Answer generated: math_basic_generator" in content
    assert "Answer generated: orchestrated_non_stream" not in content


def test_delivery_milestone_does_not_duplicate_readable_generation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="single-generation", request_type="standalone"):
        token = begin_request_summary()
        try:
            log_event(
                "generation_completed",
                component="generator",
                status="completed",
                details={"model_alias": "math_basic_generator"},
            )
            log_event(
                "answer_delivery_completed",
                component="delivery",
                status="completed",
                details={"route": "verified_replay"},
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
            )
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert content.count("Answer generated: math_basic_generator") == 1


def test_repair_generation_keeps_one_readable_completion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="repair-generation", request_type="standalone"):
        token = begin_request_summary()
        try:
            for stage in ("draft", "repair"):
                log_event(
                    "generation_completed",
                    component="generator",
                    status="completed",
                    details={
                        "model_alias": "math_basic_generator",
                        "generation_stage": stage,
                    },
                )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
            )
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert content.count("Answer generated: math_basic_generator") == 1


def test_contextual_request_log_orders_prefetch_relation_and_resolution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent-runtime.log"
    configure_readable_request_log(
        enabled=True,
        file_path=str(path),
        content_mode="preview",
    )
    with bind_request_context(request_id="context-log", request_type="pending"):
        token = begin_request_summary()
        try:
            record_local_preview("query", "how did u calculated 75%")
            record_local_preview("memory_turn_1_id", "percentage-turn")
            record_local_preview(
                "memory_turn_1_user",
                "What is the formula for calculating percentage?",
            )
            record_local_preview(
                "memory_turn_1_assistant",
                "Percentage is part divided by whole times 100. 75% is 75/100.",
            )
            log_event(
                "follow_up_context_loaded",
                component="context",
                status="completed",
                details={
                    "source": "agentcore_memory",
                    "memory_status": "succeeded",
                    "memory_event_count": 1,
                    "memory_completed_pair_count": 1,
                    "memory_returned_turn_ids": "percentage-turn",
                    "memory_duration_ms": 12,
                    "usable_recent_turns": 1,
                },
            )
            log_event(
                "conversation_relation_completed",
                component="relation",
                status="completed",
                details={
                    "relation": "follow_up",
                    "confidence": 0.97,
                    "decision_source": "combined",
                    "matched_signals": (
                        "assistant_action_reference,numeric_reference=75%"
                    ),
                },
            )
            record_local_preview("selected_turn_id", "percentage-turn")
            record_local_preview(
                "selection_reason",
                "assistant_action_reference,numeric_reference=75%",
            )
            log_event(
                "conversation_context_selected",
                component="relation",
                status="completed",
                details={
                    "selected_turn_id": "percentage-turn",
                    "selected_turn_position": "latest_turn",
                    "selection_reason": (
                        "assistant_action_reference,numeric_reference=75%"
                    ),
                    "selection_confidence": 0.97,
                },
            )
            record_local_preview(
                "resolved_query",
                "Explain how 75% was calculated in the earlier percentage question.",
            )
            log_event(
                "follow_up_resolution_completed",
                component="resolver",
                status="completed",
            )
            log_event(
                "classification_completed",
                component="classifier",
                status="completed",
                details={
                    "subject": "math",
                    "intent": "explain",
                    "difficulty": "basic",
                },
            )
            update_request_summary(
                request_type="follow_up",
                terminal_status="completed",
                terminal_reason="completed",
            )
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    sections = [
        "PAYLOAD",
        "CONTEXT PREFETCH",
        "CONVERSATION UNDERSTANDING",
        "CONTEXT CANDIDATES",
        "RESOLUTION",
        "ACADEMIC CLASSIFICATION",
    ]
    positions = [content.index(section) for section in sections]
    assert positions == sorted(positions)
    assert "| follow_up | completed" in content
    assert "Turn percenta" in content
    assert "type=latest_turn" in content


def test_local_preview_is_bounded_and_off_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    configure_readable_request_log(
        enabled=True,
        file_path=str(path),
        content_mode="preview",
    )
    with bind_request_context(request_id="preview-request", request_type="standalone"):
        token = begin_request_summary()
        try:
            record_local_preview("query", "q" * 400)
            record_local_preview("response", "a" * 700)
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
            )
        finally:
            reset_request_summary(token)
    content = path.read_text(encoding="utf-8")
    assert "Query: " + ("q" * 20) in content
    assert "Preview: " + ("a" * 20) in content
    assert len(next(line for line in content.splitlines() if line.startswith("Query:"))) <= 110
    assert len(next(line for line in content.splitlines() if line.startswith("Preview:"))) <= 110


def test_local_preview_off_never_writes_content(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    configure_readable_request_log(
        enabled=True,
        file_path=str(path),
        content_mode="off",
    )
    with bind_request_context(request_id="off-request", request_type="standalone"):
        token = begin_request_summary()
        try:
            record_local_preview("query", PRIVATE_SENTINEL)
            record_local_preview("response", PRIVATE_SENTINEL)
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
            )
        finally:
            reset_request_summary(token)

    assert PRIVATE_SENTINEL not in path.read_text(encoding="utf-8")


def test_local_preview_redacts_secret_values_and_file_is_private(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    configure_readable_request_log(
        enabled=True,
        file_path=str(path),
        content_mode="preview",
    )
    with bind_request_context(request_id="secret-request", request_type="standalone"):
        token = begin_request_summary()
        try:
            record_local_preview(
                "query",
                "api_key=TOP-SECRET authorization=Bearer-456 "
                "GOOGLE_GEMINI_API_KEY=GEMINI-SECRET "
                "AWS_SECRET_ACCESS_KEY=AWS-SECRET "
                "api key: sk-project-secretvalue safe question",
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
            )
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert "TOP-SECRET" not in content
    assert "Bearer-456" not in content
    assert "GEMINI-SECRET" not in content
    assert "AWS-SECRET" not in content
    assert "sk-project-secretvalue" not in content
    assert "[REDACTED]" in content
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_failed_request_block(tmp_path: Path) -> None:
    content = _write_request(
        tmp_path / "agent-runtime.log",
        terminal_status="failed",
        terminal_reason="unexpected_exception",
    )

    assert "| standalone | failed" in content
    assert "RESULT: Failed because an unexpected exception occurred." in content


def test_failure_block_contains_trace_code_stage_and_exception_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    trace_id = "1234567890abcdef1234567890abcdef"
    with bind_request_context(
        request_id="diagnostic-1",
        trace_id=trace_id,
        request_type="standalone",
    ):
        token = begin_request_summary()
        try:
            log_event(
                "generation_failed",
                component="generator",
                stage="generate",
                status="failed",
                error_code="PROVIDER_TIMEOUT",
                details={
                    "error_type": "TimeoutError",
                    "message": PRIVATE_SENTINEL,
                },
            )
            update_request_summary(
                terminal_status="failed",
                terminal_reason="provider_failed",
                total_duration_ms=250,
            )
            emit_request_summary()
            log_event(
                "request_failed",
                component="request.lifecycle",
                stage="failed",
                status="failed",
                error_code="PROVIDER_FAILED",
                details={"error_type": "ProviderExecutionError"},
            )
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert f"Trace: {trace_id}" in content
    assert (
        "FAIL  Answer generation failed: PROVIDER_TIMEOUT, exception=TimeoutError, "
        "stage=generate"
    ) in content
    assert "FAIL  Request failed: PROVIDER_FAILED, exception=ProviderExecutionError" in content
    assert PRIVATE_SENTINEL not in content
    assert "Traceback" not in content


def test_unexpected_exception_boundary_records_type_without_message(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)

    @observe_invocation
    def invoke(payload: dict) -> dict:
        raise ValueError(PRIVATE_SENTINEL)

    with pytest.raises(ValueError, match=PRIVATE_SENTINEL):
        invoke({})

    content = path.read_text(encoding="utf-8")
    assert "FAIL  Request failed: UNEXPECTED_EXCEPTION, exception=ValueError" in content
    assert PRIVATE_SENTINEL not in content
    assert "Traceback" not in content


def test_failed_quality_request_block(tmp_path: Path) -> None:
    content = _write_request(
        tmp_path / "agent-runtime.log",
        terminal_status="failed",
        terminal_reason="verification_failed",
        quality_status="failed_quality_gate",
    )

    assert "| standalone | failed-quality" in content
    assert "FAIL  Quality failed" in content
    assert "RESULT: Quality validation failed; answer was not persisted." in content


@pytest.mark.parametrize(
    ("terminal_status", "terminal_reason", "kind", "result"),
    [
        ("cancelled", "user_cancelled", "cancelled", "RESULT: Request cancelled."),
        (
            "cancelled",
            "client_disconnected",
            "client disconnect",
            "RESULT: Client disconnected before completion.",
        ),
        (
            "completed",
            "idempotent_replay",
            "idempotent replay",
            "RESULT: Completed from idempotent replay.",
        ),
    ],
)
def test_terminal_request_variants(
    tmp_path: Path,
    terminal_status: str,
    terminal_reason: str,
    kind: str,
    result: str,
) -> None:
    content = _write_request(
        tmp_path / "agent-runtime.log",
        terminal_status=terminal_status,
        terminal_reason=terminal_reason,
    )

    assert f"| standalone | {kind}" in content
    assert result in content


def test_clarification_request_block(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(
        request_id="clarification-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        request_type="follow_up",
    ):
        token = begin_request_summary()
        try:
            log_event(
                "follow_up_resolution_failed",
                component="follow_up",
                status="failed",
                error_code="NO_COMPLETED_PAIRS",
            )
            log_event(
                "conversation_persistence_skipped",
                component="persistence",
                status="skipped",
                details={"skip_reason": "clarification_response"},
            )
            update_request_summary(
                request_type="follow_up",
                terminal_status="clarification",
                terminal_reason="clarification_required",
                total_duration_ms=20,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert "| follow_up | clarification" in content
    assert "SKIP  Persistence skipped: clarification response" in content
    assert "RESULT: Clarification returned; no completed answer was persisted." in content


def test_follow_up_context_failure_is_concise(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="follow-up-1", request_type="follow_up"):
        token = begin_request_summary()
        try:
            log_event(
                "follow_up_context_failed",
                component="recent_context",
                status="failed",
                error_code="MEMORY_TIMEOUT",
            )
            update_request_summary(
                request_type="follow_up",
                terminal_status="failed",
                terminal_reason="context_unavailable",
                total_duration_ms=10,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    assert "WARN  Recent context unavailable: MEMORY_TIMEOUT" in path.read_text(
        encoding="utf-8"
    )


def test_persistence_partial_failure_has_individual_statuses(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="persistence-1"):
        token = begin_request_summary()
        try:
            log_event(
                "conversation_persistence_failed",
                component="persistence",
                status="failed",
                details={
                    "history_write_status": "succeeded",
                    "session_write_status": "succeeded",
                    "memory_write_status": "failed_transient",
                },
            )
            update_request_summary(
                history_write_status="succeeded",
                session_write_status="succeeded",
                memory_write_status="failed_transient",
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=30,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert "OK    History saved" in content
    assert "OK    Session updated" in content
    assert "WARN  Memory write unavailable" in content


def test_concurrent_requests_write_atomic_non_interleaved_blocks(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    barrier = threading.Barrier(3)

    def write(request_id: str, subject: str) -> None:
        with bind_request_context(request_id=request_id):
            token = begin_request_summary()
            try:
                barrier.wait()
                for _ in range(10):
                    log_event(
                        "classification_completed",
                        component="classifier",
                        details={"subject": subject},
                    )
                update_request_summary(
                    subject=subject,
                    terminal_status="completed",
                    terminal_reason="completed",
                    total_duration_ms=10,
                )
                emit_request_summary()
            finally:
                reset_request_summary(token)

    threads = [
        threading.Thread(target=write, args=("request-alpha", "alpha")),
        threading.Thread(target=write, args=("request-beta", "beta")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    blocks = [
        block for block in path.read_text(encoding="utf-8").split(DIVIDER) if block.strip()
    ]
    assert len(blocks) == 2
    assert all(not ("alpha" in block and "beta" in block) for block in blocks)


def test_lines_are_single_line_and_at_most_110_characters(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="width-1"):
        token = begin_request_summary()
        try:
            log_event(
                "classification_completed",
                component="classifier",
                details={
                    "subject": "long\nsubject " * 30,
                    "intent": "solve\r\ncarefully " * 20,
                },
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=1,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines
    assert all(len(line) <= 110 for line in lines)
    assert all("\r" not in line for line in lines)


def test_private_content_is_excluded(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="privacy-1"):
        token = begin_request_summary()
        try:
            log_event(
                "generation_completed",
                component="generator",
                details={
                    "query": PRIVATE_SENTINEL,
                    "prompt": PRIVATE_SENTINEL,
                    "answer": PRIVATE_SENTINEL,
                    "authorization": PRIVATE_SENTINEL,
                    "route": "primary",
                },
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=1,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    assert PRIVATE_SENTINEL not in path.read_text(encoding="utf-8")


def test_rotation_keeps_only_standard_backups(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    configure_readable_request_log(
        enabled=True,
        file_path=str(path),
        max_bytes=500,
        backup_count=3,
    )
    for index in range(8):
        with bind_request_context(request_id=f"rotation-{index}"):
            token = begin_request_summary()
            try:
                update_request_summary(
                    terminal_status="completed",
                    terminal_reason="completed",
                    total_duration_ms=index,
                )
                emit_request_summary()
            finally:
                reset_request_summary(token)

    assert path.exists()
    assert (tmp_path / "agent-runtime.log.1").exists()
    assert all(
        item.name == "agent-runtime.log"
        or item.name.startswith("agent-runtime.log.")
        for item in tmp_path.iterdir()
    )


def test_disabled_writer_creates_no_file(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    configure_readable_request_log(enabled=False, file_path=str(path))

    with bind_request_context(request_id="disabled-1"):
        token = begin_request_summary()
        try:
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=1,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    assert not path.exists()


def test_file_failure_does_not_affect_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import observability.readable_log as readable_log

    path = tmp_path / "agent-runtime.log"
    _configure(path)

    class BrokenHandler:
        def handle(self, record: object) -> None:
            raise OSError("disk unavailable")

    monkeypatch.setattr(readable_log, "_handler", BrokenHandler())
    with bind_request_context(request_id="failure-1"):
        token = begin_request_summary()
        try:
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=1,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    assert "local request log unavailable" in capsys.readouterr().err


def test_formatting_failure_does_not_affect_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    import observability.readable_log as readable_log

    path = tmp_path / "agent-runtime.log"
    _configure(path)
    monkeypatch.setattr(
        readable_log,
        "format_request_block",
        lambda events, summary: (_ for _ in ()).throw(ValueError("format failed")),
    )

    with bind_request_context(request_id="format-failure-1"):
        token = begin_request_summary()
        try:
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=1,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    assert "local request log unavailable" in capsys.readouterr().err


def test_exactly_one_block_per_terminal_request(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="once-1"):
        token = begin_request_summary()
        try:
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=1,
            )
            emit_request_summary()
            emit_request_summary()
        finally:
            reset_request_summary(token)

    assert path.read_text(encoding="utf-8").count(DIVIDER) == 1


def test_bounded_buffer_preserves_late_persistence_result(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="bounded-1"):
        token = begin_request_summary()
        try:
            log_event("request_started", component="request.lifecycle")
            for index in range(70):
                log_event(
                    "classification_completed",
                    component="classifier",
                    details={"subject": f"subject-{index}"},
                )
            log_event(
                "conversation_persistence_failed",
                component="persistence",
                details={"memory_write_status": "failed_transient"},
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=1,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert "WARN  Memory write unavailable" in content
    assert "subject-0" not in content


def test_reference_analysis_is_bounded_and_readable(tmp_path: Path) -> None:
    path = tmp_path / "agent-runtime.log"
    _configure(path)
    with bind_request_context(request_id="reference-log", request_type="follow_up"):
        token = begin_request_summary()
        try:
            log_event(
                "conversation_reference_analyzed",
                component="conversation.reference_resolution",
                stage="select_context",
                status="completed",
                details={
                    "local_reference": False,
                    "external_reference_detected": True,
                    "reference_types": "person,event",
                },
            )
            log_event(
                "conversation_candidate_compatibility",
                component="conversation.reference_resolution",
                stage="select_context",
                status="compatible",
                details={
                    "turn_id": "akbar-turn",
                    "compatible": True,
                    "compatibility_reason": "event_or_person",
                    "recency_rank": 1,
                },
            )
            update_request_summary(
                terminal_status="completed",
                terminal_reason="completed",
                total_duration_ms=1,
            )
            emit_request_summary()
        finally:
            reset_request_summary(token)

    content = path.read_text(encoding="utf-8")
    assert "REFERENCE ANALYSIS" in content
    assert "local=false, external=true, types=person,event" in content
    assert "reason=event_or_person" in content
