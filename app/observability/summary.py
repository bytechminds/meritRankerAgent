"""Immutable, bounded request execution summary state."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, replace
from typing import Literal

from observability.context import current_request_context
from observability.events import log_event
from observability.readable_log import (
    RequestLogBuffer,
    begin_request_log,
    finalize_request_log,
    reset_request_log,
)


@dataclass(frozen=True)
class RequestExecutionSummary:
    request_id: str
    trace_id: str
    conversation_id: str | None = None
    turn_id: str | None = None
    request_type: str = "unknown"
    subject: str | None = None
    intent: str | None = None
    difficulty: str | None = None
    classifier_source: str | None = None
    strong_classifier_used: bool = False
    context_required: bool = False
    context_source: str | None = None
    usable_recent_turns: int = 0
    retrieval_status: str | None = None
    retrieval_source: str | None = None
    generation_route: str | None = None
    generation_model: str | None = None
    generation_duration_ms: int | None = None
    quality_status: str | None = None
    rewrite_attempted: bool = False
    repair_attempted: bool = False
    history_write_status: str | None = None
    session_write_status: str | None = None
    memory_write_status: str | None = None
    terminal_status: Literal["completed", "clarification", "failed", "cancelled"] | None = None
    terminal_reason: str | None = None
    total_duration_ms: int | None = None


_summary: ContextVar[RequestExecutionSummary | None] = ContextVar(
    "agent_request_execution_summary", default=None
)
_SUMMARY_FIELDS = frozenset(RequestExecutionSummary.__dataclass_fields__)


@dataclass(frozen=True)
class RequestSummaryToken:
    summary: Token[RequestExecutionSummary | None]
    readable_log: Token[RequestLogBuffer | None]


def begin_request_summary() -> RequestSummaryToken:
    context = current_request_context()
    if context is None:
        raise RuntimeError("Request context must be bound before starting a summary.")
    readable_token = begin_request_log()
    return RequestSummaryToken(
        summary=_summary.set(
            RequestExecutionSummary(
                request_id=context.request_id,
                trace_id=context.trace_id,
                conversation_id=context.conversation_id,
                turn_id=context.turn_id,
                request_type=context.request_type,
            )
        ),
        readable_log=readable_token,
    )


def reset_request_summary(token: RequestSummaryToken) -> None:
    summary = _summary.get()
    if summary is not None and summary.terminal_status is not None:
        finalize_request_log(summary)
    _summary.reset(token.summary)
    reset_request_log(token.readable_log)


def current_request_summary() -> RequestExecutionSummary | None:
    return _summary.get()


def update_request_summary(**changes: object) -> RequestExecutionSummary | None:
    current = _summary.get()
    if current is None:
        return None
    unknown = set(changes) - _SUMMARY_FIELDS
    if unknown:
        raise ValueError(f"Unsupported request summary fields: {sorted(unknown)}")
    updated = replace(current, **changes)
    _summary.set(updated)
    return updated


def emit_request_summary() -> None:
    summary = _summary.get()
    if summary is None:
        return
    details = {
        key: value
        for key, value in asdict(summary).items()
        if key
        not in {
            "request_id",
            "trace_id",
            "conversation_id",
            "turn_id",
            "terminal_status",
            "total_duration_ms",
        }
        and value is not None
    }
    log_event(
        "request_execution_summary",
        component="request.lifecycle",
        stage="complete",
        status=summary.terminal_status,
        duration_ms=summary.total_duration_ms,
        details=details,
    )
