"""Invocation lifecycle wrapper shared by all AgentCore request paths."""

from __future__ import annotations

import functools
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any, TypeVar, cast

from observability.context import (
    RequestType,
    bind_request_context,
    current_request_context,
    safe_identifier,
)
from observability.events import log_event
from observability.readable_log import record_local_preview
from observability.summary import (
    begin_request_summary,
    emit_request_summary,
    reset_request_summary,
    update_request_summary,
)
from observability.tracing import stage_span

_F = TypeVar("_F", bound=Callable[..., Any])


def _request_type(payload: dict[str, object]) -> RequestType:
    if payload.get("image") is not None:
        return "image"
    return "pending"


def observe_invocation(function: _F) -> _F:
    @functools.wraps(function)
    def wrapper(payload: dict, *args: object, **kwargs: object) -> object:
        request_id = str(uuid.uuid4())
        started = time.monotonic()
        with bind_request_context(
            request_id=request_id,
            conversation_id=safe_identifier(payload.get("conversation_id")),
            turn_id=safe_identifier(payload.get("turn_id")),
            request_type=_request_type(payload),
        ):
            summary_token = begin_request_summary()
            try:
                record_local_preview("mode", payload.get("mode"))
                record_local_preview("language", payload.get("language"))
                record_local_preview("exam_id", payload.get("exam_id"))
                record_local_preview(
                    "query", payload.get("query") or payload.get("questionText")
                )
                with stage_span("doubt_solver.request"):
                    log_event(
                        "request_started",
                        component="request.lifecycle",
                        stage="started",
                        status="started",
                        details={
                            "request_type": current_request_context().request_type
                        },
                    )
                    result = function(payload, *args, **kwargs)
                if hasattr(result, "body_iterator"):
                    return result
                success = not isinstance(result, dict) or result.get("success", True) is not False
                duration_ms = int((time.monotonic() - started) * 1000)
                classification = (
                    result.get("classification")
                    if isinstance(result, dict)
                    and isinstance(result.get("classification"), dict)
                    else {}
                )
                clarification = bool(
                    not success
                    and isinstance(result, dict)
                    and (
                        result.get("status") == "clarification"
                        or classification.get("requires_recent_conversation")
                    )
                )
                terminal_status = (
                    "completed"
                    if success
                    else ("clarification" if clarification else "failed")
                )
                if isinstance(result, dict):
                    record_local_preview(
                        "response", result.get("answer") or result.get("error")
                    )
                    record_local_preview(
                        "response_type",
                        (
                            "answer"
                            if success
                            else ("clarification" if clarification else "error")
                        ),
                    )
                update_request_summary(
                    terminal_status=terminal_status,
                    terminal_reason=(
                        "clarification_required" if clarification else terminal_status
                    ),
                    total_duration_ms=duration_ms,
                )
                emit_request_summary()
                log_event(
                    "request_completed" if success else "request_failed",
                    component="request.lifecycle",
                    stage="complete" if success else "failed",
                    status=terminal_status,
                    duration_ms=duration_ms,
                    error_code=None if success else "REQUEST_FAILED",
                    level=logging.INFO if success else logging.ERROR,
                )
                return result
            except BaseException as exc:
                duration_ms = int((time.monotonic() - started) * 1000)
                update_request_summary(
                    terminal_status="failed",
                    terminal_reason="unexpected_exception",
                    total_duration_ms=duration_ms,
                )
                emit_request_summary()
                log_event(
                    "request_failed",
                    component="request.lifecycle",
                    stage="failed",
                    status="failed",
                    duration_ms=duration_ms,
                    error_code="UNEXPECTED_EXCEPTION",
                    details={"error_type": type(exc).__name__},
                    level=logging.ERROR,
                )
                raise
            finally:
                reset_request_summary(summary_token)

    return cast(_F, wrapper)
