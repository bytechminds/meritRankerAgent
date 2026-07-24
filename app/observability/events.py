"""Structured event contract and bounded metadata sanitization."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from observability.context import current_request_context
from observability.readable_log import collect_event

SERVICE_NAME = "meritranker-tutor"

EVENT_NAMES = frozenset(
    {
        "runtime_started",
        "request_started",
        "request_execution_summary",
        "request_completed",
        "request_failed",
        "request_cancelled",
        "classification_completed",
        "classification_fallback_used",
        "conversation_relation_completed",
        "conversation_context_selected",
        "follow_up_detected",
        "follow_up_context_loaded",
        "follow_up_context_failed",
        "follow_up_resolution_completed",
        "follow_up_resolution_failed",
        "retrieval_completed",
        "retrieval_fallback_used",
        "retrieval_failed",
        "generation_completed",
        "answer_delivery_completed",
        "generation_failed",
        "quality_validation_completed",
        "quality_rewrite_completed",
        "quality_repair_completed",
        "conversation_persistence_completed",
        "conversation_persistence_skipped",
        "conversation_persistence_failed",
    }
)

_SENSITIVE_KEY_PARTS = (
    "access_key",
    "actor_id",
    "answer",
    "api_key",
    "authorization",
    "aws_access",
    "aws_secret",
    "content",
    "conversation_context",
    "context_text",
    "credential",
    "image",
    "jwt",
    "message",
    "payload",
    "prompt",
    "query",
    "raw",
    "refresh_token",
    "secret",
    "token",
    "user_id",
)
_MAX_DETAILS = 32
_MAX_STRING = 256
_logger = logging.getLogger("agent.observability")
_environment = "local"
_detailed_logs = False


def configure_event_metadata(*, environment: str, detailed_logs: bool) -> None:
    global _environment, _detailed_logs
    _environment = environment
    _detailed_logs = detailed_logs


def _safe_key(key: object) -> str | None:
    normalized = str(key).strip().lower()
    if not normalized or any(part in normalized for part in _SENSITIVE_KEY_PARTS):
        return None
    return normalized[:64]


def _safe_value(value: object) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ")[:_MAX_STRING]
    return type(value).__name__


def sanitize_details(details: Mapping[str, object] | None) -> dict[str, object]:
    if not details:
        return {}
    safe: dict[str, object] = {}
    for key, value in list(details.items())[:_MAX_DETAILS]:
        safe_key = _safe_key(key)
        if safe_key is not None:
            safe[safe_key] = _safe_value(value)
    return safe


def build_event(
    event: str,
    *,
    component: str,
    stage: str | None = None,
    status: str | None = None,
    duration_ms: int | None = None,
    error_code: str | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if event not in EVENT_NAMES:
        raise ValueError(f"Unsupported observability event: {event}")
    context = current_request_context()
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "level": "INFO",
        "service": SERVICE_NAME,
        "environment": _environment,
        "event": event,
        "component": component[:96],
        "request_id": context.request_id if context else None,
        "trace_id": context.trace_id if context else None,
        "conversation_id": context.conversation_id if context else None,
        "turn_id": context.turn_id if context else None,
        "stage": stage,
        "status": status,
        "duration_ms": max(duration_ms, 0) if duration_ms is not None else None,
        "error_code": error_code[:96] if error_code else None,
        "details": sanitize_details(details),
    }


def log_event(
    event: str,
    *,
    component: str,
    stage: str | None = None,
    status: str | None = None,
    duration_ms: int | None = None,
    error_code: str | None = None,
    details: Mapping[str, object] | None = None,
    level: int = logging.INFO,
) -> None:
    payload = build_event(
        event,
        component=component,
        stage=stage,
        status=status,
        duration_ms=duration_ms,
        error_code=error_code,
        details=details,
    )
    payload["level"] = logging.getLevelName(level)
    context = current_request_context()
    concise = [event]
    if status:
        concise.append(f"status={status}")
    if stage:
        concise.append(f"stage={stage}")
    if context:
        concise.append(f"request_id={context.request_id[:8]}")
    if duration_ms is not None:
        concise.append(f"duration_ms={max(duration_ms, 0)}")
    if _detailed_logs:
        concise.extend(f"{key}={value}" for key, value in payload["details"].items())
    collect_event(payload)
    _logger.log(level, " ".join(concise), extra={"observability_event": payload})
