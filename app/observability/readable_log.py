"""Atomic, human-readable local request log blocks."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from observability.summary import RequestExecutionSummary

_DIVIDER = "=" * 80
_SEPARATOR = "-" * 80
_MAX_LINE_WIDTH = 110
_MAX_EVENTS = 64
_DEFAULT_MAX_BYTES = 20 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 3
_READABLE_EVENTS = frozenset(
    {
        "request_started",
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
        "generation_failed",
        "quality_validation_completed",
        "quality_rewrite_completed",
        "quality_repair_completed",
        "conversation_persistence_completed",
        "conversation_persistence_failed",
        "conversation_persistence_skipped",
    }
)


@dataclass
class RequestLogBuffer:
    events: list[dict[str, object]] = field(default_factory=list)
    previews: dict[str, str] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    flushed: bool = False


class SafeRequestBlockHandler(RotatingFileHandler):
    """Write one complete request block while holding the handler lock."""

    def handleError(self, record: logging.LogRecord) -> None:
        _warn_file_unavailable()

    def doRollover(self) -> None:
        super().doRollover()
        try:
            os.chmod(self.baseFilename, 0o600)
            for index in range(1, self.backupCount + 1):
                backup = f"{self.baseFilename}.{index}"
                if os.path.exists(backup):
                    os.chmod(backup, 0o600)
        except OSError:
            _warn_file_unavailable()


_buffer: ContextVar[RequestLogBuffer | None] = ContextVar(
    "agent_readable_request_log", default=None
)
_handler: SafeRequestBlockHandler | None = None
_handler_lock = threading.Lock()
_warning_lock = threading.Lock()
_warning_emitted = False
_content_mode = "off"
_process_started_at = datetime.now(UTC).isoformat()
_runtime_identity: dict[str, object] = {
    "application_version": os.getenv("APP_VERSION", "0.1.0"),
    "git_commit_short": "unknown",
    "process_started_at": _process_started_at,
    "pid": os.getpid(),
    "environment": "unknown",
    "region": "unknown",
    "history_configured": False,
    "session_configured": False,
    "memory_configured": False,
    "code_location": "unknown",
    "environment_file": "app/.env.local",
    "endpoint": "http://localhost:8080/invocations",
}


def configure_runtime_identity(**values: object) -> dict[str, object]:
    allowed = set(_runtime_identity)
    _runtime_identity.update({key: value for key, value in values.items() if key in allowed})
    if _runtime_identity["git_commit_short"] == "unknown":
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                check=True,
                capture_output=True,
                text=True,
                timeout=1,
            )
            _runtime_identity["git_commit_short"] = result.stdout.strip()[:12] or "unknown"
        except (OSError, subprocess.SubprocessError):
            pass
    return dict(_runtime_identity)


def _warn_file_unavailable() -> None:
    global _warning_emitted
    with _warning_lock:
        if _warning_emitted:
            return
        _warning_emitted = True
    try:
        sys.stderr.write("WARNING: local request log unavailable; console logging continues.\n")
    except OSError:
        pass


def configure_readable_request_log(
    *,
    enabled: bool,
    file_path: str,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    content_mode: str = "off",
) -> None:
    global _content_mode, _handler
    _content_mode = content_mode if content_mode in {"preview", "full"} else "off"
    with _handler_lock:
        if _handler is not None:
            try:
                _handler.close()
            except OSError:
                _warn_file_unavailable()
            _handler = None
        if not enabled:
            return
        try:
            path = Path(file_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(mode=0o600, exist_ok=True)
            os.chmod(path, 0o600)
            handler = SafeRequestBlockHandler(
                path,
                maxBytes=max(max_bytes, 1),
                backupCount=max(backup_count, 0),
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            _handler = handler
        except OSError:
            _warn_file_unavailable()


def begin_request_log() -> Token[RequestLogBuffer | None]:
    return _buffer.set(RequestLogBuffer())


def reset_request_log(token: Token[RequestLogBuffer | None]) -> None:
    _buffer.reset(token)


def collect_event(event: dict[str, object]) -> None:
    current = _buffer.get()
    if current is None or event.get("event") not in _READABLE_EVENTS:
        return
    with current.lock:
        if current.flushed:
            return
        if len(current.events) >= _MAX_EVENTS:
            current.events.pop(1 if current.events[0].get("event") == "request_started" else 0)
        current.events.append(event)


_PREVIEW_LIMITS = {
    "query": 300,
    "resolved_query": 300,
    "previous_turns": 300,
    "response": 600,
    "mode": 32,
    "language": 24,
    "exam_id": 64,
    "response_type": 32,
    "selected_turn_id": 128,
    "selection_reason": 300,
    "selected_previous_user": 250,
    "selected_previous_assistant": 400,
    "memory_turn_1_id": 128,
    "memory_turn_1_user": 250,
    "memory_turn_1_assistant": 400,
    "memory_turn_2_id": 128,
    "memory_turn_2_user": 250,
    "memory_turn_2_assistant": 400,
    "dynamodb_turn_1_id": 128,
    "dynamodb_turn_1_user": 250,
    "dynamodb_turn_1_assistant": 400,
    "dynamodb_turn_2_id": 128,
    "dynamodb_turn_2_user": 250,
    "dynamodb_turn_2_assistant": 400,
}
_FULL_LIMITS = {
    **_PREVIEW_LIMITS,
    "query": 5000,
    "resolved_query": 5000,
    "previous_turns": 7000,
    "response": 8000,
    "selected_previous_user": 5000,
    "selected_previous_assistant": 8000,
}
for _source in ("memory", "dynamodb"):
    for _index in (1, 2):
        _FULL_LIMITS[f"{_source}_turn_{_index}_user"] = 5000
        _FULL_LIMITS[f"{_source}_turn_{_index}_assistant"] = 8000
_PREVIEW_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(?:[a-z0-9_-]*(?:api[_ -]?key|secret(?:[_ -]?access)?[_ -]?key|"
        r"access[_ -]?token|refresh[_ -]?token|authorization|credentials?|password))"
        r"\s*[:=]\s*(?:bearer\s+)?[^\s,;]+"
    ),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
)


def _redact_preview_secrets(value: str) -> str:
    redacted = value
    for pattern in _PREVIEW_SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def record_local_preview(field: str, value: object) -> None:
    limits = _FULL_LIMITS if _content_mode == "full" else _PREVIEW_LIMITS
    if _content_mode == "off" or field not in limits:
        return
    current = _buffer.get()
    if current is None:
        return
    cleaned = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    cleaned = _redact_preview_secrets(cleaned)
    with current.lock:
        if not current.flushed:
            current.previews[field] = cleaned[: limits[field]]


def _short(value: object) -> str:
    text = str(value or "-").strip()
    return text[:8] if text else "-"


def _clean(value: object, *, limit: int = 48) -> str:
    text = " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())
    if len(text) <= limit:
        return text
    return f"{text[: max(limit - 3, 0)]}..."


def _line(text: str) -> str:
    cleaned = text.replace("\r", " ").replace("\n", " ")
    if len(cleaned) <= _MAX_LINE_WIDTH:
        return cleaned
    return f"{cleaned[: _MAX_LINE_WIDTH - 3]}..."


def _event_time(event: dict[str, object]) -> str:
    timestamp = str(event.get("timestamp") or "")
    try:
        return datetime.fromisoformat(timestamp).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return "--:--:--"


def _local_datetime(event: dict[str, object]) -> str:
    timestamp = str(event.get("timestamp") or "")
    try:
        return datetime.fromisoformat(timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _entry(timestamp: str, label: str, description: str) -> str:
    return _line(f"{timestamp}  {label:<4}  {description}")


def _status_entry(name: str, status: object) -> tuple[str, str]:
    normalized = str(status or "skipped")
    if normalized == "succeeded":
        descriptions = {
            "history": "History saved",
            "session": "Session updated",
            "memory": "Memory saved",
        }
        return "OK", descriptions[name]
    if normalized == "skipped":
        return "SKIP", f"{name.title()} persistence skipped"
    descriptions = {
        "history": "History write unavailable",
        "session": "Session update unavailable",
        "memory": "Memory write unavailable",
    }
    return "WARN", descriptions[name]


def _safe_count(value: object) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _failure_description(
    event: dict[str, object],
    safe: dict[str, object],
    description: str,
    *,
    extra: tuple[tuple[str, object], ...] = (),
) -> str:
    evidence: list[str] = []
    error_code = _clean(event.get("error_code"), limit=40)
    error_type = _clean(safe.get("error_type"), limit=32)
    stage = _clean(event.get("stage"), limit=24)
    if error_code:
        evidence.append(error_code)
    if error_type:
        evidence.append(f"exception={error_type}")
    if stage and stage not in {"failed", "complete"}:
        evidence.append(f"stage={stage}")
    for key, value in extra:
        cleaned = _clean(value, limit=32)
        if cleaned:
            evidence.append(f"{key}={cleaned}")
    return f"{description}: {', '.join(evidence)}" if evidence else description


def _timeline_entries(
    event: dict[str, object],
    summary: RequestExecutionSummary,
) -> list[tuple[str, str]]:
    name = str(event.get("event") or "")
    details = event.get("details")
    safe = details if isinstance(details, dict) else {}
    status = str(event.get("status") or "")

    if name == "request_failed":
        return [("FAIL", _failure_description(event, safe, "Request failed"))]
    if name == "request_cancelled":
        return [("WARN", _failure_description(event, safe, "Request cancelled"))]
    if name == "classification_completed":
        values = [
            _clean(safe.get("subject"), limit=24),
            _clean(safe.get("intent"), limit=24),
            _clean(safe.get("difficulty"), limit=16),
        ]
        return [("OK", f"Classified: {', '.join(value for value in values if value)}")]
    if name == "classification_fallback_used":
        return [("WARN", "Classification fallback used")]
    if name == "conversation_relation_completed":
        return [
            (
                "OK",
                "Relation: "
                f"{_clean(safe.get('relation'), limit=32)}, "
                f"confidence={_clean(safe.get('confidence'), limit=8)}, "
                f"source={_clean(safe.get('decision_source'), limit=24)}",
            ),
            (
                "OK",
                f"Signals: {_clean(safe.get('matched_signals'), limit=90)}",
            ),
        ]
    if name == "conversation_context_selected":
        return [
            (
                "OK",
                "Selected turn: "
                f"{_short(safe.get('selected_turn_id'))}, "
                f"position={_clean(safe.get('selected_turn_position'), limit=24)}, "
                f"confidence={_clean(safe.get('selection_confidence'), limit=8)}",
            ),
            (
                "OK",
                f"Selection reason: {_clean(safe.get('selection_reason'), limit=82)}",
            ),
        ]
    if name == "follow_up_detected":
        return [("OK", "Follow-up detected")]
    if name == "follow_up_context_loaded":
        turns = _safe_count(safe.get("usable_recent_turns"))
        source = str(safe.get("source") or "")
        if source == "dynamodb_fallback":
            return [
                (
                    "WARN",
                    "Memory: "
                    f"{_clean(safe.get('memory_status'), limit=24)}, "
                    f"reason={_clean(safe.get('memory_failure_reason'), limit=32)}, "
                    f"events={_safe_count(safe.get('memory_event_count'))}, "
                    f"duration={_safe_count(safe.get('memory_duration_ms'))}ms",
                ),
                (
                    "OK",
                    "DynamoDB: "
                    f"{_clean(safe.get('dynamodb_status'), limit=24)}, "
                    f"items={_safe_count(safe.get('dynamodb_item_count'))}, "
                    f"usable={turns}, "
                    f"duration={_safe_count(safe.get('dynamodb_duration_ms'))}ms",
                ),
            ]
        if source == "agentcore_memory":
            return [
                (
                    "OK",
                    "Memory: "
                    f"{_clean(safe.get('memory_status'), limit=24)}, "
                    f"events={_safe_count(safe.get('memory_event_count'))}, "
                    f"pairs={_safe_count(safe.get('memory_completed_pair_count'))}, "
                    f"usable={turns}, "
                    f"turns={_clean(safe.get('memory_returned_turn_ids'), limit=48)}, "
                    f"duration={_safe_count(safe.get('memory_duration_ms'))}ms",
                )
            ]
        return [("OK", f"Recent context loaded: {turns} turns")]
    if name == "follow_up_context_failed":
        return [
            (
                "WARN",
                "Memory: "
                f"{_clean(safe.get('memory_status'), limit=24)}, "
                f"reason={_clean(safe.get('memory_failure_reason'), limit=32)}, "
                f"events={_safe_count(safe.get('memory_event_count'))}",
            ),
            (
                "WARN",
                "DynamoDB: "
                f"{_clean(safe.get('dynamodb_status'), limit=24)}, "
                f"reason={_clean(safe.get('dynamodb_failure_reason'), limit=32)}, "
                f"items={_safe_count(safe.get('dynamodb_item_count'))}, usable=0",
            ),
            ("WARN", _failure_description(event, safe, "Recent context unavailable")),
        ]
    if name == "follow_up_resolution_completed":
        return [("OK", "Follow-up resolved")]
    if name == "follow_up_resolution_failed":
        return [
            ("FAIL", _failure_description(event, safe, "Follow-up resolution failed"))
        ]
    if name == "retrieval_completed":
        source = _clean(safe.get("source"), limit=32)
        return [("OK", f"Retrieval completed{f': {source}' if source else ''}")]
    if name == "retrieval_fallback_used":
        if str(event.get("stage") or "") == "load_recent_context":
            return []
        return [("WARN", "Retrieval fallback used")]
    if name == "retrieval_failed":
        return [
            (
                "WARN",
                _failure_description(
                    event,
                    safe,
                    "Retrieval unavailable",
                    extra=(("source", summary.retrieval_source),),
                ),
            )
        ]
    if name == "generation_completed":
        model = _clean(
            summary.generation_model
            or safe.get("model_alias")
            or safe.get("route"),
            limit=40,
        )
        return [("OK", f"Answer generated{f': {model}' if model else ''}")]
    if name == "generation_failed":
        return [
            (
                "FAIL",
                _failure_description(
                    event,
                    safe,
                    "Answer generation failed",
                    extra=(
                        ("model", summary.generation_model),
                        ("route", summary.generation_route),
                    ),
                ),
            )
        ]
    if name == "quality_validation_completed":
        if status == "failed_quality_gate":
            return [
                (
                    "FAIL",
                    _failure_description(
                        event,
                        safe,
                        "Quality failed",
                        extra=(("repair_attempted", summary.repair_attempted),),
                    ),
                )
            ]
        return [("OK", "Quality passed")]
    if name == "quality_rewrite_completed":
        if "failed" in status:
            return [("FAIL", _failure_description(event, safe, "Answer rewrite failed"))]
        return [("OK", "Answer rewrite completed")]
    if name == "quality_repair_completed":
        if "failed" in status:
            return [("FAIL", _failure_description(event, safe, "Answer repair failed"))]
        return [("OK", "Answer repair completed")]
    if name in {
        "conversation_persistence_completed",
        "conversation_persistence_failed",
        "conversation_persistence_skipped",
    }:
        if safe.get("skip_reason") == "clarification_response":
            return [("SKIP", "Persistence skipped: clarification response")]
        entries: list[tuple[str, str]] = []
        for key, display in (
            ("history_write_status", "history"),
            ("session_write_status", "session"),
            ("memory_write_status", "memory"),
        ):
            if key in safe:
                entries.append(_status_entry(display, safe[key]))
        if name == "conversation_persistence_failed":
            entries.append(
                ("WARN", _failure_description(event, safe, "Persistence partial failure"))
            )
        return entries
    return []


def _result_kind(
    summary: RequestExecutionSummary,
    events: list[dict[str, object]],
) -> str:
    if summary.terminal_reason == "idempotent_replay":
        return "idempotent replay"
    if summary.terminal_reason == "client_disconnected":
        return "client disconnect"
    if any(
        isinstance(event.get("details"), dict)
        and event["details"].get("skip_reason") == "clarification_response"
        for event in events
    ):
        return "clarification"
    if summary.quality_status == "failed_quality_gate":
        return "failed-quality"
    return summary.terminal_status or "failed"


def _result_text(
    summary: RequestExecutionSummary,
    events: list[dict[str, object]],
) -> str:
    reason = str(summary.terminal_reason or "")
    error_codes = {
        str(event.get("error_code") or "").lower()
        for event in events
        if event.get("error_code")
    }
    if summary.terminal_status == "clarification":
        return "Clarification returned; no completed answer was persisted."
    if "no_completed_pairs" in error_codes:
        return "Follow-up failed because no previous completed turn was found."
    if reason == "idempotent_replay":
        return "Completed from idempotent replay."
    if reason == "client_disconnected":
        return "Client disconnected before completion."
    if summary.terminal_status == "cancelled":
        return "Request cancelled."
    if summary.quality_status == "failed_quality_gate" or reason in {
        "verification_failed",
        "answer_verification_failed",
        "quality_failure",
    }:
        return "Quality validation failed; answer was not persisted."
    if summary.terminal_status == "completed":
        return "Completed successfully."
    if reason in {"unexpected_exception", "unexpected_internal_error"}:
        return "Failed because an unexpected exception occurred."
    readable_reason = _clean(reason.replace("_", " "), limit=68)
    return f"Failed: {readable_reason or 'unknown operational error'}."


def format_request_block(
    events: list[dict[str, object]],
    summary: RequestExecutionSummary,
) -> str:
    first = events[0] if events else {}
    event_time = _event_time(first)
    duration_seconds = max(summary.total_duration_ms or 0, 0) / 1000
    kind = _result_kind(summary, events)
    lines = [
        _line(
            f"REQUEST {_short(summary.request_id)} | {_local_datetime(first)} | "
            f"{_clean(summary.request_type, limit=20)} | {kind}"
        ),
        _line(
            f"App: {_runtime_identity['application_version']}@"
            f"{_runtime_identity['git_commit_short']}   "
            f"PID: {_runtime_identity['pid']}   "
            f"Environment: {_runtime_identity['environment']}   "
            f"Region: {_runtime_identity['region']}"
        ),
        _line(
            f"Conversation: {_short(summary.conversation_id)}   "
            f"Turn: {_short(summary.turn_id)}   Duration: {duration_seconds:.2f}s"
        ),
        _line(f"Trace: {_clean(summary.trace_id, limit=100)}"),
        _line(
            "Configured: "
            f"history={str(_runtime_identity['history_configured']).lower()} "
            f"session={str(_runtime_identity['session_configured']).lower()} "
            f"memory={str(_runtime_identity['memory_configured']).lower()}"
        ),
        _SEPARATOR,
        _entry(event_time, "OK", "Request received"),
    ]
    current = _buffer.get()
    previews = dict(current.previews) if current is not None else {}
    if previews:
        lines.extend(
            [
                "PAYLOAD",
                _line(
                    f"Mode: {previews.get('mode', '-')}   "
                    f"Language: {previews.get('language', '-')}   "
                    f"Exam: {previews.get('exam_id', '-')}"
                ),
                _line(f"Query: {previews.get('query', '-')}"),
            ]
        )
    last_section: str | None = None
    generation_completion_rendered = False
    for event in events:
        name = str(event.get("event") or "")
        if name.startswith("conversation_persistence_"):
            continue
        if name == "generation_completed":
            if generation_completion_rendered:
                continue
            generation_completion_rendered = True
        section = _event_section(name)
        timestamp = _event_time(event)
        entries = _timeline_entries(event, summary)
        if entries and section and section != last_section:
            lines.append(section)
            last_section = section
        for label, description in entries:
            lines.append(_entry(timestamp, label, description))
        if name in {"follow_up_context_loaded", "follow_up_context_failed"}:
            details = event.get("details")
            safe = details if isinstance(details, dict) else {}
            source = (
                "dynamodb"
                if safe.get("source") == "dynamodb_fallback"
                else "memory"
            )
            fetched = any(
                previews.get(f"{source}_turn_{index}_user")
                for index in (1, 2)
            )
            if fetched:
                lines.append(f"{source.upper()} FETCH")
                for index in (1, 2):
                    turn_id = previews.get(f"{source}_turn_{index}_id")
                    user = previews.get(f"{source}_turn_{index}_user")
                    assistant = previews.get(f"{source}_turn_{index}_assistant")
                    if not user and not assistant:
                        continue
                    lines.append(_line(f"Turn {_short(turn_id)}"))
                    lines.append(_line(f"User: {user or '-'}"))
                    lines.append(_line(f"Assistant: {assistant or '-'}"))
        if name == "conversation_context_selected":
            if previews.get("selected_previous_user"):
                lines.append(
                    _line(f"User: {previews['selected_previous_user']}")
                )
            if previews.get("selected_previous_assistant"):
                lines.append(
                    _line(
                        "Assistant: "
                        f"{previews['selected_previous_assistant']}"
                    )
                )
        if name == "follow_up_resolution_completed" and previews.get(
            "resolved_query"
        ):
            lines.append(_line(f"Resolved: {previews['resolved_query']}"))
    lines.append("PERSISTENCE")
    for event in events:
        if not str(event.get("event") or "").startswith("conversation_persistence_"):
            continue
        timestamp = _event_time(event)
        for label, description in _timeline_entries(event, summary):
            lines.append(_entry(timestamp, label, description))
    lines.append(_line(f"History: {summary.history_write_status or 'not_attempted'}"))
    lines.append(_line(f"Session: {summary.session_write_status or 'not_attempted'}"))
    lines.append(_line(f"Memory: {summary.memory_write_status or 'not_attempted'}"))
    if previews.get("response") or previews.get("response_type"):
        lines.extend(
            [
                "RESPONSE",
                _line(f"Type: {previews.get('response_type', '-')}"),
                _line(f"Preview: {previews.get('response', '-')}"),
            ]
        )
    lines.extend(
        [
            _SEPARATOR,
            _line(f"RESULT: {_result_text(summary, events)}"),
            _DIVIDER,
        ]
    )
    return "\n".join(lines)


def _event_section(event_name: str) -> str | None:
    if event_name.startswith("classification_"):
        return "ACADEMIC CLASSIFICATION"
    if event_name == "conversation_relation_completed":
        return "CONVERSATION RELATION"
    if event_name == "conversation_context_selected":
        return "SELECTED CONTEXT"
    if event_name == "follow_up_detected":
        return "FOLLOW-UP"
    if event_name.startswith("follow_up_context_"):
        return "CONTEXT PREFETCH"
    if event_name.startswith("follow_up_resolution_"):
        return "RESOLUTION"
    if event_name.startswith("retrieval_"):
        return "RETRIEVAL"
    if event_name.startswith("generation_"):
        return "GENERATION"
    if event_name.startswith("quality_"):
        return "QUALITY"
    return None


def finalize_request_log(summary: RequestExecutionSummary) -> None:
    try:
        current = _buffer.get()
        if current is None:
            return
        with current.lock:
            if current.flushed:
                return
            current.flushed = True
            events = list(current.events)
        with _handler_lock:
            handler = _handler
            if handler is None:
                return
            record = logging.LogRecord(
                "agent.readable_request_log",
                logging.INFO,
                "",
                0,
                format_request_block(events, summary),
                (),
                None,
            )
            handler.handle(record)
    except Exception:
        _warn_file_unavailable()


def reset_readable_log_for_tests() -> None:
    global _handler, _warning_emitted
    with _handler_lock:
        if _handler is not None:
            close = getattr(_handler, "close", None)
            if callable(close):
                close()
            _handler = None
    _warning_emitted = False
