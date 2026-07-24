"""Central console, JSON stdout, and local readable request-log configuration."""

from __future__ import annotations

import json
import logging
import sys

from rich.logging import RichHandler

from observability.context import current_request_context
from observability.events import configure_event_metadata
from observability.readable_log import (
    configure_readable_request_log,
    reset_readable_log_for_tests,
)
from observability.tracing import configure_tracing

_configured = False
_NOISY_LOGGERS = ("httpx", "urllib3", "botocore", "boto3", "azure", "openai")


class JsonEventFormatter(logging.Formatter):
    def __init__(self, *, environment: str) -> None:
        super().__init__()
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "observability_event", None)
        if payload is None:
            context = current_request_context()
            payload = {
                "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "service": "meritranker-tutor",
                "environment": self._environment,
                "event": "application_log",
                "component": record.name,
                "request_id": context.request_id if context else None,
                "trace_id": context.trace_id if context else None,
                "conversation_id": context.conversation_id if context else None,
                "turn_id": context.turn_id if context else None,
                "stage": None,
                "status": None,
                "duration_ms": None,
                "error_code": None,
                "details": {"message_redacted": True},
            }
        try:
            return json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
        except (TypeError, ValueError):
            return json.dumps(
                {
                    "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": "ERROR",
                    "service": "meritranker-tutor",
                    "environment": self._environment,
                    "event": "application_log",
                    "component": "observability.serializer",
                    "request_id": None,
                    "trace_id": None,
                    "conversation_id": None,
                    "turn_id": None,
                    "stage": "logging",
                    "status": "failed",
                    "duration_ms": None,
                    "error_code": "LOG_SERIALIZATION_FAILED",
                    "details": {},
                },
                separators=(",", ":"),
                sort_keys=True,
            )


def configure_logging(
    log_level: str = "INFO",
    *,
    environment: str = "local",
    log_format: str | None = None,
    file_enabled: bool | None = None,
    file_path: str = ".logs/agent-runtime.log",
    detailed_logs: bool = False,
    observability_enabled: bool = True,
    local_log_content: str = "preview",
) -> None:
    global _configured
    if _configured:
        return

    normalized_environment = environment.strip().lower() or "local"
    production = normalized_environment == "production"
    selected_format = (log_format or ("json" if production else "pretty_and_json_file")).lower()
    if selected_format not in {"pretty", "json", "pretty_and_json_file"}:
        selected_format = "json" if production else "pretty_and_json_file"
    if production:
        selected_format = "json"
        file_enabled = False
        local_log_content = "off"
    elif file_enabled is None:
        file_enabled = selected_format == "pretty_and_json_file"

    level = logging.getLevelNamesMapping().get(log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if selected_format == "json":
        stdout = logging.StreamHandler(sys.stdout)
        stdout.setFormatter(JsonEventFormatter(environment=normalized_environment))
        handlers.append(stdout)
    else:
        handlers.append(
            RichHandler(
                level=level,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                show_path=False,
                markup=False,
            )
        )

    configure_readable_request_log(
        enabled=bool(file_enabled and selected_format == "pretty_and_json_file"),
        file_path=file_path,
        content_mode=local_log_content,
    )

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
        force=True,
    )
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    configure_event_metadata(
        environment=normalized_environment,
        detailed_logs=detailed_logs,
    )
    configure_tracing(enabled=observability_enabled)
    _configured = True


def reset_logging_for_tests() -> None:
    global _configured
    for handler in logging.getLogger().handlers:
        handler.close()
    logging.getLogger().handlers.clear()
    reset_readable_log_for_tests()
    _configured = False
