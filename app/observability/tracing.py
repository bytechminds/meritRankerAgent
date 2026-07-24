"""Optional OpenTelemetry spans with a dependency-free no-op fallback."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from observability.context import update_trace_id

_enabled = False

try:
    from opentelemetry import trace as _otel_trace
except ImportError:  # AgentCore local mode does not install ADOT by default.
    _otel_trace = None


def configure_tracing(*, enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def tracing_available() -> bool:
    return _enabled and _otel_trace is not None


@contextmanager
def stage_span(name: str, **attributes: str | bool | int | float | None) -> Iterator[object | None]:
    if not tracing_available():
        yield None
        return
    try:
        tracer = _otel_trace.get_tracer("meritranker-tutor")
        manager = tracer.start_as_current_span(name)
        span = manager.__enter__()
    except Exception:  # telemetry setup is never an application failure
        yield None
        return
    try:
        span_context = span.get_span_context()
        if span_context.is_valid:
            update_trace_id(f"{span_context.trace_id:032x}")
    except Exception:
        pass
    application_error: BaseException | None = None
    try:
        for key, value in attributes.items():
            if value is not None and key not in {"query", "prompt", "context", "message", "answer"}:
                try:
                    span.set_attribute(f"agent.{key}", value)
                except Exception:
                    continue
        yield span
    except BaseException as exc:
        application_error = exc
        raise
    finally:
        try:
            manager.__exit__(
                type(application_error) if application_error is not None else None,
                application_error,
                application_error.__traceback__ if application_error is not None else None,
            )
        except Exception:
            pass
