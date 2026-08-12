"""Request identity propagation based on context-local state."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from services.llm.billing import OperationUsageAccumulator

RequestType = Literal["standalone", "follow_up", "image", "pending", "unknown"]

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,127}$")


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    trace_id: str
    conversation_id: str | None = None
    turn_id: str | None = None
    request_type: RequestType = "unknown"


_request_context: ContextVar[RequestContext | None] = ContextVar(
    "agent_request_context", default=None
)


@dataclass(frozen=True)
class ExecutionContext:
    """Safe operation correlation for work executed beneath one request."""

    activity_id: str | None = None
    batch_id: str | None = None
    slot_ids: tuple[str, ...] = ()
    operation_id: str | None = None
    feature: str | None = None
    operation_accumulator: OperationUsageAccumulator | None = None


_execution_context: ContextVar[ExecutionContext | None] = ContextVar(
    "agent_execution_context", default=None
)


def safe_identifier(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not _SAFE_IDENTIFIER.fullmatch(normalized):
        return None
    return normalized


def new_trace_id() -> str:
    try:
        from opentelemetry import trace  # noqa: PLC0415

        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            return f"{span_context.trace_id:032x}"
    except Exception:
        pass
    return uuid.uuid4().hex


def current_request_context() -> RequestContext | None:
    return _request_context.get()


def current_execution_context() -> ExecutionContext | None:
    return _execution_context.get()


def update_request_type(request_type: RequestType) -> RequestContext | None:
    current = _request_context.get()
    if current is None:
        return None
    updated = RequestContext(
        request_id=current.request_id,
        trace_id=current.trace_id,
        conversation_id=current.conversation_id,
        turn_id=current.turn_id,
        request_type=request_type,
    )
    _request_context.set(updated)
    return updated


def update_trace_id(trace_id: str) -> RequestContext | None:
    current = _request_context.get()
    safe_trace_id = safe_identifier(trace_id)
    if current is None or safe_trace_id is None:
        return current
    updated = RequestContext(
        request_id=current.request_id,
        trace_id=safe_trace_id,
        conversation_id=current.conversation_id,
        turn_id=current.turn_id,
        request_type=current.request_type,
    )
    _request_context.set(updated)
    return updated


def set_request_context(context: RequestContext) -> Token[RequestContext | None]:
    return _request_context.set(context)


def reset_request_context(token: Token[RequestContext | None]) -> None:
    _request_context.reset(token)


@contextmanager
def bind_request_context(
    *,
    request_id: str,
    trace_id: str | None = None,
    conversation_id: str | None = None,
    turn_id: str | None = None,
    request_type: RequestType = "unknown",
) -> Iterator[RequestContext]:
    context = RequestContext(
        request_id=safe_identifier(request_id) or str(uuid.uuid4()),
        trace_id=safe_identifier(trace_id) or new_trace_id(),
        conversation_id=safe_identifier(conversation_id),
        turn_id=safe_identifier(turn_id),
        request_type=request_type,
    )
    token = set_request_context(context)
    otel_token: object | None = None
    try:
        if context.conversation_id:
            from opentelemetry import baggage  # noqa: PLC0415
            from opentelemetry import context as otel_context  # noqa: PLC0415

            baggage_context = baggage.set_baggage(
                "session.id", context.conversation_id
            )
            otel_token = otel_context.attach(baggage_context)
    except Exception:
        otel_token = None
    try:
        yield context
    finally:
        if otel_token is not None:
            try:
                otel_context.detach(otel_token)
            except Exception:
                pass
        reset_request_context(token)


@contextmanager
def bind_execution_context(
    *,
    activity_id: str | None = None,
    batch_id: str | None = None,
    slot_ids: tuple[str, ...] = (),
    operation_id: str | None = None,
    feature: str | None = None,
    operation_accumulator: OperationUsageAccumulator | None = None,
) -> Iterator[ExecutionContext]:
    """Bind server-issued operation identifiers without user content."""
    parent = _execution_context.get() or ExecutionContext()
    context = ExecutionContext(
        activity_id=(
            safe_identifier(activity_id)
            if activity_id is not None
            else parent.activity_id
        ),
        batch_id=(
            safe_identifier(batch_id)
            if batch_id is not None
            else parent.batch_id
        ),
        slot_ids=(
            tuple(
                safe_slot_id
                for value in slot_ids[:100]
                if (safe_slot_id := safe_identifier(value)) is not None
            )
            if slot_ids
            else parent.slot_ids
        ),
        operation_id=(
            safe_identifier(operation_id)
            if operation_id is not None
            else parent.operation_id
        ),
        feature=(
            safe_identifier(feature)
            if feature is not None
            else parent.feature
        ),
        operation_accumulator=(
            operation_accumulator
            if operation_accumulator is not None
            else parent.operation_accumulator
        ),
    )
    token = _execution_context.set(context)
    try:
        yield context
    finally:
        _execution_context.reset(token)
