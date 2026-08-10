"""Safe, request-scoped observability primitives for the agent runtime."""

from observability.context import (
    ExecutionContext,
    RequestContext,
    bind_execution_context,
    bind_request_context,
    current_execution_context,
    current_request_context,
    update_request_type,
)
from observability.events import log_event
from observability.lifecycle import observe_invocation
from observability.llm_usage import (
    bind_llm_attempt_type,
    count_generator_calls,
    current_llm_attempt_type,
    record_llm_call,
    snapshot_llm_usage_records,
)
from observability.readable_log import configure_runtime_identity, record_local_preview
from observability.summary import (
    RequestExecutionSummary,
    begin_request_summary,
    current_request_summary,
    emit_request_summary,
    update_request_summary,
)
from observability.tracing import configure_tracing, stage_span

__all__ = [
    "RequestContext",
    "ExecutionContext",
    "RequestExecutionSummary",
    "begin_request_summary",
    "bind_execution_context",
    "bind_request_context",
    "bind_llm_attempt_type",
    "count_generator_calls",
    "configure_tracing",
    "configure_runtime_identity",
    "current_request_context",
    "current_execution_context",
    "current_request_summary",
    "current_llm_attempt_type",
    "emit_request_summary",
    "log_event",
    "observe_invocation",
    "record_local_preview",
    "record_llm_call",
    "snapshot_llm_usage_records",
    "stage_span",
    "update_request_type",
    "update_request_summary",
]
