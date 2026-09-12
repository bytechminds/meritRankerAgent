"""R-2: provider failure_kind survives into usage telemetry."""

from __future__ import annotations

import pytest

from observability.context import bind_request_context
from observability.llm_usage import record_llm_call, snapshot_llm_usage_records
from observability.summary import begin_request_summary, reset_request_summary
from schemas.llm_usage import LLMUsageRecord, ProviderTokenUsage
from services.llm.providers.errors import LlmProviderExecutionError

FAILURE_KINDS = [
    "timeout",
    "rate_limited",
    "authentication_failed",
    "provider_unavailable",
    "invalid_request",
    "insufficient_quota",
]


@pytest.fixture
def usage_scope():
    with bind_request_context(request_id="req-fk", request_type="standalone"):
        token = begin_request_summary()
        try:
            yield
        finally:
            reset_request_summary(token)


def _record(**kw):
    record_llm_call(
        request_id="req-fk",
        role="general.classifier",
        provider="gemini",
        model="gemini-3.1-flash-lite",
        deployment=None,
        attempt_type="primary",
        streaming=False,
        usage=ProviderTokenUsage(),
        duration_ms=10,
        status="failed",
        error_type="LlmProviderExecutionError",
        **kw,
    )
    return snapshot_llm_usage_records()[-1]


def test_schema_carries_failure_kind() -> None:
    assert "failure_kind" in LLMUsageRecord.model_fields


@pytest.mark.parametrize("kind", FAILURE_KINDS)
def test_each_failure_kind_is_preserved(usage_scope: None, kind: str) -> None:
    assert _record(failure_kind=kind).failure_kind == kind


def test_absent_failure_kind_stays_none(usage_scope: None) -> None:
    assert _record().failure_kind is None


def test_exception_exposes_failure_kind_for_the_recorder() -> None:
    """The attribute the execution boundary reads via getattr."""
    exc = LlmProviderExecutionError("boom", failure_kind="timeout")

    assert getattr(exc, "failure_kind", None) == "timeout"


def test_failure_kind_is_not_treated_as_a_sensitive_key() -> None:
    """It must survive event-detail sanitisation like error_type already does."""
    from observability.events import _safe_key

    assert _safe_key("failure_kind") == "failure_kind"
