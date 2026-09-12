"""Structured telemetry and request-summary visibility for credit outcomes.

CREDIT-STAB-2: the financially significant failure outcomes must be observable
without inspecting DynamoDB, while keeping the existing control flow.
CREDIT-STAB-3: the request summary shows what the credit runtime actually decided.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from observability.context import bind_execution_context, bind_request_context
from observability.llm_usage import record_llm_call
from observability.readable_log import format_request_block
from observability.summary import (
    begin_request_summary,
    current_request_summary,
    reset_request_summary,
)
from schemas.llm_usage import ProviderTokenUsage
from schemas.student_credits import StudentCreditPolicy
from services.llm.billing import OperationDescriptor, OperationUsageAccumulator
from services.student_credits.errors import (
    InsufficientStudentCreditsError,
    StudentCreditRepositoryError,
)
from services.student_credits.repository import StudentCreditRepository
from services.student_credits.runtime import StudentCreditRuntime
from tests.test_student_credits import (
    CREDIT_LEDGER_TABLE,
    USER_CREDITS_TABLE,
    FakeDynamoDBClient,
    _billing_config,
)

_USER = "student-telemetry"


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict]]:
    captured: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        "services.student_credits.runtime.log_event",
        lambda name, **kw: captured.append(
            (name, str(kw.get("status")), dict(kw.get("details") or {}))
        ),
    )
    monkeypatch.setattr("services.llm.billing.load_billing_config", _billing_config)
    return captured


def _runtime(client: Any, *, dry_run: bool = False) -> StudentCreditRuntime:
    return StudentCreditRuntime(
        policy=StudentCreditPolicy(
            enforcement_enabled=True,
            dry_run=dry_run,
            credits_per_usd=Decimal("50"),
            target_gross_margin=Decimal("0.40"),
            rounding_mode="CEIL",
        ),
        repository=StudentCreditRepository(
            user_credits_table=USER_CREDITS_TABLE,
            credit_ledger_table=CREDIT_LEDGER_TABLE,
            client=client,
        ),
    )


def _settle(runtime: StudentCreditRuntime, *, reference: str = "ref-tel") -> Any:
    accumulator = OperationUsageAccumulator(
        descriptor=OperationDescriptor(operation_id="op-tel", feature="doubt")
    )
    with bind_execution_context(
        operation_id="op-tel", feature="doubt", operation_accumulator=accumulator
    ):
        record_llm_call(
            request_id="req-tel",
            role="general.generator",
            provider="openai",
            model="gpt-test",
            deployment=None,
            attempt_type="primary",
            streaming=False,
            usage=ProviderTokenUsage(input_tokens=1_000_000, output_tokens=0),
            duration_ms=5,
            status="succeeded",
        )
        return runtime.settle(
            user_id=_USER,
            reference_id=reference,
            feature="doubt",
            operation_status="completed",
        )


def _settlement_events(events: list[tuple[str, str, dict]]) -> list[tuple[str, dict]]:
    return [
        (status, details)
        for name, status, details in events
        if name == "student_credit_settlement"
    ]


# --------------------------------------------------------------------------
# CREDIT-STAB-2
# --------------------------------------------------------------------------


def test_insufficient_credits_emits_settlement_telemetry_and_still_raises(
    events: list[tuple[str, str, dict]],
) -> None:
    client = FakeDynamoDBClient(wallets={_USER: 10})
    runtime = _runtime(client)

    with pytest.raises(InsufficientStudentCreditsError):
        _settle(runtime)

    statuses = [status for status, _ in _settlement_events(events)]
    assert "insufficient_credits" in statuses
    details = next(d for s, d in _settlement_events(events) if s == "insufficient_credits")
    assert details["credits_debited"] == 0
    assert details["student_credits_to_debit"] == 84
    assert "user_ref" in details and _USER not in str(details["user_ref"])
    # Control flow and financial state unchanged.
    assert client.wallets[_USER] == 10
    assert client.ledger == {}


def test_repository_unavailable_emits_settlement_telemetry_and_still_raises(
    events: list[tuple[str, str, dict]],
) -> None:
    class BrokenClient(FakeDynamoDBClient):
        def transact_write_items(self, **kwargs: Any) -> dict[str, Any]:
            raise StudentCreditRepositoryError("boom")

    client = BrokenClient(wallets={_USER: 500})
    runtime = _runtime(client)

    with pytest.raises(StudentCreditRepositoryError):
        _settle(runtime)

    statuses = [status for status, _ in _settlement_events(events)]
    assert "unavailable" in statuses
    assert client.wallets[_USER] == 500
    assert client.ledger == {}


def test_successful_outcomes_still_emit_their_existing_events(
    events: list[tuple[str, str, dict]],
) -> None:
    client = FakeDynamoDBClient(wallets={_USER: 500})
    runtime = _runtime(client)

    _settle(runtime)

    assert [status for status, _ in _settlement_events(events)] == ["settled"]


def test_telemetry_never_contains_the_raw_user_id(
    events: list[tuple[str, str, dict]],
) -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 500}))

    _settle(runtime)

    assert all(_USER not in str(details) for _name, _status, details in events)


# --------------------------------------------------------------------------
# CREDIT-STAB-3
# --------------------------------------------------------------------------


@pytest.fixture
def summary_scope(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("services.llm.billing.load_billing_config", _billing_config)
    with bind_request_context(request_id="req-sum", request_type="standalone"):
        token = begin_request_summary()
        try:
            yield
        finally:
            reset_request_summary(token)


def test_summary_reports_dry_run_outcome(summary_scope: None) -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 500}), dry_run=True)

    runtime.ensure_can_start(_USER)
    _settle(runtime)
    summary = current_request_summary()

    assert summary is not None
    assert summary.credit_mode == "dry_run"
    assert summary.credit_admission == "allowed"
    assert summary.credit_calculated == 84
    assert summary.credit_settlement == "skipped_dry_run"


def test_summary_reports_settled_outcome(summary_scope: None) -> None:
    client = FakeDynamoDBClient(wallets={_USER: 500})
    runtime = _runtime(client)

    runtime.ensure_can_start(_USER)
    _settle(runtime)
    summary = current_request_summary()

    assert summary is not None
    assert summary.credit_mode == "enforcing"
    assert summary.credit_admission == "allowed"
    assert summary.credit_calculated == 84
    assert summary.credit_settlement == "settled"
    assert client.wallets[_USER] == 416


def test_readable_log_renders_the_credit_section(summary_scope: None) -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 500}), dry_run=True)
    runtime.ensure_can_start(_USER)
    _settle(runtime)
    summary = current_request_summary()
    assert summary is not None

    text = format_request_block([], summary)

    assert "CREDITS" in text
    assert "Mode: dry_run" in text
    assert "Admission: allowed" in text
    assert "Calculated: 84" in text
    assert "Settlement: skipped_dry_run" in text


def test_readable_log_renders_a_settled_credit_section(summary_scope: None) -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 500}))
    runtime.ensure_can_start(_USER)
    _settle(runtime)
    summary = current_request_summary()
    assert summary is not None

    text = format_request_block([], summary)

    assert "Mode: enforcing" in text
    assert "Settlement: settled" in text


def test_readable_log_omits_the_credit_section_when_enforcement_disabled(
    summary_scope: None,
) -> None:
    """No credit runtime ran, so nothing may imply a wallet lookup happened."""
    summary = current_request_summary()
    assert summary is not None

    text = format_request_block([], summary)

    assert summary.credit_mode is None
    assert "CREDITS" not in text
    assert "Admission:" not in text


# --------------------------------------------------------------------------
# R-3 / R-4: the credit block must survive every settlement state
# --------------------------------------------------------------------------


def _incomplete_settle(runtime: StudentCreditRuntime) -> None:
    """Drive a real incomplete-usage block (a failed call with no usage)."""
    accumulator = OperationUsageAccumulator(
        descriptor=OperationDescriptor(operation_id="op-inc", feature="doubt")
    )
    with bind_execution_context(
        operation_id="op-inc", feature="doubt", operation_accumulator=accumulator
    ):
        record_llm_call(
            request_id="req-inc",
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
        )
        runtime.settle(
            user_id=_USER,
            reference_id="ref-inc",
            feature="doubt",
            operation_status="completed",
        )


def test_blocked_incomplete_usage_still_renders_the_credit_block(
    summary_scope: None,
) -> None:
    """The financially interesting no-charge case must stay visible."""
    from services.student_credits.errors import StudentCreditPricingIncompleteError

    client = FakeDynamoDBClient(wallets={_USER: 500})
    runtime = _runtime(client, dry_run=True)
    runtime.ensure_can_start(_USER)

    with pytest.raises(StudentCreditPricingIncompleteError):
        _incomplete_settle(runtime)

    summary = current_request_summary()
    assert summary is not None
    assert summary.credit_mode == "dry_run"
    assert summary.credit_settlement == "blocked_incomplete_usage"
    assert summary.credit_calculated is None

    text = format_request_block([], summary)
    assert "CREDITS" in text
    assert "Settlement: blocked_incomplete_usage" in text
    assert "Calculated: unavailable" in text
    assert "Debit: 0" in text
    assert client.wallets[_USER] == 500
    assert client.ledger == {}


def test_credit_block_renders_from_admission_alone(summary_scope: None) -> None:
    """Even if settlement never runs, admission alone must show the block."""
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 42}), dry_run=True)
    runtime.ensure_can_start(_USER)

    text = format_request_block([], current_request_summary())

    assert "CREDITS" in text
    assert "Admission: allowed" in text
    assert "Balance: 42" in text


def test_settled_block_reports_the_actual_debit(summary_scope: None) -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 500}))
    runtime.ensure_can_start(_USER)
    _settle(runtime)

    text = format_request_block([], current_request_summary())

    assert "Settlement: settled" in text
    assert "Debit: 84" in text
