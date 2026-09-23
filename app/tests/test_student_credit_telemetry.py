"""Safe request telemetry for student authorization lifecycle events."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from observability.context import bind_request_context
from observability.readable_log import format_request_block
from observability.summary import (
    begin_request_summary,
    current_request_summary,
    reset_request_summary,
)
from schemas.student_credits import StudentCreditPolicy
from services.student_credits.repository import StudentCreditRepository
from services.student_credits.runtime import StudentCreditRuntime
from tests.test_student_credits import (
    CREDIT_LEDGER_TABLE,
    USER_CREDITS_TABLE,
    FakeDynamoDBClient,
    _summary,
)

_USER = "student-telemetry"


def _runtime(client: Any, *, dry_run: bool = False) -> StudentCreditRuntime:
    return StudentCreditRuntime(
        policy=StudentCreditPolicy(
            enforcement_enabled=True,
            dry_run=dry_run,
            credits_per_usd=Decimal("50"),
            target_gross_margin=Decimal("0.40"),
            rounding_mode="CEIL",
            doubt_authorization_credits=5,
            practice_min_authorization_credits=5,
            practice_authorization_credits_per_question=1,
        ),
        repository=StudentCreditRepository(
            user_credits_table=USER_CREDITS_TABLE,
            credit_ledger_table=CREDIT_LEDGER_TABLE,
            client=client,
        ),
    )


@pytest.fixture
def summary_scope():
    with bind_request_context(request_id="req-sum", request_type="standalone"):
        token = begin_request_summary()
        try:
            yield
        finally:
            reset_request_summary(token)


def test_authorization_and_settlement_summary_is_safe_and_complete(
    monkeypatch: pytest.MonkeyPatch,
    summary_scope: None,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "services.student_credits.runtime.log_event",
        lambda name, **kwargs: events.append((name, kwargs.get("details") or {})),
    )
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 10}))
    runtime.ensure_can_start(_USER)
    runtime.authorize_doubt(user_id=_USER, reference_id="telemetry")
    runtime.settle(
        user_id=_USER,
        reference_id="telemetry",
        feature="doubt",
        operation_status="completed",
        summary=_summary(),
    )

    summary = current_request_summary()
    assert summary is not None
    assert summary.credit_preflight == "allowed"
    assert summary.credit_authorization_status == "authorized"
    assert summary.credit_authorization_credits == 5
    assert summary.credit_calculated == 1
    assert summary.credit_captured_credits == 1
    assert summary.credit_released_credits == 4
    assert _USER not in repr(events)


def test_overage_emits_high_severity_event_and_summary_flag(
    monkeypatch: pytest.MonkeyPatch,
    summary_scope: None,
) -> None:
    names: list[str] = []
    monkeypatch.setattr(
        "services.student_credits.runtime.log_event",
        lambda name, **_kwargs: names.append(name),
    )
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 10}))
    runtime.authorize_doubt(user_id=_USER, reference_id="overage")
    runtime.settle(
        user_id=_USER,
        reference_id="overage",
        feature="doubt",
        operation_status="completed",
        summary=_summary(cost="0.100"),
    )

    summary = current_request_summary()
    assert summary is not None and summary.credit_authorization_exceeded is True
    assert "STUDENT_CREDIT_AUTHORIZATION_EXCEEDED" in names


def test_dry_run_formats_the_existing_credit_section_without_wallet_mutation(
    summary_scope: None,
) -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets={_USER: 10}), dry_run=True)
    runtime.ensure_can_start(_USER)
    runtime.authorize_doubt(user_id=_USER, reference_id="dry-run")
    runtime.settle(
        user_id=_USER,
        reference_id="dry-run",
        feature="doubt",
        operation_status="completed",
        summary=_summary(),
    )

    text = format_request_block([], current_request_summary())
    assert "CREDITS" in text
    assert "Authorization: would_authorize" in text
    assert "Settlement: skipped_dry_run" in text
