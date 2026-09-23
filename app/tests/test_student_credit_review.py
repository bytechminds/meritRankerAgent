"""Independent adversarial checks for the held-credit repository boundary."""

from __future__ import annotations

from decimal import Decimal

import pytest

from schemas.student_credits import StudentCreditPolicy
from services.student_credits.errors import InsufficientStudentCreditsError
from services.student_credits.repository import StudentCreditRepository
from services.student_credits.runtime import StudentCreditRuntime
from tests.test_student_credits import (
    CREDIT_LEDGER_TABLE,
    USER_CREDITS_TABLE,
    FakeDynamoDBClient,
    _summary,
)


def _runtime(client: FakeDynamoDBClient) -> StudentCreditRuntime:
    return StudentCreditRuntime(
        policy=StudentCreditPolicy(
            enforcement_enabled=True,
            dry_run=False,
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


def test_authorization_does_not_depend_on_the_stale_preflight_balance() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    runtime.ensure_can_start("student-1")
    client.wallets["student-1"] = 4

    with pytest.raises(InsufficientStudentCreditsError):
        runtime.authorize_doubt(user_id="student-1", reference_id="stale-read")

    assert client.wallets["student-1"] == 4
    assert client.ledger == {}


def test_authorization_cannot_charge_a_different_wallet_from_its_reference() -> None:
    client = FakeDynamoDBClient(wallets={"victim": 100, "attacker": 10})
    runtime = _runtime(client)
    runtime.authorize_doubt(
        user_id="attacker",
        reference_id="student-credit:doubt:attacker:victim-turn",
    )

    assert client.wallets == {"victim": 100, "attacker": 5}


def test_release_after_settlement_does_not_refund_a_completed_operation() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    runtime.authorize_doubt(user_id="student-1", reference_id="settled")
    runtime.settle(
        user_id="student-1",
        reference_id="settled",
        feature="doubt",
        operation_status="completed",
        summary=_summary(),
    )
    runtime.release(
        user_id="student-1",
        reference_id="settled",
        feature="doubt",
        reason="late_failure",
    )

    assert client.wallets["student-1"] == 9
    assert client.ledger["settled"]["authorizationState"]["S"] == "SETTLED"


def test_authorization_then_release_never_drives_wallet_negative() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 5})
    runtime = _runtime(client)
    runtime.authorize_doubt(user_id="student-1", reference_id="exact")
    runtime.release(
        user_id="student-1",
        reference_id="exact",
        feature="doubt",
        reason="cancelled",
    )

    assert client.wallets["student-1"] == 5
    assert min(client.wallets.values()) >= 0
