"""Adversarial Phase 2 review checks for student credit enforcement.

These are deliberately hostile: they assume the implementation is wrong and try
to produce a double debit, a negative wallet, a cross-wallet debit, or a
blocked student during observational dry run.
"""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest
from botocore.exceptions import ClientError

from observability.context import bind_execution_context
from schemas.student_credits import StudentCreditPolicy
from services.student_credits.errors import (
    InsufficientStudentCreditsError,
    StudentCreditRepositoryError,
)
from services.student_credits.repository import StudentCreditRepository
from services.student_credits.runtime import StudentCreditRuntime, doubt_reference_id
from tests.test_student_credits import (
    CREDIT_LEDGER_TABLE,
    USER_CREDITS_TABLE,
    FakeDynamoDBClient,
    _billing_config,
    _operation,
    _record_usage,
)


def _runtime(
    client: FakeDynamoDBClient, *, dry_run: bool = False
) -> StudentCreditRuntime:
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


class UnreadableWalletClient(FakeDynamoDBClient):
    """Credit store that cannot serve the admission read."""

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["TableName"] == USER_CREDITS_TABLE:
            raise ClientError(
                {"Error": {"Code": "ProvisionedThroughputExceededException"}}, "GetItem"
            )
        return super().get_item(**kwargs)


# ---------------------------------------------------------------------------
# Role 8 — admission behaviour
# ---------------------------------------------------------------------------


def test_dry_run_continues_when_the_credit_store_is_unavailable() -> None:
    """Dry run is observational: an outage must not block Doubt Solver."""
    runtime = _runtime(UnreadableWalletClient(), dry_run=True)

    assert runtime.ensure_can_start("student-1") is None


def test_dry_run_outage_records_unavailable_rather_than_claiming_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any] | None]] = []
    monkeypatch.setattr(
        "services.student_credits.runtime.log_event",
        lambda name, **kwargs: events.append((name, kwargs.get("details"))),
    )
    runtime = _runtime(UnreadableWalletClient(), dry_run=True)

    runtime.ensure_can_start("student-1")

    assert [name for name, _ in events] == ["student_credit_admission_checked"]
    details = events[0][1]
    assert details is not None
    assert details["student_credit_admission_status"] == "unavailable"
    assert details["wallet_exists"] is None
    assert details["credits_balance"] is None
    assert details["dry_run"] is True


def test_enforcing_mode_fails_closed_when_the_wallet_is_unreadable() -> None:
    runtime = _runtime(UnreadableWalletClient(), dry_run=False)

    with pytest.raises(StudentCreditRepositoryError):
        runtime.ensure_can_start("student-1")


def test_admission_telemetry_never_contains_a_raw_user_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "services.student_credits.runtime.log_event",
        lambda name, **kwargs: events.append(kwargs.get("details") or {}),
    )
    runtime = _runtime(FakeDynamoDBClient(wallets={"cognito-sub-abcdef": 100}))

    runtime.ensure_can_start("cognito-sub-abcdef")

    serialized = repr(events)
    assert "cognito-sub-abcdef" not in serialized
    assert len(events[0]["user_ref"]) == 16


# ---------------------------------------------------------------------------
# Role 6 — idempotency
# ---------------------------------------------------------------------------


def test_settling_the_same_operation_three_times_debits_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    reference = doubt_reference_id(user_id="student-1", turn_id="turn-triple")
    statuses = []

    for _ in range(3):
        accumulator = _operation(monkeypatch)
        with bind_execution_context(
            operation_id="operation-credit",
            feature="doubt",
            operation_accumulator=accumulator,
        ):
            _record_usage(input_tokens=1_000_000, output_tokens=0)
            statuses.append(
                runtime.settle(
                    user_id="student-1",
                    reference_id=reference,
                    feature="doubt",
                    operation_status="completed",
                ).status
            )

    assert statuses == ["settled", "already_settled", "already_settled"]
    assert client.wallets["student-1"] == 416
    assert len(client.ledger) == 1
    assert client.ledger[reference]["amount"]["N"] == "84"


def test_lost_response_retry_returns_the_recorded_amount_not_a_recomputed_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry whose usage differs must still report the committed debit."""
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    reference = doubt_reference_id(user_id="student-1", turn_id="turn-lost")

    accumulator = _operation(monkeypatch)
    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        first = runtime.settle(
            user_id="student-1",
            reference_id=reference,
            feature="doubt",
            operation_status="completed",
        )

    retry_accumulator = _operation(monkeypatch)
    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=retry_accumulator,
    ):
        _record_usage(input_tokens=3_000_000, output_tokens=0)
        retry = runtime.settle(
            user_id="student-1",
            reference_id=reference,
            feature="doubt",
            operation_status="completed",
        )

    assert first.credits_debited == 84
    assert retry.status == "already_settled"
    assert retry.credits_debited == 84
    assert client.wallets["student-1"] == 416


# ---------------------------------------------------------------------------
# Role 6 — concurrency
# ---------------------------------------------------------------------------


def test_concurrent_credit_during_debit_does_not_invalidate_the_debit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ADD choice over CAS: an unrelated wallet CREDIT must not cancel a debit."""
    client = FakeDynamoDBClient(wallets={"student-1": 100})
    runtime = _runtime(client)

    runtime.ensure_can_start("student-1")
    # A purchase lands between the admission read and settlement. A stale
    # balance/version compare-and-set would cancel here; a conditional ADD
    # must not.
    client.wallets["student-1"] += 500

    accumulator = _operation(monkeypatch)
    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-credit-race",
            feature="doubt",
            operation_status="completed",
        )

    assert settlement.status == "settled"
    assert settlement.credits_debited == 84
    assert client.wallets["student-1"] == 100 + 500 - 84


def test_two_distinct_operations_cannot_drive_the_wallet_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 100})
    runtime = _runtime(client)
    outcomes: list[str] = []

    for reference in ("ref-p", "ref-q"):
        accumulator = _operation(monkeypatch)
        with bind_execution_context(
            operation_id="operation-credit",
            feature="doubt",
            operation_accumulator=accumulator,
        ):
            _record_usage(input_tokens=1_000_000, output_tokens=0)
            try:
                outcomes.append(
                    runtime.settle(
                        user_id="student-1",
                        reference_id=reference,
                        feature="doubt",
                        operation_status="completed",
                    ).status
                )
            except InsufficientStudentCreditsError:
                outcomes.append("insufficient")

    assert outcomes == ["settled", "insufficient"]
    assert client.wallets["student-1"] == 16
    assert client.wallets["student-1"] >= 0
    assert list(client.ledger) == ["ref-p"]


# ---------------------------------------------------------------------------
# Role 4 — identity
# ---------------------------------------------------------------------------


def test_reference_id_cannot_collide_across_users(monkeypatch: pytest.MonkeyPatch) -> None:
    """turn_id admits no colon, so the trailing segment always recovers the turn."""
    from schemas.doubt_solver import DoubtSolverRequest

    turn_pattern = DoubtSolverRequest.model_fields["turn_id"].metadata
    assert any(":" not in getattr(m, "pattern", "") for m in turn_pattern)

    # A user id may contain a colon; a turn id may not. The pair therefore
    # remains unambiguous because the turn is always the final segment.
    a = doubt_reference_id(user_id="tenant:student-1", turn_id="turn-1")
    b = doubt_reference_id(user_id="tenant", turn_id="student-1")
    assert a != b


def test_settlement_only_ever_addresses_the_supplied_wallet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    class CapturingClient(FakeDynamoDBClient):
        def transact_write_items(self, **kwargs: Any) -> dict[str, Any]:
            captured.append(kwargs)
            return super().transact_write_items(**kwargs)

    client = CapturingClient(wallets={"victim": 1000, "attacker": 500})
    runtime = _runtime(client)

    accumulator = _operation(monkeypatch)
    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        # A hostile turn id naming another wallet must not redirect the debit.
        runtime.settle(
            user_id="attacker",
            reference_id=doubt_reference_id(
                user_id="attacker", turn_id="victim-turn-1"
            ),
            feature="doubt",
            operation_status="completed",
        )

    assert client.wallets["victim"] == 1000
    assert client.wallets["attacker"] == 416
    put, update = captured[0]["TransactItems"]
    # The wallet addressed is the Key alone; the reference string is opaque data
    # and can never redirect the update.
    assert update["Update"]["Key"] == {"userId": {"S": "attacker"}}
    assert put["Put"]["Item"]["userId"] == {"S": "attacker"}
    assert "attribute_exists(#userId)" in update["Update"]["ConditionExpression"]
    assert update["Update"]["TableName"] == USER_CREDITS_TABLE


def test_wallet_is_never_created_for_an_unknown_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={})
    runtime = _runtime(client)

    accumulator = _operation(monkeypatch)
    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        with pytest.raises(InsufficientStudentCreditsError):
            runtime.settle(
                user_id="never-seen",
                reference_id="ref-ghost",
                feature="doubt",
                operation_status="completed",
            )

    assert client.wallets == {}
    assert client.ledger == {}


# ---------------------------------------------------------------------------
# Role 1 — arithmetic reconciliation at the boundary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "expected_credits"),
    [
        (1_000_000, 0, 84),
        (0, 1_000_000, 167),
        (1_000_000, 1_000_000, 250),
        (1_000, 500, 1),
    ],
)
def test_ledger_debit_equals_formula_and_wallet_delta(
    monkeypatch: pytest.MonkeyPatch,
    input_tokens: int,
    output_tokens: int,
    expected_credits: int,
) -> None:
    """§35 reconciliation: formula == ledger amount == wallet delta."""
    client = FakeDynamoDBClient(wallets={"student-1": 5000})
    runtime = _runtime(client)
    config = _billing_config()

    accumulator = _operation(monkeypatch)
    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=input_tokens, output_tokens=output_tokens)
        settlement = runtime.settle(
            user_id="student-1",
            reference_id=f"ref-{input_tokens}-{output_tokens}",
            feature="doubt",
            operation_status="completed",
        )

    rate = config.models[0]
    cost = (
        Decimal(input_tokens) * rate.input_cost_per_million_tokens
        + Decimal(output_tokens) * rate.output_cost_per_million_tokens
    ) / Decimal("1000000")
    # Independent of Decimal and of the implementation: exact rational
    # arithmetic plus the stdlib ceiling.
    independent = math.ceil(
        Fraction(cost) * Fraction(50) / Fraction(6, 10)
    )

    assert settlement.charge is not None
    assert settlement.charge.actual_llm_cost_usd == cost
    assert independent == expected_credits
    assert settlement.credits_debited == expected_credits
    ledger_amount = int(
        client.ledger[f"ref-{input_tokens}-{output_tokens}"]["amount"]["N"]
    )
    assert ledger_amount == expected_credits
    assert 5000 - client.wallets["student-1"] == expected_credits
