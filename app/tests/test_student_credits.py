"""Student runtime credit enforcement: calculation, settlement, and safety."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest
from botocore.exceptions import ClientError

from observability.context import bind_execution_context
from observability.llm_usage import record_llm_call
from schemas.billing import BillingConfig
from schemas.llm_usage import ProviderTokenUsage
from schemas.student_credits import StudentCreditPolicy, StudentWalletBalance
from services.llm.billing import OperationDescriptor, OperationUsageAccumulator
from services.student_credits.calculator import calculate_credits
from services.student_credits.errors import (
    InsufficientStudentCreditsError,
    StudentCreditLedgerConflictError,
    StudentCreditPricingIncompleteError,
    StudentCreditRepositoryError,
)
from services.student_credits.repository import StudentCreditRepository
from services.student_credits.runtime import StudentCreditRuntime, doubt_reference_id

USER_CREDITS_TABLE = "UserCredits-test"
CREDIT_LEDGER_TABLE = "CreditLedger-test"


def _policy(*, dry_run: bool = False, **overrides: Any) -> StudentCreditPolicy:
    values: dict[str, Any] = {
        "enforcement_enabled": True,
        "dry_run": dry_run,
        "credits_per_usd": Decimal("50"),
        "target_gross_margin": Decimal("0.40"),
        "rounding_mode": "CEIL",
    }
    values.update(overrides)
    return StudentCreditPolicy(**values)


def _billing_config() -> BillingConfig:
    return BillingConfig.model_validate(
        {
            "billing_version": "billing-test-v1",
            "metering_enabled": True,
            "credit_debit_enabled": False,
            "models": [
                {
                    "provider": "openai",
                    "model_or_deployment": "gpt-test",
                    "input_cost_per_million_tokens": "1.00",
                    "output_cost_per_million_tokens": "2.00",
                    "effective_from": "2026-01-01",
                }
            ],
            "infra": {
                "doubt": {"fixed_cost": "0.001"},
                "quick_practice": {"fixed_cost": "0.003"},
                "mock_test": {"fixed_cost": "0.006"},
            },
            "credits": {"usd_per_credit": "0.001", "pricing_factor": "2"},
        }
    )


class FakeDynamoDBClient:
    """Minimal DynamoDB double covering only the two calls the runtime makes."""

    def __init__(self, *, wallets: dict[str, int] | None = None) -> None:
        self.wallets: dict[str, int] = dict(wallets or {})
        self.ledger: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, int] = {}
        self.transaction_calls = 0
        self.get_item_calls = 0

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        self.get_item_calls += 1
        table = kwargs["TableName"]
        if table == USER_CREDITS_TABLE:
            user_id = kwargs["Key"]["userId"]["S"]
            if user_id not in self.wallets:
                return {}
            return {
                "Item": {
                    "userId": {"S": user_id},
                    "creditsBalance": {"N": str(self.wallets[user_id])},
                }
            }
        ledger_id = kwargs["Key"]["ledgerId"]["S"]
        item = self.ledger.get(ledger_id)
        return {"Item": item} if item else {}

    def transact_write_items(self, **kwargs: Any) -> dict[str, Any]:
        self.transaction_calls += 1
        put, update = kwargs["TransactItems"]
        ledger_item = put["Put"]["Item"]
        ledger_id = ledger_item["ledgerId"]["S"]
        user_id = update["Update"]["Key"]["userId"]["S"]
        credits = int(update["Update"]["ExpressionAttributeValues"][":credits"]["N"])

        reasons: list[dict[str, str]] = []
        ledger_conflict = ledger_id in self.ledger
        wallet_blocked = (
            user_id not in self.wallets or self.wallets[user_id] < credits
        )
        if ledger_conflict or wallet_blocked:
            reasons.append(
                {"Code": "ConditionalCheckFailed"} if ledger_conflict else {"Code": "None"}
            )
            reasons.append(
                {"Code": "ConditionalCheckFailed"} if wallet_blocked else {"Code": "None"}
            )
            raise ClientError(
                {
                    "Error": {"Code": "TransactionCanceledException"},
                    "CancellationReasons": reasons,
                },
                "TransactWriteItems",
            )

        self.ledger[ledger_id] = ledger_item
        self.wallets[user_id] -= credits
        self.versions[user_id] = self.versions.get(user_id, 0) + 1
        return {}


def _runtime(
    client: FakeDynamoDBClient,
    *,
    policy: StudentCreditPolicy | None = None,
) -> StudentCreditRuntime:
    return StudentCreditRuntime(
        policy=policy or _policy(),
        repository=StudentCreditRepository(
            user_credits_table=USER_CREDITS_TABLE,
            credit_ledger_table=CREDIT_LEDGER_TABLE,
            client=client,
        ),
    )


def _record_usage(*, input_tokens: int, output_tokens: int, provider: str = "openai") -> None:
    record_llm_call(
        request_id="request-credit",
        role="general.generator",
        provider=provider,
        model="gpt-test",
        deployment=None,
        attempt_type="primary",
        streaming=False,
        usage=ProviderTokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
        duration_ms=5,
        status="succeeded",
    )


def _operation(monkeypatch: pytest.MonkeyPatch) -> OperationUsageAccumulator:
    monkeypatch.setattr("services.llm.billing.load_billing_config", _billing_config)
    return OperationUsageAccumulator(
        descriptor=OperationDescriptor(operation_id="operation-credit", feature="doubt")
    )


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cost_usd", "expected_credits"),
    [
        ("1.00", 84),
        ("0.10", 9),
        ("0.012", 1),
        ("0.0000001", 1),
        ("0", 0),
    ],
)
def test_calculation_applies_gross_margin_then_ceiling(
    cost_usd: str, expected_credits: int
) -> None:
    charge = calculate_credits(Decimal(cost_usd), policy=_policy())

    assert charge.student_credits_to_debit == expected_credits
    assert charge.actual_llm_cost_usd == Decimal(cost_usd)


def test_gross_margin_is_not_a_markup() -> None:
    """40% gross margin divides by 0.60; it never multiplies by 1.40."""
    charge = calculate_credits(Decimal("1.00"), policy=_policy())

    assert charge.customer_equivalent_usd > Decimal("1.66")
    assert charge.customer_equivalent_usd < Decimal("1.67")
    assert charge.student_credits_to_debit != 70


def test_calculation_is_decimal_exact_for_repeating_values() -> None:
    charge = calculate_credits(Decimal("0.003"), policy=_policy())

    assert charge.customer_equivalent_usd == Decimal("0.005")
    assert charge.student_credits_to_debit == 1


def test_configuration_changes_change_the_charge() -> None:
    doubled_rate = calculate_credits(
        Decimal("1.00"), policy=_policy(credits_per_usd=Decimal("100"))
    )
    zero_margin = calculate_credits(
        Decimal("1.00"), policy=_policy(target_gross_margin=Decimal("0"))
    )

    assert doubled_rate.student_credits_to_debit == 167
    assert zero_margin.student_credits_to_debit == 50


def test_negative_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_credits(Decimal("-0.01"), policy=_policy())


def test_multiple_llm_calls_are_summed_before_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 1000})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        _record_usage(input_tokens=0, output_tokens=1_000_000)
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-multi",
            feature="doubt",
            operation_status="completed",
        )

    # $1.00 input + $2.00 output = $3.00 -> 3/0.6*50 = 250 credits exactly.
    assert settlement.charge is not None
    assert settlement.charge.actual_llm_cost_usd == Decimal("3.00")
    assert settlement.credits_debited == 250


# ---------------------------------------------------------------------------
# Successful settlement
# ---------------------------------------------------------------------------


def test_successful_settlement_debits_wallet_and_writes_one_ledger_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-1",
            feature="doubt",
            operation_status="completed",
        )

    assert settlement.status == "settled"
    assert settlement.credits_debited == 84
    assert client.wallets["student-1"] == 416
    assert len(client.ledger) == 1
    ledger = client.ledger["ref-1"]
    assert ledger["direction"]["S"] == "DEBIT"
    assert ledger["source"]["S"] == "FEATURE"
    assert ledger["amount"]["N"] == "84"
    assert ledger["userId"]["S"] == "student-1"
    assert ledger["referenceId"]["S"] == "ref-1"
    metadata = json.loads(ledger["metadata"]["S"])
    assert metadata["actualLlmCostUsd"] == "1"
    assert metadata["llmCallCount"] == 1


def test_ledger_debit_equals_wallet_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=500_000, output_tokens=250_000)
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-recon",
            feature="doubt",
            operation_status="completed",
        )

    debited = int(client.ledger["ref-recon"]["amount"]["N"])
    assert debited == settlement.credits_debited
    assert 500 - client.wallets["student-1"] == debited


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_same_turn_settled_twice_debits_once(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    reference = doubt_reference_id(user_id="student-1", turn_id="turn-9")

    for _ in range(2):
        accumulator = _operation(monkeypatch)
        with bind_execution_context(
            operation_id="operation-credit",
            feature="doubt",
            operation_accumulator=accumulator,
        ):
            _record_usage(input_tokens=1_000_000, output_tokens=0)
            runtime.settle(
                user_id="student-1",
                reference_id=reference,
                feature="doubt",
                operation_status="completed",
            )

    assert client.wallets["student-1"] == 416
    assert len(client.ledger) == 1
    assert client.transaction_calls == 2


def test_lost_response_retry_recovers_the_existing_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        runtime.settle(
            user_id="student-1",
            reference_id="ref-lost",
            feature="doubt",
            operation_status="completed",
        )

    retry_accumulator = _operation(monkeypatch)
    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=retry_accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        retry = runtime.settle(
            user_id="student-1",
            reference_id="ref-lost",
            feature="doubt",
            operation_status="completed",
        )

    assert retry.status == "already_settled"
    assert retry.credits_debited == 84
    assert client.wallets["student-1"] == 416


def test_reference_id_is_deterministic_per_user_and_turn() -> None:
    first = doubt_reference_id(user_id="student-1", turn_id="turn-1")

    assert first == doubt_reference_id(user_id="student-1", turn_id="turn-1")
    assert first != doubt_reference_id(user_id="student-2", turn_id="turn-1")
    assert first != doubt_reference_id(user_id="student-1", turn_id="turn-2")


# ---------------------------------------------------------------------------
# Non-chargeable operations
# ---------------------------------------------------------------------------


def test_zero_llm_calls_never_debits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed-turn replay reaches settlement with an empty accumulator."""
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-replay",
            feature="doubt",
            operation_status="completed",
        )

    assert settlement.status == "skipped_not_chargeable"
    assert settlement.credits_debited == 0
    assert client.wallets["student-1"] == 500
    assert client.transaction_calls == 0


def test_replay_is_not_charged_the_infrastructure_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy shadow billing bills an infra floor for zero calls; students never do."""
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-floor",
            feature="doubt",
            operation_status="completed",
        )

    assert settlement.charge is None
    assert client.ledger == {}


def test_zero_cost_usage_never_debits(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=0, output_tokens=0)
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-zero",
            feature="doubt",
            operation_status="completed",
        )

    assert settlement.status == "skipped_zero_cost"
    assert client.wallets["student-1"] == 500
    assert client.transaction_calls == 0


def test_metering_unavailable_never_debits() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})

    settlement = _runtime(client).settle(
        user_id="student-1",
        reference_id="ref-nometer",
        feature="doubt",
        operation_status="completed",
    )

    assert settlement.status == "skipped_not_chargeable"
    assert client.transaction_calls == 0


# ---------------------------------------------------------------------------
# Pricing coverage
# ---------------------------------------------------------------------------


def test_unpriced_model_blocks_settlement_instead_of_charging_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000, output_tokens=500, provider="azure_openai")
        with pytest.raises(StudentCreditPricingIncompleteError):
            runtime.settle(
                user_id="student-1",
                reference_id="ref-unpriced",
                feature="doubt",
                operation_status="completed",
            )

    assert client.transaction_calls == 0
    assert client.ledger == {}
    assert client.wallets["student-1"] == 500


# ---------------------------------------------------------------------------
# Balance, admission, and concurrency
# ---------------------------------------------------------------------------


def test_missing_wallet_is_not_created_and_blocks_admission() -> None:
    client = FakeDynamoDBClient(wallets={})
    runtime = _runtime(client)

    balance = runtime.get_balance("student-unknown")
    assert balance == StudentWalletBalance(
        user_id="student-unknown", exists=False, credits_balance=0
    )
    with pytest.raises(InsufficientStudentCreditsError):
        runtime.ensure_can_start("student-unknown")
    assert client.wallets == {}


def test_zero_balance_blocks_admission() -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets={"student-1": 0}))

    with pytest.raises(InsufficientStudentCreditsError):
        runtime.ensure_can_start("student-1")


def test_sufficient_balance_admits_with_one_read() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 1})
    runtime = _runtime(client)

    assert runtime.ensure_can_start("student-1").credits_balance == 1
    assert client.get_item_calls == 1


def test_wallet_read_failure_fails_closed() -> None:
    class BrokenClient(FakeDynamoDBClient):
        def get_item(self, **kwargs: Any) -> dict[str, Any]:
            raise ClientError({"Error": {"Code": "InternalServerError"}}, "GetItem")

    runtime = _runtime(BrokenClient())

    with pytest.raises(StudentCreditRepositoryError):
        runtime.ensure_can_start("student-1")


def test_insufficient_balance_at_settlement_leaves_wallet_and_ledger_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
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
                user_id="student-1",
                reference_id="ref-broke",
                feature="doubt",
                operation_status="completed",
            )

    assert client.wallets["student-1"] == 10
    assert client.ledger == {}


def test_concurrent_operations_cannot_overspend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two distinct operations compete for a balance that covers only one."""
    client = FakeDynamoDBClient(wallets={"student-1": 100})
    runtime = _runtime(client)
    settled = 0
    refused = 0

    for reference in ("ref-a", "ref-b"):
        accumulator = _operation(monkeypatch)
        with bind_execution_context(
            operation_id="operation-credit",
            feature="doubt",
            operation_accumulator=accumulator,
        ):
            _record_usage(input_tokens=1_000_000, output_tokens=0)
            try:
                runtime.settle(
                    user_id="student-1",
                    reference_id=reference,
                    feature="doubt",
                    operation_status="completed",
                )
                settled += 1
            except InsufficientStudentCreditsError:
                refused += 1

    assert settled == 1
    assert refused == 1
    assert client.wallets["student-1"] == 16
    assert client.wallets["student-1"] >= 0
    assert len(client.ledger) == 1


def test_exact_balance_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 84})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-exact",
            feature="doubt",
            operation_status="completed",
        )

    assert settlement.credits_debited == 84
    assert client.wallets["student-1"] == 0


def test_duplicate_settlement_wins_over_insufficient_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retry after the balance was spent elsewhere still reports already-settled."""
    client = FakeDynamoDBClient(wallets={"student-1": 84})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        runtime.settle(
            user_id="student-1",
            reference_id="ref-dup",
            feature="doubt",
            operation_status="completed",
        )
        retry = runtime.settle(
            user_id="student-1",
            reference_id="ref-dup",
            feature="doubt",
            operation_status="completed",
        )

    assert client.wallets["student-1"] == 0
    assert retry.status == "already_settled"


def test_one_student_cannot_debit_another_wallet(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500, "student-2": 500})
    runtime = _runtime(client)
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        runtime.settle(
            user_id="student-1",
            reference_id=doubt_reference_id(user_id="student-1", turn_id="turn-1"),
            feature="doubt",
            operation_status="completed",
        )

    assert client.wallets["student-2"] == 500
    assert client.ledger[
        doubt_reference_id(user_id="student-1", turn_id="turn-1")
    ]["userId"]["S"] == "student-1"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_calculates_without_touching_the_wallet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    runtime = _runtime(client, policy=_policy(dry_run=True))
    accumulator = _operation(monkeypatch)

    with bind_execution_context(
        operation_id="operation-credit",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        _record_usage(input_tokens=1_000_000, output_tokens=0)
        settlement = runtime.settle(
            user_id="student-1",
            reference_id="ref-dry",
            feature="doubt",
            operation_status="completed",
        )

    assert settlement.status == "skipped_dry_run"
    assert settlement.credits_debited == 0
    assert settlement.charge is not None
    assert settlement.charge.student_credits_to_debit == 84
    assert client.wallets["student-1"] == 500
    assert client.transaction_calls == 0
    assert client.ledger == {}


def test_dry_run_never_refuses_a_student() -> None:
    runtime = _runtime(
        FakeDynamoDBClient(wallets={"student-1": 0}), policy=_policy(dry_run=True)
    )

    assert runtime.ensure_can_start("student-1").credits_balance == 0


# ---------------------------------------------------------------------------
# Repository contract
# ---------------------------------------------------------------------------


def test_settlement_condition_expressions_are_least_privilege() -> None:
    captured: dict[str, Any] = {}

    class CapturingClient(FakeDynamoDBClient):
        def transact_write_items(self, **kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            return {}

    repository = StudentCreditRepository(
        user_credits_table=USER_CREDITS_TABLE,
        credit_ledger_table=CREDIT_LEDGER_TABLE,
        client=CapturingClient(wallets={"student-1": 500}),
    )
    repository.settle(
        reference_id="ref-guard",
        user_id="student-1",
        credits=7,
        feature_key="doubt",
        metadata={},
    )

    put, update = captured["TransactItems"]
    assert put["Put"]["ConditionExpression"] == "attribute_not_exists(#ledgerId)"
    assert (
        update["Update"]["ConditionExpression"]
        == "attribute_exists(#userId) AND #creditsBalance >= :credits"
    )
    assert "ADD #creditsBalance :debit, #version :one" in update["Update"]["UpdateExpression"]
    assert update["Update"]["ExpressionAttributeValues"][":debit"]["N"] == "-7"
    assert update["Update"]["Key"] == {"userId": {"S": "student-1"}}


def test_settle_rejects_non_positive_credits() -> None:
    repository = StudentCreditRepository(
        user_credits_table=USER_CREDITS_TABLE,
        credit_ledger_table=CREDIT_LEDGER_TABLE,
        client=FakeDynamoDBClient(),
    )

    with pytest.raises(ValueError, match="positive credit amount"):
        repository.settle(
            reference_id="ref",
            user_id="student-1",
            credits=0,
            feature_key="doubt",
            metadata={},
        )


def test_ledger_conflict_is_distinguished_from_insufficient_balance() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 500})
    repository = StudentCreditRepository(
        user_credits_table=USER_CREDITS_TABLE,
        credit_ledger_table=CREDIT_LEDGER_TABLE,
        client=client,
    )
    repository.settle(
        reference_id="ref-x",
        user_id="student-1",
        credits=5,
        feature_key="doubt",
        metadata={},
    )

    with pytest.raises(StudentCreditLedgerConflictError):
        repository.settle(
            reference_id="ref-x",
            user_id="student-1",
            credits=5,
            feature_key="doubt",
            metadata={},
        )
    assert repository.read_settlement("ref-x") == 5
