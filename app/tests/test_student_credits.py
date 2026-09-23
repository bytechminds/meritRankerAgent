"""Focused contract tests for student credit authorization and settlement."""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any

import pytest
from botocore.exceptions import ClientError

from observability.llm_usage import record_llm_call
from schemas.billing import BillingConfig, OperationBillingSummary
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
from services.student_credits.runtime import (
    StudentCreditRuntime,
    doubt_reference_id,
    practice_reference_id,
)
from student_credit_policy import load_student_credit_policy

USER_CREDITS_TABLE = "UserCredits-test"
CREDIT_LEDGER_TABLE = "CreditLedger-test"


def _policy(*, dry_run: bool = False, **overrides: Any) -> StudentCreditPolicy:
    values: dict[str, Any] = {
        "enforcement_enabled": True,
        "dry_run": dry_run,
        "credits_per_usd": Decimal("50"),
        "target_gross_margin": Decimal("0.40"),
        "rounding_mode": "CEIL",
        "doubt_authorization_credits": 5,
        "practice_min_authorization_credits": 5,
        "practice_authorization_credits_per_question": 1,
    }
    values.update(overrides)
    return StudentCreditPolicy(**values)


def _file_backed_policy(*, dry_run: bool) -> StudentCreditPolicy:
    return load_student_credit_policy(enforcement_enabled=True, dry_run=dry_run)


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


class FakeDynamoDBClient:
    """Atomic in-memory double for the two credit tables only."""

    def __init__(self, *, wallets: dict[str, int] | None = None) -> None:
        self.wallets = dict(wallets or {})
        self.ledger: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, int] = {}
        self.transaction_calls = 0
        self.get_item_calls = 0
        self.transactions: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        self.get_item_calls += 1
        if kwargs["TableName"] == USER_CREDITS_TABLE:
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
        return {"Item": item} if item is not None else {}

    def transact_write_items(self, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            self.transaction_calls += 1
            self.transactions.append(kwargs)
            for transact_item in kwargs["TransactItems"]:
                self._reject_unused_names(next(iter(transact_item.values())))
            first, wallet = kwargs["TransactItems"]
            update = wallet["Update"]
            user_id = update["Key"]["userId"]["S"]
            values = update["ExpressionAttributeValues"]
            required = int(values.get(":requiredCredits", {"N": "0"})["N"])
            wallet_failed = user_id not in self.wallets or self.wallets[user_id] < required
            if "Put" in first:
                item = first["Put"]["Item"]
                ledger_id = item["ledgerId"]["S"]
                ledger_failed = ledger_id in self.ledger
                if ledger_failed or wallet_failed:
                    self._cancel(ledger_failed, wallet_failed)
                self.ledger[ledger_id] = item
            else:
                ledger_update = first["Update"]
                ledger_id = ledger_update["Key"]["ledgerId"]["S"]
                item = self.ledger.get(ledger_id)
                attrs = ledger_update["ExpressionAttributeValues"]
                expected_state = attrs[":authorized"]["S"]
                expected_user = attrs[":userId"]["S"]
                expected_amount = int(attrs[":authorizedCredits"]["N"])
                ledger_failed = (
                    item is None
                    or item["userId"]["S"] != expected_user
                    or item.get("authorizationState", {}).get("S") != expected_state
                    or int(item["amount"]["N"]) != expected_amount
                )
                if ledger_failed or wallet_failed:
                    self._cancel(ledger_failed, wallet_failed)
                assert item is not None
                item["amount"] = attrs.get(":captured", attrs.get(":zero"))
                item["authorizationState"] = attrs.get(":settled", attrs.get(":released"))
                item["metadata"] = attrs[":metadata"]
                item["updatedAt"] = attrs[":updatedAt"]
            delta = int(values.get(":creditsDelta", {"N": "0"})["N"])
            self.wallets[user_id] += delta
            self.versions[user_id] = self.versions.get(user_id, 0) + 1
        return {}

    @staticmethod
    def _reject_unused_names(operation: dict[str, Any]) -> None:
        """DynamoDB refuses the whole request when a declared name is unused.

        The double previously accepted it, which is how a settlement that released
        nothing passed every test here and failed every real transaction.
        """
        expressions = " ".join(
            str(operation.get(key) or "")
            for key in ("UpdateExpression", "ConditionExpression")
        )
        for field in ("ExpressionAttributeNames", "ExpressionAttributeValues"):
            # Placeholders are matched whole, so ":one" never hides inside ":ones".
            unused = sorted(
                token
                for token in operation.get(field, {})
                if not re.search(rf"{re.escape(token)}(?![A-Za-z0-9_])", expressions)
            )
            if unused:
                raise ClientError(
                    {
                        "Error": {
                            "Code": "ValidationException",
                            "Message": (
                                f"Value provided in {field} unused in "
                                f"expressions: keys: {{{', '.join(unused)}}}"
                            ),
                        }
                    },
                    "TransactWriteItems",
                )

    @staticmethod
    def _cancel(ledger_failed: bool, wallet_failed: bool) -> None:
        reasons = [
            {"Code": "ConditionalCheckFailed" if ledger_failed else "None"},
            {"Code": "ConditionalCheckFailed" if wallet_failed else "None"},
        ]
        raise ClientError(
            {
                "Error": {"Code": "TransactionCanceledException"},
                "CancellationReasons": reasons,
            },
            "TransactWriteItems",
        )


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


def _summary(*, cost: str = "0.012", complete: bool = True) -> OperationBillingSummary:
    return OperationBillingSummary(
        operation_id="operation-credit",
        feature="doubt",
        operation_status="completed",
        total_input_tokens=100,
        total_output_tokens=100,
        llm_call_count=1,
        total_llm_cost_usd=Decimal(cost) if complete else None,
        infra_cost_usd=Decimal("0"),
        actual_usage_cost_usd=Decimal(cost) if complete else None,
        pricing_factor=Decimal("2"),
        usd_per_credit=Decimal("0.001"),
        calculated_credits=None,
        credit_debit_enabled=False,
        credits_debited=0,
        billing_config_version="billing-test-v1",
        cost_complete=complete,
        missing_cost_profiles=() if complete else ("provider:model",),
        missing_usage_call_count=0,
    )


def _settle(runtime: StudentCreditRuntime, *, reference: str, cost: str = "0.012"):
    return runtime.settle(
        user_id="student-1",
        reference_id=reference,
        feature="doubt",
        operation_status="completed",
        summary=_summary(cost=cost),
    )


def test_calculation_preserves_measured_cost_margin_and_ceiling() -> None:
    assert calculate_credits(Decimal("0.012"), policy=_policy()).student_credits_to_debit == 1
    assert calculate_credits(Decimal("0.060"), policy=_policy()).student_credits_to_debit == 5


def test_file_backed_policy_preserves_dry_run_calculation_without_mutation() -> None:
    policy = _file_backed_policy(dry_run=True)
    expected = calculate_credits(
        Decimal("0.012"),
        policy=_policy(
            dry_run=True,
            credits_per_usd=Decimal("100"),
            target_gross_margin=Decimal("0.60"),
        ),
    )
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client, policy=policy)

    runtime.authorize_doubt(user_id="student-1", reference_id="file-policy-dry-run")
    settlement = _settle(runtime, reference="file-policy-dry-run")

    assert settlement.charge == expected
    assert settlement.credits_debited == 0
    assert client.wallets["student-1"] == 10
    assert client.ledger == {}


def test_file_backed_policy_preserves_real_settlement_calculation() -> None:
    policy = _file_backed_policy(dry_run=False)
    expected = calculate_credits(
        Decimal("0.012"),
        policy=_policy(credits_per_usd=Decimal("100"), target_gross_margin=Decimal("0.60")),
    )
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client, policy=policy)

    runtime.authorize_doubt(user_id="student-1", reference_id="file-policy-settlement")
    settlement = _settle(runtime, reference="file-policy-settlement")

    assert settlement.charge == expected
    assert settlement.charge.student_credits_to_debit == 3  # CEIL(0.012*100/0.40)
    assert settlement.credits_debited == 3
    assert client.wallets["student-1"] == 7


def test_file_backed_policy_replay_does_not_debit_twice() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client, policy=_file_backed_policy(dry_run=False))
    reference = "file-policy-settlement-replay"

    runtime.authorize_doubt(user_id="student-1", reference_id=reference)
    first = _settle(runtime, reference=reference)
    second = _settle(runtime, reference=reference, cost="0.100")

    assert first.credits_debited == 3
    assert second.status == "already_settled"
    assert second.credits_debited == 3
    assert client.wallets["student-1"] == 7


@pytest.mark.parametrize("wallet", [{}, {"student-1": 0}])
def test_missing_or_zero_wallet_is_rejected_before_authorization(wallet: dict[str, int]) -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets=wallet))

    with pytest.raises(InsufficientStudentCreditsError):
        runtime.ensure_can_start("student-1")


@pytest.mark.parametrize("balance", [1, 4])
def test_wallet_below_the_configured_minimum_is_rejected_before_any_work(
    balance: int,
) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": balance})
    runtime = _runtime(client)

    with pytest.raises(InsufficientStudentCreditsError) as refused:
        runtime.ensure_can_start("student-1")

    assert refused.value.reason_code == "INSUFFICIENT_CREDITS"
    assert client.transaction_calls == 0
    assert client.wallets["student-1"] == balance


@pytest.mark.parametrize("balance", [5, 6, 500])
def test_wallet_at_or_above_the_configured_minimum_passes_preflight(balance: int) -> None:
    runtime = _runtime(FakeDynamoDBClient(wallets={"student-1": balance}))

    assert runtime.minimum_admission_credits == 5
    assert runtime.ensure_can_start("student-1") == StudentWalletBalance(
        user_id="student-1", exists=True, credits_balance=balance
    )


def test_admission_minimum_follows_the_lowest_configured_authorization() -> None:
    runtime = _runtime(
        FakeDynamoDBClient(wallets={"student-1": 3}),
        policy=_policy(doubt_authorization_credits=7, practice_min_authorization_credits=3),
    )

    assert runtime.minimum_admission_credits == 3
    assert runtime.ensure_can_start("student-1") is not None


def test_dry_run_observes_a_low_wallet_without_refusing_it() -> None:
    runtime = _runtime(
        FakeDynamoDBClient(wallets={"student-1": 1}), policy=_policy(dry_run=True)
    )

    balance = runtime.ensure_can_start("student-1")

    assert balance is not None and not runtime.admits(balance)


def test_doubt_authorization_is_atomic_and_persists_authorized_state() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    reference = doubt_reference_id(user_id="student-1", turn_id="turn-1")

    assert runtime.authorize_doubt(user_id="student-1", reference_id=reference) == 5

    assert client.wallets["student-1"] == 5
    assert client.ledger[reference]["authorizationState"]["S"] == "AUTHORIZED"
    assert client.ledger[reference]["amount"]["N"] == "5"
    put, update = client.transactions[0]["TransactItems"]
    assert "attribute_not_exists" in put["Put"]["ConditionExpression"]
    assert "#creditsBalance >= :requiredCredits" in update["Update"]["ConditionExpression"]


def test_duplicate_active_authorization_cannot_start_paid_work() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    reference = "doubt-idempotent"

    runtime.authorize_doubt(user_id="student-1", reference_id=reference)
    with pytest.raises(StudentCreditLedgerConflictError):
        runtime.authorize_doubt(user_id="student-1", reference_id=reference)

    assert client.wallets["student-1"] == 5
    assert len(client.ledger) == 1


@pytest.mark.parametrize("terminal", ["settled", "released"])
def test_terminal_authorization_reference_cannot_start_paid_work(terminal: str) -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    reference = f"terminal-{terminal}"
    runtime.authorize_doubt(user_id="student-1", reference_id=reference)
    if terminal == "settled":
        _settle(runtime, reference=reference)
        expected_balance = 9
    else:
        runtime.release(
            user_id="student-1",
            reference_id=reference,
            feature="doubt",
            reason="failed",
        )
        expected_balance = 10

    with pytest.raises(StudentCreditLedgerConflictError):
        runtime.authorize_doubt(user_id="student-1", reference_id=reference)

    assert client.wallets["student-1"] == expected_balance


def test_practice_authorization_uses_effective_count_not_requested_count() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 100})
    runtime = _runtime(client)
    reference = practice_reference_id(user_id="student-1", test_id="practice-100-to-50")

    assert runtime.authorize_practice(
        user_id="student-1",
        reference_id=reference,
        feature="mock_test",
        effective_count=50,
    ) == 50
    assert client.wallets["student-1"] == 50


def test_low_positive_wallet_fails_authorization_without_a_hold() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 4})
    runtime = _runtime(client)

    with pytest.raises(InsufficientStudentCreditsError):
        runtime.authorize_doubt(user_id="student-1", reference_id="low-wallet")

    assert client.wallets["student-1"] == 4
    assert client.ledger == {}


def test_concurrent_practice_authorizations_never_overdraw_the_wallet() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 50})
    runtime = _runtime(client)

    def authorize(reference: str) -> str:
        try:
            runtime.authorize_practice(
                user_id="student-1",
                reference_id=reference,
                feature="mock_test",
                effective_count=30,
            )
        except InsufficientStudentCreditsError:
            return "insufficient"
        return "authorized"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(authorize, ("practice-a", "practice-b")))

    assert sorted(outcomes) == ["authorized", "insufficient"]
    assert client.wallets["student-1"] == 20
    assert client.wallets["student-1"] >= 0


def test_settlement_refunds_unused_authorization_and_keeps_actual_ledger_amount() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    reference = "settle-refund"
    runtime.authorize_doubt(user_id="student-1", reference_id=reference)

    settlement = _settle(runtime, reference=reference, cost="0.012")

    assert settlement.status == "settled"
    assert settlement.credits_debited == 1
    assert client.wallets["student-1"] == 9
    ledger = client.ledger[reference]
    assert ledger["authorizationState"]["S"] == "SETTLED"
    assert ledger["amount"]["N"] == "1"
    assert json.loads(ledger["metadata"]["S"])["releasedCredits"] == 4


def test_settlement_at_authorization_keeps_the_full_authorized_debit() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    runtime.authorize_doubt(user_id="student-1", reference_id="settle-exact")

    assert _settle(runtime, reference="settle-exact", cost="0.060").credits_debited == 5
    assert client.wallets["student-1"] == 5


def test_overage_is_capped_without_failing_the_completed_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        "services.student_credits.runtime.log_event",
        lambda name, **_kwargs: events.append(name),
    )
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    runtime.authorize_doubt(user_id="student-1", reference_id="settle-overage")

    settlement = _settle(runtime, reference="settle-overage", cost="0.100")

    assert settlement.status == "settled"
    assert settlement.credits_debited == 5
    assert client.wallets["student-1"] == 5
    assert "STUDENT_CREDIT_AUTHORIZATION_EXCEEDED" in events


def test_duplicate_settlement_does_not_debit_or_refund_twice() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    runtime.authorize_doubt(user_id="student-1", reference_id="settle-replay")

    first = _settle(runtime, reference="settle-replay")
    second = _settle(runtime, reference="settle-replay", cost="0.100")

    assert first.status == "settled"
    assert second.status == "already_settled"
    assert second.credits_debited == 1
    assert client.wallets["student-1"] == 9


def test_failed_operation_releases_the_full_authorization_once() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    runtime.authorize_doubt(user_id="student-1", reference_id="release-on-failure")

    runtime.release(
        user_id="student-1",
        reference_id="release-on-failure",
        feature="doubt",
        reason="provider_failed",
    )
    runtime.release(
        user_id="student-1",
        reference_id="release-on-failure",
        feature="doubt",
        reason="duplicate_failure",
    )

    assert client.wallets["student-1"] == 10
    assert client.ledger["release-on-failure"]["authorizationState"]["S"] == "RELEASED"
    assert client.ledger["release-on-failure"]["amount"]["N"] == "0"


def test_incomplete_pricing_releases_the_authorization_before_raising() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)
    runtime.authorize_doubt(user_id="student-1", reference_id="incomplete")

    with pytest.raises(StudentCreditPricingIncompleteError):
        runtime.settle(
            user_id="student-1",
            reference_id="incomplete",
            feature="doubt",
            operation_status="completed",
            summary=_summary(complete=False),
        )

    assert client.wallets["student-1"] == 10
    assert client.ledger["incomplete"]["authorizationState"]["S"] == "RELEASED"


def test_settlement_without_authorization_has_no_direct_debit_fallback() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client)

    with pytest.raises(StudentCreditRepositoryError):
        _settle(runtime, reference="no-hold")

    assert client.wallets["student-1"] == 10
    assert client.ledger == {}


def test_dry_run_logs_the_authorization_and_never_mutates_wallet_or_ledger() -> None:
    client = FakeDynamoDBClient(wallets={"student-1": 10})
    runtime = _runtime(client, policy=_policy(dry_run=True))

    runtime.authorize_doubt(user_id="student-1", reference_id="dry-run")
    settlement = _settle(runtime, reference="dry-run")
    runtime.release(
        user_id="student-1",
        reference_id="dry-run",
        feature="doubt",
        reason="failed",
    )

    assert settlement.status == "skipped_dry_run"
    assert client.wallets["student-1"] == 10
    assert client.ledger == {}
