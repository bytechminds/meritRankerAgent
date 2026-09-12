"""Authoritative DynamoDB access for the student wallet and its debit ledger.

Two operations only, mirroring the reviewed backend debit precedent:

* one ``GetItem`` on ``UserCredits`` for admission;
* one ``TransactWriteItems`` that writes the deterministic ledger row and
  decrements the wallet, or does neither.

The runtime never creates a wallet row and never writes any other attribute.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from schemas.student_credits import StudentWalletBalance
from services.student_credits.errors import (
    InsufficientStudentCreditsError,
    StudentCreditLedgerConflictError,
    StudentCreditRepositoryError,
)

logger = logging.getLogger(__name__)

LEDGER_DIRECTION_DEBIT = "DEBIT"
LEDGER_SOURCE_FEATURE = "FEATURE"
_LEDGER_TYPENAME = "CreditLedger"
_CONDITIONAL_CHECK_FAILED = "ConditionalCheckFailed"
_LEDGER_ITEM_INDEX = 0
_WALLET_ITEM_INDEX = 1


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _number(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    raw = value.get("N")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    return value.get("S") if isinstance(value, dict) else None


class StudentCreditRepository:
    """Least-privilege wallet and ledger access for one deployment."""

    def __init__(
        self,
        *,
        user_credits_table: str,
        credit_ledger_table: str,
        client: Any,
    ) -> None:
        self._user_credits_table = user_credits_table
        self._credit_ledger_table = credit_ledger_table
        self._client = client

    def get_balance(self, user_id: str) -> StudentWalletBalance:
        """Read the authoritative balance. A missing row is not an error."""
        try:
            response = self._client.get_item(
                TableName=self._user_credits_table,
                Key={"userId": {"S": user_id}},
                ProjectionExpression="#userId, #creditsBalance",
                ExpressionAttributeNames={
                    "#userId": "userId",
                    "#creditsBalance": "creditsBalance",
                },
                ConsistentRead=True,
            )
        except (BotoCoreError, ClientError) as exc:
            raise StudentCreditRepositoryError("Student wallet read failed.") from exc

        item = response.get("Item")
        if not item:
            return StudentWalletBalance(user_id=user_id, exists=False, credits_balance=0)
        return StudentWalletBalance(
            user_id=user_id,
            exists=True,
            credits_balance=_number(item.get("creditsBalance")) or 0,
        )

    def read_settlement(self, reference_id: str) -> int | None:
        """Return the debited amount of an existing settlement, if any."""
        try:
            response = self._client.get_item(
                TableName=self._credit_ledger_table,
                Key={"ledgerId": {"S": reference_id}},
                ConsistentRead=True,
            )
        except (BotoCoreError, ClientError) as exc:
            raise StudentCreditRepositoryError("Student credit ledger read failed.") from exc
        item = response.get("Item")
        if not item:
            return None
        return _number(item.get("amount"))

    def settle(
        self,
        *,
        reference_id: str,
        user_id: str,
        credits: int,
        feature_key: str,
        metadata: dict[str, Any],
    ) -> None:
        """Write the ledger row and decrement the wallet atomically.

        Raises:
            StudentCreditLedgerConflictError: The reference is already settled.
            InsufficientStudentCreditsError: Wallet missing or balance too low.
            StudentCreditRepositoryError: The transaction could not be attempted.
        """
        if credits <= 0:
            raise ValueError("settle requires a positive credit amount")
        now = _now()
        try:
            self._client.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": self._credit_ledger_table,
                            "Item": {
                                "ledgerId": {"S": reference_id},
                                "userId": {"S": user_id},
                                "direction": {"S": LEDGER_DIRECTION_DEBIT},
                                "amount": {"N": str(credits)},
                                "source": {"S": LEDGER_SOURCE_FEATURE},
                                "featureKey": {"S": feature_key},
                                "referenceId": {"S": reference_id},
                                "metadata": {"S": json.dumps(metadata, sort_keys=True)},
                                "createdAt": {"S": now},
                                "__typename": {"S": _LEDGER_TYPENAME},
                            },
                            "ConditionExpression": "attribute_not_exists(#ledgerId)",
                            "ExpressionAttributeNames": {"#ledgerId": "ledgerId"},
                        }
                    },
                    {
                        "Update": {
                            "TableName": self._user_credits_table,
                            "Key": {"userId": {"S": user_id}},
                            "UpdateExpression": (
                                "SET #lastReferenceId = :referenceId, #updatedAt = :updatedAt "
                                "ADD #creditsBalance :debit, #version :one"
                            ),
                            "ConditionExpression": (
                                "attribute_exists(#userId) AND #creditsBalance >= :credits"
                            ),
                            "ExpressionAttributeNames": {
                                "#userId": "userId",
                                "#creditsBalance": "creditsBalance",
                                "#lastReferenceId": "lastReferenceId",
                                "#updatedAt": "updatedAt",
                                "#version": "version",
                            },
                            "ExpressionAttributeValues": {
                                ":referenceId": {"S": reference_id},
                                ":updatedAt": {"S": now},
                                ":debit": {"N": str(-credits)},
                                ":credits": {"N": str(credits)},
                                ":one": {"N": "1"},
                            },
                        }
                    },
                ]
            )
        except ClientError as exc:
            self._raise_for_cancellation(exc, reference_id=reference_id, credits=credits)
            raise StudentCreditRepositoryError("Student credit settlement failed.") from exc
        except BotoCoreError as exc:
            raise StudentCreditRepositoryError("Student credit settlement failed.") from exc

    def _raise_for_cancellation(
        self,
        exc: ClientError,
        *,
        reference_id: str,
        credits: int,
    ) -> None:
        error = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
        if error.get("Code") != "TransactionCanceledException":
            return
        reasons = exc.response.get("CancellationReasons") or []
        codes = [
            reason.get("Code") if isinstance(reason, dict) else None for reason in reasons
        ]

        def failed(index: int) -> bool:
            return len(codes) > index and codes[index] == _CONDITIONAL_CHECK_FAILED

        # The ledger condition is checked first so a duplicate settlement is
        # reported as already-settled even when the balance has since dropped.
        if failed(_LEDGER_ITEM_INDEX):
            raise StudentCreditLedgerConflictError(
                f"Student credit reference already settled: {reference_id}"
            ) from exc
        if failed(_WALLET_ITEM_INDEX):
            raise InsufficientStudentCreditsError(
                f"Student wallet cannot cover {credits} credits."
            ) from exc
