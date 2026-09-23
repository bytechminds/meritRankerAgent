"""Authoritative DynamoDB access for the student wallet and credit ledger."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from schemas.student_credits import StudentWalletBalance
from services.student_credits.errors import (
    InsufficientStudentCreditsError,
    StudentCreditLedgerConflictError,
    StudentCreditRepositoryError,
)

LEDGER_DIRECTION_DEBIT = "DEBIT"
LEDGER_SOURCE_FEATURE = "FEATURE"
_LEDGER_TYPENAME = "CreditLedger"
_AUTHORIZED = "AUTHORIZED"
_SETTLED = "SETTLED"
_RELEASED = "RELEASED"
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


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    raw = _text(item.get("metadata"))
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _authorization_credits(item: dict[str, Any]) -> int:
    configured = _metadata(item).get("authorizationCredits")
    if isinstance(configured, int) and configured >= 0:
        return configured
    return _number(item.get("amount")) or 0


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
        """Return a settled debit, never a temporary authorization."""
        item = self._read_ledger(reference_id)
        if not item:
            return None
        state = _text(item.get("authorizationState"))
        if state not in {None, _SETTLED}:
            return None
        return _number(item.get("amount"))

    def authorize(
        self,
        *,
        reference_id: str,
        user_id: str,
        credits: int,
        feature_key: str,
        metadata: dict[str, Any],
    ) -> tuple[str, int]:
        """Atomically hold the configured maximum student charge exactly once."""
        if credits <= 0:
            raise ValueError("authorize requires a positive credit amount")
        now = _now()
        ledger_metadata = {**metadata, "authorizationCredits": credits}
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
                                "authorizationState": {"S": _AUTHORIZED},
                                "metadata": {"S": json.dumps(ledger_metadata, sort_keys=True)},
                                "createdAt": {"S": now},
                                "__typename": {"S": _LEDGER_TYPENAME},
                            },
                            "ConditionExpression": "attribute_not_exists(#ledgerId)",
                            "ExpressionAttributeNames": {"#ledgerId": "ledgerId"},
                        }
                    },
                    self._wallet_update(
                        user_id=user_id,
                        reference_id=reference_id,
                        now=now,
                        credits_delta=-credits,
                        require_balance=credits,
                    ),
                ]
            )
        except ClientError as exc:
            return self._recover_authorization(
                exc,
                reference_id=reference_id,
                user_id=user_id,
                feature_key=feature_key,
                credits=credits,
            )
        except BotoCoreError as exc:
            raise StudentCreditRepositoryError("Student credit authorization failed.") from exc
        return "authorized", credits

    def settle_authorization(
        self,
        *,
        reference_id: str,
        user_id: str,
        actual_credits: int,
        metadata: dict[str, Any],
    ) -> tuple[str, int, int, int]:
        """Capture a measured debit and atomically return an unused authorization."""
        if actual_credits < 0:
            raise ValueError("actual_credits cannot be negative")
        item = self._require_ledger(reference_id, user_id)
        state = _text(item.get("authorizationState"))
        authorized = _authorization_credits(item)
        if state is None or state == _SETTLED:
            return "already_settled", _number(item.get("amount")) or 0, 0, authorized
        if state == _RELEASED:
            return "already_released", 0, 0, authorized
        if state != _AUTHORIZED:
            raise StudentCreditRepositoryError("Student credit ledger state is invalid.")

        captured = min(actual_credits, authorized)
        released = authorized - captured
        now = _now()
        settled_metadata = {
            **_metadata(item),
            **metadata,
            "authorizationCredits": authorized,
            "actualCalculatedCredits": actual_credits,
            "capturedCredits": captured,
            "releasedCredits": released,
        }
        try:
            self._client.transact_write_items(
                TransactItems=[
                    {
                        "Update": {
                            "TableName": self._credit_ledger_table,
                            "Key": {"ledgerId": {"S": reference_id}},
                            "UpdateExpression": (
                                "SET #amount = :captured, #authorizationState = :settled, "
                                "#metadata = :metadata, #updatedAt = :updatedAt"
                            ),
                            "ConditionExpression": (
                                "#userId = :userId AND #authorizationState = :authorized "
                                "AND #amount = :authorizedCredits"
                            ),
                            "ExpressionAttributeNames": {
                                "#amount": "amount",
                                "#authorizationState": "authorizationState",
                                "#metadata": "metadata",
                                "#updatedAt": "updatedAt",
                                "#userId": "userId",
                            },
                            "ExpressionAttributeValues": {
                                ":captured": {"N": str(captured)},
                                ":settled": {"S": _SETTLED},
                                ":metadata": {"S": json.dumps(settled_metadata, sort_keys=True)},
                                ":updatedAt": {"S": now},
                                ":userId": {"S": user_id},
                                ":authorized": {"S": _AUTHORIZED},
                                ":authorizedCredits": {"N": str(authorized)},
                            },
                        }
                    },
                    self._wallet_update(
                        user_id=user_id,
                        reference_id=reference_id,
                        now=now,
                        credits_delta=released,
                    ),
                ]
            )
        except ClientError as exc:
            return self._recover_terminal_transition(
                exc,
                reference_id=reference_id,
                user_id=user_id,
                authorized=authorized,
                action="settlement",
            )
        except BotoCoreError as exc:
            raise StudentCreditRepositoryError("Student credit settlement failed.") from exc
        return "settled", captured, released, authorized

    def release_authorization(
        self,
        *,
        reference_id: str,
        user_id: str,
        metadata: dict[str, Any],
    ) -> tuple[str, int, int]:
        """Return an abandoned authorization once, with no debit left behind."""
        item = self._read_ledger(reference_id)
        if item is None:
            return "skipped_no_authorization", 0, 0
        self._validate_ledger(item, user_id)
        state = _text(item.get("authorizationState"))
        authorized = _authorization_credits(item)
        if state == _SETTLED or state is None:
            return "already_settled", 0, authorized
        if state == _RELEASED:
            return "already_released", 0, authorized
        if state != _AUTHORIZED:
            raise StudentCreditRepositoryError("Student credit ledger state is invalid.")

        now = _now()
        released_metadata = {
            **_metadata(item),
            **metadata,
            "authorizationCredits": authorized,
            "capturedCredits": 0,
            "releasedCredits": authorized,
        }
        try:
            self._client.transact_write_items(
                TransactItems=[
                    {
                        "Update": {
                            "TableName": self._credit_ledger_table,
                            "Key": {"ledgerId": {"S": reference_id}},
                            "UpdateExpression": (
                                "SET #amount = :zero, #authorizationState = :released, "
                                "#metadata = :metadata, #updatedAt = :updatedAt"
                            ),
                            "ConditionExpression": (
                                "#userId = :userId AND #authorizationState = :authorized "
                                "AND #amount = :authorizedCredits"
                            ),
                            "ExpressionAttributeNames": {
                                "#amount": "amount",
                                "#authorizationState": "authorizationState",
                                "#metadata": "metadata",
                                "#updatedAt": "updatedAt",
                                "#userId": "userId",
                            },
                            "ExpressionAttributeValues": {
                                ":zero": {"N": "0"},
                                ":released": {"S": _RELEASED},
                                ":metadata": {"S": json.dumps(released_metadata, sort_keys=True)},
                                ":updatedAt": {"S": now},
                                ":userId": {"S": user_id},
                                ":authorized": {"S": _AUTHORIZED},
                                ":authorizedCredits": {"N": str(authorized)},
                            },
                        }
                    },
                    self._wallet_update(
                        user_id=user_id,
                        reference_id=reference_id,
                        now=now,
                        credits_delta=authorized,
                    ),
                ]
            )
        except ClientError as exc:
            status, _, _, _ = self._recover_terminal_transition(
                exc,
                reference_id=reference_id,
                user_id=user_id,
                authorized=authorized,
                action="release",
            )
            return status, 0, authorized
        except BotoCoreError as exc:
            raise StudentCreditRepositoryError("Student credit release failed.") from exc
        return "released", authorized, authorized

    def _read_ledger(self, reference_id: str) -> dict[str, Any] | None:
        try:
            response = self._client.get_item(
                TableName=self._credit_ledger_table,
                Key={"ledgerId": {"S": reference_id}},
                ConsistentRead=True,
            )
        except (BotoCoreError, ClientError) as exc:
            raise StudentCreditRepositoryError("Student credit ledger read failed.") from exc
        item = response.get("Item")
        return item if isinstance(item, dict) else None

    def _require_ledger(self, reference_id: str, user_id: str) -> dict[str, Any]:
        item = self._read_ledger(reference_id)
        if item is None:
            raise StudentCreditRepositoryError("Student credit authorization is missing.")
        self._validate_ledger(item, user_id)
        return item

    def _validate_ledger(self, item: dict[str, Any], user_id: str) -> None:
        if _text(item.get("userId")) != user_id:
            raise StudentCreditLedgerConflictError(
                "Student credit reference belongs to another user."
            )

    def _recover_authorization(
        self,
        exc: ClientError,
        *,
        reference_id: str,
        user_id: str,
        feature_key: str,
        credits: int,
    ) -> tuple[str, int]:
        if not self._transaction_cancelled(exc):
            raise StudentCreditRepositoryError("Student credit authorization failed.") from exc
        if self._transaction_item_failed(exc, _LEDGER_ITEM_INDEX):
            item = self._require_ledger(reference_id, user_id)
            if _text(item.get("featureKey")) != feature_key:
                raise StudentCreditLedgerConflictError(
                    "Student credit reference belongs to another operation."
                ) from exc
            state = _text(item.get("authorizationState"))
            if state == _AUTHORIZED:
                return "already_authorized", _authorization_credits(item)
            if state == _SETTLED or state is None:
                return "already_settled", _authorization_credits(item)
            if state == _RELEASED:
                return "already_released", _authorization_credits(item)
            raise StudentCreditRepositoryError("Student credit ledger state is invalid.") from exc
        if self._transaction_item_failed(exc, _WALLET_ITEM_INDEX):
            raise InsufficientStudentCreditsError(
                f"Student wallet cannot cover {credits} credits."
            ) from exc
        item = self._read_ledger(reference_id)
        if item is not None:
            self._validate_ledger(item, user_id)
            return "already_authorized", _authorization_credits(item)
        raise StudentCreditRepositoryError("Student credit authorization failed.") from exc

    def _recover_terminal_transition(
        self,
        exc: ClientError,
        *,
        reference_id: str,
        user_id: str,
        authorized: int,
        action: str,
    ) -> tuple[str, int, int, int]:
        if not self._transaction_cancelled(exc):
            raise StudentCreditRepositoryError(f"Student credit {action} failed.") from exc
        item = self._require_ledger(reference_id, user_id)
        state = _text(item.get("authorizationState"))
        if state is None or state == _SETTLED:
            return "already_settled", _number(item.get("amount")) or 0, 0, authorized
        if state == _RELEASED:
            return "already_released", 0, 0, authorized
        raise StudentCreditRepositoryError(f"Student credit {action} failed.") from exc

    def _wallet_update(
        self,
        *,
        user_id: str,
        reference_id: str,
        now: str,
        credits_delta: int,
        require_balance: int | None = None,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            ":referenceId": {"S": reference_id},
            ":updatedAt": {"S": now},
            ":one": {"N": "1"},
        }
        update = "SET #lastReferenceId = :referenceId, #updatedAt = :updatedAt ADD #version :one"
        if credits_delta:
            values[":creditsDelta"] = {"N": str(credits_delta)}
            update += ", #creditsBalance :creditsDelta"
        condition = "attribute_exists(#userId)"
        if require_balance is not None:
            values[":requiredCredits"] = {"N": str(require_balance)}
            condition += " AND #creditsBalance >= :requiredCredits"
        names = {
            "#userId": "userId",
            "#lastReferenceId": "lastReferenceId",
            "#updatedAt": "updatedAt",
            "#version": "version",
        }
        # DynamoDB rejects any declared name the expressions do not use. A settlement
        # that captures its whole authorization releases nothing, so the balance is
        # untouched — declaring it anyway failed every such settlement validation.
        if credits_delta or require_balance is not None:
            names["#creditsBalance"] = "creditsBalance"
        return {
            "Update": {
                "TableName": self._user_credits_table,
                "Key": {"userId": {"S": user_id}},
                "UpdateExpression": update,
                "ConditionExpression": condition,
                "ExpressionAttributeNames": names,
                "ExpressionAttributeValues": values,
            }
        }

    @staticmethod
    def _transaction_cancelled(exc: ClientError) -> bool:
        error = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
        return error.get("Code") == "TransactionCanceledException"

    @staticmethod
    def _transaction_item_failed(exc: ClientError, index: int) -> bool:
        reasons = exc.response.get("CancellationReasons") or []
        if len(reasons) <= index:
            return False
        reason = reasons[index]
        return isinstance(reason, dict) and reason.get("Code") == _CONDITIONAL_CHECK_FAILED
