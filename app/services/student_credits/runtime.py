"""The single student-credit boundary used by the Agent runtime."""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal

from observability.events import log_event
from observability.summary import update_request_summary
from schemas.billing import OperationBillingSummary
from schemas.student_credits import (
    StudentCreditCharge,
    StudentCreditPolicy,
    StudentCreditSettlement,
    StudentWalletBalance,
)
from services.llm.billing import snapshot_operation_billing
from services.student_credits.calculator import calculate_credits
from services.student_credits.errors import (
    InsufficientStudentCreditsError,
    StudentCreditLedgerConflictError,
    StudentCreditPricingIncompleteError,
    StudentCreditRepositoryError,
)
from services.student_credits.repository import StudentCreditRepository

logger = logging.getLogger(__name__)

_COMPONENT = "student_credits"


def doubt_reference_id(*, user_id: str, turn_id: str) -> str:
    """Deterministic credit reference for one billable Doubt Solver turn."""
    return f"student-credit:doubt:{user_id}:{turn_id}"


def practice_reference_id(*, user_id: str, test_id: str) -> str:
    """Deterministic credit reference for one billable Practice assessment."""
    return f"student-credit:practice:{user_id}:{test_id}"


def _user_ref(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def _reference_ref(reference_id: str) -> str:
    """Loggable form of a credit reference: the embedded user id is hashed.

    The reference itself stays the idempotency key; only its logged copy changes,
    keeping the feature and operation segments correlatable.
    """
    parts = reference_id.split(":")
    if len(parts) >= 4 and parts[0] == "student-credit":
        parts[2] = _user_ref(parts[2])
        return ":".join(parts)
    return _user_ref(reference_id)


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value.normalize(), "f")


def _store_failure_diagnostics(exc: BaseException) -> dict[str, object]:
    """Safe shape of a failed wallet/ledger write: codes only, never values.

    The repository chains the provider exception, so the cause is read from it
    rather than threaded through a second error contract.
    """
    cause = exc.__cause__
    diagnostics: dict[str, object] = {
        "exception_type": type(cause).__name__ if cause is not None else type(exc).__name__,
    }
    response = getattr(cause, "response", None)
    if isinstance(response, dict):
        error = response.get("Error") or {}
        code = error.get("Code") if isinstance(error, dict) else None
        diagnostics["aws_error_code"] = code
        diagnostics["transaction_cancelled"] = code == "TransactionCanceledException"
        reasons = response.get("CancellationReasons") or []
        diagnostics["cancellation_reason_codes"] = ",".join(
            str(reason.get("Code") or "None") if isinstance(reason, dict) else "None"
            for reason in reasons
        )
    return diagnostics


class StudentCreditRuntime:
    """Config-driven preflight, authorization, settlement, and release."""

    def __init__(
        self,
        *,
        policy: StudentCreditPolicy,
        repository: StudentCreditRepository,
    ) -> None:
        self._policy = policy
        self._repository = repository

    @property
    def policy(self) -> StudentCreditPolicy:
        return self._policy

    @property
    def minimum_admission_credits(self) -> int:
        """Smallest balance any chargeable feature can authorize against.

        Admission runs before classification, when the feature is not yet known,
        so it uses the lowest configured authorization. A wallet below it cannot
        pass any authorization, and refusing it here spares the classifier call.
        """
        return min(
            self._policy.doubt_authorization_credits,
            self._policy.practice_min_authorization_credits,
        )

    def admits(self, balance: StudentWalletBalance) -> bool:
        return (
            balance.can_start_chargeable_work
            and balance.credits_balance >= self.minimum_admission_credits
        )

    def get_balance(self, user_id: str) -> StudentWalletBalance:
        """Read the authoritative wallet for the admission decision."""
        started = time.monotonic()
        balance = self._repository.get_balance(user_id)
        self._balance_read_ms = max(int((time.monotonic() - started) * 1000), 0)
        self._log_admission(
            "allowed" if self.admits(balance) else "blocked",
            user_id=user_id,
            balance=balance,
        )
        return balance

    def ensure_can_start(self, user_id: str) -> StudentWalletBalance | None:
        """Fail early for a wallet below the configured minimum authorization."""
        try:
            balance = self.get_balance(user_id)
        except StudentCreditRepositoryError:
            self._log_admission("unavailable", user_id=user_id, balance=None)
            if self._policy.dry_run:
                return None
            raise
        if self._policy.dry_run or self.admits(balance):
            return balance
        raise InsufficientStudentCreditsError(
            "Student wallet is below the minimum credits required to start.",
            required_credits=self.minimum_admission_credits,
            available_credits=max(balance.credits_balance, 0),
        )

    def authorize_doubt(self, *, user_id: str, reference_id: str) -> int:
        return self._authorize(
            user_id=user_id,
            reference_id=reference_id,
            feature="doubt",
            credits=self._policy.doubt_authorization_credits,
        )

    def authorize_practice(
        self,
        *,
        user_id: str,
        reference_id: str,
        feature: str,
        effective_count: int,
    ) -> int:
        if effective_count <= 0:
            raise ValueError("effective_count must be positive")
        credits = max(
            self._policy.practice_min_authorization_credits,
            effective_count * self._policy.practice_authorization_credits_per_question,
        )
        return self._authorize(
            user_id=user_id,
            reference_id=reference_id,
            feature=feature,
            credits=credits,
        )

    def _authorize(
        self,
        *,
        user_id: str,
        reference_id: str,
        feature: str,
        credits: int,
    ) -> int:
        started = time.monotonic()
        if self._policy.dry_run:
            self._log_authorization(
                "would_authorize",
                user_id=user_id,
                reference_id=reference_id,
                feature=feature,
                credits=credits,
                latency_ms=max(int((time.monotonic() - started) * 1000), 0),
            )
            return credits
        try:
            status, authorized = self._repository.authorize(
                reference_id=reference_id,
                user_id=user_id,
                credits=credits,
                feature_key=feature,
                metadata={"feature": feature},
            )
        except InsufficientStudentCreditsError as exc:
            self._log_authorization(
                "insufficient_credits",
                user_id=user_id,
                reference_id=reference_id,
                feature=feature,
                credits=credits,
                latency_ms=max(int((time.monotonic() - started) * 1000), 0),
            )
            # The hold is the amount the student must have; the wallet balance is
            # not re-read here, so it is reported only when already known.
            raise InsufficientStudentCreditsError(
                str(exc), required_credits=credits
            ) from exc
        except StudentCreditRepositoryError:
            self._log_authorization(
                "unavailable",
                user_id=user_id,
                reference_id=reference_id,
                feature=feature,
                credits=credits,
                latency_ms=max(int((time.monotonic() - started) * 1000), 0),
            )
            raise
        self._log_authorization(
            status,
            user_id=user_id,
            reference_id=reference_id,
            feature=feature,
            credits=authorized,
            latency_ms=max(int((time.monotonic() - started) * 1000), 0),
        )
        if status != "authorized":
            raise StudentCreditLedgerConflictError(
                "Student credit reference is already being processed."
            )
        return authorized

    def calculate_charge(self, actual_llm_cost_usd: Decimal) -> StudentCreditCharge:
        """Convert an actual measured provider cost into whole credits."""
        return calculate_credits(actual_llm_cost_usd, policy=self._policy)

    def settle(
        self,
        *,
        user_id: str,
        reference_id: str,
        feature: str,
        operation_status: str,
        summary: OperationBillingSummary | None = None,
    ) -> StudentCreditSettlement:
        """Settle a previously authorized operation from measured provider cost."""
        if summary is None:
            summary = snapshot_operation_billing(operation_status=operation_status)
        if summary is None or summary.llm_call_count == 0:
            charge = self.calculate_charge(Decimal("0"))
        elif not summary.cost_complete or summary.total_llm_cost_usd is None:
            self.release(
                user_id=user_id,
                reference_id=reference_id,
                feature=feature,
                reason="pricing_incomplete",
            )
            self._log_pricing_incomplete(summary, user_id=user_id, reference_id=reference_id)
            raise StudentCreditPricingIncompleteError(
                "Operation contains provider calls with no reviewed price."
            )
        else:
            charge = self.calculate_charge(summary.total_llm_cost_usd)
        if self._policy.dry_run:
            self._log_settlement(
                "skipped_dry_run",
                user_id=user_id,
                reference_id=reference_id,
                summary=summary,
                charge=charge,
                authorization_credits=None,
                captured_credits=0,
                released_credits=0,
            )
            return StudentCreditSettlement(
                status="skipped_dry_run",
                reference_id=reference_id,
                credits_debited=0,
                charge=charge,
            )

        started = time.monotonic()
        try:
            status, captured, released, authorized = self._repository.settle_authorization(
                reference_id=reference_id,
                user_id=user_id,
                actual_credits=charge.student_credits_to_debit,
                metadata={
                    "operationId": summary.operation_id,
                    "feature": summary.feature,
                    "llmCallCount": summary.llm_call_count,
                    "actualLlmCostUsd": _decimal_text(summary.total_llm_cost_usd),
                    "creditsPerUsd": _decimal_text(charge.credits_per_usd),
                    "targetGrossMargin": _decimal_text(charge.target_gross_margin),
                    "billingConfigVersion": summary.billing_config_version,
                },
            )
        except StudentCreditRepositoryError as exc:
            self._log_settlement(
                "unavailable",
                user_id=user_id,
                reference_id=reference_id,
                summary=summary,
                charge=charge,
                authorization_credits=None,
                captured_credits=0,
                released_credits=0,
                diagnostics=_store_failure_diagnostics(exc),
            )
            raise
        settlement_ms = max(int((time.monotonic() - started) * 1000), 0)
        exceeded = charge.student_credits_to_debit > authorized
        self._log_settlement(
            status,
            user_id=user_id,
            reference_id=reference_id,
            summary=summary,
            charge=charge,
            authorization_credits=authorized,
            captured_credits=captured,
            released_credits=released,
            settlement_ms=settlement_ms,
        )
        if exceeded:
            log_event(
                "STUDENT_CREDIT_AUTHORIZATION_EXCEEDED",
                component=_COMPONENT,
                stage="settlement",
                status="capped",
                error_code="STUDENT_CREDIT_AUTHORIZATION_EXCEEDED",
                details={
                    "reference_id": _reference_ref(reference_id),
                    "actual_calculated_credits": charge.student_credits_to_debit,
                    "authorization_credits": authorized,
                    "captured_credits": captured,
                },
                level=logging.ERROR,
            )
        return StudentCreditSettlement(
            status=status,  # type: ignore[arg-type]
            reference_id=reference_id,
            credits_debited=captured,
            charge=charge,
        )

    def release(
        self,
        *,
        user_id: str,
        reference_id: str,
        feature: str,
        reason: str,
    ) -> None:
        """Release a failed operation's authorization once; dry run is read-free."""
        if self._policy.dry_run:
            self._log_release(
                "would_release",
                user_id=user_id,
                reference_id=reference_id,
                feature=feature,
                released_credits=0,
                authorization_credits=None,
                reason=reason,
            )
            return
        started = time.monotonic()
        try:
            status, released, authorized = self._repository.release_authorization(
                reference_id=reference_id,
                user_id=user_id,
                metadata={"feature": feature, "releaseReason": reason},
            )
        except StudentCreditRepositoryError:
            self._log_release(
                "unavailable",
                user_id=user_id,
                reference_id=reference_id,
                feature=feature,
                released_credits=0,
                authorization_credits=None,
                reason=reason,
            )
            raise
        self._log_release(
            status,
            user_id=user_id,
            reference_id=reference_id,
            feature=feature,
            released_credits=released,
            authorization_credits=authorized,
            reason=reason,
            settlement_ms=max(int((time.monotonic() - started) * 1000), 0),
        )

    def _log_admission(
        self,
        status: str,
        *,
        user_id: str,
        balance: StudentWalletBalance | None,
    ) -> None:
        log_event(
            "student_credit_admission_checked",
            component=_COMPONENT,
            stage="admission",
            status=status,
            details={
                "user_ref": _user_ref(user_id),
                "student_credit_admission_status": status,
                "wallet_exists": balance.exists if balance else None,
                "credits_balance": balance.credits_balance if balance else None,
                "dry_run": self._policy.dry_run,
                "student_credit_balance_read_ms": getattr(self, "_balance_read_ms", None),
            },
        )
        update_request_summary(
            credit_mode="dry_run" if self._policy.dry_run else "enforcing",
            credit_admission=status,
            credit_balance=balance.credits_balance if balance else None,
            credit_preflight=status,
        )

    def _log_authorization(
        self,
        status: str,
        *,
        user_id: str,
        reference_id: str,
        feature: str,
        credits: int,
        latency_ms: int,
    ) -> None:
        log_event(
            "student_credit_authorization",
            component=_COMPONENT,
            stage="authorization",
            status=status,
            duration_ms=latency_ms,
            details={
                "user_ref": _user_ref(user_id),
                "reference_id": _reference_ref(reference_id),
                "feature": feature,
                "authorization_status": status,
                "authorization_credits": credits,
                "dry_run": self._policy.dry_run,
                "authorization_latency_ms": latency_ms,
            },
        )
        update_request_summary(
            credit_mode="dry_run" if self._policy.dry_run else "enforcing",
            credit_authorization_status=status,
            credit_authorization_credits=credits,
            credit_idempotent_replay=status == "already_authorized",
        )

    def _log_settlement(
        self,
        status: str,
        *,
        user_id: str,
        reference_id: str,
        summary: OperationBillingSummary | None,
        charge: StudentCreditCharge | None,
        authorization_credits: int | None,
        captured_credits: int,
        released_credits: int,
        settlement_ms: int | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        exceeded = bool(
            charge is not None
            and authorization_credits is not None
            and charge.student_credits_to_debit > authorization_credits
        )
        log_event(
            "student_credit_settlement",
            component=_COMPONENT,
            stage="settlement",
            status=status,
            duration_ms=settlement_ms,
            details={
                "user_ref": _user_ref(user_id),
                "reference_id": _reference_ref(reference_id),
                "operation_id": summary.operation_id if summary else None,
                "feature": summary.feature if summary else None,
                "llm_call_count": summary.llm_call_count if summary else 0,
                "actual_llm_cost_usd": (
                    _decimal_text(summary.total_llm_cost_usd) if summary else None
                ),
                "actual_calculated_credits": (
                    charge.student_credits_to_debit if charge else None
                ),
                "authorization_credits": authorization_credits,
                "captured_credits": captured_credits,
                "released_credits": released_credits,
                "settlement_status": status,
                "authorization_exceeded": exceeded,
                "idempotent_replay": status in {"already_settled", "already_released"},
                "settlement_latency_ms": settlement_ms,
                "dry_run": self._policy.dry_run,
                **(diagnostics or {}),
            },
        )
        update_request_summary(
            credit_mode="dry_run" if self._policy.dry_run else "enforcing",
            credit_calculated=charge.student_credits_to_debit if charge else None,
            credit_settlement=status,
            credits_debited=captured_credits,
            credit_authorization_credits=authorization_credits,
            credit_captured_credits=captured_credits,
            credit_released_credits=released_credits,
            credit_authorization_exceeded=exceeded,
            credit_idempotent_replay=status in {"already_settled", "already_released"},
        )

    def _log_release(
        self,
        status: str,
        *,
        user_id: str,
        reference_id: str,
        feature: str,
        released_credits: int,
        authorization_credits: int | None,
        reason: str,
        settlement_ms: int | None = None,
    ) -> None:
        log_event(
            "student_credit_settlement",
            component=_COMPONENT,
            stage="release",
            status=status,
            duration_ms=settlement_ms,
            details={
                "user_ref": _user_ref(user_id),
                "reference_id": _reference_ref(reference_id),
                "feature": feature,
                "authorization_credits": authorization_credits,
                "captured_credits": 0,
                "released_credits": released_credits,
                "settlement_status": status,
                "release_reason": reason,
                "idempotent_replay": status in {"already_settled", "already_released"},
                "settlement_latency_ms": settlement_ms,
                "dry_run": self._policy.dry_run,
            },
        )
        update_request_summary(
            credit_mode="dry_run" if self._policy.dry_run else "enforcing",
            credit_settlement=status,
            credit_authorization_credits=authorization_credits,
            credit_captured_credits=0,
            credit_released_credits=released_credits,
            credit_idempotent_replay=status in {"already_settled", "already_released"},
        )

    def _log_pricing_incomplete(
        self,
        summary: OperationBillingSummary,
        *,
        user_id: str,
        reference_id: str,
    ) -> None:
        log_event(
            "student_credit_pricing_incomplete",
            component=_COMPONENT,
            stage="settlement",
            status="blocked_pricing_incomplete",
            error_code="STUDENT_CREDIT_PRICING_INCOMPLETE",
            details={
                "user_ref": _user_ref(user_id),
                "reference_id": _reference_ref(reference_id),
                "operation_id": summary.operation_id,
                "feature": summary.feature,
                "llm_call_count": summary.llm_call_count,
                "missing_cost_profiles": ",".join(summary.missing_cost_profiles),
                "missing_usage_call_count": summary.missing_usage_call_count,
                "billing_config_version": summary.billing_config_version,
            },
            level=logging.ERROR,
        )


__all__ = [
    "InsufficientStudentCreditsError",
    "StudentCreditPricingIncompleteError",
    "StudentCreditRepositoryError",
    "StudentCreditRuntime",
    "doubt_reference_id",
    "practice_reference_id",
]
