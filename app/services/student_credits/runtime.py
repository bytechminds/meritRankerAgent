"""The single student credit boundary used by the Agent runtime.

One admission read before chargeable generation, one atomic settlement after an
accepted result.  Provider and model pricing stay owned by
``services.llm.billing``; this module only converts an already-measured
operation cost into whole credits and moves the authoritative wallet.
"""

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
    """Deterministic settlement reference for one billable Doubt Solver turn."""
    return f"student-credit:doubt:{user_id}:{turn_id}"


def practice_reference_id(*, user_id: str, test_id: str) -> str:
    """Deterministic settlement reference for one billable Practice assessment.

    The test id is itself derived from the request idempotency key, so a replayed
    or resumed Practice settles against the same reference and never debits twice.
    """
    return f"student-credit:practice:{user_id}:{test_id}"


def _user_ref(user_id: str) -> str:
    """Correlatable, non-reversible wallet reference for logs."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value.normalize(), "f")


class StudentCreditRuntime:
    """Config-driven student credit enforcement for one process."""

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

    def get_balance(self, user_id: str) -> StudentWalletBalance:
        """Read the authoritative wallet for the admission decision.

        Raises:
            StudentCreditRepositoryError: The wallet could not be read.
        """
        _started = time.monotonic()
        balance = self._repository.get_balance(user_id)
        self._balance_read_ms = max(int((time.monotonic() - _started) * 1000), 0)
        self._log_admission(
            "allowed" if balance.can_start_chargeable_work else "blocked",
            user_id=user_id,
            balance=balance,
        )
        return balance

    def ensure_can_start(self, user_id: str) -> StudentWalletBalance | None:
        """Block chargeable generation for a missing or exhausted wallet.

        Dry run is observational: it records what the decision *would* be and
        never refuses a student, including when the credit store is
        unavailable. Enforcing mode fails closed on an unreadable wallet,
        because spend that settlement cannot recover must not start.

        Returns:
            The observed wallet, or None when the balance could not be read
            during dry run. None never means "check passed".

        Raises:
            InsufficientStudentCreditsError: Enforcing and the wallet is
                missing or non-positive.
            StudentCreditRepositoryError: Enforcing and the wallet is
                unreadable.
        """
        try:
            balance = self.get_balance(user_id)
        except StudentCreditRepositoryError:
            self._log_admission("unavailable", user_id=user_id, balance=None)
            if self._policy.dry_run:
                return None
            raise
        if self._policy.dry_run or balance.can_start_chargeable_work:
            return balance
        raise InsufficientStudentCreditsError(
            "Student wallet has no credits available."
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
                "student_credit_balance_read_ms": getattr(
                    self, "_balance_read_ms", None
                ),
            },
        )
        update_request_summary(
            credit_mode="dry_run" if self._policy.dry_run else "enforcing",
            credit_admission=status,
            credit_balance=balance.credits_balance if balance else None,
        )

    def calculate_charge(self, actual_llm_cost_usd: Decimal) -> StudentCreditCharge:
        """Convert an operation's actual provider cost into whole credits."""
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
        """Charge one accepted operation exactly once.

        Callers must only invoke this for an accepted chargeable terminal
        result. Replays, rejected answers, and failed operations must not reach
        it.

        ``summary`` lets a caller supply an already-priced record set — Practice
        charges only the calls that produced the delivered questions. Omitting it
        keeps the existing behaviour of pricing the whole operation.

        Raises:
            InsufficientStudentCreditsError: The wallet cannot cover the charge.
            StudentCreditPricingIncompleteError: A chargeable call is unpriced.
            StudentCreditRepositoryError: The settlement could not be attempted.
        """
        if summary is None:
            summary = snapshot_operation_billing(operation_status=operation_status)
        if summary is None or summary.llm_call_count == 0:
            return self._skipped(
                "skipped_not_chargeable",
                reference_id=reference_id,
                user_id=user_id,
                summary=summary,
            )
        if not summary.cost_complete or summary.total_llm_cost_usd is None:
            self._log_pricing_incomplete(summary, user_id=user_id, reference_id=reference_id)
            raise StudentCreditPricingIncompleteError(
                "Operation contains provider calls with no reviewed price."
            )

        charge = self.calculate_charge(summary.total_llm_cost_usd)
        if charge.student_credits_to_debit == 0:
            return self._skipped(
                "skipped_zero_cost",
                reference_id=reference_id,
                user_id=user_id,
                summary=summary,
                charge=charge,
            )
        if self._policy.dry_run:
            self._log_settlement(
                "skipped_dry_run",
                user_id=user_id,
                reference_id=reference_id,
                summary=summary,
                charge=charge,
                credits_debited=0,
            )
            return StudentCreditSettlement(
                status="skipped_dry_run",
                reference_id=reference_id,
                credits_debited=0,
                charge=charge,
            )

        try:
            self._repository.settle(
                reference_id=reference_id,
                user_id=user_id,
                credits=charge.student_credits_to_debit,
                feature_key=feature,
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
        except InsufficientStudentCreditsError:
            # Financially significant refusal: record it before the typed error
            # propagates. Control flow and the wallet are untouched.
            self._log_settlement(
                "insufficient_credits",
                user_id=user_id,
                reference_id=reference_id,
                summary=summary,
                charge=charge,
                credits_debited=0,
            )
            raise
        except StudentCreditRepositoryError:
            self._log_settlement(
                "unavailable",
                user_id=user_id,
                reference_id=reference_id,
                summary=summary,
                charge=charge,
                credits_debited=0,
            )
            raise
        except StudentCreditLedgerConflictError:
            settled_credits = self._repository.read_settlement(reference_id)
            credits = (
                settled_credits
                if settled_credits is not None
                else charge.student_credits_to_debit
            )
            self._log_settlement(
                "already_settled",
                user_id=user_id,
                reference_id=reference_id,
                summary=summary,
                charge=charge,
                credits_debited=credits,
            )
            return StudentCreditSettlement(
                status="already_settled",
                reference_id=reference_id,
                credits_debited=credits,
                charge=charge,
            )

        self._log_settlement(
            "settled",
            user_id=user_id,
            reference_id=reference_id,
            summary=summary,
            charge=charge,
            credits_debited=charge.student_credits_to_debit,
        )
        return StudentCreditSettlement(
            status="settled",
            reference_id=reference_id,
            credits_debited=charge.student_credits_to_debit,
            charge=charge,
        )

    def _skipped(
        self,
        status: str,
        *,
        reference_id: str,
        user_id: str,
        summary: OperationBillingSummary | None,
        charge: StudentCreditCharge | None = None,
    ) -> StudentCreditSettlement:
        self._log_settlement(
            status,
            user_id=user_id,
            reference_id=reference_id,
            summary=summary,
            charge=charge,
            credits_debited=0,
        )
        return StudentCreditSettlement(
            status=status,  # type: ignore[arg-type]
            reference_id=reference_id,
            credits_debited=0,
            charge=charge,
        )

    def _log_settlement(
        self,
        status: str,
        *,
        user_id: str,
        reference_id: str,
        summary: OperationBillingSummary | None,
        charge: StudentCreditCharge | None,
        credits_debited: int,
    ) -> None:
        log_event(
            "student_credit_settlement",
            component=_COMPONENT,
            stage="settlement",
            status=status,
            details={
                "user_ref": _user_ref(user_id),
                "reference_id": reference_id,
                "operation_id": summary.operation_id if summary else None,
                "feature": summary.feature if summary else None,
                "llm_call_count": summary.llm_call_count if summary else 0,
                "actual_llm_cost_usd": (
                    _decimal_text(summary.total_llm_cost_usd) if summary else None
                ),
                "credits_per_usd": _decimal_text(self._policy.credits_per_usd),
                "target_gross_margin": _decimal_text(self._policy.target_gross_margin),
                "student_credits_to_debit": (
                    charge.student_credits_to_debit if charge else 0
                ),
                "would_debit": self._policy.dry_run,
                "credits_debited": credits_debited,
            },
        )
        update_request_summary(
            credit_mode="dry_run" if self._policy.dry_run else "enforcing",
            credit_calculated=charge.student_credits_to_debit if charge else None,
            credit_settlement=status,
            credits_debited=credits_debited,
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
                "reference_id": reference_id,
                "operation_id": summary.operation_id,
                "feature": summary.feature,
                "llm_call_count": summary.llm_call_count,
                "missing_cost_profiles": ",".join(summary.missing_cost_profiles),
                "missing_usage_call_count": summary.missing_usage_call_count,
                "billing_config_version": summary.billing_config_version,
            },
            level=logging.ERROR,
        )
        update_request_summary(
            credit_mode="dry_run" if self._policy.dry_run else "enforcing",
            credit_calculated=None,
            credit_settlement="blocked_incomplete_usage",
        )


__all__ = [
    "InsufficientStudentCreditsError",
    "StudentCreditPricingIncompleteError",
    "StudentCreditRepositoryError",
    "StudentCreditRuntime",
    "doubt_reference_id",
    "practice_reference_id",
]
