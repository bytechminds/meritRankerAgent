"""Typed student credit failures surfaced to the runtime error contract."""

from __future__ import annotations


class StudentCreditError(RuntimeError):
    """Base class carrying one stable, safe reason code."""

    reason_code = "STUDENT_CREDIT_FAILED"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.reason_code)


class InsufficientStudentCreditsError(StudentCreditError):
    """Wallet is missing, empty, or cannot cover the settled charge."""

    reason_code = "INSUFFICIENT_CREDITS"


class StudentCreditPricingIncompleteError(StudentCreditError):
    """At least one chargeable provider call has no reviewed price."""

    reason_code = "STUDENT_CREDIT_PRICING_INCOMPLETE"


class StudentCreditRepositoryError(StudentCreditError):
    """The wallet or ledger could not be read or written."""

    reason_code = "STUDENT_CREDIT_STORE_UNAVAILABLE"


class StudentCreditLedgerConflictError(StudentCreditError):
    """An existing ledger row for this reference does not match this operation."""

    reason_code = "STUDENT_CREDIT_LEDGER_CONFLICT"
