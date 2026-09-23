"""Typed student credit failures surfaced to the runtime error contract."""

from __future__ import annotations


class StudentCreditError(RuntimeError):
    """Base class carrying one stable, safe reason code."""

    reason_code = "STUDENT_CREDIT_FAILED"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.reason_code)


INSUFFICIENT_CREDITS_LABEL = "Not enough credits to continue. Add credits and try again."


class InsufficientStudentCreditsError(StudentCreditError):
    """Wallet is missing, empty, or cannot cover the settled charge."""

    reason_code = "INSUFFICIENT_CREDITS"

    def __init__(
        self,
        message: str | None = None,
        *,
        required_credits: int | None = None,
        available_credits: int | None = None,
    ) -> None:
        super().__init__(message)
        self.required_credits = required_credits
        self.available_credits = available_credits

    def client_fields(self) -> dict[str, object]:
        """Stable, student-safe refusal fields; amounts only when actually known."""
        fields: dict[str, object] = {
            "code": self.reason_code,
            "retryable": False,
            "action": "ADD_CREDITS",
        }
        if self.required_credits is not None:
            fields["requiredCredits"] = self.required_credits
        if self.available_credits is not None:
            fields["availableCredits"] = self.available_credits
        return fields


class StudentCreditPricingIncompleteError(StudentCreditError):
    """At least one chargeable provider call has no reviewed price."""

    reason_code = "STUDENT_CREDIT_PRICING_INCOMPLETE"


class StudentCreditRepositoryError(StudentCreditError):
    """The wallet or ledger could not be read or written."""

    reason_code = "STUDENT_CREDIT_STORE_UNAVAILABLE"


class StudentCreditLedgerConflictError(StudentCreditError):
    """An existing ledger row for this reference does not match this operation."""

    reason_code = "STUDENT_CREDIT_LEDGER_CONFLICT"
