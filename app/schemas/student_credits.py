"""Content-free contracts for student runtime credit enforcement.

The student wallet is authoritative in DynamoDB (``UserCredits``) and every
debit is recorded once in ``CreditLedger``.  These contracts carry only safe
identifiers, integer credits, and Decimal money — never prompts, answers, or
provider payloads.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

StudentCreditSettlementStatus = Literal[
    "settled",
    "already_settled",
    "skipped_disabled",
    "skipped_dry_run",
    "skipped_not_chargeable",
    "skipped_zero_cost",
    "blocked_pricing_incomplete",
]


class StudentCreditPolicy(BaseModel):
    """Validated student pricing policy. The only student pricing multiplier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enforcement_enabled: bool
    dry_run: bool
    credits_per_usd: Decimal = Field(gt=Decimal("0"))
    target_gross_margin: Decimal = Field(ge=Decimal("0"), lt=Decimal("1"))
    rounding_mode: Literal["CEIL"]

    @property
    def margin_divisor(self) -> Decimal:
        """``1 - margin``. Guaranteed non-zero by the ``lt=1`` bound above."""
        return Decimal("1") - self.target_gross_margin


class StudentWalletBalance(BaseModel):
    """One authoritative wallet read. ``exists=False`` means no wallet row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = Field(min_length=1, max_length=128)
    exists: bool
    credits_balance: int

    @property
    def can_start_chargeable_work(self) -> bool:
        """Admission rule: a wallet must exist and hold a positive balance."""
        return self.exists and self.credits_balance > 0


class StudentCreditCharge(BaseModel):
    """The calculated student charge for one operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actual_llm_cost_usd: Decimal = Field(ge=Decimal("0"))
    customer_equivalent_usd: Decimal = Field(ge=Decimal("0"))
    credits_per_usd: Decimal = Field(gt=Decimal("0"))
    target_gross_margin: Decimal = Field(ge=Decimal("0"), lt=Decimal("1"))
    student_credits_to_debit: int = Field(ge=0)


class StudentCreditSettlement(BaseModel):
    """Terminal settlement outcome. ``credits_debited`` is what the wallet lost."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: StudentCreditSettlementStatus
    reference_id: str = Field(min_length=1, max_length=256)
    credits_debited: int = Field(ge=0)
    charge: StudentCreditCharge | None = None
    balance_before: int | None = None

    @model_validator(mode="after")
    def validate_debited_credits(self) -> StudentCreditSettlement:
        if self.status not in {"settled", "already_settled"} and self.credits_debited:
            raise ValueError("credits_debited must be zero unless the wallet was debited")
        return self
