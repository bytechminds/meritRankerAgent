"""Content-free contracts for student runtime credit enforcement.

The student wallet is authoritative in DynamoDB (``UserCredits``) and every
debit is recorded once in ``CreditLedger``.  These contracts carry only safe
identifiers, integer credits, and Decimal money — never prompts, answers, or
provider payloads.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
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
    "released",
    "already_released",
    "skipped_no_authorization",
]


class StudentCreditPolicy(BaseModel):
    """Validated student pricing policy. The only student pricing multiplier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enforcement_enabled: bool
    dry_run: bool
    credits_per_usd: Decimal = Field(gt=Decimal("0"))
    target_gross_margin: Decimal = Field(ge=Decimal("0"), lt=Decimal("1"))
    rounding_mode: Literal["CEIL"]
    doubt_authorization_credits: int = Field(ge=1)
    practice_min_authorization_credits: int = Field(ge=1)
    practice_authorization_credits_per_question: int = Field(ge=1)

    @property
    def margin_divisor(self) -> Decimal:
        """``1 - margin``. Guaranteed non-zero by the ``lt=1`` bound above."""
        return Decimal("1") - self.target_gross_margin


class StudentCreditPolicyCurrency(BaseModel):
    """Currency inputs for the versioned, server-owned credit policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_currency: Literal["INR"]
    credit_value_inr: Decimal = Field(gt=Decimal("0"))
    usd_to_inr_business_rate: Decimal = Field(gt=Decimal("0"))


class StudentCreditPolicyPricing(BaseModel):
    """Pricing inputs for the versioned, server-owned credit policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_gross_margin: Decimal = Field(ge=Decimal("0"), lt=Decimal("1"))
    rounding_mode: Literal["CEIL"]


class StudentCreditPolicyAuthorization(BaseModel):
    """Preflight authorization inputs for the versioned credit policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doubt_min_credits: int = Field(ge=1)
    practice_min_credits: int = Field(ge=1)
    practice_credits_per_question: int = Field(ge=1)


class StudentCreditPolicyDocument(BaseModel):
    """Strict on-disk source for student business and pricing policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: Literal["student-credit-v1"]
    currency: StudentCreditPolicyCurrency
    pricing: StudentCreditPolicyPricing
    authorization: StudentCreditPolicyAuthorization

    def build_runtime_policy(
        self,
        *,
        enforcement_enabled: bool,
        dry_run: bool,
    ) -> StudentCreditPolicy:
        with localcontext() as context:
            context.prec = 34
            credits_per_usd = (
                self.currency.usd_to_inr_business_rate / self.currency.credit_value_inr
            )
        return StudentCreditPolicy(
            enforcement_enabled=enforcement_enabled,
            dry_run=dry_run,
            credits_per_usd=credits_per_usd,
            target_gross_margin=self.pricing.target_gross_margin,
            rounding_mode=self.pricing.rounding_mode,
            doubt_authorization_credits=self.authorization.doubt_min_credits,
            practice_min_authorization_credits=self.authorization.practice_min_credits,
            practice_authorization_credits_per_question=(
                self.authorization.practice_credits_per_question
            ),
        )


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
