"""Immutable, content-free contracts for shadow AI usage billing."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BillingModelRate(BaseModel):
    """One provider/model rate in USD per one million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    model_or_deployment: str = Field(min_length=1, max_length=128)
    input_cost_per_million_tokens: Decimal = Field(ge=Decimal("0"))
    output_cost_per_million_tokens: Decimal = Field(ge=Decimal("0"))
    cached_input_cost_per_million_tokens: Decimal | None = Field(
        default=None, ge=Decimal("0")
    )
    effective_from: str = Field(min_length=1, max_length=32)


class BillingAliasProfile(BaseModel):
    """Auditable runtime alias identity; unknown rates are explicit in shadow mode."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1, max_length=64)
    model_or_deployment: str = Field(min_length=1, max_length=128)
    rate_status: Literal["available", "unknown"]


class BillingFeatureCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fixed_cost: Decimal = Field(ge=Decimal("0"))


class BillingCreditsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    usd_per_credit: Decimal = Field(gt=Decimal("0"))
    pricing_factor: Decimal = Field(gt=Decimal("0"))


class BillingConfig(BaseModel):
    """Cached V1 billing configuration. Credit debit remains intentionally disabled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    billing_version: str = Field(min_length=1, max_length=64)
    pricing_version: str | None = Field(default=None, min_length=1, max_length=64)
    currency: Literal["USD"] = "USD"
    metering_enabled: bool
    credit_debit_enabled: bool
    models: tuple[BillingModelRate, ...]
    model_alias_profiles: dict[str, BillingAliasProfile] = Field(default_factory=dict)
    infra: dict[str, BillingFeatureCost]
    credits: BillingCreditsConfig

    @model_validator(mode="after")
    def validate_shadow_configuration(self) -> BillingConfig:
        required_features = {"doubt", "quick_practice", "mock_test"}
        missing_features = sorted(required_features - set(self.infra))
        if missing_features:
            raise ValueError(
                "infra must configure fixed_cost for: " + ", ".join(missing_features)
            )
        if self.credit_debit_enabled:
            raise ValueError("credit_debit_enabled must remain false for billing-v1 shadow mode")
        seen: set[tuple[str, str]] = set()
        for item in self.models:
            key = (item.provider.casefold(), item.model_or_deployment.casefold())
            if key in seen:
                raise ValueError(
                    "models must not contain duplicate provider/model_or_deployment rates"
                )
            seen.add(key)
        for alias, profile in self.model_alias_profiles.items():
            if profile.rate_status != "available":
                continue
            key = (
                profile.provider.casefold(),
                profile.model_or_deployment.casefold(),
            )
            if key not in seen:
                raise ValueError(
                    f"available model_alias_profiles entry {alias!r} has no reviewed rate"
                )
        return self


class OperationBillingSummary(BaseModel):
    """Safe terminal accounting result. Null cost values mean incomplete measurement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation_id: str = Field(min_length=1, max_length=128)
    feature: str = Field(min_length=1, max_length=64)
    operation_status: str = Field(min_length=1, max_length=64)
    total_input_tokens: int = Field(ge=0)
    total_output_tokens: int = Field(ge=0)
    llm_call_count: int = Field(ge=0)
    total_llm_cost_usd: Decimal | None = Field(default=None, ge=Decimal("0"))
    infra_cost_usd: Decimal = Field(ge=Decimal("0"))
    actual_usage_cost_usd: Decimal | None = Field(default=None, ge=Decimal("0"))
    pricing_factor: Decimal = Field(gt=Decimal("0"))
    usd_per_credit: Decimal = Field(gt=Decimal("0"))
    calculated_credits: int | None = Field(default=None, ge=0)
    credit_debit_enabled: bool
    credits_debited: int = Field(ge=0)
    billing_config_version: str = Field(min_length=1, max_length=64)
    cost_complete: bool
    missing_cost_profiles: tuple[str, ...] = ()
    missing_usage_call_count: int = Field(ge=0)
