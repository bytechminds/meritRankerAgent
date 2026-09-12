"""Internal, content-free LLM usage and pricing contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

UsageSource = Literal["provider_reported", "locally_estimated", "unavailable"]
UsageStatus = Literal["succeeded", "failed", "cancelled"]
PricingStatus = Literal["available", "unavailable"]


class ProviderTokenUsage(BaseModel):
    """Normalized token counters returned by a provider."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)

    model_config = {"frozen": True, "extra": "forbid"}

    @property
    def available(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.cached_input_tokens,
                self.reasoning_tokens,
            )
        )


class LLMUsageRecord(BaseModel):
    """One safe usage record for one actual provider attempt."""

    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str | None = Field(default=None, max_length=128)
    turn_id: str | None = Field(default=None, max_length=128)
    role: str = Field(min_length=1, max_length=128)
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=128)
    deployment: str | None = Field(default=None, max_length=128)
    model_alias: str | None = Field(default=None, max_length=128)
    call_index: int = Field(default=0, ge=0)
    attempt_type: str = Field(default="primary", min_length=1, max_length=64)
    streaming: bool = False
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    usage_source: UsageSource = "unavailable"
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    pricing_version: str | None = Field(default=None, max_length=64)
    pricing_status: PricingStatus = "unavailable"
    duration_ms: int = Field(default=0, ge=0)
    status: UsageStatus
    error_type: str | None = Field(default=None, max_length=128)
    # Provider failure classification (ProviderFailureKind) preserved from the
    # originating exception so failures stay diagnosable after the fact.
    failure_kind: str | None = Field(default=None, max_length=64)

    model_config = {"frozen": True, "extra": "forbid"}

    @model_validator(mode="after")
    def validate_usage_source(self) -> LLMUsageRecord:
        has_usage = any(
            value is not None
            for value in (
                self.input_tokens,
                self.output_tokens,
                self.total_tokens,
                self.cached_input_tokens,
                self.reasoning_tokens,
            )
        )
        if self.usage_source == "unavailable" and has_usage:
            raise ValueError("unavailable usage_source cannot contain token counts")
        return self


class ModelPricing(BaseModel):
    """Versioned price for one provider model or deployment."""

    provider: str = Field(min_length=1, max_length=64)
    model_or_deployment: str = Field(min_length=1, max_length=128)
    input_cost_per_million_tokens: float = Field(ge=0.0)
    output_cost_per_million_tokens: float = Field(ge=0.0)
    cached_input_cost_per_million_tokens: float | None = Field(default=None, ge=0.0)
    reasoning_cost_policy: Literal["included_in_output"] = "included_in_output"
    currency: Literal["USD"] = "USD"
    effective_from: str = Field(min_length=1, max_length=32)

    model_config = {"frozen": True, "extra": "forbid"}


class LLMPricingConfig(BaseModel):
    """Root pricing configuration loaded outside provider adapters."""

    pricing_version: str = Field(min_length=1, max_length=64)
    billing_version: str | None = Field(default=None, min_length=1, max_length=64)
    metering_enabled: bool | None = None
    credit_debit_enabled: bool | None = None
    currency: Literal["USD"] = "USD"
    models: tuple[ModelPricing, ...] = ()
    model_alias_profiles: dict[str, object] = Field(default_factory=dict)
    infra: dict[str, object] = Field(default_factory=dict)
    credits: dict[str, object] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}


class LLMRoleUsageSummary(BaseModel):
    calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    usage_complete: bool
    cost_complete: bool

    model_config = {"frozen": True, "extra": "forbid"}


class LLMUsageSummary(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    duration_ms: int = Field(ge=0)
    usage_complete: bool
    cost_complete: bool
    roles: dict[str, LLMRoleUsageSummary] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}
