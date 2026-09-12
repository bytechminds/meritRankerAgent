"""Exact provider-cost checks for the statically priced Azure deployments.

Rates come from the project owner's authoritative Azure pricing sheet
(validated 2026-09-11). These tests pin the configured numbers so a silent edit
or a lost row cannot change what a student is charged.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from schemas.llm_usage import LLMUsageRecord
from services.llm.billing import (
    OperationDescriptor,
    calculate_operation_billing,
    load_billing_config,
)

MILLION = 1_000_000

# provider deployment -> (input per 1M, cached input per 1M, output per 1M)
AZURE_RATES: dict[str, tuple[str, str, str]] = {
    "gpt-4.1-mini": ("0.40", "0.10", "1.60"),
    "gpt-4.1": ("2.00", "0.50", "8.00"),
    "o4-mini": ("1.10", "0.28", "4.40"),
    "o3": ("2.00", "0.50", "8.00"),
    "gpt-5.4-mini": ("0.75", "0.08", "4.50"),
    "gpt-5.6-luna": ("0.20", "0.02", "1.20"),
    "gpt-5.4": ("2.50", "0.25", "15.00"),
}


def _record(deployment: str, *, input_tokens: int, output_tokens: int) -> LLMUsageRecord:
    return LLMUsageRecord(
        request_id="azure-pricing",
        role="math.generator",
        provider="azure_openai",
        model=deployment,
        deployment=deployment,
        streaming=False,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        usage_source="provider_reported",
        status="succeeded",
    )


def _llm_cost(deployment: str, *, input_tokens: int, output_tokens: int) -> Decimal:
    """Operation LLM cost only — the exact input to the student calculator."""
    summary = calculate_operation_billing(
        descriptor=OperationDescriptor(operation_id="op", feature="doubt"),
        records=(_record(deployment, input_tokens=input_tokens, output_tokens=output_tokens),),
        operation_status="completed",
    )
    assert summary.cost_complete, f"{deployment} must resolve to a reviewed rate"
    assert summary.total_llm_cost_usd is not None
    return summary.total_llm_cost_usd


@pytest.mark.parametrize("deployment", sorted(AZURE_RATES))
def test_one_million_input_tokens_costs_the_configured_input_rate(deployment: str) -> None:
    expected = Decimal(AZURE_RATES[deployment][0])

    assert _llm_cost(deployment, input_tokens=MILLION, output_tokens=0) == expected


@pytest.mark.parametrize("deployment", sorted(AZURE_RATES))
def test_one_million_output_tokens_costs_the_configured_output_rate(deployment: str) -> None:
    expected = Decimal(AZURE_RATES[deployment][2])

    assert _llm_cost(deployment, input_tokens=0, output_tokens=MILLION) == expected


@pytest.mark.parametrize("deployment", sorted(AZURE_RATES))
def test_configured_cached_input_rate_matches_the_pricing_sheet(deployment: str) -> None:
    """Cached rates are recorded for audit; V1 bills cached input at the full rate."""
    config = load_billing_config()
    rate = next(
        item
        for item in config.models
        if item.provider == "azure_openai" and item.model_or_deployment == deployment
    )

    assert rate.cached_input_cost_per_million_tokens == Decimal(AZURE_RATES[deployment][1])


@pytest.mark.parametrize("deployment", sorted(AZURE_RATES))
def test_realistic_mixed_token_operation_is_decimal_exact(deployment: str) -> None:
    input_rate, _cached, output_rate = AZURE_RATES[deployment]
    input_tokens, output_tokens = 12_500, 3_200
    expected = (
        Decimal(input_tokens) * Decimal(input_rate)
        + Decimal(output_tokens) * Decimal(output_rate)
    ) / Decimal(MILLION)

    assert _llm_cost(
        deployment, input_tokens=input_tokens, output_tokens=output_tokens
    ) == expected


def test_student_credits_for_a_realistic_azure_doubt_operation() -> None:
    """End-to-end: gpt-4.1-mini generation -> USD -> CEIL(usd * 50 / 0.60)."""
    from schemas.student_credits import StudentCreditPolicy
    from services.student_credits.calculator import calculate_credits

    cost = _llm_cost("gpt-4.1-mini", input_tokens=12_500, output_tokens=3_200)
    # 12500*0.40/1e6 + 3200*1.60/1e6 = 0.005 + 0.00512 = 0.01012
    assert cost == Decimal("0.01012")

    charge = calculate_credits(
        cost,
        policy=StudentCreditPolicy(
            enforcement_enabled=True,
            dry_run=False,
            credits_per_usd=Decimal("50"),
            target_gross_margin=Decimal("0.40"),
            rounding_mode="CEIL",
        ),
    )
    # 0.01012 / 0.60 * 50 = 0.84333... -> CEIL -> 1
    assert charge.student_credits_to_debit == 1
