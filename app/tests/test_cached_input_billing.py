"""Authoritative cached-input billing (R-1).

Cached tokens are priced at the verified cached rate. Where that rate is
absent, or provider usage violates an invariant, the operation is marked
incomplete rather than guessed or clamped — so no student debit occurs.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from schemas.billing import BillingConfig
from schemas.llm_usage import LLMUsageRecord
from services.llm.billing import OperationDescriptor, calculate_operation_billing

MILLION = Decimal("1000000")


def _config() -> BillingConfig:
    return BillingConfig.model_validate(
        {
            "billing_version": "billing-cached-test",
            "metering_enabled": True,
            "credit_debit_enabled": False,
            "models": [
                {   # verified cached rate
                    "provider": "azure_openai",
                    "model_or_deployment": "gpt-4.1-mini",
                    "input_cost_per_million_tokens": "0.40",
                    "output_cost_per_million_tokens": "1.60",
                    "cached_input_cost_per_million_tokens": "0.10",
                    "effective_from": "2026-09-11",
                },
                {   # NO cached rate configured
                    "provider": "gemini",
                    "model_or_deployment": "gemini-3.7-flash",
                    "input_cost_per_million_tokens": "0.75",
                    "output_cost_per_million_tokens": "3.75",
                    "effective_from": "2026-09-11",
                },
            ],
            "infra": {
                "doubt": {"fixed_cost": "0.001"},
                "quick_practice": {"fixed_cost": "0.003"},
                "mock_test": {"fixed_cost": "0.006"},
            },
            "credits": {"usd_per_credit": "0.001", "pricing_factor": "2"},
        }
    )


def _record(
    *,
    provider: str = "azure_openai",
    model: str = "gpt-4.1-mini",
    input_tokens: int,
    output_tokens: int,
    cached: int | None = None,
) -> LLMUsageRecord:
    return LLMUsageRecord(
        request_id="r",
        role="math.generator",
        provider=provider,
        model=model,
        deployment=model if provider == "azure_openai" else None,
        streaming=False,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_input_tokens=cached,
        usage_source="provider_reported",
        status="succeeded",
    )


def _summary(*records: LLMUsageRecord, feature: str = "doubt"):
    return calculate_operation_billing(
        descriptor=OperationDescriptor(operation_id="op", feature=feature),
        records=tuple(records),
        operation_status="completed",
        config=_config(),
    )


# --------------------------------------------------------------------------
# Case A/B — unchanged behaviour
# --------------------------------------------------------------------------


def test_case_a_no_cached_field_preserves_existing_cost() -> None:
    s = _summary(_record(input_tokens=2811, output_tokens=134, cached=None))

    expected = (Decimal(2811) * Decimal("0.40") + Decimal(134) * Decimal("1.60")) / MILLION
    assert s.cost_complete is True
    assert s.total_llm_cost_usd == expected == Decimal("0.0013388")


def test_case_b_zero_cached_preserves_existing_cost() -> None:
    s = _summary(_record(input_tokens=2811, output_tokens=134, cached=0))

    assert s.cost_complete is True
    assert s.total_llm_cost_usd == Decimal("0.0013388")


# --------------------------------------------------------------------------
# Case C/D — the fix
# --------------------------------------------------------------------------


def test_case_c_partial_cache_matches_independently_measured_provider_cost() -> None:
    """The exact live vector: in=2811 cached=2688 out=134 -> $0.0005324."""
    s = _summary(_record(input_tokens=2811, output_tokens=134, cached=2688))

    assert s.cost_complete is True
    assert s.total_llm_cost_usd == Decimal("0.0005324")


def test_case_d_fully_cached_input_uses_only_the_cached_rate() -> None:
    s = _summary(_record(input_tokens=1000, output_tokens=100, cached=1000))

    expected = (Decimal(1000) * Decimal("0.10") + Decimal(100) * Decimal("1.60")) / MILLION
    assert s.total_llm_cost_usd == expected
    assert s.cost_complete is True


def test_cached_billing_never_exceeds_uncached_billing() -> None:
    cached = _summary(_record(input_tokens=2811, output_tokens=134, cached=2688))
    uncached = _summary(_record(input_tokens=2811, output_tokens=134, cached=0))

    assert cached.total_llm_cost_usd < uncached.total_llm_cost_usd


# --------------------------------------------------------------------------
# Case E — invalid usage is never clamped
# --------------------------------------------------------------------------


def test_case_e_cached_greater_than_input_marks_incomplete_without_clamping() -> None:
    s = _summary(_record(input_tokens=100, output_tokens=10, cached=500))

    assert s.cost_complete is False
    assert s.invalid_usage_call_count == 1
    assert s.total_llm_cost_usd is None
    assert s.calculated_credits is None


def test_negative_cached_tokens_are_unrepresentable() -> None:
    """The schema forbids them, so authoritative billing cannot see one."""
    with pytest.raises(ValueError):
        _record(input_tokens=100, output_tokens=10, cached=-1)


# --------------------------------------------------------------------------
# Case F — cached tokens with no verified cached rate
# --------------------------------------------------------------------------


def test_case_f_cached_without_cached_rate_blocks_instead_of_guessing() -> None:
    s = _summary(
        _record(
            provider="gemini",
            model="gemini-3.7-flash",
            input_tokens=2000,
            output_tokens=100,
            cached=1500,
        )
    )

    assert s.cost_complete is False
    assert s.missing_cached_rate_profiles == ("gemini:gemini-3.7-flash",)
    assert s.total_llm_cost_usd is None
    assert s.calculated_credits is None


def test_case_f_model_without_cached_rate_is_unaffected_when_nothing_cached() -> None:
    s = _summary(
        _record(
            provider="gemini",
            model="gemini-3.7-flash",
            input_tokens=2000,
            output_tokens=100,
            cached=0,
        )
    )

    expected = (Decimal(2000) * Decimal("0.75") + Decimal(100) * Decimal("3.75")) / MILLION
    assert s.cost_complete is True
    assert s.total_llm_cost_usd == expected


# --------------------------------------------------------------------------
# Case G — operation total across mixed calls
# --------------------------------------------------------------------------


def test_case_g_operation_total_equals_sum_of_per_call_costs() -> None:
    cached_call = _record(input_tokens=2811, output_tokens=134, cached=2688)
    plain_call = _record(input_tokens=1000, output_tokens=200, cached=0)
    gemini_call = _record(
        provider="gemini", model="gemini-3.7-flash",
        input_tokens=1950, output_tokens=230, cached=None,
    )

    s = _summary(cached_call, plain_call, gemini_call)

    per_call = (
        Decimal("0.0005324")
        + (Decimal(1000) * Decimal("0.40") + Decimal(200) * Decimal("1.60")) / MILLION
        + (Decimal(1950) * Decimal("0.75") + Decimal(230) * Decimal("3.75")) / MILLION
    )
    assert s.cost_complete is True
    assert s.total_llm_cost_usd == per_call
    assert s.llm_call_count == 3


def test_one_blocked_call_blocks_the_whole_operation() -> None:
    good = _record(input_tokens=1000, output_tokens=100, cached=0)
    blocked = _record(
        provider="gemini", model="gemini-3.7-flash",
        input_tokens=500, output_tokens=50, cached=400,
    )

    s = _summary(good, blocked)

    assert s.cost_complete is False
    assert s.calculated_credits is None


# --------------------------------------------------------------------------
# Credit formula unchanged
# --------------------------------------------------------------------------


def test_student_credits_follow_the_unchanged_formula_on_cached_cost() -> None:
    from decimal import ROUND_CEILING

    from schemas.student_credits import StudentCreditPolicy
    from services.student_credits.calculator import calculate_credits

    s = _summary(_record(input_tokens=2811, output_tokens=134, cached=2688))
    assert s.total_llm_cost_usd is not None

    charge = calculate_credits(
        s.total_llm_cost_usd,
        policy=StudentCreditPolicy(
            enforcement_enabled=True, dry_run=False,
            credits_per_usd=Decimal("50"),
            target_gross_margin=Decimal("0.40"),
            rounding_mode="CEIL",
        ),
    )
    expected = int(
        (s.total_llm_cost_usd * Decimal("50") / Decimal("0.60")).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    assert charge.student_credits_to_debit == expected
