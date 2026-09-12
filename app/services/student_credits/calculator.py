"""Student credit calculation from an already-measured operation LLM cost.

This module deliberately knows nothing about providers, models, tokens, or
features.  Provider/model pricing is owned by ``services.llm.billing``; the only
student-facing multiplier is the configured gross margin.

    customer_equivalent_usd = actual_llm_cost_usd / (1 - TARGET_GROSS_MARGIN)
    student_credits         = CEIL(customer_equivalent_usd * CREDITS_PER_USD)
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal, localcontext

from schemas.student_credits import StudentCreditCharge, StudentCreditPolicy

# Fixed working precision so a charge never depends on the caller's ambient
# decimal context. Money is never computed in binary floating point.
_CALCULATION_PRECISION = 34


def calculate_credits(
    actual_llm_cost_usd: Decimal,
    *,
    policy: StudentCreditPolicy,
) -> StudentCreditCharge:
    """Convert one operation's actual provider cost into whole student credits.

    Args:
        actual_llm_cost_usd: Summed cost of the operation's chargeable provider
            calls, in USD. Must already exclude infrastructure allowances.
        policy: Validated pricing policy.

    Raises:
        ValueError: If the cost is negative.
    """
    if actual_llm_cost_usd < 0:
        raise ValueError("actual_llm_cost_usd cannot be negative")

    with localcontext() as context:
        context.prec = _CALCULATION_PRECISION
        customer_equivalent_usd = actual_llm_cost_usd / policy.margin_divisor
        credits = int(
            (customer_equivalent_usd * policy.credits_per_usd).to_integral_value(
                rounding=ROUND_CEILING
            )
        )

    return StudentCreditCharge(
        actual_llm_cost_usd=actual_llm_cost_usd,
        customer_equivalent_usd=customer_equivalent_usd,
        credits_per_usd=policy.credits_per_usd,
        target_gross_margin=policy.target_gross_margin,
        student_credits_to_debit=credits,
    )
