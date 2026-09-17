"""Pricing coverage gate for student credit enforcement.

A student charge is derived from provider cost, so every billable model the
runtime can actually reach must resolve to a reviewed price. An unpriced model
must never silently become ``$0``.

This inspects the *compiled* registry — after ``model_registry_env`` applies
deployment overlays — because the deployed model name can differ from the YAML
default. Coverage is therefore a property of a deployment, not of a file.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from schemas.llm_usage import LLMUsageRecord
from services.llm.billing import (
    OperationDescriptor,
    _find_rate,
    calculate_operation_billing,
    load_billing_config,
)
from services.llm.orchestration.config_registry import get_registry

# The mock provider performs no provider call and incurs no cost, so it is not
# a billable combination. Everything else the router can reach is.
NON_BILLABLE_PROVIDERS = frozenset({"mock"})


def reachable_billable_aliases() -> set[str]:
    """Every billable model alias an *active* production route can reach.

    Active-route membership comes from the registry's own
    ``_active_route_model_aliases``, which excludes the optional test-only
    difficulties (``gemini_test``, ``deepseek_test``, ``deepseek_azure_test``,
    ``cat_advanced``, ``sbi_po_complex``, ``cat_lrdi``). Those cannot be
    selected at runtime: ``normalize_difficulty`` maps every classifier value
    into {basic, intermediate, advanced, default}, so no request can resolve a
    route keyed by any other difficulty. Reusing the registry's definition
    keeps one source of truth — a second hard-coded list here could silently
    drift and exclude a genuinely reachable model from the gate.

    Provider-failure fallback chains (``fallback_models``) are expanded
    transitively, because those models do execute and do cost money.
    """
    registry = get_registry()
    model_map = registry.model_map
    reachable: set[str] = set()

    def expand(alias: str) -> None:
        if alias in reachable or alias not in model_map:
            return
        reachable.add(alias)
        for fallback_alias in model_map[alias].fallback_models or []:
            expand(fallback_alias)

    for alias in registry._active_route_model_aliases():  # noqa: SLF001
        expand(alias)

    return {
        alias
        for alias in reachable
        if model_map[alias].provider not in NON_BILLABLE_PROVIDERS
    }


def unpriced_reachable_models() -> list[tuple[str, str, str]]:
    """Return ``(alias, provider, resolved model)`` for every unpriced route."""
    registry = get_registry()
    billing_config = load_billing_config()
    unpriced: list[tuple[str, str, str]] = []

    for alias in sorted(reachable_billable_aliases()):
        model_config = registry.model_map[alias]
        resolved = model_config.model_id or model_config.deployment or alias
        rate = _find_rate(
            billing_config,
            provider=model_config.provider,
            model=resolved,
            deployment=model_config.deployment,
        )
        if rate is None:
            unpriced.append((alias, model_config.provider, resolved))
    return unpriced


def test_every_reachable_billable_model_resolves_to_a_price() -> None:
    """Release gate. Real student deductions must not be enabled while this fails."""
    unpriced = unpriced_reachable_models()
    reachable = len(reachable_billable_aliases())
    priced = reachable - len(unpriced)

    detail = "\n".join(
        f"  {alias}  provider={provider}  model={resolved}"
        for alias, provider, resolved in unpriced
    )
    assert not unpriced, (
        f"reachable billable models = {reachable}\n"
        f"priced reachable models   = {priced}\n"
        f"coverage                  = {priced / reachable:.1%}\n"
        f"unpriced reachable models:\n{detail}"
    )


def test_pricing_coverage_inspects_the_compiled_registry() -> None:
    """Guards the gate itself: it must never pass vacuously or read only YAML."""
    registry = get_registry()
    aliases = reachable_billable_aliases()

    assert aliases, "no billable models reachable — the gate would pass vacuously"
    assert all(alias in registry.model_map for alias in aliases)
    # Aliases whose deployment comes from an unset environment variable resolve
    # to no provider model at all. They must still be reported rather than
    # skipped, which is only possible when the gate reads the compiled registry.
    unresolved = {
        alias
        for alias in aliases
        if not (registry.model_map[alias].model_id or registry.model_map[alias].deployment)
    }
    reported = {alias for alias, _provider, _model in unpriced_reachable_models()}
    assert unresolved <= reported


# ---------------------------------------------------------------------------
# Exact cost verification for newly priced models (Step 7)
# ---------------------------------------------------------------------------


def _cost_usd(
    provider: str, model: str, *, input_tokens: int, output_tokens: int
) -> Decimal:
    """Operation LLM cost for one call, using the real pricing catalog."""
    record = LLMUsageRecord(
        request_id="pricing-check",
        role="pricing.check",
        provider=provider,
        model=model,
        deployment=None,
        attempt_type="primary",
        streaming=False,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        usage_source="provider_reported",
        status="succeeded",
    )
    summary = calculate_operation_billing(
        descriptor=OperationDescriptor(operation_id="pricing-check", feature="doubt"),
        records=(record,),
        operation_status="completed",
    )
    assert summary.cost_complete, f"{provider}/{model} is not priced"
    assert summary.total_llm_cost_usd is not None
    return summary.total_llm_cost_usd


@pytest.mark.parametrize(
    ("provider", "model", "input_rate", "output_rate"),
    [
        ("gemini", "gemini-3.7-flash", Decimal("0.75"), Decimal("3.75")),
        ("openai", "gpt-4o-mini", Decimal("0.15"), Decimal("0.60")),
        ("azure_openai", "gpt-5.6-terra", Decimal("2.00"), Decimal("12.00")),
        ("azure_openai", "DeepSeek-V4-Pro", Decimal("1.74"), Decimal("3.48")),
    ],
)
def test_newly_priced_model_costs_match_the_configured_rates(
    provider: str, model: str, input_rate: Decimal, output_rate: Decimal
) -> None:
    """1M input and 1M output must cost exactly the configured per-million rate."""
    assert (
        _cost_usd(provider, model, input_tokens=1_000_000, output_tokens=0) == input_rate
    )
    assert (
        _cost_usd(provider, model, input_tokens=0, output_tokens=1_000_000) == output_rate
    )
    # Mixed: 250k input + 100k output.
    mixed = _cost_usd(provider, model, input_tokens=250_000, output_tokens=100_000)
    expected = (
        Decimal("250000") * input_rate + Decimal("100000") * output_rate
    ) / Decimal("1000000")
    assert mixed == expected


@pytest.mark.parametrize(
    ("alias", "deployment", "input_rate", "output_rate", "cached_rate"),
    [
        ("openai_gpt_5_6_terra", "gpt-5.6-terra", "2.00", "12.00", "0.20"),
        ("azure_deepseek_v4_pro", "DeepSeek-V4-Pro", "1.74", "3.48", "0.145"),
        ("deepseek_v4pro", "DeepSeek-V4-Pro", "1.74", "3.48", "0.145"),
    ],
)
def test_o_series_replacement_aliases_resolve_to_reviewed_azure_rates(
    alias: str, deployment: str, input_rate: str, output_rate: str, cached_rate: str
) -> None:
    """Terra and DeepSeek-V4-Pro are reachable, so the compiled registry must price them."""
    registry = get_registry()
    assert alias in reachable_billable_aliases()
    model_config = registry.model_map[alias]
    assert (model_config.provider, model_config.deployment) == ("azure_openai", deployment)

    rate = _find_rate(
        load_billing_config(),
        provider=model_config.provider,
        model=model_config.model_id or model_config.deployment or alias,
        deployment=model_config.deployment,
    )
    assert rate is not None
    assert rate.input_cost_per_million_tokens == Decimal(input_rate)
    assert rate.output_cost_per_million_tokens == Decimal(output_rate)
    assert rate.cached_input_cost_per_million_tokens == Decimal(cached_rate)


def test_newly_priced_models_do_not_disturb_existing_rates() -> None:
    """Regression guard: the rates already in the catalog are unchanged."""
    million = {"input_tokens": 1_000_000, "output_tokens": 0}
    million_out = {"input_tokens": 0, "output_tokens": 1_000_000}
    assert _cost_usd("openai", "gpt-4.1-mini", **million) == Decimal("0.40")
    assert _cost_usd("openai", "gpt-4.1", **million_out) == Decimal("8.00")
    assert _cost_usd("gemini", "gemini-3.1-flash-lite", **million) == Decimal("0.25")
    assert _cost_usd("bedrock", "zai.glm-4.7-flash", **million_out) == Decimal("0.48")


# ---------------------------------------------------------------------------
# Scheduled provider price changes (Part C)
# ---------------------------------------------------------------------------

# Google publishes Gemini 3.7 Flash introductory rates as valid "through
# December 31, 2026", with standard rates applying from 2027-01-01.
# Verified 2026-09-11 against ai.google.dev/gemini-api/docs/pricing.
_GEMINI_3_7_FLASH_TRANSITION = date(2027, 1, 1)
_GEMINI_3_7_FLASH_INTRODUCTORY = (Decimal("0.75"), Decimal("3.75"))
_GEMINI_3_7_FLASH_STANDARD = (Decimal("1.50"), Decimal("7.50"))


def test_gemini_3_7_flash_rate_matches_the_currently_applicable_tier() -> None:
    """An expired promotional rate must fail loudly, never under-bill silently.

    ``_find_rate`` performs no date-based selection and the schema forbids two
    rows for one provider/model, so the catalog can hold only the rate that is
    applicable today. This guard flips on the published transition date and
    fails until the row is reviewed and updated. It is deliberately a date
    assertion rather than a date-based pricing engine.
    """
    expected_input, expected_output = (
        _GEMINI_3_7_FLASH_INTRODUCTORY
        if date.today() < _GEMINI_3_7_FLASH_TRANSITION
        else _GEMINI_3_7_FLASH_STANDARD
    )
    actual_input = _cost_usd(
        "gemini", "gemini-3.7-flash", input_tokens=1_000_000, output_tokens=0
    )
    actual_output = _cost_usd(
        "gemini", "gemini-3.7-flash", input_tokens=0, output_tokens=1_000_000
    )

    assert (actual_input, actual_output) == (expected_input, expected_output), (
        "gemini-3.7-flash pricing is stale. Google's introductory rate "
        f"({_GEMINI_3_7_FLASH_INTRODUCTORY[0]}/{_GEMINI_3_7_FLASH_INTRODUCTORY[1]}) "
        f"expired on {_GEMINI_3_7_FLASH_TRANSITION}; the standard rate "
        f"({_GEMINI_3_7_FLASH_STANDARD[0]}/{_GEMINI_3_7_FLASH_STANDARD[1]}) now "
        "applies. Update config/llm/model_pricing.yaml before charging students."
    )
