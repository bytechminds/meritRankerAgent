"""Versioned local LLM cost estimation; provider invoices remain authoritative."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from schemas.llm_usage import LLMPricingConfig, ModelPricing, ProviderTokenUsage

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[2] / "config" / "llm" / "model_pricing.yaml"


@lru_cache(maxsize=4)
def load_pricing_config(path: Path = DEFAULT_PRICING_PATH) -> LLMPricingConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return LLMPricingConfig.model_validate(payload)


def find_model_pricing(
    config: LLMPricingConfig,
    *,
    provider: str,
    model: str,
    deployment: str | None,
) -> ModelPricing | None:
    candidates = {
        value.strip().lower()
        for value in (deployment, model)
        if isinstance(value, str) and value.strip()
    }
    provider_key = provider.strip().lower()
    for item in config.models:
        if (
            item.provider.strip().lower() == provider_key
            and item.model_or_deployment.strip().lower() in candidates
        ):
            return item
    return None


def estimate_cost_usd(
    usage: ProviderTokenUsage,
    pricing: ModelPricing,
) -> float | None:
    """Estimate cost from reported tokens without double-charging cached input."""
    if usage.input_tokens is None or usage.output_tokens is None:
        return None
    cached = min(usage.cached_input_tokens or 0, usage.input_tokens)
    regular_input = usage.input_tokens - cached
    cached_rate = (
        pricing.cached_input_cost_per_million_tokens
        if pricing.cached_input_cost_per_million_tokens is not None
        else pricing.input_cost_per_million_tokens
    )
    cost = (
        regular_input * pricing.input_cost_per_million_tokens
        + cached * cached_rate
        + usage.output_tokens * pricing.output_cost_per_million_tokens
    ) / 1_000_000
    return round(cost, 12)
