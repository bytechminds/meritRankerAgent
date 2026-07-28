"""Focused tests for provider-reported token and estimated-cost observability."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import observability.llm_usage as usage_observability
from observability.context import bind_request_context
from observability.events import sanitize_details
from observability.llm_usage import (
    begin_llm_usage_collection,
    emit_llm_usage_summary,
    record_llm_call,
    reset_llm_usage_collection,
    snapshot_llm_usage_records,
)
from schemas.llm_usage import (
    LLMPricingConfig,
    ModelPricing,
    ProviderTokenUsage,
)
from services.llm.pricing import (
    estimate_cost_usd,
    find_model_pricing,
    load_pricing_config,
)
from services.llm.providers.usage import (
    extract_bedrock_usage,
    extract_gemini_usage,
    extract_openai_usage,
)


def test_extract_openai_usage_including_cached_and_reasoning_tokens() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            prompt_tokens_details=SimpleNamespace(cached_tokens=25),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=12),
        )
    )

    assert extract_openai_usage(response) == ProviderTokenUsage(
        input_tokens=100,
        output_tokens=40,
        total_tokens=140,
        cached_input_tokens=25,
        reasoning_tokens=12,
    )


def test_extractors_preserve_provider_reported_zero_values() -> None:
    gemini = extract_gemini_usage(
        {
            "usageMetadata": {
                "promptTokenCount": 0,
                "candidatesTokenCount": 4,
                "totalTokenCount": 4,
                "cachedContentTokenCount": 0,
                "thoughtsTokenCount": 0,
            }
        }
    )
    bedrock = extract_bedrock_usage(
        {
            "metadata": {
                "usage": {
                    "inputTokens": 7,
                    "outputTokens": 0,
                    "totalTokens": 7,
                    "cacheReadInputTokens": 0,
                }
            }
        }
    )

    assert gemini.input_tokens == 0
    assert gemini.cached_input_tokens == 0
    assert gemini.reasoning_tokens == 0
    assert bedrock.output_tokens == 0
    assert bedrock.cached_input_tokens == 0


def test_versioned_pricing_uses_cached_input_rate_without_double_counting() -> None:
    pricing = ModelPricing(
        provider="openai",
        model_or_deployment="gpt-test",
        input_cost_per_million_tokens=2.0,
        output_cost_per_million_tokens=8.0,
        cached_input_cost_per_million_tokens=0.5,
        effective_from="2026-07-27",
    )
    usage = ProviderTokenUsage(
        input_tokens=1_000,
        output_tokens=500,
        total_tokens=1_500,
        cached_input_tokens=200,
    )

    assert estimate_cost_usd(usage, pricing) == pytest.approx(0.0057)


def test_pricing_requires_exact_provider_and_model_or_deployment_match() -> None:
    config = LLMPricingConfig(
        pricing_version="test-v1",
        models=(
            ModelPricing(
                provider="azure_openai",
                model_or_deployment="deployment-a",
                input_cost_per_million_tokens=1.0,
                output_cost_per_million_tokens=2.0,
                effective_from="2026-07-27",
            ),
        ),
    )

    assert (
        find_model_pricing(
            config,
            provider="azure_openai",
            model="gpt-test",
            deployment="deployment-a",
        )
        is not None
    )
    assert (
        find_model_pricing(
            config,
            provider="openai",
            model="deployment-a",
            deployment=None,
        )
        is None
    )


@pytest.mark.parametrize(
    ("provider", "model", "input_rate", "output_rate"),
    (
        ("openai", "gpt-4.1-mini", 0.40, 1.60),
        ("openai", "gpt-4.1", 2.00, 8.00),
        ("gemini", "gemini-3.1-flash-lite", 0.25, 1.50),
    ),
)
def test_reviewed_active_model_pricing_is_loaded_exactly(
    provider: str,
    model: str,
    input_rate: float,
    output_rate: float,
) -> None:
    pricing = find_model_pricing(
        load_pricing_config(),
        provider=provider,
        model=model,
        deployment=None,
    )

    assert pricing is not None
    assert pricing.input_cost_per_million_tokens == input_rate
    assert pricing.output_cost_per_million_tokens == output_rate
    assert pricing.effective_from == "2026-07-27"


def test_unverified_azure_deployment_price_is_not_guessed() -> None:
    pricing = find_model_pricing(
        load_pricing_config(),
        provider="azure_openai",
        model="doubt_solver_classifier",
        deployment="gpt-4.1-mini",
    )

    assert pricing is None


def test_gemini_reviewed_price_calculates_reported_usage() -> None:
    pricing = find_model_pricing(
        load_pricing_config(),
        provider="gemini",
        model="gemini-3.1-flash-lite",
        deployment=None,
    )

    assert pricing is not None
    assert estimate_cost_usd(
        ProviderTokenUsage(
            input_tokens=4_511,
            output_tokens=414,
            total_tokens=4_925,
        ),
        pricing,
    ) == pytest.approx(0.00174875)


def test_request_aggregate_sums_actual_calls_and_marks_unknown_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        usage_observability,
        "load_pricing_config",
        lambda: LLMPricingConfig(pricing_version="empty"),
    )
    with bind_request_context(request_id="request-usage"):
        token = begin_llm_usage_collection()
        try:
            record_llm_call(
                request_id="request-usage",
                role="general.classifier.default",
                provider="openai",
                model="gpt-test",
                deployment=None,
                attempt_type="primary",
                streaming=False,
                usage=ProviderTokenUsage(
                    input_tokens=10,
                    output_tokens=3,
                    total_tokens=13,
                ),
                duration_ms=12,
                status="succeeded",
            )
            record_llm_call(
                request_id="request-usage",
                role="math.generator.default",
                provider="openai",
                model="gpt-test",
                deployment=None,
                attempt_type="continuation",
                streaming=True,
                usage=ProviderTokenUsage(
                    input_tokens=20,
                    output_tokens=5,
                    total_tokens=25,
                ),
                duration_ms=15,
                status="succeeded",
            )

            summary = emit_llm_usage_summary()
        finally:
            reset_llm_usage_collection(token)

    assert summary is not None
    assert summary.calls == 2
    assert summary.input_tokens == 30
    assert summary.output_tokens == 8
    assert summary.total_tokens == 38
    assert summary.estimated_cost_usd is None
    assert summary.usage_complete is True
    assert summary.cost_complete is False
    assert summary.roles["classifier"].calls == 1
    assert summary.roles["continuation"].calls == 1


def test_strong_classifier_generator_and_rewrite_are_separate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        usage_observability,
        "load_pricing_config",
        lambda: LLMPricingConfig(pricing_version="empty"),
    )
    calls = (
        ("general.classifier.default", "primary"),
        ("general.classifier.strong", "primary"),
        ("math.generator.hard", "primary"),
        ("math.generator.hard", "rewrite"),
    )
    with bind_request_context(request_id="request-multi-call"):
        token = begin_llm_usage_collection()
        try:
            for role, attempt_type in calls:
                record_llm_call(
                    request_id="request-multi-call",
                    role=role,
                    provider="azure_openai",
                    model="gpt-test",
                    deployment="deployment-test",
                    attempt_type=attempt_type,
                    streaming=False,
                    usage=ProviderTokenUsage(
                        input_tokens=10,
                        output_tokens=2,
                        total_tokens=12,
                    ),
                    duration_ms=1,
                    status="succeeded",
                )
            summary = emit_llm_usage_summary()
        finally:
            reset_llm_usage_collection(token)

    assert summary is not None
    assert summary.calls == 4
    assert summary.total_tokens == 48
    assert summary.roles["classifier"].calls == 1
    assert summary.roles["strong_classifier"].calls == 1
    assert summary.roles["generator"].calls == 1
    assert summary.roles["rewrite"].calls == 1


def test_avoided_strong_classifier_reduces_call_and_token_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        usage_observability,
        "load_pricing_config",
        lambda: LLMPricingConfig(pricing_version="empty"),
    )

    def collect(*, include_strong: bool):
        token = begin_llm_usage_collection()
        try:
            record_llm_call(
                request_id="request-classifier-cost",
                role="general.classifier.default",
                provider="azure_openai",
                model="gpt-4.1-mini",
                deployment="gpt-4.1-mini",
                attempt_type="primary",
                streaming=False,
                usage=ProviderTokenUsage(
                    input_tokens=2_450,
                    output_tokens=187,
                    total_tokens=2_637,
                ),
                duration_ms=2_544,
                status="succeeded",
            )
            if include_strong:
                record_llm_call(
                    request_id="request-classifier-cost",
                    role="general.classifier_strong.default",
                    provider="azure_openai",
                    model="gpt-4.1",
                    deployment="gpt-4.1",
                    attempt_type="primary",
                    streaming=False,
                    usage=ProviderTokenUsage(
                        input_tokens=2_438,
                        output_tokens=197,
                        total_tokens=2_635,
                    ),
                    duration_ms=1_892,
                    status="succeeded",
                )
            return emit_llm_usage_summary()
        finally:
            reset_llm_usage_collection(token)

    with bind_request_context(request_id="request-classifier-cost"):
        accepted = collect(include_strong=False)
        escalated = collect(include_strong=True)

    assert accepted is not None
    assert escalated is not None
    assert accepted.calls == 1
    assert accepted.total_tokens == 2_637
    assert escalated.calls == 2
    assert escalated.total_tokens == 5_272


def test_initial_records_transfer_between_request_collectors() -> None:
    with bind_request_context(request_id="request-transfer"):
        first = begin_llm_usage_collection()
        try:
            record_llm_call(
                request_id="request-transfer",
                role="image_question_classifier",
                provider="gemini",
                model="gemini-test",
                deployment=None,
                attempt_type="primary",
                streaming=False,
                usage=ProviderTokenUsage(
                    input_tokens=8,
                    output_tokens=2,
                    total_tokens=10,
                ),
                duration_ms=1,
                status="succeeded",
            )
            seed = snapshot_llm_usage_records()
        finally:
            reset_llm_usage_collection(first)

        second = begin_llm_usage_collection(seed)
        try:
            records = snapshot_llm_usage_records()
        finally:
            reset_llm_usage_collection(second)

    assert len(records) == 1
    assert records[0].role == "image_question_classifier"
    assert records[0].call_index == 1


def test_telemetry_failure_is_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        usage_observability,
        "_log_call",
        lambda _record: (_ for _ in ()).throw(RuntimeError("telemetry down")),
    )

    record = record_llm_call(
        request_id="request-fail-open",
        role="generator",
        provider="openai",
        model="gpt-test",
        deployment=None,
        attempt_type="primary",
        streaming=False,
        usage=ProviderTokenUsage(),
        duration_ms=1,
        status="failed",
        error_type="TimeoutError",
    )

    assert record is None


def test_usage_logging_sanitizer_keeps_counts_but_drops_content_and_endpoints() -> None:
    details = sanitize_details(
        {
            "input_tokens": 10,
            "output_tokens": 4,
            "prompt": "private prompt",
            "response_content": "private response",
            "endpoint": "https://private.example",
            "api_key": "secret",
        }
    )

    assert details == {"input_tokens": 10, "output_tokens": 4}
