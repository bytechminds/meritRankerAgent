"""Shadow AI usage billing checks for central provider-attempt telemetry."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from decimal import Decimal

import pytest

from observability.context import bind_execution_context, current_execution_context
from observability.llm_usage import record_llm_call
from schemas.billing import BillingConfig
from schemas.llm_usage import LLMUsageRecord, ProviderTokenUsage
from services.llm.billing import (
    OperationDescriptor,
    OperationUsageAccumulator,
    begin_operation,
    calculate_operation_billing,
    emit_operation_billing_summary,
)


def _config() -> BillingConfig:
    return BillingConfig.model_validate(
        {
            "billing_version": "billing-test-v1",
            "pricing_version": "pricing-test-v1",
            "metering_enabled": True,
            "credit_debit_enabled": False,
            "models": [
                {
                    "provider": "openai",
                    "model_or_deployment": "gpt-test",
                    "input_cost_per_million_tokens": "2",
                    "output_cost_per_million_tokens": "8",
                    "cached_input_cost_per_million_tokens": "0.5",
                    "effective_from": "2026-08-12",
                }
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
    provider: str = "openai",
    model: str = "gpt-test",
    deployment: str | None = None,
    input_tokens: int | None = 1_000_000,
    output_tokens: int | None = 1_000_000,
    cached_input_tokens: int | None = None,
) -> LLMUsageRecord:
    return LLMUsageRecord(
        request_id="request-billing",
        role="general.generator",
        provider=provider,
        model=model,
        deployment=deployment,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=(
            (input_tokens or 0) + (output_tokens or 0)
            if input_tokens is not None or output_tokens is not None
            else None
        ),
        cached_input_tokens=cached_input_tokens,
        usage_source="provider_reported"
        if input_tokens is not None or output_tokens is not None
        else "unavailable",
        status="succeeded",
    )


def test_billing_uses_decimal_rates_and_ceiling_credit_conversion() -> None:
    summary = calculate_operation_billing(
        descriptor=OperationDescriptor(operation_id="operation-1", feature="doubt"),
        records=(
            _record(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                cached_input_tokens=200_000,
            ),
        ),
        operation_status="completed",
        config=_config(),
    )

    assert summary.total_llm_cost_usd == Decimal("10")
    assert summary.actual_usage_cost_usd == Decimal("10.001")
    assert summary.calculated_credits == 20_002
    assert summary.credit_debit_enabled is False
    assert summary.credits_debited == 0
    assert summary.cost_complete is True


def test_unknown_provider_rate_is_incomplete_and_never_guessed() -> None:
    summary = calculate_operation_billing(
        descriptor=OperationDescriptor(operation_id="operation-2", feature="doubt"),
        records=(
            _record(
                provider="azure_openai",
                model="gpt-4.1-mini",
                deployment="gpt-4.1-mini",
            ),
        ),
        operation_status="completed",
        config=_config(),
    )

    assert summary.cost_complete is False
    assert summary.total_llm_cost_usd is None
    assert summary.actual_usage_cost_usd is None
    assert summary.calculated_credits is None
    assert summary.missing_cost_profiles == ("azure_openai:gpt-4.1-mini",)


def test_missing_provider_usage_is_incomplete_without_synthetic_tokens() -> None:
    summary = calculate_operation_billing(
        descriptor=OperationDescriptor(operation_id="operation-3", feature="doubt"),
        records=(_record(input_tokens=None, output_tokens=None),),
        operation_status="failed",
        config=_config(),
    )

    assert summary.total_input_tokens == 0
    assert summary.total_output_tokens == 0
    assert summary.missing_usage_call_count == 1
    assert summary.actual_usage_cost_usd is None


def test_execution_context_inherits_operation_accumulator_for_nested_work() -> None:
    accumulator = OperationUsageAccumulator(
        descriptor=OperationDescriptor(operation_id="operation-4", feature="mock_test")
    )

    with bind_execution_context(
        operation_id="operation-4",
        feature="mock_test",
        operation_accumulator=accumulator,
    ):
        with bind_execution_context(activity_id="test-4", batch_id="batch-4"):
            context = current_execution_context()

    assert context is not None
    assert context.operation_id == "operation-4"
    assert context.feature == "mock_test"
    assert context.operation_accumulator is accumulator


def test_copied_execution_context_keeps_operation_identity_in_worker_thread() -> None:
    accumulator = OperationUsageAccumulator(
        descriptor=OperationDescriptor(operation_id="operation-4b", feature="mock_test")
    )
    with bind_execution_context(
        operation_id="operation-4b",
        feature="mock_test",
        operation_accumulator=accumulator,
    ):
        worker_context = copy_context()
        with ThreadPoolExecutor(max_workers=1) as executor:
            observed = executor.submit(
                worker_context.run,
                current_execution_context,
            ).result()

    assert observed is not None
    assert observed.operation_id == "operation-4b"
    assert observed.operation_accumulator is accumulator


def test_shadow_config_rejects_credit_debit_enablement() -> None:
    payload = _config().model_dump(mode="json")
    payload["credit_debit_enabled"] = True

    with pytest.raises(ValueError, match="credit_debit_enabled"):
        BillingConfig.model_validate(payload)


def test_shadow_config_requires_each_feature_allowance() -> None:
    payload = _config().model_dump(mode="json")
    del payload["infra"]["doubt"]

    with pytest.raises(ValueError, match="infra must configure"):
        BillingConfig.model_validate(payload)


def test_metering_disabled_does_not_create_an_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = _config().model_copy(update={"metering_enabled": False})
    monkeypatch.setattr("services.llm.billing.load_billing_config", lambda: disabled)

    assert begin_operation(operation_id="operation-disabled", feature="doubt") is None


def test_reasoning_tokens_are_not_charged_twice() -> None:
    record = _record(input_tokens=1_000_000, output_tokens=4_000)
    record = record.model_copy(update={"reasoning_tokens": 2_500})

    summary = calculate_operation_billing(
        descriptor=OperationDescriptor(operation_id="operation-reasoning", feature="doubt"),
        records=(record,),
        operation_status="completed",
        config=_config(),
    )

    assert summary.total_llm_cost_usd == Decimal("2.032")


def test_handed_off_practice_operation_finalizes_with_durable_test_identity() -> None:
    accumulator = OperationUsageAccumulator(
        descriptor=OperationDescriptor(operation_id="request-6", feature="doubt"),
        records=[_record(input_tokens=10, output_tokens=2)],
    )
    accumulator.hand_off(operation_id="test-6", feature="mock_test")

    assert accumulator.snapshot_for_finalization() is None

    accumulator.clear_handoff()
    snapshot = accumulator.snapshot_for_finalization()

    assert snapshot is not None
    descriptor, records = snapshot
    assert descriptor == OperationDescriptor(operation_id="test-6", feature="mock_test")
    assert len(records) == 1


def test_central_llm_hook_emits_one_shadow_summary_without_debit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accumulator = OperationUsageAccumulator(
        descriptor=OperationDescriptor(operation_id="operation-5", feature="doubt")
    )
    events: list[tuple[str, dict[str, object] | None]] = []
    monkeypatch.setattr(
        "services.llm.billing.load_billing_config",
        _config,
    )
    monkeypatch.setattr(
        "services.llm.billing.log_event",
        lambda event_name, **kwargs: events.append((event_name, kwargs.get("details"))),
    )

    with bind_execution_context(
        operation_id="operation-5",
        feature="doubt",
        operation_accumulator=accumulator,
    ):
        record_llm_call(
            request_id="request-billing",
            role="general.generator",
            provider="openai",
            model="gpt-test",
            deployment=None,
            attempt_type="fallback",
            streaming=False,
            usage=ProviderTokenUsage(input_tokens=1_000, output_tokens=500),
            duration_ms=10,
            status="succeeded",
        )
        record_llm_call(
            request_id="request-billing",
            role="general.generator",
            provider="openai",
            model="gpt-test",
            deployment=None,
            attempt_type="retry_fallback",
            streaming=False,
            usage=ProviderTokenUsage(input_tokens=100, output_tokens=50),
            duration_ms=11,
            status="failed",
            error_type="TimeoutError",
        )
        first = emit_operation_billing_summary(operation_status="completed")
        second = emit_operation_billing_summary(operation_status="completed")

    assert first is not None
    assert second is None
    assert first.llm_call_count == 2
    assert first.credit_debit_enabled is False
    assert first.credits_debited == 0
    assert [event_name for event_name, _details in events] == ["AI_USAGE_SUMMARY"]
    assert events[0][1] is not None
    assert "prompt" not in events[0][1]
    assert "response" not in events[0][1]
