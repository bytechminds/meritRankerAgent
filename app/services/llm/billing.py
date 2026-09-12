"""Low-latency, fail-open operation billing for provider-reported LLM usage."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal
from functools import lru_cache
from pathlib import Path
from threading import Lock

import yaml

from observability.context import current_execution_context
from observability.events import log_event
from schemas.billing import BillingConfig, BillingModelRate, OperationBillingSummary
from schemas.llm_usage import LLMUsageRecord

logger = logging.getLogger(__name__)

DEFAULT_BILLING_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "llm" / "model_pricing.yaml"
)
_MILLION = Decimal("1000000")


class BillingConfigurationError(RuntimeError):
    """Raised when the versioned billing YAML is structurally invalid."""


@lru_cache(maxsize=4)
def load_billing_config(path: Path = DEFAULT_BILLING_CONFIG_PATH) -> BillingConfig:
    """Load and validate immutable V1 settings once per process/path."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return BillingConfig.model_validate(payload)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise BillingConfigurationError(f"Invalid billing configuration at {path}.") from exc


def validate_billing_configuration() -> BillingConfig:
    """Startup hook; callers should fail fast before accepting traffic."""
    return load_billing_config()


@dataclass(frozen=True)
class OperationDescriptor:
    operation_id: str
    feature: str


@dataclass
class OperationUsageAccumulator:
    """Thread-safe, request-lifecycle-bound storage; never a user-keyed global cache."""

    descriptor: OperationDescriptor
    records: list[LLMUsageRecord] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    handed_off: bool = False
    summary_emitted: bool = False
    # The subset of `records` that produced the delivered result. `records` stays the
    # truthful provider ledger — a failed call keeps its unknown usage there — while
    # the student is charged only for work that survived to the accepted output.
    chargeable_records: list[LLMUsageRecord] = field(default_factory=list)
    chargeable_seeded: bool = False

    def append(self, record: LLMUsageRecord) -> None:
        with self.lock:
            if not self.summary_emitted:
                self.records.append(record)

    def commit_chargeable(self, records: tuple[LLMUsageRecord, ...]) -> None:
        """Mark already-recorded calls as part of the accepted, billable path.

        Only succeeded calls are admitted: a failed attempt has no usable output,
        so it contributes nothing to the student charge regardless of what it cost.
        """
        with self.lock:
            self.chargeable_records.extend(
                record for record in records if record.status == "succeeded"
            )

    def seed_inherited_chargeable(self) -> None:
        """Admit the accepted usage the launching request already paid for, once.

        The request that launched this operation hands its accumulator over, so its
        classifier call is already present here and needs no second transport.
        """
        with self.lock:
            if self.chargeable_seeded:
                return
            self.chargeable_seeded = True
            self.chargeable_records.extend(
                record for record in self.records if record.status == "succeeded"
            )

    def snapshot_chargeable(
        self,
    ) -> tuple[OperationDescriptor, tuple[LLMUsageRecord, ...]]:
        with self.lock:
            return self.descriptor, tuple(self.chargeable_records)

    def hand_off(self, *, operation_id: str, feature: str) -> None:
        with self.lock:
            if self.summary_emitted:
                return
            self.descriptor = OperationDescriptor(operation_id=operation_id, feature=feature)
            self.handed_off = True

    def absorb(self, other: OperationUsageAccumulator) -> bool:
        if other is self:
            return True
        first, second = (self, other) if id(self) < id(other) else (other, self)
        with first.lock:
            with second.lock:
                if self.summary_emitted:
                    return False
                self.records.extend(other.records)
                self.chargeable_records.extend(other.chargeable_records)
                other.handed_off = True
                return True

    def snapshot_for_finalization(
        self,
    ) -> tuple[OperationDescriptor, tuple[LLMUsageRecord, ...]] | None:
        with self.lock:
            if self.summary_emitted or self.handed_off:
                return None
            self.summary_emitted = True
            return self.descriptor, tuple(self.records)

    def snapshot_records(
        self,
    ) -> tuple[OperationDescriptor, tuple[LLMUsageRecord, ...]]:
        """Read the operation so far without consuming the terminal summary.

        Settlement needs the operation total before the terminal event, while
        ``snapshot_for_finalization`` stays the single one-shot telemetry path.
        """
        with self.lock:
            return self.descriptor, tuple(self.records)

    def clear_handoff(self) -> None:
        with self.lock:
            self.handed_off = False


def begin_operation(
    *, operation_id: str, feature: str
) -> OperationUsageAccumulator | None:
    config = load_billing_config()
    if not config.metering_enabled:
        return None
    return OperationUsageAccumulator(
        descriptor=OperationDescriptor(operation_id=operation_id, feature=feature)
    )


def current_operation_accumulator() -> OperationUsageAccumulator | None:
    context = current_execution_context()
    if context is None:
        return None
    accumulator = context.operation_accumulator
    return accumulator if isinstance(accumulator, OperationUsageAccumulator) else None


def record_usage_for_current_operation(record: LLMUsageRecord) -> None:
    """Append an existing safe usage record without affecting provider execution."""
    try:
        accumulator = current_operation_accumulator()
        if accumulator is not None:
            accumulator.append(record)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ai_usage_meter_unavailable reason=recording_failed error_type=%s",
            type(exc).__name__,
        )


def hand_off_current_operation(
    *, operation_id: str, feature: str
) -> OperationUsageAccumulator | None:
    accumulator = current_operation_accumulator()
    if accumulator is not None:
        accumulator.hand_off(operation_id=operation_id, feature=feature)
    return accumulator


def feature_for_practice_type(practice_type: str | None) -> str:
    normalized = (practice_type or "").strip().upper()
    if normalized in {"SECTIONAL_TEST", "FULL_MOCK", "TOPIC_TEST", "CURRENT_AFFAIRS_SET"}:
        return "mock_test"
    return "quick_practice"


def _find_rate(
    config: BillingConfig,
    *,
    provider: str,
    model: str,
    deployment: str | None,
) -> BillingModelRate | None:
    candidates = {
        value.strip().casefold()
        for value in (model, deployment)
        if isinstance(value, str) and value.strip()
    }
    provider_key = provider.strip().casefold()
    for rate in config.models:
        if (
            rate.provider.casefold() == provider_key
            and rate.model_or_deployment.casefold() in candidates
        ):
            return rate
    return None


def _profile_identity(record: LLMUsageRecord) -> str:
    model = record.deployment or record.model
    return f"{record.provider}:{model}"


def calculate_operation_billing(
    *,
    descriptor: OperationDescriptor,
    records: tuple[LLMUsageRecord, ...],
    operation_status: str,
    config: BillingConfig | None = None,
) -> OperationBillingSummary:
    """Calculate one operation total from actual provider counters only."""
    active_config = config or load_billing_config()
    infra = active_config.infra[descriptor.feature].fixed_cost
    total_input = sum(record.input_tokens or 0 for record in records)
    total_output = sum(record.output_tokens or 0 for record in records)
    total_llm = Decimal("0")
    missing_profiles: set[str] = set()
    missing_cached_rates: set[str] = set()
    missing_usage = 0
    invalid_usage = 0

    for record in records:
        if record.input_tokens is None or record.output_tokens is None:
            missing_usage += 1
            continue
        rate = _find_rate(
            active_config,
            provider=record.provider,
            model=record.model,
            deployment=record.deployment,
        )
        if rate is None:
            missing_profiles.add(_profile_identity(record))
            continue
        cached = record.cached_input_tokens or 0
        if cached > record.input_tokens:
            # Invalid provider usage. Authoritative billing never clamps it:
            # an unusable counter makes the operation incomplete instead.
            invalid_usage += 1
            continue
        if cached > 0 and rate.cached_input_cost_per_million_tokens is None:
            # Cached tokens with no verified cached rate. Charging them at the
            # normal input rate would knowingly overstate provider cost.
            missing_cached_rates.add(_profile_identity(record))
            continue
        uncached = record.input_tokens - cached
        cached_rate = rate.cached_input_cost_per_million_tokens or Decimal("0")
        total_llm += (
            Decimal(uncached) * rate.input_cost_per_million_tokens
            + Decimal(cached) * cached_rate
            + Decimal(record.output_tokens) * rate.output_cost_per_million_tokens
        ) / _MILLION

    complete = (
        not missing_profiles
        and not missing_cached_rates
        and missing_usage == 0
        and invalid_usage == 0
    )
    actual_usage = total_llm + infra if complete else None
    calculated_credits = (
        int(
            ((actual_usage * active_config.credits.pricing_factor)
            / active_config.credits.usd_per_credit).to_integral_value(rounding=ROUND_CEILING)
        )
        if actual_usage is not None
        else None
    )
    return OperationBillingSummary(
        operation_id=descriptor.operation_id,
        feature=descriptor.feature,
        operation_status=operation_status,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        llm_call_count=len(records),
        total_llm_cost_usd=total_llm if complete else None,
        infra_cost_usd=infra,
        actual_usage_cost_usd=actual_usage,
        pricing_factor=active_config.credits.pricing_factor,
        usd_per_credit=active_config.credits.usd_per_credit,
        calculated_credits=calculated_credits,
        credit_debit_enabled=active_config.credit_debit_enabled,
        credits_debited=0,
        billing_config_version=active_config.billing_version,
        cost_complete=complete,
        missing_cost_profiles=tuple(sorted(missing_profiles)),
        missing_usage_call_count=missing_usage,
        missing_cached_rate_profiles=tuple(sorted(missing_cached_rates)),
        invalid_usage_call_count=invalid_usage,
    )


def snapshot_operation_billing(
    *, operation_status: str
) -> OperationBillingSummary | None:
    """Return the current operation total without emitting the terminal summary.

    Returns None when metering is inactive for this operation. Never raises:
    an unavailable meter must not change a student operation.
    """
    try:
        accumulator = current_operation_accumulator()
        if accumulator is None:
            return None
        descriptor, records = accumulator.snapshot_records()
        return calculate_operation_billing(
            descriptor=descriptor,
            records=records,
            operation_status=operation_status,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ai_usage_meter_unavailable reason=snapshot_failed error_type=%s",
            type(exc).__name__,
        )
        return None


@contextmanager
def capture_llm_usage() -> Iterator[list[LLMUsageRecord]]:
    """Isolate one call's usage records so the caller can attribute them.

    Slot groups and verifications run concurrently against one operation
    accumulator, which therefore cannot say which record belongs to which
    question. Binding a child accumulator for the duration of a single call keeps
    that attribution exact without any shared mutable billing state, and every
    captured record is forwarded to the operation accumulator afterwards so the
    provider usage summary stays exactly as it was.
    """
    from observability.context import bind_execution_context  # noqa: PLC0415

    parent = current_operation_accumulator()
    if parent is None:
        yield []
        return
    child = OperationUsageAccumulator(descriptor=parent.descriptor)
    captured: list[LLMUsageRecord] = []
    try:
        with bind_execution_context(operation_accumulator=child):
            yield captured
    finally:
        _, records = child.snapshot_records()
        captured.extend(records)
        for record in records:
            parent.append(record)


def chargeable_operation_billing(
    *, operation_status: str
) -> OperationBillingSummary | None:
    """Price only the calls that produced the delivered result.

    Same authoritative calculation as the provider total; the single difference is
    the record set, so student charging can never drift from provider pricing.
    """
    try:
        accumulator = current_operation_accumulator()
        if accumulator is None:
            return None
        descriptor, records = accumulator.snapshot_chargeable()
        return calculate_operation_billing(
            descriptor=descriptor,
            records=records,
            operation_status=operation_status,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ai_usage_meter_unavailable reason=chargeable_snapshot_failed error_type=%s",
            type(exc).__name__,
        )
        return None


def commit_chargeable_usage(records: tuple[LLMUsageRecord, ...]) -> None:
    """Attach accepted-path records to the current operation, if metering is on."""
    accumulator = current_operation_accumulator()
    if accumulator is not None:
        accumulator.commit_chargeable(records)


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value.normalize(), "f")


def emit_operation_billing_summary(
    *, operation_status: str
) -> OperationBillingSummary | None:
    """Finalize once. Accounting telemetry never changes a student operation."""
    try:
        accumulator = current_operation_accumulator()
        if accumulator is None:
            return None
        snapshot = accumulator.snapshot_for_finalization()
        if snapshot is None:
            return None
        descriptor, records = snapshot
        summary = calculate_operation_billing(
            descriptor=descriptor,
            records=records,
            operation_status=operation_status,
        )
        details = {
            "operation_id": summary.operation_id,
            "feature": summary.feature,
            "operation_status": summary.operation_status,
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "llm_call_count": summary.llm_call_count,
            "total_llm_cost_usd": _decimal_text(summary.total_llm_cost_usd),
            "infra_cost_usd": _decimal_text(summary.infra_cost_usd),
            "actual_usage_cost_usd": _decimal_text(summary.actual_usage_cost_usd),
            "pricing_factor": _decimal_text(summary.pricing_factor),
            "usd_per_credit": _decimal_text(summary.usd_per_credit),
            "calculated_credits": summary.calculated_credits,
            "credit_debit_enabled": summary.credit_debit_enabled,
            "credits_debited": summary.credits_debited,
            "billing_config_version": summary.billing_config_version,
            "cost_complete": summary.cost_complete,
            "missing_usage_call_count": summary.missing_usage_call_count,
            "missing_cost_profiles": ",".join(summary.missing_cost_profiles),
            "missing_cached_rate_profiles": ",".join(
                summary.missing_cached_rate_profiles
            ),
            "invalid_usage_call_count": summary.invalid_usage_call_count,
        }
        logger.info(
            "AI_USAGE_SUMMARY operation_id=%s feature=%s status=%s calls=%d "
            "input_tokens=%d output_tokens=%d total_llm_cost_usd=%s infra_cost_usd=%s "
            "actual_usage_cost_usd=%s calculated_credits=%s credit_debit_enabled=%s "
            "credits_debited=0 billing_config_version=%s cost_complete=%s",
            summary.operation_id,
            summary.feature,
            summary.operation_status,
            summary.llm_call_count,
            summary.total_input_tokens,
            summary.total_output_tokens,
            details["total_llm_cost_usd"] or "incomplete",
            details["infra_cost_usd"],
            details["actual_usage_cost_usd"] or "incomplete",
            summary.calculated_credits if summary.calculated_credits is not None else "incomplete",
            str(summary.credit_debit_enabled).lower(),
            summary.billing_config_version,
            str(summary.cost_complete).lower(),
        )
        log_event(
            "AI_USAGE_SUMMARY",
            component="llm.billing",
            stage="complete",
            status=operation_status.casefold(),
            details=details,
        )
        if summary.missing_cached_rate_profiles:
            log_event(
                "cached_rate_missing",
                component="llm.billing",
                stage="complete",
                status="incomplete",
                error_code="CACHED_RATE_MISSING",
                details={
                    "operation_id": summary.operation_id,
                    "feature": summary.feature,
                    "missing_cached_rate_profiles": ",".join(
                        summary.missing_cached_rate_profiles
                    ),
                    "billing_config_version": summary.billing_config_version,
                },
                level=logging.ERROR,
            )
        if summary.invalid_usage_call_count:
            log_event(
                "invalid_provider_usage",
                component="llm.billing",
                stage="complete",
                status="incomplete",
                error_code="INVALID_PROVIDER_USAGE",
                details={
                    "operation_id": summary.operation_id,
                    "feature": summary.feature,
                    "invalid_usage_call_count": summary.invalid_usage_call_count,
                    "billing_config_version": summary.billing_config_version,
                },
                level=logging.ERROR,
            )
        if summary.missing_cost_profiles:
            log_event(
                "COST_PROFILE_MISSING",
                component="llm.billing",
                stage="complete",
                status="incomplete",
                details={
                    "operation_id": summary.operation_id,
                    "feature": summary.feature,
                    "missing_cost_profiles": ",".join(summary.missing_cost_profiles),
                    "billing_config_version": summary.billing_config_version,
                },
                level=logging.WARNING,
            )
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "ai_usage_meter_unavailable reason=finalization_failed error_type=%s",
            type(exc).__name__,
        )
        return None
