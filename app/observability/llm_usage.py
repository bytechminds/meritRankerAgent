"""Fail-open request-local LLM token and estimated-cost observability."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from threading import Lock

from observability.context import current_request_context
from observability.events import log_event
from schemas.llm_usage import (
    LLMRoleUsageSummary,
    LLMUsageRecord,
    LLMUsageSummary,
    ProviderTokenUsage,
    UsageStatus,
)
from services.llm.pricing import (
    estimate_cost_usd,
    find_model_pricing,
    load_pricing_config,
)

logger = logging.getLogger(__name__)


@dataclass
class _UsageCollector:
    records: list[LLMUsageRecord] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    summary_emitted: bool = False


_collector: ContextVar[_UsageCollector | None] = ContextVar(
    "request_llm_usage_collector", default=None
)
_attempt_type: ContextVar[str] = ContextVar("llm_attempt_type", default="primary")


def begin_llm_usage_collection(
    initial_records: Sequence[LLMUsageRecord] = (),
) -> Token[_UsageCollector | None]:
    return _collector.set(_UsageCollector(records=list(initial_records)))


def reset_llm_usage_collection(token: Token[_UsageCollector | None]) -> None:
    _collector.reset(token)


def snapshot_llm_usage_records() -> tuple[LLMUsageRecord, ...]:
    current = _collector.get()
    if current is None:
        return ()
    with current.lock:
        return tuple(current.records)


def count_generator_calls() -> int:
    """Return request-local generator attempts, including provider fallbacks."""
    return sum(
        1
        for record in snapshot_llm_usage_records()
        if ".generator." in record.role or record.role.endswith(".generator")
    )


@contextmanager
def bind_llm_attempt_type(attempt_type: str) -> Iterator[None]:
    token = _attempt_type.set(attempt_type[:64] or "primary")
    try:
        yield
    finally:
        _attempt_type.reset(token)


def current_llm_attempt_type() -> str:
    return _attempt_type.get()


def _pricing(
    *,
    provider: str,
    model: str,
    deployment: str | None,
    usage: ProviderTokenUsage,
) -> tuple[float | None, str | None, str]:
    try:
        config = load_pricing_config()
        pricing = find_model_pricing(
            config,
            provider=provider,
            model=model,
            deployment=deployment,
        )
        if pricing is None:
            return None, config.pricing_version, "unavailable"
        cost = estimate_cost_usd(usage, pricing)
        if cost is None:
            return None, config.pricing_version, "unavailable"
        return cost, config.pricing_version, "available"
    except Exception:  # noqa: BLE001
        return None, None, "unavailable"


def record_llm_call(
    *,
    request_id: str,
    role: str,
    provider: str,
    model: str,
    deployment: str | None,
    attempt_type: str,
    streaming: bool,
    usage: ProviderTokenUsage,
    duration_ms: int,
    status: UsageStatus,
    error_type: str | None = None,
) -> LLMUsageRecord | None:
    """Record and log one call without allowing telemetry to affect execution."""
    try:
        context = current_request_context()
        cost, pricing_version, pricing_status = _pricing(
            provider=provider,
            model=model,
            deployment=deployment,
            usage=usage,
        )
        usage_source = "provider_reported" if usage.available else "unavailable"
        record = LLMUsageRecord(
            request_id=request_id,
            trace_id=context.trace_id if context else None,
            turn_id=context.turn_id if context else None,
            role=role,
            provider=provider,
            model=model,
            deployment=deployment,
            attempt_type=attempt_type,
            streaming=streaming,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            usage_source=usage_source,
            estimated_cost_usd=cost,
            pricing_version=pricing_version,
            pricing_status=pricing_status,
            duration_ms=max(duration_ms, 0),
            status=status,
            error_type=error_type,
        )
        current = _collector.get()
        if current is not None:
            with current.lock:
                record = record.model_copy(update={"call_index": len(current.records) + 1})
                current.records.append(record)
        _log_call(record)
        return record
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "llm_usage_unavailable reason=recording_failed error_type=%s",
            type(exc).__name__,
        )
        return None


def _log_call(record: LLMUsageRecord) -> None:
    modality = "image" if "image" in record.role.casefold() else "text"
    token_text = (
        f"in:{record.input_tokens} out:{record.output_tokens} total:{record.total_tokens}"
        if record.usage_source != "unavailable"
        else "unavailable"
    )
    cost_text = (
        f"{record.estimated_cost_usd:.12f}".rstrip("0").rstrip(".")
        if record.estimated_cost_usd is not None
        else "unavailable"
    )
    logger.info(
        "LLM CALL role=%s provider=%s model=%s modality=%s deployment=%s attempt=%s "
        "streaming=%s tokens=%s usage_source=%s cost_usd=%s duration=%dms status=%s",
        record.role,
        record.provider,
        record.model,
        modality,
        record.deployment or "none",
        record.attempt_type,
        str(record.streaming).lower(),
        token_text,
        record.usage_source,
        cost_text,
        record.duration_ms,
        record.status,
    )
    log_event(
        "llm_call_usage",
        component="llm.usage",
        stage=record.role,
        status=record.status,
        duration_ms=record.duration_ms,
        error_code=record.error_type,
        details={
            "role": record.role,
            "provider": record.provider,
            "model": record.model,
            "modality": modality,
            "deployment": record.deployment,
            "call_index": record.call_index,
            "attempt_type": record.attempt_type,
            "streaming": record.streaming,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "total_tokens": record.total_tokens,
            "cached_input_tokens": record.cached_input_tokens,
            "reasoning_tokens": record.reasoning_tokens,
            "usage_source": record.usage_source,
            "estimated_cost_usd": record.estimated_cost_usd,
            "pricing_version": record.pricing_version,
            "pricing_status": record.pricing_status,
            "error_type": record.error_type,
        },
    )


def _role_group(record: LLMUsageRecord) -> str:
    role = record.role.lower()
    attempt = record.attempt_type.lower()
    if "rewrite" in attempt:
        return "rewrite"
    if "repair" in attempt:
        return "repair"
    if "continuation" in attempt:
        return "continuation"
    if "image" in role and "classifier" in role:
        return "image_classifier"
    if (
        "classifier_strong" in role
        or "strong_classifier" in role
        or ("classifier" in role and "strong" in role)
    ):
        return "strong_classifier"
    if "classifier" in role:
        return "classifier"
    if "planner" in role:
        return "planner"
    if "verifier" in role or "judge" in role:
        return "verifier"
    if "embedding" in role:
        return "query_embedding"
    return "generator"


def _cost_text(cost: float | None) -> str:
    if cost is None:
        return "unavailable"
    return f"{cost:.12f}".rstrip("0").rstrip(".")


def _summarize_records(records: Sequence[LLMUsageRecord]) -> LLMUsageSummary:
    request_id = records[0].request_id if records else (
        current_request_context().request_id if current_request_context() else "unknown"
    )
    role_records: dict[str, list[LLMUsageRecord]] = {}
    for record in records:
        role_records.setdefault(_role_group(record), []).append(record)

    def summarize(items: Sequence[LLMUsageRecord]) -> LLMRoleUsageSummary:
        usage_complete = all(item.usage_source != "unavailable" for item in items)
        cost_complete = all(item.estimated_cost_usd is not None for item in items)
        return LLMRoleUsageSummary(
            calls=len(items),
            input_tokens=sum(item.input_tokens or 0 for item in items),
            output_tokens=sum(item.output_tokens or 0 for item in items),
            total_tokens=sum(item.total_tokens or 0 for item in items),
            estimated_cost_usd=(
                round(sum(item.estimated_cost_usd or 0.0 for item in items), 12)
                if cost_complete
                else None
            ),
            usage_complete=usage_complete,
            cost_complete=cost_complete,
        )

    roles = {name: summarize(items) for name, items in role_records.items()}
    usage_complete = all(record.usage_source != "unavailable" for record in records)
    cost_complete = all(record.estimated_cost_usd is not None for record in records)
    return LLMUsageSummary(
        request_id=request_id,
        calls=len(records),
        input_tokens=sum(record.input_tokens or 0 for record in records),
        output_tokens=sum(record.output_tokens or 0 for record in records),
        total_tokens=sum(record.total_tokens or 0 for record in records),
        estimated_cost_usd=(
            round(sum(record.estimated_cost_usd or 0.0 for record in records), 12)
            if cost_complete
            else None
        ),
        duration_ms=sum(record.duration_ms for record in records),
        usage_complete=usage_complete,
        cost_complete=cost_complete,
        roles=roles,
    )


def emit_llm_usage_summary() -> LLMUsageSummary | None:
    current = _collector.get()
    if current is None:
        return None
    with current.lock:
        if current.summary_emitted:
            return None
        current.summary_emitted = True
        records = tuple(current.records)
    try:
        summary = _summarize_records(records)
        cost_text = _cost_text(summary.estimated_cost_usd)
        role_breakdown = ";".join(
            (
                f"{role}:calls={usage.calls},tokens={usage.total_tokens},"
                f"cost={_cost_text(usage.estimated_cost_usd)}"
            )
            for role, usage in sorted(summary.roles.items())
        ) or "none"
        generator_calls = sum(
            1
            for record in records
            if ".generator." in record.role or record.role.endswith(".generator")
        )
        rewrite_count = sum(
            1
            for record in records
            if record.attempt_type in {"rewrite", "repair", "correctness_repair"}
        )
        verification_calls = sum(
            1
            for record in records
            if ".verifier." in record.role or record.role.endswith(".verifier")
        )
        logger.info(
            "LLM USAGE SUMMARY calls=%d input_tokens=%d output_tokens=%d "
            "total_tokens=%d estimated_cost_usd=%s duration_ms=%d "
            "usage_complete=%s cost_complete=%s generator_calls=%d "
            "rewrite_count=%d verification_calls=%d roles=%s",
            summary.calls,
            summary.input_tokens,
            summary.output_tokens,
            summary.total_tokens,
            cost_text,
            summary.duration_ms,
            str(summary.usage_complete).lower(),
            str(summary.cost_complete).lower(),
            generator_calls,
            rewrite_count,
            verification_calls,
            role_breakdown,
        )
        log_event(
            "llm_usage_summary",
            component="llm.usage",
            stage="complete",
            status="completed",
            duration_ms=summary.duration_ms,
            details={
                "calls": summary.calls,
                "input_tokens": summary.input_tokens,
                "output_tokens": summary.output_tokens,
                "total_tokens": summary.total_tokens,
                "estimated_cost_usd": summary.estimated_cost_usd,
                "usage_complete": summary.usage_complete,
                "cost_complete": summary.cost_complete,
                "generator_calls": generator_calls,
                "rewrite_count": rewrite_count,
                "verification_calls": verification_calls,
                "role_breakdown": role_breakdown,
            },
        )
        return summary
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "llm_usage_unavailable reason=summary_failed error_type=%s",
            type(exc).__name__,
        )
        return None


def monotonic_ms(started_at: float) -> int:
    return max(int((time.monotonic() - started_at) * 1000), 0)
