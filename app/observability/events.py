"""Structured event contract and bounded metadata sanitization."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from observability.context import current_request_context
from observability.readable_log import collect_event

SERVICE_NAME = "meritranker-tutor"

EVENT_NAMES = frozenset(
    {
        "runtime_started",
        "request_started",
        "request_execution_summary",
        "request_completed",
        "request_failed",
        "request_cancelled",
        "classification_completed",
        "classification_fallback_used",
        "classifier_primary_decision",
        "model_execution_completed",
        "llm_call_usage",
        "llm_usage_summary",
        "web_search_decision",
        "web_search_execution",
        "grounding_completed",
        "context_gate_completed",
        "conversation_candidates_prepared",
        "conversation_reference_analyzed",
        "conversation_grounded_entity_extracted",
        "conversation_candidate_compatibility",
        "conversation_entity_grouped",
        "conversation_clarification_labels",
        "conversation_relation_completed",
        "conversation_context_selected",
        "selected_generation_context_built",
        "follow_up_detected",
        "follow_up_context_loaded",
        "follow_up_context_failed",
        "follow_up_resolution_completed",
        "follow_up_resolution_failed",
        "retrieval_completed",
        "retrieval_fallback_used",
        "retrieval_failed",
        "generation_completed",
        "answer_delivery_completed",
        "generation_failed",
        "quality_decision",
        "quality_validation_completed",
        "quality_rewrite_completed",
        "quality_repair_completed",
        "correctness_verification_completed",
        "conversation_persistence_completed",
        "conversation_persistence_skipped",
        "conversation_persistence_failed",
        "PRACTICE_REQUEST_RESOLVED",
        "PRACTICE_LAUNCH_DECISION",
        "PRACTICE_DISPATCH_RESULT",
        "PRACTICE_RUNTIME_VALIDATED",
        "PRACTICE_RESOURCE_CONTRACT_LOADED",
        "PRACTICE_RESOURCE_VALIDATION_FAILED",
        "practice_request_classified",
        "practice_assessment_initialized",
        "practice_async_task_registered",
        "practice_graph_started",
        "practice_async_task_failed",
        "practice_reuse_completed",
        "practice_generation_progress",
        "practice_validation_completed",
        "practice_answer_distribution_validated",
        "practice_answer_distribution_rebalanced",
        "practice_manifest_updated",
        "practice_ready",
        "practice_failed",
        "practice_model_execution_failed",
        "practice_model_fallback_started",
        "practice_model_fallback_succeeded",
        "practice_generator_token_exhausted",
        "generator_model_attempt_started",
        "generator_model_attempt_failed",
        "generator_fallback_selected",
        "generator_fallback_started",
        "generator_fallback_succeeded",
        "generator_fallback_failed",
        "generator_fallback_exhausted",
        "planner_validation_failed",
        "planner_repair_started",
        "planner_repair_completed",
        "planner_repair_failed",
        "practice_provider_replacement_scheduled",
        "practice_validation_repair_started",
        "practice_replacement_started",
        "practice_ready_published",
        "practice_failed_published",
        "practice_generation_terminal",
        "practice_async_task_completed",
        "question_contract_validation",
        "question_contract_rejected",
        "question_contract_accepted",
        "question_repair_requested",
        "question_fallback_requested",
        "final_manifest_validation_started",
        "final_manifest_validation_failed",
        "final_manifest_playable",
        "QUESTION_BANK_QUERY_STARTED",
        "QUESTION_BANK_QUERY_COMPLETED",
        "QUESTION_BANK_REUSE_QUERY_STARTED",
        "QUESTION_BANK_REUSE_QUERY_COMPLETED",
        "QUESTION_BANK_CATEGORY_FALLBACK",
        "QUESTION_BANK_REUSE_LIMIT_REACHED",
        "QUESTION_MANIFEST_UPDATED",
        "ASSESSMENT_QUESTIONS_QUERY_COMPLETED",
        "ASSESSMENT_CREATED",
        "QUESTION_COUNT_CLAMPED",
        "BLUEPRINT_STARTED",
        "BLUEPRINT_COMPLETED",
        "BLUEPRINT_REPAIRED",
        "EXISTING_MATCH_STARTED",
        "EXISTING_MATCH_COMPLETED",
        "DEFICIT_CALCULATED",
        "GENERATION_GROUP_CREATED",
        "GENERATION_GROUP_STARTED",
        "GENERATION_ITEM_ACCEPTED",
        "GENERATION_ITEM_REJECTED",
        "GENERATION_ITEM_RETRY",
        "QUESTION_VERIFICATION_RESULT",
        "QUESTION_LINKED",
        "ASSESSMENT_PROGRESS",
        "ASSESSMENT_FINALIZING",
        "ASSESSMENT_FINALIZATION_RETRY",
        "ASSESSMENT_READY",
        "ASSESSMENT_FAILED",
        "ASSESSMENT_CANCELLED",
    }
)

_SENSITIVE_KEY_PARTS = (
    "access_key",
    "actor_id",
    "answer",
    "api_key",
    "authorization",
    "aws_access",
    "aws_secret",
    "content",
    "conversation_context",
    "context_text",
    "credential",
    "endpoint",
    "image",
    "jwt",
    "message",
    "payload",
    "prompt",
    "query",
    "raw",
    "refresh_token",
    "secret",
    "token",
    "user_id",
)
_SAFE_USAGE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
    }
)
_MAX_DETAILS = 32
_MAX_STRING = 256
_logger = logging.getLogger("agent.observability")
_environment = "local"
_detailed_logs = False


def configure_event_metadata(*, environment: str, detailed_logs: bool) -> None:
    global _environment, _detailed_logs
    _environment = environment
    _detailed_logs = detailed_logs


def _safe_key(key: object) -> str | None:
    normalized = str(key).strip().lower()
    if normalized in _SAFE_USAGE_KEYS:
        return normalized
    if not normalized or any(part in normalized for part in _SENSITIVE_KEY_PARTS):
        return None
    return normalized[:64]


def _safe_value(value: object) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value.replace("\r", " ").replace("\n", " ")[:_MAX_STRING]
    return type(value).__name__


def sanitize_details(details: Mapping[str, object] | None) -> dict[str, object]:
    if not details:
        return {}
    safe: dict[str, object] = {}
    for key, value in list(details.items())[:_MAX_DETAILS]:
        safe_key = _safe_key(key)
        if safe_key is not None:
            safe[safe_key] = _safe_value(value)
    return safe


def build_event(
    event: str,
    *,
    component: str,
    stage: str | None = None,
    status: str | None = None,
    duration_ms: int | None = None,
    error_code: str | None = None,
    details: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    if event not in EVENT_NAMES:
        raise ValueError(f"Unsupported observability event: {event}")
    context = current_request_context()
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "level": "INFO",
        "service": SERVICE_NAME,
        "environment": _environment,
        "event": event,
        "component": component[:96],
        "request_id": context.request_id if context else None,
        "trace_id": context.trace_id if context else None,
        "conversation_id": context.conversation_id if context else None,
        "turn_id": context.turn_id if context else None,
        "stage": stage,
        "status": status,
        "duration_ms": max(duration_ms, 0) if duration_ms is not None else None,
        "error_code": error_code[:96] if error_code else None,
        "details": sanitize_details(details),
    }


def log_event(
    event: str,
    *,
    component: str,
    stage: str | None = None,
    status: str | None = None,
    duration_ms: int | None = None,
    error_code: str | None = None,
    details: Mapping[str, object] | None = None,
    level: int = logging.INFO,
) -> None:
    payload = build_event(
        event,
        component=component,
        stage=stage,
        status=status,
        duration_ms=duration_ms,
        error_code=error_code,
        details=details,
    )
    payload["level"] = logging.getLevelName(level)
    context = current_request_context()
    concise = [event]
    if status:
        concise.append(f"status={status}")
    if stage:
        concise.append(f"stage={stage}")
    if context:
        concise.append(f"request_id={context.request_id[:8]}")
    if duration_ms is not None:
        concise.append(f"duration_ms={max(duration_ms, 0)}")
    if _detailed_logs:
        concise.extend(f"{key}={value}" for key, value in payload["details"].items())
    collect_event(payload)
    _logger.log(level, " ".join(concise), extra={"observability_event": payload})
