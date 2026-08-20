"""Safe assessment-level observability events."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from observability import log_event

_PRODUCTION_MILESTONE_EVENTS = frozenset(
    {
        "PRACTICE_RUNTIME_STARTED",
        "PRACTICE_RUNTIME_VALIDATED",
        "practice_assessment_initialized",
        "practice_graph_started",
        "BLUEPRINT_COMPLETED",
        "EXISTING_MATCH_COMPLETED",
        "final_manifest_validation_completed",
        "practice_ready",
        # Aggregate-only; candidate-by-candidate Pattern events stay DEBUG-default.
        "PATTERN_RETRIEVAL_COMPLETED",
    }
)
_RECOVERY_WARNING_EVENTS = frozenset(
    {
        "planner_repair_started",
        "planner_repair_completed",
        "planner_fallback_completed",
        "practice_provider_replacement_scheduled",
        "practice_validation_repair_started",
        "practice_replacement_started",
        "question_repair_requested",
        "question_repair_started",
        "question_repair_completed",
        "question_repair_failed",
        "question_replacement_started",
        "question_replacement_completed",
        "GENERATION_ITEM_RETRY",
        "manifest_recovery_started",
        "manifest_recovery_completed",
        "final_manifest_validation_failed",
        "QUESTION_BANK_CATEGORY_FALLBACK",
        "QUESTION_BANK_REUSE_LIMIT_REACHED",
        "QUESTION_BANK_REUSE_QUERY_DEBUG",
        "QUESTION_SEMANTIC_RETRIEVAL_FAILED",
        "QUESTION_SEMANTIC_CANDIDATE_DECISION",
        "QUESTION_SEMANTIC_REUSE_COMPLETED",
        "PATTERN_QUESTION_BANK_LINK_FAILED",
        "PATTERN_QUESTION_BANK_LINK_CONFLICT",
        "PATTERN_REUSE_HISTORY_UNAVAILABLE",
    }
)


def _default_practice_event_level(event_name: str) -> int:
    if event_name in _PRODUCTION_MILESTONE_EVENTS:
        return logging.INFO
    if event_name in _RECOVERY_WARNING_EVENTS:
        return logging.WARNING
    return logging.DEBUG


def safe_user_ref(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def emit_practice_event(
    event_name: str,
    *,
    test_id: str,
    status: str,
    details: dict[str, Any] | None = None,
    level: int | None = None,
) -> None:
    safe = {
        key: value
        for key, value in (details or {}).items()
        if key
        not in {
            "prompt",
            "query",
            "question",
            "questions",
            "answer",
            "answers",
            "options",
            "solution",
            "provider_response",
            "raw_provider_payload",
            "credentials",
        }
    }
    safe["testId"] = test_id
    safe["activityId"] = test_id
    log_event(
        event_name,
        component="practice_generation",
        stage=str(safe.get("phase") or "practice"),
        status=status,
        error_code=(str(safe["reasonCode"]) if safe.get("reasonCode") is not None else None),
        details=safe,
        level=(level if level is not None else _default_practice_event_level(event_name)),
    )
