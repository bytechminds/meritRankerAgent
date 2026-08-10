"""Safe assessment-level observability events."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from observability import log_event


def safe_user_ref(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def emit_practice_event(
    event_name: str,
    *,
    test_id: str,
    status: str,
    details: dict[str, Any] | None = None,
    level: int = logging.INFO,
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
        level=level,
    )
