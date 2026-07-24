"""Answer completion marker detection and bounded continuation policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import Settings, get_settings
from schemas.llm import LlmMessage
from services.doubt_solver.answer_quality import detect_final_answer

logger = logging.getLogger(__name__)

_CURRENT_AFFAIRS_REASONS: frozenset[str] = frozenset(
    {"current_affairs", "current_economy", "current_event"}
)

CONTINUATION_USER_PROMPT = (
    "Continue from the last incomplete point. Do not restart. Be concise. "
    "Finish the answer and end with <ANSWER_DONE>."
)


@dataclass(frozen=True)
class AnswerCompletionPolicy:
    """Runtime policy for completion markers and continuation."""

    marker: str
    continuation_enabled: bool
    continuation_max_attempts: int

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> AnswerCompletionPolicy:
        cfg = settings or get_settings()
        return cls(
            marker=cfg.answer_completion_marker,
            continuation_enabled=cfg.answer_continuation_enabled,
            continuation_max_attempts=cfg.answer_continuation_max_attempts,
        )


def resolve_generator_route_subject(
    *,
    subject: str,
    intent: str,
    web_search_reason: str | None = None,
) -> str:
    """Map classifier output to generator route subject for token budgets."""
    if intent == "practice":
        return "practice"
    if web_search_reason and web_search_reason.strip().lower() in _CURRENT_AFFAIRS_REASONS:
        return "current_affairs"
    return subject


def has_completion_marker(content: str, marker: str) -> bool:
    return marker in content


def strip_completion_marker(content: str, marker: str) -> str:
    if not content:
        return ""
    cleaned = content.replace(marker, "").rstrip()
    return cleaned


def needs_continuation(
    content: str,
    finish_reason: str | None,
    policy: AnswerCompletionPolicy,
) -> bool:
    # Empty/whitespace base answer: continuation would create invalid messages and
    # signals a stream-level failure, not an incomplete answer. Block it.
    if not content or not content.strip():
        return False
    if not policy.continuation_enabled or policy.continuation_max_attempts < 1:
        return False
    if finish_reason == "length":
        return True
    if has_completion_marker(content, policy.marker):
        return False
    if detect_final_answer(content):
        return False
    return True


def marker_missing_but_answer_complete(content: str, policy: AnswerCompletionPolicy) -> bool:
    """True when marker absent but a Final Answer section exists."""
    return (
        not has_completion_marker(content, policy.marker)
        and detect_final_answer(content)
    )


def should_run_continuation(
    content: str,
    finish_reason: str | None,
    policy: AnswerCompletionPolicy,
    *,
    provider: str | None,
    task_role: str | None = None,
    route_id: str | None = None,
) -> bool:
    """Apply continuation policy; mock provider skips unless truncated by length."""
    base_answer_present = bool(content and content.strip())
    if not is_answer_generation_route(task_role=task_role, route_id=route_id):
        _log_continuation_decision(
            base_answer_present=base_answer_present,
            finish_reason=finish_reason,
            continuation_allowed=False,
            reason="not_generator_route",
            content=content,
            policy=policy,
        )
        return False
    if not needs_continuation(content, finish_reason, policy):
        _log_continuation_decision(
            base_answer_present=base_answer_present,
            finish_reason=finish_reason,
            continuation_allowed=False,
            reason="not_needed",
            content=content,
            policy=policy,
        )
        return False
    if provider == "mock" and finish_reason != "length":
        _log_continuation_decision(
            base_answer_present=base_answer_present,
            finish_reason=finish_reason,
            continuation_allowed=False,
            reason="mock_provider_skip",
            content=content,
            policy=policy,
        )
        return False
    _log_continuation_decision(
        base_answer_present=base_answer_present,
        finish_reason=finish_reason,
        continuation_allowed=True,
        reason="allowed",
        content=content,
        policy=policy,
    )
    return True


def _log_continuation_decision(
    *,
    base_answer_present: bool,
    finish_reason: str | None,
    continuation_allowed: bool,
    reason: str,
    content: str,
    policy: AnswerCompletionPolicy,
) -> None:
    """Emit safe continuation decision diagnostics. Never logs content."""
    logger.debug(
        "answer_continuation_decision  base_answer_present=%s  finish_reason=%s  "
        "final_answer_detected=%s  marker_found=%s  continuation_allowed=%s  reason=%s",
        base_answer_present,
        finish_reason or "",
        detect_final_answer(content),
        has_completion_marker(content, policy.marker),
        continuation_allowed,
        reason,
    )


def is_answer_generation_route(
    *,
    task_role: str | None = None,
    route_id: str | None = None,
) -> bool:
    """True only for final answer generator routes (not classifier/JSON tasks)."""
    if task_role == "generator":
        return True
    if route_id and ".generator." in route_id:
        return True
    return False


def continuation_max_tokens(*, difficulty: str, route_subject: str) -> int:
    if difficulty == "advanced" or route_subject == "practice":
        return 1000
    if difficulty == "intermediate":
        return 700
    return 500


def build_continuation_messages(
    messages: list[LlmMessage],
    *,
    partial_content: str,
    policy: AnswerCompletionPolicy,
) -> list[LlmMessage]:
    return [
        *messages,
        LlmMessage(role="assistant", content=partial_content),
        LlmMessage(role="user", content=CONTINUATION_USER_PROMPT),
    ]


class StreamingMarkerFilter:
    """Strip completion marker from streamed chunks without leaking partial marker text."""

    def __init__(self, marker: str) -> None:
        self._marker = marker
        self._holdback = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        combined = (self._holdback + chunk).replace(self._marker, "")
        hold_len = max(0, len(self._marker) - 1)
        if len(combined) <= hold_len:
            self._holdback = combined
            return ""
        emit = combined[:-hold_len] if hold_len else combined
        self._holdback = combined[-hold_len:] if hold_len else ""
        return emit

    def flush(self) -> str:
        remaining = self._holdback.replace(self._marker, "")
        self._holdback = ""
        return remaining


def log_answer_generation_budget(
    *,
    request_id: str,
    route_id: str,
    subject: str,
    difficulty: str,
    intent: str | None,
    max_output_tokens: int,
    context_chars: int,
) -> None:
    logger.debug(
        "answer_generation_budget  request_id=%s  route_id=%s  subject=%s  "
        "difficulty=%s  intent=%s  max_output_tokens=%d  context_chars=%d",
        request_id,
        route_id,
        subject,
        difficulty,
        intent or "",
        max_output_tokens,
        context_chars,
    )


def log_answer_completion(
    *,
    request_id: str,
    finish_reason: str | None,
    completion_marker_found: bool,
    final_answer_detected: bool,
    continuation_used: bool,
    continuation_attempts: int,
    rewrite_used: bool,
    output_chars: int,
    marker_missing_but_answer_complete: bool = False,
) -> None:
    logger.debug(
        "answer_completion  request_id=%s  finish_reason=%s  "
        "completion_marker_found=%s  final_answer_detected=%s  "
        "continuation_used=%s  continuation_attempts=%d  rewrite_used=%s  "
        "output_chars=%d  completion_marker_missing_but_answer_complete=%s",
        request_id,
        finish_reason or "",
        completion_marker_found,
        final_answer_detected,
        continuation_used,
        continuation_attempts,
        rewrite_used,
        output_chars,
        marker_missing_but_answer_complete,
    )
