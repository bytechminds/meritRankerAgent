"""Resolve context-dependent student queries into standalone academic queries."""

from __future__ import annotations

import json
import logging
import time

from observability import log_event, record_local_preview, stage_span
from schemas.conversation import (
    RecentContextLoadResult,
    RecentConversationContext,
    ResolvedFollowUpQuery,
)
from schemas.llm_routing import RouteRequest
from services.conversation.persistence import ConversationPersistenceService
from services.conversation.recent_context import format_recent_conversation
from services.llm.orchestration.orchestrator import LlmOrchestrator

_MIN_CONFIDENCE = 0.75
logger = logging.getLogger(__name__)


class FollowUpResolutionError(RuntimeError):
    """A follow-up could not be resolved safely."""

    def __init__(
        self,
        message: str,
        *,
        stage: str = "resolver",
        reason: str = "invalid_output",
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.reason = reason


class FollowUpQueryResolver:
    def __init__(self, *, orchestrator: LlmOrchestrator) -> None:
        self._orchestrator = orchestrator

    def resolve(
        self,
        *,
        request_id: str,
        original_query: str,
        context: RecentConversationContext,
    ) -> ResolvedFollowUpQuery:
        if not context.turns or not context.formatted_reference:
            raise FollowUpResolutionError(
                "Recent conversation is unavailable.",
                stage="context_load",
                reason="no_completed_pairs",
            )
        started = time.monotonic()
        try:
            with stage_span("doubt_solver.resolve_follow_up"):
                result = self._orchestrator.generate(
                    route_request=RouteRequest(
                        request_id=request_id,
                        subject="general",
                        task_role="follow_up_resolver",
                        difficulty="default",
                    ),
                    query=original_query,
                    conversation_context=context.formatted_reference,
                )
        except Exception as exc:
            log_event(
                "follow_up_resolution_failed",
                component="conversation.follow_up",
                stage="resolve_follow_up",
                status="failed",
                duration_ms=int((time.monotonic() - started) * 1000),
                error_code="RESOLVER_EXECUTION_FAILED",
                details={"error_type": type(exc).__name__},
                level=logging.WARNING,
            )
            raise
        try:
            raw = json.loads(result.content)
            resolved = ResolvedFollowUpQuery.model_validate(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise FollowUpResolutionError(
                "Follow-up resolver returned invalid output.", reason="invalid_output"
            ) from exc
        if resolved.confidence < _MIN_CONFIDENCE:
            raise FollowUpResolutionError(
                "Follow-up reference is ambiguous.", reason="low_confidence"
            )
        log_event(
            "follow_up_resolution_completed",
            component="conversation.follow_up",
            stage="resolve_follow_up",
            status="completed",
            duration_ms=int((time.monotonic() - started) * 1000),
            details={"confidence": resolved.confidence},
        )
        return resolved


def resolve_follow_up_with_recent_context(
    *,
    persistence: ConversationPersistenceService,
    resolver: FollowUpQueryResolver,
    actor_id: str,
    conversation_id: str,
    request_id: str,
    original_query: str,
) -> tuple[ResolvedFollowUpQuery, RecentContextLoadResult | RecentConversationContext]:
    """Resolve from two turns, consulting a third only after controlled ambiguity."""
    loaded = persistence.load_recent_context(actor_id, conversation_id, 2)
    usable_turn_count = getattr(loaded, "usable_turn_count", len(loaded.turns))
    if usable_turn_count < 1 or not loaded.formatted_reference:
        raise FollowUpResolutionError(
            "Recent conversation is unavailable.",
            stage="context_load",
            reason=getattr(loaded, "failure_reason", None) or "no_completed_pairs",
        )
    loaded_context = (
        loaded.as_conversation_context()
        if isinstance(loaded, RecentContextLoadResult)
        else loaded
    )
    context = format_recent_conversation(
        list(loaded_context.turns[-2:]),
        source=loaded_context.source,
    )
    record_local_preview(
        "previous_turns",
        " | ".join(
            f"Q: {turn.original_query} A: {turn.final_answer}" for turn in context.turns
        ),
    )
    try:
        resolved = resolver.resolve(
            request_id=request_id,
            original_query=original_query,
            context=context,
        )
        record_local_preview("resolved_query", resolved.resolved_query)
        logger.debug("immediate_follow_up_success count=1")
        if isinstance(loaded, RecentContextLoadResult):
            used: RecentContextLoadResult | RecentConversationContext = loaded.model_copy(
                update={
                    "turns": context.turns,
                    "usable_turn_count": len(context.turns),
                    "formatted_reference": context.formatted_reference,
                }
            )
        else:
            used = context
        return resolved, used
    except FollowUpResolutionError as exc:
        logger.warning(
            "follow_up_resolver_failure count=1 failure_reason=%s",
            exc.reason,
        )
        if exc.reason != "low_confidence":
            raise
        expanded = persistence.load_recent_context(actor_id, conversation_id, 3)
        expanded_context = (
            expanded.as_conversation_context()
            if isinstance(expanded, RecentContextLoadResult)
            else expanded
        )
        if len(expanded_context.turns) <= len(context.turns):
            raise
        resolved = resolver.resolve(
            request_id=request_id,
            original_query=original_query,
            context=expanded_context,
        )
        logger.debug("immediate_follow_up_success count=1")
        return resolved, expanded
