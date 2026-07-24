"""Memory-first recent conversation loading with exact DynamoDB fallback."""

from __future__ import annotations

import logging
import time

from observability import (
    log_event,
    record_local_preview,
    stage_span,
    update_request_summary,
)
from schemas.conversation import RecentContextLoadResult
from services.conversation.history_repository import (
    ConversationHistoryError,
    ConversationHistoryIsolationError,
    ConversationHistoryRepository,
)
from services.conversation.recent_context import format_recent_conversation
from services.conversation.short_term_memory import (
    AgentCoreShortTermMemory,
    ShortTermMemoryError,
    ShortTermMemoryReadResult,
    UnavailableShortTermMemory,
)

logger = logging.getLogger(__name__)


class RecentConversationContextService:
    def __init__(
        self,
        *,
        history_repository: ConversationHistoryRepository,
        short_term_memory: AgentCoreShortTermMemory | UnavailableShortTermMemory,
    ) -> None:
        self._history_repository = history_repository
        self._short_term_memory = short_term_memory

    def load(
        self, actor_id: str, conversation_id: str, limit: int = 2
    ) -> RecentContextLoadResult:
        started = time.monotonic()
        memory_reason: str | None = None
        memory_attempted = True
        memory_event_count = 0
        memory_started = time.monotonic()
        try:
            with stage_span("doubt_solver.load_recent_context", source="agentcore_memory"):
                detailed = self._short_term_memory.read_recent_completed_turns(
                    actor_id, conversation_id, limit
                )
            if isinstance(detailed, ShortTermMemoryReadResult):
                memory_turns = list(detailed.turns)
                memory_reason = detailed.failure_reason
                memory_attempted = detailed.attempted
                memory_event_count = detailed.event_count
            else:
                memory_turns = self._short_term_memory.list_recent_completed_turns(
                    actor_id, conversation_id, limit
                )
                memory_reason = None if memory_turns else "memory_no_events"
        except ShortTermMemoryError as exc:
            memory_turns = []
            memory_reason = exc.reason
        memory_duration_ms = int((time.monotonic() - memory_started) * 1000)
        if memory_turns:
            context = format_recent_conversation(memory_turns, source="agentcore")
            result = RecentContextLoadResult(
                source="agentcore_memory",
                memory_attempted=memory_attempted,
                memory_status="succeeded",
                memory_event_count=memory_event_count,
                memory_completed_pair_count=len(context.turns),
                memory_returned_turn_ids=tuple(turn.turn_id for turn in context.turns),
                memory_latest_event_time=context.turns[-1].created_at,
                memory_duration_ms=memory_duration_ms,
                turns=context.turns,
                usable_turn_count=len(context.turns),
                latency_ms=int((time.monotonic() - started) * 1000),
                formatted_reference=context.formatted_reference,
            )
            self._log_result(result)
            return result

        logger.debug("dynamodb_fallback_count count=1")
        if memory_reason is not None:
            logger.warning(
                "follow_up_context_failure source=agentcore_memory failure_reason=%s",
                memory_reason,
            )

        dynamodb_started = time.monotonic()
        try:
            history_turns = self._history_repository.list_recent_completed_turns(
                actor_id, conversation_id, limit
            )
        except ConversationHistoryIsolationError as exc:
            result = self._empty_result(
                started,
                exc.reason,
                memory_attempted=memory_attempted,
                memory_reason=memory_reason,
                memory_event_count=memory_event_count,
                memory_duration_ms=memory_duration_ms,
                dynamodb_duration_ms=int((time.monotonic() - dynamodb_started) * 1000),
                dynamodb_status="failed_validation",
                dynamodb_reason=exc.reason,
            )
            self._log_result(result)
            return result
        except ConversationHistoryError as exc:
            result = self._empty_result(
                started,
                exc.reason,
                memory_attempted=memory_attempted,
                memory_reason=memory_reason,
                memory_event_count=memory_event_count,
                memory_duration_ms=memory_duration_ms,
                dynamodb_duration_ms=int((time.monotonic() - dynamodb_started) * 1000),
                dynamodb_status=self._history_status(exc.reason),
                dynamodb_reason=exc.reason,
            )
            self._log_result(result)
            return result

        if history_turns:
            context = format_recent_conversation(history_turns, source="dynamodb")
            result = RecentContextLoadResult(
                source="dynamodb_fallback",
                memory_attempted=memory_attempted,
                memory_status=self._memory_status(memory_reason),
                memory_event_count=memory_event_count,
                memory_duration_ms=memory_duration_ms,
                memory_failure_reason=memory_reason,
                dynamodb_attempted=True,
                dynamodb_status="succeeded",
                dynamodb_item_count=len(history_turns),
                dynamodb_returned_turn_ids=tuple(
                    turn.turn_id for turn in context.turns
                ),
                dynamodb_duration_ms=int(
                    (time.monotonic() - dynamodb_started) * 1000
                ),
                turns=context.turns,
                usable_turn_count=len(context.turns),
                latency_ms=int((time.monotonic() - started) * 1000),
                formatted_reference=context.formatted_reference,
            )
            self._log_result(result)
            return result

        failure_reason = "history_no_items"
        result = self._empty_result(
            started,
            failure_reason,
            memory_attempted=memory_attempted,
            memory_reason=memory_reason,
            memory_event_count=memory_event_count,
            memory_duration_ms=memory_duration_ms,
            dynamodb_duration_ms=int((time.monotonic() - dynamodb_started) * 1000),
            dynamodb_status="empty",
            dynamodb_reason=failure_reason,
        )
        self._log_result(result)
        return result

    @staticmethod
    def _empty_result(
        started: float,
        reason: str,
        *,
        memory_attempted: bool = True,
        memory_reason: str | None = None,
        memory_event_count: int = 0,
        memory_duration_ms: int = 0,
        dynamodb_duration_ms: int = 0,
        dynamodb_status: str = "empty",
        dynamodb_reason: str | None = None,
    ) -> RecentContextLoadResult:
        return RecentContextLoadResult(
            source="none",
            memory_attempted=memory_attempted,
            memory_status=RecentConversationContextService._memory_status(memory_reason),
            memory_event_count=memory_event_count,
            memory_duration_ms=memory_duration_ms,
            memory_failure_reason=memory_reason,
            dynamodb_attempted=True,
            dynamodb_status=dynamodb_status,
            dynamodb_duration_ms=dynamodb_duration_ms,
            dynamodb_failure_reason=dynamodb_reason,
            failure_reason=reason,
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    @staticmethod
    def _memory_status(reason: str | None) -> str:
        if reason is None:
            return "succeeded"
        if reason == "memory_not_configured":
            return "not_configured"
        if reason == "memory_no_events":
            return "empty"
        if reason == "memory_permission_denied":
            return "failed_permission"
        if reason == "memory_resource_missing":
            return "failed_configuration"
        return "failed_transient"

    @staticmethod
    def _history_status(reason: str) -> str:
        if reason == "history_not_configured":
            return "not_configured"
        if reason == "history_permission_denied":
            return "failed_permission"
        if reason in {
            "history_actor_mismatch",
            "history_conversation_mismatch",
            "history_no_completed_turns",
        }:
            return "failed_validation"
        return "failed_transient"

    @staticmethod
    def _log_result(result: RecentContextLoadResult) -> None:
        logger.debug(
            "follow_up_context_source source=%s usable_turn_count=%d latency_ms=%d",
            result.source,
            result.usable_turn_count,
            result.latency_ms,
        )
        if result.failure_reason is not None:
            logger.warning(
                "follow_up_context_failure failure_reason=%s",
                result.failure_reason,
            )
        if result.formatted_reference:
            logger.debug(
                "recent_context_token_count count=%d",
                (len(result.formatted_reference) + 3) // 4,
            )
        update_request_summary(
            context_source=result.source,
            usable_recent_turns=result.usable_turn_count,
        )
        preview_prefix = (
            "memory" if result.source == "agentcore_memory" else "dynamodb"
        )
        for index, turn in enumerate(result.turns[-2:], start=1):
            record_local_preview(
                f"{preview_prefix}_turn_{index}_id",
                turn.turn_id,
            )
            record_local_preview(
                f"{preview_prefix}_turn_{index}_user",
                turn.original_query,
            )
            record_local_preview(
                f"{preview_prefix}_turn_{index}_assistant",
                turn.final_answer,
            )
        if result.source == "none":
            log_event(
                "follow_up_context_failed",
                component="conversation.recent_context",
                stage="load_recent_context",
                status="failed",
                duration_ms=result.latency_ms,
                error_code=(result.failure_reason or "CONTEXT_UNAVAILABLE").upper(),
                details={
                    "failure_reason": result.failure_reason,
                    "memory_attempted": result.memory_attempted,
                    "memory_status": result.memory_status,
                    "memory_event_count": result.memory_event_count,
                    "memory_completed_pair_count": result.memory_completed_pair_count,
                    "memory_returned_turn_ids": ",".join(
                        result.memory_returned_turn_ids
                    ),
                    "memory_latest_event_time": (
                        result.memory_latest_event_time.isoformat()
                        if result.memory_latest_event_time
                        else None
                    ),
                    "memory_duration_ms": result.memory_duration_ms,
                    "memory_failure_reason": result.memory_failure_reason,
                    "dynamodb_attempted": result.dynamodb_attempted,
                    "dynamodb_status": result.dynamodb_status,
                    "dynamodb_item_count": result.dynamodb_item_count,
                    "dynamodb_returned_turn_ids": ",".join(
                        result.dynamodb_returned_turn_ids
                    ),
                    "dynamodb_duration_ms": result.dynamodb_duration_ms,
                    "dynamodb_failure_reason": result.dynamodb_failure_reason,
                    "usable_recent_turns": result.usable_turn_count,
                    "source": result.source,
                },
                level=logging.WARNING,
            )
        else:
            log_event(
                "follow_up_context_loaded",
                component="conversation.recent_context",
                stage="load_recent_context",
                status="completed",
                duration_ms=result.latency_ms,
                details={
                    "source": result.source,
                    "usable_recent_turns": result.usable_turn_count,
                    "memory_attempted": result.memory_attempted,
                    "memory_status": result.memory_status,
                    "memory_event_count": result.memory_event_count,
                    "memory_completed_pair_count": result.memory_completed_pair_count,
                    "memory_returned_turn_ids": ",".join(
                        result.memory_returned_turn_ids
                    ),
                    "memory_latest_event_time": (
                        result.memory_latest_event_time.isoformat()
                        if result.memory_latest_event_time
                        else None
                    ),
                    "memory_duration_ms": result.memory_duration_ms,
                    "memory_failure_reason": result.memory_failure_reason,
                    "dynamodb_attempted": result.dynamodb_attempted,
                    "dynamodb_status": result.dynamodb_status,
                    "dynamodb_item_count": result.dynamodb_item_count,
                    "dynamodb_returned_turn_ids": ",".join(
                        result.dynamodb_returned_turn_ids
                    ),
                    "dynamodb_duration_ms": result.dynamodb_duration_ms,
                    "dynamodb_failure_reason": result.dynamodb_failure_reason,
                },
            )
            if result.source == "dynamodb_fallback":
                log_event(
                    "retrieval_fallback_used",
                    component="conversation.recent_context",
                    stage="load_recent_context",
                    status="fallback",
                    details={"source": result.source, "reason": "memory_unavailable"},
                )
