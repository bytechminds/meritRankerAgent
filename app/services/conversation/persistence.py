"""Completed-turn persistence and recent-context fallback coordination."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import copy_context
from typing import TypeVar, cast

from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)
from pydantic import ValidationError

from observability import log_event, stage_span, update_request_summary
from schemas.conversation import (
    CompletedConversationTurn,
    ConversationPersistenceResult,
    PersistenceDecision,
    PersistenceSkipReason,
    PersistenceStatus,
    RecentContextLoadResult,
)
from schemas.doubt_solver import FinalAnswerResult
from services.conversation.history_repository import ConversationHistoryRepository
from services.conversation.memory_hygiene import classify_non_substantive_response
from services.conversation.recent_context_service import RecentConversationContextService
from services.conversation.session_repository import ConversationSessionRepository
from services.conversation.short_term_memory import (
    AgentCoreShortTermMemory,
    ShortTermMemoryError,
    UnavailableShortTermMemory,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
_ACCEPTED_QUALITY_STATUSES = frozenset({"checked", "passed_quality_gate"})
_TRANSIENT_ERROR_CODES = frozenset(
    {
        "InternalServerError",
        "InternalFailure",
        "RequestLimitExceeded",
        "RequestTimeout",
        "ServiceUnavailable",
        "ThrottlingException",
    }
)
_PERMISSION_ERROR_CODES = frozenset(
    {"AccessDenied", "AccessDeniedException", "UnauthorizedException"}
)
_CONFIGURATION_ERROR_CODES = frozenset(
    {"ResourceNotFound", "ResourceNotFoundException", "ValidationException"}
)


def _find_client_error(error: BaseException) -> ClientError | None:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, ClientError):
            return current
        current = current.__cause__
    return None


def _is_transient(error: BaseException) -> bool:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, ShortTermMemoryError):
            return current.reason == "memory_timeout"
        if isinstance(
            current,
            (ConnectionClosedError, EndpointConnectionError, ReadTimeoutError),
        ):
            return True
        if isinstance(current, ClientError):
            code = str(current.response.get("Error", {}).get("Code") or "")
            return code in _TRANSIENT_ERROR_CODES
        current = current.__cause__
    return False


def _status_for_exception(error: BaseException) -> PersistenceStatus:
    if isinstance(error, ShortTermMemoryError):
        if error.reason == "memory_timeout":
            return "failed_transient"
        if error.reason == "memory_permission_denied":
            return "failed_permission"
        if error.reason == "memory_resource_missing":
            return "failed_configuration"
        if error.reason == "memory_not_configured":
            return "failed_configuration"
    if _is_transient(error):
        return "failed_transient"
    client_error = _find_client_error(error)
    if client_error is not None:
        code = str(client_error.response.get("Error", {}).get("Code") or "")
        if code in _PERMISSION_ERROR_CODES:
            return "failed_permission"
        if code in _CONFIGURATION_ERROR_CODES:
            return "failed_configuration"
    if isinstance(error, (TypeError, ValueError, ValidationError)):
        return "failed_validation"
    return "failed_configuration"


def _run_with_one_transient_retry(operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except Exception as exc:  # noqa: BLE001
        if not _is_transient(exc):
            raise
        return operation()


def evaluate_completed_turn_persistence(
    final_answer: FinalAnswerResult,
    *,
    response_type: str | None = None,
    finalized: bool = True,
    request_cancelled: bool = False,
    clarification_response: bool = False,
) -> PersistenceDecision:
    if request_cancelled:
        return PersistenceDecision(persistable=False, skip_reason="request_cancelled")
    if clarification_response:
        return PersistenceDecision(persistable=False, skip_reason="clarification_response")
    if not finalized:
        return PersistenceDecision(persistable=False, skip_reason="not_finalized")
    if not final_answer.content.strip():
        return PersistenceDecision(persistable=False, skip_reason="empty_final_answer")
    response_rejection = (
        None
        if response_type == "practice_generation"
        else classify_non_substantive_response(final_answer.content)
    )
    if response_rejection in {
        "clarification_response",
        "unresolved_reference_response",
    }:
        return PersistenceDecision(persistable=False, skip_reason="clarification_response")
    if response_rejection is not None:
        return PersistenceDecision(persistable=False, skip_reason="non_substantive_answer")
    if not final_answer.language_compliant:
        return PersistenceDecision(persistable=False, skip_reason="language_non_compliant")
    if final_answer.quality_status not in _ACCEPTED_QUALITY_STATUSES:
        return PersistenceDecision(persistable=False, skip_reason="failed_quality_gate")
    return PersistenceDecision(persistable=True)


def is_completed_turn_persistable(final_answer: FinalAnswerResult) -> bool:
    return evaluate_completed_turn_persistence(final_answer).persistable


class ConversationPersistenceService:
    def __init__(
        self,
        *,
        history_repository: ConversationHistoryRepository,
        short_term_memory: AgentCoreShortTermMemory | UnavailableShortTermMemory,
        session_repository: ConversationSessionRepository | None = None,
        write_timeout_seconds: float = 2.0,
    ) -> None:
        self.history_repository = history_repository
        self.short_term_memory = short_term_memory
        self.session_repository = session_repository
        self._write_timeout_seconds = write_timeout_seconds
        self._recent_context = RecentConversationContextService(
            history_repository=history_repository,
            short_term_memory=short_term_memory,
        )

    def load_recent_context(
        self, actor_id: str, conversation_id: str, limit: int = 2
    ) -> RecentContextLoadResult:
        return self._recent_context.load(actor_id, conversation_id, limit)

    def record_skip(
        self,
        *,
        request_id: str,
        conversation_id: str,
        turn_id: str,
        skip_reason: PersistenceSkipReason,
    ) -> ConversationPersistenceResult:
        result = ConversationPersistenceResult(
            request_id=request_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            persistable=False,
            history_write_status="skipped",
            session_write_status="skipped",
            memory_write_status="skipped",
            skip_reason=skip_reason,
        )
        self._log_result(result)
        return result

    def persist_completed_turn(
        self,
        turn: CompletedConversationTurn,
        final_answer: FinalAnswerResult,
        *,
        request_id: str = "",
        finalized: bool = True,
        request_cancelled: bool = False,
        clarification_response: bool = False,
    ) -> ConversationPersistenceResult:
        decision = evaluate_completed_turn_persistence(
            final_answer,
            response_type=turn.response_type,
            finalized=finalized,
            request_cancelled=request_cancelled,
            clarification_response=clarification_response,
        )
        if not decision.persistable:
            logger.debug("persistence_skip_count count=1 reason=%s", decision.skip_reason)
            return self.record_skip(
                request_id=request_id,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                skip_reason=cast(PersistenceSkipReason, decision.skip_reason),
            )

        pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="conversation-write")
        try:
            history_future = pool.submit(
                copy_context().run, self._persist_history_and_session, turn
            )
            memory_future = pool.submit(copy_context().run, self._persist_memory, turn)
            done, not_done = wait(
                {history_future, memory_future}, timeout=self._write_timeout_seconds
            )

            if history_future in done:
                try:
                    history_status, session_status, history_ms, session_ms = history_future.result()
                except Exception as exc:  # noqa: BLE001
                    history_status = _status_for_exception(exc)
                    session_status = "skipped"
                    history_ms = 0
                    session_ms = 0
            else:
                history_status = "failed_transient"
                session_status = "skipped"
                history_ms = int(self._write_timeout_seconds * 1000)
                session_ms = 0
                history_future.cancel()

            if memory_future in done:
                try:
                    memory_status, memory_ms = memory_future.result()
                except Exception as exc:  # noqa: BLE001
                    memory_status = _status_for_exception(exc)
                    memory_ms = 0
            else:
                memory_status = "failed_transient"
                memory_ms = int(self._write_timeout_seconds * 1000)
                memory_future.cancel()

            for future in not_done:
                future.cancel()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

        result = ConversationPersistenceResult(
            request_id=request_id,
            conversation_id=turn.conversation_id,
            turn_id=turn.turn_id,
            persistable=True,
            history_write_status=history_status,
            session_write_status=session_status,
            memory_write_status=memory_status,
            history_latency_ms=history_ms,
            session_latency_ms=session_ms,
            memory_latency_ms=memory_ms,
        )
        self._log_failures(result)
        self._log_result(result)
        return result

    def _persist_history_and_session(
        self, turn: CompletedConversationTurn
    ) -> tuple[PersistenceStatus, PersistenceStatus, int, int]:
        history_started = time.monotonic()
        with stage_span("doubt_solver.persist_history"):
            created = _run_with_one_transient_retry(
                lambda: self.history_repository.save_completed_turn(turn)
            )
        history_ms = int((time.monotonic() - history_started) * 1000)
        history_status: PersistenceStatus = "idempotent_replay" if created is False else "succeeded"
        if self.session_repository is None:
            return history_status, "skipped", history_ms, 0

        session_started = time.monotonic()
        try:
            updated = _run_with_one_transient_retry(
                lambda: self.session_repository.upsert_from_completed_turn(turn)
            )
        except Exception as exc:  # noqa: BLE001
            return (
                history_status,
                _status_for_exception(exc),
                history_ms,
                int((time.monotonic() - session_started) * 1000),
            )
        session_ms = int((time.monotonic() - session_started) * 1000)
        session_status: PersistenceStatus = "idempotent_replay" if updated is False else "succeeded"
        return history_status, session_status, history_ms, session_ms

    def _persist_memory(self, turn: CompletedConversationTurn) -> tuple[PersistenceStatus, int]:
        if turn.response_type == "practice_generation":
            return "skipped", 0
        started = time.monotonic()
        with stage_span("doubt_solver.persist_memory"):
            _run_with_one_transient_retry(lambda: self.short_term_memory.save_completed_turn(turn))
        return "succeeded", int((time.monotonic() - started) * 1000)

    @staticmethod
    def _log_failures(result: ConversationPersistenceResult) -> None:
        for component, status in (
            ("history", result.history_write_status),
            ("session", result.session_write_status),
            ("memory", result.memory_write_status),
        ):
            if status.startswith("failed_"):
                logger.warning(
                    "%s_write_failure metric_count=1 status=%s "
                    "degraded_session_list_consistency=%s",
                    component,
                    status,
                    str(component == "session").lower(),
                )
        if result.memory_write_status == "succeeded" and result.history_write_status.startswith(
            "failed_"
        ):
            logger.warning("conversation_degraded_consistency metric_count=1")

    @staticmethod
    def _log_result(result: ConversationPersistenceResult) -> None:
        logger.debug(
            "conversation_persistence_result request_id=%s conversation_id=%s turn_id=%s "
            "persistable=%s history_write_status=%s session_write_status=%s "
            "memory_write_status=%s skip_reason=%s history_latency_ms=%d "
            "session_latency_ms=%d memory_latency_ms=%d",
            result.request_id,
            result.conversation_id,
            result.turn_id,
            str(result.persistable).lower(),
            result.history_write_status,
            result.session_write_status,
            result.memory_write_status,
            result.skip_reason or "",
            result.history_latency_ms,
            result.session_latency_ms,
            result.memory_latency_ms,
        )
        update_request_summary(
            history_write_status=result.history_write_status,
            session_write_status=result.session_write_status,
            memory_write_status=result.memory_write_status,
        )
        failed = any(
            status.startswith("failed_")
            for status in (
                result.history_write_status,
                result.session_write_status,
                result.memory_write_status,
            )
        )
        event = (
            "conversation_persistence_skipped"
            if not result.persistable
            else (
                "conversation_persistence_failed"
                if failed
                else "conversation_persistence_completed"
            )
        )
        log_event(
            event,
            component="conversation.persistence",
            stage="persist_history",
            status="skipped" if not result.persistable else ("failed" if failed else "completed"),
            duration_ms=max(
                result.history_latency_ms,
                result.session_latency_ms,
                result.memory_latency_ms,
            ),
            error_code="PERSISTENCE_PARTIAL_FAILURE" if failed else None,
            details={
                "history_write_status": result.history_write_status,
                "session_write_status": result.session_write_status,
                "memory_write_status": result.memory_write_status,
                "skip_reason": result.skip_reason,
            },
            level=logging.WARNING if failed else logging.INFO,
        )
