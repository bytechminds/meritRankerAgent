"""Adaptive verified streaming for the orchestrated doubt solver."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from traceback import extract_tb
from typing import Literal, Protocol

from config import get_settings
from features.practice_generation.agentcore_async import PracticeLaunchError
from features.practice_generation.planning import (
    PRACTICE_ASYNC_NOT_CONFIGURED,
    PracticeRequestCountError,
    decide_practice_launch,
    practice_route_enabled,
    resolve_practice_freshness_requirement,
    resolve_practice_request,
    validate_practice_requested_count,
)
from features.practice_generation.request_intelligence import PracticeRequestInterpreter
from features.practice_generation.schemas import (
    PracticeGenerationRequest,
    PracticeLaunchResult,
)
from graphs.doubt_solver_graph import (
    _ORCHESTRATED_FALLBACK_CLASSIFICATION,
    _orchestrated_collect_context_node,
    orchestrated_classify_query_stage,
    orchestrated_classify_query_with_delivery_signals,
)
from observability import (
    begin_request_summary,
    bind_execution_context,
    bind_llm_attempt_type,
    bind_request_context,
    count_generator_calls,
    current_request_summary,
    emit_request_summary,
    log_event,
    record_local_preview,
    snapshot_llm_usage_records,
    stage_span,
    update_request_summary,
    update_request_type,
)
from observability.summary import reset_request_summary
from retrieval.pattern_intelligence import DoubtPatternContext
from schemas.conversation import CompletedConversationTurn
from schemas.doubt_solver import (
    CanonicalLanguage,
    DoubtSolverFinalResponse,
    DoubtSolverStreamEvent,
    PracticeGenerationStartedData,
    ResponseContent,
    TerminalConversationIdentity,
)
from schemas.llm_usage import LLMUsageRecord
from services.context_retrieval.web_grounding import (
    log_grounding_status,
    required_web_answer_verified,
    required_web_context_verified,
    sanitize_required_web_answer,
    verification_limited_response,
)
from services.conversation.selected_context_builder import (
    build_context_aware_clarification,
    build_selected_generation_context,
)
from services.doubt_solver.answer_correctness import (
    requires_independent_correctness_verification,
)
from services.doubt_solver.answer_delivery_policy import (
    AnswerDeliveryPolicy,
    AnswerDeliverySignals,
)
from services.doubt_solver.answer_diagnosis import diagnose_verification_failure
from services.doubt_solver.answer_generation_adapter import AnswerGenerationAdapter
from services.doubt_solver.answer_quality import (
    AnswerQualityPolicy,
    is_presentation_only_failure,
    validate_answer_quality,
)
from services.doubt_solver.final_answer import build_final_answer_result
from services.doubt_solver.markdown_replay import iter_markdown_replay_chunks
from services.doubt_solver.question_integrity import (
    QUESTION_NEEDS_CLARIFICATION,
    ambiguous_question_message,
)
from services.doubt_solver.recovery_policy import (
    CANDIDATE_RECOVERY_INSTRUCTION,
    SERVICE_TEMPORARILY_UNAVAILABLE,
    RecoveryBudget,
    decide_verification_recovery,
    log_recovery_decision,
    recovery_failure_source,
    recovery_reason_code,
    shadow_generation_failure,
    shadow_quality_recovery,
    shadow_verification_recovery,
)
from services.doubt_solver.stream_labels import (
    LABEL_ANSWER_CONTINUATION,
    LABEL_CAREFUL_CLASSIFICATION,
    LABEL_GENERATOR_FALLBACK,
    LABEL_WEB_SEARCH,
    LABEL_WEB_SEARCH_RETRY,
    LABEL_WEB_SEARCH_WEAK,
    get_stream_label,
)
from services.doubt_solver.stream_status import StreamStatusTracker
from services.llm.billing import (
    OperationUsageAccumulator,
    begin_operation,
    emit_operation_billing_summary,
)
from services.student_credits.errors import InsufficientStudentCreditsError
from services.student_credits.runtime import (
    StudentCreditRuntime,
    doubt_reference_id,
)
from tools.web_search.models import FreshEvidenceBundle

logger = logging.getLogger(__name__)
__all__ = ["orchestrated_classify_query_with_delivery_signals"]

# `retryable` is the system's automatic retry/fallback semantics; `user_retryable`
# says whether the student may explicitly start a new attempt. Codes outside this
# set omit the field so clients keep deciding from `retryable` as before.
_USER_RETRYABLE_CODES = frozenset(
    {
        "ANSWER_VERIFICATION_FAILED",
        "ANSWER_QUALITY_FAILED",
        "ANSWER_PROVIDER_FAILED",
        # An outage is the terminal a student should retry soonest, and it says nothing
        # about their question or the answer.
        SERVICE_TEMPORARILY_UNAVAILABLE,
    }
)


def _stream_doubt_pattern_context(payload: object) -> DoubtPatternContext | None:
    """Validate the internal context before handing it to the prompt adapter."""
    if not isinstance(payload, dict):
        return None
    raw_context = payload.get("doubtPatternContext")
    if not isinstance(raw_context, dict):
        return None
    try:
        return DoubtPatternContext.model_validate(raw_context)
    except ValueError:
        return None


class PracticeLauncher(Protocol):
    def launch(self, request: PracticeGenerationRequest) -> PracticeLaunchResult: ...

    def start(self, test_id: str, delivery_id: str | None = None) -> bool: ...

    def abort(
        self,
        test_id: str,
        code: str,
        delivery_id: str | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class StreamDoubtSolverInput:
    """Safe input for orchestrated doubt solver streaming."""

    request_id: str
    query: str
    trace_id: str | None = None
    actor_id: str = "local-user"
    original_query: str = ""
    conversation_id: str = "local-conversation"
    turn_id: str = "local-turn"
    language: CanonicalLanguage = "english"
    classification: dict | None = None
    source_modality: Literal["text", "image"] = "text"
    image_confidence: float | None = None
    image_uncertain: bool = False
    classifier_confidence: float | None = None
    classifier_fallback: bool = False
    exam_id: str | None = None
    exam_stage: str | None = None
    exam_profile_id: str | None = None
    should_cancel: Callable[[], bool] | None = None
    cancellation_reason: Callable[[], str | None] | None = None
    request_started_logged: bool = False
    defer_practice_start_until_committed: bool = False
    initial_llm_usage_records: tuple[LLMUsageRecord, ...] = ()
    operation_accumulator: OperationUsageAccumulator | None = None
    # Admission runs in the invocation entrypoint, whose summary is discarded for
    # streaming responses. Carrying the outcome keeps the request report truthful.
    credit_admission: str | None = None
    credit_mode: str | None = None
    credit_balance: int | None = None
    # Entrypoint monotonic clock. The stream's own timer measures orchestration
    # only, so pre-admission work would otherwise be invisible.
    entrypoint_started_at: float | None = None


@dataclass
class _PostAnswerFinalizationProgress:
    """Request-local diagnostic state for the narrow post-answer boundary."""

    lifecycle_stage: str = "before_answer"
    answer_emitted: bool = False
    persistence_payload_build_started: bool = False
    persistence_network_call_started: bool = False


def _sanitized_exception_origin(exc: Exception) -> tuple[str, str, int, str]:
    """Return frame coordinates only; never include exception text or source lines."""
    frames = extract_tb(exc.__traceback__)
    if not frames:
        return "unknown", "unknown", 0, "unknown"
    origin = frames[-1]
    module = "/".join(Path(origin.filename).parts[-3:]) or "unknown"
    trace = " <- ".join(
        f"{'/'.join(Path(frame.filename).parts[-3:])}:{frame.name}:{frame.lineno}"
        for frame in frames[-8:]
    )
    return module, origin.name, origin.lineno, trace


def _post_answer_finalization_failure_details(
    exc: Exception,
    progress: _PostAnswerFinalizationProgress,
    *,
    operation_id: str,
) -> tuple[dict[str, object], str]:
    """Build bounded diagnostic metadata without serializing user-controlled exception text."""
    origin_module, origin_function, origin_line, trace = _sanitized_exception_origin(exc)
    return (
        {
            "operation_id": operation_id,
            "lifecycle_stage": progress.lifecycle_stage,
            "exception_type": type(exc).__name__,
            "exception_message_short": "exception message redacted",
            "origin_module": origin_module,
            "origin_function": origin_function,
            "origin_line": origin_line,
            "answer_emitted": progress.answer_emitted,
            "persistence_payload_build_started": progress.persistence_payload_build_started,
            "persistence_network_call_started": progress.persistence_network_call_started,
        },
        trace,
    )


def _cancelled(input: StreamDoubtSolverInput) -> bool:
    return input.should_cancel is not None and input.should_cancel()


def _error_event(
    request_id: str,
    *,
    code: str,
    retryable: bool,
    label: str | None = None,
    user_retryable: bool | None = None,
) -> DoubtSolverStreamEvent:
    metadata: dict[str, object] = {"retryable": retryable, "code": code}
    if user_retryable is not None:
        metadata["user_retryable"] = user_retryable
    elif code in _USER_RETRYABLE_CODES:
        metadata["user_retryable"] = True
    return DoubtSolverStreamEvent(
        type="error",
        request_id=request_id,
        stage="failed",
        label=label or "Unable to complete",
        metadata=metadata,
    )


def _record_recovery(
    decision: object,
    budget: RecoveryBudget,
    correctness: object,
    attempt: int,
    subject: str,
    difficulty: str,
    *,
    node: str,
    outcome: str,
) -> None:
    """Log one recovery the controller carried out, with its budget at that moment."""
    log_recovery_decision(
        decision,  # type: ignore[arg-type]
        applied=True,
        budget=budget.snapshot(),
        verifier_status=str(getattr(correctness, "status", "")),
        attempt_number=attempt,
        node=node,  # type: ignore[arg-type]
        subject=subject,
        difficulty=difficulty,
        outcome=outcome,
    )


def _terminal_metadata(
    input: StreamDoubtSolverInput,
    *,
    persisted: bool,
    terminal_reason: str | None = None,
) -> dict[str, object]:
    metadata = TerminalConversationIdentity(
        request_id=input.request_id,
        conversation_id=input.conversation_id,
        turn_id=input.turn_id,
        persisted=persisted,
    ).model_dump(mode="json")
    if terminal_reason is not None:
        metadata["terminal_reason"] = terminal_reason
    return metadata


def _history_persisted(result: object | None) -> bool:
    return bool(
        result is not None
        and getattr(result, "history_write_status", None)
        in {"succeeded", "idempotent_replay"}
    )


def _iter_stream_doubt_solver(
    input: StreamDoubtSolverInput,
    *,
    adapter: AnswerGenerationAdapter,
    conversation_persistence=None,
    follow_up_resolver=None,
    conversation_understanding=None,
    practice_launcher: PracticeLauncher | None = None,
    practice_request_interpreter: PracticeRequestInterpreter | None = None,
    student_credits: StudentCreditRuntime | None = None,
    post_answer_progress: _PostAnswerFinalizationProgress | None = None,
) -> Iterator[DoubtSolverStreamEvent]:
    """Yield live chunks only for low risk requests, otherwise replay approval."""
    request_id = input.request_id
    query = input.query
    record_local_preview("mode", "doubt_solver")
    record_local_preview("language", input.language)
    record_local_preview("exam_id", input.exam_id)
    record_local_preview("query", input.original_query or input.query)
    started_at = time.monotonic()
    status_tracker = StreamStatusTracker(request_id=request_id)

    def emit_status(
        *,
        stage: str,
        intent: str | None = None,
        label: str | None = None,
        reason_code: str = "stage_progress",
    ) -> DoubtSolverStreamEvent | None:
        return status_tracker.emit_direct(
            stage=stage,
            label=label or get_stream_label(stage, intent),
            reason_code=reason_code,
        )

    def emit_pending_statuses() -> Iterator[DoubtSolverStreamEvent]:
        yield from status_tracker.pending_events
        status_tracker.pending_events.clear()

    logger.debug("answer_delivery request_id=%s stream_started=true", request_id)
    event = emit_status(stage="understanding", reason_code="stream_started")
    if event is not None:
        yield event
    if _cancelled(input):
        return

    conversation_preparation = None
    conversation_context = ""
    if conversation_understanding is not None and input.source_modality == "text":
        conversation_preparation = conversation_understanding.understand(
            actor_id=input.actor_id,
            conversation_id=input.conversation_id,
            query=query,
            request_id=request_id,
            exam_id=input.exam_id,
            language=input.language,
        )

    state = {
        "request_id": request_id,
        "query": query,
        "original_query": input.original_query or query,
        "actor_id": input.actor_id,
        "language": input.language,
        "exam_id": input.exam_id,
        "exam_stage": input.exam_stage,
        "exam_profile_id": input.exam_profile_id,
        "classification": None,
        "retrieval_context": {},
        "context_text": "",
        "answer": None,
    }
    careful_status_pending = False

    def on_before_strong_classifier() -> None:
        nonlocal careful_status_pending
        careful_status_pending = True

    classification_started = time.monotonic()
    with stage_span("doubt_solver.classify"):
        classification_dict = input.classification
        raw_classification = None
        if classification_dict is None:
            stage_result = orchestrated_classify_query_stage(
                query=query,
                request_id=request_id,
                on_before_strong_classifier=on_before_strong_classifier,
                conversation=conversation_preparation,
            )
            classification_dict = stage_result.classification
            raw_classification = stage_result.raw
            classifier_confidence = stage_result.classifier_confidence
            classifier_fallback = stage_result.classifier_fallback
        else:
            classifier_confidence = input.classifier_confidence
            classifier_fallback = input.classifier_fallback

    if raw_classification is not None and conversation_preparation is not None:
        # The working query is the student's text, or that text plus its repaired
        # formatting when the request boundary normalized a malformed question.
        selected_context = build_selected_generation_context(
            current_query=query,
            classification=raw_classification,
            preparation=conversation_preparation,
        )
        if selected_context.clarification_required:
            clarification = build_context_aware_clarification(
                conversation_preparation,
                language=input.language,
                current_query=input.original_query or query,
            )
            record_local_preview("response_type", "clarification")
            record_local_preview("response", clarification)
            final = build_final_answer_result(
                content=clarification,
                language=input.language,
                quality_status="failed_quality_gate",
            )
            if conversation_persistence is not None:
                conversation_persistence.record_skip(
                    request_id=request_id,
                    conversation_id=input.conversation_id,
                    turn_id=input.turn_id,
                    skip_reason="clarification_response",
                )
            yield DoubtSolverStreamEvent(
                type="chunk",
                request_id=request_id,
                content=clarification,
            )
            yield DoubtSolverStreamEvent(
                type="complete",
                request_id=request_id,
                stage="complete",
                label=get_stream_label("complete"),
                metadata=_terminal_metadata(
                    input,
                    persisted=False,
                    terminal_reason="clarification_required",
                ),
                response=DoubtSolverFinalResponse(
                    request_id=request_id,
                    content=ResponseContent(value=clarification),
                    answer=clarification,
                    final_answer=final,
                ),
            )
            return
        query = selected_context.resolved_query
        conversation_context = selected_context.conversation_context
        state["query"] = query
        log_event(
            "selected_generation_context_built",
            component="conversation.selected_context",
            stage="prepare_follow_up",
            status="completed",
            details={
                "relation": selected_context.relation,
                "action": selected_context.requested_action,
                "selected_turn_id": selected_context.selected_turn_id,
                "context_policy": selected_context.context_policy,
                "context_characters": selected_context.context_characters,
            },
        )

    state["classification"] = classification_dict
    request_type = (
        "image"
        if input.source_modality == "image"
        else (
            "follow_up"
            if raw_classification is not None and raw_classification.relation != "NEW_QUESTION"
            else "standalone"
        )
    )
    update_request_type(request_type)
    update_request_summary(
        request_type=request_type,
        subject=str(classification_dict.get("subject") or "general"),
        intent=str(classification_dict.get("intent") or "explain"),
        difficulty=str(classification_dict.get("difficulty") or "default"),
        classifier_source=(
            "fallback"
            if classifier_fallback
            else str(classification_dict.get("classification_source") or "llm")
        ),
        strong_classifier_used=careful_status_pending,
        context_required=request_type == "follow_up",
    )
    log_event(
        "classification_completed",
        component="doubt_solver.classifier",
        stage="understanding",
        status="completed",
        duration_ms=int((time.monotonic() - classification_started) * 1000),
        details={
            "subject": classification_dict.get("subject"),
            "intent": classification_dict.get("intent"),
            "difficulty": classification_dict.get("difficulty"),
            "classifier_source": "fallback" if classifier_fallback else "llm",
            "strong_classifier_used": careful_status_pending,
            "relation": (
                raw_classification.relation if raw_classification is not None else "NEW_QUESTION"
            ),
            "action": (
                raw_classification.requested_action
                if raw_classification is not None
                else "ANSWER_CURRENT"
            ),
            "selected_turn_id": (
                raw_classification.selected_turn_id if raw_classification is not None else None
            ),
            "confidence": classifier_confidence,
            "need_web_search": bool(classification_dict.get("need_web_search")),
            "web_search_reason": classification_dict.get("web_search_reason"),
            "search_term_present": bool(classification_dict.get("web_search_query")),
        },
    )
    if classifier_fallback:
        log_event(
            "classification_fallback_used",
            component="doubt_solver.classifier",
            stage="understanding",
            status="fallback",
            details={"reason": "classifier_policy"},
        )
    if careful_status_pending:
        event = emit_status(
            stage="understanding",
            label=LABEL_CAREFUL_CLASSIFICATION,
            reason_code="classifier_fallback",
        )
        if event is not None:
            yield event
    if _cancelled(input):
        return
    if request_type == "standalone":
        logger.debug("standalone_request_count count=1")
    elif request_type == "follow_up":
        logger.debug("follow_up_request_count count=1")

    practice_decision = decide_practice_launch(query, classification_dict)
    freshness_requirement = resolve_practice_freshness_requirement(
        input.original_query or query,
        classification_dict,
    )
    if practice_route_enabled() and classification_dict.get("intent") == "practice":
        log_event(
            "PRACTICE_LAUNCH_DECISION",
            component="practice.routing",
            stage="route",
            status="eligible" if practice_decision.eligible else "ineligible",
            details={
                "eligible": practice_decision.eligible,
                "reasonCode": practice_decision.reason_code,
                "requestedArtifact": practice_decision.requested_artifact,
            },
        )
    if practice_route_enabled() and practice_decision.eligible:
        log_event(
            "practice_request_classified",
            component="practice.routing",
            stage="route",
            status="eligible",
            details={
                "requestedCountPresent": bool(practice_decision.requested_artifact),
                "language": input.language,
            },
        )
        if practice_launcher is None:
            if conversation_persistence is not None:
                conversation_persistence.record_skip(
                        request_id=request_id,
                        conversation_id=input.conversation_id,
                        turn_id=input.turn_id,
                        skip_reason="failed_quality_gate",
                )
            log_event(
                "PRACTICE_LAUNCH_DECISION",
                component="practice.routing",
                stage="route",
                status="disabled",
                error_code=PRACTICE_ASYNC_NOT_CONFIGURED,
            )
            yield _error_event(
                request_id,
                code=PRACTICE_ASYNC_NOT_CONFIGURED,
                retryable=False,
            )
            return
        fresh_evidence = None
        try:
            validate_practice_requested_count(input.original_query or input.query)
        except PracticeRequestCountError as exc:
            if conversation_persistence is not None:
                conversation_persistence.record_skip(
                    request_id=request_id,
                    conversation_id=input.conversation_id,
                    turn_id=input.turn_id,
                    skip_reason="failed_quality_gate",
                )
            yield _error_event(request_id, code=exc.reason_code, retryable=False)
            return
        if freshness_requirement.requires_fresh_evidence:
            context_update = _orchestrated_collect_context_node(
                state,
                on_before_web_search=status_tracker.hook(
                    stage="thinking",
                    label=LABEL_WEB_SEARCH,
                    reason_code="web_search_started",
                ),
                on_web_search_retry=status_tracker.hook(
                    stage="thinking",
                    label=LABEL_WEB_SEARCH_RETRY,
                    reason_code="web_search_retry_sources",
                ),
                on_web_search_weak_context=status_tracker.hook(
                    stage="thinking",
                    label=LABEL_WEB_SEARCH_WEAK,
                    reason_code="web_search_weak_context",
                ),
            )
            state.update(context_update)
            yield from emit_pending_statuses()
            raw_fresh_evidence = state.get("fresh_evidence")
            if isinstance(raw_fresh_evidence, dict):
                try:
                    fresh_evidence = FreshEvidenceBundle.model_validate(raw_fresh_evidence)
                except ValueError:
                    fresh_evidence = None
            if fresh_evidence is None:
                if conversation_persistence is not None:
                    conversation_persistence.record_skip(
                        request_id=request_id,
                        conversation_id=input.conversation_id,
                        turn_id=input.turn_id,
                        skip_reason="failed_quality_gate",
                    )
                yield _error_event(
                    request_id,
                    code="PRACTICE_FRESH_EVIDENCE_UNAVAILABLE",
                    retryable=False,
                )
                return
        try:
            practice_request = resolve_practice_request(
                request_id=request_id,
                user_id=input.actor_id,
                conversation_id=input.conversation_id,
                turn_id=input.turn_id,
                query=input.original_query or input.query,
                subject=str(classification_dict.get("subject") or "general"),
                topic=(
                    str(classification_dict["topic"]) if classification_dict.get("topic") else None
                ),
                difficulty=str(classification_dict.get("difficulty") or "default"),
                language=input.language,
                exam_id=input.exam_id,
                exam_stage=input.exam_stage,
                exam_profile_id=input.exam_profile_id,
                source_question_reference=(
                    raw_classification.selected_turn_id if raw_classification is not None else None
                ),
                freshness_requirement=freshness_requirement,
                fresh_evidence=fresh_evidence,
                request_interpreter=practice_request_interpreter,
            )
            log_event(
                "practice_language_resolved",
                component="practice.routing",
                stage="language",
                status="resolved",
                details={
                    "language": practice_request.language,
                    "source": practice_request.language_source,
                },
            )
            launch = practice_launcher.launch(practice_request)
        except PracticeLaunchError as exc:
            if conversation_persistence is not None:
                conversation_persistence.record_skip(
                    request_id=request_id,
                    conversation_id=input.conversation_id,
                    turn_id=input.turn_id,
                    skip_reason="failed_quality_gate",
                )
            yield _error_event(request_id, code=exc.code, retryable=False)
            return
        except PracticeRequestCountError as exc:
            yield _error_event(request_id, code=exc.reason_code, retryable=False)
            return
        except (TypeError, ValueError):
            yield _error_event(
                request_id,
                code="INVALID_PRACTICE_REQUEST",
                retryable=False,
            )
            return

        final_answer = build_final_answer_result(
            content=launch.message,
            language=input.language,
            quality_status="checked",
        )
        record_local_preview("response_type", "practice_generation")
        record_local_preview("response", launch.message)
        persistence_result = None
        if conversation_persistence is not None:
            turn = CompletedConversationTurn.now(
                actor_id=input.actor_id,
                conversation_id=input.conversation_id,
                turn_id=input.turn_id,
                original_query=input.original_query or input.query,
                final_answer=launch.message,
                exam_id=input.exam_id,
                exam_stage=input.exam_stage,
                language=input.language,
                subject=str(classification_dict.get("subject") or "general"),
                topic=(
                    str(classification_dict["topic"]) if classification_dict.get("topic") else None
                ),
                quality_status=final_answer.quality_status,
                was_regenerated=False,
                response_type="practice_generation",
                practice_test_id=launch.test_id,
            )
            persistence_result = conversation_persistence.persist_completed_turn(
                turn,
                final_answer,
                request_id=request_id,
            )
        history_linked = (
            persistence_result is not None
            and persistence_result.history_write_status in {"succeeded", "idempotent_replay"}
        )
        if not history_linked:
            practice_launcher.abort(
                launch.test_id,
                "PRACTICE_CONVERSATION_LINKAGE_FAILED",
                request_id,
            )
            yield _error_event(
                request_id,
                code="PRACTICE_CONVERSATION_LINKAGE_FAILED",
                retryable=False,
            )
            return
        if not input.defer_practice_start_until_committed:
            try:
                practice_launcher.start(launch.test_id, request_id)
            except PracticeLaunchError as exc:
                yield _error_event(request_id, code=exc.code, retryable=False)
                return
        yield DoubtSolverStreamEvent(
            type="practice_generation_started",
            request_id=request_id,
            stage="practice_generation_started",
            label=launch.message,
            metadata=_terminal_metadata(input, persisted=history_linked),
            data=PracticeGenerationStartedData(
                practice_test_id=launch.test_id,
                message=launch.message,
            ),
        )
        return

    event = emit_status(stage="thinking", reason_code="thinking")
    if event is not None:
        yield event
    event = emit_status(stage="retrieving", reason_code="retrieval_started")
    if event is not None:
        yield event
    need_web_search = bool(classification_dict.get("need_web_search"))
    with stage_span("doubt_solver.retrieve"):
        context_update = _orchestrated_collect_context_node(
            state,
            on_before_web_search=status_tracker.hook(
                stage="thinking",
                label=LABEL_WEB_SEARCH,
                reason_code="web_search_started",
            ),
            on_web_search_retry=status_tracker.hook(
                stage="thinking",
                label=LABEL_WEB_SEARCH_RETRY,
                reason_code="web_search_retry_sources",
            ),
            on_web_search_weak_context=(
                status_tracker.hook(
                    stage="thinking",
                    label=LABEL_WEB_SEARCH_WEAK,
                    reason_code="web_search_weak_context",
                )
                if need_web_search
                else None
            ),
        )
    state.update(context_update)
    yield from emit_pending_statuses()
    if _cancelled(input):
        return

    classification_dict = (
        state.get("classification") or _ORCHESTRATED_FALLBACK_CLASSIFICATION.copy()
    )
    subject = str(classification_dict.get("subject", "general"))
    intent = str(classification_dict.get("intent", "explain"))
    difficulty = str(classification_dict.get("difficulty", "default"))
    requested_action = (
        raw_classification.requested_action
        if raw_classification is not None
        else str(classification_dict.get("requested_action") or "ANSWER_CURRENT")
    )
    context_need = (
        conversation_preparation.gate.decision
        if conversation_preparation is not None
        else None
    )
    correctness_verification_required = requires_independent_correctness_verification(
        subject=subject,
        difficulty=difficulty,
        intent=intent,
        requested_action=requested_action,
        context_need=context_need,
    )
    context_text = str(state.get("context_text") or "")
    web_verified = required_web_context_verified(classification_dict, state)
    retrieval_context = state.get("retrieval_context") or {}
    doubt_pattern_context = _stream_doubt_pattern_context(retrieval_context)
    retrieval_mode = retrieval_context.get("mode")
    retrieval_used = retrieval_mode not in {None, "fresh_solve"}
    policy = AnswerDeliveryPolicy.from_settings()
    decision = policy.decide(
        AnswerDeliverySignals(
            classifier_confidence=classifier_confidence,
            classifier_fallback=classifier_fallback,
            source_modality=input.source_modality,
            image_confidence=input.image_confidence,
            image_uncertain=input.image_uncertain,
            difficulty=difficulty,
            current_fact_dependency=need_web_search,
            retrieval_used=retrieval_used,
            retrieval_mode=retrieval_mode,
            pattern_confidence=retrieval_context.get("confidence"),
            pattern_candidate_conflict=bool(retrieval_context.get("materialConflicts") or []),
            provider_fallback=False,
            needs_review=correctness_verification_required,
            language=input.language,
        )
    )
    logger.debug(
        "answer_delivery request_id=%s strategy=%s risk=%s reasons=%s",
        request_id,
        decision.strategy,
        decision.risk_level,
        ",".join(decision.reason_codes),
    )

    event = emit_status(stage="generating", intent=intent, reason_code="generating")
    if event is not None:
        yield event
    if _cancelled(input):
        return

    answer = ""
    verification = None
    repair_attempted = False
    presentation_only_continued = False
    generation_started = time.monotonic()
    if not web_verified:
        answer = verification_limited_response(input.language)
        log_grounding_status(classification_dict, state, verified=False)
        yield DoubtSolverStreamEvent(
            type="chunk",
            request_id=request_id,
            content=answer,
        )
    elif decision.strategy == "live_stream":
        answer_parts: list[str] = []
        visible_answer_emitted = False
        try:
            stream_kwargs = {
                "request_id": request_id,
                "query": query,
                "subject": subject,
                "intent": intent,
                "difficulty": difficulty,
                "context_text": context_text,
                "web_search_reason": (
                    str(classification_dict["web_search_reason"])
                    if classification_dict.get("web_search_reason")
                    else None
                ),
                "on_before_generator_fallback": status_tracker.hook(
                    stage="generating",
                    label=LABEL_GENERATOR_FALLBACK,
                    reason_code="generator_fallback",
                ),
                "on_before_continuation": status_tracker.hook(
                    stage="generating",
                    label=LABEL_ANSWER_CONTINUATION,
                    reason_code="answer_continuation",
                ),
                "verify_before_stream": False,
                "exam_id": input.exam_id,
                "exam_stage": input.exam_stage,
                "language": input.language,
                "conversation_context": conversation_context or None,
            }
            if input.exam_profile_id:
                stream_kwargs["exam_profile_id"] = input.exam_profile_id
            if doubt_pattern_context is not None:
                stream_kwargs["doubt_pattern_context"] = doubt_pattern_context
            for chunk in adapter.generate_stream(**stream_kwargs):
                yield from emit_pending_statuses()
                if _cancelled(input):
                    return
                if not chunk:
                    continue
                answer_parts.append(chunk)
                if chunk.strip():
                    visible_answer_emitted = True
                    logger.debug(
                        "answer_chunk_emission request_id=%s strategy=live_stream "
                        "first_visible_chunk_emitted=true chunk_chars=%d",
                        request_id,
                        len(chunk),
                    )
                yield DoubtSolverStreamEvent(type="chunk", request_id=request_id, content=chunk)
        except Exception:  # noqa: BLE001
            logger.warning(
                "answer_delivery request_id=%s stage=live_stream terminal_reason=%s",
                request_id,
                (
                    "provider_failed_after_content"
                    if visible_answer_emitted
                    else "provider_failed_before_content"
                ),
            )
            yield _error_event(
                request_id,
                code=(
                    "ANSWER_PARTIAL_STREAM_FAILED"
                    if visible_answer_emitted
                    else "ANSWER_PROVIDER_FAILED"
                ),
                retryable=not visible_answer_emitted,
            )
            return
        answer = "".join(answer_parts)
    else:
        try:
            generation_kwargs = {
                "request_id": request_id,
                "query": query,
                "subject": subject,
                "intent": intent,
                "difficulty": difficulty,
                "context": context_text,
                "web_search_reason": (
                    str(classification_dict["web_search_reason"])
                    if classification_dict.get("web_search_reason")
                    else None
                ),
                "exam_id": input.exam_id,
                "exam_stage": input.exam_stage,
                "language": input.language,
                "conversation_context": conversation_context or None,
            }
            if input.exam_profile_id:
                generation_kwargs["exam_profile_id"] = input.exam_profile_id
            if doubt_pattern_context is not None:
                generation_kwargs["doubt_pattern_context"] = doubt_pattern_context
            draft = adapter.generate(**generation_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "answer_delivery request_id=%s stage=private_generate terminal_reason=%s",
                request_id,
                "provider_failed_before_content",
            )
            shadow_generation_failure(
                exc,
                actual_terminal_code="ANSWER_PROVIDER_FAILED",
            )
            yield _error_event(
                request_id,
                code="ANSWER_PROVIDER_FAILED",
                retryable=True,
            )
            return
        if _cancelled(input):
            return

        event = emit_status(stage="verifying", reason_code="verification_started")
        if event is not None:
            yield event
        logger.debug("verification_started request_id=%s stage=verifying", request_id)
        if _cancelled(input):
            return
        settings = get_settings()
        try:
            with stage_span("doubt_solver.validate_quality"):
                verification = validate_answer_quality(
                    draft,
                    subject=subject,
                    intent=intent,
                    difficulty=difficulty,
                    query=query,
                    language=input.language,
                    policy=AnswerQualityPolicy.from_settings(settings),
                )
        except Exception:  # noqa: BLE001
            yield _error_event(
                request_id,
                code="ANSWER_VERIFICATION_FAILED",
                retryable=True,
            )
            return
        logger.debug(
            "verification_completed request_id=%s stage=verifying",
            request_id,
        )
        if _cancelled(input):
            return
        correctness_verifier = getattr(adapter, "correctness_verifier", None)
        if settings.answer_verifier_enabled and not verification.is_valid:
            if settings.answer_verifier_max_repair_attempts == 1 and count_generator_calls() < 2:
                repair_attempted = True
                logger.debug("repair_started request_id=%s stage=verifying", request_id)
                if _cancelled(input):
                    return
                try:
                    with bind_llm_attempt_type("repair"):
                        draft = adapter.generate(**generation_kwargs)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "answer_delivery request_id=%s stage=repair terminal_reason=%s",
                        request_id,
                        "repair_failed",
                    )
                    shadow_generation_failure(
                        exc,
                        actual_terminal_code="ANSWER_REPAIR_FAILED",
                    )
                    yield _error_event(
                        request_id,
                        code="ANSWER_REPAIR_FAILED",
                        retryable=True,
                    )
                    return
                if _cancelled(input):
                    return
                try:
                    with stage_span("doubt_solver.validate_quality", repair=True):
                        verification = validate_answer_quality(
                            draft,
                            subject=subject,
                            intent=intent,
                            difficulty=difficulty,
                            query=query,
                            language=input.language,
                            policy=AnswerQualityPolicy.from_settings(settings),
                        )
                except Exception:  # noqa: BLE001
                    yield _error_event(
                        request_id,
                        code="ANSWER_VERIFICATION_FAILED",
                        retryable=True,
                    )
                    return
                logger.debug("repair_completed request_id=%s stage=verifying", request_id)
                if _cancelled(input):
                    return
            # After the single rewrite, an answer rejected only for presentation is
            # still judged on correctness when the verifier runs for this request;
            # without that verifier it fails exactly as before.
            presentation_only_continued = (
                not verification.is_valid
                and is_presentation_only_failure(verification)
                and correctness_verification_required
                and correctness_verifier is not None
                and any(
                    record.attempt_type == "rewrite"
                    for record in snapshot_llm_usage_records()
                )
            )
            if presentation_only_continued:
                log_event(
                    "QUALITY_PRESENTATION_ONLY_CONTINUED",
                    component="doubt_solver.quality",
                    stage="validate_quality",
                    status="continued",
                    details={"reason_codes": ",".join(sorted(set(verification.reason_codes)))},
                )
            elif not verification.is_valid:
                logger.warning(
                    "answer_delivery request_id=%s stage=verification approved=false "
                    "repair=%s generator_calls=%d",
                    request_id,
                    repair_attempted,
                    count_generator_calls(),
                )
                # The correctness verifier runs only after this block, so naming it
                # here blamed a stage that never executed.
                shadow_quality_recovery(
                    verification,
                    actual_terminal_code="ANSWER_QUALITY_FAILED",
                )
                yield _error_event(
                    request_id,
                    code="ANSWER_QUALITY_FAILED",
                    retryable=False,
                )
                return
        if _cancelled(input):
            return
        if correctness_verification_required and correctness_verifier is not None:
            budget = RecoveryBudget()
            approved_for_delivery = False
            regenerated = False
            # Each recovery is single-use in the ledger, so at most three verifications can
            # run: the first, one verifier-local retry, and one after a regenerated
            # candidate. The range is a backstop, never the real bound.
            for attempt in range(1, 4):
                correctness = correctness_verifier.verify(
                    request_id=request_id,
                    query=query,
                    candidate_answer=verification.sanitized_text or draft,
                    subject=subject,
                    difficulty=difficulty,
                    language=input.language,
                )
                if _cancelled(input):
                    return
                if correctness.approved and (verification.is_valid or presentation_only_continued):
                    approved_for_delivery = True
                    break
                verification_error_code = (
                    "ANSWER_VERIFICATION_UNAVAILABLE"
                    if correctness.status == "unavailable"
                    else "ANSWER_VERIFICATION_FAILED"
                )
                if correctness.approved:
                    # The verifier approved text the quality gate rejected: a quality
                    # question, not a verification one, and already bounded elsewhere.
                    shadow_quality_recovery(
                        verification,
                        actual_terminal_code=verification_error_code,
                    )
                    yield _error_event(request_id, code=verification_error_code, retryable=False)
                    return
                # After a regeneration the outcome is settled either way: exactly one
                # verification follows it and no second candidate may be generated, so the
                # paid diagnosis is skipped. The verdict's own reason still decides, which
                # keeps an outage an outage and a technical failure retryable once.
                diagnosis = None if regenerated else diagnose_verification_failure(
                    adapter,
                    request_id=request_id,
                    query=query,
                    candidate_answer=verification.sanitized_text or draft,
                    verdict=correctness.status,
                    verification=correctness,
                    method=correctness.method,
                    language=input.language,
                )
                if not settings.answer_recovery_enabled:
                    shadow_verification_recovery(
                        correctness,
                        actual_terminal_code=verification_error_code,
                        diagnosis=diagnosis,
                    )
                    yield _error_event(request_id, code=verification_error_code, retryable=False)
                    return
                recovery_decision = decide_verification_recovery(
                    approved=False,
                    reason_code=recovery_reason_code(correctness, diagnosis),
                    budget=budget.snapshot(),
                    failure_source=recovery_failure_source(correctness, diagnosis),
                )

                # The slot is claimed before the work starts, so an exception on the way
                # can never hand the same recovery back to this request.
                if (
                    recovery_decision.action == "RETRY_SAME_NODE"
                    and budget.take_verifier_technical_retry()
                ):
                    # Same candidate, same upstream results: only verification runs again.
                    _record_recovery(
                        recovery_decision,
                        budget,
                        correctness,
                        attempt,
                        subject,
                        difficulty,
                        node="verifier",
                        outcome="verifier_retried",
                    )
                    continue
                if (
                    recovery_decision.action == "REGENERATE_CANDIDATE"
                    and budget.take_candidate_recovery()
                ):
                    _record_recovery(
                        recovery_decision,
                        budget,
                        correctness,
                        attempt,
                        subject,
                        difficulty,
                        node="generator",
                        outcome="candidate_regenerated",
                    )
                    try:
                        with bind_llm_attempt_type("repair"):
                            draft = adapter.generate(
                                **generation_kwargs,
                                recovery_instruction=CANDIDATE_RECOVERY_INSTRUCTION,
                            )
                    except Exception as exc:  # noqa: BLE001
                        shadow_generation_failure(exc, actual_terminal_code="ANSWER_REPAIR_FAILED")
                        yield _error_event(request_id, code="ANSWER_REPAIR_FAILED", retryable=True)
                        return
                    if _cancelled(input):
                        return
                    verification = validate_answer_quality(
                        draft,
                        subject=subject,
                        intent=intent,
                        difficulty=difficulty,
                        query=query,
                        language=input.language,
                        policy=AnswerQualityPolicy.from_settings(settings),
                    )
                    # The frozen presentation-only rule applies to this candidate too: a
                    # rewrite has already run for this request, the verifier is about to
                    # judge it, and layout alone never becomes a correctness failure.
                    regenerated = True
                    presentation_only_continued = not verification.is_valid and (
                        is_presentation_only_failure(verification)
                        and any(
                            record.attempt_type.startswith("rewrite")
                            for record in snapshot_llm_usage_records()
                        )
                    )
                    if not verification.is_valid and not presentation_only_continued:
                        # A fresh candidate earns no trust: it faces the same gate, and its
                        # own recovery capacity is already spent.
                        shadow_quality_recovery(
                            verification, actual_terminal_code="ANSWER_QUALITY_FAILED"
                        )
                        yield _error_event(
                            request_id, code="ANSWER_QUALITY_FAILED", retryable=False
                        )
                        return
                    continue
                _record_recovery(
                    recovery_decision,
                    budget,
                    correctness,
                    attempt,
                    subject,
                    difficulty,
                    node="verifier",
                    outcome="terminal",
                )
                if recovery_decision.action == "ASK_CLARIFICATION":
                    yield _error_event(
                        request_id,
                        code=QUESTION_NEEDS_CLARIFICATION,
                        retryable=False,
                        user_retryable=False,
                        label=ambiguous_question_message(
                            input.language,
                            multiple_answers=recovery_decision.reason_code
                            == "MULTIPLE_DEFENSIBLE_ANSWERS",
                        ),
                    )
                    return
                if recovery_decision.terminal_code == SERVICE_TEMPORARILY_UNAVAILABLE:
                    # The executor already walked the configured provider chain; this is
                    # an outage, not a wrong answer.
                    yield _error_event(
                        request_id, code=SERVICE_TEMPORARILY_UNAVAILABLE, retryable=True
                    )
                    return
                yield _error_event(
                    request_id,
                    code=recovery_decision.terminal_code or verification_error_code,
                    retryable=False,
                )
                return
            if not approved_for_delivery:
                # Unreachable while every budget is single-use, and it fails closed if that
                # ever stops being true: an unapproved answer must never be delivered.
                logger.error(
                    "answer_delivery request_id=%s stage=verification terminal_reason=%s",
                    request_id,
                    "recovery_attempts_exhausted",
                )
                yield _error_event(
                    request_id, code="ANSWER_VERIFICATION_FAILED", retryable=False
                )
                return
        if (
            not presentation_only_continued
            and not verification.is_valid
            and is_presentation_only_failure(verification)
        ):
            # Reached only when the quality block above is disabled: text the gate
            # rejected is kept solely for the correctness verifier and is never replayed.
            yield _error_event(
                request_id,
                code="ANSWER_QUALITY_FAILED",
                retryable=False,
            )
            return
        answer = verification.sanitized_text or draft
        grounded_answer = sanitize_required_web_answer(
            classification_dict,
            state,
            answer,
        )
        if grounded_answer is not None:
            answer = grounded_answer
        answer_grounded = grounded_answer is not None and required_web_answer_verified(
            classification_dict,
            state,
            answer,
        )
        if not answer_grounded:
            answer = verification_limited_response(input.language)
        log_grounding_status(
            classification_dict,
            state,
            verified=answer_grounded,
            answer=answer,
        )
        replay_chunks = list(
            iter_markdown_replay_chunks(
                answer,
                max_chunk_chars=settings.answer_replay_max_chunk_chars,
            )
        )
        if "".join(replay_chunks) != answer:
            logger.error(
                "answer_delivery request_id=%s stage=replay terminal_reason=%s",
                request_id,
                "replay_integrity_failed",
            )
            yield _error_event(
                request_id,
                code="ANSWER_REPLAY_FAILED",
                retryable=True,
            )
            return
        logger.debug("replay_started request_id=%s stage=verifying", request_id)
        for chunk in replay_chunks:
            if _cancelled(input):
                return
            if chunk:
                logger.debug(
                    "answer_chunk_emission request_id=%s strategy=verified_replay "
                    "first_visible_chunk_emitted=true chunk_chars=%d",
                    request_id,
                    len(chunk),
                )
                yield DoubtSolverStreamEvent(type="chunk", request_id=request_id, content=chunk)
        logger.debug("replay_completed request_id=%s stage=verifying", request_id)

    if web_verified and decision.strategy == "live_stream":
        answer_grounded = required_web_answer_verified(
            classification_dict,
            state,
            answer,
        )
        log_grounding_status(
            classification_dict,
            state,
            verified=answer_grounded,
            answer=answer,
        )
    if not answer.strip():
        yield _error_event(
            request_id,
            code="ANSWER_STREAM_FAILED",
            retryable=True,
        )
        return
    generation_duration_ms = int((time.monotonic() - generation_started) * 1000)
    summary = current_request_summary()
    update_request_summary(
        generation_route=(
            summary.generation_route if summary and summary.generation_route else decision.strategy
        ),
        generation_duration_ms=generation_duration_ms,
    )
    log_event(
        (
            "generation_completed"
            if decision.strategy == "live_stream" and web_verified
            else "answer_delivery_completed"
        ),
        component=(
            "doubt_solver.generator"
            if decision.strategy == "live_stream"
            else "doubt_solver.delivery"
        ),
        stage="generating",
        status="completed",
        duration_ms=generation_duration_ms,
        details={
            "route": decision.strategy if web_verified else "verification_limited",
            "web_context_received": bool(context_text) and web_verified,
            "web_context_characters": len(context_text) if web_verified else 0,
            "retrieval_context_characters": 0 if web_verified else len(context_text),
            "conversation_context_characters": len(conversation_context or ""),
        },
    )
    if _cancelled(input):
        return
    final_quality_policy = AnswerQualityPolicy.from_settings(get_settings())
    if verification is None:
        with stage_span("doubt_solver.validate_quality"):
            verification = validate_answer_quality(
                answer,
                subject=subject,
                intent=intent,
                difficulty=difficulty,
                query=query,
                language=input.language,
                policy=final_quality_policy,
            )
    if not final_quality_policy.validation_enabled or presentation_only_continued:
        # A presentation-only answer reaches here only after the correctness
        # verifier approved it; "checked" is the existing accepted, non-passed status.
        quality_status = "checked"
    elif verification.is_valid:
        quality_status = "passed_quality_gate"
    else:
        quality_status = "failed_quality_gate"
    final_answer = build_final_answer_result(
        content=answer,
        language=input.language,
        quality_status=quality_status,
        was_regenerated=repair_attempted,
    )
    update_request_summary(
        quality_status=final_answer.quality_status,
        repair_attempted=repair_attempted,
    )
    log_event(
        "quality_validation_completed",
        component="doubt_solver.quality",
        stage="validate_quality",
        status=final_answer.quality_status,
        error_code=(
            "QUALITY_GATE_FAILED" if final_answer.quality_status == "failed_quality_gate" else None
        ),
        details={
            "repair_attempted": repair_attempted,
            "language_compliant": final_answer.language_compliant,
        },
    )
    if repair_attempted:
        log_event(
            "quality_repair_completed",
            component="doubt_solver.quality",
            stage="validate_quality",
            status=final_answer.quality_status,
            error_code=(
                "QUALITY_REPAIR_FAILED"
                if final_answer.quality_status == "failed_quality_gate"
                else None
            ),
        )
    if final_answer.quality_status == "failed_quality_gate":
        if conversation_persistence is not None:
            conversation_persistence.record_skip(
                request_id=request_id,
                conversation_id=input.conversation_id,
                turn_id=input.turn_id,
                skip_reason="failed_quality_gate",
            )
        yield _error_event(
            request_id,
            code="ANSWER_QUALITY_FAILED",
            retryable=False,
        )
        return
    event = emit_status(stage="finalizing", reason_code="finalizing")
    if event is not None:
        yield event
    if post_answer_progress is not None:
        post_answer_progress.lifecycle_stage = "final_response_contract"
    final_response = DoubtSolverFinalResponse(
        request_id=request_id,
        content=ResponseContent(value=answer),
        answer=answer,
        final_answer=final_answer,
    )
    if post_answer_progress is not None:
        post_answer_progress.lifecycle_stage = "response_preview"
    record_local_preview("response_type", "answer")
    record_local_preview("response", answer)
    persistence_result = None
    if conversation_persistence is not None:
        if post_answer_progress is not None:
            post_answer_progress.lifecycle_stage = "persistence_payload_build"
            post_answer_progress.persistence_payload_build_started = True
        turn = CompletedConversationTurn.now(
            actor_id=input.actor_id,
            conversation_id=input.conversation_id,
            turn_id=input.turn_id,
            original_query=input.original_query or input.query,
            final_answer=final_answer.content,
            exam_id=input.exam_id,
            exam_stage=input.exam_stage,
            language=input.language,
            subject=subject,
            topic=(
                str(classification_dict.get("topic")) if classification_dict.get("topic") else None
            ),
            quality_status=final_answer.quality_status,
            was_regenerated=final_answer.was_regenerated,
        )
        if post_answer_progress is not None:
            post_answer_progress.lifecycle_stage = "persistence_coordinator_entry"
        persistence_result = conversation_persistence.persist_completed_turn(
            turn,
            final_answer,
            request_id=request_id,
        )
    if student_credits is not None:
        if post_answer_progress is not None:
            post_answer_progress.lifecycle_stage = "student_credit_settlement"
        try:
            student_credits.settle(
                user_id=input.actor_id,
                reference_id=doubt_reference_id(
                    user_id=input.actor_id, turn_id=input.turn_id
                ),
                feature="doubt",
                operation_status="completed",
            )
        except InsufficientStudentCreditsError:
            yield _error_event(request_id, code="INSUFFICIENT_CREDITS", retryable=False)
            return
        except Exception as exc:  # noqa: BLE001
            # The answer is already delivered; no credit-store, pricing, or
            # unexpected settlement failure may retract it. Nothing is charged
            # and the failure is recorded with a locatable reason code.
            logger.error(
                "student_credit_settlement_unavailable request_id=%s reason_code=%s "
                "error_type=%s",
                request_id,
                getattr(exc, "reason_code", "STUDENT_CREDIT_UNEXPECTED_ERROR"),
                type(exc).__name__,
            )
    if post_answer_progress is not None:
        post_answer_progress.lifecycle_stage = "terminal_event_build"
    logger.debug(
        "answer_delivery request_id=%s stream_completed=true strategy=%s latency_ms=%d "
        "quality_status=%s was_regenerated=%s language_compliant=%s",
        request_id,
        decision.strategy,
        int((time.monotonic() - started_at) * 1000),
        final_answer.quality_status,
        final_answer.was_regenerated,
        final_answer.language_compliant,
    )
    yield DoubtSolverStreamEvent(
        type="complete",
        request_id=request_id,
        stage="complete",
        label=get_stream_label("complete"),
        metadata=_terminal_metadata(
            input,
            persisted=_history_persisted(persistence_result),
        ),
        response=final_response,
    )


def stream_doubt_solver(
    input: StreamDoubtSolverInput,
    *,
    adapter: AnswerGenerationAdapter,
    conversation_persistence=None,
    follow_up_resolver=None,
    conversation_understanding=None,
    practice_launcher: PracticeLauncher | None = None,
    practice_request_interpreter: PracticeRequestInterpreter | None = None,
    student_credits: StudentCreditRuntime | None = None,
) -> Iterator[DoubtSolverStreamEvent]:
    """Enforce a terminal event unless cancellation is confirmed."""
    started_at = time.monotonic()
    initial_type = "image" if input.source_modality == "image" else "pending"
    operation_accumulator = input.operation_accumulator
    if operation_accumulator is None:
        try:
            operation_accumulator = begin_operation(
                operation_id=input.request_id,
                feature="doubt",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ai_usage_meter_unavailable reason=stream_start_failed error_type=%s",
                type(exc).__name__,
            )
    with bind_request_context(
        request_id=input.request_id,
        trace_id=input.trace_id,
        conversation_id=input.conversation_id,
        turn_id=input.turn_id,
        request_type=initial_type,
    ):
        with bind_execution_context(
            operation_id=input.request_id,
            feature="doubt",
            operation_accumulator=operation_accumulator,
        ):
            summary_token = begin_request_summary(
                initial_llm_usage_records=input.initial_llm_usage_records
            )
            if input.credit_admission is not None or input.credit_mode is not None:
                update_request_summary(
                    credit_admission=input.credit_admission,
                    credit_mode=input.credit_mode,
                    credit_balance=input.credit_balance,
                )
            visible = False
            terminal = False
            terminal_reason = "unexpected_internal_error"
            terminal_error_type: str | None = None
            post_answer_progress = _PostAnswerFinalizationProgress()
            try:
                with stage_span("doubt_solver.request"):
                    if not input.request_started_logged:
                        log_event(
                            "request_started",
                            component="request.lifecycle",
                            stage="started",
                            status="started",
                            details={"request_type": initial_type},
                        )
                    for event in _iter_stream_doubt_solver(
                        input,
                        adapter=adapter,
                        conversation_persistence=conversation_persistence,
                        follow_up_resolver=follow_up_resolver,
                        conversation_understanding=conversation_understanding,
                        practice_launcher=practice_launcher,
                        practice_request_interpreter=practice_request_interpreter,
                        student_credits=student_credits,
                        post_answer_progress=post_answer_progress,
                    ):
                        if _cancelled(input):
                            terminal_reason = (
                                input.cancellation_reason()
                                if input.cancellation_reason is not None
                                else "user_cancelled"
                            ) or "user_cancelled"
                            logger.debug(
                                "request_cancelled request_id=%s terminal_reason=%s",
                                input.request_id,
                                terminal_reason,
                            )
                            if conversation_persistence is not None:
                                conversation_persistence.record_skip(
                                    request_id=input.request_id,
                                    conversation_id=input.conversation_id,
                                    turn_id=input.turn_id,
                                    skip_reason="request_cancelled",
                                )
                            return
                        if terminal:
                            return
                        if event.type == "chunk":
                            visible = True
                            post_answer_progress.answer_emitted = True
                        if event.stage:
                            post_answer_progress.lifecycle_stage = event.stage
                        if event.type in {
                            "complete",
                            "error",
                            "practice_generation_started",
                        }:
                            terminal = True
                            terminal_reason = (
                                "practice_generation_started"
                                if event.type == "practice_generation_started"
                                else (
                                    str(event.metadata.get("terminal_reason") or "completed")
                                    if event.type == "complete"
                                    else str(event.metadata.get("code") or "stream_failed")
                                )
                            )
                        yield event
            except Exception as exc:  # noqa: BLE001
                terminal = True
                terminal_reason = "unexpected_internal_error"
                terminal_error_type = type(exc).__name__
                if post_answer_progress.answer_emitted:
                    details, trace = _post_answer_finalization_failure_details(
                        exc,
                        post_answer_progress,
                        operation_id=input.request_id,
                    )
                    log_event(
                        "post_answer_finalization_failed",
                        component="doubt_solver.finalization",
                        stage=post_answer_progress.lifecycle_stage,
                        status="failed",
                        error_code="POST_ANSWER_FINALIZATION_FAILED",
                        details=details,
                        level=logging.ERROR,
                    )
                    if (
                        get_settings().app_env == "local"
                        and logger.isEnabledFor(logging.DEBUG)
                    ):
                        logger.debug(
                            "post_answer_finalization_trace request_id=%s trace=%s",
                            input.request_id,
                            trace,
                        )
                yield _error_event(
                    input.request_id,
                    code=("ANSWER_PARTIAL_STREAM_FAILED" if visible else "ANSWER_STREAM_FAILED"),
                    retryable=not visible,
                )
                return
            finally:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                try:
                    cancelled = _cancelled(input)
                except Exception:  # noqa: BLE001
                    cancelled = False
                status = (
                    "cancelled"
                    if cancelled
                    else (
                        "completed"
                        if terminal_reason in {"completed", "practice_generation_started"}
                        else (
                            "clarification"
                            if terminal_reason == "clarification_required"
                            else "failed"
                        )
                    )
                )
                update_request_summary(
                    terminal_status=status,
                    terminal_reason=terminal_reason,
                    total_duration_ms=duration_ms,
                    wall_clock_duration_ms=(
                        int((time.monotonic() - input.entrypoint_started_at) * 1000)
                        if input.entrypoint_started_at is not None
                        else None
                    ),
                )
                emit_request_summary()
                emit_operation_billing_summary(operation_status=status)
                log_event(
                    (
                        "request_cancelled"
                        if status == "cancelled"
                        else ("request_completed" if status == "completed" else "request_failed")
                    ),
                    component="request.lifecycle",
                    stage="complete" if status == "completed" else status,
                    status=status,
                    duration_ms=duration_ms,
                    error_code=None if status == "completed" else terminal_reason.upper(),
                    details=(
                        {"error_type": terminal_error_type}
                        if terminal_error_type is not None
                        else None
                    ),
                    level=logging.ERROR if status == "failed" else logging.INFO,
                )
                reset_request_summary(summary_token)
        try:
            cancellation_confirmed = _cancelled(input)
        except Exception:  # noqa: BLE001
            yield _error_event(
                input.request_id,
                code=("ANSWER_PARTIAL_STREAM_FAILED" if visible else "ANSWER_STREAM_FAILED"),
                retryable=not visible,
            )
            return
        if not terminal and not cancellation_confirmed:
            yield _error_event(
                input.request_id,
                code=("ANSWER_PARTIAL_STREAM_FAILED" if visible else "ANSWER_STREAM_FAILED"),
                retryable=not visible,
            )
        elif not terminal and conversation_persistence is not None:
            conversation_persistence.record_skip(
                request_id=input.request_id,
                conversation_id=input.conversation_id,
                turn_id=input.turn_id,
                skip_reason="request_cancelled",
            )
