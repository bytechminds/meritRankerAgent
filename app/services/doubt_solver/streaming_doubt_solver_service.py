"""Adaptive verified streaming for the orchestrated doubt solver."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

from config import get_settings
from graphs.doubt_solver_graph import (
    _ORCHESTRATED_FALLBACK_CLASSIFICATION,
    _orchestrated_collect_context_node,
    orchestrated_classify_query_with_delivery_signals,
)
from observability import (
    begin_request_summary,
    bind_request_context,
    current_request_summary,
    emit_request_summary,
    log_event,
    record_local_preview,
    stage_span,
    update_request_summary,
    update_request_type,
)
from observability.summary import reset_request_summary
from schemas.conversation import CompletedConversationTurn
from schemas.doubt_solver import (
    CanonicalLanguage,
    DoubtSolverFinalResponse,
    DoubtSolverStreamEvent,
    ResponseContent,
)
from services.conversation.follow_up_query_resolver import (
    resolve_follow_up_with_recent_context,
)
from services.doubt_solver.answer_delivery_policy import (
    AnswerDeliveryPolicy,
    AnswerDeliverySignals,
)
from services.doubt_solver.answer_generation_adapter import AnswerGenerationAdapter
from services.doubt_solver.answer_quality import AnswerQualityPolicy, validate_answer_quality
from services.doubt_solver.final_answer import build_final_answer_result
from services.doubt_solver.markdown_replay import iter_markdown_replay_chunks
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

logger = logging.getLogger(__name__)


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
    should_cancel: Callable[[], bool] | None = None
    cancellation_reason: Callable[[], str | None] | None = None
    request_started_logged: bool = False


def _cancelled(input: StreamDoubtSolverInput) -> bool:
    return input.should_cancel is not None and input.should_cancel()


def _error_event(
    request_id: str,
    *,
    code: str,
    retryable: bool,
) -> DoubtSolverStreamEvent:
    return DoubtSolverStreamEvent(
        type="error",
        request_id=request_id,
        stage="failed",
        label="Unable to complete",
        metadata={"retryable": retryable, "code": code},
    )


def _iter_stream_doubt_solver(
    input: StreamDoubtSolverInput,
    *,
    adapter: AnswerGenerationAdapter,
    conversation_persistence=None,
    follow_up_resolver=None,
    conversation_understanding=None,
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

    conversation_result = None
    conversation_context = ""
    if conversation_understanding is not None:
        conversation_result = conversation_understanding.understand(
            actor_id=input.actor_id,
            conversation_id=input.conversation_id,
            query=query,
        )
        conversation_context = conversation_result.conversation_context
        if conversation_result.resolved_query is not None:
            query = conversation_result.resolved_query
        elif conversation_result.relation.requires_recent_conversation:
            clarification = {
                "english": "Please clarify which earlier question, step, or option you mean.",
                "hinglish": (
                    "Please clarify karein ki aap kis pehle question, step, ya option "
                    "ki baat kar rahe hain."
                ),
                "hindi": "कृपया स्पष्ट करें कि आप पहले के किस प्रश्न, चरण या विकल्प की बात कर रहे हैं।",
            }[input.language]
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
                metadata={
                    "request_id": request_id,
                    "terminal_reason": "clarification_required",
                },
                response=DoubtSolverFinalResponse(
                    request_id=request_id,
                    content=ResponseContent(value=clarification),
                    answer=clarification,
                    final_answer=final,
                ),
            )
            return

    state = {
        "request_id": request_id,
        "query": query,
        "original_query": input.original_query or query,
        "actor_id": input.actor_id,
        "language": input.language,
        "exam_id": input.exam_id,
        "exam_stage": input.exam_stage,
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
        classification_dict = (
            None
            if conversation_result is not None
            and conversation_result.relation.requires_recent_conversation
            else input.classification
        )
        if classification_dict is None:
            (
                classification_dict,
                classifier_confidence,
                classifier_fallback,
            ) = orchestrated_classify_query_with_delivery_signals(
                query,
                request_id=request_id,
                on_before_strong_classifier=on_before_strong_classifier,
            )
        else:
            classifier_confidence = input.classifier_confidence
            classifier_fallback = input.classifier_fallback
    if (
        conversation_result is not None
        and not conversation_result.relation.requires_recent_conversation
    ):
        classification_dict = dict(classification_dict)
        classification_dict["requires_recent_conversation"] = False
    state["classification"] = classification_dict
    request_type = (
        "follow_up"
        if (
            conversation_result is not None
            and conversation_result.relation.requires_recent_conversation
        )
        or classification_dict.get("requires_recent_conversation", False)
        else ("image" if input.source_modality == "image" else "standalone")
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

    if (
        conversation_result is None
        and classification_dict.get("requires_recent_conversation", False)
    ):
        logger.debug("follow_up_request_count count=1")
        log_event(
            "follow_up_detected",
            component="conversation.follow_up",
            stage="load_recent_context",
            status="detected",
        )
        try:
            if conversation_persistence is None or follow_up_resolver is None:
                raise RuntimeError("Conversation context is unavailable.")
            with stage_span("doubt_solver.resolve_follow_up"):
                resolved, recent = resolve_follow_up_with_recent_context(
                    persistence=conversation_persistence,
                    resolver=follow_up_resolver,
                    actor_id=input.actor_id,
                    conversation_id=input.conversation_id,
                    request_id=request_id,
                    original_query=input.original_query or query,
                )
            query = resolved.resolved_query
            record_local_preview("resolved_query", query)
            classification_dict, classifier_confidence, classifier_fallback = (
                orchestrated_classify_query_with_delivery_signals(
                    query,
                    request_id=request_id,
                    on_before_strong_classifier=on_before_strong_classifier,
                )
            )
            classification_dict["requires_recent_conversation"] = False
            state["query"] = query
            state["classification"] = classification_dict
            conversation_context = recent.formatted_reference
            update_request_summary(
                context_source=getattr(recent, "source", "unknown"),
                usable_recent_turns=getattr(recent, "usable_turn_count", len(recent.turns)),
            )
        except Exception as exc:  # noqa: BLE001
            stage = getattr(exc, "stage", "context_load")
            reason = getattr(exc, "reason", "context_unavailable")
            logger.warning(
                "follow_up_resolution_failure count=1 failure_stage=%s "
                "failure_reason=%s",
                stage,
                reason,
            )
            log_event(
                "follow_up_resolution_failed",
                component="conversation.follow_up",
                stage=str(stage),
                status="failed",
                error_code=str(reason).upper(),
                details={
                    "error_type": type(exc).__name__,
                    "failure_stage": stage,
                    "failure_reason": reason,
                },
                level=logging.WARNING,
            )
            clarification = {
                "english": "Please clarify which earlier question, step, or option you mean.",
                "hinglish": (
                    "Please clarify karein ki aap kis pehle question, step, ya option "
                    "ki baat kar rahe hain."
                ),
                "hindi": "कृपया स्पष्ट करें कि आप पहले के किस प्रश्न, चरण या विकल्प की बात कर रहे हैं।",
            }[input.language]
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
                type="chunk", request_id=request_id, content=clarification
            )
            yield DoubtSolverStreamEvent(
                type="complete",
                request_id=request_id,
                stage="complete",
                label=get_stream_label("complete"),
                metadata={
                    "request_id": request_id,
                    "terminal_reason": "clarification_required",
                },
                response=DoubtSolverFinalResponse(
                    request_id=request_id,
                    content=ResponseContent(value=clarification),
                    answer=clarification,
                    final_answer=final,
                ),
            )
            return
    elif (
        conversation_result is None
        or not conversation_result.relation.requires_recent_conversation
    ):
        logger.debug("standalone_request_count count=1")
    else:
        logger.debug("follow_up_request_count count=1")

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
    context_text = str(state.get("context_text") or "")
    retrieval_context = state.get("retrieval_context") or {}
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
            pattern_candidate_conflict=bool(
                retrieval_context.get("materialConflicts") or []
            ),
            provider_fallback=False,
            needs_review=False,
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
    generation_started = time.monotonic()
    if decision.strategy == "live_stream":
        answer_parts: list[str] = []
        visible_answer_emitted = False
        try:
            for chunk in adapter.generate_stream(
                request_id=request_id,
                query=query,
                subject=subject,
                intent=intent,
                difficulty=difficulty,
                context_text=context_text,
                web_search_reason=(
                    str(classification_dict["web_search_reason"])
                    if classification_dict.get("web_search_reason")
                    else None
                ),
                on_before_generator_fallback=status_tracker.hook(
                    stage="generating",
                    label=LABEL_GENERATOR_FALLBACK,
                    reason_code="generator_fallback",
                ),
                on_before_continuation=status_tracker.hook(
                    stage="generating",
                    label=LABEL_ANSWER_CONTINUATION,
                    reason_code="answer_continuation",
                ),
                verify_before_stream=False,
                exam_id=input.exam_id,
                exam_stage=input.exam_stage,
                language=input.language,
                conversation_context=conversation_context or None,
            ):
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
                yield DoubtSolverStreamEvent(
                    type="chunk", request_id=request_id, content=chunk
                )
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
            draft = adapter.generate(
                request_id=request_id,
                query=query,
                subject=subject,
                intent=intent,
                difficulty=difficulty,
                context=context_text,
                web_search_reason=(
                    str(classification_dict["web_search_reason"])
                    if classification_dict.get("web_search_reason")
                    else None
                ),
                exam_id=input.exam_id,
                exam_stage=input.exam_stage,
                language=input.language,
                conversation_context=conversation_context or None,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "answer_delivery request_id=%s stage=private_generate terminal_reason=%s",
                request_id,
                "provider_failed_before_content",
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
        if settings.answer_verifier_enabled and not verification.is_valid:
            if settings.answer_verifier_max_repair_attempts == 1:
                repair_attempted = True
                logger.debug("repair_started request_id=%s stage=verifying", request_id)
                if _cancelled(input):
                    return
                try:
                    draft = adapter.generate(
                        request_id=request_id,
                        query=query,
                        subject=subject,
                        intent=intent,
                        difficulty=difficulty,
                        context=context_text,
                        web_search_reason=(
                            str(classification_dict["web_search_reason"])
                            if classification_dict.get("web_search_reason")
                            else None
                        ),
                        exam_id=input.exam_id,
                        exam_stage=input.exam_stage,
                        language=input.language,
                        conversation_context=conversation_context or None,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "answer_delivery request_id=%s stage=repair terminal_reason=%s",
                        request_id,
                        "repair_failed",
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
            if not verification.is_valid:
                logger.warning(
                    "answer_delivery request_id=%s stage=verification approved=false repair=%s",
                    request_id,
                    repair_attempted,
                )
                yield _error_event(
                    request_id,
                    code="ANSWER_VERIFICATION_FAILED",
                    retryable=False,
                )
                return
        if _cancelled(input):
            return
        answer = verification.sanitized_text or draft
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
                yield DoubtSolverStreamEvent(
                    type="chunk", request_id=request_id, content=chunk
                )
        logger.debug("replay_completed request_id=%s stage=verifying", request_id)

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
            if decision.strategy == "live_stream"
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
        details={"route": decision.strategy},
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
                language=input.language,
                policy=final_quality_policy,
            )
    if not final_quality_policy.validation_enabled:
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
            "QUALITY_GATE_FAILED"
            if final_answer.quality_status == "failed_quality_gate"
            else None
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
            code="ANSWER_VERIFICATION_FAILED",
            retryable=False,
        )
        return
    event = emit_status(stage="finalizing", reason_code="finalizing")
    if event is not None:
        yield event
    final_response = DoubtSolverFinalResponse(
        request_id=request_id,
        content=ResponseContent(value=answer),
        answer=answer,
        final_answer=final_answer,
    )
    record_local_preview("response_type", "answer")
    record_local_preview("response", answer)
    if conversation_persistence is not None:
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
                str(classification_dict.get("topic"))
                if classification_dict.get("topic")
                else None
            ),
            quality_status=final_answer.quality_status,
            was_regenerated=final_answer.was_regenerated,
        )
        conversation_persistence.persist_completed_turn(
            turn,
            final_answer,
            request_id=request_id,
        )
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
        metadata={"request_id": request_id},
        response=final_response,
    )


def stream_doubt_solver(
    input: StreamDoubtSolverInput,
    *,
    adapter: AnswerGenerationAdapter,
    conversation_persistence=None,
    follow_up_resolver=None,
    conversation_understanding=None,
) -> Iterator[DoubtSolverStreamEvent]:
    """Enforce a terminal event unless cancellation is confirmed."""
    started_at = time.monotonic()
    initial_type = "image" if input.source_modality == "image" else "pending"
    with bind_request_context(
        request_id=input.request_id,
        trace_id=input.trace_id,
        conversation_id=input.conversation_id,
        turn_id=input.turn_id,
        request_type=initial_type,
    ):
        summary_token = begin_request_summary()
        visible = False
        terminal = False
        terminal_reason = "unexpected_internal_error"
        terminal_error_type: str | None = None
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
                    elif event.type in {"complete", "error"}:
                        terminal = True
                        terminal_reason = (
                            str(event.metadata.get("terminal_reason") or "completed")
                            if event.type == "complete"
                            else str(event.metadata.get("code") or "stream_failed")
                        )
                    yield event
        except Exception as exc:  # noqa: BLE001
            terminal = True
            terminal_reason = "unexpected_internal_error"
            terminal_error_type = type(exc).__name__
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
                    if terminal_reason == "completed"
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
            )
            emit_request_summary()
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
