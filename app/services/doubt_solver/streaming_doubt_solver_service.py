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
from schemas.doubt_solver import (
    DoubtSolverFinalResponse,
    DoubtSolverStreamEvent,
    ResponseContent,
)
from services.doubt_solver.answer_delivery_policy import (
    AnswerDeliveryPolicy,
    AnswerDeliverySignals,
)
from services.doubt_solver.answer_generation_adapter import AnswerGenerationAdapter
from services.doubt_solver.answer_quality import AnswerQualityPolicy, validate_answer_quality
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
) -> Iterator[DoubtSolverStreamEvent]:
    """Yield live chunks only for low risk requests, otherwise replay approval."""
    request_id = input.request_id
    query = input.query
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

    logger.info("answer_delivery request_id=%s stream_started=true", request_id)
    event = emit_status(stage="understanding", reason_code="stream_started")
    if event is not None:
        yield event
    if _cancelled(input):
        return

    state = {
        "request_id": request_id,
        "query": query,
        "classification": None,
        "retrieval_context": {},
        "context_text": "",
        "answer": None,
    }
    careful_status_pending = False

    def on_before_strong_classifier() -> None:
        nonlocal careful_status_pending
        careful_status_pending = True

    classification_dict = input.classification
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
    state["classification"] = classification_dict
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

    event = emit_status(stage="thinking", reason_code="thinking")
    if event is not None:
        yield event
    event = emit_status(stage="retrieving", reason_code="retrieval_started")
    if event is not None:
        yield event
    need_web_search = bool(classification_dict.get("need_web_search"))
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
    logger.info(
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
            ):
                yield from emit_pending_statuses()
                if _cancelled(input):
                    return
                if not chunk:
                    continue
                answer_parts.append(chunk)
                if chunk.strip():
                    visible_answer_emitted = True
                    logger.info(
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
        logger.info("verification_started request_id=%s stage=verifying", request_id)
        if _cancelled(input):
            return
        settings = get_settings()
        try:
            verification = validate_answer_quality(
                draft,
                subject=subject,
                intent=intent,
                difficulty=difficulty,
                policy=AnswerQualityPolicy.from_settings(settings),
            )
        except Exception:  # noqa: BLE001
            yield _error_event(
                request_id,
                code="ANSWER_VERIFICATION_FAILED",
                retryable=True,
            )
            return
        logger.info(
            "verification_completed request_id=%s stage=verifying",
            request_id,
        )
        if _cancelled(input):
            return
        repair_attempted = False
        if settings.answer_verifier_enabled and not verification.is_valid:
            if settings.answer_verifier_max_repair_attempts == 1:
                repair_attempted = True
                logger.info("repair_started request_id=%s stage=verifying", request_id)
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
                    verification = validate_answer_quality(
                        draft,
                        subject=subject,
                        intent=intent,
                        difficulty=difficulty,
                        policy=AnswerQualityPolicy.from_settings(settings),
                    )
                except Exception:  # noqa: BLE001
                    yield _error_event(
                        request_id,
                        code="ANSWER_VERIFICATION_FAILED",
                        retryable=True,
                    )
                    return
                logger.info("repair_completed request_id=%s stage=verifying", request_id)
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
        logger.info("replay_started request_id=%s stage=verifying", request_id)
        for chunk in replay_chunks:
            if _cancelled(input):
                return
            if chunk:
                logger.info(
                    "answer_chunk_emission request_id=%s strategy=verified_replay "
                    "first_visible_chunk_emitted=true chunk_chars=%d",
                    request_id,
                    len(chunk),
                )
                yield DoubtSolverStreamEvent(
                    type="chunk", request_id=request_id, content=chunk
                )
        logger.info("replay_completed request_id=%s stage=verifying", request_id)

    if not answer.strip():
        yield _error_event(
            request_id,
            code="ANSWER_STREAM_FAILED",
            retryable=True,
        )
        return
    if _cancelled(input):
        return
    event = emit_status(stage="finalizing", reason_code="finalizing")
    if event is not None:
        yield event
    final_response = DoubtSolverFinalResponse(
        request_id=request_id,
        content=ResponseContent(value=answer),
        answer=answer,
    )
    logger.info(
        "answer_delivery request_id=%s stream_completed=true strategy=%s latency_ms=%d",
        request_id,
        decision.strategy,
        int((time.monotonic() - started_at) * 1000),
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
) -> Iterator[DoubtSolverStreamEvent]:
    """Enforce a terminal event unless cancellation is confirmed."""
    visible = False
    terminal = False
    try:
        for event in _iter_stream_doubt_solver(input, adapter=adapter):
            if _cancelled(input):
                reason = (
                    input.cancellation_reason()
                    if input.cancellation_reason is not None
                    else "user_cancelled"
                )
                logger.info(
                    "request_cancelled request_id=%s terminal_reason=%s",
                    input.request_id,
                    reason or "user_cancelled",
                )
                return
            if terminal:
                return
            if event.type == "chunk":
                visible = True
            elif event.type in {"complete", "error"}:
                terminal = True
            yield event
    except Exception:  # noqa: BLE001
        terminal = True
        yield _error_event(
            input.request_id,
            code=("ANSWER_PARTIAL_STREAM_FAILED" if visible else "ANSWER_STREAM_FAILED"),
            retryable=not visible,
        )
        return
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
