"""
app/main.py
-----------
AgentCore entrypoint for the meritRankerTutor agent.

How this file fits into the bigger picture:
    agentcore dev  ──►  starts a local HTTP server
                   ──►  POST /invocations  ──►  invoke() below
                   ──►  agentcore deploy uploads this file to AWS

Flow:
    1. HTTP POST arrives with a JSON body.
    2. BedrockAgentCoreApp deserialises the body to a plain dict.
    3. invoke(payload) validates the dict with AgentRequest (Pydantic v2).
    4. A per-request UUID is generated so every call can be traced end-to-end.
    5. The LangGraph workflow runs and produces an answer.
    6. AgentResponse is serialised and returned as the HTTP response body.

Error handling:
    - Pydantic ValidationError  →  400-style response with "Validation error: …"
    - Any other exception       →  500-style response with "Internal error: …"
    Neither case crashes the server — AgentCore runtime stays alive.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator

from bedrock_agentcore import BedrockAgentCoreApp
from pydantic import ValidationError
from starlette.responses import Response, StreamingResponse

from config import ConfigurationError, get_settings
from features.practice_generation.agentcore_async import build_practice_async_launcher
from features.practice_generation.planning import decide_practice_launch
from features.practice_generation.schemas import PracticeControlRequest
from graphs.demo_graph import build_demo_graph
from graphs.doubt_solver_graph import (
    build_doubt_solver_graph,
    build_orchestrated_doubt_solver_graph,
    map_to_orchestrated_classification,
)
from graphs.image_question_classifier_node import (
    ImageClassifierNodeInput,
    image_classifier_node,
)
from logging_config import configure_logging
from observability import (
    begin_request_summary,
    bind_execution_context,
    bind_request_context,
    configure_runtime_identity,
    current_request_context,
    current_request_summary,
    emit_request_summary,
    log_event,
    observe_invocation,
    snapshot_llm_usage_records,
    stage_span,
    update_request_summary,
    update_request_type,
)
from observability.summary import reset_request_summary
from schemas.conversation import CompletedConversationTurn
from schemas.doubt_solver import (
    DoubtSolverFinalResponse,
    DoubtSolverRequest,
    DoubtSolverStreamEvent,
    FinalAnswerResult,
    ResponseContent,
    TerminalConversationIdentity,
)
from schemas.image_question_classification import ImageClassificationStatus
from schemas.request import AgentRequest
from schemas.response import AgentResponse
from services.classification.coordinator import ClassificationCoordinator
from services.classification.image_classification_adapter import (
    adapt_image_classification,
)
from services.conversation.bootstrap import build_conversation_persistence_service
from services.conversation.conversation_understanding import (
    ConversationUnderstandingService,
)
from services.conversation.history_repository import (
    ConversationHistoryError,
    ConversationTurnConflictError,
)
from services.conversation.runtime_config import ConversationConfigurationError
from services.doubt_solver.actor_identity import resolve_actor_id
from services.doubt_solver.answer_generation_adapter import AnswerGenerationAdapter
from services.doubt_solver.exam_profile_cache import get_exam_profile_runtime
from services.doubt_solver.exam_response_profile import (
    ExamResponseProfileConfigError,
    get_exam_response_profile_resolver,
)
from services.doubt_solver.stream_transport import (
    StreamCancellation,
    stream_events_as_sse,
)
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)
from services.llm.billing import (
    BillingConfigurationError,
    OperationUsageAccumulator,
    begin_operation,
    current_operation_accumulator,
    emit_operation_billing_summary,
    validate_billing_configuration,
)
from services.llm.runtime_factory import build_model_executor

# ---------------------------------------------------------------------------
# Bootstrap — runs once when the module is imported
# ---------------------------------------------------------------------------

settings = get_settings()
try:
    validate_billing_configuration()
except BillingConfigurationError as exc:
    raise ConfigurationError(str(exc)) from exc
configure_logging(
    settings.log_level,
    environment=settings.app_env,
    log_format=settings.agent_log_format or None,
    file_enabled=settings.agent_log_file_enabled,
    file_path=settings.agent_log_file_path,
    detailed_logs=settings.agent_detailed_logs,
    observability_enabled=settings.agent_observability_enabled,
    local_log_content=settings.agent_local_log_content,
)
try:
    get_exam_response_profile_resolver()
except ExamResponseProfileConfigError as exc:
    raise ConfigurationError(str(exc)) from exc

if settings.app_env != "test":
    # Profiles are advisory during migration.  A failed startup load preserves
    # the existing bundled exam-response guidance rather than failing requests.
    get_exam_profile_runtime().load()

logger = logging.getLogger(__name__)

# The single AgentCore application owns both doubt solving and tracked practice tasks.
app = BedrockAgentCoreApp()

conversation_persistence = None
if settings.app_env != "test":
    try:
        conversation_persistence = build_conversation_persistence_service()
    except ConversationConfigurationError as exc:
        raise ConfigurationError(str(exc)) from exc
conversation_understanding = (
    ConversationUnderstandingService(persistence=conversation_persistence)
    if conversation_persistence is not None
    else None
)

runtime_identity = configure_runtime_identity(
    environment=settings.app_env,
    region=settings.bedrock_kb_region or "unknown",
    history_configured=conversation_persistence is not None,
    session_configured=bool(
        conversation_persistence is not None
        and conversation_persistence.session_repository is not None
    ),
    memory_configured=bool(
        conversation_persistence is not None
        and conversation_persistence.short_term_memory.__class__.__name__
        != "UnavailableShortTermMemory"
    ),
    code_location=__file__,
)
# Emitted here so runtime identity is still captured when a later mandatory step
# fails. It deliberately does NOT claim readiness: Practice/Pattern resource
# contracts, DynamoDB validation, and graph construction all still follow, and any
# of them can abort startup. The authoritative ready event is emitted at the end.
log_event(
    "runtime_started",
    component="runtime.bootstrap",
    stage="startup",
    status="started",
    details=runtime_identity,
)

# Build graphs once at startup — compilation is not free.
graph = build_demo_graph()
doubt_solver_graph = build_doubt_solver_graph()

image_question_classifier = None
classification_coordinator = ClassificationCoordinator()
if settings.image_classifier_enabled:
    from services.image_question_classification.factory import (  # noqa: PLC0415
        build_image_question_classifier,
    )

    image_question_classifier = build_image_question_classifier(settings)

# Orchestrated graph — only built when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true.
# Default is false; existing tests are unaffected.
orchestrated_doubt_solver_graph = None
orchestrated_adapter: AnswerGenerationAdapter | None = None
follow_up_resolver = None
practice_async_launcher = None
practice_request_interpreter = None
if settings.enable_orchestrated_doubt_solver:
    from services.llm.orchestration.orchestrator import LlmOrchestrator  # noqa: PLC0415

    if not settings.enable_real_llm:
        # Production safety guard.
        # Mock executor must not silently serve fake answers in production.
        # Fail fast at startup so operators see a clear config error instead
        # of discovering fake answers in production traffic.
        _is_production = settings.app_env == "production"
        _mock_permitted = not _is_production or settings.enable_orchestrated_mock_llm
        if not _mock_permitted:
            raise ConfigurationError(
                "Unsafe configuration: ENABLE_ORCHESTRATED_DOUBT_SOLVER=true "
                "with ENABLE_REAL_LLM=false is not permitted when "
                "APP_ENV=production. "
                "To fix: set ENABLE_REAL_LLM=true (recommended for production), "
                "or set ENABLE_ORCHESTRATED_MOCK_LLM=true to explicitly allow "
                "mock (only for controlled internal testing — not normal production)."
            )

    _model_executor = build_model_executor(
        settings,
        mock_content=(
            "**Final Answer:** [orchestrated-mock] Mock answer — set ENABLE_REAL_LLM=true "
            "for a real LLM response."
        ),
    )

    _orchestrator = LlmOrchestrator(model_executor=_model_executor)
    from services.conversation.follow_up_query_resolver import (  # noqa: PLC0415
        FollowUpQueryResolver,
    )

    follow_up_resolver = FollowUpQueryResolver(orchestrator=_orchestrator)
    _adapter = AnswerGenerationAdapter(orchestrator=_orchestrator)
    orchestrated_adapter = _adapter
    practice_async_launcher = build_practice_async_launcher(
        task_tracker=app,
        llm_orchestrator=_orchestrator,
    )
    # Bypassed on the Practice path: the >20 deterministic router hands large requests
    # straight to the existing intelligence planner, so no separate interpretation call
    # runs ahead of it. The provider and its route stay in the repo, unwired.
    practice_request_interpreter = None
    orchestrated_doubt_solver_graph = build_orchestrated_doubt_solver_graph(
        _adapter,
        conversation_persistence=conversation_persistence,
        follow_up_resolver=follow_up_resolver,
        conversation_understanding=conversation_understanding,
        practice_launcher=(
            practice_async_launcher.launch if practice_async_launcher is not None else None
        ),
        practice_request_interpreter=practice_request_interpreter,
    )
    logger.info(
        "Orchestrated graph built  enable_real_llm=%s",
        settings.enable_real_llm,
    )

logger.info(
    "Agent initialised app_env=%s default_model_provider=%s routing=dynamic",
    settings.app_env,
    settings.model_provider,
)

# Authoritative readiness boundary. Every mandatory startup step has now
# succeeded: resource contracts, DynamoDB resource validation, Practice/Pattern
# runtime construction, and graph compilation. Any failure above raises out of
# module import, so this line is unreachable and no ready event is emitted.
log_event(
    "runtime_ready",
    component="runtime.bootstrap",
    stage="startup",
    status="ready",
    details={
        "practiceEnabled": practice_async_launcher is not None,
        "patternContextEnabled": settings.pattern_intelligence_enabled,
        "patternReuseEnabled": settings.pattern_intelligence_reuse_enabled,
        "orchestratedDoubtSolverEnabled": settings.enable_orchestrated_doubt_solver,
    },
)

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _load_completed_replay(
    *, actor_id: str, conversation_id: str, turn_id: str, original_query: str
) -> CompletedConversationTurn | None:
    if conversation_persistence is None:
        return None
    try:
        turn = conversation_persistence.history_repository.get_completed_turn(
            actor_id, conversation_id, turn_id
        )
    except ConversationTurnConflictError:
        raise
    except ConversationHistoryError as exc:
        logger.warning(
            "conversation_idempotency_read_failure error_type=%s",
            type(exc).__name__,
        )
        return None
    if turn is not None and turn.original_query != original_query:
        raise ConversationTurnConflictError(
            "Turn ID already exists with different request content."
        )
    return turn


def _replay_response(turn: CompletedConversationTurn, request_id: str) -> dict:
    response = {
        "schema_version": "1",
        "status": "completed",
        "success": True,
        "request_id": request_id,
        "mode": "doubt_solver",
        "answer": turn.final_answer,
        "content": {"format": "markdown", "value": turn.final_answer},
        "classification": {
            "intent": "unknown",
            "subject": turn.subject,
            "topic": turn.topic,
            "confidence": 0.0,
            "classification_source": "fallback",
        },
    }
    if turn.response_type == "practice_generation" and turn.practice_test_id:
        response["responseType"] = turn.response_type
        response["practiceTestId"] = turn.practice_test_id
    return response


def _stream_replayed_turn(
    turn: CompletedConversationTurn,
    request_id: str,
    *,
    trace_id: str | None = None,
    request_started_logged: bool = False,
    operation_accumulator: OperationUsageAccumulator | None = None,
) -> Iterator[DoubtSolverStreamEvent]:
    started_at = time.monotonic()
    accumulator = operation_accumulator
    if accumulator is None:
        accumulator = begin_operation(operation_id=request_id, feature="doubt")
    with bind_request_context(
        request_id=request_id,
        trace_id=trace_id,
        conversation_id=turn.conversation_id,
        turn_id=turn.turn_id,
        request_type="standalone",
    ):
        with bind_execution_context(
            operation_id=request_id,
            feature="doubt",
            operation_accumulator=accumulator,
        ):
            summary_token = begin_request_summary()
            if not request_started_logged:
                log_event(
                    "request_started",
                    component="request.lifecycle",
                    stage="started",
                    status="started",
                    details={"request_type": "standalone"},
                )
            update_request_summary(
                request_type="standalone",
                subject=turn.subject,
                quality_status=turn.quality_status,
                history_write_status="idempotent_replay",
            )
            terminal_status = "cancelled"
            terminal_reason = "stream_closed"
            try:
                final_answer = FinalAnswerResult(
                    content=turn.final_answer,
                    quality_status=turn.quality_status,
                    was_regenerated=turn.was_regenerated,
                    language_compliant=True,
                )
                yield DoubtSolverStreamEvent(
                    type="chunk",
                    request_id=request_id,
                    content=turn.final_answer,
                )
                terminal_status = "completed"
                terminal_reason = "idempotent_replay"
                yield DoubtSolverStreamEvent(
                    type="complete",
                    request_id=request_id,
                    stage="complete",
                    label="Complete",
                    metadata={
                        **TerminalConversationIdentity(
                            request_id=request_id,
                            conversation_id=turn.conversation_id,
                            turn_id=turn.turn_id,
                            persisted=True,
                        ).model_dump(mode="json"),
                        "replayed": True,
                    },
                    response=DoubtSolverFinalResponse(
                        request_id=request_id,
                        content=ResponseContent(value=turn.final_answer),
                        answer=turn.final_answer,
                        final_answer=final_answer,
                        response_type=turn.response_type,
                        practice_test_id=turn.practice_test_id,
                    ),
                )
            finally:
                duration_ms = int((time.monotonic() - started_at) * 1000)
                update_request_summary(
                    terminal_status=terminal_status,
                    terminal_reason=terminal_reason,
                    total_duration_ms=duration_ms,
                )
                emit_request_summary()
                emit_operation_billing_summary(operation_status=terminal_status)
                log_event(
                    (
                        "request_completed"
                        if terminal_status == "completed"
                        else "request_cancelled"
                    ),
                    component="request.lifecycle",
                    stage="complete" if terminal_status == "completed" else "cancelled",
                    status=terminal_status,
                    duration_ms=duration_ms,
                )
                reset_request_summary(summary_token)


def _persist_completed_result(
    *,
    request: DoubtSolverRequest,
    actor_id: str,
    original_query: str,
    result: dict,
    request_id: str,
):
    classification = result.get("classification") or {}
    final_raw = result.get("final_answer")
    practice_disabled = (
        final_raw is not None
        and str(final_raw.get("quality_status") or "") == "failed_quality_gate"
        and decide_practice_launch(original_query, classification).eligible
    )
    if practice_disabled:
        if conversation_persistence is not None:
            conversation_persistence.record_skip(
                request_id=request_id,
                conversation_id=request.conversation_id,
                turn_id=request.turn_id,
                skip_reason="failed_quality_gate",
            )
        return
    if conversation_persistence is None:
        log_event(
            "conversation_persistence_skipped",
            component="conversation.persistence",
            stage="persist_history",
            status="skipped",
            details={"skip_reason": "service_unavailable"},
        )
        return
    final_raw = result.get("final_answer")
    if final_raw is None:
        log_event(
            "conversation_persistence_skipped",
            component="conversation.persistence",
            stage="persist_history",
            status="skipped",
            details={"skip_reason": "final_answer_unavailable"},
        )
        return
    final_answer = FinalAnswerResult.model_validate(final_raw)
    classification = result.get("classification") or {}
    turn = CompletedConversationTurn.now(
        actor_id=actor_id,
        conversation_id=request.conversation_id,
        turn_id=request.turn_id,
        original_query=original_query,
        final_answer=final_answer.content,
        exam_id=request.exam_id,
        exam_stage=request.exam_stage,
        language=request.language,
        subject=str(classification.get("subject") or "unknown"),
        topic=(str(classification["topic"]) if classification.get("topic") else None),
        quality_status=final_answer.quality_status,
        was_regenerated=final_answer.was_regenerated,
        response_type=result.get("response_type"),
        practice_test_id=result.get("practice_test_id"),
    )
    return conversation_persistence.persist_completed_turn(
        turn,
        final_answer,
        request_id=request_id,
        clarification_response=bool(
            classification.get("requires_recent_conversation")
            and final_answer.quality_status == "failed_quality_gate"
        ),
    )


@app.entrypoint
@observe_invocation
def invoke(payload: dict) -> dict | Response:
    """Handle a single invocation request.

    Args:
        payload: Raw dict from the HTTP request body.

    Returns:
        Serialised AgentResponse dict.
    """
    context = current_request_context()
    request_id = context.request_id if context is not None else str(uuid.uuid4())

    try:
        mode = payload.get("mode", "demo")

        if mode == "practice_control":
            control = PracticeControlRequest.model_validate(payload)
            if practice_async_launcher is None:
                return {
                    "success": False,
                    "code": "PRACTICE_GENERATION_UNAVAILABLE",
                    "test_id": control.test_id,
                    "status": "FAILED",
                }
            result = (
                practice_async_launcher.cancel(control.test_id, control.user_id)
                if control.action == "cancel"
                else practice_async_launcher.resume(control.test_id, control.user_id)
            )
            return result.model_dump(mode="json")

        # --- Doubt Solver path --------------------------------------------
        if mode == "doubt_solver":
            ds_request = DoubtSolverRequest.model_validate(payload)
            initial_request_type = "image" if ds_request.image is not None else "unknown"
            update_request_type(initial_request_type)
            update_request_summary(request_type=initial_request_type)
            actor_id = resolve_actor_id(ds_request)
            submitted_query = ds_request.query or ""
            replay = (
                _load_completed_replay(
                    actor_id=actor_id,
                    conversation_id=ds_request.conversation_id,
                    turn_id=ds_request.turn_id,
                    original_query=submitted_query,
                )
                if submitted_query
                else None
            )
            if replay is not None:
                logger.debug(
                    "request_id=%s idempotent_replay_count count=1",
                    request_id,
                )
                if ds_request.stream:
                    cancellation = StreamCancellation()
                    return StreamingResponse(
                        stream_events_as_sse(
                            _stream_replayed_turn(
                                replay,
                                request_id,
                                trace_id=(
                                    current_request_context().trace_id
                                    if current_request_context() is not None
                                    else None
                                ),
                                request_started_logged=True,
                                operation_accumulator=current_operation_accumulator(),
                            ),
                            request_id=request_id,
                            cancellation=cancellation,
                            heartbeat_interval_seconds=(
                                settings.answer_stream_heartbeat_interval_seconds
                            ),
                        ),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                return _replay_response(replay, request_id)
            logger.debug(
                "request_id=%s  actor_id_present=%s  mode=doubt_solver  query_len=%d "
                "has_image=%s  exam_id=%s  language=%s  legacy_language_defaulted=%s "
                "— invoke started",
                request_id,
                bool(actor_id),
                len(ds_request.query or ""),
                ds_request.image is not None,
                ds_request.exam_id or "",
                ds_request.language,
                "language" not in payload,
            )

            query = ds_request.query or ""
            original_query = query
            entry_classification = None
            entry_classification_result = None
            source_modality = "text"
            image_confidence: float | None = None
            image_uncertain = False
            classifier_confidence: float | None = None
            classifier_fallback = False
            if ds_request.image is not None:
                source_modality = "image"
                if image_question_classifier is None:
                    return {
                        "success": False,
                        "answer": (
                            "Image questions are not enabled. Please enter the question as text."
                        ),
                        "request_id": request_id,
                        "mode": "doubt_solver",
                        "image_classification_status": (
                            ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE.value
                        ),
                    }
                image_result = image_classifier_node(
                    ImageClassifierNodeInput(
                        request_id=request_id,
                        image=ds_request.image,
                        instruction=ds_request.query,
                    ),
                    classifier=image_question_classifier,
                )
                if image_result.status != ImageClassificationStatus.CLASSIFIED:
                    return {
                        "success": False,
                        "answer": image_result.user_message or "The image could not be processed.",
                        "request_id": request_id,
                        "mode": "doubt_solver",
                        "image_classification_status": image_result.status.value,
                    }
                if image_result.classification is None or not image_result.normalized_query:
                    return {
                        "success": False,
                        "answer": "The image could not be processed.",
                        "request_id": request_id,
                        "mode": "doubt_solver",
                        "image_classification_status": (
                            ImageClassificationStatus.REJECTED_UNREADABLE_IMAGE.value
                        ),
                    }
                query = image_result.normalized_query
                if not original_query:
                    original_query = query
                    replay = _load_completed_replay(
                        actor_id=actor_id,
                        conversation_id=ds_request.conversation_id,
                        turn_id=ds_request.turn_id,
                        original_query=original_query,
                    )
                    if replay is not None:
                        return _replay_response(replay, request_id)
                entry_classification = image_result.classification
                entry_classification_result = adapt_image_classification(
                    image_result,
                    request_id=request_id,
                    coordinator=classification_coordinator,
                )
                entry_classification = entry_classification_result.raw
                image_metadata = image_result.image_parse_metadata
                image_confidence = min(
                    image_metadata.extraction_confidence,
                    image_metadata.classification_confidence,
                )
                image_uncertain = bool(image_metadata.warnings) or (
                    image_metadata.has_visual
                    and image_metadata.visual_context.confidence < image_confidence
                )
                classifier_confidence = entry_classification.confidence
                classifier_fallback = entry_classification.classification_source == "fallback"
                update_request_summary(
                    request_type="image",
                    subject=entry_classification.subject,
                    intent=entry_classification.intent,
                    difficulty=entry_classification.difficulty,
                    classifier_source=entry_classification.classification_source,
                )
                log_event(
                    "classification_completed",
                    component="image_question.classifier",
                    stage="understanding",
                    status="completed",
                    details={
                        "subject": entry_classification.subject,
                        "intent": entry_classification.intent,
                        "difficulty": entry_classification.difficulty,
                        "classifier_source": entry_classification.classification_source,
                        "need_web_search": entry_classification.need_web_search,
                        "web_search_reason": entry_classification.web_search_reason,
                        "search_term_present": bool(entry_classification.web_search_query),
                    },
                )

            # Orchestrated path — ENABLE_ORCHESTRATED_DOUBT_SOLVER=true
            if (
                get_settings().enable_orchestrated_doubt_solver
                and orchestrated_doubt_solver_graph is not None
            ):
                if ds_request.stream:
                    if orchestrated_adapter is None:
                        return {
                            "success": False,
                            "answer": "Streaming is unavailable.",
                            "request_id": request_id,
                            "mode": "doubt_solver",
                        }

                    logger.debug(
                        "request_id=%s  generation_path=orchestrated_stream — invoke started",
                        request_id,
                    )

                    mapped_classification = (
                        entry_classification_result.classification
                        if entry_classification_result is not None
                        and entry_classification is not None
                        else map_to_orchestrated_classification(
                            entry_classification,
                            query=query,
                            request_id=request_id,
                        )
                        if entry_classification is not None
                        else None
                    )
                    cancellation = StreamCancellation()
                    stream_input = StreamDoubtSolverInput(
                        request_id=request_id,
                        trace_id=(
                            current_request_context().trace_id
                            if current_request_context() is not None
                            else None
                        ),
                        actor_id=actor_id,
                        conversation_id=ds_request.conversation_id,
                        turn_id=ds_request.turn_id,
                        query=query,
                        original_query=original_query,
                        language=ds_request.language,
                        classification=mapped_classification,
                        source_modality=source_modality,
                        image_confidence=image_confidence,
                        image_uncertain=image_uncertain,
                        classifier_confidence=classifier_confidence,
                        classifier_fallback=classifier_fallback,
                        exam_id=ds_request.exam_id,
                        exam_stage=ds_request.exam_stage,
                        exam_profile_id=ds_request.exam_profile_id,
                        should_cancel=cancellation.is_cancelled,
                        cancellation_reason=lambda: cancellation.reason,
                        request_started_logged=True,
                        defer_practice_start_until_committed=True,
                        initial_llm_usage_records=snapshot_llm_usage_records(),
                        operation_accumulator=current_operation_accumulator(),
                    )
                    stream_kwargs = {
                        "adapter": orchestrated_adapter,
                        "conversation_persistence": conversation_persistence,
                        "follow_up_resolver": follow_up_resolver,
                        "conversation_understanding": conversation_understanding,
                        "practice_launcher": (practice_async_launcher),
                        "practice_request_interpreter": practice_request_interpreter,
                    }
                    events = stream_doubt_solver(
                        stream_input,
                        **stream_kwargs,
                    )
                    return StreamingResponse(
                        stream_events_as_sse(
                            events,
                            request_id=request_id,
                            cancellation=cancellation,
                            heartbeat_interval_seconds=(
                                get_settings().answer_stream_heartbeat_interval_seconds
                            ),
                            on_terminal_frame_committed=(
                                lambda event: practice_async_launcher.start(
                                    event.data.practice_test_id,
                                    event.request_id,
                                )
                                if practice_async_launcher is not None
                                and event.type == "practice_generation_started"
                                and event.data is not None
                                else None
                            ),
                            on_terminal_frame_abandoned=(
                                lambda event: practice_async_launcher.abort(
                                    event.data.practice_test_id,
                                    "PRACTICE_START_EVENT_DELIVERY_FAILED",
                                    event.request_id,
                                )
                                if practice_async_launcher is not None
                                and event.type == "practice_generation_started"
                                and event.data is not None
                                else None
                            ),
                        ),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "X-Accel-Buffering": "no",
                        },
                    )

                orchestrated_input = {
                    "request_id": request_id,
                    "actor_id": actor_id,
                    "conversation_id": ds_request.conversation_id,
                    "turn_id": ds_request.turn_id,
                    "query": query,
                    "original_query": original_query,
                    "language": ds_request.language,
                    "exam_id": ds_request.exam_id,
                    "exam_stage": ds_request.exam_stage,
                    "exam_profile_id": ds_request.exam_profile_id,
                    "classification": (
                        entry_classification_result.classification
                        if entry_classification_result is not None
                        and entry_classification is not None
                        else map_to_orchestrated_classification(
                            entry_classification,
                            query=query,
                            request_id=request_id,
                        )
                        if entry_classification is not None
                        else None
                    ),
                    "retrieval_context": {},
                    "context_text": "",
                    "answer": None,
                    "final_answer": None,
                    "response_type": None,
                    "practice_test_id": None,
                    "conversation_context": "",
                    "conversation_relation": None,
                    "conversation_preparation": None,
                    "query_classification": (
                        entry_classification.model_dump()
                        if entry_classification is not None
                        else None
                    ),
                    "source_modality": source_modality,
                }
                generation_started = time.monotonic()
                try:
                    with stage_span("doubt_solver.generate", route="orchestrated_non_stream"):
                        orchestrated_result = orchestrated_doubt_solver_graph.invoke(
                            orchestrated_input
                        )
                except Exception as exc:
                    log_event(
                        "generation_failed",
                        component="doubt_solver.generator",
                        stage="generate",
                        status="failed",
                        duration_ms=int((time.monotonic() - generation_started) * 1000),
                        error_code="GENERATION_FAILED",
                        details={"error_type": type(exc).__name__},
                        level=logging.ERROR,
                    )
                    raise
                final_answer = orchestrated_result.get("final_answer") or {}
                authoritative_answer = final_answer.get("content") or ""
                accepted = final_answer.get("quality_status") in {
                    "checked",
                    "passed_quality_gate",
                }
                generation_duration_ms = int((time.monotonic() - generation_started) * 1000)
                classification = orchestrated_result.get("classification") or {}
                current_summary = current_request_summary()
                resolved_request_type = (
                    "image"
                    if initial_request_type == "image"
                    else (
                        "follow_up"
                        if orchestrated_result.get("conversation_context")
                        or classification.get("requires_recent_conversation", False)
                        else "standalone"
                    )
                )
                update_request_type(resolved_request_type)
                update_request_summary(
                    request_type=resolved_request_type,
                    subject=str(classification.get("subject") or "general"),
                    intent=str(classification.get("intent") or "explain"),
                    difficulty=str(classification.get("difficulty") or "default"),
                    classifier_source=str(classification.get("classification_source") or "llm"),
                    generation_route=(
                        current_summary.generation_route
                        if current_summary and current_summary.generation_route
                        else "orchestrated_non_stream"
                    ),
                    generation_duration_ms=generation_duration_ms,
                    quality_status=str(final_answer.get("quality_status") or "unknown"),
                    repair_attempted=bool(final_answer.get("was_regenerated")),
                )
                if not accepted:
                    log_event(
                        "generation_failed",
                        component="doubt_solver.generator",
                        stage="generate",
                        status="failed",
                        duration_ms=generation_duration_ms,
                        error_code="QUALITY_GATE_FAILED",
                        details={"route": "orchestrated_non_stream"},
                        level=logging.ERROR,
                    )
                log_event(
                    "quality_validation_completed",
                    component="doubt_solver.quality",
                    stage="validate_quality",
                    status=str(final_answer.get("quality_status") or "unknown"),
                    error_code=None if accepted else "QUALITY_GATE_FAILED",
                    details={"repair_attempted": bool(final_answer.get("was_regenerated"))},
                )
                logger.debug(
                    "request_id=%s  generation_path=orchestrated_non_stream — invoke succeeded",
                    request_id,
                )
                persistence_result = _persist_completed_result(
                    request=ds_request,
                    actor_id=actor_id,
                    original_query=original_query,
                    result=orchestrated_result,
                    request_id=request_id,
                )
                response_type = orchestrated_result.get("response_type")
                practice_test_id = orchestrated_result.get("practice_test_id")
                if (
                    response_type == "practice_generation"
                    and practice_test_id
                    and practice_async_launcher is not None
                ):
                    history_linked = (
                        persistence_result is not None
                        and persistence_result.history_write_status
                        in {"succeeded", "idempotent_replay"}
                    )
                    if not history_linked:
                        practice_async_launcher.abort(
                            practice_test_id,
                            "PRACTICE_CONVERSATION_LINKAGE_FAILED",
                            request_id,
                        )
                        return {
                            "success": False,
                            "request_id": request_id,
                            "mode": "doubt_solver",
                            "error": "PRACTICE_CONVERSATION_LINKAGE_FAILED",
                        }
                    practice_async_launcher.start(practice_test_id, request_id)
                response = {
                    "schema_version": "1",
                    "status": "completed" if accepted else "failed",
                    "success": accepted,
                    "request_id": request_id,
                    "mode": "doubt_solver",
                    "answer": authoritative_answer,
                    "content": {
                        "format": "markdown",
                        "value": authoritative_answer,
                    },
                    "classification": orchestrated_result.get("classification"),
                }
                if response_type == "practice_generation" and practice_test_id:
                    response["responseType"] = response_type
                    response["practiceTestId"] = practice_test_id
                return response

            # Legacy path — ENABLE_ORCHESTRATED_DOUBT_SOLVER=false (default)
            graph_input = {
                "request_id": request_id,
                "actor_id": actor_id,
                "conversation_id": ds_request.conversation_id,
                "turn_id": ds_request.turn_id,
                "query": query,
                "original_query": original_query,
                "mode": ds_request.mode,
                "language": ds_request.language,
                "exam_id": ds_request.exam_id,
                "exam_stage": ds_request.exam_stage,
                "exam_profile_id": ds_request.exam_profile_id,
                "classification": (
                    entry_classification.model_dump() if entry_classification is not None else None
                ),
                "answer": None,
                "final_answer": None,
                "answer_source": None,
                "is_truncated": False,
                "response": None,
                # Part 9 context-pipeline fields — initialised to safe no-op defaults.
                "should_retrieve": False,
                "kb_results": None,
                "dynamodb_records": None,
                "answer_context": None,
                "context_source_count": 0,
                "used_retrieval": False,
                "context_used": False,
                "service_error": False,
                "retrieval_context": None,
                "conversation_context": "",
                "conversation_relation": None,
            }
            generation_started = time.monotonic()
            try:
                with stage_span("doubt_solver.generate", route="legacy_non_stream"):
                    result = doubt_solver_graph.invoke(graph_input)
            except Exception as exc:
                log_event(
                    "generation_failed",
                    component="doubt_solver.generator",
                    stage="generate",
                    status="failed",
                    duration_ms=int((time.monotonic() - generation_started) * 1000),
                    error_code="GENERATION_FAILED",
                    details={"error_type": type(exc).__name__},
                    level=logging.ERROR,
                )
                raise
            generation_duration_ms = int((time.monotonic() - generation_started) * 1000)
            result_classification = result.get("classification") or {}
            result_final = result.get("final_answer") or {}
            legacy_accepted = result_final.get("quality_status") in {
                "checked",
                "passed_quality_gate",
            }
            current_summary = current_request_summary()
            resolved_request_type = (
                initial_request_type
                if initial_request_type == "image"
                else (
                    "follow_up"
                    if result_classification.get("requires_recent_conversation")
                    else "standalone"
                )
            )
            update_request_type(resolved_request_type)
            update_request_summary(
                request_type=resolved_request_type,
                subject=str(result_classification.get("subject") or "general"),
                intent=str(result_classification.get("intent") or "explain"),
                difficulty=str(result_classification.get("difficulty") or "default"),
                classifier_source=str(result_classification.get("classification_source") or "rule"),
                generation_route=(
                    current_summary.generation_route
                    if current_summary and current_summary.generation_route
                    else "legacy_non_stream"
                ),
                generation_duration_ms=generation_duration_ms,
                quality_status=str(result_final.get("quality_status") or "unknown"),
                repair_attempted=bool(result_final.get("was_regenerated")),
            )
            log_event(
                "generation_completed" if legacy_accepted else "generation_failed",
                component="doubt_solver.generator",
                stage="generate",
                status="completed" if legacy_accepted else "failed",
                duration_ms=generation_duration_ms,
                error_code=None if legacy_accepted else "QUALITY_GATE_FAILED",
                details={"route": "legacy_non_stream"},
                level=logging.INFO if legacy_accepted else logging.ERROR,
            )
            _persist_completed_result(
                request=ds_request,
                actor_id=actor_id,
                original_query=original_query,
                result=result,
                request_id=request_id,
            )
            logger.debug(
                "request_id=%s  generation_path=legacy_non_stream — invoke succeeded",
                request_id,
            )
            return result["response"]

        # --- Demo / default path ------------------------------------------
        request = AgentRequest.model_validate(payload)
        logger.debug(
            "request_id=%s actor_id_present=%s mode=%s message_len=%d — invoke started",
            request_id,
            bool(request.user_id),
            request.mode,
            len(request.message),
        )
        graph_input = {
            "request_id": request_id,
            "message": request.message,
            "user_id": request.user_id,
            "mode": request.mode,
            "answer": None,
        }
        result = graph.invoke(graph_input)
        answer: str = result.get("answer") or ""
        response = AgentResponse(
            success=True,
            answer=answer,
            request_id=request_id,
            mode=request.mode,
        )
        logger.debug("request_id=%s — invoke succeeded", request_id)
        return response.model_dump()

    except ConversationTurnConflictError:
        logger.warning("request_id=%s conversation_turn_conflict=true", request_id)
        return {
            "success": False,
            "answer": "This turn ID is already associated with a different request.",
            "request_id": request_id,
            "mode": payload.get("mode", "demo"),
        }

    except ValidationError as exc:
        logger.warning(
            "request_id=%s — validation error  error_count=%d",
            request_id,
            exc.error_count(),
        )
        return {
            "success": False,
            "answer": "Validation error: request payload is invalid.",
            "request_id": request_id,
            "mode": payload.get("mode", "demo"),
        }

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "request_id=%s unexpected_error=true error_type=%s",
            request_id,
            type(exc).__name__,
        )
        return {
            "success": False,
            "answer": "Internal error: request failed safely.",
            "request_id": request_id,
            "mode": payload.get("mode", "demo"),
        }


# ---------------------------------------------------------------------------
# Local runner — lets you do `python main.py` without agentcore CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run()
