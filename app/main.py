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

# ---------------------------------------------------------------------------
# Bootstrap — runs once when the module is imported
# ---------------------------------------------------------------------------

settings = get_settings()
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

logger = logging.getLogger(__name__)

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
log_event(
    "runtime_started",
    component="runtime.bootstrap",
    stage="startup",
    status="ready",
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

        # Mock path — no real provider calls, no network I/O.
        # Used when ENABLE_REAL_LLM=false (default, tests, local dev).
        # The LlmOrchestrator still resolves routes and builds prompts
        # (file reads only); MockModelExecutor short-circuits provider calls.
        from services.llm.orchestration.orchestrator import (  # noqa: PLC0415
            MockModelExecutor,
        )

        _model_executor = MockModelExecutor(
            content=(
                "**Final Answer:** [orchestrated-mock] Mock answer — set ENABLE_REAL_LLM=true "
                "for a real LLM response."
            )
        )
    else:
        # Real provider chain — wired for production use.
        from services.llm.orchestration.model_config_resolver import (  # noqa: PLC0415
            ModelConfigResolver,
        )
        from services.llm.orchestration.model_execution import (  # noqa: PLC0415
            ProviderAdapterExecutor,
            RegistryBackedModelExecutor,
        )
        from services.llm.providers.provider_factory import (  # noqa: PLC0415
            ProviderAdapterFactory,
        )
        from services.secrets.env_secret_resolver import EnvSecretResolver  # noqa: PLC0415
        from services.secrets.provider_credentials import (  # noqa: PLC0415
            ProviderCredentialResolver,
        )

        _secret_resolver = EnvSecretResolver()
        _credential_resolver = ProviderCredentialResolver(
            secret_resolver=_secret_resolver
        )
        _adapter_executor = ProviderAdapterExecutor(
            credential_resolver=_credential_resolver,
            provider_factory=ProviderAdapterFactory(),
        )
        _model_executor = RegistryBackedModelExecutor(
            provider_executor=_adapter_executor,
            model_config_resolver=ModelConfigResolver(),
        )

        # Preflight: block startup if any Azure deployment name is still a
        # placeholder (YOUR_*, TODO*, REPLACE_ME*, PLACEHOLDER*).
        # This catches misconfiguration before the first real provider call.
        # [SECURITY] Error message includes only model alias names — no keys,
        # endpoints, prompts, or query content.
        from services.llm.orchestration.config_registry import (  # noqa: PLC0415
            get_registry,
        )
        from services.llm.orchestration.errors import LlmConfigValidationError  # noqa: PLC0415

        try:
            get_registry().validate_real_mode_deployments()
        except LlmConfigValidationError as _preflight_err:
            raise ConfigurationError(str(_preflight_err)) from _preflight_err

    _orchestrator = LlmOrchestrator(model_executor=_model_executor)
    from services.conversation.follow_up_query_resolver import (  # noqa: PLC0415
        FollowUpQueryResolver,
    )

    follow_up_resolver = FollowUpQueryResolver(orchestrator=_orchestrator)
    _adapter = AnswerGenerationAdapter(orchestrator=_orchestrator)
    orchestrated_adapter = _adapter
    orchestrated_doubt_solver_graph = build_orchestrated_doubt_solver_graph(
        _adapter,
        conversation_persistence=conversation_persistence,
        follow_up_resolver=follow_up_resolver,
        conversation_understanding=conversation_understanding,
    )
    logger.info(
        "Orchestrated graph built  enable_real_llm=%s",
        settings.enable_real_llm,
    )

# AgentCore application object
app = BedrockAgentCoreApp()

logger.info(
    "Agent initialised  app_env=%s  model_provider=%s",
    settings.app_env,
    settings.model_provider,
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
    return {
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


def _stream_replayed_turn(
    turn: CompletedConversationTurn,
    request_id: str,
    *,
    trace_id: str | None = None,
    request_started_logged: bool = False,
) -> Iterator[DoubtSolverStreamEvent]:
    started_at = time.monotonic()
    with bind_request_context(
        request_id=request_id,
        trace_id=trace_id,
        conversation_id=turn.conversation_id,
        turn_id=turn.turn_id,
        request_type="standalone",
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
                metadata={"request_id": request_id, "replayed": True},
                response=DoubtSolverFinalResponse(
                    request_id=request_id,
                    content=ResponseContent(value=turn.final_answer),
                    answer=turn.final_answer,
                    final_answer=final_answer,
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
) -> None:
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
    )
    conversation_persistence.persist_completed_turn(
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
                        "search_term_present": bool(
                            entry_classification.web_search_query
                        ),
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
                    events = stream_doubt_solver(
                        StreamDoubtSolverInput(
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
                            should_cancel=cancellation.is_cancelled,
                            cancellation_reason=lambda: cancellation.reason,
                            request_started_logged=True,
                            initial_llm_usage_records=snapshot_llm_usage_records(),
                        ),
                        adapter=orchestrated_adapter,
                        conversation_persistence=conversation_persistence,
                        follow_up_resolver=follow_up_resolver,
                        conversation_understanding=conversation_understanding,
                    )
                    return StreamingResponse(
                        stream_events_as_sse(
                            events,
                            request_id=request_id,
                            cancellation=cancellation,
                            heartbeat_interval_seconds=(
                                get_settings().answer_stream_heartbeat_interval_seconds
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
                generation_duration_ms = int(
                    (time.monotonic() - generation_started) * 1000
                )
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
                    classifier_source=str(
                        classification.get("classification_source") or "llm"
                    ),
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
                    details={
                        "repair_attempted": bool(final_answer.get("was_regenerated"))
                    },
                )
                logger.debug(
                    "request_id=%s  generation_path=orchestrated_non_stream — invoke succeeded",
                    request_id,
                )
                _persist_completed_result(
                    request=ds_request,
                    actor_id=actor_id,
                    original_query=original_query,
                    result=orchestrated_result,
                    request_id=request_id,
                )
                return {
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
                "classification": (
                    entry_classification.model_dump()
                    if entry_classification is not None
                    else None
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
                classifier_source=str(
                    result_classification.get("classification_source") or "rule"
                ),
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
