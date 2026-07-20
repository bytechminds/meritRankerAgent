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
import uuid

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
from schemas.doubt_solver import DoubtSolverRequest
from schemas.image_question_classification import ImageClassificationStatus
from schemas.request import AgentRequest
from schemas.response import AgentResponse
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
configure_logging(settings.log_level)
try:
    get_exam_response_profile_resolver()
except ExamResponseProfileConfigError as exc:
    raise ConfigurationError(str(exc)) from exc

logger = logging.getLogger(__name__)

# Build graphs once at startup — compilation is not free.
graph = build_demo_graph()
doubt_solver_graph = build_doubt_solver_graph()

image_question_classifier = None
if settings.image_classifier_enabled:
    from services.image_question_classification.factory import (  # noqa: PLC0415
        build_image_question_classifier,
    )

    image_question_classifier = build_image_question_classifier(settings)

# Orchestrated graph — only built when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true.
# Default is false; existing tests are unaffected.
orchestrated_doubt_solver_graph = None
orchestrated_adapter: AnswerGenerationAdapter | None = None
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
                "[orchestrated-mock] Mock answer — set ENABLE_REAL_LLM=true "
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
    _adapter = AnswerGenerationAdapter(orchestrator=_orchestrator)
    orchestrated_adapter = _adapter
    orchestrated_doubt_solver_graph = build_orchestrated_doubt_solver_graph(_adapter)
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


@app.entrypoint
def invoke(payload: dict) -> dict | Response:
    """Handle a single invocation request.

    Args:
        payload: Raw dict from the HTTP request body.

    Returns:
        Serialised AgentResponse dict.
    """
    request_id = str(uuid.uuid4())

    try:
        mode = payload.get("mode", "demo")

        # --- Doubt Solver path --------------------------------------------
        if mode == "doubt_solver":
            ds_request = DoubtSolverRequest.model_validate(payload)
            logger.info(
                "request_id=%s  user_id=%s  mode=doubt_solver  query_len=%d "
                "has_image=%s — invoke started",
                request_id,
                ds_request.user_id,
                len(ds_request.query or ""),
                ds_request.image is not None,
            )

            query = ds_request.query or ""
            graph_config = {
                "configurable": {
                    "exam_id": ds_request.exam_id,
                    "exam_stage": ds_request.exam_stage,
                }
            }
            entry_classification = None
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
                entry_classification = image_result.classification
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

                    logger.info(
                        "request_id=%s — invoke started (doubt_solver_orchestrated_stream)",
                        request_id,
                    )

                    mapped_classification = (
                        map_to_orchestrated_classification(
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
                            query=query,
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
                        ),
                        adapter=orchestrated_adapter,
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
                    "query": query,
                    "classification": (
                        map_to_orchestrated_classification(
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
                }
                orchestrated_result = orchestrated_doubt_solver_graph.invoke(
                    orchestrated_input,
                    config=graph_config,
                )
                logger.info(
                    "request_id=%s — invoke succeeded (doubt_solver_orchestrated)",
                    request_id,
                )
                return {
                    "schema_version": "1",
                    "status": "completed",
                    "success": True,
                    "request_id": request_id,
                    "mode": "doubt_solver",
                    "answer": orchestrated_result.get("answer") or "",
                    "content": {
                        "format": "markdown",
                        "value": orchestrated_result.get("answer") or "",
                    },
                    "classification": orchestrated_result.get("classification"),
                }

            # Legacy path — ENABLE_ORCHESTRATED_DOUBT_SOLVER=false (default)
            graph_input = {
                "request_id": request_id,
                "query": query,
                "user_id": ds_request.user_id,
                "mode": ds_request.mode,
                "language": ds_request.language,
                "classification": (
                    entry_classification.model_dump()
                    if entry_classification is not None
                    else None
                ),
                "answer": None,
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
            }
            result = doubt_solver_graph.invoke(graph_input, config=graph_config)
            logger.info("request_id=%s — invoke succeeded (doubt_solver)", request_id)
            return result["response"]

        # --- Demo / default path ------------------------------------------
        request = AgentRequest.model_validate(payload)
        logger.info(
            "request_id=%s  user_id=%s  mode=%s  message_len=%d — invoke started",
            request_id,
            request.user_id,
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
        logger.info("request_id=%s — invoke succeeded", request_id)
        return response.model_dump()

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

    except Exception:  # noqa: BLE001
        logger.exception("request_id=%s — unexpected error", request_id)
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
