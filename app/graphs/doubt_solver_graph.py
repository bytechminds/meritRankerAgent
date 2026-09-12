"""
app/graphs/doubt_solver_graph.py
---------------------------------
LangGraph StateGraph for the Doubt Solver V1 workflow.

Node layout (Part 9):
    START
    ──► classify_query
    ──► plan_context          (decides whether retrieval should run)
    ──► retrieve_kb_context   (KB retrieval — no-op when disabled or retrieval_need=none)
    ──► fetch_dynamodb_records (DynamoDB fetch — no-op when disabled or no record_ids)
    ──► build_answer_context  (assembles bounded context string)
    ──► generate_answer       (calls answer generator with optional context)
    ──► build_response
    ──► END

Feature flags:
    ENABLE_KB_RETRIEVAL=false (default)  → retrieve_kb_context no-ops
    ENABLE_DYNAMODB_FETCH=false (default) → fetch_dynamodb_records no-ops

All KB and DynamoDB calls go through services — graph nodes never touch boto3.
Retrieved content is UNTRUSTED reference material, not instructions.

Public API:
    build_doubt_solver_graph() -> CompiledGraph
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypedDict, cast

from langgraph.graph import END, START, StateGraph

from config import get_settings
from features.practice_generation.agentcore_async import PracticeLaunchError
from features.practice_generation.planning import (
    PRACTICE_ASYNC_NOT_CONFIGURED,
    PracticeRequestCountError,
    canonical_practice_subject,
    decide_practice_launch,
    practice_async_unavailable_message,
    practice_route_enabled,
    required_fresh_evidence_count,
    resolve_practice_freshness_requirement,
    resolve_practice_request,
    validate_practice_requested_count,
)
from features.practice_generation.request_intelligence import PracticeRequestInterpreter
from features.practice_generation.schemas import (
    PracticeGenerationRequest,
    PracticeLaunchResult,
)
from observability import log_event, update_request_summary
from schemas.conversation import ConversationCandidateCard, ConversationPreparation
from schemas.doubt_solver import (
    CanonicalLanguage,
    DoubtSolverResponse,
    FinalAnswerResult,
    QueryClassification,
)
from schemas.retrieval import KnowledgeBaseResult
from services.answer_generator_service import generate_answer
from services.bedrock_kb_service import (
    KnowledgeBaseConfigurationError,
    KnowledgeBaseServiceError,
    retrieve_similar_context,
)
from services.classification.academic_classifier import map_academic_classification
from services.classification.contracts import ClassificationStageResult
from services.classification.coordinator import ClassificationCoordinator
from services.context_builder_service import build_doubt_solver_context
from services.conversation.selected_context_builder import (
    build_context_aware_clarification,
    build_selected_generation_context,
)
from services.doubt_solver.answer_correctness import (
    requires_independent_correctness_verification,
)
from services.doubt_solver.answer_quality import generation_failure_message
from services.doubt_solver.final_answer import build_final_answer_result
from services.dynamodb_service import DynamoDbConfigurationError, DynamoDbServiceError
from services.query_classifier_service import classify_query
from services.question_record_service import fetch_question_records_by_ids
from tools.web_search.models import FreshEvidenceBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph internal state — plain TypedDict; Pydantic only at boundaries
# ---------------------------------------------------------------------------


class DoubtSolverGraphState(TypedDict):
    """Data carried between nodes inside the LangGraph workflow."""

    request_id: str
    query: str
    original_query: str
    actor_id: str
    mode: str
    language: str
    exam_id: str | None
    exam_stage: str | None
    exam_profile_id: str | None
    classification: dict | None  # serialised QueryClassification
    answer: str | None
    final_answer: dict | None
    response_type: str | None
    practice_test_id: str | None
    answer_source: str | None  # "mock" | "llm" | "fallback"
    is_truncated: bool
    response: dict | None  # serialised DoubtSolverResponse
    # Part 9 context-pipeline fields
    should_retrieve: bool  # set by plan_context_node
    kb_results: list | None  # list of serialised KnowledgeBaseResult dicts
    dynamodb_records: list | None  # list of raw DynamoDB record dicts
    answer_context: str | None  # bounded context string for answer generator
    context_source_count: int  # number of sources included in context
    used_retrieval: bool  # True if KB returned ≥1 result
    context_used: bool  # True if context was passed to answer generator
    service_error: bool  # True if KB or DynamoDB service error occurred
    retrieval_context: dict | None  # internal S3 Vector retrieval contract
    doubt_pattern_context: dict | None  # answer-redacted Pattern guidance only
    conversation_context: str  # selected recent turn context, when relevant
    conversation_relation: dict | None


# ---------------------------------------------------------------------------
# Nodes — original
# ---------------------------------------------------------------------------


def classify_query_node(state: DoubtSolverGraphState) -> dict:
    """Call the classifier service and write the result to state."""
    existing = state.get("classification")
    classification = (
        QueryClassification.model_validate(existing)
        if existing is not None
        else classify_query(state["query"])
    )
    relation = state.get("conversation_relation") or {}
    if relation and not relation.get("requires_recent_conversation"):
        classification = classification.model_copy(update={"requires_recent_conversation": False})
    logger.debug(
        "request_id=%s  classify_query  intent=%s  confidence=%.2f",
        state["request_id"],
        classification.intent,
        classification.confidence,
    )
    return {"classification": classification.model_dump()}


# ---------------------------------------------------------------------------
# Nodes — Part 9 context pipeline
# ---------------------------------------------------------------------------


def plan_context_node(state: DoubtSolverGraphState) -> dict:
    """Decide whether KB retrieval should run for this request.

    Sets should_retrieve=True when the classifier suggests retrieval may help
    and the query is non-empty.  The actual feature-flag check happens inside
    retrieve_similar_context (which no-ops when ENABLE_KB_RETRIEVAL=false).
    """
    classification_dict = state.get("classification") or {}
    retrieval_need: str = classification_dict.get("retrieval_need", "none")
    query: str = state.get("query", "")

    # "none" means the classifier is confident retrieval won't help.
    should_retrieve = retrieval_need != "none" and bool(query)
    logger.debug(
        "request_id=%s  plan_context  retrieval_need=%s  should_retrieve=%s",
        state.get("request_id", ""),
        retrieval_need,
        should_retrieve,
    )
    return {"should_retrieve": should_retrieve}


def retrieve_kb_context_node(state: DoubtSolverGraphState) -> dict:
    """Call KB retrieval service and store results in state.

    No-ops when:
    - should_retrieve is False (classifier said no retrieval needed)
    - ENABLE_KB_RETRIEVAL=false (service returns retrieval_source="disabled")

    On KnowledgeBaseConfigurationError or KnowledgeBaseServiceError:
    - Logs a safe warning (no query text, no raw response).
    - Sets service_error=True so build_response marks needs_review.
    - Continues with empty KB results.
    """

    if get_settings().retrieval_provider == "s3_vector":
        return _retrieve_s3_vector_context_node(state)

    if not state.get("should_retrieve"):
        return {"kb_results": None, "used_retrieval": False}

    try:
        retrieval_response = retrieve_similar_context(state["query"])

        if retrieval_response.retrieval_source == "disabled":
            # KB flag is off — service returned early without any AWS call.
            return {"kb_results": None, "used_retrieval": False}

        results_dicts = [r.model_dump() for r in retrieval_response.results]
        used = len(results_dicts) > 0
        logger.debug(
            "request_id=%s  retrieve_kb_context  result_count=%d",
            state.get("request_id", ""),
            len(results_dicts),
        )
        return {"kb_results": results_dicts, "used_retrieval": used}

    except (KnowledgeBaseConfigurationError, KnowledgeBaseServiceError) as exc:
        logger.warning(
            "request_id=%s  retrieve_kb_context  error=%s — continuing without KB context",
            state.get("request_id", ""),
            type(exc).__name__,
        )
        return {"kb_results": None, "used_retrieval": False, "service_error": True}


def _retrieve_s3_vector_context_node(state: DoubtSolverGraphState) -> dict:
    """Use the graph-facing context service for both legacy and canonical S3 paths."""
    from services.context_retrieval.context_models import ContextRetrievalRequest  # noqa: PLC0415
    from services.context_retrieval.context_retrieval_service import (  # noqa: PLC0415
        get_context_retrieval_service,
    )

    classification_dict = state.get("classification") or {}
    request = ContextRetrievalRequest(
        request_id=state["request_id"],
        query=state.get("query", ""),
        subject=str(classification_dict.get("subject", "general")),
        intent=_legacy_intent_for_retrieval(str(classification_dict.get("intent", "explain"))),
        difficulty=str(classification_dict.get("difficulty", "default")),
        confidence=classification_dict.get("confidence"),
        topic=classification_dict.get("topic"),
        topic_confidence=classification_dict.get("topic_confidence"),
        pattern_topic_candidate=classification_dict.get("pattern_topic_candidate"),
        pattern_family_candidate=classification_dict.get("pattern_family_candidate"),
        retrieval_tags=classification_dict.get("retrieval_tags") or [],
    )
    result = get_context_retrieval_service().retrieve_context(request)
    context_text = result.context_text or ""
    retrieval_used = result.retrieval_used
    doubt_pattern_context = (
        result.doubt_pattern_context.model_dump(by_alias=True)
        if result.doubt_pattern_context is not None
        else None
    )
    return {
        "kb_results": None,
        "dynamodb_records": None,
        "answer_context": context_text or None,
        "context_source_count": 1 if retrieval_used else 0,
        "used_retrieval": retrieval_used,
        "context_used": bool(context_text),
        "retrieval_context": result.retrieval_context.model_dump(by_alias=True),
        "doubt_pattern_context": doubt_pattern_context,
    }


def _legacy_intent_for_retrieval(intent: str) -> str:
    """Map the legacy classifier intent enum to the retrieval gate vocabulary."""
    return {
        "solve_question": "solve",
        "explain_concept": "explain",
        "explain_option": "explain",
        "practice_question": "practice",
        "visualize_question": "visualize",
    }.get(intent, "explain")


def _doubt_pattern_context_from_state(state: dict) -> object | None:
    """Validate internal answer-redacted Pattern guidance before prompt handoff."""
    return _doubt_pattern_context_from_payload(
        state.get("doubt_pattern_context"),
        request_id=state.get("request_id", ""),
    )


def _doubt_pattern_context_from_payload(
    raw_context: object,
    *,
    request_id: str,
) -> object | None:
    """Validate a server-only serialized Pattern context before prompt handoff."""
    if not isinstance(raw_context, dict):
        return None
    from pydantic import ValidationError  # noqa: PLC0415

    from retrieval.pattern_intelligence import DoubtPatternContext  # noqa: PLC0415

    try:
        return DoubtPatternContext.model_validate(raw_context)
    except ValidationError:
        logger.warning(
            "request_id=%s invalid_doubt_pattern_context=true",
            request_id,
        )
        return None


def fetch_dynamodb_records_node(state: DoubtSolverGraphState) -> dict:
    """Fetch DynamoDB question records referenced by KB results.

    No-ops when:
    - ENABLE_DYNAMODB_FETCH=false — returns immediately, no service call made
    - No KB results contain record_ids

    On DynamoDbConfigurationError or DynamoDbServiceError:
    - Logs a safe warning.
    - Sets service_error=True so build_response marks needs_review.
    - Continues with KB context only.

    Note: Only question records are fetched in Part 9.
    Pattern record fetching is deferred to a future part.
    """
    # Deferred import — ensures dotenv has loaded before config is read.

    if get_settings().retrieval_provider == "s3_vector":
        return {"dynamodb_records": None}

    if not get_settings().enable_dynamodb_fetch:
        return {"dynamodb_records": None}

    kb_results_raw = state.get("kb_results") or []

    # Collect all record_ids from KB results.
    record_ids: list[str] = []
    for result_dict in kb_results_raw:
        record_ids.extend(result_dict.get("record_ids", []))

    if not record_ids:
        return {"dynamodb_records": None}

    try:
        records = fetch_question_records_by_ids(record_ids)
        logger.debug(
            "request_id=%s  fetch_dynamodb_records  requested=%d  fetched=%d",
            state.get("request_id", ""),
            len(record_ids),
            len(records),
        )
        return {"dynamodb_records": records if records else None}

    except (DynamoDbConfigurationError, DynamoDbServiceError) as exc:
        logger.warning(
            "request_id=%s  fetch_dynamodb_records  error=%s — continuing without records",
            state.get("request_id", ""),
            type(exc).__name__,
        )
        return {"dynamodb_records": None, "service_error": True}


def build_answer_context_node(state: DoubtSolverGraphState) -> dict:
    """Assemble a bounded, safe context string from KB results and DynamoDB records.

    Calls context_builder_service which handles truncation and safety labelling.
    Sets context_used=True only when the context string is non-empty.
    """

    if get_settings().retrieval_provider == "s3_vector":
        context = state.get("answer_context") or ""
        return {
            "answer_context": context or None,
            "context_source_count": state.get("context_source_count") or 0,
            "context_used": bool(context),
        }

    classification_dict = state.get("classification") or {}
    classification = QueryClassification.model_validate(classification_dict)

    kb_results_raw = state.get("kb_results") or []
    kb_results = [KnowledgeBaseResult.model_validate(r) for r in kb_results_raw]

    dynamodb_records = state.get("dynamodb_records") or []

    bundle = build_doubt_solver_context(
        query=state.get("query", ""),
        classification=classification,
        kb_results=kb_results,
        dynamodb_records=list(dynamodb_records),
    )

    context_used = bool(bundle.context.strip())
    logger.debug(
        "request_id=%s  build_answer_context  context_len=%d  sources=%d  used=%s",
        state.get("request_id", ""),
        len(bundle.context),
        bundle.source_count,
        context_used,
    )
    return {
        "answer_context": bundle.context or None,
        "context_source_count": bundle.source_count,
        "context_used": context_used,
    }


# ---------------------------------------------------------------------------
# Nodes — answer generation and response assembly
# ---------------------------------------------------------------------------


def generate_answer_node(
    state: DoubtSolverGraphState,
) -> dict:
    """Call the answer generator service and write the result to state.

    Passes the bounded context string (if any) to the generator.
    The mock path ignores context; the LLM path includes it in the user message
    as reference material — clearly labelled, not as instructions.
    """
    classification = QueryClassification.model_validate(state.get("classification") or {})
    language = cast(
        CanonicalLanguage,
        {"en": "english", "hi": "hindi"}.get(
            str(state.get("language", "english")),
            state.get("language", "english"),
        ),
    )
    if classification.requires_recent_conversation:
        clarification = {
            "english": "Please clarify which earlier question, step, or option you mean.",
            "hinglish": (
                "Please clarify karein ki aap kis pehle question, step, ya option "
                "ki baat kar rahe hain."
            ),
            "hindi": "कृपया स्पष्ट करें कि आप पहले के किस प्रश्न, चरण या विकल्प की बात कर रहे हैं।",
        }[language]
        final_answer = build_final_answer_result(
            content=clarification,
            language=language,
            quality_status="failed_quality_gate",
        )
        return {
            "answer": clarification,
            "final_answer": final_answer.model_dump(),
            "answer_source": "fallback",
            "is_truncated": False,
        }
    # Convert empty string to None so generate_answer treats it as no context.
    retrieval_context = state.get("answer_context") or ""
    conversation_context = state.get("conversation_context") or ""
    context = (
        "\n\n".join(value for value in (conversation_context, retrieval_context) if value) or None
    )
    generation_kwargs: dict[str, object] = {
        "context": context,
        "exam_id": state.get("exam_id"),
        "exam_stage": state.get("exam_stage"),
        "exam_profile_id": state.get("exam_profile_id"),
        "language": state.get("language", "english"),
        "request_id": state.get("request_id", ""),
    }
    doubt_pattern_context = _doubt_pattern_context_from_state(state)
    if doubt_pattern_context is not None:
        generation_kwargs["doubt_pattern_context"] = doubt_pattern_context
    output = generate_answer(
        state["query"],
        classification,
        **generation_kwargs,
    )
    final_answer = build_final_answer_result(
        content=output.content,
        language=language,
        quality_status="checked",
    )
    if not final_answer.language_compliant:
        fallback = generation_failure_message(language)
        final_answer = build_final_answer_result(
            content=fallback,
            language=language,
            quality_status="failed_quality_gate",
        )
        output = output.model_copy(
            update={
                "content": fallback,
                "answer_source": "fallback",
            }
        )
    logger.debug(
        "request_id=%s  generate_answer  source=%s  answer_len=%d  truncated=%s",
        state.get("request_id", ""),
        output.answer_source,
        len(output.content),
        output.is_truncated,
    )
    return {
        "answer": output.content,
        "final_answer": final_answer.model_dump(),
        "answer_source": output.answer_source,
        "is_truncated": output.is_truncated,
    }


def build_response_node(state: DoubtSolverGraphState) -> dict:
    """Assemble DoubtSolverResponse from state and store it as a dict."""
    raw_classification = state.get("classification") or {}
    confidence: float = raw_classification.get("confidence", 1.0)
    answer_source: str = state.get("answer_source") or "mock"
    is_truncated: bool = state.get("is_truncated") or False
    service_error: bool = state.get("service_error") or False
    used_retrieval: bool = state.get("used_retrieval") or False
    context_used: bool = state.get("context_used") or False
    source_count: int = state.get("context_source_count") or 0

    # needs_review is True when:
    #   • classifier confidence is low (< 0.6)
    #   • generator used the fallback path (LLM failed)
    #   • model answer was truncated
    #   • KB or DynamoDB service error occurred during this request
    needs_review = confidence < 0.6 or answer_source == "fallback" or is_truncated or service_error

    classification = QueryClassification.model_validate(raw_classification)
    final_answer_raw = state.get("final_answer")
    authoritative_answer = (
        FinalAnswerResult.model_validate(final_answer_raw).content
        if final_answer_raw is not None
        else state.get("answer") or ""
    )
    final_answer_model = (
        FinalAnswerResult.model_validate(final_answer_raw) if final_answer_raw is not None else None
    )
    response = DoubtSolverResponse(
        success=bool(
            final_answer_model is None or final_answer_model.quality_status != "failed_quality_gate"
        ),
        request_id=state["request_id"],
        mode="doubt_solver",
        answer=authoritative_answer,
        classification=classification,
        needs_review=needs_review,
        answer_source=answer_source,  # type: ignore[arg-type]
        is_truncated=is_truncated,
        used_retrieval=used_retrieval,
        source_count=source_count,
        context_used=context_used,
    )
    logger.debug(
        "request_id=%s  build_response  needs_review=%s  answer_source=%s  "
        "used_retrieval=%s  source_count=%d  context_used=%s — completed",
        state["request_id"],
        needs_review,
        answer_source,
        used_retrieval,
        source_count,
        context_used,
    )
    return {"response": response.model_dump()}


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_doubt_solver_graph():
    """Construct and compile the Doubt Solver StateGraph.

    Returns a compiled graph ready to call with .invoke(state_dict).

    Example::

        graph = build_doubt_solver_graph()
        result = graph.invoke({
            "request_id": "abc",
            "query": "What is 20% of 500?",
            "user_id": "local-user",
            "mode": "doubt_solver",
            "language": "en",
            "classification": None,
            "answer": None,
            "answer_source": None,
            "is_truncated": False,
            "response": None,
            "should_retrieve": False,
            "kb_results": None,
            "dynamodb_records": None,
            "answer_context": None,
            "context_source_count": 0,
            "used_retrieval": False,
            "context_used": False,
            "service_error": False,
            "retrieval_context": None,
            "doubt_pattern_context": None,
        })
        print(result["response"]["answer"])
        print(result["response"]["used_retrieval"])
    """
    builder = StateGraph(DoubtSolverGraphState)

    builder.add_node("classify_query", classify_query_node)
    builder.add_node("plan_context", plan_context_node)
    builder.add_node("retrieve_kb_context", retrieve_kb_context_node)
    builder.add_node("fetch_dynamodb_records", fetch_dynamodb_records_node)
    builder.add_node("build_answer_context", build_answer_context_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("build_response", build_response_node)

    builder.add_edge(START, "classify_query")
    builder.add_edge("classify_query", "plan_context")
    builder.add_edge("plan_context", "retrieve_kb_context")
    builder.add_edge("retrieve_kb_context", "fetch_dynamodb_records")
    builder.add_edge("fetch_dynamodb_records", "build_answer_context")
    builder.add_edge("build_answer_context", "generate_answer")
    builder.add_edge("generate_answer", "build_response")
    builder.add_edge("build_response", END)

    return builder.compile()


# ===========================================================================
# Orchestrated Doubt Solver Graph — ENABLE_ORCHESTRATED_DOUBT_SOLVER path
# ===========================================================================
#
# Node layout:
#     START
#     ──► classify      (classify_query → DoubtSolverClassification)
#     ──► collect_context (KB retrieval if enabled + retrieval_required=True)
#     ──► generate      (LlmOrchestrator via AnswerGenerationAdapter)
#     ──► END
#
# State: OrchestratedDoubtSolverState (request_id, query, classification,
#        retrieval_context, context_text, answer)
#
# Guard: this code path is only active when
#        ENABLE_ORCHESTRATED_DOUBT_SOLVER=true.  Default is false.
#
# Graph nodes must NOT contain:
#   - model_id, deployment, provider, provider_profile
#   - API keys, env var reads for credentials
#   - direct provider SDK calls
# ===========================================================================


class OrchestratedDoubtSolverState(TypedDict):
    """Lean orchestrated graph state — only what nodes need to produce an answer.

    Fields outside this TypedDict (user_id, language, mode, answer_source,
    is_truncated, used_retrieval, etc.) belong in the legacy graph state
    or in the API response layer, not here.
    """

    request_id: str
    query: str
    original_query: str
    actor_id: str
    conversation_id: str
    turn_id: str
    language: str
    exam_id: str | None
    exam_stage: str | None
    exam_profile_id: str | None
    classification: dict | None  # serialised DoubtSolverClassification
    retrieval_context: dict  # structured internal student retrieval context
    context_text: str  # compact context string (may be "")
    answer: str | None
    final_answer: dict | None
    response_type: str | None
    practice_test_id: str | None
    conversation_context: str
    conversation_relation: dict | None
    conversation_preparation: dict | None
    query_classification: dict | None
    source_modality: str
    fresh_evidence: dict | None


# ---------------------------------------------------------------------------
# Subject/intent normalisation helpers
# ---------------------------------------------------------------------------


def _run_graph_classifier(
    query: str,
    request_id: str | None = None,
    *,
    on_before_strong_classifier: Callable[[], None] | None = None,
    conversation_candidates: str | None = None,
    candidate_cards: tuple[ConversationCandidateCard, ...] = (),
    candidate_turn_ids: tuple[str, ...] = (),
    context_gate: str = "CONTEXT_NOT_NEEDED",
) -> QueryClassification:
    """Keep existing graph monkeypatch and dependency-injection boundaries intact."""
    return classify_query(
        query,
        request_id,
        on_before_strong_classifier=on_before_strong_classifier,
        conversation_candidates=conversation_candidates,
        candidate_cards=candidate_cards,
        candidate_turn_ids=candidate_turn_ids,
        context_gate=context_gate,
    )


_classification_coordinator = ClassificationCoordinator(classifier=_run_graph_classifier)

# Safe fallback classification used when the classifier fails.
_ORCHESTRATED_FALLBACK_CLASSIFICATION: dict = {
    "subject": "general",
    "intent": "explain",
    "difficulty": "default",
    "retrieval_required": False,
    "requires_recent_conversation": False,
}


def _map_to_orchestrated_classification(
    raw: QueryClassification,
    query: str = "",
    request_id: str = "",
) -> dict:
    """Map existing QueryClassification output to orchestrated graph classification dict.

    The orchestrated state classification contains generator routing fields plus
    optional retrieval hints nested in the same classification dict (graph state
    remains 5 top-level fields).
    """
    return map_academic_classification(
        raw,
        query=query,
        request_id=request_id,
    )


def map_to_orchestrated_classification(
    raw: QueryClassification,
    *,
    query: str,
    request_id: str,
) -> dict:
    """Public boundary for mapping either text or image classification downstream."""
    return _map_to_orchestrated_classification(raw, query=query, request_id=request_id)


# ---------------------------------------------------------------------------
# Node 1: classify
# ---------------------------------------------------------------------------


def orchestrated_classify_query(
    query: str,
    request_id: str = "",
    *,
    on_before_strong_classifier: Callable[[], None] | None = None,
) -> dict:
    """Classify and map to orchestrated graph classification dict.

    Shared by the orchestrated classify graph node and streaming service.
    """
    classification, _confidence, _fallback = orchestrated_classify_query_with_delivery_signals(
        query,
        request_id=request_id,
        on_before_strong_classifier=on_before_strong_classifier,
    )
    return classification


def orchestrated_classify_query_with_delivery_signals(
    query: str,
    request_id: str = "",
    *,
    on_before_strong_classifier: Callable[[], None] | None = None,
    conversation: ConversationPreparation | None = None,
) -> tuple[dict, float | None, bool]:
    """Return graph-safe classification plus streaming-only delivery signals."""
    result = orchestrated_classify_query_stage(
        query=query,
        request_id=request_id,
        on_before_strong_classifier=on_before_strong_classifier,
        conversation=conversation,
    )
    return (
        result.classification,
        result.classifier_confidence,
        result.classifier_fallback,
    )


def orchestrated_classify_query_stage(
    *,
    query: str,
    request_id: str,
    on_before_strong_classifier: Callable[[], None] | None = None,
    conversation: ConversationPreparation | None = None,
) -> ClassificationStageResult:
    """Shared typed classification result for graph and streaming paths."""
    return _classification_coordinator.classify_text(
        query=query,
        request_id=request_id,
        on_before_strong_classifier=on_before_strong_classifier,
        conversation=conversation,
    )


# ---------------------------------------------------------------------------
# Node 1: classify
# ---------------------------------------------------------------------------


def _orchestrated_classify_node(state: OrchestratedDoubtSolverState) -> dict:
    """Classify the query and write a lean DoubtSolverClassification to state.

    On any classification error: uses safe fallback
    (subject=general, intent=explain, difficulty=default,
     retrieval_required=False).

    Does NOT:
    - Write answer.
    - Call any provider.
    - Use model_id / provider / deployment.
    """
    classification_dict = state.get("classification")
    raw_classification = state.get("query_classification")
    if classification_dict is None:
        preparation_payload = state.get("conversation_preparation")
        preparation = (
            ConversationPreparation.model_validate(preparation_payload)
            if preparation_payload
            else None
        )
        stage_result = orchestrated_classify_query_stage(
            query=state["query"],
            request_id=state.get("request_id", ""),
            conversation=preparation,
        )
        classification_dict = stage_result.classification
        raw_classification = stage_result.raw.model_dump()

    logger.debug(
        "request_id=%s  orchestrated_classify  subject=%s  intent=%s  difficulty=%s  "
        "retrieval_required=%s",
        state.get("request_id", ""),
        classification_dict.get("subject"),
        classification_dict.get("intent"),
        classification_dict.get("difficulty"),
        classification_dict.get("retrieval_required"),
    )
    log_event(
        "classification_completed",
        component="doubt_solver.classifier",
        stage="classify",
        status="completed",
        details={
            "subject": classification_dict.get("subject"),
            "intent": classification_dict.get("intent"),
            "difficulty": classification_dict.get("difficulty"),
            "relation": (
                raw_classification.relation
                if isinstance(raw_classification, QueryClassification)
                else (
                    raw_classification.get("relation")
                    if isinstance(raw_classification, dict)
                    else "NEW_QUESTION"
                )
            ),
            "action": (
                raw_classification.requested_action
                if isinstance(raw_classification, QueryClassification)
                else (
                    raw_classification.get("requested_action")
                    if isinstance(raw_classification, dict)
                    else "ANSWER_CURRENT"
                )
            ),
            "need_web_search": bool(classification_dict.get("need_web_search")),
            "web_search_reason": classification_dict.get("web_search_reason"),
            "search_term_present": bool(classification_dict.get("web_search_query")),
        },
    )
    update = {"classification": classification_dict}
    if raw_classification is not None:
        update["query_classification"] = raw_classification
    return update


# ---------------------------------------------------------------------------
# Node 2: collect_context
# ---------------------------------------------------------------------------


def _trace_field(result: object, name: str) -> object | None:
    """Read one safe retrieval-trace field.

    The trace hangs off ``result.retrieval_context.retrieval_trace``; every
    lookup is attribute-safe so test doubles without a trace stay valid.
    """
    context = getattr(result, "retrieval_context", None)
    return getattr(getattr(context, "retrieval_trace", None), name, None)


def _orchestrated_collect_context_node(
    state: OrchestratedDoubtSolverState,
    *,
    on_before_web_search: Callable[[], None] | None = None,
    on_web_search_retry: Callable[[], None] | None = None,
    on_web_search_weak_context: Callable[[], None] | None = None,
) -> dict:
    """Retrieve compact context via ContextRetrievalService.

    Delegates all KB decision, retrieval, reranking, and formatting to the
    context retrieval service.  Graph state receives only context_text.

    On failure: returns context_text="" and continues.
    """
    query: str = state.get("query", "")
    if not query:
        from retrieval.models import StudentRetrievalContext  # noqa: PLC0415

        return {
            "context_text": "",
            "retrieval_context": StudentRetrievalContext.fresh_solve("empty_query").model_dump(
                by_alias=True
            ),
        }

    classification_dict = state.get("classification") or {}

    started_at = time.monotonic()
    try:
        from services.context_retrieval.context_retrieval_service import (  # noqa: PLC0415
            ContextRequestBuilder,
            get_context_retrieval_service,
        )

        freshness_requirement = resolve_practice_freshness_requirement(
            query,
            classification_dict,
        )
        try:
            validate_practice_requested_count(state.get("original_query") or query)
        except PracticeRequestCountError:
            return {"context_text": "", "retrieval_context": {}}
        retrieval_classification = dict(classification_dict)
        if freshness_requirement.requires_fresh_evidence:
            retrieval_classification.update(
                {
                    "need_web_search": True,
                    "web_search_reason": freshness_requirement.freshness_reason,
                }
            )
        request = ContextRequestBuilder.from_query_and_classification(
            request_id=state.get("request_id", ""),
            query=query,
            classification=retrieval_classification,
            requires_fresh_evidence=freshness_requirement.requires_fresh_evidence,
            required_evidence_count=(
                required_fresh_evidence_count(state.get("original_query") or query)
                if freshness_requirement.requires_fresh_evidence
                else 0
            ),
        )
        result = get_context_retrieval_service().retrieve_context(
            request,
            on_before_web_search=on_before_web_search,
            on_web_search_retry=on_web_search_retry,
            on_web_search_weak_context=on_web_search_weak_context,
        )
        context_text = result.context_text or ""

        logger.debug(
            "request_id=%s  orchestrated_collect_context  items=%d  context_chars=%d  reason=%s",
            state.get("request_id", ""),
            result.item_count,
            len(context_text),
            result.reason,
        )
        retrieval_payload = result.retrieval_context.model_dump(by_alias=True)
        if result.doubt_pattern_context is not None:
            retrieval_payload["doubtPatternContext"] = result.doubt_pattern_context.model_dump(
                by_alias=True
            )
        retrieval_mode = str(retrieval_payload.get("mode") or "none")
        retrieval_source = (
            result.web_search_provider or "web"
            if result.reason == "web_context_selected"
            else retrieval_mode
        )
        duration_ms = int((time.monotonic() - started_at) * 1000)
        update_request_summary(
            retrieval_status="completed",
            retrieval_source=retrieval_source,
        )
        log_event(
            "retrieval_completed",
            component="doubt_solver.retrieval",
            stage="retrieve",
            status="completed",
            duration_ms=duration_ms,
            details={
                "source": retrieval_source,
                "item_count": result.item_count,
                # Already computed by the retrieval service; previously dropped
                # here, which made a slow zero-item retrieval undiagnosable.
                "fallback_reason": _trace_field(result, "fallback_reason"),
                "embedding_ms": _trace_field(result, "embedding_ms"),
                "runtime_query_ms": _trace_field(result, "runtime_query_ms"),
                "pattern_query_ms": _trace_field(result, "pattern_query_ms"),
                "rerank_ms": _trace_field(result, "rerank_ms"),
            },
        )
        fresh_evidence = result.fresh_evidence
        if freshness_requirement.requires_fresh_evidence:
            log_event(
                "practice_freshness_evidence",
                component="practice.freshness",
                stage="retrieve",
                status="attached" if fresh_evidence is not None else "unavailable",
                details={
                    "required": True,
                    "reason": freshness_requirement.freshness_reason,
                    "evidenceItemCount": (
                        len(fresh_evidence.items) if fresh_evidence is not None else 0
                    ),
                    "windowStart": (
                        fresh_evidence.requested_window.start_date
                        if fresh_evidence is not None
                        else ""
                    ),
                    "windowEnd": (
                        fresh_evidence.requested_window.end_date
                        if fresh_evidence is not None
                        else ""
                    ),
                    "temporalMode": (
                        fresh_evidence.requested_window.temporal_mode
                        if fresh_evidence is not None
                        else ""
                    ),
                },
            )
        update = {
            "context_text": context_text,
            "retrieval_context": retrieval_payload,
        }
        if fresh_evidence is not None:
            update["fresh_evidence"] = fresh_evidence.model_dump(mode="json")
        return update

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "request_id=%s orchestrated_collect_context retrieval_error=true "
            "error_type=%s phase=context_retrieve",
            state.get("request_id", ""),
            type(exc).__name__,
        )
        from retrieval.models import StudentRetrievalContext  # noqa: PLC0415

        duration_ms = int((time.monotonic() - started_at) * 1000)
        update_request_summary(
            retrieval_status="failed",
            retrieval_source="fresh_solve",
        )
        log_event(
            "retrieval_failed",
            component="doubt_solver.retrieval",
            stage="retrieve",
            status="failed",
            duration_ms=duration_ms,
            error_code="RETRIEVAL_ERROR",
            details={"error_type": type(exc).__name__},
            level=logging.WARNING,
        )
        log_event(
            "retrieval_fallback_used",
            component="doubt_solver.retrieval",
            stage="retrieve",
            status="fallback",
            duration_ms=duration_ms,
            details={"source": "fresh_solve", "reason": "retrieval_error"},
            level=logging.WARNING,
        )
        return {
            "context_text": "",
            "retrieval_context": StudentRetrievalContext.fresh_solve("retrieval_error").model_dump(
                by_alias=True
            ),
        }


# ---------------------------------------------------------------------------
# Graph factory — Orchestrated Doubt Solver
# ---------------------------------------------------------------------------


def build_orchestrated_doubt_solver_graph(
    adapter,
    *,
    conversation_persistence=None,
    follow_up_resolver=None,
    conversation_understanding=None,
    practice_launcher: Callable[[PracticeGenerationRequest], PracticeLaunchResult] | None = None,
    practice_request_interpreter: PracticeRequestInterpreter | None = None,
):
    """Construct and compile the lean Orchestrated Doubt Solver StateGraph.

    Args:
        adapter: AnswerGenerationAdapter.  Must be fully constructed before
                 calling this function — the generate node captures it as a
                 closure.  Tests inject a mock-safe adapter here.

    Returns:
        A compiled LangGraph CompiledGraph.

    Node flow:
        START → understand_conversation → classify → prepare_follow_up
        → collect_context → generate → END

    State:
        OrchestratedDoubtSolverState — retrieval_context remains internal to the graph.
        No plan, no response, no sources, no route_decision.

    Guard:
        Only call when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true.
        Default path is build_doubt_solver_graph().
    """

    def _understand_conversation_node(
        state: OrchestratedDoubtSolverState,
    ) -> dict:
        if (
            state.get("source_modality") == "image"
            or state.get("conversation_preparation")
            or conversation_understanding is None
        ):
            return {}
        preparation = conversation_understanding.understand(
            actor_id=state["actor_id"],
            conversation_id=state["conversation_id"],
            query=state["query"],
            request_id=state["request_id"],
            exam_id=state.get("exam_id"),
            language=cast(CanonicalLanguage, state.get("language", "english")),
        )
        return {"conversation_preparation": preparation.model_dump()}

    def _prepare_follow_up_node(state: OrchestratedDoubtSolverState) -> dict:
        classification = state.get("classification") or {}
        preparation_payload = state.get("conversation_preparation")
        raw_payload = state.get("query_classification")
        if preparation_payload and raw_payload:
            preparation = ConversationPreparation.model_validate(preparation_payload)
            raw = QueryClassification.model_validate(raw_payload)
            selected = build_selected_generation_context(
                current_query=state["original_query"],
                classification=raw,
                preparation=preparation,
            )
            relation = {
                "relation": selected.relation,
                "requested_action": selected.requested_action,
                "selected_turn_id": selected.selected_turn_id,
                "requires_recent_conversation": selected.context_policy != "current_only",
                "clarification_needed": selected.clarification_required,
                "confidence": raw.confidence,
                "decision_source": raw.classification_source,
            }
            if selected.clarification_required:
                language = cast(CanonicalLanguage, state.get("language", "english"))
                clarification = build_context_aware_clarification(
                    preparation,
                    language=language,
                    current_query=str(state.get("original_query") or state["query"]),
                )
                final = build_final_answer_result(
                    content=clarification,
                    language=language,
                    quality_status="failed_quality_gate",
                )
                return {
                    "answer": clarification,
                    "final_answer": final.model_dump(),
                    "conversation_relation": relation,
                    "conversation_context": "",
                }
            log_event(
                "selected_generation_context_built",
                component="conversation.selected_context",
                stage="prepare_follow_up",
                status="completed",
                details={
                    "relation": selected.relation,
                    "action": selected.requested_action,
                    "selected_turn_id": selected.selected_turn_id,
                    "context_policy": selected.context_policy,
                    "context_characters": selected.context_characters,
                },
            )
            return {
                "query": selected.resolved_query,
                "conversation_context": selected.conversation_context,
                "conversation_relation": relation,
            }
        if not classification.get("requires_recent_conversation", False):
            logger.debug("standalone_request_count count=1")
            return {"conversation_context": ""}
        logger.debug("follow_up_request_count count=1")
        log_event(
            "follow_up_detected",
            component="conversation.follow_up",
            stage="detect_follow_up",
            status="completed",
            details={"detection": "classifier_or_reference_signal"},
        )
        language = cast(CanonicalLanguage, state.get("language", "english"))
        clarification = {
            "english": "Please clarify which earlier question, step, or option you mean.",
            "hinglish": (
                "Please clarify karein ki aap kis pehle question, step, ya option "
                "ki baat kar rahe hain."
            ),
            "hindi": "कृपया स्पष्ट करें कि आप पहले के किस प्रश्न, चरण या विकल्प की बात कर रहे हैं।",
        }[language]
        final = build_final_answer_result(
            content=clarification,
            language=language,
            quality_status="failed_quality_gate",
        )
        return {"answer": clarification, "final_answer": final.model_dump()}

    def _generate_node(
        state: OrchestratedDoubtSolverState,
    ) -> dict:
        """Call AnswerGenerationAdapter and write answer string to state.

        Handles controlled provider failures by returning a safe user-facing
        message instead of propagating the error.  Unexpected programming
        errors (not LlmOrchestrationError subclasses) still propagate loudly.

        Does NOT:
        - Store OrchestrationResult in state.
        - Store prompt/messages in state.
        - Store raw provider response in state.
        - Use model_id / provider / deployment directly.
        """
        from services.llm.orchestration.errors import ProviderExecutionError  # noqa: PLC0415

        classification_dict = (
            state.get("classification") or _ORCHESTRATED_FALLBACK_CLASSIFICATION.copy()
        )
        subject: str = classification_dict.get("subject", "general")
        intent: str = classification_dict.get("intent", "explain")
        difficulty: str = classification_dict.get("difficulty", "default")
        context_text: str = state.get("context_text") or ""
        retrieval_payload = state.get("retrieval_context") or {}
        doubt_pattern_context = _doubt_pattern_context_from_payload(
            retrieval_payload.get("doubtPatternContext"),
            request_id=state.get("request_id", ""),
        )

        language = cast(CanonicalLanguage, state.get("language", "english"))
        from services.context_retrieval.web_grounding import (  # noqa: PLC0415
            log_grounding_status,
            required_web_answer_verified,
            required_web_context_verified,
            sanitize_required_web_answer,
            verification_limited_response,
        )

        web_verified = required_web_context_verified(classification_dict, state)
        if not web_verified:
            answer = verification_limited_response(language)
            log_grounding_status(classification_dict, state, verified=False)
            final_answer = build_final_answer_result(
                content=answer,
                language=language,
                quality_status="failed_quality_gate",
            )
            return {"answer": answer, "final_answer": final_answer.model_dump()}
        try:
            generation_kwargs = {
                "request_id": state["request_id"],
                "query": state["query"],
                "subject": subject,
                "intent": intent,
                "difficulty": difficulty,
                "context": context_text,
                "web_search_reason": (
                    str(classification_dict.get("web_search_reason"))
                    if classification_dict.get("web_search_reason")
                    else None
                ),
                "exam_id": state.get("exam_id"),
                "exam_stage": state.get("exam_stage"),
                "language": language,
                "conversation_context": state.get("conversation_context") or None,
            }
            if state.get("exam_profile_id"):
                generation_kwargs["exam_profile_id"] = state["exam_profile_id"]
            if doubt_pattern_context is not None:
                generation_kwargs["doubt_pattern_context"] = doubt_pattern_context
            generate_final = getattr(adapter, "generate_final", None)
            if generate_final is not None:
                final_answer = generate_final(**generation_kwargs)
                answer = final_answer.content
                relation = state.get("conversation_relation") or {}
                verification_required = requires_independent_correctness_verification(
                    subject=subject,
                    difficulty=difficulty,
                    intent=intent,
                    requested_action=str(relation.get("requested_action") or "ANSWER_CURRENT"),
                )
                correctness_verifier = getattr(adapter, "correctness_verifier", None)
                if (
                    verification_required
                    and correctness_verifier is not None
                    and final_answer.quality_status != "failed_quality_gate"
                ):
                    correctness = correctness_verifier.verify(
                        request_id=state["request_id"],
                        query=state["query"],
                        candidate_answer=answer,
                        subject=subject,
                        difficulty=difficulty,
                        language=language,
                    )
                    if not correctness.approved:
                        answer = generation_failure_message(language)
                        final_answer = build_final_answer_result(
                            content=answer,
                            language=language,
                            quality_status="failed_quality_gate",
                        )
            else:
                answer = adapter.generate(**generation_kwargs)
                final_answer = build_final_answer_result(
                    content=answer,
                    language=language,
                    quality_status="checked",
                )
        except ProviderExecutionError as exc:
            # Controlled provider failure — all fallbacks exhausted.
            # Log safely (no query/context/provider details in the message).
            logger.warning(
                "request_id=%s  orchestrated_generate  provider_failure  "
                "error_type=%s — returning safe fallback answer",
                state.get("request_id", ""),
                type(exc).__name__,
            )
            log_event(
                "generation_failed",
                component="doubt_solver.generator",
                stage="generate",
                status="failed",
                error_code="PROVIDER_EXECUTION_FAILED",
                details={"error_type": type(exc).__name__},
                level=logging.ERROR,
            )
            answer = generation_failure_message(language)
            final_answer = build_final_answer_result(
                content=answer,
                language=language,
                quality_status="failed_quality_gate",
            )

        grounded_answer = sanitize_required_web_answer(
            classification_dict,
            state,
            answer,
        )
        if grounded_answer is not None:
            answer = grounded_answer
            final_answer = build_final_answer_result(
                content=answer,
                language=language,
                quality_status=final_answer.quality_status,
            )
        answer_grounded = grounded_answer is not None and required_web_answer_verified(
            classification_dict,
            state,
            answer,
        )
        if not answer_grounded:
            answer = verification_limited_response(language)
            final_answer = build_final_answer_result(
                content=answer,
                language=language,
                quality_status="failed_quality_gate",
            )
        log_grounding_status(
            classification_dict,
            state,
            verified=answer_grounded,
            answer=answer,
        )
        logger.debug(
            "request_id=%s  orchestrated_generate  subject=%s  intent=%s  "
            "difficulty=%s  answer_len=%d",
            state.get("request_id", ""),
            subject,
            intent,
            difficulty,
            len(answer),
        )
        return {"answer": answer, "final_answer": final_answer.model_dump()}

    def _practice_launch_node(state: OrchestratedDoubtSolverState) -> dict:
        if practice_launcher is not None:
            classification = state.get("classification") or {}
            try:
                freshness_requirement = resolve_practice_freshness_requirement(
                    state.get("original_query") or state["query"],
                    classification,
                )
                fresh_evidence = (
                    FreshEvidenceBundle.model_validate(state["fresh_evidence"])
                    if freshness_requirement.requires_fresh_evidence
                    and state.get("fresh_evidence")
                    else None
                )
                if freshness_requirement.requires_fresh_evidence and fresh_evidence is None:
                    raise PracticeLaunchError("PRACTICE_FRESH_EVIDENCE_UNAVAILABLE")
                request = resolve_practice_request(
                    request_id=state["request_id"],
                    user_id=state["actor_id"],
                    conversation_id=state["conversation_id"],
                    turn_id=state["turn_id"],
                    query=state.get("original_query") or state["query"],
                    subject=canonical_practice_subject(
                        classification.get("subject"),
                        classification.get("pattern_family_candidate"),
                    ),
                    topic=(str(classification["topic"]) if classification.get("topic") else None),
                    difficulty=str(classification.get("difficulty") or "default"),
                    language=state.get("language", "english"),
                    exam_id=state.get("exam_id"),
                    exam_stage=state.get("exam_stage"),
                    exam_profile_id=state.get("exam_profile_id"),
                    source_question_reference=str(
                        (state.get("query_classification") or {}).get("selected_turn_id") or ""
                    )
                    or None,
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
                        "language": request.language,
                        "source": request.language_source,
                    },
                )
                launch = practice_launcher(request)
            except (PracticeLaunchError, PracticeRequestCountError):
                message = "Practice generation could not be started. Please try again."
                final = build_final_answer_result(
                    content=message,
                    language=cast(
                        CanonicalLanguage,
                        state.get("language", "english"),
                    ),
                    quality_status="failed_quality_gate",
                )
                return {"answer": message, "final_answer": final.model_dump()}
            final = build_final_answer_result(
                content=launch.message,
                language=cast(CanonicalLanguage, state.get("language", "english")),
                quality_status="checked",
            )
            return {
                "answer": launch.message,
                "final_answer": final.model_dump(),
                "response_type": "practice_generation",
                "practice_test_id": launch.test_id,
            }
        message = practice_async_unavailable_message(state.get("language", "english"))
        final = build_final_answer_result(
            content=message,
            language=cast(CanonicalLanguage, state.get("language", "english")),
            quality_status="failed_quality_gate",
        )
        log_event(
            "PRACTICE_LAUNCH_DECISION",
            component="practice.routing",
            stage="route",
            status="disabled",
            error_code=PRACTICE_ASYNC_NOT_CONFIGURED,
        )
        return {
            "answer": message,
            "final_answer": final.model_dump(),
        }

    def _route_after_follow_up(state: OrchestratedDoubtSolverState) -> str:
        if state.get("final_answer"):
            return "complete"
        classification = state.get("classification") or {}
        decision = decide_practice_launch(state["query"], classification)
        if practice_route_enabled() and classification.get("intent") == "practice":
            log_event(
                "PRACTICE_LAUNCH_DECISION",
                component="practice.routing",
                stage="route",
                status="eligible" if decision.eligible else "ineligible",
                details={
                    "eligible": decision.eligible,
                    "reasonCode": decision.reason_code,
                    "requestedArtifact": decision.requested_artifact,
                },
            )
        if practice_route_enabled() and decision.eligible:
            requirement = resolve_practice_freshness_requirement(
                state.get("original_query") or state["query"],
                classification,
            )
            return "practice_fresh_context" if requirement.requires_fresh_evidence else "practice"
        return "retrieve"

    def _route_after_context(state: OrchestratedDoubtSolverState) -> str:
        classification = state.get("classification") or {}
        decision = decide_practice_launch(state["query"], classification)
        if practice_route_enabled() and decision.eligible:
            return "practice"
        return "generate"

    builder: StateGraph = StateGraph(OrchestratedDoubtSolverState)
    builder.add_node("understand_conversation", _understand_conversation_node)
    builder.add_node("classify", _orchestrated_classify_node)
    builder.add_node("prepare_follow_up", _prepare_follow_up_node)
    builder.add_node("collect_context", _orchestrated_collect_context_node)
    builder.add_node("generate", _generate_node)
    if practice_route_enabled():
        builder.add_node("practice_launch", _practice_launch_node)

    builder.add_edge(START, "understand_conversation")
    builder.add_edge("understand_conversation", "classify")
    builder.add_edge("classify", "prepare_follow_up")
    builder.add_conditional_edges(
        "prepare_follow_up",
        _route_after_follow_up,
        {
            "complete": END,
            "practice": "practice_launch" if practice_route_enabled() else "collect_context",
            "practice_fresh_context": "collect_context",
            "retrieve": "collect_context",
        },
    )
    if practice_route_enabled():
        builder.add_edge("practice_launch", END)
    builder.add_conditional_edges(
        "collect_context",
        _route_after_context,
        {
            "practice": "practice_launch" if practice_route_enabled() else "generate",
            "generate": "generate",
        },
    )
    builder.add_edge("generate", END)

    return builder.compile()
