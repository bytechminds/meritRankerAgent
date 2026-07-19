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
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from schemas.doubt_solver import (
    DoubtSolverResponse,
    QueryClassification,
)
from schemas.retrieval import KnowledgeBaseResult
from services.answer_generator_service import generate_answer
from services.bedrock_kb_service import (
    KnowledgeBaseConfigurationError,
    KnowledgeBaseServiceError,
    retrieve_similar_context,
)
from services.context_builder_service import build_doubt_solver_context
from services.dynamodb_service import DynamoDbConfigurationError, DynamoDbServiceError
from services.query_classifier_service import (
    apply_classification_policy,
    apply_classification_sanity,
    classify_query,
)
from services.question_record_service import fetch_question_records_by_ids

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph internal state — plain TypedDict; Pydantic only at boundaries
# ---------------------------------------------------------------------------


class DoubtSolverGraphState(TypedDict):
    """Data carried between nodes inside the LangGraph workflow."""

    request_id: str
    query: str
    user_id: str
    mode: str
    language: str
    classification: dict | None       # serialised QueryClassification
    answer: str | None
    answer_source: str | None         # "mock" | "llm" | "fallback"
    is_truncated: bool
    response: dict | None             # serialised DoubtSolverResponse
    # Part 9 context-pipeline fields
    should_retrieve: bool             # set by plan_context_node
    kb_results: list | None           # list of serialised KnowledgeBaseResult dicts
    dynamodb_records: list | None     # list of raw DynamoDB record dicts
    answer_context: str | None        # bounded context string for answer generator
    context_source_count: int         # number of sources included in context
    used_retrieval: bool              # True if KB returned ≥1 result
    context_used: bool                # True if context was passed to answer generator
    service_error: bool               # True if KB or DynamoDB service error occurred
    retrieval_context: dict | None    # internal S3 Vector retrieval contract


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
    from config import get_settings  # noqa: PLC0415

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
    """Run the approved S3 Vector student retrieval path for the legacy graph."""
    from retrieval.context.data_context_builder import render_retrieval_context  # noqa: PLC0415
    from retrieval.retrieval_service import StudentRetrievalService  # noqa: PLC0415
    from services.context_retrieval.context_models import ContextRetrievalRequest  # noqa: PLC0415

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
    retrieval_context = StudentRetrievalService().retrieve(request)
    context_text = render_retrieval_context(retrieval_context)
    retrieval_used = retrieval_context.mode != "fresh_solve"
    return {
        "kb_results": None,
        "dynamodb_records": None,
        "answer_context": context_text or None,
        "context_source_count": 1 if retrieval_used else 0,
        "used_retrieval": retrieval_used,
        "context_used": bool(context_text),
        "retrieval_context": retrieval_context.model_dump(by_alias=True),
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
    from config import get_settings  # noqa: PLC0415

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
    from config import get_settings  # noqa: PLC0415

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


def generate_answer_node(state: DoubtSolverGraphState) -> dict:
    """Call the answer generator service and write the result to state.

    Passes the bounded context string (if any) to the generator.
    The mock path ignores context; the LLM path includes it in the user message
    as reference material — clearly labelled, not as instructions.
    """
    classification = QueryClassification.model_validate(state.get("classification") or {})
    # Convert empty string to None so generate_answer treats it as no context.
    context = state.get("answer_context") or None
    output = generate_answer(state["query"], classification, context=context)
    logger.debug(
        "request_id=%s  generate_answer  source=%s  answer_len=%d  truncated=%s",
        state.get("request_id", ""),
        output.answer_source,
        len(output.content),
        output.is_truncated,
    )
    return {
        "answer": output.content,
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
    needs_review = (
        confidence < 0.6
        or answer_source == "fallback"
        or is_truncated
        or service_error
    )

    classification = QueryClassification.model_validate(raw_classification)
    response = DoubtSolverResponse(
        success=True,
        request_id=state["request_id"],
        mode="doubt_solver",
        answer=state.get("answer") or "",
        classification=classification,
        needs_review=needs_review,
        answer_source=answer_source,  # type: ignore[arg-type]
        is_truncated=is_truncated,
        used_retrieval=used_retrieval,
        source_count=source_count,
        context_used=context_used,
    )
    logger.info(
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
    classification: dict | None   # serialised DoubtSolverClassification
    retrieval_context: dict       # structured internal student retrieval context
    context_text: str             # compact context string (may be "")
    answer: str | None


# ---------------------------------------------------------------------------
# Subject/intent normalisation helpers
# ---------------------------------------------------------------------------

_ORCHESTRATED_SUBJECT_MAP: dict[str, str] = {
    "math": "math",
    "reasoning": "reasoning",
    "english": "english",
    "general": "general",
    "science": "general",   # unmapped subjects → general
    "unknown": "general",
}

_ORCHESTRATED_INTENT_MAP: dict[str, str] = {
    "solve_question": "solve",
    "explain_concept": "explain",
    "explain_option": "explain",
    "general_doubt": "explain",
    "practice_question": "practice",
    "visualize_question": "visualize",
    "unknown": "explain",
}

# Safe fallback classification used when the classifier fails.
_ORCHESTRATED_FALLBACK_CLASSIFICATION: dict = {
    "subject": "general",
    "intent": "explain",
    "difficulty": "default",
    "retrieval_required": False,
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
    from schemas.doubt_solver import DoubtSolverClassification  # noqa: PLC0415

    subject = _ORCHESTRATED_SUBJECT_MAP.get(raw.subject, "general")
    intent = _ORCHESTRATED_INTENT_MAP.get(raw.intent, "explain")
    # Pass classifier difficulty through directly — no longer hardcoded "default".
    difficulty = raw.difficulty
    retrieval_required = raw.retrieval_need != "none"

    classification = DoubtSolverClassification(
        subject=subject,
        intent=intent,
        difficulty=difficulty,
        retrieval_required=retrieval_required,
        topic=raw.topic,
        topic_confidence=raw.topic_confidence,
        pattern_topic_candidate=raw.pattern_topic_candidate,
        pattern_family_candidate=raw.pattern_family_candidate,
        retrieval_tags=raw.retrieval_tags,
        need_web_search=raw.need_web_search,
        web_search_reason=raw.web_search_reason,
        web_search_query=raw.web_search_query,
    )
    classification_dict = classification.model_dump()
    classification_dict = apply_classification_sanity(
        query,
        classification_dict,
        request_id=request_id,
        classifier_confidence=raw.confidence,
    )
    return apply_classification_policy(
        query,
        classification_dict,
        request_id=request_id,
        classifier_confidence=raw.confidence,
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
    classification, _confidence, _fallback = (
        orchestrated_classify_query_with_delivery_signals(
            query,
            request_id=request_id,
            on_before_strong_classifier=on_before_strong_classifier,
        )
    )
    return classification


def orchestrated_classify_query_with_delivery_signals(
    query: str,
    request_id: str = "",
    *,
    on_before_strong_classifier: Callable[[], None] | None = None,
) -> tuple[dict, float | None, bool]:
    """Return graph-safe classification plus streaming-only delivery signals."""
    try:
        raw: QueryClassification = classify_query(
            query,
            request_id=request_id or None,
            on_before_strong_classifier=on_before_strong_classifier,
        )
        return (
            _map_to_orchestrated_classification(
                raw,
                query=query,
                request_id=request_id,
            ),
            raw.confidence,
            raw.classification_source == "fallback",
        )
    except Exception:  # noqa: BLE001
        logger.warning(
            "request_id=%s  orchestrated_classify  classifier raised, using fallback",
            request_id,
        )
        classification_dict = _ORCHESTRATED_FALLBACK_CLASSIFICATION.copy()
        return (
            apply_classification_policy(
                query,
                classification_dict,
                request_id=request_id,
            ),
            None,
            True,
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
    if classification_dict is None:
        classification_dict = orchestrated_classify_query(
            state["query"],
            request_id=state.get("request_id", ""),
        )

    logger.debug(
        "request_id=%s  orchestrated_classify  subject=%s  intent=%s  difficulty=%s  "
        "retrieval_required=%s",
        state.get("request_id", ""),
        classification_dict.get("subject"),
        classification_dict.get("intent"),
        classification_dict.get("difficulty"),
        classification_dict.get("retrieval_required"),
    )
    return {"classification": classification_dict}
# ---------------------------------------------------------------------------
# Node 2: collect_context
# ---------------------------------------------------------------------------


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
            "retrieval_context": StudentRetrievalContext.fresh_solve(
                "empty_query"
            ).model_dump(by_alias=True),
        }

    classification_dict = state.get("classification") or {}

    try:
        from services.context_retrieval.context_retrieval_service import (  # noqa: PLC0415
            ContextRequestBuilder,
            get_context_retrieval_service,
        )

        request = ContextRequestBuilder.from_query_and_classification(
            request_id=state.get("request_id", ""),
            query=query,
            classification=classification_dict,
        )
        result = get_context_retrieval_service().retrieve_context(
            request,
            on_before_web_search=on_before_web_search,
            on_web_search_retry=on_web_search_retry,
            on_web_search_weak_context=on_web_search_weak_context,
        )
        context_text = result.context_text or ""

        logger.debug(
            "request_id=%s  orchestrated_collect_context  items=%d  "
            "context_chars=%d  reason=%s",
            state.get("request_id", ""),
            result.item_count,
            len(context_text),
            result.reason,
        )
        return {
            "context_text": context_text,
            "retrieval_context": result.retrieval_context.model_dump(by_alias=True),
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "request_id=%s  orchestrated_collect_context  retrieval error  "
            "error_type=%s  error_message_short=%s  phase=context_retrieve",
            state.get("request_id", ""),
            type(exc).__name__,
            str(exc)[:120],
        )
        logger.debug(
            "request_id=%s  orchestrated_collect_context traceback",
            state.get("request_id", ""),
            exc_info=True,
        )
        from retrieval.models import StudentRetrievalContext  # noqa: PLC0415

        return {
            "context_text": "",
            "retrieval_context": StudentRetrievalContext.fresh_solve(
                "retrieval_error"
            ).model_dump(by_alias=True),
        }


# ---------------------------------------------------------------------------
# Graph factory — Orchestrated Doubt Solver
# ---------------------------------------------------------------------------


def build_orchestrated_doubt_solver_graph(adapter):
    """Construct and compile the lean Orchestrated Doubt Solver StateGraph.

    Args:
        adapter: AnswerGenerationAdapter.  Must be fully constructed before
                 calling this function — the generate node captures it as a
                 closure.  Tests inject a mock-safe adapter here.

    Returns:
        A compiled LangGraph CompiledGraph.

    Node flow:
        START → classify → collect_context → generate → END

    State:
        OrchestratedDoubtSolverState — retrieval_context remains internal to the graph.
        No plan, no response, no sources, no route_decision.

    Guard:
        Only call when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true.
        Default path is build_doubt_solver_graph().
    """
    def _generate_node(state: OrchestratedDoubtSolverState) -> dict:
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

        try:
            answer: str = adapter.generate(
                request_id=state["request_id"],
                query=state["query"],
                subject=subject,
                intent=intent,
                difficulty=difficulty,
                context=context_text,
                web_search_reason=str(classification_dict.get("web_search_reason"))
                if classification_dict.get("web_search_reason")
                else None,
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
            answer = (
                "I couldn't generate the answer right now because the AI provider "
                "is unavailable or quota-limited. Please try again later."
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
        return {"answer": answer}

    builder: StateGraph = StateGraph(OrchestratedDoubtSolverState)
    builder.add_node("classify", _orchestrated_classify_node)
    builder.add_node("collect_context", _orchestrated_collect_context_node)
    builder.add_node("generate", _generate_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "collect_context")
    builder.add_edge("collect_context", "generate")
    builder.add_edge("generate", END)

    return builder.compile()
