"""
app/schemas/doubt_solver.py
---------------------------
Pydantic v2 schemas for the Doubt Solver workflow.

Models:
    DoubtSolverRequest      — inbound payload (validated at API boundary)
    QueryClassification     — classifier output (validated at service boundary)
    AnswerOutput            — answer generator output (validated at service boundary)
    DoubtSolverResponse     — outbound payload
    DoubtSolverState        — Python-layer state (Pydantic); the LangGraph
                              internal TypedDict lives in doubt_solver_graph.py
    DoubtSolverClassification — lean classification for the orchestrated graph state
                              (ENABLE_ORCHESTRATED_DOUBT_SOLVER path only)
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from schemas.image_input import ImageInput


def _normalize_retrieval_tags(raw_tags: Any, *, max_tags: int = 10) -> list[str]:
    if raw_tags is None:
        return []
    items: list[str] = []
    if isinstance(raw_tags, str):
        items = [t.strip() for t in raw_tags.replace(";", ",").split(",") if t.strip()]
    elif isinstance(raw_tags, (list, tuple, set)):
        for entry in raw_tags:
            if entry is None:
                continue
            text = str(entry).strip()
            if text:
                items.append(text)
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in items:
        clean = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
        if not clean or clean in seen:
            continue
        seen.add(clean)
        normalized.append(clean)
        if len(normalized) >= max_tags:
            break
    return normalized


class DoubtSolverRequest(BaseModel):
    """Validated inbound payload for a doubt solver request."""

    mode: Literal["doubt_solver"]
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=5000,
        description="The student's text question or instruction for an attached image.",
    )
    image: ImageInput | None = Field(
        default=None,
        description="Optional image containing the source question.",
    )
    user_id: str = Field(
        default="local-user",
        min_length=1,
        max_length=128,
        description="Caller identifier — used for tracing.",
    )
    language: Literal["en", "hi", "hinglish"] = Field(
        default="en",
        description="Preferred response language.",
    )
    exam_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Optional selected exam identifier used only for answer presentation.",
    )
    exam_stage: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="Optional selected exam stage used only for answer presentation.",
    )
    stream: bool = Field(
        default=False,
        description="Request a streaming response with student-friendly status "
                    "labels and real answer chunks. Only honoured by the "
                    "orchestrated graph path.",
    )

    model_config = {"str_strip_whitespace": True}

    @model_validator(mode="after")
    def _require_query_or_image(self) -> DoubtSolverRequest:
        if self.query is None and self.image is None:
            raise ValueError("Either query or image is required.")
        return self


QueryIntent = Literal[
    "solve_question",
    "explain_concept",
    "explain_option",
    "general_doubt",
    "practice_question",
    "visualize_question",
    "unknown",
]
ResponseStyle = Literal["step_by_step", "short_answer", "simple_explanation"]
QueryDifficulty = Literal["default", "basic", "intermediate", "advanced"]
RetrievalNeed = Literal["none", "concept_context", "similar_question", "unknown"]
ClassificationSource = Literal["deterministic", "llm", "fallback"]


class QueryClassification(BaseModel):
    """Structured output from the query classifier service."""

    intent: QueryIntent = Field(description="Detected intent of the student query.")
    subject: str = Field(
        default="unknown",
        description="Detected subject area (e.g. 'math', 'reasoning').",
    )
    topic: str | None = Field(
        default=None,
        description="Human-readable exam topic label from classifier.",
    )
    topic_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence for topic/pattern hint fields.",
    )
    pattern_topic_candidate: str | None = Field(
        default=None,
        max_length=128,
        description="Canonical patternTopicKey candidate (UPPER_SNAKE) when obvious.",
    )
    pattern_family_candidate: str | None = Field(
        default=None,
        max_length=128,
        description="Canonical patternFamilyKey candidate when obvious.",
    )
    retrieval_tags: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Compact normalized tags for KB rerank signals (not strict filters).",
    )
    response_style: ResponseStyle = Field(
        default="step_by_step",
        description="Preferred response style for this intent.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Classifier confidence score.",
    )
    difficulty: QueryDifficulty = Field(
        default="default",
        description="Detected difficulty level of the query.",
    )
    retrieval_need: RetrievalNeed = Field(
        default="unknown",
        description="Whether retrieval context would improve the answer.",
    )
    classification_source: ClassificationSource = Field(
        default="deterministic",
        description="Which path produced this classification.",
    )
    reasoning_summary: str | None = Field(
        default=None,
        max_length=500,
        description="Short non-sensitive explanation of the classification (from LLM only).",
    )
    need_web_search: bool = Field(
        default=False,
        description="True when fresh web context is required for the answer.",
    )
    web_search_reason: str | None = Field(
        default=None,
        max_length=64,
        description="Compact reason enum for web search trigger (internal only).",
    )
    web_search_query: str | None = Field(
        default=None,
        max_length=256,
        description="Concise web search query without personal data (internal only).",
    )

    @field_validator("retrieval_tags", mode="before")
    @classmethod
    def _validate_retrieval_tags(cls, value: Any) -> list[str]:
        return _normalize_retrieval_tags(value, max_tags=12)

    @field_validator("pattern_topic_candidate", "pattern_family_candidate", mode="before")
    @classmethod
    def _strip_pattern_candidates(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class AnswerOutput(BaseModel):
    """Validated output from the answer generator service.

    Carries the answer content, the path that produced it, and a truncation
    flag.  Validated at the service boundary before being written to graph state.
    """

    content: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The generated answer text.",
    )
    answer_source: Literal["mock", "llm", "fallback"] = Field(
        description="Which path produced this answer.",
    )
    is_truncated: bool = Field(
        default=False,
        description="True when model output exceeded the max length and was truncated.",
    )


class ResponseContent(BaseModel):
    """Canonical answer content shared by final and streamed responses."""

    format: Literal["markdown"] = "markdown"
    value: str = Field(min_length=1, max_length=8000)


class DoubtSolverResponse(BaseModel):
    """Serialised outbound payload for a doubt solver response."""

    schema_version: Literal["1"] = "1"
    status: Literal["completed"] = "completed"
    success: bool = Field(description="True when an answer was produced.")
    request_id: str = Field(description="UUID assigned at the entrypoint.")
    mode: Literal["doubt_solver"]
    answer: str = Field(description="The generated explanation.")
    classification: QueryClassification = Field(
        description="Classification result from the classifier node."
    )
    needs_review: bool = Field(
        default=False,
        description="True when the answer requires human review (e.g. low confidence).",
    )
    answer_source: Literal["mock", "llm", "fallback"] = Field(
        default="mock",
        description="Which path produced the answer.",
    )
    is_truncated: bool = Field(
        default=False,
        description="True when the generated answer was truncated to fit the max length.",
    )
    used_retrieval: bool = Field(
        default=False,
        description="True when KB retrieval was called and returned at least one result.",
    )
    source_count: int = Field(
        default=0,
        ge=0,
        description="Number of context sources (KB results + DynamoDB records) used.",
    )
    context_used: bool = Field(
        default=False,
        description="True when retrieved context was included in the answer generation prompt.",
    )
    content: ResponseContent | None = Field(
        default=None,
        description=(
            "Canonical Markdown response content. `answer` remains its compatibility "
            "projection."
        ),
    )

    @model_validator(mode="after")
    def _project_answer_to_content(self) -> DoubtSolverResponse:
        if self.content is None:
            self.content = ResponseContent(value=self.answer)
        elif self.content.value != self.answer:
            raise ValueError("content.value must match answer")
        return self


class DoubtSolverState(BaseModel):
    """Python-layer state for the Doubt Solver workflow.

    Used outside the LangGraph graph (e.g. in main.py) to package request
    context as a typed object.  The actual LangGraph internal state uses a
    plain TypedDict (DoubtSolverGraphState in doubt_solver_graph.py) so that
    LangGraph's reducer system can serialise it freely.
    """

    request: DoubtSolverRequest = Field(description="The validated inbound request.")
    request_id: str = Field(description="UUID assigned at the entrypoint.")
    classification: QueryClassification | None = Field(
        default=None,
        description="Classifier output, populated after classify_query node.",
    )
    answer: str | None = Field(
        default=None,
        description="Generated answer string, populated after generate_answer node.",
    )
    response: DoubtSolverResponse | None = Field(
        default=None,
        description="Final structured response, populated after build_response node.",
    )


# ---------------------------------------------------------------------------
# DoubtSolverClassification — lean classification for the orchestrated graph
# (ENABLE_ORCHESTRATED_DOUBT_SOLVER path only)
# ---------------------------------------------------------------------------


class DoubtSolverClassification(BaseModel):
    """Minimal classification for the orchestrated Doubt Solver graph state.

    Intentionally small — only what the generator node needs to build a
    RouteRequest.  Internal classifier scores, topics, and style hints stay
    outside the orchestrated graph state.

    Allowed values match the route config in llm_routes.yaml:
        subject:    math | reasoning | english | general
        intent:     solve | explain | practice
        difficulty: default | basic | intermediate | advanced
        retrieval_required: True when the classifier recommends context retrieval
    """

    subject: str = Field(default="general")
    intent: str = Field(default="explain")
    difficulty: str = Field(default="default")
    retrieval_required: bool = Field(default=False)
    topic: str | None = Field(
        default=None,
        max_length=256,
        description="Human-readable topic hint for retrieval (optional).",
    )
    topic_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Classifier confidence for retrieval hints.",
    )
    pattern_topic_candidate: str | None = Field(default=None, max_length=128)
    pattern_family_candidate: str | None = Field(default=None, max_length=128)
    retrieval_tags: list[str] = Field(
        default_factory=list,
        max_length=12,
        description="Normalized retrieval tags for rerank (optional).",
    )
    need_web_search: bool = Field(
        default=False,
        description="True when fresh web context is required.",
    )
    web_search_reason: str | None = Field(default=None, max_length=64)
    web_search_query: str | None = Field(default=None, max_length=256)

    model_config = {"str_strip_whitespace": True}

    @field_validator("retrieval_tags", mode="before")
    @classmethod
    def _validate_retrieval_tags(cls, value: Any) -> list[str]:
        return _normalize_retrieval_tags(value, max_tags=12)


# ---------------------------------------------------------------------------
# DoubtSolverStreamEvent — single event in a streaming orchestrated response
# ---------------------------------------------------------------------------


_FORBIDDEN_STREAM_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "prompt",
        "system_prompt",
        "user_prompt",
        "messages",
        "context",
        "context_text",
        "query",
        "api_key",
        "secret",
        "credential",
        "authorization",
        "raw_response",
        "provider",
        "deployment",
        "model_id",
        "stack_trace",
    }
)


class DoubtSolverFinalResponse(BaseModel):
    """Authoritative final response emitted by a completed answer stream."""

    schema_version: Literal["1"] = "1"
    request_id: str = Field(min_length=1)
    status: Literal["completed"] = "completed"
    content: ResponseContent
    answer: str = Field(min_length=1, max_length=8000)

    @model_validator(mode="after")
    def _project_content_to_answer(self) -> DoubtSolverFinalResponse:
        if self.answer != self.content.value:
            raise ValueError("answer must match content.value")
        return self


class DoubtSolverStreamEvent(BaseModel):
    """A single event in an orchestrated Doubt Solver streaming response.

    Events are emitted in order:
        1. ``status``   — student-friendly progress labels (understanding, thinking, …).
        2. ``chunk``    — zero or more, each carries one text chunk of the answer.
        3. ``complete`` — exactly one, last event, may carry safe metadata.
        4. ``error``    — replaces ``complete`` when generation fails.

    Security note: ``content`` must contain ONLY answer text chunks.
    No prompt, messages, context, API keys, or provider details may appear
    in any field.
    """

    type: Literal["status", "chunk", "complete", "error"] = Field(
        description="Type of streaming event.",
    )
    request_id: str = Field(
        min_length=1,
        description="UUID of the originating request.",
    )
    stage: str | None = Field(
        default=None,
        description="Internal stage name (status/complete/error events only).",
    )
    label: str | None = Field(
        default=None,
        description="Student-facing label (status/complete/error events only).",
    )
    content: str | None = Field(
        default=None,
        description=(
            "Answer text chunk (non-None only for 'chunk' events). "
            "Must contain answer text only — no prompt, context, or credentials."
        ),
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Safe tracing metadata only (e.g. request_id echo). "
            "Must not contain secrets, prompt text, context, or provider details."
        ),
    )
    response: DoubtSolverFinalResponse | None = Field(
        default=None,
        description="Authoritative final response, present only on complete events.",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_event_shape(self) -> DoubtSolverStreamEvent:
        for key in self.metadata:
            if key.lower() in _FORBIDDEN_STREAM_METADATA_KEYS:
                raise ValueError(
                    f"metadata must not contain forbidden key: {key!r}"
                )

        if self.type != "complete" and self.response is not None:
            raise ValueError("response is only allowed on complete events")

        if self.type == "chunk":
            if self.content is None:
                raise ValueError("chunk event must have content")
        elif self.type == "status":
            if not self.stage or not self.label:
                raise ValueError("status event must have stage and label")
        elif self.type == "error":
            if not self.label:
                raise ValueError("error event must have a safe label/message")
        elif self.type == "complete":
            if self.response is None:
                raise ValueError("complete event must have response")
            if self.response.request_id != self.request_id:
                raise ValueError("complete response request_id must match event request_id")

        return self
