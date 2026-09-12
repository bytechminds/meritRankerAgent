"""Typed contracts for student S3 Vector retrieval."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RetrievalMode = Literal["runtime_ready", "pattern_assist", "fresh_solve"]
CandidateSource = Literal["s3_vector", "colbert", "cache"]


class RetrievedCandidate(BaseModel):
    """Candidate metadata returned by S3 Vectors, never final-answer authority."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    chunk_type: str | None = Field(default=None, alias="chunkType", max_length=128)
    subject: str | None = Field(default=None, max_length=64)
    topic: str | None = Field(default=None, max_length=256)
    flow_type: str | None = Field(default=None, alias="flowType", max_length=128)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    distance: float | None = Field(default=None, ge=0.0)
    vector_key: str | None = Field(default=None, alias="vectorKey", max_length=1024)
    version_hash: str | None = Field(default=None, alias="versionHash", max_length=256)
    runtime_ready: bool = Field(default=False, alias="runtimeReady")
    can_use_for_final_answer: bool = Field(default=False, alias="canUseForFinalAnswer")
    can_use_for_retrieval: bool = Field(default=False, alias="canUseForRetrieval")
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: CandidateSource = "s3_vector"


class PatternRuntimeBundle(BaseModel):
    """DynamoDB source-of-truth student runtime bundle."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    pattern_status: Literal["approved", "needs_review", "rejected"] = Field(
        alias="patternStatus"
    )
    answer_data_status: Literal[
        "answer_valid",
        "valid",
        "missing",
        "ambiguous",
        "inconsistent",
        "unknown",
    ] = Field(alias="answerDataStatus")
    solve_flow_status: Literal["approved", "needs_review", "missing", "not_required"] = (
        Field(alias="solveFlowStatus")
    )
    runtime_ready: bool = Field(alias="runtimeReady")
    student_visibility: Literal["runtime_ready", "pattern_only", "admin_only"] = Field(
        alias="studentVisibility"
    )
    can_use_for_retrieval: bool = Field(alias="canUseForRetrieval")
    can_use_for_final_answer: bool = Field(alias="canUseForFinalAnswer")
    subject: str = Field(min_length=1, max_length=64)
    topic: str = Field(default="", max_length=256)
    flow_type: Literal[
        "COMPUTATIONAL_SOLVE_FLOW",
        "REASONING_TRACE_FLOW",
        "ANSWER_EXPLANATION_FLOW",
        "FACT_CONCEPT_ANSWER_FLOW",
    ] = Field(alias="flowType")
    pattern_title: str = Field(default="", alias="patternTitle", max_length=512)
    pattern_graph_compact: dict[str, Any] = Field(default_factory=dict, alias="patternGraphCompact")
    solve_flow: dict[str, Any] = Field(default_factory=dict, alias="solveFlow")
    answer_guide: dict[str, Any] = Field(default_factory=dict, alias="answerGuide")
    not_same_when: list[str] = Field(default_factory=list, alias="notSameWhen")
    material_conflicts: list[str] = Field(default_factory=list, alias="materialConflicts")
    version_hash: str = Field(alias="versionHash", min_length=1, max_length=256)
    updated_at: str = Field(default="", alias="updatedAt", max_length=64)


class RetrievalTrace(BaseModel):
    """Safe retrieval diagnostics retained in graph state only."""

    model_config = ConfigDict(populate_by_name=True)

    runtime_candidates_count: int = Field(default=0, alias="runtimeCandidatesCount", ge=0)
    pattern_candidates_count: int = Field(default=0, alias="patternCandidatesCount", ge=0)
    rerank_used: bool = Field(default=False, alias="rerankUsed")
    graph_gate_passed: bool = Field(default=False, alias="graphGatePassed")
    fallback_reason: str | None = Field(default=None, alias="fallbackReason", max_length=128)
    # Sub-operation timings so a slow retrieval names its own bottleneck.
    embedding_ms: int = Field(default=0, alias="embeddingMs", ge=0)
    runtime_query_ms: int = Field(default=0, alias="runtimeQueryMs", ge=0)
    pattern_query_ms: int = Field(default=0, alias="patternQueryMs", ge=0)
    rerank_ms: int = Field(default=0, alias="rerankMs", ge=0)


class StudentRetrievalContext(BaseModel):
    """Internal planner-ready retrieval contract rendered into generator context."""

    model_config = ConfigDict(populate_by_name=True)

    provider: Literal["s3_vector"] = "s3_vector"
    mode: RetrievalMode = "fresh_solve"
    selected_pattern_id: str | None = Field(default=None, alias="selectedPatternId", max_length=256)
    selected_candidate: RetrievedCandidate | None = Field(default=None, exclude=True)
    runtime_bundle: PatternRuntimeBundle | None = Field(default=None, exclude=True)
    pattern_graph_only: bool = Field(default=False, exclude=True)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    can_use_for_final_answer: bool = Field(default=False, alias="canUseForFinalAnswer")
    subject: str | None = Field(default=None, max_length=64)
    topic: str | None = Field(default=None, max_length=256)
    flow_type: str | None = Field(default=None, alias="flowType", max_length=128)
    pattern_graph_compact: dict[str, Any] = Field(default_factory=dict, alias="patternGraphCompact")
    solve_flow: dict[str, Any] = Field(default_factory=dict, alias="solveFlow")
    answer_guide: dict[str, Any] = Field(default_factory=dict, alias="answerGuide")
    not_same_when: list[str] = Field(default_factory=list, alias="notSameWhen")
    material_conflicts: list[str] = Field(default_factory=list, alias="materialConflicts")
    retrieval_trace: RetrievalTrace = Field(default_factory=RetrievalTrace, alias="retrievalTrace")
    warnings: list[str] = Field(default_factory=list, max_length=12)

    @classmethod
    def fresh_solve(cls, reason: str | None = None) -> StudentRetrievalContext:
        return cls(
            retrieval_trace=RetrievalTrace(fallback_reason=reason),
            warnings=[reason] if reason else [],
        )


class GraphGateDecision(BaseModel):
    """Compatibility decision before a bundle enters student data context."""

    allowed: bool
    reason: str = Field(min_length=1, max_length=128)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
