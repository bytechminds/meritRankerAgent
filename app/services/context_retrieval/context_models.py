"""
app/services/context_retrieval/context_models.py
--------------------------------------------------
Pydantic models for the context retrieval boundary.

Graph-facing services pass ContextRetrievalRequest — never raw graph state,
prompts, or provider metadata.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from retrieval.models import StudentRetrievalContext

ContextSourceType = Literal["bedrock_kb", "none"]
PatternHintStrength = Literal["weak", "medium", "strong"]


class PatternHints(BaseModel):
    """Resolved topic/pattern hints for KB retrieval and reranking."""

    pattern_topic_key: str | None = Field(default=None, max_length=128)
    pattern_family_key: str | None = Field(default=None, max_length=128)
    topic_hint: str | None = Field(default=None, max_length=256)
    matched_signals: list[str] = Field(default_factory=list)
    strength: PatternHintStrength = "weak"
    retrieval_tags: list[str] = Field(default_factory=list)
    topic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    hint_source: str = Field(default="none", max_length=64)

    model_config = {"str_strip_whitespace": True}


class ContextRetrievalRequest(BaseModel):
    """Clean input for context retrieval — no graph state object."""

    request_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=5000)
    subject: str = Field(default="general")
    intent: str = Field(default="explain")
    difficulty: str = Field(default="default")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Classifier confidence when available outside graph state.",
    )
    topic: str | None = Field(default=None, max_length=256)
    topic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    pattern_topic_candidate: str | None = Field(default=None, max_length=128)
    pattern_family_candidate: str | None = Field(default=None, max_length=128)
    retrieval_tags: list[str] = Field(default_factory=list, max_length=12)
    exam: str | None = Field(default=None, max_length=128)
    max_context_chars: int = Field(default=2500, ge=0, le=8000)
    max_results: int = Field(default=5, ge=1, le=20)
    retrieval_version: str = Field(default="v1", min_length=1, max_length=32)
    need_web_search: bool = Field(default=False)
    web_search_reason: str | None = Field(default=None, max_length=64)
    web_search_query: str | None = Field(default=None, max_length=256)

    model_config = {"str_strip_whitespace": True}


class RetrievedContextItem(BaseModel):
    """Normalised KB chunk — no raw AWS response fields."""

    source_type: ContextSourceType = "bedrock_kb"
    text: str = Field(..., min_length=1, max_length=8000)
    score: float | None = Field(default=None, ge=0.0)
    source_id: str | None = Field(default=None, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)
    title: str | None = Field(default=None, max_length=256)
    match_lane: str | None = Field(default=None, max_length=64)
    rerank_score: float | None = Field(default=None, ge=0.0)
    rerank_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    why_candidate: str | None = Field(default=None, max_length=512)
    risk: str | None = Field(default=None, max_length=128)


class ContextRetrievalDecision(BaseModel):
    """Policy output — whether and how to retrieve KB context."""

    use_kb: bool
    reason: str = Field(..., max_length=256)
    filters: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=0, le=20)
    rerank_top_n: int = Field(default=2, ge=0, le=5)
    max_context_chars: int = Field(default=2500, ge=0, le=8000)


class ContextRetrievalResult(BaseModel):
    """Compact retrieval output for the generator."""

    context_text: str = Field(default="", max_length=8000)
    item_count: int = Field(default=0, ge=0)
    retrieval_used: bool = False
    reason: str = Field(default="", max_length=256)
    retrieval_context: StudentRetrievalContext = Field(
        default_factory=StudentRetrievalContext.fresh_solve
    )
    web_search_executed: bool = False
    web_search_status: str = Field(default="skipped", max_length=32)
    web_search_provider: str = Field(default="", max_length=64)
    web_context_chars: int = Field(default=0, ge=0, le=8000)
    web_result_count: int = Field(default=0, ge=0, le=20)
    web_citation_count: int = Field(default=0, ge=0, le=20)
