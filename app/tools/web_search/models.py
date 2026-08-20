"""Pydantic models for the web search subsystem."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from practice_limits import MAX_PRACTICE_QUESTIONS

SourceQuality = Literal["trusted", "reputed", "exam_prep", "generic", "blocked"]
SearchAttemptKind = Literal[
    "authoritative",
    "authoritative_plus_reputed",
    "exam_prep_fallback",
    "generic_fallback",
]
ContextStrength = Literal["authoritative", "mixed", "supporting_only", "weak"]
TemporalMode = Literal[
    "CURRENT",
    "LATEST",
    "RECENT",
    "EXPLICIT_YEAR",
    "EXPLICIT_MONTH",
    "EXPLICIT_DATE_RANGE",
]

MAX_FRESH_EVIDENCE_ITEMS = MAX_PRACTICE_QUESTIONS


class WebSearchRequest(BaseModel):
    """Graph-facing web search input — provider-agnostic."""

    request_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=5000)
    web_search_query: str | None = Field(default=None, max_length=500)
    subject: str = Field(default="general", max_length=64)
    topic: str | None = Field(default=None, max_length=256)
    retrieval_tags: list[str] = Field(default_factory=list, max_length=12)
    web_search_reason: str | None = Field(default=None, max_length=64)
    requires_fresh_evidence: bool = False
    required_evidence_count: int = Field(default=0, ge=0, le=MAX_FRESH_EVIDENCE_ITEMS)
    timeout_seconds: float = Field(default=8.0, ge=1.0, le=30.0)

    model_config = {"str_strip_whitespace": True}


class WebSearchProviderRequest(BaseModel):
    """Provider-neutral search request — no Tavily-specific fields."""

    query: str = Field(..., min_length=1, max_length=500)
    topic: str = Field(default="general", max_length=32)
    include_domains: list[str] = Field(default_factory=list, max_length=40)
    exclude_domains: list[str] = Field(default_factory=list, max_length=40)
    start_date: str | None = Field(default=None, max_length=16)
    end_date: str | None = Field(default=None, max_length=16)
    time_range: str | None = Field(default=None, max_length=16)
    max_results: int = Field(default=5, ge=1, le=20)
    search_depth: str = Field(default="basic", max_length=16)
    include_raw_content: bool = False
    timeout_seconds: float = Field(default=8.0, ge=1.0, le=30.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"str_strip_whitespace": True}


class WebSearchItem(BaseModel):
    """Normalised web search result item."""

    title: str = Field(default="", max_length=512)
    url: str = Field(default="", max_length=2048)
    snippet: str = Field(default="", max_length=2000)
    source: str = Field(default="", max_length=256)
    published_at: str | None = Field(default=None, max_length=64)
    score: float | None = Field(default=None, ge=0.0)
    source_quality: SourceQuality | None = None
    selected_score: float | None = Field(default=None, ge=0.0)

    model_config = {"str_strip_whitespace": True}


class FreshnessWindow(BaseModel):
    """Deterministic date range applied to a freshness-required web search."""

    start_date: str | None = Field(default=None, max_length=16)
    end_date: str | None = Field(default=None, max_length=16)
    source: Literal["default", "user_explicit", "none"] = "none"
    temporal_mode: TemporalMode = "CURRENT"
    label: str = Field(default="", max_length=64)

    model_config = {"str_strip_whitespace": True, "frozen": True}


class FreshEvidenceBundle(BaseModel):
    """Compact, selected external evidence retained for grounded Practice."""

    requested_window: FreshnessWindow
    retrieved_at: str = Field(min_length=1, max_length=64)
    search_query: str = Field(min_length=1, max_length=500)
    items: list[WebSearchItem] = Field(
        default_factory=list,
        max_length=MAX_FRESH_EVIDENCE_ITEMS,
    )

    model_config = {"str_strip_whitespace": True, "frozen": True}


class WebSearchProviderResult(BaseModel):
    """Provider adapter output."""

    items: list[WebSearchItem] = Field(
        default_factory=list,
        max_length=MAX_FRESH_EVIDENCE_ITEMS,
    )
    provider: str = Field(default="", max_length=64)
    attempt: SearchAttemptKind = "authoritative"
    error_kind: str | None = Field(default=None, max_length=64)


class WebSearchResult(BaseModel):
    """Final web search output for context retrieval."""

    used: bool = False
    provider: str = Field(default="", max_length=64)
    query: str = Field(default="", max_length=500)
    items: list[WebSearchItem] = Field(
        default_factory=list,
        max_length=MAX_FRESH_EVIDENCE_ITEMS,
    )
    context_text: str = Field(default="", max_length=8000)
    reason: str = Field(default="", max_length=128)
    error_kind: str | None = Field(default=None, max_length=64)
    weak_context: bool = False
    source_pack_name: str = Field(default="", max_length=64)
    attempt_used: SearchAttemptKind | None = None
    freshness_label: str = Field(default="", max_length=64)
    executed: bool = False
    attempt_count: int = Field(default=0, ge=0, le=4)
    candidate_count: int = Field(default=0, ge=0, le=80)
    official_count: int = Field(default=0, ge=0, le=20)
    reputable_count: int = Field(default=0, ge=0, le=20)
    duration_ms: int = Field(default=0, ge=0)
    freshness_required: bool = False
    freshness_start_date: str | None = Field(default=None, max_length=16)
    freshness_end_date: str | None = Field(default=None, max_length=16)
    temporal_mode: TemporalMode | None = None
    stale_results_rejected: int = Field(default=0, ge=0)
    eligible_evidence_count: int = Field(default=0, ge=0, le=MAX_FRESH_EVIDENCE_ITEMS)
    evidence_date_min: str | None = Field(default=None, max_length=16)
    evidence_date_max: str | None = Field(default=None, max_length=16)
    fresh_evidence: FreshEvidenceBundle | None = None

    model_config = {"str_strip_whitespace": True}
