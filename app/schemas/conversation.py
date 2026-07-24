"""Typed conversation-history and short-term-memory contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.doubt_solver import CanonicalLanguage, QualityStatus

PersistenceStatus = Literal[
    "succeeded",
    "skipped",
    "failed_transient",
    "failed_permission",
    "failed_configuration",
    "failed_validation",
    "idempotent_replay",
]
PersistenceSkipReason = Literal[
    "failed_quality_gate",
    "empty_final_answer",
    "language_non_compliant",
    "request_cancelled",
    "clarification_response",
    "not_finalized",
]
RecentContextSource = Literal["agentcore_memory", "dynamodb_fallback", "none"]
RecentContextFailureReason = Literal[
    "memory_not_configured",
    "memory_no_events",
    "memory_timeout",
    "memory_permission_denied",
    "memory_resource_missing",
    "memory_actor_mismatch",
    "memory_conversation_mismatch",
    "memory_malformed_events",
    "memory_no_completed_turns",
    "history_no_items",
    "history_not_configured",
    "history_permission_denied",
    "history_query_failed",
    "history_actor_mismatch",
    "history_conversation_mismatch",
    "history_no_completed_turns",
]
RecentContextStoreStatus = Literal[
    "not_attempted",
    "not_configured",
    "succeeded",
    "empty",
    "failed_transient",
    "failed_permission",
    "failed_configuration",
    "failed_validation",
]
ConversationRelationType = Literal[
    "independent",
    "follow_up",
    "continuation",
    "correction",
    "clarification_of_previous",
    "regeneration_request",
    "ambiguous",
]
ConversationDecisionSource = Literal[
    "contextual_classifier",
    "deterministic_signal",
    "combined",
    "no_context",
]
ReferencedTurnPosition = Literal[
    "latest_turn",
    "previous_of_two",
    "both_turns",
    "none",
]


class RecentConversationTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_id: str = Field(min_length=1, max_length=128)
    original_query: str = Field(min_length=1, max_length=5000)
    final_answer: str = Field(min_length=1, max_length=8000)
    created_at: datetime


class CompletedConversationTurn(RecentConversationTurn):
    actor_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    exam_id: str | None = Field(default=None, max_length=128)
    exam_stage: str | None = Field(default=None, max_length=64)
    language: CanonicalLanguage
    subject: str = Field(default="unknown", max_length=128)
    topic: str | None = Field(default=None, max_length=256)
    quality_status: QualityStatus
    was_regenerated: bool = False

    @classmethod
    def now(cls, **values: object) -> CompletedConversationTurn:
        return cls(created_at=datetime.now(UTC), **values)


class ResolvedFollowUpQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    resolved_query: str = Field(min_length=1, max_length=5000)
    confidence: float = Field(ge=0.0, le=1.0)


class RecentConversationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    turns: tuple[RecentConversationTurn, ...] = Field(default_factory=tuple, max_length=3)
    formatted_reference: str = Field(default="", max_length=7000)
    source: str = Field(default="none", pattern=r"^(agentcore|dynamodb|none)$")


class RecentContextLoadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: RecentContextSource = "none"
    memory_attempted: bool = False
    memory_status: RecentContextStoreStatus = "not_attempted"
    memory_event_count: int = Field(default=0, ge=0)
    memory_completed_pair_count: int = Field(default=0, ge=0)
    memory_returned_turn_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=3)
    memory_latest_event_time: datetime | None = None
    memory_duration_ms: int = Field(default=0, ge=0)
    memory_failure_reason: RecentContextFailureReason | None = None
    dynamodb_attempted: bool = False
    dynamodb_status: RecentContextStoreStatus = "not_attempted"
    dynamodb_item_count: int = Field(default=0, ge=0)
    dynamodb_returned_turn_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=3)
    dynamodb_duration_ms: int = Field(default=0, ge=0)
    dynamodb_failure_reason: RecentContextFailureReason | None = None
    turns: tuple[RecentConversationTurn, ...] = Field(default_factory=tuple, max_length=3)
    usable_turn_count: int = Field(default=0, ge=0, le=3)
    failure_reason: RecentContextFailureReason | None = None
    latency_ms: int = Field(default=0, ge=0)
    formatted_reference: str = Field(default="", max_length=7000)

    def as_conversation_context(self) -> RecentConversationContext:
        source = {
            "agentcore_memory": "agentcore",
            "dynamodb_fallback": "dynamodb",
            "none": "none",
        }[self.source]
        return RecentConversationContext(
            turns=self.turns,
            formatted_reference=self.formatted_reference,
            source=source,
        )


class ConversationRelation(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation: ConversationRelationType
    requires_recent_conversation: bool
    referenced_turn_id: str | None = Field(default=None, max_length=128)
    referenced_turn_position: ReferencedTurnPosition = "none"
    confidence: float = Field(ge=0.0, le=1.0)
    decision_source: ConversationDecisionSource
    matched_signals: tuple[str, ...] = Field(default_factory=tuple, max_length=16)


class ConversationUnderstandingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation: ConversationRelation
    context_load: RecentContextLoadResult
    selected_turns: tuple[RecentConversationTurn, ...] = Field(
        default_factory=tuple,
        max_length=2,
    )
    selection_reason: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    selection_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    resolved_query: str | None = Field(default=None, max_length=5000)
    conversation_context: str = Field(default="", max_length=7000)


class PersistenceDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    persistable: bool
    skip_reason: PersistenceSkipReason | None = None


class ConversationPersistenceResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str = Field(default="", max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=128)
    persistable: bool
    history_write_status: PersistenceStatus
    session_write_status: PersistenceStatus
    memory_write_status: PersistenceStatus
    skip_reason: PersistenceSkipReason | None = None
    history_latency_ms: int = Field(default=0, ge=0)
    session_latency_ms: int = Field(default=0, ge=0)
    memory_latency_ms: int = Field(default=0, ge=0)
