"""Typed conversation-history and short-term-memory contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from schemas.doubt_solver import (
    CanonicalLanguage,
    ConversationClassificationAction,
    ConversationClassificationRelation,
    QualityStatus,
)

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
    "non_substantive_answer",
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
ConversationTurnType = Literal[
    "academic_question_answer",
    "clarification",
    "acknowledgement",
    "error",
    "failed_quality",
    "correction_only",
    "meta_response",
]
ContextNeedDecision = Literal[
    "CONTEXT_NOT_NEEDED",
    "CONTEXT_REQUIRED",
    "UNCERTAIN",
]
GenerationContextPolicy = Literal[
    "current_only",
    "answer_with_context",
    "explain_previous",
    "continue_previous",
    "generate_similar",
    "transform_previous",
    "verify_and_correct",
    "resolve_from_scratch",
    "clarification",
]


class RecentConversationTurn(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_id: str = Field(min_length=1, max_length=128)
    original_query: str = Field(min_length=1, max_length=5000)
    final_answer: str = Field(min_length=1, max_length=8000)
    created_at: datetime
    turn_type: ConversationTurnType = "academic_question_answer"


class ContextNeedAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    decision: ContextNeedDecision
    reason_codes: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    matched_signals: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    duration_ms: int = Field(default=0, ge=0)


class ConversationCandidateCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    turn_id: str = Field(min_length=1, max_length=128)
    question_preview: str = Field(min_length=1, max_length=700)
    answer_clue: str = Field(min_length=1, max_length=500)
    subject: str = Field(default="unknown", max_length=128)
    topic: str | None = Field(default=None, max_length=256)
    difficulty: str = Field(default="default", max_length=32)
    query_match_indicators: tuple[str, ...] = Field(default_factory=tuple, max_length=6)


class ConversationPreparation(BaseModel):
    model_config = ConfigDict(frozen=True)

    gate: ContextNeedAssessment
    context_load: RecentContextLoadResult
    eligible_turns: tuple[RecentConversationTurn, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    candidates: tuple[ConversationCandidateCard, ...] = Field(
        default_factory=tuple,
        max_length=5,
    )
    rejected_turn_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    candidate_characters: int = Field(default=0, ge=0, le=7200)


class SelectedGenerationContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    relation: ConversationClassificationRelation
    requested_action: ConversationClassificationAction
    selected_turn_id: str | None = Field(default=None, max_length=128)
    resolved_reference: str | None = Field(default=None, max_length=128)
    resolved_query: str = Field(min_length=1, max_length=5000)
    conversation_context: str = Field(default="", max_length=8000)
    context_policy: GenerationContextPolicy
    context_characters: int = Field(default=0, ge=0, le=8000)
    clarification_required: bool = False


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

    turns: tuple[RecentConversationTurn, ...] = Field(default_factory=tuple, max_length=5)
    formatted_reference: str = Field(default="", max_length=7000)
    source: str = Field(default="none", pattern=r"^(agentcore|dynamodb|none)$")


class RecentContextLoadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: RecentContextSource = "none"
    memory_attempted: bool = False
    memory_status: RecentContextStoreStatus = "not_attempted"
    memory_event_count: int = Field(default=0, ge=0)
    memory_completed_pair_count: int = Field(default=0, ge=0)
    memory_returned_turn_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    memory_latest_event_time: datetime | None = None
    memory_duration_ms: int = Field(default=0, ge=0)
    memory_failure_reason: RecentContextFailureReason | None = None
    dynamodb_attempted: bool = False
    dynamodb_status: RecentContextStoreStatus = "not_attempted"
    dynamodb_item_count: int = Field(default=0, ge=0)
    dynamodb_returned_turn_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=5)
    dynamodb_duration_ms: int = Field(default=0, ge=0)
    dynamodb_failure_reason: RecentContextFailureReason | None = None
    turns: tuple[RecentConversationTurn, ...] = Field(default_factory=tuple, max_length=5)
    usable_turn_count: int = Field(default=0, ge=0, le=5)
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
