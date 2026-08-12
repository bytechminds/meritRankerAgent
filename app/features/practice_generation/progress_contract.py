"""Strict producer-side contract for AppSync practice-progress metadata."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

PRACTICE_PROGRESS_CONTRACT_VERSION = "practice-progress-meta-v1"


class PracticeProgressMeta(BaseModel):
    """The exact top-level ``meta`` fields accepted by the AppSync resolver."""

    schema_version: str | None = Field(default=None, alias="schemaVersion")
    idempotency_key: str | None = Field(default=None, alias="idempotencyKey")
    conversation_id: str | None = Field(default=None, alias="conversationId")
    turn_id: str | None = Field(default=None, alias="turnId")
    practice_type: str | None = Field(default=None, alias="practiceType")
    requested_count: int | None = Field(default=None, alias="requestedCount")
    accepted_count: int | None = Field(default=None, alias="acceptedCount")
    exam_stage: str | None = Field(default=None, alias="examStage")
    requested_language: str | None = Field(default=None, alias="requestedLanguage")
    phase: str | None = None
    playable: bool | None = None
    progress_percent: int | None = Field(default=None, alias="progressPercent")
    ready_question_count: int | None = Field(default=None, alias="readyQuestionCount")
    question_manifest_version: int | None = Field(
        default=None,
        alias="questionManifestVersion",
    )
    ready_question_ids: list[str] | None = Field(default=None, alias="readyQuestionIds")
    ready_count: int | None = Field(default=None, alias="readyCount")
    reused_count: int | None = Field(default=None, alias="reusedCount")
    generated_count: int | None = Field(default=None, alias="generatedCount")
    verified_count: int | None = Field(default=None, alias="verifiedCount")
    failed_count: int | None = Field(default=None, alias="failedCount")
    practice_request: dict[str, Any] | None = Field(default=None, alias="practiceRequest")
    resource_aliases: dict[str, Any] | None = Field(default=None, alias="resourceAliases")
    blueprint: dict[str, Any] | None = None
    planner_calls: int | None = Field(default=None, alias="plannerCalls")
    planner_tier: str | None = Field(default=None, alias="plannerTier")
    planner_repaired: bool | None = Field(default=None, alias="plannerRepaired")
    planner_deterministic_fallback: bool | None = Field(
        default=None,
        alias="plannerDeterministicFallback",
    )
    bucket_ready_counts: dict[str, int] | None = Field(
        default=None,
        alias="bucketReadyCounts",
    )
    deficits: dict[str, int] | None = None
    generation_groups: dict[str, dict[str, Any]] | None = Field(
        default=None,
        alias="generationGroups",
    )
    finalization_attempt: int | None = Field(default=None, alias="finalizationAttempt")
    finalization_reason_code: str | None = Field(
        default=None,
        alias="finalizationReasonCode",
    )
    ready_at: str | None = Field(default=None, alias="readyAt")
    error_code: str | None = Field(default=None, alias="errorCode")
    generation_version: int | None = Field(default=None, alias="generationVersion")
    started_at: str | None = Field(default=None, alias="startedAt")
    last_progress_at: str | None = Field(default=None, alias="lastProgressAt")
    recovery_attempt_count: int | None = Field(
        default=None,
        alias="recoveryAttemptCount",
    )
    last_completed_stage: str | None = Field(default=None, alias="lastCompletedStage")
    replacement_wave_count: int | None = Field(
        default=None,
        alias="replacementWaveCount",
    )
    slot_ready_counts: dict[str, int] | None = Field(default=None, alias="slotReadyCounts")

    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)


PRACTICE_PROGRESS_ALLOWED_META_KEYS = frozenset(
    field.alias or field_name
    for field_name, field in PracticeProgressMeta.model_fields.items()
)

# Legacy duplicate top-level fields are never part of the strict AppSync parent-progress mutation.
# The durable recovery source is the existing approved ``practiceRequest`` object instead.
PRACTICE_PROGRESS_INTERNAL_META_KEYS = frozenset({"examProfileId"})


class PracticeProgressContractError(RuntimeError):
    """A producer-side contract violation with metadata-key-only diagnostics."""

    def __init__(
        self,
        code: str,
        *,
        actual_keys: tuple[str, ...],
        unknown_keys: tuple[str, ...],
    ) -> None:
        super().__init__(code)
        self.code = code
        self.actual_keys = actual_keys
        self.unknown_keys = unknown_keys


@dataclass(frozen=True)
class PracticeProgressMetaProjection:
    """Typed AppSync payload plus safe key-only contract diagnostics."""

    meta: PracticeProgressMeta
    actual_keys: tuple[str, ...]
    unknown_keys: tuple[str, ...]
    contract_version: str = PRACTICE_PROGRESS_CONTRACT_VERSION

    @property
    def payload(self) -> dict[str, Any]:
        return self.meta.model_dump(by_alias=True, exclude_unset=True)

    @property
    def unapproved_keys(self) -> tuple[str, ...]:
        return tuple(
            key for key in self.unknown_keys if key not in PRACTICE_PROGRESS_INTERNAL_META_KEYS
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite Decimal")
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def project_practice_progress_meta(meta: dict[str, Any]) -> PracticeProgressMetaProjection:
    """Project durable metadata into the exact strict AppSync transport contract."""

    actual_keys = tuple(sorted(str(key) for key in meta))
    unknown_keys = tuple(
        key for key in actual_keys if key not in PRACTICE_PROGRESS_ALLOWED_META_KEYS
    )
    payload = {
        key: _json_safe(deepcopy(value))
        for key, value in meta.items()
        if isinstance(key, str) and key in PRACTICE_PROGRESS_ALLOWED_META_KEYS
    }
    try:
        typed = PracticeProgressMeta.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise PracticeProgressContractError(
            "PRACTICE_PROGRESS_INVALID_META",
            actual_keys=actual_keys,
            unknown_keys=unknown_keys,
        ) from exc
    return PracticeProgressMetaProjection(
        meta=typed,
        actual_keys=actual_keys,
        unknown_keys=unknown_keys,
    )
