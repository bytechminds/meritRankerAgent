"""Feature-gated, server-side PatternGraph guidance for practice deficits."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from features.practice_generation.schemas import PlannerSlot, PracticeGenerationRequest
from retrieval.pattern_intelligence import (
    CanonicalPlayableQuestion,
    PatternGenerationContext,
    PatternMatchTier,
    PatternRuntimeMemo,
    PatternRuntimeRequest,
    PatternRuntimeService,
)

logger = logging.getLogger(__name__)


class PatternSlotSelection(BaseModel):
    """Persistable audit data only; compact graph guidance is never persisted."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    pattern_version_hash: str = Field(
        alias="patternVersionHash",
        min_length=1,
        max_length=256,
    )
    tier: PatternMatchTier
    reason: str = Field(min_length=1, max_length=96)


class PatternSlotResolution(BaseModel):
    """Request-local contexts and non-sensitive selection audit information."""

    model_config = ConfigDict(frozen=True)

    selections_by_slot: dict[str, PatternSlotSelection] = Field(default_factory=dict)
    guidance_by_slot: dict[str, PatternGenerationContext] = Field(default_factory=dict)
    reuse_questions_by_slot: dict[str, CanonicalPlayableQuestion] = Field(
        default_factory=dict
    )
    retrieval_group_count: int = Field(default=0, ge=0, le=100)
    vector_query_count: int = Field(default=0, ge=0, le=100)
    ignored_pattern_count: int = Field(default=0, ge=0, le=1_000)
    warnings: tuple[str, ...] = Field(default_factory=tuple, max_length=32)


class PatternContextProvider(Protocol):
    """Resolve guidance after planner slots exist, never before blueprinting."""

    def resolve_slots(
        self,
        *,
        request: PracticeGenerationRequest,
        slots: Sequence[PlannerSlot],
        persisted_selections: (
            Mapping[str, PatternSlotSelection | Mapping[str, object]] | None
        ) = None,
        excluded_question_ids: Sequence[str] = (),
        seen_question_ids: Sequence[str] = (),
        student_history_checked: bool = False,
    ) -> PatternSlotResolution:
        """Return compact safe guidance or an empty result without raising."""
        ...


class NoOpPatternContextProvider:
    """Preserve the verified practice path when Pattern Intelligence is unavailable."""

    def resolve_slots(
        self,
        *,
        request: PracticeGenerationRequest,
        slots: Sequence[PlannerSlot],
        persisted_selections: (
            Mapping[str, PatternSlotSelection | Mapping[str, object]] | None
        ) = None,
        excluded_question_ids: Sequence[str] = (),
        seen_question_ids: Sequence[str] = (),
        student_history_checked: bool = False,
    ) -> PatternSlotResolution:
        del (
            request,
            slots,
            persisted_selections,
            excluded_question_ids,
            seen_question_ids,
            student_history_checked,
        )
        return PatternSlotResolution()


class RuntimePatternContextProvider:
    """Adapt immutable planner slots to the shared deterministic runtime."""

    def __init__(
        self,
        *,
        runtime: PatternRuntimeService,
        max_candidates: int,
    ) -> None:
        self._runtime = runtime
        self._max_candidates = min(max(max_candidates, 1), 24)

    def resolve_slots(
        self,
        *,
        request: PracticeGenerationRequest,
        slots: Sequence[PlannerSlot],
        persisted_selections: (
            Mapping[str, PatternSlotSelection | Mapping[str, object]] | None
        ) = None,
        excluded_question_ids: Sequence[str] = (),
        seen_question_ids: Sequence[str] = (),
        student_history_checked: bool = False,
    ) -> PatternSlotResolution:
        if not slots:
            return PatternSlotResolution()

        memo = PatternRuntimeMemo()
        normalized_persisted = _normalize_persisted_selections(persisted_selections)
        slot_groups = _group_slots(slots, normalized_persisted)
        selections: dict[str, PatternSlotSelection] = {}
        guidance: dict[str, PatternGenerationContext] = {}
        reuse_questions: dict[str, CanonicalPlayableQuestion] = {}
        vector_query_count = 0
        ignored_pattern_count = 0
        warnings: list[str] = []

        for group_key, group_slots in slot_groups:
            persisted_key, _, _compatibility_key = group_key.partition("|")
            server_pattern_id = (
                persisted_key.removeprefix("pattern:")
                if persisted_key.startswith("pattern:")
                else None
            )
            server_pattern_version_hash = next(
                (
                    normalized_persisted[slot.slot_id].pattern_version_hash
                    for slot in group_slots
                    if slot.slot_id in normalized_persisted
                ),
                None,
            )
            runtime_request = _build_runtime_request(
                request=request,
                slots=group_slots,
                max_candidates=self._max_candidates,
                server_pattern_id=server_pattern_id,
                server_pattern_version_hash=server_pattern_version_hash,
                excluded_question_ids=excluded_question_ids,
                seen_question_ids=seen_question_ids,
                student_history_checked=student_history_checked,
            )
            try:
                result = self._runtime.resolve_practice(runtime_request, memo=memo)
            except Exception as exc:  # noqa: BLE001
                # Pattern guidance is optional; the normal verified path must continue.
                logger.warning(
                    "practice_pattern_runtime_unavailable request_id=%s error_type=%s",
                    request.request_id,
                    type(exc).__name__,
                )
                warnings.append("pattern_runtime_unavailable")
                continue

            if result.plan.strategy == "vector":
                vector_query_count += 1
            warnings.extend(result.warnings)
            ignored_pattern_count += sum(
                decision.tier is PatternMatchTier.IGNORE for decision in result.decisions
            )
            if (
                result.tier
                not in {PatternMatchTier.GUIDANCE_SAFE, PatternMatchTier.REUSE_SAFE}
                or result.selected_pattern_id is None
                or result.selected_pattern_version_hash is None
                or result.generation_context is None
            ):
                continue

            decision = next(
                (
                    value
                    for value in result.decisions
                    if value.pattern_id == result.selected_pattern_id
                    and value.tier is PatternMatchTier.GUIDANCE_SAFE
                ),
                None,
            )
            if decision is None:
                continue
            selection = PatternSlotSelection(
                patternId=result.selected_pattern_id,
                patternVersionHash=result.selected_pattern_version_hash,
                tier=result.tier,
                reason=(
                    "authoritative_playable_question_compatible"
                    if result.tier is PatternMatchTier.REUSE_SAFE
                    else decision.reason
                ),
            )
            for index, slot in enumerate(group_slots):
                if (
                    result.tier is PatternMatchTier.REUSE_SAFE
                    and index == 0
                    and result.reuse_question is not None
                ):
                    selections[slot.slot_id] = selection
                    reuse_questions[slot.slot_id] = result.reuse_question
                    continue
                selections[slot.slot_id] = selection.model_copy(
                    update={
                        "tier": PatternMatchTier.GUIDANCE_SAFE,
                        "reason": decision.reason,
                    }
                )
                guidance[slot.slot_id] = result.generation_context

        return PatternSlotResolution(
            selections_by_slot=selections,
            guidance_by_slot=guidance,
            reuse_questions_by_slot=reuse_questions,
            retrieval_group_count=len(slot_groups),
            vector_query_count=vector_query_count,
            ignored_pattern_count=ignored_pattern_count,
            warnings=tuple(dict.fromkeys(warnings))[:32],
        )


def build_pattern_context_provider(
    *,
    enabled: bool,
    runtime: PatternRuntimeService | None = None,
) -> PatternContextProvider:
    """Build a no-op unless both practice and canonical runtime flags are enabled."""
    if not enabled:
        return NoOpPatternContextProvider()
    try:
        from config import get_settings  # noqa: PLC0415
        from services.pattern_intelligence_runtime import (  # noqa: PLC0415
            build_pattern_intelligence_runtime,
        )

        settings = get_settings()
        if not settings.pattern_intelligence_enabled:
            return NoOpPatternContextProvider()
        resolved_runtime = runtime or build_pattern_intelligence_runtime(
            settings=settings,
            include_linked_question_references=False,
        )
    except Exception:
        return NoOpPatternContextProvider()
    if resolved_runtime is None:
        return NoOpPatternContextProvider()
    return RuntimePatternContextProvider(
        runtime=resolved_runtime,
        max_candidates=settings.pattern_intelligence_max_candidates,
    )


def _normalize_persisted_selections(
    selections: Mapping[str, PatternSlotSelection | Mapping[str, object]] | None,
) -> dict[str, PatternSlotSelection]:
    if not selections:
        return {}
    normalized: dict[str, PatternSlotSelection] = {}
    for slot_id, value in selections.items():
        if not isinstance(slot_id, str) or not slot_id:
            continue
        try:
            selection = (
                value
                if isinstance(value, PatternSlotSelection)
                else PatternSlotSelection.model_validate(value)
            )
        except ValidationError:
            continue
        if selection.tier is PatternMatchTier.GUIDANCE_SAFE:
            normalized[slot_id] = selection
    return normalized


def _group_slots(
    slots: Sequence[PlannerSlot],
    persisted_selections: Mapping[str, PatternSlotSelection],
) -> tuple[tuple[str, tuple[PlannerSlot, ...]], ...]:
    """Group compatible slots; exact re-entry groups only server-selected IDs."""
    grouped: dict[str, list[PlannerSlot]] = defaultdict(list)
    for slot in slots:
        persisted = persisted_selections.get(slot.slot_id)
        compatibility_key = _slot_compatibility_key(slot)
        if persisted is not None:
            key = f"pattern:{persisted.pattern_id}|{compatibility_key}"
        else:
            key = f"vector|{compatibility_key}"
        grouped[key].append(slot)
    return tuple(
        (key, tuple(grouped[key]))
        for key in sorted(grouped)
    )


def _slot_compatibility_key(slot: PlannerSlot) -> str:
    """Keep exact re-entry and vector grouping equally conservative."""
    exam_ids = ",".join(
        sorted({exam_id.strip() for exam_id in slot.exam_ids if exam_id.strip()})
    )
    return "|".join(
        (
            slot.subject_id,
            slot.topic_id,
            slot.category_id,
            slot.difficulty.value,
            slot.complexity.value,
            exam_ids,
            slot.question_type.value,
            slot.target_skill,
            slot.reasoning_target or "",
            slot.generator_route_hint,
            slot.pattern_family_id or "",
        )
    )


def _build_runtime_request(
    *,
    request: PracticeGenerationRequest,
    slots: Sequence[PlannerSlot],
    max_candidates: int,
    server_pattern_id: str | None,
    server_pattern_version_hash: str | None,
    excluded_question_ids: Sequence[str],
    seen_question_ids: Sequence[str],
    student_history_checked: bool,
) -> PatternRuntimeRequest:
    representative = slots[0]
    exam_ids = tuple(
        sorted(
            {
                *(
                    exam_id.strip()
                    for slot in slots
                    for exam_id in slot.exam_ids
                    if exam_id.strip()
                ),
                *([request.exam_id] if request.exam_id else []),
            }
        )
    )
    target_skills = ", ".join(
        sorted({slot.target_skill.replace("_", " ") for slot in slots})[:4]
    )
    reasoning_targets = ", ".join(
        sorted(
            {
                slot.reasoning_target.replace("_", " ")
                for slot in slots
                if slot.reasoning_target
            }
        )[:4]
    )
    excluded_conditions = tuple(
        sorted(
            {
                condition.strip()
                for slot in slots
                for condition in slot.not_same_when
                if condition.strip()
            }
        )
    )
    required_operation_ids = tuple(
        sorted(
            {
                slot.reasoning_target.strip()
                for slot in slots
                if slot.reasoning_target and slot.reasoning_target.strip()
            }
        )
    )
    query = " | ".join(
        value
        for value in (
            f"subject: {representative.subject_id}",
            f"topic: {representative.topic_id.replace('_', ' ')}",
            f"category: {representative.category_id.replace('_', ' ')}",
            f"target skills: {target_skills}" if target_skills else "",
            f"reasoning targets: {reasoning_targets}" if reasoning_targets else "",
            f"question type: {representative.question_type.value}",
            f"difficulty: {representative.difficulty.value}",
            f"complexity: {representative.complexity.value}",
        )
        if value
    )
    return PatternRuntimeRequest(
        # Keep the externally validated request id intact; appending a slot suffix
        # can exceed the shared runtime's 128-character safety bound.
        requestId=request.request_id,
        query=query,
        subject=representative.subject_id,
        topic=representative.topic_id.replace("_", " "),
        # These values are accepted only on exact canonical parity. This deliberately
        # ignores Patterns whose source contract cannot prove the slot compatibility.
        category=representative.category_id,
        difficulty=representative.complexity.value,
        examIds=exam_ids,
        questionType=representative.question_type.value,
        patternFamilyId=representative.pattern_family_id,
        excludedConditions=excluded_conditions,
        requiredOperationIds=required_operation_ids,
        requiredTargetIds=required_operation_ids,
        serverPatternId=server_pattern_id,
        serverPatternVersionHash=server_pattern_version_hash,
        candidateLimit=max_candidates,
        maxLinkedQuestions=1,
        language=request.language,
        excludedQuestionIds=tuple(dict.fromkeys(excluded_question_ids))[:100],
        seenQuestionIds=tuple(dict.fromkeys(seen_question_ids))[:100],
        studentHistoryChecked=student_history_checked,
    )
