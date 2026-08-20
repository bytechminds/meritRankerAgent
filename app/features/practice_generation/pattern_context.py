"""Feature-gated, server-side PatternGraph guidance for practice deficits."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from features.practice_generation.config import PracticeConfigurationError
from features.practice_generation.events import emit_practice_event
from features.practice_generation.execution_control import current_practice_execution_id
from features.practice_generation.schemas import PlannerSlot, PracticeGenerationRequest
from observability import current_execution_context
from retrieval.pattern_intelligence import (
    CanonicalPlayableQuestion,
    PatternGenerationContext,
    PatternMatchTier,
    PatternRuntimeMemo,
    PatternRuntimeRequest,
    PatternRuntimeService,
)

logger = logging.getLogger(__name__)
# Only these reason codes indicate a version mismatch, not a genuinely stale/valid
# Pattern; derived from PatternCompatibilityGuard/_resolve's own reason strings.
_VERSION_INVALID_REASONS = frozenset(
    {"pattern_version_unavailable", "pattern_version_mismatch", "pattern_not_found"}
)


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
    reuse_questions_by_slot: dict[str, CanonicalPlayableQuestion] = Field(default_factory=dict)
    retrieval_group_count: int = Field(default=0, ge=0, le=100)
    vector_query_count: int = Field(default=0, ge=0, le=100)
    expansion_query_count: int = Field(default=0, ge=0, le=100)
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
        expansion_query_count = 0
        ignored_pattern_count = 0
        warnings: list[str] = []
        # One question may satisfy only one slot across the whole assessment.
        allocated_question_ids: list[str] = []
        operation_context = current_execution_context()
        operation_test_id = (operation_context.activity_id if operation_context else None) or ""
        execution_id = current_practice_execution_id() or ""

        for group_index, (group_key, group_slots) in enumerate(slot_groups):
            group_id = f"group-{group_index:03d}"
            representative = group_slots[0]
            emit_practice_event(
                "PATTERN_RETRIEVAL_STARTED",
                test_id=operation_test_id,
                status="started",
                details={
                    "groupId": group_id,
                    "executionId": execution_id,
                    "subject": representative.subject_id,
                    "topic": representative.topic_id,
                    "difficulty": representative.difficulty.value,
                    "requiredCount": len(group_slots),
                    "candidateLimit": self._max_candidates,
                },
            )
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
                excluded_question_ids=(*excluded_question_ids, *allocated_question_ids),
                seen_question_ids=seen_question_ids,
                student_history_checked=student_history_checked,
            )
            # Snapshotting the shared memo lets duplicate work be observed exactly,
            # without adding any counter to PatternRuntimeService/models.py.
            candidate_ids_before = set(memo.candidate_ids)
            playable_before = set(memo.playable_questions)
            raw_stats_before = set(memo.raw_candidate_stats)
            started_at = time.monotonic()
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
                emit_practice_event(
                    "PATTERN_VECTOR_QUERY_COMPLETED",
                    test_id=operation_test_id,
                    status="unavailable",
                    details={
                        "groupId": group_id,
                        "durationMs": int((time.monotonic() - started_at) * 1000),
                    },
                )
                continue
            duration_ms = int((time.monotonic() - started_at) * 1000)

            embedding_calls = (
                len(set(memo.candidate_ids) - candidate_ids_before)
                if result.plan.strategy == "vector"
                else 0
            )
            questionbank_probes = len(set(memo.playable_questions) - playable_before)
            new_raw_stats = [
                memo.raw_candidate_stats[key]
                for key in set(memo.raw_candidate_stats) - raw_stats_before
            ]
            vector_result_count = sum(count for count, _ in new_raw_stats)
            top_scores = [score for _, score in new_raw_stats if score is not None]
            top_vector_score = max(top_scores) if top_scores else None
            if result.plan.strategy == "vector":
                vector_query_count += 1
                if result.candidate_expansion_used:
                    expansion_query_count += 1
            warnings.extend(result.warnings)
            ignored_pattern_count += sum(
                decision.tier is PatternMatchTier.IGNORE for decision in result.decisions
            )
            emit_practice_event(
                "PATTERN_VECTOR_QUERY_COMPLETED",
                test_id=operation_test_id,
                status=(
                    "unavailable"
                    if "vector_discovery_unavailable" in result.warnings
                    else "completed"
                ),
                details={
                    "groupId": group_id,
                    "embeddingCallCount": embedding_calls,
                    # A vector strategy that added no new memo entry was served
                    # entirely from a prior identical query in this operation.
                    "embeddingMemoHit": result.plan.strategy == "vector" and embedding_calls == 0,
                    # Not "vectorQueryCount": the sanitizer's raw-query-text
                    # denylist matches "query" as a substring and would silently
                    # drop this safe integer field.
                    "vectorSearchCount": embedding_calls,
                    # candidateCount is POST confidence-floor (== strongCandidateCount);
                    # vectorResultCount is what S3 Vector actually returned, before
                    # filtering. Keeping candidateCount's existing meaning avoids
                    # breaking prior consumers. Not "rawVectorResultCount"/"topRawScore":
                    # the sanitizer's raw-payload denylist matches "raw" as a substring.
                    "candidateCount": len(result.decisions),
                    "vectorResultCount": vector_result_count,
                    "strongCandidateCount": len(result.decisions),
                    **(
                        {"topVectorScore": top_vector_score} if top_vector_score is not None else {}
                    ),
                    "candidateLimit": result.plan.candidate_limit,
                    "expansionUsed": result.candidate_expansion_used,
                    "durationMs": duration_ms,
                },
            )
            for rank, candidate_decision in enumerate(result.decisions, start=1):
                emit_practice_event(
                    "PATTERN_CANDIDATE_DECISION",
                    test_id=operation_test_id,
                    status="evaluated",
                    details={
                        "groupId": group_id,
                        "rank": rank,
                        "patternId": candidate_decision.pattern_id,
                        "tier": candidate_decision.tier.value,
                        "versionValid": candidate_decision.reason not in _VERSION_INVALID_REASONS,
                        "reasonCode": candidate_decision.reason,
                    },
                )

            if (
                result.tier not in {PatternMatchTier.GUIDANCE_SAFE, PatternMatchTier.REUSE_SAFE}
                or result.selected_pattern_id is None
                or result.selected_pattern_version_hash is None
                or result.generation_context is None
            ):
                emit_practice_event(
                    "PATTERN_RETRIEVAL_COMPLETED",
                    test_id=operation_test_id,
                    status="completed",
                    details={
                        "groupId": group_id,
                        "candidatesEvaluated": len(result.decisions),
                        "ignoredCount": sum(
                            d.tier is PatternMatchTier.IGNORE for d in result.decisions
                        ),
                        "directReuseCount": 0,
                        "generationDeficit": len(group_slots),
                        "embeddingCalls": embedding_calls,
                        "vectorQueries": embedding_calls,
                        "questionbankProbes": questionbank_probes,
                        "durationMs": duration_ms,
                    },
                )
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
            # Every slot in a group shares one compatibility key, so allocating reuse
            # to a prefix of the group cannot alter the planner's distribution.
            reusable = (
                result.reuse_questions[: len(group_slots)]
                if result.tier is PatternMatchTier.REUSE_SAFE
                else ()
            )
            for question in reusable:
                emit_practice_event(
                    "PATTERN_LINKED_QUESTION_DECISION",
                    test_id=operation_test_id,
                    status="evaluated",
                    details={
                        "groupId": group_id,
                        "patternId": question.pattern_id,
                        "questionId": question.question_id,
                        # Already excluded by seen/attempted history before selection.
                        "studentSeen": False,
                        "directReuseEligible": True,
                        "purpose": "DIRECT_REUSE",
                        "reasonCode": selection.reason,
                    },
                )
            for index, slot in enumerate(group_slots):
                if index < len(reusable):
                    selections[slot.slot_id] = selection
                    reuse_questions[slot.slot_id] = reusable[index]
                    allocated_question_ids.append(reusable[index].question_id)
                    continue
                selections[slot.slot_id] = selection.model_copy(
                    update={
                        "tier": PatternMatchTier.GUIDANCE_SAFE,
                        "reason": decision.reason,
                    }
                )
                guidance[slot.slot_id] = result.generation_context
            emit_practice_event(
                "PATTERN_RETRIEVAL_COMPLETED",
                test_id=operation_test_id,
                status="completed",
                details={
                    "groupId": group_id,
                    "candidatesEvaluated": len(result.decisions),
                    "guidanceSafeCount": sum(
                        d.tier is PatternMatchTier.GUIDANCE_SAFE for d in result.decisions
                    ),
                    "ignoredCount": sum(
                        d.tier is PatternMatchTier.IGNORE for d in result.decisions
                    ),
                    "linkedQuestionsLoaded": len(reusable),
                    "directReuseCount": len(reusable),
                    "generationDeficit": len(group_slots) - len(reusable),
                    "embeddingCalls": embedding_calls,
                    "vectorQueries": embedding_calls,
                    "questionbankProbes": questionbank_probes,
                    "durationMs": duration_ms,
                },
            )

        return PatternSlotResolution(
            selections_by_slot=selections,
            guidance_by_slot=guidance,
            reuse_questions_by_slot=reuse_questions,
            retrieval_group_count=len(slot_groups),
            vector_query_count=vector_query_count,
            expansion_query_count=expansion_query_count,
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
    except PracticeConfigurationError:
        # Pattern Intelligence was explicitly enabled but its mandatory resource
        # identity could not be resolved. Failing closed to NoOp here would hide a
        # real misconfiguration behind silently degraded generation.
        raise
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
    return tuple((key, tuple(grouped[key])) for key in sorted(grouped))


def _slot_compatibility_key(slot: PlannerSlot) -> str:
    """Keep exact re-entry and vector grouping equally conservative."""
    exam_ids = ",".join(sorted({exam_id.strip() for exam_id in slot.exam_ids if exam_id.strip()}))
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
    target_skills = ", ".join(sorted({slot.target_skill.replace("_", " ") for slot in slots})[:4])
    reasoning_targets = ", ".join(
        sorted(
            {slot.reasoning_target.replace("_", " ") for slot in slots if slot.reasoning_target}
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
        reuseTarget=len(slots),
        maxLinkedQuestions=1,
        language=request.language,
        excludedQuestionIds=tuple(dict.fromkeys(excluded_question_ids))[:100],
        seenQuestionIds=tuple(dict.fromkeys(seen_question_ids))[:100],
        studentHistoryChecked=student_history_checked,
    )
