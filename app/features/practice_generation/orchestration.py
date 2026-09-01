"""DynamoDB-backed practice-generation orchestration used by the graph."""

from __future__ import annotations

import functools
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from features.practice_generation.config import PracticeGenerationConfig
from features.practice_generation.events import emit_practice_event
from features.practice_generation.execution_control import (
    PracticeExecutionStopped,
    current_practice_execution_id,
)
from features.practice_generation.generation import (
    QuestionGenerator,
    QuestionVerifier,
    bucket_for_slot,
    build_generation_groups,
    build_slot_generation_groups,
    deterministic_question_id,
    parse_partial_generation,
    validate_final_set,
    verification_required,
)
from features.practice_generation.matching import (
    ReusableQuestion,
    build_reuse_bucket_key,
    build_reuse_difficulty_prefix,
    build_slot_reuse_bucket_key,
    match_existing_questions,
    match_existing_questions_to_slots,
    normalize_question_identity,
    reusable_question_from_item,
    shortlist_reuse_candidate_ids,
)
from features.practice_generation.pattern_context import (
    PatternContextProvider,
    PatternSlotResolution,
    PatternSlotSelection,
)
from features.practice_generation.planning import (
    BlueprintManager,
    BlueprintPlanningError,
    PlannerValidationDiagnostic,
    apply_system_bucket_policy,
    planner_tier,
    select_planner_family,
)
from features.practice_generation.progress import AppSyncAssessmentProgressRepository
from features.practice_generation.question_semantic_reuse import (
    QuestionSemanticReuseResolver,
    group_semantic_demands,
)
from features.practice_generation.repositories import (
    AssessmentRepository,
    PracticeRepositoryError,
    QuestionRepository,
)
from features.practice_generation.schemas import (
    DemandBucket,
    GeneratedQuestion,
    GenerationGroup,
    InternalPhase,
    PlannerSlot,
    PracticeBlueprint,
    PracticeGenerationRequest,
    VerificationDecision,
    VerificationResult,
)
from observability import bind_execution_context
from retrieval.pattern_intelligence import (
    CanonicalPlayableQuestion,
    PatternGenerationContext,
    PatternMatchTier,
)
from services.llm.orchestration.errors import ProviderExecutionError
from services.llm.providers.errors import FALLBACK_ELIGIBLE_FAILURE_KINDS

logger = logging.getLogger(__name__)


def _pattern_reuse_question(value: CanonicalPlayableQuestion) -> ReusableQuestion:
    return ReusableQuestion(
        question_id=value.question_id,
        question=value.question,
        options=value.options,
        correct_answer=value.correct_answer,
        solution=value.explanation,
        subject=value.subject,
        topic=value.topic,
        difficulty=value.difficulty.casefold(),
        question_type=value.question_type.casefold(),
        language=value.language.casefold(),
        source="QuestionBank.Pattern",
        source_updated_at=value.updated_at or None,
        category=value.category,
        exam_ids=value.exam_ids,
        pattern_family_id=value.pattern_family_id or None,
        confidence=1.0,
    )


@dataclass(frozen=True)
class _SlotGenerationContext:
    test_id: str
    request: PracticeGenerationRequest
    blueprint: PracticeBlueprint
    group: GenerationGroup
    bucket: DemandBucket
    slots: tuple[PlannerSlot, ...]
    excluded_texts: tuple[str, ...]
    pattern_guidance_by_slot: dict[str, PatternGenerationContext]
    pattern_selections_by_slot: dict[str, PatternSlotSelection]


@dataclass(frozen=True)
class _VerifiedSlotQuestion:
    slot: PlannerSlot
    question: GeneratedQuestion
    verification: VerificationResult


@dataclass(frozen=True)
class _SlotGenerationOutcome:
    context: _SlotGenerationContext
    accepted: tuple[_VerifiedSlotQuestion, ...]
    unresolved_slot_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    route_id: str
    model: str
    replacement_wave_count: int
    terminal_rejection: bool = False
    provider_failure_recoverable: bool = False
    provider_failure_stage: str | None = None
    cancelled: bool = False


def _meta(assessment: dict[str, Any]) -> dict[str, Any]:
    value = assessment.get("meta")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _request(assessment: dict[str, Any]) -> PracticeGenerationRequest:
    meta = _meta(assessment)
    value = meta.get("practiceRequest")
    if not isinstance(value, dict):
        raise PracticeRepositoryError("PRACTICE_REQUEST_META_MISSING")
    mixed_difficulty_requested = bool(value.get("mixedDifficultyRequested"))
    explicit_difficulty_requested = bool(value.get("explicitDifficultyRequested"))
    return PracticeGenerationRequest.model_validate(
        {
            "request_id": value.get("requestId"),
            "user_id": assessment.get("userId"),
            "conversation_id": value.get("conversationId"),
            "turn_id": value.get("turnId"),
            "original_query": (
                value.get("querySummary")
                or (
                    "Mixed difficulty practice request"
                    if mixed_difficulty_requested
                    else "Practice generation request"
                )
            ),
            "practice_type": value.get("practiceType"),
            "requested_count": value.get("requestedCount"),
            "accepted_count": value.get("acceptedCount"),
            "subject": value.get("subject"),
            "topic": value.get("topic"),
            "topics": value.get("topics"),
            "difficulty": value.get("difficulty"),
            "mixed_difficulty_requested": mixed_difficulty_requested,
            "explicit_difficulty_requested": explicit_difficulty_requested,
            "difficulty_distribution": value.get("difficultyDistribution"),
            "language": value.get("language"),
            "language_source": value.get("languageSource", "REQUEST"),
            "exam_id": value.get("examId"),
            "exam_stage": value.get("examStage"),
            "exam_profile_id": value.get("examProfileId"),
            "source_question_reference": value.get("sourceQuestionReference"),
            "requires_fresh_evidence": bool(value.get("requiresFreshEvidence")),
            "freshness_reason": value.get("freshnessReason"),
            "fresh_evidence": value.get("freshEvidence"),
            "include_solutions": value.get("includeSolutions", True),
            "assessment_title": value.get("assessmentTitle") or assessment.get("name"),
        }
    )


def _pattern_selections_for_slots(
    value: object,
    slot_ids: list[str],
) -> dict[str, PatternSlotSelection]:
    if not isinstance(value, dict):
        return {}
    selections: dict[str, PatternSlotSelection] = {}
    allowed_slot_ids = set(slot_ids)
    for slot_id, raw_selection in value.items():
        if not isinstance(slot_id, str) or slot_id not in allowed_slot_ids:
            continue
        if not isinstance(raw_selection, dict):
            continue
        try:
            selections[slot_id] = PatternSlotSelection.model_validate(raw_selection)
        except ValueError:
            continue
    return selections


class PracticeGenerationOrchestrator:
    def __init__(
        self,
        *,
        config: PracticeGenerationConfig,
        assessments: AssessmentRepository,
        progress: AppSyncAssessmentProgressRepository,
        questions: QuestionRepository,
        blueprint_manager: BlueprintManager,
        generator: QuestionGenerator,
        verifier: QuestionVerifier,
        pattern_context: PatternContextProvider,
        semantic_resolver: QuestionSemanticReuseResolver | None = None,
    ) -> None:
        self._config = config
        self._assessments = assessments
        self._progress_updates = progress
        self._questions = questions
        self._blueprints = blueprint_manager
        self._generator = generator
        self._verifier = verifier
        self._pattern_context = pattern_context
        # Absent unless Phase D is configured; "off" never constructs one.
        self._semantic_resolver = semantic_resolver

    def _require_expensive_work_allowed(self, test_id: str) -> None:
        if current_practice_execution_id() is None:
            return
        self._progress_updates.require_expensive_work_allowed(test_id)

    def _resolve_pattern_slots(
        self,
        *,
        request: PracticeGenerationRequest,
        slots: tuple[PlannerSlot, ...],
        persisted_selections: dict[str, PatternSlotSelection] | None = None,
        excluded_question_ids: tuple[str, ...] = (),
        seen_question_ids: tuple[str, ...] = (),
        student_history_checked: bool = False,
    ) -> PatternSlotResolution:
        if not slots:
            return PatternSlotResolution()
        try:
            return self._pattern_context.resolve_slots(
                request=request,
                slots=slots,
                persisted_selections=persisted_selections,
                excluded_question_ids=excluded_question_ids,
                seen_question_ids=seen_question_ids,
                student_history_checked=student_history_checked,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pattern guidance unavailable test_id=%s error_type=%s",
                request.request_id,
                type(exc).__name__,
            )
            return PatternSlotResolution()

    def _mark_failed(
        self,
        test_id: str,
        reason_code: str,
        *,
        meta_updates: dict[str, Any] | None = None,
    ) -> None:
        self._progress_updates.mark_failed(
            test_id,
            reason_code,
            meta_updates=meta_updates,
        )
        emit_practice_event(
            "practice_failed",
            test_id=test_id,
            status="failed",
            details={"reasonCode": reason_code},
            level=logging.ERROR,
        )
        emit_practice_event(
            "ASSESSMENT_FAILED",
            test_id=test_id,
            status="failed",
            details={"reasonCode": reason_code},
            level=logging.ERROR,
        )
        emit_practice_event(
            "practice_failed_published",
            test_id=test_id,
            status="failed",
            details={"reasonCode": reason_code},
            level=logging.ERROR,
        )
        emit_practice_event(
            "practice_generation_terminal",
            test_id=test_id,
            status="failed",
            details={"reasonCode": reason_code, "businessStatus": "FAILED"},
            level=logging.ERROR,
        )

    @staticmethod
    def _planner_event_base(
        request: PracticeGenerationRequest,
        *,
        fallback_invoked: bool,
        fallback_result: str,
    ) -> dict[str, Any]:
        tier = planner_tier(request)
        planner_difficulty = {
            "light": "basic",
            "standard": "intermediate",
            "strong": "advanced",
        }[tier]
        return {
            "expectedSlotCount": request.accepted_count,
            "routeId": f"{select_planner_family(request).value}.planner.{planner_difficulty}",
            "plannerTier": tier,
            "fallbackInvoked": fallback_invoked,
            "fallbackResult": fallback_result,
        }

    def _emit_planner_validation_diagnostics(
        self,
        *,
        test_id: str,
        request: PracticeGenerationRequest,
        diagnostics: tuple[PlannerValidationDiagnostic, ...],
        fallback_invoked: bool,
        fallback_result: str,
    ) -> None:
        base = self._planner_event_base(
            request,
            fallback_invoked=fallback_invoked,
            fallback_result=fallback_result,
        )
        for diagnostic in diagnostics:
            # Field ORDER is load-bearing: the operation log renders only the
            # first eight detail fields, so the diagnostics that identify the
            # failing invariant must lead. Emitted before the base context, which
            # previously consumed five of the eight slots and pushed fieldPaths
            # and errorTypes past the cut.
            details = {
                "reasonCode": diagnostic.reason_code,
                "expectedSlotCount": base.get("expectedSlotCount"),
                "actualSlotCount": diagnostic.actual_slot_count,
                "plannerAttempt": diagnostic.attempt,
                "validationStage": diagnostic.validation_stage,
                # Joined, not lists: the sanitizer replaces any non-scalar with its
                # type name, which logged the literal string "list".
                "errorTypes": ",".join(diagnostic.error_types)[:256],
                "fieldPaths": ",".join(diagnostic.field_paths)[:512],
                "validationErrorCount": diagnostic.error_count,
                **{key: value for key, value in base.items() if key != "expectedSlotCount"},
                "plannerPhase": diagnostic.phase,
                "schemaName": diagnostic.schema_name,
                "durationMs": diagnostic.duration_ms,
            }
            emit_practice_event(
                "planner_validation_failed",
                test_id=test_id,
                status="failed",
                details=details,
                level=logging.WARNING,
            )
            if diagnostic.phase == "initial" and len(diagnostics) > 1:
                emit_practice_event(
                    "planner_repair_started",
                    test_id=test_id,
                    status="started",
                    details={
                        **base,
                        "reasonCode": diagnostic.reason_code,
                        "plannerAttempt": diagnostic.attempt + 1,
                        "plannerPhase": "repair",
                    },
                )

    def plan_and_fill(self, test_id: str) -> None:
        assessment = self._assessments.get(test_id)
        if assessment is None:
            raise PracticeRepositoryError("ASSESSMENT_NOT_FOUND")
        if assessment.get("status") in {"READY", "FAILED", "ARCHIVED"}:
            return
        request = _request(assessment)
        meta = _meta(assessment)
        blueprint_payload = meta.get("blueprint")
        existing_groups = meta.get("generationGroups")
        plan_meta: dict[str, Any] = {}
        if isinstance(blueprint_payload, dict) and isinstance(existing_groups, dict):
            return
        if isinstance(blueprint_payload, dict):
            blueprint = apply_system_bucket_policy(
                PracticeBlueprint.model_validate(blueprint_payload),
                request,
            )
        else:
            emit_practice_event(
                "BLUEPRINT_STARTED",
                test_id=test_id,
                status="started",
                details={"phase": InternalPhase.PLANNING.value},
            )
            try:
                self._require_expensive_work_allowed(test_id)
                plan = self._blueprints.build(request)
            except PracticeExecutionStopped:
                raise
            except BlueprintPlanningError as exc:
                fallback_result = exc.fallback_result or "not_invoked"
                self._emit_planner_validation_diagnostics(
                    test_id=test_id,
                    request=request,
                    diagnostics=exc.diagnostics,
                    fallback_invoked=exc.fallback_invoked,
                    fallback_result=fallback_result,
                )
                if exc.fallback_invoked:
                    emit_practice_event(
                        "planner_fallback_failed",
                        test_id=test_id,
                        status="failed",
                        details={
                            **self._planner_event_base(
                                request,
                                fallback_invoked=True,
                                fallback_result=fallback_result,
                            ),
                            "reasonCode": exc.reason_code,
                        },
                        level=logging.ERROR,
                    )
                self._mark_failed(test_id, exc.reason_code)
                return
            blueprint = plan.blueprint
            fallback_result = "valid" if plan.deterministic_fallback else "not_invoked"
            self._emit_planner_validation_diagnostics(
                test_id=test_id,
                request=request,
                diagnostics=plan.validation_diagnostics,
                fallback_invoked=plan.deterministic_fallback,
                fallback_result=fallback_result,
            )
            planner_event_details = self._planner_event_base(
                request,
                fallback_invoked=plan.deterministic_fallback,
                fallback_result=fallback_result,
            )
            if plan.deterministic_fallback:
                emit_practice_event(
                    "planner_fallback_completed",
                    test_id=test_id,
                    status="completed",
                    details={
                        **planner_event_details,
                        "reasonCode": plan.validation_reason_code or "PLANNER_FALLBACK",
                    },
                )
            elif plan.repaired:
                emit_practice_event(
                    "planner_repair_completed",
                    test_id=test_id,
                    status="completed",
                    details={
                        **planner_event_details,
                        "reasonCode": plan.validation_reason_code or "PLANNER_SCHEMA_VALID",
                    },
                )
            plan_meta = {
                "blueprint": blueprint.model_dump(mode="json"),
                "plannerCalls": plan.planner_calls,
                "plannerTier": plan.tier,
                "plannerRepaired": plan.repaired,
                "plannerDeterministicFallback": plan.deterministic_fallback,
            }
            emit_practice_event(
                "BLUEPRINT_COMPLETED",
                test_id=test_id,
                status="completed",
                details={
                    "phase": InternalPhase.MATCHING_EXISTING.value,
                    "plannerCalls": plan.planner_calls,
                    "plannerTier": plan.tier,
                    "bucketCount": len(blueprint.buckets),
                    "repaired": plan.repaired,
                    "deterministicFallback": plan.deterministic_fallback,
                },
            )
            if plan.repaired and not plan.deterministic_fallback:
                emit_practice_event(
                    "BLUEPRINT_REPAIRED",
                    test_id=test_id,
                    status="completed",
                    details={
                        "plannerCalls": plan.planner_calls,
                        "plannerTier": plan.tier,
                    },
                )

        emit_practice_event(
            "EXISTING_MATCH_STARTED",
            test_id=test_id,
            status="started",
            details={"phase": InternalPhase.MATCHING_EXISTING.value},
        )
        if blueprint.schema_version == "2":
            self._plan_and_fill_slots(
                test_id,
                request=request,
                blueprint=blueprint,
                plan_meta=plan_meta,
            )
            return
        linked = self._questions.list_linked(test_id)
        already_linked_source_ids = {
            str(item.get("_practiceMeta", {}).get("sourceQuestionBankId") or "") for item in linked
        }
        selected_ids: list[str] = []
        selected_id_set = set(already_linked_source_ids)
        total_candidates = 0
        for bucket in blueprint.buckets:
            remaining_total = max(
                self._config.question_bank_max_total_candidates - total_candidates,
                0,
            )
            if remaining_total == 0:
                emit_practice_event(
                    "QUESTION_BANK_REUSE_LIMIT_REACHED",
                    test_id=test_id,
                    status="bounded",
                    details={
                        "bucketId": bucket.bucket_id,
                        "reasonCode": "ASSESSMENT_CANDIDATE_LIMIT_REACHED",
                    },
                )
                break
            bucket_limit = min(
                self._config.question_bank_max_candidates_per_bucket,
                remaining_total,
            )
            reuse_bucket_key = build_reuse_bucket_key(
                category=bucket.subject,
                topic=bucket.topic,
                question_type=bucket.question_type.value,
                language=request.language,
            )
            emit_practice_event(
                "QUESTION_BANK_REUSE_QUERY_STARTED",
                test_id=test_id,
                status="started",
                details={
                    "logicalTable": "QuestionBank",
                    "logicalIndex": "QuestionBank.reuse",
                    "bucketId": bucket.bucket_id,
                    "pageCount": 0,
                    "candidateCount": 0,
                    "selectedCount": 0,
                    "hasMorePages": False,
                    "durationMs": 0,
                    "consumedCapacity": None,
                },
            )
            bucket_blueprint = PracticeBlueprint(
                practice_type=blueprint.practice_type,
                accepted_count=bucket.required_count,
                buckets=[bucket],
            )
            bucket_candidates: list[dict[str, Any]] = []
            bucket_selected: list[str] = []
            bucket_evaluated = 0
            page_count = 0
            duration_ms = 0
            consumed_capacity = 0.0
            capacity_reported = False
            continuation_key: dict[str, Any] | None = None
            has_more_pages = True
            while (
                len(bucket_selected) < bucket.required_count
                and page_count < self._config.question_bank_max_pages
                and bucket_evaluated < bucket_limit
                and total_candidates < self._config.question_bank_max_total_candidates
                and has_more_pages
            ):
                page_limit = min(
                    self._config.question_bank_query_page_size,
                    bucket_limit - bucket_evaluated,
                    self._config.question_bank_max_total_candidates - total_candidates,
                )
                query_result = self._questions.query_topic_reuse_candidates(
                    reuse_bucket_key=reuse_bucket_key,
                    limit=page_limit,
                    exclusive_start_key=continuation_key,
                )
                evaluated = max(
                    query_result.evaluated_count,
                    len(query_result.items),
                )
                total_candidates += evaluated
                bucket_evaluated += evaluated
                page_count += query_result.page_count
                duration_ms += query_result.duration_ms
                if query_result.consumed_capacity is not None:
                    consumed_capacity += query_result.consumed_capacity
                    capacity_reported = True
                bucket_candidates.extend(query_result.items)
                bucket_selected = shortlist_reuse_candidate_ids(
                    bucket_blueprint,
                    tuple(bucket_candidates),
                    requested_language=request.language,
                    already_linked_ids=selected_id_set,
                )
                continuation_key = query_result.continuation_key
                has_more_pages = query_result.has_more_pages
                if evaluated == 0:
                    break
            if len(bucket_selected) < bucket.required_count:
                fallback_remaining = min(
                    bucket_limit - bucket_evaluated,
                    self._config.question_bank_max_total_candidates - total_candidates,
                )
                if fallback_remaining > 0 and page_count < self._config.question_bank_max_pages:
                    emit_practice_event(
                        "QUESTION_BANK_CATEGORY_FALLBACK",
                        test_id=test_id,
                        status="started",
                        details={
                            "logicalIndex": "QuestionBank.category",
                            "bucketId": bucket.bucket_id,
                            "unresolvedDeficit": (bucket.required_count - len(bucket_selected)),
                        },
                    )
                    continuation_key = None
                    has_more_pages = True
                    while (
                        len(bucket_selected) < bucket.required_count
                        and page_count < self._config.question_bank_max_pages
                        and fallback_remaining > 0
                        and has_more_pages
                    ):
                        page_limit = min(
                            self._config.question_bank_query_page_size,
                            fallback_remaining,
                        )
                        fallback = self._questions.query_reuse_candidates(
                            category=bucket.subject,
                            limit=page_limit,
                            exclusive_start_key=continuation_key,
                        )
                        evaluated = max(
                            fallback.evaluated_count,
                            len(fallback.items),
                        )
                        total_candidates += evaluated
                        bucket_evaluated += evaluated
                        fallback_remaining -= evaluated
                        page_count += fallback.page_count
                        duration_ms += fallback.duration_ms
                        if fallback.consumed_capacity is not None:
                            consumed_capacity += fallback.consumed_capacity
                            capacity_reported = True
                        bucket_candidates.extend(fallback.items)
                        bucket_selected = shortlist_reuse_candidate_ids(
                            bucket_blueprint,
                            tuple(bucket_candidates),
                            requested_language=request.language,
                            already_linked_ids=selected_id_set,
                        )
                        continuation_key = fallback.continuation_key
                        has_more_pages = fallback.has_more_pages
                        if evaluated == 0:
                            break
            selected_ids.extend(bucket_selected)
            selected_id_set.update(bucket_selected)
            emit_practice_event(
                "QUESTION_BANK_REUSE_QUERY_COMPLETED",
                test_id=test_id,
                status="completed",
                details={
                    "logicalTable": "QuestionBank",
                    "logicalIndex": "QuestionBank.reuse",
                    "bucketId": bucket.bucket_id,
                    "pageCount": page_count,
                    "candidateCount": bucket_evaluated,
                    "selectedCount": len(bucket_selected),
                    "hasMorePages": has_more_pages,
                    "durationMs": duration_ms,
                    "consumedCapacity": (consumed_capacity if capacity_reported else None),
                },
            )
        raw_candidates = self._questions.get_reuse_candidates(selected_ids)
        candidates = [
            candidate
            for item in raw_candidates
            if (
                candidate := reusable_question_from_item(
                    item,
                    requested_language=request.language,
                )
            )
            is not None
        ]
        matches = match_existing_questions(
            blueprint,
            candidates,
            already_linked_ids=already_linked_source_ids,
        )
        accepted_reuse_ids: list[str] = []
        for match in matches:
            for candidate in match.selected:
                linked_now = self._questions.link_reused(
                    test_id=test_id,
                    bucket_id=match.bucket.bucket_id,
                    question=candidate,
                )
                if linked_now:
                    accepted_reuse_ids.append(
                        deterministic_question_id(
                            test_id,
                            source_id=candidate.question_id,
                            bucket_id=match.bucket.bucket_id,
                        )
                    )
                    emit_practice_event(
                        "QUESTION_LINKED",
                        test_id=test_id,
                        status="linked",
                        details={
                            "bucketId": match.bucket.bucket_id,
                            "source": "REUSED",
                        },
                    )
        refreshed_meta = self._progress_updates.calculate_authoritative_meta(
            test_id,
            meta_updates={
                **plan_meta,
                "phase": InternalPhase.MATCHING_EXISTING.value,
                "progressPercent": 5,
                "bucketReadyCounts": {bucket.bucket_id: 0 for bucket in blueprint.buckets},
            },
            authoritative_question_ids=tuple(accepted_reuse_ids),
        )
        counts = {
            str(key): int(value)
            for key, value in dict(refreshed_meta.get("bucketReadyCounts") or {}).items()
        }
        deficits = {
            bucket.bucket_id: max(
                bucket.required_count - counts.get(bucket.bucket_id, 0),
                0,
            )
            for bucket in blueprint.buckets
        }
        groups = build_generation_groups(
            blueprint,
            deficits,
            group_size=self._config.generation_group_size,
            group_max=self._config.generation_group_max,
        )
        generation_groups = {
            group.group_id: {
                "groupId": group.group_id,
                "bucketId": group.bucket_id,
                "requiredCount": group.required_count,
                "tokenBudget": group.token_budget,
                "attempt": group.attempt,
                "state": "PENDING",
            }
            for group in groups
        }
        reused_count = int(refreshed_meta.get("reusedCount") or 0)
        ready_count = int(refreshed_meta.get("readyQuestionCount") or 0)
        progress = self._progress(ready_count, request.accepted_count)
        self._progress_updates.update(
            test_id,
            meta_updates={
                **plan_meta,
                "phase": (
                    InternalPhase.FINALIZING.value if not groups else InternalPhase.GENERATING.value
                ),
                "progressPercent": progress,
                "deficits": deficits,
                "generationGroups": generation_groups,
            },
            live=False,
            recalculate_manifest=True,
            authoritative_question_ids=tuple(accepted_reuse_ids),
        )
        emit_practice_event(
            "EXISTING_MATCH_COMPLETED",
            test_id=test_id,
            status="completed",
            details={
                "reusedCount": reused_count,
                "deficitCount": sum(deficits.values()),
            },
        )
        emit_practice_event(
            "practice_reuse_completed",
            test_id=test_id,
            status="completed",
            details={
                "reusedCount": reused_count,
                "deficitCount": sum(deficits.values()),
            },
        )
        emit_practice_event(
            "DEFICIT_CALCULATED",
            test_id=test_id,
            status="completed",
            details={
                "reusedCount": reused_count,
                "deficitCount": sum(deficits.values()),
                "groupCount": len(groups),
            },
        )
        if not groups:
            self._finalize(test_id, request, blueprint, attempt=0)
            return
        for group in groups:
            emit_practice_event(
                "GENERATION_GROUP_CREATED",
                test_id=test_id,
                status="created",
                details={
                    "groupId": group.group_id,
                    "bucketId": group.bucket_id,
                    "requiredCount": group.required_count,
                },
            )

    def _plan_and_fill_slots(
        self,
        test_id: str,
        *,
        request: PracticeGenerationRequest,
        blueprint: PracticeBlueprint,
        plan_meta: dict[str, Any],
    ) -> None:
        """Fill schema-v2 slots by exact indexed reuse, then persist exact deficits."""
        linked = self._questions.list_linked(test_id)
        linked_meta = [
            item.get("_practiceMeta")
            if isinstance(item.get("_practiceMeta"), dict)
            else _meta({"meta": item.get("meta")})
            for item in linked
        ]
        filled_slot_ids = {
            str(value.get("slotId") or "") for value in linked_meta if value.get("slotId")
        }
        already_linked_source_ids = {
            str(value.get("sourceQuestionBankId") or "")
            for value in linked_meta
            if value.get("sourceQuestionBankId")
        }
        candidate_ids: list[str] = []
        lane_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
        total_evaluated = 0
        for slot in blueprint.slots:
            if slot.slot_id in filled_slot_ids:
                continue
            reuse_key = build_slot_reuse_bucket_key(slot, language=request.language)
            difficulty_prefix = build_reuse_difficulty_prefix(slot.difficulty.value)
            if reuse_key is None or difficulty_prefix is None:
                continue
            lane = (reuse_key, difficulty_prefix)
            if lane not in lane_cache:
                emit_practice_event(
                    "QUESTION_BANK_REUSE_QUERY_STARTED",
                    test_id=test_id,
                    status="started",
                    details={
                        "logicalTable": "QuestionBank",
                        "logicalIndex": "QuestionBank.reuse",
                        "operation": "schema_v2_slot_reuse",
                        "candidateCount": 0,
                        "selectedCount": 0,
                    },
                )
                lane_items: list[dict[str, Any]] = []
                continuation: dict[str, Any] | None = None
                for _page in range(self._config.question_bank_max_pages):
                    remaining = min(
                        self._config.question_bank_max_candidates_per_bucket,
                        self._config.question_bank_max_total_candidates - total_evaluated,
                    )
                    if remaining <= 0:
                        break
                    result = self._questions.query_topic_reuse_candidates(
                        reuse_bucket_key=reuse_key,
                        difficulty_prefix=difficulty_prefix,
                        limit=min(self._config.question_bank_query_page_size, remaining),
                        exclusive_start_key=continuation,
                    )
                    lane_items.extend(result.items)
                    evaluated = max(result.evaluated_count, len(result.items))
                    total_evaluated += evaluated
                    continuation = result.continuation_key
                    if not result.has_more_pages or evaluated == 0:
                        break
                lane_cache[lane] = lane_items
                # DEBUG only: the completed event reports candidateCount without the
                # educational bucket it queried, so a zero result cannot be told apart
                # from "inventory exists under a different topic label".
                emit_practice_event(
                    "QUESTION_BANK_REUSE_QUERY_DEBUG",
                    test_id=test_id,
                    status="completed",
                    details={
                        "slotId": slot.slot_id,
                        "reuseBucketKey": reuse_key,
                        "difficultyPrefix": difficulty_prefix,
                        "returnedCount": len(lane_items),
                    },
                    level=logging.DEBUG,
                )
                emit_practice_event(
                    "QUESTION_BANK_REUSE_QUERY_COMPLETED",
                    test_id=test_id,
                    status="completed",
                    details={
                        "logicalTable": "QuestionBank",
                        "logicalIndex": "QuestionBank.reuse",
                        "operation": "schema_v2_slot_reuse",
                        "candidateCount": len(lane_items),
                        "selectedCount": 0,
                    },
                )
            for item in lane_cache[lane]:
                question_id = str(item.get("qbId") or "")
                if question_id and question_id not in already_linked_source_ids:
                    candidate_ids.append(question_id)

        raw_candidates = self._questions.get_reuse_candidates(
            list(dict.fromkeys(candidate_ids))
        )
        candidates = [
            candidate
            for item in raw_candidates
            if (
                candidate := reusable_question_from_item(
                    item,
                    requested_language=request.language,
                )
            )
            is not None
        ]
        matches = match_existing_questions_to_slots(
            blueprint,
            candidates,
            requested_language=request.language,
            already_linked_ids=already_linked_source_ids,
        )
        accepted_ids: list[str] = []
        for match in matches:
            if match.slot.slot_id in filled_slot_ids or match.selected is None:
                continue
            bucket = next(
                bucket
                for bucket in blueprint.buckets
                if bucket.subject == match.slot.subject_id
                and bucket.topic == match.slot.topic_id
                and bucket.difficulty is match.slot.difficulty
                and bucket.question_type is match.slot.question_type
            )
            if self._questions.link_reused(
                test_id=test_id,
                bucket_id=bucket.bucket_id,
                slot_id=match.slot.slot_id,
                question=match.selected,
            ):
                filled_slot_ids.add(match.slot.slot_id)
                accepted_ids.append(
                    deterministic_question_id(
                        test_id,
                        source_id=match.selected.question_id,
                        bucket_id=bucket.bucket_id,
                    )
                )

        deficit_slot_ids = {
            slot.slot_id for slot in blueprint.slots if slot.slot_id not in filled_slot_ids
        }
        seen_question_ids: set[str] = set()
        student_history_checked = False
        if self._config.pattern_reuse_enabled:
            try:
                seen_question_ids = self._questions.list_recent_seen_question_bank_ids(
                    user_id=request.user_id
                )
                student_history_checked = True
            except PracticeRepositoryError:
                emit_practice_event(
                    "PATTERN_REUSE_HISTORY_UNAVAILABLE",
                    test_id=test_id,
                    status="fallback",
                    details={"reasonCode": "PRACTICE_HISTORY_READ_FAILED"},
                    level=logging.WARNING,
                )
        # Phase D: between exact reuse and Pattern, over remaining deficits only.
        deficit_slot_ids = self._apply_question_semantic_reuse(
            test_id=test_id,
            request=request,
            blueprint=blueprint,
            deficit_slot_ids=deficit_slot_ids,
            filled_slot_ids=filled_slot_ids,
            already_linked_source_ids=already_linked_source_ids,
            accepted_ids=accepted_ids,
        )
        pattern_runtime_enabled = (
            self._config.pattern_context_enabled or self._config.pattern_reuse_enabled
        )
        pattern_resolution = PatternSlotResolution()
        if pattern_runtime_enabled:
            pattern_resolution = self._resolve_pattern_slots(
                request=request,
                slots=tuple(
                    slot for slot in blueprint.slots if slot.slot_id in deficit_slot_ids
                ),
                excluded_question_ids=tuple(sorted(already_linked_source_ids)),
                seen_question_ids=tuple(sorted(seen_question_ids)),
                student_history_checked=student_history_checked,
            )
        slots_by_id = {slot.slot_id: slot for slot in blueprint.slots}
        pattern_reused_count = 0
        for slot_id, playable in pattern_resolution.reuse_questions_by_slot.items():
            if slot_id not in deficit_slot_ids:
                continue
            slot = slots_by_id[slot_id]
            bucket = next(
                value
                for value in blueprint.buckets
                if value.subject == slot.subject_id
                and value.topic == slot.topic_id
                and value.difficulty is slot.difficulty
                and value.question_type is slot.question_type
            )
            reusable = _pattern_reuse_question(playable)
            selection = pattern_resolution.selections_by_slot.get(slot_id)
            trusted_pattern_selection = (
                selection
                if selection is not None
                and selection.tier is PatternMatchTier.REUSE_SAFE
                else None
            )
            if self._questions.link_reused(
                test_id=test_id,
                bucket_id=bucket.bucket_id,
                slot_id=slot_id,
                question=reusable,
                trusted_pattern_selection=trusted_pattern_selection,
            ):
                filled_slot_ids.add(slot_id)
                deficit_slot_ids.remove(slot_id)
                already_linked_source_ids.add(reusable.question_id)
                pattern_reused_count += 1
                accepted_ids.append(
                    deterministic_question_id(
                        test_id,
                        source_id=reusable.question_id,
                        bucket_id=bucket.bucket_id,
                    )
                )
        pattern_context_keys = {
            slot_id: selection.pattern_id
            for slot_id, selection in pattern_resolution.selections_by_slot.items()
            if slot_id in deficit_slot_ids
            and selection.tier is PatternMatchTier.GUIDANCE_SAFE
        }
        groups = build_slot_generation_groups(
            blueprint,
            deficit_slot_ids,
            group_size=self._config.generation_group_size,
            group_max=self._config.generation_group_max,
            pattern_context_keys=pattern_context_keys,
        )
        generation_groups = {
            group.group_id: {
                "groupId": group.group_id,
                "bucketId": group.bucket_id,
                "requiredCount": group.required_count,
                "slotIds": list(group.slot_ids),
                "tokenBudget": group.token_budget,
                "attempt": 0,
                "replacementWave": 0,
                "state": "PENDING",
                "patternSelections": {
                    slot_id: pattern_resolution.selections_by_slot[slot_id].model_dump(
                        by_alias=True
                    )
                    for slot_id in group.slot_ids
                    if slot_id in pattern_resolution.selections_by_slot
                },
            }
            for group in groups
        }
        slot_ready_counts = {
            slot.slot_id: int(slot.slot_id in filled_slot_ids) for slot in blueprint.slots
        }
        progress_updates: dict[str, Any] = {
            **plan_meta,
            "phase": (
                InternalPhase.FINALIZING.value
                if not groups
                else InternalPhase.GENERATING.value
            ),
            "progressPercent": self._progress(
                len(filled_slot_ids),
                request.accepted_count,
            ),
            "deficits": {slot_id: 1 for slot_id in sorted(deficit_slot_ids)},
            "slotReadyCounts": slot_ready_counts,
            "generationGroups": generation_groups,
            "lastCompletedStage": "SLOT_REUSE_MATCHED",
        }
        self._progress_updates.update(
            test_id,
            meta_updates=progress_updates,
            live=False,
            recalculate_manifest=True,
            authoritative_question_ids=tuple(accepted_ids),
        )
        emit_practice_event(
            "EXISTING_MATCH_COMPLETED",
            test_id=test_id,
            status="completed",
            details={
                "reusedCount": len(filled_slot_ids),
                "deficitCount": len(deficit_slot_ids),
            },
        )
        if pattern_runtime_enabled:
            emit_practice_event(
                "PATTERN_RETRIEVAL_COMPLETED",
                test_id=test_id,
                status="completed",
                details={
                    "retrievalGroupCount": pattern_resolution.retrieval_group_count,
                    "vectorQueryCount": pattern_resolution.vector_query_count,
                    "expansionQueryCount": pattern_resolution.expansion_query_count,
                    "guidanceSafeCount": len(pattern_resolution.guidance_by_slot),
                    "ignoredPatternCount": pattern_resolution.ignored_pattern_count,
                    "reuseSafeCount": pattern_reused_count,
                },
            )
        emit_practice_event(
            "DEFICIT_CALCULATED",
            test_id=test_id,
            status="completed",
            details={
                "reusedCount": len(filled_slot_ids),
                "deficitCount": len(deficit_slot_ids),
                "groupCount": len(groups),
            },
        )
        if not groups:
            self._finalize(test_id, request, blueprint, attempt=0)
            return
        for group in groups:
            emit_practice_event(
                "GENERATION_GROUP_CREATED",
                test_id=test_id,
                status="created",
                details={
                    "groupId": group.group_id,
                    "bucketId": group.bucket_id,
                    "requiredCount": group.required_count,
                },
            )

    def generate_group(self, test_id: str, group_id: str) -> None:
        assessment = self._assessments.get(test_id)
        if assessment is None or assessment.get("status") in {"READY", "FAILED", "ARCHIVED"}:
            return
        request = _request(assessment)
        claimed = self._assessments.claim_group(test_id, group_id)
        if claimed is None:
            return
        assessment = self._assessments.get(test_id)
        if assessment is None or assessment.get("status") != "GENERATING":
            return
        meta = _meta(assessment)
        groups = deepcopy(dict(meta.get("generationGroups") or {}))
        persisted_group = groups.get(group_id)
        if not isinstance(persisted_group, dict) or persisted_group.get("state") != "RUNNING":
            return
        claimed = persisted_group
        blueprint = apply_system_bucket_policy(
            PracticeBlueprint.model_validate(meta.get("blueprint")),
            request,
        )
        bucket_id = str(claimed.get("bucketId") or "")
        bucket = next(
            (value for value in blueprint.buckets if value.bucket_id == bucket_id),
            None,
        )
        if bucket is None:
            self._mark_failed(test_id, "GENERATION_BUCKET_UNKNOWN")
            return
        attempt = int(claimed.get("attempt") or 0)
        attempt_stage = str(claimed.get("attemptStage") or "GROUP")
        replacement = bool(claimed.get("replacement"))
        current_count = int(dict(meta.get("bucketReadyCounts") or {}).get(bucket_id, 0))
        deficit = max(bucket.required_count - current_count, 0)
        requested = min(int(claimed.get("requiredCount") or 0), deficit)
        if requested <= 0:
            groups[group_id]["state"] = "COMPLETED"
            self._update_progress(
                test_id,
                request,
                meta_updates={"generationGroups": groups},
            )
            self._maybe_finalize(test_id, request, blueprint)
            return

        emit_practice_event(
            "GENERATION_GROUP_STARTED",
            test_id=test_id,
            status="started",
            details={
                "groupId": group_id,
                "bucketId": bucket_id,
                "attempt": attempt,
                "requiredCount": requested,
            },
        )
        linked = self._questions.list_linked(test_id)
        existing_texts = {
            normalize_question_identity(
                str(item.get("question") or ""), item.get("options")
            )
            for item in linked
        }
        from features.practice_generation.schemas import GenerationGroup  # noqa: PLC0415

        group = GenerationGroup(
            group_id=group_id,
            bucket_id=bucket_id,
            required_count=requested,
            attempt=attempt,
        )
        try:
            self._require_expensive_work_allowed(test_id)
            batch = self._generator.generate(
                request=request,
                bucket=bucket,
                group=group,
                exclude_normalized_texts=tuple(sorted(existing_texts)),
            )
            parsed = parse_partial_generation(
                batch.content,
                group=group,
                bucket=bucket,
                existing_normalized_texts=existing_texts,
                requested_language=request.language,
            )
        except PracticeExecutionStopped:
            raise
        except ProviderExecutionError as exc:
            reason_code = (
                "PRACTICE_GENERATOR_OUTPUT_TOKEN_EXHAUSTED"
                if exc.failure_kind == "output_token_exhausted"
                else "PRACTICE_GENERATOR_PROVIDER_FAILED"
            )
            groups[group_id].update(
                {
                    "state": "FAILED",
                    "requiredCount": requested,
                    "attempt": attempt,
                    "errorCode": reason_code,
                    "attemptStage": "MODEL_FALLBACK",
                    "lastReasonCode": reason_code,
                    "attemptedModelAliases": list(exc.attempted_aliases),
                }
            )
            emit_practice_event(
                "practice_model_execution_failed",
                test_id=test_id,
                status="failed",
                details={
                    "groupId": group_id,
                    "bucketId": bucket_id,
                    "attempt": attempt,
                    "failureClass": exc.failure_kind,
                    "modelAlias": (
                        exc.attempted_aliases[-1] if exc.attempted_aliases else ""
                    ),
                    "remainingCount": requested,
                },
                level=logging.WARNING,
            )
            self._mark_failed(
                test_id,
                reason_code,
                meta_updates={"generationGroups": groups},
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "practice generation failed unexpectedly test_id=%s group_id=%s error_type=%s",
                test_id,
                group_id,
                type(exc).__name__,
            )
            groups[group_id].update(
                {
                    "state": "FAILED",
                    "requiredCount": requested,
                    "attempt": attempt,
                    "errorCode": "PRACTICE_GENERATION_UNEXPECTED_FAILURE",
                    "lastReasonCode": "PRACTICE_GENERATION_UNEXPECTED_FAILURE",
                }
            )
            self._mark_failed(
                test_id,
                "PRACTICE_GENERATION_UNEXPECTED_FAILURE",
                meta_updates={"generationGroups": groups},
            )
            return

        accepted_count = 0
        accepted_question_ids: list[str] = []
        if parsed is not None:
            for reason_code in parsed.rejection_reason_codes:
                emit_practice_event(
                    "question_contract_rejected",
                    test_id=test_id,
                    status="rejected",
                    details={
                        "groupId": group_id,
                        "bucketId": bucket_id,
                        "reasonCode": reason_code,
                    },
                )
            for index, question in enumerate(parsed.accepted):
                emit_practice_event(
                    "question_contract_validation",
                    test_id=test_id,
                    status="validated",
                    details={
                        "groupId": group_id,
                        "bucketId": bucket_id,
                        "reasonCode": "PLAYABLE",
                    },
                )
                approved = True
                reason = "STRUCTURE_VALID"
                required_verification = verification_required(
                    bucket,
                    index,
                    replacement=replacement,
                )
                if required_verification:
                    verifier_error_class: str | None = None
                    try:
                        self._require_expensive_work_allowed(test_id)
                        result = self._verifier.verify(
                            request=request,
                            bucket=bucket,
                            question=question,
                        )
                    except PracticeExecutionStopped:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        # Exception class only: the slot path already records this and
                        # without it here VERIFIER_UNAVAILABLE stays undiagnosable.
                        verifier_error_class = type(exc).__name__
                        result = VerificationResult(
                            generation_item_id=question.generation_item_id,
                            approved=False,
                            reason_code="VERIFIER_UNAVAILABLE",
                        )
                        logger.warning(
                            "practice verifier unavailable test_id=%s group_id=%s error_type=%s",
                            test_id,
                            group_id,
                            type(exc).__name__,
                        )
                    approved = result.approved
                    reason = "VERIFIER_APPROVED" if approved else "VERIFIER_REJECTED"
                    emit_practice_event(
                        "QUESTION_VERIFICATION_RESULT",
                        test_id=test_id,
                        status="approved" if approved else "rejected",
                        details={
                            "groupId": group_id,
                            "bucketId": bucket_id,
                            "reasonCode": reason,
                            **(
                                {"errorClass": verifier_error_class}
                                if verifier_error_class
                                else {}
                            ),
                        },
                    )
                if approved and self._questions.link_generated(
                    test_id=test_id,
                    question=question,
                    verified=True,
                    group_id=group_id,
                    generator_route=batch.route_id,
                    generator_model=batch.model,
                    verification_policy=(
                        "MANDATORY"
                        if replacement and bucket.subject in {"math", "reasoning"}
                        else bucket.verification_policy.value
                    ),
                    verification_method=(
                        "MODEL" if required_verification else "STRUCTURAL"
                    ),
                    language=request.language,
                ):
                    accepted_count += 1
                    accepted_question_ids.append(
                        deterministic_question_id(
                            test_id,
                            source_id=question.generation_item_id,
                            bucket_id=question.bucket_id,
                        )
                    )
                    emit_practice_event(
                        "question_contract_accepted",
                        test_id=test_id,
                        status="accepted",
                        details={
                            "groupId": group_id,
                            "bucketId": bucket_id,
                            "reasonCode": "PLAYABLE",
                        },
                    )
                    emit_practice_event(
                        "QUESTION_LINKED",
                        test_id=test_id,
                        status="linked",
                        details={
                            "groupId": group_id,
                            "bucketId": bucket_id,
                            "source": "GENERATED",
                        },
                    )
                    emit_practice_event(
                        "GENERATION_ITEM_ACCEPTED",
                        test_id=test_id,
                        status="accepted",
                        details={
                            "groupId": group_id,
                            "bucketId": bucket_id,
                            "reasonCode": reason,
                        },
                    )
                else:
                    emit_practice_event(
                        "GENERATION_ITEM_REJECTED",
                        test_id=test_id,
                        status="rejected",
                        details={
                            "groupId": group_id,
                            "bucketId": bucket_id,
                            "reasonCode": reason,
                        },
                    )

        missing = max(requested - accepted_count, 0)
        if missing:
            emit_practice_event(
                "question_repair_requested",
                test_id=test_id,
                status="requested",
                details={
                    "groupId": group_id,
                    "bucketId": bucket_id,
                    "reasonCode": (
                        parsed.rejection_reason_codes[0]
                        if parsed is not None and parsed.rejection_reason_codes
                        else "PARTIAL_GROUP_DEFICIT"
                    ),
                },
            )
            emit_practice_event(
                "practice_validation_repair_started",
                test_id=test_id,
                status="started",
                details={
                    "groupId": group_id,
                    "bucketId": bucket_id,
                    "attempt": attempt,
                    "remainingCount": missing,
                },
            )
        emit_practice_event(
            "practice_validation_completed",
            test_id=test_id,
            status="completed",
            details={
                "groupId": group_id,
                "acceptedCount": accepted_count,
                "deficitCount": missing,
            },
        )
        if missing:
            if attempt_stage == "GROUP":
                retry_group_ids: list[str] = []
                for item_index in range(missing):
                    retry_group_id = f"{group_id}-retry-{item_index + 1}"
                    if retry_group_id not in groups:
                        groups[retry_group_id] = {
                            "groupId": retry_group_id,
                            "bucketId": bucket_id,
                            "requiredCount": 1,
                            "attempt": 1,
                            "attemptStage": "ITEM_RETRY",
                            "itemRetryCount": 1,
                            "replacementCount": 0,
                            "replacement": False,
                            "lastReasonCode": "PARTIAL_GROUP_DEFICIT",
                            "state": "PENDING",
                        }
                    retry_group_ids.append(retry_group_id)
                    emit_practice_event(
                        "GENERATION_ITEM_RETRY",
                        test_id=test_id,
                        status="retry",
                        details={
                            "groupId": retry_group_id,
                            "bucketId": bucket_id,
                            "attempt": 1,
                            "attemptStage": "ITEM_RETRY",
                            "deficitCount": 1,
                        },
                    )
                groups[group_id].update(
                    {
                        "state": "COMPLETED",
                        "requiredCount": requested,
                        "attempt": attempt,
                        "errorCode": "PARTIAL_GROUP_DEFICIT",
                        "attemptStage": "GROUP",
                        "lastReasonCode": "PARTIAL_GROUP_DEFICIT",
                    }
                )
                self._update_progress(
                    test_id,
                    request,
                    authoritative_question_ids=tuple(accepted_question_ids),
                    meta_updates={"generationGroups": groups},
                )
                return
            if attempt_stage == "ITEM_RETRY":
                emit_practice_event(
                    "practice_replacement_started",
                    test_id=test_id,
                    status="started",
                    details={
                        "groupId": group_id,
                        "bucketId": bucket_id,
                        "attempt": 2,
                        "remainingCount": missing,
                    },
                )
                groups[group_id].update(
                    {
                        "state": "PENDING",
                        "requiredCount": 1,
                        "attempt": 2,
                        "errorCode": "ITEM_RETRY_FAILED",
                        "attemptStage": "REPLACEMENT",
                        "itemRetryCount": 1,
                        "replacementCount": 1,
                        "replacement": True,
                        "lastReasonCode": "ITEM_RETRY_FAILED",
                    }
                )
                self._update_progress(
                    test_id,
                    request,
                    authoritative_question_ids=tuple(accepted_question_ids),
                    meta_updates={"generationGroups": groups},
                )
                emit_practice_event(
                    "GENERATION_ITEM_RETRY",
                    test_id=test_id,
                    status="replacement",
                    details={
                        "groupId": group_id,
                        "bucketId": bucket_id,
                        "attempt": 2,
                        "attemptStage": "REPLACEMENT",
                        "deficitCount": 1,
                    },
                )
                return
            groups[group_id].update(
                {
                    "state": "FAILED",
                    "requiredCount": missing,
                    "attempt": attempt,
                    "errorCode": "GENERATION_DEFICIT_EXHAUSTED",
                    "attemptStage": "REPLACEMENT",
                    "itemRetryCount": 1,
                    "replacementCount": 1,
                    "replacement": True,
                    "lastReasonCode": "GENERATION_DEFICIT_EXHAUSTED",
                }
            )
            self._mark_failed(
                test_id,
                "GENERATION_DEFICIT_EXHAUSTED",
                meta_updates={"generationGroups": groups},
            )
            return
        groups[group_id]["state"] = "COMPLETED"
        self._update_progress(
            test_id,
            request,
            authoritative_question_ids=tuple(accepted_question_ids),
            meta_updates={"generationGroups": groups},
        )
        self._maybe_finalize(test_id, request, blueprint)

    def generate_wave(self, test_id: str, group_ids: list[str]) -> None:
        """Run at most two immutable schema-v2 workers; commit only in coordinator."""
        assessment = self._assessments.get(test_id)
        if assessment is None or assessment.get("status") != "GENERATING":
            return
        request = _request(assessment)
        blueprint = apply_system_bucket_policy(
            PracticeBlueprint.model_validate(_meta(assessment).get("blueprint")),
            request,
        )
        if blueprint.schema_version != "2":
            for group_id in group_ids[:1]:
                self.generate_group(test_id, group_id)
            return
        contexts: list[_SlotGenerationContext] = []
        selected_bucket_ids: set[str] = set()
        queued_groups = dict(_meta(assessment).get("generationGroups") or {})
        for group_id in list(dict.fromkeys(group_ids)):
            if len(contexts) >= 2:
                break
            queued_group = queued_groups.get(group_id)
            queued_bucket_id = (
                str(queued_group.get("bucketId") or "")
                if isinstance(queued_group, dict)
                else ""
            )
            # Parallel prompts cannot share an exclusion snapshot. Serialize groups from
            # one canonical bucket so their committed questions become exclusions first.
            if not queued_bucket_id or queued_bucket_id in selected_bucket_ids:
                continue
            if self._progress_updates.claim_group(test_id, group_id) is None:
                continue
            current = self._assessments.get(test_id)
            if current is None or current.get("status") != "GENERATING":
                return
            meta = _meta(current)
            persisted = dict(meta.get("generationGroups") or {}).get(group_id)
            if not isinstance(persisted, dict) or persisted.get("state") != "RUNNING":
                continue
            slot_ids = [str(value) for value in list(persisted.get("slotIds") or [])]
            slots_by_id = {slot.slot_id: slot for slot in blueprint.slots}
            try:
                slots = tuple(slots_by_id[slot_id] for slot_id in slot_ids)
            except KeyError:
                self._mark_failed(test_id, "GENERATION_SLOT_UNKNOWN")
                return
            if not slots:
                self._mark_failed(test_id, "GENERATION_SLOT_MISSING")
                return
            bucket_id = str(persisted.get("bucketId") or "")
            bucket = next(
                (value for value in blueprint.buckets if value.bucket_id == bucket_id),
                None,
            )
            if bucket is None:
                self._mark_failed(test_id, "GENERATION_BUCKET_UNKNOWN")
                return
            linked = self._questions.list_linked(test_id)
            excluded = tuple(
                sorted(
                    normalize_question_identity(
                        str(item.get("question") or ""), item.get("options")
                    )
                    for item in linked
                    if item.get("question")
                )
            )
            persisted_selections = _pattern_selections_for_slots(
                persisted.get("patternSelections"),
                slot_ids,
            )
            pattern_resolution = self._resolve_pattern_slots(
                request=request,
                slots=slots,
                persisted_selections=persisted_selections or None,
            ) if persisted_selections else PatternSlotResolution()
            contexts.append(
                _SlotGenerationContext(
                    test_id=test_id,
                    request=request,
                    blueprint=blueprint,
                    group=GenerationGroup(
                        group_id=group_id,
                        bucket_id=bucket_id,
                        required_count=len(slots),
                        slot_ids=slot_ids,
                    ),
                    bucket=bucket,
                    slots=slots,
                    excluded_texts=excluded,
                    pattern_guidance_by_slot={
                        slot_id: guidance
                        for slot_id, guidance in pattern_resolution.guidance_by_slot.items()
                        if slot_id in slot_ids
                    },
                    pattern_selections_by_slot={
                        slot_id: selection
                        for slot_id, selection in pattern_resolution.selections_by_slot.items()
                        if slot_id in slot_ids
                    },
                )
            )
            selected_bucket_ids.add(bucket_id)
        if not contexts:
            return
        with ThreadPoolExecutor(
            max_workers=min(2, len(contexts)),
            thread_name_prefix="practice-slot-wave",
        ) as executor:
            futures = [
                executor.submit(copy_context().run, self._execute_slot_group, context)
                for context in contexts
            ]
            outcomes = [future.result() for future in futures]
        for outcome in outcomes:
            if not self._commit_slot_outcome(outcome):
                return
        current = self._assessments.get(test_id)
        if current is not None and current.get("status") == "GENERATING":
            self._maybe_finalize(test_id, request, blueprint)

    def _verify_one_slot(
        self,
        *,
        context: _SlotGenerationContext,
        slot: PlannerSlot,
        question: GeneratedQuestion,
    ) -> VerificationResult | BaseException:
        """Run one unchanged verifier call, returning its failure instead of raising.

        Returning the exception keeps every worker terminally resolved, so no task
        is ever abandoned and the caller applies the existing failure policy in a
        single deterministic place.
        """
        try:
            self._require_expensive_work_allowed(context.test_id)
            return self._verifier.verify_slot(
                request=context.request,
                bucket=context.bucket,
                slot=slot,
                question=question,
            )
        except BaseException as exc:  # noqa: BLE001
            return exc

    def _verify_slots_bounded(
        self,
        *,
        context: _SlotGenerationContext,
        units: list[tuple[PlannerSlot, GeneratedQuestion]],
    ) -> dict[str, VerificationResult | BaseException]:
        """Fan the existing per-question verifier calls out under a bounded pool."""
        if not units:
            return {}
        limit = max(1, min(self._config.verification_max_concurrency, len(units)))
        if limit == 1:
            return {
                slot.slot_id: self._verify_one_slot(
                    context=context, slot=slot, question=question
                )
                for slot, question in units
            }
        outcomes: dict[str, VerificationResult | BaseException] = {}
        # Bounded and fully joined: the pool never exceeds `limit` in-flight calls and
        # the context manager waits for every worker before returning.
        with ThreadPoolExecutor(
            max_workers=limit,
            thread_name_prefix="practice-verify",
        ) as executor:
            futures = {
                slot.slot_id: executor.submit(
                    copy_context().run,
                    functools.partial(
                        self._verify_one_slot,
                        context=context,
                        slot=slot,
                        question=question,
                    ),
                )
                for slot, question in units
            }
            for slot_id, future in futures.items():
                try:
                    outcomes[slot_id] = future.result()
                except BaseException as exc:  # noqa: BLE001
                    outcomes[slot_id] = exc
        return outcomes

    def _execute_slot_group(
        self,
        context: _SlotGenerationContext,
    ) -> _SlotGenerationOutcome:
        pending = {slot.slot_id: slot for slot in context.slots}
        accepted: dict[str, _VerifiedSlotQuestion] = {}
        repair_candidates: dict[str, GeneratedQuestion] = {}
        repair_reasons: dict[str, tuple[str, ...]] = {}
        reasons: list[str] = []
        excluded = set(context.excluded_texts)
        route_id = ""
        model = ""
        terminal = False
        provider_failure_recoverable = False
        provider_failure_stage: str | None = None
        completed_wave = 0
        replacement_wave = 0
        cancelled = False
        while replacement_wave <= 2:
            if not pending or terminal:
                break
            try:
                self._require_expensive_work_allowed(context.test_id)
            except PracticeExecutionStopped:
                cancelled = True
                break
            completed_wave = replacement_wave
            force_regeneration = False
            wave_slots = tuple(pending.values())
            wave_group = context.group.model_copy(
                update={
                    "required_count": len(wave_slots),
                    "attempt": replacement_wave,
                    "slot_ids": [slot.slot_id for slot in wave_slots],
                }
            )
            if replacement_wave == 1:
                emit_practice_event(
                    "question_repair_started",
                    test_id=context.test_id,
                    status="started",
                    details={
                        "groupId": context.group.group_id,
                        "repairAttempt": 1,
                        "slotIds": ",".join(slot.slot_id for slot in wave_slots),
                    },
                )
            elif replacement_wave == 2:
                emit_practice_event(
                    "question_replacement_started",
                    test_id=context.test_id,
                    status="started",
                    details={
                        "groupId": context.group.group_id,
                        "replacementAttempt": 1,
                        "slotIds": ",".join(slot.slot_id for slot in wave_slots),
                    },
                )
            emit_practice_event(
                "GENERATION_GROUP_STARTED",
                test_id=context.test_id,
                status="started",
                details={
                    "groupId": context.group.group_id,
                    "batchId": context.group.group_id,
                    "bucketId": context.bucket.bucket_id,
                    "replacementWave": replacement_wave,
                    "requiredCount": len(wave_slots),
                    "slotCount": len(wave_slots),
                    "slotIds": ",".join(slot.slot_id for slot in wave_slots),
                    "routeId": wave_slots[0].generator_route_hint,
                },
            )
            try:
                with bind_execution_context(
                    activity_id=context.test_id,
                    batch_id=context.group.group_id,
                    slot_ids=tuple(slot.slot_id for slot in wave_slots),
                ):
                    batch = self._generator.generate_slots(
                        request=context.request,
                        bucket=context.bucket,
                        group=wave_group,
                        slots=wave_slots,
                        exclude_normalized_texts=tuple(sorted(excluded)),
                        repair_feedback=tuple(reasons[-8:]),
                        repair_candidates_by_slot=repair_candidates,
                        repair_reason_codes_by_slot=repair_reasons,
                        replacement_wave=replacement_wave,
                        pattern_guidance_by_slot={
                            slot.slot_id: context.pattern_guidance_by_slot[slot.slot_id]
                            for slot in wave_slots
                            if slot.slot_id in context.pattern_guidance_by_slot
                        },
                    )
                route_id = batch.route_id
                model = batch.model
                wave_excluded = set(excluded)
                parsed = parse_partial_generation(
                    batch.content,
                    group=wave_group,
                    bucket=context.bucket,
                    existing_normalized_texts=wave_excluded,
                    slots=wave_slots,
                    requested_language=context.request.language,
                )
            except PracticeExecutionStopped:
                cancelled = True
                break
            except ProviderExecutionError as exc:
                provider_failure_recoverable = (
                    exc.failure_kind in FALLBACK_ELIGIBLE_FAILURE_KINDS
                )
                reason_code = (
                    "PRACTICE_GENERATOR_FALLBACK_EXHAUSTED"
                    if provider_failure_recoverable
                    else "PRACTICE_GENERATOR_PROVIDER_FAILED"
                )
                reasons.append(reason_code)
                emit_practice_event(
                    "practice_model_execution_failed",
                    test_id=context.test_id,
                    status="failed",
                    details={
                        "groupId": context.group.group_id,
                        "batchId": context.group.group_id,
                        "bucketId": context.bucket.bucket_id,
                        "replacementWave": replacement_wave,
                        "reasonCode": reason_code,
                        "errorClass": type(exc).__name__,
                        "errorCode": exc.failure_kind,
                        "fallbackEligible": provider_failure_recoverable,
                        "retryEligible": provider_failure_recoverable,
                        "modelAlias": (
                            exc.attempted_aliases[-1] if exc.attempted_aliases else ""
                        ),
                        "routeId": wave_slots[0].generator_route_hint,
                        "slotCount": len(wave_slots),
                        "slotIds": ",".join(slot.slot_id for slot in wave_slots),
                    },
                    level=logging.WARNING,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "slot generation wave failed test_id=%s group_id=%s error_type=%s",
                    context.test_id,
                    context.group.group_id,
                    type(exc).__name__,
                )
                reasons.append("PRACTICE_GENERATION_UNEXPECTED_FAILURE")
                emit_practice_event(
                    "practice_model_execution_failed",
                    test_id=context.test_id,
                    status="failed",
                    details={
                        "groupId": context.group.group_id,
                        "bucketId": context.bucket.bucket_id,
                        "replacementWave": replacement_wave,
                        "reasonCode": "PRACTICE_GENERATION_UNEXPECTED_FAILURE",
                    },
                    level=logging.WARNING,
                )
                break
            for reason_code in parsed.rejection_reason_codes:
                emit_practice_event(
                    "question_contract_rejected",
                    test_id=context.test_id,
                    status="rejected",
                    details={
                        "groupId": context.group.group_id,
                        "bucketId": context.bucket.bucket_id,
                        "replacementWave": replacement_wave,
                        "reasonCode": reason_code,
                    },
                )
            reasons.extend(parsed.rejection_reason_codes)
            if parsed.rejection_reason_codes:
                for slot in wave_slots:
                    repair_reasons.setdefault(
                        slot.slot_id,
                        tuple(parsed.rejection_reason_codes[:4]),
                    )
            verification_units: list[tuple[PlannerSlot, GeneratedQuestion]] = []
            for question in parsed.accepted:
                slot = pending.get(str(question.slot_id or ""))
                if slot is None:
                    reasons.append("VERIFIER_SLOT_BINDING_MISMATCH")
                    continue
                verification_units.append((slot, question))
            # Transport-only fan-out: each question still gets its own verifier call
            # with the same route, model, and prompt. Outcomes are keyed by slot_id and
            # merged below in the original deterministic order, so completion order can
            # never determine identity or event ordering.
            verification_outcomes = self._verify_slots_bounded(
                context=context,
                units=verification_units,
            )
            for slot, question in verification_units:
                outcome = verification_outcomes.get(slot.slot_id)
                if isinstance(outcome, PracticeExecutionStopped):
                    cancelled = True
                    break
                if isinstance(outcome, ProviderExecutionError):
                    exc = outcome
                    provider_failure_recoverable = (
                        exc.failure_kind in FALLBACK_ELIGIBLE_FAILURE_KINDS
                    )
                    provider_failure_stage = "VERIFIER"
                    reason_code = (
                        "PRACTICE_VERIFIER_FALLBACK_EXHAUSTED"
                        if provider_failure_recoverable
                        else "PRACTICE_VERIFIER_PROVIDER_FAILED"
                    )
                    reasons.append(reason_code)
                    emit_practice_event(
                        "practice_model_execution_failed",
                        test_id=context.test_id,
                        status="failed",
                        details={
                            "groupId": context.group.group_id,
                            "batchId": context.group.group_id,
                            "bucketId": context.bucket.bucket_id,
                            "replacementWave": replacement_wave,
                            "reasonCode": reason_code,
                            "errorClass": type(exc).__name__,
                            "errorCode": exc.failure_kind,
                            "failureStage": provider_failure_stage,
                            "fallbackEligible": provider_failure_recoverable,
                            "retryEligible": provider_failure_recoverable,
                            "modelAlias": (
                                exc.attempted_aliases[-1] if exc.attempted_aliases else ""
                            ),
                            "slotCount": len(wave_slots),
                            "slotIds": ",".join(
                                candidate.slot_id for candidate in wave_slots
                            ),
                        },
                        level=logging.WARNING,
                    )
                    break
                if isinstance(outcome, BaseException) or outcome is None:
                    logger.warning(
                        "slot verifier unavailable test_id=%s group_id=%s error_type=%s",
                        context.test_id,
                        context.group.group_id,
                        type(outcome).__name__,
                    )
                    reasons.append("VERIFIER_UNAVAILABLE")
                    emit_practice_event(
                        "QUESTION_VERIFICATION_RESULT",
                        test_id=context.test_id,
                        status="rejected",
                        details={
                            "groupId": context.group.group_id,
                            "bucketId": context.bucket.bucket_id,
                            "slotId": slot.slot_id,
                            "replacementWave": replacement_wave,
                            "reasonCode": "VERIFIER_UNAVAILABLE",
                            # Exception class only: without it VERIFIER_UNAVAILABLE
                            # cannot be told apart from provider capacity limits.
                            "errorClass": type(outcome).__name__,
                        },
                    )
                    # An unavailable verifier is infrastructure failure, not a
                    # content defect: a newly generated question cannot repair it.
                    # Leave the content replacement waves alone, and stay
                    # non-recoverable so the outer commit does not schedule a
                    # PROVIDER_REPLACEMENT group and regenerate content either.
                    # The already-generated candidate is not carried past this
                    # return, so re-verifying it is not possible here.
                    provider_failure_recoverable = False
                    provider_failure_stage = "VERIFIER"
                    terminal = False
                    break
                verification = outcome
                binding_valid = (
                    verification.schema_version == "2"
                    and verification.generation_item_id == question.generation_item_id
                    and verification.slot_id == slot.slot_id
                )
                # The authority never saw the author's proposal, so this is a genuine
                # two-source agreement rather than a confirmation. Exactly one option
                # may be valid: agreement on a single id says nothing about whether a
                # second id is also correct, which is why the count is gated first.
                # Comparison is by option id only — never by answer text, whose
                # representation ("48" vs "forty-eight") varies without changing meaning.
                valid_option_ids = verification.valid_option_ids
                if len(valid_option_ids) != 1:
                    gate_reason = (
                        "NO_VALID_OPTION"
                        if not valid_option_ids
                        else "MULTIPLE_VALID_OPTIONS"
                    )
                elif valid_option_ids[0] != question.correct_option_id:
                    gate_reason = "AUTHOR_AUTHORITY_MISMATCH"
                else:
                    gate_reason = None
                answer_valid = gate_reason is None
                if (
                    binding_valid
                    and answer_valid
                    and verification.is_approved
                ):
                    accepted[slot.slot_id] = _VerifiedSlotQuestion(
                        slot=slot,
                        question=question,
                        verification=verification,
                    )
                    excluded.add(
                        normalize_question_identity(question.question, question.options)
                    )
                    pending.pop(slot.slot_id, None)
                    if replacement_wave == 1:
                        emit_practice_event(
                            "question_repair_completed",
                            test_id=context.test_id,
                            status="completed",
                            details={
                                "groupId": context.group.group_id,
                                "repairAttempt": 1,
                                "slotId": slot.slot_id,
                            },
                        )
                    elif replacement_wave == 2:
                        emit_practice_event(
                            "question_replacement_completed",
                            test_id=context.test_id,
                            status="completed",
                            details={
                                "groupId": context.group.group_id,
                                "replacementAttempt": 1,
                                "slotId": slot.slot_id,
                            },
                        )
                    emit_practice_event(
                        "QUESTION_VERIFICATION_RESULT",
                        test_id=context.test_id,
                        status="approved",
                        details={
                            "groupId": context.group.group_id,
                            "bucketId": context.bucket.bucket_id,
                            "slotId": slot.slot_id,
                            "replacementWave": replacement_wave,
                            "reasonCode": "VERIFIER_APPROVED",
                        },
                    )
                    continue
                # A deterministic gate failure is the real reason for rejection, so it
                # leads the codes: the authority may have returned ACCEPT while the
                # option count or the author comparison is what actually failed.
                gate_reason_codes = (
                    [gate_reason] if gate_reason is not None else []
                )
                if not binding_valid:
                    gate_reason_codes.append("BINDING_INVALID")
                combined_reason_codes = [
                    *gate_reason_codes,
                    *verification.reason_codes,
                ]
                reasons.extend(combined_reason_codes)
                repair_candidates[slot.slot_id] = question
                repair_reasons[slot.slot_id] = tuple(
                    combined_reason_codes[:4]
                    or ["VERIFIER_REJECTED"]
                )
                emit_practice_event(
                    "QUESTION_VERIFICATION_RESULT",
                    test_id=context.test_id,
                    status="rejected",
                    details={
                        "groupId": context.group.group_id,
                        "bucketId": context.bucket.bucket_id,
                        "slotId": slot.slot_id,
                        "replacementWave": replacement_wave,
                        "reasonCode": (
                            combined_reason_codes[0]
                            if combined_reason_codes
                            else "VERIFIER_REJECTED"
                        ),
                    },
                )
                if verification.decision is VerificationDecision.TERMINAL_REJECTION:
                    terminal = True
                    break
                if (
                    verification.decision is VerificationDecision.REGENERATE
                    or gate_reason is not None
                ):
                    force_regeneration = True
            if provider_failure_stage is not None:
                break
            if cancelled:
                break
            if pending and not terminal:
                if force_regeneration:
                    excluded.update(
                        normalize_question_identity(
                            repair_candidates[slot_id].question,
                            repair_candidates[slot_id].options,
                        )
                        for slot_id in pending
                        if slot_id in repair_candidates
                    )
                if replacement_wave == 1:
                    excluded.update(
                        normalize_question_identity(
                            repair_candidates[slot_id].question,
                            repair_candidates[slot_id].options,
                        )
                        for slot_id in pending
                        if slot_id in repair_candidates
                    )
                    emit_practice_event(
                        "question_repair_failed",
                        test_id=context.test_id,
                        status="failed",
                        details={
                            "groupId": context.group.group_id,
                            "repairAttempt": 1,
                            "reasonCode": reasons[-1] if reasons else "REPAIR_REJECTED",
                            "slotIds": ",".join(pending),
                        },
                    )
                replacement_wave = 2 if force_regeneration else replacement_wave + 1
        emit_practice_event(
            "practice_validation_completed",
            test_id=context.test_id,
            status="completed",
            details={
                "groupId": context.group.group_id,
                "acceptedCount": len(accepted),
                "remainingCount": len(pending),
                "reasonCode": reasons[-1] if reasons else "VERIFIED",
            },
        )
        return _SlotGenerationOutcome(
            context=context,
            accepted=tuple(accepted.values()),
            unresolved_slot_ids=tuple(pending),
            reason_codes=tuple(dict.fromkeys(reasons)),
            route_id=route_id,
            model=model,
            replacement_wave_count=completed_wave,
            terminal_rejection=terminal,
            provider_failure_recoverable=provider_failure_recoverable,
            provider_failure_stage=provider_failure_stage,
            cancelled=cancelled,
        )

    def _apply_question_semantic_reuse(
        self,
        *,
        test_id: str,
        request: PracticeGenerationRequest,
        blueprint: PracticeBlueprint,
        deficit_slot_ids: set[str],
        filled_slot_ids: set[str],
        already_linked_source_ids: set[str],
        accepted_ids: list[str],
    ) -> set[str]:
        """Fill remaining deficits from questions-v1, or leave them for Pattern.

        In "shadow" mode the full retrieval and authoritative validation run and are
        reported, but nothing is served and no deficit is reduced.
        """
        mode = self._config.question_semantic_reuse_mode
        if mode == "off" or not deficit_slot_ids or self._semantic_resolver is None:
            return deficit_slot_ids
        slots_by_id = {slot.slot_id: slot for slot in blueprint.slots}
        pending = tuple(
            slots_by_id[slot_id] for slot_id in sorted(deficit_slot_ids) if slot_id in slots_by_id
        )
        if not pending:
            return deficit_slot_ids
        groups = group_semantic_demands(pending, language=request.language)
        outcome = self._semantic_resolver.resolve(
            test_id=test_id,
            groups=groups,
            language=request.language,
            excluded_question_ids=set(already_linked_source_ids),
        )
        served = 0
        if mode == "on":
            for slot_id, candidate in outcome.selected_by_slot.items():
                if slot_id not in deficit_slot_ids:
                    continue
                slot = slots_by_id[slot_id]
                bucket = bucket_for_slot(blueprint, slot)
                if not self._questions.link_reused(
                    test_id=test_id,
                    bucket_id=bucket.bucket_id,
                    slot_id=slot_id,
                    question=candidate,
                ):
                    continue
                filled_slot_ids.add(slot_id)
                deficit_slot_ids.discard(slot_id)
                already_linked_source_ids.add(candidate.question_id)
                accepted_ids.append(
                    deterministic_question_id(
                        test_id,
                        source_id=candidate.question_id,
                        bucket_id=bucket.bucket_id,
                    )
                )
                served += 1
        emit_practice_event(
            "QUESTION_SEMANTIC_REUSE_COMPLETED",
            test_id=test_id,
            status="completed",
            details={
                "semanticMode": mode,
                "semanticDemandGroupCount": outcome.group_count,
                "embeddingCallCount": outcome.embedding_call_count,
                "semanticSearchCount": outcome.semantic_search_count,
                "vectorApiCallCount": outcome.vector_api_call_count,
                "nearDuplicateRejectedCount": outcome.near_duplicate_rejected_count,
                "candidateCount": outcome.candidate_count,
                "hydratedCount": outcome.hydrated_count,
                "versionParityRejectedCount": outcome.version_parity_rejected_count,
                "identityRejectedCount": outcome.identity_rejected_count,
                "trustRejectedCount": outcome.trust_rejected_count,
                "compatibilityRejectedCount": outcome.compatibility_rejected_count,
                "duplicateRejectedCount": outcome.duplicate_rejected_count,
                "thresholdRejectedCount": outcome.threshold_rejected_count,
                "wouldReuseCount": outcome.would_reuse_count,
                "selectedCount": served,
                "remainingDeficitCount": len(deficit_slot_ids),
            },
        )
        return deficit_slot_ids

    def _commit_slot_outcome(self, outcome: _SlotGenerationOutcome) -> bool:
        test_id = outcome.context.test_id
        current = self._assessments.get(test_id)
        if current is None or current.get("status") != "GENERATING":
            return False
        if outcome.terminal_rejection:
            self._mark_failed(test_id, "VERIFIER_TERMINAL_REJECTION")
            return False
        meta = _meta(current)
        groups = deepcopy(dict(meta.get("generationGroups") or {}))
        group = groups.get(outcome.context.group.group_id)
        if not isinstance(group, dict) or group.get("state") != "RUNNING":
            return False
        existing_slot_ids = {
            str(
                (
                    item.get("_practiceMeta")
                    if isinstance(item.get("_practiceMeta"), dict)
                    else _meta({"meta": item.get("meta")})
                ).get("slotId")
                or ""
            )
            for item in self._questions.list_linked(test_id)
        }
        accepted_question_ids: list[str] = []
        for verified in outcome.accepted:
            if verified.slot.slot_id in existing_slot_ids:
                continue
            source_question_bank_id: str | None = None
            selection = outcome.context.pattern_selections_by_slot.get(
                verified.slot.slot_id
            )
            if self._config.question_bank_promotion_enabled:
                try:
                    source_question_bank_id = (
                        self._questions.persist_verified_question(
                            test_id=test_id,
                            question=verified.question,
                            slot=verified.slot,
                            language=outcome.context.request.language,
                            pattern_id=(
                                selection.pattern_id if selection is not None else None
                            ),
                            pattern_version_hash=(
                                selection.pattern_version_hash
                                if selection is not None
                                else None
                            ),
                        )
                    )
                except PracticeRepositoryError:
                    emit_practice_event(
                        "PATTERN_QUESTION_BANK_LINK_FAILED",
                        test_id=test_id,
                        status="failed",
                        details={
                            "slotId": verified.slot.slot_id,
                            "reasonCode": "QUESTION_BANK_PATTERN_LINK_FAILED",
                        },
                        level=logging.WARNING,
                    )
            linked = self._questions.link_generated(
                test_id=test_id,
                question=verified.question,
                verified=True,
                group_id=outcome.context.group.group_id,
                generator_route=outcome.route_id,
                generator_model=outcome.model,
                verification_policy="MANDATORY",
                verification_method="INDEPENDENT_MODEL_V2",
                language=outcome.context.request.language,
                source_question_bank_id=source_question_bank_id,
                pattern_selection=selection,
            )
            if linked:
                existing_slot_ids.add(verified.slot.slot_id)
                accepted_question_ids.append(
                    deterministic_question_id(
                        test_id,
                        source_id=verified.slot.slot_id,
                        bucket_id=verified.question.bucket_id,
                    )
                )
        if outcome.cancelled:
            if accepted_question_ids:
                self._update_progress(
                    test_id,
                    outcome.context.request,
                    authoritative_question_ids=tuple(accepted_question_ids),
                    meta_updates={"lastCompletedStage": "CANCEL_CHECKPOINTED"},
                )
            raise PracticeExecutionStopped("PRACTICE_CANCEL_OBSERVED")
        unresolved = [
            slot_id
            for slot_id in outcome.unresolved_slot_ids
            if slot_id not in existing_slot_ids
        ]
        slot_ready_counts = dict(meta.get("slotReadyCounts") or {})
        for slot_id in existing_slot_ids:
            if slot_id:
                slot_ready_counts[slot_id] = 1
        last_reason_code = (
            outcome.reason_codes[-1] if outcome.reason_codes else "VERIFIED"
        )
        group.update(
            {
                "replacementWave": outcome.replacement_wave_count,
                "lastReasonCode": last_reason_code,
                "state": "FAILED" if unresolved else "COMPLETED",
            }
        )
        if unresolved:
            provider_replacement_count = int(group.get("providerReplacementCount") or 0)
            provider_replacement_budget = self._config.item_retry_limit
            provider_failure_reason_code = (
                "PRACTICE_VERIFIER_FALLBACK_EXHAUSTED"
                if outcome.provider_failure_stage == "VERIFIER"
                else "PRACTICE_GENERATOR_FALLBACK_EXHAUSTED"
            )
            if (
                outcome.provider_failure_recoverable
                and provider_replacement_count < provider_replacement_budget
            ):
                next_provider_replacement_count = provider_replacement_count + 1
                replacement_groups: list[dict[str, object]] = []
                primary_replacement_slot_id = unresolved[0]
                group.update(
                    {
                        "state": "PENDING",
                        "slotIds": [primary_replacement_slot_id],
                        "requiredCount": 1,
                        "attempt": int(group.get("attempt") or 0) + 1,
                        "attemptStage": "PROVIDER_REPLACEMENT",
                        "providerReplacementCount": next_provider_replacement_count,
                        "replacement": False,
                        "lastReasonCode": provider_failure_reason_code,
                    }
                )
                for item_index, slot_id in enumerate(unresolved[1:], start=2):
                    replacement_group_id = (
                        f"{outcome.context.group.group_id[:146]}"
                        f"-p{next_provider_replacement_count}-{item_index}"
                    )
                    if replacement_group_id in groups:
                        self._mark_failed(
                            test_id,
                            "PRACTICE_PROVIDER_REPLACEMENT_CONFLICT",
                        )
                        return False
                    replacement_group = dict(group)
                    replacement_group.update(
                        {
                            "groupId": replacement_group_id,
                            "slotIds": [slot_id],
                            "requiredCount": 1,
                        }
                    )
                    groups[replacement_group_id] = replacement_group
                    replacement_groups.append(replacement_group)
                self._update_progress(
                    test_id,
                    outcome.context.request,
                    authoritative_question_ids=tuple(accepted_question_ids),
                    meta_updates={
                        "generationGroups": groups,
                        "slotReadyCounts": slot_ready_counts,
                        "replacementWaveCount": max(
                            int(meta.get("replacementWaveCount") or 0),
                            outcome.replacement_wave_count,
                        ),
                        "lastCompletedStage": "SLOT_GROUP_PROVIDER_REPLACEMENT_SCHEDULED",
                    },
                )
                emit_practice_event(
                    "practice_provider_replacement_scheduled",
                    test_id=test_id,
                    status="scheduled",
                    details={
                        "groupId": outcome.context.group.group_id,
                        "batchId": outcome.context.group.group_id,
                        "slotCount": len(unresolved),
                        "slotIds": ",".join(unresolved),
                        "routeId": outcome.context.slots[0].generator_route_hint,
                        "splitGroupCount": len(replacement_groups) + 1,
                        "acceptedCount": len(existing_slot_ids),
                        "reusedCount": int(meta.get("reusedCount") or 0),
                        "remainingCount": len(unresolved),
                        "attempt": next_provider_replacement_count,
                        "reasonCode": provider_failure_reason_code,
                    },
                    level=logging.WARNING,
                )
                emit_practice_event(
                    "DEFICIT_CALCULATED",
                    test_id=test_id,
                    status="recalculated",
                    details={
                        "reusedCount": int(meta.get("reusedCount") or 0),
                        "acceptedCount": len(existing_slot_ids),
                        "deficitCount": len(unresolved),
                        "groupId": outcome.context.group.group_id,
                    },
                )
                return True

            if outcome.provider_failure_recoverable:
                terminal_reason_code = "PRACTICE_REPLACEMENT_EXHAUSTED"
            elif outcome.provider_failure_stage == "VERIFIER":
                # Verification never completed; the questions themselves were
                # never shown to be deficient.
                terminal_reason_code = "PRACTICE_VERIFIER_FALLBACK_EXHAUSTED"
            elif "STRUCTURED_PARSE_INVALID" in outcome.reason_codes:
                terminal_reason_code = "PRACTICE_GENERATOR_OUTPUT_INVALID"
            else:
                terminal_reason_code = "GENERATION_DEFICIT_EXHAUSTED"
            group.update(
                {
                    "state": "FAILED",
                    "errorCode": terminal_reason_code,
                    "lastReasonCode": terminal_reason_code,
                }
            )
            self._update_progress(
                test_id,
                outcome.context.request,
                authoritative_question_ids=tuple(accepted_question_ids),
                meta_updates={
                    "generationGroups": groups,
                    "slotReadyCounts": slot_ready_counts,
                    "replacementWaveCount": max(
                        int(meta.get("replacementWaveCount") or 0),
                        outcome.replacement_wave_count,
                    ),
                    "lastCompletedStage": "SLOT_GROUP_FAILED_AFTER_COMMIT",
                },
            )
            self._mark_failed(
                test_id,
                terminal_reason_code,
            )
            return False
        self._update_progress(
            test_id,
            outcome.context.request,
            authoritative_question_ids=tuple(accepted_question_ids),
            meta_updates={
                "generationGroups": groups,
                "slotReadyCounts": slot_ready_counts,
                "replacementWaveCount": max(
                    int(meta.get("replacementWaveCount") or 0),
                    outcome.replacement_wave_count,
                ),
                "lastCompletedStage": "SLOT_GROUP_COMMITTED",
            },
        )
        return True

    def _maybe_finalize(
        self,
        test_id: str,
        request: PracticeGenerationRequest,
        blueprint: PracticeBlueprint,
    ) -> None:
        assessment = self._assessments.get(test_id)
        if assessment is None or assessment.get("status") != "GENERATING":
            return
        groups = _meta(assessment).get("generationGroups")
        if isinstance(groups, dict) and any(
            str(group.get("state")) not in {"COMPLETED"}
            for group in groups.values()
            if isinstance(group, dict)
        ):
            return
        self._finalize(test_id, request, blueprint, attempt=0)

    def _finalize(
        self,
        test_id: str,
        request: PracticeGenerationRequest,
        blueprint: PracticeBlueprint,
        *,
        attempt: int,
    ) -> None:
        assessment = self._assessments.get(test_id)
        if assessment is None or assessment.get("status") != "GENERATING":
            return
        meta = _meta(assessment)
        groups = meta.get("generationGroups")
        if isinstance(groups, dict) and any(
            not isinstance(group, dict) or group.get("state") != "COMPLETED"
            for group in groups.values()
        ):
            return
        ready_count = int(meta.get("readyQuestionCount") or 0)
        failed_count = int(meta.get("failedCount") or 0)
        reused_count = int(meta.get("reusedCount") or 0)
        generated_count = int(meta.get("generatedCount") or 0)
        verified_count = int(meta.get("verifiedCount") or 0)
        bucket_counts = {
            str(key): int(value) for key, value in dict(meta.get("bucketReadyCounts") or {}).items()
        }
        counters_agree = (
            reused_count + generated_count == ready_count
            and verified_count == ready_count
            and sum(bucket_counts.values()) == ready_count
        )
        if ready_count != request.accepted_count or failed_count != 0 or not counters_agree:
            self._mark_failed(
                test_id,
                "AUTHORITATIVE_COUNTER_MISMATCH",
            )
            return
        emit_practice_event(
            "final_manifest_validation_started",
            test_id=test_id,
            status="started",
            details={"acceptedCount": request.accepted_count},
        )
        emit_practice_event(
            "ASSESSMENT_FINALIZING",
            test_id=test_id,
            status="finalizing",
            details={
                "readyCount": int(meta.get("readyCount") or 0),
                "acceptedCount": request.accepted_count,
            },
        )
        manifest_ids = [
            str(question_id)
            for question_id in list(meta.get("readyQuestionIds") or [])
            if str(question_id)
        ]
        manifest_count = int(meta.get("readyCount") or 0)
        if (
            manifest_count != request.accepted_count
            or len(manifest_ids) != request.accepted_count
            or len(set(manifest_ids)) != request.accepted_count
        ):
            self._mark_failed(test_id, "QUESTION_MANIFEST_MISMATCH")
            return
        linked = self._questions.get_questions_by_ids(manifest_ids)
        if len(linked) != request.accepted_count:
            next_attempt = attempt + 1
            if next_attempt < self._config.finalization_max_attempts:
                retry_recorded = self._progress_updates.record_finalization_retry(
                    test_id,
                    next_attempt=next_attempt,
                    reason_code="MANIFEST_QUESTIONS_NOT_YET_RESOLVED",
                )
                if not retry_recorded:
                    return
                return
            self._mark_failed(
                test_id,
                "FINALIZATION_CONSISTENCY_TIMEOUT",
            )
            return
        if any(str(item.get("testId") or "") != test_id for item in linked):
            self._mark_failed(
                test_id,
                "QUESTION_MANIFEST_OWNERSHIP_MISMATCH",
            )
            return
        for item in linked:
            if "_practiceMeta" not in item:
                item["_practiceMeta"] = _meta({"meta": item.get("meta")})
        validation = validate_final_set(
            blueprint=blueprint,
            linked_questions=linked,
            requested_language=request.language,
        )
        if not validation.ready:
            emit_practice_event(
                "final_manifest_validation_failed",
                test_id=test_id,
                status="failed",
                details={
                    "reasonCode": validation.reason_code,
                    "failedSlotIds": ",".join(validation.failed_slot_ids),
                    "expectedQuestionCount": validation.expected_question_count,
                    "actualQuestionCount": validation.actual_question_count,
                    "recoverable": validation.recoverable,
                },
            )
            if (
                attempt == 0
                and validation.recoverable
                and validation.reason_code == "BLUEPRINT_DISTRIBUTION_MISMATCH"
                and validation.failed_slot_ids
            ):
                slots_by_id = {slot.slot_id: slot for slot in blueprint.slots}
                assignments: dict[str, tuple[str, str]] = {}
                linked_by_slot: dict[str, dict[str, object]] = {}
                for item in linked:
                    practice_meta = item.get("_practiceMeta")
                    if isinstance(practice_meta, dict):
                        linked_by_slot[str(practice_meta.get("slotId") or "")] = item
                for slot_id in validation.failed_slot_ids:
                    slot = slots_by_id.get(slot_id)
                    item = linked_by_slot.get(slot_id)
                    question_id = str(item.get("questionId") or "") if item else ""
                    if slot is None or not question_id:
                        self._mark_failed(test_id, validation.reason_code)
                        return
                    assignments[question_id] = (
                        slot_id,
                        bucket_for_slot(blueprint, slot).bucket_id,
                    )
                emit_practice_event(
                    "manifest_recovery_started",
                    test_id=test_id,
                    status="started",
                    details={
                        "reasonCode": validation.reason_code,
                        "repairAttempt": 1,
                        "failedSlotIds": ",".join(validation.failed_slot_ids),
                    },
                )
                try:
                    repaired_count = self._questions.repair_bucket_assignments(
                        test_id,
                        assignments,
                    )
                except PracticeRepositoryError as exc:
                    emit_practice_event(
                        "manifest_recovery_failed",
                        test_id=test_id,
                        status="failed",
                        details={
                            "reasonCode": exc.code,
                            "repairAttempt": 1,
                            "failedSlotIds": ",".join(validation.failed_slot_ids),
                        },
                        level=logging.ERROR,
                    )
                    self._mark_failed(test_id, exc.code)
                    return
                if repaired_count != len(assignments):
                    self._mark_failed(test_id, "MANIFEST_RECOVERY_INCOMPLETE")
                    return
                retry_recorded = self._progress_updates.record_finalization_retry(
                    test_id,
                    next_attempt=1,
                    reason_code="MANIFEST_BUCKET_ASSIGNMENTS_REPAIRED",
                )
                if retry_recorded:
                    emit_practice_event(
                        "manifest_recovery_completed",
                        test_id=test_id,
                        status="completed",
                        details={
                            "reasonCode": "MANIFEST_BUCKET_ASSIGNMENTS_REPAIRED",
                            "repairAttempt": 1,
                            "repairedSlotCount": repaired_count,
                        },
                    )
                return
            if attempt > 0 and validation.recoverable:
                emit_practice_event(
                    "manifest_recovery_failed",
                    test_id=test_id,
                    status="failed",
                    details={
                        "reasonCode": validation.reason_code,
                        "repairAttempt": 1,
                        "failedSlotIds": ",".join(validation.failed_slot_ids),
                    },
                    level=logging.ERROR,
                )
            self._mark_failed(test_id, validation.reason_code)
            return
        try:
            rebalanced, correct_positions = self._questions.rebalance_answer_positions(
                test_id,
                linked,
            )
        except PracticeRepositoryError as exc:
            self._mark_failed(test_id, str(exc))
            return
        revalidated = validate_final_set(
            blueprint=blueprint,
            linked_questions=linked,
            requested_language=request.language,
        )
        if not revalidated.ready:
            self._mark_failed(test_id, revalidated.reason_code)
            return
        emit_practice_event(
            (
                "practice_answer_distribution_rebalanced"
                if rebalanced
                else "practice_answer_distribution_validated"
            ),
            test_id=test_id,
            status="completed",
            details={
                "questionCount": len(linked),
                **{
                    f"correctPosition{position}Count": correct_positions.count(position)
                    for position in range(4)
                },
            },
        )
        self._questions.assign_positions(linked)
        reused_count, generated_count = self._source_counts(linked)
        ready_updated = self._progress_updates.mark_ready(
            test_id,
            request.accepted_count,
            {
                "phase": InternalPhase.READY.value,
                "playable": True,
                "progressPercent": 100,
                "readyQuestionCount": request.accepted_count,
                "readyCount": request.accepted_count,
                "verifiedCount": request.accepted_count,
                "reusedCount": reused_count,
                "generatedCount": generated_count,
                "readyAt": datetime.now(UTC).isoformat(),
                "errorCode": None,
            },
        )
        if not ready_updated:
            return
        emit_practice_event(
            "final_manifest_validation_completed",
            test_id=test_id,
            status="valid",
            details={
                "expectedQuestionCount": request.accepted_count,
                "actualQuestionCount": len(linked),
                "uniqueQuestionCount": len({str(item.get("questionId") or "") for item in linked}),
                "expectedSlotCount": len(blueprint.slots),
                "resolvedSlotCount": len(blueprint.slots),
                "failedSlotCount": 0,
                "recoverable": False,
                "recoveryAttempt": attempt,
            },
        )
        emit_practice_event(
            "practice_ready",
            test_id=test_id,
            status="ready",
            details={
                "acceptedCount": request.accepted_count,
                "reusedCount": reused_count,
                "generatedCount": generated_count,
            },
        )
        emit_practice_event(
            "practice_ready_published",
            test_id=test_id,
            status="ready",
            details={
                "acceptedCount": request.accepted_count,
                "businessStatus": "READY",
            },
        )
        emit_practice_event(
            "practice_generation_terminal",
            test_id=test_id,
            status="ready",
            details={
                "acceptedCount": request.accepted_count,
                "businessStatus": "READY",
            },
        )
        emit_practice_event(
            "ASSESSMENT_READY",
            test_id=test_id,
            status="ready",
            details={
                "acceptedCount": request.accepted_count,
                "reusedCount": reused_count,
                "generatedCount": generated_count,
                "verifiedCount": request.accepted_count,
            },
        )

    def _update_progress(
        self,
        test_id: str,
        request: PracticeGenerationRequest,
        *,
        authoritative_question_ids: tuple[str, ...] = (),
        meta_updates: dict[str, Any] | None = None,
    ) -> None:
        assessment = self._assessments.get(test_id)
        if assessment is None:
            raise PracticeRepositoryError("ASSESSMENT_NOT_FOUND")
        meta = _meta(assessment)
        ready_count = int(meta.get("readyQuestionCount") or 0)
        manifest_count = int(meta.get("readyCount") or 0)
        progress = self._progress(ready_count, request.accepted_count)
        self._progress_updates.update(
            test_id,
            meta_updates={
                **(meta_updates or {}),
                "phase": InternalPhase.GENERATING.value,
                "progressPercent": progress,
            },
            live=False,
            recalculate_manifest=True,
            authoritative_question_ids=authoritative_question_ids,
        )
        emit_practice_event(
            "practice_manifest_updated",
            test_id=test_id,
            status="updated",
            details={
                "readyCount": manifest_count,
                "acceptedCount": request.accepted_count,
            },
        )
        emit_practice_event(
            "QUESTION_MANIFEST_UPDATED",
            test_id=test_id,
            status="updated",
            details={
                "readyCount": manifest_count,
                "acceptedCount": request.accepted_count,
            },
        )
        emit_practice_event(
            "ASSESSMENT_PROGRESS",
            test_id=test_id,
            status="generating",
            details={
                "readyQuestionCount": ready_count,
                "acceptedCount": request.accepted_count,
                "progressPercent": progress,
            },
        )
        emit_practice_event(
            "practice_generation_progress",
            test_id=test_id,
            status="generating",
            details={
                "readyQuestionCount": ready_count,
                "acceptedCount": request.accepted_count,
                "progressPercent": progress,
            },
        )

    @staticmethod
    def _bucket_counts(linked: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in linked:
            bucket_id = str(item.get("_practiceMeta", {}).get("bucketId") or "")
            if bucket_id:
                counts[bucket_id] = counts.get(bucket_id, 0) + 1
        return counts

    @staticmethod
    def _source_counts(linked: list[dict[str, Any]]) -> tuple[int, int]:
        reused = sum(
            1 for item in linked if item.get("_practiceMeta", {}).get("source") == "REUSED"
        )
        generated = sum(
            1 for item in linked if item.get("_practiceMeta", {}).get("source") == "GENERATED"
        )
        return reused, generated

    @staticmethod
    def _progress(ready: int, accepted: int) -> int:
        if ready <= 0:
            return 10
        return min(95, 10 + int(85 * ready / accepted))

    def finalize(self, test_id: str, *, attempt: int) -> None:
        assessment = self._assessments.get(test_id)
        if assessment is None or assessment.get("status") != "GENERATING":
            return
        request = _request(assessment)
        blueprint = apply_system_bucket_policy(
            PracticeBlueprint.model_validate(_meta(assessment).get("blueprint")),
            request,
        )
        self._finalize(test_id, request, blueprint, attempt=attempt)
