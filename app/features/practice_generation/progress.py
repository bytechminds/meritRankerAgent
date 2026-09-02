"""Authoritative AppSync parent-state updates for practice generation."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from features.practice_generation.appsync_progress_client import (
    AppSyncPracticeProgressClient,
    PracticeProgressError,
)
from features.practice_generation.execution_control import (
    PracticeExecutionStopped,
    current_practice_execution_id,
    local_cancellation_requested,
)
from features.practice_generation.progress_contract import (
    PracticeProgressContractError,
    PracticeProgressMeta,
    PracticeProgressMetaProjection,
    project_practice_progress_meta,
)
from features.practice_generation.repositories import (
    AssessmentRepository,
    PracticeRepositoryError,
    QuestionRepository,
)
from features.practice_generation.schemas import InternalPhase, PracticeBlueprint

logger = logging.getLogger(__name__)

_FALLBACKABLE_PROGRESS_FAILURE_CODES = frozenset(
    {
        "PRACTICE_PROGRESS_CREDENTIALS_UNAVAILABLE",
        "PRACTICE_PROGRESS_TRANSPORT_FAILED",
    }
)
_RESUMABLE_ERROR_CODES = frozenset(
    {
        "USER_CANCELLED",
        "PRACTICE_GENERATOR_PROVIDER_FAILED",
        "PRACTICE_GENERATOR_FALLBACK_EXHAUSTED",
        "PRACTICE_VERIFIER_PROVIDER_FAILED",
        "PRACTICE_VERIFIER_FALLBACK_EXHAUSTED",
        "PRACTICE_GENERATION_STALLED",
        "PRACTICE_RECOVERY_EXHAUSTED",
        "PRACTICE_MAX_WALL_TIME_EXCEEDED",
    }
)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _is_resumable_error(value: object) -> bool:
    return value is None or str(value) in _RESUMABLE_ERROR_CODES


def _meta(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("meta")
    if isinstance(value, dict):
        return deepcopy(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _remove_legacy_query_summary(meta: dict[str, Any]) -> None:
    """Keep raw student text out of the parent progress payload."""
    practice_request = meta.get("practiceRequest")
    if isinstance(practice_request, dict):
        practice_request.pop("querySummary", None)


class AppSyncAssessmentProgressRepository:
    """Merge and publish complete MockTestQuiz metadata through AppSync only."""

    def __init__(
        self,
        *,
        assessments: AssessmentRepository,
        questions: QuestionRepository,
        client: AppSyncPracticeProgressClient,
    ) -> None:
        self._assessments = assessments
        self._questions = questions
        self._client = client

    def update(
        self,
        test_id: str,
        *,
        meta_updates: dict[str, Any],
        status: str | None = None,
        live: bool | None = None,
        expected_updated_at: str | None = None,
        recalculate_manifest: bool = False,
        authoritative_question_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        assessment = self._required_assessment(test_id)
        execution_id = current_practice_execution_id()
        if execution_id is not None and _meta(assessment).get(
            "activeExecutionId"
        ) != execution_id:
            raise PracticeRepositoryError("PRACTICE_EXECUTION_FENCED")
        if expected_updated_at is not None and assessment.get("updatedAt") != expected_updated_at:
            raise PracticeRepositoryError("ASSESSMENT_CONCURRENT_UPDATE")
        current_status = str(assessment.get("status") or "GENERATING")
        target_status = status or current_status
        target_live = bool(assessment.get("live")) if live is None else live
        meta = _meta(assessment)
        meta.update(deepcopy(meta_updates))
        meta["lastProgressAt"] = datetime.now(UTC).isoformat()
        _remove_legacy_query_summary(meta)
        if recalculate_manifest:
            meta = self._authoritative_meta(
                test_id,
                assessment,
                meta,
                authoritative_question_ids=authoritative_question_ids,
            )
        self._validate_monotonic(assessment, meta, target_status, target_live)
        projection = self._project_transport_meta(meta)
        if projection.unapproved_keys:
            self._log_transport_projection(meta, projection, logging.ERROR)
            raise PracticeRepositoryError(
                "PRACTICE_PROGRESS_UNKNOWN_META_FIELD",
                operation="appsync.updatePracticeGenerationProgress",
                logical_table="MockTestQuiz",
                fallback_decision="fatal_progress_publication_failure",
            )
        if projection.unknown_keys:
            self._log_transport_projection(meta, projection, logging.WARNING)

        try:
            result = asyncio.run(
                self._client.update_progress(
                    test_id=test_id,
                    user_id=str(assessment.get("userId") or ""),
                    status=target_status,
                    meta=projection.payload,
                    live=target_live,
                    expected_updated_at=str(assessment.get("updatedAt") or ""),
                )
            )
        except PracticeProgressError as exc:
            if exc.code == "PRACTICE_PROGRESS_CONFLICT":
                self._authoritative_meta(
                    test_id,
                    self._required_assessment(test_id),
                    meta,
                )
                raise PracticeRepositoryError(
                    "ASSESSMENT_CONCURRENT_UPDATE",
                    operation="appsync.updatePracticeGenerationProgress",
                    logical_table="MockTestQuiz",
                    fallback_decision="fatal_progress_publication_failure",
                ) from exc
            raise PracticeRepositoryError(
                exc.code,
                operation="appsync.updatePracticeGenerationProgress",
                logical_table="MockTestQuiz",
                fallback_decision="fatal_progress_publication_failure",
                progress_detail=exc.safe_detail,
                progress_error_type=exc.error_type,
                progress_retryable=exc.retryable,
            ) from exc
        updated = dict(assessment)
        updated.update(
            {
                "status": result.status,
                "live": result.live,
                "updatedAt": result.updated_at,
                "meta": meta,
            }
        )
        return updated

    def claim_execution(
        self,
        test_id: str,
        *,
        user_id: str,
        execution_id: str,
        lease_seconds: int,
        resume_reason: str,
    ) -> dict[str, Any] | None:
        """CAS-claim initial/resumed execution and reclaim only unfinished groups."""
        assessment = self._required_assessment(test_id)
        if (
            assessment.get("userId") != user_id
            or assessment.get("origin") != "AI_CUSTOM"
            or assessment.get("visibility") != "PRIVATE"
        ):
            raise PracticeRepositoryError("PRACTICE_EXECUTION_UNAUTHORIZED")
        status = str(assessment.get("status") or "")
        if status == "READY":
            raise PracticeRepositoryError("PRACTICE_ALREADY_READY")
        meta = _meta(assessment)
        active_execution_id = meta.get("activeExecutionId")
        lease_expires_at = _parse_timestamp(meta.get("executionLeaseExpiresAt"))
        now = datetime.now(UTC)
        if (
            active_execution_id
            and active_execution_id != execution_id
            and lease_expires_at is not None
            and lease_expires_at > now
        ):
            return None
        if status == "FAILED" and not _is_resumable_error(meta.get("errorCode")):
            raise PracticeRepositoryError("PRACTICE_NOT_RESUMABLE")
        groups = deepcopy(dict(meta.get("generationGroups") or {}))
        valid_slot_ids: set[str] = set()
        invalid_question_ids: tuple[str, ...] = ()
        if resume_reason != "INITIAL":
            practice_request = meta.get("practiceRequest")
            request_data = practice_request if isinstance(practice_request, dict) else {}
            valid_slot_ids, invalid_question_ids = self._questions.inspect_reusable_slots(
                test_id,
                language=str(request_data.get("language") or "english"),
                solution_required=bool(request_data.get("includeSolutions", True)),
            )
        for group in groups.values():
            group_slot_ids = {
                str(slot_id) for slot_id in list(group.get("slotIds") or []) if str(slot_id)
            } if isinstance(group, dict) else set()
            if isinstance(group, dict) and (
                group.get("state") in {"RUNNING", "FAILED"}
                or (resume_reason != "INITIAL" and bool(group_slot_ids - valid_slot_ids))
            ):
                group["state"] = "PENDING"
                group["lastReasonCode"] = "INTERRUPTED_BEFORE_COMMIT"
        attempt = int(meta.get("executionAttempt") or 0) + 1
        lease = (now + timedelta(seconds=lease_seconds)).isoformat()
        try:
            claimed = self.update(
                test_id,
                meta_updates={
                    "activeExecutionId": execution_id,
                    "executionLeaseExpiresAt": lease,
                    "executionStartedAt": now.isoformat(),
                    "executionAttempt": attempt,
                    "cancelRequested": False,
                    "resumeReason": resume_reason,
                    "generationGroups": groups,
                    "phase": InternalPhase.GENERATING.value,
                    "playable": False,
                    "errorCode": None,
                    "lastCompletedStage": "EXECUTION_CLAIMED",
                },
                status="GENERATING",
                live=False,
                expected_updated_at=str(assessment.get("updatedAt") or ""),
                recalculate_manifest=not invalid_question_ids,
            )
        except PracticeRepositoryError as exc:
            if exc.code == "ASSESSMENT_CONCURRENT_UPDATE":
                return None
            raise
        if not invalid_question_ids:
            return claimed
        self._questions.delete_invalid_resume_questions(
            test_id,
            execution_id,
            invalid_question_ids,
        )
        return self.update(
            test_id,
            meta_updates={"lastCompletedStage": "RESUME_RECONSTRUCTED"},
            expected_updated_at=str(claimed.get("updatedAt") or ""),
            live=False,
            recalculate_manifest=True,
        )

    def request_cancellation(self, test_id: str, *, user_id: str) -> dict[str, Any]:
        assessment = self._required_assessment(test_id)
        if (
            assessment.get("userId") != user_id
            or assessment.get("origin") != "AI_CUSTOM"
            or assessment.get("visibility") != "PRIVATE"
        ):
            raise PracticeRepositoryError("PRACTICE_EXECUTION_UNAUTHORIZED")
        status = str(assessment.get("status") or "")
        if status == "READY":
            raise PracticeRepositoryError("PRACTICE_ALREADY_READY")
        meta = _meta(assessment)
        if bool(meta.get("cancelRequested")):
            return assessment
        if status != "GENERATING":
            raise PracticeRepositoryError("PRACTICE_NOT_ACTIVE")
        return self.update(
            test_id,
            meta_updates={
                "cancelRequested": True,
                "phase": InternalPhase.CANCELLED.value,
                "playable": False,
                "errorCode": "USER_CANCELLED",
                "lastCompletedStage": "CANCEL_REQUESTED",
            },
            expected_updated_at=str(assessment.get("updatedAt") or ""),
            live=False,
            recalculate_manifest=True,
        )

    def finish_cancellation(self, test_id: str, execution_id: str) -> None:
        assessment = self._required_assessment(test_id)
        meta = _meta(assessment)
        if meta.get("activeExecutionId") != execution_id:
            return
        if bool(meta.get("cancelRequested")) and assessment.get("status") == "GENERATING":
            self.update(
                test_id,
                meta_updates={
                    "phase": InternalPhase.CANCELLED.value,
                    "playable": False,
                    "errorCode": "USER_CANCELLED",
                    "lastCompletedStage": "CANCEL_COMPLETED",
                },
                status="FAILED",
                live=False,
                expected_updated_at=str(assessment.get("updatedAt") or ""),
                recalculate_manifest=True,
            )

    def require_expensive_work_allowed(
        self,
        test_id: str,
        *,
        lease_seconds: int = 300,
    ) -> None:
        execution_id = current_practice_execution_id()
        if execution_id is None:
            return
        if local_cancellation_requested():
            raise PracticeExecutionStopped("PRACTICE_CANCEL_OBSERVED")
        lease = (datetime.now(UTC) + timedelta(seconds=lease_seconds)).isoformat()
        if not self._assessments.renew_execution_lease(test_id, execution_id, lease):
            raise PracticeExecutionStopped("PRACTICE_EXECUTION_FENCED")

    def release_execution(self, test_id: str, execution_id: str) -> bool:
        return self._assessments.release_execution(test_id, execution_id)

    def calculate_authoritative_meta(
        self,
        test_id: str,
        *,
        meta_updates: dict[str, Any] | None = None,
        authoritative_question_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Recalculate persisted-question state without publishing a checkpoint."""
        assessment = self._required_assessment(test_id)
        meta = _meta(assessment)
        meta.update(deepcopy(meta_updates or {}))
        return self._authoritative_meta(
            test_id,
            assessment,
            meta,
            authoritative_question_ids=authoritative_question_ids,
        )

    def set_blueprint(
        self,
        test_id: str,
        blueprint: PracticeBlueprint,
        *,
        planner_calls: int,
        planner_tier: str,
        planner_repaired: bool,
        deterministic_fallback: bool,
    ) -> None:
        self.update(
            test_id,
            meta_updates={
                "phase": InternalPhase.MATCHING_EXISTING.value,
                "blueprint": blueprint.model_dump(mode="json"),
                "plannerCalls": planner_calls,
                "plannerTier": planner_tier,
                "plannerRepaired": planner_repaired,
                "plannerDeterministicFallback": deterministic_fallback,
                "progressPercent": 5,
                "bucketReadyCounts": {bucket.bucket_id: 0 for bucket in blueprint.buckets},
            },
            live=False,
        )

    def mark_failed(
        self,
        test_id: str,
        error_code: str,
        *,
        meta_updates: dict[str, Any] | None = None,
    ) -> None:
        assessment = self._required_assessment(test_id)
        if str(assessment.get("status") or "") != "GENERATING":
            return
        meta = _meta(assessment)
        updates = deepcopy(meta_updates or {})
        updates.update(
            {
                "phase": InternalPhase.FAILED.value,
                "playable": False,
                "errorCode": error_code,
                "failedCount": int(meta.get("failedCount") or 0) + 1,
            }
        )
        try:
            self.update(
                test_id,
                meta_updates=updates,
                status="FAILED",
                live=False,
                recalculate_manifest=True,
            )
        except PracticeRepositoryError as exc:
            # AppSync is the primary channel.  If it is unavailable, the existing
            # conditional DynamoDB failure write prevents a permanent stale activity.
            if exc.code not in _FALLBACKABLE_PROGRESS_FAILURE_CODES:
                raise
            self._assessments.mark_failed(test_id, error_code)

    def claim_group(self, test_id: str, group_id: str) -> dict[str, Any] | None:
        assessment = self._required_assessment(test_id)
        meta = _meta(assessment)
        groups = deepcopy(dict(meta.get("generationGroups") or {}))
        group = groups.get(group_id)
        if not isinstance(group, dict) or group.get("state") != "PENDING":
            return None
        group["state"] = "RUNNING"
        group["claimCount"] = int(group.get("claimCount") or 0) + 1
        groups[group_id] = group
        self.update(
            test_id,
            meta_updates={
                "generationGroups": groups,
                "phase": InternalPhase.GENERATING.value,
            },
            expected_updated_at=str(assessment.get("updatedAt") or ""),
            live=False,
        )
        return group

    def claim_recovery(self, test_id: str) -> bool:
        """CAS-claim the single allowed continuation and reset only unfinished work."""
        assessment = self._required_assessment(test_id)
        if str(assessment.get("status") or "") != "GENERATING":
            return False
        meta = _meta(assessment)
        if int(meta.get("recoveryAttemptCount") or 0) >= 1:
            return False
        groups = deepcopy(dict(meta.get("generationGroups") or {}))
        for group in groups.values():
            if isinstance(group, dict) and group.get("state") == "RUNNING":
                group["state"] = "PENDING"
                group["lastReasonCode"] = "INTERRUPTED_BEFORE_COMMIT"
        try:
            self.update(
                test_id,
                meta_updates={
                    "generationGroups": groups,
                    "recoveryAttemptCount": 1,
                    "lastCompletedStage": "RECOVERY_CLAIMED",
                    "phase": InternalPhase.GENERATING.value,
                },
                expected_updated_at=str(assessment.get("updatedAt") or ""),
                live=False,
                recalculate_manifest=True,
            )
        except PracticeRepositoryError as exc:
            if str(exc) == "ASSESSMENT_CONCURRENT_UPDATE":
                return False
            raise
        return True

    def create_retry_group(
        self,
        test_id: str,
        *,
        group_id: str,
        parent_group_id: str,
        bucket_id: str,
        attempt_stage: str,
        attempt: int,
        replacement: bool,
    ) -> bool:
        assessment = self._required_assessment(test_id)
        meta = _meta(assessment)
        groups = deepcopy(dict(meta.get("generationGroups") or {}))
        if group_id in groups:
            return False
        parent = groups.get(parent_group_id)
        if not isinstance(parent, dict) or parent.get("state") != "RUNNING":
            return False
        groups[group_id] = {
            "groupId": group_id,
            "bucketId": bucket_id,
            "requiredCount": 1,
            "attempt": attempt,
            "attemptStage": attempt_stage,
            "itemRetryCount": 1,
            "replacementCount": 1 if replacement else 0,
            "replacement": replacement,
            "lastReasonCode": "PARTIAL_GROUP_DEFICIT",
            "state": "PENDING",
        }
        self.update(
            test_id,
            meta_updates={"generationGroups": groups},
            expected_updated_at=str(assessment.get("updatedAt") or ""),
            live=False,
        )
        return True

    def update_group(
        self,
        test_id: str,
        group_id: str,
        *,
        state: str,
        required_count: int | None = None,
        attempt: int | None = None,
        error_code: str | None = None,
        attempt_stage: str | None = None,
        item_retry_count: int | None = None,
        replacement_count: int | None = None,
        replacement: bool | None = None,
        last_reason_code: str | None = None,
    ) -> None:
        assessment = self._required_assessment(test_id)
        meta = _meta(assessment)
        groups = deepcopy(dict(meta.get("generationGroups") or {}))
        group = groups.get(group_id)
        if not isinstance(group, dict):
            raise PracticeRepositoryError("GENERATION_GROUP_NOT_FOUND")
        group["state"] = state
        for key, value in {
            "requiredCount": required_count,
            "attempt": attempt,
            "errorCode": error_code,
            "attemptStage": attempt_stage,
            "itemRetryCount": item_retry_count,
            "replacementCount": replacement_count,
            "replacement": replacement,
            "lastReasonCode": last_reason_code,
        }.items():
            if value is not None:
                group[key] = value
        groups[group_id] = group
        self.update(
            test_id,
            meta_updates={"generationGroups": groups},
            expected_updated_at=str(assessment.get("updatedAt") or ""),
            live=False,
            recalculate_manifest=True,
        )

    def record_finalization_retry(
        self,
        test_id: str,
        *,
        next_attempt: int,
        reason_code: str,
    ) -> bool:
        assessment = self._required_assessment(test_id)
        meta = _meta(assessment)
        if int(meta.get("finalizationAttempt") or 0) >= next_attempt:
            return False
        self.update(
            test_id,
            meta_updates={
                "phase": InternalPhase.FINALIZING.value,
                "finalizationAttempt": next_attempt,
                "finalizationReasonCode": reason_code,
            },
            expected_updated_at=str(assessment.get("updatedAt") or ""),
            live=False,
            recalculate_manifest=True,
        )
        return True

    def mark_ready(
        self,
        test_id: str,
        accepted_count: int,
        meta_updates: dict[str, Any],
    ) -> bool:
        assessment = self._required_assessment(test_id)
        meta = self._authoritative_meta(test_id, assessment, _meta(assessment))
        if (
            int(meta.get("readyQuestionCount") or 0) != accepted_count
            or int(meta.get("readyCount") or 0) != accepted_count
            or len(meta.get("readyQuestionIds") or []) != accepted_count
            or int(meta.get("failedCount") or 0) != 0
        ):
            return False
        self.update(
            test_id,
            meta_updates=deepcopy(meta_updates),
            status="READY",
            live=True,
            recalculate_manifest=True,
        )
        return True

    def _required_assessment(self, test_id: str) -> dict[str, Any]:
        assessment = self._assessments.get(test_id)
        if assessment is None:
            raise PracticeRepositoryError("ASSESSMENT_NOT_FOUND")
        return assessment

    @staticmethod
    def _project_transport_meta(meta: dict[str, Any]) -> PracticeProgressMetaProjection:
        try:
            return project_practice_progress_meta(meta)
        except PracticeProgressContractError as exc:
            projection = PracticeProgressMetaProjection(
                meta=PracticeProgressMeta(),
                actual_keys=exc.actual_keys,
                unknown_keys=exc.unknown_keys,
            )
            AppSyncAssessmentProgressRepository._log_transport_projection(
                meta,
                projection,
                logging.ERROR,
            )
            raise PracticeRepositoryError(
                exc.code,
                operation="appsync.updatePracticeGenerationProgress",
                logical_table="MockTestQuiz",
                fallback_decision="fatal_progress_publication_failure",
            ) from exc

    @staticmethod
    def _log_transport_projection(
        meta: dict[str, Any],
        projection: PracticeProgressMetaProjection,
        level: int,
    ) -> None:
        stage = str(meta.get("lastCompletedStage") or meta.get("phase") or "UNKNOWN")
        logger.log(
            level,
            "practice_progress_transport stage=%s actual_keys=%s unknown_keys=%s "
            "contract_version=%s",
            stage,
            projection.actual_keys,
            projection.unknown_keys,
            projection.contract_version,
        )

    def _authoritative_meta(
        self,
        test_id: str,
        assessment: dict[str, Any],
        meta: dict[str, Any],
        *,
        authoritative_question_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        linked = self._questions.list_linked(test_id)
        existing_ids = [
            str(value) for value in list(meta.get("readyQuestionIds") or []) if str(value)
        ]
        existing_ids.extend(authoritative_question_ids)
        existing_ids = list(dict.fromkeys(existing_ids))
        if existing_ids:
            resolved = self._questions.get_questions_by_ids(existing_ids)
            by_id = {str(item.get("questionId") or ""): item for item in resolved}
        else:
            by_id = {}
        for item in linked:
            question_id = str(item.get("questionId") or "")
            if question_id:
                by_id[question_id] = item
        ordered = [by_id[key] for key in sorted(by_id)]
        total = int(assessment.get("totalQuestions") or meta.get("acceptedCount") or 0)
        if len(ordered) > total:
            raise PracticeRepositoryError("PRACTICE_PROGRESS_INVALID_COUNTS")
        bucket_counts: dict[str, int] = {}
        reused = 0
        generated = 0
        for item in ordered:
            practice_meta = item.get("_practiceMeta")
            if not isinstance(practice_meta, dict):
                practice_meta = _meta({"meta": item.get("meta")})
            bucket_id = str(practice_meta.get("bucketId") or "")
            if bucket_id:
                bucket_counts[bucket_id] = bucket_counts.get(bucket_id, 0) + 1
            if practice_meta.get("source") == "REUSED":
                reused += 1
            elif practice_meta.get("source") == "GENERATED":
                generated += 1
        count = len(ordered)
        meta.update(
            {
                "readyQuestionIds": [str(item["questionId"]) for item in ordered],
                "readyCount": count,
                "readyQuestionCount": count,
                "verifiedCount": count,
                "reusedCount": reused,
                "generatedCount": generated,
                "bucketReadyCounts": bucket_counts,
            }
        )
        if str(assessment.get("status") or "GENERATING") == "GENERATING":
            calculated_progress = 10 if count == 0 else min(95, 10 + int(85 * count / total))
            meta["progressPercent"] = max(
                int(meta.get("progressPercent") or 0),
                calculated_progress,
            )
        return meta

    @staticmethod
    def _validate_monotonic(
        assessment: dict[str, Any],
        meta: dict[str, Any],
        status: str,
        live: bool,
    ) -> None:
        previous = _meta(assessment)
        total = int(assessment.get("totalQuestions") or meta.get("acceptedCount") or 0)
        ready = int(meta.get("readyCount") or 0)
        progress = int(meta.get("progressPercent") or 0)
        if ready < int(previous.get("readyCount") or 0):
            raise PracticeRepositoryError("PRACTICE_PROGRESS_READY_COUNT_DECREASED")
        if progress < int(previous.get("progressPercent") or 0):
            raise PracticeRepositoryError("PRACTICE_PROGRESS_PERCENT_DECREASED")
        if ready > total or len(list(meta.get("readyQuestionIds") or [])) != ready:
            raise PracticeRepositoryError("PRACTICE_PROGRESS_INVALID_MANIFEST")
        if status == "GENERATING" and (bool(meta.get("playable")) or live):
            raise PracticeRepositoryError("PRACTICE_PROGRESS_INVALID_GENERATING_STATE")
        if status == "FAILED":
            meta["playable"] = False
        if status == "READY":
            meta["readyAt"] = meta.get("readyAt") or datetime.now(UTC).isoformat()
