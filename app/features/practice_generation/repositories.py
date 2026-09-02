"""DynamoDB repositories mapped to existing MockTestQuiz, QuestionBank, and Question."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

from features.practice_generation.events import emit_practice_event
from features.practice_generation.execution_control import current_practice_execution_id
from features.practice_generation.generation import deterministic_question_id
from features.practice_generation.matching import (
    ReusableQuestion,
    build_question_bank_id,
    build_reuse_difficulty_prefix,
    build_slot_reuse_bucket_key,
    question_bank_version_hash_from_item,
)
from features.practice_generation.option_distribution import (
    reorder_options,
    target_correct_positions,
    validate_answer_position_distribution,
)
from features.practice_generation.pattern_context import PatternSlotSelection
from features.practice_generation.planning import request_idempotency_key
from features.practice_generation.question_contract import validate_persisted_playable_question
from features.practice_generation.resource_validation import IndexProjection
from features.practice_generation.schemas import (
    AssessmentProgress,
    GeneratedQuestion,
    InternalPhase,
    PlannerSlot,
    PracticeBlueprint,
    PracticeGenerationRequest,
)

_SERIALIZER = TypeSerializer()
_DESERIALIZER = TypeDeserializer()
_SUBJECTS = {
    "math": "MATH",
    "reasoning": "REASONING",
    "science": "SCIENCE",
    "history": "HISTORY",
    "geography": "GEOGRAPHY",
    "english": "ENGLISH",
    "physics": "PHYSICS",
    "chemistry": "CHEMISTRY",
    "biology": "BIOLOGY",
    "computer_science": "COMPUTER_SCIENCE",
    "economics": "ECONOMICS",
    "polity": "POLITY",
    "general": "GENERAL",
}
logger = logging.getLogger(__name__)


def _storage_language(language: str) -> str:
    return {"english": "en", "hindi": "hi", "hinglish": "hinglish"}.get(
        language.casefold(),
        "",
    )
_ASSESSMENT_SIZE_ENVELOPE: dict[str, Any] = {
    "testId": "x" * 128,
    "userId": "x" * 128,
    "name": "x" * 180,
    "subject": "COMPUTER_SCIENCE",
    "topic": "x" * 128,
    "type": "mockTest",
    "activityKind": "MINI_MOCK",
    "assessmentMode": "PRACTICE",
    "status": "GENERATING",
    "visibility": "PRIVATE",
    "origin": "AI_CUSTOM",
    "language": "en",
    "durationMinutes": 180,
    "totalQuestions": 100,
    "exams": ["x" * 128],
    "live": False,
    "createdAt": "2026-07-28T00:00:00.000000Z",
    "updatedAt": "2026-07-28T00:00:00.000000Z",
    "__typename": "MockTestQuiz",
}


class PracticeRepositoryError(RuntimeError):
    """Typed practice persistence failure."""

    def __init__(
        self,
        code: str,
        *,
        operation: str | None = None,
        logical_table: str | None = None,
        logical_index: str | None = None,
        aws_exception_type: str | None = None,
        aws_error_code: str | None = None,
        fallback_decision: str | None = None,
        progress_detail: str | None = None,
        progress_error_type: str | None = None,
        progress_retryable: bool | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.operation = operation
        self.logical_table = logical_table
        self.logical_index = logical_index
        self.aws_exception_type = aws_exception_type
        self.aws_error_code = aws_error_code
        self.fallback_decision = fallback_decision
        self.progress_detail = progress_detail
        self.progress_error_type = progress_error_type
        self.progress_retryable = progress_retryable

    def safe_details(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "repositoryOperation": self.operation,
                "logicalTable": self.logical_table,
                "logicalIndex": self.logical_index,
                "awsExceptionType": self.aws_exception_type,
                "awsErrorCode": self.aws_error_code,
                "fallbackDecision": self.fallback_decision,
                "progressDetail": self.progress_detail,
                "progressErrorType": self.progress_error_type,
                "progressRetryable": (
                    None
                    if self.progress_retryable is None
                    else str(self.progress_retryable).casefold()
                ),
            }.items()
            if value
        }


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _item(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: _SERIALIZER.serialize(entry) for key, entry in value.items()}


def _plain(value: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {key: _DESERIALIZER.deserialize(entry) for key, entry in value.items()}


def _is_conditional_failure(exc: ClientError) -> bool:
    return exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException"


def _parse_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _attribute_value_size(value: dict[str, Any]) -> int:
    if "S" in value:
        return len(str(value["S"]).encode("utf-8")) + 1
    if "N" in value:
        return len(str(value["N"]).encode("utf-8")) + 1
    if "B" in value:
        return len(bytes(value["B"])) + 1
    if "BOOL" in value or "NULL" in value:
        return 2
    if "SS" in value or "NS" in value or "BS" in value:
        key = next(key for key in ("SS", "NS", "BS") if key in value)
        return 3 + sum(
            _attribute_value_size(
                {
                    {"SS": "S", "NS": "N", "BS": "B"}[key]: entry,
                }
            )
            + 1
            for entry in value[key]
        )
    if "L" in value:
        return 3 + sum(_attribute_value_size(entry) + 1 for entry in value["L"])
    if "M" in value:
        return 3 + sum(
            len(str(name).encode("utf-8")) + _attribute_value_size(entry) + 1
            for name, entry in value["M"].items()
        )
    raise PracticeRepositoryError("DYNAMODB_ITEM_SIZE_ESTIMATION_FAILED")


def estimate_dynamodb_item_size(item: dict[str, Any]) -> int:
    """Conservatively estimate DynamoDB item bytes from encoded attribute values."""
    encoded = _item(item)
    return sum(
        len(name.encode("utf-8")) + _attribute_value_size(value) + 1
        for name, value in encoded.items()
    )


@dataclass(frozen=True)
class IndexedQueryResult:
    items: tuple[dict[str, Any], ...]
    page_count: int
    has_more_pages: bool
    consumed_capacity: float | None
    duration_ms: int
    continuation_key: dict[str, Any] | None = None
    evaluated_count: int = 0


class AssessmentRepository:
    def __init__(
        self,
        client: Any,
        *,
        table_name: str,
        meta_safe_size_bytes: int = 280_000,
    ) -> None:
        self._client = client
        self._table = table_name
        self._meta_safe_size_bytes = min(max(meta_safe_size_bytes, 64_000), 290_000)

    def _guard_meta(self, meta: dict[str, Any]) -> None:
        intended = dict(_ASSESSMENT_SIZE_ENVELOPE)
        intended["meta"] = meta
        if estimate_dynamodb_item_size(intended) >= self._meta_safe_size_bytes:
            raise PracticeRepositoryError("ASSESSMENT_META_SIZE_LIMIT_EXCEEDED")

    def _guard_item(self, item: dict[str, Any]) -> None:
        if estimate_dynamodb_item_size(item) >= self._meta_safe_size_bytes:
            raise PracticeRepositoryError("ASSESSMENT_ITEM_SIZE_LIMIT_EXCEEDED")

    def get(self, test_id: str, *, consistent: bool = True) -> dict[str, Any] | None:
        try:
            response = self._client.get_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                ConsistentRead=consistent,
            )
        except ClientError as exc:
            raise PracticeRepositoryError("ASSESSMENT_READ_FAILED") from exc
        raw = response.get("Item")
        return _plain(raw) if raw else None

    def create_or_get(
        self,
        test_id: str,
        request: PracticeGenerationRequest,
    ) -> tuple[dict[str, Any], bool]:
        timestamp = _now()
        key = request_idempotency_key(request)
        progress = AssessmentProgress(
            requested_count=request.requested_count,
            accepted_count=request.accepted_count,
        )
        meta = {
            "schemaVersion": "1",
            "idempotencyKey": key,
            "conversationId": request.conversation_id,
            "turnId": request.turn_id,
            "practiceType": request.practice_type.value,
            "requestedCount": request.requested_count,
            "acceptedCount": request.accepted_count,
            "examStage": request.exam_stage,
            "requestedLanguage": request.language,
            "phase": progress.phase.value,
            "playable": False,
            "progressPercent": 0,
            "readyQuestionCount": 0,
            "questionManifestVersion": 1,
            "readyQuestionIds": [],
            "readyCount": 0,
            "reusedCount": 0,
            "generatedCount": 0,
            "verifiedCount": 0,
            "failedCount": 0,
            "generationVersion": 1,
            "startedAt": timestamp,
            "lastProgressAt": timestamp,
            "recoveryAttemptCount": 0,
            "lastCompletedStage": "INITIALIZED",
            "replacementWaveCount": 0,
            "slotReadyCounts": {},
            "activeExecutionId": None,
            "executionLeaseExpiresAt": None,
            "executionStartedAt": None,
            "executionAttempt": 0,
            "cancelRequested": False,
            "resumeReason": None,
            "practiceRequest": {
                # Provenance for post-hoc correctness audit and replanning only. This
                # is never a retrieval or matching key: reuse continues to match on the
                # structured slot constraints alone.
                "originalQuery": request.original_query,
                "requestId": request.request_id,
                "conversationId": request.conversation_id,
                "turnId": request.turn_id,
                "practiceType": request.practice_type.value,
                "requestedCount": request.requested_count,
                "acceptedCount": request.accepted_count,
                "subject": request.subject,
                "topic": request.topic,
                "topics": request.topics,
                "difficulty": request.difficulty.value,
                "mixedDifficultyRequested": request.mixed_difficulty_requested,
                "explicitDifficultyRequested": request.explicit_difficulty_requested,
                "difficultyDistribution": (
                    {level.value: count for level, count in request.difficulty_distribution.items()}
                    if request.difficulty_distribution is not None
                    else None
                ),
                "language": request.language,
                "languageSource": request.language_source,
                "examId": request.exam_id,
                "examStage": request.exam_stage,
                "examProfileId": request.exam_profile_id,
                "sourceQuestionReference": request.source_question_reference,
                "requiresFreshEvidence": request.requires_fresh_evidence,
                "freshnessReason": request.freshness_reason,
                "freshEvidence": (
                    request.fresh_evidence.model_dump(mode="json")
                    if request.fresh_evidence is not None
                    else None
                ),
                "includeSolutions": request.include_solutions,
                "assessmentTitle": request.assessment_title,
            },
            "resourceAliases": {
                "assessment": "MockTestQuiz",
                "questions": "Question",
                "questionBank": "QuestionBank",
            },
        }
        is_mock = request.practice_type.value in {
            "SECTIONAL_TEST",
            "FULL_MOCK",
            "TOPIC_TEST",
        }
        item = {
            "testId": test_id,
            "userId": request.user_id,
            "name": request.assessment_title,
            "subject": _SUBJECTS.get(request.subject.casefold(), "OTHER"),
            "topic": request.topic or request.subject,
            "type": "mockTest" if is_mock else "quiz",
            "activityKind": "MINI_MOCK" if is_mock else "QUICK_QUIZ",
            # AI Tutor launch currently supports practice conditions only. This is an
            # explicit server-owned contract, never a browser-selected inference.
            "assessmentMode": "PRACTICE",
            **({"examProfileId": request.exam_profile_id} if request.exam_profile_id else {}),
            "status": "GENERATING",
            "visibility": "PRIVATE",
            "origin": "AI_CUSTOM",
            "language": _storage_language(request.language),
            "durationMinutes": max(1, min(request.accepted_count * 2, 180)),
            "totalQuestions": request.accepted_count,
            "exams": [request.exam_id] if request.exam_id else ["General"],
            "live": False,
            "meta": meta,
            "createdAt": timestamp,
            "updatedAt": timestamp,
            "__typename": "MockTestQuiz",
        }
        self._guard_meta(meta)
        self._guard_item(item)
        try:
            self._client.put_item(
                TableName=self._table,
                Item=_item(item),
                ConditionExpression="attribute_not_exists(testId)",
            )
            return item, False
        except ClientError as exc:
            if not _is_conditional_failure(exc):
                raise PracticeRepositoryError("ASSESSMENT_CREATE_FAILED") from exc
        existing = self.get(test_id)
        if existing is None or _parse_meta(existing.get("meta")).get("idempotencyKey") != key:
            raise PracticeRepositoryError("ASSESSMENT_IDEMPOTENCY_CONFLICT")
        return existing, True

    def update(
        self,
        test_id: str,
        *,
        meta_updates: dict[str, Any],
        status: str | None = None,
        live: bool | None = None,
        expected_updated_at: str | None = None,
    ) -> dict[str, Any]:
        self._guard_meta(meta_updates)
        timestamp = _now()
        names = {"#meta": "meta", "#updated": "updatedAt"}
        values: dict[str, Any] = {":updated": timestamp}
        assignments = ["#updated = :updated"]
        for index, (key, value) in enumerate(meta_updates.items()):
            name_key = f"#field{index}"
            value_key = f":value{index}"
            names[name_key] = key
            values[value_key] = value
            assignments.append(f"#meta.{name_key} = {value_key}")
        if status is not None:
            names["#status"] = "status"
            values[":status"] = status
            assignments.append("#status = :status")
        if live is not None:
            names["#live"] = "live"
            values[":live"] = live
            assignments.append("#live = :live")
        condition = "attribute_exists(testId)"
        if expected_updated_at is not None:
            values[":previous"] = expected_updated_at
            condition += " AND #updated = :previous"
        try:
            response = self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression="SET " + ", ".join(assignments),
                ConditionExpression=condition,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=_item(values),
                ReturnValues="ALL_NEW",
            )
            return _plain(response["Attributes"])
        except ClientError as exc:
            if _is_conditional_failure(exc):
                raise PracticeRepositoryError("ASSESSMENT_CONCURRENT_UPDATE") from exc
            raise PracticeRepositoryError("ASSESSMENT_UPDATE_FAILED") from exc

    def renew_execution_lease(
        self,
        test_id: str,
        execution_id: str,
        lease_expires_at: str,
    ) -> bool:
        """Renew one coarse lease only while this execution still owns the assessment."""
        try:
            self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression=(
                    "SET #meta.#lease = :lease, #meta.#last = :now, #updated = :now"
                ),
                ConditionExpression=(
                    "#status = :generating AND #meta.#execution = :execution "
                    "AND #meta.#cancel = :false"
                ),
                ExpressionAttributeNames={
                    "#meta": "meta",
                    "#lease": "executionLeaseExpiresAt",
                    "#last": "lastProgressAt",
                    "#execution": "activeExecutionId",
                    "#cancel": "cancelRequested",
                    "#status": "status",
                    "#updated": "updatedAt",
                },
                ExpressionAttributeValues=_item(
                    {
                        ":lease": lease_expires_at,
                        ":now": _now(),
                        ":execution": execution_id,
                        ":false": False,
                        ":generating": "GENERATING",
                    }
                ),
            )
            return True
        except ClientError as exc:
            if _is_conditional_failure(exc):
                return False
            raise PracticeRepositoryError("PRACTICE_EXECUTION_LEASE_UPDATE_FAILED") from exc

    def release_execution(self, test_id: str, execution_id: str) -> bool:
        """Release only the matching execution; a stale executor cannot release its successor."""
        try:
            self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression=(
                    "SET #meta.#execution = :empty, #meta.#lease = :empty, "
                    "#meta.#last = :now, #updated = :now"
                ),
                ConditionExpression="#meta.#execution = :execution",
                ExpressionAttributeNames={
                    "#meta": "meta",
                    "#execution": "activeExecutionId",
                    "#lease": "executionLeaseExpiresAt",
                    "#last": "lastProgressAt",
                    "#updated": "updatedAt",
                },
                ExpressionAttributeValues=_item(
                    {":execution": execution_id, ":empty": None, ":now": _now()}
                ),
            )
            return True
        except ClientError as exc:
            if _is_conditional_failure(exc):
                return False
            raise PracticeRepositoryError("PRACTICE_EXECUTION_RELEASE_FAILED") from exc

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
        )

    def mark_failed(self, test_id: str, error_code: str) -> None:
        timestamp = _now()
        execution_id = current_practice_execution_id()
        condition = "attribute_exists(testId) AND #status = :generating_status"
        names = {
            "#meta": "meta",
            "#phase": "phase",
            "#playable": "playable",
            "#error": "errorCode",
            "#failed": "failedCount",
            "#status": "status",
            "#live": "live",
            "#updated": "updatedAt",
        }
        values: dict[str, Any] = {
            ":phase": InternalPhase.FAILED.value,
            ":false": False,
            ":error": error_code,
            ":zero": 0,
            ":one": 1,
            ":failed_status": "FAILED",
            ":generating_status": "GENERATING",
            ":updated": timestamp,
        }
        if execution_id is not None:
            condition += " AND #meta.#execution = :execution"
            names["#execution"] = "activeExecutionId"
            values[":execution"] = execution_id
        try:
            self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression=(
                    "SET #meta.#phase = :phase, #meta.#playable = :false, "
                    "#meta.#error = :error, "
                    "#meta.#failed = if_not_exists(#meta.#failed, :zero) + :one, "
                    "#status = :failed_status, #live = :false, #updated = :updated"
                ),
                ConditionExpression=condition,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=_item(values),
            )
        except ClientError as exc:
            if not _is_conditional_failure(exc):
                raise PracticeRepositoryError("ASSESSMENT_FAILURE_UPDATE_FAILED") from exc

    def claim_group(self, test_id: str, group_id: str) -> dict[str, Any] | None:
        now_value = _now()
        names = {
            "#meta": "meta",
            "#groups": "generationGroups",
            "#group": group_id,
            "#state": "state",
            "#claims": "claimCount",
            "#phase": "phase",
            "#status": "status",
            "#updated": "updatedAt",
        }
        try:
            response = self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression=(
                    "SET #meta.#groups.#group.#state = :running, "
                    "#meta.#groups.#group.#claims = "
                    "if_not_exists(#meta.#groups.#group.#claims, :zero) + :one, "
                    "#meta.#phase = :phase, #updated = :now"
                ),
                ConditionExpression=(
                    "#status = :generating AND "
                    "attribute_exists(#meta.#groups.#group) AND "
                    "#meta.#groups.#group.#state = :pending"
                ),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=_item(
                    {
                        ":running": "RUNNING",
                        ":pending": "PENDING",
                        ":generating": "GENERATING",
                        ":now": now_value,
                        ":zero": 0,
                        ":one": 1,
                        ":phase": InternalPhase.GENERATING.value,
                    }
                ),
                ReturnValues="ALL_NEW",
            )
        except ClientError as exc:
            if _is_conditional_failure(exc):
                return None
            raise PracticeRepositoryError("GENERATION_GROUP_CLAIM_FAILED") from exc
        meta = _parse_meta(_plain(response["Attributes"]).get("meta"))
        group = meta.get("generationGroups", {}).get(group_id)
        return dict(group) if isinstance(group, dict) else None

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
        timestamp = _now()
        group = {
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
        try:
            self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression=("SET #meta.#groups.#group = :group, #updated = :updated"),
                ConditionExpression=(
                    "#status = :generating AND "
                    "attribute_not_exists(#meta.#groups.#group) AND "
                    "#meta.#groups.#parent.#state = :running"
                ),
                ExpressionAttributeNames={
                    "#meta": "meta",
                    "#groups": "generationGroups",
                    "#group": group_id,
                    "#parent": parent_group_id,
                    "#state": "state",
                    "#status": "status",
                    "#updated": "updatedAt",
                },
                ExpressionAttributeValues=_item(
                    {
                        ":group": group,
                        ":generating": "GENERATING",
                        ":running": "RUNNING",
                        ":updated": timestamp,
                    }
                ),
            )
            return True
        except ClientError as exc:
            if _is_conditional_failure(exc):
                return False
            raise PracticeRepositoryError("GENERATION_RETRY_GROUP_CREATE_FAILED") from exc

    def record_finalization_retry(
        self,
        test_id: str,
        *,
        next_attempt: int,
        reason_code: str,
    ) -> bool:
        try:
            self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression=(
                    "SET #meta.#phase = :phase, #meta.#final_attempt = :attempt, "
                    "#meta.#final_reason = :reason, #updated = :updated"
                ),
                ConditionExpression=(
                    "#status = :generating AND "
                    "(attribute_not_exists(#meta.#final_attempt) OR "
                    "#meta.#final_attempt < :attempt)"
                ),
                ExpressionAttributeNames={
                    "#meta": "meta",
                    "#phase": "phase",
                    "#final_attempt": "finalizationAttempt",
                    "#final_reason": "finalizationReasonCode",
                    "#status": "status",
                    "#updated": "updatedAt",
                },
                ExpressionAttributeValues=_item(
                    {
                        ":phase": InternalPhase.FINALIZING.value,
                        ":attempt": next_attempt,
                        ":reason": reason_code,
                        ":generating": "GENERATING",
                        ":updated": _now(),
                    }
                ),
            )
            return True
        except ClientError as exc:
            if _is_conditional_failure(exc):
                return False
            raise PracticeRepositoryError("FINALIZATION_RETRY_RECORD_FAILED") from exc

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
        names = {
            "#meta": "meta",
            "#groups": "generationGroups",
            "#group": group_id,
            "#state": "state",
            "#updated": "updatedAt",
        }
        values: dict[str, Any] = {
            ":state": state,
            ":updated": _now(),
        }
        condition = "attribute_exists(#meta.#groups.#group)"
        assignments = [
            "#meta.#groups.#group.#state = :state",
            "#updated = :updated",
        ]
        for field, value in {
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
                alias = f"#{field}"
                token = f":{field}"
                names[alias] = field
                values[token] = value
                assignments.append(f"#meta.#groups.#group.{alias} = {token}")
        try:
            self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression="SET " + ", ".join(assignments),
                ConditionExpression=condition,
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=_item(values),
            )
        except ClientError as exc:
            if _is_conditional_failure(exc):
                raise PracticeRepositoryError("GENERATION_GROUP_NOT_FOUND") from exc
            raise PracticeRepositoryError("GENERATION_GROUP_UPDATE_FAILED") from exc

    def mark_ready(self, test_id: str, accepted_count: int, meta_updates: dict[str, Any]) -> bool:
        self._guard_meta(meta_updates)
        names = {
            "#meta": "meta",
            "#ready_count": "readyQuestionCount",
            "#manifest_ready": "readyCount",
            "#manifest": "readyQuestionIds",
            "#failed_count": "failedCount",
            "#status": "status",
            "#live": "live",
            "#updated": "updatedAt",
        }
        values: dict[str, Any] = {
            ":accepted": accepted_count,
            ":zero": 0,
            ":generating": "GENERATING",
            ":ready_status": "READY",
            ":true": True,
            ":updated": _now(),
        }
        assignments = [
            "#status = :ready_status",
            "#live = :true",
            "#updated = :updated",
        ]
        for index, (key, value) in enumerate(meta_updates.items()):
            name = f"#ready_field{index}"
            token = f":ready_value{index}"
            names[name] = key
            values[token] = value
            assignments.append(f"#meta.{name} = {token}")
        try:
            self._client.update_item(
                TableName=self._table,
                Key=_item({"testId": test_id}),
                UpdateExpression="SET " + ", ".join(assignments),
                ConditionExpression=(
                    "#status = :generating AND "
                    "#meta.#ready_count = :accepted AND "
                    "#meta.#manifest_ready = :accepted AND "
                    "size(#meta.#manifest) = :accepted AND "
                    "#meta.#failed_count = :zero"
                ),
                ExpressionAttributeNames=names,
                ExpressionAttributeValues=_item(values),
            )
            return True
        except ClientError as exc:
            if _is_conditional_failure(exc):
                return False
            raise PracticeRepositoryError("ASSESSMENT_READY_UPDATE_FAILED") from exc

    def cancel(self, test_id: str) -> None:
        self.update(
            test_id,
            meta_updates={
                "phase": InternalPhase.CANCELLED.value,
                "playable": False,
            },
            status="ARCHIVED",
            live=False,
        )


class QuestionRepository:
    def __init__(
        self,
        client: Any,
        *,
        assessment_table: str,
        question_table: str,
        question_bank_table: str,
        question_test_index: str,
        question_bank_category_index: str,
        pattern_context_enabled: bool = False,
        pattern_reuse_enabled: bool = False,
        practice_attempt_table: str = "",
        practice_attempt_user_index: str = "",
        question_bank_reuse_index: str = "",
        question_bank_reuse_projection: IndexProjection | None = None,
        question_bank_category_projection: IndexProjection | None = None,
        query_page_size: int = 25,
        query_max_pages: int = 2,
        query_latency_warning_ms: int = 1_000,
    ) -> None:
        self._client = client
        self._assessment_table = assessment_table
        self._question_table = question_table
        self._question_bank_table = question_bank_table
        self._question_test_index = question_test_index
        self._question_bank_category_index = question_bank_category_index
        self._pattern_context_enabled = pattern_context_enabled
        self._pattern_reuse_enabled = pattern_reuse_enabled
        self._practice_attempt_table = practice_attempt_table
        self._practice_attempt_user_index = practice_attempt_user_index
        self._question_bank_reuse_index = question_bank_reuse_index
        self._question_bank_reuse_projection = question_bank_reuse_projection
        self._question_bank_category_projection = question_bank_category_projection
        self._query_page_size = min(max(query_page_size, 1), 50)
        self._query_max_pages = min(max(query_max_pages, 1), 5)
        self._query_latency_warning_ms = min(
            max(query_latency_warning_ms, 100),
            10_000,
        )

    def _query_reuse_candidates(
        self,
        *,
        index_name: str,
        key_name: str,
        key_value: str,
        projection: IndexProjection | None,
        limit: int,
        sort_key_name: str | None = None,
        sort_key_prefix: str | None = None,
        exclusive_start_key: dict[str, Any] | None = None,
    ) -> IndexedQueryResult:
        if not index_name:
            raise PracticeRepositoryError("QUESTION_BANK_INDEX_NOT_CONFIGURED")
        remaining = min(max(limit, 1), 100)
        items: list[dict[str, Any]] = []
        start_key = exclusive_start_key
        page_count = 0
        evaluated_count = 0
        consumed_capacity = 0.0
        consumed_reported = False
        started = time.monotonic()
        try:
            while remaining and page_count < self._query_max_pages:
                kwargs: dict[str, Any] = {
                    "TableName": self._question_bank_table,
                    "IndexName": index_name,
                    "KeyConditionExpression": "#lookup = :lookup",
                    "ExpressionAttributeNames": {"#lookup": key_name},
                    "ExpressionAttributeValues": _item({":lookup": key_value}),
                    "Limit": min(remaining, self._query_page_size),
                    "ScanIndexForward": False,
                    "ReturnConsumedCapacity": "TOTAL",
                }
                if sort_key_prefix is not None:
                    if not sort_key_name:
                        raise PracticeRepositoryError("QUESTION_BANK_SORT_KEY_NOT_CONFIGURED")
                    kwargs["KeyConditionExpression"] += (
                        " AND begins_with(#sort_key, :sort_key_prefix)"
                    )
                    kwargs["ExpressionAttributeNames"]["#sort_key"] = sort_key_name
                    kwargs["ExpressionAttributeValues"].update(
                        _item({":sort_key_prefix": sort_key_prefix})
                    )
                metadata_projected = (
                    projection is None or projection.query_mode == "PROJECTED_METADATA"
                )
                if metadata_projected:
                    kwargs["ProjectionExpression"] = (
                        "qbId, category, difficulty, tags, #meta, updatedAt, #source"
                    )
                    kwargs["ExpressionAttributeNames"].update(
                        {"#meta": "meta", "#source": "source"}
                    )
                else:
                    kwargs["ProjectionExpression"] = "qbId"
                if start_key:
                    kwargs["ExclusiveStartKey"] = start_key
                response = self._client.query(**kwargs)
                page_count += 1
                page = [_plain(raw) for raw in response.get("Items", [])]
                evaluated_count += len(page)
                if metadata_projected:
                    items.extend(page)
                else:
                    hydrated = self.get_reuse_candidates(
                        [str(item.get("qbId") or "") for item in page]
                    )
                    items.extend(hydrated)
                remaining -= len(page)
                start_key = response.get("LastEvaluatedKey")
                capacity = response.get("ConsumedCapacity")
                if isinstance(capacity, dict) and capacity.get("CapacityUnits") is not None:
                    consumed_capacity += float(capacity["CapacityUnits"])
                    consumed_reported = True
                if not start_key or not page:
                    break
        except ClientError as exc:
            error = exc.response.get("Error", {})
            emit_practice_event(
                "PRACTICE_REPOSITORY_FAILURE",
                test_id="runtime",
                status="failed",
                details={
                    "reasonCode": "QUESTION_BANK_QUERY_FAILED",
                    "operation": "Query",
                    "logicalTable": "QuestionBank",
                    "logicalIndex": (
                        "QuestionBank.reuse"
                        if index_name == self._question_bank_reuse_index
                        else "QuestionBank.category"
                    ),
                    "awsExceptionType": type(exc).__name__,
                    "awsErrorCode": str(error.get("Code") or "unknown"),
                    "patternContextEnabled": self._pattern_context_enabled,
                    "patternReuseEnabled": self._pattern_reuse_enabled,
                    "fallbackDecision": "fatal_legacy_repository_failure",
                },
                level=logging.ERROR,
            )
            raise PracticeRepositoryError(
                "QUESTION_BANK_QUERY_FAILED",
                operation="dynamodb.Query",
                logical_table="QuestionBank",
                logical_index=(
                    "QuestionBank.reuse"
                    if index_name == self._question_bank_reuse_index
                    else "QuestionBank.category"
                ),
                aws_exception_type=type(exc).__name__,
                aws_error_code=str(error.get("Code") or "unknown"),
                fallback_decision="fatal_legacy_repository_failure",
            ) from exc
        duration_ms = int((time.monotonic() - started) * 1000)
        if (
            page_count >= self._query_max_pages and start_key
        ) or duration_ms >= self._query_latency_warning_ms:
            logger.warning(
                "bounded practice QuestionBank query warning pages=%d duration_ms=%d",
                page_count,
                duration_ms,
            )
        return IndexedQueryResult(
            items=tuple(items[:limit]),
            page_count=page_count,
            has_more_pages=bool(start_key),
            consumed_capacity=consumed_capacity if consumed_reported else None,
            duration_ms=duration_ms,
            continuation_key=start_key,
            evaluated_count=evaluated_count,
        )

    def query_topic_reuse_candidates(
        self,
        *,
        reuse_bucket_key: str,
        limit: int,
        difficulty_prefix: str | None = None,
        exclusive_start_key: dict[str, Any] | None = None,
    ) -> IndexedQueryResult:
        return self._query_reuse_candidates(
            index_name=self._question_bank_reuse_index,
            key_name="reuseBucketKey",
            key_value=reuse_bucket_key,
            projection=self._question_bank_reuse_projection,
            limit=limit,
            sort_key_name="reuseSortKey",
            sort_key_prefix=difficulty_prefix,
            exclusive_start_key=exclusive_start_key,
        )

    def query_reuse_candidates(
        self,
        *,
        category: str,
        limit: int,
        exclusive_start_key: dict[str, Any] | None = None,
    ) -> IndexedQueryResult:
        return self._query_reuse_candidates(
            index_name=self._question_bank_category_index,
            key_name="category",
            key_value=category,
            projection=self._question_bank_category_projection,
            limit=limit,
            exclusive_start_key=exclusive_start_key,
        )

    def get_reuse_candidates(
        self,
        question_ids: list[str],
    ) -> list[dict[str, Any]]:
        unique_ids = list(dict.fromkeys(question_ids))[:100]
        if not unique_ids:
            return []
        request_items = {
            self._question_bank_table: {
                "Keys": [_item({"qbId": question_id}) for question_id in unique_ids],
                "ConsistentRead": True,
            }
        }
        records: list[dict[str, Any]] = []
        try:
            for _attempt in range(2):
                response = self._client.batch_get_item(RequestItems=request_items)
                records.extend(
                    dict(response.get("Responses") or {}).get(
                        self._question_bank_table,
                        [],
                    )
                )
                unprocessed = dict(response.get("UnprocessedKeys") or {})
                if not unprocessed:
                    break
                request_items = unprocessed
        except ClientError as exc:
            error = exc.response.get("Error", {})
            raise PracticeRepositoryError(
                "QUESTION_BANK_BATCH_GET_FAILED",
                operation="dynamodb.BatchGetItem",
                logical_table="QuestionBank",
                aws_exception_type=type(exc).__name__,
                aws_error_code=str(error.get("Code") or "unknown"),
                fallback_decision="fatal_legacy_repository_failure",
            ) from exc
        return [_plain(raw) for raw in records]

    def list_recent_seen_question_bank_ids(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> set[str]:
        """Resolve bounded cross-activity QuestionBank history without scans or N+1 reads."""
        if not self._practice_attempt_table or not self._practice_attempt_user_index:
            raise PracticeRepositoryError("PRACTICE_HISTORY_INDEX_NOT_CONFIGURED")
        try:
            response = self._client.query(
                TableName=self._practice_attempt_table,
                IndexName=self._practice_attempt_user_index,
                KeyConditionExpression="#user = :user",
                ExpressionAttributeNames={"#user": "userId"},
                ExpressionAttributeValues=_item({":user": user_id}),
                ProjectionExpression="activityId",
                Limit=min(max(limit, 1), 50),
                ScanIndexForward=False,
            )
            activity_ids = list(
                dict.fromkeys(
                    str(_plain(item).get("activityId") or "")
                    for item in response.get("Items", [])
                    if str(_plain(item).get("activityId") or "")
                )
            )[:50]
            if not activity_ids:
                return set()
            assessments = self._client.batch_get_item(
                RequestItems={
                    self._assessment_table: {
                        "Keys": [_item({"testId": value}) for value in activity_ids],
                        "ConsistentRead": True,
                    }
                }
            )
            if dict(assessments.get("UnprocessedKeys") or {}):
                raise PracticeRepositoryError("PRACTICE_HISTORY_BATCH_INCOMPLETE")
            question_ids: list[str] = []
            for raw in dict(assessments.get("Responses") or {}).get(
                self._assessment_table,
                [],
            ):
                meta = _parse_meta(_plain(raw).get("meta"))
                question_ids.extend(
                    str(value)
                    for value in list(meta.get("readyQuestionIds") or [])
                    if str(value)
                )
            questions = self.get_questions_by_ids(list(dict.fromkeys(question_ids))[:100])
            seen: set[str] = set()
            for question in questions:
                source_id = _parse_meta(question.get("meta")).get("sourceQuestionBankId")
                if source_id:
                    seen.add(str(source_id))
            return seen
        except ClientError as exc:
            raise PracticeRepositoryError("PRACTICE_HISTORY_READ_FAILED") from exc

    def list_linked(
        self,
        test_id: str,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        start_key: dict[str, Any] | None = None
        page_count = 0
        consumed_capacity = 0.0
        consumed_reported = False
        started = time.monotonic()
        try:
            while len(results) < min(max(limit, 1), 100) and page_count < 5:
                kwargs: dict[str, Any] = {
                    "TableName": self._question_table,
                    "IndexName": self._question_test_index,
                    "KeyConditionExpression": "#test = :test",
                    "ExpressionAttributeNames": {"#test": "testId"},
                    "ExpressionAttributeValues": _item({":test": test_id}),
                    "Limit": min(25, max(limit - len(results), 1)),
                    "ConsistentRead": False,
                    "ReturnConsumedCapacity": "TOTAL",
                }
                if start_key:
                    kwargs["ExclusiveStartKey"] = start_key
                response = self._client.query(**kwargs)
                page_count += 1
                for raw in response.get("Items", []):
                    item = _plain(raw)
                    item["_practiceMeta"] = _parse_meta(item.get("meta"))
                    results.append(item)
                capacity = response.get("ConsumedCapacity")
                if isinstance(capacity, dict) and capacity.get("CapacityUnits") is not None:
                    consumed_capacity += float(capacity["CapacityUnits"])
                    consumed_reported = True
                start_key = response.get("LastEvaluatedKey")
                if not start_key or not response.get("Items"):
                    break
        except ClientError as exc:
            error = exc.response.get("Error", {})
            raise PracticeRepositoryError(
                "ASSESSMENT_QUESTIONS_QUERY_FAILED",
                operation="dynamodb.Query",
                logical_table="Question",
                logical_index="Question.testId",
                aws_exception_type=type(exc).__name__,
                aws_error_code=str(error.get("Code") or "unknown"),
                fallback_decision="fatal_legacy_repository_failure",
            ) from exc
        duration_ms = int((time.monotonic() - started) * 1000)
        emit_practice_event(
            "ASSESSMENT_QUESTIONS_QUERY_COMPLETED",
            test_id=test_id,
            status="completed",
            details={
                "logicalTable": "Question",
                "logicalIndex": "Question.testId",
                "operation": "list_linked",
                "pageCount": page_count,
                "candidateCount": len(results),
                "selectedCount": len(results),
                "hasMorePages": bool(start_key),
                "durationMs": duration_ms,
                "consumedCapacity": (consumed_capacity if consumed_reported else None),
            },
        )
        return results[:limit]

    def list_playable(
        self,
        test_id: str,
        *,
        user_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        try:
            response = self._client.get_item(
                TableName=self._assessment_table,
                Key=_item({"testId": test_id}),
                ConsistentRead=True,
            )
        except ClientError as exc:
            raise PracticeRepositoryError("ASSESSMENT_READ_FAILED") from exc
        assessment = _plain(response["Item"]) if response.get("Item") else {}
        meta = _parse_meta(assessment.get("meta"))
        if (
            assessment.get("status") != "READY"
            or assessment.get("userId") != user_id
            or assessment.get("live") is not True
            or meta.get("playable") is not True
        ):
            return []
        accepted_count = int(meta.get("acceptedCount") or 0)
        manifest_count = int(meta.get("readyCount") or 0)
        manifest_ids = [
            str(question_id)
            for question_id in list(meta.get("readyQuestionIds") or [])
            if str(question_id)
        ]
        if (
            accepted_count < 1
            or manifest_count != accepted_count
            or len(manifest_ids) != accepted_count
            or len(set(manifest_ids)) != accepted_count
        ):
            raise PracticeRepositoryError("TEST_CONTENT_NOT_YET_CONSISTENT")
        linked = self.list_linked(test_id, limit=limit)
        linked_ids = {
            str(item.get("questionId") or "")
            for item in linked
            if str(item.get("questionId") or "")
        }
        if len(linked) == accepted_count and linked_ids == set(manifest_ids):
            by_id = {str(item.get("questionId") or ""): item for item in linked}
        else:
            resolved = self.get_questions_by_ids(manifest_ids)
            by_id = {
                str(item.get("questionId") or ""): item
                for item in resolved
                if str(item.get("testId") or "") == test_id
            }
            if len(by_id) != accepted_count:
                raise PracticeRepositoryError("TEST_CONTENT_NOT_YET_CONSISTENT")
        practice_request = meta.get("practiceRequest")
        requested_language = (
            str(practice_request.get("language") or "")
            if isinstance(practice_request, dict)
            else ""
        )
        if not requested_language:
            raise PracticeRepositoryError("TEST_CONTENT_NOT_YET_CONSISTENT")
        requires_storage_language = isinstance(practice_request, dict) and (
            "languageSource" in practice_request
        )
        if requires_storage_language and assessment.get("language") != _storage_language(
            requested_language
        ):
            raise PracticeRepositoryError("TEST_CONTENT_NOT_YET_CONSISTENT")
        solution_required = bool(practice_request.get("includeSolutions", True))
        playable: list[dict[str, Any]] = []
        for question_id in manifest_ids:
            item = by_id[question_id]
            item["_practiceMeta"] = _parse_meta(item.get("meta"))
            contract = validate_persisted_playable_question(
                item,
                expected_question_type="mcq",
                expected_language=requested_language,
                solution_required=solution_required,
                expected_storage_language=(
                    _storage_language(requested_language)
                    if requires_storage_language
                    else None
                ),
            )
            if not contract.valid:
                raise PracticeRepositoryError("TEST_CONTENT_NOT_YET_CONSISTENT")
            playable.append(item)
        return sorted(
            playable,
            key=lambda item: int(item.get("position") or 0),
        )

    def inspect_reusable_slots(
        self,
        test_id: str,
        *,
        language: str,
        solution_required: bool,
    ) -> tuple[set[str], tuple[str, ...]]:
        """Reuse the canonical playable validator for durable resume reconstruction."""
        valid_slot_ids: set[str] = set()
        invalid_question_ids: list[str] = []
        for item in self.list_linked(test_id):
            practice_meta = item.get("_practiceMeta")
            if not isinstance(practice_meta, dict):
                practice_meta = _parse_meta(item.get("meta"))
            slot_id = str(practice_meta.get("slotId") or "")
            if not slot_id:
                continue
            item["_practiceMeta"] = practice_meta
            # Schema-v2 questions are authored without a solution by contract, so a
            # resume must not invalidate slots that were already verified. v1/legacy
            # items keep the original requirement.
            item_schema_v2 = str(practice_meta.get("schemaVersion") or "") == "2"
            contract = validate_persisted_playable_question(
                item,
                expected_question_type=str(practice_meta.get("questionType") or "mcq"),
                expected_language=language,
                solution_required=solution_required and not item_schema_v2,
            )
            if (
                contract.valid
                and practice_meta.get("verified") is True
                and practice_meta.get("status") == "READY"
            ):
                valid_slot_ids.add(slot_id)
            else:
                question_id = str(item.get("questionId") or "")
                if question_id:
                    invalid_question_ids.append(question_id)
        return valid_slot_ids, tuple(dict.fromkeys(invalid_question_ids))

    def delete_invalid_resume_questions(
        self,
        test_id: str,
        execution_id: str,
        question_ids: tuple[str, ...],
    ) -> None:
        """Delete only invalid assessment-owned rows under the new execution fence."""
        for question_id in question_ids:
            try:
                self._client.transact_write_items(
                    TransactItems=[
                        {
                            "ConditionCheck": {
                                "TableName": self._assessment_table,
                                "Key": _item({"testId": test_id}),
                                "ConditionExpression": "#meta.#execution = :execution",
                                "ExpressionAttributeNames": {
                                    "#meta": "meta",
                                    "#execution": "activeExecutionId",
                                },
                                "ExpressionAttributeValues": _item(
                                    {":execution": execution_id}
                                ),
                            }
                        },
                        {
                            "Delete": {
                                "TableName": self._question_table,
                                "Key": _item({"questionId": question_id}),
                                "ConditionExpression": "#test = :test",
                                "ExpressionAttributeNames": {"#test": "testId"},
                                "ExpressionAttributeValues": _item({":test": test_id}),
                            }
                        },
                    ]
                )
            except ClientError as exc:
                raise PracticeRepositoryError(
                    "PRACTICE_RESUME_INVALID_QUESTION_CLEANUP_FAILED"
                ) from exc

    def get_questions_by_ids(
        self,
        question_ids: list[str],
    ) -> list[dict[str, Any]]:
        unique_ids = list(dict.fromkeys(question_ids))[:100]
        if not unique_ids:
            return []
        records: list[dict[str, Any]] = []
        for offset in range(0, len(unique_ids), 100):
            request_items = {
                self._question_table: {
                    "Keys": [
                        _item({"questionId": question_id})
                        for question_id in unique_ids[offset : offset + 100]
                    ],
                    "ConsistentRead": True,
                }
            }
            try:
                for _attempt in range(2):
                    response = self._client.batch_get_item(RequestItems=request_items)
                    records.extend(
                        dict(response.get("Responses") or {}).get(
                            self._question_table,
                            [],
                        )
                    )
                    unprocessed = dict(response.get("UnprocessedKeys") or {})
                    if not unprocessed:
                        break
                    request_items = unprocessed
            except ClientError as exc:
                raise PracticeRepositoryError("ASSESSMENT_QUESTIONS_BATCH_GET_FAILED") from exc
        return [_plain(raw) for raw in records]

    def link_reused(
        self,
        *,
        test_id: str,
        bucket_id: str,
        question: ReusableQuestion,
        slot_id: str | None = None,
        trusted_pattern_selection: PatternSlotSelection | None = None,
    ) -> bool:
        pattern_id = None
        pattern_version_hash = None
        if (
            trusted_pattern_selection is not None
            and trusted_pattern_selection.tier.value == "REUSE_SAFE"
        ):
            pattern_id = trusted_pattern_selection.pattern_id
            pattern_version_hash = trusted_pattern_selection.pattern_version_hash
        question_id = deterministic_question_id(
            test_id,
            source_id=question.question_id,
            bucket_id=bucket_id,
        )
        return self._put_question(
            question_id=question_id,
            test_id=test_id,
            question=question.question,
            options=list(question.options),
            correct_answer=question.correct_answer,
            solution=question.solution,
            subject=question.subject,
            topic=question.topic,
            difficulty=question.difficulty,
            answer_contract={
                "schemaVersion": "2",
                "optionIdentity": "INDEX_V1",
                "options": [
                    {"optionId": index, "value": value}
                    for index, value in enumerate(question.options)
                ],
                "correctOptionId": list(question.options).index(question.correct_answer),
                "correctAnswer": question.correct_answer,
                "answerExplanation": question.solution,
                "answerStatus": "VERIFIED",
                "answerVersion": 1,
            },
            meta={
                "schemaVersion": "2" if slot_id else "1",
                "source": "REUSED",
                "sourceType": "QUESTION_BANK",
                "sourceQuestionBankId": question.question_id,
                "sourceVersion": question.source_updated_at,
                "bucketId": bucket_id,
                **({"slotId": slot_id} if slot_id else {}),
                "verified": True,
                "verificationPolicy": "STORED_AUTHORITY",
                "verificationMethod": "QUESTION_BANK_QUALITY",
                "questionType": question.question_type,
                "optionIdentity": "INDEX_V1",
                "language": question.language,
                "status": "READY",
            },
            pattern_id=pattern_id,
            pattern_version_hash=pattern_version_hash,
            language=question.language,
        )

    def link_generated(
        self,
        *,
        test_id: str,
        question: GeneratedQuestion,
        verified: bool,
        group_id: str,
        generator_route: str,
        generator_model: str,
        verification_policy: str,
        verification_method: str,
        language: str,
        source_question_bank_id: str | None = None,
        pattern_selection: PatternSlotSelection | None = None,
    ) -> bool:
        if not verified:
            raise PracticeRepositoryError("UNVERIFIED_QUESTION_REJECTED")
        question_id = deterministic_question_id(
            test_id,
            source_id=question.slot_id or question.generation_item_id,
            bucket_id=question.bucket_id,
        )
        canonical_solution = (
            question.answer_explanation
            if question.schema_version == "2"
            else question.solution
        )
        return self._put_question(
            question_id=question_id,
            test_id=test_id,
            question=question.question,
            options=question.options,
            correct_answer=question.correct_answer,
            solution=canonical_solution,
            subject=question.subject,
            topic=question.topic,
            difficulty=question.difficulty.value,
            answer_contract=(
                {
                    "schemaVersion": "2",
                    "optionIdentity": "INDEX_V1",
                    "options": [
                        {"optionId": int(option.option_id), "value": option.value}
                        for option in question.canonical_options
                    ],
                    "correctOptionId": int(question.correct_option_id),
                    "correctAnswer": question.correct_answer,
                    "answerExplanation": question.answer_explanation,
                    "answerStatus": "VERIFIED",
                    "answerVersion": 1,
                }
                if question.schema_version == "2"
                else None
            ),
            meta={
                "schemaVersion": question.schema_version,
                "source": "GENERATED",
                "sourceType": "AI_GENERATED",
                "generationItemId": question.generation_item_id,
                "generationGroupId": group_id,
                "generatorRoute": generator_route,
                "generatorModel": generator_model,
                "bucketId": question.bucket_id,
                **({"slotId": question.slot_id} if question.slot_id else {}),
                "verified": True,
                "verificationPolicy": verification_policy,
                "verificationMethod": verification_method,
                "questionType": question.question_type.value,
                "optionIdentity": "INDEX_V1",
                "language": language,
                "status": "READY",
                "reusable": False,
                "visibility": "PRIVATE",
                **(
                    {"sourceQuestionBankId": source_question_bank_id}
                    if source_question_bank_id
                    else {}
                ),
            },
            pattern_id=(
                pattern_selection.pattern_id if pattern_selection is not None else None
            ),
            pattern_version_hash=(
                pattern_selection.pattern_version_hash
                if pattern_selection is not None
                else None
            ),
            language=language,
        )

    def persist_verified_question(
        self,
        *,
        test_id: str,
        question: GeneratedQuestion,
        slot: PlannerSlot,
        language: str,
        pattern_id: str | None = None,
        pattern_version_hash: str | None = None,
    ) -> str | None:
        """Create an idempotent reusable QuestionBank row after verifier acceptance.

        Trust comes from independent verifier acceptance of system-generated content,
        so Pattern linkage is optional and only decorates the row when the server
        itself selected a Pattern for the slot.
        """
        reuse_bucket_key = build_slot_reuse_bucket_key(slot, language=language)
        difficulty_prefix = build_reuse_difficulty_prefix(slot.difficulty.value)
        if not reuse_bucket_key or not difficulty_prefix:
            return None
        qb_id = build_question_bank_id(
            subject=slot.subject_id,
            difficulty=slot.difficulty.value,
            exam_ids=slot.exam_ids,
            language=language,
            question_type=question.question_type.value,
            question=question.question,
            options=tuple(question.options),
            correct_answer=question.correct_answer,
        )
        if qb_id is None:
            return None
        pattern_linked = bool(pattern_id and pattern_version_hash)
        timestamp = _now()
        difficulty = {
            "basic": "EASY",
            "intermediate": "MEDIUM",
            "advanced": "HARD",
        }[slot.difficulty.value]
        item = {
            "qbId": qb_id,
            "question": question.question,
            "answers": _json({"options": question.options}),
            "correctAnswer": question.correct_answer,
            "explanation": question.answer_explanation or question.solution,
            "category": slot.category_id,
            "difficulty": difficulty,
            "source": (
                "PATTERN_VERIFIED_PRACTICE" if pattern_linked else "SYSTEM_VERIFIED_PRACTICE"
            ),
            "meta": {
                "status": "ACTIVE",
                "qualityStatus": "VERIFIED",
                "reusable": True,
                "visibility": "PLATFORM",
                "subject": slot.subject_id,
                "topic": slot.topic_id,
                "questionType": slot.question_type.value,
                "language": language,
                "examIds": list(slot.exam_ids),
                "patternFamilyId": slot.pattern_family_id,
                "confidence": 1,
                "solutionAvailable": True,
                "answerAvailable": True,
                "verificationMethod": "INDEPENDENT_MODEL_V2",
            },
            "reuseBucketKey": reuse_bucket_key,
            "reuseSortKey": (
                f"{difficulty_prefix}"
                f"{quote(timestamp.casefold(), safe='-_.!~*')}#"
                f"{quote(qb_id.casefold(), safe='-_.!~*')}"
            ),
            **(
                {
                    "patternId": pattern_id,
                    "patternVersionHash": pattern_version_hash,
                    "patternLinkEvidence": "VERIFIED_GENERATION",
                }
                if pattern_linked
                else {}
            ),
            "createdAt": timestamp,
            "updatedAt": timestamp,
            "__typename": "QuestionBank",
        }
        # Derived from the constructed row through the same function every reader
        # uses, so the writer cannot drift from recomputation.
        version_hash = question_bank_version_hash_from_item(item)
        if version_hash is None:
            return None
        item["versionHash"] = version_hash
        try:
            self._client.put_item(
                TableName=self._question_bank_table,
                Item=_item(item),
                ConditionExpression="attribute_not_exists(qbId)",
            )
            return qb_id
        except ClientError as exc:
            if _is_conditional_failure(exc):
                if pattern_linked:
                    self.attach_verified_pattern_link_if_absent(
                        test_id=test_id,
                        qb_id=qb_id,
                        pattern_id=str(pattern_id),
                        pattern_version_hash=str(pattern_version_hash),
                    )
                return qb_id
            raise PracticeRepositoryError("QUESTION_BANK_PATTERN_LINK_FAILED") from exc

    def attach_verified_pattern_link_if_absent(
        self,
        *,
        test_id: str,
        qb_id: str,
        pattern_id: str,
        pattern_version_hash: str,
    ) -> bool:
        """Attach authoritative Pattern evidence to an existing reusable Question.

        Enrichment only.  The condition keeps the write idempotent for the same
        Pattern, never replaces a different Pattern already recorded, and never
        creates a row.  Playable content, classification, and trust fields are
        outside the update expression by construction.
        """
        if not qb_id or not pattern_id or not pattern_version_hash:
            return False
        try:
            self._client.update_item(
                TableName=self._question_bank_table,
                Key=_item({"qbId": qb_id}),
                UpdateExpression=(
                    "SET patternId = :pattern, patternVersionHash = :version, "
                    "patternLinkEvidence = :evidence, updatedAt = :now"
                ),
                ConditionExpression=(
                    "attribute_exists(qbId) AND "
                    "(attribute_not_exists(patternId) OR patternId = :pattern)"
                ),
                ExpressionAttributeValues=_item(
                    {
                        ":pattern": pattern_id,
                        ":version": pattern_version_hash,
                        ":evidence": "VERIFIED_GENERATION",
                        ":now": _now(),
                    }
                ),
            )
            return True
        except ClientError as exc:
            if _is_conditional_failure(exc):
                emit_practice_event(
                    "PATTERN_QUESTION_BANK_LINK_CONFLICT",
                    test_id=test_id,
                    status="skipped",
                    details={"reasonCode": "QUESTION_BANK_PATTERN_LINK_NOT_ABSENT"},
                    level=logging.WARNING,
                )
                return False
            raise PracticeRepositoryError("QUESTION_BANK_PATTERN_LINK_FAILED") from exc

    def _put_question(
        self,
        *,
        question_id: str,
        test_id: str,
        question: str,
        options: list[str],
        correct_answer: str,
        solution: str,
        subject: str,
        topic: str,
        difficulty: str,
        meta: dict[str, Any],
        language: str = "",
        answer_contract: dict[str, Any] | None = None,
        pattern_id: str | None = None,
        pattern_version_hash: str | None = None,
    ) -> bool:
        timestamp = _now()
        item = {
            "questionId": question_id,
            "testId": test_id,
            "question": question,
            "options": options,
            "answers": _json(
                answer_contract
                or {"correctAnswer": correct_answer, "options": options}
            ),
            "correctAnswer": correct_answer,
            "marks": 1,
            "negativeMarks": 0,
            "section": subject,
            "difficulty": difficulty.upper(),
            "topic": topic,
            "subject": _SUBJECTS.get(subject.casefold(), "OTHER"),
            **(
                {"language": _storage_language(language)}
                if _storage_language(language)
                else {}
            ),
            "format": "standard",
            "meta": meta,
            "createdAt": timestamp,
            "updatedAt": timestamp,
            "__typename": "Question",
        }
        if pattern_id and pattern_version_hash:
            item["patternId"] = pattern_id
            item["patternVersionHash"] = pattern_version_hash
        if solution:
            item["explanation"] = solution
        try:
            execution_id = current_practice_execution_id()
            if execution_id is None:
                self._client.put_item(
                    TableName=self._question_table,
                    Item=_item(item),
                    ConditionExpression="attribute_not_exists(questionId)",
                )
            else:
                self._client.transact_write_items(
                    TransactItems=[
                        {
                            "ConditionCheck": {
                                "TableName": self._assessment_table,
                                "Key": _item({"testId": test_id}),
                                "ConditionExpression": "#meta.#execution = :execution",
                                "ExpressionAttributeNames": {
                                    "#meta": "meta",
                                    "#execution": "activeExecutionId",
                                },
                                "ExpressionAttributeValues": _item(
                                    {":execution": execution_id}
                                ),
                            }
                        },
                        {
                            "Put": {
                                "TableName": self._question_table,
                                "Item": _item(item),
                                "ConditionExpression": "attribute_not_exists(questionId)",
                            }
                        },
                    ]
                )
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if _is_conditional_failure(exc) or code == "TransactionCanceledException":
                cancellation_reasons = exc.response.get("CancellationReasons")
                if isinstance(cancellation_reasons, list) and any(
                    isinstance(reason, dict)
                    and reason.get("Code") not in {None, "None", "ConditionalCheckFailed"}
                    for reason in cancellation_reasons
                ):
                    raise PracticeRepositoryError("QUESTION_LINK_FAILED") from exc
                if execution_id is not None:
                    current = self._client.get_item(
                        TableName=self._assessment_table,
                        Key=_item({"testId": test_id}),
                        ConsistentRead=True,
                    ).get("Item")
                    assessment = _plain(current) if current else {}
                    if _parse_meta(assessment.get("meta")).get(
                        "activeExecutionId"
                    ) != execution_id:
                        raise PracticeRepositoryError("PRACTICE_EXECUTION_FENCED") from exc
                return False
            raise PracticeRepositoryError("QUESTION_LINK_FAILED") from exc

    def _update_question_with_execution_fence(
        self,
        *,
        test_id: str,
        question_id: str,
        update_expression: str,
        condition_expression: str,
        names: dict[str, str],
        values: dict[str, Any],
    ) -> None:
        execution_id = current_practice_execution_id()
        update = {
            "TableName": self._question_table,
            "Key": _item({"questionId": question_id}),
            "UpdateExpression": update_expression,
            "ConditionExpression": condition_expression,
            "ExpressionAttributeNames": names,
            "ExpressionAttributeValues": _item(values),
        }
        if execution_id is None:
            self._client.update_item(**update)
            return
        self._client.transact_write_items(
            TransactItems=[
                {
                    "ConditionCheck": {
                        "TableName": self._assessment_table,
                        "Key": _item({"testId": test_id}),
                        "ConditionExpression": "#meta.#execution = :execution",
                        "ExpressionAttributeNames": {
                            "#meta": "meta",
                            "#execution": "activeExecutionId",
                        },
                        "ExpressionAttributeValues": _item(
                            {":execution": execution_id}
                        ),
                    }
                },
                {"Update": update},
            ]
        )

    def assign_positions(self, questions: list[dict[str, Any]]) -> None:
        ordered = sorted(
            questions,
            key=lambda item: (
                str(item.get("_practiceMeta", {}).get("bucketId") or ""),
                str(item.get("questionId") or ""),
            ),
        )
        for position, question in enumerate(ordered, start=1):
            try:
                self._update_question_with_execution_fence(
                    test_id=str(question.get("testId") or ""),
                    question_id=str(question["questionId"]),
                    update_expression="SET #position = :position, #updated = :updated",
                    names={
                        "#position": "position",
                        "#updated": "updatedAt",
                    },
                    values={":position": position, ":updated": _now()},
                    condition_expression="attribute_exists(questionId)",
                )
            except ClientError as exc:
                raise PracticeRepositoryError("QUESTION_ORDERING_FAILED") from exc

    def repair_bucket_assignments(
        self,
        test_id: str,
        assignments: dict[str, tuple[str, str]],
    ) -> int:
        """Correct verified schema-v2 bucket metadata under immutable slot ownership."""
        repaired = 0
        for question_id, (slot_id, bucket_id) in assignments.items():
            if not question_id or not slot_id or not bucket_id:
                raise PracticeRepositoryError("MANIFEST_RECOVERY_INVALID_ASSIGNMENT")
            try:
                self._update_question_with_execution_fence(
                    test_id=test_id,
                    question_id=question_id,
                    update_expression=(
                        "SET #meta.#bucket = :bucket, #updated = :updated"
                    ),
                    names={
                        "#meta": "meta",
                        "#bucket": "bucketId",
                        "#slot": "slotId",
                        "#verified": "verified",
                        "#test": "testId",
                        "#updated": "updatedAt",
                    },
                    values={
                        ":bucket": bucket_id,
                        ":slot": slot_id,
                        ":verified": True,
                        ":test": test_id,
                        ":updated": _now(),
                    },
                    condition_expression=(
                        "#test = :test AND #meta.#slot = :slot "
                        "AND #meta.#verified = :verified"
                    ),
                )
            except ClientError as exc:
                raise PracticeRepositoryError("MANIFEST_RECOVERY_CONFLICT") from exc
            repaired += 1
        return repaired

    def rebalance_answer_positions(
        self,
        test_id: str,
        questions: list[dict[str, Any]],
    ) -> tuple[bool, tuple[int, ...]]:
        ordered = sorted(
            questions,
            key=lambda item: (
                str(item.get("_practiceMeta", {}).get("bucketId") or ""),
                str(item.get("questionId") or ""),
            ),
        )
        question_ids = [str(item.get("questionId") or "") for item in ordered]
        if not all(question_ids) or len(question_ids) != len(set(question_ids)):
            raise PracticeRepositoryError("ANSWER_POSITION_DISTRIBUTION_INVALID")
        changed = False
        for item, target_position in zip(
            ordered,
            target_correct_positions(test_id, question_ids),
            strict=True,
        ):
            options = item.get("options")
            if not isinstance(options, list):
                raise PracticeRepositoryError("ANSWER_POSITION_DISTRIBUTION_INVALID")
            reordered = reorder_options(
                test_id=test_id,
                question_id=str(item["questionId"]),
                options=[str(option) for option in options],
                correct_answer=str(item.get("correctAnswer") or ""),
                target_position=target_position,
            )
            if reordered is None:
                raise PracticeRepositoryError("ANSWER_POSITION_DISTRIBUTION_INVALID")
            if reordered != options:
                practice_meta = item.get("_practiceMeta")
                is_schema_v2 = (
                    isinstance(practice_meta, dict)
                    and str(practice_meta.get("schemaVersion") or "") == "2"
                )
                if is_schema_v2:
                    correct_answer = str(item.get("correctAnswer") or "")
                    correct_option_id = reordered.index(correct_answer)
                    current_answer_version = 1
                    raw_answers = item.get("answers")
                    if isinstance(raw_answers, str):
                        try:
                            parsed_answers = json.loads(raw_answers)
                        except (TypeError, ValueError):
                            parsed_answers = {}
                        if isinstance(parsed_answers, dict):
                            current_answer_version = int(
                                parsed_answers.get("answerVersion") or 1
                            )
                    answer_contract = {
                        "schemaVersion": "2",
                        "optionIdentity": "INDEX_V1",
                        "options": [
                            {"optionId": index, "value": value}
                            for index, value in enumerate(reordered)
                        ],
                        "correctOptionId": correct_option_id,
                        "correctAnswer": correct_answer,
                        "answerExplanation": str(item.get("explanation") or ""),
                        "answerStatus": "VERIFIED",
                        "answerVersion": current_answer_version + 1,
                    }
                else:
                    answer_contract = {
                        "correctAnswer": item.get("correctAnswer"),
                        "options": reordered,
                    }
                try:
                    self._update_question_with_execution_fence(
                        test_id=test_id,
                        question_id=str(item["questionId"]),
                        update_expression=(
                            "SET #options = :options, #answers = :answers, #updated = :updated"
                        ),
                        names={
                            "#options": "options",
                            "#answers": "answers",
                            "#updated": "updatedAt",
                        },
                        values={
                            ":options": reordered,
                            ":answers": _json(answer_contract),
                            ":updated": _now(),
                        },
                        condition_expression="attribute_exists(questionId)",
                    )
                except ClientError as exc:
                    raise PracticeRepositoryError(
                        "ANSWER_POSITION_DISTRIBUTION_INVALID"
                    ) from exc
                item["options"] = reordered
                item["answers"] = _json(answer_contract)
                changed = True
        validation = validate_answer_position_distribution(ordered)
        if not validation.valid:
            raise PracticeRepositoryError(validation.reason_code)
        return changed, validation.correct_positions
