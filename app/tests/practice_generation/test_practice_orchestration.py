"""In-memory integration and reliability tests for durable graph orchestration."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from features.practice_generation import orchestration as orchestration_module
from features.practice_generation.config import PracticeGenerationConfig
from features.practice_generation.generation import deterministic_question_id
from features.practice_generation.graph import PracticeGraphRunner
from features.practice_generation.matching import (
    build_reuse_difficulty_prefix,
    build_slot_reuse_bucket_key,
)
from features.practice_generation.option_distribution import (
    reorder_options,
    target_correct_positions,
    validate_answer_position_distribution,
)
from features.practice_generation.orchestration import PracticeGenerationOrchestrator
from features.practice_generation.pattern_context import NoOpPatternContextProvider
from features.practice_generation.planning import (
    BlueprintManager,
    BlueprintPlanResult,
    deterministic_blueprint,
)
from features.practice_generation.progress_contract import (
    PRACTICE_PROGRESS_ALLOWED_META_KEYS,
    PRACTICE_PROGRESS_INTERNAL_META_KEYS,
)
from features.practice_generation.repositories import IndexedQueryResult
from features.practice_generation.schemas import (
    GeneratedBatch,
    PracticeBlueprint,
    PracticeGenerationRequest,
    PracticeGraphCommand,
    PracticeType,
    VerificationResult,
)
from services.llm.orchestration.errors import ProviderExecutionError


class FakeAssessments:
    def __init__(self, request: PracticeGenerationRequest, test_id: str = "test-1") -> None:
        self.phase_history = ["QUEUED"]
        self.group_progress_publications = 0
        self.ready_publications = 0
        self.failed_publications = 0
        self.claimed_group_ids: list[str] = []
        self.item = {
            "testId": test_id,
            "userId": request.user_id,
            "name": request.assessment_title,
            "status": "GENERATING",
            "live": False,
            "updatedAt": "1",
            "meta": json.dumps(
                {
                    "phase": "QUEUED",
                    "practiceRequest": {
                        "requestId": request.request_id,
                        "conversationId": request.conversation_id,
                        "turnId": request.turn_id,
                        "practiceType": request.practice_type.value,
                        "requestedCount": request.requested_count,
                        "acceptedCount": request.accepted_count,
                        "subject": request.subject,
                        "topic": request.topic,
                        "difficulty": request.difficulty.value,
                        "language": request.language,
                        "examId": request.exam_id,
                        "examStage": request.exam_stage,
                        "includeSolutions": request.include_solutions,
                        "assessmentTitle": request.assessment_title,
                        "querySummary": request.original_query,
                    },
                    "readyQuestionCount": 0,
                    "questionManifestVersion": 1,
                    "readyQuestionIds": [],
                    "readyCount": 0,
                    "reusedCount": 0,
                    "generatedCount": 0,
                    "verifiedCount": 0,
                    "failedCount": 0,
                }
            ),
        }

    def get(self, _test_id: str, *, consistent: bool = True):
        return deepcopy(self.item)

    def update(
        self,
        _test_id: str,
        *,
        meta_updates,
        status=None,
        live=None,
        expected_updated_at=None,
        recalculate_manifest=False,
        authoritative_question_ids=(),
    ):
        if expected_updated_at is not None:
            assert expected_updated_at == self.item["updatedAt"]
        if recalculate_manifest and authoritative_question_ids:
            self.group_progress_publications += 1
        meta = json.loads(self.item["meta"])
        meta.update(deepcopy(meta_updates))
        if meta_updates.get("phase"):
            self.phase_history.append(str(meta_updates["phase"]))
        self.item["meta"] = json.dumps(meta)
        self.item["updatedAt"] = str(int(self.item["updatedAt"]) + 1)
        if status is not None:
            self.item["status"] = status
        if live is not None:
            self.item["live"] = live
        return deepcopy(self.item)

    def calculate_authoritative_meta(
        self,
        _test_id,
        *,
        meta_updates=None,
        authoritative_question_ids=(),
    ):
        meta = json.loads(self.item["meta"])
        updates = deepcopy(meta_updates or {})
        if meta.get("bucketReadyCounts"):
            updates.pop("bucketReadyCounts", None)
        meta.update(updates)
        return meta

    def set_blueprint(
        self,
        test_id,
        blueprint,
        *,
        planner_calls,
        planner_tier,
        planner_repaired,
        deterministic_fallback,
    ):
        self.update(
            test_id,
            meta_updates={
                "blueprint": blueprint.model_dump(mode="json"),
                "plannerCalls": planner_calls,
                "plannerTier": planner_tier,
                "plannerRepaired": planner_repaired,
                "plannerDeterministicFallback": deterministic_fallback,
                "bucketReadyCounts": {bucket.bucket_id: 0 for bucket in blueprint.buckets},
            },
        )

    def claim_group(self, _test_id, group_id):
        meta = json.loads(self.item["meta"])
        group = meta["generationGroups"][group_id]
        if group["state"] in {"RUNNING", "COMPLETED", "FAILED"}:
            return None
        group["state"] = "RUNNING"
        self.claimed_group_ids.append(group_id)
        self.item["meta"] = json.dumps(meta)
        return deepcopy(group)

    def update_group(
        self,
        test_id,
        group_id,
        *,
        state,
        required_count=None,
        attempt=None,
        error_code=None,
        attempt_stage=None,
        item_retry_count=None,
        replacement_count=None,
        replacement=None,
        last_reason_code=None,
    ):
        meta = json.loads(self.item["meta"])
        group = meta["generationGroups"][group_id]
        group["state"] = state
        if required_count is not None:
            group["requiredCount"] = required_count
        if attempt is not None:
            group["attempt"] = attempt
        if error_code is not None:
            group["errorCode"] = error_code
        for key, value in {
            "attemptStage": attempt_stage,
            "itemRetryCount": item_retry_count,
            "replacementCount": replacement_count,
            "replacement": replacement,
            "lastReasonCode": last_reason_code,
        }.items():
            if value is not None:
                group[key] = value
        self.item["meta"] = json.dumps(meta)

    def create_retry_group(
        self,
        _test_id,
        *,
        group_id,
        parent_group_id,
        bucket_id,
        attempt_stage,
        attempt,
        replacement,
    ):
        meta = json.loads(self.item["meta"])
        groups = meta["generationGroups"]
        if group_id in groups:
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
            "state": "PENDING",
        }
        self.item["meta"] = json.dumps(meta)
        return True

    def record_finalization_retry(self, _test_id, *, next_attempt, reason_code):
        meta = json.loads(self.item["meta"])
        if int(meta.get("finalizationAttempt") or 0) >= next_attempt:
            return False
        meta["finalizationAttempt"] = next_attempt
        meta["finalizationReasonCode"] = reason_code
        self.item["meta"] = json.dumps(meta)
        return True

    def mark_failed(self, test_id, error_code, *, meta_updates=None):
        self.failed_publications += 1
        meta = json.loads(self.item["meta"])
        meta.update(deepcopy(meta_updates or {}))
        self.item["meta"] = json.dumps(meta)
        meta["failedCount"] = int(meta.get("failedCount") or 0) + 1
        self.item["meta"] = json.dumps(meta)
        self.update(
            test_id,
            meta_updates={"phase": "FAILED", "errorCode": error_code, "playable": False},
            status="FAILED",
            live=False,
        )

    def mark_ready(self, test_id, accepted_count, meta_updates):
        meta = json.loads(self.item["meta"])
        if (
            int(meta.get("readyQuestionCount") or 0) != accepted_count
            or int(meta.get("readyCount") or 0) != accepted_count
            or len(meta.get("readyQuestionIds") or []) != accepted_count
            or int(meta.get("failedCount") or 0) != 0
        ):
            return False
        self.update(
            test_id,
            meta_updates=meta_updates,
            status="READY",
            live=True,
        )
        self.ready_publications += 1
        return True

    def cancel(self, test_id):
        self.update(
            test_id,
            meta_updates={"phase": "CANCELLED", "playable": False},
            status="ARCHIVED",
            live=False,
        )


class FakeQuestions:
    def __init__(
        self,
        assessments,
        reusable_items=None,
        *,
        final_gsi_lag_once: bool = False,
        final_gsi_lag_always: bool = False,
        drop_counter_once: bool = False,
        corrupt_manifest_ownership: bool = False,
        topic_pages: list[list[dict]] | None = None,
    ) -> None:
        self.assessments = assessments
        self.promoted: list[dict] = []
        self.topic_pages = deepcopy(topic_pages)
        self.reusable_items = list(reusable_items or [])
        if self.topic_pages is not None:
            self.reusable_items = [item for page in self.topic_pages for item in page]
        self.linked: dict[str, dict] = {}
        self.final_gsi_lag_once = final_gsi_lag_once
        self.final_gsi_lag_always = final_gsi_lag_always
        self.final_gsi_lag_used = False
        self.drop_counter_once = drop_counter_once
        self.corrupt_manifest_ownership = corrupt_manifest_ownership
        self.reuse_query_order: list[str] = []

    def query_reuse_candidates(
        self,
        *,
        category,
        limit,
        exclusive_start_key=None,
    ):
        assert exclusive_start_key is None
        self.reuse_query_order.append("category")
        return IndexedQueryResult(
            items=tuple(deepcopy(self.reusable_items[:limit])),
            page_count=1,
            has_more_pages=False,
            consumed_capacity=0.5,
            duration_ms=1,
        )

    def query_topic_reuse_candidates(
        self,
        *,
        reuse_bucket_key,
        limit,
        difficulty_prefix=None,
        exclusive_start_key=None,
    ):
        del difficulty_prefix
        self.reuse_query_order.append("topic")
        if self.topic_pages is not None:
            page_index = int((exclusive_start_key or {}).get("page") or 0)
            page = self.topic_pages[page_index][:limit]
            has_more = page_index + 1 < len(self.topic_pages)
            return IndexedQueryResult(
                items=tuple(deepcopy(page)),
                page_count=1,
                has_more_pages=has_more,
                consumed_capacity=0.5,
                duration_ms=1,
                continuation_key=({"page": page_index + 1} if has_more else None),
                evaluated_count=len(page),
            )
        assert exclusive_start_key is None
        return IndexedQueryResult(
            items=tuple(deepcopy(self.reusable_items[:limit])),
            page_count=1,
            has_more_pages=False,
            consumed_capacity=0.5,
            duration_ms=1,
        )

    def get_reuse_candidates(self, question_ids):
        selected = set(question_ids)
        return deepcopy(
            [item for item in self.reusable_items if str(item.get("qbId") or "") in selected]
        )

    def list_linked(self, _test_id, *, limit=100):
        items = list(self.linked.values())[:limit]
        meta = json.loads(self.assessments.item["meta"])
        if (
            (self.final_gsi_lag_always or self.final_gsi_lag_once)
            and (self.final_gsi_lag_always or not self.final_gsi_lag_used)
            and len(items) == int(meta["practiceRequest"]["acceptedCount"])
        ):
            self.final_gsi_lag_used = True
            items = items[:-1]
        return deepcopy(items)

    def _increment(self, bucket_id, source):
        meta = json.loads(self.assessments.item["meta"])
        question_id = next(reversed(self.linked))
        meta["readyQuestionCount"] = int(meta.get("readyQuestionCount") or 0) + 1
        meta["readyCount"] = int(meta.get("readyCount") or 0) + 1
        meta.setdefault("readyQuestionIds", []).append(question_id)
        meta["verifiedCount"] = int(meta.get("verifiedCount") or 0) + 1
        meta[source] = int(meta.get(source) or 0) + 1
        counts = dict(meta.get("bucketReadyCounts") or {})
        counts[bucket_id] = int(counts.get(bucket_id) or 0) + 1
        meta["bucketReadyCounts"] = counts
        self.assessments.item["meta"] = json.dumps(meta)

    def link_reused(self, *, test_id, bucket_id, question, **_kwargs):
        question_id = deterministic_question_id(
            test_id,
            source_id=question.question_id,
            bucket_id=bucket_id,
        )
        if question_id in self.linked:
            return False
        self.linked[question_id] = {
            "questionId": question_id,
            "testId": test_id,
            "question": question.question,
            "options": list(question.options),
            "answers": json.dumps(
                {"correctAnswer": question.correct_answer, "options": list(question.options)}
            ),
            "correctAnswer": question.correct_answer,
            "explanation": question.solution,
            "_practiceMeta": {
                "source": "REUSED",
                "sourceType": "QUESTION_BANK",
                "sourceQuestionBankId": question.question_id,
                "bucketId": bucket_id,
                "verified": True,
                "verificationMethod": "QUESTION_BANK_QUALITY",
                "questionType": question.question_type,
                "language": question.language,
            },
        }
        self._increment(bucket_id, "reusedCount")
        return True

    def get_questions_by_ids(self, question_ids):
        resolved = deepcopy(
            [self.linked[question_id] for question_id in question_ids if question_id in self.linked]
        )
        if self.corrupt_manifest_ownership and resolved:
            resolved[0]["testId"] = "other-test"
        return resolved

    def repair_bucket_assignments(self, test_id, assignments):
        repaired = 0
        for question_id, (slot_id, bucket_id) in assignments.items():
            item = self.linked[question_id]
            assert item["testId"] == test_id
            assert item["_practiceMeta"]["slotId"] == slot_id
            assert item["_practiceMeta"]["verified"] is True
            item["_practiceMeta"]["bucketId"] = bucket_id
            repaired += 1
        return repaired

    def persist_verified_question(
        self,
        *,
        test_id,
        question,
        slot,
        language,
        pattern_id=None,
        pattern_version_hash=None,
    ):
        self.promoted.append(
            {
                "slotId": slot.slot_id,
                "language": language,
                "patternId": pattern_id,
                "patternVersionHash": pattern_version_hash,
            }
        )
        return f"qb-v2-{slot.slot_id}"

    def link_generated(
        self,
        *,
        test_id,
        question,
        verified,
        group_id,
        generator_route,
        generator_model,
        verification_policy,
        verification_method,
        language,
        **_kwargs,
    ):
        assert verified is True
        question_id = deterministic_question_id(
            test_id,
            source_id=question.generation_item_id,
            bucket_id=question.bucket_id,
        )
        if question_id in self.linked:
            return False
        self.linked[question_id] = {
            "questionId": question_id,
            "testId": test_id,
            "question": question.question,
            "options": list(question.options),
            "answers": json.dumps(
                {"correctAnswer": question.correct_answer, "options": question.options}
            ),
            "correctAnswer": question.correct_answer,
            "explanation": question.solution,
            "_practiceMeta": {
                "source": "GENERATED",
                "sourceType": "AI_GENERATED",
                "bucketId": question.bucket_id,
                "verified": True,
                "generationGroupId": group_id,
                "generatorRoute": generator_route,
                "generatorModel": generator_model,
                "verificationPolicy": verification_policy,
                "verificationMethod": verification_method,
                "questionType": question.question_type.value,
                "language": language,
            },
        }
        if self.drop_counter_once:
            self.drop_counter_once = False
            meta = json.loads(self.assessments.item["meta"])
            meta["readyQuestionCount"] = int(meta.get("readyQuestionCount") or 0) + 1
            meta["readyCount"] = int(meta.get("readyCount") or 0) + 1
            meta.setdefault("readyQuestionIds", []).append(question_id)
            meta["verifiedCount"] = int(meta.get("verifiedCount") or 0) + 1
            counts = dict(meta.get("bucketReadyCounts") or {})
            counts[question.bucket_id] = int(counts.get(question.bucket_id) or 0) + 1
            meta["bucketReadyCounts"] = counts
            self.assessments.item["meta"] = json.dumps(meta)
        else:
            self._increment(question.bucket_id, "generatedCount")
        return True

    def assign_positions(self, questions):
        for position, question in enumerate(
            sorted(questions, key=lambda item: item["questionId"]),
            start=1,
        ):
            self.linked[question["questionId"]]["position"] = position

    def rebalance_answer_positions(self, test_id, questions):
        ordered = sorted(questions, key=lambda item: item["questionId"])
        question_ids = [question["questionId"] for question in ordered]
        changed = False
        for question, target_position in zip(
            ordered,
            target_correct_positions(test_id, question_ids),
            strict=True,
        ):
            reordered = reorder_options(
                test_id=test_id,
                question_id=question["questionId"],
                options=list(question["options"]),
                correct_answer=question["correctAnswer"],
                target_position=target_position,
            )
            assert reordered is not None
            if reordered != question["options"]:
                question["options"] = reordered
                question["answers"] = json.dumps(
                    {
                        "correctAnswer": question["correctAnswer"],
                        "options": reordered,
                    }
                )
                self.linked[question["questionId"]] = deepcopy(question)
                changed = True
        validation = validate_answer_position_distribution(ordered)
        assert validation.valid
        return changed, validation.correct_positions


class Planner:
    def plan(self, request, *, tier, repair_feedback=None):
        return json.dumps(
            {
                "schema_version": "1",
                "practice_type": request.practice_type.value,
                "accepted_count": request.accepted_count,
                "buckets": [
                    {
                        "bucket_id": "bucket-1",
                        "subject": request.subject,
                        "topic": request.topic or request.subject,
                        "difficulty": request.difficulty.value,
                        "question_type": "mcq",
                        "required_count": request.accepted_count,
                        "keywords": [request.topic or request.subject],
                        "question_intent": "legacy compatibility orchestration test",
                        "verification_policy": "MANDATORY",
                    }
                ],
            }
        )


class LegacyBlueprintManager:
    def build(self, request):
        blueprint = PracticeBlueprint.model_validate(
            json.loads(Planner().plan(request, tier="light"))
        )
        return BlueprintPlanResult(
            blueprint=blueprint,
            planner_calls=1,
            repaired=False,
            deterministic_fallback=False,
            tier="light",
        )


_LAST_GENERATOR: dict[str, Generator] = {}


class Generator:
    def __init__(
        self,
        *,
        partial_once: bool = False,
        fail_always: bool = False,
        fail_item_retry_once: bool = False,
    ) -> None:
        self.partial_once = partial_once
        self.fail_always = fail_always
        self.fail_item_retry_once = fail_item_retry_once
        self.calls: dict[str, int] = {}
        # (group_id, attempt, required_count) per invocation, so tests can prove a
        # retry asks only for the deficit rather than the whole group.
        self.requests: list[tuple[str, int, int]] = []
        # (group_id, attempt, language, exam_id, exam_stage) per invocation, so tests
        # can prove a replacement wave carries the same delivery context as attempt 0.
        self.contexts: list[tuple[str, int, str, str | None, str | None]] = []

    def generate(self, *, request, bucket, group, exclude_normalized_texts):
        self.requests.append((group.group_id, group.attempt, group.required_count))
        self.contexts.append(
            (
                group.group_id,
                group.attempt,
                request.language,
                request.exam_id,
                request.exam_stage,
            )
        )
        if self.fail_always:
            raise TimeoutError("injected")
        if self.fail_item_retry_once and "-retry-" in group.group_id and group.attempt == 1:
            self.fail_item_retry_once = False
            raise TimeoutError("injected item retry failure")
        calls = self.calls.get(group.group_id, 0)
        self.calls[group.group_id] = calls + 1
        count = group.required_count
        if self.partial_once and calls == 0 and count > 1:
            count -= 1
        questions = []
        for index in range(count):
            item_id = f"{group.group_id}-a{group.attempt}-i{index}"
            answer = str(index + 2)
            questions.append(
                {
                    "generation_item_id": item_id,
                    "bucket_id": bucket.bucket_id,
                    "question": (
                        # Devanagari stem when Hindi is requested: the production
                        # delivery-language gate rejects ASCII-only Hindi content.
                        f"{'प्रश्न: ' if request.language == 'hindi' else ''}"
                        f"For generated item {item_id}, what is {index + 1} plus one?"
                    ),
                    "question_type": bucket.question_type.value,
                    "options": [answer, "10", "11", "12"],
                    "correct_answer": answer,
                    "solution": "Add one to the stated integer.",
                    "subject": bucket.subject,
                    "topic": bucket.topic,
                    "difficulty": bucket.difficulty.value,
                }
            )
        return GeneratedBatch(
            content=json.dumps({"questions": questions}),
            route_id=f"{bucket.subject}.generator.{bucket.difficulty.value}",
            model="test-generator",
        )


class TokenExhaustedGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, request, bucket, group, exclude_normalized_texts):
        self.calls += 1
        raise ProviderExecutionError(
            "configured model fallbacks exhausted",
            failure_kind="output_token_exhausted",
            attempted_aliases=("reasoning_advanced_generator", "openai_o3"),
        )


class ProviderFailureGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *, request, bucket, group, exclude_normalized_texts):
        self.calls += 1
        raise ProviderExecutionError(
            "configured model fallbacks exhausted",
            failure_kind="timeout",
            attempted_aliases=("math_generator", "fallback_generator"),
        )


class Verifier:
    def __init__(self, *, reject_once: bool = False) -> None:
        self.reject_once = reject_once
        self.calls = 0

    def verify(self, *, request, bucket, question):
        self.calls += 1
        if self.reject_once:
            self.reject_once = False
            return VerificationResult(
                generation_item_id=question.generation_item_id,
                approved=False,
                reason_code="INJECTED_REJECTION",
            )
        return VerificationResult(
            generation_item_id=question.generation_item_id,
            approved=True,
            reason_code="MATCH",
        )


class SensitiveReasonVerifier:
    def verify(self, *, request, bucket, question):
        return VerificationResult(
            generation_item_id=question.generation_item_id,
            approved=True,
            reason_code="THE_ANSWER_IS_SECRET",
        )


def build_orchestrator(
    assessments,
    questions,
    *,
    partial_once: bool = False,
    fail_always: bool = False,
    fail_item_retry_once: bool = False,
    reject_once: bool = False,
    verifier=None,
    pattern_context=None,
    question_bank_promotion_enabled: bool = False,
):
    generator = Generator(
        partial_once=partial_once,
        fail_always=fail_always,
        fail_item_retry_once=fail_item_retry_once,
    )
    _LAST_GENERATOR["generator"] = generator
    return PracticeGenerationOrchestrator(
        config=PracticeGenerationConfig(
            enabled=True,
            pattern_context_enabled=False,
            pattern_reuse_enabled=False,
            question_bank_promotion_enabled=question_bank_promotion_enabled,
            assessment_table="assessment",
            question_table="question",
            question_bank_table="bank",
            question_bank_category_index="category-index",
            question_test_index="test-index",
            aws_region="ap-south-1",
            generation_group_size=3,
            generation_group_max=5,
            item_retry_limit=1,
            planner_repair_limit=1,
        ),
        assessments=assessments,
        progress=assessments,
        questions=questions,
        blueprint_manager=LegacyBlueprintManager(),
        generator=generator,
        verifier=verifier or Verifier(reject_once=reject_once),
        pattern_context=pattern_context or NoOpPatternContextProvider(),
    )


def make_request(
    count: int,
    *,
    full_mock: bool = False,
    language: str = "english",
    exam_id: str | None = "CAT",
    exam_stage: str | None = None,
) -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id=f"request-{count}",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id=f"turn-{count}",
        original_query=f"Create {count} questions",
        practice_type="FULL_MOCK" if full_mock else "QUIZ",
        requested_count=count,
        accepted_count=min(count, 100),
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language=language,
        exam_id=exam_id,
        exam_stage=exam_stage,
        assessment_title="Algebra Practice",
    )


def reusable_item(index: int) -> dict:
    answer = str(index + 2)
    return {
        "qbId": f"bank-{index}",
        "question": f"Reusable algebra question number {index}?",
        "answers": json.dumps({"options": [answer, "20", "21", "22"]}),
        "correctAnswer": answer,
        "explanation": "Verified reusable solution.",
        "category": "math",
        "difficulty": "MEDIUM",
        "source": "verified-bank",
        "meta": json.dumps(
            {
                "status": "ACTIVE",
                "qualityStatus": "VERIFIED",
                "reusable": True,
                "visibility": "PLATFORM",
                "language": "english",
                "subject": "math",
                "topic": "algebra",
                "questionType": "mcq",
            }
        ),
    }


def run_job(
    count: int,
    *,
    reuse_count: int = 0,
    partial_once: bool = False,
    fail_always: bool = False,
    fail_item_retry_once: bool = False,
    final_gsi_lag_once: bool = False,
    final_gsi_lag_always: bool = False,
    drop_counter_once: bool = False,
    corrupt_manifest_ownership: bool = False,
    topic_pages: list[list[dict]] | None = None,
    reject_once: bool = False,
    verifier=None,
    question_bank_promotion_enabled: bool = False,
    language: str = "english",
    exam_id: str | None = "CAT",
    exam_stage: str | None = None,
):
    request = make_request(
        count,
        full_mock=count >= 50,
        language=language,
        exam_id=exam_id,
        exam_stage=exam_stage,
    )
    assessments = FakeAssessments(request)
    questions = FakeQuestions(
        assessments,
        [reusable_item(index) for index in range(reuse_count)],
        final_gsi_lag_once=final_gsi_lag_once,
        final_gsi_lag_always=final_gsi_lag_always,
        drop_counter_once=drop_counter_once,
        corrupt_manifest_ownership=corrupt_manifest_ownership,
        topic_pages=topic_pages,
    )
    orchestrator = build_orchestrator(
        assessments,
        questions,
        partial_once=partial_once,
        fail_always=fail_always,
        fail_item_retry_once=fail_item_retry_once,
        reject_once=reject_once,
        verifier=verifier,
        question_bank_promotion_enabled=question_bank_promotion_enabled,
    )
    runner = PracticeGraphRunner(orchestrator)
    runner.process(PracticeGraphCommand(operation="plan_and_fill", test_id="test-1"))
    processed = 1
    while processed < 500 and assessments.item["status"] == "GENERATING":
        meta = json.loads(assessments.item["meta"])
        pending = [
            (group_id, group)
            for group_id, group in dict(meta.get("generationGroups") or {}).items()
            if isinstance(group, dict) and group.get("state") == "PENDING"
        ]
        if pending:
            for group_id, group in pending:
                command = PracticeGraphCommand(
                    operation="generate_group",
                    test_id="test-1",
                    group_id=group_id,
                    attempt=int(group.get("attempt") or 0),
                )
                runner.process(command)
                runner.process(command)
                processed += 1
            continue
        next_attempt = int(meta.get("finalizationAttempt") or 0)
        if next_attempt:
            runner.process(
                PracticeGraphCommand(
                    operation="finalize",
                    test_id="test-1",
                    attempt=next_attempt,
                )
            )
            processed += 1
            continue
        break
    return assessments, questions, processed


def test_mixed_reuse_and_generation_reaches_exact_ready_count() -> None:
    assessments, questions, _ = run_job(10, reuse_count=4)
    assert (assessments.item["status"], assessments.item["live"], len(questions.linked)) == (
        "READY",
        True,
        10,
    )


def test_finalization_repairs_wrong_bucket_metadata_once_and_revalidates() -> None:
    practice_request = make_request(3)
    blueprint_data = deterministic_blueprint(practice_request).model_dump(mode="json")
    blueprint_data["slots"][0]["category_id"] = "basic_fundamentals"
    blueprint_data["slots"][1]["category_id"] = "tricky_concept"
    blueprint_data["slots"][2]["category_id"] = "tricky_concept"
    blueprint = PracticeBlueprint.model_validate(blueprint_data)
    assessments = FakeAssessments(practice_request)
    questions = FakeQuestions(assessments)
    question_ids: list[str] = []
    for slot in blueprint.slots:
        question_id = f"question-{slot.slot_id}"
        question_ids.append(question_id)
        questions.linked[question_id] = {
            "questionId": question_id,
            "testId": "test-1",
            "question": f"Question for {slot.slot_id}?",
            "options": ["A", "B", "C", "D"],
            "answers": json.dumps(
                {"correctAnswer": "A", "options": ["A", "B", "C", "D"]}
            ),
            "correctAnswer": "A",
            "explanation": "A is correct.",
            "topic": slot.topic_id,
            "difficulty": slot.difficulty.value,
            "_practiceMeta": {
                "bucketId": "slot-bucket-001",
                "slotId": slot.slot_id,
                "verified": True,
                "sourceType": "AI_GENERATED",
                "verificationMethod": "INDEPENDENT_MODEL_V2",
                "questionType": "mcq",
                "language": "english",
            },
        }
    meta = json.loads(assessments.item["meta"])
    meta.update(
        {
            "blueprint": blueprint.model_dump(mode="json"),
            "generationGroups": {
                "g1": {"state": "COMPLETED"},
                "g2": {"state": "COMPLETED"},
            },
            "readyQuestionIds": question_ids,
            "readyCount": 3,
            "readyQuestionCount": 3,
            "generatedCount": 3,
            "reusedCount": 0,
            "verifiedCount": 3,
            "failedCount": 0,
            "bucketReadyCounts": {"slot-bucket-001": 3},
        }
    )
    assessments.item["meta"] = json.dumps(meta)
    orchestrator = build_orchestrator(assessments, questions)

    orchestrator._finalize("test-1", practice_request, blueprint, attempt=0)

    recovered_meta = json.loads(assessments.item["meta"])
    assert recovered_meta["finalizationAttempt"] == 1
    assert assessments.item["status"] == "GENERATING"
    assert [
        questions.linked[question_id]["_practiceMeta"]["bucketId"]
        for question_id in question_ids
    ] == ["slot-bucket-001", "slot-bucket-002", "slot-bucket-002"]

    orchestrator._finalize("test-1", practice_request, blueprint, attempt=1)

    assert assessments.item["status"] == "READY"


def test_pattern_flags_off_reaches_legacy_five_question_generation_without_pattern_resources(
) -> None:
    assessments, questions, _ = run_job(5)
    meta = json.loads(assessments.item["meta"])

    assert assessments.item["status"] == "READY"
    assert len(questions.linked) == 5
    assert meta["generatedCount"] == 5
    assert "patternRuntime" not in meta


def test_pattern_flags_off_never_calls_the_pattern_provider() -> None:
    class FailIfCalledPatternProvider:
        def resolve_slots(self, **_kwargs):
            raise AssertionError("Pattern provider must not run while both flags are off")

    request = make_request(5)
    assessments = FakeAssessments(request)
    questions = FakeQuestions(assessments)
    orchestrator = build_orchestrator(
        assessments,
        questions,
        pattern_context=FailIfCalledPatternProvider(),
    )

    orchestrator.plan_and_fill("test-1")

    meta = json.loads(assessments.item["meta"])
    assert assessments.item["status"] == "GENERATING"
    assert len(meta["generationGroups"]) == 2


def test_schema_v2_reasoning_cat_fallback_reaches_generation_started_with_backend_meta_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failing Dev request must not publish Pattern-only metadata when flags are off."""

    allowed_meta_keys = (
        PRACTICE_PROGRESS_ALLOWED_META_KEYS | PRACTICE_PROGRESS_INTERNAL_META_KEYS
    )

    class ContractAssessments(FakeAssessments):
        def update(self, *args, meta_updates, **kwargs):
            current = json.loads(self.item["meta"])
            candidate = {**current, **deepcopy(meta_updates)}
            assert set(candidate).issubset(allowed_meta_keys)
            return super().update(*args, meta_updates=meta_updates, **kwargs)

    class InvalidPlanner:
        def plan(self, *_args, **_kwargs):
            return '{"slots":[]}'

    request = make_request(5).model_copy(
        update={
            "original_query": (
                "Create a 5-question Quick Practice on seating arrangements in Reasoning "
                "for CAT Management — Pre."
            ),
            "practice_type": PracticeType.QUICK_PRACTICE,
            "subject": "reasoning",
            "topic": "seating_arrangements",
            "exam_stage": "PRE",
            "exam_profile_id": "cat_management_pre",
            "assessment_title": "Reasoning Quick Practice",
        }
    )
    assessments = ContractAssessments(request)
    questions = FakeQuestions(assessments)
    events: list[str] = []
    monkeypatch.setattr(
        orchestration_module,
        "emit_practice_event",
        lambda name, **_kwargs: events.append(name),
    )
    orchestrator = PracticeGenerationOrchestrator(
        config=PracticeGenerationConfig(
            enabled=True,
            pattern_context_enabled=False,
            pattern_reuse_enabled=False,
            assessment_table="assessment",
            question_table="question",
            question_bank_table="bank",
            question_bank_category_index="category-index",
            question_test_index="test-index",
            aws_region="ap-south-1",
        ),
        assessments=assessments,
        progress=assessments,
        questions=questions,
        blueprint_manager=BlueprintManager(InvalidPlanner(), repair_limit=1),
        generator=Generator(),
        verifier=Verifier(),
        pattern_context=NoOpPatternContextProvider(),
    )

    orchestrator.plan_and_fill("test-1")

    meta = json.loads(assessments.item["meta"])
    assert assessments.item["status"] == "GENERATING"
    # Quick Practice now reaches the deterministic compiler directly rather than
    # by falling back from a failed planner call.
    assert meta["plannerDeterministicFallback"] is False
    assert meta["blueprint"]["schema_version"] == "2"
    slots = PracticeBlueprint.model_validate(meta["blueprint"]).slots
    assert [slot.slot_id for slot in slots] == [
        "slot-001",
        "slot-002",
        "slot-003",
        "slot-004",
        "slot-005",
    ]
    assert all(
        slot.subject_id == "reasoning"
        and slot.topic_id == "seating_arrangements"
        and slot.category_id == "seating_arrangements"
        and slot.difficulty.value == "intermediate"
        and slot.question_type.value == "mcq"
        and slot.exam_ids == ["CAT"]
        and slot.pattern_family_id is None
        and slot.reasoning_target is None
        and slot.not_same_when == []
        for slot in slots
    )
    assert {
        build_slot_reuse_bucket_key(slot, language=request.language) for slot in slots
    } == {"v1#seating_arrangements#seating_arrangements#mcq#english"}
    assert {build_reuse_difficulty_prefix(slot.difficulty.value) for slot in slots} == {
        "v1#medium#"
    }
    assert len(meta["generationGroups"]) == 2
    assert "patternRuntime" not in meta
    assert "QUESTION_BANK_REUSE_QUERY_STARTED" in events
    assert "QUESTION_BANK_REUSE_QUERY_COMPLETED" in events
    assert "EXISTING_MATCH_COMPLETED" in events
    assert "DEFICIT_CALCULATED" in events
    assert "GENERATION_GROUP_CREATED" in events
    assert "PATTERN_RETRIEVAL_COMPLETED" not in events

    first_group_id = next(iter(meta["generationGroups"]))
    orchestrator.generate_group("test-1", first_group_id)

    assert "GENERATION_GROUP_STARTED" in events


def test_invalid_deterministic_fallback_fails_with_safe_planner_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidPlanner:
        def plan(self, *_args, **_kwargs):
            return '{"slots": []}'

    request = make_request(5).model_copy(update={"topic": "///"})
    assessments = FakeAssessments(request)
    questions = FakeQuestions(assessments)
    emitted: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        orchestration_module,
        "emit_practice_event",
        lambda name, **kwargs: emitted.append((name, kwargs.get("details") or {})),
    )
    orchestrator = PracticeGenerationOrchestrator(
        config=PracticeGenerationConfig(
            enabled=True,
            pattern_context_enabled=False,
            pattern_reuse_enabled=False,
            assessment_table="assessment",
            question_table="question",
            question_bank_table="bank",
            question_bank_category_index="category-index",
            question_test_index="test-index",
            aws_region="ap-south-1",
        ),
        assessments=assessments,
        progress=assessments,
        questions=questions,
        blueprint_manager=BlueprintManager(InvalidPlanner(), repair_limit=1),
        generator=Generator(),
        verifier=Verifier(),
        pattern_context=NoOpPatternContextProvider(),
    )

    orchestrator.plan_and_fill("test-1")

    meta = json.loads(assessments.item["meta"])
    assert assessments.item["status"] == "FAILED"
    # C1: OLD CONTRACT — the deterministic fallback ran and was itself invalid
    # (PRACTICE_PLANNER_FALLBACK_INVALID). NEW CONTRACT — the request carries no
    # trustworthy structured topics, so planning stops before the fallback can widen
    # the composition. Both end FAILED and non-playable; only the reason differs.
    assert meta["errorCode"] == "PRACTICE_PLANNER_SEMANTIC_FALLBACK_UNSAFE"
    diagnostics = [details for name, details in emitted if name == "planner_validation_failed"]
    assert {
        key: value
        for key, value in diagnostics[-1].items()
        if key != "validationOrigin"
    } == {
        "expectedSlotCount": 5,
        "routeId": "quant_reasoning.planner.basic",
        "plannerTier": "light",
        "fallbackInvoked": False,
        "fallbackResult": "not_invoked",
        "reasonCode": "PLANNER_SLOT_COUNT_MISMATCH",
        "actualSlotCount": 0,
        "plannerAttempt": 2,
        "plannerPhase": "repair",
        "schemaName": "PracticeBlueprint",
        "validationErrorCount": 1,
        # Joined strings, not lists: the sanitizer collapses non-scalars to "list".
        "fieldPaths": "$",
        "errorTypes": "value_error",
        # Which layer rejected the response, so the failing invariant is
        # identifiable without the raw planner output.
        "validationStage": "schema",
        "durationMs": 0,
    }
    # Raise site inside our own source. Code location only — no message text, no model
    # output, no student content. The line number is deliberately not pinned: it moves
    # with any edit, and the safety property is the shape, not the literal.
    origin = diagnostics[-1]["validationOrigin"]
    assert origin.startswith("planning.py:")
    assert origin.split(":")[1].isdigit()
    # C1: the fallback is no longer invoked, so no planner_fallback_failed event.
    # The assessment still terminates FAILED through the existing bounded lifecycle.
    emitted_names = [name for name, _details in emitted]
    assert "planner_fallback_failed" not in emitted_names
    assert "practice_failed" in emitted_names
    assert "ASSESSMENT_FAILED" in emitted_names


def test_duplicate_graph_operation_and_restart_are_idempotent() -> None:
    assessments, questions, processed = run_job(20, reuse_count=5)
    assert processed > 1
    assert len(questions.linked) == len(set(questions.linked)) == 20
    assert assessments.item["status"] == "READY"


def test_orchestrator_restart_resumes_persisted_groups_without_duplicate_links() -> None:
    request = make_request(10)
    assessments = FakeAssessments(request)
    questions = FakeQuestions(assessments)
    first_runner = PracticeGraphRunner(build_orchestrator(assessments, questions))
    first_runner.process(PracticeGraphCommand(operation="plan_and_fill", test_id="test-1"))
    meta = json.loads(assessments.item["meta"])
    first_group_id = next(iter(meta["generationGroups"]))
    first_runner.process(
        PracticeGraphCommand(
            operation="generate_group",
            test_id="test-1",
            group_id=first_group_id,
        )
    )

    restarted_runner = PracticeGraphRunner(build_orchestrator(assessments, questions))
    while assessments.item["status"] == "GENERATING":
        meta = json.loads(assessments.item["meta"])
        pending = [
            group_id
            for group_id, group in meta["generationGroups"].items()
            if group["state"] == "PENDING"
        ]
        if not pending:
            break
        for group_id in pending:
            restarted_runner.process(
                PracticeGraphCommand(
                    operation="generate_group",
                    test_id="test-1",
                    group_id=group_id,
                )
            )

    assert assessments.item["status"] == "READY"
    assert len(questions.linked) == len(set(questions.linked)) == 10


def test_generation_group_is_durably_claimed_before_generation() -> None:
    request = make_request(5)
    assessments = FakeAssessments(request)
    questions = FakeQuestions(assessments)
    runner = PracticeGraphRunner(build_orchestrator(assessments, questions))
    runner.process(PracticeGraphCommand(operation="plan_and_fill", test_id="test-1"))
    groups = json.loads(assessments.item["meta"])["generationGroups"]
    group_id = next(iter(groups))

    runner.process(
        PracticeGraphCommand(operation="generate_group", test_id="test-1", group_id=group_id)
    )

    assert assessments.claimed_group_ids == [group_id]


def test_partial_group_keeps_valid_items_and_repairs_only_deficit() -> None:
    assessments, questions, _ = run_job(5, partial_once=True)
    assert (assessments.item["status"], len(questions.linked)) == ("READY", 5)
    groups = json.loads(assessments.item["meta"])["generationGroups"]
    assert any(
        group.get("attemptStage") == "ITEM_RETRY"
        and group.get("itemRetryCount") == 1
        and group.get("state") == "COMPLETED"
        for group in groups.values()
    )


def test_provider_exception_during_item_retry_fails_without_content_replacement() -> None:
    assessments, questions, _ = run_job(
        5,
        partial_once=True,
        fail_item_retry_once=True,
    )
    groups = json.loads(assessments.item["meta"])["generationGroups"]
    assert (assessments.item["status"], len(questions.linked)) == ("FAILED", 3)
    assert json.loads(assessments.item["meta"])["errorCode"] == (
        "PRACTICE_GENERATION_UNEXPECTED_FAILURE"
    )
    assert not any(group.get("attemptStage") == "REPLACEMENT" for group in groups.values())


def test_provider_failure_never_makes_short_assessment_playable() -> None:
    assessments, questions, _ = run_job(5, fail_always=True)
    assert (assessments.item["status"], assessments.item["live"], len(questions.linked)) == (
        "FAILED",
        False,
        0,
    )
    assert assessments.failed_publications == 1
    assert assessments.ready_publications == 0


def test_exhausted_generation_emits_one_standard_failed_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        orchestration_module,
        "emit_practice_event",
        lambda event_name, **kwargs: events.append((event_name, kwargs.get("details") or {})),
    )

    assessments, _questions, _ = run_job(5, fail_always=True)

    assert assessments.item["status"] == "FAILED"
    assert [name for name, _details in events if name == "practice_generation_terminal"] == [
        "practice_generation_terminal"
    ]
    assert [name for name, _details in events if name == "practice_failed"] == [
        "practice_failed"
    ]


def test_verifier_reason_is_redacted_at_practice_event_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_reason = "THE_ANSWER_IS_SECRET"
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        orchestration_module,
        "emit_practice_event",
        lambda event_name, **kwargs: events.append((event_name, kwargs.get("details") or {})),
    )

    assessments, _questions, _ = run_job(5, verifier=SensitiveReasonVerifier())

    assert assessments.item["status"] == "READY"
    assert raw_reason not in str(events)
    verification_reasons = [
        details.get("reasonCode")
        for name, details in events
        if name == "QUESTION_VERIFICATION_RESULT"
    ]
    assert verification_reasons
    assert set(verification_reasons) == {"VERIFIER_APPROVED"}


def test_token_exhaustion_fails_without_question_repair_or_regeneration() -> None:
    request = make_request(5)
    assessments = FakeAssessments(request)
    questions = FakeQuestions(assessments, [reusable_item(0), reusable_item(1)])
    orchestrator = build_orchestrator(assessments, questions)
    generator = TokenExhaustedGenerator()
    orchestrator._generator = generator
    runner = PracticeGraphRunner(orchestrator)
    runner.process(PracticeGraphCommand(operation="plan_and_fill", test_id="test-1"))
    groups = json.loads(assessments.item["meta"])["generationGroups"]
    group_id = next(iter(groups))

    runner.process(
        PracticeGraphCommand(operation="generate_group", test_id="test-1", group_id=group_id)
    )

    meta = json.loads(assessments.item["meta"])
    assert (assessments.item["status"], len(questions.linked), generator.calls) == (
        "FAILED",
        2,
        1,
    )
    assert meta["errorCode"] == "PRACTICE_GENERATOR_OUTPUT_TOKEN_EXHAUSTED"
    assert meta["generationGroups"][group_id]["state"] == "FAILED"
    assert all("-retry-" not in generated_group for generated_group in meta["generationGroups"])


def test_provider_failure_fails_without_content_repair_or_regeneration(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        orchestration_module,
        "emit_practice_event",
        lambda event_name, **_kwargs: events.append(event_name),
    )
    request = make_request(5)
    assessments = FakeAssessments(request)
    questions = FakeQuestions(assessments, [reusable_item(0), reusable_item(1)])
    orchestrator = build_orchestrator(assessments, questions)
    generator = ProviderFailureGenerator()
    orchestrator._generator = generator
    runner = PracticeGraphRunner(orchestrator)
    runner.process(PracticeGraphCommand(operation="plan_and_fill", test_id="test-1"))
    group_id = next(iter(json.loads(assessments.item["meta"])["generationGroups"]))

    runner.process(
        PracticeGraphCommand(operation="generate_group", test_id="test-1", group_id=group_id)
    )

    meta = json.loads(assessments.item["meta"])
    assert (assessments.item["status"], generator.calls) == ("FAILED", 1)
    assert meta["errorCode"] == "PRACTICE_GENERATOR_PROVIDER_FAILED"
    assert "practice_model_execution_failed" in events
    assert "question_repair_requested" not in events
    assert "practice_validation_repair_started" not in events
    assert all("-retry-" not in group for group in meta["generationGroups"])


def test_each_accepted_generation_group_and_ready_publish_once() -> None:
    assessments, questions, _ = run_job(5)

    assert len(questions.linked) == 5
    assert assessments.group_progress_publications == 2
    assert assessments.ready_publications == 1
    assert assessments.failed_publications == 0


def test_temporary_gsi_lag_does_not_block_manifest_authoritative_ready() -> None:
    assessments, questions, _ = run_job(5, final_gsi_lag_once=True)
    meta = json.loads(assessments.item["meta"])
    assert assessments.item["status"] == "READY"
    assert questions.final_gsi_lag_used is False
    assert len(meta["readyQuestionIds"]) == 5


def test_persistent_gsi_lag_cannot_make_complete_manifest_fail() -> None:
    assessments, _questions, _ = run_job(5, final_gsi_lag_always=True)
    meta = json.loads(assessments.item["meta"])
    assert (assessments.item["status"], meta["errorCode"]) == ("READY", None)


def test_duplicate_finalization_attempt_is_harmless() -> None:
    request = make_request(5)
    assessments = FakeAssessments(request)
    assert assessments.record_finalization_retry(
        "test-1",
        next_attempt=1,
        reason_code="QUESTION_GSI_NOT_YET_CONSISTENT",
    )
    assert not assessments.record_finalization_retry(
        "test-1",
        next_attempt=1,
        reason_code="QUESTION_GSI_NOT_YET_CONSISTENT",
    )


def test_authoritative_counter_mismatch_fails_without_false_ready() -> None:
    assessments, _questions, _ = run_job(5, drop_counter_once=True)
    meta = json.loads(assessments.item["meta"])
    assert assessments.item["status"] == "FAILED"
    assert meta["errorCode"] == "AUTHORITATIVE_COUNTER_MISMATCH"


def test_manifest_question_owned_by_another_test_never_becomes_ready() -> None:
    assessments, _questions, _ = run_job(
        5,
        corrupt_manifest_ownership=True,
    )
    meta = json.loads(assessments.item["meta"])
    assert assessments.item["status"] == "FAILED"
    assert meta["errorCode"] == "QUESTION_MANIFEST_OWNERSHIP_MISMATCH"


def test_mandatory_verification_rejection_repairs_only_missing_item() -> None:
    assessments, questions, _ = run_job(5, reject_once=True)
    assert assessments.item["status"] == "READY"
    assert len(questions.linked) == 5


def test_all_reusable_uses_zero_generation_groups() -> None:
    assessments, questions, processed = run_job(10, reuse_count=10)
    meta = json.loads(assessments.item["meta"])
    assert (assessments.item["status"], len(questions.linked), meta["generatedCount"]) == (
        "READY",
        10,
        0,
    )
    assert processed == 1
    assert questions.reuse_query_order
    assert questions.reuse_query_order[0] == "topic"
    assert "category" not in questions.reuse_query_order


def test_topic_reuse_paginates_until_later_page_fills_deficit() -> None:
    ineligible = reusable_item(0)
    ineligible_meta = json.loads(ineligible["meta"])
    ineligible_meta["status"] = "INACTIVE"
    ineligible["meta"] = json.dumps(ineligible_meta)
    later_matches = [reusable_item(index) for index in range(1, 6)]

    assessments, questions, _ = run_job(
        5,
        topic_pages=[[ineligible], later_matches],
    )
    meta = json.loads(assessments.item["meta"])

    assert assessments.item["status"] == "READY"
    assert meta["reusedCount"] == 5
    assert questions.reuse_query_order == ["topic", "topic"]


def test_topic_reuse_page_bound_exhaustion_generates_only_remaining_deficit() -> None:
    pages: list[list[dict]] = []
    for index in range(3):
        item = reusable_item(index)
        details = json.loads(item["meta"])
        details["status"] = "INACTIVE"
        item["meta"] = json.dumps(details)
        pages.append([item])

    assessments, questions, _ = run_job(5, topic_pages=pages)
    meta = json.loads(assessments.item["meta"])

    assert assessments.item["status"] == "READY"
    assert meta["generatedCount"] == 5
    assert questions.reuse_query_order == ["topic", "topic"]


def test_cancellation_stops_future_group_claims() -> None:
    request = make_request(5)
    assessments = FakeAssessments(request)
    assessments.cancel("test-1")
    assert (assessments.item["status"], assessments.item["live"]) == ("ARCHIVED", False)


def test_repeated_five_question_reliability_20_runs() -> None:
    for _ in range(20):
        assessments, questions, _processed = run_job(5, reuse_count=2, partial_once=True)
        assert assessments.item["status"] == "READY"
        assert len(questions.linked) == 5


def test_repeated_fifty_question_reliability_10_runs() -> None:
    def execute(_index):
        return run_job(
            50,
            reuse_count=20,
            partial_once=True,
            final_gsi_lag_once=True,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(execute, range(10)))
    for assessments, questions, _processed in results:
        assert assessments.item["status"] == "READY"
        assert len(questions.linked) == 50
        assert len(assessments.item["meta"].encode("utf-8")) < 50_000


def test_repeated_hundred_question_reliability_3_runs() -> None:
    for _ in range(3):
        assessments, questions, _processed = run_job(
            100,
            reuse_count=40,
            partial_once=True,
            final_gsi_lag_once=True,
            reject_once=True,
        )
        assert assessments.item["status"] == "READY"
        assert len(questions.linked) == 100
        assert len(assessments.item["meta"].encode("utf-8")) < 100_000


def test_legacy_bucket_path_never_promotes_even_when_promotion_is_enabled() -> None:
    """generate_group() has no planner slot and may only be STRUCTURAL-verified.

    Promotion is reserved for the planner-slot path, which is always
    MANDATORY / INDEPENDENT_MODEL_V2, so the legacy path must never write a
    reusable QuestionBank row.
    """
    _, questions, _ = run_job(3, question_bank_promotion_enabled=True)

    assert len(questions.linked) == 3
    assert questions.promoted == []


class _SlotCommitQuestions:
    """Minimal question repository for the planner-slot commit path."""

    def __init__(self) -> None:
        self.promoted: list[dict] = []
        self.linked: list[dict] = []

    def list_linked(self, _test_id, *, limit=100):
        return []

    def persist_verified_question(
        self, *, test_id, question, slot, language, pattern_id=None, pattern_version_hash=None
    ):
        self.promoted.append(
            {"slotId": slot.slot_id, "patternId": pattern_id, "language": language}
        )
        return f"qb-v2-{slot.slot_id}"

    def link_generated(self, *, test_id, question, source_question_bank_id=None, **_kwargs):
        self.linked.append(
            {"slotId": _kwargs.get("slot_id"), "sourceQuestionBankId": source_question_bank_id}
        )
        return True


def _slot_commit_fixture(promotion_enabled: bool, *, selection=None):
    from features.practice_generation.orchestration import (
        _SlotGenerationContext,
        _SlotGenerationOutcome,
        _VerifiedSlotQuestion,
    )
    from features.practice_generation.schemas import (
        DemandBucket,
        GeneratedQuestion,
        GenerationGroup,
        PlannerSlot,
        VerificationResult,
    )

    request = make_request(1)
    assessments = FakeAssessments(request)
    assessments.item["status"] = "GENERATING"
    assessments.item["meta"] = json.dumps(
        {"generationGroups": {"group-1": {"groupId": "group-1", "state": "RUNNING"}}}
    )
    questions = _SlotCommitQuestions()
    orchestrator = build_orchestrator(
        assessments, questions, question_bank_promotion_enabled=promotion_enabled
    )
    slot = PlannerSlot(
        slot_id="slot-001",
        subject_id="math",
        topic_id="geometry",
        category_id="geometry",
        difficulty="basic",
        complexity="low",
        exam_ids=["CAT"],
        question_type="mcq",
        target_skill="triangle_angle_sum",
        variation_hint="vary_angles",
        generator_route_hint="math.generator.basic",
    )
    question = GeneratedQuestion(
        generation_item_id="item-slot-001",
        bucket_id="slot-bucket-001",
        slot_id="slot-001",
        question="Two angles of a triangle are 55 and 75 degrees. Find the third.",
        question_type="mcq",
        options=["40", "50", "60", "70"],
        correct_answer="50",
        solution="Angles of a triangle sum to 180 degrees.",
        subject="math",
        topic="geometry",
        difficulty="basic",
    )
    bucket = DemandBucket(
        bucket_id="slot-bucket-001",
        subject="math",
        topic="geometry",
        difficulty="basic",
        question_type="mcq",
        required_count=1,
        question_intent="Find an unknown interior angle.",
        verification_policy="MANDATORY",
    )
    context = _SlotGenerationContext(
        test_id="test-1",
        request=request,
        blueprint=None,
        group=GenerationGroup(
            group_id="group-1",
            bucket_id="slot-bucket-001",
            required_count=1,
            slot_ids=["slot-001"],
        ),
        bucket=bucket,
        slots=(slot,),
        excluded_texts=(),
        pattern_guidance_by_slot={},
        pattern_selections_by_slot=({"slot-001": selection} if selection else {}),
    )
    outcome = _SlotGenerationOutcome(
        context=context,
        accepted=(
            _VerifiedSlotQuestion(
                slot=slot,
                question=question,
                verification=VerificationResult(
                    generation_item_id="item-slot-001", approved=True, reason_code="OK"
                ),
            ),
        ),
        unresolved_slot_ids=(),
        reason_codes=("VERIFIED",),
        route_id="route-1",
        model="model-1",
        replacement_wave_count=0,
    )
    return orchestrator, questions, outcome


def test_slot_path_promotes_with_pattern_reuse_off_and_no_pattern_selection() -> None:
    orchestrator, questions, outcome = _slot_commit_fixture(True)

    orchestrator._commit_slot_outcome(outcome)

    assert len(questions.promoted) == 1
    assert questions.promoted[0]["patternId"] is None
    assert questions.linked[0]["sourceQuestionBankId"] == "qb-v2-slot-001"


def test_slot_path_skips_promotion_while_the_flag_is_off() -> None:
    orchestrator, questions, outcome = _slot_commit_fixture(False)

    orchestrator._commit_slot_outcome(outcome)

    assert questions.promoted == []
    assert questions.linked[0]["sourceQuestionBankId"] is None


def test_promotion_only_ever_covers_independently_accepted_questions() -> None:
    """Promotion iterates outcome.accepted, which _execute_slot_group only fills
    when verification binding, independent answer agreement, and approval all hold."""
    import dataclasses

    orchestrator, questions, outcome = _slot_commit_fixture(True)
    empty = dataclasses.replace(outcome, accepted=(), unresolved_slot_ids=("slot-001",))

    orchestrator._commit_slot_outcome(empty)

    assert questions.promoted == []


def test_terminal_verifier_rejection_promotes_nothing() -> None:
    import dataclasses

    orchestrator, questions, outcome = _slot_commit_fixture(True)
    rejected = dataclasses.replace(outcome, terminal_rejection=True)

    orchestrator._commit_slot_outcome(rejected)

    assert questions.promoted == []


# --- localized replacement characterization (TEST C / TEST D) ----------------

def test_item_retry_requests_only_the_missing_item_not_approved_siblings() -> None:
    """TEST C: a partial group must re-request only the deficit.

    Extends test_partial_group_keeps_valid_items_and_repairs_only_deficit, which
    proved the end state but not what the retry actually asked the generator for.
    """
    assessments, questions, _ = run_job(5, partial_once=True)
    generator = _LAST_GENERATOR["generator"]

    assert (assessments.item["status"], len(questions.linked)) == ("READY", 5)
    initial = [entry for entry in generator.requests if entry[1] == 0]
    retries = [entry for entry in generator.requests if entry[1] > 0]

    # The five questions are planned across the normal groups first.
    assert sum(entry[2] for entry in initial) == 5
    assert retries, "a retry must have happened"
    # Only the single missing item is re-requested; valid siblings are not.
    assert all(entry[2] == 1 for entry in retries), generator.requests


def test_verification_rejection_replacement_requests_only_the_rejected_item() -> None:
    """TEST C (verification variant): approved siblings are not regenerated."""
    assessments, questions, _ = run_job(5, reject_once=True)
    generator = _LAST_GENERATOR["generator"]

    assert (assessments.item["status"], len(questions.linked)) == ("READY", 5)
    # attempt > 0 is a retry/replacement wave; attempt 0 entries are the initial groups.
    follow_ups = [entry for entry in generator.requests if entry[1] > 0]

    assert follow_ups, "a replacement must have happened"
    assert all(entry[2] == 1 for entry in follow_ups), generator.requests


def test_structured_parse_invalid_rejects_the_whole_model_response() -> None:
    """TEST D: characterization only — no per-candidate localization is claimed.

    A structurally invalid batch is rejected as a unit; recovery comes from the
    existing retry/replacement waves, not from salvaging valid siblings.
    """
    from features.practice_generation.generation import parse_partial_generation
    from features.practice_generation.schemas import GenerationGroup

    request = make_request(2)
    blueprint = LegacyBlueprintManager().build(request).blueprint
    bucket = blueprint.buckets[0]
    group = GenerationGroup(
        group_id="group-1", bucket_id=bucket.bucket_id, required_count=2
    )

    def _question(index: int, *, valid: bool) -> dict:
        item = {
            "generation_item_id": f"item-{index}",
            "bucket_id": bucket.bucket_id,
            "question": f"Valid sibling question number {index} about addition?",
            "question_type": bucket.question_type.value,
            "options": ["2", "3", "4", "5"],
            "correct_answer": "2",
            "solution": "Add one to the stated integer.",
            "subject": bucket.subject,
            "topic": bucket.topic,
            "difficulty": bucket.difficulty.value,
        }
        if not valid:
            del item["options"]
            del item["correct_answer"]
        return item

    mixed = parse_partial_generation(
        json.dumps({"questions": [_question(1, valid=True), _question(2, valid=False)]}),
        group=group,
        bucket=bucket,
        existing_normalized_texts=set(),
    )
    malformed = parse_partial_generation(
        "{not json",
        group=group,
        bucket=bucket,
        existing_normalized_texts=set(),
    )

    # A per-question schema failure is localized: the valid sibling survives.
    assert len(mixed.accepted) == 1
    assert mixed.rejected_count == 1
    # An unparseable response has no salvageable candidates at all.
    assert malformed.accepted == ()
    assert "STRUCTURED_PARSE_INVALID" in malformed.rejection_reason_codes


def test_replacement_wave_preserves_language_exam_stage_and_group_identity() -> None:
    """A rejected candidate must be regenerated in the same delivery context.

    Losing language/exam/stage on the replacement wave would silently deliver a
    question in the wrong language or for the wrong exam while the approved
    siblings stay correct, which no downstream gate would catch.
    """
    assessments, questions, _ = run_job(
        5,
        reject_once=True,
        language="hindi",
        exam_id="SSC_CGL",
        exam_stage="TIER_2",
    )
    generator = _LAST_GENERATOR["generator"]

    assert (assessments.item["status"], len(questions.linked)) == ("READY", 5)

    initial = [entry for entry in generator.contexts if entry[1] == 0]
    replacements = [entry for entry in generator.contexts if entry[1] > 0]
    assert initial, "an initial generation wave must have happened"
    assert replacements, "a replacement wave must have happened"

    for _group_id, _attempt, language, exam_id, exam_stage in generator.contexts:
        assert (language, exam_id, exam_stage) == ("hindi", "SSC_CGL", "TIER_2")

    # The replacement stays inside an existing group rather than opening a new one.
    initial_groups = {entry[0] for entry in initial}
    for group_id, *_ in replacements:
        assert group_id.split("-retry-")[0] in initial_groups, group_id

    # Sibling preservation: a replacement asks only for the rejected deficit.
    follow_ups = [entry for entry in generator.requests if entry[1] > 0]
    assert all(entry[2] == 1 for entry in follow_ups), generator.requests
