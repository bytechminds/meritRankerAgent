"""Guard regressions for schema-v2 slot generation and coordinator-only commits."""

from __future__ import annotations

import json
import threading
from copy import deepcopy

from features.practice_generation.config import PracticeGenerationConfig
from features.practice_generation.orchestration import PracticeGenerationOrchestrator
from features.practice_generation.pattern_context import NoOpPatternContextProvider
from features.practice_generation.planning import deterministic_blueprint
from features.practice_generation.schemas import (
    GeneratedBatch,
    PracticeGenerationRequest,
    VerificationResult,
)
from services.llm.orchestration.errors import ProviderExecutionError


def request() -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="request-v2",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-v2",
        original_query="Create three algebra questions",
        practice_type="QUIZ",
        requested_count=3,
        accepted_count=3,
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        assessment_title="Algebra Quiz",
    )


class Assessments:
    def __init__(self) -> None:
        blueprint = deterministic_blueprint(request())
        bucket_id = blueprint.buckets[0].bucket_id
        groups = {
            f"g{index}": {
                "groupId": f"g{index}",
                "bucketId": bucket_id,
                "requiredCount": 1,
                "slotIds": [f"slot-{index:03d}"],
                "state": "PENDING",
            }
            for index in range(1, 4)
        }
        self.item = {
            "testId": "test-v2",
            "userId": "user-1",
            "name": "Algebra Quiz",
            "status": "GENERATING",
            "live": False,
            "totalQuestions": 3,
            "updatedAt": "1",
            "meta": {
                "practiceRequest": {
                    "requestId": "request-v2",
                    "conversationId": "conversation-1",
                    "turnId": "turn-v2",
                    "practiceType": "QUIZ",
                    "requestedCount": 3,
                    "acceptedCount": 3,
                    "subject": "math",
                    "topic": "algebra",
                    "difficulty": "intermediate",
                    "language": "english",
                    "examId": "CAT",
                    "includeSolutions": True,
                    "assessmentTitle": "Algebra Quiz",
                },
                "blueprint": blueprint.model_dump(mode="json"),
                "generationGroups": groups,
                "slotReadyCounts": {f"slot-{index:03d}": 0 for index in range(1, 4)},
                "readyQuestionIds": [],
                "readyCount": 0,
                "readyQuestionCount": 0,
                "progressPercent": 10,
                "failedCount": 0,
                "playable": False,
            },
        }

    def get(self, _test_id: str):
        return deepcopy(self.item)


class Progress:
    def __init__(self, assessments: Assessments, questions: Questions) -> None:
        self.assessments = assessments
        self.questions = questions

    def claim_group(self, _test_id: str, group_id: str):
        group = self.assessments.item["meta"]["generationGroups"][group_id]
        if group["state"] != "PENDING":
            return None
        group["state"] = "RUNNING"
        return deepcopy(group)

    def update(self, _test_id: str, *, meta_updates, **_kwargs):
        self.assessments.item["meta"].update(deepcopy(meta_updates))
        linked = list(self.questions.linked.values())
        meta = self.assessments.item["meta"]
        meta["readyQuestionIds"] = sorted(self.questions.linked)
        meta["readyCount"] = len(linked)
        meta["readyQuestionCount"] = len(linked)
        meta["verifiedCount"] = len(linked)
        meta["generatedCount"] = len(linked)
        meta["reusedCount"] = 0
        counts: dict[str, int] = {}
        for item in linked:
            bucket = item["_practiceMeta"]["bucketId"]
            counts[bucket] = counts.get(bucket, 0) + 1
        meta["bucketReadyCounts"] = counts
        return deepcopy(self.assessments.item)

    def mark_failed(self, _test_id: str, error_code: str, **_kwargs):
        self.assessments.item["status"] = "FAILED"
        self.assessments.item["meta"]["errorCode"] = error_code


class Questions:
    def __init__(self) -> None:
        self.linked: dict[str, dict] = {}

    def list_linked(self, _test_id: str):
        return deepcopy(list(self.linked.values()))

    def link_generated(self, *, test_id, question, group_id, **_kwargs):
        question_id = f"question-{question.slot_id}"
        if question_id in self.linked:
            return False
        self.linked[question_id] = {
            "questionId": question_id,
            "testId": test_id,
            "question": question.question,
            "topic": question.topic,
            "difficulty": question.difficulty.value,
            "_practiceMeta": {
                "bucketId": question.bucket_id,
                "slotId": question.slot_id,
                "source": "GENERATED",
                "sourceType": "AI_GENERATED",
                "verified": True,
                "questionType": "mcq",
                "verificationMethod": "INDEPENDENT_MODEL_V2",
                "generationGroupId": group_id,
                "language": "english",
            },
        }
        return True


class ReverseGenerator:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.finish_order: list[str] = []

    def generate_slots(self, *, bucket, group, slots, replacement_wave, **_kwargs):
        assert replacement_wave == 0
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        slot = slots[0]
        question = {
            "schema_version": "2",
            "generation_item_id": f"item-{slot.slot_id}",
            "bucket_id": bucket.bucket_id,
            "slot_id": slot.slot_id,
            "question": f"For {slot.slot_id}, what is two plus two?",
            "question_type": "mcq",
            "options": [
                {"option_id": "0", "value": "1"},
                {"option_id": "1", "value": "2"},
                {"option_id": "2", "value": "3"},
                {"option_id": "3", "value": "4"},
            ],
            "correct_option_id": "3",
            "correct_answer": "4",
            "answer_explanation": "Two plus two equals four.",
            "solution": "Two plus two equals four.",
            "subject": slot.subject_id,
            "topic": slot.topic_id,
            "difficulty": slot.difficulty.value,
        }
        with self.lock:
            self.active -= 1
            self.finish_order.append(group.group_id)
        return GeneratedBatch(
            content=json.dumps({"questions": [question]}),
            route_id=slot.generator_route_hint,
            model="test-model",
        )


class Verifier:
    def verify_slot(self, *, slot, question, **_kwargs):
        return VerificationResult(
            schema_version="2",
            generation_item_id=question.generation_item_id,
            slot_id=slot.slot_id,
            decision="ACCEPT",
            independently_solved_option_id=question.correct_option_id,
            reason_codes=["INDEPENDENT_SOLUTION_MATCH"],
        )


def generated_question(*, bucket, slot) -> dict:
    return {
        "schema_version": "2",
        "generation_item_id": f"item-{slot.slot_id}",
        "bucket_id": bucket.bucket_id,
        "slot_id": slot.slot_id,
        "question": f"For {slot.slot_id}, what is two plus two?",
        "question_type": "mcq",
        "options": [
            {"option_id": "0", "value": "1"},
            {"option_id": "1", "value": "2"},
            {"option_id": "2", "value": "3"},
            {"option_id": "3", "value": "4"},
        ],
        "correct_option_id": "3",
        "correct_answer": "4",
        "answer_explanation": "Two plus two equals four.",
        "solution": "Two plus two equals four.",
        "subject": slot.subject_id,
        "topic": slot.topic_id,
        "difficulty": slot.difficulty.value,
    }


class WaveGenerator:
    def __init__(self, *, fail_provider: bool = False) -> None:
        self.fail_provider = fail_provider
        self.waves: list[int] = []

    def generate_slots(self, *, bucket, slots, replacement_wave, **_kwargs):
        self.waves.append(replacement_wave)
        if self.fail_provider:
            raise ProviderExecutionError("provider failed", failure_kind="timeout")
        question = generated_question(bucket=bucket, slot=slots[0])
        if replacement_wave == 2:
            question["question"] = (
                f"For {slots[0].slot_id}, in a fresh scenario, what is two plus two?"
            )
        return GeneratedBatch(
            content=json.dumps({"questions": [question]}),
            route_id=slots[0].generator_route_hint,
            model="test-model",
        )


class SelectiveProviderFailureGenerator:
    def __init__(self, *, failing_slot_id: str) -> None:
        self.failing_slot_id: str | None = failing_slot_id
        self.calls: list[tuple[str, ...]] = []

    def generate_slots(self, *, bucket, slots, replacement_wave, **_kwargs):
        self.calls.append(tuple(slot.slot_id for slot in slots))
        assert replacement_wave == 0
        if self.failing_slot_id in self.calls[-1]:
            raise ProviderExecutionError("provider failed", failure_kind="timeout")
        return GeneratedBatch(
            content=json.dumps(
                {
                    "questions": [
                        generated_question(bucket=bucket, slot=slot) for slot in slots
                    ]
                }
            ),
            route_id=slots[0].generator_route_hint,
            model="test-model",
        )


class MultiSlotProviderFailureGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def generate_slots(self, *, bucket, slots, replacement_wave, **_kwargs):
        self.calls.append(tuple(slot.slot_id for slot in slots))
        assert replacement_wave == 0
        if len(slots) > 1:
            raise ProviderExecutionError(
                "provider response exceeded output budget",
                failure_kind="output_token_exhausted",
            )
        return GeneratedBatch(
            content=json.dumps(
                {
                    "questions": [
                        generated_question(bucket=bucket, slot=slot) for slot in slots
                    ]
                }
            ),
            route_id=slots[0].generator_route_hint,
            model="test-model",
        )


class InvalidStructuredGenerator:
    def __init__(self) -> None:
        self.waves: list[int] = []

    def generate_slots(self, *, slots, replacement_wave, **_kwargs):
        self.waves.append(replacement_wave)
        return GeneratedBatch(
            content="{",
            route_id=slots[0].generator_route_hint,
            model="test-model",
        )


class DecisionVerifier:
    def __init__(self, first_decision: str) -> None:
        self.first_decision = first_decision
        self.calls = 0

    def verify_slot(self, *, slot, question, **_kwargs):
        self.calls += 1
        decision = self.first_decision if self.calls == 1 else "ACCEPT"
        return VerificationResult(
            schema_version="2",
            generation_item_id=question.generation_item_id,
            slot_id=slot.slot_id,
            decision=decision,
            independently_solved_option_id=(
                question.correct_option_id if decision == "ACCEPT" else None
            ),
            reason_codes=[
                "INDEPENDENT_SOLUTION_MATCH"
                if decision == "ACCEPT"
                else "ANSWER_EXPLANATION_MISMATCH"
            ],
        )


class SelectiveProviderFailureVerifier:
    def __init__(self, *, failing_slot_id: str) -> None:
        self.failing_slot_id = failing_slot_id
        self.calls: list[str] = []
        self._failed = False

    def verify_slot(self, *, slot, question, **_kwargs):
        self.calls.append(slot.slot_id)
        if slot.slot_id == self.failing_slot_id and not self._failed:
            self._failed = True
            raise ProviderExecutionError(
                "verifier output exceeded budget",
                failure_kind="output_token_exhausted",
            )
        return VerificationResult(
            schema_version="2",
            generation_item_id=question.generation_item_id,
            slot_id=slot.slot_id,
            decision="ACCEPT",
            independently_solved_option_id=question.correct_option_id,
            reason_codes=["INDEPENDENT_SOLUTION_MATCH"],
        )


def build_orchestrator(*, generator, verifier):
    assessments = Assessments()
    questions = Questions()
    progress = Progress(assessments, questions)
    orchestrator = PracticeGenerationOrchestrator(
        config=PracticeGenerationConfig(
            enabled=True,
            pattern_context_enabled=False,
            pattern_reuse_enabled=False,
            assessment_table="assessment",
            question_table="question",
            question_bank_table="bank",
            question_bank_category_index="category",
            question_test_index="test",
            aws_region="ap-south-1",
        ),
        assessments=assessments,
        progress=progress,
        questions=questions,
        blueprint_manager=None,
        generator=generator,
        verifier=verifier,
        pattern_context=NoOpPatternContextProvider(),
    )
    return orchestrator, assessments


def test_same_bucket_groups_are_serialized_before_their_exclusions_are_committed() -> None:
    assessments = Assessments()
    questions = Questions()
    progress = Progress(assessments, questions)
    generator = ReverseGenerator()
    orchestrator = PracticeGenerationOrchestrator(
        config=PracticeGenerationConfig(
            enabled=True,
            pattern_context_enabled=False,
            pattern_reuse_enabled=False,
            assessment_table="assessment",
            question_table="question",
            question_bank_table="bank",
            question_bank_category_index="category",
            question_test_index="test",
            aws_region="ap-south-1",
        ),
        assessments=assessments,
        progress=progress,
        questions=questions,
        blueprint_manager=None,
        generator=generator,
        verifier=Verifier(),
        pattern_context=NoOpPatternContextProvider(),
    )

    orchestrator.generate_wave("test-v2", ["g1", "g2"])

    groups = assessments.item["meta"]["generationGroups"]
    assert generator.max_active == 1
    assert generator.finish_order == ["g1"]
    assert [groups["g1"]["state"], groups["g2"]["state"], groups["g3"]["state"]] == [
        "COMPLETED",
        "PENDING",
        "PENDING",
    ]
    assert set(assessments.item["meta"]["readyQuestionIds"]) == {
        "question-slot-001",
    }


def test_repairable_question_uses_one_repair_wave_then_accepts() -> None:
    generator = WaveGenerator()
    orchestrator, assessments = build_orchestrator(
        generator=generator,
        verifier=DecisionVerifier("REPAIRABLE"),
    )

    orchestrator.generate_wave("test-v2", ["g1"])

    assert generator.waves == [0, 1]
    assert assessments.item["meta"]["generationGroups"]["g1"]["state"] == "COMPLETED"


def test_regenerate_decision_skips_repair_prompt_and_uses_replacement_wave() -> None:
    generator = WaveGenerator()
    orchestrator, assessments = build_orchestrator(
        generator=generator,
        verifier=DecisionVerifier("REGENERATE"),
    )

    orchestrator.generate_wave("test-v2", ["g1"])

    assert generator.waves == [0, 2]
    assert assessments.item["meta"]["generationGroups"]["g1"]["state"] == "COMPLETED"


def test_provider_failure_does_not_enter_content_repair() -> None:
    generator = WaveGenerator(fail_provider=True)
    orchestrator, assessments = build_orchestrator(
        generator=generator,
        verifier=Verifier(),
    )

    orchestrator.generate_wave("test-v2", ["g1"])

    assert generator.waves == [0]
    group = assessments.item["meta"]["generationGroups"]["g1"]
    assert assessments.item["status"] == "GENERATING"
    assert group["state"] == "PENDING"
    assert group["attemptStage"] == "PROVIDER_REPLACEMENT"
    assert group["providerReplacementCount"] == 1
    assert group["slotIds"] == ["slot-001"]

    orchestrator.generate_wave("test-v2", ["g1"])

    assert generator.waves == [0, 0]
    assert assessments.item["status"] == "FAILED"
    assert assessments.item["meta"]["errorCode"] == "PRACTICE_REPLACEMENT_EXHAUSTED"


def test_structured_generator_exhaustion_has_a_specific_terminal_code() -> None:
    generator = InvalidStructuredGenerator()
    orchestrator, assessments = build_orchestrator(
        generator=generator,
        verifier=Verifier(),
    )

    orchestrator.generate_wave("test-v2", ["g1"])

    group = assessments.item["meta"]["generationGroups"]["g1"]
    assert generator.waves == [0, 1, 2]
    assert assessments.item["status"] == "FAILED"
    assert group["errorCode"] == "PRACTICE_GENERATOR_OUTPUT_INVALID"
    assert assessments.item["meta"]["errorCode"] == "PRACTICE_GENERATOR_OUTPUT_INVALID"


def test_provider_replacement_preserves_verified_slots_and_regenerates_only_deficit() -> None:
    generator = SelectiveProviderFailureGenerator(failing_slot_id="slot-002")
    orchestrator, assessments = build_orchestrator(
        generator=generator,
        verifier=Verifier(),
    )

    orchestrator.generate_wave("test-v2", ["g1"])
    orchestrator.generate_wave("test-v2", ["g2"])

    groups = assessments.item["meta"]["generationGroups"]
    assert groups["g1"]["state"] == "COMPLETED"
    assert groups["g2"]["state"] == "PENDING"
    assert groups["g2"]["slotIds"] == ["slot-002"]
    assert set(orchestrator._questions.linked) == {"question-slot-001"}

    generator.failing_slot_id = None
    orchestrator.generate_wave("test-v2", ["g2"])

    assert generator.calls == [
        ("slot-001",),
        ("slot-002",),
        ("slot-002",),
    ]
    assert set(orchestrator._questions.linked) == {
        "question-slot-001",
        "question-slot-002",
    }


def test_provider_replacement_splits_a_multi_slot_deficit_once_per_slot() -> None:
    generator = MultiSlotProviderFailureGenerator()
    orchestrator, assessments = build_orchestrator(
        generator=generator,
        verifier=Verifier(),
    )
    groups = assessments.item["meta"]["generationGroups"]
    groups["g1"].update(
        {
            "slotIds": ["slot-001", "slot-002"],
            "requiredCount": 2,
        }
    )
    del groups["g2"]

    orchestrator.generate_wave("test-v2", ["g1"])

    groups = assessments.item["meta"]["generationGroups"]
    replacement_group_ids = sorted(groups)
    assert replacement_group_ids == ["g1", "g1-p1-2", "g3"]
    assert groups["g1"]["state"] == "PENDING"
    assert groups["g1"]["slotIds"] == ["slot-001"]
    assert groups["g1"]["providerReplacementCount"] == 1
    assert groups["g1-p1-2"]["state"] == "PENDING"
    assert groups["g1-p1-2"]["slotIds"] == ["slot-002"]
    assert groups["g1-p1-2"]["providerReplacementCount"] == 1

    orchestrator.generate_wave("test-v2", ["g1"])
    orchestrator.generate_wave("test-v2", ["g1-p1-2"])

    assert generator.calls == [
        ("slot-001", "slot-002"),
        ("slot-001",),
        ("slot-002",),
    ]
    assert set(orchestrator._questions.linked) == {
        "question-slot-001",
        "question-slot-002",
    }


def test_verifier_provider_failure_preserves_verified_slots_without_content_repair() -> None:
    generator = SelectiveProviderFailureGenerator(failing_slot_id="not-a-slot")
    verifier = SelectiveProviderFailureVerifier(failing_slot_id="slot-002")
    orchestrator, assessments = build_orchestrator(
        generator=generator,
        verifier=verifier,
    )
    groups = assessments.item["meta"]["generationGroups"]
    groups["g1"].update(
        {
            "slotIds": ["slot-001", "slot-002"],
            "requiredCount": 2,
        }
    )
    del groups["g2"]

    orchestrator.generate_wave("test-v2", ["g1"])

    group = assessments.item["meta"]["generationGroups"]["g1"]
    assert assessments.item["status"] == "GENERATING"
    assert group["state"] == "PENDING"
    assert group["attemptStage"] == "PROVIDER_REPLACEMENT"
    assert group["lastReasonCode"] == "PRACTICE_VERIFIER_FALLBACK_EXHAUSTED"
    assert group["slotIds"] == ["slot-002"]
    assert set(orchestrator._questions.linked) == {"question-slot-001"}
    assert generator.calls == [("slot-001", "slot-002")]

    orchestrator.generate_wave("test-v2", ["g1"])

    assert generator.calls == [
        ("slot-001", "slot-002"),
        ("slot-002",),
    ]
    assert verifier.calls == ["slot-001", "slot-002", "slot-002"]
    assert set(orchestrator._questions.linked) == {
        "question-slot-001",
        "question-slot-002",
    }
