"""Existing model-routing adapters for structured practice model calls."""

from __future__ import annotations

import json
from typing import Any

from features.practice_generation.planning import select_planner_family
from features.practice_generation.schemas import (
    DemandBucket,
    GeneratedBatch,
    GeneratedQuestion,
    GenerationGroup,
    PlannerSlot,
    PracticeGenerationRequest,
    VerificationResult,
)
from schemas.llm_routing import RouteRequest
from services.doubt_solver.exam_profile_cache import get_exam_profile_runtime
from services.llm.orchestration.orchestrator import LlmOrchestrator


def _execute(
    orchestrator: LlmOrchestrator,
    *,
    request_id: str,
    subject: str,
    task_role: str,
    difficulty: str,
    language: str,
    prompt_name: str,
    payload: dict[str, Any],
) -> GeneratedBatch:
    result = orchestrator.generate_structured(
        route_request=RouteRequest(
            request_id=request_id,
            subject=subject,
            task_role=task_role,
            difficulty=difficulty,
            intent="practice",
            language=language,
        ),
        user_content=json.dumps(payload, separators=(",", ":"), sort_keys=True),
        prompt=f"{prompt_name}.md",
        overlays=["practice_generation/shared_contract.md"],
    )
    return GeneratedBatch(
        content=result.content,
        route_id=result.route_decision.route_id,
        model=result.model,
    )


class RoutedPlannerProvider:
    def __init__(self, orchestrator: LlmOrchestrator) -> None:
        self._orchestrator = orchestrator

    def plan(
        self,
        request: PracticeGenerationRequest,
        *,
        tier: str,
        repair_feedback: str | None = None,
    ) -> str:
        difficulty = {"light": "basic", "standard": "intermediate", "strong": "advanced"}[tier]
        family = select_planner_family(request)
        profile_resolution = get_exam_profile_runtime().resolve(
            exam_profile_id=request.exam_profile_id,
            exam_id=request.exam_id,
            exam_stage=request.exam_stage,
            subject=request.subject,
            full_mock=request.practice_type.value == "FULL_MOCK",
        )
        payload = {
            "schema_version": "2",
            "planner_family": family.value,
            "practice_type": request.practice_type.value,
            "accepted_count": request.accepted_count,
            "subject": request.subject,
            "topic": request.topic,
            "difficulty": request.difficulty.value,
            "exam_id": request.exam_id,
            "exam_stage": request.exam_stage,
            "exam_profile_id": request.exam_profile_id,
            "request_constraints": request.original_query[:1000],
            "repair_reason": repair_feedback,
            "supported_question_types": ["mcq"],
            "required_slot_ids": [
                f"slot-{index:03d}" for index in range(1, request.accepted_count + 1)
            ],
        }
        if profile_resolution.context is not None:
            payload["exam_profile"] = profile_resolution.context.as_planner_payload()
        return _execute(
            self._orchestrator,
            request_id=request.request_id,
            subject=family.value,
            task_role="planner",
            difficulty=difficulty,
            language=request.language,
            prompt_name=f"practice_generation/planners/{family.value}",
            payload=payload,
        ).content


class RoutedQuestionGenerator:
    def __init__(self, orchestrator: LlmOrchestrator) -> None:
        self._orchestrator = orchestrator

    def generate(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        group: GenerationGroup,
        exclude_normalized_texts: tuple[str, ...],
    ) -> GeneratedBatch:
        return _execute(
            self._orchestrator,
            request_id=request.request_id,
            subject=bucket.subject,
            task_role="generator",
            difficulty=bucket.difficulty.value,
            language=request.language,
            prompt_name="practice_generation/question_generator",
            payload={
                "bucket": bucket.model_dump(mode="json"),
                "group_id": group.group_id,
                "count": group.required_count,
                "language": request.language,
                "exam_id": request.exam_id,
                "exam_stage": request.exam_stage,
                "include_solutions": request.include_solutions,
                "excluded_question_snippets": [
                    value[:180] for value in exclude_normalized_texts[-20:]
                ],
            },
        )

    def generate_slots(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        group: GenerationGroup,
        slots: tuple[PlannerSlot, ...],
        exclude_normalized_texts: tuple[str, ...],
        repair_feedback: tuple[str, ...] = (),
        replacement_wave: int = 0,
    ) -> GeneratedBatch:
        prompt_name = (
            "practice_generation/question_generator_v2"
            if replacement_wave == 0
            else "practice_generation/question_repair"
            if replacement_wave == 1
            else "practice_generation/question_regenerator"
        )
        return _execute(
            self._orchestrator,
            request_id=request.request_id,
            subject=bucket.subject,
            task_role="generator",
            difficulty=bucket.difficulty.value,
            language=request.language,
            prompt_name=prompt_name,
            payload={
                "schema_version": "2",
                "bucket": bucket.model_dump(mode="json"),
                "group_id": group.group_id,
                "count": len(slots),
                "slots": [slot.model_dump(mode="json") for slot in slots],
                "language": request.language,
                "exam_id": request.exam_id,
                "exam_stage": request.exam_stage,
                "include_solutions": request.include_solutions,
                "repair_reason_codes": list(repair_feedback)[:8],
                "replacement_wave": replacement_wave,
                "excluded_question_snippets": [
                    value[:180] for value in exclude_normalized_texts[-20:]
                ],
                "required_answer_contract": {
                    "option_ids": ["0", "1", "2", "3"],
                    "answer_status": "PENDING_VERIFICATION",
                    "answer_version": 1,
                },
            },
        )


class RoutedQuestionVerifier:
    def __init__(self, orchestrator: LlmOrchestrator) -> None:
        self._orchestrator = orchestrator

    def verify(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        question: GeneratedQuestion,
    ) -> VerificationResult:
        raw = _execute(
            self._orchestrator,
            request_id=request.request_id,
            subject="general",
            task_role="verifier",
            difficulty="default",
            language=request.language,
            prompt_name="practice_generation/question_verifier",
            payload={
                "bucket": bucket.model_dump(mode="json"),
                "question": question.model_dump(mode="json"),
                "language": request.language,
            },
        )
        return VerificationResult.model_validate(json.loads(raw.content))

    def verify_slot(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        slot: PlannerSlot,
        question: GeneratedQuestion,
    ) -> VerificationResult:
        raw = _execute(
            self._orchestrator,
            request_id=request.request_id,
            subject="general",
            task_role="verifier",
            difficulty="default",
            language=request.language,
            prompt_name="practice_generation/question_verifier_v2",
            payload={
                "schema_version": "2",
                "slot": slot.model_dump(mode="json"),
                "question": {
                    "schema_version": question.schema_version,
                    "generation_item_id": question.generation_item_id,
                    "slot_id": question.slot_id,
                    "question": question.question,
                    "question_type": question.question_type.value,
                    "options": [
                        option.model_dump(mode="json") for option in question.canonical_options
                    ],
                    "submitted_answer": {
                        "correct_option_id": question.correct_option_id,
                        "correct_answer": question.correct_answer,
                    },
                    "answer_explanation": question.answer_explanation,
                },
                "language": request.language,
                "instruction": "Solve independently before comparing the submitted option ID.",
            },
        )
        return VerificationResult.model_validate(json.loads(raw.content))
