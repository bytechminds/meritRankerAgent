"""Existing model-routing adapters for structured practice model calls."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
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
from retrieval.pattern_intelligence import PatternGenerationContext
from schemas.llm_routing import PracticeGenerationWorkload, RouteRequest
from services.doubt_solver.exam_profile_cache import get_exam_profile_runtime
from services.llm.orchestration.orchestrator import LlmOrchestrator
from services.llm.orchestration.prompt_budget import PromptInputBudget

logger = logging.getLogger(__name__)

_PRACTICE_OVERLAYS = ["practice_generation/shared_contract.md"]
_PRACTICE_GENERATOR_OVERLAYS = [
    *_PRACTICE_OVERLAYS,
    "practice_generation/generator_output_policy.md",
]
_DEFAULT_PATTERN_INPUT_MAX_TOKENS = 3_800


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
    prompt_input_budget: PromptInputBudget | None = None,
    practice_generation_workload: PracticeGenerationWorkload | None = None,
) -> GeneratedBatch:
    overlays = (
        _PRACTICE_GENERATOR_OVERLAYS if task_role == "generator" else _PRACTICE_OVERLAYS
    )
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
        overlays=overlays,
        practice_generation_workload=practice_generation_workload,
    )
    return GeneratedBatch(
        content=result.content,
        route_id=result.route_decision.route_id,
        model=result.model,
        prompt_input_budget=prompt_input_budget,
    )


def _pattern_guidance_payload(
    guidance_by_slot: Mapping[str, PatternGenerationContext] | None,
) -> list[dict[str, object]]:
    """Serialize one redacted Pattern context once for each compatible slot set."""
    if not guidance_by_slot:
        return []
    grouped: dict[str, tuple[PatternGenerationContext, list[str]]] = {}
    for slot_id, context in guidance_by_slot.items():
        if not slot_id or not isinstance(context, PatternGenerationContext):
            continue
        current = grouped.get(context.pattern_id)
        if current is None:
            grouped[context.pattern_id] = (context, [slot_id])
        else:
            current[1].append(slot_id)
    return [
        {
            "slot_ids": sorted(slot_ids),
            "context": context.model_dump(by_alias=True, exclude_none=True),
        }
        for _pattern_id, (context, slot_ids) in sorted(grouped.items())
    ]


def _canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _combine_prompt_budgets(budgets: list[PromptInputBudget]) -> PromptInputBudget | None:
    if not budgets:
        return None
    max_input_tokens = budgets[0].max_input_tokens
    return PromptInputBudget(
        static_instruction_tokens=sum(item.static_instruction_tokens for item in budgets),
        current_question_tokens=sum(item.current_question_tokens for item in budgets),
        exam_context_tokens=sum(item.exam_context_tokens for item in budgets),
        slot_context_tokens=sum(item.slot_context_tokens for item in budgets),
        pattern_context_tokens=sum(item.pattern_context_tokens for item in budgets),
        reference_tokens=sum(item.reference_tokens for item in budgets),
        other_dynamic_tokens=sum(item.other_dynamic_tokens for item in budgets),
        total_input_tokens=sum(item.total_input_tokens for item in budgets),
        max_input_tokens=max_input_tokens,
        within_budget=all(item.within_budget for item in budgets),
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
            "planner_phase": "repair" if repair_feedback is not None else "initial",
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
    def __init__(
        self,
        orchestrator: LlmOrchestrator,
        *,
        pattern_input_max_tokens: int = _DEFAULT_PATTERN_INPUT_MAX_TOKENS,
    ) -> None:
        self._orchestrator = orchestrator
        self._pattern_input_max_tokens = max(pattern_input_max_tokens, 1)

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
            practice_generation_workload=PracticeGenerationWorkload(
                complexity="medium",
                slot_count=group.required_count,
            ),
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
        repair_candidates_by_slot: Mapping[str, GeneratedQuestion] | None = None,
        repair_reason_codes_by_slot: Mapping[str, tuple[str, ...]] | None = None,
        replacement_wave: int = 0,
        pattern_guidance_by_slot: Mapping[str, PatternGenerationContext] | None = None,
    ) -> GeneratedBatch:
        prompt_name = (
            "practice_generation/question_generator_v2"
            if replacement_wave == 0
            else "practice_generation/question_repair"
            if replacement_wave == 1
            else "practice_generation/question_regenerator"
        )
        guidance = dict(pattern_guidance_by_slot or {})
        initial_payload = self._slot_payload(
            request=request,
            bucket=bucket,
            group=group,
            slots=slots,
            exclude_normalized_texts=exclude_normalized_texts,
            repair_feedback=repair_feedback,
            repair_candidates_by_slot=repair_candidates_by_slot,
            repair_reason_codes_by_slot=repair_reason_codes_by_slot,
            replacement_wave=replacement_wave,
            guidance_by_slot=guidance,
        )
        initial_budget = self._measure_slot_budget(
            request=request,
            bucket=bucket,
            prompt_name=prompt_name,
            payload=initial_payload,
        )
        if not guidance or initial_budget.within_budget:
            return self._execute_slot_payload(
                request=request,
                bucket=bucket,
                prompt_name=prompt_name,
                payload=initial_payload,
                budget=initial_budget,
                slots=slots,
            )

        chunks = self._split_pattern_aware_slots(
            request=request,
            bucket=bucket,
            group=group,
            slots=slots,
            exclude_normalized_texts=exclude_normalized_texts,
            repair_feedback=repair_feedback,
            repair_candidates_by_slot=repair_candidates_by_slot,
            repair_reason_codes_by_slot=repair_reason_codes_by_slot,
            replacement_wave=replacement_wave,
            guidance_by_slot=guidance,
            prompt_name=prompt_name,
        )
        if len(chunks) == 1:
            return self._execute_bounded_pattern_chunk(
                request=request,
                bucket=bucket,
                group=group,
                slots=chunks[0],
                exclude_normalized_texts=exclude_normalized_texts,
                repair_feedback=repair_feedback,
                repair_candidates_by_slot=repair_candidates_by_slot,
                repair_reason_codes_by_slot=repair_reason_codes_by_slot,
                replacement_wave=replacement_wave,
                guidance_by_slot=guidance,
                prompt_name=prompt_name,
            )

        batches: list[GeneratedBatch] = []
        for chunk in chunks:
            chunk_group = group.model_copy(
                update={
                    "required_count": len(chunk),
                    "slot_ids": [slot.slot_id for slot in chunk],
                }
            )
            batches.append(
                self._execute_bounded_pattern_chunk(
                    request=request,
                    bucket=bucket,
                    group=chunk_group,
                    slots=chunk,
                    exclude_normalized_texts=exclude_normalized_texts,
                    repair_feedback=repair_feedback,
                    repair_candidates_by_slot=repair_candidates_by_slot,
                    repair_reason_codes_by_slot=repair_reason_codes_by_slot,
                    replacement_wave=replacement_wave,
                    guidance_by_slot={
                        slot.slot_id: guidance[slot.slot_id]
                        for slot in chunk
                        if slot.slot_id in guidance
                    },
                    prompt_name=prompt_name,
                )
            )
        return self._merge_pattern_batches(batches)

    def _slot_payload(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        group: GenerationGroup,
        slots: tuple[PlannerSlot, ...],
        exclude_normalized_texts: tuple[str, ...],
        repair_feedback: tuple[str, ...],
        repair_candidates_by_slot: Mapping[str, GeneratedQuestion] | None,
        repair_reason_codes_by_slot: Mapping[str, tuple[str, ...]] | None,
        replacement_wave: int,
        guidance_by_slot: Mapping[str, PatternGenerationContext],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
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
        }
        if replacement_wave == 1:
            candidates = repair_candidates_by_slot or {}
            reasons_by_slot = repair_reason_codes_by_slot or {}
            payload["repair_context"] = [
                {
                    "slot_id": slot.slot_id,
                    "reason_codes": list(reasons_by_slot.get(slot.slot_id, ()))[:4],
                    **(
                        {"candidate": candidates[slot.slot_id].model_dump(mode="json")}
                        if slot.slot_id in candidates
                        else {}
                    ),
                }
                for slot in slots
            ]
        pattern_guidance = _pattern_guidance_payload(guidance_by_slot)
        if pattern_guidance:
            payload["pattern_guidance"] = pattern_guidance
        return payload

    def _measure_slot_budget(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        prompt_name: str,
        payload: dict[str, Any],
    ) -> PromptInputBudget:
        exam_context = {
            "exam_id": payload.get("exam_id"),
            "exam_stage": payload.get("exam_stage"),
        }
        slot_context = {"slots": payload.get("slots", [])}
        pattern_context = {"pattern_guidance": payload.get("pattern_guidance", [])}
        other_dynamic = {
            key: value
            for key, value in payload.items()
            if key not in {"exam_id", "exam_stage", "slots", "pattern_guidance"}
        }
        return self._orchestrator.measure_structured_input(
            route_request=RouteRequest(
                request_id=request.request_id,
                subject=bucket.subject,
                task_role="generator",
                difficulty=bucket.difficulty.value,
                intent="practice",
                language=request.language,
            ),
            user_content=_canonical_json(payload),
            prompt=f"{prompt_name}.md",
            overlays=_PRACTICE_GENERATOR_OVERLAYS,
            exam_context=_canonical_json(exam_context),
            slot_context=_canonical_json(slot_context),
            pattern_context=_canonical_json(pattern_context),
            other_dynamic_input=_canonical_json(other_dynamic),
            max_input_tokens=self._pattern_input_max_tokens,
        )

    def _execute_slot_payload(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        prompt_name: str,
        payload: dict[str, Any],
        budget: PromptInputBudget,
        slots: tuple[PlannerSlot, ...],
    ) -> GeneratedBatch:
        logger.info(
            "practice_pattern_prompt_budget request_id=%s total_input_tokens=%d "
            "pattern_context_tokens=%d reference_tokens=%d within_budget=%s",
            request.request_id,
            budget.total_input_tokens,
            budget.pattern_context_tokens,
            budget.reference_tokens,
            budget.within_budget,
        )
        return _execute(
            self._orchestrator,
            request_id=request.request_id,
            subject=bucket.subject,
            task_role="generator",
            difficulty=bucket.difficulty.value,
            language=request.language,
            prompt_name=prompt_name,
            payload=payload,
            prompt_input_budget=budget,
            practice_generation_workload=PracticeGenerationWorkload(
                complexity=slots[0].complexity.value,
                slot_count=len(slots),
            ),
        )

    def _split_pattern_aware_slots(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        group: GenerationGroup,
        slots: tuple[PlannerSlot, ...],
        exclude_normalized_texts: tuple[str, ...],
        repair_feedback: tuple[str, ...],
        repair_candidates_by_slot: Mapping[str, GeneratedQuestion] | None,
        repair_reason_codes_by_slot: Mapping[str, tuple[str, ...]] | None,
        replacement_wave: int,
        guidance_by_slot: Mapping[str, PatternGenerationContext],
        prompt_name: str,
    ) -> tuple[tuple[PlannerSlot, ...], ...]:
        chunks: list[tuple[PlannerSlot, ...]] = []
        current: list[PlannerSlot] = []
        for slot in slots:
            candidate = tuple((*current, slot))
            candidate_group = group.model_copy(
                update={
                    "required_count": len(candidate),
                    "slot_ids": [item.slot_id for item in candidate],
                }
            )
            candidate_payload = self._slot_payload(
                request=request,
                bucket=bucket,
                group=candidate_group,
                slots=candidate,
                exclude_normalized_texts=exclude_normalized_texts,
                repair_feedback=repair_feedback,
                repair_candidates_by_slot=repair_candidates_by_slot,
                repair_reason_codes_by_slot=repair_reason_codes_by_slot,
                replacement_wave=replacement_wave,
                guidance_by_slot={
                    item.slot_id: guidance_by_slot[item.slot_id]
                    for item in candidate
                    if item.slot_id in guidance_by_slot
                },
            )
            budget = self._measure_slot_budget(
                request=request,
                bucket=bucket,
                prompt_name=prompt_name,
                payload=candidate_payload,
            )
            if current and not budget.within_budget:
                chunks.append(tuple(current))
                current = [slot]
            else:
                current = list(candidate)
        if current:
            chunks.append(tuple(current))
        return tuple(chunks)

    def _execute_bounded_pattern_chunk(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        group: GenerationGroup,
        slots: tuple[PlannerSlot, ...],
        exclude_normalized_texts: tuple[str, ...],
        repair_feedback: tuple[str, ...],
        repair_candidates_by_slot: Mapping[str, GeneratedQuestion] | None,
        repair_reason_codes_by_slot: Mapping[str, tuple[str, ...]] | None,
        replacement_wave: int,
        guidance_by_slot: Mapping[str, PatternGenerationContext],
        prompt_name: str,
    ) -> GeneratedBatch:
        payload = self._slot_payload(
            request=request,
            bucket=bucket,
            group=group,
            slots=slots,
            exclude_normalized_texts=exclude_normalized_texts,
            repair_feedback=repair_feedback,
            repair_candidates_by_slot=repair_candidates_by_slot,
            repair_reason_codes_by_slot=repair_reason_codes_by_slot,
            replacement_wave=replacement_wave,
            guidance_by_slot=guidance_by_slot,
        )
        budget = self._measure_slot_budget(
            request=request,
            bucket=bucket,
            prompt_name=prompt_name,
            payload=payload,
        )
        if not budget.within_budget and guidance_by_slot:
            reduced_guidance = {
                slot_id: context.model_copy(
                    update={
                        "trap_cues": (),
                        "variation_focus": "",
                        "complexity_level": "",
                    }
                )
                for slot_id, context in guidance_by_slot.items()
            }
            payload = self._slot_payload(
                request=request,
                bucket=bucket,
                group=group,
                slots=slots,
                exclude_normalized_texts=exclude_normalized_texts,
                repair_feedback=repair_feedback,
                repair_candidates_by_slot=repair_candidates_by_slot,
                repair_reason_codes_by_slot=repair_reason_codes_by_slot,
                replacement_wave=replacement_wave,
                guidance_by_slot=reduced_guidance,
            )
            budget = self._measure_slot_budget(
                request=request,
                bucket=bucket,
                prompt_name=prompt_name,
                payload=payload,
            )
            if not budget.within_budget:
                logger.warning(
                    "practice_pattern_guidance_rejected request_id=%s group_id=%s "
                    "reason=single_slot_budget_exceeded total_input_tokens=%d "
                    "max_input_tokens=%s",
                    request.request_id,
                    group.group_id,
                    budget.total_input_tokens,
                    budget.max_input_tokens,
                )
                raise ValueError(
                    "PATTERN_GUIDANCE_CORE_PROMPT_BUDGET_EXCEEDED"
                )
        return self._execute_slot_payload(
            request=request,
            bucket=bucket,
            prompt_name=prompt_name,
            payload=payload,
            budget=budget,
            slots=slots,
        )

    def _merge_pattern_batches(self, batches: list[GeneratedBatch]) -> GeneratedBatch:
        if not batches:
            raise ValueError("Pattern-aware generation requires at least one batch.")
        questions: list[object] = []
        valid_batches: list[GeneratedBatch] = []
        for batch in batches:
            try:
                decoded = json.loads(batch.content)
            except json.JSONDecodeError:
                continue
            raw_questions = decoded.get("questions") if isinstance(decoded, dict) else None
            if not isinstance(raw_questions, list):
                continue
            questions.extend(raw_questions)
            valid_batches.append(batch)
        if not valid_batches:
            return batches[0]
        first = valid_batches[0]
        return GeneratedBatch(
            content=_canonical_json({"questions": questions}),
            route_id=first.route_id,
            model=first.model,
            prompt_input_budget=_combine_prompt_budgets(
                [
                    batch.prompt_input_budget
                    for batch in valid_batches
                    if batch.prompt_input_budget
                ]
            ),
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
