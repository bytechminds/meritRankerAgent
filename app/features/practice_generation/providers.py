"""Existing model-routing adapters for structured practice model calls."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from features.practice_generation.execution_control import (
    current_expensive_attempt_guard,
)
from features.practice_generation.planning import select_planner_family
from features.practice_generation.request_intelligence import (
    token_id,
    tokenize_query,
)
from features.practice_generation.schemas import (
    DemandBucket,
    GeneratedBatch,
    GeneratedQuestion,
    GenerationGroup,
    PlannerSlot,
    PracticeGenerationRequest,
    VerificationDecision,
    VerificationResult,
)
from retrieval.pattern_intelligence import PatternGenerationContext
from schemas.llm_routing import PracticeGenerationWorkload, RouteRequest
from services.doubt_solver.exam_profile_cache import get_exam_profile_runtime
from services.llm.orchestration.orchestrator import LlmOrchestrator
from services.llm.orchestration.prompt_budget import PromptInputBudget
from services.llm.orchestration.prompt_resolver import DEFAULT_PROMPT_ROOT
from services.llm.orchestration.route_resolver import normalize_subject, resolve_route

logger = logging.getLogger(__name__)

_PRACTICE_OVERLAYS = ["practice_generation/shared_contract.md"]
_PRACTICE_GENERATOR_OVERLAYS = [
    *_PRACTICE_OVERLAYS,
    "practice_generation/generator_output_policy.md",
]
# Factual authoring replaces the generic role prompt and output-policy overlay rather
# than stacking on top of them, so the composed prompt stays inside the frozen budget.
# The family comes from the central route resolver, so no second taxonomy exists here
# and a newly qualified family is a routing-config change, not a code change.
_FACTUAL_GENERATOR_PROMPT = "practice_generation/question_generator_factual"


def _factual_subject_profile(subject: str) -> str | None:
    """Return the subject's authoring profile when it belongs to the factual family."""
    if normalize_subject(subject) != "factual":
        return None
    overlay = f"practice_generation/subjects/{subject}.md"
    if (DEFAULT_PROMPT_ROOT / overlay).is_file():
        return overlay
    return None
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
    subject_profile: str | None = None,
) -> GeneratedBatch:
    if subject_profile is not None:
        # Replacement, not addition: the subject profile takes the generic output
        # policy's place so the composed prompt stays inside the frozen budget.
        overlays = [*_PRACTICE_OVERLAYS, subject_profile]
    else:
        overlays = (
            _PRACTICE_GENERATOR_OVERLAYS
            if task_role == "generator"
            else _PRACTICE_OVERLAYS
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
        attempt_guard=current_expensive_attempt_guard(),
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


def _freshness_metadata_payload(request: PracticeGenerationRequest) -> dict[str, object] | None:
    if not request.requires_fresh_evidence or request.fresh_evidence is None:
        return None
    return {
        "required": True,
        "reason": request.freshness_reason,
        "requested_window": request.fresh_evidence.requested_window.model_dump(mode="json"),
        "evidence_item_count": len(request.fresh_evidence.items),
    }


def _fresh_evidence_payload(
    request: PracticeGenerationRequest,
    *,
    slot_ids: tuple[str, ...] = (),
) -> dict[str, object] | None:
    if not request.requires_fresh_evidence or request.fresh_evidence is None:
        return None
    items = request.fresh_evidence.items
    evidence_by_slot: list[dict[str, object]] = []
    if slot_ids:
        selected = []
        for slot_id in slot_ids:
            try:
                index = int(slot_id.rsplit("-", 1)[-1]) - 1
            except ValueError:
                continue
            if 0 <= index < len(items):
                evidence_item = items[index]
                selected.append(evidence_item)
                evidence_by_slot.append(
                    {
                        "slot_id": slot_id,
                        "items": [evidence_item.model_dump(mode="json")],
                    }
                )
        items = selected
    payload: dict[str, object] = {
        "requested_window": request.fresh_evidence.requested_window.model_dump(mode="json"),
        "items": [item.model_dump(mode="json") for item in items],
    }
    if evidence_by_slot:
        payload["evidence_by_slot"] = evidence_by_slot
    return payload


def _enforce_evidence_citations(
    request: PracticeGenerationRequest,
    result: VerificationResult,
    *,
    evidence: dict[str, object] | None,
) -> VerificationResult:
    if not request.requires_fresh_evidence or not result.is_approved:
        return result
    raw_items = evidence.get("items") if isinstance(evidence, dict) else None
    allowed_urls = {
        str(item.get("url") or "").strip()
        for item in raw_items
        if isinstance(item, dict) and str(item.get("url") or "").strip()
    }
    cited_urls = {url.strip() for url in result.evidence_urls if url.strip()}
    if cited_urls and cited_urls.issubset(allowed_urls):
        return result
    if result.schema_version == "2":
        return result.model_copy(
            update={
                "decision": VerificationDecision.REGENERATE,
                "reason_codes": ["UNSUPPORTED_FACT"],
                "evidence_urls": [],
            }
        )
    return result.model_copy(
        update={
            "approved": False,
            "reason_code": "UNSUPPORTED_FACT",
            "evidence_urls": [],
        }
    )


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


class RoutedRequestIntelligenceProvider:
    """Interpret one free-text Practice request through the shared LLM runtime.

    The provider only transports; it applies no business rule. Shape is guaranteed
    by the route's native structured-output schema, and every semantic invariant is
    checked deterministically by ``features.practice_generation.request_intelligence``.
    """

    def __init__(self, orchestrator: LlmOrchestrator) -> None:
        self._orchestrator = orchestrator

    def interpret(
        self,
        *,
        request_id: str,
        query: str,
        subject: str,
        language: str,
        exam_id: str | None,
        exam_stage: str | None,
    ) -> str:
        payload = {
            # The untranslated query is the semantic source of truth; the rest are
            # hints the classifier already paid for. query_tokens carries the same
            # query already split deterministically, each token labelled with the id
            # a topic selects it by — so the model picks labels it can see rather
            # than retyping the student's characters or counting positions.
            "query": query,
            "query_tokens": [
                {"id": token_id(token.index), "text": token.text}
                for token in tokenize_query(query)
            ],
            "classified_subject": subject,
            "detected_language": language,
            "exam_id": exam_id,
            "exam_stage": exam_stage,
        }
        return self._orchestrator.generate_structured(
            route_request=RouteRequest(
                request_id=request_id,
                subject="general",
                task_role="request_intelligence",
                difficulty="default",
                intent="practice",
                language=language,
            ),
            user_content=json.dumps(payload, separators=(",", ":"), sort_keys=True),
            prompt="practice_generation/request_intelligence.md",
            overlays=[],
        ).content


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
        freshness = _freshness_metadata_payload(request)
        if freshness is not None:
            payload["freshness_requirement"] = freshness
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
        payload: dict[str, Any] = {
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
        }
        evidence = _fresh_evidence_payload(request)
        if evidence is not None:
            payload["fresh_evidence"] = evidence
            logger.info(
                "practice_fresh_evidence_generator request_id=%s evidence_count=%d",
                request.request_id,
                len(evidence["items"]),
            )
        return _execute(
            self._orchestrator,
            request_id=request.request_id,
            subject=bucket.subject,
            task_role="generator",
            difficulty=bucket.difficulty.value,
            language=request.language,
            prompt_name="practice_generation/question_generator",
            payload=payload,
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
        subject_profile = _factual_subject_profile(bucket.subject)
        if replacement_wave == 0:
            prompt_name = (
                _FACTUAL_GENERATOR_PROMPT
                if subject_profile is not None
                else "practice_generation/question_generator_v2"
            )
        elif replacement_wave == 1:
            prompt_name = "practice_generation/question_repair"
        else:
            prompt_name = "practice_generation/question_regenerator"
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
        evidence = _fresh_evidence_payload(
            request,
            slot_ids=tuple(slot.slot_id for slot in slots),
        )
        if evidence is not None:
            payload["fresh_evidence"] = evidence
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
        # The subject profile travels with the factual role prompt that expects it, so
        # repair and regeneration waves keep their own prompts and overlays unchanged.
        subject_profile = (
            _factual_subject_profile(bucket.subject)
            if prompt_name == _FACTUAL_GENERATOR_PROMPT
            else None
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
            subject_profile=subject_profile,
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


class PracticeAuthorityUnavailableError(RuntimeError):
    """No qualified Answer Authority is routed for this Practice subject family.

    Practice must never verify a student-facing question with an Authority that has
    not passed the role's qualification gate. Falling through to the shared
    ``general.verifier.default`` entry would do exactly that — it exists for other
    callers and carries no Practice qualification — so Practice fails closed instead.
    """


def _require_qualified_authority(*, request_id: str, subject: str, language: str) -> str:
    """Return the qualified Authority route id, or fail closed.

    The check is on routing metadata only: an exact subject match means a family
    Authority was deliberately configured, while any fallback means none was. No
    subject or model name is tested here, so adding a newly qualified family is a
    configuration change and never a code change.
    """
    decision = resolve_route(
        RouteRequest(
            request_id=request_id,
            subject=subject,
            task_role="verifier",
            difficulty="default",
            intent="practice",
            language=language,
        )
    )
    # ``general`` is the shared cross-feature verifier entry, not a Practice family
    # Authority, and normalize_subject() collapses every unknown or malformed subject
    # onto it. Requiring a non-general exact match therefore makes unknown input fail
    # closed for the same reason an unqualified family does, without guessing.
    if decision.route_source != "exact" or decision.subject == "general":
        logger.warning(
            "practice_authority_unavailable request_id=%s subject=%s language=%s "
            "resolved_route=%s route_source=%s",
            request_id,
            subject,
            language,
            decision.route_id,
            decision.route_source,
        )
        raise PracticeAuthorityUnavailableError(
            f"PRACTICE_AUTHORITY_NOT_QUALIFIED subject={subject}"
        )
    return decision.route_id


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
        evidence = _fresh_evidence_payload(request)
        payload: dict[str, Any] = {
            "bucket": bucket.model_dump(mode="json"),
            "question": question.model_dump(mode="json"),
            "language": request.language,
        }
        if evidence is not None:
            payload["fresh_evidence"] = evidence
            logger.info(
                "practice_fresh_evidence_verifier request_id=%s evidence_count=%d",
                request.request_id,
                len(evidence["items"]),
            )
        raw = _execute(
            self._orchestrator,
            request_id=request.request_id,
            subject="general",
            task_role="verifier",
            difficulty="default",
            language=request.language,
            prompt_name="practice_generation/question_verifier",
            payload=payload,
        )
        return _enforce_evidence_citations(
            request,
            VerificationResult.model_validate(json.loads(raw.content)),
            evidence=evidence,
        )

    def verify_slot(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        slot: PlannerSlot,
        question: GeneratedQuestion,
    ) -> VerificationResult:
        evidence = _fresh_evidence_payload(request, slot_ids=(slot.slot_id,))
        payload: dict[str, Any] = {
            "schema_version": "2",
            "slot": slot.model_dump(mode="json"),
            # Deliberately blind: the author's proposed option id, its free-text
            # answer, its explanation and its solution are all withheld. The authority
            # cannot confirm a key it has never seen, so its verdict is evidence
            # independent of the author rather than agreement with it.
            "question": {
                "schema_version": question.schema_version,
                "generation_item_id": question.generation_item_id,
                "slot_id": question.slot_id,
                "question": question.question,
                "question_type": question.question_type.value,
                "options": [
                    option.model_dump(mode="json") for option in question.canonical_options
                ],
            },
            "language": request.language,
            "instruction": (
                "Solve independently, then evaluate every supplied option and return "
                "the id of every option that satisfies the question."
            ),
        }
        if evidence is not None:
            payload["fresh_evidence"] = evidence
            logger.info(
                "practice_fresh_evidence_verifier request_id=%s evidence_count=%d",
                request.request_id,
                len(evidence["items"]),
            )
        _require_qualified_authority(
            request_id=request.request_id,
            subject=bucket.subject,
            language=request.language,
        )
        raw = _execute(
            self._orchestrator,
            request_id=request.request_id,
            # The bucket's canonical subject, not a literal, so the central resolver can
            # place a subject-qualified Authority. A single literal previously sent every
            # Practice subject and language to one model, which made evidence-based
            # selection impossible. Nothing is re-classified here: this value is the
            # planner's own normalized subject.
            subject=bucket.subject,
            task_role="verifier",
            difficulty="default",
            language=request.language,
            prompt_name="practice_generation/question_verifier_v2",
            payload=payload,
        )
        return _enforce_evidence_citations(
            request,
            VerificationResult.model_validate(json.loads(raw.content)),
            evidence=evidence,
        )
