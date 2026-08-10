"""Bounded generation groups, partial parsing, verification policy, and final gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from features.practice_generation.matching import normalize_question_text
from features.practice_generation.metadata_normalization import (
    normalize_subject,
    normalize_topic,
)
from features.practice_generation.question_contract import (
    validate_persisted_playable_question,
    validate_playable_question,
)
from features.practice_generation.schemas import (
    DemandBucket,
    GeneratedBatch,
    GeneratedQuestion,
    GenerationGroup,
    PlannerSlot,
    PracticeBlueprint,
    PracticeGenerationRequest,
    VerificationPolicy,
    VerificationResult,
)
from schemas.llm_routing import RouteRequest
from services.llm.orchestration.errors import (
    LlmRouteNotFoundError,
    LlmRouteResolutionError,
)
from services.llm.orchestration.route_resolver import resolve_route

TokenBudgetResolver = Callable[[str, str], int]


def _structured_parse_rejection_code(error: ValidationError | TypeError) -> str:
    """Map schema failures to safe, stable diagnostics without exposing model output."""
    if isinstance(error, TypeError):
        return "STRUCTURED_PARSE_INVALID"

    locations = {
        str(location[0])
        for item in error.errors(include_url=False)
        if (location := item.get("loc"))
    }
    if locations & {"generation_item_id", "bucket_id", "schema_version"}:
        return "SCHEMA_V2_IDENTITY_CONTRACT_INVALID"
    if locations & {"slot_id", "subject", "topic", "difficulty", "question_type"}:
        return "SCHEMA_V2_SLOT_CONTRACT_INVALID"
    if locations & {"options", "canonical_options"}:
        return "SCHEMA_V2_OPTION_CONTRACT_INVALID"
    if locations & {"correct_option_id", "correct_answer", "answer_explanation", "solution"}:
        return "SCHEMA_V2_ANSWER_CONTRACT_INVALID"
    return "STRUCTURED_PARSE_INVALID"


def _route_output_token_budget(subject: str, difficulty: str) -> int:
    """Read the configured shared generator-route budget without a model call."""
    try:
        return resolve_route(
            RouteRequest(
                request_id="practice-generation-batch-capacity",
                subject=subject,
                task_role="generator",
                difficulty=difficulty,
                intent="practice",
                language="english",
            )
        ).max_tokens
    except (LlmRouteNotFoundError, LlmRouteResolutionError):
        # A route-resolution failure must make batching safer, never larger.
        return 0


def _expected_question_output_tokens(
    *,
    subject: str,
    complexity: str,
) -> int:
    """Conservative structured-output estimate for one full playable MCQ."""
    base = {
        "math": 300,
        "reasoning": 310,
        "english": 280,
    }.get(subject, 260)
    complexity_adjustment = {"low": 0, "medium": 50, "high": 160}[complexity]
    return base + complexity_adjustment


def _model_safe_batch_capacity(
    *,
    subject: str,
    difficulty: str,
    complexity: str,
    token_budget_resolver: TokenBudgetResolver,
) -> tuple[int, int]:
    """Return a route-budgeted batch capacity and the source output budget."""
    output_budget = max(token_budget_resolver(subject, difficulty), 0)
    if output_budget <= 0:
        return 1, output_budget
    # Preserve enough output for the JSON envelope and stop sequence.
    usable_budget = max(output_budget - 120, 1)
    expected_per_question = _expected_question_output_tokens(
        subject=subject,
        complexity=complexity,
    )
    return max(1, min(5, usable_budget // expected_per_question)), output_budget


def _slot_batch_limit(
    slot: PlannerSlot,
    *,
    effective_size: int,
    token_budget_resolver: TokenBudgetResolver,
) -> tuple[int, int]:
    difficulty_cap = (
        {"basic": 5, "intermediate": 4, "advanced": 2}[slot.difficulty.value]
        if slot.subject_id in {"math", "reasoning"}
        else 5
    )
    complexity_cap = {"low": 5, "medium": 4, "high": 2}[slot.complexity.value]
    model_capacity, output_budget = _model_safe_batch_capacity(
        subject=slot.subject_id,
        difficulty=slot.difficulty.value,
        complexity=slot.complexity.value,
        token_budget_resolver=token_budget_resolver,
    )
    cap = min(difficulty_cap, complexity_cap, model_capacity)
    if (
        slot.subject_id == "reasoning"
        and slot.difficulty.value == "advanced"
        and slot.complexity.value == "high"
    ):
        cap = 1
    if slot.generation_group_hint is not None:
        cap = min(cap, slot.generation_group_hint)
    return min(effective_size, cap), output_budget


class QuestionGenerator(Protocol):
    def generate(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        group: GenerationGroup,
        exclude_normalized_texts: tuple[str, ...],
    ) -> GeneratedBatch: ...


class QuestionVerifier(Protocol):
    def verify(
        self,
        *,
        request: PracticeGenerationRequest,
        bucket: DemandBucket,
        question: GeneratedQuestion,
    ) -> VerificationResult: ...


def bucket_for_slot(
    blueprint: PracticeBlueprint,
    slot: PlannerSlot,
) -> DemandBucket:
    for bucket in blueprint.buckets:
        if (
            bucket.subject == slot.subject_id
            and bucket.topic == slot.topic_id
            and bucket.difficulty is slot.difficulty
            and bucket.question_type is slot.question_type
        ):
            return bucket
    raise ValueError("PLANNER_SLOT_BUCKET_MISSING")


def build_slot_generation_groups(
    blueprint: PracticeBlueprint,
    deficit_slot_ids: set[str],
    *,
    group_size: int,
    group_max: int,
    token_budget_resolver: TokenBudgetResolver = _route_output_token_budget,
) -> tuple[GenerationGroup, ...]:
    """Build deterministic homogeneous batches from exact unfilled slots."""
    effective_size = min(max(group_size, 1), min(max(group_max, 1), 5))
    homogeneous: dict[tuple[str, ...], list[PlannerSlot]] = {}
    batch_limits: dict[tuple[str, ...], tuple[int, int]] = {}
    for slot in blueprint.slots:
        if slot.slot_id not in deficit_slot_ids:
            continue
        batch_size, output_budget = _slot_batch_limit(
            slot,
            effective_size=effective_size,
            token_budget_resolver=token_budget_resolver,
        )
        key = (
            slot.subject_id,
            slot.topic_id,
            slot.category_id,
            slot.difficulty.value,
            slot.complexity.value,
            slot.question_type.value,
            slot.generator_route_hint,
            str(batch_size),
        )
        homogeneous.setdefault(key, []).append(slot)
        batch_limits[key] = (batch_size, output_budget)
    groups: list[GenerationGroup] = []
    sequence = 1
    for key, slots in homogeneous.items():
        batch_size, output_budget = batch_limits[key]
        bucket = bucket_for_slot(blueprint, slots[0])
        for offset in range(0, len(slots), batch_size):
            batch = slots[offset : offset + batch_size]
            groups.append(
                GenerationGroup(
                    group_id=f"slot-group-{sequence:03d}",
                    bucket_id=bucket.bucket_id,
                    required_count=len(batch),
                    slot_ids=[slot.slot_id for slot in batch],
                    token_budget=output_budget or None,
                )
            )
            sequence += 1
    return tuple(groups)


@dataclass(frozen=True)
class ParsedGeneration:
    accepted: tuple[GeneratedQuestion, ...]
    rejected_count: int
    rejection_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinalValidation:
    ready: bool
    reason_code: str


def build_generation_groups(
    blueprint: PracticeBlueprint,
    deficits: dict[str, int],
    *,
    group_size: int,
    group_max: int,
) -> tuple[GenerationGroup, ...]:
    effective_size = min(max(group_size, 1), min(max(group_max, 1), 5))
    groups: list[GenerationGroup] = []
    for bucket in blueprint.buckets:
        difficulty_cap = {
            "basic": 5,
            "intermediate": 4,
            "advanced": 2,
        }[bucket.difficulty.value]
        if bucket.generation_group_hint is not None:
            difficulty_cap = min(difficulty_cap, bucket.generation_group_hint)
        bucket_group_size = min(effective_size, difficulty_cap)
        remaining = max(deficits.get(bucket.bucket_id, 0), 0)
        sequence = 1
        while remaining:
            count = min(remaining, bucket_group_size)
            groups.append(
                GenerationGroup(
                    group_id=f"{bucket.bucket_id}-g{sequence}",
                    bucket_id=bucket.bucket_id,
                    required_count=count,
                )
            )
            remaining -= count
            sequence += 1
    return tuple(groups)


def parse_partial_generation(
    raw: str,
    *,
    group: GenerationGroup,
    bucket: DemandBucket,
    existing_normalized_texts: set[str],
    slots: tuple[PlannerSlot, ...] = (),
) -> ParsedGeneration:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ParsedGeneration(
            accepted=(),
            rejected_count=group.required_count,
            rejection_reason_codes=("STRUCTURED_PARSE_INVALID",),
        )
    raw_questions = payload.get("questions", []) if isinstance(payload, dict) else []
    if not isinstance(raw_questions, list):
        return ParsedGeneration(
            accepted=(),
            rejected_count=group.required_count,
            rejection_reason_codes=("STRUCTURED_PARSE_INVALID",),
        )

    accepted: list[GeneratedQuestion] = []
    seen_ids: set[str] = set()
    rejected = 0
    rejection_reason_codes: list[str] = []
    slots_by_id = {slot.slot_id: slot for slot in slots}
    for raw_question in raw_questions[: group.required_count]:
        try:
            question = GeneratedQuestion.model_validate(raw_question)
        except (ValidationError, TypeError) as error:
            rejected += 1
            rejection_reason_codes.append(_structured_parse_rejection_code(error))
            continue
        contract = validate_playable_question(
            question_type=question.question_type,
            question=question.question,
            options=question.options,
            correct_answer=question.correct_answer,
            solution=question.solution,
            solution_required=bucket.solution_required,
        )
        if not contract.valid:
            rejected += 1
            rejection_reason_codes.append(contract.reason_code)
            continue
        normalized = normalize_question_text(question.question)
        slot = slots_by_id.get(str(question.slot_id or "")) if slots else None
        canonical_subject = normalize_subject(question.subject)
        canonical_topic = normalize_topic(question.topic)
        if (
            question.bucket_id != bucket.bucket_id
            or question.generation_item_id in seen_ids
            or normalized in existing_normalized_texts
            or canonical_subject != bucket.subject
            or canonical_topic != bucket.topic
            or question.difficulty is not bucket.difficulty
            or question.question_type is not bucket.question_type
            or bucket.solution_required
            and not question.solution
            or slots
            and (
                question.schema_version != "2"
                or slot is None
                or question.slot_id in seen_ids
                or canonical_subject != slot.subject_id
                or canonical_topic != slot.topic_id
                or question.difficulty is not slot.difficulty
                or question.question_type is not slot.question_type
            )
        ):
            rejected += 1
            rejection_reason_codes.append("GENERATION_BUCKET_CONTRACT_MISMATCH")
            continue
        accepted.append(question)
        seen_ids.add(question.generation_item_id)
        if question.slot_id:
            seen_ids.add(question.slot_id)
        existing_normalized_texts.add(normalized)
    rejected += max(0, group.required_count - len(accepted) - rejected)
    return ParsedGeneration(
        accepted=tuple(accepted),
        rejected_count=rejected,
        rejection_reason_codes=tuple(rejection_reason_codes),
    )


def verification_required(
    bucket: DemandBucket,
    index_in_group: int,
    *,
    replacement: bool = False,
) -> bool:
    del bucket, index_in_group, replacement
    return True


def deterministic_question_id(
    test_id: str,
    *,
    source_id: str,
    bucket_id: str,
) -> str:
    digest = hashlib.sha256(f"{test_id}|{source_id}|{bucket_id}".encode()).hexdigest()
    return f"pq-{digest[:32]}"


def validate_final_set(
    *,
    blueprint: PracticeBlueprint,
    linked_questions: list[dict[str, object]],
    requested_language: str = "english",
) -> FinalValidation:
    if len(linked_questions) != blueprint.accepted_count:
        return FinalValidation(False, "FINAL_COUNT_MISMATCH")
    ids = [str(item.get("questionId") or "") for item in linked_questions]
    if not all(ids) or len(ids) != len(set(ids)):
        return FinalValidation(False, "DUPLICATE_OR_MISSING_QUESTION_ID")
    normalized_texts = [
        normalize_question_text(str(item.get("question") or "")) for item in linked_questions
    ]
    if not all(normalized_texts) or len(normalized_texts) != len(set(normalized_texts)):
        return FinalValidation(False, "DUPLICATE_OR_EMPTY_QUESTION_TEXT")
    bucket_ids = {bucket.bucket_id for bucket in blueprint.buckets}
    buckets_by_id = {bucket.bucket_id: bucket for bucket in blueprint.buckets}
    counts = {bucket_id: 0 for bucket_id in bucket_ids}
    source_question_bank_ids: list[str] = []
    for item in linked_questions:
        meta = item.get("_practiceMeta")
        if not isinstance(meta, dict):
            return FinalValidation(False, "QUESTION_META_INVALID")
        bucket_id = str(meta.get("bucketId") or "")
        if bucket_id not in counts or meta.get("verified") is not True:
            return FinalValidation(False, "QUESTION_NOT_VERIFIED_OR_UNKNOWN_BUCKET")
        bucket = buckets_by_id[bucket_id]
        contract = validate_persisted_playable_question(
            item,
            expected_question_type=bucket.question_type.value,
            expected_language=requested_language,
            solution_required=bucket.solution_required,
        )
        if not contract.valid:
            return FinalValidation(False, contract.reason_code)
        source_type = str(meta.get("sourceType") or "")
        if source_type == "QUESTION_BANK":
            source_id = str(meta.get("sourceQuestionBankId") or "")
            if not source_id:
                return FinalValidation(False, "QUESTION_BANK_PROVENANCE_MISSING")
            source_question_bank_ids.append(source_id)
        elif source_type == "AI_GENERATED":
            required_method = (
                "INDEPENDENT_MODEL_V2" if blueprint.schema_version == "2" else "MODEL"
            )
            if (
                bucket.verification_policy is VerificationPolicy.MANDATORY
                and meta.get("verificationMethod") != required_method
            ):
                return FinalValidation(False, "MANDATORY_VERIFICATION_MISSING")
        else:
            return FinalValidation(False, "QUESTION_PROVENANCE_INVALID")
        counts[bucket_id] += 1
    if len(source_question_bank_ids) != len(set(source_question_bank_ids)):
        return FinalValidation(False, "DUPLICATE_QUESTION_BANK_SOURCE")
    expected = {bucket.bucket_id: bucket.required_count for bucket in blueprint.buckets}
    if counts != expected:
        return FinalValidation(False, "BLUEPRINT_DISTRIBUTION_MISMATCH")
    if blueprint.schema_version == "2":
        expected_slots = {slot.slot_id for slot in blueprint.slots}
        actual_slots = {
            str(item.get("_practiceMeta", {}).get("slotId") or "")
            for item in linked_questions
        }
        if actual_slots != expected_slots:
            return FinalValidation(False, "PLANNER_SLOT_MANIFEST_MISMATCH")
        slots_by_id = {slot.slot_id: slot for slot in blueprint.slots}
        for item in linked_questions:
            meta = item["_practiceMeta"]
            slot = slots_by_id[str(meta["slotId"])]
            if (
                str(item.get("topic") or "").casefold() != slot.topic_id
                or str(item.get("difficulty") or "").casefold() != slot.difficulty.value
                or str(meta.get("questionType") or "").casefold() != slot.question_type.value
                or meta.get("verified") is not True
            ):
                return FinalValidation(False, "PLANNER_SLOT_CONTRACT_MISMATCH")
    return FinalValidation(True, "READY")
