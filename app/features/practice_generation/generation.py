"""Bounded generation groups, partial parsing, verification policy, and final gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import ValidationError

from features.practice_generation.matching import (
    normalize_question_identity,
)
from features.practice_generation.metadata_normalization import (
    normalize_difficulty,
    normalize_subject,
    normalize_topic,
)
from features.practice_generation.planning import (
    _structural_error_bucket,
    practice_generator_route_subject,
)
from features.practice_generation.question_contract import (
    SemanticBasis,
    classify_semantic_basis,
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
from schemas.llm_routing import PracticeGenerationWorkload, RouteRequest
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.errors import (
    LlmRouteNotFoundError,
    LlmRouteResolutionError,
)
from services.llm.orchestration.practice_generation_capacity import (
    PracticeGenerationCapacityPolicy,
)
from services.llm.orchestration.route_resolver import resolve_route
from services.llm.providers.finish_reasons import normalize_completion_outcome


@dataclass(frozen=True)
class RouteOutputCapacity:
    """Configured completion budget and measured reasoning reserve for one route."""

    configured_output_tokens: int
    expected_reasoning_tokens: int = 0


TokenBudgetResolver = Callable[[str, str], int | RouteOutputCapacity]


# Bucket → generator-prefixed reason code, mirroring the planner's identical mapping
# (features.practice_generation.planning._PLANNER_STRUCTURAL_REASON_CODES) with its
# own prefix. Consulted only after the more specific SCHEMA_V2_*_CONTRACT_INVALID
# location-based codes below have had a chance to match, so a field-specific code is
# never coarsened just because this set also covers it.
_GENERATOR_STRUCTURAL_REASON_CODES = {
    "parse_failure": "GENERATOR_PARSE_FAILURE",
    "missing": "GENERATOR_MISSING_FIELD",
    "constraint": "GENERATOR_SCHEMA_CONSTRAINT_INVALID",
    "type": "GENERATOR_SCHEMA_TYPE_INVALID",
}

# Every technical/structural rejection code parse_partial_generation can emit — the
# location-based SCHEMA_V2_*_CONTRACT_INVALID codes plus every value
# _GENERATOR_STRUCTURAL_REASON_CODES can produce, plus the envelope/truncation/
# fallback codes emitted directly. A semantic rejection (e.g. from
# validate_playable_question, or GENERATION_BUCKET_CONTRACT_MISMATCH) is never a
# member: the caller uses this set to tell "the output was structurally wrong" apart
# from "the output was well-formed but rejected", which decide different terminal
# reason codes.
STRUCTURAL_GENERATOR_REASON_CODES = frozenset(
    {
        "SCHEMA_V2_IDENTITY_CONTRACT_INVALID",
        "SCHEMA_V2_SLOT_CONTRACT_INVALID",
        "SCHEMA_V2_OPTION_CONTRACT_INVALID",
        "SCHEMA_V2_ANSWER_CONTRACT_INVALID",
        "GENERATOR_CONTRACT_INVALID",
        "GENERATOR_OUTPUT_TRUNCATED",
        "GENERATOR_BATCH_COUNT_MISMATCH",
        "GENERATOR_OUTPUT_INVALID",
        *_GENERATOR_STRUCTURAL_REASON_CODES.values(),
    }
)


def _structured_parse_rejection_code(error: ValidationError | TypeError) -> str:
    """Map schema failures to safe, stable diagnostics without exposing model output."""
    if isinstance(error, TypeError):
        return "GENERATOR_OUTPUT_INVALID"

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
    bucket = _structural_error_bucket(error)
    return _GENERATOR_STRUCTURAL_REASON_CODES.get(bucket, "GENERATOR_OUTPUT_INVALID")


def _route_output_token_budget(subject: str, difficulty: str) -> RouteOutputCapacity:
    """Read the configured generator route capacity without a model call."""
    try:
        route = resolve_route(
            RouteRequest(
                request_id="practice-generation-batch-capacity",
                subject=practice_generator_route_subject(subject, difficulty),
                task_role="generator",
                difficulty=difficulty,
                intent="practice",
                language="english",
            )
        )
        model_config = LlmConfigRegistry().model_map.get(route.model)
        return RouteOutputCapacity(
            configured_output_tokens=route.max_tokens,
            expected_reasoning_tokens=(
                model_config.structured_output_reasoning_reserve_tokens
                if model_config is not None
                else 0
            ),
        )
    except (LlmRouteNotFoundError, LlmRouteResolutionError):
        # A route-resolution failure must make batching safer, never larger.
        return RouteOutputCapacity(configured_output_tokens=0)


def _practice_capacity_for_slot(
    slot: PlannerSlot,
    *,
    slot_count: int,
):
    """Resolve central Practice capacity from an exact route and model alias."""
    try:
        route = resolve_route(
            RouteRequest(
                request_id="practice-generation-capacity",
                subject=practice_generator_route_subject(
                    slot.subject_id,
                    slot.difficulty,
                ),
                task_role="generator",
                difficulty=slot.difficulty.value,
                intent="practice",
                language="english",
            )
        )
        model_config = LlmConfigRegistry().model_map.get(route.model)
        if model_config is None:
            return None
        return PracticeGenerationCapacityPolicy.resolve(
            route_decision=route,
            model_config=model_config,
            workload=PracticeGenerationWorkload(
                complexity=slot.complexity.value,
                slot_count=slot_count,
            ),
        )
    except (LlmRouteNotFoundError, LlmRouteResolutionError, ValueError):
        return None


def _practice_capacity_for_bucket(
    bucket: DemandBucket,
    *,
    slot_count: int,
):
    """Resolve the same central policy for schema-v1 compatibility batches."""
    try:
        route = resolve_route(
            RouteRequest(
                request_id="practice-generation-capacity",
                subject=practice_generator_route_subject(
                    bucket.subject,
                    bucket.difficulty,
                ),
                task_role="generator",
                difficulty=bucket.difficulty.value,
                intent="practice",
                language="english",
            )
        )
        model_config = LlmConfigRegistry().model_map.get(route.model)
        if model_config is None:
            return None
        return PracticeGenerationCapacityPolicy.resolve(
            route_decision=route,
            model_config=model_config,
            workload=PracticeGenerationWorkload(
                complexity="medium",
                slot_count=slot_count,
            ),
        )
    except (LlmRouteNotFoundError, LlmRouteResolutionError, ValueError):
        return None


def _normalized_route_output_capacity(
    capacity: int | RouteOutputCapacity,
) -> RouteOutputCapacity:
    """Retain the legacy integer test seam while applying route-local reserves."""
    if isinstance(capacity, RouteOutputCapacity):
        return capacity
    return RouteOutputCapacity(configured_output_tokens=capacity)


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
    route_capacity = _normalized_route_output_capacity(
        token_budget_resolver(subject, difficulty)
    )
    output_budget = max(route_capacity.configured_output_tokens, 0)
    if output_budget <= 0:
        return 1, output_budget
    # Preserve the measured reasoning reserve, then JSON envelope and stop sequence.
    usable_budget = max(
        output_budget - max(route_capacity.expected_reasoning_tokens, 0) - 120,
        1,
    )
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
    if token_budget_resolver is _route_output_token_budget:
        capacity = _practice_capacity_for_slot(slot, slot_count=1)
        if capacity is not None:
            cap = min(effective_size, capacity.max_slots_per_batch)
            if slot.generation_group_hint is not None:
                cap = min(cap, slot.generation_group_hint)
            return cap, capacity.initial_max_output_tokens

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
    signatures: dict[tuple[str, ...], int] = {}
    for candidate in blueprint.slots:
        signature = (
            candidate.subject_id,
            candidate.topic_id,
            candidate.category_id,
            candidate.difficulty.value,
            candidate.question_type.value,
            candidate.generator_route_hint,
        )
        signatures.setdefault(signature, len(signatures))
        if candidate.slot_id != slot.slot_id:
            continue
        bucket_index = signatures[signature]
        if bucket_index >= len(blueprint.buckets):
            break
        bucket = blueprint.buckets[bucket_index]
        expected_count = sum(
            1
            for planned in blueprint.slots
            if (
                planned.subject_id,
                planned.topic_id,
                planned.category_id,
                planned.difficulty.value,
                planned.question_type.value,
                planned.generator_route_hint,
            )
            == signature
        )
        if bucket.required_count == expected_count:
            return bucket
        break
    raise ValueError("PLANNER_SLOT_BUCKET_MISSING")


def build_slot_generation_groups(
    blueprint: PracticeBlueprint,
    deficit_slot_ids: set[str],
    *,
    group_size: int,
    group_max: int,
    token_budget_resolver: TokenBudgetResolver = _route_output_token_budget,
    pattern_context_keys: dict[str, str] | None = None,
) -> tuple[GenerationGroup, ...]:
    """Build deterministic homogeneous batches from exact unfilled slots.

    A selected Pattern identity is an additional batch boundary.  This keeps one
    compact PatternGraph projection from being repeated for unrelated slots and
    lets the generator's input budget split otherwise-compatible work safely.
    """
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
            ",".join(
                sorted({exam_id.strip() for exam_id in slot.exam_ids if exam_id.strip()})
            ),
            slot.target_skill,
            slot.reasoning_target or "",
            slot.pattern_family_id or "",
            slot.generator_route_hint,
            (pattern_context_keys or {}).get(slot.slot_id, ""),
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
            capacity = (
                _practice_capacity_for_slot(batch[0], slot_count=len(batch))
                if token_budget_resolver is _route_output_token_budget
                else None
            )
            groups.append(
                GenerationGroup(
                    group_id=f"slot-group-{sequence:03d}",
                    bucket_id=bucket.bucket_id,
                    required_count=len(batch),
                    slot_ids=[slot.slot_id for slot in batch],
                    token_budget=(
                        capacity.initial_max_output_tokens
                        if capacity is not None
                        else output_budget or None
                    ),
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
    failed_slot_ids: tuple[str, ...] = ()
    expected_question_count: int = 0
    actual_question_count: int = 0
    recoverable: bool = False


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
        capacity = _practice_capacity_for_bucket(bucket, slot_count=1)
        difficulty_cap = (
            capacity.max_slots_per_batch
            if capacity is not None
            else {"basic": 5, "intermediate": 4, "advanced": 2}[bucket.difficulty.value]
        )
        if bucket.generation_group_hint is not None:
            difficulty_cap = min(difficulty_cap, bucket.generation_group_hint)
        bucket_group_size = min(effective_size, difficulty_cap)
        remaining = max(deficits.get(bucket.bucket_id, 0), 0)
        sequence = 1
        while remaining:
            count = min(remaining, bucket_group_size)
            batch_capacity = _practice_capacity_for_bucket(bucket, slot_count=count)
            groups.append(
                GenerationGroup(
                    group_id=f"{bucket.bucket_id}-g{sequence}",
                    bucket_id=bucket.bucket_id,
                    required_count=count,
                    token_budget=(
                        batch_capacity.initial_max_output_tokens
                        if batch_capacity is not None
                        else None
                    ),
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
    requested_language: str | None = None,
    finish_reason: str | None = None,
) -> ParsedGeneration:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # A truncated completion (finish_reason=length) is a bounded-recovery signal,
        # not a malformed-output one: the model was cut off mid-JSON, so the existing
        # replacement wave should run, not be told the shape itself was wrong.
        truncated = (
            finish_reason is not None
            and normalize_completion_outcome(finish_reason) == "output_token_exhausted"
        )
        return ParsedGeneration(
            accepted=(),
            rejected_count=group.required_count,
            rejection_reason_codes=(
                "GENERATOR_OUTPUT_TRUNCATED" if truncated else "GENERATOR_PARSE_FAILURE",
            ),
        )
    raw_questions = payload.get("questions", []) if isinstance(payload, dict) else []
    if not isinstance(raw_questions, list):
        return ParsedGeneration(
            accepted=(),
            rejected_count=group.required_count,
            rejection_reason_codes=("GENERATOR_CONTRACT_INVALID",),
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
        # Schema v2 authors emit no solution or explanation, so the bucket's
        # solution requirement applies only to the legacy v1 contract that still does.
        solution_required = bucket.solution_required and question.schema_version != "2"
        contract = validate_playable_question(
            question_type=question.question_type,
            question=question.question,
            options=question.options,
            correct_answer=question.correct_answer,
            solution=question.solution,
            solution_required=solution_required,
        )
        if not contract.valid:
            rejected += 1
            rejection_reason_codes.append(contract.reason_code)
            continue
        if requested_language and not _is_delivery_language_compliant(
            question,
            requested_language=requested_language,
            subject=bucket.subject,
        ):
            rejected += 1
            rejection_reason_codes.append("QUESTION_LANGUAGE_MISMATCH")
            continue
        # Lexical-equivalence English items rest on a lexicon rather than a rule, and
        # several options can be defensible. The Authority is necessary but has been
        # observed to miss a co-valid alternative, so availability must not depend on
        # it alone. No trusted lexical basis exists in the system, so these fail closed.
        if (
            normalize_subject(question.subject) == "english"
            and classify_semantic_basis(question.question)
            is SemanticBasis.EVIDENCE_REQUIRED
        ):
            rejected += 1
            rejection_reason_codes.append("SOFT_SEMANTIC_WITHOUT_TRUSTED_BASIS")
            continue
        normalized = normalize_question_identity(question.question, question.options)
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
            or solution_required
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
    # The model returned fewer items than required_count: slots with no corresponding
    # item at all, distinct from an item that was present and rejected — each such gap
    # previously counted toward rejected_count with no reason code attached.
    missing = max(0, group.required_count - len(accepted) - rejected)
    rejected += missing
    rejection_reason_codes.extend(["GENERATOR_BATCH_COUNT_MISMATCH"] * missing)
    return ParsedGeneration(
        accepted=tuple(accepted),
        rejected_count=rejected,
        rejection_reason_codes=tuple(rejection_reason_codes),
    )


def _is_delivery_language_compliant(
    question: GeneratedQuestion,
    *,
    requested_language: str,
    subject: str,
) -> bool:
    """Use a bounded script signal before the independent language-aware verifier."""
    text = " ".join(
        (
            question.question,
            *question.options,
            question.answer_explanation,
            question.solution,
        )
    )
    devanagari_count = sum("\u0900" <= character <= "\u097f" for character in text)
    if requested_language == "english":
        return devanagari_count < 8
    if requested_language == "hinglish":
        return devanagari_count < 4
    if requested_language == "hindi" and subject.casefold() != "english":
        return devanagari_count >= 4
    return True


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


def _is_persisted_schema_v2(practice_meta: object) -> bool:
    """True when a persisted question was authored under the schema-v2 contract.

    The v2 Author deliberately emits no `solution`, but a verified persisted v2 item
    carries the mandatory Authority-derived explanation in both answer snapshots.
    The existing v2 answer-contract validator enforces that requirement independently
    of the legacy solution flag.
    """
    return (
        isinstance(practice_meta, dict)
        and str(practice_meta.get("schemaVersion") or "") == "2"
    )


def validate_final_set(
    *,
    blueprint: PracticeBlueprint,
    linked_questions: list[dict[str, object]],
    requested_language: str = "english",
) -> FinalValidation:
    expected_count = blueprint.accepted_count
    actual_count = len(linked_questions)
    expected_slots = {slot.slot_id for slot in blueprint.slots}
    slots_by_id = {slot.slot_id: slot for slot in blueprint.slots}

    def result(
        ready: bool,
        reason_code: str,
        *,
        failed_slot_ids: set[str] | tuple[str, ...] = (),
        recoverable: bool = False,
    ) -> FinalValidation:
        return FinalValidation(
            ready=ready,
            reason_code=reason_code,
            failed_slot_ids=tuple(sorted(failed_slot_ids)),
            expected_question_count=expected_count,
            actual_question_count=actual_count,
            recoverable=recoverable,
        )

    question_slots: list[str] = []
    for item in linked_questions:
        meta = item.get("_practiceMeta")
        question_slots.append(
            str(meta.get("slotId") or "") if isinstance(meta, dict) else ""
        )
    if len(linked_questions) != blueprint.accepted_count:
        missing_slots = expected_slots - {slot_id for slot_id in question_slots if slot_id}
        return result(
            False,
            "FINAL_COUNT_MISMATCH",
            failed_slot_ids=missing_slots,
        )
    ids = [str(item.get("questionId") or "") for item in linked_questions]
    if not all(ids) or len(ids) != len(set(ids)):
        duplicate_slots = {
            question_slots[index]
            for index, question_id in enumerate(ids)
            if question_slots[index]
            and (not question_id or question_id in ids[:index])
        }
        return result(
            False,
            "DUPLICATE_OR_MISSING_QUESTION_ID",
            failed_slot_ids=duplicate_slots,
        )
    # Same identity as generation dedup and replacement exclusion. A generic
    # instructional stem is legitimate exam construction — English error-detection items
    # carry their content in the options — so keying the final set on the stem alone
    # collapsed genuinely distinct questions and rejected the whole set.
    normalized_texts = [
        normalize_question_identity(
            str(item.get("question") or ""), item.get("options")
        )
        for item in linked_questions
    ]
    if not all(normalized_texts) or len(normalized_texts) != len(set(normalized_texts)):
        duplicate_slots = {
            question_slots[index]
            for index, text in enumerate(normalized_texts)
            if question_slots[index] and (not text or text in normalized_texts[:index])
        }
        return result(
            False,
            "DUPLICATE_OR_EMPTY_QUESTION_TEXT",
            failed_slot_ids=duplicate_slots,
        )
    bucket_ids = {bucket.bucket_id for bucket in blueprint.buckets}
    buckets_by_id = {bucket.bucket_id: bucket for bucket in blueprint.buckets}
    counts = {bucket_id: 0 for bucket_id in bucket_ids}
    source_question_bank_ids: list[str] = []
    for item in linked_questions:
        meta = item.get("_practiceMeta")
        if not isinstance(meta, dict):
            return result(False, "QUESTION_META_INVALID")
        slot_id = str(meta.get("slotId") or "")
        bucket_id = str(meta.get("bucketId") or "")
        if bucket_id not in counts or meta.get("verified") is not True:
            return result(
                False,
                "QUESTION_NOT_VERIFIED_OR_UNKNOWN_BUCKET",
                failed_slot_ids={slot_id} if slot_id in expected_slots else (),
            )
        bucket = buckets_by_id[bucket_id]
        contract = validate_persisted_playable_question(
            item,
            expected_question_type=bucket.question_type.value,
            expected_language=requested_language,
            solution_required=(
                bucket.solution_required and not _is_persisted_schema_v2(meta)
            ),
        )
        if not contract.valid:
            return result(
                False,
                contract.reason_code,
                failed_slot_ids={slot_id} if slot_id in expected_slots else (),
            )
        source_type = str(meta.get("sourceType") or "")
        if source_type == "QUESTION_BANK":
            source_id = str(meta.get("sourceQuestionBankId") or "")
            if not source_id:
                return result(False, "QUESTION_BANK_PROVENANCE_MISSING")
            source_question_bank_ids.append(source_id)
        elif source_type == "AI_GENERATED":
            required_method = (
                "INDEPENDENT_MODEL_V2" if blueprint.schema_version == "2" else "MODEL"
            )
            if (
                bucket.verification_policy is VerificationPolicy.MANDATORY
                and meta.get("verificationMethod") != required_method
            ):
                return result(
                    False,
                    "MANDATORY_VERIFICATION_MISSING",
                    failed_slot_ids={slot_id} if slot_id in expected_slots else (),
                )
        else:
            return result(False, "QUESTION_PROVENANCE_INVALID")
        counts[bucket_id] += 1
    if len(source_question_bank_ids) != len(set(source_question_bank_ids)):
        return result(False, "DUPLICATE_QUESTION_BANK_SOURCE")
    expected = {bucket.bucket_id: bucket.required_count for bucket in blueprint.buckets}
    if counts != expected:
        failed_slots = {
            slot_id
            for slot_id, item in zip(question_slots, linked_questions, strict=True)
            if slot_id in slots_by_id
            and isinstance(item.get("_practiceMeta"), dict)
            and str(item["_practiceMeta"].get("bucketId") or "")
            != bucket_for_slot(blueprint, slots_by_id[slot_id]).bucket_id
        }
        return result(
            False,
            "BLUEPRINT_DISTRIBUTION_MISMATCH",
            failed_slot_ids=failed_slots,
            recoverable=bool(failed_slots),
        )
    if blueprint.schema_version == "2":
        actual_slots = set(question_slots)
        if actual_slots != expected_slots:
            failed_slots = expected_slots - actual_slots
            failed_slots.update(
                slot_id
                for index, slot_id in enumerate(question_slots)
                if slot_id and slot_id in question_slots[:index]
            )
            return result(
                False,
                "PLANNER_SLOT_MANIFEST_MISMATCH",
                failed_slot_ids=failed_slots,
            )
        for item in linked_questions:
            meta = item["_practiceMeta"]
            slot = slots_by_id[str(meta["slotId"])]
            # Compared through the same canonical normalizers the admission-time
            # matcher uses.  QuestionBank stores backend difficulty vocabulary
            # (EASY/MEDIUM/HARD) while planner slots use basic/intermediate/advanced,
            # so a raw casefold comparison rejected every reused Question even though
            # it had already satisfied strict slot compatibility.  Topic carries the
            # same alias risk.  This is a vocabulary fix, not a relaxation: the
            # normalizers are exact alias maps and still reject genuine mismatches.
            if (
                normalize_topic(item.get("topic")) != slot.topic_id
                or normalize_difficulty(item.get("difficulty")) != slot.difficulty.value
                or str(meta.get("questionType") or "").casefold() != slot.question_type.value
                or meta.get("verified") is not True
            ):
                return result(
                    False,
                    "PLANNER_SLOT_CONTRACT_MISMATCH",
                    failed_slot_ids={slot.slot_id},
                )
    return result(True, "READY")
