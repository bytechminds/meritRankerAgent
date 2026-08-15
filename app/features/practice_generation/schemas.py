"""Typed contracts for practice planning, generation, and progress."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from services.llm.orchestration.prompt_budget import PromptInputBudget
from tools.web_search.models import FreshEvidenceBundle


class PracticeType(StrEnum):
    SIMILAR_QUESTION = "SIMILAR_QUESTION"
    QUICK_PRACTICE = "QUICK_PRACTICE"
    QUIZ = "QUIZ"
    TOPIC_TEST = "TOPIC_TEST"
    SECTIONAL_TEST = "SECTIONAL_TEST"
    FULL_MOCK = "FULL_MOCK"
    CURRENT_AFFAIRS_SET = "CURRENT_AFFAIRS_SET"


class InternalPhase(StrEnum):
    QUEUED = "QUEUED"
    PLANNING = "PLANNING"
    MATCHING_EXISTING = "MATCHING_EXISTING"
    GENERATING = "GENERATING"
    VERIFYING = "VERIFYING"
    FINALIZING = "FINALIZING"
    READY = "READY"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class Difficulty(StrEnum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Complexity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PlannerFamily(StrEnum):
    QUANT_REASONING = "quant_reasoning"
    ENGLISH = "english"
    FACTUAL = "factual"


class QuestionType(StrEnum):
    MCQ = "mcq"
    MSQ = "msq"
    NUMERICAL = "numerical"
    DESCRIPTIVE = "descriptive"


class VerificationDecision(StrEnum):
    ACCEPT = "ACCEPT"
    REPAIRABLE = "REPAIRABLE"
    REGENERATE = "REGENERATE"
    TERMINAL_REJECTION = "TERMINAL_REJECTION"


class VerificationPolicy(StrEnum):
    NONE = "NONE"
    SELECTIVE = "SELECTIVE"
    MANDATORY = "MANDATORY"


class PracticeGenerationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    request_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=100)
    turn_id: str = Field(min_length=1, max_length=128)
    original_query: str = Field(min_length=1, max_length=5000)
    practice_type: PracticeType
    requested_count: int = Field(ge=1)
    accepted_count: int = Field(ge=1, le=100)
    subject: str = Field(min_length=1, max_length=64)
    topic: str | None = Field(default=None, max_length=128)
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    mixed_difficulty_requested: bool = False
    explicit_difficulty_requested: bool = False
    language: Literal["english", "hinglish", "hindi"] = "english"
    exam_id: str | None = Field(default=None, max_length=128)
    exam_stage: str | None = Field(default=None, max_length=64)
    exam_profile_id: str | None = Field(default=None, max_length=160)
    source_question_reference: str | None = Field(default=None, max_length=128)
    requires_fresh_evidence: bool = False
    freshness_reason: str | None = Field(default=None, max_length=64)
    fresh_evidence: FreshEvidenceBundle | None = None
    include_solutions: bool = True
    assessment_title: str = Field(min_length=1, max_length=180)

    @model_validator(mode="after")
    def _accepted_count_is_clamped(self) -> PracticeGenerationRequest:
        if self.accepted_count != min(self.requested_count, 100):
            raise ValueError("accepted_count must equal min(requested_count, 100)")
        if self.requires_fresh_evidence:
            if not self.freshness_reason:
                raise ValueError("freshness-required practice needs a reason")
            if self.fresh_evidence is None:
                raise ValueError("freshness-required practice needs fresh evidence")
            if len(self.fresh_evidence.items) < self.accepted_count:
                raise ValueError("fresh evidence is insufficient for the accepted count")
        elif self.fresh_evidence is not None:
            raise ValueError("static practice must not retain fresh evidence")
        return self


class DemandBucket(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    bucket_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    subject: str = Field(min_length=1, max_length=64)
    topic: str = Field(min_length=1, max_length=128)
    difficulty: Difficulty
    question_type: QuestionType
    required_count: int = Field(ge=1, le=100)
    keywords: list[str] = Field(default_factory=list, max_length=8)
    question_intent: str = Field(min_length=1, max_length=240)
    excluded_variants: list[str] = Field(default_factory=list, max_length=8)
    verification_policy: VerificationPolicy
    solution_required: bool = True
    generation_group_hint: int | None = Field(default=None, ge=1, le=5)

    @field_validator("subject")
    @classmethod
    def _validate_subject(cls, value: str) -> str:
        normalized = value.casefold().replace(" ", "_").replace("-", "_")
        allowed = {
            "math",
            "reasoning",
            "science",
            "history",
            "geography",
            "english",
            "physics",
            "chemistry",
            "biology",
            "computer_science",
            "economics",
            "polity",
            "general",
            "other",
        }
        if normalized not in allowed:
            raise ValueError("unsupported demand-bucket subject")
        return normalized

    @field_validator("topic")
    @classmethod
    def _validate_topic(cls, value: str) -> str:
        normalized = value.casefold().replace(" ", "_").replace("-", "_")
        if not normalized or not all(
            character.isalnum() or character == "_" for character in normalized
        ):
            raise ValueError("topic must be a canonical alphanumeric key")
        return normalized

    @field_validator("keywords", "excluded_variants")
    @classmethod
    def _bound_list_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip()[:96] for value in values if value.strip()]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("list values must be unique")
        return cleaned


class PlannerSlot(BaseModel):
    """One immutable assessment-design requirement for one playable question."""

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    slot_id: str = Field(pattern=r"^slot-(?:00[1-9]|0[1-9][0-9]|100)$")
    subject_id: str = Field(min_length=1, max_length=64)
    topic_id: str = Field(min_length=1, max_length=128)
    category_id: str = Field(min_length=1, max_length=128)
    difficulty: Difficulty
    complexity: Complexity
    exam_ids: list[str] = Field(default_factory=list, max_length=8)
    question_type: QuestionType = QuestionType.MCQ
    target_skill: str = Field(min_length=1, max_length=160)
    variation_hint: str = Field(min_length=1, max_length=160)
    pattern_family_id: str | None = Field(default=None, max_length=160)
    generator_route_hint: str = Field(min_length=1, max_length=160)
    reasoning_target: str | None = Field(default=None, max_length=240)
    trap_type: str | None = Field(default=None, max_length=96)
    not_same_when: list[str] = Field(default_factory=list, max_length=8)
    generation_group_hint: int | None = Field(default=None, ge=1, le=5)

    @field_validator("subject_id", "topic_id", "category_id")
    @classmethod
    def _canonical_identifier(cls, value: str) -> str:
        normalized = value.casefold().replace("&", "and")
        normalized = "_".join(normalized.replace("-", " ").split())
        if not normalized or not all(
            character.isalnum() or character == "_" for character in normalized
        ):
            raise ValueError("planner slot metadata must use canonical identifiers")
        return normalized

    @field_validator("subject_id")
    @classmethod
    def _supported_subject(cls, value: str) -> str:
        if value not in {
            "math",
            "reasoning",
            "science",
            "history",
            "geography",
            "english",
            "physics",
            "chemistry",
            "biology",
            "computer_science",
            "economics",
            "polity",
            "general",
            "other",
        }:
            raise ValueError("unsupported planner-slot subject")
        return value

    @field_validator("exam_ids")
    @classmethod
    def _canonical_exam_ids(cls, values: list[str]) -> list[str]:
        normalized = [
            "_".join(value.upper().replace("-", " ").split())
            for value in values
            if value.strip()
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("planner-slot exam IDs must be unique")
        return normalized

    @field_validator("not_same_when")
    @classmethod
    def _unique_variation_constraints(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip()[:160] for value in values if value.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("planner-slot variation constraints must be unique")
        return cleaned


class PracticeBlueprint(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    schema_version: Literal["1", "2"] = "1"
    practice_type: PracticeType
    accepted_count: int = Field(ge=1, le=100)
    planner_family: PlannerFamily | None = None
    slots: list[PlannerSlot] = Field(default_factory=list, max_length=100)
    buckets: list[DemandBucket] = Field(default_factory=list, max_length=100)

    @model_validator(mode="before")
    @classmethod
    def _derive_compatibility_buckets(cls, value: Any) -> Any:
        if not isinstance(value, dict) or str(value.get("schema_version") or "1") != "2":
            return value
        updated = dict(value)
        slots = [PlannerSlot.model_validate(slot) for slot in list(updated.get("slots") or [])]
        grouped: dict[tuple[str, ...], list[PlannerSlot]] = {}
        for slot in slots:
            key = (
                slot.subject_id,
                slot.topic_id,
                slot.category_id,
                slot.difficulty.value,
                slot.question_type.value,
                slot.generator_route_hint,
            )
            grouped.setdefault(key, []).append(slot)
        buckets: list[dict[str, Any]] = []
        for index, compatible_slots in enumerate(grouped.values(), start=1):
            first = compatible_slots[0]
            keywords = list(
                dict.fromkeys(
                    [first.topic_id, first.category_id]
                    + [slot.target_skill for slot in compatible_slots]
                )
            )[:8]
            hints = [
                slot.generation_group_hint
                for slot in compatible_slots
                if slot.generation_group_hint is not None
            ]
            buckets.append(
                {
                    "bucket_id": f"slot-bucket-{index:03d}",
                    "subject": first.subject_id,
                    "topic": first.topic_id,
                    "difficulty": first.difficulty.value,
                    "question_type": first.question_type.value,
                    "required_count": len(compatible_slots),
                    "keywords": keywords,
                    "question_intent": (
                        f"Assess {first.target_skill.replace('_', ' ')} with "
                        f"{first.variation_hint.replace('_', ' ')} variation."
                    ),
                    "excluded_variants": list(first.not_same_when),
                    "verification_policy": VerificationPolicy.NONE.value,
                    "generation_group_hint": min(hints) if hints else None,
                }
            )
        updated["buckets"] = buckets
        return updated

    @model_validator(mode="after")
    def _validate_distribution(self) -> PracticeBlueprint:
        if self.schema_version == "2":
            if self.planner_family is None:
                raise ValueError("schema-v2 blueprint requires planner_family")
            if len(self.slots) != self.accepted_count:
                raise ValueError("schema-v2 slot count must equal accepted_count")
            slot_ids = [slot.slot_id for slot in self.slots]
            if len(slot_ids) != len(set(slot_ids)):
                raise ValueError("planner slot IDs must be unique")
            signatures = [
                (
                    slot.subject_id,
                    slot.topic_id,
                    slot.category_id,
                    slot.difficulty,
                    slot.complexity,
                    slot.question_type,
                    slot.target_skill.casefold(),
                    slot.variation_hint.casefold(),
                )
                for slot in self.slots
            ]
            if len(signatures) != len(set(signatures)):
                raise ValueError("planner slots must be intentionally distinct")
        elif self.slots or self.planner_family is not None:
            raise ValueError("schema-v1 blueprint cannot contain schema-v2 planner fields")
        if not self.buckets:
            raise ValueError("blueprint requires compatibility buckets")
        if sum(bucket.required_count for bucket in self.buckets) != self.accepted_count:
            raise ValueError("bucket required_count sum must equal accepted_count")
        ids = [bucket.bucket_id for bucket in self.buckets]
        if len(ids) != len(set(ids)):
            raise ValueError("bucket IDs must be unique")
        return self


class GenerationGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    group_id: str = Field(min_length=1, max_length=160)
    bucket_id: str = Field(min_length=1, max_length=128)
    required_count: int = Field(ge=1, le=5)
    attempt: int = Field(default=0, ge=0, le=2)
    slot_ids: list[str] = Field(default_factory=list, max_length=5)
    token_budget: int | None = Field(default=None, ge=1, le=32_000)


class QuestionOption(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    option_id: Literal["0", "1", "2", "3"]
    value: str = Field(min_length=1, max_length=1000)


class GeneratedQuestion(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    schema_version: Literal["1", "2"] = "1"
    generation_item_id: str = Field(min_length=1, max_length=160)
    bucket_id: str = Field(min_length=1, max_length=128)
    slot_id: str | None = Field(
        default=None,
        pattern=r"^slot-(?:00[1-9]|0[1-9][0-9]|100)$",
    )
    question: str = Field(min_length=8, max_length=4000)
    question_type: QuestionType
    options: list[str] = Field(default_factory=list, max_length=8)
    canonical_options: list[QuestionOption] = Field(default_factory=list, max_length=4)
    correct_option_id: Literal["0", "1", "2", "3"] | None = None
    correct_answer: str = Field(min_length=1, max_length=1000)
    answer_explanation: str = Field(default="", max_length=5000)
    solution: str = Field(default="", max_length=5000)
    subject: str = Field(min_length=1, max_length=64)
    topic: str = Field(min_length=1, max_length=128)
    difficulty: Difficulty

    @model_validator(mode="before")
    @classmethod
    def _normalize_option_contract(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        updated = dict(value)
        raw_options = list(updated.get("options") or [])
        if raw_options and all(isinstance(option, dict) for option in raw_options):
            canonical = [QuestionOption.model_validate(option) for option in raw_options]
            updated["canonical_options"] = [option.model_dump() for option in canonical]
            updated["options"] = [option.value for option in canonical]
        if updated.get("schema_version") == "2":
            if not updated.get("answer_explanation") and updated.get("solution"):
                updated["answer_explanation"] = updated["solution"]
            if not updated.get("solution") and updated.get("answer_explanation"):
                updated["solution"] = updated["answer_explanation"]
        return updated

    @model_validator(mode="after")
    def _validate_answer_structure(self) -> GeneratedQuestion:
        if self.question_type in {QuestionType.MCQ, QuestionType.MSQ}:
            if len(self.options) < 2 or len(set(self.options)) != len(self.options):
                raise ValueError("choice questions require unique options")
            if self.question_type == QuestionType.MCQ and self.correct_answer not in self.options:
                raise ValueError("MCQ correct_answer must be an option")
        if self.schema_version == "2":
            if self.question_type is not QuestionType.MCQ or self.slot_id is None:
                raise ValueError("schema-v2 generated questions require an MCQ slot")
            if [option.option_id for option in self.canonical_options] != ["0", "1", "2", "3"]:
                raise ValueError("schema-v2 options require exact stable index IDs")
            if [option.value for option in self.canonical_options] != self.options:
                raise ValueError("schema-v2 option projection mismatch")
            if self.correct_option_id is None:
                raise ValueError("schema-v2 correct_option_id is required")
            if self.options[int(self.correct_option_id)] != self.correct_answer:
                raise ValueError("schema-v2 correct answer does not match correct_option_id")
            if not self.answer_explanation:
                raise ValueError("schema-v2 answer explanation is required")
        return self


class AssessmentProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    requested_count: int = Field(ge=1)
    accepted_count: int = Field(ge=1, le=100)
    reused_count: int = Field(default=0, ge=0, le=100)
    generated_count: int = Field(default=0, ge=0, le=100)
    verified_count: int = Field(default=0, ge=0, le=100)
    failed_count: int = Field(default=0, ge=0)
    ready_question_count: int = Field(default=0, ge=0, le=100)
    progress_percent: int = Field(default=0, ge=0, le=100)
    phase: InternalPhase = InternalPhase.QUEUED
    playable: bool = False
    error_code: str | None = Field(default=None, max_length=96)

    @model_validator(mode="after")
    def _ready_invariants(self) -> AssessmentProgress:
        if self.ready_question_count > self.accepted_count:
            raise ValueError("ready_question_count cannot exceed accepted_count")
        if self.playable and (
            self.phase is not InternalPhase.READY
            or self.ready_question_count != self.accepted_count
        ):
            raise ValueError("playable requires READY with the exact accepted count")
        return self


class PracticeLaunchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    test_id: str = Field(min_length=1, max_length=128)
    status: Literal["GENERATING", "READY", "FAILED"]
    requested_count: int = Field(ge=1)
    accepted_count: int = Field(ge=1, le=100)
    count_clamped: bool
    progress_percent: int = Field(ge=0, le=100)
    playable: bool
    duplicate_request: bool = False
    message: str = Field(min_length=1, max_length=500)
    execution_id: str | None = Field(default=None, max_length=128)


class PracticeControlRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    mode: Literal["practice_control"]
    action: Literal["cancel", "resume"]
    test_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=100)


class PracticeControlResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    success: bool
    test_id: str = Field(min_length=1, max_length=128)
    status: Literal["GENERATING", "CANCEL_REQUESTED", "CANCELLED", "READY", "FAILED"]
    execution_id: str | None = Field(default=None, max_length=128)
    already_active: bool = False
    ready_count: int = Field(default=0, ge=0, le=100)
    requested_count: int = Field(default=0, ge=0, le=100)


class PracticeLaunchDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    eligible: bool
    reason_code: str = Field(min_length=1, max_length=96)
    requested_artifact: str | None = Field(default=None, max_length=32)


class PracticeFreshnessRequirement(BaseModel):
    """Deterministic pre-launch freshness decision for one Practice request."""

    model_config = ConfigDict(frozen=True)

    requires_fresh_evidence: bool = False
    freshness_reason: str | None = Field(default=None, max_length=64)


class GeneratedBatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    content: str
    route_id: str = Field(min_length=1, max_length=160)
    model: str = Field(min_length=1, max_length=160)
    prompt_input_budget: PromptInputBudget | None = None


class PlannerEnvelope(BaseModel):
    blueprint: PracticeBlueprint


class GenerationEnvelope(BaseModel):
    questions: list[GeneratedQuestion] = Field(default_factory=list, max_length=5)


class VerificationResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    schema_version: Literal["1", "2"] = "1"
    generation_item_id: str = Field(min_length=1, max_length=160)
    slot_id: str | None = Field(default=None, max_length=32)
    decision: VerificationDecision | None = None
    independently_solved_option_id: Literal["0", "1", "2", "3"] | None = None
    reason_codes: list[str] = Field(default_factory=list, max_length=8)
    evidence_urls: list[str] = Field(default_factory=list, max_length=4)
    approved: bool | None = None
    reason_code: str | None = Field(default=None, max_length=96)

    @model_validator(mode="after")
    def _verification_contract(self) -> VerificationResult:
        if self.schema_version == "2":
            if self.slot_id is None or self.decision is None or not self.reason_codes:
                raise ValueError("schema-v2 verifier binding is incomplete")
            if self.decision is VerificationDecision.ACCEPT and (
                self.independently_solved_option_id is None
            ):
                raise ValueError("accepted verification requires an independent answer")
        elif self.approved is None or not self.reason_code:
            raise ValueError("legacy verification result is incomplete")
        return self

    @property
    def is_approved(self) -> bool:
        if self.schema_version == "2":
            return self.decision is VerificationDecision.ACCEPT
        return self.approved is True


class PracticeGraphCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: Literal["plan_and_fill", "generate_group", "generate_wave", "finalize"]
    test_id: str = Field(min_length=1, max_length=128)
    group_id: str | None = Field(default=None, max_length=160)
    group_ids: list[str] = Field(default_factory=list, max_length=2)
    attempt: int = Field(default=0, ge=0, le=2)
