"""Typed contracts for practice planning, generation, and progress."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from practice_limits import (
    MAX_PRACTICE_QUESTIONS,
    MAX_REQUESTED_PRACTICE_QUESTIONS,
    PRACTICE_SLOT_ID_PATTERN,
    effective_practice_question_count,
)
from schemas.doubt_solver import CanonicalLanguage, normalize_question_language
from schemas.practice_limit import PracticeLimitation
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


class RequestedPracticeConstraint(BaseModel):
    """Accepted composition identity and its already-grounded source span."""

    model_config = ConfigDict(str_strip_whitespace=True, frozen=True, populate_by_name=True)

    subject_id: str = Field(min_length=1, max_length=64, alias="subjectId")
    topic_id: str = Field(min_length=1, max_length=128, alias="topicId")
    source_text: str = Field(min_length=1, max_length=160, alias="sourceText")


class PracticeGenerationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    request_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=100)
    turn_id: str = Field(min_length=1, max_length=128)
    original_query: str = Field(min_length=1, max_length=5000)
    practice_type: PracticeType
    requested_count: int = Field(ge=1, le=MAX_REQUESTED_PRACTICE_QUESTIONS)
    accepted_count: int = Field(ge=1, le=MAX_PRACTICE_QUESTIONS)
    limitation: PracticeLimitation | None = None
    subject: str = Field(min_length=1, max_length=64)
    topic: str | None = Field(default=None, max_length=128)
    topics: list[str] | None = Field(default=None, max_length=12)
    trusted_constraints: tuple[RequestedPracticeConstraint, ...] = ()
    difficulty: Difficulty = Difficulty.INTERMEDIATE
    mixed_difficulty_requested: bool = False
    explicit_difficulty_requested: bool = False
    # Explicit per-level counts. Already validated to sum to accepted_count before
    # the request is built, so slot construction consumes it without re-deriving.
    difficulty_distribution: dict[Difficulty, int] | None = None
    language: CanonicalLanguage = "english"
    language_source: Literal["REQUEST", "EXPLICIT_QUERY"] = "REQUEST"
    exam_id: str | None = Field(default=None, max_length=128)
    exam_stage: str | None = Field(default=None, max_length=64)
    exam_profile_id: str | None = Field(default=None, max_length=160)
    source_question_reference: str | None = Field(default=None, max_length=128)
    requires_fresh_evidence: bool = False
    freshness_reason: str | None = Field(default=None, max_length=64)
    fresh_evidence: FreshEvidenceBundle | None = None
    include_solutions: bool = True
    assessment_title: str = Field(min_length=1, max_length=180)

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: object) -> CanonicalLanguage:
        return normalize_question_language(value)

    @model_validator(mode="after")
    def _accepted_count_matches_requested_count(self) -> PracticeGenerationRequest:
        effective_count = effective_practice_question_count(self.requested_count)
        if self.accepted_count != effective_count:
            raise ValueError("accepted_count must equal the effective requested count")
        if self.requested_count == self.accepted_count and self.limitation is not None:
            raise ValueError("unlimited request must not carry a limitation")
        if self.requested_count != self.accepted_count:
            if self.limitation is None:
                raise ValueError("limited request must carry a limitation")
            if (
                self.limitation.type != "QUESTION_COUNT_LIMIT"
                or self.limitation.requested_value != self.requested_count
                or self.limitation.effective_value != self.accepted_count
                or self.limitation.maximum_value != MAX_PRACTICE_QUESTIONS
                or self.limitation.message_key != "PRACTICE_MAX_QUESTIONS_LIMITED"
            ):
                raise ValueError("limitation does not match the question-count contract")
        if self.difficulty_distribution is not None:
            counts = self.difficulty_distribution.values()
            if any(count < 0 for count in counts):
                raise ValueError("difficulty distribution counts must not be negative")
            if sum(counts) != self.accepted_count:
                raise ValueError("difficulty distribution must sum to accepted_count")
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

    @property
    def effective_count(self) -> int:
        """The count supplied to the planner and all downstream lifecycle stages."""
        return self.accepted_count


class DemandBucket(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    bucket_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    subject: str = Field(min_length=1, max_length=64)
    topic: str = Field(min_length=1, max_length=128)
    difficulty: Difficulty
    question_type: QuestionType
    required_count: int = Field(ge=1, le=MAX_PRACTICE_QUESTIONS)
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

    slot_id: str = Field(pattern=PRACTICE_SLOT_ID_PATTERN)
    constraint_ref: str | None = Field(default=None, min_length=1, max_length=32)
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


class TopicEvidence(BaseModel):
    """One requested topic, grounded by a short exact span the student actually wrote.

    ``source_text`` is a short exact substring of the original query, own script,
    sufficient for ``_grounded_topic_ids`` to confirm the topic was actually requested;
    the planner supplies the semantic normalization in ``topic_id``. It is deliberately
    NOT required to reproduce the full source question or the complete relevant phrase —
    grounding (``planning.py:_grounded_topic_ids``) only ever tests literal substring
    membership, so the shortest span that still uniquely names the topic satisfies it.
    Audited: the only production consumer of this field is that substring check; no
    caller needs the complete original question here.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True, frozen=True, populate_by_name=True
    )

    source_text: str = Field(min_length=1, max_length=160, alias="sourceText")
    topic_id: str = Field(min_length=1, max_length=128, alias="topicId")


class PracticeBlueprint(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True, frozen=True, populate_by_name=True
    )

    schema_version: Literal["1", "2"] = "1"
    practice_type: PracticeType
    accepted_count: int = Field(ge=1, le=MAX_PRACTICE_QUESTIONS)
    planner_family: PlannerFamily | None = None
    slots: list[PlannerSlot] = Field(
        default_factory=list,
        max_length=MAX_PRACTICE_QUESTIONS,
    )
    buckets: list[DemandBucket] = Field(
        default_factory=list,
        max_length=MAX_PRACTICE_QUESTIONS,
    )
    # Emitted once per distinct requested topic, never repeated inside slots, so a
    # 50-slot blueprint stays bounded.
    requested_topic_evidence: list[TopicEvidence] = Field(
        default_factory=list,
        alias="requestedTopicEvidence",
        max_length=MAX_PRACTICE_QUESTIONS,
    )

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


# ---------------------------------------------------------------------------------
# Provider-native structured-output generation schema (schema-v2 planner wire shape).
#
# This is NOT ``PlannerSlot``/``TopicEvidence``.model_json_schema(): that compiles
# optional fields to ``anyOf``/``$ref``/``$defs``, which strict-mode providers reject
# (see the identical constraint documented next to the Gemini verifier schema). This
# is the smallest JSON-Schema subset that constrains SHAPE only — type, enum,
# required, additionalProperties — for a provider's own strict/constrained-decoding
# mode. Business rules a provider schema cannot express (``max_length``, item-count
# bounds, uniqueness, grounding, cross-field invariants) stay enforced locally by
# ``PracticeBlueprint``/``PlannerSlot``/``TopicEvidence`` after parsing, unchanged by
# whichever path produced the JSON. Shared by every schema-v2 planner family
# (quant_reasoning, english, factual): their wire shape is identical; only the prompt
# guidance differs per family.
# ---------------------------------------------------------------------------------


def planner_generation_schema() -> dict[str, Any]:
    """Return the schema-v2 planner output shape as a strict-mode JSON Schema.

    Every property is listed in ``required`` per provider strict-mode semantics
    (required means "present in the object", not "non-null"); a field that is
    optional in ``PlannerSlot``/``TopicEvidence`` is typed nullable here instead.
    """
    nullable_string = {"type": ["string", "null"]}
    slot = {
        "type": "object",
        "properties": {
            "slot_id": {"type": "string"},
            "constraint_ref": nullable_string,
            "subject_id": {"type": "string"},
            "topic_id": {"type": "string"},
            "category_id": {"type": "string"},
            "difficulty": {"type": "string", "enum": [value.value for value in Difficulty]},
            "complexity": {"type": "string", "enum": [value.value for value in Complexity]},
            "exam_ids": {"type": "array", "items": {"type": "string"}},
            "question_type": {"type": "string", "enum": [QuestionType.MCQ.value]},
            "target_skill": {"type": "string"},
            "variation_hint": {"type": "string"},
            "pattern_family_id": nullable_string,
            "generator_route_hint": {"type": "string"},
            "reasoning_target": nullable_string,
            "trap_type": nullable_string,
            "not_same_when": {"type": "array", "items": {"type": "string"}},
            "generation_group_hint": {"type": ["integer", "null"]},
        },
        "required": [
            "slot_id",
            "constraint_ref",
            "subject_id",
            "topic_id",
            "category_id",
            "difficulty",
            "complexity",
            "exam_ids",
            "question_type",
            "target_skill",
            "variation_hint",
            "pattern_family_id",
            "generator_route_hint",
            "reasoning_target",
            "trap_type",
            "not_same_when",
            "generation_group_hint",
        ],
        "additionalProperties": False,
    }
    evidence = {
        "type": "object",
        "properties": {
            "sourceText": {"type": "string"},
            "topicId": {"type": "string"},
        },
        "required": ["sourceText", "topicId"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "slots": {"type": "array", "items": slot},
            "requestedTopicEvidence": {
                "type": ["array", "null"],
                "items": evidence,
            },
        },
        "required": ["slots", "requestedTopicEvidence"],
        "additionalProperties": False,
    }


def practice_generator_generation_schema() -> dict[str, Any]:
    """Return the schema-v2 question-generator output shape as a strict-mode JSON Schema.

    Shared by every generator route (math/reasoning/english/factual/general) and every
    replacement wave: ``question_generator_v2.md``, ``question_generator_factual.md``,
    ``question_repair.md``, and ``question_regenerator.md`` all specify this identical
    wire shape, so one schema covers initial generation and both replacement waves.
    Constrains structure only; ``GeneratedQuestion`` stays the authority for
    ``max_length``, option-count, and cross-field invariants after parsing.
    """
    option = {
        "type": "object",
        "properties": {
            "option_id": {"type": "string", "enum": ["0", "1", "2", "3"]},
            "value": {"type": "string"},
        },
        "required": ["option_id", "value"],
        "additionalProperties": False,
    }
    question = {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["2"]},
            "bucket_id": {"type": "string"},
            "slot_id": {"type": "string"},
            "question": {"type": "string"},
            "question_type": {"type": "string", "enum": [QuestionType.MCQ.value]},
            "options": {"type": "array", "items": option},
            "correct_option_id": {"type": "string", "enum": ["0", "1", "2", "3"]},
            "subject": {"type": "string"},
            "topic": {"type": "string"},
            "difficulty": {"type": "string", "enum": [value.value for value in Difficulty]},
        },
        "required": [
            "schema_version",
            "bucket_id",
            "slot_id",
            "question",
            "question_type",
            "options",
            "correct_option_id",
            "subject",
            "topic",
            "difficulty",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"questions": {"type": "array", "items": question}},
        "required": ["questions"],
        "additionalProperties": False,
    }


def practice_verifier_generation_schema() -> dict[str, Any]:
    """Return the schema-v2 Answer Authority output shape as a strict-mode JSON Schema.

    Shared by every Practice verifier route (math/reasoning/english/factual) regardless
    of provider: ``question_verifier_v2.md`` specifies this identical wire shape.
    Includes ``evidence_urls`` — present on ``VerificationResult`` and required by
    ``_enforce_evidence_citations`` for fresh-evidence questions, but absent from the
    Gemini adapter's previous hand-written copy of this schema; adapters should build
    their native response format from this one function rather than duplicating it, so
    the two cannot drift apart again. Constrains structure only; ``VerificationResult``
    stays the authority for ``max_length``, item-count bounds, and cross-field
    invariants after parsing.
    """
    option_id = {"type": "string", "enum": ["0", "1", "2", "3"]}
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["2"]},
            "generation_item_id": {"type": "string"},
            "slot_id": {"type": "string"},
            "decision": {
                "type": "string",
                "enum": ["ACCEPT", "REPAIRABLE", "REGENERATE", "TERMINAL_REJECTION"],
            },
            "valid_option_ids": {"type": "array", "items": option_id},
            "answer_explanation": {"type": "string"},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "evidence_urls": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "schema_version",
            "generation_item_id",
            "slot_id",
            "decision",
            "valid_option_ids",
            "answer_explanation",
            "reason_codes",
            "evidence_urls",
        ],
        "additionalProperties": False,
    }


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
        pattern=PRACTICE_SLOT_ID_PATTERN,
    )
    question: str = Field(min_length=8, max_length=4000)
    question_type: QuestionType
    options: list[str] = Field(default_factory=list, max_length=8)
    canonical_options: list[QuestionOption] = Field(default_factory=list, max_length=4)
    # AUTHOR-INTENDED, PENDING_VERIFICATION, NON-AUTHORITATIVE. This is the author's
    # proposal only. It becomes student-facing truth solely when a blind Answer
    # Authority independently finds this same id to be the one valid option.
    correct_option_id: Literal["0", "1", "2", "3"] | None = None
    # Derived from ``correct_option_id`` for schema v2 — not authored, not compared.
    correct_answer: str = Field(min_length=1, max_length=1000)
    # Schema-v2 authors omit prose. The blind Answer Authority supplies this only after
    # independently selecting one canonical option; persistence then writes that exact
    # value to both canonical explanation snapshots.
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
            # Bookkeeping the code owns. A schema-v2 item is addressed by its slot, so
            # the item id is derivable and the author never needs to invent one. Models
            # copied the prompt's placeholder verbatim and the duplicate check then
            # discarded two of every three questions, which is a contract defect rather
            # than a model defect. Derived unconditionally so a stale or repeated value
            # cannot reach the dedupe set.
            slot_id = updated.get("slot_id")
            if isinstance(slot_id, str) and slot_id:
                updated["generation_item_id"] = f"item-{slot_id}"
            # The author emits only the option id. ``correct_answer`` is the exact
            # value at that id, so deriving it here removes a field the author could
            # contradict its own options with, and keeps every downstream consumer
            # (persistence, review, matching) on the shape they already expect.
            if not updated.get("correct_answer"):
                canonical = updated.get("canonical_options") or []
                option_id = updated.get("correct_option_id")
                if isinstance(option_id, str) and option_id.isdigit():
                    index = int(option_id)
                    if 0 <= index < len(canonical):
                        updated["correct_answer"] = canonical[index]["value"]
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
        return self


class AssessmentProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    requested_count: int = Field(ge=1, le=MAX_REQUESTED_PRACTICE_QUESTIONS)
    accepted_count: int = Field(ge=1, le=MAX_PRACTICE_QUESTIONS)
    reused_count: int = Field(default=0, ge=0, le=MAX_PRACTICE_QUESTIONS)
    generated_count: int = Field(default=0, ge=0, le=MAX_PRACTICE_QUESTIONS)
    verified_count: int = Field(default=0, ge=0, le=MAX_PRACTICE_QUESTIONS)
    failed_count: int = Field(default=0, ge=0)
    ready_question_count: int = Field(default=0, ge=0, le=MAX_PRACTICE_QUESTIONS)
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
    requested_count: int = Field(ge=1, le=MAX_REQUESTED_PRACTICE_QUESTIONS)
    accepted_count: int = Field(ge=1, le=MAX_PRACTICE_QUESTIONS)
    count_clamped: bool
    limitation: PracticeLimitation | None = None
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
    ready_count: int = Field(default=0, ge=0, le=MAX_PRACTICE_QUESTIONS)
    requested_count: int = Field(default=0, ge=0, le=MAX_REQUESTED_PRACTICE_QUESTIONS)


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
    # The provider's own stop reason ("stop", "length", ...), carried through so a
    # truncated response can be told apart from a malformed one before either is
    # treated as a schema failure. None for callers/executors that do not report it.
    finish_reason: str | None = None


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
    # Every option the authority independently found to satisfy the stem. Exactly one
    # entry is the only acceptable outcome: an empty list means no supplied option is
    # correct, and more than one means the item has multiple valid answers. Agreement
    # on a single id can never prove that a second id is not also valid, so this list
    # — not a scalar answer — is what the deterministic gate consumes.
    valid_option_ids: list[Literal["0", "1", "2", "3"]] = Field(
        default_factory=list, max_length=4
    )
    # Required by the strict schema-v2 provider response for ACCEPT. It is produced by
    # the independent Answer Authority, never copied from the author.
    answer_explanation: str = Field(default="", max_length=5000)
    reason_codes: list[str] = Field(default_factory=list, max_length=8)
    evidence_urls: list[str] = Field(default_factory=list, max_length=4)
    approved: bool | None = None
    reason_code: str | None = Field(default=None, max_length=96)

    @model_validator(mode="after")
    def _verification_contract(self) -> VerificationResult:
        if self.schema_version == "2":
            if self.slot_id is None or self.decision is None or not self.reason_codes:
                raise ValueError("schema-v2 verifier binding is incomplete")
            if len(set(self.valid_option_ids)) != len(self.valid_option_ids):
                raise ValueError("valid_option_ids must not repeat an option id")
            if self.decision is VerificationDecision.ACCEPT:
                if len(self.valid_option_ids) != 1:
                    raise ValueError(
                        "accepted verification requires exactly one valid option id"
                    )
                if not self.answer_explanation:
                    raise ValueError(
                        "accepted verification requires an answer explanation"
                    )
                if self.independently_solved_option_id is None:
                    object.__setattr__(
                        self,
                        "independently_solved_option_id",
                        self.valid_option_ids[0],
                    )
                if self.independently_solved_option_id != self.valid_option_ids[0]:
                    raise ValueError(
                        "independent answer must be the single valid option id"
                    )
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
