"""Schema-first models and narrow raw-record adapters for Pattern intelligence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PatternRuntimeMode(StrEnum):
    """Trusted workflow that will consume the selected Pattern context."""

    PRACTICE = "practice"
    DOUBT = "doubt"


class PatternMatchTier(StrEnum):
    """Safe handling tier determined only after authoritative record validation."""

    REUSE_SAFE = "REUSE_SAFE"
    GUIDANCE_SAFE = "GUIDANCE_SAFE"
    IGNORE = "IGNORE"


class VectorPatternCandidate(BaseModel):
    """Minimal vector-discovery result; score is a negative relevance gate only."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)

    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    version_hash: str | None = Field(default=None, alias="versionHash", max_length=256)

    @classmethod
    def from_raw(
        cls,
        value: VectorPatternCandidate | Mapping[str, Any],
    ) -> VectorPatternCandidate | None:
        if isinstance(value, cls):
            return value
        pattern_id = _text(value.get("patternId"))
        if not pattern_id:
            return None
        score = value.get("score")
        try:
            parsed_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            parsed_score = None
        try:
            return cls(
                patternId=pattern_id,
                score=parsed_score,
                versionHash=_text(value.get("versionHash")) or None,
            )
        except ValidationError:
            return None


class PatternRuntimeRequest(BaseModel):
    """A server-created retrieval request for one practice group or doubt turn."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)

    request_id: str = Field(alias="requestId", min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=8000)
    subject: str = Field(min_length=1, max_length=64)
    mode: PatternRuntimeMode = PatternRuntimeMode.PRACTICE
    topic: str | None = Field(default=None, max_length=256)
    category: str | None = Field(default=None, max_length=128)
    difficulty: str | None = Field(default=None, max_length=64)
    exam_ids: tuple[str, ...] = Field(default_factory=tuple, alias="examIds", max_length=12)
    question_type: str | None = Field(default=None, alias="questionType", max_length=64)
    pattern_family_id: str | None = Field(
        default=None,
        alias="patternFamilyId",
        max_length=256,
    )
    excluded_conditions: tuple[str, ...] = Field(
        default_factory=tuple,
        alias="excludedConditions",
        max_length=8,
    )
    required_operation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        alias="requiredOperationIds",
        max_length=8,
    )
    required_target_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        alias="requiredTargetIds",
        max_length=8,
    )
    server_pattern_id: str | None = Field(
        default=None,
        alias="serverPatternId",
        max_length=256,
    )
    server_pattern_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        alias="serverPatternIds",
        max_length=24,
    )
    server_pattern_version_hash: str | None = Field(
        default=None,
        alias="serverPatternVersionHash",
        max_length=256,
    )
    candidate_limit: int = Field(default=12, ge=1, le=24, alias="candidateLimit")
    # One demand group may need several directly reusable questions. The default of
    # one preserves the original single-reuse contract for every existing caller.
    reuse_target: int = Field(default=1, ge=1, le=100, alias="reuseTarget")
    max_linked_questions: int = Field(
        default=1,
        ge=0,
        le=2,
        alias="maxLinkedQuestions",
    )
    language: str | None = Field(default=None, max_length=32)
    excluded_question_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        alias="excludedQuestionIds",
        max_length=100,
    )
    seen_question_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        alias="seenQuestionIds",
        max_length=100,
    )
    student_history_checked: bool = Field(
        default=False,
        alias="studentHistoryChecked",
    )


class PatternRetrievalPlan(BaseModel):
    """Deterministic plan that makes discovery and authority separate."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    mode: PatternRuntimeMode
    strategy: str = Field(pattern="^(exact|vector)$")
    candidate_limit: int = Field(alias="candidateLimit", ge=1, le=24)
    pattern_ids: tuple[str, ...] = Field(default_factory=tuple, alias="patternIds")


class CanonicalPatternRecord(BaseModel):
    """Whitelisted representation of the canonical Pattern record.

    `PatternQuestion` records are intentionally not converted into student-playable
    questions here. Their relationship is provenance-only until a verified QuestionBank
    mapping is supplied by the owning service.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)

    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    status: str = Field(min_length=1, max_length=64)
    subject: str = Field(min_length=1, max_length=64)
    topic: str = Field(default="", max_length=256)
    core_concept: str = Field(default="", alias="coreConcept", max_length=256)
    complexity_level: str = Field(default="", alias="complexityLevel", max_length=64)
    exams: tuple[str, ...] = Field(default_factory=tuple)
    category: str = Field(default="", max_length=128)
    question_type: str = Field(default="", alias="questionType", max_length=64)
    pattern_family_id: str = Field(default="", alias="patternFamilyId", max_length=256)
    pattern_graph: dict[str, Any] = Field(default_factory=dict, alias="patternGraph")
    solve_flow: dict[str, Any] = Field(default_factory=dict, alias="solveFlow")
    solve_flow_status: str = Field(default="", alias="solveFlowStatus", max_length=64)
    solve_flow_meta: dict[str, Any] = Field(default_factory=dict, alias="solveFlowMeta")
    current_version_hash: str = Field(
        default="",
        alias="kbCurrentVersionHash",
        max_length=256,
    )

    @classmethod
    def from_raw(
        cls,
        value: CanonicalPatternRecord | Mapping[str, Any],
    ) -> CanonicalPatternRecord | None:
        if isinstance(value, cls):
            return value
        pattern_graph = _mapping(value.get("patternGraph"))
        taxonomy = _mapping(value.get("patternTaxonomy")) or _mapping(
            pattern_graph.get("patternTaxonomy")
        )
        family = (
            _family_id(value.get("patternFamily"))
            or _family_id(pattern_graph.get("patternFamily"))
            or _family_id(taxonomy)
        )
        try:
            return cls(
                patternId=_text(value.get("patternId")),
                status=_text(value.get("status")),
                subject=_text(value.get("subject")),
                topic=_text(value.get("topic")),
                coreConcept=_text(value.get("coreConcept")),
                complexityLevel=_metadata_text(value.get("complexityLevel")),
                exams=tuple(_texts(value.get("exams"))),
                category=_text(value.get("category")),
                questionType=_text(value.get("questionType")),
                patternFamilyId=family,
                patternGraph=pattern_graph,
                solveFlow=_mapping(value.get("solveFlow")),
                solveFlowStatus=_text(value.get("solveFlowStatus")),
                solveFlowMeta=_mapping(value.get("solveFlowMeta")),
                kbCurrentVersionHash=_text(value.get("kbCurrentVersionHash")),
            )
        except ValidationError:
            return None


class CanonicalPatternQuestion(BaseModel):
    """Whitelisted linked PatternQuestion provenance record.

    Correct-answer and solution fields are deliberately absent from this schema.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)

    question_id: str = Field(alias="questionId", min_length=1, max_length=256)
    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    question_text: str = Field(alias="questionText", min_length=1, max_length=12000)
    options: tuple[str, ...] = Field(default_factory=tuple)
    status: str = Field(default="", max_length=64)
    stage: str = Field(default="", max_length=64)
    answer_contract_status: str = Field(
        default="",
        alias="answerContractStatus",
        max_length=64,
    )
    solution_status: str = Field(default="", alias="solutionStatus", max_length=64)

    @property
    def is_verified_reference(self) -> bool:
        return (
            self.status.strip().casefold() == "completed"
            and self.stage.strip().casefold() == "save_completed"
            and self.answer_contract_status.strip().casefold() == "resolved"
            and self.solution_status.strip().casefold() == "verified"
        )

    @classmethod
    def from_raw(
        cls,
        value: Mapping[str, Any],
    ) -> CanonicalPatternQuestion | None:
        try:
            return cls(
                questionId=_text(value.get("questionId")),
                patternId=_text(value.get("patternId")),
                questionText=_text(value.get("questionText")),
                options=tuple(_option_texts(value.get("options"))),
                status=_text(value.get("status")),
                stage=_text(value.get("stage")),
                answerContractStatus=_text(value.get("answerContractStatus")),
                solutionStatus=_text(value.get("solutionStatus")),
            )
        except ValidationError:
            return None


class CanonicalPlayableQuestion(BaseModel):
    """Internal authoritative QuestionBank row; never rendered into Pattern prompts."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore", frozen=True)

    question_id: str = Field(alias="qbId", min_length=1, max_length=256)
    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    pattern_version_hash: str = Field(
        alias="patternVersionHash",
        min_length=1,
        max_length=256,
    )
    pattern_link_evidence: str = Field(alias="patternLinkEvidence", max_length=64)
    question: str = Field(min_length=1, max_length=12000)
    options: tuple[str, ...] = Field(min_length=2, max_length=8)
    correct_answer: str = Field(alias="correctAnswer", min_length=1, max_length=4000)
    explanation: str = Field(default="", max_length=12000)
    category: str = Field(min_length=1, max_length=128)
    difficulty: str = Field(min_length=1, max_length=64)
    subject: str = Field(min_length=1, max_length=64)
    topic: str = Field(min_length=1, max_length=256)
    question_type: str = Field(alias="questionType", min_length=1, max_length=64)
    language: str = Field(min_length=1, max_length=32)
    exam_ids: tuple[str, ...] = Field(default_factory=tuple, alias="examIds")
    pattern_family_id: str = Field(default="", alias="patternFamilyId", max_length=256)
    updated_at: str = Field(default="", alias="updatedAt", max_length=64)

    @classmethod
    def from_raw(cls, value: Mapping[str, Any]) -> CanonicalPlayableQuestion | None:
        meta = _mapping(value.get("meta"))
        if (
            _text(meta.get("status")).upper() != "ACTIVE"
            or _text(meta.get("qualityStatus")).upper() != "VERIFIED"
            or meta.get("reusable") is not True
            or _text(meta.get("visibility")).upper() not in {"PLATFORM", "PUBLIC"}
            or meta.get("needsReview") is True
        ):
            return None
        options = _question_bank_options(value.get("answers"))
        correct_answer = _text(value.get("correctAnswer"))
        if correct_answer not in options:
            return None
        try:
            return cls(
                qbId=_text(value.get("qbId")),
                patternId=_text(value.get("patternId")),
                patternVersionHash=_text(value.get("patternVersionHash")),
                patternLinkEvidence=_text(value.get("patternLinkEvidence")),
                question=_text(value.get("question")),
                options=tuple(options),
                correctAnswer=correct_answer,
                explanation=_text(value.get("explanation")),
                category=_text(value.get("category")),
                difficulty=_text(value.get("difficulty")),
                subject=_text(meta.get("subject")),
                topic=_text(meta.get("topic")),
                questionType=_text(meta.get("questionType")),
                language=_text(meta.get("language")),
                examIds=tuple(_texts(meta.get("examIds"))),
                patternFamilyId=_text(meta.get("patternFamilyId")),
                updatedAt=_text(value.get("updatedAt")),
            )
        except ValidationError:
            return None


class PatternGenerationContext(BaseModel):
    """Answer-redacted PatternGraph guidance for a generator prompt."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    target: tuple[str, ...] = Field(default_factory=tuple)
    givens: tuple[str, ...] = Field(default_factory=tuple)
    conditions: tuple[str, ...] = Field(default_factory=tuple)
    operation_hints: tuple[str, ...] = Field(default_factory=tuple, alias="operationHints")
    not_same_when: tuple[str, ...] = Field(default_factory=tuple, alias="notSameWhen")
    trap_cues: tuple[str, ...] = Field(default_factory=tuple, alias="trapCues")
    complexity_level: str = Field(default="", alias="complexityLevel", max_length=64)
    variation_focus: str = Field(default="", alias="variationFocus", max_length=256)


class DoubtQuestionReference(BaseModel):
    """A bounded, answer-redacted linked question reference for a doubt prompt."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    question_id: str = Field(alias="questionId", min_length=1, max_length=256)
    question_text: str = Field(alias="questionText", min_length=1, max_length=4000)
    options: tuple[str, ...] = Field(default_factory=tuple)


class DoubtSolveFlowStep(BaseModel):
    """One replay-safe, answer-redacted SolveFlow instruction for a doubt prompt."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    action: str = Field(min_length=1, max_length=80)
    target: str = Field(min_length=1, max_length=80)
    instruction: str = Field(min_length=1, max_length=160)


class DoubtPatternContext(PatternGenerationContext):
    """Compact PatternGraph guidance plus vetted linked-question references."""

    question_references: tuple[DoubtQuestionReference, ...] = Field(
        default_factory=tuple,
        alias="questionReferences",
        max_length=2,
    )
    solve_flow_steps: tuple[DoubtSolveFlowStep, ...] = Field(
        default_factory=tuple,
        alias="solveFlowSteps",
        max_length=6,
    )


class PatternMatchDecision(BaseModel):
    """Auditable, non-sensitive outcome for one authoritative Pattern record."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    pattern_id: str = Field(alias="patternId", min_length=1, max_length=256)
    tier: PatternMatchTier
    reason: str = Field(min_length=1, max_length=96)


class PatternRuntimeResult(BaseModel):
    """Trusted result that callers may render only at their prompt boundary."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)

    plan: PatternRetrievalPlan
    tier: PatternMatchTier = PatternMatchTier.IGNORE
    selected_pattern_id: str | None = Field(default=None, alias="selectedPatternId", max_length=256)
    selected_pattern_version_hash: str | None = Field(
        default=None,
        alias="selectedPatternVersionHash",
        max_length=256,
    )
    generation_context: PatternGenerationContext | None = Field(
        default=None,
        alias="generationContext",
    )
    doubt_context: DoubtPatternContext | None = Field(default=None, alias="doubtContext")
    reuse_questions: tuple[CanonicalPlayableQuestion, ...] = Field(
        default_factory=tuple,
        alias="reuseQuestions",
        max_length=100,
    )
    decisions: tuple[PatternMatchDecision, ...] = Field(default_factory=tuple, max_length=24)
    warnings: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    rerank_used: bool = Field(default=False, alias="rerankUsed")
    candidate_expansion_used: bool = Field(default=False, alias="candidateExpansionUsed")

    @property
    def reuse_question(self) -> CanonicalPlayableQuestion | None:
        """Keep the original single-reuse read contract available to callers."""
        return self.reuse_questions[0] if self.reuse_questions else None


@dataclass(slots=True)
class PatternRuntimeMemo:
    """Request-local memo supplied by the caller; never shared across requests."""

    candidate_ids: dict[tuple[str, str, int], tuple[str, ...]] = field(default_factory=dict)
    candidate_versions: dict[tuple[str, str, int], dict[str, str]] = field(
        default_factory=dict
    )
    # Diagnostics only: raw vector-search yield before the confidence floor, and
    # its top score, so callers can distinguish "nothing indexed" from "indexed
    # but below threshold" without changing what is selected or hydrated.
    raw_candidate_stats: dict[tuple[str, str, int], tuple[int, float | None]] = field(
        default_factory=dict
    )
    patterns: dict[str, CanonicalPatternRecord | None] = field(default_factory=dict)
    playable_questions: dict[str, tuple[CanonicalPlayableQuestion, ...]] = field(
        default_factory=dict
    )


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _metadata_text(value: object) -> str:
    """Keep canonical scalar metadata without inventing a semantic conversion."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def _texts(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple, set)):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _family_id(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, Mapping):
        return ""
    for key in ("familyKey", "key", "patternFamilyId", "id"):
        text = _text(value.get(key))
        if text:
            return text
    return ""


def _option_texts(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        value = decoded
    if not isinstance(value, (list, tuple)):
        return []
    texts: list[str] = []
    for option in value:
        if isinstance(option, str) and option.strip():
            texts.append(option.strip()[:500])
            continue
        if not isinstance(option, Mapping):
            continue
        for key in ("text", "optionText", "value", "label"):
            text = _text(option.get(key))
            if text:
                texts.append(text[:500])
                break
    return texts[:8]


def _question_bank_options(value: object) -> list[str]:
    decoded: object = value
    for _ in range(2):
        if not isinstance(decoded, str):
            break
        try:
            decoded = json.loads(decoded)
        except json.JSONDecodeError:
            return []
    if isinstance(decoded, Mapping):
        decoded = decoded.get("options")
    if not isinstance(decoded, (list, tuple)):
        return []
    return [item.strip() for item in decoded if isinstance(item, str) and item.strip()][:8]
