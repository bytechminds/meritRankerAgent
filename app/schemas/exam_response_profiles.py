"""Validated schemas for compact exam-response category guidance."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ID_SEPARATOR = re.compile(r"[^A-Z0-9]+")
_INSTRUCTION_PREFIX = "EXAM RESPONSE GUIDANCE\n"

ResponseCategory = Literal[
    "BASIC_OBJECTIVE",
    "STANDARD_OBJECTIVE",
    "ANALYTICAL_OBJECTIVE",
    "CONCEPTUAL_OBJECTIVE",
    "DESCRIPTIVE_ANALYTICAL",
    "PEDAGOGY_CONCEPTUAL",
    "TECHNICAL_PROBLEM_SOLVING",
    "ADVANCED_APTITUDE",
]
ResolutionProvenance = Literal["exam_stage", "exam", "family", "default"]


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


def normalize_profile_id(value: str) -> str:
    """Normalize configured and runtime exam identifiers deterministically."""
    return _ID_SEPARATOR.sub("_", value.strip().upper()).strip("_")


def build_compact_instruction(parts: list[str]) -> str:
    """Join unique resolved clauses into one model-facing guidance block."""
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = " ".join(part.split())
        key = clean.casefold().rstrip(".")
        if clean and key not in seen:
            unique.append(clean)
            seen.add(key)
    return _INSTRUCTION_PREFIX + " ".join(unique)


class _AliasedModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )


class ExamResponseProfileLimits(_AliasedModel):
    """Character budgets for category, override, and resolved guidance."""

    max_category_guide_chars: int = Field(gt=0, le=500)
    max_append_guide_chars: int = Field(gt=0, le=250)
    max_resolved_context_chars: int = Field(gt=0, le=1000)


class ExamResponseCategoryGuide(BaseModel):
    """One reusable presentation guide."""

    model_config = ConfigDict(extra="forbid")

    guide: str = Field(min_length=1)

    @field_validator("guide")
    @classmethod
    def _normalize_guide(cls, value: str) -> str:
        clean = " ".join(value.split())
        if not clean:
            raise ValueError("Category guide must not be blank.")
        return clean


class ExamResponseFamilyMapping(_AliasedModel):
    """Family default plus its canonical exam membership."""

    category: ResponseCategory
    exams: list[str] = Field(default_factory=list)


class ExamResponseStageOverride(_AliasedModel):
    """Sparse stage-specific category or append-only adjustment."""

    category: ResponseCategory | None = None
    append_guide: str | None = None

    @field_validator("append_guide")
    @classmethod
    def _normalize_append_guide(cls, value: str | None) -> str | None:
        if value is None:
            return None
        clean = " ".join(value.split())
        if not clean:
            raise ValueError("Stage appendGuide must not be blank.")
        return clean

    @model_validator(mode="after")
    def _require_adjustment(self) -> ExamResponseStageOverride:
        if self.category is None and self.append_guide is None:
            raise ValueError("Stage override must define category or appendGuide.")
        return self


class ExamResponseExamMapping(_AliasedModel):
    """Sparse exam-level category and stage exceptions."""

    category: ResponseCategory | None = None
    append_guide: str | None = None
    stage_overrides: dict[str, ExamResponseStageOverride] = Field(default_factory=dict)

    @field_validator("append_guide")
    @classmethod
    def _normalize_append_guide(cls, value: str | None) -> str | None:
        if value is None:
            return None
        clean = " ".join(value.split())
        if not clean:
            raise ValueError("Exam appendGuide must not be blank.")
        return clean

    @model_validator(mode="after")
    def _require_adjustment(self) -> ExamResponseExamMapping:
        if self.category is None and self.append_guide is None and not self.stage_overrides:
            raise ValueError("Exam mapping must define category, appendGuide, or stageOverrides.")
        return self


class ExamResponseProfilesConfig(_AliasedModel):
    """Versioned static exam-response category configuration."""

    schema_version: int = Field(ge=2)
    default_category: ResponseCategory
    limits: ExamResponseProfileLimits
    categories: dict[ResponseCategory, ExamResponseCategoryGuide] = Field(min_length=1)
    family_mappings: dict[str, ExamResponseFamilyMapping] = Field(min_length=1)
    exam_mappings: dict[str, ExamResponseExamMapping] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_catalog(self) -> ExamResponseProfilesConfig:
        self.family_mappings = self._normalize_named_models(
            self.family_mappings,
            "family",
        )
        self.exam_mappings = self._normalize_named_models(
            self.exam_mappings,
            "exam mapping",
        )
        self.aliases = {
            normalize_profile_id(alias): normalize_profile_id(target)
            for alias, target in self.aliases.items()
        }

        if self.default_category not in self.categories:
            raise ValueError("defaultCategory must reference a configured category.")

        canonical_to_family: dict[str, str] = {}
        for family, mapping in self.family_mappings.items():
            if mapping.category not in self.categories:
                raise ValueError(f"Family {family!r} references an unconfigured category.")
            normalized_exams: list[str] = []
            for raw_exam in mapping.exams:
                exam = normalize_profile_id(raw_exam)
                if not exam:
                    raise ValueError("Mapped exam identifiers must not be blank.")
                existing = canonical_to_family.get(exam)
                if existing is not None:
                    raise ValueError(
                        f"Canonical exam {exam!r} is mapped to both {existing!r} and {family!r}."
                    )
                canonical_to_family[exam] = family
                normalized_exams.append(exam)
            mapping.exams = normalized_exams

        normalized_exam_mappings: dict[str, ExamResponseExamMapping] = {}
        for raw_exam, mapping in self.exam_mappings.items():
            exam = normalize_profile_id(raw_exam)
            if not exam:
                raise ValueError("Exam mapping identifiers must not be blank.")
            if exam in normalized_exam_mappings:
                raise ValueError(f"Duplicate normalized exam mapping {exam!r}.")
            if exam not in canonical_to_family:
                raise ValueError(f"examMappings references unknown exam {exam!r}.")
            if mapping.category is not None and mapping.category not in self.categories:
                raise ValueError(f"Exam mapping {exam!r} references an unconfigured category.")
            normalized_stages: dict[str, ExamResponseStageOverride] = {}
            for raw_stage, override in mapping.stage_overrides.items():
                stage = normalize_profile_id(raw_stage)
                if not stage:
                    raise ValueError("Stage identifiers must not be blank.")
                if stage in normalized_stages:
                    raise ValueError(f"Duplicate normalized stage override {exam!r}/{stage!r}.")
                if override.category is not None and override.category not in self.categories:
                    raise ValueError(
                        f"Stage override {exam!r}/{stage!r} references an unconfigured category."
                    )
                self._validate_append_guide(
                    override.append_guide,
                    f"Stage appendGuide for {exam!r}/{stage!r}",
                )
                normalized_stages[stage] = override
            mapping.stage_overrides = normalized_stages
            self._validate_append_guide(
                mapping.append_guide,
                f"Exam appendGuide for {exam!r}",
            )
            normalized_exam_mappings[exam] = mapping
        self.exam_mappings = normalized_exam_mappings

        for alias in self.aliases:
            if not alias:
                raise ValueError("Alias identifiers must not be blank.")
            if alias in canonical_to_family:
                raise ValueError(f"Alias {alias!r} conflicts with a canonical exam.")
            visited: set[str] = set()
            current = alias
            while current in self.aliases:
                if current in visited:
                    raise ValueError(f"Alias cycle detected at {current!r}.")
                visited.add(current)
                current = self.aliases[current]
            if current not in canonical_to_family:
                raise ValueError(f"Alias {alias!r} resolves to unknown canonical exam {current!r}.")

        for category, profile in self.categories.items():
            if len(profile.guide) > self.limits.max_category_guide_chars:
                raise ValueError(f"Category guide for {category!r} exceeds its character limit.")

        for exam in canonical_to_family:
            mapping = self.exam_mappings.get(exam)
            stages: list[str | None] = [None]
            if mapping is not None:
                stages.extend(mapping.stage_overrides)
            for stage in stages:
                category, append_guide = self.resolve_configured_values(
                    exam,
                    stage,
                    canonical_to_family,
                )
                instruction = build_compact_instruction(
                    [
                        self.categories[category].guide,
                        *([append_guide] if append_guide else []),
                    ]
                )
                if len(instruction) > self.limits.max_resolved_context_chars:
                    raise ValueError(
                        f"Resolved context for {exam!r}/{stage or 'default'} exceeds "
                        "maxResolvedContextChars."
                    )
        return self

    def resolve_configured_values(
        self,
        exam: str,
        stage: str | None,
        canonical_to_family: dict[str, str],
    ) -> tuple[ResponseCategory, str | None]:
        """Resolve category and at most one append guide for a canonical exam."""
        family = canonical_to_family[exam]
        family_category = self.family_mappings[family].category
        mapping = self.exam_mappings.get(exam)
        stage_mapping = (
            mapping.stage_overrides.get(stage)
            if mapping is not None and stage is not None
            else None
        )
        category = (
            stage_mapping.category
            if stage_mapping is not None and stage_mapping.category is not None
            else mapping.category
            if mapping is not None and mapping.category is not None
            else family_category
        )
        append_guide = (
            stage_mapping.append_guide
            if stage_mapping is not None and stage_mapping.append_guide is not None
            else mapping.append_guide
            if mapping is not None
            else None
        )
        return category, append_guide

    @staticmethod
    def _normalize_named_models(
        values: dict[str, object],
        label: str,
    ) -> dict[str, object]:
        normalized: dict[str, object] = {}
        for raw_name, value in values.items():
            name = normalize_profile_id(raw_name)
            if not name:
                raise ValueError(f"{label.capitalize()} identifiers must not be blank.")
            if name in normalized:
                raise ValueError(f"Duplicate normalized {label} {name!r}.")
            normalized[name] = value
        return normalized

    def _validate_append_guide(self, value: str | None, label: str) -> None:
        if value is not None and len(value) > self.limits.max_append_guide_chars:
            raise ValueError(f"{label} exceeds its character limit.")


class ResolvedExamResponseProfile(BaseModel):
    """Safe deterministic output used only by answer prompt composition."""

    model_config = ConfigDict(frozen=True)

    exam_id: str | None
    canonical_exam_id: str | None
    exam_family: str | None
    stage: str | None
    response_category: ResponseCategory
    category_guide: str
    optional_override: str | None
    compact_instruction: str
    provenance: ResolutionProvenance
    alias_applied: bool
    fallback_used: bool
    source: Literal["canonical", "alias", "missing_fallback", "unknown_fallback"]
    schema_version: int
