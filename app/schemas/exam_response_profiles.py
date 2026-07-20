"""Validated schemas for compact exam-aware answer response profiles."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ID_SEPARATOR = re.compile(r"[^A-Z0-9]+")
_INSTRUCTION_PREFIX = "Exam response guidance: "


def _to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


def normalize_profile_id(value: str) -> str:
    """Normalize configured and runtime exam identifiers deterministically."""
    return _ID_SEPARATOR.sub("_", value.strip().upper()).strip("_")


def build_compact_instruction(parts: list[str]) -> str:
    """Join unique profile clauses without semantic rewriting."""
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        clean = " ".join(part.split())
        key = clean.casefold().rstrip(".")
        if clean and key not in seen:
            unique.append(clean)
            seen.add(key)
    return _INSTRUCTION_PREFIX + " ".join(unique)


class ExamResponseProfileLimits(BaseModel):
    """Character budgets for configuration text and resolved instructions."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    max_family_response_chars: int = Field(gt=0, le=500)
    max_exam_override_chars: int = Field(gt=0, le=500)
    max_stage_override_chars: int = Field(gt=0, le=500)
    max_resolved_context_chars: int = Field(gt=0, le=1000)


class ExamResponseFamily(BaseModel):
    """One broad exam-family presentation direction."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(min_length=1)

    @field_validator("response")
    @classmethod
    def _normalize_response(cls, value: str) -> str:
        clean = " ".join(value.split())
        if not clean:
            raise ValueError("Family response must not be blank.")
        return clean


class ExamResponseProfilesConfig(BaseModel):
    """Versioned static exam response profile configuration."""

    model_config = ConfigDict(
        alias_generator=_to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    version: int = Field(ge=1)
    fallback_family: str = Field(min_length=1)
    limits: ExamResponseProfileLimits
    families: dict[str, ExamResponseFamily] = Field(min_length=1)
    exam_mappings: dict[str, list[str]] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)
    exam_overrides: dict[str, str] = Field(default_factory=dict)
    stage_overrides: dict[str, dict[str, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_catalog(self) -> ExamResponseProfilesConfig:
        self.fallback_family = normalize_profile_id(self.fallback_family)
        normalized_families: dict[str, ExamResponseFamily] = {}
        for name, family in self.families.items():
            normalized_name = normalize_profile_id(name)
            if not normalized_name:
                raise ValueError("Family identifiers must not be blank.")
            if normalized_name in normalized_families:
                raise ValueError(f"Duplicate normalized family {normalized_name!r}.")
            normalized_families[normalized_name] = family
        self.families = normalized_families
        self.exam_mappings = {
            normalize_profile_id(family): [normalize_profile_id(exam) for exam in exams]
            for family, exams in self.exam_mappings.items()
        }
        self.aliases = {
            normalize_profile_id(alias): normalize_profile_id(target)
            for alias, target in self.aliases.items()
        }
        self.exam_overrides = {
            normalize_profile_id(exam): " ".join(text.split())
            for exam, text in self.exam_overrides.items()
        }
        self.stage_overrides = {
            normalize_profile_id(exam): {
                normalize_profile_id(stage): " ".join(text.split())
                for stage, text in stages.items()
            }
            for exam, stages in self.stage_overrides.items()
        }

        if self.fallback_family not in self.families:
            raise ValueError("fallbackFamily must reference a configured family.")

        canonical_to_family: dict[str, str] = {}
        for family, exams in self.exam_mappings.items():
            if family not in self.families:
                raise ValueError(f"examMappings references unknown family {family!r}.")
            for exam in exams:
                if not exam:
                    raise ValueError("Mapped exam identifiers must not be blank.")
                existing = canonical_to_family.get(exam)
                if existing is not None:
                    raise ValueError(
                        f"Canonical exam {exam!r} is mapped to both {existing!r} and {family!r}."
                    )
                canonical_to_family[exam] = family

        for alias in self.aliases:
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
                raise ValueError(
                    f"Alias {alias!r} resolves to unknown canonical exam {current!r}."
                )

        for family, profile in self.families.items():
            if len(profile.response) > self.limits.max_family_response_chars:
                raise ValueError(f"Family response for {family!r} exceeds its character limit.")

        for exam, text in self.exam_overrides.items():
            if exam not in canonical_to_family:
                raise ValueError(f"examOverrides references unknown exam {exam!r}.")
            self._validate_direction(
                text,
                self.limits.max_exam_override_chars,
                f"Exam override for {exam!r}",
            )

        for exam, stages in self.stage_overrides.items():
            if exam not in canonical_to_family:
                raise ValueError(f"stageOverrides references unknown exam {exam!r}.")
            if not stages:
                raise ValueError(f"Stage overrides for {exam!r} must not be empty.")
            for stage, text in stages.items():
                if not stage:
                    raise ValueError("Stage identifiers must not be blank.")
                self._validate_direction(
                    text,
                    self.limits.max_stage_override_chars,
                    f"Stage override for {exam!r}/{stage!r}",
                )

        combinations: list[tuple[str, str | None]] = [
            (exam, None) for exam in canonical_to_family
        ]
        combinations.extend(
            (exam, stage)
            for exam, stages in self.stage_overrides.items()
            for stage in stages
        )
        for exam, stage in combinations:
            family = canonical_to_family[exam]
            parts = [self.families[family].response]
            if exam in self.exam_overrides:
                parts.append(self.exam_overrides[exam])
            if stage is not None:
                parts.append(self.stage_overrides[exam][stage])
            instruction = build_compact_instruction(parts)
            if len(instruction) > self.limits.max_resolved_context_chars:
                raise ValueError(
                    f"Resolved context for {exam!r}/{stage or 'default'} exceeds "
                    "maxResolvedContextChars."
                )
        return self

    @staticmethod
    def _validate_direction(text: str, limit: int, label: str) -> None:
        if not text:
            raise ValueError(f"{label} must not be blank.")
        if len(text) > limit:
            raise ValueError(f"{label} exceeds its character limit.")


class ResolvedExamResponseProfile(BaseModel):
    """Safe deterministic output used by answer prompt composition."""

    model_config = ConfigDict(frozen=True)

    exam_id: str | None
    canonical_exam_id: str | None
    exam_family: str
    stage: str | None
    family_response: str
    exam_override: str | None
    stage_override: str | None
    compact_instruction: str
    override_applied: bool
    alias_applied: bool
    exam_override_applied: bool
    stage_override_applied: bool
    fallback_used: bool
    source: Literal["canonical", "alias", "missing_fallback", "unknown_fallback"]
    version: int
