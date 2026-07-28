"""Cached deterministic resolver for compact exam-response category guidance."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from schemas.exam_response_profiles import (
    ExamResponseProfilesConfig,
    ResolutionProvenance,
    ResolvedExamResponseProfile,
    ResponseCategory,
    build_compact_instruction,
    normalize_profile_id,
)

logger = logging.getLogger(__name__)

_APP_DIR = Path(__file__).resolve().parents[2]
DEFAULT_EXAM_RESPONSE_PROFILES_PATH = _APP_DIR / "config" / "exam_response_profiles.yaml"
_STAGE_ALIASES = {
    "PRELIM": "PRELIMS",
    "PRELIMINARY": "PRELIMS",
    "PRELIMINARIES": "PRELIMS",
    "MAIN": "MAINS",
}


class ExamResponseProfileConfigError(ValueError):
    """Raised when the bundled profile configuration is missing or invalid."""


class ExamResponseProfileResolver:
    """Resolve an optional exam and stage without I/O after construction."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = (config_path or DEFAULT_EXAM_RESPONSE_PROFILES_PATH).resolve()
        self._config = self._load_config(self._config_path)
        self._exam_to_family = {
            exam: family
            for family, mapping in self._config.family_mappings.items()
            for exam in mapping.exams
        }

    @property
    def config(self) -> ExamResponseProfilesConfig:
        return self._config

    def resolve(
        self,
        exam_id: str | None,
        stage: str | None,
        *,
        request_id: str = "",
    ) -> ResolvedExamResponseProfile:
        normalized_exam = normalize_profile_id(exam_id) if exam_id else None
        normalized_stage = self._normalize_stage(stage)
        alias_applied = False
        fallback_used = False

        if normalized_exam is None:
            canonical_exam = None
            family = None
            source = "missing_fallback"
            fallback_used = True
        else:
            canonical_exam = normalized_exam
            while canonical_exam in self._config.aliases:
                canonical_exam = self._config.aliases[canonical_exam]
                alias_applied = True
            family = self._exam_to_family.get(canonical_exam)
            if family is None:
                canonical_exam = None
                source = "unknown_fallback"
                fallback_used = True
            else:
                source = "alias" if alias_applied else "canonical"

        category: ResponseCategory = self._config.default_category
        optional_override: str | None = None
        provenance: ResolutionProvenance = "default"
        if canonical_exam is not None and family is not None:
            category = self._config.family_mappings[family].category
            provenance = "family"
            mapping = self._config.exam_mappings.get(canonical_exam)
            if mapping is not None:
                if mapping.category is not None:
                    category = mapping.category
                optional_override = mapping.append_guide
                if mapping.category is not None or mapping.append_guide is not None:
                    provenance = "exam"
                stage_mapping = (
                    mapping.stage_overrides.get(normalized_stage)
                    if normalized_stage is not None
                    else None
                )
                if stage_mapping is not None:
                    if stage_mapping.category is not None:
                        category = stage_mapping.category
                    if stage_mapping.append_guide is not None:
                        optional_override = stage_mapping.append_guide
                    provenance = "exam_stage"

        category_guide = self._config.categories[category].guide
        compact_instruction = build_compact_instruction(
            [
                category_guide,
                *([optional_override] if optional_override else []),
            ]
        )
        if len(compact_instruction) > self._config.limits.max_resolved_context_chars:
            raise ExamResponseProfileConfigError(
                "Resolved exam response guidance exceeds its configured character limit."
            )

        result = ResolvedExamResponseProfile(
            exam_id=normalized_exam,
            canonical_exam_id=canonical_exam,
            exam_family=family,
            stage=normalized_stage,
            response_category=category,
            category_guide=category_guide,
            optional_override=optional_override,
            compact_instruction=compact_instruction,
            provenance=provenance,
            alias_applied=alias_applied,
            fallback_used=fallback_used,
            source=source,
            schema_version=self._config.schema_version,
        )
        logger.debug(
            "exam_response_profile_resolved request_id=%s canonical_exam_id=%s "
            "exam_family=%s stage=%s response_category=%s provenance=%s "
            "alias_applied=%s optional_override_applied=%s "
            "compact_instruction_chars=%d profile_schema_version=%d fallback_used=%s",
            request_id,
            result.canonical_exam_id or "",
            result.exam_family or "",
            result.stage or "",
            result.response_category,
            result.provenance,
            result.alias_applied,
            result.optional_override is not None,
            len(result.compact_instruction),
            result.schema_version,
            result.fallback_used,
        )
        return result

    @staticmethod
    def _normalize_stage(stage: str | None) -> str | None:
        if not stage:
            return None
        normalized = normalize_profile_id(stage)
        return _STAGE_ALIASES.get(normalized, normalized) or None

    @staticmethod
    def _load_config(path: Path) -> ExamResponseProfilesConfig:
        try:
            raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ExamResponseProfileConfigError(
                f"Unable to load exam response profile configuration: {type(exc).__name__}."
            ) from exc
        if not isinstance(raw, dict):
            raise ExamResponseProfileConfigError(
                "Exam response profile configuration must be a YAML mapping."
            )
        try:
            return ExamResponseProfilesConfig.model_validate(raw)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error['loc']) or 'config'}: {error['msg']}"
                for error in exc.errors(include_input=False)[:5]
            )
            raise ExamResponseProfileConfigError(
                f"Invalid exam response profile configuration: {details}"
            ) from exc


_resolver: ExamResponseProfileResolver | None = None
_resolver_lock = threading.Lock()


def get_exam_response_profile_resolver() -> ExamResponseProfileResolver:
    """Return the process-level validated resolver singleton."""
    global _resolver  # noqa: PLW0603
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = ExamResponseProfileResolver()
    return _resolver


def reset_exam_response_profile_resolver() -> None:
    """Reset the singleton for isolated tests."""
    global _resolver  # noqa: PLW0603
    with _resolver_lock:
        _resolver = None
