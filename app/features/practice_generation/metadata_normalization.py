"""Canonical metadata normalization for exact practice-question reuse."""

from __future__ import annotations

import re
from dataclasses import dataclass

from features.practice_generation.schemas import PracticeType

_CANONICAL_ID = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_CANONICAL_EXAM_ID = re.compile(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*$")

_SUBJECT_ALIASES = {
    "quant": "math",
    "quantitative aptitude": "math",
    "logical reasoning": "reasoning",
    "verbal reasoning": "reasoning",
    "english grammar": "english",
}
_SUPPORTED_SUBJECTS = frozenset(
    {
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
)
_TOPIC_ALIASES = {
    "time work": "time_and_work",
    "time and work": "time_and_work",
    "work and time": "time_and_work",
}
_DIFFICULTY_ALIASES = {
    "basic": "basic",
    "easy": "basic",
    "intermediate": "intermediate",
    "medium": "intermediate",
    "advanced": "advanced",
    "hard": "advanced",
}
_LANGUAGE_ALIASES = {
    "english": "english",
    "en": "english",
    "hinglish": "hinglish",
    "hindi": "hindi",
    "hi": "hindi",
}
_EXAM_ALIASES = {
    "cat": "CAT",
    "ssc cgl": "SSC_CGL",
}
_ACTIVITY_TYPES = frozenset(member.value for member in PracticeType)


def _words(value: object) -> str:
    if not isinstance(value, str):
        return ""
    normalized = value.strip().casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


def _canonical_lower(value: object, aliases: dict[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().casefold()
    alias = aliases.get(_words(value))
    if alias is not None:
        return alias
    if _CANONICAL_ID.fullmatch(stripped):
        return stripped
    return None


def normalize_subject(value: object) -> str | None:
    normalized = _canonical_lower(value, _SUBJECT_ALIASES)
    return normalized if normalized in _SUPPORTED_SUBJECTS else None


def normalize_topic(value: object) -> str | None:
    return _canonical_lower(value, _TOPIC_ALIASES)


def normalize_category(value: object) -> str | None:
    return _canonical_lower(value, {})


def normalize_difficulty(value: object) -> str | None:
    return _DIFFICULTY_ALIASES.get(_words(value))


def normalize_exam(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip().upper()
    if _CANONICAL_EXAM_ID.fullmatch(stripped):
        return stripped
    return _EXAM_ALIASES.get(_words(value))


def normalize_language(value: object) -> str | None:
    return _LANGUAGE_ALIASES.get(_words(value))


def normalize_activity_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = "_".join(_words(value).upper().split())
    return normalized if normalized in _ACTIVITY_TYPES else None


def normalize_pattern_family(value: object) -> str | None:
    return _canonical_lower(value, {})


def normalize_exam_ids(values: object) -> tuple[str, ...] | None:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple, set, frozenset)):
        return None
    normalized = tuple(normalize_exam(value) for value in values)
    if any(value is None for value in normalized):
        return None
    return tuple(sorted(set(value for value in normalized if value is not None)))


@dataclass(frozen=True)
class CanonicalQuestionMetadata:
    subject: str
    topic: str
    category: str
    difficulty: str
    exam_ids: tuple[str, ...]
    language: str
    pattern_family_id: str | None


def normalize_question_metadata(
    *,
    subject: object,
    topic: object,
    category: object,
    difficulty: object,
    exam_ids: object,
    language: object,
    pattern_family_id: object = None,
) -> CanonicalQuestionMetadata | None:
    normalized_subject = normalize_subject(subject)
    normalized_topic = normalize_topic(topic)
    normalized_category = normalize_category(category)
    normalized_difficulty = normalize_difficulty(difficulty)
    normalized_exams = normalize_exam_ids(exam_ids)
    normalized_language = normalize_language(language)
    normalized_pattern = (
        normalize_pattern_family(pattern_family_id)
        if pattern_family_id not in {None, ""}
        else None
    )
    if (
        normalized_subject is None
        or normalized_topic is None
        or normalized_category is None
        or normalized_difficulty is None
        or normalized_exams is None
        or normalized_language is None
        or (pattern_family_id not in {None, ""} and normalized_pattern is None)
    ):
        return None
    return CanonicalQuestionMetadata(
        subject=normalized_subject,
        topic=normalized_topic,
        category=normalized_category,
        difficulty=normalized_difficulty,
        exam_ids=normalized_exams,
        language=normalized_language,
        pattern_family_id=normalized_pattern,
    )
