"""Deterministic reusable-question eligibility, ranking, and deficit logic."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from features.practice_generation.metadata_normalization import (
    CanonicalQuestionMetadata,
    normalize_category,
    normalize_difficulty,
    normalize_exam_ids,
    normalize_language,
    normalize_pattern_family,
    normalize_question_metadata,
    normalize_subject,
    normalize_topic,
)
from features.practice_generation.question_contract import validate_playable_question
from features.practice_generation.schemas import DemandBucket, PlannerSlot, PracticeBlueprint

MIN_DECLARED_REUSE_CONFIDENCE = 0.9


def normalize_question_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold())).strip()


def normalize_reuse_key_segment(value: str) -> str:
    """Normalize one reuse-key segment under contract version 1."""
    normalized = re.sub(r"\s+", " ", value.strip()).lower()
    return quote(normalized, safe="-_.!~*'()")


def build_reuse_bucket_key(
    *,
    category: str,
    topic: str,
    question_type: str,
    language: str,
) -> str:
    parts = (
        "v1",
        normalize_reuse_key_segment(category),
        normalize_reuse_key_segment(topic),
        normalize_reuse_key_segment(question_type),
        normalize_reuse_key_segment(language),
    )
    if any(not part for part in parts):
        raise ValueError("Reuse bucket key contains an empty normalized segment.")
    return "#".join(parts)


def build_slot_reuse_bucket_key(slot: PlannerSlot, *, language: str) -> str | None:
    """Build the exact sparse-index key for one canonical planner slot."""
    category = normalize_category(slot.category_id)
    topic = normalize_topic(slot.topic_id)
    normalized_language = normalize_language(language)
    if category is None or topic is None or normalized_language is None:
        return None
    return build_reuse_bucket_key(
        category=category,
        topic=topic,
        question_type=slot.question_type.value,
        language=normalized_language,
    )


def build_reuse_difficulty_prefix(difficulty: object) -> str | None:
    """Build the reuseSortKey prefix owned by QuestionBank persistence v1."""
    normalized = normalize_difficulty(difficulty)
    backend_value = {
        "basic": "easy",
        "intermediate": "medium",
        "advanced": "hard",
    }.get(normalized or "")
    return f"v1#{backend_value}#" if backend_value else None


def _parse_meta(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


@dataclass(frozen=True)
class ReusableQuestion:
    question_id: str
    question: str
    options: tuple[str, ...]
    correct_answer: str
    solution: str
    subject: str
    topic: str
    difficulty: str
    question_type: str
    language: str
    source: str
    source_updated_at: str | None
    category: str = ""
    exam_ids: tuple[str, ...] = ()
    pattern_family_id: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class BucketMatch:
    bucket: DemandBucket
    selected: tuple[ReusableQuestion, ...]
    deficit: int


@dataclass(frozen=True)
class SlotMatch:
    slot: PlannerSlot
    selected: ReusableQuestion | None

    @property
    def deficit(self) -> int:
        return int(self.selected is None)


def _confidence(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return -1.0
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return -1.0
    return parsed if 0.0 <= parsed <= 1.0 else -1.0


def _declared_confidence_is_eligible(value: object) -> bool:
    parsed = _confidence(value)
    return parsed is None or parsed >= MIN_DECLARED_REUSE_CONFIDENCE


def _difficulty_matches(candidate: str, bucket: DemandBucket) -> bool:
    normalized = candidate.casefold()
    return normalized in {
        bucket.difficulty.value,
        {"basic": "easy", "intermediate": "medium", "advanced": "hard"}[
            bucket.difficulty.value
        ],
    }


def shortlist_reuse_candidate_ids(
    blueprint: PracticeBlueprint,
    items: tuple[dict[str, Any], ...],
    *,
    requested_language: str,
    already_linked_ids: set[str],
) -> list[str]:
    """Rank projected GSI metadata and return only IDs requiring full reads."""
    selected: list[str] = []
    used = set(already_linked_ids)
    for bucket in blueprint.buckets:
        ranked: list[tuple[int, str]] = []
        for item in items:
            question_id = str(item.get("qbId") or "")
            meta = _parse_meta(item.get("meta"))
            if (
                not question_id
                or question_id in used
                or str(meta.get("status") or "").upper() != "ACTIVE"
                or str(meta.get("qualityStatus") or "").upper() != "VERIFIED"
                or meta.get("reusable") is not True
                or meta.get("needsReview") is True
                or str(meta.get("visibility") or "PLATFORM").upper() not in {"PLATFORM", "PUBLIC"}
                or str(meta.get("language") or "english").casefold()
                != requested_language.casefold()
                or str(meta.get("subject") or item.get("category") or "").casefold()
                != bucket.subject.casefold()
                or str(meta.get("topic") or "").casefold() != bucket.topic.casefold()
                or str(meta.get("questionType") or "mcq").casefold() != bucket.question_type.value
                or not _difficulty_matches(str(item.get("difficulty") or ""), bucket)
                or (bucket.solution_required and meta.get("solutionAvailable") is False)
                or meta.get("answerAvailable") is False
            ):
                continue
            score = 0
            if str(meta.get("topic") or "").casefold() == bucket.topic.casefold():
                score += 8
            score += 4
            tags = {str(tag).casefold() for tag in list(item.get("tags") or []) if str(tag).strip()}
            score += min(
                sum(1 for keyword in bucket.keywords if keyword.casefold() in tags),
                3,
            )
            ranked.append((-score, question_id))
        ranked.sort()
        for _, question_id in ranked[: bucket.required_count * 2]:
            if question_id not in used:
                selected.append(question_id)
                used.add(question_id)
    return selected


def reusable_question_from_item(
    item: dict[str, Any],
    *,
    requested_language: str,
) -> ReusableQuestion | None:
    meta = _parse_meta(item.get("meta"))
    if str(meta.get("status", "")).upper() != "ACTIVE":
        return None
    if str(meta.get("qualityStatus", "")).upper() != "VERIFIED":
        return None
    if meta.get("reusable") is not True or meta.get("needsReview") is True:
        return None
    visibility = str(meta.get("visibility", "PLATFORM")).upper()
    if visibility not in {"PLATFORM", "PUBLIC"}:
        return None
    language = normalize_language(meta.get("language", "english"))
    if language is None or language != normalize_language(requested_language):
        return None
    canonical = normalize_question_metadata(
        subject=meta.get("subject") or item.get("category"),
        topic=meta.get("topic"),
        category=item.get("category"),
        difficulty=item.get("difficulty"),
        exam_ids=meta.get("examIds", []),
        language=meta.get("language", "english"),
        pattern_family_id=meta.get("patternFamilyId"),
    )
    exam_ids = normalize_exam_ids(meta.get("examIds", []))
    declared_pattern = meta.get("patternFamilyId")
    pattern_family_id = (
        normalize_pattern_family(declared_pattern)
        if declared_pattern not in {None, ""}
        else None
    )
    confidence = _confidence(meta.get("confidence"))
    if (
        exam_ids is None
        or (declared_pattern not in {None, ""} and pattern_family_id is None)
        or not _declared_confidence_is_eligible(meta.get("confidence"))
    ):
        return None
    question = str(item.get("question") or "").strip()
    correct_answer = str(item.get("correctAnswer") or "").strip()
    solution = str(item.get("explanation") or "").strip()
    if not question or not correct_answer:
        return None
    raw_answers = item.get("answers")
    if isinstance(raw_answers, str):
        try:
            raw_answers = json.loads(raw_answers)
        except json.JSONDecodeError:
            raw_answers = {}
    options_value = raw_answers.get("options", []) if isinstance(raw_answers, dict) else []
    options = tuple(str(option).strip() for option in options_value if str(option).strip())
    question_type = str(meta.get("questionType", "mcq")).casefold()
    contract = validate_playable_question(
        question_type=question_type,
        question=question,
        options=options,
        correct_answer=correct_answer,
        solution=solution,
        solution_required=False,
    )
    if not contract.valid:
        return None
    question_id = str(item.get("qbId") or "").strip()
    if not question_id:
        return None
    return ReusableQuestion(
        question_id=question_id,
        question=question,
        options=options,
        correct_answer=correct_answer,
        solution=solution,
        subject=str(meta.get("subject") or item.get("category") or "").casefold(),
        topic=str(meta.get("topic") or "").casefold(),
        difficulty=str(item.get("difficulty") or "").casefold(),
        question_type=question_type,
        language=language,
        source=str(item.get("source") or "QuestionBank"),
        source_updated_at=(str(item["updatedAt"]) if item.get("updatedAt") is not None else None),
        category=canonical.category if canonical is not None else "",
        exam_ids=exam_ids,
        pattern_family_id=pattern_family_id,
        confidence=confidence,
    )


def _slot_metadata(slot: PlannerSlot, *, language: str) -> CanonicalQuestionMetadata | None:
    return normalize_question_metadata(
        subject=slot.subject_id,
        topic=slot.topic_id,
        category=slot.category_id,
        difficulty=slot.difficulty.value,
        exam_ids=slot.exam_ids,
        language=language,
        pattern_family_id=slot.pattern_family_id,
    )


def _candidate_matches_slot(
    candidate: ReusableQuestion,
    slot: PlannerSlot,
    *,
    requested_language: str,
) -> bool:
    expected = _slot_metadata(slot, language=requested_language)
    candidate_exams = normalize_exam_ids(candidate.exam_ids)
    candidate_pattern = (
        normalize_pattern_family(candidate.pattern_family_id)
        if candidate.pattern_family_id
        else None
    )
    if expected is None or candidate_exams is None:
        return False
    if candidate.confidence is None or candidate.confidence < MIN_DECLARED_REUSE_CONFIDENCE:
        return False
    if (
        normalize_subject(candidate.subject) != expected.subject
        or normalize_topic(candidate.topic) != expected.topic
        or normalize_category(candidate.category) != expected.category
        or normalize_difficulty(candidate.difficulty) != expected.difficulty
        or normalize_language(candidate.language) != expected.language
        or candidate.question_type != slot.question_type.value
    ):
        return False
    if expected.exam_ids and not set(expected.exam_ids).issubset(candidate_exams):
        return False
    if expected.pattern_family_id is not None and candidate_pattern != expected.pattern_family_id:
        return False
    return bool(candidate.solution)


def match_existing_questions_to_slots(
    blueprint: PracticeBlueprint,
    candidates: list[ReusableQuestion],
    *,
    requested_language: str,
    already_linked_ids: set[str] | None = None,
) -> tuple[SlotMatch, ...]:
    """Assign exact candidates one-to-one to schema-v2 slots deterministically."""
    if blueprint.schema_version != "2":
        raise ValueError("slot matching requires a schema-v2 blueprint")
    used = set(already_linked_ids or ())
    by_id = {
        candidate.question_id: candidate
        for candidate in candidates
        if candidate.question_id and candidate.question_id not in used
    }
    matches: list[SlotMatch] = []
    for slot in blueprint.slots:
        eligible = [
            candidate
            for candidate in by_id.values()
            if candidate.question_id not in used
            and _candidate_matches_slot(
                candidate,
                slot,
                requested_language=requested_language,
            )
        ]
        eligible.sort(
            key=lambda candidate: (
                -(candidate.confidence if candidate.confidence is not None else 0.0),
                candidate.question_id,
            )
        )
        selected = eligible[0] if eligible else None
        if selected is not None:
            used.add(selected.question_id)
        matches.append(SlotMatch(slot=slot, selected=selected))
    return tuple(matches)


def _candidate_score(candidate: ReusableQuestion, bucket: DemandBucket) -> tuple[int, str]:
    score = 0
    if candidate.subject == bucket.subject.casefold():
        score += 8
    if candidate.topic == bucket.topic.casefold():
        score += 6
    if candidate.difficulty in {
        bucket.difficulty.value,
        {"basic": "easy", "intermediate": "medium", "advanced": "hard"}[bucket.difficulty.value],
    }:
        score += 4
    if candidate.question_type == bucket.question_type.value:
        score += 3
    normalized = normalize_question_text(candidate.question)
    keyword_hits = sum(1 for keyword in bucket.keywords if keyword.casefold() in normalized)
    score += min(keyword_hits, 3)
    return (-score, candidate.question_id)


def match_existing_questions(
    blueprint: PracticeBlueprint,
    candidates: list[ReusableQuestion],
    *,
    already_linked_ids: set[str] | None = None,
) -> tuple[BucketMatch, ...]:
    used = set(already_linked_ids or ())
    matches: list[BucketMatch] = []
    for bucket in blueprint.buckets:
        eligible_by_id = {
            candidate.question_id: candidate
            for candidate in candidates
            if candidate.question_id not in used
            and candidate.subject == bucket.subject.casefold()
            and candidate.topic == bucket.topic.casefold()
            and candidate.question_type == bucket.question_type.value
            and _difficulty_matches(candidate.difficulty, bucket)
            and (not bucket.solution_required or bool(candidate.solution))
        }
        eligible = list(eligible_by_id.values())
        eligible.sort(key=lambda candidate: _candidate_score(candidate, bucket))
        selected = tuple(eligible[: bucket.required_count])
        used.update(candidate.question_id for candidate in selected)
        matches.append(
            BucketMatch(
                bucket=bucket,
                selected=selected,
                deficit=bucket.required_count - len(selected),
            )
        )
    return tuple(matches)


def total_deficit(matches: tuple[BucketMatch, ...]) -> int:
    return sum(match.deficit for match in matches)
