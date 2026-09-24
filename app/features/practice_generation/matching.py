"""Deterministic reusable-question eligibility, ranking, and deficit logic."""

from __future__ import annotations

import hashlib
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


def normalize_question_identity(value: str, options: object = None) -> str:
    """Identity of a generated MCQ candidate: its stem plus its option set.

    A generic instructional stem ("Which sentence is grammatically correct?") is
    legitimate exam construction, and the question it actually asks then lives in the
    options. Keying identity on the stem alone collapsed such items into one another.
    Options are normalized and sorted, so a reordered option set is still a duplicate.

    Reuse, PatternGraph and ingestion identity are unaffected: they keep using
    ``normalize_question_text``.
    """
    stem = normalize_question_text(value)
    if not options:
        return stem
    values: list[str] = []
    for option in options:
        if isinstance(option, dict):
            candidate = option.get("value")
        else:
            candidate = getattr(option, "value", option)
        normalized = normalize_question_text(str(candidate or ""))
        if normalized:
            values.append(normalized)
    if not values:
        return stem
    return stem + "|" + "|".join(sorted(values))


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
    """Build the exact sparse-index key for one canonical planner slot.

    The category segment is the slot's subject family, as for schema-v1 buckets. The
    planner's ``category_id`` is per-run variety, not identity: equivalent retries
    choose different values, so keying on it stranded verified inventory.
    """
    category = normalize_subject(slot.subject_id)
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


QB_IDENTITY_NAMESPACE = "qbid-v2"
QB_ID_PREFIX = "qb-v2-"


def normalize_question_for_identity(value: str) -> str:
    """Case/whitespace-only normalization: digits and symbols stay significant."""
    return " ".join(value.casefold().split())


def _identity_segment(value: str) -> str:
    """Length-prefix a segment so a separator inside content cannot forge identity."""
    normalized = normalize_question_for_identity(value)
    return f"{len(normalized)}:{normalized}"


def build_question_bank_id(
    *,
    subject: object,
    difficulty: object,
    exam_ids: object,
    language: object,
    question_type: str,
    question: str,
    options: tuple[str, ...],
    correct_answer: str,
) -> str | None:
    """Derive the Pattern-independent identity of one reusable Question scope.

    Identity is the strict compatibility scope that authorizes reuse plus the exact
    playable content.  Including the strict dimensions keeps one immutable row per
    scope, so a Question first accepted as ``basic``/``CAT`` never decides the
    classification of an independently accepted ``intermediate``/``GMAT`` occurrence.
    Topic, category, and concept are deliberately excluded: planner taxonomy wording
    varies, and bridging that variance is questions-v1's job, not durable identity's.

    Returns ``None`` when any strict dimension fails canonicalization, so an
    unclassifiable Question is never promoted under a guessed identity.
    """
    canonical_subject = normalize_subject(subject)
    canonical_difficulty = normalize_difficulty(difficulty)
    canonical_exams = normalize_exam_ids(exam_ids)
    canonical_language = normalize_language(language)
    if (
        canonical_subject is None
        or canonical_difficulty is None
        or canonical_exams is None
        or canonical_language is None
        or not question_type
    ):
        return None
    segments = (
        QB_IDENTITY_NAMESPACE,
        question_type,
        canonical_subject,
        canonical_difficulty,
        ",".join(canonical_exams),
        canonical_language,
        question,
        str(len(options)),
        *options,
        correct_answer,
    )
    digest = hashlib.sha256(
        "|".join(_identity_segment(segment) for segment in segments).encode("utf-8")
    ).hexdigest()[:32]
    return f"{QB_ID_PREFIX}{digest}"


QB_VERSION_NAMESPACE = "qbver-v1"


def build_question_bank_version_hash(
    *,
    subject: object,
    difficulty: object,
    exam_ids: object,
    language: object,
    question_type: str,
    topic: object,
    category: object,
    question: str,
    options: tuple[str, ...],
    correct_answer: str,
    solution: str,
) -> str | None:
    """Digest the current authoritative version of one reusable Question.

    Coverage is the union of three sets, which is what makes a stale questions-v1
    entry fail parity: what Phase D will embed (subject, topic, category, stem,
    options), what will become vector metadata (subject, difficulty, exam scope,
    language), and what is served to the student (correct answer, solution).

    Pattern linkage is excluded because it is neither embedded nor vector metadata
    nor student-facing, so enrichment must not invalidate an otherwise current
    vector.  ``patternFamilyId`` is excluded for the same reason: it gates reuse,
    but it is re-checked live against the row at hydration, so parity adds nothing.
    Identifiers, timestamps, derived reuse keys, and eligibility flags are excluded
    as transient or independently validated.
    """
    canonical_subject = normalize_subject(subject)
    canonical_difficulty = normalize_difficulty(difficulty)
    canonical_exams = normalize_exam_ids(exam_ids)
    canonical_language = normalize_language(language)
    canonical_topic = normalize_topic(topic)
    canonical_category = normalize_category(category)
    if (
        canonical_subject is None
        or canonical_difficulty is None
        or canonical_exams is None
        or canonical_language is None
        or canonical_topic is None
        or canonical_category is None
        or not question_type
    ):
        return None
    segments = (
        QB_VERSION_NAMESPACE,
        question_type,
        canonical_subject,
        canonical_difficulty,
        ",".join(canonical_exams),
        canonical_language,
        canonical_topic,
        canonical_category,
        question,
        str(len(options)),
        *options,
        correct_answer,
        solution,
    )
    return hashlib.sha256(
        "|".join(_identity_segment(segment) for segment in segments).encode("utf-8")
    ).hexdigest()


def _item_options(item: dict[str, Any]) -> tuple[str, ...]:
    raw_answers = item.get("answers")
    if isinstance(raw_answers, str):
        try:
            raw_answers = json.loads(raw_answers)
        except json.JSONDecodeError:
            raw_answers = {}
    values = raw_answers.get("options", []) if isinstance(raw_answers, dict) else []
    return tuple(str(option).strip() for option in values if str(option).strip())


def question_bank_version_hash_from_item(item: dict[str, Any]) -> str | None:
    """Recompute the version hash from a stored row.

    The writer and every future reader go through this one function, so a stored
    ``versionHash`` is only ever advisory: an administrative edit that forgets to
    refresh it still fails parity against an indexed vector.
    """
    meta = _parse_meta(item.get("meta"))
    return build_question_bank_version_hash(
        subject=meta.get("subject") or item.get("category"),
        difficulty=item.get("difficulty"),
        exam_ids=meta.get("examIds", []),
        language=meta.get("language"),
        question_type=str(meta.get("questionType") or "").casefold(),
        topic=meta.get("topic"),
        category=item.get("category"),
        question=str(item.get("question") or ""),
        options=_item_options(item),
        correct_answer=str(item.get("correctAnswer") or ""),
        solution=str(item.get("explanation") or ""),
    )


def question_bank_identity_is_intact(item: dict[str, Any]) -> bool:
    """Recompute the identity of a versioned row and compare it to the stored id.

    Rows created before this identity scheme cannot be recomputed, so legacy ids keep
    their existing eligibility.  Versioned rows fail closed: an in-place edit of any
    identity-bearing field leaves the stored id unreachable and the row unusable for
    reuse rather than silently serving mutated content.
    """
    qb_id = str(item.get("qbId") or "").strip()
    if not qb_id.startswith(QB_ID_PREFIX):
        return True
    meta = _parse_meta(item.get("meta"))
    expected = build_question_bank_id(
        subject=meta.get("subject") or item.get("category"),
        difficulty=item.get("difficulty"),
        exam_ids=meta.get("examIds", []),
        language=meta.get("language"),
        question_type=str(meta.get("questionType") or "").casefold(),
        question=str(item.get("question") or ""),
        options=_item_options(item),
        correct_answer=str(item.get("correctAnswer") or ""),
    )
    return expected == qb_id


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
    pattern_id: str | None = None
    pattern_version_hash: str | None = None
    pattern_link_evidence: str | None = None


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
                or normalize_language(meta.get("language"))
                != normalize_language(requested_language)
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
    language = normalize_language(meta.get("language"))
    if language is None or language != normalize_language(requested_language):
        return None
    canonical = normalize_question_metadata(
        subject=meta.get("subject") or item.get("category"),
        topic=meta.get("topic"),
        category=item.get("category"),
        difficulty=item.get("difficulty"),
        exam_ids=meta.get("examIds", []),
        language=meta.get("language"),
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
    if not question_bank_identity_is_intact(item):
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
