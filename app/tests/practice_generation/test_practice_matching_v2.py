"""Exact schema-v2 slot reuse and canonical metadata tests."""

from __future__ import annotations

import json

from features.practice_generation.matching import (
    ReusableQuestion,
    build_reuse_difficulty_prefix,
    build_slot_reuse_bucket_key,
    match_existing_questions_to_slots,
    reusable_question_from_item,
)
from features.practice_generation.metadata_normalization import (
    normalize_exam,
    normalize_subject,
    normalize_topic,
)
from features.practice_generation.schemas import (
    Complexity,
    Difficulty,
    PlannerFamily,
    PlannerSlot,
    PracticeBlueprint,
    PracticeType,
    QuestionType,
)


def _slot(
    slot_id: str,
    *,
    category: str = "work_efficiency",
    exam_ids: list[str] | None = None,
    pattern_family_id: str | None = None,
) -> PlannerSlot:
    return PlannerSlot(
        slot_id=slot_id,
        subject_id="math",
        topic_id="time_and_work",
        category_id=category,
        difficulty=Difficulty.INTERMEDIATE,
        complexity=Complexity.MEDIUM,
        exam_ids=exam_ids or ["SSC_CGL"],
        question_type=QuestionType.MCQ,
        target_skill="calculate_combined_work_rate",
        variation_hint=f"variation_{slot_id}",
        pattern_family_id=pattern_family_id,
        generator_route_hint="math.generator.intermediate",
    )


def _blueprint(*slots: PlannerSlot) -> PracticeBlueprint:
    return PracticeBlueprint(
        schema_version="2",
        practice_type=PracticeType.QUICK_PRACTICE,
        accepted_count=len(slots),
        planner_family=PlannerFamily.QUANT_REASONING,
        slots=list(slots),
    )


def _candidate(
    question_id: str,
    *,
    category: str = "work_efficiency",
    exam_ids: tuple[str, ...] = ("SSC_CGL",),
    pattern_family_id: str | None = None,
    confidence: float | None = 0.97,
) -> ReusableQuestion:
    return ReusableQuestion(
        question_id=question_id,
        question="A and B complete a job together. Find their combined rate.",
        options=("1/4", "1/5", "1/6", "1/7"),
        correct_answer="1/4",
        solution="Add the two individual work rates.",
        subject="math",
        topic="time_and_work",
        difficulty="medium",
        question_type="mcq",
        language="english",
        source="QuestionBank",
        source_updated_at="2026-08-06T00:00:00Z",
        category=category,
        exam_ids=exam_ids,
        pattern_family_id=pattern_family_id,
        confidence=confidence,
    )


def test_approved_topic_aliases_resolve_but_unknown_alias_does_not() -> None:
    assert normalize_subject("Quant") == "math"
    assert {
        normalize_topic("time work"),
        normalize_topic("time_and_work"),
        normalize_topic("Time & Work"),
        normalize_topic("work and time"),
    } == {"time_and_work"}
    assert normalize_topic("invented topic alias") is None
    assert normalize_exam("SSC CGL") == "SSC_CGL"
    assert normalize_exam("invented exam") is None


def test_slot_reuse_key_uses_category_not_subject_and_difficulty_prefix() -> None:
    slot = _slot("slot-001")
    assert build_slot_reuse_bucket_key(slot, language="english") == (
        "v1#work_efficiency#time_and_work#mcq#english"
    )
    assert build_reuse_difficulty_prefix(slot.difficulty.value) == "v1#medium#"


def test_exact_slot_matching_is_one_to_one_and_deterministic() -> None:
    blueprint = _blueprint(_slot("slot-001"), _slot("slot-002"))
    lower = _candidate("question-b", confidence=0.94)
    higher = _candidate("question-a", confidence=0.99)

    matches = match_existing_questions_to_slots(
        blueprint,
        [lower, higher, higher],
        requested_language="english",
    )

    assert [match.selected.question_id if match.selected else None for match in matches] == [
        "question-a",
        "question-b",
    ]
    assert sum(match.deficit for match in matches) == 0


def test_category_exam_pattern_and_confidence_constraints_fail_closed() -> None:
    blueprint = _blueprint(
        _slot("slot-001", pattern_family_id="combined_work_rate"),
    )
    candidates = [
        _candidate(
            "wrong-category",
            category="worker_scheduling",
            pattern_family_id="combined_work_rate",
        ),
        _candidate("missing-exam", exam_ids=(), pattern_family_id="combined_work_rate"),
        _candidate("wrong-pattern", pattern_family_id="pipes_and_cisterns"),
        _candidate("low-confidence", pattern_family_id="combined_work_rate", confidence=0.89),
        _candidate("missing-confidence", pattern_family_id="combined_work_rate", confidence=None),
    ]

    match = match_existing_questions_to_slots(
        blueprint,
        candidates,
        requested_language="english",
    )[0]

    assert match.selected is None
    assert match.deficit == 1


def test_hydration_rejects_low_or_malformed_declared_confidence() -> None:
    item = {
        "qbId": "question-1",
        "question": "If A works in four days, what fraction is completed daily?",
        "answers": json.dumps({"options": ["1/2", "1/3", "1/4", "1/5"]}),
        "correctAnswer": "1/4",
        "explanation": "One job divided by four days is one fourth per day.",
        "category": "work_efficiency",
        "difficulty": "MEDIUM",
        "meta": {
            "status": "ACTIVE",
            "qualityStatus": "VERIFIED",
            "reusable": True,
            "visibility": "PLATFORM",
            "language": "english",
            "subject": "math",
            "topic": "time_and_work",
            "questionType": "mcq",
            "examIds": ["SSC_CGL"],
            "confidence": 0.89,
        },
    }
    assert reusable_question_from_item(item, requested_language="english") is None
    item["meta"]["confidence"] = "not-a-number"
    assert reusable_question_from_item(item, requested_language="english") is None
