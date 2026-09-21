"""Generator structured-output validation: actionable taxonomy + finish_reason handling.

Mirrors the planner's equivalent regression coverage
(tests/practice_generation/test_planning_route.py) for the same class of defect: a
generic STRUCTURED_PARSE_INVALID catch-all previously hid whether a batch failure was
a parse failure, a missing field, a wrong type, a constraint violation, truncation, or
a bad envelope shape.
"""

from __future__ import annotations

import json

from features.practice_generation.generation import (
    STRUCTURAL_GENERATOR_REASON_CODES,
    parse_partial_generation,
)
from features.practice_generation.schemas import (
    DemandBucket,
    Difficulty,
    GenerationGroup,
    QuestionType,
    VerificationPolicy,
)


def _bucket(*, required_count: int = 1) -> DemandBucket:
    return DemandBucket(
        bucket_id="bucket-1",
        subject="math",
        topic="addition",
        difficulty=Difficulty.BASIC,
        question_type=QuestionType.MCQ,
        required_count=required_count,
        question_intent="practice",
        verification_policy=VerificationPolicy.SELECTIVE,
    )


def _group(*, required_count: int = 1) -> GenerationGroup:
    return GenerationGroup(
        group_id="group-1", bucket_id="bucket-1", required_count=required_count
    )


def _valid_question(index: int = 1) -> dict:
    return {
        "schema_version": "2",
        "bucket_id": "bucket-1",
        "slot_id": f"slot-{index:03d}",
        "question": f"What is {index} + {index}?",
        "question_type": "mcq",
        "options": [
            {"option_id": "0", "value": str(index * 2)},
            {"option_id": "1", "value": str(index * 2 + 1)},
            {"option_id": "2", "value": str(index * 2 + 2)},
            {"option_id": "3", "value": str(index * 2 + 3)},
        ],
        "correct_option_id": "0",
        "subject": "math",
        "topic": "addition",
        "difficulty": "basic",
    }


def test_valid_batch_is_accepted() -> None:
    parsed = parse_partial_generation(
        json.dumps({"questions": [_valid_question()]}),
        group=_group(),
        bucket=_bucket(),
        existing_normalized_texts=set(),
    )
    assert len(parsed.accepted) == 1
    assert parsed.rejected_count == 0


def test_missing_required_field_is_a_missing_field_code() -> None:
    question = _valid_question()
    del question["correct_option_id"]
    parsed = parse_partial_generation(
        json.dumps({"questions": [question]}),
        group=_group(),
        bucket=_bucket(),
        existing_normalized_texts=set(),
    )
    assert parsed.accepted == ()
    # correct_option_id lands under the existing SCHEMA_V2_ANSWER_CONTRACT_INVALID
    # location bucket, which stays specific rather than being coarsened.
    assert parsed.rejection_reason_codes == ("SCHEMA_V2_ANSWER_CONTRACT_INVALID",)


def test_wrong_scalar_type_is_a_schema_type_invalid_code() -> None:
    question = _valid_question()
    question["question"] = 12345  # must be a string
    parsed = parse_partial_generation(
        json.dumps({"questions": [question]}),
        group=_group(),
        bucket=_bucket(),
        existing_normalized_texts=set(),
    )
    assert parsed.accepted == ()
    assert parsed.rejection_reason_codes[0] in STRUCTURAL_GENERATOR_REASON_CODES


def test_wrong_array_shape_for_options_is_structural() -> None:
    question = _valid_question()
    question["options"] = "not-a-list"
    parsed = parse_partial_generation(
        json.dumps({"questions": [question]}),
        group=_group(),
        bucket=_bucket(),
        existing_normalized_texts=set(),
    )
    assert parsed.accepted == ()
    assert parsed.rejection_reason_codes == ("SCHEMA_V2_OPTION_CONTRACT_INVALID",)


def test_truncated_output_is_distinguished_from_parse_failure() -> None:
    truncated = parse_partial_generation(
        "{not valid json",
        group=_group(),
        bucket=_bucket(),
        existing_normalized_texts=set(),
        finish_reason="length",
    )
    assert truncated.accepted == ()
    assert truncated.rejection_reason_codes == ("GENERATOR_OUTPUT_TRUNCATED",)

    ordinary = parse_partial_generation(
        "{not valid json",
        group=_group(),
        bucket=_bucket(),
        existing_normalized_texts=set(),
        finish_reason="stop",
    )
    assert ordinary.rejection_reason_codes == ("GENERATOR_PARSE_FAILURE",)

    unknown = parse_partial_generation(
        "{not valid json",
        group=_group(),
        bucket=_bucket(),
        existing_normalized_texts=set(),
    )
    assert unknown.rejection_reason_codes == ("GENERATOR_PARSE_FAILURE",)


def test_wrong_typed_questions_envelope_is_a_contract_invalid_code() -> None:
    """"questions" present but not an array — a missing key defaults to zero
    questions (a GENERATOR_BATCH_COUNT_MISMATCH deficit), but a present, wrong-typed
    envelope is a distinct, more specific signal."""
    parsed = parse_partial_generation(
        json.dumps({"questions": "not-an-array"}),
        group=_group(),
        bucket=_bucket(),
        existing_normalized_texts=set(),
    )
    assert parsed.accepted == ()
    assert parsed.rejection_reason_codes == ("GENERATOR_CONTRACT_INVALID",)


def test_fewer_questions_than_required_is_a_batch_count_mismatch() -> None:
    """A model that returns 1 question for a 3-required group must not silently
    lose the other 2 slots with no reason code at all."""
    parsed = parse_partial_generation(
        json.dumps({"questions": [_valid_question(1)]}),
        group=_group(required_count=3),
        bucket=_bucket(required_count=3),
        existing_normalized_texts=set(),
    )
    assert len(parsed.accepted) == 1
    assert parsed.rejected_count == 2
    assert parsed.rejection_reason_codes.count("GENERATOR_BATCH_COUNT_MISMATCH") == 2


def test_structural_reason_codes_never_include_a_semantic_rejection() -> None:
    """Semantic rejections (validate_playable_question, bucket mismatch, ...) must
    never be classified as structural — the two decide different terminal codes."""
    assert "GENERATION_BUCKET_CONTRACT_MISMATCH" not in STRUCTURAL_GENERATOR_REASON_CODES
    assert "QUESTION_LANGUAGE_MISMATCH" not in STRUCTURAL_GENERATOR_REASON_CODES
    assert "SOFT_SEMANTIC_WITHOUT_TRUSTED_BASIS" not in STRUCTURAL_GENERATOR_REASON_CODES
