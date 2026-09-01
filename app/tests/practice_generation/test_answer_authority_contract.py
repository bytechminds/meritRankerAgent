"""
app/tests/practice_generation/test_answer_authority_contract.py
---------------------------------------------------------------
The v2 author/authority contract: a blind authority, an option-count gate, and an
option-id comparison that never depends on how an answer is spelled.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from features.practice_generation.question_contract import (
    _duplicate_numeric_option,
    _numeric_identity,
    validate_persisted_playable_question,
    validate_playable_question,
)
from features.practice_generation.schemas import (
    GeneratedQuestion,
    VerificationResult,
)


def _question(**overrides: object) -> GeneratedQuestion:
    payload: dict[str, object] = {
        "schema_version": "2",
        "generation_item_id": "item-1",
        "bucket_id": "bucket-1",
        "slot_id": "slot-001",
        "question": "Which option equals two plus two?",
        "question_type": "mcq",
        "options": [
            {"option_id": "0", "value": "1"},
            {"option_id": "1", "value": "2"},
            {"option_id": "2", "value": "3"},
            {"option_id": "3", "value": "4"},
        ],
        "correct_option_id": "3",
        "answer_explanation": "Two plus two is four.",
        "subject": "math",
        "topic": "algebra",
        "difficulty": "basic",
    }
    payload.update(overrides)
    return GeneratedQuestion.model_validate(payload)


def _result(valid_option_ids: list[str], decision: str = "ACCEPT") -> VerificationResult:
    return VerificationResult(
        schema_version="2",
        generation_item_id="item-1",
        slot_id="slot-001",
        decision=decision,
        valid_option_ids=valid_option_ids,
        reason_codes=["SINGLE_VALID_OPTION"],
    )


class TestAuthorOutputIsMinimal:
    def test_author_need_not_emit_correct_answer_or_solution(self) -> None:
        """Both are identities of the chosen option, so the author never states them."""
        question = _question()
        assert question.correct_answer == "4"
        assert question.solution == "Two plus two is four."

    def test_derived_answer_tracks_the_option_id_not_the_author(self) -> None:
        question = _question(correct_option_id="1")
        assert question.correct_answer == "2"

    def test_an_explicit_correct_answer_is_still_accepted(self) -> None:
        """Backward compatibility: a caller supplying the old field is not broken."""
        assert _question(correct_answer="4").correct_answer == "4"

    def test_a_question_is_complete_without_an_explanation(self) -> None:
        """Stem, options and a verified option id are all a playable question needs."""
        question = _question(answer_explanation="")
        assert question.answer_explanation == ""
        assert question.solution == ""
        assert question.correct_option_id == "3"
        assert question.correct_answer == "4"

    def test_an_explanation_is_still_carried_when_supplied(self) -> None:
        """Reuse, QuestionBank, PYQ and educator content keep their prose."""
        question = _question(answer_explanation="Two plus two is four.")
        assert question.answer_explanation == "Two plus two is four."


class TestExplanationIsOptionalForReady:
    @staticmethod
    def _item(*, explanation: str, contract_explanation: str) -> dict[str, object]:
        return {
            "question": "What is two plus two?",
            "options": ["1", "2", "3", "4"],
            "correctAnswer": "4",
            "explanation": explanation,
            "answers": json.dumps(
                {
                    "schemaVersion": "2",
                    "optionIdentity": "INDEX_V1",
                    "options": [
                        {"optionId": 0, "value": "1"},
                        {"optionId": 1, "value": "2"},
                        {"optionId": 2, "value": "3"},
                        {"optionId": 3, "value": "4"},
                    ],
                    "correctOptionId": 3,
                    "correctAnswer": "4",
                    "answerExplanation": contract_explanation,
                    "answerStatus": "VERIFIED",
                    "answerVersion": 1,
                }
            ),
            "_practiceMeta": {"questionType": "mcq", "language": "english"},
        }

    def test_persisted_v2_contract_accepts_an_absent_explanation(self) -> None:
        validation = validate_persisted_playable_question(
            self._item(explanation="", contract_explanation=""),
            expected_question_type="mcq",
            expected_language="english",
            solution_required=False,
        )
        assert (validation.valid, validation.reason_code) == (True, "PLAYABLE")

    def test_persisted_v2_contract_keeps_an_existing_explanation_valid(self) -> None:
        validation = validate_persisted_playable_question(
            self._item(
                explanation="Two plus two equals four.",
                contract_explanation="Two plus two equals four.",
            ),
            expected_question_type="mcq",
            expected_language="english",
            solution_required=False,
        )
        assert (validation.valid, validation.reason_code) == (True, "PLAYABLE")

    def test_persisted_v2_contract_accepts_a_wholly_absent_explanation_key(self) -> None:
        """The production shape: _put_question omits the attribute when it is empty."""
        item = self._item(explanation="", contract_explanation="")
        del item["explanation"]

        validation = validate_persisted_playable_question(
            item,
            expected_question_type="mcq",
            expected_language="english",
            solution_required=False,
        )
        assert (validation.valid, validation.reason_code) == (True, "PLAYABLE")

    def test_persisted_v2_contract_rejects_disagreeing_explanations(self) -> None:
        """Optional, but the item and its answer contract must not diverge."""
        validation = validate_persisted_playable_question(
            self._item(
                explanation="Two plus two equals four.",
                contract_explanation="Something else entirely.",
            ),
            expected_question_type="mcq",
            expected_language="english",
            solution_required=False,
        )
        assert validation.valid is False
        assert validation.reason_code == "ANSWER_CONTRACT_MISMATCH"

    def test_v1_questions_still_require_their_solution(self) -> None:
        """Legacy behaviour is untouched: only schema v2 drops the requirement."""
        validation = validate_playable_question(
            question_type="mcq",
            question="Legacy question text here.",
            options=["a", "b", "c", "d"],
            correct_answer="a",
            solution="",
            solution_required=True,
        )
        assert validation.valid is False
        assert validation.reason_code == "QUESTION_SOLUTION_REQUIRED"


class TestAuthorityOutputContract:
    def test_exactly_one_valid_option_is_acceptable(self) -> None:
        assert _result(["3"]).is_approved is True

    def test_independent_answer_is_derived_from_the_single_valid_option(self) -> None:
        assert _result(["3"]).independently_solved_option_id == "3"

    def test_accept_with_no_valid_option_is_rejected_by_the_schema(self) -> None:
        with pytest.raises(ValidationError, match="exactly one valid option"):
            _result([])

    def test_accept_with_several_valid_options_is_rejected_by_the_schema(self) -> None:
        with pytest.raises(ValidationError, match="exactly one valid option"):
            _result(["1", "3"])

    def test_repeated_option_ids_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not repeat"):
            _result(["3", "3"], decision="REGENERATE")

    def test_a_rejecting_decision_may_carry_no_valid_option(self) -> None:
        assert _result([], decision="REGENERATE").is_approved is False

    def test_a_rejecting_decision_may_carry_several_valid_options(self) -> None:
        result = _result(["1", "3"], decision="REGENERATE")
        assert result.is_approved is False
        assert result.valid_option_ids == ["1", "3"]


class TestAgreementIsByOptionIdOnly:
    """The gate compares ids, so notation can never decide correctness."""

    @pytest.mark.parametrize("option_value", ["4", "four", "4.0"])
    def test_agreement_holds_regardless_of_answer_spelling(
        self, option_value: str
    ) -> None:
        question = _question(
            options=[
                {"option_id": "0", "value": "1"},
                {"option_id": "1", "value": "2"},
                {"option_id": "2", "value": "3"},
                {"option_id": "3", "value": option_value},
            ]
        )
        result = _result(["3"])
        assert result.valid_option_ids[0] == question.correct_option_id

    def test_disagreement_is_visible_as_an_id_mismatch(self) -> None:
        question = _question(correct_option_id="3")
        assert _result(["1"], decision="REGENERATE").valid_option_ids != [
            question.correct_option_id
        ]


class TestDeterministicOptionEquivalence:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("48", 48),
            ("forty-eight", 48),
            ("forty eight", 48),
            ("1,000", 1000),
            ("0.5", 0.5),
            ("1/2", 0.5),
            ("twenty", 20),
            ("ninety-nine", 99),
        ],
    )
    def test_recognised_numeric_notations(self, text: str, expected: float) -> None:
        identity = _numeric_identity(text)
        assert identity is not None
        assert float(identity) == expected

    @pytest.mark.parametrize("text", ["5 km", "fast", "", "12/0", "rupees", "1/2/3"])
    def test_non_numeric_options_are_left_alone(self, text: str) -> None:
        assert _numeric_identity(text) is None

    @pytest.mark.parametrize(
        "options",
        [
            ["48", "forty-eight", "50", "60"],
            ["0.5", "1/2", "0.25", "2"],
            ["1,000", "1000", "2000", "3000"],
        ],
    )
    def test_equivalent_numeric_options_are_caught(self, options: list[str]) -> None:
        assert _duplicate_numeric_option(options) is True

    @pytest.mark.parametrize(
        "options",
        [
            ["5 km", "5 m", "10 km", "2 km"],
            ["fast", "rapid", "slow", "still"],
            ["1", "2", "3", "4"],
        ],
    )
    def test_no_false_positives_on_units_or_synonyms(
        self, options: list[str]
    ) -> None:
        """Units change meaning and synonyms need context — both are the authority's job."""
        assert _duplicate_numeric_option(options) is False

    def test_playable_validation_rejects_equivalent_numeric_options(self) -> None:
        validation = validate_playable_question(
            question_type="mcq",
            question="Find the next term: 3, 6, 12, 24, ?",
            options=["48", "forty-eight", "50", "60"],
            correct_answer="48",
            solution="Each term doubles.",
            solution_required=False,
        )
        assert validation.valid is False
        assert validation.reason_code == "SEMANTIC_DUPLICATE_OPTIONS"

    def test_exact_duplicate_detection_is_unchanged(self) -> None:
        validation = validate_playable_question(
            question_type="mcq",
            question="Pick the vegetable.",
            options=["carrot", "carrot", "apple", "mango"],
            correct_answer="carrot",
            solution="A carrot is a vegetable.",
            solution_required=False,
        )
        assert validation.valid is False
        assert validation.reason_code == "DUPLICATE_OPTIONS"


class TestEnglishSoftSemanticGate:
    """Soft-semantic English is gated per item, not per category.

    Synonym/antonym/near-synonym/idiomatic items are unsafe only when more than one
    option is defensible. The Authority already reports every defensible option in
    `valid_option_ids`, and an ACCEPT carrying more than one is refused outright. That
    is strictly more precise than a category-level RULE_BOUND/EVIDENCE_REQUIRED enum,
    which would also block wide-gap synonym items that are genuinely single-answer.

    Live evidence (Gemini 3.7 Authority, production v2 path): five crafted
    near-synonym items — happy/glad/joyful/cheerful, quick/fast/rapid/swift,
    begin/start/commence/initiate, died/passed away/expired/perished, and
    increase/decrease/reduce/diminish — each returned ["0","1","2"] and were rejected;
    abundant/plentiful and fragile/durable returned ["0"] and were accepted.
    """

    @pytest.mark.parametrize(
        "valid_option_ids",
        [["0", "1"], ["0", "1", "2"], ["0", "1", "2", "3"]],
    )
    def test_accept_cannot_carry_several_defensible_options(
        self, valid_option_ids: list[str]
    ) -> None:
        """A soft-semantic item with several readings can never be an ACCEPT."""
        with pytest.raises(ValueError):
            _result(valid_option_ids, decision="ACCEPT")

    @pytest.mark.parametrize(
        "valid_option_ids",
        [["0", "1"], ["0", "1", "2"]],
    )
    def test_several_defensible_options_survive_only_as_regenerate(
        self, valid_option_ids: list[str]
    ) -> None:
        result = _result(valid_option_ids, decision="REGENERATE")
        assert len(result.valid_option_ids) > 1

    def test_no_defensible_option_cannot_reach_ready(self) -> None:
        assert _result([], decision="REGENERATE").valid_option_ids == []

    def test_single_defensible_option_is_the_only_ready_shape(self) -> None:
        result = _result(["0"], decision="ACCEPT")
        assert result.valid_option_ids == ["0"]
