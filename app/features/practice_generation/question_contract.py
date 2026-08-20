"""Deterministic renderer contract for the current single-choice practice player."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

_CURRENT_PLAYER_QUESTION_TYPE = "mcq"
_CURRENT_PLAYER_OPTION_COUNT = 4


@dataclass(frozen=True)
class PlayableQuestionValidation:
    valid: bool
    reason_code: str


def _normalized(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _integer_index(value: object) -> int | None:
    """Accept exact DynamoDB numeric values without accepting untrusted strings or floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal) and value.is_finite() and value == value.to_integral_value():
        return int(value)
    return None


def _validated_options(options: object) -> tuple[list[str] | None, PlayableQuestionValidation]:
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        return None, PlayableQuestionValidation(False, "MISSING_OPTIONS")
    if len(options) != _CURRENT_PLAYER_OPTION_COUNT:
        return None, PlayableQuestionValidation(False, "INVALID_OPTION_COUNT")
    values = [str(option).strip() for option in options]
    if any(not value for value in values):
        return None, PlayableQuestionValidation(False, "EMPTY_OPTION_TEXT")
    if len({_normalized(value) for value in values}) != len(values):
        return None, PlayableQuestionValidation(False, "DUPLICATE_OPTIONS")
    return values, PlayableQuestionValidation(True, "PLAYABLE")


def validate_playable_question(
    *,
    question_type: object,
    question: object,
    options: object,
    correct_answer: object,
    solution: object,
    solution_required: bool,
) -> PlayableQuestionValidation:
    """Validate the one persisted question shape that the current player can render."""
    if str(question_type or "").casefold() != _CURRENT_PLAYER_QUESTION_TYPE:
        return PlayableQuestionValidation(False, "UNSUPPORTED_QUESTION_TYPE")
    if not _normalized(question):
        return PlayableQuestionValidation(False, "QUESTION_TEXT_EMPTY")
    if re.search(r"\b(?:correct\s+(?:answer|option)|answer)\s*(?:is|:)\b", str(question), re.I):
        return PlayableQuestionValidation(False, "ANSWER_EXPOSED")
    values, option_validation = _validated_options(options)
    if values is None:
        return option_validation
    normalized_answer = _normalized(correct_answer)
    answer_matches = sum(
        _normalized(option) == normalized_answer for option in values
    )
    if not normalized_answer or answer_matches != 1:
        return PlayableQuestionValidation(False, "CORRECT_OPTION_NOT_FOUND")
    if solution_required and not _normalized(solution):
        return PlayableQuestionValidation(False, "QUESTION_SOLUTION_REQUIRED")
    return PlayableQuestionValidation(True, "PLAYABLE")


def validate_persisted_playable_question(
    item: dict[str, object],
    *,
    expected_question_type: str,
    expected_language: str,
    solution_required: bool,
    expected_storage_language: str | None = None,
) -> PlayableQuestionValidation:
    """Cross-check a stored Question record before it is delivered or made READY."""
    meta = item.get("_practiceMeta")
    if not isinstance(meta, dict):
        return PlayableQuestionValidation(False, "QUESTION_META_INVALID")
    question_type = str(meta.get("questionType") or "")
    if question_type.casefold() != expected_question_type.casefold():
        return PlayableQuestionValidation(False, "RENDERER_CONTRACT_INVALID")
    if str(meta.get("language") or "").casefold() != expected_language.casefold():
        return PlayableQuestionValidation(False, "QUESTION_LANGUAGE_INVALID")
    if (
        expected_storage_language is not None
        and str(item.get("language") or "").casefold()
        != expected_storage_language.casefold()
    ):
        return PlayableQuestionValidation(False, "QUESTION_LANGUAGE_STORAGE_INVALID")
    raw_answers = item.get("answers")
    try:
        answers = json.loads(raw_answers) if isinstance(raw_answers, str) else raw_answers
    except json.JSONDecodeError:
        return PlayableQuestionValidation(False, "MALFORMED_QUESTION_ANSWERS")
    if not isinstance(answers, dict):
        return PlayableQuestionValidation(False, "MALFORMED_QUESTION_ANSWERS")
    options = item.get("options")
    answer_options = answers.get("options")
    if not isinstance(options, list) or not isinstance(answer_options, list):
        return PlayableQuestionValidation(False, "MISSING_OPTIONS")
    if str(answers.get("schemaVersion") or "") == "2":
        contract_validation = _validate_v2_answer_contract(
            item=item,
            answers=answers,
            options=options,
            answer_options=answer_options,
        )
        if not contract_validation.valid:
            return contract_validation
    elif options != answer_options or item.get("correctAnswer") != answers.get("correctAnswer"):
        return PlayableQuestionValidation(False, "ANSWER_CONTRACT_MISMATCH")
    return validate_playable_question(
        question_type=question_type,
        question=item.get("question"),
        options=options,
        correct_answer=item.get("correctAnswer"),
        solution=item.get("explanation"),
        solution_required=solution_required,
    )


def _validate_v2_answer_contract(
    *,
    item: dict[str, object],
    answers: dict[str, object],
    options: list[object],
    answer_options: list[object],
) -> PlayableQuestionValidation:
    """Validate the private V2 answer payload against the deployed INDEX_V1 player contract."""
    answer_version = _integer_index(answers.get("answerVersion"))
    if (
        answers.get("optionIdentity") != "INDEX_V1"
        or answers.get("answerStatus") != "VERIFIED"
        or answer_version is None
        or answer_version < 1
        or len(options) != len(answer_options)
    ):
        return PlayableQuestionValidation(False, "ANSWER_CONTRACT_MISMATCH")

    normalized_options = [str(option).strip() for option in options]
    for index, answer_option in enumerate(answer_options):
        if (
            not isinstance(answer_option, dict)
            or _integer_index(answer_option.get("optionId")) != index
            or not isinstance(answer_option.get("value"), str)
            or answer_option["value"].strip() != normalized_options[index]
        ):
            return PlayableQuestionValidation(False, "ANSWER_CONTRACT_MISMATCH")

    correct_option_id = _integer_index(answers.get("correctOptionId"))
    correct_answer = answers.get("correctAnswer")
    explanation = answers.get("answerExplanation")
    if (
        correct_option_id is None
        or correct_option_id < 0
        or correct_option_id >= len(answer_options)
        or not isinstance(correct_answer, str)
        or answer_options[correct_option_id].get("value") != correct_answer
        or item.get("correctAnswer") != correct_answer
        or not isinstance(explanation, str)
        or explanation.strip() == ""
        or item.get("explanation") != explanation
    ):
        return PlayableQuestionValidation(False, "ANSWER_CONTRACT_MISMATCH")
    return PlayableQuestionValidation(True, "PLAYABLE")
