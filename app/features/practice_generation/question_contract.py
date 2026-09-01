"""Deterministic renderer contract for the current single-choice practice player."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

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


# Number words only up to the range an exam option realistically uses as a bare
# quantity. This table exists to catch one proven defect class — the same value
# offered twice in two notations ("48" and "forty-eight") — and deliberately stops
# well short of general language understanding.
_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}


def _numeric_identity(value: str) -> Decimal | None:
    """Return the number an option denotes, or None when it is not purely a number.

    Only whole-string numbers qualify. An option carrying any other content — a unit,
    a currency symbol, a sentence — returns None, because equal magnitudes there do
    not imply equal meaning ("5 km" and "5 m" must stay distinct).
    """
    text = value.strip().casefold().replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except (ArithmeticError, ValueError):
        pass
    if re.fullmatch(r"-?\d+\s*/\s*\d+", text):
        numerator, denominator = (part.strip() for part in text.split("/"))
        try:
            if Decimal(denominator) == 0:
                return None
            return Decimal(numerator) / Decimal(denominator)
        except (ArithmeticError, ValueError):
            return None
    words = [word for word in re.split(r"[\s-]+", text) if word and word != "and"]
    if not words or not all(word in _NUMBER_WORDS for word in words):
        return None
    if len(words) == 1:
        return Decimal(_NUMBER_WORDS[words[0]])
    if len(words) == 2:
        tens, units = (_NUMBER_WORDS[word] for word in words)
        if tens >= 20 and tens % 10 == 0 and 1 <= units <= 9:
            return Decimal(tens + units)
    return None


def _duplicate_numeric_option(values: Sequence[str]) -> bool:
    """True when two options denote the same number in different notations."""
    seen: list[Decimal] = []
    for value in values:
        number = _numeric_identity(value)
        if number is None:
            continue
        if any(number == previous for previous in seen):
            return True
        seen.append(number)
    return False


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
    if _duplicate_numeric_option(values):
        return None, PlayableQuestionValidation(False, "SEMANTIC_DUPLICATE_OPTIONS")
    return values, PlayableQuestionValidation(True, "PLAYABLE")


class SemanticBasis(StrEnum):
    """How a question's single correct answer can be established.

    RULE_BOUND items are decidable from grammar, usage in a supplied context, or an
    explicit passage. EVIDENCE_REQUIRED items assert lexical equivalence — synonym,
    antonym, near-synonym, substitution, idiomatic equivalence — where correctness
    rests on a lexicon rather than a rule, and several options can be defensible.
    """

    RULE_BOUND = "RULE_BOUND"
    EVIDENCE_REQUIRED = "EVIDENCE_REQUIRED"


# Closed set of RULE_BOUND question forms: those whose single correct answer is
# decidable from grammar, sentence structure, an explicit passage, or the usage of a
# word supplied in the stem. This is an allowlist, so any form not recognised here —
# including every future or unseen one — falls through to EVIDENCE_REQUIRED and fails
# closed. It classifies the question's FORM, never its topic: it reads no topic_id,
# target_skill or model-authored label, and adds no model call.
_RULE_BOUND_STEM = re.compile(
    r"\bgrammatical(?:ly)?\b"
    r"|\bgrammar\b"
    r"|\bcontains?\s+(?:an?\s+)?error\b"
    r"|\berror\s+(?:in|detection|part)\b"
    r"|\bidentify\s+the\s+(?:part|segment|portion)\b"
    r"|\b(?:best\s+)?order\s+(?:for|of)\s+the\s+sentences?\b"
    r"|\b(?:correct|proper|logical)\s+(?:order|sequence)\b"
    r"|\brearrange\b"
    r"|\bread\s+the\s+(?:passage|paragraph|text)\b"
    r"|\buses?\s+the\s+word\s+.{0,40}?\s*correctly\b"
    r"|\bused\s+correctly\b",
    re.IGNORECASE,
)


def classify_semantic_basis(question: object) -> SemanticBasis:
    """Deterministically classify a question stem's semantic basis.

    Safe by default: only recognised rule-bound forms are RULE_BOUND. Anything else,
    recognised lexical-equivalence operator or not, is EVIDENCE_REQUIRED.
    """
    stem = str(question or "")
    if stem and _RULE_BOUND_STEM.search(stem):
        return SemanticBasis.RULE_BOUND
    return SemanticBasis.EVIDENCE_REQUIRED


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


def _explanation_consistent(*, item: dict[str, object], explanation: object) -> bool:
    """A v2 explanation is optional, but the two copies must never disagree.

    Scoring reads the canonical option id, never prose, so an absent explanation does
    not make a question unplayable. What would be a defect is the item and its answer
    contract carrying different explanations, so that stays enforced.
    """
    stored = item.get("explanation")
    written = "" if explanation is None else explanation
    if not isinstance(written, str) or not isinstance(stored, (str, type(None))):
        return False
    return (stored or "").strip() == written.strip()


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
        or not _explanation_consistent(item=item, explanation=explanation)
    ):
        return PlayableQuestionValidation(False, "ANSWER_CONTRACT_MISMATCH")
    return PlayableQuestionValidation(True, "PLAYABLE")
