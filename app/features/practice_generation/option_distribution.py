"""Deterministic final option ordering for private practice questions."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptionDistribution:
    valid: bool
    reason_code: str
    correct_positions: tuple[int, ...]


def _normalized(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def correct_option_index(options: list[str], correct_answer: object) -> int | None:
    target = _normalized(correct_answer)
    matches = [index for index, option in enumerate(options) if _normalized(option) == target]
    return matches[0] if len(matches) == 1 else None


def target_correct_positions(test_id: str, question_ids: list[str]) -> tuple[int, ...]:
    if not question_ids:
        return ()
    offset = int(hashlib.sha256(test_id.encode("utf-8")).hexdigest()[:8], 16) % 4
    return tuple((offset + index) % 4 for index in range(len(question_ids)))


def reorder_options(
    *,
    test_id: str,
    question_id: str,
    options: list[str],
    correct_answer: str,
    target_position: int,
) -> list[str] | None:
    if len(options) != 4 or target_position not in range(4):
        return None
    correct_index = correct_option_index(options, correct_answer)
    if correct_index is None:
        return None
    shuffled = list(options)
    seed = int(
        hashlib.sha256(f"{test_id}|{question_id}".encode()).hexdigest(),
        16,
    )
    random.Random(seed).shuffle(shuffled)
    shuffled_correct_index = correct_option_index(shuffled, correct_answer)
    if shuffled_correct_index is None:
        return None
    shuffled[shuffled_correct_index], shuffled[target_position] = (
        shuffled[target_position],
        shuffled[shuffled_correct_index],
    )
    return shuffled


def validate_answer_position_distribution(
    questions: list[dict[str, Any]],
) -> OptionDistribution:
    positions: list[int] = []
    for question in questions:
        options = question.get("options")
        if not isinstance(options, list) or len(options) != 4:
            return OptionDistribution(False, "ANSWER_POSITION_DISTRIBUTION_INVALID", ())
        correct_index = correct_option_index(
            [str(option) for option in options], question.get("correctAnswer")
        )
        if correct_index is None:
            return OptionDistribution(False, "ANSWER_POSITION_DISTRIBUTION_INVALID", ())
        positions.append(correct_index)
    maximum_per_position = (len(positions) + 3) // 4
    if any(positions.count(position) > maximum_per_position for position in range(4)):
        return OptionDistribution(False, "ANSWER_POSITION_DISTRIBUTION_INVALID", tuple(positions))
    if any(
        positions[index] == positions[index + 1] == positions[index + 2]
        for index in range(max(len(positions) - 2, 0))
    ):
        return OptionDistribution(False, "ANSWER_POSITION_DISTRIBUTION_INVALID", tuple(positions))
    return OptionDistribution(True, "ANSWER_POSITION_DISTRIBUTION_VALID", tuple(positions))
