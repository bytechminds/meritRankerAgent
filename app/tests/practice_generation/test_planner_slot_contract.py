"""The planner slot contract for `not_same_when` and `generation_group_hint`.

The planner prompt names both fields; these tests pin the types the schema
actually accepts, so a prompt that drifts back to scalars fails here first.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from features.practice_generation.planning import parse_blueprint
from features.practice_generation.schemas import (
    Difficulty,
    PracticeGenerationRequest,
    PracticeType,
)

_SLOT: dict = {
    "slot_id": "slot-001",
    "subject_id": "math",
    "topic_id": "percentage",
    "category_id": "basic_percentage",
    "difficulty": "intermediate",
    "complexity": "medium",
    "exam_ids": [],
    "question_type": "mcq",
    "target_skill": "calculate_percentage_of_number",
    "variation_hint": "direct calculation",
    "pattern_family_id": None,
    "generator_route_hint": "math.generator.intermediate",
    "reasoning_target": "numerical",
    "trap_type": "misplaced_decimal",
    "not_same_when": [],
    "generation_group_hint": None,
}


def _request(count: int = 2) -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="r",
        user_id="u",
        conversation_id="c",
        turn_id="t",
        original_query="Create questions on percentage",
        practice_type=PracticeType.QUICK_PRACTICE,
        requested_count=count,
        accepted_count=count,
        subject="math",
        topic="percentage",
        difficulty=Difficulty.INTERMEDIATE,
        assessment_title="T",
    )


def _raw(**overrides: object) -> str:
    slots = [
        dict(
            _SLOT,
            slot_id=f"slot-{index:03d}",
            target_skill=f"skill_{index}",
            variation_hint=f"variation {index}",
            **overrides,
        )
        for index in (1, 2)
    ]
    return json.dumps({"slots": slots})


# --- accepted shapes ---------------------------------------------------------

def test_empty_exclusions_and_null_group_hint_are_accepted() -> None:
    blueprint = parse_blueprint(
        _raw(not_same_when=[], generation_group_hint=None), _request()
    )

    assert [slot.not_same_when for slot in blueprint.slots] == [[], []]
    assert [slot.generation_group_hint for slot in blueprint.slots] == [None, None]


def test_populated_exclusions_and_integer_group_hint_are_accepted() -> None:
    blueprint = parse_blueprint(
        _raw(not_same_when=["avoid exact repetition"], generation_group_hint=3),
        _request(),
    )

    assert blueprint.slots[0].not_same_when == ["avoid exact repetition"]
    assert blueprint.slots[0].generation_group_hint == 3


@pytest.mark.parametrize("hint", [1, 5])
def test_the_group_hint_bounds_themselves_are_accepted(hint: int) -> None:
    blueprint = parse_blueprint(_raw(generation_group_hint=hint), _request())

    assert blueprint.slots[0].generation_group_hint == hint


# --- rejections that must stay rejections ------------------------------------

def test_a_scalar_string_is_not_an_exclusion_list() -> None:
    """The exact shape the planner emitted before the prompt stated the type."""
    with pytest.raises(ValidationError):
        parse_blueprint(_raw(not_same_when="variation_hint"), _request())


def test_a_text_label_is_not_a_group_hint() -> None:
    with pytest.raises(ValidationError):
        parse_blueprint(_raw(generation_group_hint="basic_percentage"), _request())


@pytest.mark.parametrize("hint", [0, 6])
def test_group_hints_outside_one_through_five_are_rejected(hint: int) -> None:
    with pytest.raises(ValidationError):
        parse_blueprint(_raw(generation_group_hint=hint), _request())


def test_more_than_eight_exclusions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_blueprint(
            _raw(not_same_when=[f"rule {index}" for index in range(9)]), _request()
        )


# --- the prompts state the contract the schema enforces ----------------------

def test_the_shared_contract_states_both_field_types() -> None:
    """One statement, composed into every planner that emits these slots."""
    from pathlib import Path

    prompt = Path("prompts/practice_generation/shared_contract.md").read_text()

    assert "`not_same_when` is a JSON array" in prompt
    assert "`generation_group_hint` is a JSON integer from 1 through 5" in prompt
