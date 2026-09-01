"""All three Practice generation waves share one Author output contract.

Wave 0 was reduced to a minimal Author contract: no `correct_answer`, no
`answer_explanation`, no `solution`, and `generation_item_id` derived in code as
`item-{slot_id}`. Waves 1 and 2 still instructed the model to author all four, which
cost output tokens on prose the schema discards and re-opened the authored-id path
that previously produced duplicate ids. These guards assert the contract, not wording.
"""

from __future__ import annotations

import pytest

from services.llm.orchestration.prompt_resolver import DEFAULT_PROMPT_ROOT

WAVE_PROMPTS = [
    "practice_generation/question_generator_v2.md",
    "practice_generation/question_generator_factual.md",
    "practice_generation/question_repair.md",
    "practice_generation/question_regenerator.md",
]

DERIVED_OR_DISCARDED_FIELDS = [
    "correct_answer",
    "answer_explanation",
    "solution",
    "generation_item_id",
]


def _text(name: str) -> str:
    return (DEFAULT_PROMPT_ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("prompt", WAVE_PROMPTS)
@pytest.mark.parametrize("field", DERIVED_OR_DISCARDED_FIELDS)
def test_no_wave_asks_the_author_for_a_code_owned_field(
    prompt: str, field: str
) -> None:
    """The JSON shape each wave shows the model must not contain these keys."""
    body = _text(prompt)
    shape_start = body.find("{\"schema_version\"")
    if shape_start == -1:
        pytest.skip(f"{prompt} shows no literal JSON shape")
    shape = body[shape_start : body.find("}", body.rfind("difficulty")) + 1]
    assert f'"{field}"' not in shape, f"{prompt} still asks for {field}"


@pytest.mark.parametrize("prompt", WAVE_PROMPTS)
def test_every_wave_states_the_omission_explicitly(prompt: str) -> None:
    body = _text(prompt)
    assert "Omit `correct_answer`" in body
    assert "`answer_explanation`" in body


@pytest.mark.parametrize("prompt", WAVE_PROMPTS)
def test_every_wave_keeps_the_required_author_fields(prompt: str) -> None:
    """Reducing the contract must not drop what downstream parsing needs."""
    body = _text(prompt)
    for field in ("schema_version", "slot_id", "question", "options", "correct_option_id"):
        assert field in body, f"{prompt} lost {field}"


@pytest.mark.parametrize(
    "prompt",
    [
        "practice_generation/question_repair.md",
        "practice_generation/question_regenerator.md",
    ],
)
def test_replacement_waves_keep_their_evidence_and_language_rules(prompt: str) -> None:
    """Trimming the contract must not remove a safety rule."""
    body = _text(prompt)
    assert "fresh_evidence" in body
    assert "evidence_by_slot" in body
    assert "supplied `language`" in body
