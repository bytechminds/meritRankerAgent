"""Gemini 3.7 hidden-reasoning capacity for the factual generator route.

Measured over 695 production-path calls, gemini-3.7-flash spends hidden reasoning
against the completion budget on essentially every call (p50 156, p95 565, p99 1042,
max 1441) while visible output stays at 149-344 tokens. Every other alias in the
registry measures 0. The registry declared no reserve, so batching treated the whole
completion budget as available for visible slot output, over-packed factual groups,
and failed with output_token_exhausted.

These pin the two halves of the correction: the reserve is declared, and the route cap
is large enough that recording it does not shrink the batches the route already made.
"""

from __future__ import annotations

import pytest

from features.practice_generation.generation import (
    _expected_question_output_tokens,
    _model_safe_batch_capacity,
    _route_output_token_budget,
)
from services.llm.orchestration.config_registry import LlmConfigRegistry

MEASURED_MAX_REASONING_TOKENS = 1441
ENVELOPE_TOKENS = 120


def test_registry_declares_the_measured_reasoning_reserve() -> None:
    model = LlmConfigRegistry().model_map["gemini_3_7_flash"]
    assert model.structured_output_reasoning_reserve_tokens >= MEASURED_MAX_REASONING_TOKENS


def test_reserve_reaches_the_batching_calculation() -> None:
    """The reserve is inert unless the route resolver carries it into batching."""
    capacity = _route_output_token_budget("science", "intermediate")
    registry_reserve = LlmConfigRegistry().model_map[
        "gemini_3_7_flash"
    ].structured_output_reasoning_reserve_tokens
    assert capacity.expected_reasoning_tokens == registry_reserve
    assert capacity.expected_reasoning_tokens > 0


@pytest.mark.parametrize(
    ("complexity", "expected_batch"),
    [("low", 3), ("medium", 2), ("high", 2)],
)
def test_recording_the_reserve_preserves_factual_batch_sizes(
    complexity: str,
    expected_batch: int,
) -> None:
    """Reserving capacity must not multiply Author calls.

    These are the batch sizes the route produced before the reserve existed; the cap
    was raised by exactly enough to keep them.
    """
    batch, _ = _model_safe_batch_capacity(
        subject="science",
        difficulty="intermediate",
        complexity=complexity,
        token_budget_resolver=_route_output_token_budget,
    )
    assert batch == expected_batch


def test_route_cap_covers_reserve_envelope_and_the_largest_batch() -> None:
    """The cap is derived, not guessed: reserve + envelope + visible output."""
    capacity = _route_output_token_budget("science", "intermediate")
    worst = max(
        _expected_question_output_tokens(subject="science", complexity=complexity)
        * batch
        for complexity, batch in (("low", 3), ("medium", 2), ("high", 2))
    )
    required = capacity.expected_reasoning_tokens + ENVELOPE_TOKENS + worst
    assert capacity.configured_output_tokens >= required


def test_no_other_alias_claims_a_reserve_it_does_not_spend() -> None:
    """Only measured reasoning consumers declare a reserve; the rest measured 0."""
    models = LlmConfigRegistry().model_map
    declared = {
        alias
        for alias, model in models.items()
        if model.structured_output_reasoning_reserve_tokens > 0
    }
    assert "gemini_3_7_flash" in declared
    assert declared <= {"gemini_3_7_flash", "deepseek_reasoner"} | {
        alias
        for alias in declared
        if models[alias].supports_thinking or models[alias].supports_reasoning
    }
