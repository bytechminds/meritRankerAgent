"""Advanced-reasoning capacity stabilization.

Observed live behaviour: every advanced reasoning generation spent a full model
call at the 4000 initial cap, exhausted output tokens, then retried at the 5600
cap the policy already defines. This pins the stabilized policy: start at the
capacity that works, and never spend a model call on an escalation that cannot
add capacity.
"""

from __future__ import annotations

import pytest

from schemas.llm_routing import PracticeGenerationWorkload, RouteRequest
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.practice_generation_capacity import (
    PracticeGenerationCapacityPolicy,
)


def _capacity(subject: str, difficulty: str, *, complexity: str, slot_count: int):
    from services.llm.orchestration.route_resolver import resolve_route

    registry = LlmConfigRegistry()
    route = resolve_route(
        RouteRequest(
            request_id=f"capacity-{subject}-{difficulty}",
            subject=subject,
            task_role="generator",
            difficulty=difficulty,
            intent="practice",
            language="english",
        )
    )
    return PracticeGenerationCapacityPolicy.resolve(
        route_decision=route,
        model_config=registry.model_map[route.model],
        workload=PracticeGenerationWorkload(complexity=complexity, slot_count=slot_count),
    )


def test_advanced_reasoning_starts_at_the_capacity_that_actually_works() -> None:
    capacity = _capacity("reasoning", "advanced", complexity="high", slot_count=1)

    # No preliminary 4000 attempt: the observed sample failed it 5/5 times.
    assert capacity.initial_max_output_tokens == 5600


@pytest.mark.parametrize("slot_count", [1, 2])
@pytest.mark.parametrize("complexity", ["low", "medium", "high"])
def test_every_legal_advanced_reasoning_group_starts_at_5600(
    slot_count: int, complexity: str
) -> None:
    capacity = _capacity("reasoning", "advanced", complexity=complexity, slot_count=slot_count)

    assert capacity.initial_max_output_tokens == 5600


def test_advanced_reasoning_escalation_cannot_add_capacity() -> None:
    """With initial == escalation, the existing no-op guard in the executor skips
    the escalation call entirely and the request proceeds to the fallback."""
    capacity = _capacity("reasoning", "advanced", complexity="high", slot_count=1)

    assert capacity.escalation_max_output_tokens <= capacity.initial_max_output_tokens


def test_advanced_reasoning_ceilings_are_untouched() -> None:
    capacity = _capacity("reasoning", "advanced", complexity="high", slot_count=1)

    # This patch removes a wasted call; it does not buy more capacity.
    assert capacity.product_hard_max_output_tokens == 5600
    assert capacity.model_hard_max_output_tokens == 8000
    # capacity.reasoning_effort mirrors the alias default (Terra declares "high") and is
    # not sent; the payload effort is the route's provider option, which stays "medium".
    route = LlmConfigRegistry().get_route("reasoning", "generator", "advanced")
    assert route.provider_options["reasoning_effort"] == "medium"


def test_math_advanced_capacity_is_unchanged() -> None:
    capacity = _capacity("math", "advanced", complexity="high", slot_count=1)

    # Different provider, different measured reserve: must not move.
    # GPT-4.1 reserves nothing for hidden reasoning, so the initial budget is the
    # route's own figure; a single advanced question measured 150-600 output tokens.
    assert capacity.initial_max_output_tokens == 2600
    assert capacity.escalation_max_output_tokens == 3600
    assert capacity.product_hard_max_output_tokens == 4400


@pytest.mark.parametrize(
    ("subject", "difficulty", "complexity", "expected_initial"),
    [
        # Basic Reasoning now routes to a reasoning-capable Author, so the capacity
        # policy selects its reasoning band (1600) instead of the non-reasoning one
        # (900). The policy itself is unchanged — the band is keyed on the model's
        # declared capability, and a model that spends tokens on hidden reasoning
        # needs the larger initial budget. This row tracks that deliberate route
        # change; the other three rows still guard against capacity drift.
        ("reasoning", "basic", "low", 1600),
        ("reasoning", "intermediate", "medium", 2600),
        ("math", "basic", "low", 900),
        ("math", "intermediate", "medium", 1600),
    ],
)
def test_non_advanced_routes_are_unchanged(
    subject: str, difficulty: str, complexity: str, expected_initial: int
) -> None:
    capacity = _capacity(subject, difficulty, complexity=complexity, slot_count=1)

    assert capacity.initial_max_output_tokens == expected_initial


def test_advanced_reasoning_group_size_limits_are_unchanged() -> None:
    high = _capacity("reasoning", "advanced", complexity="high", slot_count=1)
    low = _capacity("reasoning", "advanced", complexity="low", slot_count=1)

    assert high.max_slots_per_batch == 1
    assert low.max_slots_per_batch == 2
