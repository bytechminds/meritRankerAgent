"""Per-model reasoning-effort capability.

Widening the ReasoningEffort type alone must grant nothing: a model may only receive
an effort it has explicitly declared, so an undeclared model still rejects `xhigh`
before the provider is ever called.
"""

from __future__ import annotations

import pytest

from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.errors import ModelExecutionConfigError
from services.llm.orchestration.model_config_resolver import ModelConfigResolver


@pytest.fixture
def resolver() -> ModelConfigResolver:
    return ModelConfigResolver(LlmConfigRegistry())


def _resolve(resolver: ModelConfigResolver, alias: str, effort: str) -> None:
    registry = LlmConfigRegistry()
    resolver.validate_provider_options(
        provider_options={"reasoning_effort": effort},
        model_config=registry.model_map[alias],
        model_alias=alias,
    )


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_the_conservative_default_set_is_unchanged(
    resolver: ModelConfigResolver, effort: str
) -> None:
    """Models that declare nothing keep exactly today's accepted values."""
    _resolve(resolver, "openai_o4_mini", effort)


def test_an_undeclared_model_still_rejects_xhigh(
    resolver: ModelConfigResolver,
) -> None:
    with pytest.raises(ModelExecutionConfigError) as excinfo:
        _resolve(resolver, "openai_o4_mini", "xhigh")

    assert "reasoning_effort" in str(excinfo.value)


def test_luna_accepts_xhigh_because_it_declares_it(
    resolver: ModelConfigResolver,
) -> None:
    _resolve(resolver, "openai_gpt_5_6_luna", "xhigh")


def test_terra_did_not_silently_gain_xhigh(resolver: ModelConfigResolver) -> None:
    """Declaring capability on one model must not widen any other."""
    with pytest.raises(ModelExecutionConfigError):
        _resolve(resolver, "openai_gpt_5_6_terra", "xhigh")


def test_max_is_not_granted_to_luna(resolver: ModelConfigResolver) -> None:
    """Luna declares up to xhigh only; `max` remains unauthorised."""
    with pytest.raises(ModelExecutionConfigError):
        _resolve(resolver, "openai_gpt_5_6_luna", "max")


def test_a_non_reasoning_model_still_rejects_any_effort(
    resolver: ModelConfigResolver,
) -> None:
    with pytest.raises(ModelExecutionConfigError):
        _resolve(resolver, "openai_gpt_4_1", "high")
