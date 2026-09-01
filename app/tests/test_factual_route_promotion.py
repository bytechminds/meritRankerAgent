"""Promoted production routes for the factual family.

Gemini 3.7 authors and GPT-5.6 Luna is the blind Answer Authority. These pin the
promotion so the pair cannot drift back to a qualification-only override, collapse
onto one model, or leak to families that never qualified.
"""

from __future__ import annotations

import pytest

from schemas.llm_routing import RouteRequest
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.route_resolver import resolve_route

FACTUAL_SUBJECTS = (
    "science",
    "history",
    "geography",
    "physics",
    "chemistry",
    "biology",
    "computer_science",
    "economics",
    "polity",
)


def _route(subject: str, task_role: str):
    return resolve_route(
        RouteRequest(request_id="promo", subject=subject, task_role=task_role)
    )


def test_factual_author_is_gemini() -> None:
    decision = _route("factual", "generator")
    assert decision.model == "gemini_3_7_flash"
    assert decision.route_source == "exact"


def test_factual_authority_is_luna() -> None:
    decision = _route("factual", "verifier")
    assert decision.model == "openai_gpt_5_6_luna"
    assert decision.route_source == "exact"


def test_author_and_authority_are_different_models() -> None:
    """Agreement is only evidence when two independent models produce it."""
    assert _route("factual", "generator").model != _route("factual", "verifier").model


@pytest.mark.parametrize("subject", FACTUAL_SUBJECTS)
def test_every_factual_subject_reaches_the_promoted_pair(subject: str) -> None:
    """All nine product subjects normalize onto the factual family, not general."""
    author = _route(subject, "generator")
    authority = _route(subject, "verifier")
    assert authority.model == "openai_gpt_5_6_luna"
    assert authority.route_source == "exact"
    assert author.model == "gemini_3_7_flash"


@pytest.mark.parametrize("subject", ["general", "other", "not_a_subject"])
def test_unsupported_subjects_do_not_inherit_the_factual_authority(
    subject: str,
) -> None:
    decision = _route(subject, "verifier")
    assert decision.model != "openai_gpt_5_6_luna"
    assert decision.route_id == "general.verifier.default"


def test_english_does_not_inherit_the_factual_authority() -> None:
    """English has its own qualified Authority (Gemini), never Luna."""
    decision = _route("english", "verifier")
    assert decision.model != "openai_gpt_5_6_luna"
    assert decision.route_id == "english.verifier.default"


@pytest.mark.parametrize("subject", ["math", "reasoning"])
def test_math_and_reasoning_authorities_are_unchanged(subject: str) -> None:
    decision = _route(subject, "verifier")
    assert decision.model == "gemini_3_7_flash"
    assert decision.route_source == "exact"


def test_factual_generator_route_carries_the_computed_capacity() -> None:
    """The cap is derived from the batching formula, not a round number."""
    route = LlmConfigRegistry().get_route("factual", "generator", "default")
    assert route is not None
    assert route.max_tokens == 2460


class TestEnglishAuthorityRoute:
    """Gemini 3.7 is the qualified English Answer Authority for rule-bound items."""

    def test_english_authority_is_gemini(self) -> None:
        decision = _route("english", "verifier")
        assert decision.model == "gemini_3_7_flash"
        assert decision.route_id == "english.verifier.default"

    def test_english_author_is_not_the_authority(self) -> None:
        assert _route("english", "generator").model != _route("english", "verifier").model

    def test_english_authority_qualifies_and_does_not_fail_closed(self) -> None:
        from features.practice_generation.providers import _require_qualified_authority

        assert (
            _require_qualified_authority(
                request_id="eng", subject="english", language="english"
            )
            == "english.verifier.default"
        )

    def test_factual_authority_is_unchanged_by_the_english_route(self) -> None:
        assert _route("factual", "verifier").model == "openai_gpt_5_6_luna"
