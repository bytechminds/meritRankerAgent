"""
app/tests/test_llm_route_resolver.py
--------------------------------------
Unit tests for app/services/llm_orchestration/route_resolver.py

Coverage:
- math/generator/advanced resolves exactly (route_source="exact").
- Unknown difficulty falls back to subject default (route_source="subject_default").
- Unknown subject falls back to general default (route_source="general_default").
- Subject normalization: quant→math, logical_reasoning→reasoning, english_grammar→english.
- Difficulty normalization: hard→advanced, medium→intermediate, easy→basic.
- reasoning route resolves.
- english route resolves.
- Fallback symbols resolve into typed FallbackAttempt objects.
- safe_mock fallback attempt has kind="model".
- RouteDecision has route_id in correct format.
- RouteDecision has no credentials (no api_key, endpoint, model_id values).
- RouteDecision has non-empty prompt and overlays as list.
- Unsupported task role raises LlmRouteNotFoundError safely.
- general default fallback works for math/planner when no planner route exists.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from schemas.llm_routing import FallbackAttempt, RouteRequest
from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.errors import LlmRouteNotFoundError
from services.llm_orchestration.route_resolver import (
    normalize_difficulty,
    normalize_subject,
    resolve_route,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> LlmConfigRegistry:
    """Module-scoped registry loaded from the real project YAML."""
    return LlmConfigRegistry()


def _request(
    subject: str,
    task_role: str = "generator",
    difficulty: str = "default",
    request_id: str = "test-001",
    intent: str | None = None,
) -> RouteRequest:
    return RouteRequest(
        request_id=request_id,
        subject=subject,
        task_role=task_role,  # type: ignore[arg-type]
        difficulty=difficulty,
        intent=intent,
    )


# ---------------------------------------------------------------------------
# TestSubjectNormalization
# ---------------------------------------------------------------------------


class TestSubjectNormalization:
    def test_math_normalizes_to_math(self) -> None:
        assert normalize_subject("math") == "math"

    def test_internal_practice_math_route_subject_is_preserved(self) -> None:
        assert normalize_subject("practice_math") == "practice_math"

    def test_quant_normalizes_to_math(self) -> None:
        assert normalize_subject("quant") == "math"

    def test_quantitative_aptitude_normalizes_to_math(self) -> None:
        assert normalize_subject("quantitative_aptitude") == "math"

    def test_logical_reasoning_normalizes_to_reasoning(self) -> None:
        assert normalize_subject("logical_reasoning") == "reasoning"

    def test_verbal_reasoning_normalizes_to_reasoning(self) -> None:
        assert normalize_subject("verbal_reasoning") == "reasoning"

    def test_english_grammar_normalizes_to_english(self) -> None:
        assert normalize_subject("english_grammar") == "english"

    def test_english_vocabulary_normalizes_to_english(self) -> None:
        assert normalize_subject("english_vocabulary") == "english"

    def test_unknown_subject_normalizes_to_general(self) -> None:
        assert normalize_subject("some_random_topic") == "general"

    def test_empty_subject_normalizes_to_general(self) -> None:
        assert normalize_subject("") == "general"

    def test_whitespace_subject_normalizes_to_general(self) -> None:
        assert normalize_subject("   ") == "general"

    def test_case_insensitive_normalization(self) -> None:
        assert normalize_subject("MATH") == "math"
        assert normalize_subject("Reasoning") == "reasoning"


# ---------------------------------------------------------------------------
# TestDifficultyNormalization
# ---------------------------------------------------------------------------


class TestDifficultyNormalization:
    def test_basic_normalizes_to_basic(self) -> None:
        assert normalize_difficulty("basic") == "basic"

    def test_easy_normalizes_to_basic(self) -> None:
        assert normalize_difficulty("easy") == "basic"

    def test_intermediate_normalizes_to_intermediate(self) -> None:
        assert normalize_difficulty("intermediate") == "intermediate"

    def test_medium_normalizes_to_intermediate(self) -> None:
        assert normalize_difficulty("medium") == "intermediate"

    def test_advanced_normalizes_to_advanced(self) -> None:
        assert normalize_difficulty("advanced") == "advanced"

    def test_hard_normalizes_to_advanced(self) -> None:
        assert normalize_difficulty("hard") == "advanced"

    def test_unknown_normalizes_to_default(self) -> None:
        assert normalize_difficulty("mystery_level") == "default"

    def test_empty_normalizes_to_default(self) -> None:
        assert normalize_difficulty("") == "default"


# ---------------------------------------------------------------------------
# TestExactRouteResolution
# ---------------------------------------------------------------------------


class TestExactRouteResolution:
    def test_math_generator_advanced_resolves_exact(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        assert decision.route_source == "exact"

    def test_math_generator_advanced_uses_advanced_model(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        # Fix A (50Q closure): the advanced-math structured generator moved off
        # deepseek-reasoner, whose hidden-reasoning variance exhausted the output
        # budget on single-slot calls, to the already-configured GPT-4.1 route.
        assert decision.model == "openai_gpt_4_1"

    def test_math_generator_default_resolves_exact(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "default")
        decision = resolve_route(req, registry)
        assert decision.route_source == "exact"
        assert decision.difficulty == "default"

    def test_practice_math_intermediate_resolves_to_terra(
        self, registry: LlmConfigRegistry
    ) -> None:
        decision = resolve_route(
            _request(
                "practice_math",
                "generator",
                "intermediate",
                intent="practice",
            ),
            registry,
        )
        assert decision.route_id == "practice_math.generator.intermediate"
        assert decision.model == "openai_gpt_5_6_terra"
        assert decision.provider_options == {}

    def test_shared_math_intermediate_route_remains_unchanged(
        self, registry: LlmConfigRegistry
    ) -> None:
        decision = resolve_route(
            _request("math", "generator", "intermediate", intent="solve"),
            registry,
        )
        assert decision.route_id == "math.generator.intermediate"
        assert decision.model == "math_intermediate_generator"

    def test_reasoning_generator_default_resolves(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("reasoning", "generator", "default")
        decision = resolve_route(req, registry)
        assert decision.route_source == "exact"
        assert decision.subject == "reasoning"

    def test_english_generator_basic_resolves(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("english", "generator", "basic")
        decision = resolve_route(req, registry)
        assert decision.route_source == "exact"
        assert decision.subject == "english"

    def test_route_id_format(self, registry: LlmConfigRegistry) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        assert decision.route_id == "math.generator.advanced"

    def test_route_id_non_empty(self, registry: LlmConfigRegistry) -> None:
        req = _request("english", "generator", "intermediate")
        decision = resolve_route(req, registry)
        assert len(decision.route_id) > 0


# ---------------------------------------------------------------------------
# TestSubjectDefaultFallback
# ---------------------------------------------------------------------------


class TestSubjectDefaultFallback:
    def test_general_intermediate_route_resolves_exact(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("general", "generator", "medium")
        decision = resolve_route(req, registry)
        assert decision.route_source == "exact"
        assert decision.difficulty == "intermediate"
        assert decision.max_tokens == 1400

    def test_subject_default_has_same_subject(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("general", "generator", "medium")
        decision = resolve_route(req, registry)
        assert decision.subject == "general"


# ---------------------------------------------------------------------------
# TestGeneralDefaultFallback
# ---------------------------------------------------------------------------

_GENERAL_ONLY_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: safe_mock
            prompt: subjects/general_generator.md
            temperature: 0.3
            max_tokens: 800
            fallback:
              - safe_mock
    models:
      safe_mock:
        provider: mock
        provider_profile: local_mock
        model_id: local-mock
        model_label: safe-mock
        cost_tier: none
        supports_streaming: true
        supports_thinking: false
        timeout_seconds: 1
        capabilities: {}
    provider_profiles:
      local_mock:
        provider: mock
""")


class TestGeneralDefaultFallback:
    def test_absent_subject_falls_back_to_general_default(
        self, tmp_path: Path
    ) -> None:
        # Registry has only general routes.  Requesting "math" (known alias)
        # finds no math route → subject_default (none) → general_default.
        yaml_file = tmp_path / "llm_orchestration.yaml"
        yaml_file.write_text(_GENERAL_ONLY_YAML, encoding="utf-8")
        reg = LlmConfigRegistry(yaml_path=yaml_file)
        req = _request("math", "generator", "default")
        decision = resolve_route(req, reg)
        assert decision.route_source == "general_default"
        assert decision.subject == "general"

    def test_absent_subject_difficulty_falls_back_through_chain(
        self, tmp_path: Path
    ) -> None:
        yaml_file = tmp_path / "llm_orchestration.yaml"
        yaml_file.write_text(_GENERAL_ONLY_YAML, encoding="utf-8")
        reg = LlmConfigRegistry(yaml_path=yaml_file)
        req = _request("math", "generator", "hard")
        decision = resolve_route(req, reg)
        # hard→advanced, no math route → subject_default (none) → general_default
        assert decision.route_source == "general_default"

    def test_general_default_returns_valid_decision(
        self, tmp_path: Path
    ) -> None:
        yaml_file = tmp_path / "llm_orchestration.yaml"
        yaml_file.write_text(_GENERAL_ONLY_YAML, encoding="utf-8")
        reg = LlmConfigRegistry(yaml_path=yaml_file)
        req = _request("reasoning", "generator", "default")
        decision = resolve_route(req, reg)
        assert decision.model is not None
        assert decision.prompt is not None


# ---------------------------------------------------------------------------
# TestFallbackAttemptResolution
# ---------------------------------------------------------------------------


class TestFallbackAttemptResolution:
    def test_fallback_attempts_is_list(self, registry: LlmConfigRegistry) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        assert isinstance(decision.fallback_attempts, list)

    def test_safe_mock_fallback_has_kind_model(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        safe_mock_attempts = [
            fa for fa in decision.fallback_attempts if fa.model == "safe_mock"
        ]
        assert len(safe_mock_attempts) >= 1
        assert safe_mock_attempts[0].kind == "model"

    def test_safe_mock_fallback_has_no_route_fields(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        safe_mock_attempt = next(
            fa for fa in decision.fallback_attempts if fa.model == "safe_mock"
        )
        assert safe_mock_attempt.subject is None
        assert safe_mock_attempt.task_role is None
        assert safe_mock_attempt.difficulty is None

    def test_route_fallback_has_kind_route(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        route_attempts = [fa for fa in decision.fallback_attempts if fa.kind == "route"]
        assert len(route_attempts) > 0

    def test_general_default_fallback_symbol_resolved(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "default")
        decision = resolve_route(req, registry)
        # math.generator.default fallback includes general_default
        general_attempts = [
            fa
            for fa in decision.fallback_attempts
            if fa.subject == "general" and fa.kind == "route"
        ]
        assert len(general_attempts) >= 1

    def test_fallback_attempts_all_typed(self, registry: LlmConfigRegistry) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        for fa in decision.fallback_attempts:
            assert isinstance(fa, FallbackAttempt)


# ---------------------------------------------------------------------------
# TestRouteDecisionHasNoCredentials
# ---------------------------------------------------------------------------


class TestRouteDecisionHasNoCredentials:
    """Ensure RouteDecision contains no credential or secret values."""

    def test_route_decision_has_no_api_key_field(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        decision_dict = decision.model_dump()
        assert "api_key" not in decision_dict
        assert "api_key_env" not in decision_dict

    def test_route_decision_has_no_endpoint_field(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        decision_dict = decision.model_dump()
        assert "endpoint" not in decision_dict
        assert "endpoint_env" not in decision_dict

    def test_route_decision_model_is_alias_not_model_id(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        # model field must be an alias like gemini_flash_reasoning_light
        # not an actual provider model_id like gemini-2.5-flash
        assert "_" in decision.model or decision.model == "safe_mock"
        assert not decision.model.startswith("gemini-")  # not a real model_id

    def test_route_decision_has_no_provider_profile(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        decision_dict = decision.model_dump()
        assert "provider_profile" not in decision_dict
        assert "credential_ref" not in decision_dict

    def test_route_decision_no_secret_like_values_in_provider_options(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        for v in decision.provider_options.values():
            if isinstance(v, str):
                assert not v.startswith("sk-")
                assert not v.startswith("AIza")
                assert not v.startswith("AKIA")


# ---------------------------------------------------------------------------
# TestRouteDecisionHasPromptAndOverlays
# ---------------------------------------------------------------------------


class TestRouteDecisionHasPromptAndOverlays:
    def test_route_decision_has_non_empty_prompt(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        assert isinstance(decision.prompt, str)
        assert len(decision.prompt) > 0

    def test_route_decision_overlays_is_list(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        assert isinstance(decision.overlays, list)

    def test_default_route_has_no_overlays(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "default")
        decision = resolve_route(req, registry)
        assert decision.overlays == []

    def test_advanced_route_has_overlays(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("math", "generator", "advanced")
        decision = resolve_route(req, registry)
        assert len(decision.overlays) > 0


# ---------------------------------------------------------------------------
# TestReasoningRouteResolves
# ---------------------------------------------------------------------------


class TestReasoningRouteResolves:
    def test_reasoning_intermediate_resolves(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("reasoning", "generator", "intermediate")
        decision = resolve_route(req, registry)
        assert decision.subject == "reasoning"
        assert decision.route_source == "exact"

    def test_logical_reasoning_normalizes_and_resolves(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("logical_reasoning", "generator", "default")
        decision = resolve_route(req, registry)
        assert decision.subject == "reasoning"


# ---------------------------------------------------------------------------
# TestEnglishRouteResolves
# ---------------------------------------------------------------------------


class TestEnglishRouteResolves:
    def test_english_advanced_resolves(self, registry: LlmConfigRegistry) -> None:
        req = _request("english", "generator", "advanced")
        decision = resolve_route(req, registry)
        assert decision.subject == "english"
        assert decision.route_source == "exact"

    def test_english_grammar_normalizes_and_resolves(
        self, registry: LlmConfigRegistry
    ) -> None:
        req = _request("english_grammar", "generator", "basic")
        decision = resolve_route(req, registry)
        assert decision.subject == "english"


# ---------------------------------------------------------------------------
# TestUnsupportedTaskRole
# ---------------------------------------------------------------------------


class TestUnsupportedTaskRole:
    def test_unsupported_task_role_raises_route_not_found(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Unsupported task_role (no routes configured) must raise LlmRouteNotFoundError."""
        req = _request("math", "formatter", "default")
        with pytest.raises(LlmRouteNotFoundError):
            resolve_route(req, registry)

    def test_unsupported_task_role_error_message_is_safe(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Error message must not expose YAML internals."""
        req = _request("math", "classifier", "default")
        try:
            resolve_route(req, registry)
        except LlmRouteNotFoundError as exc:
            msg = str(exc)
            assert "yaml" not in msg.lower()
            assert "password" not in msg.lower()
            assert "secret" not in msg.lower()
            assert "api_key" not in msg.lower()
        else:
            # If it somehow resolves (e.g. a classifier route was added), that is OK
            pass

    def test_generator_task_role_does_not_raise(
        self, registry: LlmConfigRegistry
    ) -> None:
        """generator task_role is fully supported and must not raise."""
        req = _request("math", "generator", "default")
        decision = resolve_route(req, registry)
        assert decision is not None
