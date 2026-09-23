"""GPT-6 Sol on the Practice advanced planner routes only.

Pins the alias contract, the exact route migration and its blast radius, the
provider payload, and that a GPT-6 Sol failure still reaches the unchanged
planner fallback semantics (deterministic when trusted constraints exist, fail
closed when they do not).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from features.practice_generation.planning import BlueprintManager, BlueprintPlanningError
from features.practice_generation.providers import RoutedPlannerProvider
from features.practice_generation.schemas import (
    Difficulty,
    PracticeGenerationRequest,
    PracticeType,
    RequestedPracticeConstraint,
    planner_generation_schema,
)
from schemas.llm_orchestration import LlmMessage, ModelExecutionResult, ProviderExecutionRequest
from schemas.llm_routing import RouteRequest
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.errors import LlmConfigValidationError
from services.llm.orchestration.model_config_resolver import ModelConfigResolver
from services.llm.orchestration.model_execution import RegistryBackedModelExecutor
from services.llm.orchestration.orchestrator import LlmOrchestrator
from services.llm.orchestration.route_resolver import resolve_route
from services.llm.providers.errors import LlmProviderExecutionError
from services.llm.providers.payload_shaping import build_azure_openai_chat_completion_kwargs

ALIAS = "openai_gpt_6_sol"
MIGRATED_FAMILIES = ("factual", "quant_reasoning", "english")


def _route(subject: str, difficulty: str):
    return resolve_route(
        RouteRequest(
            request_id="gpt-6-sol-route",
            subject=subject,
            task_role="planner",
            difficulty=difficulty,
            intent="practice",
            language="english",
        )
    )


class TestAlias:
    def test_alias_resolves_to_existing_azure_foundry_profile(self) -> None:
        model = LlmConfigRegistry().model_map[ALIAS]

        assert model.provider == "azure_openai"
        assert model.provider_profile == "azure_foundry_v1"
        assert model.deployment == "gpt-6-sol"
        assert model.supports_reasoning is True
        assert model.reasoning_effort == "medium"
        assert model.supported_reasoning_efforts == ["medium", "high"]
        assert model.token_budget_param == "max_completion_tokens"
        assert model.supports_temperature is False
        assert model.fallback_models == []

    def test_env_overrides_deployment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_GPT_6_SOL", "gpt-6-sol-canary")

        assert LlmConfigRegistry().model_map[ALIAS].deployment == "gpt-6-sol-canary"

    def test_blank_env_keeps_default_deployment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_GPT_6_SOL", "")

        assert LlmConfigRegistry().model_map[ALIAS].deployment == "gpt-6-sol"

    def test_active_route_with_blank_deployment_fails_preflight(self) -> None:
        registry = LlmConfigRegistry()
        registry.validate_real_mode_deployments()
        registry._model_map[ALIAS] = registry.model_map[ALIAS].model_copy(
            update={"deployment": ""}
        )

        with pytest.raises(LlmConfigValidationError, match=f"'{ALIAS}' has empty Azure"):
            registry.validate_real_mode_deployments()


class TestRoutes:
    @pytest.mark.parametrize("family", MIGRATED_FAMILIES)
    def test_advanced_planner_uses_gpt_6_sol_medium_with_preserved_contract(
        self, family: str
    ) -> None:
        route = _route(family, "advanced")

        assert route.model == ALIAS
        assert route.provider_options == {"reasoning_effort": "medium"}
        assert route.prompt == f"practice_generation/planners/{family}.md"
        assert route.max_tokens == 8000
        assert LlmConfigRegistry().get_route(family, "planner", "advanced").fallback == [
            "intermediate",
            "default",
            "safe_mock",
        ]

    @pytest.mark.parametrize("family", MIGRATED_FAMILIES)
    @pytest.mark.parametrize(
        ("difficulty", "model"),
        [
            ("default", "general_fast_generator"),
            ("basic", "general_fast_generator"),
            ("intermediate", "openai_gpt_4_1"),
        ],
    )
    def test_lower_planner_tiers_unchanged(
        self, family: str, difficulty: str, model: str
    ) -> None:
        route = _route(family, difficulty)

        assert route.model == model
        assert route.provider_options == {}

    def test_only_the_three_advanced_planner_routes_use_gpt_6_sol(self) -> None:
        registry = LlmConfigRegistry()

        routed = {
            key for key, route in registry._route_map.items() if route.model == ALIAS
        }

        assert routed == {(family, "planner", "advanced") for family in MIGRATED_FAMILIES}
        assert registry.get_route("general", "planner", "advanced").model == "openai_gpt_4_1"


def _planner_execution_request(family: str) -> ProviderExecutionRequest:
    route = _route(family, "advanced")
    return ProviderExecutionRequest(
        route_decision=route,
        model_resolution=ModelConfigResolver().resolve(route),
        messages=[LlmMessage(role="user", content="{}")],
        temperature=route.temperature,
        max_tokens=route.max_tokens,
        provider_options=dict(route.provider_options),
    )


class TestPayload:
    @pytest.mark.parametrize("family", MIGRATED_FAMILIES)
    def test_payload_sends_medium_effort_completion_budget_and_planner_schema(
        self, family: str
    ) -> None:
        kwargs, metadata = build_azure_openai_chat_completion_kwargs(
            request=_planner_execution_request(family),
            deployment="gpt-6-sol",
        )

        assert set(kwargs) == {
            "model",
            "messages",
            "max_completion_tokens",
            "reasoning_effort",
            "response_format",
        }
        assert kwargs["model"] == "gpt-6-sol"
        assert kwargs["max_completion_tokens"] == 8000
        assert kwargs["reasoning_effort"] == "medium"
        assert kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "planner_response",
                "schema": planner_generation_schema(),
                "strict": True,
            },
        }
        assert metadata.reasoning_param_sent is True
        assert {"max_tokens", "temperature"} <= set(metadata.dropped_params)


class _FailingProvider:
    """GPT-6 Sol unavailable: records every provider attempt, then fails."""

    def __init__(self) -> None:
        self.attempts: list[tuple[str, dict]] = []

    def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
        self.attempts.append(
            (request.model_resolution.model_alias, dict(request.provider_options))
        )
        raise LlmProviderExecutionError(
            "planner provider timed out",
            failure_kind="timeout",
            provider="azure_openai",
            model_alias=request.model_resolution.model_alias,
        )

    def execute_stream(self, request: ProviderExecutionRequest) -> Iterator[str]:
        raise AssertionError("Planner does not stream.")


def _forty_question_request(
    constraints: tuple[RequestedPracticeConstraint, ...] = (),
) -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="r", user_id="u", conversation_id="c", turn_id="t",
        original_query=(
            "create mock test of 40 questions. from indian constitution, "
            "math arithmetic, reasoning, general science"
        ),
        practice_type=PracticeType.QUICK_PRACTICE,
        requested_count=40, accepted_count=40, subject="general",
        difficulty=Difficulty.INTERMEDIATE, assessment_title="T",
        trusted_constraints=constraints,
    )


def _build(request: PracticeGenerationRequest, provider: _FailingProvider):
    executor = RegistryBackedModelExecutor(provider_executor=provider)
    planner = RoutedPlannerProvider(LlmOrchestrator(model_executor=executor))
    return BlueprintManager(planner).build(request)


class TestProviderFailureFallback:
    def test_failure_without_trusted_constraints_fails_closed(self) -> None:
        provider = _FailingProvider()

        with pytest.raises(BlueprintPlanningError) as excinfo:
            _build(_forty_question_request(), provider)

        assert excinfo.value.reason_code == "PRACTICE_PLANNER_SEMANTIC_FALLBACK_UNSAFE"
        assert provider.attempts == [(ALIAS, {"reasoning_effort": "medium"})]

    def test_failure_with_trusted_constraints_uses_deterministic_fallback(self) -> None:
        provider = _FailingProvider()
        constraints = (
            RequestedPracticeConstraint(
                subject_id="geography", topic_id="geography", source_text="geography"
            ),
            RequestedPracticeConstraint(
                subject_id="polity", topic_id="polity", source_text="polity"
            ),
        )

        result = _build(_forty_question_request(constraints), provider)

        assert result.deterministic_fallback is True
        assert result.planner_calls == 1
        assert len(result.blueprint.slots) == 40
        assert {slot.topic_id for slot in result.blueprint.slots} == {"geography", "polity"}
        assert provider.attempts == [(ALIAS, {"reasoning_effort": "medium"})]
