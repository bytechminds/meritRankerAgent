"""Retired o3 / o4-mini must not be reachable from any live production route.

Live routes are the registry's own active set: ``_active_route_model_aliases`` excludes
the optional difficulties that ``normalize_difficulty`` can never produce. Reachability
is checked on the compiled registry (after environment deployment overlays), through the
primary alias and every ``fallback_models`` entry, expanded transitively.

The replacement policy is Terra primary with Azure DeepSeek-V4-Pro as the bounded
infrastructure fallback. Content-quality failures stay with verification, never with
provider fallback.
"""

from __future__ import annotations

import pytest

from schemas.llm import LlmMessage
from schemas.llm_orchestration import ModelExecutionResult, ProviderExecutionRequest
from schemas.llm_routing import RouteRequest
from services.doubt_solver.answer_correctness import AnswerCorrectnessVerifier
from services.llm.orchestration.config_registry import get_registry
from services.llm.orchestration.model_execution import RegistryBackedModelExecutor
from services.llm.orchestration.route_resolver import normalize_difficulty, resolve_route
from services.llm.providers.errors import LlmProviderExecutionError

RETIRED_IDENTITIES = frozenset({"o3", "o4-mini"})
DORMANT_DIFFICULTIES = frozenset(
    {
        "gemini_test",
        "deepseek_test",
        "deepseek_azure_test",
        "cat_advanced",
        "sbi_po_complex",
        "cat_lrdi",
    }
)
TERRA = "openai_gpt_5_6_terra"
DEEPSEEK = "azure_deepseek_v4_pro"


def _identity(alias: str) -> str:
    config = get_registry().model_map[alias]
    return config.deployment or config.model_id or ""


def _reachable_from(alias: str) -> list[str]:
    model_map = get_registry().model_map
    seen: list[str] = []

    def expand(current: str) -> None:
        if current in seen or current not in model_map:
            return
        seen.append(current)
        for fallback in model_map[current].fallback_models or []:
            expand(fallback)

    expand(alias)
    return seen


def _live_routes() -> list[tuple[str, str, str]]:
    return [
        key for key in get_registry().route_map if key[2] not in DORMANT_DIFFICULTIES
    ]


class TestNoLiveRouteReachesTheOSeries:
    def test_no_live_route_primary_or_fallback_resolves_to_o3_or_o4_mini(self) -> None:
        offending = [
            f"{'.'.join(key)} -> {alias} ({_identity(alias)})"
            for key in _live_routes()
            for alias in _reachable_from(get_registry().route_map[key].model)
            if _identity(alias) in RETIRED_IDENTITIES
        ]
        assert offending == []

    def test_the_registry_active_set_matches_the_live_routes_checked_here(self) -> None:
        registry = get_registry()
        live_primaries = {registry.route_map[key].model for key in _live_routes()}
        assert live_primaries == registry._active_route_model_aliases()  # noqa: SLF001

    def test_no_reachable_alias_takes_its_deployment_from_an_o_series_env_var(self) -> None:
        from services.llm.orchestration import model_registry_env  # noqa: PLC0415

        o_series_env = {"AZURE_OPENAI_DEPLOYMENT_O4_MINI", "AZURE_OPENAI_DEPLOYMENT_O3"}
        reachable = {
            alias
            for key in _live_routes()
            for alias in _reachable_from(get_registry().route_map[key].model)
        }
        overlaid = {
            alias
            for alias, (env_var, _default) in model_registry_env._AZURE_DEPLOYMENT_ENV.items()  # noqa: SLF001
            if env_var in o_series_env
        }
        assert reachable.isdisjoint(overlaid)

    @pytest.mark.parametrize("difficulty", sorted(DORMANT_DIFFICULTIES))
    def test_dormant_difficulties_can_never_be_selected(self, difficulty: str) -> None:
        assert normalize_difficulty(difficulty) == "default"


class TestApprovedReplacementPolicy:
    @pytest.mark.parametrize(
        ("subject", "task_role", "difficulty"),
        [
            ("general", "verifier", "default"),
            ("reasoning", "generator", "intermediate"),
            ("reasoning", "generator", "advanced"),
        ],
    )
    def test_former_o_series_primaries_now_use_terra(
        self, subject: str, task_role: str, difficulty: str
    ) -> None:
        assert get_registry().get_route(subject, task_role, difficulty).model == TERRA

    def test_terra_falls_back_to_azure_deepseek_v4_pro_only(self) -> None:
        model_map = get_registry().model_map
        assert model_map[TERRA].fallback_models == [DEEPSEEK]
        deepseek = model_map[DEEPSEEK]
        assert deepseek.provider == "azure_openai"
        assert deepseek.deployment == "DeepSeek-V4-Pro"
        assert "deepseek-reasoner" not in _identity(DEEPSEEK)

    @pytest.mark.parametrize(
        "alias",
        ["math_basic_generator", "math_intermediate_generator", "reasoning_basic_generator"],
    )
    def test_non_o_series_primaries_keep_their_model_and_gain_terra_then_v4_pro(
        self, alias: str
    ) -> None:
        config = get_registry().model_map[alias]
        assert config.deployment == "gpt-4.1-mini"
        assert config.fallback_models[:2] == [TERRA, "deepseek_v4pro"]
        assert _identity("deepseek_v4pro") == "DeepSeek-V4-Pro"

    def test_the_doubt_solver_verifier_contract_is_unchanged(self) -> None:
        route = get_registry().get_route("general", "verifier", "default")
        assert route.prompt == "answer_correctness_verifier.md"
        assert route.temperature == 0.0
        assert route.max_tokens == 5000
        assert route.provider_options == {"reasoning_effort": "medium"}


class _AliasFake:
    def __init__(self, *, raise_for: dict[str, Exception], content: str) -> None:
        self._raise_for = raise_for
        self._content = content
        self.call_log: list[str] = []

    def execute(self, request: ProviderExecutionRequest) -> ModelExecutionResult:
        alias = request.model_resolution.model_alias
        self.call_log.append(alias)
        if alias in self._raise_for:
            raise self._raise_for[alias]
        return ModelExecutionResult(
            content=self._content, model=alias, provider="azure_openai", finish_reason="stop"
        )


def _messages() -> list[LlmMessage]:
    return [LlmMessage(role="system", content="System."), LlmMessage(role="user", content="Q.")]


def _verifier_decision():
    return resolve_route(
        RouteRequest(
            request_id="o-series-guard",
            subject="general",
            task_role="verifier",
            difficulty="default",
            intent="solve",
            language="english",
        )
    )


class TestFallbackSemantics:
    @pytest.mark.parametrize("failure_kind", ["provider_unavailable", "timeout", "rate_limited"])
    def test_terra_infrastructure_failure_invokes_deepseek_exactly_once(
        self, failure_kind: str
    ) -> None:
        verdict = (
            '{"status":"MATCH","independent_answer":"55",'
            '"single_defensible_answer":true,"reason":"ok"}'
        )
        fake = _AliasFake(
            raise_for={TERRA: LlmProviderExecutionError("down", failure_kind=failure_kind)},
            content=verdict,
        )
        result = RegistryBackedModelExecutor(provider_executor=fake).execute(
            route_decision=_verifier_decision(), messages=_messages()
        )

        assert fake.call_log == [TERRA, DEEPSEEK]
        assert result.model == DEEPSEEK
        assert result.fallback_used is True
        assert result.content == verdict

    def test_a_non_infrastructure_provider_failure_does_not_fall_back(self) -> None:
        fake = _AliasFake(
            raise_for={TERRA: LlmProviderExecutionError("refused", failure_kind="safety_blocked")},
            content="unused",
        )
        with pytest.raises(Exception):  # noqa: B017, PT011
            RegistryBackedModelExecutor(provider_executor=fake).execute(
                route_decision=_verifier_decision(), messages=_messages()
            )
        assert fake.call_log == [TERRA]

    def test_a_successful_but_wrong_answer_never_triggers_provider_fallback(self) -> None:
        """Content quality is not an infrastructure failure: one call, no fallback."""
        fake = _AliasFake(raise_for={}, content="**Answer:** 7 (wrong)")
        result = RegistryBackedModelExecutor(provider_executor=fake).execute(
            route_decision=_verifier_decision(), messages=_messages()
        )
        assert fake.call_log == [TERRA]
        assert result.fallback_used is False

    def test_a_verifier_mismatch_is_final_and_calls_no_other_model(self) -> None:
        calls: list[str] = []

        class _Orchestrator:
            def generate(self, **_kwargs):
                calls.append("generate")
                return type(
                    "R",
                    (),
                    {
                        "content": '{"status":"MISMATCH","independent_answer":"50",'
                        '"single_defensible_answer":true,"reason":"differs"}'
                    },
                )()

        verification = AnswerCorrectnessVerifier(orchestrator=_Orchestrator()).verify(  # type: ignore[arg-type]
            request_id="quality",
            query="A train question with no deterministic answer.",
            candidate_answer="**Answer:** 40 km/hr",
            subject="math",
            difficulty="intermediate",
            language="english",
        )
        assert calls == ["generate"]
        assert verification.status == "mismatch"
        assert verification.approved is False
