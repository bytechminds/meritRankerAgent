"""
app/tests/test_model_config_closure.py
--------------------------------------
Deployment identity and provider-path guards for the models closed out this round.

The costly failure these prevent is a model alias quietly resolving to the wrong
provider: DeepSeek V4-Pro on the direct api.deepseek.com path returned HTTP 402 and
triggered expensive fallbacks, and a Mini deployment name landing on the full GPT-5.4
variable would route a different model entirely.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

import pytest

from schemas.llm_routing import RouteRequest
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.route_resolver import resolve_route


@pytest.fixture(scope="module")
def registry() -> LlmConfigRegistry:
    return LlmConfigRegistry()


class TestGpt54MiniDeployment:
    def test_alias_resolves_to_the_authoritative_deployment(
        self, registry: LlmConfigRegistry
    ) -> None:
        assert registry.model_map["openai_gpt_5_4_mini"].deployment == "gpt-5.4-mini"

    def test_it_uses_the_azure_v1_provider_profile(
        self, registry: LlmConfigRegistry
    ) -> None:
        config = registry.model_map["openai_gpt_5_4_mini"]
        assert config.provider == "azure_openai"
        assert config.provider_profile == "azure_foundry_v1"

    def test_reasoning_metadata_matches_probed_behaviour(
        self, registry: LlmConfigRegistry
    ) -> None:
        """The deployment rejects max_tokens and reports reasoning tokens."""
        config = registry.model_map["openai_gpt_5_4_mini"]
        assert config.token_budget_param == "max_completion_tokens"
        assert config.supports_reasoning is True
        assert config.supported_reasoning_efforts == ["low", "medium", "high", "xhigh"]

    def test_temperature_stays_enabled(self, registry: LlmConfigRegistry) -> None:
        """gpt-5.4-mini accepts temperature; gpt-5-mini does not. Do not copy siblings."""
        assert registry.model_map["openai_gpt_5_4_mini"].supports_temperature is True

    def test_the_full_gpt_5_4_alias_did_not_inherit_the_mini_deployment(
        self, registry: LlmConfigRegistry
    ) -> None:
        """GPT-5.4 and GPT-5.4 Mini are different models."""
        full = registry.model_map["openai_gpt_5_4"].deployment
        assert full != "gpt-5.4-mini"


class TestDeepSeekV4ProProviderPath:
    def test_alias_resolves_to_the_authoritative_deployment(
        self, registry: LlmConfigRegistry
    ) -> None:
        config = registry.model_map["azure_deepseek_v4_pro"]
        assert config.deployment == "DeepSeek-V4-Pro"

    def test_it_uses_azure_and_never_the_direct_deepseek_provider(
        self, registry: LlmConfigRegistry
    ) -> None:
        config = registry.model_map["azure_deepseek_v4_pro"]
        assert config.provider == "azure_openai"
        assert config.provider_profile == "azure_foundry_v1"
        assert config.model_id is None

    def test_the_azure_alias_carries_no_deepseek_reasoner_identity(
        self, registry: LlmConfigRegistry
    ) -> None:
        config = registry.model_map["azure_deepseek_v4_pro"]
        assert "deepseek-reasoner" not in (config.model_id or "")
        assert "deepseek-reasoner" not in (config.deployment or "")

    @pytest.mark.parametrize("alias", ["deepseek_v4pro", "math_advanced_generator"])
    def test_legacy_direct_aliases_are_behaviourally_unchanged(
        self, registry: LlmConfigRegistry, alias: str
    ) -> None:
        """Both still read DEEPSEEK_V4PRO_MODEL and both still target the direct API.

        Remapping either one would silently move their existing callers and fallback
        chains onto a different provider, which is why the Azure entry is additive.
        """
        config = registry.model_map[alias]
        assert config.provider == "deepseek"
        assert config.provider_profile == "deepseek_primary"
        assert config.deployment is None


class TestAuthorityRoute:
    def test_authority_resolves_to_the_qualified_terra_alias(
        self, registry: LlmConfigRegistry
    ) -> None:
        route = registry.get_route("general", "verifier", "default")
        assert route is not None
        assert route.model == "openai_gpt_5_6_terra"

    def test_authority_keeps_the_configuration_that_passed_the_gold_corpus(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Changing effort or budget would invalidate the qualification result."""
        route = registry.get_route("general", "verifier", "default")
        assert route is not None
        assert route.provider_options.get("reasoning_effort") == "medium"
        assert route.max_tokens == 5000

    def test_no_route_uses_terra_as_an_author(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Author and Authority must stay different models or agreement proves nothing."""
        authoring = [
            f"{subject}.{task_role}.{difficulty}"
            for (subject, task_role, difficulty), entry in registry.route_map.items()
            if entry.model == "openai_gpt_5_6_terra" and task_role != "verifier"
        ]
        assert authoring == []

    def test_verifier_requests_resolve_through_the_single_shared_route(self) -> None:
        """Documents the blast radius: every verifier caller shares this one route."""
        for subject in ("general", "math", "reasoning", "english"):
            decision = resolve_route(
                RouteRequest(
                    request_id="cfg-1", subject=subject, task_role="verifier"
                )
            )
            assert decision.route_id == "general.verifier.default"
            assert decision.model == "openai_gpt_5_6_terra"


class TestFrozenRoutesPreserved:
    @pytest.mark.parametrize(
        ("subject", "task_role", "difficulty", "expected"),
        [
            ("math", "generator", "advanced", "openai_gpt_4_1"),
            ("quant_reasoning", "planner", "advanced", "openai_gpt_4_1"),
        ],
    )
    def test_frozen_gpt_4_1_routes_are_untouched(
        self,
        registry: LlmConfigRegistry,
        subject: str,
        task_role: str,
        difficulty: str,
        expected: str,
    ) -> None:
        route = registry.get_route(subject, task_role, difficulty)
        assert route is not None
        assert route.model == expected


class TestBedrockAliasesUnchanged:
    @pytest.mark.parametrize(
        ("alias", "model_id"),
        [
            ("glm47_flash_bedrock", "zai.glm-4.7-flash"),
            ("glm5_bedrock", "zai.glm-5"),
            ("qwen3_next_80b_bedrock", "qwen.qwen3-next-80b-a3b"),
            ("qwen3_235b_bedrock", "qwen.qwen3-235b-a22b-2507-v1:0"),
            ("practice_request_intelligence", "zai.glm-4.7-flash"),
        ],
    )
    def test_bedrock_identities_are_stable(
        self, registry: LlmConfigRegistry, alias: str, model_id: str
    ) -> None:
        config = registry.model_map[alias]
        assert config.provider == "bedrock"
        assert config.model_id == model_id
