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

from services.llm.orchestration.config_registry import LlmConfigRegistry


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

    def test_the_v4pro_legacy_alias_now_resolves_to_the_azure_deployment(
        self, registry: LlmConfigRegistry
    ) -> None:
        """deepseek_v4pro claimed V4-Pro but served deepseek-reasoner; it is corrected."""
        config = registry.model_map["deepseek_v4pro"]
        assert config.provider == "azure_openai"
        assert config.provider_profile == "azure_foundry_v1"
        assert config.deployment == "DeepSeek-V4-Pro"
        assert config.model_id is None

    @pytest.mark.parametrize("alias", ["math_advanced_generator"])
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
    """Authority selection lives in tests/test_authority_subject_routing.py.

    Kept here: the retirement and isolation invariants this file is responsible for.
    """

    def test_no_verifier_route_uses_the_retired_terra_alias(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Terra is RETIRED_FOR_PRACTICE_AUTHORITY.

        Across three runs of a 49-item Math and 52-item Reasoning gold corpus it
        accepted 15 and 15 defective items, almost all by naming one valid option and
        not evaluating the rest. The alias stays registered for rollback and other
        roles; it must serve no verifier route.
        """
        serving = [
            f"{s}.{r}.{d}"
            for (s, r, d), entry in registry.route_map.items()
            if r == "verifier" and entry.model == "openai_gpt_5_6_terra" and s != "general"
        ]
        assert serving == []
        assert "openai_gpt_5_6_terra" in registry.model_map

    def test_every_verifier_route_keeps_a_bounded_output_budget(
        self, registry: LlmConfigRegistry
    ) -> None:
        for (subject, task_role, difficulty), entry in registry.route_map.items():
            if task_role != "verifier":
                continue
            assert entry.max_tokens == 5000, f"{subject}.{task_role}.{difficulty}"
            assert entry.temperature == 0.0

    def test_doubt_solver_keeps_its_own_verifier_entry(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Practice Authority selection must never move Doubt Solver."""
        assert ("general", "verifier", "default") in registry.route_map
        general = registry.get_route("general", "verifier", "default")
        assert general is not None
        assert general.model == "openai_gpt_5_6_terra"


class TestGeminiIntegration:
    def test_the_classifier_route_is_unchanged(
        self, registry: LlmConfigRegistry
    ) -> None:
        route = registry.get_route("general", "classifier", "default")
        assert route is not None
        assert route.model == "doubt_solver_classifier_gemini"

    def test_gemini_3_7_reuses_the_existing_provider_and_credentials(
        self, registry: LlmConfigRegistry
    ) -> None:
        """One Gemini provider path only — no second abstraction, no new credential."""
        candidate = registry.model_map["gemini_3_7_flash"]
        classifier = registry.model_map["doubt_solver_classifier_gemini"]
        assert candidate.provider == "gemini"
        assert candidate.provider_profile == classifier.provider_profile
        assert candidate.model_id == "gemini-3.7-flash"

    def test_gemini_3_7_declares_no_unprobed_effort_levels(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Nothing is claimed from documentation alone."""
        candidate = registry.model_map["gemini_3_7_flash"]
        assert candidate.supported_reasoning_efforts is None
        assert candidate.supports_reasoning is False

    def test_gemini_3_7_serves_only_its_qualified_authority_families(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Authority for Math, Reasoning and English; Author for the factual family."""
        routed = sorted(
            f"{s}.{r}.{d}"
            for (s, r, d), entry in registry.route_map.items()
            if entry.model == "gemini_3_7_flash"
        )
        assert routed == [
            "english.verifier.default",
            "factual.generator.default",
            "math.verifier.default",
            "reasoning.verifier.default",
        ]


class TestDeterministicInternalIds:
    def test_generation_item_id_is_derived_not_authored(self) -> None:
        """The author copying a placeholder must not cost the question."""
        from features.practice_generation.schemas import GeneratedQuestion

        base = {
            "schema_version": "2",
            "bucket_id": "b",
            "question": "Which option equals two plus two?",
            "question_type": "mcq",
            "options": [
                {"option_id": "0", "value": "1"},
                {"option_id": "1", "value": "2"},
                {"option_id": "2", "value": "3"},
                {"option_id": "3", "value": "4"},
            ],
            "correct_option_id": "3",
            "subject": "math",
            "topic": "t",
            "difficulty": "basic",
        }
        derived = [
            GeneratedQuestion.model_validate(
                {**base, "slot_id": f"slot-00{n}", "generation_item_id": "i"}
            ).generation_item_id
            for n in (1, 2, 3)
        ]
        assert derived == ["item-slot-001", "item-slot-002", "item-slot-003"]
        assert len(set(derived)) == 3


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
