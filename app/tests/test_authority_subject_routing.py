"""
app/tests/test_authority_subject_routing.py
-------------------------------------------
Subject-aware Practice Authority routing, and native structured output for the
verifier role.

Two proven defects motivate these: verify_slot() previously sent a literal subject, so
one Authority served every Practice subject and language and no evidence-based choice
was possible; and the Gemini adapter enabled native response schemas only for the
classifier, so ~6% of Authority calls returned unparseable text.

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


class TestSubjectAwareAuthorityRouting:
    @pytest.mark.parametrize("subject", ["math", "reasoning"])
    def test_qualified_families_reach_the_qualified_authority(
        self, subject: str
    ) -> None:
        decision = resolve_route(
            RouteRequest(request_id="auth-1", subject=subject, task_role="verifier")
        )
        assert decision.model == "gemini_3_7_flash"
        assert decision.route_source == "exact"

    def test_general_does_not_inherit_a_family_authority(self) -> None:
        """Families qualify one at a time; `general` never inherits one of theirs."""
        decision = resolve_route(
            RouteRequest(request_id="auth-2", subject="general", task_role="verifier")
        )
        assert decision.model not in {"gemini_3_7_flash", "openai_gpt_5_6_luna"}
        assert decision.route_id == "general.verifier.default"
        assert decision.route_source == "exact"

    def test_doubt_solver_verifier_is_untouched(
        self, registry: LlmConfigRegistry
    ) -> None:
        route = registry.get_route("general", "verifier", "default")
        assert route is not None
        # Retired o4-mini replaced by Terra; prompt and budget stay the Doubt Solver's own.
        assert route.model == "openai_gpt_5_6_terra"
        assert route.prompt == "answer_correctness_verifier.md"

    def test_no_topic_level_verifier_routes_exist(
        self, registry: LlmConfigRegistry
    ) -> None:
        """Routing dimensions stay subject/role/difficulty/language."""
        subjects = {
            subject
            for (subject, task_role, _difficulty) in registry.route_map
            if task_role == "verifier"
        }
        assert subjects == {"general", "math", "reasoning", "factual", "english"}

    def test_retired_models_serve_no_verifier_route(
        self, registry: LlmConfigRegistry
    ) -> None:
        retired = {"openai_gpt_5_6_terra", "glm5_bedrock"}
        # Terra's retirement is for the Practice Answer Authority; ``general`` is the
        # Doubt Solver answer-correctness verifier, a different contract.
        serving = {
            entry.model
            for (subject, task_role, _d), entry in registry.route_map.items()
            if task_role == "verifier" and subject != "general"
        }
        assert serving.isdisjoint(retired)

    def test_authority_routes_send_no_undeclared_reasoning_effort(
        self, registry: LlmConfigRegistry
    ) -> None:
        for subject in ("math", "reasoning"):
            route = registry.get_route(subject, "verifier", "default")
            assert route is not None
            model = registry.model_map[route.model]
            if not model.supports_reasoning:
                assert "reasoning_effort" not in route.provider_options


class TestSubjectToAuthorityFamilyMapping:
    """Every supported product subject must resolve to an Authority family.

    Ten factual subjects previously collapsed onto `general`, which is the shared
    cross-feature entry and not a Practice family — that mismatch, not any model, is
    why they had no Authority. The mapping is routing-domain only: no classifier, no
    per-subject models, no topic routes.
    """

    EXPECTED = {
        "math": "math",
        "reasoning": "reasoning",
        "english": "english",
        "science": "factual",
        "physics": "factual",
        "chemistry": "factual",
        "biology": "factual",
        "history": "factual",
        "geography": "factual",
        "polity": "factual",
        "economics": "factual",
        "computer_science": "factual",
        "general": "general",
        "other": "general",
    }

    def test_every_supported_subject_is_mapped(self) -> None:
        from features.practice_generation.metadata_normalization import (
            _SUPPORTED_SUBJECTS,
        )

        assert set(self.EXPECTED) == set(_SUPPORTED_SUBJECTS)

    @pytest.mark.parametrize("subject,family", sorted(EXPECTED.items()))
    def test_subject_resolves_to_its_family(self, subject: str, family: str) -> None:
        from services.llm.orchestration.route_resolver import normalize_subject

        assert normalize_subject(subject) == family

    def test_other_is_not_a_dumping_ground(self) -> None:
        """`other` must not inherit a family it was never qualified for."""
        from services.llm.orchestration.route_resolver import normalize_subject

        assert normalize_subject("other") != "factual"

    @pytest.mark.parametrize("subject", ["", "   ", "nonsense", "not_a_subject"])
    def test_unknown_input_never_gains_a_family(self, subject: str) -> None:
        from services.llm.orchestration.route_resolver import normalize_subject

        assert normalize_subject(subject) == "general"


class TestPracticeFailsClosedWithoutQualifiedAuthority:
    """Practice must never verify with an Authority that has not passed the gate."""

    @pytest.mark.parametrize("subject", ["math", "reasoning"])
    def test_qualified_families_resolve(self, subject: str) -> None:
        from features.practice_generation.providers import (
            _require_qualified_authority,
        )

        route_id = _require_qualified_authority(
            request_id="fc-1", subject=subject, language="english"
        )
        assert route_id == f"{subject}.verifier.default"

    @pytest.mark.parametrize(
        "subject",
        ["general", "other", "not_a_subject"],
    )
    def test_unqualified_or_unknown_subjects_fail_closed(self, subject: str) -> None:
        """Includes unknown input: normalize_subject collapses it onto the shared
        general entry, which is not a Practice family Authority.

        Science/history/polity left this set when the factual family qualified, and
        English left it when Gemini was qualified as its Authority; all of them now
        normalize onto a family that has one.
        """
        from features.practice_generation.providers import (
            PracticeAuthorityUnavailableError,
            _require_qualified_authority,
        )

        with pytest.raises(PracticeAuthorityUnavailableError, match="NOT_QUALIFIED"):
            _require_qualified_authority(
                request_id="fc-2", subject=subject, language="english"
            )

    def test_the_guard_names_no_model_and_no_subject_allowlist(self) -> None:
        """Adding a qualified family must be config-only, never a code change."""
        from pathlib import Path

        source = Path("features/practice_generation/providers.py").read_text()
        guard = source[source.index("def _require_qualified_authority") :]
        guard = guard[: guard.index("class RoutedQuestionVerifier")]
        for name in ("gemini", "openai", "o4_mini", "math", "reasoning", "english"):
            assert name not in guard.lower().replace("general", "")


class TestGeminiNativeVerifierSchema:
    def test_the_verifier_role_has_a_native_response_schema(self) -> None:
        from services.llm.providers.gemini_provider import _NATIVE_RESPONSE_SCHEMAS

        assert "verifier" in _NATIVE_RESPONSE_SCHEMAS
        assert "classifier" in _NATIVE_RESPONSE_SCHEMAS
        assert "generator" in _NATIVE_RESPONSE_SCHEMAS

    def test_schemas_are_keyed_by_role_never_by_model(self) -> None:
        """No model-specific repair branch may exist in the adapter."""
        from pathlib import Path

        source = Path("services/llm/providers/gemini_provider.py").read_text()
        assert "gemini-3.7" not in source
        assert "gemini_3_7_flash" not in source

    def test_the_verifier_schema_carries_the_v2_contract(self) -> None:
        from services.llm.providers.gemini_provider import _verifier_response_schema

        schema = _verifier_response_schema()
        assert set(schema["required"]) == {
            "schema_version",
            "generation_item_id",
            "slot_id",
            "decision",
            "valid_option_ids",
            "reason_codes",
            # Present on VerificationResult and required by
            # _enforce_evidence_citations for fresh-evidence questions; missing from
            # this schema's previous hand-written copy before it started delegating
            # to the canonical practice_verifier_generation_schema().
            "evidence_urls",
        }
        options = schema["properties"]["valid_option_ids"]
        assert options["items"]["enum"] == ["0", "1", "2", "3"]
        # Item-count bounds (maxItems) are no longer encoded in the provider schema —
        # the canonical schema constrains shape only; VerificationResult's own
        # max_length=4 stays the authority, unchanged after parsing.

    def test_the_verifier_schema_avoids_constructs_the_api_rejects(self) -> None:
        """Derived-from-model schemas emit anyOf/$ref, which this API refuses."""
        import json

        from services.llm.providers.gemini_provider import _verifier_response_schema

        raw = json.dumps(_verifier_response_schema())
        for construct in ("anyOf", "$ref", "$defs", "allOf"):
            assert construct not in raw

    def test_the_wire_schema_still_validates_against_the_canonical_model(self) -> None:
        """The canonical model stays the authority — it re-validates every response."""
        from features.practice_generation.schemas import VerificationResult

        result = VerificationResult.model_validate(
            {
                "schema_version": "2",
                "generation_item_id": "item-slot-001",
                "slot_id": "slot-001",
                "decision": "ACCEPT",
                "valid_option_ids": ["2"],
                "reason_codes": ["SINGLE_VALID_OPTION"],
            }
        )
        assert result.is_approved is True
        assert result.independently_solved_option_id == "2"


class TestGeminiNativeGeneratorSchema:
    def test_the_generator_schema_carries_the_v2_contract(self) -> None:
        from services.llm.providers.gemini_provider import _generator_response_schema

        schema = _generator_response_schema()
        question_schema = schema["properties"]["questions"]["items"]
        assert set(question_schema["required"]) == {
            "schema_version", "bucket_id", "slot_id", "question", "question_type",
            "options", "correct_option_id", "subject", "topic", "difficulty",
        }

    def test_the_generator_schema_validates_against_the_canonical_model(self) -> None:
        from features.practice_generation.schemas import GeneratedQuestion

        question = GeneratedQuestion.model_validate(
            {
                "schema_version": "2", "bucket_id": "b1", "slot_id": "slot-001",
                "question": "What is 2+2?", "question_type": "mcq",
                "options": [
                    {"option_id": "0", "value": "3"}, {"option_id": "1", "value": "4"},
                    {"option_id": "2", "value": "5"}, {"option_id": "3", "value": "6"},
                ],
                "correct_option_id": "1", "subject": "math", "topic": "addition",
                "difficulty": "basic",
            }
        )
        assert question.correct_answer == "4"


class TestGeminiNativeSchemaDoubtSolverIsolation:
    """generator/verifier are shared task_roles with Doubt Solver; the native schema
    must fire only for Practice's own schema-v2 prompts, gated on the exact prompt
    path rather than intent.

    Gating on intent=="practice" alone is unsafe: the classifier maps a
    practice_question intent to the literal string "practice"
    (academic_classifier.ACADEMIC_INTENT_MAP) even for requests that fall through to
    Doubt Solver's ordinary free-text "generate" node (Practice launch ineligible or
    disabled), so a naive intent gate would force a free-text Doubt Solver answer into
    the strict Practice MCQ JSON schema. Prompt-path gating also correctly excludes
    the legacy schema-v1 Practice generator/verifier prompts, which use a different
    wire shape than schema-v2.
    """

    @staticmethod
    def _request(task_role: str, prompt: str, intent: str | None = None):
        from types import SimpleNamespace

        return SimpleNamespace(
            route_decision=SimpleNamespace(
                task_role=task_role, prompt=prompt, intent=intent
            )
        )

    @pytest.mark.parametrize(
        ("task_role", "prompt"),
        [
            ("generator", "practice_generation/question_generator_v2.md"),
            ("generator", "practice_generation/question_generator_factual.md"),
            ("generator", "practice_generation/question_repair.md"),
            ("generator", "practice_generation/question_regenerator.md"),
            ("verifier", "practice_generation/question_verifier_v2.md"),
        ],
    )
    def test_v2_practice_prompts_activate_the_native_schema(
        self, task_role: str, prompt: str
    ) -> None:
        from services.llm.providers.gemini_provider import _active_native_schema_builder

        assert _active_native_schema_builder(self._request(task_role, prompt)) is not None

    @pytest.mark.parametrize(
        ("task_role", "prompt"),
        [
            ("generator", "subjects/math_generator.md"),
            ("generator", "subjects/general_generator.md"),
            ("verifier", "answer_correctness_verifier.md"),
        ],
    )
    def test_doubt_solver_prompts_never_activate_the_native_schema_even_with_intent_practice(
        self, task_role: str, prompt: str
    ) -> None:
        from services.llm.providers.gemini_provider import _active_native_schema_builder

        request = self._request(task_role, prompt, intent="practice")
        assert _active_native_schema_builder(request) is None

    @pytest.mark.parametrize(
        ("task_role", "prompt"),
        [
            ("generator", "practice_generation/question_generator.md"),
            ("verifier", "practice_generation/question_verifier.md"),
        ],
    )
    def test_legacy_v1_practice_prompts_never_activate_the_native_schema(
        self, task_role: str, prompt: str
    ) -> None:
        from services.llm.providers.gemini_provider import _active_native_schema_builder

        request = self._request(task_role, prompt, intent="practice")
        assert _active_native_schema_builder(request) is None

    def test_classifier_is_unaffected_by_the_prompt_gate(self) -> None:
        """classifier has exactly one caller and needs no prompt gate."""
        from services.llm.providers.gemini_provider import _active_native_schema_builder

        request = self._request("classifier", "classification_semantics.md")
        assert _active_native_schema_builder(request) is not None
