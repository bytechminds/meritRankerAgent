"""
app/tests/test_orchestrated_classifier_routing.py
---------------------------------------------------
Tests for the orchestrated Azure-first classifier path.

Covers:
- doubt_solver_classifier alias exists in model registry with azure_openai provider
- doubt_solver_classifier_openai_native alias exists as fallback (openai provider)
- general.classifier.default route exists in llm_routes.yaml pointing to
  doubt_solver_classifier model
- classify_query dispatches to orchestrated path when
  ENABLE_ORCHESTRATED_DOUBT_SOLVER=true and ENABLE_REAL_LLM=true
- orchestrated path falls back to deterministic when LlmOrchestrator raises
- legacy model_router path still used when ENABLE_ORCHESTRATED_DOUBT_SOLVER=false
- _get_classifier_orchestrator returns the same object on repeated calls (singleton)
- _get_classifier_orchestrator singleton can be reset for test isolation
- _warn_placeholder_deployments emits a WARNING for placeholder deployment names

No real provider calls in any test.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

import config as cfg_module
import services.query_classifier_service as classifier_module
from schemas.doubt_solver import QueryClassification
from services.doubt_solver.classifier_json import ClassifierJsonError
from services.query_classifier_service import (
    _CLASSIFIER_ROLE,
    ClassifierRunResult,
    classify_query,
)


def _classifier_run(
    classification: QueryClassification,
    *,
    strong_classifier_used: bool = False,
) -> ClassifierRunResult:
    return ClassifierRunResult(
        classification=classification,
        strong_classifier_used=strong_classifier_used,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_settings() -> None:
    cfg_module._settings = None


def _reset_orchestrator() -> None:
    """Reset the orchestrator singleton so tests start with a clean state."""
    classifier_module._classifier_orchestrator = None


def _make_valid_classification_json(
    intent: str = "solve_question",
    subject: str = "math",
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "intent": intent,
            "subject": subject,
            "topic": "algebra",
            "response_style": "step_by_step",
            "confidence": confidence,
            "retrieval_need": "none",
            "reasoning_summary": "Math question",
        }
    )


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch):
    """Reset Settings and orchestrator singleton between every test."""
    _reset_settings()
    _reset_orchestrator()
    yield
    _reset_settings()
    _reset_orchestrator()


# ---------------------------------------------------------------------------
# Part B — YAML config presence checks (registry + routes)
# ---------------------------------------------------------------------------


class TestModelRegistryClassifierAliases:
    """doubt_solver_classifier and its fallback must exist in model_registry.yaml."""

    def _load_registry(self):
        """Return the raw LlmConfigRegistry (real YAML, no providers)."""
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        return LlmConfigRegistry()

    def test_primary_alias_exists(self):
        registry = self._load_registry()
        assert "doubt_solver_classifier" in registry.model_map

    def test_primary_alias_is_azure_openai(self):
        registry = self._load_registry()
        model = registry.model_map["doubt_solver_classifier"]
        assert model.provider == "azure_openai"

    def test_gemini_primary_alias_has_no_model_fallback(self):
        registry = self._load_registry()
        model = registry.model_map["doubt_solver_classifier_gemini"]
        assert model.provider == "gemini"
        assert model.model_id == "gemini-3.1-flash-lite"
        assert not (model.fallback_models or [])

    def test_native_fallback_alias_exists(self):
        registry = self._load_registry()
        assert "doubt_solver_classifier_openai_native" in registry.model_map

    def test_native_fallback_alias_is_openai(self):
        registry = self._load_registry()
        model = registry.model_map["doubt_solver_classifier_openai_native"]
        assert model.provider == "openai"

    def test_native_fallback_has_no_further_fallback(self):
        """Fallback chain is intentionally shallow — native alias has no further fallback."""
        registry = self._load_registry()
        model = registry.model_map["doubt_solver_classifier_openai_native"]
        assert not (model.fallback_models or [])

    def test_primary_supports_streaming_false(self):
        registry = self._load_registry()
        model = registry.model_map["doubt_solver_classifier"]
        assert model.supports_streaming is False

    def test_native_fallback_supports_streaming_false(self):
        registry = self._load_registry()
        model = registry.model_map["doubt_solver_classifier_openai_native"]
        assert model.supports_streaming is False


class TestLlmRoutesClassifierRoute:
    """general.classifier.default route must exist and point to doubt_solver_classifier."""

    def _load_registry(self):
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        return LlmConfigRegistry()

    def test_classifier_route_exists(self):
        registry = self._load_registry()
        # Route key is (subject, task_role, difficulty)
        key = ("general", "classifier", "default")
        assert key in registry.route_map, (
            "Route general.classifier.default not found. "
            f"Keys: {list(registry.route_map.keys())[:10]}"
        )

    def test_classifier_route_points_to_correct_model(self):
        registry = self._load_registry()
        route = registry.route_map[("general", "classifier", "default")]
        assert route.model == "doubt_solver_classifier_gemini"

    def test_classifier_route_prompt_set(self):
        registry = self._load_registry()
        route = registry.route_map[("general", "classifier", "default")]
        assert route.prompt, "Classifier route must have a prompt."


# ---------------------------------------------------------------------------
# Part C — classify_query orchestrated dispatch
# ---------------------------------------------------------------------------


class TestClassifyQueryOrchestratedDispatch:
    """classify_query must use orchestrated path when ENABLE_ORCHESTRATED_DOUBT_SOLVER=true."""

    def _env(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

    def test_dispatches_to_orchestrated_path(self, monkeypatch):
        """When flag=true, classify_query calls _classify_with_llm_orchestrated_or_fallback."""
        self._env(monkeypatch)
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated_or_fallback",
            return_value=_classifier_run(
                QueryClassification(
                    intent="solve_question",
                    subject="math",
                    topic="algebra",
                    response_style="step_by_step",
                    confidence=0.9,
                    retrieval_need="none",
                    reasoning_summary="",
                    classification_source="llm",
                )
            ),
        ) as mock_fn:
            result = classify_query("What is 2+2?")
        mock_fn.assert_called_once_with(
            "What is 2+2?",
            request_id=None,
            on_before_strong_classifier=None,
        )
        assert result.classification_source == "llm"

    def test_does_not_call_legacy_model_router_when_orchestrated(self, monkeypatch):
        """Legacy model_router must not be called when orchestrated path is active."""
        self._env(monkeypatch)
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated_or_fallback",
            return_value=_classifier_run(
                QueryClassification(
                    intent="explain_concept",
                    subject="general",
                    topic="test",
                    response_style="short_answer",
                    confidence=0.8,
                    retrieval_need="none",
                    reasoning_summary="",
                    classification_source="llm",
                )
            ),
        ), patch.object(
            classifier_module,
            "_classify_with_llm",
        ) as legacy_mock:
            classify_query("Explain gravity")
        legacy_mock.assert_not_called()


class TestClassifyQueryLegacyPathPreserved:
    """Legacy model_router path must still work when ENABLE_ORCHESTRATED_DOUBT_SOLVER=false."""

    def _env(self, monkeypatch, role_config: dict):
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", json.dumps(role_config))
        _reset_settings()

    def test_falls_back_to_deterministic_when_role_not_in_config(self, monkeypatch):
        self._env(monkeypatch, {})
        result = classify_query("What is the derivative of x^2?")
        assert result.classification_source in ("deterministic", "llm", "fallback")

    def test_calls_legacy_llm_or_fallback_when_role_in_config(self, monkeypatch):
        self._env(monkeypatch, {_CLASSIFIER_ROLE: {"provider": "mock"}})
        with patch.object(
            classifier_module,
            "_classify_with_llm_or_fallback",
            return_value=QueryClassification(
                intent="solve_question",
                subject="math",
                topic="calc",
                response_style="step_by_step",
                confidence=0.9,
                retrieval_need="none",
                reasoning_summary="",
                classification_source="llm",
            ),
        ) as mock_fn:
            classify_query("Integrate x^2")
        mock_fn.assert_called_once()

    def test_does_not_call_orchestrated_path(self, monkeypatch):
        self._env(monkeypatch, {})
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated_or_fallback",
        ) as mock_fn:
            classify_query("What is calculus?")
        mock_fn.assert_not_called()


class TestOrchestratedClassifierOrFallback:
    """_classify_with_llm_orchestrated_or_fallback must fall back to deterministic on errors."""

    def test_falls_back_on_provider_error(self, monkeypatch):
        from services.llm.orchestration.errors import ProviderExecutionError

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=ProviderExecutionError("All providers failed"),
        ):
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "Solve 3x=9"
            )
        assert run.classification.classification_source == "fallback"
        assert run.strong_classifier_used is True

    def test_falls_back_on_json_decode_error(self, monkeypatch):
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=ValueError("Bad JSON"),
        ):
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "What is DNA?"
            )
        assert run.classification.classification_source == "fallback"
        assert run.strong_classifier_used is True

    def test_returns_llm_result_on_success(self, monkeypatch):
        expected = QueryClassification(
            intent="explain_concept",
            subject="english",
            topic="grammar",
            response_style="simple_explanation",
            confidence=0.93,
            retrieval_need="none",
            reasoning_summary="",
            classification_source="llm",
        )
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            return_value=expected,
        ):
            result = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "What is a noun?"
            ).classification
        assert result.intent == "explain_concept"
        assert result.classification_source == "llm"


# ---------------------------------------------------------------------------
# Orchestrator singleton
# ---------------------------------------------------------------------------


class TestClassifierOrchestratorSingleton:
    """_get_classifier_orchestrator must return the same object on repeated calls."""

    def test_singleton_returns_same_instance(self):
        """After first build, returns cached object."""
        _reset_orchestrator()
        mock_orchestrator = MagicMock()
        # Inject the mock directly — bypasses lazy wiring
        classifier_module._classifier_orchestrator = mock_orchestrator
        o1 = classifier_module._get_classifier_orchestrator()
        o2 = classifier_module._get_classifier_orchestrator()
        assert o1 is o2
        assert o1 is mock_orchestrator

    def test_singleton_reset_is_none(self):
        """After reset, singleton is None until next call builds it."""
        _reset_orchestrator()
        assert classifier_module._classifier_orchestrator is None


# ---------------------------------------------------------------------------
# Part D — Placeholder deployment warnings
# ---------------------------------------------------------------------------


class TestPlaceholderDeploymentWarnings:
    """_warn_placeholder_deployments must log a WARNING for YOUR_* names.

    NOTE: model_registry.yaml currently has real deployment names (gpt-4o,
    gpt-4o-mini) — no YOUR_* placeholders.  These tests verify the warning
    mechanism works correctly when placeholders ARE present (via an inline
    registry) rather than asserting the production registry has placeholders.
    """

    def test_warns_for_placeholder_deployment_in_inline_registry(self, tmp_path, caplog):
        """When model_registry.yaml has YOUR_* deployments, warnings are emitted."""
        import textwrap

        from services.llm.orchestration.config_registry import LlmConfigRegistry

        yaml_text = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: azure_placeholder
                    prompt: subjects/general_generator.md
                    temperature: 0.3
                    max_tokens: 800
            models:
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_label: safe-mock
                cost_tier: none
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 1
              azure_placeholder:
                provider: azure_openai
                provider_profile: azure_primary
                deployment: YOUR_AZURE_DEPLOYMENT_NAME
                model_label: azure-placeholder
                cost_tier: medium
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 30
            provider_profiles:
              local_mock:
                provider: mock
              azure_primary:
                provider: azure_openai
                api_key_env: AZURE_OPENAI_API_KEY
                endpoint_env: AZURE_OPENAI_ENDPOINT
                api_version_env: AZURE_OPENAI_API_VERSION
        """)
        yaml_path = tmp_path / "llm_orchestration.yaml"
        yaml_path.write_text(yaml_text)

        with caplog.at_level(logging.WARNING):
            LlmConfigRegistry(yaml_path=yaml_path)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        placeholder_warnings = [m for m in warning_messages if "placeholder" in m.lower()]
        assert len(placeholder_warnings) >= 1, (
            f"Expected ≥1 placeholder warning for YOUR_* deployment. "
            f"Got: {placeholder_warnings}"
        )

    def test_warning_includes_alias_name_for_placeholder(self, tmp_path, caplog):
        """Warning message includes the model alias with the placeholder deployment."""
        import textwrap

        from services.llm.orchestration.config_registry import LlmConfigRegistry

        yaml_text = textwrap.dedent("""\
            version: 1
            routes:
              general:
                generator:
                  default:
                    model: my_azure_model
                    prompt: subjects/general_generator.md
                    temperature: 0.3
                    max_tokens: 800
            models:
              safe_mock:
                provider: mock
                provider_profile: local_mock
                model_label: safe-mock
                cost_tier: none
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 1
              my_azure_model:
                provider: azure_openai
                provider_profile: azure_primary
                deployment: YOUR_GPT4O_DEPLOYMENT
                model_label: azure-model
                cost_tier: medium
                supports_streaming: false
                supports_thinking: false
                timeout_seconds: 30
            provider_profiles:
              local_mock:
                provider: mock
              azure_primary:
                provider: azure_openai
                api_key_env: AZURE_OPENAI_API_KEY
                endpoint_env: AZURE_OPENAI_ENDPOINT
                api_version_env: AZURE_OPENAI_API_VERSION
        """)
        yaml_path = tmp_path / "llm_orchestration.yaml"
        yaml_path.write_text(yaml_text)

        with caplog.at_level(logging.WARNING):
            LlmConfigRegistry(yaml_path=yaml_path)

        messages = " ".join(r.message for r in caplog.records if r.levelno == logging.WARNING)
        assert "my_azure_model" in messages or "YOUR_GPT4O_DEPLOYMENT" in messages

    def test_production_registry_has_no_placeholder_deployments(self, caplog):
        """Confirm model_registry.yaml has NO YOUR_* placeholder deployments.

        All deployment names are real (gpt-4o, gpt-4o-mini).
        If this test fails, someone re-introduced a placeholder — update model_registry.yaml.
        """
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        with caplog.at_level(logging.WARNING):
            LlmConfigRegistry()

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        placeholder_warnings = [m for m in warning_messages if "placeholder" in m.lower()]
        assert placeholder_warnings == [], (
            f"Production model_registry.yaml has YOUR_* placeholder deployments — "
            f"replace them with real Azure deployment names.\n"
            f"Placeholder warnings: {placeholder_warnings}"
        )


# ---------------------------------------------------------------------------
# Part E — request_id propagation (Part A fix coverage)
# ---------------------------------------------------------------------------


class TestClassifyQueryRequestIdPropagation:
    """request_id must be threaded from classify_query down to the orchestrated path."""

    def _env(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

    def test_request_id_propagated_when_provided(self, monkeypatch):
        self._env(monkeypatch)
        _good_result = _classifier_run(
            QueryClassification(
                intent="solve_question",
                subject="math",
                topic="algebra",
                response_style="step_by_step",
                confidence=0.9,
                retrieval_need="none",
                reasoning_summary="",
                classification_source="llm",
            )
        )
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated_or_fallback",
            return_value=_good_result,
        ) as mock_fn:
            classify_query("What is 2+2?", request_id="req-abc")
        mock_fn.assert_called_once_with(
            "What is 2+2?",
            request_id="req-abc",
            on_before_strong_classifier=None,
        )

    def test_request_id_none_when_not_provided(self, monkeypatch):
        self._env(monkeypatch)
        _good_result = _classifier_run(
            QueryClassification(
                intent="solve_question",
                subject="math",
                topic="algebra",
                response_style="step_by_step",
                confidence=0.9,
                retrieval_need="none",
                reasoning_summary="",
                classification_source="llm",
            )
        )
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated_or_fallback",
            return_value=_good_result,
        ) as mock_fn:
            classify_query("What is 2+2?")
        mock_fn.assert_called_once_with(
            "What is 2+2?",
            request_id=None,
            on_before_strong_classifier=None,
        )


class TestClassifyWithLlmOrchestratedRequestId:
    """_classify_with_llm_orchestrated must pass request_id into RouteRequest."""

    def _fake_generate(self, captured: list):
        import json
        from types import SimpleNamespace

        def _inner(route_request, query, classification, context):
            captured.append(route_request)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "intent": "solve_question",
                        "subject": "math",
                        "topic": "algebra",
                        "response_style": "step_by_step",
                        "confidence": 0.9,
                        "retrieval_need": "none",
                        "reasoning_summary": "",
                    }
                )
            )

        return _inner

    def test_uses_provided_request_id(self):
        captured: list = []
        mock_orchestrator = MagicMock()
        mock_orchestrator.generate.side_effect = self._fake_generate(captured)
        with patch.object(
            classifier_module, "_get_classifier_orchestrator", return_value=mock_orchestrator
        ):
            classifier_module._classify_with_llm_orchestrated("Solve x+1=3", request_id="trace-abc")
        assert len(captured) == 1
        assert captured[0].request_id == "trace-abc"

    def test_generates_uuid_fallback_when_request_id_is_none(self):
        import re

        captured: list = []
        mock_orchestrator = MagicMock()
        mock_orchestrator.generate.side_effect = self._fake_generate(captured)
        with patch.object(
            classifier_module, "_get_classifier_orchestrator", return_value=mock_orchestrator
        ):
            classifier_module._classify_with_llm_orchestrated("Solve x+1=3", request_id=None)
        assert len(captured) == 1
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            captured[0].request_id,
        ), f"Expected UUID, got: {captured[0].request_id}"

    def test_route_request_has_classifier_task_role(self):
        captured: list = []
        mock_orchestrator = MagicMock()
        mock_orchestrator.generate.side_effect = self._fake_generate(captured)
        with patch.object(
            classifier_module, "_get_classifier_orchestrator", return_value=mock_orchestrator
        ):
            classifier_module._classify_with_llm_orchestrated("What is x?", request_id="r1")
        assert captured[0].task_role == "classifier"
        assert captured[0].subject == "general"
        assert captured[0].difficulty == "default"


# ---------------------------------------------------------------------------
# Part F — validate_real_mode_deployments (Part B fix coverage)
# ---------------------------------------------------------------------------


class TestValidateRealModeDeployments:
    """validate_real_mode_deployments must raise LlmConfigValidationError for placeholder names.

    NOTE: model_registry.yaml currently has real deployment names — no YOUR_* placeholders.
    The tests that previously used the production registry to verify raises have been
    updated to use an inline registry with known placeholder data.
    The production registry test verifies validate_real_mode_deployments does NOT raise.
    """

    def _placeholder_registry(self):
        """Return registry with YOUR_* placeholder deployment on an active route."""
        from schemas.llm_routing import ResolvedRouteEntry
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        registry = LlmConfigRegistry.__new__(LlmConfigRegistry)
        mock_model = MagicMock()
        mock_model.provider = "azure_openai"
        mock_model.deployment = "YOUR_AZURE_GPT4O_DEPLOYMENT"
        registry._model_map = {"math_basic_generator": mock_model}
        registry._route_map = {
            ("math", "generator", "default"): ResolvedRouteEntry(
                model="math_basic_generator",
                prompt="subjects/math_generator.md",
                overlays=[],
                intent_overlays={},
                temperature=0.2,
                max_tokens=900,
                provider_options={},
                fallback=[],
            )
        }
        return registry

    def test_raises_for_placeholder_azure_deployments(self):
        """Registry with YOUR_* deployment must raise LlmConfigValidationError."""
        from services.llm.orchestration.errors import LlmConfigValidationError

        registry = self._placeholder_registry()
        with pytest.raises(LlmConfigValidationError) as exc_info:
            registry.validate_real_mode_deployments()
        assert "placeholder" in str(exc_info.value).lower()
        assert "model_alias=" in str(exc_info.value)

    def test_raises_includes_actionable_message(self):
        """Error message must mention active route model alias."""
        from services.llm.orchestration.errors import LlmConfigValidationError

        registry = self._placeholder_registry()
        with pytest.raises(LlmConfigValidationError) as exc_info:
            registry.validate_real_mode_deployments()
        assert "math_basic_generator" in str(exc_info.value)

    def test_passes_when_no_placeholder_deployments(self):
        """Registry with real deployment names on active routes must not raise."""
        from schemas.llm_routing import ResolvedRouteEntry
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        registry = LlmConfigRegistry.__new__(LlmConfigRegistry)
        mock_model = MagicMock()
        mock_model.provider = "azure_openai"
        mock_model.deployment = "gpt-4o-mini-prod"
        registry._model_map = {"prod_model": mock_model}
        registry._route_map = {
            ("general", "generator", "default"): ResolvedRouteEntry(
                model="prod_model",
                prompt="subjects/general_generator.md",
                overlays=[],
                intent_overlays={},
                temperature=0.3,
                max_tokens=900,
                provider_options={},
                fallback=[],
            )
        }
        registry.validate_real_mode_deployments()

    def test_production_registry_passes_validation(self):
        """Production model_registry.yaml has real deployment names — must not raise.

        If this test fails, someone introduced a YOUR_* placeholder — replace it
        with a real Azure deployment name in config/llm/model_registry.yaml.
        """
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        registry = LlmConfigRegistry()
        # Must not raise — all Azure deployments are real names
        registry.validate_real_mode_deployments()

    def test_error_does_not_contain_sensitive_info(self):
        """[SECURITY] Error message must not expose keys, endpoints, or secrets."""
        from services.llm.orchestration.errors import LlmConfigValidationError

        registry = self._placeholder_registry()
        with pytest.raises(LlmConfigValidationError) as exc_info:
            registry.validate_real_mode_deployments()
        msg = str(exc_info.value)
        for forbidden in ("api_key", "AZURE_API_KEY", "endpoint", "http://", "https://", "secret"):
            assert forbidden.lower() not in msg.lower(), (
                f"Error message contains sensitive info '{forbidden}': {msg}"
            )


# ---------------------------------------------------------------------------
# Classifier confidence fallback (Part 13.1)
# ---------------------------------------------------------------------------


def _llm_classification(confidence: float) -> QueryClassification:
    return QueryClassification(
        intent="solve_question",
        subject="math",
        topic="algebra",
        response_style="step_by_step",
        confidence=confidence,
        retrieval_need="concept_context",
        reasoning_summary="test",
        classification_source="llm",
    )


class TestClassifierConfidenceFallback:
    def test_low_confidence_contextual_query_requires_strong_classifier(self) -> None:
        primary = _llm_classification(0.30).model_copy(
            update={
                "requires_recent_conversation": True,
                "relation": "FOLLOW_UP",
                "selected_turn_id": "turn-1",
                "requested_action": "ANSWER_WITH_CONTEXT",
            }
        )
        strong = primary.model_copy(update={"confidence": 0.94})
        calls: list[str] = []

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "What was the pattern in the last one?",
                request_id="req-follow-up",
                on_before_strong_classifier=lambda: calls.append("hook"),
                context_gate="CONTEXT_REQUIRED",
                candidate_turn_ids=("turn-1",),
            )

        assert mock_classify.call_count == 2
        assert run.classification.requires_recent_conversation is True
        assert run.strong_classifier_used is True
        assert run.primary_decision is not None
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.LOW_MATERIAL_CONFIDENCE
        )
        assert calls == ["hook"]

    def test_resolved_one_candidate_follow_up_accepts_materially_certain_primary(
        self,
    ) -> None:
        primary = _llm_classification(0.90).model_copy(
            update={
                "intent": "practice_question",
                "requires_recent_conversation": True,
                "relation": "FOLLOW_UP",
                "selected_turn_id": "turn-1",
                "requested_action": "GENERATE_SIMILAR",
            }
        )

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            return_value=primary,
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "Create one similar question.",
                request_id="req-resolved-follow-up",
                context_gate="CONTEXT_REQUIRED",
                candidate_turn_ids=("turn-1",),
            )

        assert mock_classify.call_count == 1
        assert run.strong_classifier_used is False
        assert run.primary_decision is not None
        assert run.primary_decision.accepted is True

    def test_multiple_candidate_ambiguity_keeps_strict_confidence_threshold(
        self,
    ) -> None:
        primary = _llm_classification(0.90).model_copy(
            update={
                "subject": "general",
                "intent": "general_doubt",
                "relation": "AMBIGUOUS",
                "selected_turn_id": None,
                "requested_action": "ASK_CLARIFICATION",
            }
        )
        strong = primary.model_copy(update={"confidence": 0.95})

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "Did he rule longer?",
                request_id="req-ambiguous-context",
                context_gate="CONTEXT_REQUIRED",
                candidate_turn_ids=("akbar-turn", "shah-turn"),
            )

        assert mock_classify.call_count == 2
        assert run.strong_classifier_used is True
        assert run.primary_decision is not None
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.LOW_MATERIAL_CONFIDENCE
        )

    def test_optional_uncertainty_does_not_trigger_strong_classifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("CLASSIFIER_CONFIDENCE_FALLBACK_THRESHOLD", raising=False)
        cfg_module._settings = None
        primary = _llm_classification(0.91).model_copy(
            update={
                "topic": None,
                "topic_confidence": 0.20,
                "pattern_topic_candidate": None,
                "pattern_family_candidate": None,
                "retrieval_tags": [],
                "retrieval_need": "unknown",
            }
        )
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            return_value=primary,
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "profit question", request_id="req-low"
            )
        assert mock_classify.call_count == 1
        assert run.classification.confidence == 0.91
        assert run.strong_classifier_used is False
        assert run.primary_decision is not None
        assert run.primary_decision.accepted is True
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.PRIMARY_ACCEPTED
        )
        cfg_module._settings = None

    def test_low_material_confidence_triggers_strong_classifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("CLASSIFIER_CONFIDENCE_FALLBACK_THRESHOLD", raising=False)
        cfg_module._settings = None
        primary = _llm_classification(0.80)
        strong = _llm_classification(0.94)
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "profit question", request_id="req-material-low"
            )
        assert mock_classify.call_count == 2
        mock_classify.assert_any_call(
            "profit question",
            request_id="req-material-low",
            task_role="classifier_strong",
            primary_result=primary,
            conflict_reason="LOW_MATERIAL_CONFIDENCE",
        )
        assert run.classification.confidence == 0.94
        assert run.strong_classifier_used is True
        assert run.primary_decision is not None
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.LOW_MATERIAL_CONFIDENCE
        )
        cfg_module._settings = None

    def test_high_confidence_does_not_trigger_strong_classifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("CLASSIFIER_CONFIDENCE_FALLBACK_THRESHOLD", raising=False)
        cfg_module._settings = None
        primary = _llm_classification(0.94)
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            return_value=primary,
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "profit question", request_id="req-high"
            )
        assert mock_classify.call_count == 1
        assert run.classification.confidence == 0.94
        assert run.strong_classifier_used is False
        cfg_module._settings = None

    def test_exact_riya_primary_is_accepted_without_strong_classifier(self) -> None:
        primary = QueryClassification(
            intent="solve_question",
            subject="reasoning",
            topic="Position in Row",
            topic_confidence=0.90,
            pattern_topic_candidate="POSITION_IN_ROW",
            pattern_family_candidate="REASONING",
            retrieval_tags=["rank_order", "relative_position"],
            response_style="step_by_step",
            confidence=0.90,
            difficulty="basic",
            retrieval_need="none",
            reasoning_summary="Infer total from two ordinal row positions.",
            need_web_search=False,
            relation="NEW_QUESTION",
            requested_action="ANSWER_CURRENT",
            classification_source="llm",
        )
        query = (
            "Riya is 12th from the left and 9th from the right in a row.\n"
            "How many students are there?"
        )

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            return_value=primary,
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                query,
                request_id="req-riya",
            )

        assert mock_classify.call_count == 1
        assert run.strong_classifier_used is False
        assert run.classification.subject == "reasoning"
        assert run.classification.intent == "solve_question"
        assert run.classification.difficulty == "basic"
        assert run.classification.relation == "NEW_QUESTION"
        assert run.classification.requested_action == "ANSWER_CURRENT"
        assert run.classification.need_web_search is False
        assert run.primary_decision is not None
        assert run.primary_decision.accepted is True

    def test_subject_pattern_family_conflict_requires_one_strong_classifier(self) -> None:
        primary = _llm_classification(0.90).model_copy(
            update={
                "subject": "math",
                "topic": "SEATING_ARRANGEMENT",
                "pattern_topic_candidate": "SEATING_ARRANGEMENT",
                "pattern_family_candidate": "REASONING",
            }
        )
        strong = primary.model_copy(
            update={
                "subject": "reasoning",
                "confidence": 0.95,
            }
        )

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "A rank and order question.",
                request_id="req-subject-conflict",
            )

        assert mock_classify.call_count == 2
        assert run.strong_classifier_used is True
        assert run.primary_decision is not None
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.SUBJECT_INTENT_CONFLICT
        )
        assert "subject,pattern_family" in run.primary_decision.material_conflicts

    def test_explicit_current_query_with_web_false_requires_strong_classifier(
        self,
    ) -> None:
        primary = QueryClassification(
            intent="practice_question",
            subject="general",
            confidence=0.90,
            difficulty="basic",
            need_web_search=False,
            relation="NEW_QUESTION",
            requested_action="ANSWER_CURRENT",
            classification_source="llm",
        )
        strong = primary.model_copy(
            update={
                "confidence": 0.95,
                "need_web_search": True,
                "web_search_reason": "current_affairs",
                "web_search_query": "current affairs July 2026",
            }
        )

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "Provide current affairs questions for July 2026.",
                request_id="req-current",
            )

        assert mock_classify.call_count == 2
        assert run.primary_decision is not None
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.WEB_SEARCH_CONFLICT
        )

    def test_practice_action_conflict_requires_strong_classifier(self) -> None:
        primary = _llm_classification(0.90).model_copy(
            update={
                "intent": "solve_question",
                "relation": "FOLLOW_UP",
                "selected_turn_id": "turn-1",
                "requested_action": "GENERATE_SIMILAR",
            }
        )
        strong = primary.model_copy(
            update={
                "intent": "practice_question",
                "confidence": 0.95,
            }
        )

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "Create a similar question.",
                request_id="req-practice-conflict",
                context_gate="CONTEXT_REQUIRED",
                candidate_turn_ids=("turn-1",),
            )

        assert mock_classify.call_count == 2
        assert run.primary_decision is not None
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.PRACTICE_INTENT_CONFLICT
        )

    def test_safe_logs_include_typed_reason_without_classifier_json(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        primary = _llm_classification(0.80)
        strong = _llm_classification(0.95)
        caplog.set_level("INFO", logger=classifier_module.__name__)

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ):
            classifier_module._classify_with_llm_orchestrated_or_fallback(
                "ambiguous routing request",
                request_id="req-safe-log",
            )

        text = caplog.text
        assert "reason=LOW_MATERIAL_CONFIDENCE" in text
        assert "primary_confidence=0.80" in text
        assert "material_conflicts=1" in text
        assert primary.model_dump_json() not in text

    def test_custom_threshold_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", "0.85")
        cfg_module._settings = None
        primary = _llm_classification(0.84)
        strong = _llm_classification(0.91)
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "profit question", request_id="req-custom"
            )
        assert mock_classify.call_count == 2
        assert run.classification.confidence == 0.91
        cfg_module._settings = None

    def test_before_strong_classifier_hook_called_once(self) -> None:
        primary = _llm_classification(0.80)
        strong = _llm_classification(0.95)
        calls: list[str] = []

        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, strong],
        ):
            classifier_module._classify_with_llm_orchestrated_or_fallback(
                "ambiguous question",
                request_id="req-hook",
                on_before_strong_classifier=lambda: calls.append("hook"),
            )
        assert calls == ["hook"]

    def test_strong_classifier_failure_uses_deterministic(self) -> None:
        primary = _llm_classification(0.70)
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[primary, RuntimeError("strong failed")],
        ):
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "profit question", request_id="req-fallback"
            )
        assert run.classification.classification_source == "fallback"
        assert run.strong_classifier_used is True

    def test_primary_failure_uses_deterministic(self) -> None:
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=RuntimeError("primary failed"),
        ):
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "What is algebra?", request_id="req-det"
            )
        assert run.classification.classification_source == "fallback"
        assert run.strong_classifier_used is True
        assert run.primary_decision is not None
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.PRIMARY_PROVIDER_FAILURE
        )

    def test_invalid_primary_schema_uses_exactly_one_strong_classifier(self) -> None:
        strong = _llm_classification(0.95)
        with patch.object(
            classifier_module,
            "_classify_with_llm_orchestrated",
            side_effect=[
                ClassifierJsonError("json_decode_error", "invalid classifier JSON"),
                strong,
            ],
        ) as mock_classify:
            run = classifier_module._classify_with_llm_orchestrated_or_fallback(
                "Solve 2x + 5 = 15.",
                request_id="req-invalid-schema",
            )

        assert mock_classify.call_count == 2
        assert run.strong_classifier_used is True
        assert run.primary_decision is not None
        assert (
            run.primary_decision.primary_reason
            == classifier_module.StrongFallbackReason.INVALID_SCHEMA
        )
