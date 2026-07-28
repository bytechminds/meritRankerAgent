"""
app/tests/test_query_classifier_service.py
--------------------------------------------
Unit tests for services/query_classifier_service.py.

Tests cover:
- deterministic path when ENABLE_REAL_LLM=false
- LLM path with monkeypatched model_router (valid JSON response)
- malformed JSON → fallback
- invalid enum value in JSON → fallback
- model_router raises exception → fallback
- classification_source is set correctly for all paths
- deterministic path when ENABLE_REAL_LLM=true but role not in config
- no real network calls in any test

Settings singleton is reset between tests to honour monkeypatched env vars.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import config as cfg_module
import services.query_classifier_service as classifier_service
from schemas.conversation import ConversationCandidateCard
from schemas.doubt_solver import QueryClassification
from services.query_classifier_service import (
    _CLASSIFIER_ROLE,
    _build_classifier_input,
    _classifier_prompt_character_budget,
    _classify_deterministic,
    _load_classifier_prompt,
    apply_classification_policy,
    apply_classification_sanity,
    classify_query,
    get_classifier_confidence_threshold,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_settings():
    """Reset the Settings singleton so monkeypatched env vars take effect."""
    cfg_module._settings = None


def _make_valid_llm_response_content(
    intent: str = "solve_question",
    subject: str = "math",
    confidence: float = 0.9,
) -> str:
    """Return a JSON string that passes QueryClassification validation."""
    return json.dumps(
        {
            "intent": intent,
            "subject": subject,
            "topic": "percentage",
            "topic_confidence": 0.88,
            "pattern_topic_candidate": "PERCENTAGE",
            "pattern_family_candidate": None,
            "retrieval_tags": ["ratio_parts", "weighted_percentage"],
            "difficulty": "intermediate",
            "response_style": "step_by_step",
            "confidence": confidence,
            "retrieval_need": "none",
            "reasoning_summary": "Query is a calculation task.",
        }
    )


# ---------------------------------------------------------------------------
# Deterministic path (ENABLE_REAL_LLM=false)
# ---------------------------------------------------------------------------


class TestDeterministicPath:
    def test_returns_result_when_real_llm_false(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

        result = classify_query("What is 20% of 500?")

        assert isinstance(result, QueryClassification)
        assert result.classification_source == "deterministic"
        _reset_settings()

    def test_intent_detected_correctly(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

        result = classify_query("Calculate the percentage gain")

        assert result.intent == "solve_question"
        assert result.classification_source == "deterministic"
        _reset_settings()

    def test_general_doubt_fallback_intent(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

        result = classify_query("xyzzy random gibberish")

        assert result.intent == "general_doubt"
        assert result.confidence == 0.55
        assert result.classification_source == "deterministic"
        _reset_settings()

    def test_deterministic_is_default_when_no_env(self, monkeypatch):
        """Default (no ENABLE_REAL_LLM set) should use deterministic."""
        monkeypatch.delenv("ENABLE_REAL_LLM", raising=False)
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        _reset_settings()

        result = classify_query("Explain the concept of ratio")

        assert result.classification_source == "deterministic"
        _reset_settings()


# ---------------------------------------------------------------------------
# Direct _classify_deterministic tests (no env dependency)
# ---------------------------------------------------------------------------


class TestClassifyDeterministic:
    def test_solve_question_intent(self):
        result = _classify_deterministic("Solve for x in 2x + 5 = 13")
        assert result.intent == "solve_question"
        assert result.confidence == 0.75

    def test_explain_concept_intent(self):
        result = _classify_deterministic("Explain the concept of osmosis")
        assert result.intent == "explain_concept"

    def test_explain_option_intent(self):
        result = _classify_deterministic("Which choice is correct here?")
        assert result.intent == "explain_option"

    def test_math_subject(self):
        result = _classify_deterministic("Find the profit percentage here")
        assert result.subject == "math"

    def test_classification_source_is_deterministic(self):
        result = _classify_deterministic("any query")
        assert result.classification_source == "deterministic"

    def test_response_style_step_by_step_for_solve(self):
        result = _classify_deterministic("Solve this equation")
        assert result.response_style == "step_by_step"

    def test_response_style_short_answer_keyword(self):
        result = _classify_deterministic("Give a short answer: what is osmosis?")
        assert result.response_style == "short_answer"

    def test_response_style_simple_explanation_keyword(self):
        result = _classify_deterministic("Explain in simple terms what ratio means")
        assert result.response_style == "simple_explanation"


# ---------------------------------------------------------------------------
# LLM path — valid response (monkeypatched)
# ---------------------------------------------------------------------------


class TestLlmPath:
    def _setup_llm_env(self, monkeypatch, intent: str = "solve_question"):
        """Configure env for LLM path and monkeypatch model_router.generate."""
        import services.model_router as model_router_module

        role_cfg = {
            _CLASSIFIER_ROLE: {
                "provider": "mock",
                "model_label": "test-classifier",
            }
        }
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", json.dumps(role_cfg))
        _reset_settings()

        from schemas.llm import LlmResponse

        def _fake_generate(role, messages):
            return LlmResponse(
                role=role,
                provider="mock",
                model_label="test-classifier",
                content=_make_valid_llm_response_content(intent=intent),
                finish_reason="stop",
            )

        monkeypatch.setattr(model_router_module, "generate", _fake_generate)

    def test_llm_path_returns_classification(self, monkeypatch):
        self._setup_llm_env(monkeypatch)

        result = classify_query("Calculate profit on selling price")

        assert isinstance(result, QueryClassification)
        assert result.classification_source == "llm"
        _reset_settings()

    def test_llm_path_intent_from_model(self, monkeypatch):
        self._setup_llm_env(monkeypatch, intent="explain_concept")

        result = classify_query("anything")

        assert result.intent == "explain_concept"
        assert result.classification_source == "llm"
        _reset_settings()

    def test_llm_path_high_confidence(self, monkeypatch):
        self._setup_llm_env(monkeypatch)

        result = classify_query("anything")

        assert result.confidence == 0.9
        assert result.classification_source == "llm"
        _reset_settings()

    def test_llm_path_retrieval_need_populated(self, monkeypatch):
        self._setup_llm_env(monkeypatch)

        result = classify_query("anything")

        assert result.retrieval_need == "none"
        _reset_settings()

    def test_llm_path_reasoning_summary_populated(self, monkeypatch):
        self._setup_llm_env(monkeypatch)

        result = classify_query("anything")

        assert result.reasoning_summary == "Query is a calculation task."
        _reset_settings()


# ---------------------------------------------------------------------------
# Fallback paths (malformed/invalid LLM output, errors)
# ---------------------------------------------------------------------------


class TestLlmFallbackPaths:
    def _setup_llm_env_with_fake(self, monkeypatch, response_content: str):
        import services.model_router as model_router_module

        role_cfg = {
            _CLASSIFIER_ROLE: {
                "provider": "mock",
                "model_label": "test-classifier",
            }
        }
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", json.dumps(role_cfg))
        _reset_settings()

        from schemas.llm import LlmResponse

        def _fake_generate(role, messages):
            return LlmResponse(
                role=role,
                provider="mock",
                model_label="test-classifier",
                content=response_content,
                finish_reason="stop",
            )

        monkeypatch.setattr(model_router_module, "generate", _fake_generate)

    def test_malformed_json_falls_back(self, monkeypatch):
        """Non-JSON model output → classification_source=fallback."""
        self._setup_llm_env_with_fake(monkeypatch, "NOT VALID JSON AT ALL")

        result = classify_query("Solve for x")

        assert result.classification_source == "fallback"
        _reset_settings()

    def test_malformed_json_confidence_capped(self, monkeypatch):
        """Fallback confidence must not exceed 0.55."""
        self._setup_llm_env_with_fake(monkeypatch, "NOT JSON")

        result = classify_query("Solve for x")

        assert result.confidence <= 0.55
        _reset_settings()

    def test_invalid_enum_intent_falls_back(self, monkeypatch):
        """Invalid intent value in JSON → validation error → fallback."""
        bad_json = json.dumps(
            {
                "intent": "do_homework",  # not an allowed value
                "subject": "math",
                "topic": None,
                "response_style": "step_by_step",
                "confidence": 0.9,
                "retrieval_need": "none",
                "reasoning_summary": None,
            }
        )
        self._setup_llm_env_with_fake(monkeypatch, bad_json)

        result = classify_query("anything")

        assert result.classification_source == "fallback"
        _reset_settings()

    def test_invalid_confidence_range_falls_back(self, monkeypatch):
        """Out-of-range confidence → validation error → fallback."""
        bad_json = json.dumps(
            {
                "intent": "solve_question",
                "subject": "math",
                "topic": None,
                "response_style": "step_by_step",
                "confidence": 1.5,  # > 1.0
                "retrieval_need": "none",
                "reasoning_summary": None,
            }
        )
        self._setup_llm_env_with_fake(monkeypatch, bad_json)

        result = classify_query("anything")

        assert result.classification_source == "fallback"
        _reset_settings()

    def test_model_router_exception_falls_back(self, monkeypatch):
        """model_router.generate raising an exception → fallback."""
        import services.model_router as model_router_module

        role_cfg = {
            _CLASSIFIER_ROLE: {
                "provider": "mock",
                "model_label": "test-classifier",
            }
        }
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", json.dumps(role_cfg))
        _reset_settings()

        def _failing_generate(role, messages):
            raise RuntimeError("Simulated provider failure")

        monkeypatch.setattr(model_router_module, "generate", _failing_generate)

        result = classify_query("Explain photosynthesis")

        assert result.classification_source == "fallback"
        _reset_settings()

    def test_fallback_intent_is_sensible(self, monkeypatch):
        """Fallback still produces a sensible intent from deterministic path."""
        self._setup_llm_env_with_fake(monkeypatch, "BAD JSON")

        result = classify_query("Calculate the profit here")

        # Deterministic path would return solve_question for "calculate"
        assert result.intent == "solve_question"
        assert result.classification_source == "fallback"
        _reset_settings()


# ---------------------------------------------------------------------------
# Role not configured when ENABLE_REAL_LLM=true → deterministic (not error)
# ---------------------------------------------------------------------------


class TestRoleNotConfigured:
    def test_uses_deterministic_when_role_missing(self, monkeypatch):
        """ENABLE_REAL_LLM=true but role not in config → deterministic, not error."""
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")  # no classifier role
        _reset_settings()

        result = classify_query("Explain the concept of osmosis")

        assert isinstance(result, QueryClassification)
        assert result.classification_source == "deterministic"
        _reset_settings()

    def test_malformed_role_config_json_uses_fallback(self, monkeypatch):
        """Malformed LLM_ROLE_CONFIG_JSON with ENABLE_REAL_LLM=true → fallback, not silent."""
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "NOT_JSON")
        _reset_settings()

        result = classify_query("Explain photosynthesis")

        assert result.classification_source == "fallback"
        assert result.confidence <= 0.55
        _reset_settings()

    def test_malformed_role_config_json_fallback_still_classifies(self, monkeypatch):
        """Fallback from malformed config still returns a sensible classification."""
        monkeypatch.setenv("ENABLE_REAL_LLM", "true")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "NOT_JSON")
        _reset_settings()

        result = classify_query("Solve for x in 2x = 10")

        assert result.intent == "solve_question"
        assert result.classification_source == "fallback"
        assert result.confidence <= 0.55
        _reset_settings()


class TestClassificationPolicy:
    def test_high_confidence_default_becomes_advanced_for_sbi_po(self) -> None:
        classification = {
            "subject": "math",
            "intent": "explain",
            "difficulty": "default",
            "retrieval_required": True,
        }
        result = apply_classification_policy(
            "simple explanation of SBI PO coded inequality",
            classification,
            request_id="req-policy-1",
        )
        assert result["difficulty"] == "advanced"

    def test_sbi_po_signal_forces_advanced(self) -> None:
        classification = {"subject": "reasoning", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "Explain seating arrangement for SBI PO",
            classification,
        )
        assert result["difficulty"] == "advanced"

    def test_floor_puzzle_forces_advanced(self) -> None:
        classification = {"subject": "reasoning", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "How to solve floor puzzle?",
            classification,
        )
        assert result["difficulty"] == "advanced"

    def test_basic_explanation_stays_basic_without_exam_signal(self) -> None:
        classification = {"subject": "math", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "Give a basic explanation of percentage",
            classification,
        )
        assert result["difficulty"] == "basic"

    def test_advanced_signal_wins_over_simple_wording(self) -> None:
        classification = {"subject": "math", "intent": "explain", "difficulty": "basic"}
        result = apply_classification_policy(
            "simple explanation of IBPS PO caselet",
            classification,
        )
        assert result["difficulty"] == "advanced"

    def test_no_change_when_no_policy_signal(self) -> None:
        classification = {"subject": "math", "intent": "explain", "difficulty": "intermediate"}
        result = apply_classification_policy("Explain ratio basics", classification)
        assert result["difficulty"] == "intermediate"

    def test_profit_loss_query_becomes_math(self) -> None:
        classification = {"subject": "general", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "Explain profit and loss discount marked price trap",
            classification,
        )
        assert result["subject"] == "math"

    def test_mixture_alligation_becomes_math(self) -> None:
        classification = {"subject": "general", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "How to solve mixture and alligation?",
            classification,
        )
        assert result["subject"] == "math"

    def test_coded_inequality_becomes_reasoning(self) -> None:
        classification = {"subject": "general", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "Explain coded inequality for bank exam",
            classification,
        )
        assert result["subject"] == "reasoning"
        assert result["difficulty"] == "advanced"

    def test_seating_arrangement_becomes_reasoning(self) -> None:
        classification = {"subject": "general", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "Explain circular seating arrangement",
            classification,
        )
        assert result["subject"] == "reasoning"
        assert result["difficulty"] == "advanced"

    def test_floor_puzzle_becomes_advanced(self) -> None:
        classification = {"subject": "reasoning", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "Solve floor puzzle with building numbered floors",
            classification,
        )
        assert result["difficulty"] == "advanced"

    def test_structural_complexity_becomes_advanced(self) -> None:
        classification = {"subject": "reasoning", "intent": "explain", "difficulty": "default"}
        query = (
            "A sits left of B; C who is not facing north; D given that E cannot sit "
            "with F; G which is neither adjacent to H; I such that J does not face center."
        )
        result = apply_classification_policy(query, classification)
        assert result["difficulty"] == "advanced"

    def test_simple_direction_not_overcorrected_to_advanced(self) -> None:
        classification = {"subject": "reasoning", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "What is direction sense?",
            classification,
        )
        assert result["difficulty"] in {"default", "intermediate"}

    def test_policy_summary_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        classification = {"subject": "reasoning", "intent": "explain", "difficulty": "default"}
        with caplog.at_level("INFO"):
            apply_classification_policy(
                "Explain coded inequality for bank exam",
                classification,
                request_id="req-summary",
            )
        summaries = [
            r.message for r in caplog.records if "classification_policy_summary" in r.message
        ]
        assert summaries
        assert "pattern_topic=CODED_INEQUALITY" in summaries[0]

    def test_grammar_becomes_english(self) -> None:
        classification = {"subject": "general", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy(
            "Explain grammar rules for sentence correction",
            classification,
        )
        assert result["subject"] == "english"

    def test_vague_query_not_overcorrected(self) -> None:
        classification = {"subject": "math", "intent": "explain", "difficulty": "default"}
        result = apply_classification_policy("What is the answer?", classification)
        assert result["subject"] == "math"
        assert result["difficulty"] == "default"

    def test_policy_always_logs_checked(self, caplog: pytest.LogCaptureFixture) -> None:
        classification = {"subject": "math", "intent": "explain", "difficulty": "default"}
        with caplog.at_level("INFO"):
            apply_classification_policy(
                "What is the answer?",
                classification,
                request_id="req-policy-log",
            )
        policy_logs = [r.message for r in caplog.records if "classification_policy" in r.message]
        assert policy_logs
        assert "policy_checked=true" in policy_logs[0]
        assert "policy_applied=false" in policy_logs[0]


class TestClassifierPromptFoundation:
    def test_prompt_uses_solving_method_principle(self) -> None:
        prompt = _load_classifier_prompt()
        assert "solving method" in prompt.lower()
        assert "superficial keywords" in prompt.lower()
        assert "not" in prompt.lower()

    def test_prompt_covers_broad_exam_prep_domains(self) -> None:
        prompt = _load_classifier_prompt()
        assert "government" in prompt.lower() or "competitive exam" in prompt.lower()
        assert "quantitative" in prompt.lower() or "`math`" in prompt
        assert "reasoning" in prompt.lower()
        assert "english" in prompt.lower()
        assert "general studies" in prompt.lower() or "GK" in prompt

    def test_prompt_defines_classification_dimensions(self) -> None:
        prompt = _load_classifier_prompt()
        assert "solving method" in prompt.lower()
        assert "exam" in prompt.lower()
        assert "difficulty" in prompt.lower()
        assert "confidence" in prompt.lower()

    def test_prompt_has_method_domain_mapping_table(self) -> None:
        prompt = _load_classifier_prompt()
        assert "Method → domain rules" in prompt
        assert "TIME_SPEED_DISTANCE" in prompt
        assert "BLOOD_RELATION" in prompt
        assert "GRAMMAR" in prompt

    def test_prompt_marks_boundary_illustrations_non_exhaustive(self) -> None:
        prompt = _load_classifier_prompt()
        assert "Boundary illustrations" in prompt
        assert "non-exhaustive" in prompt.lower()

    def test_prompt_contains_confidence_calibration_rules(self) -> None:
        prompt = _load_classifier_prompt()
        assert ">= 0.93" in prompt
        assert "1.00" in prompt
        assert "below 0.93" in prompt.lower()

    def test_prompt_requires_json_only_output(self) -> None:
        prompt = _load_classifier_prompt()
        assert "only" in prompt.lower()
        assert "JSON" in prompt


class TestExamClassificationMatrix:
    """Representative exam domains — not a fixed example-fix list."""

    @pytest.mark.parametrize(
        ("query", "expected_subject"),
        [
            (
                (
                    "A shopkeeper marks an article 40% above cost and gives 10% discount. "
                    "Find profit percent."
                ),
                "math",
            ),
            (
                "Syllogism: All cats are dogs. Some dogs are birds. Which conclusions follow?",
                "reasoning",
            ),
            (
                "Choose the grammatically correct sentence from grammar options below.",
                "english",
            ),
            (
                "Who is the current RBI governor?",
                "general",
            ),
            (
                "Mixture of milk and water in ratio 3:2. How much water to add?",
                "math",
            ),
        ],
    )
    def test_low_confidence_general_can_be_corrected_by_policy(
        self, query: str, expected_subject: str
    ) -> None:
        result = apply_classification_policy(
            query,
            {"subject": "general", "intent": "solve", "difficulty": "default"},
            classifier_confidence=0.72,
        )
        assert result["subject"] == expected_subject

    @pytest.mark.parametrize(
        ("query", "initial_subject", "expected_subject"),
        [
            (
                "Syllogism with statements: All A are B. Conclusions: Some B are C. Which follows?",
                "general",
                "reasoning",
            ),
            (
                "Find synonym of ephemeral in the given options.",
                "general",
                "english",
            ),
        ],
    )
    def test_explicit_pattern_family_corrections(
        self, query: str, initial_subject: str, expected_subject: str
    ) -> None:
        result = apply_classification_policy(
            query,
            {"subject": initial_subject, "intent": "explain", "difficulty": "default"},
            classifier_confidence=0.75,
        )
        assert result["subject"] == expected_subject


class TestRegressionDisambiguationBoundaries:
    """Regression-only boundary cases — not the design target."""

    def test_regression_quant_motion_not_overridden_to_reasoning(self) -> None:
        query = (
            "Two trains 120m and 180m long run in opposite directions at 54 km/hr "
            "and 72 km/hr. How long do they take to cross each other?"
        )
        result = apply_classification_policy(
            query,
            {"subject": "math", "intent": "solve", "difficulty": "intermediate"},
            classifier_confidence=0.93,
        )
        assert result["subject"] == "math"

    def test_regression_age_equation_not_overridden_to_blood_relation(self) -> None:
        query = (
            "Nisha is thrice as old as Deepak. Tanya is 5 years younger than Nisha. "
            "If Deepak is 12 years old, find Tanya's age."
        )
        result = apply_classification_policy(
            query,
            {"subject": "math", "intent": "solve", "difficulty": "default"},
            classifier_confidence=0.92,
        )
        assert result["subject"] == "math"

    def test_regression_navigation_inference_classifies_reasoning(self) -> None:
        query = "A person walked 5 km north then turned right and walked 3 km. Where is he now?"
        result = apply_classification_policy(
            query,
            {"subject": "general", "intent": "solve", "difficulty": "default"},
            classifier_confidence=0.75,
        )
        assert result["subject"] == "reasoning"

    def test_regression_pure_relation_inference_classifies_reasoning(self) -> None:
        query = "How is Amit related to Sunita if Amit is Sunita's mother's brother?"
        result = apply_classification_policy(
            query,
            {"subject": "general", "intent": "explain", "difficulty": "default"},
            classifier_confidence=0.70,
        )
        assert result["subject"] == "reasoning"

    def test_regression_high_confidence_quant_not_overridden_by_broad_words(self) -> None:
        query = "Trains moving in opposite directions at 60 km/hr and 80 km/hr"
        result = apply_classification_policy(
            query,
            {"subject": "math", "intent": "solve", "difficulty": "intermediate"},
            classifier_confidence=0.96,
        )
        assert result["subject"] == "math"

    def test_regression_father_age_equation_stays_math(self) -> None:
        query = (
            "Father is twice as old as daughter. Daughter is 12 years old. "
            "How old is the father?"
        )
        result = apply_classification_policy(
            query,
            {"subject": "math", "intent": "solve", "difficulty": "default"},
            classifier_confidence=0.94,
        )
        assert result["subject"] == "math"


class TestClassifierThresholdSettings:
    def test_get_classifier_confidence_threshold_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("CLASSIFIER_CONFIDENCE_FALLBACK_THRESHOLD", raising=False)
        _reset_settings()
        assert get_classifier_confidence_threshold() == 0.93

    def test_get_classifier_confidence_threshold_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", "0.88")
        _reset_settings()
        assert get_classifier_confidence_threshold() == 0.88


class TestPolicyGuardrailBehavior:
    def test_policy_still_upgrades_advanced_reasoning_pattern(self) -> None:
        result = apply_classification_policy(
            "Solve coded inequality with multiple constraints",
            {"subject": "reasoning", "intent": "explain", "difficulty": "default"},
            classifier_confidence=0.95,
        )
        assert result["difficulty"] == "advanced"


class TestClassifierRetrievalHintsSchema:
    def test_optional_hint_fields_parse(self) -> None:
        parsed = QueryClassification.model_validate(
            {
                "intent": "solve_question",
                "subject": "math",
                "topic": "Age Problem",
                "topic_confidence": 0.91,
                "pattern_topic_candidate": "AGE",
                "pattern_family_candidate": None,
                "retrieval_tags": ["age_equation", "birth_age"],
                "confidence": 0.92,
            }
        )
        assert parsed.pattern_topic_candidate == "AGE"
        assert parsed.retrieval_tags == ["age_equation", "birth_age"]

    def test_missing_hint_fields_backward_compatible(self) -> None:
        parsed = QueryClassification.model_validate(
            {"intent": "explain_concept", "subject": "english", "confidence": 0.8}
        )
        assert parsed.retrieval_tags == []
        assert parsed.topic_confidence is None

    def test_invalid_topic_confidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QueryClassification.model_validate(
                {
                    "intent": "solve_question",
                    "subject": "math",
                    "topic_confidence": 1.5,
                    "confidence": 0.9,
                }
            )

    def test_retrieval_tags_normalized_and_capped(self) -> None:
        parsed = QueryClassification.model_validate(
            {
                "intent": "solve_question",
                "subject": "math",
                "confidence": 0.9,
                "retrieval_tags": [
                    "Train-Crossing",
                    "train_crossing",
                    "Ratio Parts",
                ],
            }
        )
        assert parsed.retrieval_tags == ["train_crossing", "ratio_parts"]


class TestClassifierWebSearchSchema:
    def test_need_web_search_defaults_false(self) -> None:
        parsed = QueryClassification.model_validate(
            {"intent": "explain_concept", "subject": "general", "confidence": 0.9}
        )
        assert parsed.need_web_search is False

    def test_need_web_search_parses_true(self) -> None:
        parsed = QueryClassification.model_validate(
            {
                "intent": "general_doubt",
                "subject": "general",
                "confidence": 0.93,
                "need_web_search": True,
                "web_search_reason": "current_affairs",
                "web_search_query": "latest current affairs UPSC",
            }
        )
        assert parsed.need_web_search is True
        assert parsed.web_search_reason == "current_affairs"
        assert parsed.web_search_query == "latest current affairs UPSC"

    def test_web_search_optional_fields(self) -> None:
        parsed = QueryClassification.model_validate(
            {
                "intent": "general_doubt",
                "subject": "general",
                "confidence": 0.9,
                "need_web_search": False,
            }
        )
        assert parsed.web_search_reason is None
        assert parsed.web_search_query is None


class TestClassifierPromptRetrievalHints:
    def test_prompt_asks_for_topic_and_tags(self) -> None:
        prompt = _load_classifier_prompt()
        assert "topic_confidence" in prompt
        assert "retrieval_tags" in prompt
        assert "pattern_topic_candidate" in prompt

    def test_prompt_canonical_keys_only_when_obvious(self) -> None:
        prompt = _load_classifier_prompt()
        assert "obvious" in prompt.lower()
        assert "Do **not** invent obscure keys" in prompt

    def test_prompt_json_only(self) -> None:
        prompt = _load_classifier_prompt()
        assert "JSON" in prompt
        assert "chain-of-thought" in prompt.lower()


class TestClassifierPromptWebSearch:
    def test_prompt_instructs_current_affairs_behavior(self) -> None:
        prompt = _load_classifier_prompt()
        assert "need_web_search" in prompt
        assert "current affairs" in prompt.lower()

    def test_prompt_says_not_for_static_gk(self) -> None:
        prompt = _load_classifier_prompt()
        assert "static GK" in prompt or "static GK / general studies" in prompt

    def test_prompt_says_not_for_normal_math_reasoning(self) -> None:
        prompt = _load_classifier_prompt()
        lower = prompt.lower()
        assert "normal **math / quant** problem" in prompt or "math / quant" in lower
        assert "reasoning" in lower

    def test_prompt_requires_concise_web_search_query(self) -> None:
        prompt = _load_classifier_prompt()
        assert "web_search_query" in prompt
        assert "concise" in prompt.lower()


class TestDeterministicRetrievalHints:
    def test_train_question_produces_tsd_hint(self) -> None:
        result = _classify_deterministic(
            "Two trains 120m and 180m cross each other in 10 seconds. Find speed."
        )
        assert result.pattern_topic_candidate == "TIME_SPEED_DISTANCE"
        assert result.retrieval_tags

    def test_coded_inequality_produces_hint(self) -> None:
        result = _classify_deterministic(
            "Statements: A > B, B >= C. Which conclusions follow using coded symbols?"
        )
        assert result.pattern_topic_candidate == "CODED_INEQUALITY" or result.retrieval_tags


JOURNEY_SPEED_QUERY = (
    "Gopesh starts on his journey at 5 pm. He travels at 40 km/hr for 2 hours. "
    "Then he increases his speed to 60 km/hr. What is his average speed for the "
    "whole journey?"
)


class TestDeterministicFallbackHardening:
    def test_journey_speed_query_math_solve_intermediate(self) -> None:
        result = _classify_deterministic(JOURNEY_SPEED_QUERY)
        assert result.subject == "math"
        assert result.intent == "solve_question"
        assert result.difficulty == "intermediate"
        assert result.confidence >= 0.75

    def test_age_equation_math_solve(self) -> None:
        query = "Father is twice as old as son. Son is 12 years old. Find father's age."
        result = _classify_deterministic(query)
        assert result.subject == "math"
        assert result.intent == "solve_question"

    def test_direction_sense_reasoning_solve(self) -> None:
        query = "A man walks 5 km north, turns right and walks 3 km. Which direction is he facing?"
        result = _classify_deterministic(query)
        assert result.subject == "reasoning"
        assert result.intent == "solve_question"

    def test_grammar_english(self) -> None:
        result = _classify_deterministic("Choose the correct grammar form in this sentence.")
        assert result.subject == "english"

    def test_static_gk_general(self) -> None:
        result = _classify_deterministic("Who was the first President of India?")
        assert result.subject in {"unknown", "general"}


class TestClassificationSanity:
    def test_low_confidence_general_numeric_reroutes_math(self) -> None:
        updated = apply_classification_sanity(
            JOURNEY_SPEED_QUERY,
            {"subject": "general", "intent": "explain", "difficulty": "default"},
            classifier_confidence=0.55,
        )
        assert updated["subject"] == "math"
        assert updated["intent"] == "solve"
        assert updated["difficulty"] == "intermediate"

    def test_high_confidence_general_unchanged(self) -> None:
        updated = apply_classification_sanity(
            "Who was the first President of India?",
            {"subject": "general", "intent": "explain", "difficulty": "default"},
            classifier_confidence=0.85,
        )
        assert updated["subject"] == "general"

    def test_reasoning_signals_reroute(self) -> None:
        query = "How is A related to B if A is father of B's sister?"
        updated = apply_classification_sanity(
            query,
            {"subject": "general", "intent": "explain", "difficulty": "default"},
            classifier_confidence=0.60,
        )
        assert updated["subject"] == "reasoning"
        assert updated["intent"] == "solve"


class TestClassifierModelRouting:
    def test_primary_classifier_uses_native_gemini_flash_lite(self) -> None:
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        reg = LlmConfigRegistry()
        cfg = reg.model_map["doubt_solver_classifier_gemini"]
        assert cfg.provider == "gemini"
        assert cfg.model_id == "gemini-3.1-flash-lite"
        assert not cfg.fallback_models
        route = reg.get_route("general", "classifier", "default")
        assert route is not None
        assert route.model == "doubt_solver_classifier_gemini"

    def test_strong_classifier_uses_safe_gpt_41(self) -> None:
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        reg = LlmConfigRegistry()
        cfg = reg.model_map["doubt_solver_classifier_strong"]
        assert cfg.deployment == "gpt-4.1"
        assert cfg.deployment != "gpt-5.4"
        route = reg.get_route("general", "classifier_strong", "default")
        assert route is not None
        assert route.model == "doubt_solver_classifier_strong"

    def test_classifier_route_max_tokens(self) -> None:
        from services.llm.orchestration.config_registry import LlmConfigRegistry

        reg = LlmConfigRegistry()
        primary = reg.get_route("general", "classifier", "default")
        strong = reg.get_route("general", "classifier_strong", "default")
        assert primary is not None and primary.max_tokens == 650
        assert strong is not None and strong.max_tokens == 800


def test_incomplete_topic_solve_conflict_invokes_strong_classifier_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    primary = QueryClassification(
        intent="solve_question",
        subject="math",
        confidence=0.96,
        relation="NEW_QUESTION",
        requested_action="ANSWER_CURRENT",
    )
    strong = QueryClassification(
        intent="explain_concept",
        subject="general",
        confidence=0.95,
        relation="NEW_QUESTION",
        requested_action="ANSWER_CURRENT",
    )

    def classify_once(
        query: str,
        _request_id: str | None = None,
        *,
        task_role: str,
        candidate_turn_ids: tuple[str, ...] = (),
        context_gate: str,
        **_: object,
    ) -> QueryClassification:
        calls.append(task_role)
        result = primary if task_role == "classifier" else strong
        return classifier_service._validate_conversation_contract(
            result,
            query=query,
            candidate_turn_ids=candidate_turn_ids,
            context_gate=context_gate,
        )

    monkeypatch.setattr(
        classifier_service,
        "_classify_with_llm_orchestrated",
        classify_once,
    )

    result = classifier_service._classify_with_llm_orchestrated_or_fallback(
        "acid-water solution",
        request_id="topic-phrase",
        context_gate="UNCERTAIN",
    )

    assert calls == ["classifier", "classifier_strong"]
    assert result.strong_classifier_used is True
    assert result.classification.intent == "explain_concept"
    assert result.classification.relation == "NEW_QUESTION"


def test_classifier_prompt_defines_uncertain_topic_phrase_policy() -> None:
    prompt = _load_classifier_prompt()

    assert "not a numerical problem" in prompt
    assert "Numeric, percentage, option, or topic overlap by itself" in prompt
    assert "Do not force a fresh solve" in prompt
    assert "newest supplied substantive turn" in prompt


def test_classifier_base_prompt_is_materially_below_observed_baseline() -> None:
    prompt = _load_classifier_prompt()

    assert len(prompt) <= 12_000
    assert len(prompt) <= int(19_085 * 0.75)


def test_classifier_prompt_preserves_compaction_and_clarifies_rank_confidence() -> None:
    prompt = _load_classifier_prompt()

    assert "rank/order and relative positions" in prompt
    assert "Ordinal rank/order or relative-position counting" in prompt
    assert "Confidence is routing/classification certainty" in prompt
    assert "not answer certainty" in prompt
    assert len(prompt) <= int(9_596 * 1.05)


@pytest.mark.parametrize(
    ("query", "context_gate"),
    (
        ("Explain photosynthesis.", "CONTEXT_NOT_NEEDED"),
        ("Did he introduce any reforms?", "CONTEXT_REQUIRED"),
        ("Your answer is wrong.", "CONTEXT_REQUIRED"),
        ("Provide current affairs questions for July 2026.", "CONTEXT_NOT_NEEDED"),
        ("Give me five similar questions.", "CONTEXT_REQUIRED"),
        ("yeh percentage kaise nikala?", "CONTEXT_REQUIRED"),
    ),
)
def test_classifier_prompt_budget_covers_required_scenarios(
    query: str,
    context_gate: str,
) -> None:
    candidates = (
        "[RECENT_CANDIDATES_UNTRUSTED_DATA]\n"
        "<candidate turn_id='turn-1'>\n"
        "question_preview: Explain Akbar's reign.\n"
        "answer_clue: Akbar introduced administrative and revenue reforms.\n"
        "subject: general\n"
        "topic: Mughal history\n"
        "</candidate>\n"
        "[/RECENT_CANDIDATES_UNTRUSTED_DATA]"
        if context_gate != "CONTEXT_NOT_NEEDED"
        else None
    )

    budget = _classifier_prompt_character_budget(
        query=query,
        conversation_candidates=candidates,
        context_gate=context_gate,
    )

    assert budget["total_chars"] <= 13_000
    assert budget["query_chars"] == len(query)
    assert budget["candidate_context_chars"] == len(candidates or "")


def test_standalone_classifier_input_omits_candidate_placeholder() -> None:
    classifier_input = _build_classifier_input(
        "Explain photosynthesis.",
        conversation_candidates=None,
        context_gate="CONTEXT_NOT_NEEDED",
    )

    assert "RECENT_CANDIDATES" not in classifier_input
    assert "CONTEXT_NOT_NEEDED" in classifier_input


def test_strong_classifier_adds_only_compact_rejection_summary() -> None:
    primary = QueryClassification(
        intent="explain_concept",
        subject="general",
        confidence=0.80,
        relation="NEW_QUESTION",
        requested_action="ANSWER_CURRENT",
    )
    base = _build_classifier_input(
        "Explain photosynthesis.",
        conversation_candidates=None,
        context_gate="CONTEXT_NOT_NEEDED",
    )
    strong = _build_classifier_input(
        "Explain photosynthesis.",
        conversation_candidates=None,
        context_gate="CONTEXT_NOT_NEEDED",
        primary_result=primary,
        conflict_reason="primary_low_confidence",
    )

    assert len(strong) - len(base) < 400
    assert "PRIMARY_CLASSIFIER_RESULT_REJECTED" in strong


def test_generic_resolve_rejects_an_older_selected_turn() -> None:
    classification = QueryClassification(
        intent="solve_question",
        subject="math",
        confidence=0.95,
        relation="RESOLVE_AGAIN",
        selected_turn_id="older-turn",
        requested_action="RESOLVE_FROM_SCRATCH",
    )

    with pytest.raises(
        classifier_service.ConversationClassificationConflict,
        match="generic_latest_reference_selected_older_turn",
    ):
        classifier_service._validate_conversation_contract(
            classification,
            query="Again wrong, solve it from scratch.",
            candidate_turn_ids=("older-turn", "latest-turn"),
            context_gate="CONTEXT_REQUIRED",
        )


def test_short_correction_phrase_may_keep_solve_intent() -> None:
    classification = QueryClassification(
        intent="solve_question",
        subject="math",
        confidence=0.95,
        relation="CORRECTION",
        selected_turn_id="latest-turn",
        requested_action="VERIFY_AND_CORRECT",
    )

    result = classifier_service._validate_conversation_contract(
        classification,
        query="Your answer is wrong.",
        candidate_turn_ids=("latest-turn",),
        context_gate="CONTEXT_REQUIRED",
    )

    assert result.intent == "solve_question"
    assert result.relation == "CORRECTION"


def test_generic_resolve_accepts_newest_hygienic_turn_despite_reference_scoring() -> None:
    card = ConversationCandidateCard(
        turn_id="latest-turn",
        question_preview="Solve 2x + 5 = 15.",
        answer_clue="Subtract 5 and divide by 2 to obtain x = 5.",
        subject="math",
        topic="Algebra",
        difficulty="basic",
    )
    classification = QueryClassification(
        intent="solve_question",
        subject="math",
        confidence=0.93,
        relation="RESOLVE_AGAIN",
        selected_turn_id=card.turn_id,
        requested_action="RESOLVE_FROM_SCRATCH",
    )

    result = classifier_service._validate_conversation_contract(
        classification,
        query="The previous answer is wrong; solve again from scratch.",
        candidate_cards=(card,),
        candidate_turn_ids=(card.turn_id,),
        context_gate="CONTEXT_REQUIRED",
    )

    assert result.relation == "RESOLVE_AGAIN"
    assert result.selected_turn_id == card.turn_id
    assert result.requested_action == "RESOLVE_FROM_SCRATCH"


def test_deterministic_failure_fallback_keeps_clear_correction_on_latest_turn() -> None:
    result = classifier_service._deterministic_classifier_fallback(
        "Your answer is wrong.",
        strong_classifier_used=True,
        context_gate="CONTEXT_REQUIRED",
        candidate_turn_ids=("older-turn", "latest-turn"),
    )

    assert result.strong_classifier_used is True
    assert result.classification.relation == "CORRECTION"
    assert result.classification.requested_action == "VERIFY_AND_CORRECT"
    assert result.classification.selected_turn_id == "latest-turn"


def test_deterministic_failure_fallback_keeps_clear_required_follow_up() -> None:
    result = classifier_service._deterministic_classifier_fallback(
        "Highlight the water-acid solution.",
        strong_classifier_used=True,
        context_gate="CONTEXT_REQUIRED",
        candidate_turn_ids=("older-turn", "latest-turn"),
    )

    assert result.classification.relation == "FOLLOW_UP"
    assert result.classification.requested_action == "EXPLAIN_PREVIOUS"
    assert result.classification.selected_turn_id == "latest-turn"


def test_deterministic_failure_fallback_clarifies_uncertain_topic_phrase() -> None:
    result = classifier_service._deterministic_classifier_fallback(
        "water acid solution",
        strong_classifier_used=True,
        context_gate="UNCERTAIN",
        candidate_turn_ids=("candidate-turn",),
    )

    assert result.classification.relation == "AMBIGUOUS"
    assert result.classification.requested_action == "ASK_CLARIFICATION"
    assert result.classification.selected_turn_id is None
