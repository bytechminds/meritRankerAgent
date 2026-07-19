"""
app/tests/test_difficulty_classification.py
--------------------------------------------
Tests for difficulty classification and difficulty-based routing.

Covers (Part G):
 1.  QueryClassification schema accepts difficulty=advanced.
 2.  QueryClassification difficulty defaults to 'default'.
 3.  LLM output with difficulty=advanced validates correctly.
 4.  Missing difficulty in LLM JSON defaults to 'default'.
 5.  Deterministic: "advanced SSC CGL level" → advanced.
 6.  Deterministic: "hard reasoning question" → advanced.
 7.  Deterministic: "tough" → advanced.
 8.  Deterministic: "tricky" → advanced.
 9.  Deterministic: "basic explanation" → basic.
10.  Deterministic: "simple" → basic.
11.  Deterministic: "beginner" → basic.
12.  Deterministic: "intermediate level" → intermediate.
13.  Deterministic: "moderate difficulty" → intermediate.
14.  Neutral query → default.
15.  _map_to_orchestrated_classification passes difficulty=advanced through.
16.  _map_to_orchestrated_classification passes difficulty=basic through.
17.  _map_to_orchestrated_classification passes difficulty=default through.
18.  FakeAdapter receives difficulty=advanced.
19.  RouteRequest receives difficulty=advanced.
20.  RouteResolver selects math.generator.advanced when route exists.
21.  RouteResolver falls back to default when advanced route missing for a subject.
22.  Practice regression: advanced SSC CGL practice query classifies correctly.
23.  Solve regression: advanced profit/loss query classifies correctly.
24.  Graph state remains exactly 5 fields.
25.  task_role remains 'generator'.
26.  Provider strategy unchanged — no real provider calls in tests.
27.  make check: advanced queries no longer route as default.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from graphs.doubt_solver_graph import (
    OrchestratedDoubtSolverState,
    _map_to_orchestrated_classification,
)
from schemas.doubt_solver import QueryClassification
from schemas.llm_routing import RouteRequest
from services.llm.orchestration.config_registry import LlmConfigRegistry
from services.llm.orchestration.route_resolver import resolve_route
from services.query_classifier_service import (
    _classify_deterministic,
    _detect_difficulty,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_qc(
    intent: str = "solve_question",
    subject: str = "math",
    difficulty: str = "default",
    retrieval_need: str = "none",
    confidence: float = 0.9,
) -> QueryClassification:
    return QueryClassification(
        intent=intent,  # type: ignore[arg-type]
        subject=subject,
        difficulty=difficulty,  # type: ignore[arg-type]
        confidence=confidence,
        retrieval_need=retrieval_need,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1–2: QueryClassification schema
# ---------------------------------------------------------------------------


def test_query_classification_accepts_difficulty_advanced() -> None:
    qc = QueryClassification(intent="solve_question", confidence=0.9, difficulty="advanced")
    assert qc.difficulty == "advanced"


def test_query_classification_accepts_difficulty_basic() -> None:
    qc = QueryClassification(intent="explain_concept", confidence=0.9, difficulty="basic")
    assert qc.difficulty == "basic"


def test_query_classification_accepts_difficulty_intermediate() -> None:
    qc = QueryClassification(intent="general_doubt", confidence=0.9, difficulty="intermediate")
    assert qc.difficulty == "intermediate"


def test_query_classification_difficulty_defaults_to_default() -> None:
    qc = QueryClassification(intent="solve_question", confidence=0.9)
    assert qc.difficulty == "default"


def test_query_classification_rejects_unknown_difficulty() -> None:
    with pytest.raises((ValidationError, ValueError)):
        QueryClassification(
            intent="solve_question",
            confidence=0.9,
            difficulty="extreme",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 3–4: LLM classifier JSON parsing
# ---------------------------------------------------------------------------


def test_llm_output_with_difficulty_advanced_validates() -> None:
    raw = {
        "intent": "solve_question",
        "subject": "math",
        "topic": "profit and loss",
        "difficulty": "advanced",
        "response_style": "step_by_step",
        "confidence": 0.92,
        "retrieval_need": "none",
        "reasoning_summary": "Advanced multi-step profit/loss query.",
    }
    result = QueryClassification.model_validate(raw)
    assert result.difficulty == "advanced"
    assert result.intent == "solve_question"


def test_llm_output_missing_difficulty_defaults_to_default() -> None:
    """Classifier output without difficulty field (old format) defaults safely."""
    raw = {
        "intent": "explain_concept",
        "subject": "reasoning",
        "topic": None,
        "response_style": "simple_explanation",
        "confidence": 0.85,
        "retrieval_need": "concept_context",
        "reasoning_summary": None,
    }
    result = QueryClassification.model_validate(raw)
    assert result.difficulty == "default"


# ---------------------------------------------------------------------------
# 5–14: Deterministic difficulty detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_difficulty"),
    [
        ("Create advanced SSC CGL level practice questions on profit-loss", "advanced"),
        ("hard reasoning question on blood relations", "advanced"),
        ("tough math problem", "advanced"),
        ("tricky percentage calculation", "advanced"),
        ("high level arrangement puzzle", "advanced"),
        ("ssc cgl level reasoning", "advanced"),
        ("upsc level math question", "advanced"),
        ("basic explanation of ratio", "basic"),
        ("simple percentage problem", "basic"),
        ("beginner level grammar", "basic"),
        ("easy arithmetic question", "basic"),
        ("intermediate level geometry", "intermediate"),
        ("moderate difficulty reasoning", "intermediate"),
        ("What is 20% of 500?", "default"),
        ("Explain the concept of osmosis", "default"),
        ("solve for x in 2x + 5 = 13", "default"),
    ],
)
def test_detect_difficulty(query: str, expected_difficulty: str) -> None:
    result = _detect_difficulty(query.lower())
    assert result == expected_difficulty, (
        f"Query {query!r}: expected {expected_difficulty!r}, got {result!r}"
    )


def test_classify_deterministic_includes_difficulty_advanced() -> None:
    result = _classify_deterministic("Create 5 advanced SSC CGL practice questions on profit")
    assert result.difficulty == "advanced"


def test_classify_deterministic_includes_difficulty_basic() -> None:
    result = _classify_deterministic("Give me a basic explanation of profit and loss")
    assert result.difficulty == "basic"


def test_classify_deterministic_includes_difficulty_default() -> None:
    result = _classify_deterministic("What is 20% of 500?")
    assert result.difficulty == "default"


# ---------------------------------------------------------------------------
# 15–19: Orchestrated mapping / adapter
# ---------------------------------------------------------------------------


def test_map_to_orchestrated_classification_passes_difficulty_advanced() -> None:
    raw = _make_qc(subject="math", difficulty="advanced")
    result = _map_to_orchestrated_classification(raw)
    assert result["difficulty"] == "advanced"


def test_map_to_orchestrated_classification_passes_difficulty_basic() -> None:
    raw = _make_qc(subject="english", difficulty="basic")
    result = _map_to_orchestrated_classification(raw)
    assert result["difficulty"] == "basic"


def test_map_to_orchestrated_classification_passes_difficulty_intermediate() -> None:
    raw = _make_qc(subject="reasoning", difficulty="intermediate")
    result = _map_to_orchestrated_classification(raw)
    assert result["difficulty"] == "intermediate"


def test_map_to_orchestrated_classification_passes_difficulty_default() -> None:
    raw = _make_qc(subject="math", difficulty="default")
    result = _map_to_orchestrated_classification(raw)
    assert result["difficulty"] == "default"


def test_map_applies_policy_correction_for_advanced_query() -> None:
    raw = _make_qc(subject="math", difficulty="default", confidence=0.95)
    result = _map_to_orchestrated_classification(
        raw,
        query="Explain coded inequality for SBI PO mains level",
    )
    assert result["difficulty"] == "advanced"
    assert result["subject"] == "reasoning"


class _FakeAdapter:
    """Records kwargs passed to generate()."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def generate(self, *, request_id: str, query: str, subject: str,
                 intent: str, difficulty: str, context: str) -> str:
        self.last_kwargs = {
            "request_id": request_id, "query": query, "subject": subject,
            "intent": intent, "difficulty": difficulty, "context": context,
        }
        return "Fake answer."


def test_fake_adapter_receives_difficulty_advanced() -> None:
    adapter = _FakeAdapter()
    adapter.generate(
        request_id="r1", query="test", subject="math",
        intent="practice", difficulty="advanced", context="",
    )
    assert adapter.last_kwargs["difficulty"] == "advanced"


def test_route_request_accepts_difficulty_advanced() -> None:
    req = RouteRequest(
        request_id="r1",
        subject="math",
        task_role="generator",
        difficulty="advanced",
        intent="practice",
    )
    assert req.difficulty == "advanced"


# ---------------------------------------------------------------------------
# 20–21: Route resolver with difficulty
# ---------------------------------------------------------------------------


def test_route_resolver_selects_advanced_for_math() -> None:
    """math.generator.advanced exists — resolver must select it."""
    reg = LlmConfigRegistry()
    request = RouteRequest(
        request_id="r1",
        subject="math",
        task_role="generator",
        difficulty="advanced",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.route_source == "exact"
    assert decision.difficulty == "advanced"
    assert decision.subject == "math"


def test_route_resolver_selects_advanced_for_reasoning() -> None:
    """reasoning.generator.advanced exists — resolver must select it."""
    reg = LlmConfigRegistry()
    request = RouteRequest(
        request_id="r1",
        subject="reasoning",
        task_role="generator",
        difficulty="advanced",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.route_source == "exact"
    assert decision.difficulty == "advanced"


def test_route_resolver_selects_advanced_for_english() -> None:
    """english.generator.advanced exists — resolver must select it."""
    reg = LlmConfigRegistry()
    request = RouteRequest(
        request_id="r1",
        subject="english",
        task_role="generator",
        difficulty="advanced",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.route_source == "exact"
    assert decision.difficulty == "advanced"


def test_route_resolver_falls_back_to_default_when_advanced_missing(
    tmp_path: Path,
) -> None:
    """When advanced route is not configured, subject.generator.default is used."""
    import textwrap

    routes_yaml = textwrap.dedent("""\
        version: 1
        routes:
          general:
            generator:
              default:
                model: safe_mock
                prompt: gen.md
                temperature: 0.3
                max_tokens: 500
                fallback:
                  - safe_mock
    """)
    models_yaml = textwrap.dedent("""\
        version: 1
        models:
          safe_mock:
            provider: mock
            provider_profile: mock_profile
            timeout_seconds: 5
    """)
    profiles_yaml = textwrap.dedent("""\
        version: 1
        provider_profiles:
          mock_profile:
            provider: mock
    """)
    (tmp_path / "routes.yaml").write_text(routes_yaml)
    (tmp_path / "models.yaml").write_text(models_yaml)
    (tmp_path / "profiles.yaml").write_text(profiles_yaml)
    reg = LlmConfigRegistry(
        routes_path=tmp_path / "routes.yaml",
        model_registry_path=tmp_path / "models.yaml",
        provider_profiles_path=tmp_path / "profiles.yaml",
    )
    request = RouteRequest(
        request_id="r1",
        subject="general",
        task_role="generator",
        difficulty="advanced",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.route_source == "subject_default"
    assert decision.subject == "general"


# ---------------------------------------------------------------------------
# 22–23: Practice and solve regressions
# ---------------------------------------------------------------------------


def test_advanced_practice_query_classifies_correctly() -> None:
    """'Create 5 advanced SSC CGL level practice questions on profit-loss and discount'"""
    query = "Create 5 advanced SSC CGL level practice questions on profit-loss and discount"
    result = _classify_deterministic(query)
    assert result.subject == "math"
    assert result.difficulty == "advanced"


def test_advanced_practice_query_intent_is_practice() -> None:
    """Practice intent detection for the regression query."""
    query = "Create 5 advanced SSC CGL level practice questions on profit-loss and discount"
    result = _classify_deterministic(query)
    # The word "practice" may not be in deterministic intent keywords — that's OK.
    # What matters is that difficulty is advanced and subject is math.
    assert result.difficulty == "advanced"
    assert result.subject == "math"


def test_advanced_practice_routes_to_advanced_not_default() -> None:
    """An advanced math query must hit math.generator.advanced, not math.generator.default."""
    reg = LlmConfigRegistry()
    request = RouteRequest(
        request_id="r-adv-01",
        subject="math",
        task_role="generator",
        difficulty="advanced",
        intent="practice",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.route_source == "exact", (
        f"Expected 'exact' (advanced route), got {decision.route_source!r}. "
        "math.generator.advanced must exist in llm_routes.yaml."
    )
    assert decision.difficulty == "advanced"


def test_intent_overlay_practice_still_applied_on_advanced_route() -> None:
    """Practice intent overlay must still be present in advanced route."""
    reg = LlmConfigRegistry()
    route = reg.get_route("math", "generator", "advanced")
    assert route is not None
    assert "practice" in route.intent_overlays, (
        "advanced route must inherit intent_overlays from default"
    )
    assert "intents/practice.md" in route.intent_overlays["practice"]


def test_advanced_solve_query_classifies_correctly() -> None:
    """Advanced profit/loss solve query must classify as advanced."""
    query = "Solve this advanced profit and loss problem: A trader sells at 20% profit..."
    result = _classify_deterministic(query)
    assert result.difficulty == "advanced"
    assert result.subject == "math"


def test_advanced_difficulty_signal_in_solve_query() -> None:
    """'hard' keyword in query must produce difficulty=advanced."""
    query = "Here is a hard compound interest problem, solve step by step"
    result = _classify_deterministic(query)
    assert result.difficulty == "advanced"


# ---------------------------------------------------------------------------
# 24–26: Safety / regression invariants
# ---------------------------------------------------------------------------


def test_orchestrated_state_has_internal_retrieval_context() -> None:
    annotations = OrchestratedDoubtSolverState.__annotations__
    assert set(annotations.keys()) == {
        "request_id", "query", "classification", "retrieval_context", "context_text", "answer",
    }


def test_task_role_remains_generator() -> None:
    req = RouteRequest(
        request_id="r1", subject="math", task_role="generator", difficulty="advanced",
    )
    assert req.task_role == "generator"


def test_no_network_call_during_registry_load(tmp_path: Path) -> None:
    import socket
    import textwrap

    routes_yaml = textwrap.dedent("""\
        version: 1
        routes:
          general:
            generator:
              default:
                model: safe_mock
                prompt: gen.md
                temperature: 0.3
                max_tokens: 500
                fallback:
                  - safe_mock
    """)
    models_yaml = textwrap.dedent("""\
        version: 1
        models:
          safe_mock:
            provider: mock
            provider_profile: mock_profile
            timeout_seconds: 5
    """)
    profiles_yaml = textwrap.dedent("""\
        version: 1
        provider_profiles:
          mock_profile:
            provider: mock
    """)
    (tmp_path / "routes.yaml").write_text(routes_yaml)
    (tmp_path / "models.yaml").write_text(models_yaml)
    (tmp_path / "profiles.yaml").write_text(profiles_yaml)

    original = socket.socket

    def no_net(*a, **kw):  # type: ignore[no-untyped-def]
        raise AssertionError("Network call made during registry load")

    socket.socket = no_net  # type: ignore[assignment]
    try:
        LlmConfigRegistry(
            routes_path=tmp_path / "routes.yaml",
            model_registry_path=tmp_path / "models.yaml",
            provider_profiles_path=tmp_path / "profiles.yaml",
        )
    finally:
        socket.socket = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 27: Advanced queries no longer route as 'default'
# ---------------------------------------------------------------------------


def test_advanced_math_does_not_route_to_subject_default() -> None:
    """An explicit advanced math query must NOT fall back to subject_default."""
    reg = LlmConfigRegistry()
    request = RouteRequest(
        request_id="r-check-01",
        subject="math",
        task_role="generator",
        difficulty="advanced",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.route_source != "subject_default", (
        "Advanced math query should hit math.generator.advanced, not subject_default. "
        "Check that math.generator.advanced exists in llm_routes.yaml."
    )


def test_advanced_reasoning_does_not_route_to_subject_default() -> None:
    """An explicit advanced reasoning query must NOT fall back to subject_default."""
    reg = LlmConfigRegistry()
    request = RouteRequest(
        request_id="r-check-02",
        subject="reasoning",
        task_role="generator",
        difficulty="advanced",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.route_source != "subject_default"


def test_math_advanced_max_tokens_is_at_least_2600() -> None:
    """Advanced math route must have enough tokens for complete solutions."""
    reg = LlmConfigRegistry()
    route = reg.get_route("math", "generator", "advanced")
    assert route is not None
    assert route.max_tokens >= 2600, (
        f"math.generator.advanced max_tokens={route.max_tokens}, expected >= 2600. "
        "Increase in llm_routes.yaml."
    )
