"""
app/tests/test_intent_overlay.py
----------------------------------
Tests for intent-aware generator prompt overlays (Parts B/F/G/H).

Covers:
 1.  QueryClassification accepts practice_question.
 2.  QueryClassification accepts visualize_question.
 3.  QueryClassification rejects unknown intent values.
 4.  _ORCHESTRATED_INTENT_MAP covers all QueryClassification intent values.
 5.  practice_question maps to 'practice'.
 6.  visualize_question maps to 'visualize'.
 7.  unknown maps to 'explain'.
 8.  solve_question maps to 'solve'.
 9.  explain_concept maps to 'explain'.
10.  explain_option maps to 'explain'.
11.  general_doubt maps to 'explain'.
12.  RouteEntry accepts valid intent_overlays.
13.  RouteEntry rejects unknown intent key in intent_overlays.
14.  RouteEntry rejects unsafe path in intent_overlays.
15.  ResolvedRouteEntry carries intent_overlays through.
16.  LlmConfigRegistry resolves intent_overlays from llm_routes.yaml.
17.  Production registry math.generator.default has solve/explain/practice/visualize overlays.
18.  Production registry reasoning.generator.default has intent overlays.
19.  Production registry english.generator.default has intent overlays.
20.  Production registry general.generator.default has intent overlays.
21.  RouteDecision carries intent_overlays from route resolver.
22.  DoubtSolverClassification intent field reflects normalized intents.
23.  graph state remains exactly 5 fields in OrchestratedDoubtSolverState.
24.  task_role remains 'generator' (regression).
25.  RouteResolver route selection unchanged with subject+generator+difficulty.
26.  No real provider calls in any test here.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from graphs.doubt_solver_graph import (
    OrchestratedDoubtSolverState,
    _map_to_orchestrated_classification,
)
from schemas.doubt_solver import QueryClassification
from schemas.llm_routing import (
    ResolvedRouteEntry,
    RouteEntry,
    RouteRequest,
)
from services.classification.academic_classifier import ACADEMIC_INTENT_MAP
from services.llm_orchestration.config_registry import LlmConfigRegistry
from services.llm_orchestration.route_resolver import resolve_route

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_ROUTES_YAML = textwrap.dedent("""\
    version: 1
    routes:
      general:
        generator:
          default:
            model: safe_mock
            prompt: gen.md
            temperature: 0.3
            max_tokens: 500
            intent_overlays:
              solve:
                - intents/solve.md
              explain:
                - intents/explain.md
              practice:
                - intents/practice.md
              visualize:
                - intents/visualize.md
            fallback:
              - safe_mock
""")

_MINIMAL_MODELS_YAML = textwrap.dedent("""\
    version: 1
    models:
      safe_mock:
        provider: mock
        provider_profile: mock_profile
        timeout_seconds: 5
""")

_MINIMAL_PROFILES_YAML = textwrap.dedent("""\
    version: 1
    provider_profiles:
      mock_profile:
        provider: mock
""")


def _write(tmp_path: Path, rel: str, content: str) -> None:
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8")


def _make_minimal_registry(tmp_path: Path) -> LlmConfigRegistry:
    routes = tmp_path / "routes.yaml"
    models = tmp_path / "models.yaml"
    profiles = tmp_path / "profiles.yaml"
    routes.write_text(_MINIMAL_ROUTES_YAML, encoding="utf-8")
    models.write_text(_MINIMAL_MODELS_YAML, encoding="utf-8")
    profiles.write_text(_MINIMAL_PROFILES_YAML, encoding="utf-8")
    return LlmConfigRegistry(
        routes_path=routes,
        model_registry_path=models,
        provider_profiles_path=profiles,
    )


# ---------------------------------------------------------------------------
# Tests 1–2: QueryClassification schema
# ---------------------------------------------------------------------------


def test_query_classification_accepts_practice_question() -> None:
    cls = QueryClassification(intent="practice_question", confidence=0.8)
    assert cls.intent == "practice_question"


def test_query_classification_accepts_visualize_question() -> None:
    cls = QueryClassification(intent="visualize_question", confidence=0.8)
    assert cls.intent == "visualize_question"


def test_query_classification_rejects_unknown_intent() -> None:
    with pytest.raises((ValidationError, ValueError)):
        QueryClassification(intent="make_a_sandwich", confidence=0.8)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests 4–11: _ORCHESTRATED_INTENT_MAP completeness
# ---------------------------------------------------------------------------

_ALL_RAW_INTENTS = [
    "solve_question",
    "explain_concept",
    "explain_option",
    "general_doubt",
    "practice_question",
    "visualize_question",
    "unknown",
]


@pytest.mark.parametrize("raw_intent", _ALL_RAW_INTENTS)
def test_orchestrated_intent_map_covers_all_raw_intents(raw_intent: str) -> None:
    assert raw_intent in ACADEMIC_INTENT_MAP, (
        f"ACADEMIC_INTENT_MAP is missing key {raw_intent!r}. "
        "Add it to the map in doubt_solver_graph.py."
    )


def test_practice_question_maps_to_practice() -> None:
    assert ACADEMIC_INTENT_MAP["practice_question"] == "practice"


def test_visualize_question_maps_to_visualize() -> None:
    assert ACADEMIC_INTENT_MAP["visualize_question"] == "visualize"


def test_unknown_maps_to_explain() -> None:
    assert ACADEMIC_INTENT_MAP["unknown"] == "explain"


def test_solve_question_maps_to_solve() -> None:
    assert ACADEMIC_INTENT_MAP["solve_question"] == "solve"


def test_explain_concept_maps_to_explain() -> None:
    assert ACADEMIC_INTENT_MAP["explain_concept"] == "explain"


def test_explain_option_maps_to_explain() -> None:
    assert ACADEMIC_INTENT_MAP["explain_option"] == "explain"


def test_general_doubt_maps_to_explain() -> None:
    assert ACADEMIC_INTENT_MAP["general_doubt"] == "explain"


# ---------------------------------------------------------------------------
# Tests 12–14: RouteEntry intent_overlays schema validation
# ---------------------------------------------------------------------------


def test_route_entry_accepts_valid_intent_overlays() -> None:
    entry = RouteEntry(
        model="safe_mock",
        prompt="gen.md",
        temperature=0.2,
        max_tokens=500,
        intent_overlays={
            "solve": ["intents/solve.md"],
            "explain": ["intents/explain.md"],
            "practice": ["intents/practice.md"],
            "visualize": ["intents/visualize.md"],
        },
    )
    assert "solve" in entry.intent_overlays
    assert entry.intent_overlays["solve"] == ["intents/solve.md"]


def test_route_entry_rejects_unknown_intent_key() -> None:
    with pytest.raises((ValidationError, ValueError)):
        RouteEntry(
            model="safe_mock",
            prompt="gen.md",
            temperature=0.2,
            max_tokens=500,
            intent_overlays={"ask": ["intents/ask.md"]},  # type: ignore[arg-type]
        )


def test_route_entry_rejects_unsafe_intent_overlay_path() -> None:
    """Overlay paths inside intent_overlays must also be safe relative .md paths."""
    with pytest.raises((ValidationError, ValueError)):
        RouteEntry(
            model="safe_mock",
            prompt="gen.md",
            temperature=0.2,
            max_tokens=500,
            intent_overlays={"solve": ["../../../etc/passwd.md"]},
        )


def test_route_entry_rejects_absolute_intent_overlay_path() -> None:
    with pytest.raises((ValidationError, ValueError)):
        RouteEntry(
            model="safe_mock",
            prompt="gen.md",
            temperature=0.2,
            max_tokens=500,
            intent_overlays={"solve": ["/absolute/path.md"]},
        )


# ---------------------------------------------------------------------------
# Test 15: ResolvedRouteEntry carries intent_overlays
# ---------------------------------------------------------------------------


def test_resolved_route_entry_carries_intent_overlays() -> None:
    entry = ResolvedRouteEntry(
        model="safe_mock",
        prompt="gen.md",
        overlays=[],
        intent_overlays={
            "solve": ["intents/solve.md"],
            "practice": ["intents/practice.md"],
        },
        temperature=0.2,
        max_tokens=500,
        fallback=[],
    )
    assert entry.intent_overlays["solve"] == ["intents/solve.md"]
    assert entry.intent_overlays["practice"] == ["intents/practice.md"]


# ---------------------------------------------------------------------------
# Test 16: LlmConfigRegistry resolves intent_overlays from YAML
# ---------------------------------------------------------------------------


def test_registry_resolves_intent_overlays_from_yaml(tmp_path: Path) -> None:
    reg = _make_minimal_registry(tmp_path)
    route = reg.get_route("general", "generator", "default")
    assert route is not None
    assert "solve" in route.intent_overlays
    assert route.intent_overlays["solve"] == ["intents/solve.md"]
    assert route.intent_overlays["practice"] == ["intents/practice.md"]
    assert route.intent_overlays["visualize"] == ["intents/visualize.md"]


# ---------------------------------------------------------------------------
# Tests 17–20: Production registry intent overlays
# ---------------------------------------------------------------------------


def test_production_math_generator_has_intent_overlays() -> None:
    reg = LlmConfigRegistry()
    route = reg.get_route("math", "generator", "default")
    assert route is not None
    assert "solve" in route.intent_overlays
    assert "explain" in route.intent_overlays
    assert "practice" in route.intent_overlays
    assert "visualize" in route.intent_overlays
    assert "intents/solve.md" in route.intent_overlays["solve"]
    assert "intents/explain.md" in route.intent_overlays["explain"]
    assert "intents/practice.md" in route.intent_overlays["practice"]
    assert "intents/visualize.md" in route.intent_overlays["visualize"]


def test_production_reasoning_generator_has_intent_overlays() -> None:
    reg = LlmConfigRegistry()
    route = reg.get_route("reasoning", "generator", "default")
    assert route is not None
    for intent in ("solve", "explain", "practice", "visualize"):
        assert intent in route.intent_overlays, f"Missing intent overlay key: {intent}"


def test_production_english_generator_has_intent_overlays() -> None:
    reg = LlmConfigRegistry()
    route = reg.get_route("english", "generator", "default")
    assert route is not None
    for intent in ("solve", "explain", "practice", "visualize"):
        assert intent in route.intent_overlays, f"Missing intent overlay key: {intent}"


def test_production_general_generator_has_intent_overlays() -> None:
    reg = LlmConfigRegistry()
    route = reg.get_route("general", "generator", "default")
    assert route is not None
    for intent in ("solve", "explain", "practice", "visualize"):
        assert intent in route.intent_overlays, f"Missing intent overlay key: {intent}"


# ---------------------------------------------------------------------------
# Test 21: RouteDecision carries intent_overlays through route resolver
# ---------------------------------------------------------------------------


def test_route_decision_carries_intent_overlays(tmp_path: Path) -> None:
    reg = _make_minimal_registry(tmp_path)
    request = RouteRequest(
        request_id="test-01",
        subject="general",
        task_role="generator",
        difficulty="default",
        intent="solve",
    )
    decision = resolve_route(request, registry=reg)
    assert "solve" in decision.intent_overlays
    assert decision.intent_overlays["solve"] == ["intents/solve.md"]
    assert decision.intent == "solve"


# ---------------------------------------------------------------------------
# Test 22: DoubtSolverClassification intent reflects normalized intents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_intent", "expected"),
    [
        ("solve_question", "solve"),
        ("explain_concept", "explain"),
        ("explain_option", "explain"),
        ("general_doubt", "explain"),
        ("practice_question", "practice"),
        ("visualize_question", "visualize"),
        ("unknown", "explain"),
    ],
)
def test_map_to_orchestrated_classification_normalizes_intent(
    raw_intent: str, expected: str
) -> None:
    raw = QueryClassification(intent=raw_intent, confidence=0.8)  # type: ignore[arg-type]
    result = _map_to_orchestrated_classification(raw)
    assert result["intent"] == expected


# ---------------------------------------------------------------------------
# Test 23: OrchestratedDoubtSolverState includes retrieval_context
# ---------------------------------------------------------------------------


def test_orchestrated_state_has_internal_retrieval_context() -> None:
    annotations = OrchestratedDoubtSolverState.__annotations__
    assert set(annotations.keys()) == {
        "request_id",
        "query",
        "original_query",
        "actor_id",
        "conversation_id",
        "turn_id",
        "language",
        "exam_id",
        "exam_stage",
        "exam_profile_id",
        "classification",
        "retrieval_context",
        "context_text",
        "answer",
        "final_answer",
        "credit_error",
        "credit_error_details",
        "conversation_context",
        "conversation_relation",
        "conversation_preparation",
        "query_classification",
        "source_modality",
        "response_type",
        "practice_test_id",
        "fresh_evidence",
    }, f"State fields changed: {set(annotations.keys())}"


# ---------------------------------------------------------------------------
# Test 24: task_role remains 'generator' (regression)
# ---------------------------------------------------------------------------


def test_route_request_task_role_is_generator(tmp_path: Path) -> None:
    reg = _make_minimal_registry(tmp_path)
    request = RouteRequest(
        request_id="test-02",
        subject="general",
        task_role="generator",
        difficulty="default",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.task_role == "generator"


# ---------------------------------------------------------------------------
# Test 25: RouteResolver route selection unchanged (regression)
# ---------------------------------------------------------------------------


def test_route_resolver_selects_correct_route_with_subject_difficulty(
    tmp_path: Path,
) -> None:
    reg = _make_minimal_registry(tmp_path)
    request = RouteRequest(
        request_id="test-03",
        subject="general",
        task_role="generator",
        difficulty="default",
    )
    decision = resolve_route(request, registry=reg)
    assert decision.subject == "general"
    assert decision.task_role == "generator"
    assert decision.difficulty == "default"
    assert decision.model == "safe_mock"


# ---------------------------------------------------------------------------
# Test 26: No real provider calls (structural guard)
# ---------------------------------------------------------------------------


def test_no_provider_calls_in_registry_load(tmp_path: Path) -> None:
    """Registry build must not call any network or provider SDK."""
    import socket

    original = socket.socket

    def no_network(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Network call made during registry load")

    socket.socket = no_network  # type: ignore[assignment]
    try:
        _make_minimal_registry(tmp_path)
    finally:
        socket.socket = original  # type: ignore[assignment]
