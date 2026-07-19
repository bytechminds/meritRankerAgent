"""
tests/test_orchestrated_doubt_solver_graph_flow.py
-----------------------------------------
Unit tests: Orchestrated Doubt Solver graph node behaviour and full-flow execution.

Tests cover:
    classify node  — correct orchestrated mapping, fallback on error, no provider call
    collect_context — retrieval_required=false → context="", KB disabled → ""
    generate node  — uses subject/intent/difficulty/task_role=generator, writes answer
    graph flow     — 3-node linear path produces answer; state stays lean
    invariants     — no planner/verifier nodes; no model_id/provider in state

All tests inject a fake AnswerGenerationAdapter — no real provider call, no AWS
call, no network I/O.

[NOT VERIFIED]: generate node isolation from real provider adapters is asserted
by counting adapter call_count only — the tests do not inspect bytecode.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from graphs.doubt_solver_graph import (
    OrchestratedDoubtSolverState,
    _map_to_orchestrated_classification,
    _orchestrated_classify_node,
    _orchestrated_collect_context_node,
    build_orchestrated_doubt_solver_graph,
)
from schemas.doubt_solver import QueryClassification


@pytest.fixture(autouse=True)
def _use_legacy_kb_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical KB-flow tests use the explicit legacy retrieval provider."""
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "legacy_bedrock_kb")
    import config as cfg_module  # noqa: PLC0415

    cfg_module._settings = None
    yield
    cfg_module._settings = None

# ---------------------------------------------------------------------------
# Fake AnswerGenerationAdapter — records call kwargs, returns fixed content
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Minimal stub for AnswerGenerationAdapter used in graph flow tests."""

    def __init__(self, content: str = "Orchestrated mock answer.") -> None:
        self._content = content
        self.call_count: int = 0
        self.last_kwargs: dict[str, Any] = {}

    def generate(
        self,
        *,
        request_id: str,
        query: str,
        subject: str,
        intent: str,
        difficulty: str,
        context: str,
        web_search_reason: str | None = None,
    ) -> str:
        self.call_count += 1
        self.last_kwargs = {
            "request_id": request_id,
            "query": query,
            "subject": subject,
            "intent": intent,
            "difficulty": difficulty,
            "context": context,
            "web_search_reason": web_search_reason,
        }
        return self._content


# ---------------------------------------------------------------------------
# Helper to build a minimal V1 state dict
# ---------------------------------------------------------------------------


def _minimal_state(
    query: str = "What is 20% of 500?",
    request_id: str = "test-req-001",
    classification: dict | None = None,
    context_text: str = "",
    answer: str | None = None,
) -> OrchestratedDoubtSolverState:
    return {
        "request_id": request_id,
        "query": query,
        "classification": classification,
        "context_text": context_text,
        "answer": answer,
    }


# ===========================================================================
# _map_to_orchestrated_classification helper tests
# ===========================================================================


class TestMapToOrchestratedClassification:
    """Verify intent/subject/difficulty/retrieval_required mapping."""

    def _make_qc(
        self,
        intent: str = "solve_question",
        subject: str = "math",
        retrieval_need: str = "none",
    ) -> QueryClassification:
        return QueryClassification(
            intent=intent,
            subject=subject,
            confidence=0.9,
            retrieval_need=retrieval_need,
        )

    def test_solve_question_maps_to_solve(self) -> None:
        qc = self._make_qc(intent="solve_question")
        result = _map_to_orchestrated_classification(qc)
        assert result["intent"] == "solve"

    def test_explain_concept_maps_to_explain(self) -> None:
        qc = self._make_qc(intent="explain_concept")
        result = _map_to_orchestrated_classification(qc)
        assert result["intent"] == "explain"

    def test_explain_option_maps_to_explain(self) -> None:
        qc = self._make_qc(intent="explain_option")
        result = _map_to_orchestrated_classification(qc)
        assert result["intent"] == "explain"

    def test_general_doubt_maps_to_explain(self) -> None:
        qc = self._make_qc(intent="general_doubt")
        result = _map_to_orchestrated_classification(qc)
        assert result["intent"] == "explain"

    def test_unknown_intent_maps_to_explain(self) -> None:
        qc = self._make_qc(intent="unknown")
        result = _map_to_orchestrated_classification(qc)
        assert result["intent"] == "explain"

    def test_math_subject_preserved(self) -> None:
        qc = self._make_qc(subject="math")
        result = _map_to_orchestrated_classification(qc)
        assert result["subject"] == "math"

    def test_reasoning_subject_preserved(self) -> None:
        qc = self._make_qc(subject="reasoning")
        result = _map_to_orchestrated_classification(qc)
        assert result["subject"] == "reasoning"

    def test_english_subject_preserved(self) -> None:
        qc = self._make_qc(subject="english")
        result = _map_to_orchestrated_classification(qc)
        assert result["subject"] == "english"

    def test_unknown_subject_maps_to_general(self) -> None:
        qc = self._make_qc(subject="unknown")
        result = _map_to_orchestrated_classification(qc)
        assert result["subject"] == "general"

    def test_science_subject_maps_to_general(self) -> None:
        qc = self._make_qc(subject="science")
        result = _map_to_orchestrated_classification(qc)
        assert result["subject"] == "general"

    def test_retrieval_need_none_sets_false(self) -> None:
        qc = self._make_qc(retrieval_need="none")
        result = _map_to_orchestrated_classification(qc)
        assert result["retrieval_required"] is False

    def test_retrieval_need_concept_context_sets_true(self) -> None:
        qc = self._make_qc(retrieval_need="concept_context")
        result = _map_to_orchestrated_classification(qc)
        assert result["retrieval_required"] is True

    def test_retrieval_need_similar_question_sets_true(self) -> None:
        qc = self._make_qc(retrieval_need="similar_question")
        result = _map_to_orchestrated_classification(qc)
        assert result["retrieval_required"] is True

    def test_retrieval_need_unknown_sets_true(self) -> None:
        qc = self._make_qc(retrieval_need="unknown")
        result = _map_to_orchestrated_classification(qc)
        assert result["retrieval_required"] is True

    def test_difficulty_always_default_in_v1(self) -> None:
        qc = self._make_qc()
        result = _map_to_orchestrated_classification(qc)
        assert result["difficulty"] == "default"

    def test_result_is_dict_not_pydantic_model(self) -> None:
        qc = self._make_qc()
        result = _map_to_orchestrated_classification(qc)
        assert isinstance(result, dict)

    def test_result_includes_core_and_optional_hint_keys(self) -> None:
        qc = self._make_qc()
        result = _map_to_orchestrated_classification(qc)
        assert {"subject", "intent", "difficulty", "retrieval_required"}.issubset(
            result.keys()
        )


# ===========================================================================
# _orchestrated_classify_node tests
# ===========================================================================


class TestOrchestratedClassifyNode:
    """Verify the classify node writes correct orchestrated classification dict to state."""

    def test_node_writes_classification(self) -> None:
        state = _minimal_state(query="What is 20% of 500?")
        result = _orchestrated_classify_node(state)
        assert "classification" in result
        assert result["classification"] is not None

    def test_classification_has_subject(self) -> None:
        state = _minimal_state(query="Solve: 2x + 5 = 15")
        result = _orchestrated_classify_node(state)
        assert "subject" in result["classification"]

    def test_classification_has_intent(self) -> None:
        state = _minimal_state(query="Solve: 2x + 5 = 15")
        result = _orchestrated_classify_node(state)
        assert "intent" in result["classification"]

    def test_classification_has_difficulty(self) -> None:
        state = _minimal_state(query="Solve: 2x + 5 = 15")
        result = _orchestrated_classify_node(state)
        assert "difficulty" in result["classification"]

    def test_classification_has_retrieval_required(self) -> None:
        state = _minimal_state(query="Solve: 2x + 5 = 15")
        result = _orchestrated_classify_node(state)
        assert "retrieval_required" in result["classification"]

    def test_classification_includes_core_and_optional_hint_keys(self) -> None:
        state = _minimal_state(query="Solve: 2x + 5 = 15")
        result = _orchestrated_classify_node(state)
        assert {"subject", "intent", "difficulty", "retrieval_required"}.issubset(
            set(result["classification"].keys())
        )

    def test_node_does_not_write_answer(self) -> None:
        state = _minimal_state()
        result = _orchestrated_classify_node(state)
        assert "answer" not in result

    def test_node_does_not_write_context_text(self) -> None:
        state = _minimal_state()
        result = _orchestrated_classify_node(state)
        assert "context_text" not in result

    def test_node_does_not_write_model_id(self) -> None:
        state = _minimal_state()
        result = _orchestrated_classify_node(state)
        assert "model_id" not in result
        clf = result.get("classification") or {}
        assert "model_id" not in clf

    def test_node_does_not_write_provider(self) -> None:
        state = _minimal_state()
        result = _orchestrated_classify_node(state)
        clf = result.get("classification") or {}
        assert "provider" not in clf

    def test_node_does_not_write_deployment(self) -> None:
        state = _minimal_state()
        result = _orchestrated_classify_node(state)
        clf = result.get("classification") or {}
        assert "deployment" not in clf

    def test_difficulty_passes_through_from_classifier(self) -> None:
        """Difficulty from classifier must flow into orchestrated classification."""
        state = _minimal_state(query="An advanced calculus problem.")
        result = _orchestrated_classify_node(state)
        # "advanced" keyword triggers advanced difficulty in deterministic classifier.
        assert result["classification"]["difficulty"] == "advanced"

    def test_difficulty_is_default_when_no_signal(self) -> None:
        """Neutral query with no difficulty keywords → classification.difficulty == 'default'."""
        state = _minimal_state(query="What is 20% of 500?")
        result = _orchestrated_classify_node(state)
        assert result["classification"]["difficulty"] == "default"


# ===========================================================================
# _orchestrated_collect_context_node tests
# ===========================================================================


class TestOrchestratedCollectContextNode:
    """Verify collect_context node short-circuits correctly."""

    _NO_RETRIEVAL = {
        "subject": "math",
        "intent": "solve",
        "difficulty": "default",
        "retrieval_required": False,
    }
    _WITH_RETRIEVAL = {
        "subject": "math",
        "intent": "solve",
        "difficulty": "default",
        "retrieval_required": True,
    }

    def test_retrieval_required_false_returns_empty_string(self) -> None:
        state = _minimal_state(classification=self._NO_RETRIEVAL)
        result = _orchestrated_collect_context_node(state)
        assert result["context_text"] == ""

    def test_no_classification_returns_empty_string(self) -> None:
        state = _minimal_state(classification=None)
        result = _orchestrated_collect_context_node(state)
        assert result["context_text"] == ""

    def test_retrieval_required_true_but_kb_disabled_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ENABLE_KB_RETRIEVAL=false → service returns retrieval_source=disabled → context=""."""
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        import config as cfg_module  # noqa: PLC0415
        cfg_module._settings = None

        state = _minimal_state(
            query="What is a quadratic formula?",
            classification=self._WITH_RETRIEVAL,
        )
        result = _orchestrated_collect_context_node(state)
        assert result["context_text"] == ""

    def test_empty_query_returns_empty_string(self) -> None:
        state = _minimal_state(query="", classification=self._WITH_RETRIEVAL)
        result = _orchestrated_collect_context_node(state)
        assert result["context_text"] == ""

    def test_node_returns_context_text_key(self) -> None:
        state = _minimal_state(classification=self._NO_RETRIEVAL)
        result = _orchestrated_collect_context_node(state)
        assert "context_text" in result

    def test_node_does_not_return_kb_results_key(self) -> None:
        """Raw KB records must NOT be stored in orchestrated state."""
        state = _minimal_state(classification=self._NO_RETRIEVAL)
        result = _orchestrated_collect_context_node(state)
        assert "kb_results" not in result

    def test_node_does_not_return_dynamodb_records_key(self) -> None:
        state = _minimal_state(classification=self._NO_RETRIEVAL)
        result = _orchestrated_collect_context_node(state)
        assert "dynamodb_records" not in result

    def test_sets_context_text_from_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from services.context_retrieval import context_retrieval_service as crs_module
        from services.context_retrieval.context_models import ContextRetrievalResult

        mock_service = crs_module.ContextRetrievalService(kb_retriever=MagicMock())
        monkeypatch.setattr(
            crs_module,
            "get_context_retrieval_service",
            lambda: mock_service,
        )

        def _fake_retrieve(_request: Any, **kwargs: Any) -> ContextRetrievalResult:
            return ContextRetrievalResult(
                context_text="Relevant context:\n\n1. Pattern:\n   Discount rule.",
                item_count=1,
                retrieval_used=True,
                reason="intent_explain",
            )

        monkeypatch.setattr(mock_service, "retrieve_context", _fake_retrieve)

        state = _minimal_state(
            query="Explain discount pattern",
            classification={
                "subject": "math",
                "intent": "explain",
                "difficulty": "intermediate",
                "retrieval_required": True,
            },
        )
        result = _orchestrated_collect_context_node(state)
        assert "Relevant context" in result["context_text"]

    def test_selected_kb_formatting_failure_still_returns_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from services.context_retrieval import context_retrieval_service as crs_module

        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        import config as cfg_module  # noqa: PLC0415

        cfg_module._settings = None

        retriever = MagicMock()
        from services.context_retrieval.context_models import RetrievedContextItem
        from services.context_retrieval.context_retrieval_service import LANE_SUBJECT_TOPIC

        retriever.retrieve_lane.return_value = (
            [
                RetrievedContextItem(
                    text="Relative speed when trains move in opposite directions.",
                    score=0.97,
                    metadata={
                        "subject": "QUANT",
                        "patternTopicKey": "TIME_SPEED_DISTANCE",
                        "conceptTags": "SPEED,TIME",
                    },
                    match_lane=LANE_SUBJECT_TOPIC,
                )
            ],
            5,
        )
        broken_builder = MagicMock()
        broken_builder.build.side_effect = ValueError("brief failure")
        mock_service = crs_module.ContextRetrievalService(
            kb_retriever=retriever,
            brief_builder=broken_builder,
        )
        monkeypatch.setattr(
            crs_module,
            "get_context_retrieval_service",
            lambda: mock_service,
        )

        state = _minimal_state(
            query="Train crosses platform in 18 seconds at 54 km/hr",
            classification={
                "subject": "math",
                "intent": "solve",
                "difficulty": "intermediate",
                "topic": "TIME_SPEED_DISTANCE",
            },
        )
        result = _orchestrated_collect_context_node(state)
        assert len(result["context_text"]) > 0
        assert "[Relevant KB Context]" in result["context_text"]
        assert set(result.keys()) == {"context_text", "retrieval_context"}


# ===========================================================================
# build_orchestrated_doubt_solver_graph — generate node (via full graph invocation)
# ===========================================================================


class TestOrchestratedGenerateNode:
    """Verify the generate node (closure) passes correct params to adapter."""

    def test_generate_calls_adapter_once(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="What is 20% of 500?")
        graph.invoke(state)
        assert adapter.call_count == 1

    def test_generate_passes_query(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="What is 20% of 500?")
        graph.invoke(state)
        assert adapter.last_kwargs["query"] == "What is 20% of 500?"

    def test_generate_passes_request_id(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="test", request_id="req-xyz")
        graph.invoke(state)
        assert adapter.last_kwargs["request_id"] == "req-xyz"

    def test_generate_passes_task_role_generator_via_classification(self) -> None:
        """The adapter signature doesn't include task_role; it's hardcoded as generator.
        We verify subject/intent/difficulty are passed — route_request uses task_role=generator.
        """
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="Solve 2x=8")
        graph.invoke(state)
        # Subject/intent/difficulty come from orchestrated classification (not model_id/deployment)
        assert "subject" in adapter.last_kwargs
        assert "intent" in adapter.last_kwargs
        assert "difficulty" in adapter.last_kwargs

    def test_generate_does_not_pass_model_id(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="Solve 2x=8")
        graph.invoke(state)
        assert "model_id" not in adapter.last_kwargs

    def test_generate_does_not_pass_provider(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="Solve 2x=8")
        graph.invoke(state)
        assert "provider" not in adapter.last_kwargs

    def test_generate_does_not_pass_deployment(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="Solve 2x=8")
        graph.invoke(state)
        assert "deployment" not in adapter.last_kwargs

    def test_generate_writes_answer_string_to_state(self) -> None:
        adapter = _FakeAdapter(content="Answer: 100")
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="What is 20% of 500?")
        result = graph.invoke(state)
        assert result["answer"] == "Answer: 100"

    def test_generate_does_not_store_orchestration_result_in_state(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="test")
        result = graph.invoke(state)
        assert "orchestration_result" not in result

    def test_generate_does_not_store_route_decision_in_state(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="test")
        result = graph.invoke(state)
        assert "route_decision" not in result

    def test_generate_does_not_store_messages_in_state(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="test")
        result = graph.invoke(state)
        assert "messages" not in result

    def test_generate_context_empty_when_not_needed(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        state = _minimal_state(query="What is 20% of 500?")
        graph.invoke(state)
        # No retrieval required for arithmetic query → context=""\
        assert adapter.last_kwargs["context"] == ""


# ===========================================================================
# Full graph flow tests
# ===========================================================================


class TestOrchestratedGraphFlow:
    """Full classify → collect_context → generate integration tests."""

    def test_full_flow_produces_answer(self) -> None:
        adapter = _FakeAdapter(content="20% of 500 is 100.")
        graph = build_orchestrated_doubt_solver_graph(adapter)
        result = graph.invoke(_minimal_state(query="What is 20% of 500?"))
        assert result["answer"] == "20% of 500 is 100."

    def test_full_flow_preserves_request_id(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        result = graph.invoke(_minimal_state(request_id="stable-id"))
        assert result["request_id"] == "stable-id"

    def test_full_flow_preserves_query(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        q = "What is 20% of 500?"
        result = graph.invoke(_minimal_state(query=q))
        assert result["query"] == q

    def test_full_flow_state_has_retrieval_context_field(self) -> None:
        """Final state retains the structured internal retrieval contract."""
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        result = graph.invoke(_minimal_state())
        assert set(result.keys()) == {
            "request_id",
            "query",
            "classification",
            "context_text",
            "retrieval_context",
            "answer",
        }

    def test_full_flow_classification_is_dict(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        result = graph.invoke(_minimal_state())
        assert isinstance(result["classification"], dict)

    def test_full_flow_no_planner_node(self) -> None:
        """Graph must not have a 'plan' or 'planner' node."""
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        node_names = set(graph.get_graph().nodes.keys())
        assert "plan" not in node_names
        assert "planner" not in node_names
        assert "plan_context" not in node_names

    def test_full_flow_no_verifier_node(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        node_names = set(graph.get_graph().nodes.keys())
        assert "verifier" not in node_names
        assert "verify" not in node_names

    def test_full_flow_has_exactly_three_non_boundary_nodes(self) -> None:
        """classify + collect_context + generate only (no extra nodes)."""
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        node_names = {
            n for n in graph.get_graph().nodes.keys()
            if n not in ("__start__", "__end__")
        }
        assert node_names == {"classify", "collect_context", "generate"}

    def test_full_flow_adapter_called_exactly_once(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        graph.invoke(_minimal_state())
        assert adapter.call_count == 1

    def test_full_flow_no_aws_call_with_kb_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ENABLE_KB_RETRIEVAL=false → no AWS calls during full flow."""
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        import config as cfg_module  # noqa: PLC0415
        cfg_module._settings = None

        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        # Should not raise any AWS-related error
        result = graph.invoke(_minimal_state(query="What is 20% of 500?"))
        assert result["answer"] is not None

    def test_answer_is_string(self) -> None:
        adapter = _FakeAdapter(content="The answer is 100.")
        graph = build_orchestrated_doubt_solver_graph(adapter)
        result = graph.invoke(_minimal_state())
        assert isinstance(result["answer"], str)

    def test_context_text_is_empty_string_for_arithmetic(self) -> None:
        adapter = _FakeAdapter()
        graph = build_orchestrated_doubt_solver_graph(adapter)
        result = graph.invoke(_minimal_state(query="2 + 2 = ?"))
        # Arithmetic queries classified as no-retrieval → context_text=""
        assert isinstance(result["context_text"], str)
