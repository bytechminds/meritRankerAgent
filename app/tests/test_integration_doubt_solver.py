"""
app/tests/test_integration_doubt_solver.py
--------------------------------------------
Full pipeline integration tests for the Doubt Solver workflow.

Tests the complete end-to-end path from graph invocation through to the
final DoubtSolverResponse dict, covering:
    - all flags disabled (regression — identical to pre-Part-9 behaviour)
    - fake KB retrieval enabled
    - fake KB + DynamoDB enabled
    - complete response shape (all 12 expected fields present)
    - no full retrieved content exposed in response
    - main.py invoke() routing (doubt_solver and demo modes)

No real AWS credentials, network, or LLM calls required.
All external services are replaced with deterministic fakes/spies.

[NOT VERIFIED] AgentCore HTTP runtime E2E — these tests call invoke() directly.
               The HTTP transport layer (make dev → POST /invocations) is NOT
               exercised here; run manually with: make smoke-doubt-solver
"""

from __future__ import annotations

import pytest

import config as cfg_module
from graphs.doubt_solver_graph import build_doubt_solver_graph
from schemas.doubt_solver import DoubtSolverResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All fields that every DoubtSolverResponse must contain.
_REQUIRED_RESPONSE_FIELDS = frozenset(
    {
        "success",
        "request_id",
        "mode",
        "answer",
        "classification",
        "answer_source",
        "is_truncated",
        "needs_review",
        "used_retrieval",
        "context_used",
        "source_count",
    }
)


def _reset_settings() -> None:
    cfg_module._settings = None


@pytest.fixture(autouse=True)
def _use_legacy_kb_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Part 9 integration coverage remains on the explicit legacy provider."""
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "legacy_bedrock_kb")
    _reset_settings()
    yield
    _reset_settings()


def _make_graph_input(query: str = "Explain what percentage means") -> dict:
    """Return a fully initialised graph state dict (all Part 9 fields included)."""
    return {
        "request_id": "integration-test-req",
        "query": query,
        "user_id": "test-user",
        "mode": "doubt_solver",
        "language": "en",
        "classification": None,
        "answer": None,
        "answer_source": None,
        "is_truncated": False,
        "response": None,
        # Part 9 context-pipeline fields
        "should_retrieve": False,
        "kb_results": None,
        "dynamodb_records": None,
        "answer_context": None,
        "context_source_count": 0,
        "used_retrieval": False,
        "context_used": False,
        "service_error": False,
    }


def _fake_kb_with_content(content: str = "Context material."):
    """Return a fake retrieve_similar_context function yielding one result."""
    from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse  # noqa: PLC0415

    def _fake(query, max_results=None):
        return RetrievalResponse(
            query=query,
            results=[KnowledgeBaseResult(content=content, score=0.9)],
            result_count=1,
            retrieval_source="bedrock_kb",
        )

    return _fake


def _fake_empty_kb():
    """Return a fake retrieve_similar_context that returns no results."""
    from schemas.retrieval import RetrievalResponse  # noqa: PLC0415

    def _fake(query, max_results=None):
        return RetrievalResponse(
            query=query, results=[], result_count=0, retrieval_source="bedrock_kb"
        )

    return _fake


# ---------------------------------------------------------------------------
# Complete response shape
# ---------------------------------------------------------------------------


class TestFullResponseShape:
    """All flags disabled — verify every response field is present and typed."""

    @pytest.fixture(autouse=True)
    def _flags_off(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

    @pytest.fixture
    def response(self):
        graph = build_doubt_solver_graph()
        return graph.invoke(_make_graph_input("Explain what percentage means"))[
            "response"
        ]

    def test_all_required_fields_present(self, response):
        for field in _REQUIRED_RESPONSE_FIELDS:
            assert field in response, f"Missing response field: {field}"

    def test_success_is_true(self, response):
        assert response["success"] is True

    def test_mode_is_doubt_solver(self, response):
        assert response["mode"] == "doubt_solver"

    def test_request_id_preserved(self, response):
        assert response["request_id"] == "integration-test-req"

    def test_answer_is_non_empty_string(self, response):
        assert isinstance(response["answer"], str)
        assert len(response["answer"]) > 0

    def test_answer_source_is_valid_value(self, response):
        assert response["answer_source"] in {"mock", "llm", "fallback"}

    def test_answer_source_is_mock_by_default(self, response):
        assert response["answer_source"] == "mock"

    def test_is_truncated_is_bool(self, response):
        assert isinstance(response["is_truncated"], bool)

    def test_needs_review_is_bool(self, response):
        assert isinstance(response["needs_review"], bool)

    def test_used_retrieval_false_when_disabled(self, response):
        assert response["used_retrieval"] is False

    def test_context_used_false_when_disabled(self, response):
        assert response["context_used"] is False

    def test_source_count_zero_when_disabled(self, response):
        assert response["source_count"] == 0

    def test_classification_has_required_subfields(self, response):
        cls = response["classification"]
        assert isinstance(cls, dict)
        for subfield in ("intent", "subject", "confidence", "retrieval_need"):
            assert subfield in cls, f"Missing classification subfield: {subfield}"

    def test_response_passes_pydantic_validation(self, response):
        """Response dict must deserialise cleanly into DoubtSolverResponse."""
        model = DoubtSolverResponse.model_validate(response)
        assert model.success is True

    def test_response_does_not_contain_raw_context_fields(self, response):
        """Internal context strings must never appear in the public response."""
        response_str = str(response)
        for forbidden_key in ("answer_context", "kb_results", "dynamodb_records"):
            assert forbidden_key not in response_str, (
                f"Internal field '{forbidden_key}' leaked into response"
            )


# ---------------------------------------------------------------------------
# All flags disabled — regression guard
# ---------------------------------------------------------------------------


class TestAllFlagsDisabledRegression:
    """Regression: all-flags-off path must be unchanged from pre-Part-9."""

    @pytest.fixture(autouse=True)
    def _flags_off(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

    def test_dynamodb_never_called_when_flag_off(self, monkeypatch):
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415

        dynamo_called: list = []

        def _spy_dynamo(ids):
            dynamo_called.append(ids)
            return []

        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _spy_dynamo)
        build_doubt_solver_graph().invoke(_make_graph_input("Solve: x + 2 = 5"))

        assert dynamo_called == [], "DynamoDB must not be called when flag is off"

    def test_graph_completes_with_success_true(self):
        result = build_doubt_solver_graph().invoke(_make_graph_input("What is 30% of 120?"))
        assert result["response"]["success"] is True

    def test_needs_review_false_for_matched_keyword(self):
        result = build_doubt_solver_graph().invoke(_make_graph_input("Solve: 2x = 8"))
        # "solve" keyword → confidence=0.75 ≥ 0.6 → needs_review=False
        assert result["response"]["needs_review"] is False

    def test_low_confidence_query_gives_needs_review_true(self):
        result = build_doubt_solver_graph().invoke(
            _make_graph_input("zzz totally unrecognised query zzz")
        )
        assert result["response"]["needs_review"] is True


# ---------------------------------------------------------------------------
# KB enabled with fake service
# ---------------------------------------------------------------------------


class TestPipelineWithFakeKB:
    """Full pipeline integration with fake KB — no real AWS calls."""

    @pytest.fixture(autouse=True)
    def _kb_on(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "fake-kb-id-integration")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

    def test_kb_results_set_used_retrieval_true(self, monkeypatch):
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415

        monkeypatch.setattr(
            graph_module,
            "retrieve_similar_context",
            _fake_kb_with_content("Percentage means per hundred."),
        )
        result = build_doubt_solver_graph().invoke(_make_graph_input("Explain percentage"))
        assert result["response"]["used_retrieval"] is True
        assert result["response"]["source_count"] >= 1
        assert result["response"]["context_used"] is True

    def test_kb_results_context_passed_to_answer_generator(self, monkeypatch):
        """Context assembled from KB must actually reach the answer generator."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415
        from schemas.doubt_solver import AnswerOutput  # noqa: PLC0415

        monkeypatch.setattr(
            graph_module, "retrieve_similar_context", _fake_kb_with_content("Ratio facts.")
        )

        captured_context: list[str | None] = []

        def _spy_generate(query, classification, context=None):
            captured_context.append(context)
            return AnswerOutput(content="Answer.", answer_source="mock", is_truncated=False)

        monkeypatch.setattr(graph_module, "generate_answer", _spy_generate)

        build_doubt_solver_graph().invoke(_make_graph_input("Explain ratio"))

        assert len(captured_context) == 1
        assert captured_context[0] is not None
        # Safety header must be present.
        assert "reference material" in captured_context[0].lower()

    def test_kb_empty_results_graph_still_answers(self, monkeypatch):
        """KB returns no results → graph answers safely, context_used=False."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415

        monkeypatch.setattr(
            graph_module, "retrieve_similar_context", _fake_empty_kb()
        )
        result = build_doubt_solver_graph().invoke(_make_graph_input("Explain ratio"))
        assert result["response"]["success"] is True
        assert result["response"]["used_retrieval"] is False
        assert result["response"]["context_used"] is False
        assert len(result["response"]["answer"]) > 0

    def test_kb_service_error_graph_answers_with_needs_review(self, monkeypatch):
        """KnowledgeBaseServiceError → answer present, needs_review=True."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415
        from services.bedrock_kb_service import KnowledgeBaseServiceError  # noqa: PLC0415

        def _failing_kb(query, max_results=None):
            raise KnowledgeBaseServiceError("Integration test simulated failure")

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _failing_kb)
        result = build_doubt_solver_graph().invoke(_make_graph_input("Explain ratio"))
        assert result["response"]["success"] is True
        assert result["response"]["needs_review"] is True
        assert len(result["response"]["answer"]) > 0

    def test_kb_content_not_in_response(self, monkeypatch):
        """Full KB content text must NOT appear verbatim in the response."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415

        secret_marker = "UNIQUE_KB_CONTENT_MARKER_9g7h2j"
        monkeypatch.setattr(
            graph_module,
            "retrieve_similar_context",
            _fake_kb_with_content(f"KB article: {secret_marker}"),
        )
        result = build_doubt_solver_graph().invoke(_make_graph_input("Explain percentage"))
        response_str = str(result["response"])
        assert secret_marker not in response_str

    def test_all_response_fields_present_with_kb(self, monkeypatch):
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415

        monkeypatch.setattr(
            graph_module, "retrieve_similar_context", _fake_kb_with_content("Some context.")
        )
        response = build_doubt_solver_graph().invoke(
            _make_graph_input("Explain percentage")
        )["response"]

        for field in _REQUIRED_RESPONSE_FIELDS:
            assert field in response, f"Missing response field: {field}"


# ---------------------------------------------------------------------------
# KB + DynamoDB both enabled with fake services
# ---------------------------------------------------------------------------


class TestPipelineWithFakeKBAndDynamoDB:
    """Full pipeline with both KB and DynamoDB replaced by fake services."""

    @pytest.fixture(autouse=True)
    def _both_on(self, monkeypatch):
        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "fake-kb-id-integration")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "true")
        monkeypatch.setenv("DYNAMODB_QUESTION_TABLE", "fake-questions-table")
        monkeypatch.setenv("DYNAMODB_PATTERN_TABLE", "fake-patterns-table")
        _reset_settings()

    def _fake_kb_with_record_ids(self, record_ids: list[str]):
        from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse  # noqa: PLC0415

        def _fake(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[
                    KnowledgeBaseResult(
                        content="Algebra reference.", score=0.9, record_ids=record_ids
                    )
                ],
                result_count=1,
                retrieval_source="bedrock_kb",
            )

        return _fake

    def test_kb_record_ids_trigger_dynamodb_fetch(self, monkeypatch):
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415

        fetched_ids: list[str] = []

        def _fake_dynamo(ids):
            fetched_ids.extend(ids)
            return [{"question_id": "q-100", "text": "What is algebra?"}]

        monkeypatch.setattr(
            graph_module,
            "retrieve_similar_context",
            self._fake_kb_with_record_ids(["q-100"]),
        )
        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _fake_dynamo)

        result = build_doubt_solver_graph().invoke(_make_graph_input("Explain algebra"))

        assert "q-100" in fetched_ids
        assert result["response"]["source_count"] >= 1

    def test_dynamodb_error_graph_still_answers(self, monkeypatch):
        """DynamoDbServiceError → graph answers safely, needs_review=True."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415
        from services.dynamodb_service import DynamoDbServiceError  # noqa: PLC0415

        def _failing_dynamo(ids):
            raise DynamoDbServiceError("Integration test DB failure")

        monkeypatch.setattr(
            graph_module,
            "retrieve_similar_context",
            self._fake_kb_with_record_ids(["q-1"]),
        )
        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _failing_dynamo)

        result = build_doubt_solver_graph().invoke(_make_graph_input("Explain algebra"))
        assert result["response"]["success"] is True
        assert result["response"]["needs_review"] is True

    def test_dynamodb_record_text_not_in_response(self, monkeypatch):
        """DynamoDB record text must NOT appear verbatim in the response."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415

        secret_marker = "DYNAMO_CONTENT_NOT_FOR_RESPONSE_4k8m3n"

        def _fake_dynamo(ids):
            return [{"question_id": "q-200", "text": secret_marker}]

        monkeypatch.setattr(
            graph_module,
            "retrieve_similar_context",
            self._fake_kb_with_record_ids(["q-200"]),
        )
        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _fake_dynamo)

        result = build_doubt_solver_graph().invoke(_make_graph_input("Explain algebra"))
        response_str = str(result["response"])
        assert secret_marker not in response_str

    def test_dynamodb_disabled_no_call_even_with_record_ids(self, monkeypatch):
        """When DynamoDB flag is off, no fetch occurs even if KB has record_ids."""
        import graphs.doubt_solver_graph as graph_module  # noqa: PLC0415

        # Override the autouse fixture to turn DynamoDB off.
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

        dynamo_called: list = []

        def _spy_dynamo(ids):
            dynamo_called.append(ids)
            return []

        monkeypatch.setattr(
            graph_module,
            "retrieve_similar_context",
            self._fake_kb_with_record_ids(["q-1", "q-2"]),
        )
        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _spy_dynamo)

        build_doubt_solver_graph().invoke(_make_graph_input("Explain algebra"))
        assert dynamo_called == []


# ---------------------------------------------------------------------------
# main.py invoke() routing integration
# ---------------------------------------------------------------------------


class TestMainInvokeIntegration:
    """Tests for main.py invoke() routing and complete response shape."""

    def test_doubt_solver_returns_all_required_fields(self):
        import main  # noqa: PLC0415

        result = main.invoke(
            {
                "mode": "doubt_solver",
                "query": "Explain what ratio means",
                "user_id": "test-user",
                "language": "en",
            }
        )
        for field in _REQUIRED_RESPONSE_FIELDS:
            assert field in result, f"main.invoke() missing field: {field}"

    def test_doubt_solver_response_valid_pydantic(self):
        import main  # noqa: PLC0415

        result = main.invoke(
            {
                "mode": "doubt_solver",
                "query": "Solve: 3x = 9",
                "user_id": "test-user",
            }
        )
        assert result["success"] is True
        model = DoubtSolverResponse.model_validate(result)
        assert model.success is True

    def test_doubt_solver_default_flags_used_retrieval_false(self):
        """With default flags, used_retrieval must be False."""
        import main  # noqa: PLC0415

        result = main.invoke(
            {"mode": "doubt_solver", "query": "Explain percentage", "user_id": "u1"}
        )
        assert result["used_retrieval"] is False
        assert result["context_used"] is False
        assert result["source_count"] == 0

    def test_doubt_solver_missing_query_validation_error(self):
        import main  # noqa: PLC0415

        result = main.invoke({"mode": "doubt_solver", "user_id": "test-user"})
        assert result["success"] is False
        assert "Validation error" in result["answer"]

    def test_demo_mode_unaffected_by_part9_changes(self):
        """Non-doubt_solver mode must still work identically."""
        import main  # noqa: PLC0415

        result = main.invoke({"mode": "demo", "message": "Hello", "user_id": "u1"})
        assert result["success"] is True
