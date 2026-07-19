"""
app/tests/test_doubt_solver_graph.py
--------------------------------------
Tests for the Doubt Solver V1 graph, classifier service, and answer generator.

No AWS credentials, network, or LLM calls required.
All services are deterministic stubs — no mocking needed.
"""

from __future__ import annotations

import pytest

import config as cfg_module
from graphs.doubt_solver_graph import build_doubt_solver_graph
from schemas.doubt_solver import QueryClassification
from services.answer_generator_service import generate_answer
from services.query_classifier_service import classify_query

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_settings():
    cfg_module._settings = None


@pytest.fixture(autouse=True)
def _use_legacy_kb_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Part 9 graph tests exercise the explicitly retained legacy KB provider."""
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "legacy_bedrock_kb")
    _reset_settings()
    yield
    _reset_settings()


def _make_state(query: str = "What is 20% of 500?", **overrides) -> dict:
    base = {
        "request_id": "test-req-id",
        "query": query,
        "user_id": "test-user",
        "mode": "doubt_solver",
        "language": "en",
        "classification": None,
        "answer": None,
        "answer_source": None,
        "is_truncated": False,
        "response": None,
        # Part 9 fields — safe defaults keep pre-Part-9 behavior unchanged.
        "should_retrieve": False,
        "kb_results": None,
        "dynamodb_records": None,
        "answer_context": None,
        "context_source_count": 0,
        "used_retrieval": False,
        "context_used": False,
        "service_error": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Classifier service tests
# ---------------------------------------------------------------------------


class TestQueryClassifierService:
    def test_solve_intent_via_solve_keyword(self):
        result = classify_query("Solve this equation: 2x = 10")
        assert result.intent == "solve_question"
        assert result.confidence == 0.75

    def test_solve_intent_via_calculate_keyword(self):
        result = classify_query("Calculate the percentage gain")
        assert result.intent == "solve_question"

    def test_solve_intent_via_find_keyword(self):
        result = classify_query("Find the value of x")
        assert result.intent == "solve_question"

    def test_solve_intent_via_answer_keyword(self):
        result = classify_query("What is the answer to this problem?")
        assert result.intent == "solve_question"

    def test_explain_intent_via_explain_keyword(self):
        result = classify_query("Explain what ratio means")
        assert result.intent == "explain_concept"
        assert result.confidence == 0.75

    def test_explain_intent_via_concept_keyword(self):
        result = classify_query("I need to understand the concept of profit")
        assert result.intent == "explain_concept"

    def test_explain_intent_via_why_keyword(self):
        result = classify_query("Why is this formula used?")
        assert result.intent == "explain_concept"

    def test_explain_option_via_option_keyword(self):
        result = classify_query("I need to select an option here")
        assert result.intent == "explain_option"

    def test_explain_option_via_choice_keyword(self):
        result = classify_query("Which choice is correct here?")
        assert result.intent == "explain_option"

    def test_explain_option_via_correct_keyword(self):
        result = classify_query("This one is correct")
        assert result.intent == "explain_option"

    def test_general_doubt_fallback(self):
        result = classify_query("asdfghjkl random gibberish xyz")
        assert result.intent == "general_doubt"
        assert result.confidence == 0.55

    def test_math_subject_via_percentage(self):
        result = classify_query("calculate the percentage of 40 out of 200")
        assert result.subject == "math"

    def test_math_subject_via_profit(self):
        result = classify_query("What is the profit margin here?")
        assert result.subject == "math"

    def test_math_subject_via_loss(self):
        result = classify_query("How do I calculate loss?")
        assert result.subject == "math"

    def test_response_style_keyword_short(self):
        result = classify_query("Give a short answer about ratio")
        assert result.response_style == "short_answer"

    def test_response_style_keyword_simple(self):
        result = classify_query("Explain in simple terms what ratio is")
        assert result.response_style == "simple_explanation"

    def test_returns_query_classification_instance(self):
        result = classify_query("What is 5 + 5?")
        assert isinstance(result, QueryClassification)

    def test_confidence_in_valid_range(self):
        result = classify_query("Solve this: x + 2 = 5")
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Answer generator service tests
# ---------------------------------------------------------------------------


class TestAnswerGeneratorService:
    def _classification(self, intent: str, subject: str = "math") -> QueryClassification:
        return QueryClassification(intent=intent, subject=subject, confidence=0.75)  # type: ignore[arg-type]

    def test_answer_output_is_answer_output_instance(self):
        from schemas.doubt_solver import AnswerOutput

        c = self._classification("solve_question")
        result = generate_answer("What is 10 + 10?", c)
        assert isinstance(result, AnswerOutput)

    def test_answer_content_is_non_empty(self):
        c = self._classification("solve_question")
        result = generate_answer("What is 10 + 10?", c)
        assert len(result.content) > 0

    def test_solve_answer_contains_query(self):
        query = "What is 10 + 10?"
        c = self._classification("solve_question")
        result = generate_answer(query, c)
        assert query in result.content

    def test_explain_answer_contains_query(self):
        query = "Explain what ratio means"
        c = self._classification("explain_concept", subject="math")
        result = generate_answer(query, c)
        assert query in result.content

    def test_explain_option_answer_contains_query(self):
        query = "Why is option B correct?"
        c = self._classification("explain_option")
        result = generate_answer(query, c)
        assert query in result.content

    def test_unknown_intent_returns_fallback(self):
        c = self._classification("unknown", subject="unknown")
        result = generate_answer("gibberish text", c)
        assert "rephrase" in result.content.lower()

    def test_general_doubt_returns_non_empty(self):
        c = self._classification("general_doubt")
        result = generate_answer("I have a doubt", c)
        assert len(result.content) > 0


# ---------------------------------------------------------------------------
# Graph tests
# ---------------------------------------------------------------------------


class TestDoubtSolverGraph:
    @pytest.fixture(scope="class")
    def ds_graph(self):
        return build_doubt_solver_graph()

    def test_graph_returns_success_true(self, ds_graph):
        result = ds_graph.invoke(_make_state("What is 20% of 500?"))
        assert result["response"]["success"] is True

    def test_graph_returns_answer(self, ds_graph):
        result = ds_graph.invoke(_make_state("What is 20% of 500?"))
        assert result["response"]["answer"]
        assert len(result["response"]["answer"]) > 0

    def test_graph_returns_classification(self, ds_graph):
        result = ds_graph.invoke(_make_state("Explain what percentage means"))
        classification = result["response"]["classification"]
        assert classification is not None
        assert "intent" in classification
        assert "confidence" in classification

    def test_answer_contains_original_query(self, ds_graph):
        query = "Solve this equation: 3x = 9"
        result = ds_graph.invoke(_make_state(query))
        assert query in result["response"]["answer"]

    def test_solve_query_classified_as_solve_question(self, ds_graph):
        result = ds_graph.invoke(_make_state("Solve for x: 2x = 8"))
        assert result["response"]["classification"]["intent"] == "solve_question"

    def test_explain_query_classified_as_explain_concept(self, ds_graph):
        result = ds_graph.invoke(_make_state("Explain the concept of percentage"))
        assert result["response"]["classification"]["intent"] == "explain_concept"

    def test_needs_review_false_for_high_confidence(self, ds_graph):
        # Matched keyword → confidence 0.75, which is ≥ 0.6 → needs_review = False
        result = ds_graph.invoke(_make_state("Solve this: x + 1 = 5"))
        assert result["response"]["needs_review"] is False

    def test_needs_review_true_for_low_confidence(self, ds_graph):
        # general_doubt → confidence 0.55, which is < 0.6 → needs_review = True
        result = ds_graph.invoke(_make_state("zzz totally unrecognised input zzz"))
        assert result["response"]["needs_review"] is True

    def test_unknown_query_returns_answer_not_crash(self, ds_graph):
        result = ds_graph.invoke(_make_state("asdkjhaskdjhaskdj completely unknown"))
        assert result["response"]["answer"] is not None
        assert result["response"]["success"] is True

    def test_graph_preserves_request_id(self, ds_graph):
        result = ds_graph.invoke(_make_state("Explain the concept of ratio"))
        assert result["response"]["request_id"] == "test-req-id"

    def test_response_mode_is_doubt_solver(self, ds_graph):
        result = ds_graph.invoke(_make_state("What is ratio?"))
        assert result["response"]["mode"] == "doubt_solver"


# ---------------------------------------------------------------------------
# Smoke test — graph builds without error
# ---------------------------------------------------------------------------


class TestModeRouting:
    def test_doubt_solver_graph_builds_without_error(self):
        g = build_doubt_solver_graph()
        assert g is not None

    def test_doubt_solver_graph_invocable(self):
        g = build_doubt_solver_graph()
        result = g.invoke(_make_state("What is 10% of 200?"))
        assert "response" in result
        assert result["response"]["success"] is True


# ---------------------------------------------------------------------------
# Part 4: answer_source, is_truncated, extended needs_review
# ---------------------------------------------------------------------------


class TestAnswerSourceAndTruncation:
    @pytest.fixture(scope="class")
    def ds_graph(self):
        return build_doubt_solver_graph()

    def test_response_includes_answer_source(self, ds_graph):
        result = ds_graph.invoke(_make_state("Solve for x: 2x = 8"))
        assert result["response"]["answer_source"] in ("mock", "llm", "fallback")

    def test_response_answer_source_is_mock_by_default(self, ds_graph):
        """Default ENABLE_REAL_LLM=false -> mock path -> answer_source='mock'."""
        result = ds_graph.invoke(_make_state("What is 20% of 500?"))
        assert result["response"]["answer_source"] == "mock"

    def test_response_includes_is_truncated(self, ds_graph):
        result = ds_graph.invoke(_make_state("What is 20% of 500?"))
        assert result["response"]["is_truncated"] is False

    def test_needs_review_false_when_high_confidence_mock(self, ds_graph):
        # confidence=0.75 >= 0.6, source=mock, not truncated -> needs_review=False
        result = ds_graph.invoke(_make_state("Solve this: x + 1 = 5"))
        assert result["response"]["needs_review"] is False

    def test_needs_review_true_when_fallback_source(self, monkeypatch):
        """answer_source=fallback forces needs_review=True regardless of confidence."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.doubt_solver import AnswerOutput

        ds_graph = build_doubt_solver_graph()

        def _fake_generate(query, classification, context=None):
            return AnswerOutput(
                content="Fallback answer content.",
                answer_source="fallback",
                is_truncated=False,
            )

        monkeypatch.setattr(graph_module, "generate_answer", _fake_generate)

        result = ds_graph.invoke(_make_state("Solve x: x + 1 = 5"))
        assert result["response"]["needs_review"] is True

    def test_needs_review_true_when_is_truncated(self, monkeypatch):
        """is_truncated=True forces needs_review=True regardless of confidence."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.doubt_solver import AnswerOutput

        ds_graph = build_doubt_solver_graph()

        def _fake_generate(query, classification, context=None):
            return AnswerOutput(
                content="A" * 100,  # short but is_truncated=True
                answer_source="llm",
                is_truncated=True,
            )

        monkeypatch.setattr(graph_module, "generate_answer", _fake_generate)

        result = ds_graph.invoke(_make_state("Explain ratio"))
        assert result["response"]["needs_review"] is True

    def test_needs_review_true_for_low_confidence_mock(self, ds_graph):
        # general_doubt -> confidence=0.55 < 0.6 -> needs_review=True
        result = ds_graph.invoke(_make_state("zzz totally unrecognised input zzz"))
        assert result["response"]["needs_review"] is True

    def test_answer_source_propagated_to_response(self, monkeypatch):
        """answer_source set by generate_answer_node reaches the response dict."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.doubt_solver import AnswerOutput

        ds_graph = build_doubt_solver_graph()

        def _fake_generate(query, classification, context=None):
            return AnswerOutput(
                content="LLM answer here.",
                answer_source="llm",
                is_truncated=False,
            )

        monkeypatch.setattr(graph_module, "generate_answer", _fake_generate)

        result = ds_graph.invoke(_make_state("What is ratio?"))
        assert result["response"]["answer_source"] == "llm"

    def test_is_truncated_propagated_to_response(self, monkeypatch):
        import graphs.doubt_solver_graph as graph_module
        from schemas.doubt_solver import AnswerOutput

        ds_graph = build_doubt_solver_graph()

        def _fake_generate(query, classification, context=None):
            return AnswerOutput(
                content="Truncated answer.",
                answer_source="llm",
                is_truncated=True,
            )

        monkeypatch.setattr(graph_module, "generate_answer", _fake_generate)

        result = ds_graph.invoke(_make_state("Explain percentage"))
        assert result["response"]["is_truncated"] is True


# ---------------------------------------------------------------------------
# Part 9: new response fields present with default values
# ---------------------------------------------------------------------------


class TestPart9ResponseFields:
    """Verify the 3 new DoubtSolverResponse fields are present with safe defaults."""

    @pytest.fixture(scope="class")
    def ds_graph(self):
        return build_doubt_solver_graph()

    def test_used_retrieval_present_in_response(self, ds_graph):
        result = ds_graph.invoke(_make_state("What is 20% of 500?"))
        assert "used_retrieval" in result["response"]

    def test_used_retrieval_default_false_when_kb_disabled(self, ds_graph):
        # ENABLE_KB_RETRIEVAL defaults to false → no retrieval → used_retrieval=False
        result = ds_graph.invoke(_make_state("Explain ratio"))
        assert result["response"]["used_retrieval"] is False

    def test_source_count_present_in_response(self, ds_graph):
        result = ds_graph.invoke(_make_state("What is 20% of 500?"))
        assert "source_count" in result["response"]

    def test_source_count_zero_when_no_retrieval(self, ds_graph):
        result = ds_graph.invoke(_make_state("Solve: 2x = 10"))
        assert result["response"]["source_count"] == 0

    def test_context_used_present_in_response(self, ds_graph):
        result = ds_graph.invoke(_make_state("What is 20% of 500?"))
        assert "context_used" in result["response"]

    def test_context_used_false_when_no_retrieval(self, ds_graph):
        result = ds_graph.invoke(_make_state("Explain algebra"))
        assert result["response"]["context_used"] is False


# ---------------------------------------------------------------------------
# Part 9: KB retrieval integration
# ---------------------------------------------------------------------------


class TestPart9KbRetrieval:
    """Tests for the KB retrieval nodes — all using fake services."""

    def _setup_kb_enabled(self, monkeypatch, table_not_needed: bool = True):
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "fake-kb-id")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

    def test_kb_disabled_keeps_previous_behavior(self, monkeypatch):
        """Default flags off → graph completes identically to pre-Part-9."""
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

        result = build_doubt_solver_graph().invoke(_make_state("Solve: x + 1 = 5"))

        assert result["response"]["success"] is True
        assert result["response"]["used_retrieval"] is False
        assert result["response"]["source_count"] == 0
        assert result["response"]["context_used"] is False

    def test_kb_enabled_with_fake_retrieval_adds_context(self, monkeypatch):
        """When KB returns results, context is built and passed to answer generator."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse

        self._setup_kb_enabled(monkeypatch)

        fake_result = KnowledgeBaseResult(
            content="Algebra is a branch of mathematics dealing with equations.",
            score=0.95,
            source_id="doc-1",
        )

        def _fake_retrieve(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[fake_result],
                result_count=1,
                retrieval_source="bedrock_kb",
            )

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _fake_retrieve)

        captured_context: list = []

        def _spy_generate(query, classification, context=None):
            captured_context.append(context)
            from schemas.doubt_solver import AnswerOutput

            return AnswerOutput(
                content="Answer with context.",
                answer_source="mock",
                is_truncated=False,
            )

        monkeypatch.setattr(graph_module, "generate_answer", _spy_generate)

        result = build_doubt_solver_graph().invoke(_make_state("Explain algebra"))

        assert result["response"]["used_retrieval"] is True
        assert result["response"]["source_count"] >= 1
        assert result["response"]["context_used"] is True
        # Context was passed to answer generator.
        assert len(captured_context) == 1
        assert captured_context[0] is not None
        assert "Algebra" in captured_context[0]

    def test_kb_service_error_does_not_crash_graph(self, monkeypatch):
        """KnowledgeBaseServiceError → needs_review=True, graph completes."""
        import graphs.doubt_solver_graph as graph_module
        from services.bedrock_kb_service import KnowledgeBaseServiceError

        self._setup_kb_enabled(monkeypatch)

        def _failing_retrieve(query, max_results=None):
            raise KnowledgeBaseServiceError("Simulated KB failure")

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _failing_retrieve)

        result = build_doubt_solver_graph().invoke(_make_state("Explain ratio"))

        assert result["response"]["success"] is True
        assert result["response"]["needs_review"] is True
        assert result["response"]["used_retrieval"] is False

    def test_kb_config_error_does_not_crash_graph(self, monkeypatch):
        """KnowledgeBaseConfigurationError → graph completes with needs_review=True."""
        import graphs.doubt_solver_graph as graph_module
        from services.bedrock_kb_service import KnowledgeBaseConfigurationError

        self._setup_kb_enabled(monkeypatch)

        def _config_error_retrieve(query, max_results=None):
            raise KnowledgeBaseConfigurationError("Missing KB ID")

        monkeypatch.setattr(
            graph_module, "retrieve_similar_context", _config_error_retrieve
        )

        result = build_doubt_solver_graph().invoke(_make_state("Explain algebra"))

        assert result["response"]["success"] is True
        assert result["response"]["needs_review"] is True

    def test_kb_disabled_flag_results_in_no_retrieval(self, monkeypatch):
        """ENABLE_KB_RETRIEVAL=false → retrieve_similar_context returns disabled."""
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        _reset_settings()

        result = build_doubt_solver_graph().invoke(_make_state("Solve: 3x = 12"))

        assert result["response"]["used_retrieval"] is False
        assert result["response"]["source_count"] == 0

    def test_kb_results_with_no_content_gives_context_used_false(self, monkeypatch):
        """KB returns empty results list → no context built → context_used=False."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.retrieval import RetrievalResponse

        self._setup_kb_enabled(monkeypatch)

        def _empty_retrieve(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[],
                result_count=0,
                retrieval_source="bedrock_kb",
            )

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _empty_retrieve)

        result = build_doubt_solver_graph().invoke(_make_state("Explain osmosis"))

        assert result["response"]["used_retrieval"] is False
        assert result["response"]["context_used"] is False


# ---------------------------------------------------------------------------
# Part 9: DynamoDB fetch integration
# ---------------------------------------------------------------------------


class TestPart9DynamoDbFetch:
    """Tests for the DynamoDB fetch node — all using fake services."""

    def _setup_kb_and_dynamo_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "fake-kb-id")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "true")
        monkeypatch.setenv("DYNAMODB_QUESTION_TABLE", "questions-table")
        monkeypatch.setenv("DYNAMODB_PATTERN_TABLE", "patterns-table")
        _reset_settings()

    def test_dynamodb_disabled_no_op_even_with_record_ids(self, monkeypatch):
        """ENABLE_DYNAMODB_FETCH=false → DynamoDB never called even if KB has record_ids."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse

        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "fake-kb-id")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

        fake_result = KnowledgeBaseResult(
            content="Some content.",
            score=0.9,
            record_ids=["q-1", "q-2"],
        )

        def _fake_retrieve(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[fake_result],
                result_count=1,
                retrieval_source="bedrock_kb",
            )

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _fake_retrieve)

        # Patch DynamoDB service to verify it's never called.
        dynamo_called: list = []

        def _spy_fetch(ids):
            dynamo_called.append(ids)
            return []

        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _spy_fetch)

        result = build_doubt_solver_graph().invoke(_make_state("Explain algebra"))

        assert dynamo_called == [], "DynamoDB should not be called when ENABLE_DYNAMODB_FETCH=false"
        assert result["response"]["success"] is True

    def test_dynamodb_enabled_and_kb_has_record_ids_triggers_fetch(self, monkeypatch):
        """When KB returns record_ids and DynamoDB is enabled, fetch is called."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse

        self._setup_kb_and_dynamo_enabled(monkeypatch)

        fake_kb_result = KnowledgeBaseResult(
            content="Algebra content.",
            score=0.9,
            record_ids=["q-10"],
        )

        def _fake_retrieve(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[fake_kb_result],
                result_count=1,
                retrieval_source="bedrock_kb",
            )

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _fake_retrieve)

        fetched_ids: list = []

        def _fake_fetch(ids):
            fetched_ids.extend(ids)
            return [{"question_id": "q-10", "text": "What is algebra?"}]

        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _fake_fetch)

        result = build_doubt_solver_graph().invoke(_make_state("Explain algebra"))

        assert "q-10" in fetched_ids
        assert result["response"]["source_count"] >= 1

    def test_dynamodb_service_error_does_not_crash_graph(self, monkeypatch):
        """DynamoDbServiceError → graph completes with needs_review=True."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse
        from services.dynamodb_service import DynamoDbServiceError

        self._setup_kb_and_dynamo_enabled(monkeypatch)

        fake_kb_result = KnowledgeBaseResult(
            content="Some KB content.",
            score=0.9,
            record_ids=["q-5"],
        )

        def _fake_retrieve(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[fake_kb_result],
                result_count=1,
                retrieval_source="bedrock_kb",
            )

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _fake_retrieve)

        def _failing_fetch(ids):
            raise DynamoDbServiceError("Simulated DynamoDB failure")

        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _failing_fetch)

        result = build_doubt_solver_graph().invoke(_make_state("Explain algebra"))

        assert result["response"]["success"] is True
        assert result["response"]["needs_review"] is True

    def test_dynamodb_no_record_ids_in_kb_results_skips_fetch(self, monkeypatch):
        """KB results with empty record_ids → DynamoDB not called."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse

        self._setup_kb_and_dynamo_enabled(monkeypatch)

        fake_kb_result = KnowledgeBaseResult(
            content="Generic content with no linked records.",
            score=0.8,
            record_ids=[],  # empty
        )

        def _fake_retrieve(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[fake_kb_result],
                result_count=1,
                retrieval_source="bedrock_kb",
            )

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _fake_retrieve)

        dynamo_called: list = []

        def _spy_fetch(ids):
            dynamo_called.append(ids)
            return []

        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _spy_fetch)

        build_doubt_solver_graph().invoke(_make_state("Explain concept"))

        assert dynamo_called == []

    def test_dynamodb_config_error_does_not_crash_graph(self, monkeypatch):
        """DynamoDbConfigurationError → graph completes, needs_review=True."""
        import graphs.doubt_solver_graph as graph_module
        from schemas.retrieval import KnowledgeBaseResult, RetrievalResponse
        from services.dynamodb_service import DynamoDbConfigurationError

        self._setup_kb_and_dynamo_enabled(monkeypatch)

        def _fake_retrieve(query, max_results=None):
            return RetrievalResponse(
                query=query,
                results=[KnowledgeBaseResult(content="c.", record_ids=["q-1"])],
                result_count=1,
                retrieval_source="bedrock_kb",
            )

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _fake_retrieve)

        def _config_err_fetch(ids):
            raise DynamoDbConfigurationError("Missing table name")

        monkeypatch.setattr(graph_module, "fetch_question_records_by_ids", _config_err_fetch)

        result = build_doubt_solver_graph().invoke(_make_state("Explain algebra"))

        assert result["response"]["success"] is True
        assert result["response"]["needs_review"] is True


# ---------------------------------------------------------------------------
# Part 9: needs_review with service_error
# ---------------------------------------------------------------------------


class TestPart9NeedsReviewWithServiceError:
    def test_service_error_sets_needs_review_true(self, monkeypatch):
        """service_error=True in state propagates to needs_review=True in response."""
        # Build a state where service_error is already True (simulate mid-graph).
        # We test through build_response_node indirectly by running the graph
        # with a patched KB service that fails.
        import graphs.doubt_solver_graph as graph_module
        from services.bedrock_kb_service import KnowledgeBaseServiceError

        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "fake-kb-id")
        _reset_settings()

        def _failing_retrieve(query, max_results=None):
            raise KnowledgeBaseServiceError("Failure")

        monkeypatch.setattr(graph_module, "retrieve_similar_context", _failing_retrieve)

        result = build_doubt_solver_graph().invoke(_make_state("Explain osmosis"))

        assert result["response"]["needs_review"] is True

    def test_no_service_error_high_confidence_mock_needs_review_false(self, monkeypatch):
        """Baseline: no errors + high confidence + mock → needs_review=False."""
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        monkeypatch.setenv("ENABLE_DYNAMODB_FETCH", "false")
        _reset_settings()

        result = build_doubt_solver_graph().invoke(_make_state("Solve: x + 1 = 5"))

        # "Solve" keyword → confidence=0.75 (≥ 0.6), source=mock, not truncated, no error.
        assert result["response"]["needs_review"] is False
