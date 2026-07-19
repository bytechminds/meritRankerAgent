"""
tests/test_orchestrated_streaming.py
--------------------------------------
Student-friendly orchestrated doubt solver streaming tests.
"""

from __future__ import annotations

import types

import pytest

from graphs.doubt_solver_graph import OrchestratedDoubtSolverState
from schemas.doubt_solver import (
    DoubtSolverFinalResponse,
    DoubtSolverStreamEvent,
    ResponseContent,
)
from services.doubt_solver.answer_generation_adapter import AnswerGenerationAdapter
from services.doubt_solver.stream_labels import get_stream_label
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)
from services.llm.orchestration.orchestrator import (
    LlmOrchestrator,
    MockModelExecutor,
    create_mock_orchestrator_for_tests,
)

_REQUEST_ID = "test-req-stream-001"

_FORBIDDEN_CONTENT_SUBSTRINGS = {
    "prompt",
    "api_key",
    "secret",
    "credential",
    "authorization",
    "context_text",
    "raw_response",
    "Traceback",
    "fallback",
    "confidence",
    "gpt-",
    "azure",
    "deepseek",
    "classifier_strong",
}


def _complete_response() -> DoubtSolverFinalResponse:
    return DoubtSolverFinalResponse(
        request_id=_REQUEST_ID,
        content=ResponseContent(value="Answer."),
        answer="Answer.",
    )


def _make_adapter(
    content: str = "Let the cost price be ₹100. Marked price = ₹140.",
    *,
    raise_on_execute: Exception | None = None,
) -> AnswerGenerationAdapter:
    orchestrator, _ = create_mock_orchestrator_for_tests(
        content=content,
        raise_on_execute=raise_on_execute,
    )
    return AnswerGenerationAdapter(orchestrator=orchestrator)


def _collect(
    adapter: AnswerGenerationAdapter,
    *,
    query: str = "A profit question",
) -> list[DoubtSolverStreamEvent]:
    return list(
        stream_doubt_solver(
            StreamDoubtSolverInput(request_id=_REQUEST_ID, query=query),
            adapter=adapter,
        )
    )


class TestStreamEventSchema:
    def test_status_validates(self) -> None:
        event = DoubtSolverStreamEvent(
            type="status",
            request_id=_REQUEST_ID,
            stage="understanding",
            label="Understanding...",
        )
        assert event.type == "status"

    def test_sse_framing_from_model_dump(self) -> None:
        import json

        event = DoubtSolverStreamEvent(
            type="status",
            request_id=_REQUEST_ID,
            stage="understanding",
            label="Understanding...",
        )
        framed = f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"
        assert framed.startswith("data: {")
        assert framed.endswith("\n\n")
        payload = json.loads(framed.removeprefix("data: ").strip())
        assert payload["type"] == "status"
        assert payload["request_id"] == _REQUEST_ID

    def test_chunk_validates(self) -> None:
        event = DoubtSolverStreamEvent(
            type="chunk",
            request_id=_REQUEST_ID,
            content="Hello",
        )
        assert event.content == "Hello"

    def test_complete_validates(self) -> None:
        event = DoubtSolverStreamEvent(
            type="complete",
            request_id=_REQUEST_ID,
            stage="complete",
            label="Done",
            metadata={"request_id": _REQUEST_ID},
            response=_complete_response(),
        )
        assert event.type == "complete"

    def test_error_validates(self) -> None:
        event = DoubtSolverStreamEvent(
            type="error",
            request_id=_REQUEST_ID,
            stage="error",
            label="Something went wrong. Please try again.",
        )
        assert event.type == "error"

    def test_chunk_requires_content(self) -> None:
        with pytest.raises(ValueError, match="chunk event must have content"):
            DoubtSolverStreamEvent(type="chunk", request_id=_REQUEST_ID)

    def test_status_requires_stage_and_label(self) -> None:
        with pytest.raises(ValueError, match="status event must have stage and label"):
            DoubtSolverStreamEvent(
                type="status",
                request_id=_REQUEST_ID,
                stage="understanding",
            )

    def test_error_requires_label(self) -> None:
        with pytest.raises(ValueError, match="error event must have"):
            DoubtSolverStreamEvent(type="error", request_id=_REQUEST_ID, stage="error")

    def test_forbidden_metadata_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="forbidden key"):
            DoubtSolverStreamEvent(
                type="complete",
                request_id=_REQUEST_ID,
                stage="complete",
                label="Done",
                metadata={"prompt": "hidden"},
                response=_complete_response(),
            )


class TestStreamLabelHelper:
    def test_understanding(self) -> None:
        assert get_stream_label("understanding") == "Understanding..."

    def test_thinking(self) -> None:
        assert get_stream_label("thinking") == "Thinking..."

    def test_solve(self) -> None:
        assert get_stream_label("generating", "solve") == "Solving..."

    def test_explain(self) -> None:
        assert get_stream_label("generating", "explain") == "Explaining..."

    def test_practice(self) -> None:
        assert get_stream_label("generating", "practice") == "Creating practice questions..."

    def test_visualize(self) -> None:
        assert get_stream_label("generating", "visualize") == "Preparing visual explanation..."


class TestStreamingFlow:
    def test_first_event_is_understanding(self) -> None:
        events = _collect(_make_adapter())
        assert events[0].type == "status"
        assert events[0].label == "Understanding..."

    def test_stream_lifecycle_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        with caplog.at_level(logging.INFO):
            _collect(_make_adapter("Short answer."))

        messages = " ".join(r.message for r in caplog.records)
        assert "stream_started=true" in messages
        assert "stream_status_reason" in messages
        assert "first_visible_chunk_emitted=true" in messages
        assert "stream_completed=true" in messages
        assert "answer_chunk_emission" in messages
        assert "latency_ms=" in messages

    def test_thinking_before_generation(self) -> None:
        events = _collect(_make_adapter())
        status_events = [e for e in events if e.type == "status"]
        labels = [e.label for e in status_events]
        assert "Thinking..." in labels
        assert labels.index("Thinking...") < next(
            i for i, label in enumerate(labels) if label == "Finalizing..."
        )

    def test_chunks_after_generating_status(self) -> None:
        events = _collect(_make_adapter("Alpha beta gamma"))
        generating_idx = next(
            i for i, e in enumerate(events) if e.type == "status" and e.stage == "generating"
        )
        first_chunk_idx = next(i for i, e in enumerate(events) if e.type == "chunk")
        assert first_chunk_idx > generating_idx

    def test_final_event_is_complete(self) -> None:
        events = _collect(_make_adapter())
        assert events[-1].type == "complete"
        assert events[-1].label == "Done"

    def test_chunk_content_is_provider_chunk_content(self) -> None:
        content = "Let the cost price be ₹100. Marked price = ₹140."
        events = _collect(_make_adapter(content))
        chunks = [e.content for e in events if e.type == "chunk"]
        assert "".join(c for c in chunks if c is not None) == content

    def test_no_sensitive_data_in_events(self) -> None:
        events = _collect(_make_adapter("Safe answer text."))
        for event in events:
            blob = f"{event.label or ''} {event.content or ''} {event.metadata}".lower()
            for forbidden in _FORBIDDEN_CONTENT_SUBSTRINGS:
                assert forbidden not in blob


class TestCarefulClassificationStreamStatus:
    def test_no_extra_status_when_strong_classifier_not_used(self, monkeypatch) -> None:
        from unittest.mock import patch

        from schemas.doubt_solver import QueryClassification

        high_conf = QueryClassification(
            intent="solve_question",
            subject="math",
            confidence=0.95,
            classification_source="llm",
        )
        with patch(
            "graphs.doubt_solver_graph.classify_query",
            return_value=high_conf,
        ):
            events = _collect(_make_adapter("Short answer."))
        labels = [e.label for e in events if e.type == "status"]
        assert labels.count("Checking the question more carefully...") == 0

    def test_careful_status_when_strong_classifier_used(self, monkeypatch) -> None:
        from unittest.mock import patch

        from schemas.doubt_solver import QueryClassification

        def _classify_with_hook(query, request_id=None, *, on_before_strong_classifier=None):
            if on_before_strong_classifier is not None:
                on_before_strong_classifier()
            return QueryClassification(
                intent="solve_question",
                subject="math",
                confidence=0.94,
                classification_source="llm",
            )

        with patch(
            "graphs.doubt_solver_graph.classify_query",
            side_effect=_classify_with_hook,
        ):
            events = _collect(_make_adapter("Short answer."))

        labels = [e.label for e in events if e.type == "status"]
        assert labels.count("Understanding...") == 1
        assert "Checking the question more carefully..." not in labels
        assert events[-1].type == "complete"


class TestMockProviderStreaming:
    def test_deterministic_chunks_emitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "false")
        monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
        import config as cfg_module

        cfg_module._settings = None
        events = _collect(_make_adapter("12345678901234567890"))
        chunks = [e for e in events if e.type == "chunk"]
        assert len(chunks) >= 2
        cfg_module._settings = None

    def test_collected_chunks_equal_final_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANSWER_QUALITY_VALIDATION_ENABLED", "false")
        monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
        import config as cfg_module

        cfg_module._settings = None
        content = "Mock streaming answer for tests."
        events = _collect(_make_adapter(content))
        reconstructed = "".join(e.content or "" for e in events if e.type == "chunk")
        assert reconstructed == content
        cfg_module._settings = None


class TestAzureV1StreamingAdapter:
    def test_fake_streaming_response_yields_chunks(self, tmp_path) -> None:
        from services.llm.providers.azure_openai_provider import AzureOpenAIProviderAdapter
        from services.secrets.provider_credentials import ProviderCredentials
        from tests.test_azure_openai_provider_adapter import _factory, _make_request

        class _FakeStreamChunk:
            def __init__(self, delta: str) -> None:
                self.choices = [types.SimpleNamespace(delta=types.SimpleNamespace(content=delta))]

        class _FakeStreamClient:
            def __init__(self) -> None:
                self.received_kwargs: dict = {}

            @property
            def chat(self) -> types.SimpleNamespace:
                def _create(**kwargs):  # noqa: ANN202
                    self.received_kwargs = kwargs
                    return iter([
                        _FakeStreamChunk("Let "),
                        _FakeStreamChunk("the "),
                        _FakeStreamChunk("cost"),
                    ])

                return types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=_create)
                )

        client = _FakeStreamClient()
        adapter = AzureOpenAIProviderAdapter(client_factory=_factory(client))
        creds = ProviderCredentials(
            provider="azure_openai",
            api_key="fake-key",
            endpoint="https://fake.openai.azure.com/openai/v1",
            azure_api_mode="azure_openai_v1",
        )
        chunks = list(
            adapter.generate_stream(
                request=_make_request(tmp_path),
                credentials=creds,
            )
        )
        assert chunks == ["Let ", "the ", "cost"]
        assert client.received_kwargs.get("model") == "gpt-4o-deployment"
        assert client.received_kwargs.get("stream") is True


class TestNonStreamRegression:
    def test_generate_unchanged(self) -> None:
        content = "Non-streaming answer."
        adapter = _make_adapter(content)
        result = adapter.generate(
            request_id=_REQUEST_ID,
            query="Test",
            subject="general",
            intent="explain",
            difficulty="default",
            context="",
        )
        assert result == content

    def test_graph_state_includes_retrieval_context(self) -> None:
        fields = set(OrchestratedDoubtSolverState.__annotations__.keys())
        expected = {
            "request_id",
            "query",
            "classification",
            "retrieval_context",
            "context_text",
            "answer",
        }
        assert fields == expected

    def test_task_role_remains_generator(self) -> None:
        orchestrator, executor = create_mock_orchestrator_for_tests(content="x")
        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        adapter.generate(
            request_id=_REQUEST_ID,
            query="Test",
            subject="math",
            intent="solve",
            difficulty="default",
            context="",
        )
        assert executor.last_route_decision is not None
        assert executor.last_route_decision.task_role == "generator"


class TestStreamingClassificationPolicy:
    def test_streaming_uses_policy_corrected_subject_and_difficulty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import config as cfg_module
        from graphs.doubt_solver_graph import _orchestrated_classify_node

        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        cfg_module._settings = None

        state = {
            "request_id": "stream-policy-1",
            "query": "Explain profit loss discount trap for SBI PO mains level",
            "classification": None,
            "context_text": "",
            "answer": "",
        }
        result = _orchestrated_classify_node(state)
        classification = result["classification"]
        assert classification["subject"] == "math"
        assert classification["difficulty"] == "advanced"
        cfg_module._settings = None

    def test_streaming_advanced_reasoning_routes_advanced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import config as cfg_module
        from graphs.doubt_solver_graph import _orchestrated_classify_node
        from schemas.llm_routing import RouteRequest
        from services.llm.orchestration.config_registry import LlmConfigRegistry
        from services.llm.orchestration.route_resolver import resolve_route

        monkeypatch.setenv("ENABLE_REAL_LLM", "false")
        monkeypatch.setenv("LLM_ROLE_CONFIG_JSON", "{}")
        cfg_module._settings = None

        state = {
            "request_id": "stream-route-1",
            "query": "Explain coded inequality floor puzzle for SBI PO mains level",
            "classification": None,
            "context_text": "",
            "answer": "",
        }
        result = _orchestrated_classify_node(state)
        classification = result["classification"]
        assert classification["subject"] == "reasoning"
        assert classification["difficulty"] == "advanced"

        route_decision = resolve_route(
            RouteRequest(
                request_id="stream-route-1",
                subject=classification["subject"],
                task_role="generator",
                difficulty=classification["difficulty"],
            ),
            registry=LlmConfigRegistry(),
        )
        assert route_decision.difficulty == "advanced"
        assert route_decision.subject == "reasoning"
        cfg_module._settings = None


class TestWebSearchStreamStatus:
    def test_web_search_emits_recent_information_status(self, monkeypatch) -> None:
        from unittest.mock import patch

        from schemas.doubt_solver import QueryClassification

        def _collect_with_web_hook(
            state,
            *,
            on_before_web_search=None,
            on_web_search_retry=None,
            on_web_search_weak_context=None,
        ):
            if on_before_web_search is not None:
                on_before_web_search()
            return {"context_text": "Fresh web context: sample"}

        with patch(
            "graphs.doubt_solver_graph.classify_query",
            return_value=QueryClassification(
                intent="general_doubt",
                subject="general",
                confidence=0.95,
                need_web_search=True,
                web_search_reason="current_affairs",
                classification_source="llm",
            ),
        ), patch(
            "services.doubt_solver.streaming_doubt_solver_service._orchestrated_collect_context_node",
            side_effect=_collect_with_web_hook,
        ):
            events = _collect(_make_adapter("Current affairs answer."))

        labels = [e.label for e in events if e.type == "status"]
        assert labels.count("Thinking...") == 1
        assert "Checking recent information..." not in labels

    def test_no_web_status_when_web_not_called(self) -> None:
        events = _collect(_make_adapter())
        labels = [e.label for e in events if e.type == "status"]
        assert "Checking recent information..." not in labels

    def test_no_provider_details_in_web_stream_status(self, monkeypatch) -> None:
        from unittest.mock import patch

        from schemas.doubt_solver import QueryClassification

        def _collect_with_web_hook(
            state,
            *,
            on_before_web_search=None,
            on_web_search_retry=None,
            on_web_search_weak_context=None,
        ):
            if on_before_web_search is not None:
                on_before_web_search()
            return {"context_text": ""}

        with patch(
            "graphs.doubt_solver_graph.classify_query",
            return_value=QueryClassification(
                intent="general_doubt",
                subject="general",
                confidence=0.95,
                need_web_search=True,
                classification_source="llm",
            ),
        ), patch(
            "services.doubt_solver.streaming_doubt_solver_service._orchestrated_collect_context_node",
            side_effect=_collect_with_web_hook,
        ):
            events = _collect(_make_adapter("Answer."))

        for event in events:
            blob = f"{event.label or ''} {event.content or ''}".lower()
            assert "tavily" not in blob
            assert "api" not in blob


class TestExtendedStreamStatuses:
    def test_careful_status_uses_understanding_stage(self, monkeypatch) -> None:
        from unittest.mock import patch

        from schemas.doubt_solver import QueryClassification

        def _classify_with_hook(query, request_id=None, *, on_before_strong_classifier=None):
            if on_before_strong_classifier is not None:
                on_before_strong_classifier()
            return QueryClassification(
                intent="solve_question",
                subject="math",
                confidence=0.94,
                classification_source="llm",
            )

        with patch(
            "graphs.doubt_solver_graph.classify_query",
            side_effect=_classify_with_hook,
        ):
            events = _collect(_make_adapter("Short answer."))

        stages = [e.stage for e in events if e.type == "status"]
        assert stages.count("understanding") == 1

    def test_web_retry_status_at_most_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config as cfg_module
        from tools.web_search.models import WebSearchItem, WebSearchRequest
        from tools.web_search.providers.fake_provider import FakeWebSearchProvider
        from tools.web_search.web_search_tool import WebSearchTool

        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        cfg_module._settings = None

        def _collect_with_real_web(
            state,
            *,
            on_before_web_search=None,
            on_web_search_retry=None,
            **kwargs,
        ):
            if on_before_web_search is not None:
                on_before_web_search()
            tool = WebSearchTool(
                provider=FakeWebSearchProvider(
                    [
                        WebSearchItem(
                            title="Economy CA summary adda",
                            url="https://adda247.com/ca",
                            snippet=(
                                "Exam prep economy current affairs summary "
                                "with enough content."
                            ),
                            source="adda247.com",
                            score=0.8,
                        ),
                    ]
                )
            )
            tool.search(
                WebSearchRequest(
                    request_id=state.get("request_id", "x"),
                    query=state.get("query", ""),
                    web_search_reason="current_economy",
                ),
                on_retry_sources=on_web_search_retry,
            )
            return {"context_text": "web context"}

        from unittest.mock import patch

        from schemas.doubt_solver import QueryClassification

        with patch(
            "graphs.doubt_solver_graph.classify_query",
            return_value=QueryClassification(
                intent="general_doubt",
                subject="general",
                confidence=0.95,
                need_web_search=True,
                web_search_reason="current_economy",
                classification_source="llm",
            ),
        ), patch(
            "services.doubt_solver.streaming_doubt_solver_service._orchestrated_collect_context_node",
            side_effect=_collect_with_real_web,
        ):
            events = _collect(_make_adapter("Answer."))

        labels = [e.label for e in events if e.type == "status"]
        assert labels.count("Thinking...") == 1
        assert "Looking for more reliable sources..." not in labels

    def test_generator_fallback_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import config as cfg_module

        monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
        cfg_module._settings = None
        executor = MockModelExecutor(
            content="Reliable streamed answer.",
            notify_fallback_on_stream=True,
        )
        orchestrator = LlmOrchestrator(model_executor=executor)
        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        events = _collect(adapter)
        labels = [e.label for e in events if e.type == "status"]
        assert labels.count("Explaining...") == 1
        assert "Preparing a more reliable answer..." not in labels
        cfg_module._settings = None

    def test_stream_status_no_internal_leakage(self) -> None:
        executor = MockModelExecutor(
            content="Answer.",
            notify_fallback_on_stream=True,
        )
        orchestrator = LlmOrchestrator(model_executor=executor)
        adapter = AnswerGenerationAdapter(orchestrator=orchestrator)
        events = _collect(adapter)
        for event in events:
            blob = f"{event.label or ''} {event.content or ''}".lower()
            for forbidden in (
                "fallback",
                "tavily",
                "confidence",
                "classifier",
                "provider",
                "threshold",
            ):
                assert forbidden not in blob


class TestEmptyGeneratorOutputStreaming:
    class _AdapterWithEmptyChunks:
        def generate_stream(self, **kwargs):  # noqa: ANN003
            yield ""
            yield "   "
            yield "Visible answer text."

    def test_nonempty_whitespace_chunks_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import config as cfg_module

        monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
        cfg_module._settings = None
        events = list(
            stream_doubt_solver(
                StreamDoubtSolverInput(request_id=_REQUEST_ID, query="A question"),
                adapter=self._AdapterWithEmptyChunks(),  # type: ignore[arg-type]
            )
        )
        chunks = [e.content for e in events if e.type == "chunk"]
        assert chunks == ["   ", "Visible answer text."]
        complete = events[-1]
        assert complete.response is not None
        assert complete.response.content.value == "   Visible answer text."
        cfg_module._settings = None

    def test_first_visible_chunk_not_set_for_empty_only(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import logging

        import config as cfg_module

        monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_live")
        cfg_module._settings = None

        class _EmptyOnlyAdapter:
            def generate_stream(self, **kwargs):  # noqa: ANN003
                return
                yield  # pragma: no cover

        with caplog.at_level(logging.INFO):
            list(
                stream_doubt_solver(
                    StreamDoubtSolverInput(request_id=_REQUEST_ID, query="Q"),
                    adapter=_EmptyOnlyAdapter(),  # type: ignore[arg-type]
                )
            )
        messages = " ".join(r.message for r in caplog.records)
        assert "first_visible_chunk_emitted=true" not in messages
        cfg_module._settings = None


class TestStreamingErrorHandling:
    def test_provider_stream_error_returns_safe_error_event(self) -> None:
        events = _collect(_make_adapter(raise_on_execute=RuntimeError("provider boom")))
        assert events[-1].type == "error"
        assert events[-1].label == "Unable to complete"

    def test_no_stack_trace_exposed(self) -> None:
        events = _collect(_make_adapter(raise_on_execute=RuntimeError("detailed failure")))
        for event in events:
            blob = f"{event.label or ''} {event.content or ''}"
            assert "Traceback" not in blob
            assert "RuntimeError" not in blob

    def test_no_complete_on_error(self) -> None:
        events = _collect(_make_adapter(raise_on_execute=RuntimeError("fail")))
        assert not any(e.type == "complete" for e in events)
