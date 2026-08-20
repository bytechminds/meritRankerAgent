"""
tests/test_web_search_tool.py
--------------------------------
Unit tests for conditional web search tool and decision rules.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import config as cfg_module
from services.context_retrieval.context_models import ContextRetrievalRequest
from services.context_retrieval.context_retrieval_service import (
    ContextRetrievalService,
    reset_context_retrieval_service,
)
from services.context_retrieval.web_search_decision import (
    evaluate_web_search_decision,
    has_freshness_signal,
    should_attempt_web_fallback,
    should_skip_kb_for_direct_web,
)
from tools.web_search.formatter import format_web_context
from tools.web_search.models import WebSearchItem, WebSearchProviderRequest, WebSearchRequest
from tools.web_search.providers.fake_provider import FakeWebSearchProvider
from tools.web_search.providers.tavily_provider import TavilyWebSearchProvider
from tools.web_search.query_builder import WebSearchQueryBuilder
from tools.web_search.source_policy import WebSourcePolicyResolver
from tools.web_search.web_search_tool import WebSearchTool, build_fake_web_search_tool


def _reset_settings() -> None:
    cfg_module._settings = None


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Web/KB compatibility tests use the explicit legacy retrieval provider."""
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "legacy_bedrock_kb")
    _reset_settings()
    reset_context_retrieval_service()
    yield
    _reset_settings()
    reset_context_retrieval_service()


def _request(**overrides: Any) -> ContextRetrievalRequest:
    base = {
        "request_id": "req-web-1",
        "query": "Latest current affairs for UPSC prelims",
        "subject": "general",
        "intent": "explain",
        "difficulty": "intermediate",
    }
    base.update(overrides)
    return ContextRetrievalRequest(**base)


class TestWebSearchDecisionRules:
    def test_explicit_latest_current_affairs_direct_web(self) -> None:
        req = _request(
            need_web_search=True,
            web_search_reason="current_affairs",
            web_search_query="latest current affairs UPSC",
        )
        assert should_skip_kb_for_direct_web(req) is True

    def test_current_events_today_freshness_signal(self) -> None:
        assert has_freshness_signal("What are the current events today?") is True

    def test_latest_rbi_repo_rate_freshness(self) -> None:
        req = _request(
            query="What is the latest RBI repo rate?",
            need_web_search=True,
            web_search_reason="current_economy",
        )
        assert should_skip_kb_for_direct_web(req) is True

    def test_static_history_no_web_fallback(self) -> None:
        req = _request(
            query="Who founded the Maurya Empire?",
            subject="general",
            difficulty="basic",
        )
        assert should_attempt_web_fallback(req, kb_selected=False) is False

    def test_static_polity_no_web_fallback(self) -> None:
        req = _request(
            query="Explain the basic structure doctrine of the Indian Constitution",
            subject="general",
            difficulty="basic",
        )
        assert should_attempt_web_fallback(req, kb_selected=False) is False

    def test_normal_math_no_web_fallback(self) -> None:
        req = _request(
            query="Find 25% of 480",
            subject="math",
            difficulty="intermediate",
        )
        assert should_attempt_web_fallback(req, kb_selected=False) is False

    def test_normal_reasoning_no_web_fallback(self) -> None:
        req = _request(
            query="If A > B and B > C, which conclusion follows?",
            subject="reasoning",
            difficulty="advanced",
        )
        assert should_attempt_web_fallback(req, kb_selected=False) is False

    def test_english_grammar_no_web_fallback(self) -> None:
        req = _request(
            query="Correct the sentence: He don't like apples",
            subject="english",
            difficulty="intermediate",
        )
        assert should_attempt_web_fallback(req, kb_selected=False) is False

    def test_kb_selected_blocks_fallback(self) -> None:
        req = _request(
            query="Latest economic survey highlights",
            subject="general",
            difficulty="advanced",
        )
        assert should_attempt_web_fallback(req, kb_selected=True) is False

    def test_kb_miss_general_intermediate_with_freshness_calls_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        _reset_settings()
        req = _request(
            query="Latest economic indicators for India",
            subject="general",
            difficulty="intermediate",
        )
        assert should_attempt_web_fallback(req, kb_selected=False) is True


class TestWebSearchContextIntegration:
    def test_s3_vector_mode_honors_classifier_web_search_demand(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RETRIEVAL_PROVIDER", "s3_vector")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        _reset_settings()

        student_retrieval = MagicMock()
        service = ContextRetrievalService(
            web_search_tool=build_fake_web_search_tool(),
            student_retrieval_service=student_retrieval,
        )
        result = service.retrieve_context(
            _request(
                need_web_search=True,
                web_search_reason="current_affairs",
                web_search_query="latest current affairs",
            )
        )

        student_retrieval.retrieve.assert_not_called()
        assert result.retrieval_used is True
        assert "[Web Context]" in result.context_text

    def test_s3_vector_mode_static_question_does_not_call_web(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from retrieval.models import StudentRetrievalContext

        monkeypatch.setenv("RETRIEVAL_PROVIDER", "s3_vector")
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        _reset_settings()

        student_retrieval = MagicMock()
        student_retrieval.retrieve.return_value = StudentRetrievalContext.fresh_solve(
            "no_candidate"
        )
        web_tool = MagicMock()
        service = ContextRetrievalService(
            web_search_tool=web_tool,
            student_retrieval_service=student_retrieval,
        )
        result = service.retrieve_context(
            _request(
                query="Explain Akbar's revenue administration.",
                subject="general",
                difficulty="basic",
                need_web_search=False,
            )
        )

        web_tool.search.assert_not_called()
        student_retrieval.retrieve.assert_called_once()
        assert result.reason == "no_candidate"

    def test_search_start_hook_runs_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        _reset_settings()

        hook = MagicMock()
        service = ContextRetrievalService(web_search_tool=build_fake_web_search_tool())
        service.retrieve_context(
            _request(
                need_web_search=True,
                web_search_reason="current_affairs",
                web_search_query="latest current affairs",
            ),
            on_before_web_search=hook,
        )

        hook.assert_called_once_with()

    def test_need_web_search_skips_kb_for_current_affairs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "kb-test")
        _reset_settings()

        kb_mock = MagicMock()
        tool = build_fake_web_search_tool()
        service = ContextRetrievalService(kb_retriever=kb_mock, web_search_tool=tool)
        req = _request(
            need_web_search=True,
            web_search_reason="current_affairs",
            web_search_query="latest current affairs",
        )
        result = service.retrieve_context(req)
        kb_mock.retrieve_lane.assert_not_called()
        assert result.retrieval_used is True
        assert "[Solution Brief]" in result.context_text or "[Web Context]" in result.context_text

    def test_kb_selected_context_prevents_web_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "kb-test")
        _reset_settings()

        from services.context_retrieval.context_models import RetrievedContextItem
        from services.context_retrieval.context_retrieval_service import (
            LANE_SUBJECT_ONLY,
        )

        high_item = RetrievedContextItem(
            text="Pattern context for profit loss.",
            score=0.92,
            source_id="pat-1",
            metadata={
                "patternId": "pat-1",
                "subject": "QUANT",
                "patternTopicKey": "PROFIT_LOSS_DISCOUNT",
                "patternFamilyKey": "DISCOUNT",
                "complexityLevel": "3",
                "confidence": "1.00",
                "taxonomyReviewRequired": "false",
                "schemaVersion": "v2",
            },
            match_lane=LANE_SUBJECT_ONLY,
        )
        kb_mock = MagicMock()
        kb_mock.retrieve_lane = MagicMock(return_value=([high_item], 1))
        web_tool = MagicMock()
        service = ContextRetrievalService(kb_retriever=kb_mock, web_search_tool=web_tool)
        req = _request(
            query="Explain profit loss discount trap for SBI PO",
            subject="math",
            difficulty="advanced",
            need_web_search=False,
        )
        result = service.retrieve_context(req)
        web_tool.search.assert_not_called()
        assert result.retrieval_used is True
        assert "[Solution Brief]" in result.context_text

    def test_kb_miss_math_does_not_call_web(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "kb-test")
        _reset_settings()

        kb_mock = MagicMock()
        kb_mock.retrieve_lane = MagicMock(return_value=([], 0))
        web_tool = MagicMock()
        service = ContextRetrievalService(kb_retriever=kb_mock, web_search_tool=web_tool)
        req = _request(
            query="Solve this age problem with equations",
            subject="math",
            difficulty="intermediate",
            need_web_search=False,
        )
        service.retrieve_context(req)
        web_tool.search.assert_not_called()

    def test_kb_miss_general_intermediate_calls_web_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        _reset_settings()

        tool = build_fake_web_search_tool()
        service = ContextRetrievalService(web_search_tool=tool)
        req = _request(
            query="Latest RBI monetary policy update",
            subject="general",
            difficulty="intermediate",
            need_web_search=False,
            web_search_reason="freshness_required",
        )
        result = service.retrieve_context(req)
        assert result.retrieval_used is True
        assert "[Solution Brief]" in result.context_text or "[Web Context]" in result.context_text

    def test_disabled_web_returns_safe_no_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
        _reset_settings()
        tool = build_fake_web_search_tool()
        service = ContextRetrievalService(web_search_tool=tool)
        req = _request(need_web_search=True, web_search_reason="current_affairs")
        result = service.retrieve_context(req)
        assert result.context_text == ""
        assert result.reason == "web_search_disabled"

    def test_missing_api_key_returns_safe_no_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "")
        _reset_settings()
        service = ContextRetrievalService(web_search_tool=WebSearchTool())
        req = _request(need_web_search=True, web_search_reason="current_affairs")
        result = service.retrieve_context(req)
        assert result.context_text == ""
        assert result.reason == "missing_credentials"


class TestWebSearchToolProvider:
    def test_absolute_month_window_does_not_send_relative_time_range(self) -> None:
        settings = cfg_module.get_settings()
        policy = WebSourcePolicyResolver().resolve(
            query="provide current affairs question july 2026",
            web_search_query="current affairs July 2026",
            subject="general",
            topic="Current Affairs",
            retrieval_tags=["current_affairs", "july_2026"],
            web_search_reason="current_affairs",
            source_strictness=settings.web_search_source_strictness,
            default_recent_days=settings.web_search_default_recent_days,
        )
        attempt = WebSearchQueryBuilder.plan_attempts(
            policy,
            allow_generic_fallback=False,
            allow_exam_prep_fallback=False,
            exam_prep_suitable=False,
            official_only=False,
        )[0]
        request = WebSearchQueryBuilder.build_provider_request(
            search_query="current affairs July 2026",
            policy=policy,
            attempt=attempt,
            max_results=5,
            search_depth="basic",
            timeout_seconds=8,
        )

        assert request.start_date == "2026-07-01"
        assert request.end_date == "2026-07-31"
        assert request.time_range is None

    def test_fake_provider_returns_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        _reset_settings()
        tool = build_fake_web_search_tool()
        result = tool.search(
            WebSearchRequest(
                request_id="req-1",
                query="latest RBI repo rate",
                web_search_reason="current_economy",
                timeout_seconds=5,
            )
        )
        assert result.used is True
        assert len(result.items) >= 1

    def test_fresh_evidence_filters_undated_and_stale_results_before_reranking(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        _reset_settings()

        class CountingProvider(FakeWebSearchProvider):
            def __init__(self, items: list[WebSearchItem]) -> None:
                super().__init__(items)
                self.calls = 0

            def search(self, request: WebSearchProviderRequest):
                self.calls += 1
                return super().search(request)

        provider = CountingProvider(
            [
                WebSearchItem(
                    title="2024 current affairs award update",
                    url="https://pib.gov.in/fresh-one",
                    snippet="A verified 2024 national award event with sufficient factual detail.",
                    source="pib.gov.in",
                    published_at="2024-07-20",
                    score=0.8,
                ),
                WebSearchItem(
                    title="2023 stale current affairs award update",
                    url="https://pib.gov.in/stale",
                    snippet="An older award event that must not satisfy a 2024 evidence request.",
                    source="pib.gov.in",
                    published_at="2023-12-31",
                    score=0.99,
                ),
                WebSearchItem(
                    title="Undated current affairs award update",
                    url="https://pib.gov.in/undated",
                    snippet=(
                        "An undated article whose temporal relevance cannot be established safely."
                    ),
                    source="pib.gov.in",
                    score=0.98,
                ),
                WebSearchItem(
                    title="Second 2024 current affairs event",
                    url="https://pib.gov.in/fresh-two",
                    snippet=(
                        "A second verified 2024 event with enough evidence for exam preparation."
                    ),
                    source="pib.gov.in",
                    published_at="2024-06-15",
                    score=0.75,
                ),
                WebSearchItem(
                    title="Duplicate 2024 event",
                    url="https://pib.gov.in/fresh-one",
                    snippet="The duplicate URL must not inflate usable fresh evidence count.",
                    source="pib.gov.in",
                    published_at="2024-07-20",
                    score=0.7,
                ),
            ]
        )

        result = WebSearchTool(provider=provider).search(
            WebSearchRequest(
                request_id="fresh-filter",
                query="Create 2 current affairs questions from 2024 for SSC GD",
                subject="general",
                topic="current_affairs",
                web_search_reason="current_affairs",
                requires_fresh_evidence=True,
                required_evidence_count=2,
            )
        )

        assert provider.calls == 1
        assert result.used is True
        assert result.temporal_mode == "EXPLICIT_YEAR"
        assert "current affairs" in result.query.lower()
        assert "explicit_year coverage 2024-01-01 to 2024-12-31" in result.query
        assert "ssc exam preparation" in result.query.lower()
        assert result.stale_results_rejected == 2
        assert {item.url for item in result.items} == {
            "https://pib.gov.in/fresh-one",
            "https://pib.gov.in/fresh-two",
        }
        assert result.fresh_evidence is not None
        assert result.fresh_evidence.requested_window.temporal_mode == "EXPLICIT_YEAR"

    def test_hundred_slot_fresh_request_uses_one_bounded_search_and_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        _reset_settings()

        class CountingProvider(FakeWebSearchProvider):
            def __init__(self) -> None:
                super().__init__(
                    [
                        WebSearchItem(
                            title="2024 current-affairs source",
                            url="https://pib.gov.in/one",
                            snippet="A verified current-affairs fact with enough factual detail.",
                            source="pib.gov.in",
                            published_at="2024-06-01",
                        )
                    ]
                )
                self.calls = 0

            def search(self, request: WebSearchProviderRequest):
                self.calls += 1
                return super().search(request)

        provider = CountingProvider()
        result = WebSearchTool(provider=provider).search(
            WebSearchRequest(
                request_id="fresh-100",
                query="Create 100 current affairs questions from 2024",
                subject="general",
                topic="current_affairs",
                web_search_reason="current_affairs",
                requires_fresh_evidence=True,
                required_evidence_count=100,
            )
        )

        assert provider.calls == 1
        assert result.used is False
        assert result.weak_context is True
        assert result.eligible_evidence_count == 1
        assert result.fresh_evidence is None

    def test_formatter_truncates_max_chars(self) -> None:
        items = [
            WebSearchItem(
                title=f"Title {idx}",
                url=f"https://example.com/{idx}",
                snippet="x" * 500,
                source="example.com",
            )
            for idx in range(5)
        ]
        text = format_web_context(items, max_chars=400)
        assert len(text) <= 400

    def test_tavily_request_built_without_logging_secrets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        class _FakeResponse:
            def read(self) -> bytes:
                payload = (
                    b'{"results": [{"title": "Headline", "url": "https://a.com", '
                    b'"content": "Snippet"}]}'
                )
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def _fake_urlopen(request, timeout=8):  # noqa: ANN001
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = request.data
            return _FakeResponse()

        monkeypatch.setattr(
            "tools.web_search.providers.tavily_provider.urllib.request.urlopen",
            _fake_urlopen,
        )
        provider = TavilyWebSearchProvider(api_key="secret-key")
        result = provider.search(
            WebSearchProviderRequest(
                query="latest RBI repo rate",
                max_results=2,
                timeout_seconds=5,
                metadata={"attempt": "authoritative"},
            )
        )
        assert len(result.items) == 1
        assert captured["authorization"] == "Bearer secret-key"
        assert b"api_key" not in captured["body"]

    def test_disabled_tool_does_not_call_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
        _reset_settings()
        provider = MagicMock()
        tool = WebSearchTool(provider=provider)
        result = tool.search(
            WebSearchRequest(
                request_id="req-1",
                query="latest news",
                timeout_seconds=5,
            )
        )
        provider.search.assert_not_called()
        assert result.used is False


class TestWebSearchDecisionLogging:
    def test_evaluate_decision_will_call_requires_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("TAVILY_API_KEY", "")
        _reset_settings()
        settings = cfg_module.get_settings()
        req = _request(need_web_search=True, web_search_reason="current_affairs")
        decision = evaluate_web_search_decision(req, settings)
        assert decision.will_call is False
