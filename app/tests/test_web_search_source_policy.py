"""
tests/test_web_search_source_policy.py
---------------------------------------
Source pack loader, policy resolver, query builder, and provider abstraction tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config as cfg_module
from tools.web_search.models import WebSearchItem, WebSearchProviderRequest, WebSearchRequest
from tools.web_search.providers.fake_provider import FakeWebSearchProvider
from tools.web_search.providers.tavily_provider import TavilyWebSearchProvider
from tools.web_search.query_builder import WebSearchQueryBuilder
from tools.web_search.scope_policy import detect_source_scope_policy
from tools.web_search.search_query_builder import build_scope_aware_search_query
from tools.web_search.source_pack_loader import load_source_pack_catalog
from tools.web_search.source_policy import WebSourcePolicyResolver
from tools.web_search.web_search_tool import WebSearchTool, build_fake_web_search_tool


def _reset_settings() -> None:
    cfg_module._settings = None


@pytest.fixture(autouse=True)
def _clean() -> None:
    _reset_settings()
    yield
    _reset_settings()


class TestSourcePackLoader:
    def test_loads_starter_packs(self) -> None:
        catalog = load_source_pack_catalog()
        assert "economy_india" in catalog.packs
        assert "international_current_affairs" in catalog.packs
        assert "youtube.com" in catalog.global_blocked
        economy = catalog.packs["economy_india"]
        assert "finance.gov.in" in economy.trusted_domains
        assert "drishtiias.com" in economy.exam_prep_domains
        assert "drishtiias.com" not in economy.trusted_domains

    def test_exam_prep_domains_defaults_empty(self, tmp_path: Path) -> None:
        pack_file = tmp_path / "packs.yaml"
        pack_file.write_text(
            "global_blocked:\n  domains: []\npacks:\n  test_pack:\n"
            "    topic: news\n    trusted_domains: [example.gov.in]\n"
            "    reputed_domains: []\n    blocked_domains: []\n",
            encoding="utf-8",
        )
        catalog = load_source_pack_catalog(pack_file)
        pack = catalog.get_pack("test_pack")
        assert pack.exam_prep_domains == ()

    def test_validates_required_keys(self, tmp_path: Path) -> None:
        pack_file = tmp_path / "packs.yaml"
        pack_file.write_text(
            "global_blocked:\n  domains: []\npacks:\n  test_pack:\n"
            "    topic: news\n    trusted_domains: [example.gov.in]\n"
            "    reputed_domains: []\n    blocked_domains: []\n",
            encoding="utf-8",
        )
        catalog = load_source_pack_catalog(pack_file)
        pack = catalog.get_pack("test_pack")
        assert pack.topic == "news"
        assert pack.trusted_domains == ("example.gov.in",)

    def test_unknown_pack_safe_fallback(self) -> None:
        catalog = load_source_pack_catalog()
        pack = catalog.get_pack("does_not_exist")
        assert pack.name in {"current_affairs_india", "default"}


class TestWebSourcePolicyResolver:
    def test_economy_current_affairs_pack(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="What is the latest RBI repo rate?",
            web_search_query="latest RBI repo rate",
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_economy",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.source_pack_name in {"economy_india", "economy_india_mixed"}
        assert "rbi.org.in" in policy.trusted_domains

    def test_latest_scheme_pack(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="Explain the latest government scheme for farmers",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.source_pack_name in {
            "government_schemes_india",
            "government_schemes_india_mixed",
        }

    def test_exam_update_pack(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="Latest UPSC exam notification and admit card update",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="latest_exam_update",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.source_pack_name == "exam_updates_india"

    def test_monthly_current_affairs_ssc_not_exam_updates(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="Monthly current affairs summary for SSC preparation",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.source_pack_name == "current_affairs_mixed"
        assert policy.scope == "mixed"
        assert policy.india_weight == 70
        assert policy.world_weight == 30
        assert policy.source_need == "practice_current_affairs"

    def test_government_scheme_benefits_pack(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="Latest government scheme benefit amount for farmers",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.source_pack_name in {
            "government_schemes_india",
            "government_schemes_india_mixed",
        }

    def test_sports_current_affairs_pack(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="Recent cricket world cup current affairs",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.source_pack_name in {
            "sports_current_affairs",
            "sports_current_affairs_mixed",
        }

    def test_recent_without_date_uses_default_recent_days(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="Latest current affairs update",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.start_date is not None
        assert policy.end_date is not None

    def test_may_2026_month_range(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="Current affairs for May 2026",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.start_date == "2026-05-01"
        assert policy.end_date == "2026-05-31"

    def test_today_and_yesterday(self) -> None:
        resolver = WebSourcePolicyResolver()
        today_policy = resolver.resolve(
            query="Current events today",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_event",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert today_policy.start_date == today_policy.end_date
        yesterday_policy = resolver.resolve(
            query="News from yesterday",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_event",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert yesterday_policy.start_date == yesterday_policy.end_date


class TestQueryBuilderAttempts:
    def test_evidence_required_current_affairs_tries_trusted_alone_first(self) -> None:
        """Evidence-required current affairs must satisfy min_trusted_results.

        A merged trusted+reputed query lets high-volume news domains outrank
        government ones, so official sources never surfaced and every such request
        failed closed. Broad exam-prep digests keep the cheaper merged-first plan.
        """
        policy = WebSourcePolicyResolver().resolve(
            query="current affairs India August 2026",
            web_search_query="current affairs India August 2026",
            subject="polity",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_event",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        grounded = WebSearchQueryBuilder.plan_attempts(
            policy,
            allow_generic_fallback=True,
            allow_exam_prep_fallback=True,
            exam_prep_suitable=True,
            official_only=False,
            requires_fresh_evidence=True,
        )
        assert grounded[0].kind == "authoritative"
        assert grounded[0].include_domains == policy.trusted_domains
        assert grounded[1].kind == "authoritative_plus_reputed"

        ungrounded = WebSearchQueryBuilder.plan_attempts(
            policy,
            allow_generic_fallback=True,
            allow_exam_prep_fallback=True,
            exam_prep_suitable=True,
            official_only=False,
        )
        assert ungrounded[0].kind == "authoritative_plus_reputed"

    def test_broad_current_affairs_starts_with_combined_trusted_sources(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="monthly current affairs summary for SSC",
            web_search_query="monthly current affairs SSC",
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        attempts = WebSearchQueryBuilder.plan_attempts(
            policy,
            allow_generic_fallback=True,
            allow_exam_prep_fallback=True,
            exam_prep_suitable=True,
            official_only=False,
        )
        assert [a.kind for a in attempts] == [
            "authoritative_plus_reputed",
            "exam_prep_fallback",
            "generic_fallback",
        ]
        assert set(policy.trusted_domains).issubset(attempts[0].include_domains)
        assert set(policy.reputed_domains).issubset(attempts[0].include_domains)
        assert attempts[1].include_domains == policy.exam_prep_domains
        assert "youtube.com" in attempts[0].exclude_domains

    def test_exam_prep_not_planned_for_official_only(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="Latest UPSC admit card download link",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="latest_exam_update",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        attempts = WebSearchQueryBuilder.plan_attempts(
            policy,
            allow_generic_fallback=True,
            allow_exam_prep_fallback=True,
            exam_prep_suitable=False,
            official_only=True,
        )
        assert "exam_prep_fallback" not in [a.kind for a in attempts]

    def test_exam_prep_disabled_skips_attempt(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="economy current affairs summary",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_economy",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        attempts = WebSearchQueryBuilder.plan_attempts(
            policy,
            allow_generic_fallback=False,
            allow_exam_prep_fallback=False,
            exam_prep_suitable=True,
            official_only=False,
        )
        assert [a.kind for a in attempts] == [
            "authoritative",
            "authoritative_plus_reputed",
        ]


class TestProviderAbstraction:
    def test_context_retrieval_service_has_no_tavily_params(self) -> None:
        service_path = (
            Path(__file__).resolve().parents[1]
            / "services/context_retrieval/context_retrieval_service.py"
        )
        source = service_path.read_text(encoding="utf-8")
        assert "Tavily" not in source
        assert "include_raw_content" not in source

    def test_tavily_maps_neutral_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        class _FakeResponse:
            def read(self) -> bytes:
                return b'{"results": []}'

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def _fake_urlopen(request, timeout=8):  # noqa: ANN001
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["authorization"] = request.get_header("Authorization")
            return _FakeResponse()

        monkeypatch.setattr(
            "tools.web_search.providers.tavily_provider.urllib.request.urlopen",
            _fake_urlopen,
        )
        provider = TavilyWebSearchProvider(api_key="secret-key")
        provider.search(
            WebSearchProviderRequest(
                query="latest RBI repo rate",
                topic="finance",
                include_domains=["rbi.org.in"],
                exclude_domains=["youtube.com"],
                start_date="2026-05-01",
                end_date="2026-05-31",
                max_results=3,
                search_depth="basic",
                include_raw_content=False,
                metadata={"attempt": "authoritative"},
            )
        )
        body = captured["body"]
        assert body["include_raw_content"] is False
        assert body["include_answer"] is False
        assert body["include_images"] is False
        assert body["include_domains"] == ["rbi.org.in"]
        assert body["exclude_domains"] == ["youtube.com"]
        assert body["topic"] == "finance"
        assert "api_key" not in body
        assert captured["authorization"] == "Bearer secret-key"


class TestSearchAttemptsIntegration:
    def test_authoritative_attempt_used_for_trusted_domain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()
        tool = build_fake_web_search_tool()
        result = tool.search(
            WebSearchRequest(
                request_id="attempt-1",
                query="Latest RBI repo rate update",
                web_search_reason="current_economy",
            )
        )
        assert result.used is True
        assert result.source_pack_name in {"economy_india", "economy_india_mixed"}
        assert result.attempt_used in {"authoritative", "authoritative_plus_reputed"}
        assert "Content:" in result.context_text

    def test_generic_fallback_blocked_when_trusted_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_ALLOW_GENERIC_FALLBACK", "true")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()

        generic_only = [
            WebSearchItem(
                title="Random blog RBI",
                url="https://random-blog.example/post",
                snippet="Some unofficial commentary about RBI repo rate changes today.",
                source="random-blog.example",
                score=0.4,
            )
        ]
        tool = WebSearchTool(provider=FakeWebSearchProvider(generic_only))
        result = tool.search(
            WebSearchRequest(
                request_id="attempt-2",
                query="Latest RBI repo rate",
                web_search_reason="current_economy",
            )
        )
        assert result.weak_context is True
        assert "limited" in result.context_text
        assert "Content:" not in result.context_text


class TestScopePolicy:
    def test_upsc_current_affairs_mixed_default(self) -> None:
        scope = detect_source_scope_policy(
            query="current affairs questions for UPSC",
            web_search_query=None,
            web_search_reason="current_affairs",
        )
        assert scope.scope == "mixed"
        assert scope.india_weight == 70
        assert scope.world_weight == 30
        assert scope.source_need == "practice_current_affairs"
        assert scope.exam_context == "UPSC"
        assert scope.official_exam_lifecycle is False

    def test_usa_iran_world_scope(self) -> None:
        scope = detect_source_scope_policy(
            query="current affairs questions for USA and Iran updates for UPSC",
            web_search_query=None,
            web_search_reason="current_affairs",
        )
        assert scope.scope == "world"
        assert scope.world_weight == 100
        assert scope.india_weight == 0
        assert scope.source_need == "practice_current_affairs"

    def test_india_explicit_scope(self) -> None:
        scope = detect_source_scope_policy(
            query="India current affairs May 2026 for SSC",
            web_search_query=None,
            web_search_reason="current_affairs",
        )
        assert scope.scope == "india"
        assert scope.india_weight == 100
        assert scope.world_weight == 0

    def test_global_current_affairs_world_scope(self) -> None:
        scope = detect_source_scope_policy(
            query="global current affairs for UPSC",
            web_search_query=None,
            web_search_reason="current_affairs",
        )
        assert scope.scope == "world"
        assert scope.world_weight == 100

    def test_exam_name_not_exam_updates(self) -> None:
        scope = detect_source_scope_policy(
            query="UPSC current affairs questions",
            web_search_query=None,
            web_search_reason="current_affairs",
        )
        assert scope.source_need == "practice_current_affairs"
        assert scope.official_exam_lifecycle is False

    def test_ssc_economy_not_exam_updates(self) -> None:
        scope = detect_source_scope_policy(
            query="SSC economy current affairs",
            web_search_query=None,
            web_search_reason="current_economy",
        )
        assert scope.source_need == "economy"
        assert scope.official_exam_lifecycle is False

    def test_admit_card_lifecycle(self) -> None:
        scope = detect_source_scope_policy(
            query="latest SSC admit card",
            web_search_query=None,
            web_search_reason="latest_exam_update",
        )
        assert scope.source_need == "official_exam_update"
        assert scope.official_exam_lifecycle is True

    def test_ielts_result_lifecycle_future_safe(self) -> None:
        scope = detect_source_scope_policy(
            query="IELTS result date latest",
            web_search_query=None,
            web_search_reason="latest_exam_update",
        )
        assert scope.source_need == "official_exam_update"

    def test_mixed_pack_includes_india_and_world_domains(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="current affairs questions for UPSC",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.source_pack_name == "current_affairs_mixed"
        assert "pib.gov.in" in policy.trusted_domains
        assert "un.org" in policy.trusted_domains

    def test_world_query_uses_international_pack(self) -> None:
        policy = WebSourcePolicyResolver().resolve(
            query="current affairs questions for USA and Iran updates for UPSC",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="current_affairs",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        assert policy.source_pack_name == "international_current_affairs"
        assert "un.org" in policy.trusted_domains
        assert "pib.gov.in" not in policy.trusted_domains

    def test_query_builder_no_pack_names(self) -> None:
        scope = detect_source_scope_policy(
            query="current affairs questions for USA and Iran updates for UPSC",
            web_search_query=None,
            web_search_reason="current_affairs",
        )
        built = build_scope_aware_search_query(
            "current affairs questions for USA and Iran updates for UPSC",
            None,
            scope,
        )
        assert "international" in built.lower()
        assert "current_affairs_mixed" not in built
        assert "international_current_affairs" not in built

    def test_query_builder_india_scope(self) -> None:
        scope = detect_source_scope_policy(
            query="India economy current affairs May 2026 for SBI PO",
            web_search_query=None,
            web_search_reason="current_economy",
        )
        built = build_scope_aware_search_query(
            "India economy current affairs May 2026 for SBI PO",
            None,
            scope,
        )
        assert "India" in built or "india" in built.lower()
        assert "international relations" not in built.lower()

    def test_explicit_month_query_does_not_add_relative_latest_terms(self) -> None:
        scope = detect_source_scope_policy(
            query="provide current affairs questions July 2026",
            web_search_query="current affairs July 2026",
            web_search_reason="current_affairs",
        )

        built = build_scope_aware_search_query(
            "provide current affairs questions July 2026",
            "current affairs July 2026",
            scope,
        )

        assert built == "current affairs July 2026"


class TestOfficialOnlyGuard:
    def test_admit_card_is_official_only(self) -> None:
        from tools.web_search.official_only_guard import is_official_only_query

        assert is_official_only_query("Latest UPSC admit card download") is True

    def test_monthly_summary_is_exam_prep_suitable(self) -> None:
        from tools.web_search.official_only_guard import (
            is_exam_prep_suitable_query,
            is_official_only_query,
        )

        assert is_official_only_query("monthly current affairs for SSC") is False
        assert is_exam_prep_suitable_query(
            "monthly current affairs for SSC",
            web_search_reason="current_affairs",
        )


class TestExamPrepFallbackIntegration:
    def test_exam_prep_used_when_official_weak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_ALLOW_EXAM_PREP_FALLBACK", "true")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true")
        _reset_settings()

        exam_prep_only = [
            WebSearchItem(
                title="Monthly economy current affairs summary for SSC",
                url="https://adda247.com/economy-ca",
                snippet=(
                    "Adda247 summary of recent RBI repo rate and inflation "
                    "updates for exam prep."
                ),
                source="adda247.com",
                published_at="2026-05-15",
                score=0.75,
            )
        ]
        tool = WebSearchTool(provider=FakeWebSearchProvider(exam_prep_only))
        result = tool.search(
            WebSearchRequest(
                request_id="exam-prep-1",
                query="Monthly economy current affairs summary for SSC",
                web_search_reason="current_economy",
            )
        )
        assert result.attempt_used == "exam_prep_fallback"
        assert result.used is True
        assert "supporting_only" in result.context_text
        assert "exam-prep/supporting" in result.context_text

    def test_exam_update_does_not_trust_exam_prep_alone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
        monkeypatch.setenv("WEB_SEARCH_ALLOW_EXAM_PREP_FALLBACK", "true")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        monkeypatch.setenv("WEB_SEARCH_REQUIRE_OFFICIAL_FOR_EXAM_UPDATES", "true")
        _reset_settings()

        exam_prep_only = [
            WebSearchItem(
                title="UPSC admit card update",
                url="https://testbook.com/upsc-admit",
                snippet="Testbook article about UPSC admit card release date and download steps.",
                source="testbook.com",
                score=0.8,
            )
        ]
        tool = WebSearchTool(provider=FakeWebSearchProvider(exam_prep_only))
        result = tool.search(
            WebSearchRequest(
                request_id="exam-prep-2",
                query="Latest UPSC admit card download",
                web_search_reason="latest_exam_update",
            )
        )
        assert result.weak_context is True
        assert "limited" in result.context_text
        assert "Content:" not in result.context_text
        assert "exam_prep_fallback" not in (result.attempt_used or "")
