"""
tests/test_web_search_rerank.py
---------------------------------
Deterministic web search rerank and formatter tests.
"""

from __future__ import annotations

import pytest

import config as cfg_module
from tools.web_search.formatter import format_selected_web_context
from tools.web_search.models import WebSearchItem
from tools.web_search.reranker import (
    WebSearchReranker,
    WebSearchRerankInput,
    tag_items_with_source_quality,
)
from tools.web_search.source_policy import WebSourcePolicyResolver


def _reset_settings() -> None:
    cfg_module._settings = None


@pytest.fixture(autouse=True)
def _clean() -> None:
    _reset_settings()
    yield
    _reset_settings()


def _item(**overrides) -> WebSearchItem:
    base = {
        "title": "Generic headline",
        "url": "https://example.com/a",
        "snippet": "Unrelated content without query overlap at all here.",
        "source": "example.com",
        "score": 0.4,
    }
    base.update(overrides)
    return WebSearchItem(**base)


def _policy():
    return WebSourcePolicyResolver().resolve(
        query="latest RBI repo rate",
        web_search_query="latest RBI repo rate",
        subject="general",
        topic=None,
        retrieval_tags=[],
        web_search_reason="current_economy",
        source_strictness="authoritative_first",
        default_recent_days=30,
    )


class TestWebSearchReranker:
    def test_title_content_overlap_boosts_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.20")
        _reset_settings()
        reranker = WebSearchReranker()
        items = [
            _item(title="Weather today", snippet="Unrelated weather forecast details here."),
            _item(
                title="Latest RBI repo rate decision",
                url="https://example.com/rbi",
                snippet="The RBI maintained the repo rate in the latest monetary policy review.",
                score=0.5,
            ),
        ]
        result = reranker.rerank(
            items,
            WebSearchRerankInput(
                request_id="r1",
                query="latest RBI repo rate",
                web_search_reason="current_economy",
            ),
        )
        assert len(result.selected) >= 1
        assert "rbi" in result.selected[0].title.lower()

    def test_retrieval_tag_overlap_boosts_selection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.25")
        _reset_settings()
        reranker = WebSearchReranker()
        items = [
            _item(
                title="Policy update",
                snippet="General policy update without overlap.",
            ),
            _item(
                title="Repo rate update",
                url="https://news.example.com/2",
                snippet="Repo rate unchanged; monetary policy stance remains accommodative.",
                source="news.example.com",
            ),
        ]
        result = reranker.rerank(
            items,
            WebSearchRerankInput(
                request_id="r2",
                query="monetary policy repo rate",
                retrieval_tags=["repo_rate", "monetary_policy"],
            ),
        )
        assert result.selected
        assert "repo" in result.selected[0].title.lower()

    def test_duplicate_urls_penalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.20")
        monkeypatch.setenv("WEB_SEARCH_MAX_SELECTED_RESULTS", "3")
        _reset_settings()
        reranker = WebSearchReranker()
        items = [
            _item(
                title="Latest RBI repo rate",
                url="https://example.com/same",
                snippet="RBI repo rate maintained in latest policy review announcement.",
            ),
            _item(
                title="Duplicate RBI repo rate",
                url="https://example.com/same",
                snippet="RBI repo rate maintained again in duplicate source content.",
            ),
        ]
        result = reranker.rerank(
            items,
            WebSearchRerankInput(request_id="r3", query="latest RBI repo rate"),
        )
        assert len(result.selected) == 1

    def test_low_score_irrelevant_not_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.60")
        _reset_settings()
        reranker = WebSearchReranker()
        items = [
            _item(
                title="Cricket score update",
                snippet="Unrelated sports headline with no overlap to economics query.",
            )
        ]
        result = reranker.rerank(
            items,
            WebSearchRerankInput(request_id="r4", query="latest RBI repo rate"),
        )
        assert result.selected == []
        assert result.weak_context is True

    def test_title_only_result_not_selected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()
        reranker = WebSearchReranker()
        items = [
            WebSearchItem(
                title="Latest RBI repo rate",
                url="https://example.com/rbi",
                snippet="short",
                source="example.com",
            )
        ]
        result = reranker.rerank(
            items,
            WebSearchRerankInput(request_id="r5", query="latest RBI repo rate"),
        )
        assert result.selected == []

    def test_max_selected_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_MAX_SELECTED_RESULTS", "2")
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()
        reranker = WebSearchReranker()
        items = [
            _item(
                title=f"RBI repo rate update {idx}",
                url=f"https://example.com/{idx}",
                snippet=f"RBI repo rate policy update number {idx} with enough content.",
            )
            for idx in range(5)
        ]
        result = reranker.rerank(
            items,
            WebSearchRerankInput(request_id="r6", query="RBI repo rate"),
        )
        assert len(result.selected) <= 2

    def test_trusted_outranks_reputed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()
        policy = _policy()
        items = tag_items_with_source_quality(
            [
                _item(
                    title="RBI repo rate reputed news",
                    url="https://thehindu.com/rbi",
                    snippet="RBI repo rate maintained in latest monetary policy review update.",
                    source="thehindu.com",
                    score=0.7,
                ),
                _item(
                    title="RBI repo rate official",
                    url="https://rbi.org.in/rate",
                    snippet="RBI repo rate maintained in latest monetary policy review update.",
                    source="rbi.org.in",
                    score=0.6,
                ),
            ],
            policy=policy,
            attempt_kind="authoritative_plus_reputed",
        )
        result = WebSearchReranker().rerank(
            items,
            WebSearchRerankInput(
                request_id="rq1",
                query="latest RBI repo rate",
                attempt_used="authoritative_plus_reputed",
            ),
        )
        assert result.selected[0].source_quality == "trusted"

    def test_reputed_outranks_exam_prep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()
        policy = _policy()
        items = tag_items_with_source_quality(
            [
                _item(
                    title="RBI repo rate exam prep summary",
                    url="https://adda247.com/rbi",
                    snippet="RBI repo rate maintained summary for SSC exam preparation students.",
                    source="adda247.com",
                    score=0.9,
                ),
                _item(
                    title="RBI repo rate news",
                    url="https://livemint.com/rbi",
                    snippet="RBI repo rate maintained in latest monetary policy review update.",
                    source="livemint.com",
                    score=0.5,
                ),
            ],
            policy=policy,
            attempt_kind="authoritative_plus_reputed",
        )
        result = WebSearchReranker().rerank(
            items,
            WebSearchRerankInput(
                request_id="rq2",
                query="latest RBI repo rate",
                attempt_used="authoritative_plus_reputed",
            ),
        )
        assert result.selected[0].source_quality == "reputed"

    def test_exam_prep_only_supporting_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()
        policy = _policy()
        items = tag_items_with_source_quality(
            [
                _item(
                    title="RBI repo rate exam prep",
                    url="https://adda247.com/rbi",
                    snippet=(
                        "RBI repo rate maintained summary for exam preparation "
                        "with enough text."
                    ),
                    source="adda247.com",
                    score=0.8,
                ),
            ],
            policy=policy,
            attempt_kind="exam_prep_fallback",
        )
        result = WebSearchReranker().rerank(
            items,
            WebSearchRerankInput(
                request_id="rq3",
                query="latest RBI repo rate",
                web_search_reason="current_economy",
                attempt_used="exam_prep_fallback",
                exam_prep_suitable=True,
            ),
        )
        assert result.context_strength == "supporting_only"
        assert result.exam_prep_selected_count >= 1

    def test_official_required_exam_prep_only_weak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()
        policy = WebSourcePolicyResolver().resolve(
            query="Latest UPSC admit card",
            web_search_query=None,
            subject="general",
            topic=None,
            retrieval_tags=[],
            web_search_reason="latest_exam_update",
            source_strictness="authoritative_first",
            default_recent_days=30,
        )
        items = tag_items_with_source_quality(
            [
                _item(
                    title="UPSC admit card testbook",
                    url="https://testbook.com/admit",
                    snippet="Testbook article about UPSC admit card release with enough content.",
                    source="testbook.com",
                    score=0.85,
                ),
            ],
            policy=policy,
            attempt_kind="exam_prep_fallback",
        )
        result = WebSearchReranker().rerank(
            items,
            WebSearchRerankInput(
                request_id="rq4",
                query="Latest UPSC admit card",
                web_search_reason="latest_exam_update",
                attempt_used="exam_prep_fallback",
                official_required=True,
            ),
        )
        assert result.weak_context is True

    def test_youtube_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.10")
        _reset_settings()
        policy = _policy()
        items = tag_items_with_source_quality(
            [
                WebSearchItem(
                    title="RBI repo rate video",
                    url="https://youtube.com/watch",
                    snippet="RBI repo rate explained in video with enough snippet text here.",
                    source="youtube.com",
                    score=0.99,
                ),
            ],
            policy=policy,
            attempt_kind="generic_fallback",
        )
        result = WebSearchReranker().rerank(
            items,
            WebSearchRerankInput(request_id="rq5", query="latest RBI repo rate"),
        )
        assert result.selected == []


class TestWebSearchFormatter:
    def test_uses_content_and_url_fields(self) -> None:
        text = format_selected_web_context(
            [
                WebSearchItem(
                    title="Headline",
                    url="https://example.com/x",
                    snippet="Compact Tavily content snippet for generator grounding.",
                    source="example.com",
                    published_at="2026-05-01",
                    score=0.8,
                )
            ],
            reason="current_affairs",
            search_query="latest current affairs",
            max_chars=1200,
        )
        assert "[Web Context]" in text
        assert "Content:" in text
        assert "URL:" in text
        assert "score" not in text.lower()
        assert "{" not in text

    def test_truncates_max_chars(self) -> None:
        text = format_selected_web_context(
            [
                WebSearchItem(
                    title="Headline",
                    url="https://example.com/x",
                    snippet="x" * 800,
                    source="example.com",
                )
            ],
            reason="current_affairs",
            search_query="query",
            max_chars=300,
        )
        assert len(text) <= 300

    def test_skips_weak_title_only_block(self) -> None:
        text = format_selected_web_context(
            [
                WebSearchItem(
                    title="Only title",
                    url="https://example.com/x",
                    snippet="tiny",
                    source="example.com",
                )
            ],
            reason="current_affairs",
            search_query="query",
            max_chars=500,
        )
        assert "Content:" not in text

    def test_supporting_only_instruction_for_exam_prep(self) -> None:
        text = format_selected_web_context(
            [
                WebSearchItem(
                    title="Economy CA summary",
                    url="https://adda247.com/ca",
                    snippet=(
                        "Exam prep summary of recent economy current affairs "
                        "for SSC preparation."
                    ),
                    source="adda247.com",
                )
            ],
            reason="current_affairs",
            search_query="monthly current affairs",
            max_chars=1200,
            context_strength="supporting_only",
        )
        assert "Source strength: supporting_only" in text
        assert "exam-prep/supporting" in text
        assert "official dates" in text.lower()

    def test_authoritative_context_no_exam_prep_warning(self) -> None:
        text = format_selected_web_context(
            [
                WebSearchItem(
                    title="PIB release",
                    url="https://pib.gov.in/x",
                    snippet="Official press release with enough content for grounding purposes.",
                    source="pib.gov.in",
                )
            ],
            reason="current_affairs",
            search_query="latest news",
            max_chars=1200,
            context_strength="authoritative",
        )
        assert "exam-prep/supporting" not in text


class TestEvidenceArticleDeduplication:
    """One article must not fill several evidence slots.

    Live PIB results returned the same press release under two URLs differing only by
    `&reg=3&lang=1`, so a three-item bundle carried only two distinct facts and two
    slots were handed identical evidence.
    """

    @staticmethod
    def _item(url: str, title: str, published_at: str) -> WebSearchItem:
        return WebSearchItem(
            title=title,
            url=url,
            snippet=(
                "India current affairs August 2026: government announcement "
                "covering the current affairs of India in August 2026."
            ),
            source="pib.gov.in",
            published_at=published_at,
            score=0.9,
        )

    def test_same_article_under_tracking_variants_is_selected_once(self) -> None:
        base = "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2293440"
        items = [
            self._item(
                f"{base}&reg=3&lang=1",
                "India current affairs August 2026 innovation update - PIB",
                "2026-08-02",
            ),
            self._item(
                base,
                "India current affairs August 2026 innovation update - pib.gov.in",
                "2026-08-02",
            ),
            self._item(
                "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2293020",
                "India current affairs August 2026 self-reliance update",
                "2026-08-01",
            ),
        ]
        result = WebSearchReranker().rerank(
            items,
            WebSearchRerankInput(
                request_id="dedup",
                query="current affairs India August 2026",
                web_search_query="current affairs India August 2026",
                topic="current affairs",
                retrieval_tags=["current affairs", "india"],
                web_search_reason="current_event",
                source_pack_name="current_affairs_india",
                attempt_used="authoritative",
                requires_fresh_evidence=True,
            ),
        )
        urls = [item.url for item in result.selected]
        assert len(result.selected) == 2, urls
        assert sum("PRID=2293440" in url for url in urls) == 1, urls

    def test_distinct_articles_are_both_kept(self) -> None:
        """Dedup keys on the article, so different releases must survive."""
        items = [
            self._item(
                "https://www.pib.gov.in/PressReleasePage.aspx?PRID=1",
                "India current affairs August 2026 first release",
                "2026-08-02",
            ),
            self._item(
                "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2",
                "India current affairs August 2026 second release",
                "2026-08-02",
            ),
        ]
        result = WebSearchReranker().rerank(
            items,
            WebSearchRerankInput(
                request_id="dedup2",
                query="current affairs India August 2026",
                web_search_query="current affairs India August 2026",
                topic="current affairs",
                retrieval_tags=["current affairs", "india"],
                web_search_reason="current_event",
                source_pack_name="current_affairs_india",
                attempt_used="authoritative",
                requires_fresh_evidence=True,
            ),
        )
        assert len({item.title for item in result.selected}) == 2
