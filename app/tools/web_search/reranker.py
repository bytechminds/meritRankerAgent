"""Deterministic reranker with source-quality gate."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

from config import Settings, get_settings
from tools.web_search.models import ContextStrength, SourceQuality, WebSearchItem
from tools.web_search.source_policy import WebSourcePolicy

logger = logging.getLogger(__name__)

_FRESHNESS_REASONS: frozenset[str] = frozenset(
    {
        "current_affairs",
        "current_economy",
        "current_event",
        "latest_exam_update",
        "explicit_latest_request",
        "freshness_required",
    }
)

_QUALITY_BOOST: dict[SourceQuality | None, float] = {
    "trusted": 0.40,
    "reputed": 0.25,
    "exam_prep": 0.12,
    "generic": 0.0,
    None: 0.0,
    "blocked": -1.0,
}


@dataclass(frozen=True)
class WebSearchRerankInput:
    """Signals for deterministic web reranking."""

    request_id: str
    query: str
    web_search_query: str = ""
    topic: str | None = None
    retrieval_tags: list[str] | None = None
    web_search_reason: str | None = None
    source_pack_name: str = ""
    attempt_used: str = ""
    official_required: bool = False
    exam_prep_suitable: bool = False
    max_selected_results: int | None = None


@dataclass(frozen=True)
class WebSearchRerankResult:
    """Selected web items after reranking."""

    selected: list[WebSearchItem]
    weak_context: bool
    top_score: float
    trusted_selected_count: int = 0
    reputed_selected_count: int = 0
    exam_prep_selected_count: int = 0
    generic_selected_count: int = 0
    context_strength: ContextStrength = "weak"
    official_required: bool = False


class WebSearchReranker:
    """Score and select top web results without LLM calls."""

    def rerank(
        self,
        items: list[WebSearchItem],
        rerank_input: WebSearchRerankInput,
        *,
        policy: WebSourcePolicy | None = None,
        settings: Settings | None = None,
    ) -> WebSearchRerankResult:
        settings = settings or get_settings()
        if not items:
            self._log_empty(rerank_input)
            return WebSearchRerankResult(
                selected=[],
                weak_context=True,
                top_score=0.0,
                official_required=rerank_input.official_required,
            )

        query = (rerank_input.web_search_query or rerank_input.query).strip()
        query_tokens = _tokenize(query)
        tag_tokens = _normalize_tags(rerank_input.retrieval_tags or [])
        topic_tokens = _tokenize(rerank_input.topic or "")
        min_score = settings.web_search_rerank_min_score
        max_selected = (
            rerank_input.max_selected_results
            if rerank_input.max_selected_results is not None
            else settings.web_search_max_selected_results
        )
        max_selected = min(max(max_selected, 1), 20)
        if rerank_input.attempt_used == "exam_prep_fallback":
            max_selected = min(
                max_selected,
                settings.web_search_exam_prep_max_selected_results,
            )
        allow_generic = settings.web_search_allow_generic_fallback
        require_trusted = settings.web_search_require_trusted_for_current_affairs
        min_trusted = settings.web_search_min_trusted_results
        is_current = (rerank_input.web_search_reason or "") in _FRESHNESS_REASONS
        attempt_kind = rerank_input.attempt_used

        scored: list[tuple[float, WebSearchItem]] = []
        for item in items:
            if item.source_quality == "blocked":
                continue
            if not _has_usable_content(item):
                continue
            if item.source_quality == "generic" and not allow_generic:
                continue
            if item.source_quality == "exam_prep" and attempt_kind != "exam_prep_fallback":
                continue

            title_overlap = _overlap_score(query_tokens, _tokenize(item.title))
            content_overlap = _overlap_score(query_tokens, _tokenize(item.snippet))
            tag_overlap = _overlap_score(tag_tokens, _tokenize(item.title + " " + item.snippet))
            topic_overlap = _overlap_score(topic_tokens, _tokenize(item.title + " " + item.snippet))
            provider_score = min(max(item.score or 0.0, 0.0), 1.0) * 0.12
            quality_boost = _QUALITY_BOOST.get(item.source_quality, 0.0)
            freshness_boost = (
                0.08
                if (rerank_input.web_search_reason or "") in _FRESHNESS_REASONS
                or item.published_at
                else 0.0
            )

            score = (
                title_overlap * 0.22
                + content_overlap * 0.33
                + tag_overlap * 0.12
                + topic_overlap * 0.08
                + provider_score
                + quality_boost
                + freshness_boost
            )
            scored.append((score, item.model_copy(update={"selected_score": score})))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected: list[WebSearchItem] = []
        selected_urls: set[str] = set()
        for score, item in scored:
            if score < min_score:
                continue
            url_key = item.url.strip().lower()
            if url_key and url_key in selected_urls:
                continue
            selected.append(item)
            if url_key:
                selected_urls.add(url_key)
            if len(selected) >= max_selected:
                break

        top_score = scored[0][0] if scored else 0.0
        trusted_count = sum(1 for item in selected if item.source_quality == "trusted")
        reputed_count = sum(1 for item in selected if item.source_quality == "reputed")
        exam_prep_count = sum(1 for item in selected if item.source_quality == "exam_prep")
        generic_count = sum(1 for item in selected if item.source_quality == "generic")
        official_count = trusted_count + reputed_count

        context_strength = _compute_context_strength(
            trusted_count=trusted_count,
            reputed_count=reputed_count,
            exam_prep_count=exam_prep_count,
            generic_count=generic_count,
        )

        weak_context = not selected or top_score < min_score

        if rerank_input.official_required and official_count < min_trusted:
            weak_context = True

        if is_current and require_trusted and official_count < min_trusted:
            if (
                rerank_input.exam_prep_suitable
                and not rerank_input.official_required
                and exam_prep_count >= 1
                and top_score >= min_score
                and selected
            ):
                weak_context = False
            else:
                weak_context = True

        domains = ",".join(sorted({_domain(item) for item in selected if _domain(item)}))
        logger.info(
            "web_search_rerank  request_id=%s  candidate_count=%d  "
            "selected_count=%d  top_score=%.2f  selected_domains=%s  "
            "trusted_selected_count=%d  reputed_selected_count=%d  "
            "exam_prep_selected_count=%d  generic_selected_count=%d  "
            "weak_context=%s  context_strength=%s  official_required=%s  "
            "source_pack_name=%s  attempt_used=%s",
            rerank_input.request_id,
            len(items),
            len(selected),
            top_score,
            domains,
            trusted_count,
            reputed_count,
            exam_prep_count,
            generic_count,
            weak_context,
            context_strength,
            rerank_input.official_required,
            rerank_input.source_pack_name,
            rerank_input.attempt_used,
        )
        return WebSearchRerankResult(
            selected=selected,
            weak_context=weak_context,
            top_score=top_score,
            trusted_selected_count=trusted_count,
            reputed_selected_count=reputed_count,
            exam_prep_selected_count=exam_prep_count,
            generic_selected_count=generic_count,
            context_strength=context_strength,
            official_required=rerank_input.official_required,
        )

    @staticmethod
    def _log_empty(rerank_input: WebSearchRerankInput) -> None:
        logger.info(
            "web_search_rerank  request_id=%s  candidate_count=0  "
            "selected_count=0  top_score=0.00  selected_domains=  "
            "trusted_selected_count=0  reputed_selected_count=0  "
            "exam_prep_selected_count=0  generic_selected_count=0  "
            "weak_context=true  context_strength=weak  official_required=%s  "
            "source_pack_name=%s  attempt_used=%s",
            rerank_input.request_id,
            rerank_input.official_required,
            rerank_input.source_pack_name,
            rerank_input.attempt_used,
        )


def tag_items_with_source_quality(
    items: list[WebSearchItem],
    *,
    policy: WebSourcePolicy,
    attempt_kind: str,
) -> list[WebSearchItem]:
    """Assign trusted/reputed/exam_prep/generic/blocked quality from policy packs."""
    trusted = {d.lower().removeprefix("www.") for d in policy.trusted_domains}
    reputed = {d.lower().removeprefix("www.") for d in policy.reputed_domains}
    exam_prep = {d.lower().removeprefix("www.") for d in policy.exam_prep_domains}
    blocked = {d.lower().removeprefix("www.") for d in policy.global_blocked}
    tagged: list[WebSearchItem] = []
    for item in items:
        domain = _domain(item)
        if domain in blocked or any(domain.endswith(b) for b in blocked):
            quality: SourceQuality = "blocked"
        elif _domain_in_set(domain, trusted):
            quality = "trusted"
        elif _domain_in_set(domain, reputed):
            quality = "reputed"
        elif attempt_kind == "exam_prep_fallback" and _domain_in_set(domain, exam_prep):
            quality = "exam_prep"
        elif attempt_kind == "generic_fallback":
            quality = "generic"
        else:
            quality = "generic"
        tagged.append(item.model_copy(update={"source_quality": quality}))
    return tagged


def _compute_context_strength(
    *,
    trusted_count: int,
    reputed_count: int,
    exam_prep_count: int,
    generic_count: int,
) -> ContextStrength:
    official = trusted_count + reputed_count
    if official > 0 and exam_prep_count > 0:
        return "mixed"
    if official > 0:
        return "authoritative"
    if exam_prep_count > 0:
        return "supporting_only"
    if generic_count > 0:
        return "weak"
    return "weak"


def _domain_in_set(domain: str, domains: set[str]) -> bool:
    if domain in domains:
        return True
    return any(domain.endswith(entry) or entry in domain for entry in domains)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _normalize_tags(tags: list[str]) -> set[str]:
    return {tag.replace("_", " ").lower() for tag in tags if tag}


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _domain(item: WebSearchItem) -> str:
    source = (item.source or "").strip().lower()
    if source:
        return source.removeprefix("www.")
    try:
        return urlparse(item.url).netloc.lower().removeprefix("www.")
    except ValueError:
        return ""


def _has_usable_content(item: WebSearchItem) -> bool:
    return len(item.snippet.strip()) >= 20
