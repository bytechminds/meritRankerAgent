"""Web search tool — provider-agnostic orchestration entry point."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

from config import Settings, get_settings
from tools.web_search.formatter import WebContextFormatter
from tools.web_search.models import (
    FreshEvidenceBundle,
    FreshnessWindow,
    SearchAttemptKind,
    WebSearchItem,
    WebSearchRequest,
    WebSearchResult,
)
from tools.web_search.official_only_guard import (
    is_exam_prep_suitable_query,
    is_official_only_query,
)
from tools.web_search.providers.base import WebSearchProvider
from tools.web_search.providers.fake_provider import FakeWebSearchProvider
from tools.web_search.providers.tavily_provider import TavilyWebSearchProvider
from tools.web_search.query_builder import SearchAttemptPlan, WebSearchQueryBuilder
from tools.web_search.reranker import (
    WebSearchReranker,
    WebSearchRerankInput,
    tag_items_with_source_quality,
)
from tools.web_search.scope_policy import SourceScopePolicy
from tools.web_search.search_query_builder import build_scope_aware_search_query
from tools.web_search.source_policy import WebSourcePolicyResolver

logger = logging.getLogger(__name__)

_CURRENT_AFFAIRS_REASONS = frozenset(
    {"current_affairs", "current_economy", "current_event"}
)


def credentials_ready(settings: Settings | None = None) -> bool:
    """Return True when configured provider credentials are available."""
    settings = settings or get_settings()
    if not settings.web_search_enabled:
        return False
    if settings.web_search_provider == "tavily":
        return bool(settings.tavily_api_key.strip())
    return False


class WebSearchTool:
    """Search the web with source-policy-driven progressive attempts."""

    def __init__(
        self,
        *,
        provider: WebSearchProvider | None = None,
        settings: Settings | None = None,
        policy_resolver: WebSourcePolicyResolver | None = None,
        reranker: WebSearchReranker | None = None,
    ) -> None:
        self._provider_override = provider
        self._settings = settings
        self._policy_resolver = policy_resolver or WebSourcePolicyResolver()
        self._reranker = reranker or WebSearchReranker()
        self._query_builder = WebSearchQueryBuilder()

    def search(
        self,
        request: WebSearchRequest,
        *,
        on_retry_sources: Callable[[], None] | None = None,
    ) -> WebSearchResult:
        started_at = time.monotonic()
        settings = self._settings or get_settings()
        provider_name = settings.web_search_provider

        if not settings.web_search_enabled:
            return WebSearchResult(
                used=False,
                provider=provider_name,
                query=(request.web_search_query or request.query).strip(),
                reason="web_search_disabled",
            )

        if self._provider_override is None and not credentials_ready(settings):
            return WebSearchResult(
                used=False,
                provider=provider_name,
                query=(request.web_search_query or request.query).strip(),
                reason="missing_credentials",
                error_kind="missing_credentials",
            )

        policy = self._policy_resolver.resolve(
            query=request.query,
            web_search_query=request.web_search_query,
            subject=request.subject,
            topic=request.topic,
            retrieval_tags=request.retrieval_tags,
            web_search_reason=request.web_search_reason,
            source_strictness=settings.web_search_source_strictness,
            default_recent_days=settings.web_search_default_recent_days,
        )
        scope_policy = SourceScopePolicy(
            scope=policy.scope,  # type: ignore[arg-type]
            india_weight=policy.india_weight,
            world_weight=policy.world_weight,
            source_need=policy.source_need,  # type: ignore[arg-type]
            exam_context=policy.exam_context,
            explicit_scope=False,
            official_exam_lifecycle=policy.source_need == "official_exam_update",
        )
        search_query = build_scope_aware_search_query(
            request.query,
            request.web_search_query,
            scope_policy,
        )

        allow_generic = settings.web_search_allow_generic_fallback
        if (
            settings.web_search_require_trusted_for_current_affairs
            and (request.web_search_reason or "") in _CURRENT_AFFAIRS_REASONS
        ):
            allow_generic = False

        official_only = is_official_only_query(
            request.query,
            web_search_query=request.web_search_query,
            web_search_reason=request.web_search_reason,
        )
        if settings.web_search_require_official_for_exam_updates and official_only:
            allow_generic = False

        exam_prep_suitable = is_exam_prep_suitable_query(
            request.query,
            web_search_query=request.web_search_query,
            web_search_reason=request.web_search_reason,
        )
        allow_exam_prep = settings.web_search_allow_exam_prep_fallback

        attempts = self._query_builder.plan_attempts(
            policy,
            allow_generic_fallback=allow_generic,
            allow_exam_prep_fallback=allow_exam_prep,
            exam_prep_suitable=exam_prep_suitable,
            official_only=official_only,
        )

        best_items = []
        best_attempt: SearchAttemptKind | None = None
        best_rerank = None
        retry_status_sent = False
        attempt_count = 0
        candidate_count = 0

        for attempt_idx, attempt in enumerate(attempts):
            if (
                attempt_idx == 1
                and on_retry_sources is not None
                and not retry_status_sent
            ):
                on_retry_sources()
                retry_status_sent = True
            provider_result = self._run_attempt(
                request,
                settings=settings,
                policy=policy,
                attempt=attempt,
                search_query=search_query,
            )
            attempt_count += 1
            candidate_count += len(provider_result.items)
            tagged = tag_items_with_source_quality(
                provider_result.items,
                policy=policy,
                attempt_kind=attempt.kind,
            )
            rerank_result = self._reranker.rerank(
                tagged,
                WebSearchRerankInput(
                    request_id=request.request_id,
                    query=request.query,
                    web_search_query=search_query,
                    topic=request.topic,
                    retrieval_tags=request.retrieval_tags,
                    web_search_reason=request.web_search_reason,
                    source_pack_name=policy.source_pack_name,
                    attempt_used=attempt.kind,
                    official_required=official_only
                    and settings.web_search_require_official_for_exam_updates,
                    exam_prep_suitable=exam_prep_suitable,
                    max_selected_results=(
                        request.required_evidence_count
                        if request.requires_fresh_evidence
                        and request.required_evidence_count > 0
                        else None
                    ),
                ),
                policy=policy,
                settings=settings,
            )
            if rerank_result.selected and not rerank_result.weak_context:
                best_items = rerank_result.selected
                best_attempt = attempt.kind
                best_rerank = rerank_result
                break
            if rerank_result.selected and not best_items:
                best_items = rerank_result.selected
                best_attempt = attempt.kind
                best_rerank = rerank_result

        if best_rerank is None:
            best_rerank = WebSearchReranker().rerank(
                [],
                WebSearchRerankInput(
                    request_id=request.request_id,
                    query=request.query,
                    source_pack_name=policy.source_pack_name,
                ),
                settings=settings,
            )

        weak_context = best_rerank.weak_context or not best_items
        fresh_evidence: FreshEvidenceBundle | None = None
        if request.requires_fresh_evidence:
            fresh_evidence = _build_fresh_evidence_bundle(
                policy=policy,
                search_query=search_query,
                items=best_items,
            )
            best_items = list(fresh_evidence.items)
            if len(best_items) < request.required_evidence_count:
                weak_context = True
        context_strength = best_rerank.context_strength
        evidence_context_chars = 0
        if weak_context:
            context_text = WebContextFormatter.format_weak_safe_note(
                max_chars=settings.web_search_max_context_chars,
            )
            safe_note_chars = len(context_text)
            logger.info(
                "web_search_weak_context  request_id=%s  evidence_context_chars=0  "
                "safe_note_chars=%d  final_context_chars=%d  weak_context_discarded=true",
                request.request_id,
                safe_note_chars,
                safe_note_chars,
            )
        else:
            context_text = WebContextFormatter.format(
                best_items,
                source_pack_name=policy.source_pack_name,
                attempt_label=best_attempt or "authoritative",
                freshness_label=policy.freshness_label,
                reason=request.web_search_reason or "freshness_required",
                search_query=search_query,
                max_chars=settings.web_search_max_context_chars,
                weak_context=False,
                context_strength=context_strength,
            )
            evidence_context_chars = len(context_text)
            safe_note_chars = 0

        logger.info(
            "web_search_result  request_id=%s  provider=%s  attempts=%d  "
            "candidate_count=%d  used=%s  result_count=%d  "
            "context_chars=%d  evidence_context_chars=%d  safe_note_chars=%d  "
            "source_pack=%s  scope=%s  source_need=%s  attempt=%s  weak_context=%s  "
            "duration_ms=%d",
            request.request_id,
            provider_name,
            attempt_count,
            candidate_count,
            bool(best_items) and not weak_context,
            len(best_items),
            len(context_text),
            evidence_context_chars if not weak_context else 0,
            len(context_text) if weak_context else 0,
            policy.source_pack_name,
            policy.scope,
            policy.source_need,
            best_attempt or "",
            weak_context,
            int((time.monotonic() - started_at) * 1000),
        )

        official_count = sum(
            item.source_quality == "trusted" for item in best_items
        )
        reputable_count = sum(
            item.source_quality == "reputed" for item in best_items
        )
        return WebSearchResult(
            used=bool(best_items) and not weak_context,
            provider=provider_name,
            query=search_query,
            items=best_items,
            context_text=context_text,
            reason="weak_web_context" if weak_context else "web_context_selected",
            weak_context=weak_context,
            source_pack_name=policy.source_pack_name,
            attempt_used=best_attempt,
            freshness_label=policy.freshness_label,
            executed=True,
            attempt_count=attempt_count,
            candidate_count=candidate_count,
            official_count=official_count,
            reputable_count=reputable_count,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            fresh_evidence=(fresh_evidence if not weak_context else None),
        )

    def _run_attempt(
        self,
        request: WebSearchRequest,
        *,
        settings: Settings,
        policy,
        attempt: SearchAttemptPlan,
        search_query: str,
    ):
        provider_request = self._query_builder.build_provider_request(
            search_query=search_query,
            policy=policy,
            attempt=attempt,
            max_results=self._max_results_for_attempt(attempt, request, settings),
            search_depth=settings.web_search_search_depth,
            timeout_seconds=request.timeout_seconds,
        )
        provider = self._provider_override or self._build_provider(settings)
        return provider.search(provider_request)

    @staticmethod
    def _max_results_for_attempt(
        attempt: SearchAttemptPlan,
        request: WebSearchRequest,
        settings: Settings,
    ) -> int:
        if attempt.kind == "exam_prep_fallback":
            return min(
                settings.web_search_max_results,
                settings.web_search_exam_prep_max_selected_results,
            )
        if request.requires_fresh_evidence:
            return min(
                max(settings.web_search_max_results, request.required_evidence_count),
                20,
            )
        return settings.web_search_max_results

    @staticmethod
    def _build_provider(settings: Settings) -> WebSearchProvider:
        if settings.web_search_provider == "tavily":
            return TavilyWebSearchProvider(
                api_key=settings.tavily_api_key,
                provider_name=settings.web_search_provider,
            )
        raise RuntimeError("unsupported_provider")


def build_fake_web_search_tool(items=None) -> WebSearchTool:
    """Test helper — inject a fake provider."""
    from tools.web_search.models import WebSearchItem  # noqa: PLC0415

    default_items = items
    if default_items is None:
        default_items = [
            WebSearchItem(
                title="Latest RBI repo rate update",
                url="https://rbi.org.in/policy-update",
                snippet="The RBI maintained the repo rate in the latest monetary policy review.",
                source="rbi.org.in",
                published_at="2026-05-15",
                score=0.9,
            ),
            WebSearchItem(
                title="Latest government current affairs update",
                url="https://pib.gov.in/current-affairs",
                snippet="Press Information Bureau release on recent government policy updates.",
                source="pib.gov.in",
                published_at="2026-05-15",
                score=0.88,
            ),
        ]
    return WebSearchTool(provider=FakeWebSearchProvider(default_items))


def _build_fresh_evidence_bundle(
    *,
    policy,
    search_query: str,
    items: list[WebSearchItem],
) -> FreshEvidenceBundle:
    """Keep only compact, usable, in-window evidence for Practice grounding."""
    selected: list[WebSearchItem] = []
    seen_urls: set[str] = set()
    for item in items:
        url = _normalized_url(item.url)
        if not url or len(item.snippet.strip()) < 20:
            continue
        published_date = _published_date(item.published_at)
        if published_date is not None:
            if policy.start_date and published_date < policy.start_date:
                continue
            if policy.end_date and published_date > policy.end_date:
                continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        selected.append(
            item.model_copy(
                update={
                    "title": item.title[:256],
                    "url": url,
                    "snippet": item.snippet[:600],
                    "source": item.source[:128],
                }
            )
        )

    return FreshEvidenceBundle(
        requested_window=FreshnessWindow(
            start_date=policy.start_date,
            end_date=policy.end_date,
            source=policy.freshness_source,
            label=policy.freshness_label,
        ),
        retrieved_at=datetime.now(UTC).isoformat(),
        search_query=search_query,
        items=selected,
    )


def _normalized_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.query,
            "",
        )
    )


def _published_date(value: str | None) -> str | None:
    if not value:
        return None
    match = value.strip()[:10]
    try:
        return datetime.fromisoformat(match).date().isoformat()
    except ValueError:
        return None
