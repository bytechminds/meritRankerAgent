"""Tavily Search API provider adapter."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from tools.web_search.models import (
    SearchAttemptKind,
    WebSearchItem,
    WebSearchProviderRequest,
    WebSearchProviderResult,
)
from tools.web_search.providers.base import WebSearchProvider

logger = logging.getLogger(__name__)

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilyWebSearchProvider(WebSearchProvider):
    """Maps provider-neutral requests to Tavily search parameters."""

    def __init__(self, *, api_key: str, provider_name: str = "tavily") -> None:
        self._api_key = api_key
        self._provider_name = provider_name

    def search(self, request: WebSearchProviderRequest) -> WebSearchProviderResult:
        attempt = str(request.metadata.get("attempt") or "authoritative")
        attempt_kind: SearchAttemptKind
        if attempt in {"authoritative", "authoritative_plus_reputed", "generic_fallback"}:
            attempt_kind = attempt  # type: ignore[assignment]
        else:
            attempt_kind = "authoritative"

        payload: dict[str, Any] = {
            "query": request.query,
            "max_results": request.max_results,
            "search_depth": request.search_depth,
            "include_answer": False,
            "include_images": False,
            "include_raw_content": request.include_raw_content,
        }
        if request.topic:
            payload["topic"] = request.topic
        if request.include_domains:
            payload["include_domains"] = request.include_domains
        if request.exclude_domains:
            payload["exclude_domains"] = request.exclude_domains
        if request.start_date:
            payload["start_date"] = request.start_date
        if request.end_date:
            payload["end_date"] = request.end_date
        if request.time_range:
            payload["time_range"] = request.time_range

        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            _TAVILY_SEARCH_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=request.timeout_seconds,
            ) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.warning("web_search_provider  http_error=%s", exc.code)
            raise RuntimeError("provider_http_error") from exc
        except urllib.error.URLError as exc:
            logger.warning("web_search_provider  url_error=%s", type(exc).__name__)
            raise RuntimeError("provider_url_error") from exc
        except TimeoutError as exc:
            logger.warning("web_search_provider  timeout=true")
            raise RuntimeError("provider_timeout") from exc

        items = _parse_results(raw)
        return WebSearchProviderResult(
            items=items,
            provider=self._provider_name,
            attempt=attempt_kind,
        )


def _parse_results(raw: dict[str, Any]) -> list[WebSearchItem]:
    results = raw.get("results") or []
    items: list[WebSearchItem] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "").strip()
        title = str(entry.get("title") or "").strip()
        snippet = str(entry.get("content") or entry.get("snippet") or "").strip()
        source = str(entry.get("source") or _domain_from_url(url)).strip()
        published_at = entry.get("published_date") or entry.get("published_at")
        score_raw = entry.get("score")
        score = float(score_raw) if score_raw is not None else None
        if not url and not snippet:
            continue
        items.append(
            WebSearchItem(
                title=title,
                url=url,
                snippet=snippet,
                source=source,
                published_at=str(published_at).strip() if published_at else None,
                score=score,
            )
        )
    return items


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except ValueError:
        return ""
