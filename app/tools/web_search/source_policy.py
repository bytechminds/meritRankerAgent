"""Resolve source packs and freshness filters from classifier signals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from tools.web_search.scope_policy import SourceScopePolicy, detect_source_scope_policy
from tools.web_search.source_pack_loader import (
    SourcePackCatalog,
    get_source_pack_catalog,
)

_MAX_INCLUDE_DOMAINS = 25

_SOURCE_NEED_TO_INDIA_PACK: dict[str, str] = {
    "official_exam_update": "exam_updates_india",
    "economy": "economy_india",
    "government_schemes": "government_schemes_india",
    "sports": "sports_current_affairs",
    "polity": "polity_india",
    "geography": "geography_india",
    "science": "science_current",
    "environment": "environment_india",
    "practice_current_affairs": "current_affairs_india",
    "general": "current_affairs_india",
}

_WORLD_PACK = "international_current_affairs"
_DEFAULT_INDIA_PACK = "current_affairs_india"


@dataclass(frozen=True)
class WebSourcePolicy:
    """Resolved source policy for one web search call."""

    source_pack_name: str
    scope: str
    india_weight: int
    world_weight: int
    source_need: str
    exam_context: str | None
    topic: str
    trusted_domains: tuple[str, ...]
    reputed_domains: tuple[str, ...]
    exam_prep_domains: tuple[str, ...]
    blocked_domains: tuple[str, ...]
    global_blocked: tuple[str, ...]
    start_date: str | None
    end_date: str | None
    time_range: str | None
    source_strictness: str
    freshness_label: str


class WebSourcePolicyResolver:
    """Select source pack and freshness window from query/classifier signals."""

    def __init__(self, catalog: SourcePackCatalog | None = None) -> None:
        self._catalog = catalog or get_source_pack_catalog()

    def resolve(
        self,
        *,
        query: str,
        web_search_query: str | None,
        subject: str,
        topic: str | None,
        retrieval_tags: list[str] | None,
        web_search_reason: str | None,
        source_strictness: str,
        default_recent_days: int,
    ) -> WebSourcePolicy:
        combined = " ".join(
            part
            for part in (
                query,
                web_search_query or "",
                topic or "",
                subject or "",
                " ".join(retrieval_tags or []),
            )
            if part
        ).lower()

        scope_policy = detect_source_scope_policy(
            query=query,
            web_search_query=web_search_query,
            web_search_reason=web_search_reason,
        )
        india_pack_name, world_pack_name, pack_label = _select_packs(scope_policy)
        india_pack = self._catalog.get_pack(india_pack_name)
        world_pack = self._catalog.get_pack(world_pack_name)

        trusted = _merge_weighted_domains(
            india_pack.trusted_domains,
            world_pack.trusted_domains,
            scope_policy.india_weight,
            scope_policy.world_weight,
            _MAX_INCLUDE_DOMAINS,
        )
        reputed = _merge_weighted_domains(
            india_pack.reputed_domains,
            world_pack.reputed_domains,
            scope_policy.india_weight,
            scope_policy.world_weight,
            _MAX_INCLUDE_DOMAINS,
        )
        exam_prep = _merge_weighted_domains(
            india_pack.exam_prep_domains,
            world_pack.exam_prep_domains,
            scope_policy.india_weight,
            scope_policy.world_weight,
            _MAX_INCLUDE_DOMAINS,
        )

        if scope_policy.india_weight >= scope_policy.world_weight:
            topic_value = india_pack.topic
        else:
            topic_value = world_pack.topic
        start_date, end_date, time_range, freshness_label = _parse_freshness(
            combined,
            default_recent_days=default_recent_days,
        )
        blocked = tuple(
            dict.fromkeys(
                [
                    *self._catalog.global_blocked,
                    *india_pack.blocked_domains,
                    *world_pack.blocked_domains,
                ]
            ).keys()
        )
        return WebSourcePolicy(
            source_pack_name=pack_label,
            scope=scope_policy.scope,
            india_weight=scope_policy.india_weight,
            world_weight=scope_policy.world_weight,
            source_need=scope_policy.source_need,
            exam_context=scope_policy.exam_context,
            topic=topic_value,
            trusted_domains=trusted,
            reputed_domains=reputed,
            exam_prep_domains=exam_prep,
            blocked_domains=india_pack.blocked_domains,
            global_blocked=blocked,
            start_date=start_date,
            end_date=end_date,
            time_range=time_range,
            source_strictness=source_strictness,
            freshness_label=freshness_label,
        )


def _select_packs(
    scope_policy: SourceScopePolicy,
) -> tuple[str, str, str]:
    """Return india pack name, world pack name, and descriptive policy label."""
    if scope_policy.source_need == "official_exam_update":
        return "exam_updates_india", _WORLD_PACK, "exam_updates_india"

    india_pack = _SOURCE_NEED_TO_INDIA_PACK.get(
        scope_policy.source_need,
        _DEFAULT_INDIA_PACK,
    )

    if scope_policy.scope == "world":
        if scope_policy.source_need == "economy":
            return _WORLD_PACK, _WORLD_PACK, "international_current_affairs"
        if scope_policy.source_need == "practice_current_affairs":
            return _WORLD_PACK, _WORLD_PACK, "international_current_affairs"
        return india_pack, _WORLD_PACK, f"{india_pack}_world"

    if scope_policy.scope == "india":
        return india_pack, _WORLD_PACK, india_pack

    if scope_policy.source_need == "practice_current_affairs":
        return _DEFAULT_INDIA_PACK, _WORLD_PACK, "current_affairs_mixed"

    return india_pack, _WORLD_PACK, f"{india_pack}_mixed"


def _merge_weighted_domains(
    india_domains: tuple[str, ...],
    world_domains: tuple[str, ...],
    india_weight: int,
    world_weight: int,
    max_domains: int,
) -> tuple[str, ...]:
    if india_weight == 100 or not world_domains:
        return india_domains[:max_domains]
    if world_weight == 100 or not india_domains:
        return world_domains[:max_domains]

    total = max(india_weight + world_weight, 1)
    india_slots = max(1, round(max_domains * india_weight / total)) if india_domains else 0
    world_slots = max(1, max_domains - india_slots) if world_domains else 0
    merged = list(
        dict.fromkeys(
            [
                *india_domains[:india_slots],
                *world_domains[:world_slots],
            ]
        ).keys()
    )
    return tuple(merged[:max_domains])


def _parse_freshness(
    text: str,
    *,
    default_recent_days: int,
) -> tuple[str | None, str | None, str | None, str]:
    today = date.today()
    lower = text.lower()

    if "today" in lower:
        iso = today.isoformat()
        return iso, iso, None, today.strftime("%B %Y")

    if "yesterday" in lower:
        day = today - timedelta(days=1)
        iso = day.isoformat()
        return iso, iso, None, day.strftime("%B %Y")

    if "last week" in lower or "past week" in lower:
        start = today - timedelta(days=7)
        return start.isoformat(), today.isoformat(), None, today.strftime("%B %Y")

    month_match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+(20\d{2})\b",
        lower,
    )
    if month_match:
        month_name, year_str = month_match.groups()
        month_num = _month_number(month_name)
        start = date(int(year_str), month_num, 1)
        if month_num == 12:
            end = date(int(year_str), 12, 31)
        else:
            end = date(int(year_str), month_num + 1, 1) - timedelta(days=1)
        return start.isoformat(), end.isoformat(), None, start.strftime("%B %Y")

    year_match = re.search(r"\b(20\d{2})\b", lower)
    if year_match and any(token in lower for token in ("year", "annual", "yearly")):
        year = int(year_match.group(1))
        return f"{year}-01-01", f"{year}-12-31", None, str(year)

    if any(token in lower for token in ("recent", "latest", "current", "this month", "this year")):
        start = today - timedelta(days=default_recent_days)
        return start.isoformat(), today.isoformat(), None, today.strftime("%B %Y")

    return None, None, None, today.strftime("%B %Y")


def _month_number(name: str) -> int:
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    return months[name]
