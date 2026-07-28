"""Build scope-aware provider search queries."""

from __future__ import annotations

import re

from tools.web_search.scope_policy import SourceScopePolicy

_MONTH_YEAR = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+20\d{2}\b",
    re.IGNORECASE,
)


def build_scope_aware_search_query(
    query: str,
    web_search_query: str | None,
    scope_policy: SourceScopePolicy,
) -> str:
    """Compose a provider search query from user text, scope, and exam context."""
    base = (web_search_query or query).strip()
    lower = base.lower()
    parts: list[str] = [base]

    if scope_policy.source_need == "practice_current_affairs":
        if "current affairs" not in lower:
            parts.append("current affairs")
        if scope_policy.exam_context:
            parts.append(f"{scope_policy.exam_context} exam preparation practice questions")
        if scope_policy.scope == "world":
            if not any(token in lower for token in ("international", "global", "world")):
                parts.append("international relations latest updates")
        elif scope_policy.scope == "mixed" and _MONTH_YEAR.search(base) is None:
            parts.append("latest updates")
        elif scope_policy.scope == "india" and "india" not in lower:
            parts.append("India")

    elif scope_policy.source_need == "economy":
        if scope_policy.scope == "india" and "india" not in lower:
            parts.append("India economy")
        elif scope_policy.scope == "world":
            parts.append("global economy latest updates")
        if scope_policy.exam_context:
            parts.append(f"{scope_policy.exam_context} banking awareness")

    elif scope_policy.source_need == "official_exam_update":
        if scope_policy.exam_context:
            parts.append(f"{scope_policy.exam_context} official update")

    text = " ".join(part for part in parts if part).strip()
    return text[:500]
