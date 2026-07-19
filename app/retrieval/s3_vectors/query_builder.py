"""Build bounded S3 Vector query metadata filters."""

from __future__ import annotations


def runtime_filter(subject: str | None = None) -> dict[str, object]:
    """Return the approved runtime-ready metadata filter."""
    filters: dict[str, object] = {
        "patternStatus": "approved",
        "solveFlowStatus": "approved",
        "runtimeReady": True,
        "canUseForFinalAnswer": True,
    }
    if subject:
        filters["subject"] = subject
    return filters


def pattern_filter(subject: str | None = None) -> dict[str, object]:
    """Return the approved pattern-assist candidate metadata filter."""
    filters: dict[str, object] = {
        "patternStatus": "approved",
        "canUseForRetrieval": True,
    }
    if subject:
        filters["subject"] = subject
    return filters
