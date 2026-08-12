"""Provider-agnostic completion outcome normalization."""

from __future__ import annotations

from typing import Literal

CompletionOutcome = Literal[
    "completed",
    "output_token_exhausted",
    "empty_response",
    "content_filtered",
    "provider_failure",
    "unknown",
]


def normalize_completion_outcome(reason: object | None) -> CompletionOutcome:
    """Map provider stop/finish indicators into one content-safe internal outcome."""
    if reason is None:
        return "unknown"
    value = getattr(reason, "value", reason)
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"stop", "completed", "complete", "end_turn", "finished"}:
        return "completed"
    if normalized in {
        "length",
        "max_tokens",
        "max_token",
        "max_output_tokens",
        "max_completion_tokens",
        "token_limit",
    }:
        return "output_token_exhausted"
    if normalized in {
        "content_filter",
        "content_filtered",
        "safety",
        "safety_block",
        "blocked",
        "recitation",
    }:
        return "content_filtered"
    if normalized in {"empty", "empty_response", "no_content"}:
        return "empty_response"
    if normalized in {"error", "failed", "provider_failure"}:
        return "provider_failure"
    return "unknown"
