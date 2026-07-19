"""Shared reranker errors."""

from __future__ import annotations


class RerankerUnavailableError(Exception):
    """Raised when optional ColBERT cannot be used safely."""
