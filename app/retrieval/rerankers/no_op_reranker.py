"""Disabled reranker implementation."""

from __future__ import annotations

from retrieval.models import RetrievedCandidate


class NoOpReranker:
    """Preserve S3 Vector ranking when ColBERT is disabled."""

    def rerank(
        self, *, query: str, candidates: list[RetrievedCandidate]
    ) -> tuple[list[RetrievedCandidate], bool, str | None]:
        _ = query
        return candidates, False, None
