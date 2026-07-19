"""Protocol boundaries for retrieval integrations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from retrieval.models import PatternRuntimeBundle, RetrievedCandidate


@runtime_checkable
class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]:
        """Return a normalized 1024-dimensional query embedding."""
        ...


@runtime_checkable
class VectorCandidateFinder(Protocol):
    def query_runtime(
        self, *, query_vector: list[float], subject: str | None
    ) -> list[RetrievedCandidate]:
        """Return candidate-only runtime index matches."""
        ...

    def query_pattern(
        self, *, query_vector: list[float], subject: str | None
    ) -> list[RetrievedCandidate]:
        """Return candidate-only pattern index matches."""
        ...


@runtime_checkable
class PatternBundleStore(Protocol):
    def fetch(self, pattern_id: str) -> PatternRuntimeBundle | None:
        """Fetch the current source-of-truth runtime bundle."""
        ...


@runtime_checkable
class CandidateReranker(Protocol):
    def rerank(
        self, *, query: str, candidates: list[RetrievedCandidate]
    ) -> tuple[list[RetrievedCandidate], bool, str | None]:
        """Return candidates, whether reranking was used, and an optional warning."""
        ...
