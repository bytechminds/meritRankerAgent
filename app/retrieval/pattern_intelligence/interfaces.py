"""Injected integration boundaries for Pattern intelligence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from retrieval.pattern_intelligence.models import (
    CanonicalPatternRecord,
    DoubtQuestionReference,
    VectorPatternCandidate,
)


@runtime_checkable
class PatternVectorCandidateFinder(Protocol):
    """Discover bounded Pattern identifiers only; vector metadata is never authority."""

    def find_candidates(
        self,
        *,
        query: str,
        subject: str | None,
        limit: int,
        embedding_cache: dict[str, list[float]] | None = None,
    ) -> Sequence[VectorPatternCandidate | Mapping[str, Any]]:
        """Return Pattern identifiers discovered from a vector index."""
        ...


@runtime_checkable
class CanonicalPatternStore(Protocol):
    """Fetch authoritative, unmarshalled Pattern records in one bounded batch."""

    def batch_fetch(
        self,
        *,
        pattern_ids: Sequence[str],
    ) -> Sequence[CanonicalPatternRecord | Mapping[str, Any]]:
        """Return only records currently available to the trusted runtime."""
        ...


@runtime_checkable
class LinkedPatternQuestionStore(Protocol):
    """Read PatternQuestion provenance through the canonical questionsByPattern index."""

    def list_by_pattern_id(
        self,
        *,
        pattern_id: str,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Return a bounded set of linked PatternQuestion records."""
        ...


@runtime_checkable
class PlayablePatternQuestionStore(Protocol):
    """Read authoritative QuestionBank rows through the exact Pattern GSI."""

    def list_by_pattern_id(
        self,
        *,
        pattern_id: str,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Return a bounded set of explicitly linked playable records."""
        ...


@runtime_checkable
class LinkedQuestionReranker(Protocol):
    """Optional reranker constrained to already-linked question references."""

    def rerank(
        self,
        *,
        query: str,
        candidates: list[DoubtQuestionReference],
    ) -> tuple[list[DoubtQuestionReference], bool, str | None]:
        """Return ordered references, use flag, and a safe fallback warning."""
        ...
