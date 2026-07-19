"""Minimal retrieval score policy for runtime selection and rerank escalation."""

from __future__ import annotations

from retrieval.models import RetrievedCandidate

_TRAP_TERMS: frozenset[str] = frozenset(
    {"not", "except", "least", "greatest", "only", "must", "possible", "definite"}
)


def is_strong_runtime_match(candidates: list[RetrievedCandidate]) -> bool:
    """Require a clearly strong top candidate before bypassing pattern fallback."""
    if not candidates:
        return False
    top = candidates[0]
    if top.score < 0.80:
        return False
    if len(candidates) == 1:
        return True
    return top.score - candidates[1].score >= 0.08


def should_rerank(*, query: str, candidates: list[RetrievedCandidate]) -> bool:
    """Use expensive reranking only for close or trap-sensitive candidate sets."""
    if len(candidates) < 2:
        return False
    normalized = set(query.lower().split())
    if normalized.intersection(_TRAP_TERMS):
        return True
    return candidates[0].score - candidates[1].score < 0.08
