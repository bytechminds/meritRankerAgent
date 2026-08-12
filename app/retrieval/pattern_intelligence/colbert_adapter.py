"""Composition adapter that limits legacy ColBERT to linked question references."""

from __future__ import annotations

from retrieval.interfaces import CandidateReranker
from retrieval.models import RetrievedCandidate
from retrieval.pattern_intelligence.models import DoubtQuestionReference


class ColbertLinkedQuestionReranker:
    """Adapt the existing ColBERT contract without exposing answer-bearing fields."""

    def __init__(self, reranker: CandidateReranker) -> None:
        self._reranker = reranker

    def rerank(
        self,
        *,
        query: str,
        candidates: list[DoubtQuestionReference],
    ) -> tuple[list[DoubtQuestionReference], bool, str | None]:
        if not candidates:
            return [], False, None
        legacy_candidates = [
            RetrievedCandidate(
                patternId=reference.question_id,
                metadata={"compactText": _reference_text(reference)},
            )
            for reference in candidates
        ]
        reranked, used, warning = self._reranker.rerank(
            query=query,
            candidates=legacy_candidates,
        )
        by_question_id = {reference.question_id: reference for reference in candidates}
        ordered: list[DoubtQuestionReference] = []
        used_ids: set[str] = set()
        for candidate in reranked:
            reference = by_question_id.get(candidate.pattern_id)
            if reference is not None and reference.question_id not in used_ids:
                ordered.append(reference)
                used_ids.add(reference.question_id)
        ordered.extend(
            reference for reference in candidates if reference.question_id not in used_ids
        )
        return ordered, used, warning


def _reference_text(reference: DoubtQuestionReference) -> str:
    return "\n".join((reference.question_text, *reference.options))[:4000]
