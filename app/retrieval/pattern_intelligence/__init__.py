"""Deterministic, source-of-truth Pattern intelligence contracts.

This package intentionally has no AWS client construction or process-wide cache. Callers
provide the vector and DynamoDB adapters for one request and keep the returned contexts
inside trusted server-side workflow state.
"""

from retrieval.pattern_intelligence.colbert_adapter import ColbertLinkedQuestionReranker
from retrieval.pattern_intelligence.interfaces import (
    CanonicalPatternStore,
    LinkedPatternQuestionStore,
    LinkedQuestionReranker,
    PatternVectorCandidateFinder,
    PlayablePatternQuestionStore,
)
from retrieval.pattern_intelligence.models import (
    CanonicalPatternQuestion,
    CanonicalPatternRecord,
    CanonicalPlayableQuestion,
    DoubtPatternContext,
    DoubtQuestionReference,
    DoubtSolveFlowStep,
    PatternGenerationContext,
    PatternMatchDecision,
    PatternMatchTier,
    PatternRetrievalPlan,
    PatternRuntimeMemo,
    PatternRuntimeMode,
    PatternRuntimeRequest,
    PatternRuntimeResult,
    VectorPatternCandidate,
)
from retrieval.pattern_intelligence.service import PatternCompatibilityGuard, PatternRuntimeService

__all__ = [
    "CanonicalPatternQuestion",
    "CanonicalPatternRecord",
    "CanonicalPatternStore",
    "CanonicalPlayableQuestion",
    "ColbertLinkedQuestionReranker",
    "DoubtPatternContext",
    "DoubtQuestionReference",
    "DoubtSolveFlowStep",
    "LinkedPatternQuestionStore",
    "LinkedQuestionReranker",
    "PatternCompatibilityGuard",
    "PatternGenerationContext",
    "PatternMatchDecision",
    "PatternMatchTier",
    "PatternRetrievalPlan",
    "PatternRuntimeMemo",
    "PatternRuntimeMode",
    "PatternRuntimeRequest",
    "PatternRuntimeResult",
    "PatternRuntimeService",
    "PatternVectorCandidateFinder",
    "PlayablePatternQuestionStore",
    "VectorPatternCandidate",
]
