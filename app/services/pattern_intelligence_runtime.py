"""Concrete, fail-closed adapters for the canonical Pattern Intelligence runtime.

The runtime package owns matching, safety gates, and prompt-safe projections.  This
module owns only bounded AWS integration construction and deliberately contains no
process-wide retrieval cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from config import Settings, get_settings
from features.practice_generation.pattern_resource_contract import (
    load_pattern_resource_contract,
)
from retrieval.embeddings.bedrock_titan_embedder import BedrockTitanEmbedder
from retrieval.interfaces import QueryEmbedder
from retrieval.pattern_intelligence import (
    CanonicalPatternStore,
    ColbertLinkedQuestionReranker,
    LinkedPatternQuestionStore,
    LinkedQuestionReranker,
    PatternRuntimeService,
    PatternVectorCandidateFinder,
    PlayablePatternQuestionStore,
    VectorPatternCandidate,
)
from retrieval.rerankers.colbert_reranker import ColbertReranker
from retrieval.s3_vectors.client import S3VectorClient
from services import dynamodb_service

_CANONICAL_PATTERN_SUBJECTS = {
    "math": "QUANT",
    "quant": "QUANT",
    "quantitative": "QUANT",
    "reasoning": "REASONING",
    "english": "ENGLISH",
    "general": "GK",
    "gk": "GK",
    "science": "SCIENCE",
    "polity": "POLITY",
    "history": "HISTORY",
    "geography": "GEOGRAPHY",
    "economics": "ECONOMICS",
    "computer": "COMPUTER",
    "computer_science": "COMPUTER",
    "statistics": "STATISTICS",
    "current_affairs": "CURRENT_AFFAIRS",
    "evs": "EVS",
}


class S3PatternIntelligenceCandidateFinder:
    """Discover candidate Pattern ids through the direct S3 Vectors contract only."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedder: QueryEmbedder | None = None,
        vector_client: S3VectorClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or BedrockTitanEmbedder(settings=self._settings)
        self._vector_client = vector_client or S3VectorClient(settings=self._settings)

    def find_candidates(
        self,
        *,
        query: str,
        subject: str | None,
        limit: int,
        embedding_cache: dict[str, list[float]] | None = None,
    ) -> Sequence[VectorPatternCandidate]:
        """Return bounded candidate identifiers plus an advisory relevance score."""
        normalized_query = " ".join(query.split())
        if not normalized_query or limit < 1:
            return ()
        cached_vector = None if embedding_cache is None else embedding_cache.get(normalized_query)
        query_vector = cached_vector or self._embedder.embed_query(normalized_query)
        if embedding_cache is not None and cached_vector is None:
            embedding_cache[normalized_query] = query_vector
        if len(query_vector) != self._settings.s3_vector_dimensions:
            raise ValueError("Pattern Intelligence embedding dimension does not match S3 Vectors.")
        candidates = self._vector_client.query_pattern_intelligence_candidates(
            query_vector=query_vector,
            subject=_canonical_pattern_subject(subject),
            top_k=limit,
        )
        return tuple(
            VectorPatternCandidate(
                patternId=candidate.pattern_id,
                score=candidate.score,
                versionHash=candidate.version_hash,
            )
            for candidate in candidates
            if candidate.pattern_id
        )


class DynamoDbCanonicalPatternStore:
    """Hydrate canonical Pattern records in one bounded BatchGetItem call."""

    def __init__(
        self,
        *,
        table_name: str,
        partition_key: str,
    ) -> None:
        self._table_name = table_name.strip()
        self._partition_key = partition_key.strip()

    def batch_fetch(
        self,
        *,
        pattern_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        pattern_ids = _bounded_unique_ids(pattern_ids)
        if not pattern_ids:
            return ()
        return dynamodb_service.batch_get_items(
            self._table_name,
            [{self._partition_key: pattern_id} for pattern_id in pattern_ids],
            max_unprocessed_retries=1,
        )


class DynamoDbLinkedPatternQuestionStore:
    """Read provenance references only through the bounded questionsByPattern GSI."""

    def __init__(
        self,
        *,
        table_name: str,
        pattern_index_name: str,
    ) -> None:
        self._table_name = table_name.strip()
        self._pattern_index_name = pattern_index_name.strip()

    def list_by_pattern_id(
        self,
        *,
        pattern_id: str,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        normalized_pattern_id = pattern_id.strip()
        if not normalized_pattern_id or limit < 1:
            return ()
        return dynamodb_service.query_by_index(
            self._table_name,
            self._pattern_index_name,
            "patternId",
            normalized_pattern_id,
            limit=min(limit, 5),
            projection_fields=(
                "questionId",
                "patternId",
                "questionText",
                "options",
                "status",
                "stage",
                "answerContractStatus",
                "solutionStatus",
            ),
        )


class DynamoDbPlayablePatternQuestionStore:
    """Read authoritative playable rows through QuestionBank.getByPatternId."""

    def __init__(self, *, table_name: str, pattern_index_name: str) -> None:
        self._table_name = table_name.strip()
        self._pattern_index_name = pattern_index_name.strip()

    def list_by_pattern_id(
        self,
        *,
        pattern_id: str,
        limit: int,
    ) -> Sequence[Mapping[str, Any]]:
        if not pattern_id.strip() or limit < 1:
            return ()
        return dynamodb_service.query_by_index(
            self._table_name,
            self._pattern_index_name,
            "patternId",
            pattern_id.strip(),
            limit=min(limit, 5),
            projection_fields=(
                "qbId",
                "patternId",
                "patternVersionHash",
                "patternLinkEvidence",
                "question",
                "answers",
                "correctAnswer",
                "explanation",
                "category",
                "difficulty",
                "meta",
                "updatedAt",
            ),
        )


def build_pattern_intelligence_runtime(
    *,
    settings: Settings | None = None,
    vector_finder: PatternVectorCandidateFinder | None = None,
    pattern_store: CanonicalPatternStore | None = None,
    linked_question_store: LinkedPatternQuestionStore | None = None,
    linked_question_reranker: LinkedQuestionReranker | None = None,
    playable_question_store: PlayablePatternQuestionStore | None = None,
    include_linked_question_references: bool = True,
) -> PatternRuntimeService | None:
    """Build a request-local-safe runtime or return ``None`` when not configured.

    Explicit injected adapters keep this function unit-testable without AWS.  The
    direct S3 Vector and canonical Pattern table dependencies are required for the
    concrete production path.  PatternQuestion references are optional and omitted
    when their table/index contract is not configured.
    """
    resolved_settings = settings or get_settings()
    if not resolved_settings.pattern_intelligence_enabled:
        return None

    # Pattern Intelligence is explicitly enabled, so its resource identity is now
    # mandatory. Explicit environment values still win; SSM supplies only the rest.
    # A missing identifier is a configuration failure, never a silent NoOp.
    needs_resources = (
        vector_finder is None or pattern_store is None or playable_question_store is None
    )
    contract = load_pattern_resource_contract() if needs_resources else None
    if contract is not None:
        resolved_settings = replace(
            resolved_settings,
            s3_vector_pattern_index_arn=contract.pattern_vector_index_arn,
            s3_vector_pattern_index_name=contract.pattern_vector_index_name,
            dynamodb_pattern_table=contract.pattern_table_name,
            dynamodb_question_bank_table=contract.question_bank_table_name,
            dynamodb_question_bank_pattern_index=(contract.question_bank_pattern_index_name),
        )

    resolved_vector_finder = vector_finder or S3PatternIntelligenceCandidateFinder(
        settings=resolved_settings
    )

    resolved_pattern_store = pattern_store
    if resolved_pattern_store is None:
        resolved_pattern_store = DynamoDbCanonicalPatternStore(
            table_name=resolved_settings.dynamodb_pattern_table,
            partition_key=resolved_settings.dynamodb_pattern_pk,
        )

    resolved_linked_store = linked_question_store if include_linked_question_references else None
    if include_linked_question_references and resolved_linked_store is None and (
        resolved_settings.dynamodb_pattern_question_table
        and resolved_settings.dynamodb_pattern_question_by_pattern_index
    ):
        resolved_linked_store = DynamoDbLinkedPatternQuestionStore(
            table_name=resolved_settings.dynamodb_pattern_question_table,
            pattern_index_name=resolved_settings.dynamodb_pattern_question_by_pattern_index,
        )

    resolved_reranker = linked_question_reranker if include_linked_question_references else None
    if (
        include_linked_question_references
        and resolved_reranker is None
        and resolved_settings.enable_colbert_rerank
    ):
        resolved_reranker = ColbertLinkedQuestionReranker(
            ColbertReranker(
                model_name=resolved_settings.colbert_model_name,
                timeout_ms=resolved_settings.retrieval_max_latency_ms,
                top_k=resolved_settings.colbert_top_k,
            )
        )

    resolved_playable_store = playable_question_store
    if resolved_playable_store is None and (
        resolved_settings.pattern_intelligence_reuse_enabled
        or include_linked_question_references
    ):
        resolved_playable_store = DynamoDbPlayablePatternQuestionStore(
            table_name=resolved_settings.dynamodb_question_bank_table,
            pattern_index_name=resolved_settings.dynamodb_question_bank_pattern_index,
        )

    return PatternRuntimeService(
        vector_finder=resolved_vector_finder,
        pattern_store=resolved_pattern_store,
        linked_question_store=resolved_linked_store,
        linked_question_reranker=resolved_reranker,
        playable_question_store=resolved_playable_store,
        reuse_enabled=resolved_settings.pattern_intelligence_reuse_enabled,
    )


def _bounded_unique_ids(pattern_ids: Sequence[str]) -> tuple[str, ...]:
    """Keep the DynamoDB BatchGet key set deterministic and bounded."""
    unique_ids = {pattern_id.strip() for pattern_id in pattern_ids if pattern_id.strip()}
    return tuple(sorted(unique_ids)[:24])


def _canonical_pattern_subject(subject: str | None) -> str | None:
    """Map trusted runtime labels to the canonical Pattern subject enum."""
    if not isinstance(subject, str):
        return None
    normalized = subject.strip().casefold().replace("-", "_").replace(" ", "_")
    if not normalized:
        return None
    return _CANONICAL_PATTERN_SUBJECTS.get(normalized, normalized.upper())
