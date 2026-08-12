"""Unit tests for bounded concrete adapters around Pattern Intelligence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from retrieval.models import RetrievedCandidate
from services import dynamodb_service
from services.pattern_intelligence_runtime import (
    DynamoDbCanonicalPatternStore,
    DynamoDbLinkedPatternQuestionStore,
    S3PatternIntelligenceCandidateFinder,
    build_pattern_intelligence_runtime,
)


def test_canonical_pattern_store_uses_one_bounded_batch_with_opt_in_retry(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _batch_get(table_name: str, keys: list[dict[str, Any]], **kwargs: Any):
        calls.append({"table_name": table_name, "keys": keys, **kwargs})
        return [{"patternId": "pattern-1"}]

    monkeypatch.setattr(dynamodb_service, "batch_get_items", _batch_get)
    store = DynamoDbCanonicalPatternStore(table_name="Patterns", partition_key="patternId")

    records = store.batch_fetch(pattern_ids=("pattern-2", "pattern-1", "pattern-1"))

    assert records == [{"patternId": "pattern-1"}]
    assert calls == [
        {
            "table_name": "Patterns",
            "keys": [{"patternId": "pattern-1"}, {"patternId": "pattern-2"}],
            "max_unprocessed_retries": 1,
        }
    ]


def test_linked_question_store_uses_pattern_gsi_without_scan(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _query(*args: Any, **kwargs: Any):
        calls.append({"args": args, "kwargs": kwargs})
        return [{"questionId": "reference-1"}]

    monkeypatch.setattr(dynamodb_service, "query_by_index", _query)
    store = DynamoDbLinkedPatternQuestionStore(
        table_name="PatternQuestions",
        pattern_index_name="questionsByPattern",
    )

    records = store.list_by_pattern_id(pattern_id="pattern-1", limit=10)

    assert records == [{"questionId": "reference-1"}]
    assert calls == [
        {
            "args": ("PatternQuestions", "questionsByPattern", "patternId", "pattern-1"),
            "kwargs": {
                "limit": 5,
                "projection_fields": (
                    "questionId",
                    "patternId",
                    "questionText",
                    "options",
                    "status",
                    "stage",
                    "answerContractStatus",
                    "solutionStatus",
                ),
            },
        }
    ]


def test_s3_candidate_finder_returns_identifiers_without_metadata_authority() -> None:
    class _Embedder:
        def embed_query(self, _query: str) -> list[float]:
            return [0.1, 0.2, 0.3]

    class _VectorClient:
        def query_pattern_intelligence_candidates(self, **kwargs: Any):
            assert kwargs["subject"] == "QUANT"
            assert kwargs["top_k"] == 2
            return [
                RetrievedCandidate(
                    patternId="pattern-1",
                    score=0.9,
                    versionHash="version-1",
                    metadata={"answer": "must not become authority"},
                )
            ]

    finder = S3PatternIntelligenceCandidateFinder(
        settings=SimpleNamespace(s3_vector_dimensions=3),
        embedder=_Embedder(),
        vector_client=_VectorClient(),
    )

    candidates = finder.find_candidates(query=" Compare ratios ", subject="math", limit=2)

    assert [candidate.pattern_id for candidate in candidates] == ["pattern-1"]
    assert candidates[0].version_hash == "version-1"


def test_runtime_factory_fails_closed_when_feature_is_disabled() -> None:
    runtime = build_pattern_intelligence_runtime(
        settings=SimpleNamespace(pattern_intelligence_enabled=False)
    )

    assert runtime is None
