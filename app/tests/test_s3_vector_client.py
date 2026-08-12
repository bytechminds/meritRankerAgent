"""Focused tests for the direct S3 Vector client contract."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import config as cfg_module
from config import Settings, get_settings
from retrieval.s3_vectors.client import S3VectorClient


def _settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "s3_vector")
    monkeypatch.setenv("S3_VECTOR_BUCKET_NAME", "pattern-vectors")
    monkeypatch.setenv("S3_VECTOR_RUNTIME_INDEX_NAME", "runtime-index")
    monkeypatch.setenv("S3_VECTOR_PATTERN_INDEX_NAME", "pattern-index")
    monkeypatch.setenv("BEDROCK_EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("BEDROCK_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("S3_VECTOR_DIMENSIONS", "1024")
    monkeypatch.setenv("S3_VECTOR_DISTANCE_METRIC", "cosine")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_MAX_CANDIDATES", "3")
    cfg_module._settings = None
    return get_settings()


def _candidate_response() -> dict[str, object]:
    return {
        "vectors": [
            {
                "key": "patterns/pattern-1",
                "distance": 0.1,
                "metadata": {
                    "patternId": "pattern-1",
                    "subject": "QUANT",
                    "versionHash": "version-1",
                },
            }
        ]
    }


def test_pattern_intelligence_query_returns_candidates_without_legacy_approval_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.query_vectors.return_value = _candidate_response()
    vector_client = S3VectorClient(settings=_settings(monkeypatch), client_factory=lambda _: client)

    candidates = vector_client.query_pattern_intelligence_candidates(
        query_vector=[0.1] * 1024,
        subject=" QUANT ",
        top_k=99,
    )

    request = client.query_vectors.call_args.kwargs
    assert request["vectorBucketName"] == "pattern-vectors"
    assert request["indexName"] == "pattern-index"
    assert request["topK"] == 3
    assert request["filter"] == {"subject": "QUANT"}
    assert "patternStatus" not in request["filter"]
    assert "canUseForRetrieval" not in request["filter"]
    assert [candidate.pattern_id for candidate in candidates] == ["pattern-1"]
    assert candidates[0].version_hash == "version-1"


def test_pattern_intelligence_query_omits_filter_without_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.query_vectors.return_value = _candidate_response()
    vector_client = S3VectorClient(settings=_settings(monkeypatch), client_factory=lambda _: client)

    vector_client.query_pattern_intelligence_candidates(
        query_vector=[0.1] * 1024,
        subject=None,
    )

    assert "filter" not in client.query_vectors.call_args.kwargs


def test_legacy_runtime_and_pattern_queries_keep_existing_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = MagicMock()
    client.query_vectors.return_value = _candidate_response()
    vector_client = S3VectorClient(settings=_settings(monkeypatch), client_factory=lambda _: client)

    vector_client.query_runtime(query_vector=[0.1] * 1024, subject="QUANT")
    runtime_request = client.query_vectors.call_args.kwargs
    vector_client.query_pattern(query_vector=[0.1] * 1024, subject="QUANT")
    pattern_request = client.query_vectors.call_args.kwargs

    assert runtime_request["filter"] == {
        "patternStatus": "approved",
        "solveFlowStatus": "approved",
        "runtimeReady": True,
        "canUseForFinalAnswer": True,
        "subject": "QUANT",
    }
    assert pattern_request["filter"] == {
        "patternStatus": "approved",
        "canUseForRetrieval": True,
        "subject": "QUANT",
    }
