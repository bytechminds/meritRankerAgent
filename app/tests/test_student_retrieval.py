"""Unit tests for the approved S3 Vector student retrieval path."""

from __future__ import annotations

import io
import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

import config as cfg_module
from config import Settings, get_settings
from retrieval.context.data_context_builder import render_retrieval_context
from retrieval.embeddings.bedrock_titan_embedder import BedrockTitanEmbedder
from retrieval.models import PatternRuntimeBundle, RetrievedCandidate, StudentRetrievalContext
from retrieval.rerankers.colbert_reranker import ColbertReranker
from retrieval.retrieval_service import StudentRetrievalService, build_retrieval_query_text
from retrieval.s3_vectors.query_builder import pattern_filter, runtime_filter
from services.context_retrieval.context_models import ContextRetrievalRequest
from services.context_retrieval.context_retrieval_service import ContextRetrievalService


class _Embedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.01] * 1024


class _VectorClient:
    def __init__(
        self,
        runtime: list[RetrievedCandidate] | None = None,
        pattern: list[RetrievedCandidate] | None = None,
    ) -> None:
        self.runtime = runtime or []
        self.pattern = pattern or []
        self.runtime_subject: str | None = None
        self.pattern_subject: str | None = None

    def query_runtime(
        self, *, query_vector: list[float], subject: str | None
    ) -> list[RetrievedCandidate]:
        assert len(query_vector) == 1024
        self.runtime_subject = subject
        return self.runtime

    def query_pattern(
        self, *, query_vector: list[float], subject: str | None
    ) -> list[RetrievedCandidate]:
        assert len(query_vector) == 1024
        self.pattern_subject = subject
        return self.pattern


class _Store:
    def __init__(self, bundles: dict[str, PatternRuntimeBundle]) -> None:
        self.bundles = bundles
        self.calls: list[str] = []

    def fetch(self, pattern_id: str) -> PatternRuntimeBundle | None:
        self.calls.append(pattern_id)
        return self.bundles.get(pattern_id)


class _StaticStudentRetrieval:
    def __init__(self, context: StudentRetrievalContext) -> None:
        self.context = context

    def retrieve(self, request: ContextRetrievalRequest) -> StudentRetrievalContext:
        return self.context


def _request(**overrides: Any) -> ContextRetrievalRequest:
    values = {
        "request_id": "student-retrieval-1",
        "query": "A train crosses a platform in 18 seconds at 54 km/hr.",
        "subject": "math",
        "topic": "time speed distance",
        "confidence": 0.95,
        "intent": "solve",
    }
    values.update(overrides)
    return ContextRetrievalRequest(**values)


def _candidate(pattern_id: str = "pattern-1", **overrides: Any) -> RetrievedCandidate:
    values = {
        "patternId": pattern_id,
        "score": 0.94,
        "subject": "QUANT",
        "topic": "time speed distance",
        "flowType": "COMPUTATIONAL_SOLVE_FLOW",
        "versionHash": "v1",
        "runtimeReady": True,
        "canUseForRetrieval": True,
        "canUseForFinalAnswer": True,
    }
    values.update(overrides)
    return RetrievedCandidate(**values)


def _bundle(pattern_id: str = "pattern-1", **overrides: Any) -> PatternRuntimeBundle:
    values = {
        "patternId": pattern_id,
        "patternStatus": "approved",
        "answerDataStatus": "valid",
        "solveFlowStatus": "approved",
        "runtimeReady": True,
        "studentVisibility": "runtime_ready",
        "canUseForRetrieval": True,
        "canUseForFinalAnswer": True,
        "subject": "QUANT",
        "topic": "time speed distance",
        "flowType": "COMPUTATIONAL_SOLVE_FLOW",
        "patternTitle": "Relative speed",
        "patternGraphCompact": {"method": "distance equals speed times time"},
        "solveFlow": {"steps": ["convert speed", "apply distance equals speed times time"]},
        "answerGuide": {"verify": "check unit conversion"},
        "notSameWhen": [],
        "materialConflicts": [],
        "versionHash": "v1",
        "updatedAt": "2026-07-15T00:00:00Z",
    }
    values.update(overrides)
    return PatternRuntimeBundle(**values)


@pytest.fixture
def s3_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "s3_vector")
    monkeypatch.setenv("S3_VECTOR_BUCKET_NAME", "student-vectors")
    monkeypatch.setenv("DYNAMODB_PATTERN_TABLE", "patterns")
    monkeypatch.setenv("BEDROCK_EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    monkeypatch.setenv("BEDROCK_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("BEDROCK_EMBEDDING_NORMALIZE", "true")
    monkeypatch.setenv("S3_VECTOR_DIMENSIONS", "1024")
    monkeypatch.setenv("S3_VECTOR_DISTANCE_METRIC", "cosine")
    cfg_module._settings = None
    return get_settings()


def _service(
    settings: Settings,
    *,
    runtime: list[RetrievedCandidate] | None = None,
    pattern: list[RetrievedCandidate] | None = None,
    bundles: dict[str, PatternRuntimeBundle] | None = None,
) -> tuple[StudentRetrievalService, _VectorClient, _Store]:
    vector_client = _VectorClient(runtime=runtime, pattern=pattern)
    store = _Store(bundles or {})
    return (
        StudentRetrievalService(
            settings=settings,
            embedder=_Embedder(),
            vector_client=vector_client,
            pattern_store=store,
        ),
        vector_client,
        store,
    )


def test_titan_embedder_uses_approved_1024_dimension_normalized_request(
    s3_settings: Settings,
) -> None:
    client = MagicMock()
    response_body = json.dumps({"embedding": [0.1] * 1024}).encode()
    client.invoke_model.return_value = {"body": io.BytesIO(response_body)}
    embedder = BedrockTitanEmbedder(settings=s3_settings, client_factory=lambda region: client)

    embedding = embedder.embed_query("  solve   this  ")

    assert len(embedding) == 1024
    assert json.loads(client.invoke_model.call_args.kwargs["body"]) == {
        "inputText": "solve this",
        "dimensions": 1024,
        "normalize": True,
    }


def test_retrieval_query_text_includes_subject_topic_and_numeric_operator_cues() -> None:
    query_text = build_retrieval_query_text(_request(query="Solve 12% of 250 = x"))

    assert "subject: math" in query_text
    assert "topic: time speed distance" in query_text
    assert "entities: 12%, 250, =" in query_text


def test_s3_vector_filters_match_the_approved_runtime_and_pattern_contracts() -> None:
    assert runtime_filter("QUANT") == {
        "patternStatus": "approved",
        "solveFlowStatus": "approved",
        "runtimeReady": True,
        "canUseForFinalAnswer": True,
        "subject": "QUANT",
    }
    assert pattern_filter() == {
        "patternStatus": "approved",
        "canUseForRetrieval": True,
    }


def test_runtime_ready_bundle_is_selected_only_when_all_strict_states_match(
    s3_settings: Settings,
) -> None:
    service, vector_client, _ = _service(
        s3_settings,
        runtime=[_candidate()],
        bundles={"pattern-1": _bundle()},
    )

    context = service.retrieve(_request())

    assert context.mode == "runtime_ready"
    assert context.can_use_for_final_answer is True
    assert context.selected_pattern_id == "pattern-1"
    assert vector_client.runtime_subject == "QUANT"


def test_pattern_assist_never_exposes_solve_flow_or_final_answer_authority(
    s3_settings: Settings,
) -> None:
    candidate = _candidate(
        runtimeReady=False,
        canUseForFinalAnswer=False,
        canUseForRetrieval=True,
    )
    bundle = _bundle(
        answerDataStatus="missing",
        solveFlowStatus="needs_review",
        runtimeReady=False,
        studentVisibility="pattern_only",
        canUseForFinalAnswer=False,
    )
    service, _, _ = _service(
        s3_settings,
        pattern=[candidate],
        bundles={"pattern-1": bundle},
    )

    context = service.retrieve(_request())

    assert context.mode == "pattern_assist"
    assert context.can_use_for_final_answer is False
    assert context.solve_flow == {}
    assert context.answer_guide == {}


def test_stale_vector_metadata_does_not_override_dynamodb_bundle(
    s3_settings: Settings,
) -> None:
    stale_candidate = _candidate(versionHash="stale")
    service, _, store = _service(
        s3_settings,
        runtime=[stale_candidate],
        bundles={"pattern-1": _bundle(versionHash="v1")},
    )

    context = service.retrieve(_request())

    assert context.mode == "fresh_solve"
    assert store.calls == ["pattern-1"]


def test_needs_review_bundle_is_not_runtime_ready(
    s3_settings: Settings,
) -> None:
    service, _, _ = _service(
        s3_settings,
        runtime=[_candidate()],
        bundles={"pattern-1": _bundle(solveFlowStatus="needs_review")},
    )

    context = service.retrieve(_request())

    assert context.mode == "fresh_solve"
    assert context.can_use_for_final_answer is False


def test_renderer_includes_internal_retrieval_context_for_runtime_ready_bundle() -> None:
    context = StudentRetrievalContext(
        mode="runtime_ready",
        selectedPatternId="pattern-1",
        canUseForFinalAnswer=True,
        subject="QUANT",
        flowType="COMPUTATIONAL_SOLVE_FLOW",
        patternGraphCompact={"method": "convert units"},
    )
    rendered = render_retrieval_context(context)

    assert "[RETRIEVAL_CONTEXT]" in rendered
    assert "mode: runtime_ready" in rendered
    assert "canUseForFinalAnswer: true" in rendered
    assert "selectedPatternId: pattern-1" in rendered


def test_fresh_solve_does_not_force_context_into_generator_input() -> None:
    assert render_retrieval_context(StudentRetrievalContext.fresh_solve("no_match")) == ""


def test_s3_provider_never_calls_injected_bedrock_kb_retriever(
    s3_settings: Settings,
) -> None:
    kb_retriever = MagicMock()
    service = ContextRetrievalService(
        kb_retriever=kb_retriever,
        student_retrieval_service=_StaticStudentRetrieval(StudentRetrievalContext.fresh_solve("none")),
    )

    result = service.retrieve_context(_request())

    assert result.retrieval_context.mode == "fresh_solve"
    kb_retriever.retrieve_lane.assert_not_called()


def test_colbert_timeout_returns_original_s3_order(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = ColbertReranker(model_name="colbert", timeout_ms=1, top_k=2)

    class _SlowModel:
        def rerank(self, **kwargs: Any) -> list[dict[str, int]]:
            time.sleep(0.03)
            return [{"document_id": 1}, {"document_id": 0}]

    monkeypatch.setattr(reranker, "_get_model", lambda: _SlowModel())
    candidates = [_candidate("pattern-1"), _candidate("pattern-2")]

    result, used, warning = reranker.rerank(query="train speed", candidates=candidates)

    assert result == candidates
    assert used is False
    assert warning == "colbert_timeout"
