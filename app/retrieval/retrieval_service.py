"""Student retrieval orchestration: Titan embeddings, S3 Vectors, DynamoDB truth, and gates."""

from __future__ import annotations

import logging
import re
import time

from config import Settings, get_settings
from retrieval.cache.retrieval_cache import RetrievalCache
from retrieval.embeddings.base import EmbeddingConfigurationError, QueryEmbeddingError
from retrieval.embeddings.bedrock_titan_embedder import BedrockTitanEmbedder
from retrieval.gates.graph_compatibility_gate import GraphCompatibilityGate
from retrieval.gates.score_policy import is_strong_runtime_match, should_rerank
from retrieval.interfaces import (
    CandidateReranker,
    PatternBundleStore,
    QueryEmbedder,
    VectorCandidateFinder,
)
from retrieval.models import (
    PatternRuntimeBundle,
    RetrievalTrace,
    RetrievedCandidate,
    StudentRetrievalContext,
)
from retrieval.rerankers.colbert_reranker import ColbertReranker
from retrieval.rerankers.no_op_reranker import NoOpReranker
from retrieval.s3_vectors.client import S3VectorClient, S3VectorQueryError
from retrieval.stores.dynamodb_pattern_store import DynamoDbPatternStore, PatternBundleStoreError
from services.context_retrieval.context_models import ContextRetrievalRequest

logger = logging.getLogger(__name__)


def _elapsed_ms(started: float) -> int:
    return max(int((time.monotonic() - started) * 1000), 0)

_SUBJECT_TO_RUNTIME_SUBJECT: dict[str, str] = {
    "math": "QUANT",
    "quant": "QUANT",
    "quantitative": "QUANT",
    "reasoning": "REASONING",
    "english": "ENGLISH",
    "general": "GK",
}
_SUBJECT_CONFIDENCE_THRESHOLD = 0.85


class StudentRetrievalService:
    """Resolve only approved student context; otherwise fail closed to fresh solve."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        embedder: QueryEmbedder | None = None,
        vector_client: VectorCandidateFinder | None = None,
        pattern_store: PatternBundleStore | None = None,
        graph_gate: GraphCompatibilityGate | None = None,
        reranker: CandidateReranker | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._embedder = embedder or BedrockTitanEmbedder(settings=self._settings)
        self._vector_client = vector_client or S3VectorClient(settings=self._settings)
        self._pattern_store = pattern_store or DynamoDbPatternStore(settings=self._settings)
        self._graph_gate = graph_gate or GraphCompatibilityGate()
        self._reranker = reranker or self._build_reranker()
        self._candidate_cache: RetrievalCache[list[RetrievedCandidate]] = RetrievalCache(
            self._settings.retrieval_cache_ttl_seconds
        )
        self._bundle_cache: RetrievalCache[PatternRuntimeBundle] = RetrievalCache(
            self._settings.retrieval_cache_ttl_seconds
        )

    def retrieve(self, request: ContextRetrievalRequest) -> StudentRetrievalContext:
        """Build an approved runtime-ready, pattern-assist, or fresh-solve context."""
        started = time.monotonic()
        trace = RetrievalTrace()
        logger.info(
            "STUDENT_RETRIEVAL_STARTED queryId=%s subject=%s topic=%s",
            request.request_id,
            request.subject,
            request.topic or "",
        )

        if self._settings.retrieval_provider != "s3_vector":
            return self._fresh(request, trace, started, "retrieval_provider_not_s3_vector")
        if not self._settings.s3_vector_bucket_name or not self._settings.dynamodb_pattern_table:
            return self._fresh(request, trace, started, "retrieval_configuration_missing")

        try:
            query_text = build_retrieval_query_text(request)
            _embed_started = time.monotonic()
            query_vector = self._embedder.embed_query(query_text)
            trace = trace.model_copy(
                update={"embedding_ms": _elapsed_ms(_embed_started)}
            )
            if len(query_vector) != 1024:
                raise EmbeddingConfigurationError("Query embedding dimension did not equal 1024.")
            subject_filter = _subject_filter(request)
            _runtime_started = time.monotonic()
            runtime_candidates = self._query_runtime(query_vector, subject_filter, query_text)
            trace = trace.model_copy(
                update={
                    "runtime_candidates_count": len(runtime_candidates),
                    "runtime_query_ms": _elapsed_ms(_runtime_started),
                }
            )
            logger.info(
                "STUDENT_S3_VECTOR_RUNTIME_QUERY_COMPLETED queryId=%s candidates=%d",
                request.request_id,
                len(runtime_candidates),
            )

            runtime_context = self._select_runtime(
                request=request,
                candidates=runtime_candidates,
                trace=trace,
            )
            if runtime_context is not None and not self._exceeded_budget(started):
                return self._finalize(request, runtime_context, started)

            if self._exceeded_budget(started):
                return self._fresh(request, trace, started, "latency_budget_exceeded")

            _pattern_started = time.monotonic()
            pattern_candidates = self._query_pattern(query_vector, subject_filter, query_text)
            trace = trace.model_copy(
                update={
                    "pattern_candidates_count": len(pattern_candidates),
                    "pattern_query_ms": _elapsed_ms(_pattern_started),
                }
            )
            logger.info(
                "STUDENT_S3_VECTOR_PATTERN_QUERY_COMPLETED queryId=%s candidates=%d",
                request.request_id,
                len(pattern_candidates),
            )

            rerank_used = False
            _rerank_started = time.monotonic()
            warnings: list[str] = []
            if self._settings.enable_colbert_rerank and should_rerank(
                query=request.query,
                candidates=pattern_candidates,
            ):
                logger.info(
                    "STUDENT_COLBERT_RERANK_STARTED queryId=%s candidates=%d",
                    request.request_id,
                    len(pattern_candidates),
                )
                _rerank_started = time.monotonic()
                pattern_candidates, rerank_used, rerank_warning = self._reranker.rerank(
                    query=request.query,
                    candidates=pattern_candidates,
                )
                if rerank_warning:
                    warnings.append(rerank_warning)
                logger.info(
                    "STUDENT_COLBERT_RERANK_COMPLETED queryId=%s used=%s warning=%s",
                    request.request_id,
                    str(rerank_used).lower(),
                    rerank_warning or "",
                )

            trace = trace.model_copy(
                update={
                    "rerank_used": rerank_used,
                    "rerank_ms": _elapsed_ms(_rerank_started),
                }
            )
            pattern_context = self._select_pattern_assist(
                request=request,
                candidates=pattern_candidates,
                trace=trace,
                warnings=warnings,
            )
            if pattern_context is not None and not self._exceeded_budget(started):
                return self._finalize(request, pattern_context, started)
            return self._fresh(request, trace, started, "no_compatible_candidate", warnings)
        except (EmbeddingConfigurationError, QueryEmbeddingError) as exc:
            logger.error(
                "STUDENT_RETRIEVAL_ERROR queryId=%s stage=embedding error_type=%s",
                request.request_id,
                type(exc).__name__,
            )
            return self._fresh(request, trace, started, "embedding_unavailable")
        except S3VectorQueryError as exc:
            logger.warning(
                "STUDENT_RETRIEVAL_ERROR queryId=%s stage=s3_vector error_type=%s",
                request.request_id,
                type(exc).__name__,
            )
            return self._fresh(request, trace, started, "s3_vector_unavailable")
        except PatternBundleStoreError as exc:
            logger.warning(
                "STUDENT_RETRIEVAL_ERROR queryId=%s stage=dynamodb error_type=%s",
                request.request_id,
                type(exc).__name__,
            )
            return self._fresh(request, trace, started, "dynamodb_unavailable")
        except Exception as exc:
            logger.error(
                "STUDENT_RETRIEVAL_ERROR queryId=%s stage=unexpected error_type=%s",
                request.request_id,
                type(exc).__name__,
            )
            return self._fresh(request, trace, started, "retrieval_unavailable")

    def _query_runtime(
        self, vector: list[float], subject: str | None, query_text: str
    ) -> list[RetrievedCandidate]:
        key = RetrievalCache.key("runtime", query_text, subject or "")
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return [candidate.model_copy(update={"source": "cache"}) for candidate in cached]
        candidates = self._vector_client.query_runtime(
            query_vector=vector,
            subject=subject,
        )
        self._candidate_cache.put(key, candidates)
        return candidates

    def _query_pattern(
        self, vector: list[float], subject: str | None, query_text: str
    ) -> list[RetrievedCandidate]:
        key = RetrievalCache.key("pattern", query_text, subject or "")
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return [candidate.model_copy(update={"source": "cache"}) for candidate in cached]
        candidates = self._vector_client.query_pattern(
            query_vector=vector,
            subject=subject,
        )
        self._candidate_cache.put(key, candidates)
        return candidates

    def _select_runtime(
        self,
        *,
        request: ContextRetrievalRequest,
        candidates: list[RetrievedCandidate],
        trace: RetrievalTrace,
    ) -> StudentRetrievalContext | None:
        if not is_strong_runtime_match(candidates):
            return None
        candidate = candidates[0]
        bundle = self._fetch_bundle(candidate.pattern_id)
        if bundle is None or not _version_matches(candidate, bundle):
            return None
        decision = self._graph_gate.evaluate(
            query=request.query,
            subject=request.subject,
            intent=request.intent,
            bundle=bundle,
        )
        self._log_gate(request.request_id, candidate.pattern_id, decision.allowed, decision.reason)
        if not decision.allowed or not _is_runtime_ready(bundle):
            return None
        return _build_context(
            mode="runtime_ready",
            candidate=candidate,
            bundle=bundle,
            confidence=min(candidate.score, decision.confidence),
            trace=trace.model_copy(update={"graph_gate_passed": True}),
        )

    def _select_pattern_assist(
        self,
        *,
        request: ContextRetrievalRequest,
        candidates: list[RetrievedCandidate],
        trace: RetrievalTrace,
        warnings: list[str],
    ) -> StudentRetrievalContext | None:
        for candidate in candidates:
            bundle = self._fetch_bundle(candidate.pattern_id)
            if bundle is None or not _version_matches(candidate, bundle):
                continue
            decision = self._graph_gate.evaluate(
                query=request.query,
                subject=request.subject,
                intent=request.intent,
                bundle=bundle,
            )
            self._log_gate(
                request.request_id,
                candidate.pattern_id,
                decision.allowed,
                decision.reason,
            )
            if not decision.allowed or not _is_pattern_assist(bundle):
                continue
            return _build_context(
                mode="pattern_assist",
                candidate=candidate,
                bundle=bundle,
                confidence=min(candidate.score, decision.confidence),
                trace=trace.model_copy(update={"graph_gate_passed": True}),
                warnings=warnings,
            )
        return None

    def _fetch_bundle(self, pattern_id: str) -> PatternRuntimeBundle | None:
        key = RetrievalCache.key("bundle", pattern_id)
        cached = self._bundle_cache.get(key)
        if cached is not None:
            return cached
        bundle = self._pattern_store.fetch(pattern_id)
        if bundle is not None:
            self._bundle_cache.put(key, bundle)
            logger.info("STUDENT_DYNAMODB_BUNDLE_FETCHED patternId=%s", pattern_id)
        return bundle

    def _fresh(
        self,
        request: ContextRetrievalRequest,
        trace: RetrievalTrace,
        started: float,
        reason: str,
        warnings: list[str] | None = None,
    ) -> StudentRetrievalContext:
        context = StudentRetrievalContext(
            retrieval_trace=trace.model_copy(update={"fallback_reason": reason}),
            warnings=warnings or [reason],
        )
        logger.info(
            "STUDENT_RETRIEVAL_FALLBACK_FRESH_SOLVE queryId=%s reason=%s latencyMs=%d",
            request.request_id,
            reason,
            _latency_ms(started),
        )
        return context

    def _finalize(
        self,
        request: ContextRetrievalRequest,
        context: StudentRetrievalContext,
        started: float,
    ) -> StudentRetrievalContext:
        trace = context.retrieval_trace
        logger.info(
            "STUDENT_RETRIEVAL_CONTEXT_SELECTED queryId=%s subject=%s topic=%s "
            "runtimeCandidatesCount=%d patternCandidatesCount=%d rerankUsed=%s "
            "selectedPatternId=%s mode=%s canUseForFinalAnswer=%s confidence=%.2f "
            "latencyMs=%d fallbackReason=%s",
            request.request_id,
            context.subject or "",
            context.topic or "",
            trace.runtime_candidates_count,
            trace.pattern_candidates_count,
            str(trace.rerank_used).lower(),
            context.selected_pattern_id or "",
            context.mode,
            str(context.can_use_for_final_answer).lower(),
            context.confidence,
            _latency_ms(started),
            trace.fallback_reason or "",
        )
        return context

    def _exceeded_budget(self, started: float) -> bool:
        return _latency_ms(started) > self._settings.retrieval_max_latency_ms

    def _build_reranker(self) -> CandidateReranker:
        if not self._settings.enable_colbert_rerank:
            return NoOpReranker()
        return ColbertReranker(
            model_name=self._settings.colbert_model_name,
            timeout_ms=self._settings.retrieval_max_latency_ms,
            top_k=self._settings.colbert_top_k,
        )

    @staticmethod
    def _log_gate(request_id: str, pattern_id: str, allowed: bool, reason: str) -> None:
        logger.info(
            "STUDENT_GRAPH_GATE_DECISION queryId=%s patternId=%s allowed=%s reason=%s",
            request_id,
            pattern_id,
            str(allowed).lower(),
            reason,
        )


def build_retrieval_query_text(request: ContextRetrievalRequest) -> str:
    """Compose the approved query embedding input without introducing answer choices."""
    normalized_question = " ".join(request.query.split())
    parts = [f"question: {request.query}", f"normalized_question: {normalized_question}"]
    if request.subject:
        parts.append(f"subject: {request.subject}")
    if request.topic:
        parts.append(f"topic: {request.topic}")
    if request.pattern_topic_candidate:
        parts.append(f"pattern_topic: {request.pattern_topic_candidate}")
    if request.retrieval_tags:
        parts.append(f"signals: {', '.join(request.retrieval_tags[:10])}")
    entities = _retrieval_entities(request.query)
    if entities:
        parts.append(f"entities: {', '.join(entities)}")
    return "\n".join(parts)


def _subject_filter(request: ContextRetrievalRequest) -> str | None:
    confidence = request.confidence if request.confidence is not None else request.topic_confidence
    if confidence is None or confidence < _SUBJECT_CONFIDENCE_THRESHOLD:
        return None
    return _SUBJECT_TO_RUNTIME_SUBJECT.get(request.subject.strip().lower())


def _version_matches(candidate: RetrievedCandidate, bundle: PatternRuntimeBundle) -> bool:
    return candidate.version_hash is None or candidate.version_hash == bundle.version_hash


def _is_runtime_ready(bundle: PatternRuntimeBundle) -> bool:
    return (
        bundle.pattern_status == "approved"
        and bundle.answer_data_status == "valid"
        and bundle.solve_flow_status == "approved"
        and bundle.runtime_ready
        and bundle.student_visibility == "runtime_ready"
        and bundle.can_use_for_retrieval
        and bundle.can_use_for_final_answer
    )


def _is_pattern_assist(bundle: PatternRuntimeBundle) -> bool:
    return (
        bundle.pattern_status == "approved"
        and bundle.answer_data_status not in {"ambiguous", "inconsistent"}
        and bundle.solve_flow_status in {"missing", "needs_review", "not_required"}
        and not bundle.runtime_ready
        and bundle.student_visibility == "pattern_only"
        and bundle.can_use_for_retrieval
        and not bundle.can_use_for_final_answer
    )


def _build_context(
    *,
    mode: str,
    candidate: RetrievedCandidate,
    bundle: PatternRuntimeBundle,
    confidence: float,
    trace: RetrievalTrace,
    warnings: list[str] | None = None,
) -> StudentRetrievalContext:
    runtime_ready = mode == "runtime_ready"
    return StudentRetrievalContext(
        mode=mode,  # type: ignore[arg-type]
        selected_pattern_id=bundle.pattern_id,
        selected_candidate=candidate,
        runtime_bundle=bundle,
        pattern_graph_only=not runtime_ready,
        confidence=confidence,
        can_use_for_final_answer=runtime_ready,
        subject=bundle.subject,
        topic=bundle.topic or None,
        flow_type=bundle.flow_type,
        pattern_graph_compact=bundle.pattern_graph_compact,
        solve_flow=bundle.solve_flow if runtime_ready else {},
        answer_guide=bundle.answer_guide if runtime_ready else {},
        not_same_when=bundle.not_same_when,
        material_conflicts=bundle.material_conflicts,
        retrieval_trace=trace,
        warnings=warnings or [],
    )


def _latency_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _retrieval_entities(query: str) -> list[str]:
    """Extract bounded numeric and operator cues without changing the student question."""
    values = re.findall(r"\d+(?:\.\d+)?%?|[+\-*/=]", query)
    return values[:20]
