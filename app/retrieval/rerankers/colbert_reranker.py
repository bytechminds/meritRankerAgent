"""Optional bounded RAGatouille/ColBERT reranker."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from retrieval.models import RetrievedCandidate
from retrieval.rerankers.base import RerankerUnavailableError


class ColbertReranker:
    """Rerank only the S3 Vector candidate set and fall back on any failure."""

    def __init__(self, *, model_name: str, timeout_ms: int, top_k: int) -> None:
        self._model_name = model_name.strip()
        self._timeout_seconds = max(timeout_ms, 1) / 1000
        self._top_k = max(top_k, 1)
        self._model: Any | None = None

    def rerank(
        self, *, query: str, candidates: list[RetrievedCandidate]
    ) -> tuple[list[RetrievedCandidate], bool, str | None]:
        if not candidates:
            return [], False, None
        try:
            model = self._get_model()
            documents = [self._candidate_text(candidate) for candidate in candidates]
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    model.rerank,
                    query=query,
                    documents=documents,
                    k=min(self._top_k, len(documents)),
                )
                result = future.result(timeout=self._timeout_seconds)
            reranked = self._apply_result(candidates, result)
            return reranked, True, None
        except TimeoutError:
            return candidates, False, "colbert_timeout"
        except RerankerUnavailableError:
            return candidates, False, "colbert_unavailable"
        except Exception:
            return candidates, False, "colbert_failed"

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self._model_name:
            raise RerankerUnavailableError(
                "COLBERT_MODEL_NAME is required when reranking is enabled."
            )
        try:
            from ragatouille import RAGPretrainedModel  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RerankerUnavailableError("RAGatouille is not installed.") from exc
        self._model = RAGPretrainedModel.from_pretrained(self._model_name)
        return self._model

    @staticmethod
    def _candidate_text(candidate: RetrievedCandidate) -> str:
        text = candidate.metadata.get("compactText")
        if isinstance(text, str) and text.strip():
            return text[:4000]
        return " ".join(
            part
            for part in (
                candidate.subject,
                candidate.topic,
                candidate.flow_type,
                candidate.chunk_type,
            )
            if part
        )[:4000]

    @staticmethod
    def _apply_result(
        candidates: list[RetrievedCandidate], result: object
    ) -> list[RetrievedCandidate]:
        if not isinstance(result, list):
            return candidates
        ranked: list[RetrievedCandidate] = []
        used_indexes: set[int] = set()
        for item in result:
            if not isinstance(item, dict):
                continue
            index = item.get("document_id")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            if index in used_indexes:
                continue
            used_indexes.add(index)
            ranked.append(candidates[index].model_copy(update={"source": "colbert"}))
        ranked.extend(
            candidate for index, candidate in enumerate(candidates) if index not in used_indexes
        )
        return ranked
