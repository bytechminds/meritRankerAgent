"""Map S3 Vectors QueryVectors results to candidate-only models."""

from __future__ import annotations

from typing import Any

from retrieval.models import RetrievedCandidate


def map_query_vectors_response(response: dict[str, Any]) -> list[RetrievedCandidate]:
    """Normalize S3 Vector output without treating metadata as authoritative."""
    vectors = response.get("vectors")
    if not isinstance(vectors, list):
        return []

    candidates: list[RetrievedCandidate] = []
    for vector in vectors:
        if not isinstance(vector, dict):
            continue
        metadata = vector.get("metadata")
        safe_metadata = metadata if isinstance(metadata, dict) else {}
        pattern_id = safe_metadata.get("patternId")
        if not isinstance(pattern_id, str) or not pattern_id.strip():
            continue
        raw_distance = vector.get("distance")
        distance = float(raw_distance) if isinstance(raw_distance, (int, float)) else None
        score = max(0.0, min(1.0, 1.0 - distance)) if distance is not None else 0.0
        try:
            candidates.append(
                RetrievedCandidate(
                    patternId=pattern_id.strip(),
                    chunkType=_string_or_none(safe_metadata.get("chunkType")),
                    subject=_string_or_none(safe_metadata.get("subject")),
                    topic=_string_or_none(safe_metadata.get("topic")),
                    flowType=_string_or_none(safe_metadata.get("flowType")),
                    score=score,
                    distance=distance,
                    vectorKey=_string_or_none(vector.get("key")),
                    versionHash=_string_or_none(safe_metadata.get("versionHash")),
                    runtimeReady=safe_metadata.get("runtimeReady") is True,
                    canUseForFinalAnswer=safe_metadata.get("canUseForFinalAnswer") is True,
                    canUseForRetrieval=safe_metadata.get("canUseForRetrieval") is True,
                    metadata=safe_metadata,
                )
            )
        except ValueError:
            continue
    return candidates


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None
