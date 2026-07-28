"""Internal contracts for the shared classification stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from schemas.doubt_solver import QueryClassification

ClassificationModality = Literal["text", "image"]
ClassificationStageStatus = Literal[
    "validated",
    "validated_fallback",
    "technical_fallback",
]


@dataclass(frozen=True)
class ClassificationStageResult:
    raw: QueryClassification
    classification: dict[str, object]
    modality: ClassificationModality
    status: ClassificationStageStatus
    classifier_confidence: float | None
    classifier_fallback: bool
