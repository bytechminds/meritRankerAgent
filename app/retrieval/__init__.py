"""Student retrieval boundary backed by S3 Vectors and DynamoDB."""

from typing import TYPE_CHECKING

from retrieval.models import (
    PatternRuntimeBundle,
    RetrievalMode,
    RetrievedCandidate,
    StudentRetrievalContext,
)

if TYPE_CHECKING:
    from retrieval.retrieval_service import StudentRetrievalService

__all__ = [
    "PatternRuntimeBundle",
    "RetrievedCandidate",
    "RetrievalMode",
    "StudentRetrievalContext",
    "StudentRetrievalService",
]


def __getattr__(name: str) -> object:
    """Avoid importing the graph-facing service while model contracts initialize."""
    if name == "StudentRetrievalService":
        from retrieval.retrieval_service import StudentRetrievalService

        return StudentRetrievalService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
