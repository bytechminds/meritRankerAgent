"""Authoritative final-answer construction for all delivery paths."""

from schemas.doubt_solver import CanonicalLanguage, FinalAnswerResult, QualityStatus
from services.doubt_solver.language_policy import is_language_compliant


def build_final_answer_result(
    *,
    content: str,
    language: CanonicalLanguage,
    quality_status: QualityStatus,
    was_regenerated: bool = False,
) -> FinalAnswerResult:
    """Build the immutable answer object after generation and quality handling."""
    return FinalAnswerResult(
        content=content,
        quality_status=quality_status,
        was_regenerated=was_regenerated,
        language_compliant=is_language_compliant(content, language),
    )
