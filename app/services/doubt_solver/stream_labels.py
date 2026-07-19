"""
app/services/doubt_solver/stream_labels.py
------------------------------------------
Deterministic student-facing labels for orchestrated doubt solver streaming.

Labels are UX status text only — not chain-of-thought, routing details, or
provider internals.
"""

from __future__ import annotations

_GENERATING_LABELS: dict[str, str] = {
    "solve": "Solving...",
    "explain": "Explaining...",
    "practice": "Creating practice questions...",
    "visualize": "Preparing visual explanation...",
}

_STAGE_LABELS: dict[str, str] = {
    "understanding": "Understanding...",
    "retrieving": "Retrieving context...",
    "thinking": "Thinking...",
    "finalizing": "Finalizing...",
    "verifying": "Verifying answer...",
    "complete": "Done",
    "error": "Something went wrong. Please try again.",
}

LABEL_CAREFUL_CLASSIFICATION = "Checking the question more carefully..."

LABEL_WEB_SEARCH = "Checking recent information..."

LABEL_WEB_SEARCH_RETRY = "Looking for more reliable sources..."

LABEL_WEB_SEARCH_WEAK = "Reliable recent sources were limited, answering carefully..."

LABEL_GENERATOR_FALLBACK = "Preparing a more reliable answer..."

LABEL_ANSWER_CONTINUATION = "Finishing the answer..."


def get_stream_label(stage: str, intent: str | None = None) -> str:
    """Return the student-facing label for an internal streaming stage.

    Args:
        stage: Internal stage name (understanding, thinking, generating, etc.).
        intent: Normalised intent (solve, explain, practice, visualize) used
                only when stage is ``generating``.

    Returns:
        Deterministic student-facing label string.
    """
    if stage == "generating":
        if intent and intent in _GENERATING_LABELS:
            return _GENERATING_LABELS[intent]
        return "Preparing answer..."

    return _STAGE_LABELS.get(stage, "Preparing answer...")
