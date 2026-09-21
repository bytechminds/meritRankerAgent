"""Canonical limits for the Practice-generation domain."""

# A student may ask for a larger set, but V1 always builds at most this many slots.
MAX_PRACTICE_QUESTIONS = 50
# Count parsing intentionally recognizes up to three digits. Retain that request provenance
# while keeping the generated assessment bounded by ``MAX_PRACTICE_QUESTIONS``.
MAX_REQUESTED_PRACTICE_QUESTIONS = 999
# The persisted slot-id grammar remains backwards-compatible with existing assessments.
# New requests cannot create slots beyond ``MAX_PRACTICE_QUESTIONS``.
PRACTICE_SLOT_ID_PATTERN = r"^slot-(?:00[1-9]|0[1-9][0-9]|100)$"


def effective_practice_question_count(requested_count: int) -> int:
    """Return the V1-effective count for an already validated request count."""
    return min(requested_count, MAX_PRACTICE_QUESTIONS)
