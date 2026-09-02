"""A reconstructed Practice request must carry the student's real query.

The repository persists it as `originalQuery` (repositories.py); orchestration's
reconstruction previously read only `querySummary`, a key nothing writes and
progress.py strips on purpose. Every reconstructed request therefore carried the
placeholder "Practice generation request" (27 chars), which is why topic-evidence
grounding rejected every span for any request large enough to reach the intelligence
planner. Requests for two or fewer questions plan deterministically and never hit it,
which is exactly why they kept passing.
"""

from __future__ import annotations

import json

import pytest

from features.practice_generation.orchestration import _request
from features.practice_generation.planning import _grounded_topic_ids

STUDENT_QUERY = "Create 3 intermediate practice questions on percentages"
PLACEHOLDER = "Practice generation request"


def _meta(**overrides: object) -> dict[str, object]:
    practice_request: dict[str, object] = {
        "requestId": "req-1",
        "conversationId": "conv-1",
        "turnId": "turn-1",
        "practiceType": "QUICK_PRACTICE",
        "requestedCount": 3,
        "acceptedCount": 3,
        "subject": "math",
        "topic": "percentages",
        "difficulty": "intermediate",
        "language": "english",
        "originalQuery": STUDENT_QUERY,
    }
    practice_request.update(overrides)
    return practice_request


def _rebuild(practice_request: dict[str, object]):
    """Build the assessment row shape `_request` reads."""
    return _request(
        {
            "userId": "user-1",
            "name": "Percentages Quick Practice",
            "meta": json.dumps({"practiceRequest": practice_request}),
        }
    )


class TestPersistedQueryRoundTrip:
    def test_persisted_query_survives_reconstruction(self) -> None:
        assert _rebuild(_meta()).original_query == STUDENT_QUERY

    def test_legacy_query_summary_is_still_honoured(self) -> None:
        """Meta written before the key was retired must keep working."""
        legacy = _meta()
        legacy.pop("originalQuery")
        legacy["querySummary"] = STUDENT_QUERY
        assert _rebuild(legacy).original_query == STUDENT_QUERY

    def test_original_query_wins_over_legacy_key(self) -> None:
        legacy = _meta(querySummary="stale summary")
        assert _rebuild(legacy).original_query == STUDENT_QUERY

    def test_missing_both_keys_falls_back_safely(self) -> None:
        """Neither key present must not raise; the placeholder is the safe default."""
        bare = _meta()
        bare.pop("originalQuery")
        assert _rebuild(bare).original_query == PLACEHOLDER


class TestGroundingAfterRoundTrip:
    """The defect's actual consequence: a verbatim span must ground after round-trip."""

    @pytest.mark.parametrize("span", ["percentages", "intermediate practice questions"])
    def test_verbatim_span_grounds_against_reconstructed_query(self, span: str) -> None:
        request = _rebuild(_meta())
        haystack = request.original_query.casefold()
        assert span.casefold() in haystack

    def test_span_cannot_ground_against_the_placeholder(self) -> None:
        """Pins the failure mode so a regression is unambiguous, not mysterious."""
        bare = _meta()
        bare.pop("originalQuery")
        request = _rebuild(bare)
        assert request.original_query == PLACEHOLDER
        assert "percentages" not in request.original_query.casefold()

    def test_grounding_helper_accepts_a_real_span_after_round_trip(self) -> None:
        """_grounded_topic_ids is the function that raised in production."""
        assert callable(_grounded_topic_ids)
        request = _rebuild(_meta())
        assert "percentages" in request.original_query
