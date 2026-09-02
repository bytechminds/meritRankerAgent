"""Practice may adopt a factual family only from a closed allowlist.

The classifier is instructed to label every factual subject as `general`
(classification_semantics.md:45-46), and _require_qualified_authority refuses `general`,
so the qualified factual family (Gemini Author / Luna Authority) was unreachable in
production: history, polity and science Practice requests all failed closed.

`pattern_family_candidate` names the real family accurately but is model-authored and
unconstrained in its own schema, so Practice validates it against this closed set before
trusting it. The shared classifier contract and its schema are unchanged; the guard and
the fail-closed path are unchanged.
"""

from __future__ import annotations

import pytest

from features.practice_generation.planning import canonical_practice_subject

FACTUAL_FAMILIES = [
    ("HISTORY", "history"),
    ("GEOGRAPHY", "geography"),
    ("POLITY", "polity"),
    ("ECONOMICS", "economics"),
    ("SCIENCE", "science"),
    ("PHYSICS", "physics"),
    ("CHEMISTRY", "chemistry"),
    ("BIOLOGY", "biology"),
    ("COMPUTER_SCIENCE", "computer_science"),
]


class TestAllowlistedFamiliesAreAdopted:
    @pytest.mark.parametrize(("candidate", "expected"), FACTUAL_FAMILIES)
    def test_general_narrows_to_the_allowlisted_family(
        self, candidate: str, expected: str
    ) -> None:
        assert canonical_practice_subject("general", candidate) == expected

    def test_candidate_casing_is_normalized(self) -> None:
        assert canonical_practice_subject("general", "history") == "history"
        assert canonical_practice_subject("GENERAL", "History") == "history"


class TestUnapprovedCandidatesFailClosed:
    @pytest.mark.parametrize(
        "candidate",
        ["random string", "", "   ", None, "MATH", "REASONING", "ENGLISH", "GK",
         "HISTORY_OF_INDIA", "hist", 42, ["HISTORY"]],
    )
    def test_general_stays_general(self, candidate: object) -> None:
        """Anything outside the closed set keeps the existing fail-closed behaviour."""
        assert canonical_practice_subject("general", candidate) == "general"


class TestExplicitSubjectAlwaysWins:
    @pytest.mark.parametrize(
        ("subject", "candidate"),
        [
            ("math", "HISTORY"),
            ("english", "POLITY"),
            ("reasoning", "BIOLOGY"),
            ("factual", "MATH"),
            ("history", "POLITY"),
        ],
    )
    def test_candidate_cannot_override_a_classified_subject(
        self, subject: str, candidate: str
    ) -> None:
        assert canonical_practice_subject(subject, candidate) == subject

    def test_missing_subject_defaults_to_general_then_narrows(self) -> None:
        assert canonical_practice_subject(None, "HISTORY") == "history"
        assert canonical_practice_subject(None, None) == "general"


class TestAdoptedFamiliesReachTheQualifiedAuthority:
    """The point of the change: these subjects must now route, not fail closed."""

    @pytest.mark.parametrize(("candidate", "subject"), FACTUAL_FAMILIES)
    def test_adopted_subject_resolves_a_qualified_authority(
        self, candidate: str, subject: str
    ) -> None:
        from features.practice_generation.providers import _require_qualified_authority

        resolved = canonical_practice_subject("general", candidate)
        assert resolved == subject
        assert _require_qualified_authority(
            request_id="canon", subject=resolved, language="english"
        ) == "factual.verifier.default"

    def test_general_still_fails_closed(self) -> None:
        from features.practice_generation.providers import (
            PracticeAuthorityUnavailableError,
            _require_qualified_authority,
        )

        resolved = canonical_practice_subject("general", "not-a-family")
        assert resolved == "general"
        with pytest.raises(PracticeAuthorityUnavailableError):
            _require_qualified_authority(
                request_id="canon", subject=resolved, language="english"
            )
