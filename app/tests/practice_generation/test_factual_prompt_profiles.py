"""
app/tests/practice_generation/test_factual_prompt_profiles.py
-------------------------------------------------------------
Factual authoring prompt composition.

The frozen composed budget is 2500 characters, so factual guidance REPLACES the generic
role prompt and output-policy overlay rather than stacking on them. These tests pin both
halves: that every factual subject composes inside the budget, and that non-factual
subjects are untouched.

No network calls. No LLM calls. No AWS calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from features.practice_generation.providers import (
    _FACTUAL_GENERATOR_PROMPT,
    _PRACTICE_GENERATOR_OVERLAYS,
    _PRACTICE_OVERLAYS,
    _factual_subject_profile,
)
from services.llm.orchestration.prompt_resolver import DEFAULT_PROMPT_ROOT

FACTUAL_SUBJECTS = [
    "science", "history", "geography", "physics", "chemistry",
    "biology", "computer_science", "economics", "polity",
]
COMPOSED_BUDGET = 2_500
PER_FILE_BUDGET = 1_700


def _chars(relative: str) -> int:
    return len((DEFAULT_PROMPT_ROOT / relative).read_text(encoding="utf-8"))


class TestEveryFactualSubjectHasAProfile:
    @pytest.mark.parametrize("subject", FACTUAL_SUBJECTS)
    def test_profile_exists(self, subject: str) -> None:
        profile = _factual_subject_profile(subject)
        assert profile == f"practice_generation/subjects/{subject}.md"
        assert (DEFAULT_PROMPT_ROOT / profile).is_file()

    @pytest.mark.parametrize("subject", ["math", "reasoning", "english", "general", "other"])
    def test_non_factual_subjects_have_no_profile(self, subject: str) -> None:
        """Only the factual family swaps its prompt; everything else is unchanged."""
        assert _factual_subject_profile(subject) is None

    def test_profiles_are_selected_by_family_not_a_local_list(self) -> None:
        """Adding a qualified family must stay a routing-config change."""
        source = Path("features/practice_generation/providers.py").read_text()
        helper = source[source.index("def _factual_subject_profile") :]
        helper = helper[: helper.index("\n\n\n")]
        assert "normalize_subject" in helper
        for subject in FACTUAL_SUBJECTS:
            assert f'"{subject}"' not in helper


class TestComposedPromptStaysInsideTheFrozenBudget:
    @pytest.mark.parametrize("subject", FACTUAL_SUBJECTS)
    def test_factual_composition_fits(self, subject: str) -> None:
        profile = _factual_subject_profile(subject)
        assert profile is not None
        composed = (
            _chars("practice_generation/shared_contract.md")
            + _chars(f"{_FACTUAL_GENERATOR_PROMPT}.md")
            + _chars(profile)
        )
        assert composed <= COMPOSED_BUDGET, f"{subject} composes to {composed}"

    def test_factual_role_prompt_respects_the_per_file_cap(self) -> None:
        assert _chars(f"{_FACTUAL_GENERATOR_PROMPT}.md") < PER_FILE_BUDGET

    @pytest.mark.parametrize("subject", FACTUAL_SUBJECTS)
    def test_profiles_stay_compact(self, subject: str) -> None:
        assert _chars(f"practice_generation/subjects/{subject}.md") <= 300


class TestReplacementNotAddition:
    def test_the_generic_output_policy_is_replaced_for_factual(self) -> None:
        """Stacking the profile on top of the generic overlay would breach the budget."""
        import inspect

        from features.practice_generation.providers import _execute

        source = inspect.getsource(_execute)
        assert "subject_profile is not None" in source
        assert "[*_PRACTICE_OVERLAYS, subject_profile]" in source

    def test_non_factual_overlays_are_unchanged(self) -> None:
        assert _PRACTICE_OVERLAYS == ["practice_generation/shared_contract.md"]
        assert _PRACTICE_GENERATOR_OVERLAYS == [
            "practice_generation/shared_contract.md",
            "practice_generation/generator_output_policy.md",
        ]


class TestFactualPromptContract:
    def test_author_output_stays_minimal(self) -> None:
        """The minimal Author contract must not regress into prose."""
        text = (DEFAULT_PROMPT_ROOT / f"{_FACTUAL_GENERATOR_PROMPT}.md").read_text()
        assert "Omit `correct_answer`, `solution`, and `answer_explanation`" in text
        for banned in ("timeline", "flowchart", "markdown", "### "):
            assert banned not in text.lower()

    def test_the_key_is_declared_non_authoritative(self) -> None:
        text = (DEFAULT_PROMPT_ROOT / f"{_FACTUAL_GENERATOR_PROMPT}.md").read_text()
        assert "PENDING_VERIFICATION" in text

    def test_guessing_and_ungrounded_current_facts_are_forbidden(self) -> None:
        text = (DEFAULT_PROMPT_ROOT / f"{_FACTUAL_GENERATOR_PROMPT}.md").read_text()
        assert "Never guess an exact date" in text
        assert "time-sensitive claim from memory" in text

    def test_no_uncertainty_tool_is_promised_to_the_model(self) -> None:
        """Model tool-calling does not exist in the LLM layer and is out of release
        scope, so the prompt must not offer a fact-check call it cannot make."""
        text = (DEFAULT_PROMPT_ROOT / f"{_FACTUAL_GENERATOR_PROMPT}.md").read_text()
        assert "fact check" not in text.lower()
        assert "search" not in text.lower()
        assert "do not build that item" in text


class TestFactualPromptSemanticInvariants:
    """Behaviour the factual prompt must retain, independent of its wording.

    The composed prompt sits at the frozen 2500-char bound, so every edit is a
    compression. Twice a required rule was lost that way — the language instruction and
    (earlier) an author-contract clause — and neither was caught by a size check. These
    guards assert the behaviour, not the phrasing.
    """

    def test_evidence_grounded_authoring_is_instructed(self) -> None:
        """Forbidding memory is not enough; the Author must be told to use the evidence.

        The factual prompt replaces ``question_generator_v2.md`` for all nine factual
        subjects, so it must carry that file's ``fresh_evidence`` rule too. Without it a
        time-sensitive request produced correct-but-static facts that the Answer
        Authority then had to reject as ANSWER_NOT_SUPPORTED.
        """
        text = self._role_text()
        assert "fresh_evidence" in text
        assert "evidence_by_slot" in text

    @staticmethod
    def _role_text() -> str:
        return (DEFAULT_PROMPT_ROOT / f"{_FACTUAL_GENERATOR_PROMPT}.md").read_text()

    def test_supplied_language_controls_student_visible_output(self) -> None:
        """Lost once already: without it the model picked script non-deterministically
        and Hinglish economics produced Devanagari, which the language gate rejected."""
        text = self._role_text().lower()
        assert "language" in text
        assert "supplied `language`" in self._role_text()

    def test_uncertainty_must_abandon_the_item(self) -> None:
        text = self._role_text().lower()
        assert "do not guess" in text
        assert "do not build that item" in text

    def test_single_defensible_answer_is_required(self) -> None:
        text = self._role_text()
        assert "the only option satisfying the stem" in text

    def test_distractors_must_be_checked_for_equivalence(self) -> None:
        text = self._role_text().lower()
        assert "equivalence" in text or "equivalent" in text

    def test_time_sensitive_claims_may_not_come_from_memory(self) -> None:
        assert "time-sensitive claim from memory" in self._role_text()

    def test_exact_values_may_not_be_guessed(self) -> None:
        assert "Never guess an exact date" in self._role_text()

    @pytest.mark.parametrize("subject", FACTUAL_SUBJECTS)
    def test_a_subject_profile_is_always_composed(self, subject: str) -> None:
        profile = _factual_subject_profile(subject)
        assert profile is not None
        assert (DEFAULT_PROMPT_ROOT / profile).read_text().strip()

    def test_non_factual_role_prompt_is_untouched(self) -> None:
        """The generic prompt must not drift while the factual one is edited."""
        text = (DEFAULT_PROMPT_ROOT / "practice_generation/question_generator_v2.md").read_text()
        assert "Use supplied `language` for every student-visible value." in text
        assert "Omit `correct_answer`, `solution`, and `answer_explanation`" in text
        assert len(text) == 1617
