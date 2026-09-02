"""Duplicate identity for generated Practice candidates.

A generic instructional stem is legitimate exam construction for SSC/Banking English
("Which sentence is grammatically correct?"), and the question it actually asks then
lives in the options. Keying identity on the stem alone collapsed materially different
items into one another and cost roughly two thirds of every error-detection batch.

Reuse, PatternGraph and ingestion identity are deliberately untouched; they continue
to use ``normalize_question_text``.
"""

from __future__ import annotations

from features.practice_generation.matching import (
    normalize_question_identity,
    normalize_question_text,
)

GENERIC_STEM = "Which sentence is grammatically correct?"
OPTIONS_A = [
    {"value": "He go to school every day."},
    {"value": "He goes to school every day."},
    {"value": "He going to school every day."},
    {"value": "He gone to school every day."},
]
OPTIONS_B = [
    {"value": "She don't like tea."},
    {"value": "She doesn't like tea."},
    {"value": "She not like tea."},
    {"value": "She no like tea."},
]


def test_same_generic_stem_with_different_options_is_distinct() -> None:
    assert normalize_question_identity(
        GENERIC_STEM, OPTIONS_A
    ) != normalize_question_identity(GENERIC_STEM, OPTIONS_B)


def test_same_stem_and_same_options_is_duplicate() -> None:
    assert normalize_question_identity(
        GENERIC_STEM, OPTIONS_A
    ) == normalize_question_identity(GENERIC_STEM, list(OPTIONS_A))


def test_reordered_options_are_still_duplicate() -> None:
    """Shuffling the options does not make a new question."""
    assert normalize_question_identity(
        GENERIC_STEM, OPTIONS_A
    ) == normalize_question_identity(GENERIC_STEM, list(reversed(OPTIONS_A)))


def test_cosmetic_text_variation_is_still_duplicate() -> None:
    assert normalize_question_identity(
        "  which SENTENCE is grammatically correct??  ", OPTIONS_A
    ) == normalize_question_identity(GENERIC_STEM, OPTIONS_A)


def test_different_stems_stay_distinct_even_with_shared_options() -> None:
    assert normalize_question_identity(
        GENERIC_STEM, OPTIONS_A
    ) != normalize_question_identity("Choose the incorrect sentence.", OPTIONS_A)


def test_options_may_be_objects_or_plain_strings() -> None:
    """Generated questions carry option objects; persisted rows may carry strings."""
    plain = [option["value"] for option in OPTIONS_A]
    assert normalize_question_identity(
        GENERIC_STEM, OPTIONS_A
    ) == normalize_question_identity(GENERIC_STEM, plain)


def test_missing_options_fall_back_to_the_stem() -> None:
    """Absent options must not silently make everything unique."""
    assert normalize_question_identity(GENERIC_STEM, None) == normalize_question_text(
        GENERIC_STEM
    )
    assert normalize_question_identity(GENERIC_STEM, []) == normalize_question_text(
        GENERIC_STEM
    )


def test_reuse_identity_is_unchanged() -> None:
    """Semantic duplicate protection for reuse/ingestion must not be weakened."""
    assert normalize_question_text(GENERIC_STEM) == "which sentence is grammatically correct"


class TestFinalSetUsesTheSameIdentity:
    """validate_final_set was the last stage still keying on the stem alone.

    Generation dedup and replacement exclusion already used stem + canonicalized
    options; the final gate did not, so a legitimate English set with a shared generic
    stem was rejected as DUPLICATE_OR_EMPTY_QUESTION_TEXT after passing every earlier
    stage. Reuse identity (matching.py) deliberately still uses normalize_question_text.
    """

    @staticmethod
    def _linked(question: str, options: list[str], slot: str) -> dict[str, object]:
        meta = {
            "slotId": slot,
            "bucketId": "bucket-1",
            "verified": True,
            "schemaVersion": "2",
            "questionType": "mcq",
            "language": "english",
        }
        return {
            "questionId": f"q-{slot}",
            "question": question,
            "options": list(options),
            "correctAnswer": options[0],
            "_practiceMeta": meta,
        }

    def _texts(self, items: list[dict[str, object]]) -> list[str]:
        return [
            normalize_question_identity(
                str(item.get("question") or ""), item.get("options")
            )
            for item in items
        ]

    def test_generic_stem_with_different_options_is_distinct(self) -> None:
        items = [
            self._linked(GENERIC_STEM, [o["value"] for o in OPTIONS_A], "slot-001"),
            self._linked(GENERIC_STEM, [o["value"] for o in OPTIONS_B], "slot-002"),
        ]
        texts = self._texts(items)
        assert len(texts) == len(set(texts))

    def test_identical_stem_and_options_is_duplicate(self) -> None:
        values = [o["value"] for o in OPTIONS_A]
        items = [
            self._linked(GENERIC_STEM, values, "slot-001"),
            self._linked(GENERIC_STEM, values, "slot-002"),
        ]
        texts = self._texts(items)
        assert len(texts) != len(set(texts))

    def test_reordered_options_is_duplicate(self) -> None:
        values = [o["value"] for o in OPTIONS_A]
        items = [
            self._linked(GENERIC_STEM, values, "slot-001"),
            self._linked(GENERIC_STEM, list(reversed(values)), "slot-002"),
        ]
        texts = self._texts(items)
        assert len(texts) != len(set(texts))

    def test_empty_question_text_is_still_rejected(self) -> None:
        """Emptiness must remain detectable, so the falsy check still fires."""
        item = self._linked("", [o["value"] for o in OPTIONS_A], "slot-001")
        assert not normalize_question_identity("", item["options"]).split("|")[0]
