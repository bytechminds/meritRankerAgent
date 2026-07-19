"""Regression tests for Markdown spacing, symbol integrity, and exact replay."""

from __future__ import annotations

import pytest

from services.doubt_solver.answer_quality import (
    AnswerQualityPolicy,
    validate_answer_quality,
)
from services.doubt_solver.markdown_replay import iter_markdown_replay_chunks


@pytest.fixture
def policy() -> AnswerQualityPolicy:
    return AnswerQualityPolicy(
        validation_enabled=True,
        rewrite_enabled=True,
        max_rewrite_attempts=1,
        math_intermediate_max_chars=2200,
        max_visible_steps=8,
        max_display_math_blocks=6,
        max_math_line_chars=300,
        completion_marker="<ANSWER_DONE>",
    )


def _validate(content: str, policy: AnswerQualityPolicy):
    return validate_answer_quality(
        content,
        subject="general",
        difficulty="default",
        intent="solve",
        policy=policy,
    )


@pytest.mark.parametrize(
    "statement",
    [
        "Adopted on 26 November 1949",
        "coming into effect on 26 January 1950",
        "India as a sovereign, socialist, secular, democratic republic",
        "Dr. B. R. Ambedkar",
        "Article 32",
        "India's Constitution preserves citizens' rights.",
    ],
)
def test_readable_factual_and_english_text_is_clean(
    statement: str, policy: AnswerQualityPolicy
) -> None:
    result = _validate(f"**Answer:** {statement}\n<ANSWER_DONE>", policy)

    assert result.is_valid
    assert result.severity == "clean"


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("**Answer:** Adopted on26thNovember1949\n<ANSWER_DONE>", "joined_date_tokens"),
        ("**Answer:** Article32\n<ANSWER_DONE>", "suspicious_word_number_join"),
        ("**Answer:** [source](https://example.test/a\n<ANSWER_DONE>", "markdown_malformed_link"),
        ("**Answer:** ```text\nopen\n<ANSWER_DONE>", "markdown_unclosed_fence"),
    ],
)
def test_serious_formatting_defects_are_rejected(
    content: str, reason: str, policy: AnswerQualityPolicy
) -> None:
    result = _validate(content, policy)

    assert not result.is_valid
    assert reason in result.reason_codes


def test_conflicting_answer_values_are_rejected(policy: AnswerQualityPolicy) -> None:
    result = _validate(
        "**Answer:** 7%\n\n**Answer:** 70%\n<ANSWER_DONE>",
        policy,
    )

    assert not result.is_valid
    assert "duplicate_final_answer" in result.reason_codes
    assert "conflicting_answer_values" in result.reason_codes


def test_single_consistent_percentage_is_clean(policy: AnswerQualityPolicy) -> None:
    result = _validate("**Answer:** 70%\n<ANSWER_DONE>", policy)

    assert result.is_valid
    assert "conflicting_answer_values" not in result.reason_codes


def test_math_statistics_chemistry_and_reasoning_symbols_are_clean(
    policy: AnswerQualityPolicy,
) -> None:
    content = (
        "**Answer:** \\(\\frac{1}{\\sqrt{2}}\\), \\(\\mu = 4\\), "
        "\\(\\sigma^2 = 9\\), \\(H_2O\\), \\(Na^+\\), and \\(A \\to B\\).\n"
        "<ANSWER_DONE>"
    )

    result = _validate(content, policy)

    assert result.is_valid
    assert result.severity == "clean"


def test_spacing_checks_ignore_code_urls_formulae_and_identifiers(
    policy: AnswerQualityPolicy,
) -> None:
    content = (
        "**Answer:** Use `Article32`, visit https://example.test/Article32, "
        "keep H2O and model_v2 unchanged.\n"
        "```python\nvalue = 'on26thNovember1949'\n```\n"
        "<ANSWER_DONE>"
    )

    result = _validate(content, policy)

    assert result.is_valid
    assert "joined_date_tokens" not in result.reason_codes
    assert "suspicious_word_number_join" not in result.reason_codes


def test_invalid_utf8_surrogate_is_rejected(policy: AnswerQualityPolicy) -> None:
    result = _validate("**Answer:** bad \ud800 value", policy)

    assert not result.is_valid
    assert result.reason_codes == ["invalid_utf8"]


def test_replay_preserves_spaces_symbols_and_exact_content() -> None:
    content = (
        "**Answer:** Adopted on 26 November 1949.\n\n"
        "\\[P(A) = \\frac{1}{\\sqrt{2}}\\]\n"
        "Reaction: \\(2H_2 + O_2 \\to 2H_2O\\)."
    )
    chunks = list(iter_markdown_replay_chunks(content, max_chunk_chars=16))

    assert "".join(chunks) == content
    assert any(chunk.endswith(" ") for chunk in chunks)


def test_replay_preserves_leading_space_in_a_chunk() -> None:
    content = " Leading space remains intact."
    chunks = list(iter_markdown_replay_chunks(content, max_chunk_chars=10))

    assert chunks[0].startswith(" ")
    assert "".join(chunks) == content


def test_replay_does_not_split_unicode_grapheme_cluster() -> None:
    family = "👨‍👩‍👧‍👦"
    flag = "🇮🇳"
    content = f"A {family} family in {flag}"
    chunks = list(iter_markdown_replay_chunks(content, max_chunk_chars=3))

    assert "".join(chunks) == content
    assert family in chunks
    assert flag in chunks
