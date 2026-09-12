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
        "**Final Answer:** 7%\n\n**Final Answer:** 70%\n<ANSWER_DONE>",
        policy,
    )

    assert not result.is_valid
    assert "duplicate_final_answer" in result.reason_codes
    assert "conflicting_answer_values" in result.reason_codes


@pytest.mark.parametrize(
    "content",
    [
        (
            "The triangle rows use 2, 4, and 8.\n\n"
            "**Answer:** The missing number is 16.\n<ANSWER_DONE>"
        ),
        (
            "A is B's brother and B is C's mother.\n\n"
            "**Final Answer:** A is C's maternal uncle.\n<ANSWER_DONE>"
        ),
        (
            "Option A gives 12 and option B gives 18, so both are rejected.\n\n"
            "**Answer:** Option C is correct.\n<ANSWER_DONE>"
        ),
        (
            r"First \(x + 3 = 9\), so \(x = 6\). Then \(2x = 12\)." "\n\n"
            r"**Final Answer:** \(12\)" "\n<ANSWER_DONE>"
        ),
        (
            "**Answer:** Eliminate options A and B.\n\n"
            "**Final Answer:** Option C is correct.\n<ANSWER_DONE>"
        ),
    ],
)
def test_intermediate_reasoning_is_not_a_conflicting_final_answer(
    content: str, policy: AnswerQualityPolicy
) -> None:
    result = _validate(content, policy)

    assert "duplicate_final_answer" not in result.reason_codes
    assert "conflicting_answer_values" not in result.reason_codes


def test_duplicate_explicit_final_answer_blocks_are_rejected(
    policy: AnswerQualityPolicy,
) -> None:
    result = _validate(
        "**Final Answer:** Option C\n\n**Final Answer:** Option C\n<ANSWER_DONE>",
        policy,
    )

    assert "duplicate_final_answer" in result.reason_codes
    assert "conflicting_answer_values" not in result.reason_codes


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


def test_scalar_headline_contradicting_the_final_answer_is_rejected(
    policy: AnswerQualityPolicy,
) -> None:
    """The student reads the headline, so 3 seconds over a 12-second solution is unsafe."""
    result = _validate(
        "**Answer:** 3 seconds\n\n**Solution:**\n\nSpeed = 36 km/h = 10 m/s\n\n"
        "Time = 120/10 = 12 seconds\n\n**Final Answer:** 12 seconds\n<ANSWER_DONE>",
        policy,
    )

    assert not result.is_valid
    assert "conflicting_answer_values" in result.reason_codes


@pytest.mark.parametrize(
    "content",
    [
        "**Answer:** C\n\nthe working derives B\n\n**Final Answer:** B\n<ANSWER_DONE>",
        "**Answer:** 40\n\nthe sum is 64\n\n**Final Answer:** 24\n<ANSWER_DONE>",
        r"**Answer:** \(15\)" "\n\n" r"**Final Answer:** \(12\)" "\n<ANSWER_DONE>",
    ],
)
def test_option_numeric_and_latex_headline_conflicts_are_rejected(
    content: str, policy: AnswerQualityPolicy
) -> None:
    result = _validate(content, policy)

    assert "conflicting_answer_values" in result.reason_codes


@pytest.mark.parametrize(
    "content",
    [
        # the same value written two ways
        "**Answer:** 12 s\n\n**Final Answer:** 12 seconds\n<ANSWER_DONE>",
        "**Answer:** 20%\n\n**Final Answer:** 20 percent\n<ANSWER_DONE>",
        "**Answer:** B\n\n**Final Answer:** Option B\n<ANSWER_DONE>",
        "**Answer:** B\n\n**Final Answer:** Option B — New Delhi\n<ANSWER_DONE>",
        # a single heading has nothing to disagree with
        "**Answer:** 70%\n<ANSWER_DONE>",
    ],
)
def test_equivalent_or_single_answer_surfaces_stay_clean(
    content: str, policy: AnswerQualityPolicy
) -> None:
    result = _validate(content, policy)

    assert "conflicting_answer_values" not in result.reason_codes
