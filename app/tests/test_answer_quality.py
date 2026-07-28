"""Tests for generator answer quality validation, sanitization, and rewrite."""

from __future__ import annotations

from schemas.llm import LlmMessage
from services.doubt_solver.answer_quality import (
    REWRITE_USER_PROMPT,
    AnswerQualityPolicy,
    build_rewrite_messages,
    detect_final_answer,
    rewrite_max_tokens,
    validate_answer_quality,
)


def _policy(**overrides: object) -> AnswerQualityPolicy:
    base = dict(
        validation_enabled=True,
        rewrite_enabled=True,
        max_rewrite_attempts=1,
        math_intermediate_max_chars=2200,
        max_visible_steps=8,
        max_display_math_blocks=6,
        max_math_line_chars=300,
        completion_marker="<ANSWER_DONE>",
    )
    base.update(overrides)
    return AnswerQualityPolicy(**base)  # type: ignore[arg-type]


class TestAnswerQualityValidator:
    def test_valid_markdown_passes(self) -> None:
        text = (
            "**Given:**\n- speed changes\n\n**Steps:**\n1. Compare\n\n"
            "**Final Answer:**\n\\(15\\) km/h\n<ANSWER_DONE>"
        )
        result = validate_answer_quality(
            text,
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert result.is_valid
        assert result.severity == "clean"

    def test_single_dollar_fails(self) -> None:
        result = validate_answer_quality(
            "Final Answer: $15$ km/h <ANSWER_DONE>",
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert not result.is_valid
        assert "math_single_dollar" in result.reason_codes

    def test_double_dollar_fails(self) -> None:
        result = validate_answer_quality(
            "Answer $$15$$ <ANSWER_DONE>",
            subject="math",
            difficulty="basic",
            intent="solve",
            policy=_policy(),
        )
        assert "math_double_dollar" in result.reason_codes

    def test_quad_dollar_fails(self) -> None:
        result = validate_answer_quality(
            "Bad $$$$ math",
            subject="math",
            difficulty="basic",
            intent="solve",
            policy=_policy(),
        )
        assert "math_quad_dollar" in result.reason_codes

    def test_unbalanced_inline_paren_fails(self) -> None:
        result = validate_answer_quality(
            "Speed \\(15 km/h <ANSWER_DONE>",
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert "math_unbalanced_inline" in result.reason_codes

    def test_unbalanced_display_bracket_fails(self) -> None:
        result = validate_answer_quality(
            "Value \\[15 <ANSWER_DONE>",
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert "math_unbalanced_display" in result.reason_codes

    def test_raw_html_fails(self) -> None:
        result = validate_answer_quality(
            "<script>alert(1)</script> Final Answer: 1 <ANSWER_DONE>",
            subject="general",
            difficulty="default",
            intent="explain",
            policy=_policy(),
        )
        assert result.severity == "unsafe"

    def test_valid_math_inequality_is_not_misread_as_html(self) -> None:
        text = (
            "From the extrema, \\(-4c^2 + 8c > 2.25c^2 - 2c\\), so "
            "\\(0 < c < 1.6\\).\n\n"
            "**Final Answer:** None of the listed options.\n<ANSWER_DONE>"
        )

        result = validate_answer_quality(
            text,
            subject="math",
            difficulty="intermediate",
            intent="solve",
            query="Find c and show the working.",
            policy=_policy(),
        )

        assert result.is_valid is True
        assert "raw_html_tag" not in result.reason_codes

    def test_actual_non_script_html_still_requires_rewrite(self) -> None:
        result = validate_answer_quality(
            "<div>Formatted answer</div>\n**Final Answer:** 1\n<ANSWER_DONE>",
            subject="math",
            difficulty="intermediate",
            intent="solve",
            query="Find the value and show the working.",
            policy=_policy(),
        )

        assert result.is_valid is False
        assert "raw_html_tag" in result.reason_codes

    def test_duplicate_final_answer_detected(self) -> None:
        text = (
            "**Final Answer:**\n1\n\n**Final Answer:**\n1\n<ANSWER_DONE>"
        )
        result = validate_answer_quality(
            text,
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert "duplicate_final_answer" in result.reason_codes

    def test_bad_phrase_detected(self) -> None:
        result = validate_answer_quality(
            "Actually, check the setup again. Final Answer: 2 <ANSWER_DONE>",
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert any(r.startswith("bad_phrase_") for r in result.reason_codes)

    def test_long_intermediate_math_flagged(self) -> None:
        body = "x" * 2300 + "\nFinal Answer: 1 <ANSWER_DONE>"
        result = validate_answer_quality(
            body,
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(math_intermediate_max_chars=2200),
        )
        assert "math_intermediate_too_long" in result.reason_codes

    def test_missing_final_answer_for_solve(self) -> None:
        result = validate_answer_quality(
            "Some steps only. <ANSWER_DONE>",
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert "missing_final_answer" in result.reason_codes

    def test_marker_missing_with_final_answer_not_rewrite_by_itself(self) -> None:
        result = validate_answer_quality(
            "**Final Answer:**\n\\(15\\) km/h",
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert "missing_final_answer" not in result.reason_codes
        assert result.severity == "clean"


class TestEmptyGeneratorOutput:
    def test_empty_content_is_invalid(self) -> None:
        result = validate_answer_quality(
            "",
            subject="reasoning",
            difficulty="advanced",
            intent="solve",
            policy=_policy(),
        )
        assert not result.is_valid
        assert result.severity == "error"
        assert "empty_answer" in result.reason_codes

    def test_whitespace_content_is_invalid(self) -> None:
        result = validate_answer_quality(
            "   \n\t  ",
            subject="reasoning",
            difficulty="advanced",
            intent="solve",
            policy=_policy(),
        )
        assert not result.is_valid
        assert result.severity == "error"
        assert "empty_answer" in result.reason_codes

    def test_empty_content_is_not_clean(self) -> None:
        result = validate_answer_quality(
            "",
            subject="math",
            difficulty="basic",
            intent="solve",
            policy=_policy(),
        )
        assert result.severity != "clean"

    def test_non_empty_valid_answer_still_clean(self) -> None:
        text = "**Final Answer:**\n\\(15\\) km/h\n<ANSWER_DONE>"
        result = validate_answer_quality(
            text,
            subject="math",
            difficulty="intermediate",
            intent="solve",
            policy=_policy(),
        )
        assert result.is_valid
        assert result.severity == "clean"


class TestRewriteHelpers:
    def test_rewrite_prompt_bans_dollar_delimiters(self) -> None:
        assert "$" in REWRITE_USER_PROMPT
        assert "Do not use $" in REWRITE_USER_PROMPT

    def test_rewrite_max_attempts_config(self) -> None:
        assert _policy().max_rewrite_attempts == 1

    def test_rewrite_budgets(self) -> None:
        assert rewrite_max_tokens(difficulty="basic", route_subject="general") == 500
        assert rewrite_max_tokens(difficulty="intermediate", route_subject="math") == 700
        assert rewrite_max_tokens(difficulty="advanced", route_subject="practice") == 1000

    def test_build_rewrite_messages(self) -> None:
        msgs = build_rewrite_messages(
            [LlmMessage(role="user", content="q")],
            draft_answer="bad draft",
        )
        assert msgs[-1].role == "user"
        assert "Rewrite" in msgs[-1].content


class TestPromptContent:
    def test_math_prompt_exam_style_no_teacher_names(self) -> None:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "prompts/subjects/math_generator.md"
        text = path.read_text()
        assert "competitive-exam shortcut style" in text
        assert "Do not show failed attempts" in text.lower() or "do not show failed" in text.lower()
        for name in ("Gopesh", "Rakesh", "teacher", "Sir ", "Ma'am"):
            assert name not in text

    def test_global_contract_bans_dollar_math(self) -> None:
        from pathlib import Path

        text = (
            Path(__file__).resolve().parents[1] / "prompts/generator_answer_contract.md"
        ).read_text()
        assert "Do not use `$...$`" in text or "Do not use `$" in text
        assert "Visual generation is deferred" in text


class TestDetectFinalAnswer:
    def test_detects_header(self) -> None:
        assert detect_final_answer("**Final Answer:**\n\\(5\\)")
