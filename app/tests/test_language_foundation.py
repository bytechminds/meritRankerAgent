"""Canonical language and actor-identity foundation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.doubt_solver import (
    DoubtSolverFinalResponse,
    DoubtSolverRequest,
    FinalAnswerResult,
    ResponseContent,
)
from services.doubt_solver.actor_identity import resolve_actor_id
from services.doubt_solver.answer_quality import AnswerQualityPolicy, validate_answer_quality
from services.doubt_solver.final_answer import build_final_answer_result
from services.doubt_solver.language_policy import (
    LanguagePolicyResolver,
    is_language_compliant,
    language_policy_char_count,
)
from services.llm.orchestration.prompt_resolver import PromptResolver

_REQUEST_IDS = {
    "user_id": "local-user",
    "conversation_id": "conversation-1",
    "turn_id": "turn-1",
}


@pytest.mark.parametrize("language", ["english", "hinglish", "hindi"])
def test_canonical_languages_are_accepted(language: str) -> None:
    request = DoubtSolverRequest(
        mode="doubt_solver",
        query="Question",
        language=language,  # type: ignore[arg-type]
        **_REQUEST_IDS,
    )

    assert request.language == language


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("en", "english"), ("hi", "hindi")],
)
def test_legacy_language_aliases_normalize_once(alias: str, canonical: str) -> None:
    request = DoubtSolverRequest(
        mode="doubt_solver",
        query="Question",
        language=alias,  # type: ignore[arg-type]
        **_REQUEST_IDS,
    )

    assert request.language == canonical
    assert request.model_dump()["language"] == canonical


def test_missing_language_defaults_to_canonical_english() -> None:
    request = DoubtSolverRequest(mode="doubt_solver", query="Question", **_REQUEST_IDS)

    assert request.language == "english"


def test_unsupported_language_is_rejected() -> None:
    with pytest.raises(ValidationError, match="english, hinglish, or hindi"):
        DoubtSolverRequest(
            mode="doubt_solver",
            query="Question",
            language="french",  # type: ignore[arg-type]
            **_REQUEST_IDS,
        )


def test_validated_request_is_immutable() -> None:
    request = DoubtSolverRequest(mode="doubt_solver", query="Question", **_REQUEST_IDS)

    with pytest.raises(ValidationError):
        request.language = "hindi"  # type: ignore[misc]


def test_actor_resolver_returns_validated_compatibility_user_id() -> None:
    request = DoubtSolverRequest(
        mode="doubt_solver",
        query="Question",
        **{**_REQUEST_IDS, "user_id": "student-123"},
    )

    assert resolve_actor_id(request) == "student-123"


@pytest.mark.parametrize("language", ["english", "hinglish", "hindi"])
def test_language_policy_is_compact(language: str) -> None:
    resolved = LanguagePolicyResolver().resolve(language)  # type: ignore[arg-type]

    assert resolved.language == language
    assert resolved.instruction
    assert language_policy_char_count(language) == len(resolved.instruction)  # type: ignore[arg-type]
    assert len(resolved.instruction) <= 240


def test_hinglish_policy_requires_latin_script_and_preserves_symbols() -> None:
    instruction = LanguagePolicyResolver().resolve("hinglish").instruction

    assert "Latin-script Hinglish" in instruction
    assert "formulas" in instruction
    assert "units" in instruction
    assert "option labels" in instruction


def test_hindi_policy_preserves_formula_units_and_option_labels() -> None:
    instruction = LanguagePolicyResolver().resolve("hindi").instruction

    assert "Devanagari" in instruction
    assert "formulas" in instruction
    assert "units" in instruction
    assert "option labels" in instruction


@pytest.mark.parametrize(
    ("content", "language", "expected"),
    [
        ("**Answer:** The correct option is B because velocity is constant.", "english", True),
        ("**उत्तर:** सही विकल्प B है क्योंकि वेग स्थिर है।", "english", False),
        ("**Answer:** Sahi option B hai because velocity constant hai.", "hinglish", True),
        ("**उत्तर:** सही विकल्प B है क्योंकि वेग स्थिर है।", "hinglish", False),
        ("**उत्तर:** सही option B है क्योंकि velocity constant है।", "hindi", True),
        ("**Answer:** The correct option is B because velocity is constant.", "hindi", False),
        ("**उत्तर:** \\(v = 20\\,m/s\\), option B.", "hindi", True),
    ],
)
def test_obvious_script_compliance(
    content: str,
    language: str,
    expected: bool,
) -> None:
    assert is_language_compliant(content, language) is expected  # type: ignore[arg-type]


def test_generator_prompt_gets_exactly_one_language_policy() -> None:
    resolver = PromptResolver()
    policy = LanguagePolicyResolver().resolve("hindi").instruction

    composed = resolver.compose_generator_system_prompt(
        "Base generator prompt",
        task_role="generator",
        exam_id=None,
        exam_stage=None,
        language="hindi",
    )

    assert composed.count(policy) == 1


def test_non_generator_prompt_does_not_get_language_policy() -> None:
    resolver = PromptResolver()

    composed = resolver.compose_generator_system_prompt(
        "Classifier prompt",
        task_role="classifier",
        exam_id=None,
        exam_stage=None,
        language="hindi",
    )

    assert composed == "Classifier prompt"


def test_language_mismatch_requires_rewrite() -> None:
    policy = AnswerQualityPolicy(
        validation_enabled=True,
        rewrite_enabled=True,
        max_rewrite_attempts=1,
        math_intermediate_max_chars=4000,
        max_visible_steps=20,
        max_display_math_blocks=20,
        max_math_line_chars=300,
        completion_marker="<ANSWER_DONE>",
    )

    result = validate_answer_quality(
        "**Answer:** The correct option is B because velocity is constant.",
        subject="general",
        difficulty="default",
        intent="explain",
        language="hindi",
        policy=policy,
    )

    assert result.is_valid is False
    assert result.language_compliant is False
    assert "language_mismatch" in result.reason_codes


def test_final_answer_is_immutable_and_excluded_from_public_response() -> None:
    final_answer = build_final_answer_result(
        content="**Answer:** Sahi option B hai.",
        language="hinglish",
        quality_status="passed_quality_gate",
        was_regenerated=True,
    )
    response = DoubtSolverFinalResponse(
        request_id="request-1",
        content=ResponseContent(value=final_answer.content),
        answer=final_answer.content,
        final_answer=final_answer,
    )

    assert response.final_answer == final_answer
    assert "final_answer" not in response.model_dump()
    with pytest.raises(ValidationError):
        final_answer.quality_status = "checked"  # type: ignore[misc]


def test_final_response_rejects_non_authoritative_answer_projection() -> None:
    final_answer = FinalAnswerResult(
        content="Authoritative answer",
        quality_status="checked",
        language_compliant=True,
    )

    with pytest.raises(ValidationError, match="final_answer.content"):
        DoubtSolverFinalResponse(
            request_id="request-1",
            content=ResponseContent(value="Different answer"),
            answer="Different answer",
            final_answer=final_answer,
        )
