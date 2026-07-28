"""Canonical language and actor-identity foundation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import graphs.doubt_solver_graph as graph_module
from schemas.doubt_solver import (
    AnswerOutput,
    DoubtSolverFinalResponse,
    DoubtSolverRequest,
    FinalAnswerResult,
    QueryClassification,
    ResponseContent,
)
from schemas.llm_routing import RouteDecision
from services.doubt_solver.actor_identity import resolve_actor_id
from services.doubt_solver.answer_completion import (
    AnswerCompletionPolicy,
    build_continuation_messages,
)
from services.doubt_solver.answer_quality import (
    AnswerQualityPolicy,
    build_rewrite_messages,
    validate_answer_quality,
)
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


def _generator_route(language: str, *, intent: str = "explain") -> RouteDecision:
    return RouteDecision(
        route_id="general.generator.default",
        subject="general",
        task_role="generator",
        difficulty="default",
        intent=intent,
        language=language,  # type: ignore[arg-type]
        model="safe_mock",
        prompt="subjects/general_generator.md",
        overlays=[],
        intent_overlays={},
        temperature=0.0,
        max_tokens=800,
        route_source="exact",
    )


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
    assert language_policy_char_count(language) == len(resolved.instruction)  # type: ignore[arg-type]
    assert len(resolved.instruction) <= 360


def test_english_policy_adds_no_prompt_content_or_token_overhead() -> None:
    resolver = PromptResolver()

    composed = resolver.compose_generator_system_prompt(
        "Base generator prompt",
        task_role="generator",
        exam_id=None,
        exam_stage=None,
        language="english",
    )

    assert LanguagePolicyResolver().resolve("english").instruction == ""
    assert language_policy_char_count("english") == 0
    assert composed == "Base generator prompt"


def test_hinglish_policy_requires_latin_script_and_preserves_symbols() -> None:
    instruction = LanguagePolicyResolver().resolve("hinglish").instruction

    assert "Roman script only" in instruction
    assert "formulas" in instruction
    assert "option labels" in instruction
    assert "citations" in instruction
    assert "URLs" in instruction


def test_hindi_policy_preserves_formula_units_and_option_labels() -> None:
    instruction = LanguagePolicyResolver().resolve("hindi").instruction

    assert "देवनागरी" in instruction
    assert "सूत्र" in instruction
    assert "विकल्प लेबल" in instruction
    assert "citations" in instruction
    assert "URLs" in instruction


@pytest.mark.parametrize(
    ("content", "language", "expected"),
    [
        ("**Answer:** The correct option is B because velocity is constant.", "english", True),
        ("**उत्तर:** सही विकल्प B है क्योंकि वेग स्थिर है।", "english", False),
        ("**Answer:** Sahi option B hai because velocity constant hai.", "hinglish", True),
        ("**उत्तर:** सही विकल्प B है क्योंकि वेग स्थिर है।", "hinglish", False),
        (
            "**Answer:** The correct response follows from the standard interest "
            "formula and the values supplied in the question.",
            "hinglish",
            False,
        ),
        (
            "**Answer:** Simple interest calculate karne ke liye principal, rate aur "
            "time ko formula mein use karein.",
            "hinglish",
            True,
        ),
        ("**उत्तर:** सही option B है क्योंकि velocity constant है।", "hindi", True),
        ("**Answer:** The correct option is B because velocity is constant.", "hindi", False),
        ("**Answer:** \\(x = 5\\)", "hindi", False),
        ("\\(x = 5\\)", "hindi", True),
        (
            "**उत्तर:** सही है। The complete explanation is primarily in English "
            "and does not provide a Hindi explanation for the student.",
            "hindi",
            False,
        ),
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


@pytest.mark.parametrize(
    "action",
    [
        "ANSWER_CURRENT",
        "ANSWER_WITH_CONTEXT",
        "EXPLAIN_PREVIOUS",
        "CONTINUE_PREVIOUS",
        "GENERATE_SIMILAR",
        "TRANSFORM_PREVIOUS",
        "VERIFY_AND_CORRECT",
        "RESOLVE_FROM_SCRATCH",
    ],
)
def test_every_generation_action_uses_one_shared_language_policy(action: str) -> None:
    policy = LanguagePolicyResolver().resolve("hindi").instruction
    messages = PromptResolver().resolve(
        _generator_route("hindi"),
        "Question",
        conversation_context=f"Requested action: {action}",
    )

    assert messages[0].content.count(policy) == 1
    assert action in messages[1].content


def test_continuation_and_rewrite_keep_one_language_policy() -> None:
    policy = LanguagePolicyResolver().resolve("hinglish").instruction
    base = PromptResolver().resolve(_generator_route("hinglish"), "Question")
    continuation = build_continuation_messages(
        base,
        partial_content="Partial answer",
        policy=AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        ),
    )
    rewrite = build_rewrite_messages(base, draft_answer="English-only draft")

    assert sum(message.content.count(policy) for message in continuation) == 1
    assert sum(message.content.count(policy) for message in rewrite) == 1


@pytest.mark.parametrize("language", ["hindi", "hinglish"])
def test_english_web_context_is_preserved_beneath_requested_language_policy(
    language: str,
) -> None:
    url = "https://example.test/current-affairs"
    context = f"[Web Context]\nSource: Reserve Bank of India\nURL: {url}"

    messages = PromptResolver().resolve(
        _generator_route(language, intent="practice"),
        "Current affairs questions",
        context=context,
    )

    assert LanguagePolicyResolver().resolve(language).instruction in messages[0].content  # type: ignore[arg-type]
    assert "Reserve Bank of India" in messages[1].content
    assert url in messages[1].content


def test_classifier_prompt_supports_multilingual_input_without_language_output() -> None:
    messages = PromptResolver().resolve(
        RouteDecision(
            route_id="general.classifier.default",
            subject="general",
            task_role="classifier",
            difficulty="default",
            model="safe_mock",
            prompt="classification_semantics.md",
            overlays=["query_classifier_text.md"],
            temperature=0.0,
            max_tokens=800,
            route_source="exact",
        ),
        "profit percentage ka formula batao",
    )

    assert "English, Hindi, Hinglish" in messages[0].content
    assert "Do not infer, select" in messages[0].content
    assert "language" not in QueryClassification.model_fields


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


def test_legacy_generator_never_returns_noncompliant_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        graph_module,
        "generate_answer",
        lambda *args, **kwargs: AnswerOutput(  # noqa: ARG005
            content="The answer is entirely in English.",
            answer_source="llm",
        ),
    )

    result = graph_module.generate_answer_node(
        {
            "request_id": "language-legacy",
            "query": "Explain simple interest",
            "language": "hindi",
            "classification": {
                "intent": "explain_concept",
                "subject": "math",
                "difficulty": "basic",
                "confidence": 0.99,
            },
        }
    )

    assert result["answer"] == (
        "इस प्रश्न का विश्वसनीय उत्तर तैयार नहीं हो सका। कृपया फिर प्रयास करें।"
    )
    assert result["answer_source"] == "fallback"
    assert result["final_answer"]["language_compliant"] is True
    assert result["final_answer"]["quality_status"] == "failed_quality_gate"


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
