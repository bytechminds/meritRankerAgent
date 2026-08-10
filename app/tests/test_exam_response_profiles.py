"""Exam response profile resolution, validation, and prompt integration tests."""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import graphs.doubt_solver_graph as graph_module
from schemas.doubt_solver import (
    AnswerOutput,
    CanonicalLanguage,
    DoubtSolverRequest,
    QueryClassification,
)
from schemas.exam_response_profiles import (
    ExamResponseProfilesConfig,
    ResolvedExamResponseProfile,
)
from schemas.llm_routing import RouteDecision, RouteRequest
from services.answer_generator_service import _build_answer_messages
from services.doubt_solver.answer_completion import (
    AnswerCompletionPolicy,
    build_continuation_messages,
)
from services.doubt_solver.answer_generation_adapter import AnswerGenerationAdapter
from services.doubt_solver.answer_quality import build_rewrite_messages
from services.doubt_solver.exam_response_profile import (
    DEFAULT_EXAM_RESPONSE_PROFILES_PATH,
    ExamResponseProfileConfigError,
    ExamResponseProfileResolver,
)
from services.llm.orchestration.orchestrator import create_mock_orchestrator_for_tests
from services.llm.orchestration.prompt_resolver import PromptResolver

_PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


@pytest.fixture(scope="module")
def resolver() -> ExamResponseProfileResolver:
    return ExamResponseProfileResolver()


@pytest.fixture
def raw_config() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_EXAM_RESPONSE_PROFILES_PATH.read_text(encoding="utf-8"))


def _route(
    *,
    task_role: str = "generator",
    exam: str | None = None,
    exam_stage: str | None = None,
    language: CanonicalLanguage = "english",
) -> RouteDecision:
    return RouteDecision(
        route_id=f"math.{task_role}.default",
        subject="math",
        task_role=task_role,
        difficulty="default",
        intent="solve",
        exam=exam,
        exam_stage=exam_stage,
        language=language,
        model="safe_mock",
        prompt=(
            "subjects/math_generator.md" if task_role == "generator" else "query_classifier.md"
        ),
        overlays=[],
        intent_overlays={},
        temperature=0.0,
        max_tokens=800,
        provider_options={},
        fallback_attempts=[],
        route_source="exact",
    )


@pytest.mark.parametrize(
    ("exam_id", "stage", "family", "category", "provenance"),
    [
        ("SSC_MTS", None, "SSC", "BASIC_OBJECTIVE", "exam"),
        ("SSC_CHSL", None, "SSC", "STANDARD_OBJECTIVE", "family"),
        ("SSC_CGL", "TIER_1", "SSC", "STANDARD_OBJECTIVE", "family"),
        ("SSC_CGL", "TIER_2", "SSC", "ANALYTICAL_OBJECTIVE", "exam_stage"),
        ("RRB_GROUP_D", None, "RAILWAY", "BASIC_OBJECTIVE", "family"),
        ("RRB_NTPC", None, "RAILWAY", "STANDARD_OBJECTIVE", "exam"),
        ("IBPS_CLERK", None, "BANKING", "STANDARD_OBJECTIVE", "family"),
        ("SBI_PO", None, "BANKING", "ANALYTICAL_OBJECTIVE", "exam"),
        ("UPSC_CSE", "PRELIMS", "UPSC", "CONCEPTUAL_OBJECTIVE", "exam_stage"),
        (
            "UPSC_CSE",
            "MAINS",
            "UPSC",
            "DESCRIPTIVE_ANALYTICAL",
            "exam_stage",
        ),
        (
            "STATE_PSC",
            "MAINS",
            "STATE_PSC",
            "DESCRIPTIVE_ANALYTICAL",
            "exam_stage",
        ),
        ("CTET", None, "TEACHING", "PEDAGOGY_CONCEPTUAL", "family"),
        (
            "GATE",
            None,
            "ENGINEERING",
            "TECHNICAL_PROBLEM_SOLVING",
            "family",
        ),
        ("CAT", None, "MANAGEMENT", "ADVANCED_APTITUDE", "family"),
    ],
)
def test_representative_exam_category_resolution(
    resolver: ExamResponseProfileResolver,
    exam_id: str,
    stage: str | None,
    family: str,
    category: str,
    provenance: str,
) -> None:
    result = resolver.resolve(exam_id, stage)

    assert result.exam_family == family
    assert result.response_category == category
    assert result.provenance == provenance


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("IBPS_CLERK", "IBPS_CSA"),
        ("SBI_CLERK", "SBI_JUNIOR_ASSOCIATE"),
    ],
)
def test_approved_aliases_resolve(
    resolver: ExamResponseProfileResolver,
    alias: str,
    canonical: str,
) -> None:
    result = resolver.resolve(alias, None)

    assert result.canonical_exam_id == canonical
    assert result.alias_applied
    assert result.source == "alias"


def test_upsc_stages_apply_distinct_guidance(
    resolver: ExamResponseProfileResolver,
) -> None:
    prelims = resolver.resolve("UPSC_CSE", "preliminary")
    mains = resolver.resolve("upsc-cse", "main")

    assert prelims.exam_family == "UPSC"
    assert prelims.stage == "PRELIMS"
    assert mains.stage == "MAINS"
    assert prelims.response_category == "CONCEPTUAL_OBJECTIVE"
    assert mains.response_category == "DESCRIPTIVE_ANALYTICAL"
    assert prelims.provenance == mains.provenance == "exam_stage"
    assert prelims.compact_instruction != mains.compact_instruction


def test_unknown_exam_uses_default_category(
    resolver: ExamResponseProfileResolver,
) -> None:
    result = resolver.resolve("UNSUPPORTED_STATE_EXAM", "MAINS")

    assert result.exam_family is None
    assert result.exam_id == "UNSUPPORTED_STATE_EXAM"
    assert result.canonical_exam_id is None
    assert result.response_category == "STANDARD_OBJECTIVE"
    assert result.provenance == "default"
    assert result.fallback_used
    assert result.source == "unknown_fallback"


def test_missing_exam_resolves_but_is_not_implicitly_selected(
    resolver: ExamResponseProfileResolver,
) -> None:
    result = resolver.resolve(None, None)

    assert result.exam_family is None
    assert result.canonical_exam_id is None
    assert result.response_category == "STANDARD_OBJECTIVE"
    assert result.provenance == "default"
    assert result.source == "missing_fallback"


def test_resolved_profile_has_no_difficulty_field() -> None:
    assert "difficulty" not in ResolvedExamResponseProfile.model_fields


def test_duplicate_exam_mapping_fails(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["familyMappings"]["ENGINEERING"]["exams"].append("SSC_CGL")

    with pytest.raises(ValidationError, match="mapped to both"):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_alias_cycle_fails(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["aliases"]["CYCLE_A"] = "CYCLE_B"
    invalid["aliases"]["CYCLE_B"] = "CYCLE_A"

    with pytest.raises(ValidationError, match="Alias cycle"):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_unknown_exam_override_fails(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["examMappings"]["EXAM_X"] = {"category": "STANDARD_OBJECTIVE"}

    with pytest.raises(ValidationError, match="unknown exam"):
        ExamResponseProfilesConfig.model_validate(invalid)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("categories", "STANDARD_OBJECTIVE", "guide"), "Category guide"),
        (
            ("examMappings", "SSC_CGL", "stageOverrides", "TIER_2", "appendGuide"),
            "Stage appendGuide",
        ),
    ],
)
def test_whitespace_only_directions_fail_validation(
    raw_config: dict[str, Any],
    path: tuple[str, ...],
    message: str,
) -> None:
    invalid = copy.deepcopy(raw_config)
    target: dict[str, Any] = invalid
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = " \n\t "

    with pytest.raises(ValidationError, match=message):
        ExamResponseProfilesConfig.model_validate(invalid)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("categories", "STANDARD_OBJECTIVE", "guide"),
            "x" * 191,
            "Category guide",
        ),
        (
            ("examMappings", "SSC_CGL", "stageOverrides", "TIER_2", "appendGuide"),
            "x" * 91,
            "Stage appendGuide",
        ),
    ],
)
def test_direction_character_limits_are_enforced(
    raw_config: dict[str, Any],
    path: tuple[str, ...],
    value: str,
    message: str,
) -> None:
    invalid = copy.deepcopy(raw_config)
    target: dict[str, Any] = invalid
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_resolved_character_limit_is_enforced(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["limits"]["maxResolvedContextChars"] = 120

    with pytest.raises(ValidationError, match="Resolved context"):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_unsupported_category_is_rejected(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["familyMappings"]["SSC"]["category"] = "SSC_SPECIAL"

    with pytest.raises(ValidationError, match="literal_error"):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_empty_exam_mapping_is_rejected(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["examMappings"]["SSC_CHSL"] = {}

    with pytest.raises(ValidationError, match="must define"):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_all_production_profiles_fit_character_and_token_budget(
    resolver: ExamResponseProfileResolver,
) -> None:
    profiles: list[ResolvedExamResponseProfile] = []
    for family in resolver.config.family_mappings.values():
        for exam in family.exams:
            profiles.append(resolver.resolve(exam, None))
            mapping = resolver.config.exam_mappings.get(exam)
            for stage in mapping.stage_overrides if mapping is not None else {}:
                profiles.append(resolver.resolve(exam, stage))

    lengths = [len(profile.compact_instruction) for profile in profiles]
    guide_token_estimates = [
        math.ceil(len(profile.guide) / 4) for profile in resolver.config.categories.values()
    ]
    override_token_estimates = [
        math.ceil(len(override) / 4)
        for mapping in resolver.config.exam_mappings.values()
        for override in [
            mapping.append_guide,
            *(stage.append_guide for stage in mapping.stage_overrides.values()),
        ]
        if override is not None
    ]
    assert max(lengths) <= resolver.config.limits.max_resolved_context_chars
    assert max(guide_token_estimates) <= 60
    assert max(override_token_estimates) <= 25
    assert sum(lengths) / len(lengths) < 140


def test_all_supported_exams_have_one_family_membership(
    resolver: ExamResponseProfileResolver,
) -> None:
    exams = [exam for mapping in resolver.config.family_mappings.values() for exam in mapping.exams]

    assert len(exams) == 90
    assert len(exams) == len(set(exams))


def test_yaml_is_loaded_once_per_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = Path.read_text

    def counting_read(path: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read)
    local = ExamResponseProfileResolver()
    local.resolve("SSC_CGL", None)
    local.resolve("UPSC_CSE", "PRELIMS")

    assert calls == 1


def test_generator_prompt_receives_guidance_exactly_once(
    resolver: ExamResponseProfileResolver,
) -> None:
    messages = PromptResolver(
        prompt_root=_PROMPT_ROOT,
        exam_profile_resolver=resolver,
    ).resolve(
        _route(exam="SSC_CHSL"),
        "Only answer: What is 20% of 50?",
        request_id="exam-profile-test",
    )

    system = messages[0].content
    assert system.count("EXAM RESPONSE GUIDANCE") == 1
    assert resolver.resolve("SSC_CHSL", None).compact_instruction in system
    assert "only answer" in messages[1].content.lower()
    assert "SSC_CHSL" not in "\n".join(message.content for message in messages)
    assert "STANDARD_OBJECTIVE" not in system
    assert "familyMappings" not in system
    assert "competitive-exam shortcut style" in system


def test_missing_exam_leaves_generator_prompt_without_dynamic_guidance(
    resolver: ExamResponseProfileResolver,
) -> None:
    messages = PromptResolver(
        prompt_root=_PROMPT_ROOT,
        exam_profile_resolver=resolver,
    ).resolve(_route(), "Question")

    assert "EXAM RESPONSE GUIDANCE" not in messages[0].content


def test_classifier_prompt_never_receives_exam_guidance(
    resolver: ExamResponseProfileResolver,
) -> None:
    messages = PromptResolver(
        prompt_root=_PROMPT_ROOT,
        exam_profile_resolver=resolver,
    ).resolve(_route(task_role="classifier", exam="SSC_CGL"), "Question")

    assert "EXAM RESPONSE GUIDANCE" not in messages[0].content


def test_stage_override_is_appended_once_and_language_remains_separate(
    resolver: ExamResponseProfileResolver,
) -> None:
    profile = resolver.resolve("SSC_CGL", "TIER_2")
    messages = PromptResolver(
        prompt_root=_PROMPT_ROOT,
        exam_profile_resolver=resolver,
    ).resolve(
        _route(exam="SSC_CGL", exam_stage="TIER_2", language="hindi"),
        "Question",
    )

    system = messages[0].content
    assert profile.optional_override is not None
    assert system.count(profile.category_guide) == 1
    assert system.count(profile.optional_override) == 1
    assert system.count("EXAM RESPONSE GUIDANCE") == 1
    assert "देवनागरी प्रयोग करें" in system
    assert system.index("EXAM RESPONSE GUIDANCE") < system.index("देवनागरी प्रयोग करें")


def test_legacy_and_orchestrated_prompts_use_same_compact_profile(
    resolver: ExamResponseProfileResolver,
) -> None:
    expected = resolver.resolve("UPSC_CSE", "PRELIMS").compact_instruction
    orchestrated = PromptResolver(
        prompt_root=_PROMPT_ROOT,
        exam_profile_resolver=resolver,
    ).resolve(_route(exam="UPSC_CSE", exam_stage="PRELIMS"), "Question")
    legacy = _build_answer_messages(
        "Question",
        QueryClassification(intent="solve_question", confidence=0.99),
        exam_id="UPSC_CSE",
        exam_stage="PRELIMS",
    )

    assert orchestrated[0].content.count(expected) == 1
    assert legacy[0].content.count(expected) == 1


def test_continuation_and_rewrite_preserve_guidance_exactly_once(
    resolver: ExamResponseProfileResolver,
) -> None:
    expected = resolver.resolve("SSC_CGL", "TIER_2").compact_instruction
    base_messages = PromptResolver(
        prompt_root=_PROMPT_ROOT,
        exam_profile_resolver=resolver,
    ).resolve(_route(exam="SSC_CGL", exam_stage="TIER_2"), "Question")
    continued = build_continuation_messages(
        base_messages,
        partial_content="Partial answer",
        policy=AnswerCompletionPolicy(
            marker="<ANSWER_DONE>",
            continuation_enabled=True,
            continuation_max_attempts=1,
        ),
    )
    rewritten = build_rewrite_messages(base_messages, draft_answer="Draft answer")

    assert sum(message.content.count(expected) for message in continued) == 1
    assert sum(message.content.count(expected) for message in rewritten) == 1


def test_adapter_carries_exam_without_changing_route_selection() -> None:
    class _Orchestrator:
        def __init__(self) -> None:
            self.route_request = None

        def generate(self, *, route_request, **kwargs):  # noqa: ANN001, ANN003
            del kwargs
            self.route_request = route_request
            return type("Result", (), {"content": "Answer", "model": "safe_mock"})()

    orchestrator = _Orchestrator()
    adapter = AnswerGenerationAdapter(orchestrator=orchestrator)  # type: ignore[arg-type]

    adapter.generate(
        request_id="request-1",
        query="Question",
        subject="math",
        intent="solve",
        difficulty="default",
        context="",
        exam_id="ssc-cgl",
        exam_stage="tier 2",
    )

    assert orchestrator.route_request.exam == "ssc-cgl"
    assert orchestrator.route_request.exam_stage == "tier 2"
    assert orchestrator.route_request.subject == "math"
    assert orchestrator.route_request.difficulty == "default"


def test_legacy_graph_passes_state_exam_without_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_generate(
        query: str,
        classification: QueryClassification,
        context: str | None = None,
        **kwargs: Any,
    ) -> AnswerOutput:
        del query, classification, context
        captured.update(kwargs)
        return AnswerOutput(content="Answer", answer_source="mock")

    monkeypatch.setattr(graph_module, "generate_answer", fake_generate)
    state = {
        "request_id": "request-legacy",
        "query": "Question",
        "language": "english",
        "exam_id": "SSC_CGL",
        "exam_stage": "TIER_2",
        "exam_profile_id": "SSC_CGL#TIER_2",
        "classification": QueryClassification(
            intent="solve_question", confidence=0.99
        ).model_dump(),
        "answer_context": None,
    }

    graph_module.generate_answer_node(state)  # type: ignore[arg-type]

    assert captured == {
        "exam_id": "SSC_CGL",
        "exam_stage": "TIER_2",
        "exam_profile_id": "SSC_CGL#TIER_2",
        "language": "english",
        "request_id": "request-legacy",
    }
    assert state["exam_id"] == "SSC_CGL"
    assert state["exam_stage"] == "TIER_2"


def test_orchestrated_graph_passes_state_exam_without_reconstruction() -> None:
    class _Adapter:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def generate(self, **kwargs: Any) -> str:
            self.kwargs = kwargs
            return "Answer"

    adapter = _Adapter()
    graph = graph_module.build_orchestrated_doubt_solver_graph(adapter)
    state = {
        "request_id": "request-orchestrated",
        "actor_id": "student-1",
        "query": "Question",
        "original_query": "Question",
        "language": "english",
        "exam_id": "UPSC_CSE",
        "exam_stage": "PRELIMS",
        "exam_profile_id": "UPSC_CSE#PRELIMS",
        "classification": {
            "subject": "math",
            "intent": "solve",
            "difficulty": "default",
            "retrieval_required": False,
        },
        "retrieval_context": {},
        "context_text": "",
        "answer": None,
    }

    result = graph.invoke(state)

    assert adapter.kwargs["exam_id"] == "UPSC_CSE"
    assert adapter.kwargs["exam_stage"] == "PRELIMS"
    assert adapter.kwargs["exam_profile_id"] == "UPSC_CSE#PRELIMS"
    assert result["answer"] == "Answer"
    assert result["exam_id"] == "UPSC_CSE"
    assert result["exam_stage"] == "PRELIMS"


def test_full_isolated_orchestrated_flow_applies_profile_safely(
    tmp_path: Path,
    resolver: ExamResponseProfileResolver,
) -> None:
    prompt_file = tmp_path / "generator.md"
    prompt_file.write_text("Global correctness policy.", encoding="utf-8")

    def resolve_route(request: RouteRequest) -> RouteDecision:
        return RouteDecision(
            route_id="math.generator.default",
            subject=request.subject,
            task_role=request.task_role,
            difficulty=request.difficulty,
            intent=request.intent,
            exam=request.exam,
            exam_stage=request.exam_stage,
            model="safe_mock",
            prompt="generator.md",
            temperature=0.0,
            max_tokens=500,
            route_source="safe_mock",
        )

    orchestrator, executor = create_mock_orchestrator_for_tests(
        content="**Answer:** 10\n\nUsing percentage = part per hundred, 20% of 50 is 10.",
        prompt_resolver=PromptResolver(
            prompt_root=tmp_path,
            exam_profile_resolver=resolver,
        ),
        route_resolver_fn=resolve_route,
    )
    graph = graph_module.build_orchestrated_doubt_solver_graph(
        AnswerGenerationAdapter(orchestrator=orchestrator)
    )

    result = graph.invoke(
        {
            "request_id": "isolated-exam-flow",
            "actor_id": "student-1",
            "query": "What is 20% of 50?",
            "original_query": "What is 20% of 50?",
            "language": "english",
            "exam_id": "SSC_CGL",
            "exam_stage": None,
            "classification": {
                "subject": "math",
                "intent": "solve",
                "difficulty": "default",
                "retrieval_required": False,
            },
            "retrieval_context": {},
            "context_text": "",
            "answer": None,
        },
    )

    assert result["answer"] == (
        "**Answer:** 10\n\nUsing percentage = part per hundred, 20% of 50 is 10."
    )
    assert executor.call_count == 1
    assert executor.last_messages is not None
    prompt = "\n".join(message.content for message in executor.last_messages)
    assert prompt.count("EXAM RESPONSE GUIDANCE") == 1
    assert "SSC_CGL" not in prompt
    assert result["exam_id"] == "SSC_CGL"
    assert result["exam_stage"] is None


def test_exam_request_fields_are_optional_and_response_neutral() -> None:
    request_ids = {
        "user_id": "local-user",
        "conversation_id": "conversation-1",
        "turn_id": "turn-1",
    }
    baseline = DoubtSolverRequest(mode="doubt_solver", query="Question", **request_ids)
    selected = DoubtSolverRequest(
        mode="doubt_solver",
        query="Question",
        exam_id="SSC_CGL",
        exam_stage="TIER_2",
        **request_ids,
    )

    assert baseline.exam_id is None
    assert baseline.exam_stage is None
    assert selected.exam_id == "SSC_CGL"
    assert selected.exam_stage == "TIER_2"


def test_malformed_yaml_fails_loader_clearly(tmp_path: Path) -> None:
    invalid = tmp_path / "exam_response_profiles.yaml"
    invalid.write_text("families: [", encoding="utf-8")

    with pytest.raises(ExamResponseProfileConfigError, match="Unable to load"):
        ExamResponseProfileResolver(invalid)


def test_resolution_log_contains_only_safe_profile_metadata(
    resolver: ExamResponseProfileResolver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_text = "do-not-log-this-guidance"
    with caplog.at_level(
        logging.DEBUG,
        logger="services.doubt_solver.exam_response_profile",
    ):
        result = resolver.resolve("SSC_CGL", "TIER_2", request_id="safe-request")

    message = caplog.records[-1].getMessage()
    assert "safe-request" in message
    assert "canonical_exam_id=SSC_CGL" in message
    assert "exam_family=SSC" in message
    assert "response_category=ANALYTICAL_OBJECTIVE" in message
    assert "provenance=exam_stage" in message
    assert "compact_instruction_chars=" in message
    assert result.compact_instruction not in message
    assert secret_text not in message


def test_static_guidance_does_not_claim_trends_or_force_shortcuts(
    resolver: ExamResponseProfileResolver,
) -> None:
    text = " ".join(
        [profile.guide for profile in resolver.config.categories.values()]
        + [
            append_guide
            for mapping in resolver.config.exam_mappings.values()
            for append_guide in [
                mapping.append_guide,
                *(stage.append_guide for stage in mapping.stage_overrides.values()),
            ]
            if append_guide is not None
        ]
    ).lower()

    assert "latest trend" not in text
    assert "pyq" not in text
    assert "syllabus" not in text
    assert "always use" not in text
    assert "fully reliable" in text
