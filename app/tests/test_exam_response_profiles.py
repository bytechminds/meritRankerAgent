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
from schemas.doubt_solver import AnswerOutput, DoubtSolverRequest, QueryClassification
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
    return yaml.safe_load(
        DEFAULT_EXAM_RESPONSE_PROFILES_PATH.read_text(encoding="utf-8")
    )


def _route(
    *,
    task_role: str = "generator",
    exam: str | None = None,
    exam_stage: str | None = None,
) -> RouteDecision:
    return RouteDecision(
        route_id=f"math.{task_role}.default",
        subject="math",
        task_role=task_role,
        difficulty="default",
        intent="solve",
        exam=exam,
        exam_stage=exam_stage,
        model="safe_mock",
        prompt=(
            "subjects/math_generator.md"
            if task_role == "generator"
            else "query_classifier.md"
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
    ("exam_id", "family", "override"),
    [
        ("SSC_CGL", "SSC", False),
        ("SSC_CHSL", "SSC", True),
        ("RRB_NTPC", "RAILWAY_BASIC", True),
        ("SSC_JE", "TECHNICAL", False),
        ("RRB_JE", "TECHNICAL", False),
    ],
)
def test_exam_family_and_override_resolution(
    resolver: ExamResponseProfileResolver,
    exam_id: str,
    family: str,
    override: bool,
) -> None:
    result = resolver.resolve(exam_id, None)

    assert result.exam_family == family
    assert result.exam_override_applied is override


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

    assert prelims.exam_family == "UPSC_PSC"
    assert prelims.exam_override_applied
    assert prelims.stage_override_applied
    assert prelims.stage == "PRELIMS"
    assert mains.stage == "MAINS"
    assert prelims.compact_instruction != mains.compact_instruction


def test_unknown_exam_uses_general_fallback(
    resolver: ExamResponseProfileResolver,
) -> None:
    result = resolver.resolve("UNSUPPORTED_STATE_EXAM", "MAINS")

    assert result.exam_family == "GENERAL_GOVT"
    assert result.exam_id == "UNSUPPORTED_STATE_EXAM"
    assert result.canonical_exam_id is None
    assert result.fallback_used
    assert result.source == "unknown_fallback"
    assert not result.override_applied


def test_missing_exam_resolves_but_is_not_implicitly_selected(
    resolver: ExamResponseProfileResolver,
) -> None:
    result = resolver.resolve(None, None)

    assert result.exam_family == "GENERAL_GOVT"
    assert result.canonical_exam_id is None
    assert result.source == "missing_fallback"


def test_resolved_profile_has_no_difficulty_field() -> None:
    assert "difficulty" not in ResolvedExamResponseProfile.model_fields


def test_duplicate_exam_mapping_fails(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["examMappings"]["TECHNICAL"].append("SSC_CGL")

    with pytest.raises(ValidationError, match="mapped to both"):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_alias_cycle_fails(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["aliases"]["CYCLE_A"] = "CYCLE_B"
    invalid["aliases"]["CYCLE_B"] = "CYCLE_A"

    with pytest.raises(ValidationError, match="Alias cycle"):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_invalid_family_reference_fails(raw_config: dict[str, Any]) -> None:
    invalid = copy.deepcopy(raw_config)
    invalid["examMappings"]["UNKNOWN_FAMILY"] = ["EXAM_X"]

    with pytest.raises(ValidationError, match="unknown family"):
        ExamResponseProfilesConfig.model_validate(invalid)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("families", "SSC", "response"), "Family response"),
        (("examOverrides", "SSC_CHSL"), "Exam override"),
        (("stageOverrides", "UPSC_CSE", "PRELIMS"), "Stage override"),
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
        (("families", "SSC", "response"), "x" * 191, "Family response"),
        (("examOverrides", "SSC_CHSL"), "x" * 121, "Exam override"),
        (
            ("stageOverrides", "UPSC_CSE", "PRELIMS"),
            "x" * 121,
            "Stage override",
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
    invalid["limits"]["maxResolvedContextChars"] = 100

    with pytest.raises(ValidationError, match="Resolved context"):
        ExamResponseProfilesConfig.model_validate(invalid)


def test_all_production_profiles_fit_character_and_token_budget(
    resolver: ExamResponseProfileResolver,
) -> None:
    profiles = []
    for exams in resolver.config.exam_mappings.values():
        for exam in exams:
            profiles.append(resolver.resolve(exam, None))
            for stage in resolver.config.stage_overrides.get(exam, {}):
                profiles.append(resolver.resolve(exam, stage))

    lengths = [len(profile.compact_instruction) for profile in profiles]
    rough_token_estimates = [math.ceil(length / 4) for length in lengths]
    assert max(lengths) <= resolver.config.limits.max_resolved_context_chars
    assert max(rough_token_estimates) <= 65
    assert sum(lengths) / len(lengths) < 150


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
    assert system.count("Exam response guidance:") == 1
    assert resolver.resolve("SSC_CHSL", None).compact_instruction in system
    assert "only answer" in messages[1].content.lower()
    assert "SSC_CHSL" not in "\n".join(message.content for message in messages)
    assert "competitive-exam shortcut style" in system


def test_missing_exam_leaves_generator_prompt_without_dynamic_guidance(
    resolver: ExamResponseProfileResolver,
) -> None:
    messages = PromptResolver(
        prompt_root=_PROMPT_ROOT,
        exam_profile_resolver=resolver,
    ).resolve(_route(), "Question")

    assert "Exam response guidance:" not in messages[0].content


def test_classifier_prompt_never_receives_exam_guidance(
    resolver: ExamResponseProfileResolver,
) -> None:
    messages = PromptResolver(
        prompt_root=_PROMPT_ROOT,
        exam_profile_resolver=resolver,
    ).resolve(_route(task_role="classifier", exam="SSC_CGL"), "Question")

    assert "Exam response guidance:" not in messages[0].content


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


def test_legacy_graph_passes_run_config_exam_without_state_drift(
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
        "classification": QueryClassification(
            intent="solve_question", confidence=0.99
        ).model_dump(),
        "answer_context": None,
    }

    graph_module.generate_answer_node(
        state,  # type: ignore[arg-type]
        {"configurable": {"exam_id": "SSC_CGL", "exam_stage": "TIER_2"}},
    )

    assert captured == {
        "exam_id": "SSC_CGL",
        "exam_stage": "TIER_2",
        "request_id": "request-legacy",
    }
    assert "exam_id" not in state
    assert "exam_stage" not in state


def test_orchestrated_graph_passes_run_config_exam_without_state_drift() -> None:
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
        "query": "Question",
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

    result = graph.invoke(
        state,
        config={
            "configurable": {"exam_id": "UPSC_CSE", "exam_stage": "PRELIMS"}
        },
    )

    assert adapter.kwargs["exam_id"] == "UPSC_CSE"
    assert adapter.kwargs["exam_stage"] == "PRELIMS"
    assert result["answer"] == "Answer"
    assert "exam_id" not in result
    assert "exam_stage" not in result


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
        content="**Answer:** 10",
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
            "query": "What is 20% of 50?",
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
        config={"configurable": {"exam_id": "SSC_CGL"}},
    )

    assert result["answer"] == "**Answer:** 10"
    assert executor.call_count == 1
    assert executor.last_messages is not None
    prompt = "\n".join(message.content for message in executor.last_messages)
    assert prompt.count("Exam response guidance:") == 1
    assert "SSC_CGL" not in prompt
    assert "exam_id" not in result
    assert "exam_stage" not in result


def test_request_schema_additions_are_optional_and_response_neutral() -> None:
    baseline = DoubtSolverRequest(mode="doubt_solver", query="Question")
    selected = DoubtSolverRequest(
        mode="doubt_solver",
        query="Question",
        exam_id="SSC_CGL",
        exam_stage="TIER_2",
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
        logging.INFO,
        logger="services.doubt_solver.exam_response_profile",
    ):
        result = resolver.resolve("SSC_CGL", "TIER_2", request_id="safe-request")

    message = caplog.records[-1].getMessage()
    assert "safe-request" in message
    assert "canonical_exam_id=SSC_CGL" in message
    assert "exam_family=SSC" in message
    assert "compact_instruction_chars=" in message
    assert result.compact_instruction not in message
    assert secret_text not in message


def test_static_guidance_does_not_claim_trends_or_force_shortcuts(
    resolver: ExamResponseProfileResolver,
) -> None:
    text = " ".join(
        [family.response for family in resolver.config.families.values()]
        + list(resolver.config.exam_overrides.values())
        + [
            direction
            for stages in resolver.config.stage_overrides.values()
            for direction in stages.values()
        ]
    ).lower()

    assert "latest trend" not in text
    assert "pyq" not in text
    assert "syllabus" not in text
    assert "always use" not in text
    assert "validated" in text
