"""Focused contract tests for feature-gated Pattern Intelligence doubt guidance."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import config as config_module
from graphs.doubt_solver_graph import _retrieve_s3_vector_context_node
from retrieval.models import StudentRetrievalContext
from retrieval.pattern_intelligence import (
    DoubtPatternContext,
    DoubtQuestionReference,
    PatternMatchTier,
    PatternRetrievalPlan,
    PatternRuntimeMode,
    PatternRuntimeResult,
)
from schemas.doubt_solver import QueryClassification
from schemas.llm_routing import RouteDecision, RouteRequest
from services.answer_generator_service import _build_answer_messages
from services.context_retrieval.context_models import (
    ContextRetrievalRequest,
    ContextRetrievalResult,
)
from services.context_retrieval.context_retrieval_service import ContextRetrievalService
from services.doubt_solver.answer_generation_adapter import AnswerGenerationAdapter
from services.llm.orchestration.orchestrator import LlmOrchestrator, MockModelExecutor
from services.llm.orchestration.prompt_budget import estimate_text_tokens
from services.llm.orchestration.prompt_resolver import (
    PromptResolver,
    render_doubt_pattern_context,
)


def _pattern_context() -> DoubtPatternContext:
    return DoubtPatternContext(
        patternId="pattern-1",
        target=("find the ratio",),
        givens=("two quantities",),
        conditions=("keep units consistent",),
        operationHints=("simplify before comparing",),
        notSameWhen=("units are incompatible",),
        trapCues=("do not add percentages directly",),
        complexityLevel="medium",
        variationFocus="ratio comparison",
        questionReferences=(
            DoubtQuestionReference(
                questionId="reference-1",
                questionText="A related ratio question",
                options=("2:3", "3:2"),
            ),
        ),
    )


def _runtime_result() -> PatternRuntimeResult:
    return PatternRuntimeResult(
        plan=PatternRetrievalPlan(
            mode=PatternRuntimeMode.DOUBT,
            strategy="vector",
            candidateLimit=2,
        ),
        tier=PatternMatchTier.GUIDANCE_SAFE,
        selectedPatternId="pattern-1",
        doubtContext=_pattern_context(),
    )


class _FakeRuntime:
    def __init__(self) -> None:
        self.request = None

    def resolve_doubt(self, request):
        self.request = request
        return _runtime_result()


@pytest.fixture
def _pattern_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "s3_vector")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "true")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_MAX_REFERENCES", "1")
    config_module._settings = None
    yield
    config_module._settings = None


def test_context_service_keeps_pattern_guidance_structured_until_prompt_boundary(
    _pattern_enabled: None,
) -> None:
    runtime = _FakeRuntime()
    service = ContextRetrievalService(pattern_runtime=runtime)

    result = service.retrieve_context(
        ContextRetrievalRequest(
            request_id="request-1",
            query="How do I compare these ratios?",
            subject="math",
            intent="explain",
            difficulty="default",
            topic="ratio",
        )
    )

    assert result.context_text == ""
    assert result.retrieval_used is True
    assert result.doubt_pattern_context == _pattern_context()
    assert result.retrieval_context.selected_pattern_id == "pattern-1"
    assert runtime.request.max_linked_questions == 1
    assert runtime.request.difficulty is None


def test_context_service_preserves_legacy_s3_retrieval_when_pattern_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "s3_vector")
    monkeypatch.setenv("PATTERN_INTELLIGENCE_ENABLED", "false")
    config_module._settings = None

    class _LegacyRetrieval:
        def __init__(self) -> None:
            self.called = False

        def retrieve(self, _request: ContextRetrievalRequest) -> StudentRetrievalContext:
            self.called = True
            return StudentRetrievalContext.fresh_solve("legacy_path")

    legacy = _LegacyRetrieval()
    service = ContextRetrievalService(
        student_retrieval_service=legacy,
        pattern_runtime_factory=lambda: (_ for _ in ()).throw(AssertionError("must not build")),
    )

    result = service.retrieve_context(
        ContextRetrievalRequest(request_id="request-2", query="Explain this", subject="math")
    )

    assert legacy.called is True
    assert result.reason == "legacy_path"
    assert result.doubt_pattern_context is None
    config_module._settings = None


def test_context_service_falls_back_to_legacy_s3_when_canonical_guidance_is_ignored(
    _pattern_enabled: None,
) -> None:
    class _IgnoredRuntime:
        def resolve_doubt(self, _request: object) -> PatternRuntimeResult:
            return PatternRuntimeResult(
                plan=PatternRetrievalPlan(
                    mode=PatternRuntimeMode.DOUBT,
                    strategy="vector",
                    candidateLimit=2,
                ),
                warnings=("vector_confidence_insufficient",),
            )

    class _LegacyRetrieval:
        def __init__(self) -> None:
            self.called = False

        def retrieve(self, _request: ContextRetrievalRequest) -> StudentRetrievalContext:
            self.called = True
            return StudentRetrievalContext.fresh_solve("legacy_path")

    legacy = _LegacyRetrieval()
    service = ContextRetrievalService(
        student_retrieval_service=legacy,
        pattern_runtime=_IgnoredRuntime(),
    )

    result = service.retrieve_context(
        ContextRetrievalRequest(
            request_id="request-fallback-1",
            query="Explain this",
            subject="math",
        )
    )

    assert legacy.called is True
    assert result.reason == "vector_confidence_insufficient"
    assert result.doubt_pattern_context is None


def test_prompt_resolver_renders_only_answer_redacted_pattern_projection(tmp_path) -> None:
    (tmp_path / "main.md").write_text("system", encoding="utf-8")
    context = DoubtPatternContext.model_validate(
        {
            **_pattern_context().model_dump(by_alias=True),
            "answer": "ANSWER_SECRET_MUST_NOT_APPEAR",
            "solution": "SOLUTION_SECRET_MUST_NOT_APPEAR",
        }
    )
    route = RouteDecision(
        route_id="math.generator.default",
        subject="math",
        task_role="generator",
        difficulty="default",
        model="gemini_flash_light",
        prompt="main.md",
        overlays=[],
        temperature=0.2,
        max_tokens=800,
        route_source="exact",
    )

    messages = PromptResolver(prompt_root=tmp_path).resolve(
        route,
        query="Compare these ratios",
        doubt_pattern_context=context,
    )

    assert "PATTERN GUIDANCE" in messages[1].content
    assert "simplify before comparing" in messages[1].content
    assert "A related ratio question" in messages[1].content
    assert "ANSWER_SECRET_MUST_NOT_APPEAR" not in messages[1].content
    assert "SOLUTION_SECRET_MUST_NOT_APPEAR" not in messages[1].content
    assert "PATTERN GUIDANCE" not in messages[0].content


def test_orchestrated_doubt_budget_drops_references_before_core_pattern_guidance(
    _pattern_enabled: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PATTERN_INTELLIGENCE_PROMPT_MAX_INPUT_TOKENS", "1024")
    config_module._settings = None
    (tmp_path / "main.md").write_text(
        "Use concise tutoring guidance. " + "x" * 1_800,
        encoding="utf-8",
    )
    route = RouteDecision(
        route_id="math.generator.default",
        subject="math",
        task_role="generator",
        difficulty="default",
        model="gemini_flash_light",
        prompt="main.md",
        overlays=[],
        temperature=0.2,
        max_tokens=800,
        route_source="exact",
    )
    context = _pattern_context().model_copy(
        update={
            "question_references": (
                DoubtQuestionReference(
                    questionId="reference-1",
                    questionText="FIRST_REFERENCE_MARKER " + "x" * 2_400,
                ),
                DoubtQuestionReference(
                    questionId="reference-2",
                    questionText="SECOND_REFERENCE_MARKER " + "y" * 2_400,
                ),
            )
        }
    )
    executor = MockModelExecutor(content="Answer. <ANSWER_DONE>")
    orchestrator = LlmOrchestrator(
        model_executor=executor,
        route_resolver_fn=lambda _request: route,
        prompt_resolver=PromptResolver(prompt_root=tmp_path),
    )

    result = orchestrator.generate(
        route_request=RouteRequest(
            request_id="request-budget-1",
            subject="math",
            task_role="generator",
            difficulty="default",
        ),
        query="How do I compare these ratios?",
        doubt_pattern_context=context,
    )

    assert executor.last_messages is not None
    user_content = executor.last_messages[1].content
    budget = result.metadata["doubtPatternPromptBudget"]
    assert "find the ratio" in user_content
    assert "FIRST_REFERENCE_MARKER" not in user_content
    assert "SECOND_REFERENCE_MARKER" not in user_content
    assert budget["referenceTokens"] == 0
    assert result.metadata["doubtPatternGuidanceOmittedForBudget"] is False
    config_module._settings = None


def test_legacy_answer_path_applies_the_same_reference_first_budget_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classification = QueryClassification(
        intent="solve_question",
        subject="math",
        confidence=0.9,
    )
    context = _pattern_context().model_copy(
        update={
            "question_references": (
                DoubtQuestionReference(
                    questionId="reference-1",
                    questionText="LEGACY_REFERENCE_MARKER " + "x" * 2_400,
                ),
            )
        }
    )
    baseline = _build_answer_messages(
        "How do I compare these ratios?",
        classification,
        request_id="request-legacy-budget-1",
    )
    core_context = context.model_copy(update={"question_references": ()})
    max_input_tokens = (
        estimate_text_tokens(baseline[0].content)
        + estimate_text_tokens(baseline[1].content)
        + estimate_text_tokens(render_doubt_pattern_context(core_context))
        + 1
    )
    monkeypatch.setattr(
        config_module,
        "get_settings",
        lambda: SimpleNamespace(
            pattern_intelligence_prompt_max_input_tokens=max_input_tokens,
        ),
    )

    messages = _build_answer_messages(
        "How do I compare these ratios?",
        classification,
        request_id="request-legacy-budget-1",
        doubt_pattern_context=context,
    )

    assert "find the ratio" in messages[1].content
    assert "LEGACY_REFERENCE_MARKER" not in messages[1].content


def test_adapter_forwards_typed_pattern_context_only_when_present() -> None:
    calls: list[dict[str, Any]] = []

    class _Orchestrator:
        def generate(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                model="mock",
                route_decision=None,
                final_answer=None,
                content="answer",
            )

    adapter = AnswerGenerationAdapter(orchestrator=_Orchestrator())
    answer = adapter.generate(
        request_id="request-3",
        query="Compare these ratios",
        subject="math",
        intent="explain",
        difficulty="default",
        context="",
        doubt_pattern_context=_pattern_context(),
    )

    assert answer == "answer"
    assert calls[0]["doubt_pattern_context"] == _pattern_context()


def test_legacy_s3_graph_node_keeps_pattern_context_out_of_answer_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import services.context_retrieval.context_retrieval_service as retrieval_module

    class _ContextService:
        def retrieve_context(self, _request: ContextRetrievalRequest) -> ContextRetrievalResult:
            return ContextRetrievalResult(
                context_text="",
                item_count=1,
                retrieval_used=True,
                reason="pattern_guidance_selected",
                retrieval_context=StudentRetrievalContext(
                    mode="pattern_assist",
                    selectedPatternId="pattern-1",
                ),
                doubt_pattern_context=_pattern_context(),
            )

    monkeypatch.setattr(
        retrieval_module,
        "get_context_retrieval_service",
        lambda: _ContextService(),
    )
    result = _retrieve_s3_vector_context_node(
        {
            "request_id": "request-4",
            "query": "Compare these ratios",
            "classification": {"subject": "math", "intent": "explain"},
        }
    )

    assert result["answer_context"] is None
    assert result["doubt_pattern_context"]["patternId"] == "pattern-1"
