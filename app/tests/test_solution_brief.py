"""
tests/test_solution_brief.py
-----------------------------
SolutionBrief schema and deterministic builder tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import config as cfg_module
from services.context_retrieval.context_models import ContextRetrievalRequest, RetrievedContextItem
from services.context_retrieval.context_retrieval_service import (
    LANE_SUBJECT_ONLY,
    LANE_SUBJECT_TOPIC,
    ContextRetrievalService,
)
from services.solution_brief.models import SolutionBrief
from services.solution_brief.planner_policy import should_run_llm_planner
from services.solution_brief.solution_brief_builder import SolutionBriefBuilder
from tools.web_search.models import WebSearchItem


def _request(**overrides) -> ContextRetrievalRequest:
    base = {
        "request_id": "brief-1",
        "query": "Explain average over overlapping periods for SSC",
        "subject": "math",
        "intent": "explain",
        "difficulty": "intermediate",
    }
    base.update(overrides)
    return ContextRetrievalRequest(**base)


def _kb_item(**meta_overrides) -> RetrievedContextItem:
    meta = {
        "subject": "QUANT",
        "patternTopicKey": "AVERAGE",
        "patternFamilyKey": "OVERLAP",
        "conceptTags": "TOTAL,CUMULATIVE_AVERAGE",
        "traps": "Do not average the averages directly",
        "solvingStyle": "Convert each average into total",
        "answerStyle": "step_by_step",
    }
    meta.update(meta_overrides)
    return RetrievedContextItem(
        text="Use cumulative totals for overlapping ranges.",
        score=0.9,
        source_id="pat-internal-001",
        metadata=meta,
        match_lane=LANE_SUBJECT_ONLY,
        risk="family_mismatch",
    )


class TestSolutionBriefSchema:
    def test_required_minimal_keys(self) -> None:
        fields = set(SolutionBrief.model_fields.keys())
        assert fields == {
            "subject",
            "topic",
            "given",
            "find",
            "context",
            "core_concepts",
            "solution_approach",
            "risk_flags",
            "generator_instructions",
        }

    def test_uses_solution_approach_not_approach(self) -> None:
        assert "solution_approach" in SolutionBrief.model_fields
        assert "approach" not in SolutionBrief.model_fields

    def test_rejects_noisy_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            SolutionBrief.model_validate(
                {
                    "subject": "Math",
                    "patternTopicKey": "AVERAGE",
                }
            )

    def test_empty_fields_are_safe(self) -> None:
        brief = SolutionBrief()
        assert brief.subject == ""
        assert brief.given == []
        assert brief.solution_approach == []


class TestSolutionBriefBuilderKb:
    def test_maps_subject_topic_and_concepts(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(topic="Average"),
            kb_items=[_kb_item()],
        )
        assert result.used is True
        assert result.brief is not None
        assert result.brief.subject == "Math"
        assert result.brief.topic == "Average"
        assert any("Total" in c or "Cumulative Average" in c for c in result.brief.core_concepts)

    def test_traps_and_solving_style_mapped(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(),
            kb_items=[_kb_item()],
        )
        brief = result.brief
        assert brief is not None
        assert any("average" in flag.lower() for flag in brief.risk_flags)
        assert brief.solution_approach
        assert brief.generator_instructions

    def test_no_internal_ids_or_scores_in_output(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(),
            kb_items=[_kb_item()],
        )
        text = result.brief_text.lower()
        assert "pattern id" not in text
        assert "pat-internal" not in text
        assert "patterntopickey" not in text
        assert "patternfamilykey" not in text
        assert "bedrock" not in text
        assert "tavily" not in text

    def test_basic_no_context_skips_brief(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(difficulty="basic", query="2 + 2"),
            kb_items=[],
            web_items=[],
        )
        assert result.used is False
        assert result.brief_text == ""

    def test_advanced_always_builds_brief(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(difficulty="advanced", query="Hard puzzle"),
            kb_items=[],
            web_items=[],
        )
        assert result.used is True
        assert "[Solution Brief]" in result.brief_text


class TestPlannerPolicy:
    def test_no_planner_for_basic(self) -> None:
        assert should_run_llm_planner(_request(difficulty="basic")) is False

    def test_no_planner_for_intermediate(self) -> None:
        assert should_run_llm_planner(_request(difficulty="intermediate")) is False

    def test_no_planner_for_advanced(self) -> None:
        assert should_run_llm_planner(_request(difficulty="advanced")) is False


class TestSolutionBriefWebComposition:
    def test_web_and_brief_compose(self) -> None:
        web_items = [
            WebSearchItem(
                title="Latest RBI policy update",
                url="https://example.com/rbi",
                snippet="RBI kept the repo rate unchanged in the latest review.",
                source="example.com",
                published_at="2026-05-01",
            )
        ]
        builder = SolutionBriefBuilder()
        brief_result = builder.build(
            _request(subject="general", need_web_search=True, web_search_reason="current_economy"),
            web_items=web_items,
        )
        from tools.web_search.formatter import format_selected_web_context

        web_section = format_selected_web_context(
            web_items,
            reason="current_economy",
            search_query="latest RBI repo rate",
            max_chars=1200,
        )
        composed = builder.compose_context_text(
            brief_text=brief_result.brief_text,
            web_section=web_section,
            max_chars=2000,
        )
        assert "[Solution Brief]" in composed
        assert "[Web Context]" in composed
        assert "Content:" in composed


class TestSolutionBriefMetadataRobustness:
    def test_sparse_metadata_does_not_throw(self) -> None:
        item = RetrievedContextItem(
            text="Relative speed when objects move in opposite directions.",
            score=0.97,
            metadata={"patternTopicKey": "TIME_SPEED_DISTANCE"},
            match_lane=LANE_SUBJECT_ONLY,
        )
        result = SolutionBriefBuilder().build(
            _request(topic="TIME_SPEED_DISTANCE", difficulty="intermediate"),
            kb_items=[item],
        )
        assert result.used is True
        assert len(result.brief_text) > 0

    def test_concept_tags_as_list(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(),
            kb_items=[_kb_item(conceptTags=["SPEED", "TIME", "DISTANCE"])],
        )
        assert result.brief is not None
        assert result.brief.core_concepts

    def test_extra_metadata_ignored(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(),
            kb_items=[
                _kb_item(
                    patternId="internal-id",
                    score=0.99,
                    metadata={"extraField": {"nested": True}},
                )
            ],
        )
        text = result.brief_text.lower()
        assert "internal-id" not in text
        assert "extrafield" not in text
        assert "nested" not in text

    def test_invalid_metadata_types_skipped_safely(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(),
            kb_items=[
                _kb_item(
                    traps={"bad": "dict"},
                    solvingStyle=["step", "two"],
                    answerStyle=123,
                    notSameWhen=None,
                )
            ],
        )
        assert result.used is True
        assert len(result.brief_text) > 0

    def test_long_topic_truncated_not_validation_error(self) -> None:
        long_topic = "T" * 200
        result = SolutionBriefBuilder().build(
            _request(topic=long_topic, difficulty="intermediate"),
            kb_items=[_kb_item()],
        )
        assert result.brief is not None
        assert len(result.brief.topic) <= 128

    def test_snake_case_metadata_keys(self) -> None:
        result = SolutionBriefBuilder().build(
            _request(),
            kb_items=[
                _kb_item(
                    metadata={
                        "concept_tags": "SPEED,TIME",
                        "not_same_when": "Same direction movement",
                        "answer_style": "step_by_step",
                        "solving_style": "Convert units first",
                    }
                )
            ],
        )
        brief = result.brief
        assert brief is not None
        assert brief.risk_flags or brief.solution_approach or brief.generator_instructions


class TestSolutionBriefExtractGivenIndexError:
    """Regression: mixed-case 'If' in query must not raise IndexError."""

    def test_mixed_case_if_clause_extracted_safely(self) -> None:
        from services.solution_brief.solution_brief_builder import _extract_given

        given = _extract_given(
            "Find relative speed If two trains move in opposite directions at 54 km/hr"
        )
        assert given
        assert any("two trains" in item.lower() for item in given)

    def test_lowercase_if_still_works(self) -> None:
        from services.solution_brief.solution_brief_builder import _extract_given

        given = _extract_given("What if speed doubles when time is halved")
        assert any("speed doubles" in item.lower() for item in given)

    def test_no_if_clause_no_error(self) -> None:
        from services.solution_brief.solution_brief_builder import _extract_given

        given = _extract_given("A train 120m long crosses a platform in 18 seconds at 54 km/hr")
        assert len(given) == 1
        assert "18" in given[0]


class TestTimeSpeedDistanceSolutionBriefPath:
    def test_two_selected_items_build_solution_brief_not_fallback(self) -> None:
        items = [
            RetrievedContextItem(
                text="When two objects move in opposite directions, add their speeds.",
                score=1.0,
                metadata={
                    "subject": "QUANT",
                    "patternTopicKey": "TIME_SPEED_DISTANCE",
                    "patternFamilyKey": "RELATIVE_SPEED",
                    "conceptTags": "SPEED,TIME,DISTANCE",
                },
                match_lane=LANE_SUBJECT_ONLY,
                risk="family_mismatch",
            ),
            RetrievedContextItem(
                text="Platform crossing uses train length plus platform length.",
                score=0.99,
                metadata={
                    "subject": "QUANT",
                    "patternTopicKey": "TIME_SPEED_DISTANCE",
                    "conceptTags": "PLATFORM_CROSSING",
                },
                match_lane=LANE_SUBJECT_ONLY,
                source_id="tsd-2",
            ),
        ]
        query = (
            "Find relative speed If two trains move in opposite directions. "
            "A train 120m long crosses a platform in 18 seconds at 54 km/hr."
        )
        result = SolutionBriefBuilder().build(
            _request(
                query=query,
                topic="TIME_SPEED_DISTANCE",
                intent="solve",
                difficulty="intermediate",
            ),
            kb_items=items,
        )
        assert result.used is True
        assert "[Solution Brief]" in result.brief_text
        assert "[Relevant KB Context]" not in result.brief_text
        assert len(result.brief_text) > 0
        lowered = result.brief_text.lower()
        assert "patternid" not in lowered
        assert "patterntopickey" not in lowered

    def test_compose_uses_solution_brief_for_if_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RETRIEVAL_PROVIDER", "legacy_bedrock_kb")
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        cfg_module._settings = None
        retriever = MagicMock()
        retriever.retrieve_lane.return_value = (
            [
                RetrievedContextItem(
                    text="Relative speed when trains move in opposite directions.",
                    score=1.0,
                    metadata={
                        "subject": "QUANT",
                        "patternTopicKey": "TIME_SPEED_DISTANCE",
                        "conceptTags": "SPEED,TIME",
                    },
                    match_lane=LANE_SUBJECT_TOPIC,
                )
            ],
            5,
        )
        svc = ContextRetrievalService(kb_retriever=retriever)

        result = svc.retrieve_context(
            ContextRetrievalRequest(
                request_id="tsd-if-1",
                query=(
                    "Find platform length If a train 120m long crosses in 18 seconds at 54 km/hr"
                ),
                subject="math",
                intent="solve",
                difficulty="intermediate",
                topic="TIME_SPEED_DISTANCE",
            )
        )
        assert len(result.context_text) > 0
        assert "[Solution Brief]" in result.context_text
        assert "[Relevant KB Context]" not in result.context_text
