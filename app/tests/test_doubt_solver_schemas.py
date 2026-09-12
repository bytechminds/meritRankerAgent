"""
app/tests/test_doubt_solver_schemas.py
---------------------------------------
Unit tests for DoubtSolverRequest, QueryClassification, and DoubtSolverResponse.

No AWS credentials, network, or LLM calls required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.doubt_solver import (
    DoubtSolverRequest,
    DoubtSolverResponse,
    DoubtSolverState,
    QueryClassification,
)

_REQUEST_IDS = {
    "user_id": "local-user",
    "conversation_id": "conversation-1",
    "turn_id": "turn-1",
}


class TestDoubtSolverRequest:
    def test_valid_request(self):
        req = DoubtSolverRequest(
            mode="doubt_solver", query="What is 20% of 500?", **_REQUEST_IDS
        )
        assert req.query == "What is 20% of 500?"
        assert req.mode == "doubt_solver"
        assert req.user_id == "local-user"
        assert req.language == "english"

    def test_query_whitespace_stripped(self):
        req = DoubtSolverRequest(mode="doubt_solver", query="  hello  ", **_REQUEST_IDS)
        assert req.query == "hello"

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            DoubtSolverRequest(mode="doubt_solver", query="")

    def test_whitespace_only_query_rejected(self):
        with pytest.raises(ValidationError):
            DoubtSolverRequest(mode="doubt_solver", query="   ")

    def test_query_at_max_length_accepted(self):
        req = DoubtSolverRequest(mode="doubt_solver", query="a" * 5000, **_REQUEST_IDS)
        assert len(req.query) == 5000

    def test_query_over_max_length_rejected(self):
        with pytest.raises(ValidationError):
            DoubtSolverRequest(mode="doubt_solver", query="a" * 5001)

    def test_wrong_mode_rejected(self):
        with pytest.raises(ValidationError):
            DoubtSolverRequest(mode="demo", query="hello")  # type: ignore[arg-type]

    def test_language_default(self):
        req = DoubtSolverRequest(mode="doubt_solver", query="hello", **_REQUEST_IDS)
        assert req.language == "english"

    def test_language_hi_accepted(self):
        req = DoubtSolverRequest(
            mode="doubt_solver", query="hello", language="hi", **_REQUEST_IDS
        )
        assert req.language == "hindi"

    def test_language_devanagari_alias_is_accepted(self):
        req = DoubtSolverRequest(
            mode="doubt_solver", query="hello", language="हिंदी", **_REQUEST_IDS
        )
        assert req.language == "hindi"

    def test_invalid_language_rejected(self):
        with pytest.raises(ValidationError):
            DoubtSolverRequest(mode="doubt_solver", query="hello", language="fr")  # type: ignore[arg-type]

    def test_canonical_taxonomy_pair_is_retained_as_passive_request_context(self):
        baseline = DoubtSolverRequest(
            mode="doubt_solver", query="Create practice questions", **_REQUEST_IDS
        )
        req = DoubtSolverRequest(
            mode="doubt_solver",
            query="Create practice questions",
            canonical_subject_id="QUANTITATIVE_APTITUDE",
            canonical_topic_id="PERCENTAGE",
            **_REQUEST_IDS,
        )

        assert req.canonical_subject_id == "QUANTITATIVE_APTITUDE"
        assert req.canonical_topic_id == "PERCENTAGE"
        assert req.model_dump()["canonical_subject_id"] == "QUANTITATIVE_APTITUDE"
        assert req.model_dump()["canonical_topic_id"] == "PERCENTAGE"
        assert {
            key: value
            for key, value in req.model_dump().items()
            if key not in {"canonical_subject_id", "canonical_topic_id"}
        } == {
            key: value
            for key, value in baseline.model_dump().items()
            if key not in {"canonical_subject_id", "canonical_topic_id"}
        }

    @pytest.mark.parametrize(
        ("subject_id", "topic_id"),
        [
            ("QUANTITATIVE_APTITUDE", None),
            (None, "PERCENTAGE"),
            ("quantitative aptitude", "PERCENTAGE"),
            ("QUANTITATIVE_APTITUDE", "percentage"),
        ],
    )
    def test_incomplete_or_malformed_canonical_taxonomy_is_dropped(
        self, subject_id, topic_id
    ):
        req = DoubtSolverRequest(
            mode="doubt_solver",
            query="Create practice questions",
            canonical_subject_id=subject_id,
            canonical_topic_id=topic_id,
            **_REQUEST_IDS,
        )

        assert req.canonical_subject_id is None
        assert req.canonical_topic_id is None


class TestQueryClassification:
    def test_valid_classification(self):
        c = QueryClassification(intent="solve_question", confidence=0.7)
        assert c.intent == "solve_question"
        assert c.subject == "unknown"
        assert c.topic is None
        assert c.response_style == "step_by_step"
        assert c.confidence == 0.7

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            QueryClassification(intent="solve_question", confidence=1.5)

    def test_confidence_negative_rejected(self):
        with pytest.raises(ValidationError):
            QueryClassification(intent="solve_question", confidence=-0.1)

    def test_invalid_intent_rejected(self):
        with pytest.raises(ValidationError):
            QueryClassification(intent="do_homework", confidence=0.5)  # type: ignore[arg-type]

    def test_reasoning_summary_over_max_length_rejected(self):
        """reasoning_summary longer than 500 chars must be rejected by schema."""
        with pytest.raises(ValidationError):
            QueryClassification(
                intent="solve_question",
                confidence=0.9,
                reasoning_summary="x" * 501,
            )

    def test_reasoning_summary_at_max_length_accepted(self):
        """reasoning_summary of exactly 500 chars must be accepted."""
        c = QueryClassification(
            intent="solve_question",
            confidence=0.9,
            reasoning_summary="y" * 500,
        )
        assert len(c.reasoning_summary) == 500  # type: ignore[arg-type]


class TestDoubtSolverResponse:
    def _make_classification(self) -> QueryClassification:
        return QueryClassification(intent="solve_question", confidence=0.7)

    def test_valid_response(self):
        c = self._make_classification()
        resp = DoubtSolverResponse(
            success=True,
            request_id="test-id",
            mode="doubt_solver",
            answer="The answer is 100.",
            classification=c,
        )
        assert resp.success is True
        assert resp.needs_review is False

    def test_response_fields_present(self):
        c = self._make_classification()
        resp = DoubtSolverResponse(
            success=True,
            request_id="abc",
            mode="doubt_solver",
            answer="ans",
            classification=c,
        )
        dumped = resp.model_dump()
        for field in ("success", "request_id", "mode", "answer", "classification", "needs_review"):
            assert field in dumped


class TestDoubtSolverState:
    def _make_request(self) -> DoubtSolverRequest:
        return DoubtSolverRequest(
            mode="doubt_solver", query="Explain ratio", **_REQUEST_IDS
        )

    def test_valid_state(self):
        req = self._make_request()
        state = DoubtSolverState(request=req, request_id="abc")
        assert state.request.query == "Explain ratio"
        assert state.request_id == "abc"
        assert state.classification is None
        assert state.answer is None
        assert state.response is None

    def test_state_with_classification(self):
        req = self._make_request()
        c = QueryClassification(intent="explain_concept", confidence=0.75)
        state = DoubtSolverState(request=req, request_id="abc", classification=c)
        assert state.classification is not None
        assert state.classification.intent == "explain_concept"

    def test_state_model_dump_serialises(self):
        req = self._make_request()
        state = DoubtSolverState(request=req, request_id="xyz")
        dumped = state.model_dump()
        assert dumped["request_id"] == "xyz"
        assert dumped["request"]["query"] == "Explain ratio"
