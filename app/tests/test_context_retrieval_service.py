"""
tests/test_context_retrieval_service.py
----------------------------------------
Unit tests for context retrieval — metadata mapping, lanes, rerank, formatter.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

import config as cfg_module
from services.context_retrieval.bedrock_kb_retriever import (
    build_metadata_filter,
    normalize_kb_metadata,
)
from services.context_retrieval.context_models import (
    ContextRetrievalDecision,
    ContextRetrievalRequest,
    RetrievedContextItem,
)
from services.context_retrieval.context_retrieval_service import (
    LANE_BROAD_SEMANTIC,
    LANE_RELAXED_SUBJECT_ONLY,
    LANE_SUBJECT_ONLY,
    LANE_SUBJECT_TOPIC,
    LANE_SUBJECT_TOPIC_FAMILY,
    ContextRequestBuilder,
    ContextRetrievalService,
    derive_pattern_hints,
    infer_pattern_topic_key,
    map_app_subject_to_kb,
    normalize_retrieval_tags,
    reset_context_retrieval_service,
    resolve_retrieval_hints,
)
from services.query_classifier_service import apply_classification_policy


def _reset_settings() -> None:
    cfg_module._settings = None


@pytest.fixture(autouse=True)
def _clean_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Part 13.1 KB compatibility tests on their explicit legacy provider."""
    monkeypatch.setenv("RETRIEVAL_PROVIDER", "legacy_bedrock_kb")
    _reset_settings()
    reset_context_retrieval_service()
    yield
    _reset_settings()
    reset_context_retrieval_service()


def _request(**overrides: Any) -> ContextRetrievalRequest:
    base = {
        "request_id": "req-1",
        "query": "Explain successive discount trap for SBI PO",
        "subject": "math",
        "intent": "explain",
        "difficulty": "advanced",
    }
    base.update(overrides)
    return ContextRetrievalRequest(**base)


def _high_confidence_item(**overrides: Any) -> RetrievedContextItem:
    meta = {
        "patternId": "pat-001",
        "subject": "QUANT",
        "patternTopicKey": "PROFIT_LOSS_DISCOUNT",
        "patternFamilyKey": "DISCOUNT",
        "complexityLevel": "3",
        "confidence": "1.00",
        "taxonomyReviewRequired": "false",
        "schemaVersion": "v2",
    }
    meta.update(overrides.get("metadata", {}))
    overrides.pop("metadata", None)
    base = {
        "text": "Successive discount pattern rule for banking exams.",
        "score": 0.88,
        "source_id": "pat-001",
        "metadata": meta,
        "match_lane": LANE_SUBJECT_ONLY,
    }
    base.update(overrides)
    return RetrievedContextItem(**base)


class TestPatternHintExtraction:
    def test_coded_inequality(self) -> None:
        hints = derive_pattern_hints(
            "Solve coded inequality with conclusions follow",
            "reasoning",
        )
        assert hints.pattern_topic_key == "CODED_INEQUALITY"
        assert hints.strength in {"medium", "strong"}

    def test_seating_arrangement(self) -> None:
        hints = derive_pattern_hints(
            "Circular seating arrangement facing north",
            "reasoning",
        )
        assert hints.pattern_topic_key == "SEATING_ARRANGEMENT"

    def test_floor_puzzle(self) -> None:
        hints = derive_pattern_hints("Floor puzzle in building numbered floors", "reasoning")
        assert hints.pattern_topic_key == "FLOOR_PUZZLE"

    def test_direction_sense(self) -> None:
        hints = derive_pattern_hints(
            "Direction sense with turns left and turns right",
            "reasoning",
        )
        assert hints.pattern_topic_key == "DIRECTION_SENSE"

    def test_profit_loss_discount(self) -> None:
        hints = derive_pattern_hints("Profit loss discount marked price", "math")
        assert hints.pattern_topic_key == "PROFIT_LOSS_DISCOUNT"

    def test_mixture_alligation(self) -> None:
        hints = derive_pattern_hints("Mixture and alligation concentration", "math")
        assert hints.pattern_topic_key == "MIXTURE_ALLIGATION"

    def test_vague_query_no_topic(self) -> None:
        hints = derive_pattern_hints("What is the answer?", "reasoning")
        assert hints.pattern_topic_key is None
        assert hints.strength == "weak"


class TestTopicAwareLanes:
    def test_coded_inequality_uses_subject_topic_lane(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = svc._build_retrieval_lanes(
            kb_subject="REASONING",
            pattern_topic_key="CODED_INEQUALITY",
            pattern_family_key=None,
        )
        lane_names = [name for name, _ in lanes]
        assert lane_names[0] == LANE_SUBJECT_TOPIC
        assert LANE_SUBJECT_TOPIC in lane_names
        assert lane_names.index(LANE_SUBJECT_TOPIC) < lane_names.index(LANE_SUBJECT_ONLY)

    def test_profit_loss_uses_subject_topic_lane(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = svc._build_retrieval_lanes(
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
        )
        lane_names = [name for name, _ in lanes]
        assert LANE_SUBJECT_TOPIC in lane_names
        assert lane_names.index(LANE_SUBJECT_TOPIC) < lane_names.index(LANE_SUBJECT_ONLY)

    def test_unknown_topic_falls_back_to_subject_only(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = svc._build_retrieval_lanes(
            kb_subject="REASONING",
            pattern_topic_key=None,
            pattern_family_key=None,
        )
        lane_names = [name for name, _ in lanes]
        assert lane_names[0] == LANE_SUBJECT_ONLY


class TestMetadataMapping:
    def test_math_maps_to_quant(self) -> None:
        assert map_app_subject_to_kb("math") == "QUANT"

    def test_reasoning_maps(self) -> None:
        assert map_app_subject_to_kb("reasoning") == "REASONING"

    def test_english_maps(self) -> None:
        assert map_app_subject_to_kb("english") == "ENGLISH"

    def test_general_maps_gk(self) -> None:
        assert map_app_subject_to_kb("general") == "GK"

    def test_unknown_subject_returns_none(self) -> None:
        assert map_app_subject_to_kb("biology") is None

    def test_profit_keyword_infers_topic(self) -> None:
        key = infer_pattern_topic_key("Explain profit and loss discount trap", None)
        assert key == "PROFIT_LOSS_DISCOUNT"

    def test_canonical_topic_passthrough(self) -> None:
        assert infer_pattern_topic_key("query", "MIXTURE_ALLIGATION") == "MIXTURE_ALLIGATION"


class TestFilterConstruction:
    def test_subject_filter_uses_quant_for_math(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = svc._build_retrieval_lanes(
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
        )
        filters = dict(lanes)[LANE_SUBJECT_ONLY]
        assert filters["subject"] == "QUANT"

    def test_taxonomy_review_required_is_string_false(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        production = svc._production_safe_filters(cfg_module.get_settings())
        assert production["taxonomyReviewRequired"] == "false"

    def test_schema_version_is_string_v2(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        production = svc._production_safe_filters(cfg_module.get_settings())
        assert production["schemaVersion"] == "v2"

    def test_difficulty_not_in_decision_filters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        decision = svc.decide_retrieval(_request(difficulty="advanced"))
        assert "difficulty" not in decision.filters
        assert "complexityLevel" not in decision.filters
        assert "confidence" not in decision.filters
        assert "conceptTags" not in decision.filters

    def test_build_metadata_filter_uses_string_values(self) -> None:
        filt = build_metadata_filter(
            {"taxonomyReviewRequired": "false", "schemaVersion": "v2", "subject": "QUANT"}
        )
        assert filt is not None
        serialized = str(filt)
        assert "false" in serialized
        assert "v2" in serialized


class TestRetrievalLanes:
    def test_lane_fallback_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "kb-test")
        _reset_settings()

        retriever = MagicMock()
        retriever.retrieve_lane.side_effect = [
            ([], 0),
            ([], 0),
            ([_high_confidence_item()], 1),
        ]
        svc = ContextRetrievalService(kb_retriever=retriever)
        items, _aws, _lane = svc._retrieve_with_lanes(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", top_k=3),
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
            retrieval_query="Question: test",
        )
        assert len(items) == 1
        assert retriever.retrieve_lane.call_count == 3
        lanes_called = [call.kwargs["lane"] for call in retriever.retrieve_lane.call_args_list]
        assert lanes_called[0] == LANE_SUBJECT_TOPIC
        assert lanes_called[1] == LANE_SUBJECT_ONLY
        assert LANE_BROAD_SEMANTIC not in lanes_called[:2]

    def test_broad_lane_after_relaxed_subject_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "kb-test")
        _reset_settings()

        retriever = MagicMock()
        retriever.retrieve_lane.side_effect = [
            ([], 0),
            ([], 0),
            ([], 0),
            ([_high_confidence_item(match_lane=LANE_BROAD_SEMANTIC)], 1),
        ]
        svc = ContextRetrievalService(kb_retriever=retriever)
        svc._retrieve_with_lanes(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", top_k=3),
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
            retrieval_query="Question: test",
        )
        lanes_called = [call.kwargs["lane"] for call in retriever.retrieve_lane.call_args_list]
        assert LANE_RELAXED_SUBJECT_ONLY in lanes_called
        relaxed_idx = lanes_called.index(LANE_RELAXED_SUBJECT_ONLY)
        broad_idx = lanes_called.index(LANE_BROAD_SEMANTIC)
        assert relaxed_idx < broad_idx
        assert retriever.retrieve_lane.call_args_list[-1].kwargs["lane"] == LANE_BROAD_SEMANTIC

    def test_stops_after_candidates_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "kb-test")
        _reset_settings()

        retriever = MagicMock()
        retriever.retrieve_lane.return_value = ([_high_confidence_item()], 1)
        svc = ContextRetrievalService(kb_retriever=retriever)
        svc._retrieve_with_lanes(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", top_k=3),
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
            retrieval_query="Question: test",
        )
        assert retriever.retrieve_lane.call_count == 1

    def test_max_five_lanes(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = svc._build_retrieval_lanes(
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key="DISCOUNT",
        )
        assert len(lanes) <= 5
        assert lanes[0][0] == LANE_SUBJECT_TOPIC_FAMILY
        lane_names = [name for name, _ in lanes]
        assert lane_names.index(LANE_RELAXED_SUBJECT_ONLY) < lane_names.index(LANE_BROAD_SEMANTIC)
        assert lanes[-1][0] == LANE_BROAD_SEMANTIC

    def test_broad_lane_has_no_taxonomy_filter(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = dict(
            svc._build_retrieval_lanes(
                kb_subject="QUANT",
                pattern_topic_key="PROFIT_LOSS_DISCOUNT",
                pattern_family_key=None,
            )
        )
        assert "taxonomyReviewRequired" not in lanes[LANE_BROAD_SEMANTIC]
        assert "subject" not in lanes[LANE_BROAD_SEMANTIC]

    def test_relaxed_lane_subject_only(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = dict(
            svc._build_retrieval_lanes(
                kb_subject="QUANT",
                pattern_topic_key="PROFIT_LOSS_DISCOUNT",
                pattern_family_key=None,
            )
        )
        relaxed = lanes[LANE_RELAXED_SUBJECT_ONLY]
        assert relaxed == {"subject": "QUANT"}
        assert "schemaVersion" not in relaxed
        assert "taxonomyReviewRequired" not in relaxed

    def test_relaxed_lane_before_broad_on_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "kb-test")
        _reset_settings()

        retriever = MagicMock()
        retriever.retrieve_lane.side_effect = [
            ([], 0),
            ([], 0),
            ([_high_confidence_item(match_lane=LANE_RELAXED_SUBJECT_ONLY)], 1),
        ]
        svc = ContextRetrievalService(kb_retriever=retriever)
        items, _aws, _lane = svc._retrieve_with_lanes(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", top_k=3),
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
            retrieval_query="Question: test",
        )
        assert len(items) == 1
        lanes_called = [call.kwargs["lane"] for call in retriever.retrieve_lane.call_args_list]
        assert LANE_RELAXED_SUBJECT_ONLY in lanes_called
        assert LANE_BROAD_SEMANTIC not in lanes_called
        assert lanes_called[-1] == LANE_RELAXED_SUBJECT_ONLY

    def test_relaxed_lane_attempted_after_strict_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        monkeypatch.setenv("BEDROCK_KB_ID", "kb-test")
        _reset_settings()

        retriever = MagicMock()
        retriever.retrieve_lane.side_effect = [
            ([], 0),
            ([], 0),
            ([_high_confidence_item(match_lane=LANE_RELAXED_SUBJECT_ONLY)], 1),
        ]
        svc = ContextRetrievalService(kb_retriever=retriever)
        items, _aws, _lane = svc._retrieve_with_lanes(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", top_k=3),
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
            retrieval_query="Question: test",
        )
        assert len(items) == 1
        last_lane = retriever.retrieve_lane.call_args_list[-1].kwargs["lane"]
        assert last_lane == LANE_RELAXED_SUBJECT_ONLY
        assert retriever.retrieve_lane.call_count <= 5
        assert LANE_BROAD_SEMANTIC not in [
            call.kwargs["lane"] for call in retriever.retrieve_lane.call_args_list
        ]

    def test_build_lane_order_subject_before_broad(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = svc._build_retrieval_lanes(
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
        )
        lane_names = [name for name, _ in lanes]
        assert lane_names == [
            LANE_SUBJECT_TOPIC,
            LANE_SUBJECT_ONLY,
            LANE_RELAXED_SUBJECT_ONLY,
            LANE_BROAD_SEMANTIC,
        ]

    def test_skips_topic_lanes_when_topic_unknown(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        lanes = svc._build_retrieval_lanes(
            kb_subject="QUANT",
            pattern_topic_key=None,
            pattern_family_key=None,
        )
        lane_names = [name for name, _ in lanes]
        assert LANE_SUBJECT_TOPIC not in lane_names
        assert LANE_SUBJECT_ONLY in lane_names
        assert lane_names.index(LANE_RELAXED_SUBJECT_ONLY) < lane_names.index(LANE_BROAD_SEMANTIC)


class TestReranking:
    def test_exact_metadata_match_ranks_higher(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        request = _request(subject="math")
        decision = ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2)
        generic = RetrievedContextItem(
            text="generic discount profit content",
            score=0.75,
            metadata={"subject": "QUANT", "taxonomyReviewRequired": "false", "confidence": "1.00"},
        )
        matched = _high_confidence_item(
            metadata={
                "subject": "QUANT",
                "patternTopicKey": "PROFIT_LOSS_DISCOUNT",
                "taxonomyReviewRequired": "false",
                "confidence": "1.00",
            }
        )
        selected = svc._rerank_and_select(
            request,
            decision,
            [generic, matched],
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
        )
        assert selected[0].metadata.get("patternTopicKey") == "PROFIT_LOSS_DISCOUNT"

    def test_taxonomy_review_true_rejected_on_strict_lane(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        item = _high_confidence_item(
            metadata={"taxonomyReviewRequired": "true"},
            match_lane=LANE_SUBJECT_ONLY,
        )
        selected = svc._rerank_and_select(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2),
            [item],
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
        )
        assert selected == []

    def test_missing_subject_downranked_not_rejected(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        item = RetrievedContextItem(
            text="successive discount profit pattern for banking exams",
            score=0.88,
            metadata={
                "patternId": "p1",
                "taxonomyReviewRequired": "false",
                "confidence": "1.00",
            },
            match_lane=LANE_RELAXED_SUBJECT_ONLY,
        )
        selected = svc._rerank_and_select(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2),
            [item],
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
        )
        assert len(selected) == 1
        assert "missing_subject_metadata" in (selected[0].risk or "")

    def test_missing_taxonomy_not_hard_rejected_on_relaxed_lane(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        item = RetrievedContextItem(
            text="successive discount profit pattern for banking exams",
            score=0.88,
            metadata={"patternId": "p1", "subject": "QUANT", "confidence": "1.00"},
            match_lane=LANE_RELAXED_SUBJECT_ONLY,
        )
        selected = svc._rerank_and_select(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2),
            [item],
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
        )
        assert len(selected) == 1

    def test_low_bedrock_score_still_reaches_reranker(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        item = _high_confidence_item(
            score=0.20,
            text=(
                "Explain successive discount trap pattern for SBI PO banking exams "
                "with profit loss discount rules."
            ),
        )
        selected = svc._rerank_and_select(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2),
            [item],
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key="DISCOUNT",
        )
        assert len(selected) == 1
        assert (selected[0].rerank_confidence or 0.0) >= 0.85

    def test_below_threshold_produces_empty(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        low = RetrievedContextItem(
            text="weak",
            score=0.10,
            metadata={"subject": "ENGLISH", "taxonomyReviewRequired": "false"},
        )
        selected = svc._rerank_and_select(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2),
            [low],
            kb_subject="QUANT",
            pattern_topic_key=None,
            pattern_family_key=None,
        )
        assert selected == []

    def test_top_two_selected_when_confident(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        items = [
            _high_confidence_item(metadata={"patternId": "p1"}, source_id="p1"),
            _high_confidence_item(metadata={"patternId": "p2"}, source_id="p2"),
        ]
        selected = svc._rerank_and_select(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2),
            items,
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key=None,
        )
        assert len(selected) == 2

    def test_rerank_breakdown_logged_safely(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        item = RetrievedContextItem(
            text="pattern hint alpha",
            score=0.50,
            metadata={
                "patternId": "p-near",
                "subject": "REASONING",
                "patternTopicKey": "CODED_INEQUALITY",
                "taxonomyReviewRequired": "false",
            },
            match_lane=LANE_SUBJECT_ONLY,
        )
        with caplog.at_level("INFO"):
            selected = svc._rerank_and_select(
                _request(
                    query="Explain coded inequality for bank exam",
                    subject="reasoning",
                ),
                ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2),
                [item],
                kb_subject="REASONING",
                pattern_topic_key="CODED_INEQUALITY",
                pattern_family_key=None,
            )
        assert selected == []
        breakdown_logs = [
            r.message for r in caplog.records if "context_rerank_breakdown" in r.message
        ]
        assert breakdown_logs
        assert "patternId=p-near" in breakdown_logs[0]
        assert "topic_match=true" in breakdown_logs[0]
        assert "pattern hint alpha" not in caplog.text
        assert "near_miss=true" in caplog.text

    def test_near_miss_does_not_pass_context(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        item = RetrievedContextItem(
            text="pattern hint beta",
            score=0.50,
            metadata={
                "patternId": "p-near2",
                "subject": "REASONING",
                "patternTopicKey": "CODED_INEQUALITY",
                "taxonomyReviewRequired": "false",
            },
            match_lane=LANE_SUBJECT_ONLY,
        )
        selected = svc._rerank_and_select(
            _request(query="coded inequality puzzle", subject="reasoning"),
            ContextRetrievalDecision(use_kb=True, reason="test", rerank_top_n=2),
            [item],
            kb_subject="REASONING",
            pattern_topic_key="CODED_INEQUALITY",
            pattern_family_key=None,
        )
        assert selected == []


class TestFormatter:
    def test_compact_pattern_context_format(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        item = _high_confidence_item(why_candidate="subject_match, topic_match")
        text = svc._format_context(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", max_context_chars=2500),
            [item],
        )
        assert "[Solution Brief]" in text
        assert "Subject: Math" in text
        assert "Pattern ID:" not in text
        assert "patternTopicKey" not in text
        assert "{" not in text

    def test_respects_max_chars(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        item = _high_confidence_item(text="x" * 500)
        text = svc._format_context(
            _request(),
            ContextRetrievalDecision(use_kb=True, reason="test", max_context_chars=120),
            [item],
        )
        assert len(text) <= 120

    def test_low_confidence_items_not_formatted(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        assert (
            svc._format_context(
                _request(difficulty="intermediate"),
                ContextRetrievalDecision(use_kb=True, reason="test"),
                [],
            )
            == ""
        )


class TestServiceIntegration:
    def test_full_retrieval_with_topic_aware_lane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        retriever = MagicMock()
        item = _high_confidence_item(
            metadata={
                "subject": "REASONING",
                "patternTopicKey": "CODED_INEQUALITY",
            },
            match_lane=LANE_SUBJECT_TOPIC,
        )
        retriever.retrieve_lane.return_value = ([item], 3)
        svc = ContextRetrievalService(kb_retriever=retriever)
        result = svc.retrieve_context(
            _request(
                query="Explain coded inequality conclusions follow",
                subject="reasoning",
                difficulty="advanced",
            )
        )
        assert retriever.retrieve_lane.call_args_list[0].kwargs["lane"] == LANE_SUBJECT_TOPIC
        assert result.context_text != ""
        assert result.reason == "context_selected"

    def test_full_retrieval_with_high_confidence_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        retriever = MagicMock()
        retriever.retrieve_lane.return_value = ([_high_confidence_item()], 1)
        svc = ContextRetrievalService(kb_retriever=retriever)
        result = svc.retrieve_context(_request())
        assert result.context_text != ""
        assert result.reason == "context_selected"
        assert "[Solution Brief]" in result.context_text
        assert retriever.retrieve_lane.call_count >= 1

    def test_regression_quant_motion_maps_to_quant_kb_subject(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        classification = apply_classification_policy(
            "Two trains in opposite directions at 54 km/hr and 72 km/hr",
            {"subject": "math", "intent": "solve", "difficulty": "intermediate"},
            classifier_confidence=0.94,
        )
        assert classification["subject"] == "math"
        assert map_app_subject_to_kb(classification["subject"]) == "QUANT"

    def test_reasoning_direction_maps_to_reasoning_kb_subject(self) -> None:
        classification = apply_classification_policy(
            "A person walked north then turned right. Where is he facing?",
            {"subject": "reasoning", "intent": "solve", "difficulty": "default"},
            classifier_confidence=0.91,
        )
        assert classification["subject"] == "reasoning"
        assert map_app_subject_to_kb(classification["subject"]) == "REASONING"

    def test_normalization_drop_reason_when_aws_returns_but_parse_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        retriever = MagicMock()
        retriever.retrieve_lane.return_value = ([], 5)
        svc = ContextRetrievalService(kb_retriever=retriever)
        result = svc.retrieve_context(_request(difficulty="intermediate"))
        assert result.context_text == ""
        assert result.reason == "normalization_dropped_all_candidates"

    def test_below_threshold_returns_no_high_confidence_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        retriever = MagicMock()
        retriever.retrieve_lane.return_value = (
            [
                RetrievedContextItem(
                    text="weak",
                    score=0.10,
                    metadata={"subject": "ENGLISH", "taxonomyReviewRequired": "false"},
                    match_lane=LANE_BROAD_SEMANTIC,
                )
            ],
            1,
        )
        svc = ContextRetrievalService(kb_retriever=retriever)
        result = svc.retrieve_context(_request(difficulty="intermediate"))
        assert result.context_text == ""
        assert result.reason == "no_high_confidence_context"

    def test_no_kb_candidates_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        retriever = MagicMock()
        retriever.retrieve_lane.return_value = ([], 0)
        svc = ContextRetrievalService(kb_retriever=retriever)
        result = svc.retrieve_context(_request(difficulty="intermediate"))
        assert result.context_text == ""
        assert result.reason == "no_kb_candidates"

    def test_no_cache_runtime_path(self) -> None:
        source = inspect.getsource(ContextRetrievalService.retrieve_context)
        assert "cache" not in source.lower()

    def test_retrieval_query_includes_metadata(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        query = svc._build_retrieval_query(
            _request(),
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
        )
        assert "Question:" in query
        assert "Subject: QUANT" in query
        assert "Topic hint: PROFIT_LOSS_DISCOUNT" in query


class TestDecisionLayer:
    def test_explain_intent_uses_kb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        assert svc.decide_retrieval(_request(intent="explain")).use_kb is True

    def test_kb_disabled_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "false")
        _reset_settings()
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        assert svc.decide_retrieval(_request()).use_kb is False


class TestContextRequestBuilder:
    def test_builds_from_classification_dict(self) -> None:
        req = ContextRequestBuilder.from_query_and_classification(
            request_id="r1",
            query="Explain profit for SSC CGL",
            classification={
                "subject": "math",
                "intent": "explain",
                "difficulty": "advanced",
            },
        )
        assert req.subject == "math"
        assert req.difficulty == "advanced"
        assert req.exam == "ssc cgl"


class TestMetadataNormalization:
    def test_quant_subject_normalized(self) -> None:
        meta = normalize_kb_metadata({"subject": "quant"})
        assert meta["subject"] == "QUANT"

    def test_taxonomy_false_normalized(self) -> None:
        meta = normalize_kb_metadata({"taxonomyReviewRequired": "False"})
        assert meta["taxonomyReviewRequired"] == "false"

    def test_concept_tags_uppercased(self) -> None:
        meta = normalize_kb_metadata({"conceptTags": "profit, discount"})
        assert meta["conceptTags"] == "PROFIT,DISCOUNT"


class TestResolveResultReason:
    def test_no_kb_candidates(self) -> None:
        reason = ContextRetrievalService._resolve_result_reason(
            ContextRetrievalDecision(use_kb=True, reason="intent_explain"),
            [],
            [],
            any_aws_results=False,
        )
        assert reason == "no_kb_candidates"

    def test_normalization_dropped_all_candidates(self) -> None:
        reason = ContextRetrievalService._resolve_result_reason(
            ContextRetrievalDecision(use_kb=True, reason="intent_explain"),
            [],
            [],
            any_aws_results=True,
        )
        assert reason == "normalization_dropped_all_candidates"

    def test_no_high_confidence_context(self) -> None:
        reason = ContextRetrievalService._resolve_result_reason(
            ContextRetrievalDecision(use_kb=True, reason="intent_explain"),
            [_high_confidence_item()],
            [],
        )
        assert reason == "no_high_confidence_context"

    def test_context_selected(self) -> None:
        item = _high_confidence_item()
        reason = ContextRetrievalService._resolve_result_reason(
            ContextRetrievalDecision(use_kb=True, reason="intent_explain"),
            [item],
            [item],
        )
        assert reason == "context_selected"


class TestRetrievalHintResolution:
    def test_high_confidence_canonical_topic_becomes_pattern_key(self) -> None:
        hints = resolve_retrieval_hints(
            "A train crosses a platform",
            "math",
            {
                "pattern_topic_candidate": "TIME_SPEED_DISTANCE",
                "topic_confidence": 0.92,
                "retrieval_tags": ["train_crossing", "relative_speed"],
            },
        )
        assert hints.pattern_topic_key == "TIME_SPEED_DISTANCE"
        assert hints.hint_source == "classifier_pattern_topic"
        assert "train_crossing" in hints.retrieval_tags

    def test_low_confidence_topic_does_not_strict_filter(self) -> None:
        hints = resolve_retrieval_hints(
            "ratio and percentage departments",
            "math",
            {
                "pattern_topic_candidate": "RATIO_PROPORTION",
                "topic_confidence": 0.60,
                "retrieval_tags": ["ratio_parts", "weighted_percentage"],
            },
        )
        assert hints.pattern_topic_key is None or hints.hint_source != "classifier_pattern_topic"
        assert hints.retrieval_tags

    def test_deterministic_fallback_when_classifier_hints_missing(self) -> None:
        hints = resolve_retrieval_hints(
            "Two trains cross each other at relative speed",
            "math",
            {},
        )
        assert hints.pattern_topic_key == "TIME_SPEED_DISTANCE"
        assert hints.hint_source == "deterministic_derive"

    def test_normalize_retrieval_tags_dedupes_and_caps(self, monkeypatch) -> None:
        monkeypatch.setenv("CONTEXT_MAX_RETRIEVAL_TAGS", "3")
        _reset_settings()
        tags = normalize_retrieval_tags(
            ["Train-Crossing", "train_crossing", "Ratio Parts", "ratio_parts", "age"]
        )
        assert tags == ["train_crossing", "ratio_parts", "age"]


class TestRetrievalTagRerank:
    def test_retrieval_tags_overlap_boosts_confidence(self) -> None:
        service = ContextRetrievalService(kb_retriever=MagicMock())
        item = RetrievedContextItem(
            text="Relative speed when two trains cross.",
            score=0.72,
            metadata={
                "patternId": "pat-tsd",
                "subject": "QUANT",
                "patternTopicKey": "TIME_SPEED_DISTANCE",
                "conceptTags": "RELATIVE_SPEED,TRAIN_CROSSING",
                "taxonomyReviewRequired": "false",
                "confidence": "0.90",
            },
            match_lane=LANE_SUBJECT_TOPIC,
        )
        _score, confidence, _why, _risk, breakdown = service._score_candidate(
            _request(query="train crossing relative speed"),
            item,
            kb_subject="QUANT",
            pattern_topic_key="TIME_SPEED_DISTANCE",
            pattern_family_key=None,
            query_tokens={"train", "crossing", "relative", "speed"},
            retrieval_tags=["relative_speed", "train_crossing"],
            is_relaxed_lane=False,
            taxonomy_approved_only=True,
        )
        assert breakdown.tag_overlap_score > 0
        assert breakdown.matched_tags_count >= 1
        assert confidence >= 0.85

    def test_no_tag_overlap_does_not_hard_reject(self) -> None:
        service = ContextRetrievalService(kb_retriever=MagicMock())
        item = _high_confidence_item()
        _score, confidence, why, _risk, _bd = service._score_candidate(
            _request(),
            item,
            kb_subject="QUANT",
            pattern_topic_key="PROFIT_LOSS_DISCOUNT",
            pattern_family_key="DISCOUNT",
            query_tokens={"successive", "discount", "pattern"},
            retrieval_tags=["unrelated_tag"],
            is_relaxed_lane=False,
            taxonomy_approved_only=True,
        )
        assert why != "rejected"
        assert confidence >= 0.85


class TestContextRequestBuilderHints:
    def test_builder_passes_classifier_hints(self) -> None:
        req = ContextRequestBuilder.from_query_and_classification(
            request_id="req-hints",
            query="Age problem equation",
            classification={
                "subject": "math",
                "intent": "solve",
                "difficulty": "intermediate",
                "retrieval_required": True,
                "topic": "Age Problem",
                "topic_confidence": 0.9,
                "pattern_topic_candidate": "AGE",
                "retrieval_tags": ["age_equation", "birth_age"],
            },
        )
        assert req.pattern_topic_candidate == "AGE"
        assert req.topic_confidence == 0.9
        assert "age_equation" in req.retrieval_tags


class TestMixtureAlligationRerankRegression:
    def test_family_mismatch_does_not_hard_reject_topic_matched_candidate(self) -> None:
        service = ContextRetrievalService(kb_retriever=MagicMock())
        item = RetrievedContextItem(
            text="Mixture and alligation rule for two solutions.",
            score=0.78,
            metadata={
                "patternId": "pat-mix-1",
                "subject": "QUANT",
                "patternTopicKey": "MIXTURE_ALLIGATION",
                "patternFamilyKey": "ALLIGATION",
                "conceptTags": "MIXTURE,ALLIGATION",
                "taxonomyReviewRequired": "false",
                "confidence": "0.90",
            },
            match_lane=LANE_SUBJECT_TOPIC,
        )
        _score, confidence, why, risk, breakdown = service._score_candidate(
            _request(query="Mixture alligation two solutions concentration"),
            item,
            kb_subject="QUANT",
            pattern_topic_key="MIXTURE_ALLIGATION",
            pattern_family_key="MIXTURE",
            query_tokens={"mixture", "alligation", "concentration"},
            retrieval_tags=["mixture", "alligation"],
            is_relaxed_lane=False,
            taxonomy_approved_only=True,
        )
        assert why != "rejected"
        assert confidence > 0.0
        assert "family_mismatch" in risk
        assert breakdown.topic_match is True

    def test_mixture_topic_match_can_reach_selection_threshold(self) -> None:
        service = ContextRetrievalService(kb_retriever=MagicMock())
        item = RetrievedContextItem(
            text="Alligation cross method for mixture concentration problems.",
            score=0.82,
            metadata={
                "patternId": "pat-mix-2",
                "subject": "QUANT",
                "patternTopicKey": "MIXTURE_ALLIGATION",
                "patternFamilyKey": "ALLIGATION",
                "taxonomyReviewRequired": "false",
                "confidence": "0.95",
            },
            match_lane=LANE_SUBJECT_TOPIC,
        )
        _score, confidence, why, _risk, _bd = service._score_candidate(
            _request(query="Mixture alligation concentration cross method"),
            item,
            kb_subject="QUANT",
            pattern_topic_key="MIXTURE_ALLIGATION",
            pattern_family_key=None,
            query_tokens={"mixture", "alligation", "concentration", "cross"},
            retrieval_tags=["mixture", "alligation"],
            is_relaxed_lane=False,
            taxonomy_approved_only=True,
        )
        assert why != "rejected"
        assert confidence >= 0.85


class TestKbContextFormattingFallback:
    def test_time_speed_distance_selected_items_produce_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        retriever = MagicMock()
        items = [
            _high_confidence_item(
                text="When two objects move in opposite directions, add their speeds.",
                metadata={
                    "subject": "QUANT",
                    "patternTopicKey": "TIME_SPEED_DISTANCE",
                    "patternFamilyKey": "RELATIVE_SPEED",
                    "conceptTags": ["SPEED", "TIME", "DISTANCE"],
                    "traps": ["Do not add speeds when same direction"],
                    "solvingStyle": {"steps": ["convert km/hr to m/s"]},
                },
                match_lane=LANE_SUBJECT_TOPIC,
            ),
            _high_confidence_item(
                text="Platform crossing uses train length plus platform length.",
                metadata={
                    "subject": "QUANT",
                    "patternTopicKey": "TIME_SPEED_DISTANCE",
                    "conceptTags": "PLATFORM_CROSSING",
                },
                match_lane=LANE_SUBJECT_TOPIC,
                source_id="pat-tsd-2",
            ),
        ]
        retriever.retrieve_lane.return_value = (items, 5)
        svc = ContextRetrievalService(kb_retriever=retriever)
        result = svc.retrieve_context(
            _request(
                query="A train 120m long crosses a platform in 18 seconds at 54 km/hr.",
                subject="math",
                intent="solve",
                difficulty="intermediate",
                topic="TIME_SPEED_DISTANCE",
            )
        )
        assert result.item_count >= 1
        assert len(result.context_text) > 0
        lowered = result.context_text.lower()
        assert "patternid" not in lowered
        assert "pat-tsd" not in lowered
        assert "patterntopickey" not in lowered

    def test_fallback_used_when_solution_brief_builder_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENABLE_KB_RETRIEVAL", "true")
        _reset_settings()
        retriever = MagicMock()
        retriever.retrieve_lane.return_value = ([_high_confidence_item()], 3)
        broken_builder = MagicMock()
        broken_builder.build.side_effect = ValueError("brief build failed")
        svc = ContextRetrievalService(
            kb_retriever=retriever,
            brief_builder=broken_builder,
        )
        result = svc.retrieve_context(_request())
        assert len(result.context_text) > 0
        assert "[Relevant KB Context]" in result.context_text
        assert "patternId" not in result.context_text
        assert result.item_count >= 1

    def test_compose_generator_context_fallback_direct(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        broken_builder = MagicMock()
        broken_builder.build.side_effect = RuntimeError("compose failure")
        svc._brief_builder = broken_builder
        items = [_high_confidence_item(text="Compact KB excerpt for fallback.")]
        text = svc._compose_generator_context(
            _request(),
            kb_items=items,
            web_items=[],
            max_chars=800,
        )
        assert len(text) > 0
        assert "[Relevant KB Context]" in text

    def test_no_kb_selected_keeps_empty_context(self) -> None:
        svc = ContextRetrievalService(kb_retriever=MagicMock())
        text = svc._compose_generator_context(
            _request(difficulty="intermediate"),
            kb_items=[],
            web_items=[],
            max_chars=800,
        )
        assert text == ""
