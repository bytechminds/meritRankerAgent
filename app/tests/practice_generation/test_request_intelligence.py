"""Practice Request Intelligence: interpretation in, deterministic planning out.

Every model response here is a fixture. These tests prove the contract, the
deterministic validation, and the wiring — they prove nothing about the real
model's semantic quality, which requires live qualification.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from features.practice_generation.planning import (
    PracticeRequestCountError,
    deterministic_blueprint,
    explicit_requested_count,
    resolve_practice_request,
    resolve_practice_type,
    resolve_requested_count,
)
from features.practice_generation.request_intelligence import (
    MAX_INTERPRETED_TOPICS,
    RequestIntelligenceError,
    parse_request_intelligence,
    reconstruct_source_text,
    token_id,
    token_index,
    tokenize_query,
)
from features.practice_generation.schemas import Difficulty, PracticeLaunchResult
from graphs.doubt_solver_graph import build_orchestrated_doubt_solver_graph
from schemas.doubt_solver import QueryClassification
from schemas.practice_request_intelligence import (
    PRACTICE_REQUEST_INTELLIGENCE_SCHEMA,
    PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME,
)
from services.llm.orchestration.config_registry import LlmConfigRegistry

_APP_DIR = Path(__file__).resolve().parents[2]

_QUANT_QUERY = (
    "Create 50 questions on Number System, Percentage, Profit and Loss, "
    "Ratio and Proportion, Average, Time and Work, Time Speed and Distance, Geometry, "
    "Mensuration, Probability and Trigonometry with mixed difficulty"
)
_QUANT_TOPICS = (
    ("Number System", "Number System"),
    ("Percentage", "Percentage"),
    ("Profit and Loss", "Profit & Loss"),
    ("Ratio and Proportion", "Ratio & Proportion"),
    ("Average", "Average"),
    ("Time and Work", "Time & Work"),
    ("Time Speed and Distance", "Time, Speed & Distance"),
    ("Geometry", "Geometry"),
    ("Mensuration", "Mensuration"),
    ("Probability", "Probability"),
    ("Trigonometry", "Trigonometry"),
)


def _response(
    *,
    status: str = "RESOLVED",
    count: int | None = None,
    topics: tuple[tuple[str, str], ...] = (),
    mode: str = "UNSPECIFIED",
    single: str | None = None,
    distribution: dict[str, int] | None = None,
    query: str | None = None,
    raw_topics: list[dict[str, Any]] | None = None,
) -> str:
    """Build a v3 wire response. Topics are given as (phrase, name) for readability
    and resolved here to the token range that phrase occupies in `query`, exactly as
    the model would have to select it."""
    return json.dumps(
        {
            "interpretationStatus": status,
            "requestedCount": count,
            "topics": (
                raw_topics
                if raw_topics is not None
                else [_span(query, phrase, name) for phrase, name in topics]
            ),
            "difficulty": _difficulty(mode, single, distribution),
        }
    )


def _span(query: str | None, phrase: str, name: str) -> dict[str, Any]:
    """Return the inclusive token range `phrase` occupies in `query`."""
    assert query is not None, "topics require the query to resolve token ranges"
    haystack = [token.text for token in tokenize_query(query)]
    needle = [token.text for token in tokenize_query(phrase)]
    assert needle, f"phrase {phrase!r} has no tokens"
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return {
                "tokenIds": [token_id(i) for i in range(index, index + len(needle))],
                "normalizedName": name,
            }
    raise AssertionError(f"phrase {phrase!r} is not present in {query!r}")


def _difficulty(
    mode: str,
    level: str | None = None,
    distribution: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the v2 mutually exclusive difficulty payload for a mode."""
    if mode == "SINGLE":
        return {"mode": "SINGLE", "level": level}
    if mode == "CUSTOM":
        return {"mode": "CUSTOM", "distribution": distribution}
    return {"mode": mode}


class _RecordingInterpreter:
    """Stand-in for the Bedrock-routed interpreter; counts every model call."""

    def __init__(self, raw: str | Exception) -> None:
        self._raw = raw
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def interpret(self, **kwargs: Any) -> str:
        self.calls += 1
        self.last_kwargs = kwargs
        if isinstance(self._raw, Exception):
            raise self._raw
        return self._raw


def _resolve(query: str, interpreter: Any, **overrides: Any):
    kwargs: dict[str, Any] = {
        "request_id": "request-1",
        "user_id": "user-1",
        "conversation_id": "conversation-1",
        "turn_id": "turn-1",
        "query": query,
        "subject": "math",
        "topic": "Broad Subject",
        "difficulty": "intermediate",
        "language": "english",
        "exam_id": "CAT",
        "exam_stage": None,
        "request_interpreter": interpreter,
    }
    kwargs.update(overrides)
    return resolve_practice_request(**kwargs)


# ---------------------------------------------------------------------------
# 1 / 19 — the stable classifier is untouched
# ---------------------------------------------------------------------------


class TestClassifierFreeze:
    def test_classification_contract_has_no_intelligence_fields(self) -> None:
        fields = set(QueryClassification.model_fields)

        assert "topics" not in fields
        assert "needsIntelligence" not in fields
        assert "needs_intelligence" not in fields
        assert "requestIntelligence" not in fields

    def test_classifier_route_still_resolves_to_its_own_model_and_prompt(self) -> None:
        route = LlmConfigRegistry().get_route("general", "classifier", "default")

        assert route is not None
        assert route.model == "doubt_solver_classifier_gemini"
        assert route.prompt == "classification_semantics.md"
        assert route.overlays == ["query_classifier_text.md"]

    def test_classifier_prompts_carry_no_request_intelligence_instruction(self) -> None:
        for name in (
            "classification_semantics.md",
            "query_classifier_text.md",
            "query_classifier.md",
        ):
            text = (_APP_DIR / "prompts" / name).read_text(encoding="utf-8").casefold()
            assert "request intelligence" not in text
            assert "needsintelligence" not in text


# ---------------------------------------------------------------------------
# 2 / 20 — a caller that already holds trusted constraints pays nothing
# ---------------------------------------------------------------------------


class TestStructuredRequestBypass:
    def test_supplied_topics_skip_the_interpreter_entirely(self) -> None:
        interpreter = _RecordingInterpreter(_response(count=10))

        request = _resolve(
            "Create 10 questions",
            interpreter,
            topics=["Percentage", "Geometry"],
        )

        assert interpreter.calls == 0
        assert request.topics == ["Percentage", "Geometry"]

    def test_no_interpreter_configured_costs_no_call_and_changes_nothing(self) -> None:
        request = _resolve("Create 10 questions", None)

        assert request.topics is None
        assert request.difficulty_distribution is None


# ---------------------------------------------------------------------------
# 3 / 4 / 6 / 16 — free text reaches the seam and lands in the planner
# ---------------------------------------------------------------------------


class TestFreeTextReachesTheInterpreter:
    def test_free_text_practice_calls_the_interpreter_once_with_the_original_query(
        self,
    ) -> None:
        interpreter = _RecordingInterpreter(
            _response(count=20, topics=(("percentge", "Percentage"),),
                      query="Create 20 percentge questions")
        )

        _resolve("Create 20 percentge questions", interpreter)

        assert interpreter.calls == 1
        assert interpreter.last_kwargs is not None
        assert interpreter.last_kwargs["query"] == "Create 20 percentge questions"
        assert interpreter.last_kwargs["subject"] == "math"

    def test_multi_topic_interpretation_reaches_the_deterministic_planner(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(count=50, topics=_QUANT_TOPICS, mode="MIXED", query=_QUANT_QUERY)
        )

        request = _resolve(_QUANT_QUERY, interpreter)
        blueprint = deterministic_blueprint(request)

        assert request.topics == [name for _source, name in _QUANT_TOPICS]
        topic_counts = Counter(slot.topic_id for slot in blueprint.slots)
        assert len(topic_counts) == len(_QUANT_TOPICS)
        assert sum(topic_counts.values()) == 50
        # 50 slots over 11 topics: the planner, not the model, chose 5/5/…/4.
        assert sorted(topic_counts.values(), reverse=True) == [5] * 6 + [4] * 5

    def test_single_named_topic_is_forwarded_as_one_constraint(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(count=10, topics=(("Percentage", "Percentage"),),
                      query="Create 10 Percentage questions")
        )

        request = _resolve("Create 10 Percentage questions", interpreter)
        blueprint = deterministic_blueprint(request)

        assert request.topics == ["Percentage"]
        assert {slot.topic_id for slot in blueprint.slots} == {"percentage"}
        assert len(blueprint.slots) == 10


# ---------------------------------------------------------------------------
# 5 — a broad request must not become a fabricated syllabus
# ---------------------------------------------------------------------------


class TestBroadRequest:
    def test_broad_request_forwards_no_topics(self) -> None:
        interpreter = _RecordingInterpreter(_response(status="BROAD", count=50))

        request = _resolve("Give me 50 Quant questions", interpreter, topic="Quant")

        assert request.topics is None
        assert {slot.topic_id for slot in deterministic_blueprint(request).slots} == {"quant"}

    def test_broad_status_carrying_topics_is_rejected(self) -> None:
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(status="BROAD", count=50, topics=(("Quant", "Quant"),),
                          query="Give me 50 Quant questions"),
                query="Give me 50 Quant questions",
                explicit_count=50,
            )

        assert exc_info.value.reason_code == "PRACTICE_INTELLIGENCE_BROAD_TOPICS_PRESENT"


# ---------------------------------------------------------------------------
# 7 / 8 / 9 — difficulty: the model states, the planner distributes
# ---------------------------------------------------------------------------


class TestDifficultyConstraints:
    def test_mixed_difficulty_preserves_the_existing_planner_split(self) -> None:
        interpreter = _RecordingInterpreter(_response(count=50, topics=(), mode="MIXED"))
        # A query with no textual "mixed" signal, so the split can only come from
        # the interpretation reaching the existing deterministic mixed policy.
        request = _resolve("Create 50 questions", interpreter, topic="Quant")

        counts = Counter(slot.difficulty.value for slot in deterministic_blueprint(request).slots)

        assert request.mixed_difficulty_requested is True
        assert counts == {"basic": 17, "intermediate": 17, "advanced": 16}

    def test_single_difficulty_is_applied_to_every_slot(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(count=10, mode="SINGLE", single="ADVANCED", status="BROAD")
        )

        request = _resolve("Create 10 questions", interpreter, topic="Quant")

        assert request.difficulty is Difficulty.ADVANCED
        assert request.explicit_difficulty_requested is True
        assert all(
            slot.difficulty is Difficulty.ADVANCED
            for slot in deterministic_blueprint(request).slots
        )

    def test_custom_distribution_with_valid_counts_is_honoured_exactly(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(
                count=50,
                topics=(("fundamental rights", "Fundamental Rights"),),
                mode="CUSTOM",
                distribution={"basic": 10, "intermediate": 30, "advanced": 10},
                query=(
                    "Create 50 questions on fundamental rights: "
                    "10 easy, 30 medium, 10 hard"
                ),
            )
        )

        request = _resolve(
            "Create 50 questions on fundamental rights: 10 easy, 30 medium, 10 hard",
            interpreter,
        )
        counts = Counter(slot.difficulty.value for slot in deterministic_blueprint(request).slots)

        assert request.difficulty_distribution == {
            Difficulty.BASIC: 10,
            Difficulty.INTERMEDIATE: 30,
            Difficulty.ADVANCED: 10,
        }
        assert counts == {"basic": 10, "intermediate": 30, "advanced": 10}

    def test_custom_distribution_with_invalid_arithmetic_is_rejected(self) -> None:
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(
                    count=50,
                    mode="CUSTOM",
                    status="BROAD",
                    distribution={"basic": 10, "intermediate": 30, "advanced": 5},
                ),
                query="Create 50 questions: 10 easy, 30 medium, 5 hard",
                explicit_count=50,
            )

        assert (
            exc_info.value.reason_code
            == "PRACTICE_INTELLIGENCE_DIFFICULTY_ARITHMETIC_INVALID"
        )

    def test_invalid_arithmetic_falls_back_without_a_second_model_call(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(
                count=50,
                status="BROAD",
                mode="CUSTOM",
                distribution={"basic": 10, "intermediate": 30, "advanced": 5},
            )
        )

        request = _resolve("Create 50 questions", interpreter, topic="Quant")

        assert interpreter.calls == 1
        assert request.difficulty_distribution is None
        assert request.topics is None

class TestTokenIdGrounding:
    """v4: grounding is a selection of labels the model can see."""

    def test_model_cannot_fabricate_source_text(self) -> None:
        """v2 blocker: a model asked to copy "औसत" returned "ओसत"."""
        query = "मुझे औसत और प्रतिशत के 20 कठिन सवाल दो"
        intelligence = parse_request_intelligence(
            _response(count=20, topics=(("औसत", "Average"), ("प्रतिशत", "Percentage")),
                      mode="SINGLE", single="ADVANCED", query=query),
            query=query,
            explicit_count=None,
        )

        assert [t.source_text for t in intelligence.topics] == ["औसत", "प्रतिशत"]
        assert [t.normalized_name for t in intelligence.topics] == ["Average", "Percentage"]

    def test_source_text_is_reconstructed_verbatim_from_a_misspelled_query(self) -> None:
        query = "creat 20 questin on percentge and profit n loss"
        intelligence = parse_request_intelligence(
            _response(count=20,
                      topics=(("percentge", "Percentage"), ("profit n loss", "Profit & Loss")),
                      query=query),
            query=query,
            explicit_count=None,
        )

        assert [t.source_text for t in intelligence.topics] == ["percentge", "profit n loss"]

    def test_compound_topic_selection_keeps_its_separators(self) -> None:
        query = "Create 20 questions on Time, Speed & Distance"
        intelligence = parse_request_intelligence(
            _response(count=20,
                      topics=(("Time, Speed & Distance", "Time, Speed & Distance"),),
                      query=query),
            query=query,
            explicit_count=20,
        )

        assert intelligence.topics[0].source_text == "Time, Speed & Distance"

    @pytest.mark.parametrize(
        ("ids", "reason", "why"),
        (
            (["T99"], "PRACTICE_INTELLIGENCE_TOPIC_TOKEN_UNKNOWN", "id past the last token"),
            (["X1"], "PRACTICE_INTELLIGENCE_TOPIC_TOKEN_UNKNOWN", "not a token label"),
            (["T-1"], "PRACTICE_INTELLIGENCE_TOPIC_TOKEN_UNKNOWN", "negative"),
            ([], "PRACTICE_INTELLIGENCE_TOPIC_TOKEN_EMPTY", "no ids at all"),
            (["T4", "T4"], "PRACTICE_INTELLIGENCE_TOPIC_TOKEN_DUPLICATED", "same id twice"),
            (["T5", "T4"], "PRACTICE_INTELLIGENCE_TOPIC_TOKEN_UNORDERED", "reverse order"),
            (["T4", "T6"], "PRACTICE_INTELLIGENCE_TOPIC_TOKEN_NONCONTIGUOUS", "gap"),
        ),
    )
    def test_invalid_selections_are_rejected(
        self, ids: list[str], reason: str, why: str
    ) -> None:
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(count=20, raw_topics=[
                    {"tokenIds": ids, "normalizedName": "Percentage"}]),
                query="Create 20 questions on percentage and geometry",
                explicit_count=20,
            )

        assert exc_info.value.reason_code == reason

    def test_duplicate_selections_collapse_to_one_constraint(self) -> None:
        query = "Create 20 questions on percentage and geometry"
        intelligence = parse_request_intelligence(
            _response(count=20, raw_topics=[
                {"tokenIds": ["T4"], "normalizedName": "Percentage"},
                {"tokenIds": ["T4"], "normalizedName": "Percentage"},
                {"tokenIds": ["T6"], "normalizedName": "Geometry"}]),
            query=query,
            explicit_count=20,
        )

        assert [t.normalized_name for t in intelligence.topics] == ["Percentage", "Geometry"]

    def test_partially_overlapping_selections_are_rejected(self) -> None:
        """Two readings of the same words is a contradiction, not a repetition."""
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(count=20, raw_topics=[
                    {"tokenIds": ["T4", "T5", "T6"], "normalizedName": "Time & Speed"},
                    {"tokenIds": ["T5", "T6", "T7"], "normalizedName": "Speed & Distance"}]),
                query="Create 20 questions on Time Speed and Distance",
                explicit_count=20,
            )

        assert exc_info.value.reason_code == "PRACTICE_INTELLIGENCE_TOPIC_SPAN_OVERLAP"

    def test_duplicate_normalized_names_collapse(self) -> None:
        intelligence = parse_request_intelligence(
            _response(count=20, raw_topics=[
                {"tokenIds": ["T4"], "normalizedName": "Percentage"},
                {"tokenIds": ["T6"], "normalizedName": "percentage"}]),
            query="Create 20 questions on Percentage and percentage",
            explicit_count=20,
        )

        assert len(intelligence.topics) == 1

    def test_empty_normalized_name_is_rejected(self) -> None:
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(count=20, raw_topics=[
                    {"tokenIds": ["T4"], "normalizedName": "   "}]),
                query="Create 20 questions on percentage",
                explicit_count=20,
            )

        assert exc_info.value.reason_code == "PRACTICE_INTELLIGENCE_TOPIC_NAME_EMPTY"

    def test_absurdly_large_topic_list_is_rejected(self) -> None:
        query = "Create 50 questions on " + " ".join(
            f"topic{index}" for index in range(MAX_INTERPRETED_TOPICS + 1))
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(count=50, raw_topics=[
                    {"tokenIds": [token_id(4 + i)], "normalizedName": f"Topic {i}"}
                    for i in range(MAX_INTERPRETED_TOPICS + 1)]),
                query=query,
                explicit_count=50,
            )

        assert exc_info.value.reason_code == "PRACTICE_INTELLIGENCE_TOPIC_LIMIT_EXCEEDED"

    def test_token_labels_round_trip(self) -> None:
        assert token_id(0) == "T0"
        assert token_index("T7") == 7
        assert token_index("nope") is None
        assert token_index("T") is None


class TestDeterministicTokenizer:
    @pytest.mark.parametrize(
        ("query", "expected"),
        (
            ("Create 20 questions on percentage",
             ["Create", "20", "questions", "on", "percentage"]),
            ("मुझे औसत और प्रतिशत के 20 कठिन सवाल दो",
             ["मुझे", "औसत", "और", "प्रतिशत", "के", "20", "कठिन", "सवाल", "दो"]),
            ("percentage aur ratio proporton ke 20 sawal",
             ["percentage", "aur", "ratio", "proporton", "ke", "20", "sawal"]),
            ("30 sawal chahiye:  10 easy,   15 medium!",
             ["30", "sawal", "chahiye", "10", "easy", "15", "medium"]),
            ("मुझे percentage aur average ke 20 mixed sawal do",
             ["मुझे", "percentage", "aur", "average", "ke", "20", "mixed", "sawal", "do"]),
        ),
        ids=("english", "hindi", "hinglish", "punctuation-and-whitespace", "mixed-script"),
    )
    def test_tokenization(self, query: str, expected: list[str]) -> None:
        assert [t.text for t in tokenize_query(query)] == expected

    def test_combining_marks_stay_inside_their_token(self) -> None:
        """Devanagari matras and virama are marks, not alphanumerics — they must not split."""
        tokens = tokenize_query("प्रतिशत")

        assert len(tokens) == 1
        assert tokens[0].text == "प्रतिशत"

    def test_tokenization_is_deterministic(self) -> None:
        query = "मुझे औसत और प्रतिशत के 20 कठिन सवाल दो"

        assert tokenize_query(query) == tokenize_query(query)

    def test_token_bounds_index_the_original_string(self) -> None:
        query = "Create 20 questions on Time, Speed & Distance"
        tokens = tokenize_query(query)

        for token in tokens:
            assert query[token.start : token.end] == token.text

    def test_reconstruction_spans_original_separators(self) -> None:
        query = "Create 20 questions on Time, Speed & Distance"
        tokens = tokenize_query(query)

        assert reconstruct_source_text(query, tokens, [4, 5, 6, 7]) == "Time, Speed & Distance"

    def test_empty_query_yields_no_tokens(self) -> None:
        assert tokenize_query("   ") == ()


# ---------------------------------------------------------------------------
# 12 — ambiguity and other unusable interpretations
# ---------------------------------------------------------------------------


class TestUnusableInterpretations:
    def test_ambiguous_status_is_discarded_and_never_retried(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(status="AMBIGUOUS", topics=(), count=None)
        )

        request = _resolve("kuch bhi bana do", interpreter, topic="Quant")

        assert interpreter.calls == 1
        assert request.topics is None
        assert request.difficulty_distribution is None

    def test_conflicting_explicit_totals_are_ambiguous(self) -> None:
        query = "Create 10 questions, actually make it 20 questions from Geography"
        intelligence = parse_request_intelligence(
            _response(status="AMBIGUOUS", topics=(), count=None),
            query=query,
            explicit_count=None,
        )

        assert intelligence.interpretation_status == "AMBIGUOUS"
        assert intelligence.requested_count is None
        assert intelligence.topics == []

    def test_count_disagreement_with_the_deterministic_authority_is_rejected(self) -> None:
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(count=30, status="BROAD"),
                query="Create 50 questions",
                explicit_count=50,
            )

        assert exc_info.value.reason_code == "PRACTICE_INTELLIGENCE_COUNT_MISMATCH"

    def test_out_of_bounds_count_is_rejected(self) -> None:
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(count=1000, status="BROAD"),
                query="Create 1000 questions",
                explicit_count=1000,
            )

        assert exc_info.value.reason_code == "PRACTICE_INTELLIGENCE_COUNT_OUT_OF_BOUNDS"

    def test_malformed_payload_falls_back_safely(self) -> None:
        interpreter = _RecordingInterpreter("not json at all")

        request = _resolve("Create 10 questions", interpreter, topic="Quant")

        assert interpreter.calls == 1
        assert request.topics is None

    def test_provider_failure_falls_back_to_the_deterministic_request(self) -> None:
        interpreter = _RecordingInterpreter(RuntimeError("bedrock unavailable"))

        request = _resolve("Create 10 questions", interpreter, topic="Quant")

        assert interpreter.calls == 1
        assert request.topics is None
        assert len(deterministic_blueprint(request).slots) == 10


# ---------------------------------------------------------------------------
# Language — one model, one prompt, one schema for EN / HI / Hinglish
# ---------------------------------------------------------------------------


class TestLanguageFixtures:
    @pytest.mark.parametrize(
        ("query", "topics"),
        (
            (
                "Create 20 questions on percentage and ratio",
                (("percentage", "Percentage"), ("ratio", "Ratio & Proportion")),
            ),
            (
                "प्रतिशत और अनुपात पर 20 प्रश्न बनाओ",
                (("प्रतिशत", "Percentage"), ("अनुपात", "Ratio & Proportion")),
            ),
            (
                "percentge aur ratio proporton ke 20 sawal banao",
                (
                    ("percentge", "Percentage"),
                    ("ratio proporton", "Ratio & Proportion"),
                ),
            ),
        ),
        ids=("english", "hindi", "hinglish"),
    )
    def test_every_language_uses_the_same_validation_path(
        self,
        query: str,
        topics: tuple[tuple[str, str], ...],
    ) -> None:
        intelligence = parse_request_intelligence(
            _response(count=20, topics=topics, query=query),
            query=query,
            explicit_count=20,
        )

        assert [topic.normalized_name for topic in intelligence.topics] == [
            "Percentage",
            "Ratio & Proportion",
        ]


# ---------------------------------------------------------------------------
# 17 / 18 — existing behaviour is unchanged
# ---------------------------------------------------------------------------


class TestExistingBehaviourPreserved:
    def test_broad_status_is_only_valid_without_explicit_composition(self) -> None:
        query = "Give me 50 Geography questions"
        intelligence = parse_request_intelligence(
            _response(status="BROAD", count=50), query=query, explicit_count=50
        )
        assert intelligence.interpretation_status == "BROAD"
        assert intelligence.topics == []

    def test_multi_subject_composition_is_resolved_and_grounded(self) -> None:
        query = "Give me 50 Geography and Polity questions"
        intelligence = parse_request_intelligence(
            _response(
                status="RESOLVED",
                count=50,
                query=query,
                raw_topics=[
                    {**_span(query, "Geography", "Geography"), "subjectId": "geography"},
                    {**_span(query, "Polity", "Polity"), "subjectId": "polity"},
                ],
            ),
            query=query,
            explicit_count=50,
        )
        assert [(item.normalized_name, item.subject_id) for item in intelligence.topics] == [
            ("Geography", "geography"),
            ("Polity", "polity"),
        ]

    def test_subject_with_topics_and_multi_topic_same_subject_are_resolved(self) -> None:
        subject_query = "50 Polity questions on Parliament and Fundamental Rights"
        topics_query = "20 Percentage and Ratio questions"
        assert parse_request_intelligence(
            _response(
                status="RESOLVED",
                count=50,
                query=subject_query,
                topics=(("Parliament", "Parliament"), ("Fundamental Rights", "Fundamental Rights")),
            ),
            query=subject_query,
            explicit_count=50,
        ).interpretation_status == "RESOLVED"
        assert parse_request_intelligence(
            _response(
                status="RESOLVED",
                count=20,
                query=topics_query,
                topics=(("Percentage", "Percentage"), ("Ratio", "Ratio")),
            ),
            query=topics_query,
            explicit_count=20,
        ).interpretation_status == "RESOLVED"

    def test_repeated_selection_with_another_family_is_contradictory(self) -> None:
        query = "Create 10 questions from Geography"
        payload = json.loads(
            _response(count=10, topics=(("Geography", "Geography"),), query=query)
        )
        payload["topics"].append({**payload["topics"][0], "subjectId": "economics"})
        with pytest.raises(RequestIntelligenceError, match="TOPIC_SPAN_OVERLAP"):
            parse_request_intelligence(
                json.dumps(payload), query=query, explicit_count=10
            )

    def test_explicit_multi_subjects_preserve_per_constraint_routing(self) -> None:
        query = "Create 50 questions from Geography, Polity, History, Science and Economy"
        subject_ids = {
            "Geography": "geography",
            "Polity": "polity",
            "History": "history",
            "Science": "science",
            "Economy": "economics",
        }
        payload = json.loads(
            _response(
                count=50,
                query=query,
                topics=tuple((name, name) for name in subject_ids),
            )
        )
        for item in payload["topics"]:
            item["subjectId"] = subject_ids[item["normalizedName"]]
        request = _resolve(
            query,
            _RecordingInterpreter(json.dumps(payload)),
            subject="general",
            topic=None,
        )

        assert [
            (item.topic_id, item.subject_id, item.source_text)
            for item in request.trusted_constraints
        ] == [
            ("geography", "geography", "Geography"),
            ("polity", "polity", "Polity"),
            ("history", "history", "History"),
            ("science", "science", "Science"),
            ("economy", "economics", "Economy"),
        ]
        blueprint = deterministic_blueprint(request)
        assert {slot.subject_id for slot in blueprint.slots} == set(subject_ids.values())
        assert {
            (slot.topic_id, slot.subject_id) for slot in blueprint.slots
        } == set((item.topic_id, item.subject_id) for item in request.trusted_constraints)

    def test_exact_hindi_subject_labels_are_grounded(self) -> None:
        query = "भूगोल, राजनीति, इतिहास, विज्ञान और अर्थशास्त्र से 50 प्रश्न बनाइए।"
        subject_ids = {
            "भूगोल": "geography",
            "राजनीति": "polity",
            "इतिहास": "history",
            "विज्ञान": "science",
            "अर्थशास्त्र": "economics",
        }
        intelligence = parse_request_intelligence(
            _response(
                count=50,
                query=query,
                raw_topics=[
                    {**_span(query, name, name), "subjectId": subject_id}
                    for name, subject_id in subject_ids.items()
                ],
            ),
            query=query,
            explicit_count=50,
        )

        assert [topic.subject_id for topic in intelligence.topics] == list(subject_ids.values())

    def test_planner_regression_without_any_interpretation(self) -> None:
        """The deterministic planner behaves exactly as before when no model runs."""
        request = _resolve("Create 12 questions with mixed difficulty", None, topic="Algebra")
        blueprint = deterministic_blueprint(request)

        assert len(blueprint.slots) == 12
        assert {slot.topic_id for slot in blueprint.slots} == {"algebra"}
        assert Counter(slot.difficulty.value for slot in blueprint.slots) == {
            "basic": 4,
            "intermediate": 4,
            "advanced": 4,
        }

    def test_non_practice_request_never_reaches_the_interpreter(self, monkeypatch) -> None:
        monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
        interpreter = _RecordingInterpreter(_response(status="BROAD", count=10))

        class _Adapter:
            def generate_final(self, **_kwargs):
                return _final_answer()

        build_orchestrated_doubt_solver_graph(
            _Adapter(),
            practice_launcher=lambda _request: _launch_result(),
            practice_request_interpreter=interpreter,
        ).invoke(_solve_state())

        assert interpreter.calls == 0

    def test_practice_creation_request_reaches_the_interpreter_through_the_graph(
        self,
        monkeypatch,
    ) -> None:
        monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
        interpreter = _RecordingInterpreter(
            _response(count=5, topics=(("algebra", "Algebra"),),
                      query="Create five algebra questions")
        )
        launched: list[Any] = []

        class _Adapter:
            def generate_final(self, **_kwargs):
                raise AssertionError("practice creation must not generate an answer")

        def _launcher(request):
            launched.append(request)
            return _launch_result()

        build_orchestrated_doubt_solver_graph(
            _Adapter(),
            practice_launcher=_launcher,
            practice_request_interpreter=interpreter,
        ).invoke(_practice_state())

        assert interpreter.calls == 1
        assert launched and launched[0].topics == ["Algebra"]


# ---------------------------------------------------------------------------
# Static schema and route wiring
# ---------------------------------------------------------------------------


class TestStaticSchemaAndRoute:
    def test_schema_uses_only_bedrock_supported_constructs(self) -> None:
        banned = {"minimum", "maximum", "minLength", "maxLength", "maxItems", "minItems"}
        serialized = json.dumps(PRACTICE_REQUEST_INTELLIGENCE_SCHEMA)

        assert not any(f'"{key}"' in serialized for key in banned)
        assert PRACTICE_REQUEST_INTELLIGENCE_SCHEMA["additionalProperties"] is False
        assert (
            PRACTICE_REQUEST_INTELLIGENCE_SCHEMA["properties"]["topics"]["items"][
                "additionalProperties"
            ]
            is False
        )

    def test_route_resolves_to_the_qualified_terra_model(self) -> None:
        registry = LlmConfigRegistry()
        route = registry.get_route("general", "request_intelligence", "default")

        assert route is not None
        assert route.prompt == "practice_generation/request_intelligence.md"
        assert route.fallback == []
        model = registry.get_model(route.model)
        assert model is not None
        assert model.provider == "azure_openai"
        assert model.deployment == "gpt-5.6-terra"
        # Its existing Azure fallback is generator/verifier-only, so the normal
        # role gate retains deterministic recovery for an unavailable interpreter.
        assert model.fallback_models == ["azure_deepseek_v4_pro"]
        fallback = registry.get_model(model.fallback_models[0])
        assert fallback is not None
        assert fallback.allowed_task_roles == ["generator", "verifier"]

    def test_prompt_never_asks_the_model_to_emit_json_itself(self) -> None:
        text = (
            _APP_DIR / "prompts" / "practice_generation" / "request_intelligence.md"
        ).read_text(encoding="utf-8").casefold()

        assert "json" not in text
        assert "```" not in text


# ---------------------------------------------------------------------------
# Shared graph fixtures
# ---------------------------------------------------------------------------


def _launch_result() -> PracticeLaunchResult:
    return PracticeLaunchResult(
        test_id="practice-test-1",
        status="GENERATING",
        requested_count=5,
        accepted_count=5,
        count_clamped=False,
        progress_percent=0,
        playable=False,
        message="Your practice set is being created.",
    )


def _final_answer() -> Any:
    from services.doubt_solver.final_answer import build_final_answer_result

    return build_final_answer_result(
        content="**Final Answer:** 20",
        language="english",
        quality_status="checked",
    )


def _base_state(query: str, intent: str) -> dict:
    return {
        "request_id": "request-1",
        "actor_id": "user-1",
        "conversation_id": "conversation-1",
        "turn_id": "turn-1",
        "query": query,
        "original_query": query,
        "language": "english",
        "exam_id": "CAT",
        "exam_stage": None,
        "classification": {
            "intent": intent,
            "subject": "math",
            "topic": "algebra",
            "difficulty": "intermediate",
            "retrieval_required": False,
        },
        "query_classification": None,
        "retrieval_context": {},
        "context_text": "",
        "answer": None,
        "final_answer": None,
        "conversation_context": "",
        "conversation_relation": None,
        "conversation_preparation": None,
        "source_modality": "text",
    }


def _practice_state() -> dict:
    return _base_state("Create five algebra questions", "practice")


def _solve_state() -> dict:
    return _base_state("What is 20 percent of 100?", "solve")


# ---------------------------------------------------------------------------
# V2 — difficulty is mutually exclusive by construction
# ---------------------------------------------------------------------------


class TestDifficultyContract:
    def test_schema_name_is_versioned(self) -> None:
        assert PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME == (
            "practice_request_intelligence_v5"
        )

    def test_difficulty_is_a_closed_anyof_branch_per_mode(self) -> None:
        branches = PRACTICE_REQUEST_INTELLIGENCE_SCHEMA["properties"]["difficulty"]["anyOf"]

        modes = {branch["properties"]["mode"]["const"] for branch in branches}
        assert modes == {"UNSPECIFIED", "SINGLE", "MIXED", "CUSTOM"}
        for branch in branches:
            assert branch["additionalProperties"] is False
            assert branch["properties"]["mode"]["type"] == "string"

    def test_only_single_carries_a_level_and_only_custom_a_distribution(self) -> None:
        branches = {
            branch["properties"]["mode"]["const"]: branch
            for branch in PRACTICE_REQUEST_INTELLIGENCE_SCHEMA["properties"]["difficulty"]["anyOf"]
        }

        assert set(branches["SINGLE"]["required"]) == {"mode", "level"}
        assert set(branches["CUSTOM"]["required"]) == {"mode", "distribution"}
        for mode in ("UNSPECIFIED", "MIXED"):
            assert set(branches[mode]["properties"]) == {"mode"}
            assert branches[mode]["required"] == ["mode"]

    def test_schema_uses_only_bedrock_supported_constructs(self) -> None:
        banned = {"minimum", "maximum", "minLength", "maxLength", "maxItems",
                  "minItems", "minProperties", "maxProperties", "pattern",
                  "oneOf", "allOf", "not", "$ref", "if", "then", "else"}
        serialized = json.dumps(PRACTICE_REQUEST_INTELLIGENCE_SCHEMA)

        assert not [key for key in banned if f'"{key}"' in serialized]

    @pytest.mark.parametrize(
        "payload",
        (
            {"mode": "UNSPECIFIED"},
            {"mode": "SINGLE", "level": "ADVANCED"},
            {"mode": "MIXED"},
            {"mode": "CUSTOM", "distribution": {"basic": 2, "intermediate": 2, "advanced": 1}},
        ),
        ids=("unspecified", "single", "mixed", "custom"),
    )
    def test_every_valid_mode_deserializes(self, payload: dict[str, Any]) -> None:
        intelligence = parse_request_intelligence(
            json.dumps({
                "interpretationStatus": "BROAD", "requestedCount": 5,
                "topics": [], "difficulty": payload,
            }),
            query="Create 5 questions",
            explicit_count=5,
        )

        assert intelligence.difficulty.mode == payload["mode"]

    @pytest.mark.parametrize(
        ("payload", "why"),
        (
            ({"mode": "SINGLE"}, "SINGLE without a level"),
            ({"mode": "MIXED", "level": "BASIC"}, "MIXED with a single level"),
            ({"mode": "UNSPECIFIED", "level": "BASIC"}, "UNSPECIFIED with a single level"),
            ({"mode": "CUSTOM"}, "CUSTOM without its distribution"),
            (
                {"mode": "CUSTOM", "level": "BASIC",
                 "distribution": {"basic": 5, "intermediate": 0, "advanced": 0}},
                "CUSTOM with a single level",
            ),
        ),
    )
    def test_impossible_combinations_are_rejected_by_the_contract(
        self,
        payload: dict[str, Any],
        why: str,
    ) -> None:
        """These shapes are unrepresentable in the grammar; the model mirrors it."""
        with pytest.raises(ValidationError):
            parse_request_intelligence(
                json.dumps({
                    "interpretationStatus": "BROAD", "requestedCount": 5,
                    "topics": [], "difficulty": payload,
                }),
                query="Create 5 questions",
                explicit_count=5,
            )

    def test_negative_distribution_counts_are_rejected_deterministically(self) -> None:
        """The grammar cannot express non-negativity, so Python still must."""
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(
                    count=5, status="BROAD", mode="CUSTOM",
                    distribution={"basic": -1, "intermediate": 3, "advanced": 3},
                ),
                query="Create 5 questions",
                explicit_count=5,
            )

        assert exc_info.value.reason_code == "PRACTICE_INTELLIGENCE_DIFFICULTY_INVALID"


# ---------------------------------------------------------------------------
# Fix 2 — a default is not evidence; only a written count is
# ---------------------------------------------------------------------------


class TestCountAuthority:
    def test_explicit_count_matching_the_interpretation_is_accepted(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(count=20, topics=(("percentage", "Percentage"),),
                      query="Create 20 questions on percentage")
        )

        request = _resolve("Create 20 questions on percentage", interpreter)

        assert request.accepted_count == 20
        assert request.requested_count == 20
        assert request.topics == ["Percentage"]

    def test_explicit_count_conflicting_with_the_interpretation_is_rejected(self) -> None:
        """Neither value is silently preferred; the whole interpretation is dropped."""
        interpreter = _RecordingInterpreter(
            _response(count=30, topics=(("percentage", "Percentage"),),
                      query="Create 20 questions on percentage")
        )

        request = _resolve("Create 20 questions on percentage", interpreter)

        assert interpreter.calls == 1
        assert request.accepted_count == 20
        assert request.topics is None

    def test_hindi_count_is_read_deterministically_and_agrees_with_the_model(self) -> None:
        query = "मुझे औसत और प्रतिशत के 20 कठिन सवाल दो"
        assert explicit_requested_count(query) == 20, "the parser reads the Devanagari unit"
        interpreter = _RecordingInterpreter(
            _response(
                count=20,
                topics=(("औसत", "Average"), ("प्रतिशत", "Percentage")),
                mode="SINGLE",
                single="ADVANCED",
                query=query,
            )
        )

        request = _resolve(query, interpreter, subject="math")

        assert request.accepted_count == 20
        assert request.topics == ["Average", "Percentage"]
        assert request.difficulty is Difficulty.ADVANCED

    def test_typo_unit_count_is_read_deterministically_and_agrees_with_the_model(self) -> None:
        query = "creat 20 questin on percentge and profit n loss"
        assert explicit_requested_count(query) == 20, "the parser reads the typo'd unit"
        interpreter = _RecordingInterpreter(
            _response(
                count=20,
                topics=(("percentge", "Percentage"), ("profit n loss", "Profit & Loss")),
                query=query,
            )
        )

        request = _resolve(query, interpreter)

        assert request.accepted_count == 20
        assert request.topics == ["Percentage", "Profit & Loss"]

    def test_no_count_anywhere_keeps_the_existing_default(self) -> None:
        query = "reasoning practice karwa do"
        interpreter = _RecordingInterpreter(_response(status="BROAD", count=None))

        request = _resolve(query, interpreter, topic="Reasoning")

        assert request.accepted_count == resolve_requested_count(
            query, resolve_practice_type(query)
        )
        assert request.accepted_count == 5

    def test_trusted_structured_count_needs_no_interpretation(self) -> None:
        interpreter = _RecordingInterpreter(_response(count=99))

        request = _resolve(
            "Create 20 questions on percentage", interpreter, requested_count=30
        )

        assert interpreter.calls == 0
        assert request.accepted_count == 30

    def test_out_of_bounds_resolved_count_is_rejected_deterministically(self) -> None:
        interpreter = _RecordingInterpreter(_response(count=None))

        with pytest.raises(PracticeRequestCountError):
            _resolve("Create 10 questions", interpreter, requested_count=0)


class TestCustomDistributionTotals:
    def test_distribution_matching_the_total_is_honoured(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(
                count=20, topics=(("geometry", "Geometry"),), mode="CUSTOM",
                distribution={"basic": 5, "intermediate": 10, "advanced": 5},
                query="Create 20 questions on geometry",
            )
        )

        request = _resolve("Create 20 questions on geometry", interpreter)
        counts = Counter(s.difficulty.value for s in deterministic_blueprint(request).slots)

        assert request.accepted_count == 20
        assert counts == {"basic": 5, "intermediate": 10, "advanced": 5}

    def test_distribution_conflicting_with_the_stated_total_is_rejected(self) -> None:
        with pytest.raises(RequestIntelligenceError) as exc_info:
            parse_request_intelligence(
                _response(
                    count=20, status="BROAD", mode="CUSTOM",
                    distribution={"basic": 5, "intermediate": 10, "advanced": 1},
                ),
                query="Create 20 questions",
                explicit_count=20,
            )

        assert (
            exc_info.value.reason_code
            == "PRACTICE_INTELLIGENCE_DIFFICULTY_ARITHMETIC_INVALID"
        )

    def test_total_is_implied_by_the_distribution_when_nobody_states_one(self) -> None:
        """The breakdown is itself a statement of the total; we sum it, not the model."""
        query = "Polity questions on fundamental rights and parliament: 10 easy, 30 medium, 10 hard"
        assert explicit_requested_count(query) is None, "premise: parser finds no count"
        interpreter = _RecordingInterpreter(
            _response(
                count=None,
                topics=(("fundamental rights", "Fundamental Rights"), ("parliament", "Parliament")),
                mode="CUSTOM",
                distribution={"basic": 10, "intermediate": 30, "advanced": 10},
                query=query,
            )
        )

        request = _resolve(query, interpreter, subject="general")
        counts = Counter(s.difficulty.value for s in deterministic_blueprint(request).slots)

        assert request.accepted_count == 50
        assert counts == {"basic": 10, "intermediate": 30, "advanced": 10}

    def test_implied_total_above_the_v1_cap_is_proportionally_preserved(self) -> None:
        query = "Give me a practice set: 100 easy, 100 medium and 100 hard"
        interpreter = _RecordingInterpreter(
            _response(
                count=None,
                status="BROAD",
                mode="CUSTOM",
                distribution={"basic": 100, "intermediate": 100, "advanced": 100},
            )
        )

        request = _resolve(query, interpreter)
        counts = Counter(slot.difficulty.value for slot in deterministic_blueprint(request).slots)

        assert request.requested_count == 300
        assert request.accepted_count == 50
        assert counts == {"basic": 17, "intermediate": 17, "advanced": 16}


class TestHardeningPreservesExistingBehaviour:
    def test_single_topic_practice_is_unchanged(self) -> None:
        interpreter = _RecordingInterpreter(
            _response(count=10, topics=(("Percentage", "Percentage"),),
                      query="Create 10 Percentage questions")
        )

        request = _resolve("Create 10 Percentage questions", interpreter)
        blueprint = deterministic_blueprint(request)

        assert request.topics == ["Percentage"]
        assert {slot.topic_id for slot in blueprint.slots} == {"percentage"}
        assert len(blueprint.slots) == 10

    def test_planner_remains_deterministic_without_any_interpretation(self) -> None:
        request = _resolve("Create 12 questions with mixed difficulty", None, topic="Algebra")
        blueprint = deterministic_blueprint(request)

        assert len(blueprint.slots) == 12
        assert Counter(s.difficulty.value for s in blueprint.slots) == {
            "basic": 4, "intermediate": 4, "advanced": 4,
        }

    def test_resolved_count_topics_and_distribution_survive_persistence(self) -> None:
        """A resumed Practice must rehydrate the same constraints it was planned with."""
        from features.practice_generation.orchestration import _request as rehydrate

        interpreter = _RecordingInterpreter(
            _response(
                count=20,
                topics=(("geometry", "Geometry"), ("percentage", "Percentage")),
                mode="CUSTOM",
                distribution={"basic": 5, "intermediate": 10, "advanced": 5},
                query="creat 20 questin on geometry and percentage",
            )
        )
        original = _resolve(
            "creat 20 questin on geometry and percentage", interpreter
        )
        assert original.accepted_count == 20

        stored = {
            "userId": original.user_id,
            "name": original.assessment_title,
            "meta": {
                "practiceRequest": {
                    "requestId": original.request_id,
                    "conversationId": original.conversation_id,
                    "turnId": original.turn_id,
                    "querySummary": original.original_query,
                    "practiceType": original.practice_type.value,
                    "requestedCount": original.requested_count,
                    "acceptedCount": original.accepted_count,
                    "subject": original.subject,
                    "topic": original.topic,
                    "topics": original.topics,
                    "difficulty": original.difficulty.value,
                    "mixedDifficultyRequested": original.mixed_difficulty_requested,
                    "explicitDifficultyRequested": original.explicit_difficulty_requested,
                    "difficultyDistribution": {
                        level.value: count
                        for level, count in original.difficulty_distribution.items()
                    },
                    "language": original.language,
                    "languageSource": original.language_source,
                    "examId": original.exam_id,
                    "examStage": original.exam_stage,
                    "examProfileId": original.exam_profile_id,
                    "sourceQuestionReference": original.source_question_reference,
                    "requiresFreshEvidence": original.requires_fresh_evidence,
                    "freshnessReason": original.freshness_reason,
                    "freshEvidence": None,
                    "includeSolutions": original.include_solutions,
                    "assessmentTitle": original.assessment_title,
                }
            },
        }

        restored = rehydrate(stored)

        assert restored.accepted_count == original.accepted_count == 20
        assert restored.topics == original.topics
        assert restored.difficulty_distribution == original.difficulty_distribution
        assert deterministic_blueprint(restored).slots == deterministic_blueprint(original).slots
