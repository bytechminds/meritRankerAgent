"""Deterministic planning route: count decides which planner runs."""

from __future__ import annotations

import json

import pytest

from features.practice_generation.planning import (
    INTELLIGENCE_PLANNING_COUNT_THRESHOLD,
    deterministic_blueprint,
    select_planning_mode,
)
from features.practice_generation.schemas import (
    Difficulty,
    PracticeGenerationRequest,
    PracticeType,
)


def _request(
    count: int,
    *,
    query: str = "Create questions on math",
    practice_type: PracticeType = PracticeType.QUICK_PRACTICE,
    subject: str = "math",
) -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="r",
        user_id="u",
        conversation_id="c",
        turn_id="t",
        original_query=query,
        practice_type=practice_type,
        requested_count=count,
        accepted_count=count,
        subject=subject,
        difficulty=Difficulty.INTERMEDIATE,
        assessment_title="T",
    )


@pytest.mark.parametrize("count", [1, 2, 5, 10, 19, 20])
def test_counts_up_to_the_threshold_stay_deterministic(count: int) -> None:
    mode, _ = select_planning_mode(_request(count))
    assert mode == "DETERMINISTIC"


@pytest.mark.parametrize("count", [21, 30, 50])
def test_counts_above_the_threshold_use_the_intelligence_planner(count: int) -> None:
    mode, reason = select_planning_mode(_request(count))
    assert mode == "INTELLIGENCE"
    assert reason == "COUNT_OVER_THRESHOLD"


def test_the_threshold_boundary_is_exclusive() -> None:
    assert INTELLIGENCE_PLANNING_COUNT_THRESHOLD == 20
    assert select_planning_mode(_request(20))[0] == "DETERMINISTIC"
    assert select_planning_mode(_request(21))[0] == "INTELLIGENCE"


def test_large_count_overrides_the_quick_practice_deterministic_rule() -> None:
    """A QUICK_PRACTICE request is deterministic only while it stays small."""
    assert select_planning_mode(_request(10))[0] == "DETERMINISTIC"
    assert select_planning_mode(_request(30))[0] == "INTELLIGENCE"


def test_small_non_quick_requests_keep_their_existing_route() -> None:
    mode, reason = select_planning_mode(
        _request(10, practice_type=PracticeType.TOPIC_TEST)
    )
    assert (mode, reason) == ("INTELLIGENCE", "EXISTING_PRACTICE_TYPE_RULE")


MULTILINGUAL_QUERIES = [
    "Create 30 questions on percentage and ratio proportion",
    "मुझे औसत और प्रतिशत के 30 कठिन सवाल दो",
    "30 sawal banao average aur percentage pe mixed difficulty ke saath",
    "Create 30 questions on प्रतिशत and Time & Work",
    "creat 30 questin on percentge and ratio proporton",
]


@pytest.mark.parametrize("query", MULTILINGUAL_QUERIES)
def test_the_original_query_reaches_the_planner_unmodified(query: str) -> None:
    """The planner reads the student's own wording, not a rewritten copy."""
    request = _request(30, query=query)
    assert request.original_query == query
    assert select_planning_mode(request)[0] == "INTELLIGENCE"


@pytest.mark.parametrize("count", [1, 10, 20])
def test_deterministic_blueprint_still_produces_the_requested_total(count: int) -> None:
    blueprint = deterministic_blueprint(_request(count))
    assert len(blueprint.slots) == count


class _CountingInterpreter:
    """Fails the test if the Practice path ever asks it to interpret."""

    def __init__(self) -> None:
        self.calls = 0

    def interpret(self, **_kwargs: object) -> str:
        self.calls += 1
        raise AssertionError("the Practice path must not run an interpretation call")


def test_practice_resolution_runs_no_interpretation_call() -> None:
    """Large requests reach the planner directly; nothing interprets them first."""
    from features.practice_generation.planning import resolve_practice_request

    interpreter = _CountingInterpreter()
    request = resolve_practice_request(
        request_id="r",
        user_id="u",
        conversation_id="c",
        turn_id="t",
        query="Make 30 practice questions on Time and Work and Percentage",
        subject="math",
        topic="Broad Subject",
        difficulty="intermediate",
        language="english",
        exam_id=None,
        exam_stage=None,
        request_interpreter=None,
    )

    assert interpreter.calls == 0
    assert request.accepted_count == 30
    assert select_planning_mode(request)[0] == "INTELLIGENCE"


# --- explicit count extraction across scripts and spellings -------------------

@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Create 30 questions on percentage and ratio proportion", 30),
        ("Make 30 practice questions on Time and Work and Percentage", 30),
        ("मुझे औसत और प्रतिशत के 30 कठिन सवाल दो", 30),
        ("30 प्रश्न दीजिए प्रतिशत पर", 30),
        ("30 sawal banao average aur percentage pe mixed difficulty ke saath", 30),
        ("Create 30 questions on प्रतिशत and Time & Work", 30),
        ("creat 30 questin on percentge and ratio proporton", 30),
        ("give me 5 ques on ratio", 5),
        # transposed unit spelling; the digit is explicit and unambiguous
        ("creat 50 quesitons on percentge and ratio proporton", 50),
    ],
)
def test_explicit_count_is_read_whatever_unit_word_the_student_wrote(
    query: str, expected: int
) -> None:
    from features.practice_generation.planning import explicit_requested_count

    assert explicit_requested_count(query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # An earlier unrelated number must not reach past the real request.
        ("In 2 weeks. Create 10 questions on algebra", 10),
        # A later constraint number must not displace the stated total.
        ("Give me 30 questions: last 10 hard", 30),
        ("मुझे 30 सवाल दो, उनमें 10 कठिन हों", 30),
        # A unit word with no count attached is not a count.
        ("Solve question 5 for me", None),
        ("Explain this question", None),
    ],
)
def test_count_precedence_survives_multiple_numbers(
    query: str, expected: int | None
) -> None:
    from features.practice_generation.planning import explicit_requested_count

    assert explicit_requested_count(query) == expected


# --- grounded topic evidence outranks a broad classifier topic ---------------

def _blueprint_with(evidence, topic_ids, query):
    from features.practice_generation.planning import parse_blueprint

    request = _request_for(query, len(topic_ids))
    subject = request.subject
    slots = [
        dict(
            slot_id=f"slot-{i:03d}",
            subject_id=subject,
            topic_id=t,
            category_id=t,
            difficulty="intermediate",
            complexity="medium",
            exam_ids=[],
            question_type="mcq",
            target_skill=f"skill_{i}",
            variation_hint=f"variation {i}",
            pattern_family_id=None,
            generator_route_hint=f"{subject}.generator.intermediate",
            reasoning_target="numerical",
            trap_type=None,
            not_same_when=[],
            generation_group_hint=None,
        )
        for i, t in enumerate(topic_ids, start=1)
    ]
    payload = {"slots": slots, "requestedTopicEvidence": evidence}
    return parse_blueprint(json.dumps(payload), request)


def _request_for(query: str, count: int):
    subject = "reasoning" if "Blood" in query else "math"
    return PracticeGenerationRequest(
        request_id="r", user_id="u", conversation_id="c", turn_id="t",
        original_query=query, practice_type=PracticeType.QUICK_PRACTICE,
        requested_count=count, accepted_count=count, subject=subject,
        topic="Logical Reasoning" if subject == "reasoning" else "Quantitative Aptitude",
        difficulty=Difficulty.INTERMEDIATE, assessment_title="T",
    )


def test_explicit_topics_survive_a_broad_classifier_topic() -> None:
    """The student named three topics; the classifier answered one broad label."""
    query = "Create questions on Blood Relations, Direction Sense and Syllogism"
    blueprint = _blueprint_with(
        [
            {"sourceText": "Blood Relations", "topicId": "blood_relations"},
            {"sourceText": "Direction Sense", "topicId": "direction_sense"},
            {"sourceText": "Syllogism", "topicId": "syllogism"},
        ],
        ["blood_relations", "direction_sense", "syllogism"],
        query,
    )

    assert {s.topic_id for s in blueprint.slots} == {
        "blood_relations", "direction_sense", "syllogism"
    }


@pytest.mark.parametrize(
    ("source", "topic_id", "query"),
    [
        # normalization the planner supplies; code verifies only the source span
        ("percentge", "percentage", "creat 50 quesitons on percentge and avg"),
        ("avg", "average", "creat 50 quesitons on percentge and avg"),
        ("औसत", "average", "मुझे औसत के 50 सवाल दो"),
        ("प्रतिशत", "percentage", "मुझे प्रतिशत के 50 सवाल दो"),
        # compound concepts are matched whole, never split on "and"
        ("Profit and Loss", "profit_and_loss", "50 questions on Profit and Loss"),
        ("Time and Work", "time_and_work", "50 questions on Time and Work"),
        ("Statement and Conclusion", "statement_and_conclusion",
         "50 on Statement and Conclusion"),
    ],
)
def test_typo_and_multilingual_spans_ground_their_normalized_topic(
    source: str, topic_id: str, query: str
) -> None:
    blueprint = _blueprint_with(
        [{"sourceText": source, "topicId": topic_id}], [topic_id, topic_id], query
    )

    assert blueprint.slots[0].topic_id == topic_id


def test_a_topic_the_student_never_wrote_is_rejected() -> None:
    with pytest.raises(ValueError):
        _blueprint_with(
            [{"sourceText": "trigonometry", "topicId": "trigonometry"}],
            ["trigonometry", "trigonometry"],
            "Create 50 questions on Percentage",
        )


def test_duplicate_evidence_is_rejected() -> None:
    with pytest.raises(ValueError):
        _blueprint_with(
            [
                {"sourceText": "Percentage", "topicId": "percentage"},
                {"sourceText": "Percentage", "topicId": "percentage"},
            ],
            ["percentage", "percentage"],
            "Create 50 questions on Percentage",
        )


# --- coverage decides before grounding is consulted --------------------------
#
# A plan whose slots already cover every requested topic is complete on its own
# terms; the grounding result is never read for it. Evaluating grounding first
# turned an unusable evidence span into a fatal planner error for such plans.


def _reasoning_request(query: str, count: int):
    return PracticeGenerationRequest(
        request_id="r", user_id="u", conversation_id="c", turn_id="t",
        original_query=query, practice_type=PracticeType.QUICK_PRACTICE,
        requested_count=count, accepted_count=count, subject="reasoning",
        topic="Logical Reasoning", difficulty=Difficulty.INTERMEDIATE,
        assessment_title="T",
    )


def _math_request(query: str, count: int):
    return PracticeGenerationRequest(
        request_id="r", user_id="u", conversation_id="c", turn_id="t",
        original_query=query, practice_type=PracticeType.QUICK_PRACTICE,
        requested_count=count, accepted_count=count, subject="math",
        topic="Quantitative Aptitude", difficulty=Difficulty.INTERMEDIATE,
        assessment_title="T",
    )


def _parse_for(request, evidence, topic_ids):
    from features.practice_generation.planning import parse_blueprint

    slots = [
        dict(
            slot_id=f"slot-{i:03d}",
            subject_id=request.subject,
            topic_id=t,
            category_id=t,
            difficulty="intermediate",
            complexity="medium",
            exam_ids=[],
            question_type="mcq",
            target_skill=f"skill_{i}",
            variation_hint=f"variation {i}",
            pattern_family_id=None,
            generator_route_hint=f"{request.subject}.generator.intermediate",
            reasoning_target="numerical",
            trap_type=None,
            not_same_when=[],
            generation_group_hint=None,
        )
        for i, t in enumerate(topic_ids, start=1)
    ]
    payload = {"slots": slots, "requestedTopicEvidence": evidence}
    return parse_blueprint(json.dumps(payload), request)


def _grounding_spy(monkeypatch):
    """Wrap _grounded_topic_ids so a test can assert whether it was consulted."""
    from features.practice_generation import planning

    calls: list[int] = []
    original = planning._grounded_topic_ids

    def _spy(blueprint, request):
        calls.append(1)
        return original(blueprint, request)

    monkeypatch.setattr(planning, "_grounded_topic_ids", _spy)
    return calls


def test_complete_coverage_does_not_consult_grounding(monkeypatch) -> None:
    """The exact production failure: 6 reasoning slots, an unusable evidence span.

    Every requested topic is already planned, so grounding cannot change the
    outcome and must not be able to fail the plan.
    """
    calls = _grounding_spy(monkeypatch)
    request = _reasoning_request(
        "Create a 6 Reasoning mini mock for CAT Management — Pre.", 6
    )

    blueprint = _parse_for(
        request,
        # The span the planner returned twice in production; it never occurs in
        # the student's words, so _grounded_topic_ids would reject it.
        [{"sourceText": "Logical Reasoning", "topicId": "logical_reasoning"}],
        ["logical_reasoning"] * 6,
    )

    assert calls == []
    assert len(blueprint.slots) == 6
    assert {s.topic_id for s in blueprint.slots} == {"logical_reasoning"}


def test_incomplete_coverage_still_consults_grounding(monkeypatch) -> None:
    calls = _grounding_spy(monkeypatch)
    request = _math_request("Create questions on Percentage and Average", 2)

    blueprint = _parse_for(
        request,
        [
            {"sourceText": "Percentage", "topicId": "percentage"},
            {"sourceText": "Average", "topicId": "average"},
        ],
        ["percentage", "average"],
    )

    assert calls == [1]
    assert {s.topic_id for s in blueprint.slots} == {"percentage", "average"}


def test_incomplete_coverage_still_rejects_an_ungrounded_span() -> None:
    """Fix B must not weaken safety where the grounding result is actually read."""
    request = _math_request("Create questions on Percentage and Average", 2)

    with pytest.raises(ValueError, match="PLANNER_TOPIC_EVIDENCE_UNGROUNDED"):
        _parse_for(
            request,
            [
                {"sourceText": "Percentage", "topicId": "percentage"},
                {"sourceText": "Trigonometry", "topicId": "average"},
            ],
            ["percentage", "average"],
        )


def test_incomplete_coverage_still_rejects_duplicate_evidence() -> None:
    request = _math_request("Create questions on Percentage and Average", 2)

    with pytest.raises(ValueError, match="PLANNER_TOPIC_EVIDENCE_DUPLICATE"):
        _parse_for(
            request,
            [
                {"sourceText": "Percentage", "topicId": "percentage"},
                {"sourceText": "Percentage", "topicId": "percentage"},
            ],
            ["percentage", "average"],
        )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # The two invariants that previously reported as PLANNER_SCHEMA_INVALID.
        ("PLANNER_TOPIC_EVIDENCE_UNGROUNDED", "PLANNER_TOPIC_EVIDENCE_UNGROUNDED"),
        ("PLANNER_TOPIC_EVIDENCE_DUPLICATE", "PLANNER_TOPIC_EVIDENCE_DUPLICATE"),
        # Pre-existing mappings must keep their reason codes.
        ("PLANNER_TOPIC_COVERAGE_INVALID", "PLANNER_TOPIC_COVERAGE_INVALID"),
        ("slot count must equal accepted_count", "PLANNER_SLOT_COUNT_MISMATCH"),
        ("planner_family is required", "PLANNER_FAMILY_MISSING"),
        ("slots must be intentionally distinct", "PLANNER_SLOTS_NOT_DISTINCT"),
    ],
)
def test_planner_validation_reason_names_the_failed_invariant(
    message: str, expected: str
) -> None:
    from features.practice_generation.planning import _planner_validation_reason

    assert _planner_validation_reason(ValueError(message), raw="")[0] == expected


def test_repair_feedback_carries_the_named_invariant() -> None:
    """The single repair attempt must be told which invariant failed."""
    from features.practice_generation.planning import (
        _planner_repair_feedback,
        _planner_validation_diagnostic,
    )

    diagnostic = _planner_validation_diagnostic(
        ValueError("PLANNER_TOPIC_EVIDENCE_UNGROUNDED"),
        raw="",
        attempt=1,
        phase="plan",
        duration_ms=1,
    )

    assert diagnostic.reason_code == "PLANNER_TOPIC_EVIDENCE_UNGROUNDED"
    assert "reason=PLANNER_TOPIC_EVIDENCE_UNGROUNDED" in _planner_repair_feedback(
        diagnostic
    )
