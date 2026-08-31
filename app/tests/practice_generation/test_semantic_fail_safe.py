"""C1: an intelligence-planner failure must never broaden the student's request."""

from __future__ import annotations

import pytest

from features.practice_generation.planning import (
    BlueprintManager,
    BlueprintPlanningError,
    _trusted_topic_constraints,
)
from features.practice_generation.schemas import (
    Difficulty,
    PracticeGenerationRequest,
    PracticeType,
)

UNSAFE = "PRACTICE_PLANNER_SEMANTIC_FALLBACK_UNSAFE"


def _request(
    query: str,
    *,
    count: int = 50,
    subject: str = "math",
    topic: str | None = None,
    topics: list[str] | None = None,
) -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="r", user_id="u", conversation_id="c", turn_id="t",
        original_query=query, practice_type=PracticeType.QUICK_PRACTICE,
        requested_count=count, accepted_count=count, subject=subject,
        topic=topic, topics=topics, difficulty=Difficulty.INTERMEDIATE,
        assessment_title="T",
    )


class _SchemaFailurePlanner:
    def plan(self, *_a, **_k):
        return '{"slots": []}'


class _JsonFailurePlanner:
    def plan(self, *_a, **_k):
        return "not json at all"


class _MalformedPlanner:
    def plan(self, *_a, **_k):
        return '{"slots": [{"slot_id": "nope"}]}'


FAILING_PLANNERS = [
    pytest.param(_SchemaFailurePlanner, id="schema-rejection"),
    pytest.param(_JsonFailurePlanner, id="malformed-json"),
    pytest.param(_MalformedPlanner, id="semantic-rejection"),
]

EXPLICIT_REQUESTS = [
    pytest.param(
        "50 questions: Number System, Geometry, Probability, Trigonometry, "
        "Average, Ratio and Proportion",
        "math", id="english-multi-topic",
    ),
    pytest.param(
        "50 questions on Blood Relations, Direction Sense and Syllogism",
        "reasoning", id="reasoning-explicit",
    ),
    pytest.param(
        "50 questions on Time and Work, Profit and Loss, Ratio and Proportion "
        "and Statement and Conclusion",
        "math", id="compound-topics",
    ),
    pytest.param("मुझे औसत और प्रतिशत के 50 सवाल दो", "math", id="hindi"),
    pytest.param(
        "50 sawal banao average aur percentage pe", "math", id="hinglish"
    ),
    pytest.param(
        "creat 50 quesitons on percentge and ratio proporton", "math", id="noisy"
    ),
]


@pytest.mark.parametrize("planner_cls", FAILING_PLANNERS)
@pytest.mark.parametrize(("query", "subject"), EXPLICIT_REQUESTS)
def test_planner_failure_never_broadens_an_explicit_request(
    planner_cls: type, query: str, subject: str
) -> None:
    """Whatever the planner failure mode, the request is never widened."""
    broad_label = "Quantitative Aptitude" if subject == "math" else "Logical Reasoning"

    with pytest.raises(BlueprintPlanningError) as excinfo:
        BlueprintManager(planner_cls(), repair_limit=1).build(
            _request(query, subject=subject, topic=broad_label)
        )

    assert excinfo.value.reason_code == UNSAFE


@pytest.mark.parametrize("planner_cls", FAILING_PLANNERS)
def test_a_broad_request_also_fails_closed(planner_cls: type) -> None:
    """Documented trade-off: a genuinely broad request is indistinguishable.

    Nothing in the request records whether the student named topics that the
    planner then lost, so availability is sacrificed rather than correctness.
    """
    with pytest.raises(BlueprintPlanningError) as excinfo:
        BlueprintManager(planner_cls(), repair_limit=1).build(
            _request("Create 50 quantitative aptitude questions",
                     topic="Quantitative Aptitude")
        )

    assert excinfo.value.reason_code == UNSAFE


def test_trusted_structured_topics_still_allow_a_deterministic_fallback() -> None:
    """Explicit constraints that did not come from the planner survive its failure."""
    request = _request(
        "50 questions on Time and Work and Profit and Loss",
        topics=["Time and Work", "Profit and Loss"],
        topic="Quantitative Aptitude",
    )

    result = BlueprintManager(_SchemaFailurePlanner(), repair_limit=1).build(request)

    assert result.deterministic_fallback is True
    assert {slot.topic_id for slot in result.blueprint.slots} == {
        "time_and_work", "profit_and_loss"
    }
    assert len(result.blueprint.slots) == 50


def test_compound_topics_are_never_split_on_and() -> None:
    assert _trusted_topic_constraints(
        _request("x", topics=["Time and Work", "Statement and Conclusion"])
    ) == ("Time and Work", "Statement and Conclusion")


def test_the_classifier_label_is_not_a_trusted_constraint() -> None:
    """The broad label is exactly what must not be treated as the composition."""
    assert _trusted_topic_constraints(
        _request("x", topic="Quantitative Aptitude")
    ) == ()


@pytest.mark.parametrize("count", [10, 20])
def test_the_deterministic_path_is_untouched(count: int) -> None:
    """At or below the threshold the planner is never called, so nothing changes."""
    result = BlueprintManager(_SchemaFailurePlanner()).build(
        _request("Create questions on percentage", count=count, topic="Percentage")
    )

    assert result.planner_calls == 0
    assert len(result.blueprint.slots) == count


# --- request provenance -------------------------------------------------------

def test_the_original_query_is_persisted_for_audit() -> None:
    """Without it, delivered questions cannot be graded against the real request.

    Guards the projection itself: the assessment record is the only durable place the
    student's wording survives, and its absence is what made a past run unauditable.
    """
    import inspect

    from features.practice_generation import repositories

    source = inspect.getsource(repositories)

    assert '"originalQuery": request.original_query,' in source


def test_the_original_query_is_not_used_as_a_reuse_key() -> None:
    """Provenance only: retrieval must keep matching on structured slot constraints."""
    import inspect

    from features.practice_generation.repositories import QuestionRepository

    for name in ("get_reuse_candidates", "list_recent_seen_question_bank_ids"):
        source = inspect.getsource(getattr(QuestionRepository, name))
        assert "original_query" not in source
        assert "originalQuery" not in source


# --- C2: explicit topics survive the <=20 deterministic path -------------------

C2_QUERY = (
    "Create a 20-question  direction, decoding encoding, "
    "alphabeta revwerse positioning  Quick Practice."
)


def _routed(query: str, count: int, **kw) -> tuple[str, str]:
    from features.practice_generation.planning import select_planning_mode

    return select_planning_mode(_request(query, count=count, **kw))


def test_the_frozen_c2_request_escalates_instead_of_collapsing() -> None:
    """Three named topics vs the broad label 'Logical Reasoning Practice'."""
    assert _routed(
        C2_QUERY, 20, subject="reasoning", topic="Logical Reasoning Practice"
    ) == ("INTELLIGENCE", "CLASSIFIER_TOPIC_UNGROUNDED")


def test_an_ordinary_grounded_request_keeps_the_cheap_path() -> None:
    mode, reason = _routed(
        "Create 20 questions on Percentage", 20, topic="Percentage"
    )
    assert (mode, reason) == ("DETERMINISTIC", "EXISTING_DETERMINISTIC_RULE")


def test_structured_topics_keep_the_cheap_path() -> None:
    mode, reason = _routed(
        C2_QUERY, 20, subject="reasoning", topic="Logical Reasoning Practice",
        topics=["Direction Sense", "Coding-Decoding"],
    )
    assert (mode, reason) == ("DETERMINISTIC", "TRUSTED_TOPIC_CONSTRAINTS")


@pytest.mark.parametrize("count", [1, 2])
def test_forced_deterministic_counts_are_never_escalated(count: int) -> None:
    """Precedence guard: the ungrounded rule must not capture count <= 2."""
    mode, reason = _routed(
        C2_QUERY, count, subject="reasoning", topic="Logical Reasoning Practice"
    )
    assert (mode, reason) == ("DETERMINISTIC", "EXISTING_DETERMINISTIC_RULE")


@pytest.mark.parametrize(
    ("count", "expected"), [(20, "DETERMINISTIC"), (21, "INTELLIGENCE")]
)
def test_the_count_boundary_is_unchanged(count: int, expected: str) -> None:
    assert _routed(
        f"{count} questions on Percentage", count, topic="Percentage"
    )[0] == expected


def test_an_escalated_request_fails_closed_when_the_planner_cannot_ground() -> None:
    """C1 must own the escalated path: never a broad blueprint, never READY."""
    request = _request(
        C2_QUERY, count=20, subject="reasoning", topic="Logical Reasoning Practice"
    )

    with pytest.raises(BlueprintPlanningError) as excinfo:
        BlueprintManager(_SchemaFailurePlanner(), repair_limit=1).build(request)

    assert excinfo.value.reason_code == UNSAFE
