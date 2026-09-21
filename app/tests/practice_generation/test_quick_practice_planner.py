"""Deterministic Quick Practice planning.

Characterization first: these pin what the existing deterministic compiler
already produces, so the routing switch can be proven behaviour-preserving.
"""

from __future__ import annotations

import pytest

from features.practice_generation.planning import (
    BlueprintManager,
    _requested_topic_ids,
    deterministic_blueprint,
    resolve_practice_request,
)
from features.practice_generation.schemas import Difficulty, PracticeType, QuestionType


def request_for(
    query: str,
    *,
    subject: str = "math",
    topic: str | None = "time_and_work",
    difficulty: str = "basic",
    language: str = "english",
    exam_id: str | None = None,
):
    return resolve_practice_request(
        request_id="r1", user_id="u1", conversation_id="c1", turn_id="t1",
        query=query, subject=subject, topic=topic, difficulty=difficulty,
        language=language, exam_id=exam_id, exam_stage=None,
    )


class _ExplodingPlanner:
    """Any planner LLM call in deterministic mode is a defect."""

    def __init__(self) -> None:
        self.call_count = 0

    def plan(self, request, *, tier, repair_feedback=None):  # noqa: ANN001
        self.call_count += 1
        raise AssertionError("planner LLM must not be called for Quick Practice")


# --- A. single subject / single topic, exact totals ---------------------------

@pytest.mark.parametrize("count", [1, 2, 3, 5, 6, 10, 20, 50, 100])
def test_slot_count_always_equals_effective_requested_count(count: int) -> None:
    blueprint = deterministic_blueprint(request_for(f"Create {count} questions on Time and Work"))

    assert len(blueprint.slots) == min(count, 50)
    assert blueprint.accepted_count == min(count, 50)
    assert sum(bucket.required_count for bucket in blueprint.buckets) == min(count, 50)


@pytest.mark.parametrize("count", [1, 2])
def test_quick_practice_never_calls_the_planner_llm(count: int) -> None:
    planner = _ExplodingPlanner()
    manager = BlueprintManager(planner)

    result = manager.build(request_for(f"Create {count} questions on Time and Work"))

    assert planner.call_count == 0
    assert result.planner_calls == 0
    assert result.deterministic_fallback is False
    assert len(result.blueprint.slots) == count


# --- B/C. difficulty ----------------------------------------------------------

@pytest.mark.parametrize("difficulty", ["basic", "intermediate", "advanced"])
@pytest.mark.parametrize("count", [1, 3, 10])
def test_explicit_difficulty_applies_to_every_slot(difficulty: str, count: int) -> None:
    blueprint = deterministic_blueprint(
        request_for(f"Create {count} {difficulty} questions on Time and Work",
                    difficulty=difficulty)
    )

    assert {slot.difficulty.value for slot in blueprint.slots} == {difficulty}


@pytest.mark.parametrize("count", [1, 2, 3, 5, 10, 20, 100])
def test_mixed_difficulty_request_distributes_and_still_totals_exactly(count: int) -> None:
    blueprint = deterministic_blueprint(
        request_for(f"Create {count} mixed difficulty questions on Time and Work")
    )

    assert len(blueprint.slots) == min(count, 50)
    bands = {slot.difficulty for slot in blueprint.slots}
    assert bands <= {Difficulty.BASIC, Difficulty.INTERMEDIATE, Difficulty.ADVANCED}
    if count >= 3:
        assert len(bands) == 3, "a mixed request should span every band once it can"


# --- D. multiple topics, same subject ----------------------------------------

@pytest.mark.parametrize(
    ("topic", "expected_topics"),
    [
        ("percentage, ratio", 2),
        ("percentage, ratio, average", 3),
        ("a, b, c, d, e", 5),
    ],
)
@pytest.mark.parametrize("count", [5, 7, 10, 20, 100])
def test_multiple_topics_allocate_without_losing_the_total(
    topic: str, expected_topics: int, count: int
) -> None:
    blueprint = deterministic_blueprint(
        request_for(f"Create {count} questions", topic=topic)
    )

    assert len(blueprint.slots) == min(count, 50)
    distinct = {slot.topic_id for slot in blueprint.slots}
    assert len(distinct) == min(count, 50, expected_topics)
    # Equal apportionment: no topic may exceed another by more than one slot.
    per_topic = [sum(1 for s in blueprint.slots if s.topic_id == t) for t in distinct]
    assert max(per_topic) - min(per_topic) <= 1


def test_a_phrase_topic_containing_and_is_never_split() -> None:
    blueprint = deterministic_blueprint(
        request_for("Create 4 questions", topic="profit and loss and percentage")
    )

    # "and" must not be a separator: this is one topic phrase, not three.
    assert {slot.topic_id for slot in blueprint.slots} == {"profit_and_loss_and_percentage"}


def test_comma_separated_topics_are_the_supported_multi_topic_form() -> None:
    blueprint = deterministic_blueprint(
        request_for("Create 4 questions", topic="profit and loss, percentage")
    )

    assert {slot.topic_id for slot in blueprint.slots} == {"profit_and_loss", "percentage"}


# --- E/F. subject association ------------------------------------------------

def test_every_slot_carries_the_single_requested_subject() -> None:
    blueprint = deterministic_blueprint(
        request_for("Create 10 questions", subject="reasoning", topic="seating, series")
    )

    # One subject in the contract means no cross-subject pairing is possible.
    assert {slot.subject_id for slot in blueprint.slots} == {"reasoning"}


# --- G. topic absent ----------------------------------------------------------

def test_absent_topic_falls_back_to_the_subject_without_inventing_taxonomy() -> None:
    blueprint = deterministic_blueprint(request_for("Create 6 math questions", topic=None))

    topics = {slot.topic_id for slot in blueprint.slots}
    assert topics == {"math"}, "must reuse the subject, never invent a curriculum topic"


# --- I/J. exam, language, question type --------------------------------------

@pytest.mark.parametrize("exam_id", ["CAT", "SSC_CGL", None])
def test_requested_exam_is_preserved_on_every_slot(exam_id: str | None) -> None:
    blueprint = deterministic_blueprint(
        request_for("Create 7 questions on Time and Work", exam_id=exam_id)
    )

    expected = [exam_id] if exam_id else []
    for slot in blueprint.slots:
        assert slot.exam_ids == expected


def test_question_type_and_slot_ids_are_stable() -> None:
    blueprint = deterministic_blueprint(request_for("Create 3 questions on Time and Work"))

    assert {slot.question_type for slot in blueprint.slots} == {QuestionType.MCQ}
    assert [slot.slot_id for slot in blueprint.slots] == ["slot-001", "slot-002", "slot-003"]


# --- determinism --------------------------------------------------------------

@pytest.mark.parametrize("count", [7, 20, 100])
def test_identical_input_produces_an_identical_blueprint(count: int) -> None:
    query = f"Create {count} questions"
    first = deterministic_blueprint(request_for(query, topic="percentage, ratio, average"))
    second = deterministic_blueprint(request_for(query, topic="percentage, ratio, average"))

    assert first.model_dump() == second.model_dump()


def test_no_zero_count_buckets_are_emitted() -> None:
    blueprint = deterministic_blueprint(
        request_for("Create 6 questions", topic="a, b, c, d, e")
    )

    assert all(bucket.required_count > 0 for bucket in blueprint.buckets)
    assert sum(b.required_count for b in blueprint.buckets) == 6


# --- practice type scope ------------------------------------------------------

def test_quick_practice_is_the_default_practice_type() -> None:
    assert request_for("Create 5 questions on Time and Work").practice_type is (
        PracticeType.QUICK_PRACTICE
    )


# --- feasibility-aware topic coverage ----------------------------------------

@pytest.mark.parametrize(
    ("count", "topic", "distinct_topics"),
    [
        (1, "percentage", 1),
        (1, "percentage, ratio", 2),
        (1, "percentage, ratio, average, speed, time", 5),
        (2, "percentage, ratio", 2),
        (2, "percentage, ratio, average", 3),
        (2, "percentage, ratio, average, speed, time", 5),
        (3, "percentage, ratio, average", 3),
        (3, "percentage, ratio", 2),
    ],
)
def test_topic_coverage_is_feasibility_aware(
    count: int, topic: str, distinct_topics: int
) -> None:
    blueprint = deterministic_blueprint(request_for(f"Create {count} questions", topic=topic))
    planned = {slot.topic_id for slot in blueprint.slots}
    requested = set(_requested_topic_ids(request_for(f"Create {count} questions", topic=topic)))

    assert len(blueprint.slots) == count
    # Never invent a topic, whichever side of the feasibility boundary we are on.
    assert planned <= requested
    if count >= distinct_topics:
        assert requested <= planned, "every requested topic must appear when it can"
    else:
        assert len(planned) == count, "one distinct topic per slot when N < T"


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_subset_selection_follows_canonical_request_order(count: int) -> None:
    blueprint = deterministic_blueprint(
        request_for(f"Create {count} questions", topic="alpha, beta, gamma, delta, epsilon")
    )
    planned = [slot.topic_id for slot in blueprint.slots]

    assert planned == ["alpha", "beta", "gamma", "delta"][:count]


@pytest.mark.parametrize("count", [5, 20, 100])
def test_balanced_allocation_when_every_topic_fits(count: int) -> None:
    blueprint = deterministic_blueprint(
        request_for(f"Create {count} questions", topic="percentage, ratio, average")
    )
    per_topic = [
        sum(1 for s in blueprint.slots if s.topic_id == t)
        for t in {s.topic_id for s in blueprint.slots}
    ]

    assert sum(per_topic) == min(count, 50)
    assert len(per_topic) == 3
    assert max(per_topic) - min(per_topic) <= 1


def test_twenty_across_three_topics_is_seven_seven_six() -> None:
    blueprint = deterministic_blueprint(
        request_for("Create 20 questions", topic="percentage, ratio, average")
    )
    counts = [sum(1 for s in blueprint.slots if s.topic_id == t)
              for t in ("percentage", "ratio", "average")]

    assert counts == [7, 7, 6]
    assert sum(counts) == 20


# --- deterministic Quick Practice routing ------------------------------------

# Above INTELLIGENCE_PLANNING_COUNT_THRESHOLD a Quick Practice request routes to the
# intelligence planner instead; that side of the boundary is covered in
# tests/practice_generation/test_planning_route.py.
@pytest.mark.parametrize("count", [1, 2, 3, 5, 6, 10, 20])
def test_quick_practice_routes_deterministically_up_to_the_threshold(count: int) -> None:
    planner = _ExplodingPlanner()

    result = BlueprintManager(planner).build(
        request_for(f"Create {count} questions on Time and Work")
    )

    assert planner.call_count == 0, "planner LLM must not be called"
    assert result.planner_calls == 0
    assert result.repaired is False
    assert result.deterministic_fallback is False
    assert len(result.blueprint.slots) == count


@pytest.mark.parametrize("count", [1, 2, 5, 20])
def test_quick_practice_routing_survives_fewer_questions_than_topics(count: int) -> None:
    planner = _ExplodingPlanner()

    result = BlueprintManager(planner).build(
        request_for(
            f"Create {count} questions on percentage, ratio and average",
            topic="percentage, ratio, average",
        )
    )

    assert planner.call_count == 0
    assert len(result.blueprint.slots) == count


# --- scope containment and rollback ------------------------------------------

class _RecordingPlanner:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.call_count = 0

    def plan(self, request, *, tier, repair_feedback=None):  # noqa: ANN001
        self.call_count += 1
        return self.raw


def _valid_llm_payload(count: int, *, subject: str = "math", topic: str = "time_and_work") -> str:
    import json

    return json.dumps({
        "slots": [
            {
                "slot_id": f"slot-{i:03d}", "subject_id": subject, "topic_id": topic,
                "category_id": topic, "difficulty": "basic", "complexity": "low",
                "exam_ids": [], "question_type": "mcq", "target_skill": f"skill_{i}",
                "variation_hint": f"vary_{i}", "pattern_family_id": None,
                "generator_route_hint": f"{subject}.generator.basic",
                "reasoning_target": None, "trap_type": None, "not_same_when": [],
                "generation_group_hint": 1,
            }
            for i in range(1, count + 1)
        ]
    })


def test_non_quick_practice_still_uses_the_llm_planner() -> None:
    planner = _RecordingPlanner(_valid_llm_payload(5))
    request = request_for("Create a full mock test with 5 questions on Time and Work")
    assert request.practice_type is PracticeType.FULL_MOCK

    result = BlueprintManager(planner).build(request)

    # Mock/adaptive planning is explicitly out of scope and must be untouched.
    assert planner.call_count == 1
    assert result.planner_calls == 1


def test_similar_question_requests_still_use_the_llm_planner() -> None:
    planner = _RecordingPlanner(_valid_llm_payload(5))
    request = request_for("Create 5 similar questions on Time and Work")
    assert request.practice_type is PracticeType.SIMILAR_QUESTION

    BlueprintManager(planner).build(request)

    assert planner.call_count == 1


def test_rollback_to_the_llm_planner_is_a_config_flip(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_PLANNER_MODE", "llm")
    planner = _RecordingPlanner(_valid_llm_payload(5))

    result = BlueprintManager(planner).build(
        request_for("Create 5 questions on Time and Work")
    )

    assert planner.call_count == 1
    assert result.planner_calls == 1
    assert result.deterministic_fallback is False


def test_small_counts_stay_deterministic_even_when_rolled_back(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_PLANNER_MODE", "llm")
    planner = _ExplodingPlanner()

    result = BlueprintManager(planner).build(
        request_for("Create 2 questions on Time and Work")
    )

    # The pre-existing count<=2 short-circuit is independent of the new mode.
    assert planner.call_count == 0
    assert len(result.blueprint.slots) == 2


def test_generation_group_hint_cap_is_unchanged() -> None:
    for count in (1, 3, 5, 20, 100):
        blueprint = deterministic_blueprint(
            request_for(f"Create {count} questions on Time and Work")
        )
        assert all(slot.generation_group_hint <= 5 for slot in blueprint.slots)


# --- topic list conjunctions (PATCH A) ---------------------------------------

def _topics(topic: str):
    return _requested_topic_ids(request_for("Create 12 questions", topic=topic))


@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        # Plain list: already correct today.
        ("percentage, ratio, average", ("percentage", "ratio", "average")),
        # Oxford-comma list: the observed live corruption was "and_average".
        ("percentage, ratio, and average", ("percentage", "ratio", "average")),
        ("profit and loss, ratio, and average", ("profit_and_loss", "ratio", "average")),
        ("percentage, ratio and average", ("percentage", "ratio_and_average")),
        # Compound academic names must survive untouched.
        ("profit and loss", ("profit_and_loss",)),
        ("permutations and combinations", ("permutations_and_combinations",)),
        ("pipes and cisterns, time and work",
         ("pipes_and_cisterns", "time_and_work")),
        ("profit and loss, percentage", ("profit_and_loss", "percentage")),
        ("ratio and proportion, average", ("ratio_and_proportion", "average")),
    ],
)
def test_topic_list_conjunctions(topic: str, expected: tuple[str, ...]) -> None:
    assert _topics(topic) == expected


def test_leading_conjunction_strip_never_empties_a_topic() -> None:
    # A bare conjunction is not a topic; it must not become an empty identifier.
    assert _topics("percentage, and") == ("percentage", "and")


# --- generation grouping reconciliation (PART B: no production change) --------

def _groups_for(count: int, topic: str, reused: list[str]):
    from features.practice_generation.generation import build_slot_generation_groups

    blueprint = deterministic_blueprint(request_for(f"Create {count} questions", topic=topic))
    deficits = {s.slot_id for s in blueprint.slots} - set(reused)
    groups = build_slot_generation_groups(blueprint, deficits, group_size=3, group_max=5)
    return blueprint, deficits, groups


@pytest.mark.parametrize(
    "reused",
    [
        [],
        ["slot-001"],
        ["slot-001", "slot-002", "slot-003"],
        ["slot-001", "slot-004", "slot-002"],
        ["slot-001", "slot-004", "slot-007"],
    ],
)
def test_generation_groups_reconcile_with_the_deficit(reused: list[str]) -> None:
    blueprint, deficits, groups = _groups_for(12, "percentage, ratio, average", reused)

    assert sum(g.required_count for g in groups) == len(deficits)
    assert all(1 <= g.required_count <= 5 for g in groups)
    grouped_slots = [slot_id for g in groups for slot_id in g.slot_ids]
    assert sorted(grouped_slots) == sorted(deficits), "every deficit in exactly one group"
    assert len(grouped_slots) == len(set(grouped_slots))
    assert not (set(grouped_slots) & set(reused)), "a reused slot never enters generation"


def test_a_group_never_mixes_incompatible_topics() -> None:
    blueprint, _deficits, groups = _groups_for(12, "percentage, ratio, average", ["slot-001"])
    topic_of = {s.slot_id: s.topic_id for s in blueprint.slots}

    for group in groups:
        assert len({topic_of[slot_id] for slot_id in group.slot_ids}) == 1


def test_grouping_is_stable_for_identical_input() -> None:
    first = _groups_for(12, "percentage, ratio, average", ["slot-001"])[2]
    second = _groups_for(12, "percentage, ratio, average", ["slot-001"])[2]

    assert [g.model_dump() for g in first] == [g.model_dump() for g in second]


@pytest.mark.parametrize("count", [1, 5, 12, 20, 100])
def test_group_cap_holds_at_every_supported_count(count: int) -> None:
    _blueprint, deficits, groups = _groups_for(count, "percentage, ratio, average", [])

    assert sum(g.required_count for g in groups) == len(deficits) == min(count, 50)
    assert all(g.required_count <= 5 for g in groups)
