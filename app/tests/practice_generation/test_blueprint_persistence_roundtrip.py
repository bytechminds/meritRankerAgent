"""PracticeBlueprint.requested_topic_evidence must survive a persist/reload round trip.

Proven defect: orchestration.py's plan_and_fill persists the blueprint via
``blueprint.model_dump(mode="json")`` (field names, no ``by_alias``), but
``requested_topic_evidence`` is declared with ``alias="requestedTopicEvidence"``, and
``PracticeBlueprint.model_config`` did not set ``populate_by_name``. Every later reload
(generate_wave/finalize) calls ``PracticeBlueprint.model_validate(...)`` on that same
JSON, which only recognized the alias — never the field name a plain ``model_dump``
actually wrote — so the field silently reset to its empty default_factory on every
reload. This broke topic-evidence grounding (apply_system_bucket_policy) for any
request whose topic identity depends on grounded evidence rather than an explicit
`topic`/`topics` field (e.g. SIMILAR_QUESTION requests), raising
PLANNER_SLOT_TOPIC_COVERAGE_INVALID at generation time even though planning succeeded.
"""

from __future__ import annotations

from features.practice_generation.schemas import (
    Complexity,
    Difficulty,
    PlannerFamily,
    PlannerSlot,
    PracticeBlueprint,
    PracticeType,
    QuestionType,
    TopicEvidence,
)


def _blueprint() -> PracticeBlueprint:
    return PracticeBlueprint(
        schema_version="2",
        practice_type=PracticeType.QUICK_PRACTICE,
        accepted_count=1,
        planner_family=PlannerFamily.QUANT_REASONING,
        slots=[
            PlannerSlot(
                slot_id="slot-001",
                subject_id="math",
                topic_id="simple_interest",
                category_id="math",
                difficulty=Difficulty.INTERMEDIATE,
                complexity=Complexity.MEDIUM,
                question_type=QuestionType.MCQ,
                target_skill="apply_formula",
                variation_hint="vary_principal",
                generator_route_hint="math.generator.intermediate",
                not_same_when=[],
            )
        ],
        requested_topic_evidence=[
            TopicEvidence(source_text="simple interest", topic_id="simple_interest")
        ],
    )


def test_model_dump_json_uses_the_field_name_not_the_alias() -> None:
    """Characterizes the actual persistence call (orchestration.py:541): plain
    model_dump(mode="json") with no by_alias, so the dumped key is the field name."""
    dumped = _blueprint().model_dump(mode="json")
    assert "requested_topic_evidence" in dumped
    assert "requestedTopicEvidence" not in dumped


def test_requested_topic_evidence_survives_persist_and_reload() -> None:
    blueprint = _blueprint()
    dumped = blueprint.model_dump(mode="json")
    reloaded = PracticeBlueprint.model_validate(dumped)
    assert reloaded.requested_topic_evidence == blueprint.requested_topic_evidence
    assert [
        (item.source_text, item.topic_id) for item in reloaded.requested_topic_evidence
    ] == [("simple interest", "simple_interest")]


def test_reload_also_accepts_the_alias_form() -> None:
    """populate_by_name=True must not break validating the alias-keyed shape either
    (e.g. a request body built with by_alias=True, or hand-authored JSON)."""
    dumped = _blueprint().model_dump(mode="json", by_alias=True)
    assert "requestedTopicEvidence" in dumped
    reloaded = PracticeBlueprint.model_validate(dumped)
    assert len(reloaded.requested_topic_evidence) == 1
