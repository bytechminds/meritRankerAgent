"""A verifier outage must not be repaired by generating new questions.

An unavailable verifier is infrastructure failure, not a content-quality defect:
regenerating the question cannot fix it, so it must not consume the content
replacement waves reserved for genuinely rejected questions.
"""

from __future__ import annotations

import json

from features.practice_generation.config import PracticeGenerationConfig
from features.practice_generation.orchestration import (
    PracticeGenerationOrchestrator,
    _SlotGenerationContext,
)
from features.practice_generation.planning import (
    deterministic_blueprint,
    resolve_practice_request,
)
from features.practice_generation.schemas import VerificationResult


class _CountingGenerator:
    """Returns one structurally valid question per requested slot."""

    def __init__(self) -> None:
        self.call_count = 0
        self.waves: list[int] = []

    def generate_slots(self, *, request, bucket, group, slots, **kwargs):  # noqa: ANN001
        self.call_count += 1
        self.waves.append(kwargs.get("replacement_wave", 0))

        class _Batch:
            route_id = "math.generator.basic"
            model = "test-model"
            content = json.dumps({
                "questions": [
                    {
                        "schema_version": "2",
                        "generation_item_id": f"item-{slot.slot_id}",
                        "bucket_id": bucket.bucket_id,
                        "slot_id": slot.slot_id,
                        "question": f"Question for {slot.slot_id} about percentages?",
                        "question_type": "mcq",
                        "options": [
                            {"option_id": str(index), "value": value}
                            for index, value in enumerate(["10", "20", "30", "40"])
                        ],
                        "correct_option_id": "1",
                        "correct_answer": "20",
                        "answer_explanation": "Twenty is correct.",
                        "solution": "Twenty is correct.",
                        "subject": slot.subject_id,
                        "topic": slot.topic_id,
                        "difficulty": slot.difficulty.value,
                    }
                    for slot in slots
                ]
            })

        return _Batch()


class _Verifier:
    def __init__(self, outcome: str) -> None:
        self.outcome = outcome
        self.call_count = 0

    def verify_slot(self, *, request, bucket, slot, question):  # noqa: ANN001
        self.call_count += 1
        if self.outcome == "unavailable":
            raise RuntimeError("verifier provider unavailable")
        if self.outcome == "approved":
            return VerificationResult(
                generation_item_id=question.generation_item_id,
                slot_id=slot.slot_id,
                approved=True,
                reason_code="VERIFIER_APPROVED",
                valid_option_ids=[question.correct_option_id],
            )
        return VerificationResult(
            generation_item_id=question.generation_item_id,
            slot_id=slot.slot_id,
            approved=False,
            reason_code=self.outcome,
        )


def _config() -> PracticeGenerationConfig:
    return PracticeGenerationConfig(
        enabled=True, pattern_context_enabled=False, pattern_reuse_enabled=False,
        assessment_table="a", question_table="q", question_bank_table="qb",
        question_bank_category_index="ci", question_test_index="ti",
        aws_region="ap-south-1", verification_max_concurrency=1,
    )


def _run(outcome: str, *, slot_count: int = 1):
    request = resolve_practice_request(
        request_id="r1", user_id="u1", conversation_id="c1", turn_id="t1",
        query=f"Create {slot_count} questions on percentage", subject="math",
        topic="percentage", difficulty="basic", language="english",
        exam_id=None, exam_stage=None,
    )
    blueprint = deterministic_blueprint(request)
    generator = _CountingGenerator()
    verifier = _Verifier(outcome)

    orchestrator = object.__new__(PracticeGenerationOrchestrator)
    orchestrator._config = _config()
    orchestrator._generator = generator
    orchestrator._verifier = verifier
    orchestrator._progress_updates = None

    from features.practice_generation.generation import bucket_for_slot

    slots = tuple(blueprint.slots[:slot_count])
    context = _SlotGenerationContext(
        test_id="test-1", request=request, blueprint=blueprint,
        group=blueprint_group(blueprint, slots), bucket=bucket_for_slot(blueprint, slots[0]),
        slots=slots, excluded_texts=(), pattern_guidance_by_slot={},
        pattern_selections_by_slot={},
    )
    result = orchestrator._execute_slot_group(context=context)
    return generator, verifier, result


def blueprint_group(blueprint, slots):
    from features.practice_generation.generation import bucket_for_slot
    from features.practice_generation.schemas import GenerationGroup

    return GenerationGroup(
        group_id="slot-group-001",
        bucket_id=bucket_for_slot(blueprint, slots[0]).bucket_id,
        required_count=len(slots),
        slot_ids=[slot.slot_id for slot in slots],
    )


def test_verifier_outage_does_not_generate_replacement_content() -> None:
    generator, verifier, outcome = _run("unavailable")

    # One generation attempt only: regenerating cannot repair an outage.
    assert generator.call_count == 1, f"waves generated: {generator.waves}"
    assert generator.waves == [0]
    assert outcome.accepted == ()


def test_content_rejection_still_uses_the_replacement_waves() -> None:
    generator, _verifier, outcome = _run("INCORRECT_KEY")

    # Unchanged behaviour: a genuinely rejected question is regenerated.
    assert generator.call_count > 1, "content rejection must still replace"
    assert outcome.accepted == ()


# Approval end-to-end (including the binding/independent-answer predicate) is
# already covered by tests/practice_generation/test_practice_orchestration.py; this
# file is scoped to how a REJECTED result is routed.


# --- outer orchestration seam ------------------------------------------------

def test_verifier_outage_is_not_marked_provider_recoverable() -> None:
    """The outer commit turns a recoverable provider failure into a
    PROVIDER_REPLACEMENT group, which regenerates content. A verifier outage must
    therefore not be reported as provider-recoverable."""
    _generator, _verifier, outcome = _run("unavailable")

    assert outcome.provider_failure_stage == "VERIFIER"
    assert outcome.provider_failure_recoverable is False


def test_verifier_outage_schedules_no_provider_replacement_group() -> None:
    """End-to-end across the outer seam: no regeneration is scheduled, and the
    terminal reason names verifier exhaustion rather than generic content deficit."""
    import json

    from features.practice_generation.orchestration import PracticeGenerationOrchestrator

    generator, _verifier, outcome = _run("unavailable")
    assert generator.call_count == 1

    committed: dict[str, object] = {}

    class _Assessments:
        def get(self, _test_id):
            return {
                "testId": "test-1",
                "status": "GENERATING",
                "meta": json.dumps({
                    "generationGroups": {
                        "slot-group-001": {
                            "groupId": "slot-group-001",
                            "bucketId": outcome.context.bucket.bucket_id,
                            "state": "RUNNING",
                        }
                    }
                }),
            }

    class _Questions:
        def list_linked(self, _test_id, *, limit=100):
            return []

        def link_generated(self, **_kwargs):
            return True

    orchestrator = object.__new__(PracticeGenerationOrchestrator)
    orchestrator._config = _config()
    orchestrator._assessments = _Assessments()
    orchestrator._questions = _Questions()
    orchestrator._semantic_resolver = None

    def _capture(test_id, reason_code, **kwargs):
        committed["failed_reason"] = reason_code
        committed["meta"] = kwargs.get("meta_updates")

    orchestrator._mark_failed = _capture
    orchestrator._update_progress = lambda *a, **k: None

    orchestrator._commit_slot_outcome(outcome)

    groups = (committed.get("meta") or {}).get("generationGroups", {})
    stages = [group.get("attemptStage") for group in groups.values()]

    assert "PROVIDER_REPLACEMENT" not in stages, groups
    assert committed["failed_reason"] == "PRACTICE_VERIFIER_FALLBACK_EXHAUSTED", (
        committed["failed_reason"]
    )
