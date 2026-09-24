"""The content replacement waves must be bounded, whatever the generator returns.

Wave 0 is the original attempt, wave 1 the repair, wave 2 the replacement. A
content rejection cannot be repaired, so it skips wave 1 and goes straight to
wave 2 — but the skip must still move forward. Pinning the counter to the
replacement wave meant a rejection *at* wave 2 left the state unchanged, and the
group regenerated without bound. These tests pin the wave sequence itself, not
the verification semantics, which are unchanged.
"""

from __future__ import annotations

import json

from features.practice_generation.config import PracticeGenerationConfig
from features.practice_generation.matching import normalize_question_identity
from features.practice_generation.orchestration import (
    PracticeGenerationOrchestrator,
    _SlotGenerationContext,
)
from features.practice_generation.planning import (
    deterministic_blueprint,
    resolve_practice_request,
)
from features.practice_generation.schemas import (
    GenerationGroup,
    VerificationDecision,
    VerificationResult,
)
from services.llm.orchestration.errors import ProviderExecutionError

# Far beyond the intended three waves: reaching it means the loop is unbounded.
_ATTEMPT_PROBE_CAP = 12


class _LoopProbe(RuntimeError):
    """Raised by the generator double if the wave loop ever exceeds its bound."""


class _DistinctGenerator:
    """Returns a genuinely new question every attempt.

    A generator that repeats itself is filtered out by the existing near-duplicate
    exclusion, which used to hide the unbounded loop; a real generator does not
    repeat itself, so these tests deliberately do not either.
    """

    def __init__(self, *, raise_on_wave: dict[int, BaseException] | None = None) -> None:
        self._raise_on_wave = raise_on_wave or {}
        self.waves: list[int] = []
        self.slot_ids_by_wave: list[tuple[int, tuple[str, ...]]] = []

    def generate_slots(self, *, request, bucket, group, slots, **kwargs):  # noqa: ANN001
        del request, group
        wave = int(kwargs.get("replacement_wave", 0))
        self.waves.append(wave)
        self.slot_ids_by_wave.append((wave, tuple(slot.slot_id for slot in slots)))
        if len(self.waves) > _ATTEMPT_PROBE_CAP:
            raise _LoopProbe(f"unbounded generation: waves={self.waves}")
        if wave in self._raise_on_wave:
            raise self._raise_on_wave[wave]
        attempt = len(self.waves)

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
                        "question": (
                            f"Attempt {attempt}: a distinct percentage question "
                            f"for {slot.slot_id}?"
                        ),
                        "question_type": "mcq",
                        "options": [
                            {"option_id": str(index), "value": f"{attempt}{index}"}
                            for index in range(4)
                        ],
                        "correct_option_id": "1",
                        "correct_answer": f"{attempt}1",
                        "answer_explanation": "Explanation.",
                        "solution": "Solution.",
                        "subject": slot.subject_id,
                        "topic": slot.topic_id,
                        "difficulty": slot.difficulty.value,
                    }
                    for slot in slots
                ]
            })

        return _Batch()


class _Verifier:
    """Rejects the first N verifications of each slot, then approves."""

    def __init__(self, *, reject_first: int = 0) -> None:
        self._reject_first = reject_first
        self.calls = 0

    def verify_slot(self, *, request, bucket, slot, question):  # noqa: ANN001
        del request, bucket
        self.calls += 1
        if self.calls <= self._reject_first:
            return VerificationResult(
                schema_version="2",
                generation_item_id=question.generation_item_id,
                slot_id=slot.slot_id,
                decision=VerificationDecision.REGENERATE,
                valid_option_ids=[question.correct_option_id],
                reason_codes=["INCORRECT_KEY"],
            )
        return VerificationResult(
            schema_version="2",
            generation_item_id=question.generation_item_id,
            slot_id=slot.slot_id,
            decision=VerificationDecision.ACCEPT,
            valid_option_ids=[question.correct_option_id],
            answer_explanation="The independently selected option is correct.",
            reason_codes=["MATCH"],
        )


def _config() -> PracticeGenerationConfig:
    return PracticeGenerationConfig(
        enabled=True, pattern_context_enabled=False, pattern_reuse_enabled=False,
        assessment_table="a", question_table="q", question_bank_table="qb",
        question_bank_category_index="ci", question_test_index="ti",
        aws_region="ap-south-1", verification_max_concurrency=1,
    )


def _run(generator, verifier, *, slot_count: int = 1):
    from features.practice_generation.generation import bucket_for_slot

    request = resolve_practice_request(
        request_id="r1", user_id="u1", conversation_id="c1", turn_id="t1",
        query=f"Create {slot_count} questions on percentage", subject="math",
        topic="percentage", difficulty="basic", language="english",
        exam_id=None, exam_stage=None,
    )
    blueprint = deterministic_blueprint(request)
    slots = tuple(blueprint.slots[:slot_count])
    orchestrator = object.__new__(PracticeGenerationOrchestrator)
    orchestrator._config = _config()
    orchestrator._generator = generator
    orchestrator._verifier = verifier
    orchestrator._progress_updates = None
    orchestrator._student_credits = None
    orchestrator._update_progress = lambda *_args, **_kwargs: None
    context = _SlotGenerationContext(
        test_id="test-1", request=request, blueprint=blueprint,
        group=GenerationGroup(
            group_id="slot-group-001",
            bucket_id=bucket_for_slot(blueprint, slots[0]).bucket_id,
            required_count=len(slots),
            slot_ids=[slot.slot_id for slot in slots],
        ),
        bucket=bucket_for_slot(blueprint, slots[0]),
        slots=slots, excluded_texts=(), pattern_guidance_by_slot={},
        pattern_selections_by_slot={},
    )
    return orchestrator._execute_slot_group(context=context)


# --- 1-3. the sequence below the limit is unchanged --------------------------


def test_an_approved_first_attempt_uses_no_replacement_wave() -> None:
    generator, verifier = _DistinctGenerator(), _Verifier()

    outcome = _run(generator, verifier)

    assert generator.waves == [0]
    assert verifier.calls == 1
    assert len(outcome.accepted) == 1
    assert outcome.unresolved_slot_ids == ()
    assert outcome.replacement_wave_count == 0


def test_a_content_rejection_skips_the_repair_wave_and_replaces_once() -> None:
    """Unchanged behaviour: a rejection goes straight from wave 0 to wave 2."""
    generator, verifier = _DistinctGenerator(), _Verifier(reject_first=1)

    outcome = _run(generator, verifier)

    assert generator.waves == [0, 2]
    assert verifier.calls == 2
    assert len(outcome.accepted) == 1, "the replacement attempt must still be accepted"
    assert outcome.unresolved_slot_ids == ()
    assert outcome.replacement_wave_count == 2


def test_an_approved_replacement_still_completes_the_group() -> None:
    generator, verifier = _DistinctGenerator(), _Verifier(reject_first=1)

    outcome = _run(generator, verifier)

    accepted = outcome.accepted[0]
    assert accepted.slot.slot_id == "slot-001"
    assert accepted.verification.decision is VerificationDecision.ACCEPT
    assert outcome.terminal_rejection is False
    assert outcome.provider_failure_stage is None


# --- 4. the regression this fix exists for -----------------------------------


def test_a_rejection_at_the_replacement_wave_exhausts_instead_of_looping() -> None:
    """The critical case: every attempt is rejected and every question is new."""
    generator, verifier = _DistinctGenerator(), _Verifier(reject_first=99)

    outcome = _run(generator, verifier)

    assert generator.waves == [0, 2], f"wave sequence did not advance: {generator.waves}"
    assert verifier.calls == 2
    assert outcome.accepted == ()
    assert outcome.unresolved_slot_ids == ("slot-001",)
    assert "INCORRECT_KEY" in outcome.reason_codes


# --- 5. earlier accepted questions are untouched -----------------------------


def test_one_exhausting_slot_leaves_the_other_accepted_questions_intact() -> None:
    class _RejectOneSlot(_Verifier):
        def verify_slot(self, *, request, bucket, slot, question):  # noqa: ANN001
            del request, bucket
            self.calls += 1
            if slot.slot_id == "slot-002":
                return VerificationResult(
                    schema_version="2",
                    generation_item_id=question.generation_item_id,
                    slot_id=slot.slot_id,
                    decision=VerificationDecision.REGENERATE,
                    valid_option_ids=[question.correct_option_id],
                    reason_codes=["INCORRECT_KEY"],
                )
            return VerificationResult(
                schema_version="2",
                generation_item_id=question.generation_item_id,
                slot_id=slot.slot_id,
                decision=VerificationDecision.ACCEPT,
                valid_option_ids=[question.correct_option_id],
                answer_explanation="The independently selected option is correct.",
                reason_codes=["MATCH"],
            )

    generator = _DistinctGenerator()

    outcome = _run(generator, _RejectOneSlot(), slot_count=3)

    assert generator.waves == [0, 2]
    accepted_slots = sorted(item.slot.slot_id for item in outcome.accepted)
    assert accepted_slots == ["slot-001", "slot-003"], "accepted questions were lost"
    assert outcome.unresolved_slot_ids == ("slot-002",)


def test_mixed_semantic_recovery_corrects_authority_key_and_repairs_no_valid_option() -> None:
    class _MixedRecoveryVerifier:
        def __init__(self) -> None:
            self.calls_by_slot: dict[str, int] = {}

        def verify_slot(self, *, request, bucket, slot, question):  # noqa: ANN001
            del request, bucket
            calls = self.calls_by_slot.get(slot.slot_id, 0) + 1
            self.calls_by_slot[slot.slot_id] = calls
            if slot.slot_id == "slot-001" and calls == 1:
                return VerificationResult(
                    schema_version="2",
                    generation_item_id=question.generation_item_id,
                    slot_id=slot.slot_id,
                    decision=VerificationDecision.ACCEPT,
                    valid_option_ids=["0"],
                    answer_explanation="Option 0 is the independently solved answer.",
                    reason_codes=["SINGLE_VALID_OPTION"],
                )
            if slot.slot_id == "slot-002" and calls == 1:
                return VerificationResult(
                    schema_version="2",
                    generation_item_id=question.generation_item_id,
                    slot_id=slot.slot_id,
                    decision=VerificationDecision.REGENERATE,
                    valid_option_ids=[],
                    reason_codes=["NO_VALID_OPTION"],
                )
            return VerificationResult(
                schema_version="2",
                generation_item_id=question.generation_item_id,
                slot_id=slot.slot_id,
                decision=VerificationDecision.ACCEPT,
                valid_option_ids=[question.correct_option_id],
                answer_explanation="The independently selected option is correct.",
                reason_codes=["MATCH"],
            )

    generator = _DistinctGenerator()
    outcome = _run(generator, _MixedRecoveryVerifier(), slot_count=2)

    assert generator.slot_ids_by_wave == [
        (0, ("slot-001", "slot-002")),
        (1, ("slot-002",)),
    ]
    assert sorted(item.slot.slot_id for item in outcome.accepted) == [
        "slot-001",
        "slot-002",
    ]


# --- 6-7. provider failure and the attempt budget ----------------------------


def test_a_generator_provider_failure_stays_bounded() -> None:
    generator = _DistinctGenerator(
        raise_on_wave={0: ProviderExecutionError("down", failure_kind="timeout")}
    )
    verifier = _Verifier()

    outcome = _run(generator, verifier)

    assert generator.waves == [0], "a provider failure must not consume content waves"
    assert verifier.calls == 0
    assert outcome.provider_failure_stage is None
    assert outcome.provider_failure_recoverable is True
    assert outcome.unresolved_slot_ids == ("slot-001",)


def test_no_slot_group_ever_exceeds_three_generation_attempts() -> None:
    """The wave loop admits waves 0, 1 and 2 — never more, on any rejection path."""
    for reject_first in range(0, 6):
        generator = _DistinctGenerator()
        _run(generator, _Verifier(reject_first=reject_first))
        assert len(generator.waves) <= 3, (
            f"reject_first={reject_first} produced {generator.waves}"
        )
        assert generator.waves == sorted(generator.waves), generator.waves
        assert len(set(generator.waves)) == len(generator.waves), (
            f"a wave repeated: {generator.waves}"
        )


# --- 8. the exhausted group reaches a deterministic terminal state -----------


def test_an_exhausted_group_fails_only_its_group_and_lets_generation_continue() -> None:
    outcome = _run(_DistinctGenerator(), _Verifier(reject_first=99))
    committed: dict[str, object] = {}

    class _Assessments:
        @staticmethod
        def get(_test_id):
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
        @staticmethod
        def list_linked(_test_id, *, limit=100):
            del limit
            return []

    orchestrator = object.__new__(PracticeGenerationOrchestrator)
    orchestrator._config = _config()
    orchestrator._assessments = _Assessments()
    orchestrator._questions = _Questions()
    orchestrator._semantic_resolver = None
    orchestrator._mark_failed = lambda test_id, reason_code, **kwargs: committed.update(
        {"reason": reason_code, "meta": kwargs.get("meta_updates")}
    )
    orchestrator._update_progress = lambda *a, **k: committed.setdefault(
        "meta", k.get("meta_updates")
    )

    proceed = orchestrator._commit_slot_outcome(outcome)

    assert proceed is True, "other independent groups must still run"
    assert "reason" not in committed, "the parent is failed once, at finalization"
    group = committed["meta"]["generationGroups"]["slot-group-001"]
    assert (group["state"], group["errorCode"]) == ("FAILED", "GENERATION_DEFICIT_EXHAUSTED")


class _AmbiguityVerifier:
    """Names the author's own key yet rejects: right under one reading only."""

    def __init__(self) -> None:
        self.calls = 0

    def verify_slot(self, *, request, bucket, slot, question):  # noqa: ANN001
        del request, bucket
        self.calls += 1
        return VerificationResult(
            schema_version="2",
            generation_item_id=question.generation_item_id,
            slot_id=slot.slot_id,
            decision=VerificationDecision.REGENERATE,
            valid_option_ids=[question.correct_option_id],
            reason_codes=["AMBIGUOUS"],
        )


def test_an_ambiguous_verdict_matching_the_author_key_is_never_accepted() -> None:
    """Id agreement alone is not approval: the authority's decision must be ACCEPT."""
    generator, verifier = _DistinctGenerator(), _AmbiguityVerifier()

    outcome = _run(generator, verifier)

    assert outcome.accepted == ()
    assert outcome.unresolved_slot_ids == ("slot-001",)
    assert generator.waves == [0, 2]


# --- NO_VALID_OPTION: one semantic repair, then one fresh replacement ----------


class _RecordingGenerator(_DistinctGenerator):
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        super().__init__(**kwargs)
        self.excluded_by_wave: dict[int, tuple[str, ...]] = {}
        self.repair_context_by_wave: dict[int, dict] = {}

    def generate_slots(self, *, request, bucket, group, slots, **kwargs):  # noqa: ANN001
        wave = int(kwargs.get("replacement_wave", 0))
        self.excluded_by_wave[wave] = tuple(kwargs.get("exclude_normalized_texts", ()))
        self.repair_context_by_wave[wave] = {
            "candidates": dict(kwargs.get("repair_candidates_by_slot") or {}),
            "reasons": dict(kwargs.get("repair_reason_codes_by_slot") or {}),
        }
        return super().generate_slots(
            request=request, bucket=bucket, group=group, slots=slots, **kwargs
        )


class _NoValidOptionVerifier:
    """The Authority finds no valid option in the first N candidates it sees."""

    def __init__(self, *, reject_first: int) -> None:
        self._reject_first = reject_first
        self.verified_questions: list[str] = []

    def verify_slot(self, *, request, bucket, slot, question):  # noqa: ANN001
        del request, bucket
        self.verified_questions.append(question.question)
        if len(self.verified_questions) <= self._reject_first:
            return VerificationResult(
                schema_version="2",
                generation_item_id=question.generation_item_id,
                slot_id=slot.slot_id,
                decision=VerificationDecision.REGENERATE,
                valid_option_ids=[],
                reason_codes=["NO_VALID_OPTION"],
            )
        return VerificationResult(
            schema_version="2",
            generation_item_id=question.generation_item_id,
            slot_id=slot.slot_id,
            decision=VerificationDecision.ACCEPT,
            valid_option_ids=[question.correct_option_id],
            answer_explanation="The independently selected option is correct.",
            reason_codes=["MATCH"],
        )


def test_no_valid_option_is_repaired_and_the_repair_is_reverified() -> None:
    generator = _RecordingGenerator()
    verifier = _NoValidOptionVerifier(reject_first=1)

    outcome = _run(generator, verifier)

    assert generator.waves == [0, 1]
    assert [text.split(":")[0] for text in verifier.verified_questions] == [
        "Attempt 1",
        "Attempt 2",
    ]
    assert outcome.accepted[0].question.question.startswith("Attempt 2")
    repair = generator.repair_context_by_wave[1]
    assert repair["candidates"]["slot-001"].question.startswith("Attempt 1")
    assert repair["reasons"]["slot-001"] == ("NO_VALID_OPTION", "NO_VALID_OPTION")


def test_failed_repair_takes_one_fresh_replacement_excluding_both_candidates() -> None:
    generator = _RecordingGenerator()
    verifier = _NoValidOptionVerifier(reject_first=2)

    outcome = _run(generator, verifier)

    assert generator.waves == [0, 1, 2]
    assert len(verifier.verified_questions) == 3
    assert outcome.accepted[0].question.question.startswith("Attempt 3")
    excluded = set(generator.excluded_by_wave[2])
    for failed in verifier.verified_questions[:2]:
        assert any(
            normalize_question_identity(failed).split("|")[0] in value for value in excluded
        )


def test_all_three_no_valid_option_attempts_end_bounded_without_a_fourth() -> None:
    generator = _RecordingGenerator()
    verifier = _NoValidOptionVerifier(reject_first=99)

    outcome = _run(generator, verifier)

    assert generator.waves == [0, 1, 2]
    assert len(verifier.verified_questions) == 3
    assert outcome.accepted == ()
    assert outcome.unresolved_slot_ids == ("slot-001",)
    assert outcome.terminal_rejection is False
    assert outcome.reason_codes[0] == "NO_VALID_OPTION"


def test_empty_valid_options_without_authority_no_valid_option_still_skip_repair() -> None:
    class _AmbiguousVerifier(_NoValidOptionVerifier):
        def verify_slot(self, *, request, bucket, slot, question):  # noqa: ANN001
            result = super().verify_slot(
                request=request, bucket=bucket, slot=slot, question=question
            )
            if result.decision is VerificationDecision.REGENERATE:
                return result.model_copy(update={"reason_codes": ["AMBIGUOUS"]})
            return result

    generator = _RecordingGenerator()

    _run(generator, _AmbiguousVerifier(reject_first=1))

    assert generator.waves == [0, 2]


def test_rejection_event_reports_the_authority_code_that_chose_the_path(
    monkeypatch,  # noqa: ANN001
) -> None:
    import features.practice_generation.orchestration as orchestration_module

    events: list[dict] = []
    monkeypatch.setattr(
        orchestration_module,
        "emit_practice_event",
        lambda name, **kwargs: events.append({"name": name, **kwargs.get("details", {})}),
    )

    _run(_RecordingGenerator(), _NoValidOptionVerifier(reject_first=1))

    rejected = [e for e in events if e["name"] == "QUESTION_VERIFICATION_RESULT"
                and e.get("reasonCode") != "VERIFIER_APPROVED"]
    assert rejected[0]["reasonCode"] == "NO_VALID_OPTION"
    assert rejected[0]["authorityReasonCode"] == "NO_VALID_OPTION"
    assert rejected[0]["replacementWave"] == 0
