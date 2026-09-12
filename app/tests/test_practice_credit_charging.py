"""Practice student-credit charging: only the accepted path reaches the wallet.

Provider usage telemetry is deliberately untouched by these tests — every call a
Practice makes, including a failed one, stays in the operation's record ledger.
What is asserted here is the *second*, narrower set: the calls that produced the
delivered questions, which is what the student pays for.
"""

from __future__ import annotations

import threading
from decimal import Decimal
from typing import Any

import pytest

from features.practice_generation.config import PracticeGenerationConfig
from features.practice_generation.generation import ParsedGeneration
from features.practice_generation.orchestration import PracticeGenerationOrchestrator
from features.practice_generation.schemas import (
    Complexity,
    DemandBucket,
    Difficulty,
    GeneratedQuestion,
    GenerationGroup,
    PlannerSlot,
    PracticeGenerationRequest,
    QuestionType,
    VerificationDecision,
    VerificationPolicy,
    VerificationResult,
)
from observability import bind_execution_context
from observability.llm_usage import record_llm_call
from schemas.billing import BillingConfig
from schemas.llm_usage import ProviderTokenUsage
from schemas.student_credits import StudentCreditPolicy
from services.llm.billing import (
    OperationDescriptor,
    OperationUsageAccumulator,
    calculate_operation_billing,
    chargeable_operation_billing,
)
from services.llm.orchestration.errors import ProviderExecutionError
from services.student_credits import (
    StudentCreditRepository,
    StudentCreditRuntime,
    practice_reference_id,
)
from tests.test_student_credits import (
    CREDIT_LEDGER_TABLE,
    USER_CREDITS_TABLE,
    FakeDynamoDBClient,
)

GENERATOR_TOKENS = (1000, 400)
VERIFIER_TOKENS = (500, 50)
PLANNER_TOKENS = (700, 300)


def _billing_config() -> BillingConfig:
    return BillingConfig.model_validate(
        {
            "billing_version": "billing-test-v1",
            "metering_enabled": True,
            "credit_debit_enabled": False,
            "models": [
                {
                    "provider": "openai",
                    "model_or_deployment": "gpt-test",
                    "input_cost_per_million_tokens": "1.00",
                    "output_cost_per_million_tokens": "2.00",
                    "effective_from": "2026-01-01",
                },
                {
                    "provider": "openai",
                    "model_or_deployment": "gpt-unpriced",
                    "input_cost_per_million_tokens": "1.00",
                    "output_cost_per_million_tokens": "2.00",
                    "effective_from": "2026-01-01",
                },
            ],
            "infra": {
                "doubt": {"fixed_cost": "0.001"},
                "quick_practice": {"fixed_cost": "0.003"},
                "mock_test": {"fixed_cost": "0.006"},
            },
            "credits": {"usd_per_credit": "0.001", "pricing_factor": "2"},
        }
    )


def _emit_usage(
    role: str,
    tokens: tuple[int, int],
    *,
    status: str = "succeeded",
    model: str = "gpt-test",
    usage_available: bool = True,
) -> None:
    """Record one provider call exactly the way the real executor does."""
    record_llm_call(
        request_id="request-practice-credit",
        role=role,
        provider="openai",
        model=model,
        deployment=None,
        attempt_type="primary",
        streaming=False,
        usage=(
            ProviderTokenUsage(input_tokens=tokens[0], output_tokens=tokens[1])
            if usage_available
            else ProviderTokenUsage()
        ),
        duration_ms=5,
        status=status,  # type: ignore[arg-type]
        error_type=None if status == "succeeded" else "LlmProviderExecutionError",
    )


def _cost(calls: list[tuple[int, int]]) -> Decimal:
    """Independent expected cost, from the same rates the config declares."""
    return sum(
        (Decimal(i) * Decimal("1.00") + Decimal(o) * Decimal("2.00")) / Decimal(1_000_000)
        for i, o in calls
    )


# ---------------------------------------------------------------------------
# Practice doubles
# ---------------------------------------------------------------------------


class _Generator:
    """Emits one authoring call per wave, then returns the requested questions."""

    def __init__(self, *, raise_on_wave: dict[int, BaseException] | None = None) -> None:
        self._raise_on_wave = raise_on_wave or {}
        self.calls = 0

    def generate_slots(self, **kwargs: Any) -> Any:
        wave = int(kwargs["replacement_wave"])
        self.calls += 1
        _emit_usage("math.generator.default", GENERATOR_TOKENS)
        if wave in self._raise_on_wave:
            raise self._raise_on_wave[wave]

        class _Batch:
            route_id = "math.generator.default"
            model = "gpt-test"
            content = ""

        return _Batch()


class _Verifier:
    def __init__(
        self,
        *,
        reject_first: dict[str, int] | None = None,
        terminal_reject: frozenset[str] = frozenset(),
        raise_for: dict[str, BaseException] | None = None,
        usage_unavailable_for: frozenset[str] = frozenset(),
    ) -> None:
        # Rejects the first N verifications of a slot; later attempts are approved.
        self._reject_first = dict(reject_first or {})
        self._terminal_reject = terminal_reject
        self._raise_for = raise_for or {}
        self._usage_unavailable_for = usage_unavailable_for
        self._lock = threading.Lock()
        self.calls: list[str] = []

    def verify_slot(self, *, request: Any, bucket: Any, slot: Any, question: Any) -> Any:
        del request, bucket
        with self._lock:
            self.calls.append(slot.slot_id)
        failure = self._raise_for.pop(slot.slot_id, None)
        if failure is not None:
            _emit_usage(
                "math.verifier.default", VERIFIER_TOKENS, status="failed",
                usage_available=False,
            )
            raise failure
        _emit_usage(
            "math.verifier.default",
            VERIFIER_TOKENS,
            usage_available=slot.slot_id not in self._usage_unavailable_for,
            model=("gpt-unpriced" if slot.slot_id in self._usage_unavailable_for else "gpt-test"),
        )
        if slot.slot_id in self._terminal_reject:
            return VerificationResult(
                schema_version="2",
                generation_item_id=question.generation_item_id,
                slot_id=slot.slot_id,
                decision=VerificationDecision.TERMINAL_REJECTION,
                valid_option_ids=[],
                reason_codes=["NO_VALID_OPTION"],
            )
        remaining = self._reject_first.get(slot.slot_id, 0)
        if remaining > 0:
            self._reject_first[slot.slot_id] = remaining - 1
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
            reason_codes=["MATCH"],
        )


def _slot(slot_id: str) -> PlannerSlot:
    return PlannerSlot(
        slot_id=slot_id,
        subject_id="math",
        topic_id="algebra",
        category_id="algebra",
        difficulty=Difficulty.INTERMEDIATE,
        complexity=Complexity.MEDIUM,
        exam_ids=["CAT"],
        question_type=QuestionType.MCQ,
        target_skill="solve_linear_equation",
        variation_hint=slot_id,
        generator_route_hint="math.generator.intermediate",
        reasoning_target="isolate_variable",
    )


def _question(slot_id: str, wave: int) -> GeneratedQuestion:
    return GeneratedQuestion(
        schema_version="2",
        generation_item_id=f"item-{slot_id}-{wave}",
        bucket_id="bucket-1",
        slot_id=slot_id,
        question=f"Solve {slot_id} variant {wave}.",
        question_type=QuestionType.MCQ,
        options=[
            {"option_id": "0", "value": "1"},
            {"option_id": "1", "value": "2"},
            {"option_id": "2", "value": "3"},
            {"option_id": "3", "value": "4"},
        ],
        correct_option_id="1",
        correct_answer="2",
        answer_explanation="Isolate the variable.",
        solution="Isolate the variable.",
        subject="math",
        topic="algebra",
        difficulty=Difficulty.INTERMEDIATE,
    )


def _request() -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        original_query="Create algebra questions",
        practice_type="QUIZ",
        requested_count=3,
        accepted_count=3,
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        assessment_title="Algebra practice",
    )


def _bucket() -> DemandBucket:
    return DemandBucket(
        bucket_id="bucket-1",
        subject="math",
        topic="algebra",
        difficulty=Difficulty.INTERMEDIATE,
        question_type=QuestionType.MCQ,
        required_count=1,
        question_intent="Solve a linear equation for the unknown.",
        verification_policy=VerificationPolicy.MANDATORY,
    )


def _orchestrator(generator: Any, verifier: Any) -> Any:
    orchestrator = object.__new__(PracticeGenerationOrchestrator)
    orchestrator._config = PracticeGenerationConfig(
        enabled=True,
        pattern_context_enabled=False,
        pattern_reuse_enabled=False,
        assessment_table="assessment",
        question_table="question",
        question_bank_table="bank",
        question_bank_category_index="category-index",
        question_test_index="test-index",
        aws_region="ap-south-1",
        verification_max_concurrency=3,
    )
    orchestrator._generator = generator
    orchestrator._verifier = verifier
    orchestrator._progress_updates = None
    orchestrator._student_credits = None
    return orchestrator


def _group(slot_ids: list[str]) -> GenerationGroup:
    return GenerationGroup(
        group_id="group-1",
        bucket_id="bucket-1",
        required_count=len(slot_ids),
        attempt=0,
        slot_ids=slot_ids,
    )


def _context(slot_ids: list[str]) -> Any:
    from features.practice_generation.orchestration import _SlotGenerationContext

    return _SlotGenerationContext(
        test_id="practice-credit-1",
        request=_request(),
        blueprint=None,  # type: ignore[arg-type]
        group=_group(slot_ids),
        bucket=_bucket(),
        slots=tuple(_slot(slot_id) for slot_id in slot_ids),
        excluded_texts=(),
        pattern_guidance_by_slot={},
        pattern_selections_by_slot={},
    )


@pytest.fixture()
def operation(monkeypatch: pytest.MonkeyPatch) -> OperationUsageAccumulator:
    monkeypatch.setattr("services.llm.billing.load_billing_config", _billing_config)
    monkeypatch.setattr(
        "features.practice_generation.orchestration.parse_partial_generation",
        lambda content, **kwargs: _parsed(kwargs["slots"], kwargs["group"]),
    )
    return OperationUsageAccumulator(
        descriptor=OperationDescriptor(
            operation_id="practice-credit-1", feature="quick_practice"
        )
    )


def _parsed(slots: Any, group: Any) -> ParsedGeneration:
    """Stand in for the real parser: every returned question is well formed."""
    wave = int(group.attempt)
    skip = _SKIP.get(wave, frozenset())
    return ParsedGeneration(
        accepted=tuple(
            _question(slot.slot_id, wave) for slot in slots if slot.slot_id not in skip
        ),
        rejected_count=len(skip),
    )


_SKIP: dict[int, frozenset[str]] = {}


def _run_group(operation: OperationUsageAccumulator, orchestrator: Any, slot_ids: list[str]):
    with bind_execution_context(
        operation_id="practice-credit-1",
        feature="quick_practice",
        operation_accumulator=operation,
    ):
        return orchestrator._execute_slot_group(_context(slot_ids))


def _chargeable_cost(operation: OperationUsageAccumulator) -> Decimal | None:
    descriptor, records = operation.snapshot_chargeable()
    return calculate_operation_billing(
        descriptor=descriptor, records=records, operation_status="ready"
    ).total_llm_cost_usd


# ---------------------------------------------------------------------------
# 1-5. Chargeable path selection
# ---------------------------------------------------------------------------


def test_earlier_accepted_questions_survive_a_later_generator_failure(
    operation: OperationUsageAccumulator,
) -> None:
    """A failing group must never reset what previous groups already earned."""
    _SKIP.clear()
    good = _orchestrator(_Generator(), _Verifier())
    _run_group(operation, good, ["slot-001"])
    _run_group(operation, good, ["slot-002"])
    after_two = _chargeable_cost(operation)

    failing = _orchestrator(
        _Generator(
            raise_on_wave={
                0: ProviderExecutionError("boom", failure_kind="output_token_exhausted")
            }
        ),
        _Verifier(),
    )
    outcome = _run_group(operation, failing, ["slot-003"])

    assert outcome.accepted == ()
    assert after_two == _cost([GENERATOR_TOKENS, VERIFIER_TOKENS] * 2)
    assert _chargeable_cost(operation) == after_two, "a failed group reset earned cost"


def test_generator_success_with_verifier_rejection_charges_nothing(
    operation: OperationUsageAccumulator,
) -> None:
    _SKIP.clear()
    generator = _Generator()
    verifier = _Verifier(terminal_reject=frozenset({"slot-001"}))
    orchestrator = _orchestrator(generator, verifier)

    outcome = _run_group(operation, orchestrator, ["slot-001"])

    assert outcome.accepted == ()
    assert generator.calls == 1 and len(verifier.calls) == 1
    # Both calls are still in the truthful provider ledger …
    assert len(operation.records) == 2
    # … and neither reaches the student.
    assert _chargeable_cost(operation) == Decimal(0)


def test_only_the_accepted_replacement_attempt_is_charged(
    operation: OperationUsageAccumulator,
) -> None:
    """Attempt A is rejected, attempt B is approved: only B reaches the student."""
    _SKIP.clear()
    generator = _Generator()
    verifier = _Verifier(reject_first={"slot-001": 1})
    orchestrator = _orchestrator(generator, verifier)

    outcome = _run_group(operation, orchestrator, ["slot-001"])

    assert len(outcome.accepted) == 1
    assert generator.calls == 2 and len(verifier.calls) == 2
    assert len(operation.records) == 4, "telemetry must keep all four calls"
    assert _chargeable_cost(operation) == _cost([GENERATOR_TOKENS, VERIFIER_TOKENS])


def test_generator_provider_failure_contributes_zero(
    operation: OperationUsageAccumulator,
) -> None:
    _SKIP.clear()
    good = _orchestrator(_Generator(), _Verifier())
    _run_group(operation, good, ["slot-001"])
    earned = _chargeable_cost(operation)

    failing = _orchestrator(
        _Generator(
            raise_on_wave={
                0: ProviderExecutionError("down", failure_kind="provider_unavailable")
            }
        ),
        _Verifier(),
    )
    _run_group(operation, failing, ["slot-002"])

    assert _chargeable_cost(operation) == earned


def test_verifier_provider_failure_contributes_zero(
    operation: OperationUsageAccumulator,
) -> None:
    """The candidate is discarded, and the unusable verifier call is not charged."""
    _SKIP.clear()
    good = _orchestrator(_Generator(), _Verifier())
    _run_group(operation, good, ["slot-001"])
    earned = _chargeable_cost(operation)

    failing = _orchestrator(
        _Generator(),
        _Verifier(
            raise_for={
                "slot-002": ProviderExecutionError(
                    "verifier down", failure_kind="unknown_provider_error"
                )
            }
        ),
    )
    outcome = _run_group(operation, failing, ["slot-002"])

    assert outcome.provider_failure_stage == "VERIFIER"
    assert outcome.accepted == ()
    assert _chargeable_cost(operation) == earned
    # The failed call keeps its unknown usage in the provider ledger …
    assert any(record.usage_source == "unavailable" for record in operation.records)
    # … and stays out of the chargeable set, so the charge is still complete.
    _, chargeable = operation.snapshot_chargeable()
    assert all(record.usage_source == "provider_reported" for record in chargeable)


# ---------------------------------------------------------------------------
# 6-7. Planner and incomplete pricing
# ---------------------------------------------------------------------------


def test_only_the_accepted_planner_attempt_is_charged(
    operation: OperationUsageAccumulator,
) -> None:
    """A rejected blueprint attempt still returns content; it is not billable."""
    from services.llm.billing import commit_chargeable_usage

    with bind_execution_context(
        operation_id="practice-credit-1",
        feature="quick_practice",
        operation_accumulator=operation,
    ):
        from services.llm.billing import capture_llm_usage

        with capture_llm_usage() as planner_usage:
            _emit_usage("math.planner.default", (2000, 900))  # rejected attempt
            _emit_usage("math.planner.default", PLANNER_TOKENS)  # accepted repair
        accepted = [r for r in planner_usage if r.status == "succeeded"]
        commit_chargeable_usage((accepted[-1],))

    assert len(operation.records) == 2
    assert _chargeable_cost(operation) == _cost([PLANNER_TOKENS])


def test_an_unpriced_call_on_the_accepted_path_blocks_the_debit(
    monkeypatch: pytest.MonkeyPatch, operation: OperationUsageAccumulator
) -> None:
    _SKIP.clear()
    orchestrator = _orchestrator(
        _Generator(), _Verifier(usage_unavailable_for=frozenset({"slot-001"}))
    )
    _run_group(operation, orchestrator, ["slot-001"])

    with bind_execution_context(
        operation_id="practice-credit-1",
        feature="quick_practice",
        operation_accumulator=operation,
    ):
        summary = chargeable_operation_billing(operation_status="ready")

    assert summary is not None
    assert summary.cost_complete is False
    assert summary.total_llm_cost_usd is None
    assert summary.calculated_credits is None

    client = FakeDynamoDBClient(wallets={"user-1": 500})
    runtime = _runtime(client)
    with pytest.raises(Exception) as raised:
        runtime.settle(
            user_id="user-1",
            reference_id=practice_reference_id(user_id="user-1", test_id="t-1"),
            feature="quick_practice",
            operation_status="ready",
            summary=summary,
        )
    assert type(raised.value).__name__ == "StudentCreditPricingIncompleteError"
    assert client.wallets["user-1"] == 500
    assert client.ledger == {}


# ---------------------------------------------------------------------------
# 8-14. Settlement
# ---------------------------------------------------------------------------


def _runtime(
    client: FakeDynamoDBClient, *, dry_run: bool = False
) -> StudentCreditRuntime:
    return StudentCreditRuntime(
        policy=StudentCreditPolicy(
            enforcement_enabled=True,
            dry_run=dry_run,
            credits_per_usd=Decimal("50"),
            target_gross_margin=Decimal("0.40"),
            rounding_mode="CEIL",
        ),
        repository=StudentCreditRepository(
            user_credits_table=USER_CREDITS_TABLE,
            credit_ledger_table=CREDIT_LEDGER_TABLE,
            client=client,
        ),
    )


def _ready_summary(operation: OperationUsageAccumulator) -> Any:
    with bind_execution_context(
        operation_id="practice-credit-1",
        feature="quick_practice",
        operation_accumulator=operation,
    ):
        return chargeable_operation_billing(operation_status="ready")


def _one_accepted_question(operation: OperationUsageAccumulator) -> None:
    _SKIP.clear()
    _run_group(operation, _orchestrator(_Generator(), _Verifier()), ["slot-001"])


def test_a_failed_practice_never_settles(operation: OperationUsageAccumulator) -> None:
    """Internal chargeable work exists, but an unusable Practice costs nothing."""
    _one_accepted_question(operation)
    assert _chargeable_cost(operation) > 0

    client = FakeDynamoDBClient(wallets={"user-1": 500})
    orchestrator = _orchestrator(_Generator(), _Verifier())
    orchestrator._student_credits = _runtime(client)

    class _Assessments:
        @staticmethod
        def get(test_id: str) -> dict[str, Any]:
            del test_id
            return {"status": "FAILED"}

    orchestrator._assessments = _Assessments()
    with bind_execution_context(
        operation_id="practice-credit-1",
        feature="quick_practice",
        operation_accumulator=operation,
    ):
        orchestrator._finalize(
            "practice-credit-1", _request(), None, attempt=0  # type: ignore[arg-type]
        )

    assert client.wallets["user-1"] == 500
    assert client.ledger == {}


def test_dry_run_calculates_the_exact_charge_without_touching_the_wallet(
    operation: OperationUsageAccumulator,
) -> None:
    _one_accepted_question(operation)
    client = FakeDynamoDBClient(wallets={"user-1": 500})
    runtime = _runtime(client, dry_run=True)
    summary = _ready_summary(operation)

    settlement = runtime.settle(
        user_id="user-1",
        reference_id=practice_reference_id(user_id="user-1", test_id="t-1"),
        feature="quick_practice",
        operation_status="ready",
        summary=summary,
    )

    expected_cost = _cost([GENERATOR_TOKENS, VERIFIER_TOKENS])
    assert summary.total_llm_cost_usd == expected_cost
    assert settlement.status == "skipped_dry_run"
    assert settlement.credits_debited == 0
    assert settlement.charge.student_credits_to_debit == 1  # CEIL(cost*50/0.60)
    assert client.wallets["user-1"] == 500
    assert client.ledger == {}


def test_real_settlement_debits_exactly_the_calculated_credits(
    operation: OperationUsageAccumulator,
) -> None:
    for _ in range(40):
        _one_accepted_question(operation)
    client = FakeDynamoDBClient(wallets={"user-1": 500})
    runtime = _runtime(client)
    summary = _ready_summary(operation)
    expected = int(
        (summary.total_llm_cost_usd * Decimal(50) / Decimal("0.60")).to_integral_value(
            rounding="ROUND_CEILING"
        )
    )

    settlement = runtime.settle(
        user_id="user-1",
        reference_id=practice_reference_id(user_id="user-1", test_id="t-1"),
        feature="quick_practice",
        operation_status="ready",
        summary=summary,
    )

    assert expected > 1
    assert settlement.status == "settled"
    assert settlement.credits_debited == expected
    assert 500 - client.wallets["user-1"] == expected
    assert len(client.ledger) == 1
    assert next(iter(client.ledger)) == "student-credit:practice:user-1:t-1"


def test_replaying_the_same_practice_never_debits_twice(
    operation: OperationUsageAccumulator,
) -> None:
    for _ in range(40):
        _one_accepted_question(operation)
    client = FakeDynamoDBClient(wallets={"user-1": 500})
    runtime = _runtime(client)
    summary = _ready_summary(operation)
    reference_id = practice_reference_id(user_id="user-1", test_id="t-1")

    first = runtime.settle(
        user_id="user-1", reference_id=reference_id, feature="quick_practice",
        operation_status="ready", summary=summary,
    )
    balance_after_first = client.wallets["user-1"]
    second = runtime.settle(
        user_id="user-1", reference_id=reference_id, feature="quick_practice",
        operation_status="ready", summary=summary,
    )

    assert first.status == "settled"
    assert second.status == "already_settled"
    assert second.credits_debited == first.credits_debited
    assert client.wallets["user-1"] == balance_after_first
    assert len(client.ledger) == 1


def test_a_wallet_holding_exactly_the_required_credits_settles_to_zero(
    operation: OperationUsageAccumulator,
) -> None:
    for _ in range(40):
        _one_accepted_question(operation)
    summary = _ready_summary(operation)
    required = int(
        (summary.total_llm_cost_usd * Decimal(50) / Decimal("0.60")).to_integral_value(
            rounding="ROUND_CEILING"
        )
    )
    client = FakeDynamoDBClient(wallets={"user-1": required})
    runtime = _runtime(client)

    settlement = runtime.settle(
        user_id="user-1",
        reference_id=practice_reference_id(user_id="user-1", test_id="t-1"),
        feature="quick_practice",
        operation_status="ready",
        summary=summary,
    )

    assert settlement.status == "settled"
    assert client.wallets["user-1"] == 0


def test_insufficient_balance_leaves_the_wallet_and_ledger_intact(
    operation: OperationUsageAccumulator,
) -> None:
    for _ in range(40):
        _one_accepted_question(operation)
    summary = _ready_summary(operation)
    required = int(
        (summary.total_llm_cost_usd * Decimal(50) / Decimal("0.60")).to_integral_value(
            rounding="ROUND_CEILING"
        )
    )
    client = FakeDynamoDBClient(wallets={"user-1": required - 1})
    runtime = _runtime(client)

    with pytest.raises(Exception) as raised:
        runtime.settle(
            user_id="user-1",
            reference_id=practice_reference_id(user_id="user-1", test_id="t-1"),
            feature="quick_practice",
            operation_status="ready",
            summary=summary,
        )

    assert type(raised.value).__name__ == "InsufficientStudentCreditsError"
    assert client.wallets["user-1"] == required - 1
    assert client.ledger == {}


def test_enforcement_disabled_leaves_practice_with_no_credit_dependency(
    operation: OperationUsageAccumulator,
) -> None:
    _one_accepted_question(operation)
    orchestrator = _orchestrator(_Generator(), _Verifier())
    orchestrator._student_credits = None

    assert orchestrator._settle_student_credits("practice-credit-1", _request()) is True


def test_the_practice_settlement_reference_is_deterministic() -> None:
    first = practice_reference_id(user_id="user-1", test_id="practice-abc")
    second = practice_reference_id(user_id="user-1", test_id="practice-abc")
    assert first == second == "student-credit:practice:user-1:practice-abc"
    assert practice_reference_id(user_id="user-2", test_id="practice-abc") != first
