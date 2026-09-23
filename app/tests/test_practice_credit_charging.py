"""Practice-specific credit settlement keeps the established worker boundary."""

from __future__ import annotations

from decimal import Decimal

import pytest

import features.practice_generation.orchestration as orchestration_module
from features.practice_generation.orchestration import PracticeGenerationOrchestrator
from features.practice_generation.schemas import PracticeGenerationRequest
from schemas.student_credits import StudentCreditPolicy
from services.student_credits import (
    StudentCreditRepository,
    StudentCreditRuntime,
    practice_reference_id,
)
from tests.test_student_credits import (
    CREDIT_LEDGER_TABLE,
    USER_CREDITS_TABLE,
    FakeDynamoDBClient,
    _summary,
)


def _request(*, requested: int = 3, accepted: int = 3) -> PracticeGenerationRequest:
    return PracticeGenerationRequest(
        request_id="request-1",
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        original_query="Create algebra questions",
        practice_type="QUIZ",
        requested_count=requested,
        accepted_count=accepted,
        limitation=(
            None
            if requested == accepted
            else {
                "type": "QUESTION_COUNT_LIMIT",
                "requested_value": requested,
                "effective_value": accepted,
                "maximum_value": 50,
                "message_key": "PRACTICE_MAX_QUESTIONS_LIMITED",
            }
        ),
        subject="math",
        topic="algebra",
        difficulty="intermediate",
        language="english",
        exam_id="CAT",
        assessment_title="Algebra practice",
    )


def _runtime(client: FakeDynamoDBClient) -> StudentCreditRuntime:
    return StudentCreditRuntime(
        policy=StudentCreditPolicy(
            enforcement_enabled=True,
            dry_run=False,
            credits_per_usd=Decimal("50"),
            target_gross_margin=Decimal("0.40"),
            rounding_mode="CEIL",
            doubt_authorization_credits=5,
            practice_min_authorization_credits=5,
            practice_authorization_credits_per_question=1,
        ),
        repository=StudentCreditRepository(
            user_credits_table=USER_CREDITS_TABLE,
            credit_ledger_table=CREDIT_LEDGER_TABLE,
            client=client,
        ),
    )


def test_practice_authorization_precedes_worker_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    test_id = "practice-credit-1"
    client = FakeDynamoDBClient(wallets={"user-1": 10})
    runtime = _runtime(client)
    reference = practice_reference_id(user_id=request.user_id, test_id=test_id)
    runtime.authorize_practice(
        user_id=request.user_id,
        reference_id=reference,
        feature="quick_practice",
        effective_count=request.effective_count,
    )
    monkeypatch.setattr(
        orchestration_module,
        "chargeable_operation_billing",
        lambda **_kwargs: _summary(cost="0.012"),
    )
    orchestrator = object.__new__(PracticeGenerationOrchestrator)
    orchestrator._student_credits = runtime

    assert orchestrator._settle_student_credits(test_id, request) is True
    assert client.wallets["user-1"] == 9
    assert client.ledger[reference]["amount"]["N"] == "1"


def test_failed_practice_releases_its_hold() -> None:
    request = _request()
    client = FakeDynamoDBClient(wallets={"user-1": 10})
    runtime = _runtime(client)
    reference = practice_reference_id(user_id=request.user_id, test_id="failed-practice")
    runtime.authorize_practice(
        user_id=request.user_id,
        reference_id=reference,
        feature="quick_practice",
        effective_count=request.effective_count,
    )

    runtime.release(
        user_id=request.user_id,
        reference_id=reference,
        feature="quick_practice",
        reason="practice_failed",
    )

    assert client.wallets["user-1"] == 10
    assert client.ledger[reference]["authorizationState"]["S"] == "RELEASED"


def test_100_requested_uses_normalized_50_effective_authorization() -> None:
    request = _request(requested=100, accepted=50)
    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime = _runtime(client)

    runtime.authorize_practice(
        user_id=request.user_id,
        reference_id=practice_reference_id(user_id=request.user_id, test_id="50-cap"),
        feature="quick_practice",
        effective_count=request.effective_count,
    )

    assert client.wallets["user-1"] == 50


# --- 20Q incident: settlement above authorization, and the leaked hold -------


def test_practice_settling_above_its_authorization_captures_the_hold() -> None:
    """The incident shape: 20 authorized, 99 calculated. It must settle, not fail."""
    client = FakeDynamoDBClient(wallets={"user-1": 48})
    runtime = _runtime(client)
    reference = practice_reference_id(user_id="user-1", test_id="practice-20q")
    runtime.authorize_practice(
        user_id="user-1", reference_id=reference, feature="mock_test", effective_count=20
    )
    assert client.wallets["user-1"] == 28

    settlement = runtime.settle(
        user_id="user-1",
        reference_id=reference,
        feature="mock_test",
        operation_status="ready",
        summary=_summary(cost="1.188"),  # CEIL(1.188 * 50 / 0.60) = 99 credits
    )

    assert settlement.status == "settled"
    assert settlement.charge.student_credits_to_debit == 99
    assert settlement.credits_debited == 20, "capture is capped at the authorization"
    assert client.wallets["user-1"] == 28, "a full capture releases nothing"
    assert client.ledger[reference]["authorizationState"]["S"] == "SETTLED"


def test_a_failed_practice_releases_its_hold_through_the_launcher() -> None:
    """The owner is on the assessment row; the persisted request never carries it."""
    import json

    from features.practice_generation.agentcore_async import AgentCorePracticeAsyncLauncher

    client = FakeDynamoDBClient(wallets={"user-1": 48})
    runtime = _runtime(client)
    reference = practice_reference_id(user_id="user-1", test_id="practice-leak")
    runtime.authorize_practice(
        user_id="user-1", reference_id=reference, feature="mock_test", effective_count=20
    )

    class _Assessments:
        @staticmethod
        def get(_test_id):
            return {
                "testId": "practice-leak",
                "userId": "user-1",
                "meta": json.dumps({"practiceRequest": {"practiceType": "SECTIONAL_TEST"}}),
            }

    launcher = object.__new__(AgentCorePracticeAsyncLauncher)
    launcher._student_credits = runtime
    launcher._assessments = _Assessments()

    launcher._release_student_credits("practice-leak", "PRACTICE_CREDIT_SETTLEMENT_FAILED")

    assert client.ledger[reference]["authorizationState"]["S"] == "RELEASED"
    assert client.wallets["user-1"] == 48, "the full hold must return to the wallet"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            {"Error": {"Code": "ValidationException"}},
            {
                "exception_type": "ClientError",
                "aws_error_code": "ValidationException",
                "transaction_cancelled": False,
                "cancellation_reason_codes": "",
            },
        ),
        (
            {
                "Error": {"Code": "TransactionCanceledException"},
                "CancellationReasons": [{"Code": "None"}, {"Code": "ThrottlingError"}],
            },
            {
                "exception_type": "ClientError",
                "aws_error_code": "TransactionCanceledException",
                "transaction_cancelled": True,
                "cancellation_reason_codes": "None,ThrottlingError",
            },
        ),
    ],
)
def test_an_unavailable_settlement_logs_the_store_failure_shape(
    monkeypatch: pytest.MonkeyPatch, error: dict, expected: dict
) -> None:
    from botocore.exceptions import ClientError

    from services.student_credits.errors import StudentCreditRepositoryError

    captured: list[dict] = []
    monkeypatch.setattr(
        "services.student_credits.runtime.log_event",
        lambda name, **kwargs: captured.append({"name": name, **kwargs}),
    )
    client = FakeDynamoDBClient(wallets={"user-1": 48})
    runtime = _runtime(client)
    reference = practice_reference_id(user_id="user-1", test_id="practice-diag")
    runtime.authorize_practice(
        user_id="user-1", reference_id=reference, feature="mock_test", effective_count=20
    )

    def _fail(**_kwargs):
        raise ClientError(error, "TransactWriteItems")

    client.transact_write_items = _fail
    balance_before = client.wallets["user-1"]

    with pytest.raises(StudentCreditRepositoryError):
        runtime.settle(
            user_id="user-1",
            reference_id=reference,
            feature="mock_test",
            operation_status="ready",
            summary=_summary(cost="1.188"),
        )

    event = next(
        item for item in captured
        if item["name"] == "student_credit_settlement" and item["status"] == "unavailable"
    )
    details = event["details"]
    assert {key: details[key] for key in expected} == expected
    assert details["actual_calculated_credits"] == 99
    assert "user-1" not in str(details), "no raw user id may reach the settlement log"
    assert details["reference_id"].endswith(":practice-diag")
    assert client.wallets["user-1"] == balance_before, "an uncertain write moves nothing"
    assert client.ledger[reference]["authorizationState"]["S"] == "AUTHORIZED"


# --- hold sizing from the locked, file-backed policy -------------------------


def _file_policy_runtime(client: FakeDynamoDBClient, *, dry_run: bool = False):
    from student_credit_policy import load_student_credit_policy

    return StudentCreditRuntime(
        policy=load_student_credit_policy(enforcement_enabled=True, dry_run=dry_run),
        repository=StudentCreditRepository(
            user_credits_table=USER_CREDITS_TABLE,
            credit_ledger_table=CREDIT_LEDGER_TABLE,
            client=client,
        ),
    )


def test_locked_policy_values_come_from_config() -> None:
    policy = _file_policy_runtime(FakeDynamoDBClient()).policy

    assert policy.credits_per_usd == Decimal("100")
    assert policy.target_gross_margin == Decimal("0.60")
    assert policy.rounding_mode == "CEIL"
    assert policy.practice_authorization_credits_per_question == 5
    assert policy.practice_min_authorization_credits == 5


@pytest.mark.parametrize(("count", "hold"), [(1, 5), (5, 25), (10, 50), (20, 100), (50, 250)])
def test_practice_hold_is_max_of_minimum_and_count_times_rate(count: int, hold: int) -> None:
    client = FakeDynamoDBClient(wallets={"user-1": 1000})
    runtime = _file_policy_runtime(client)
    reference = practice_reference_id(user_id="user-1", test_id=f"practice-hold-{count}")

    authorized = runtime.authorize_practice(
        user_id="user-1", reference_id=reference, feature="mock_test", effective_count=count
    )

    assert authorized == hold
    assert client.wallets["user-1"] == 1000 - hold
    assert client.ledger[reference]["authorizationState"]["S"] == "AUTHORIZED"
    assert client.ledger[reference]["amount"]["N"] == str(hold)


def test_exact_hold_balance_authorizes_to_zero() -> None:
    client = FakeDynamoDBClient(wallets={"user-1": 25})
    runtime = _file_policy_runtime(client)
    reference = practice_reference_id(user_id="user-1", test_id="practice-exact")

    assert runtime.authorize_practice(
        user_id="user-1", reference_id=reference, feature="quick_practice", effective_count=5
    ) == 25
    assert client.wallets["user-1"] == 0


def test_insufficient_hold_balance_moves_nothing_and_reports_the_requirement() -> None:
    from services.student_credits.errors import InsufficientStudentCreditsError

    client = FakeDynamoDBClient(wallets={"user-1": 24})
    runtime = _file_policy_runtime(client)
    reference = practice_reference_id(user_id="user-1", test_id="practice-short")

    with pytest.raises(InsufficientStudentCreditsError) as refused:
        runtime.authorize_practice(
            user_id="user-1", reference_id=reference, feature="quick_practice", effective_count=5
        )

    assert refused.value.client_fields() == {
        "code": "INSUFFICIENT_CREDITS",
        "retryable": False,
        "action": "ADD_CREDITS",
        "requiredCredits": 25,
    }
    assert client.wallets["user-1"] == 24
    assert client.ledger == {}


def test_duplicate_practice_authorization_takes_no_second_hold() -> None:
    from services.student_credits.errors import StudentCreditLedgerConflictError

    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime = _file_policy_runtime(client)
    reference = practice_reference_id(user_id="user-1", test_id="practice-dup")
    runtime.authorize_practice(
        user_id="user-1", reference_id=reference, feature="quick_practice", effective_count=5
    )

    with pytest.raises(StudentCreditLedgerConflictError):
        runtime.authorize_practice(
            user_id="user-1", reference_id=reference, feature="quick_practice", effective_count=5
        )

    assert client.wallets["user-1"] == 75


# --- settlement and release state transitions --------------------------------


def _held(client: FakeDynamoDBClient, test_id: str, count: int = 5):
    runtime = _file_policy_runtime(client)
    reference = practice_reference_id(user_id="user-1", test_id=test_id)
    runtime.authorize_practice(
        user_id="user-1", reference_id=reference, feature="quick_practice", effective_count=count
    )
    return runtime, reference


def _settle_cost(runtime, reference: str, cost: str):
    return runtime.settle(
        user_id="user-1",
        reference_id=reference,
        feature="quick_practice",
        operation_status="ready",
        summary=_summary(cost=cost),
    )


@pytest.mark.parametrize(
    ("cost", "actual", "charged", "refund"),
    [
        ("0.032", 8, 8, 17),    # actual < authorized: CEIL(0.032*100/0.40) = 8
        ("0.100", 25, 25, 0),   # actual == authorized: zero refund
        ("0.160", 40, 25, 0),   # actual >  authorized: capped, zero refund
    ],
)
def test_settlement_charges_actual_capped_at_hold_and_refunds_the_rest(
    cost: str, actual: int, charged: int, refund: int
) -> None:
    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime, reference = _held(client, "practice-settle")
    assert client.wallets["user-1"] == 75

    settlement = _settle_cost(runtime, reference, cost)

    assert settlement.status == "settled"
    assert settlement.charge.student_credits_to_debit == actual
    assert settlement.credits_debited == charged
    assert client.wallets["user-1"] == 75 + refund
    assert client.ledger[reference]["authorizationState"]["S"] == "SETTLED"
    assert client.ledger[reference]["amount"]["N"] == str(charged)
    wallet_update = client.transactions[-1]["TransactItems"][1]["Update"]
    assert ("#creditsBalance" in wallet_update["ExpressionAttributeNames"]) is (refund > 0)

    replay = _settle_cost(runtime, reference, cost)
    assert replay.status == "already_settled"
    assert replay.credits_debited == charged
    assert client.wallets["user-1"] == 75 + refund, "a replayed settlement moves nothing"


def test_release_returns_the_full_hold_once() -> None:
    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime, reference = _held(client, "practice-release")

    runtime.release(
        user_id="user-1", reference_id=reference, feature="quick_practice", reason="failed"
    )
    runtime.release(
        user_id="user-1", reference_id=reference, feature="quick_practice", reason="failed"
    )

    assert client.wallets["user-1"] == 100
    assert client.ledger[reference]["authorizationState"]["S"] == "RELEASED"
    assert client.ledger[reference]["amount"]["N"] == "0"


def test_settled_authorization_cannot_be_released() -> None:
    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime, reference = _held(client, "practice-settled-release")
    _settle_cost(runtime, reference, "0.032")
    wallet_after_settle = client.wallets["user-1"]

    runtime.release(
        user_id="user-1", reference_id=reference, feature="quick_practice", reason="late"
    )

    assert client.wallets["user-1"] == wallet_after_settle
    assert client.ledger[reference]["authorizationState"]["S"] == "SETTLED"


def test_released_authorization_can_never_charge() -> None:
    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime, reference = _held(client, "practice-released-settle")
    runtime.release(
        user_id="user-1", reference_id=reference, feature="quick_practice", reason="failed"
    )

    settlement = _settle_cost(runtime, reference, "0.160")

    assert settlement.status == "already_released"
    assert settlement.credits_debited == 0
    assert client.wallets["user-1"] == 100
    assert client.ledger[reference]["authorizationState"]["S"] == "RELEASED"


def test_a_release_that_loses_the_race_to_settlement_refunds_nothing() -> None:
    """Both paths read AUTHORIZED; the ledger condition lets exactly one commit."""
    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime, reference = _held(client, "practice-race")
    repository = runtime._repository
    stale = dict(client.ledger[reference])
    _settle_cost(runtime, reference, "0.032")
    wallet_after_settle = client.wallets["user-1"]

    real_read = repository._read_ledger
    reads = {"count": 0}

    def _stale_first_read(ref):
        reads["count"] += 1
        return stale if reads["count"] == 1 else real_read(ref)

    repository._read_ledger = _stale_first_read
    status, released, _ = repository.release_authorization(
        reference_id=reference, user_id="user-1", metadata={"releaseReason": "race"}
    )

    assert status == "already_settled"
    assert released == 0
    assert client.wallets["user-1"] == wallet_after_settle
    assert client.ledger[reference]["authorizationState"]["S"] == "SETTLED"


def test_settlement_without_a_hold_fails_closed_without_charging() -> None:
    from services.student_credits.errors import StudentCreditRepositoryError

    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime = _file_policy_runtime(client)

    with pytest.raises(StudentCreditRepositoryError):
        _settle_cost(runtime, "student-credit:practice:user-1:never-held", "0.032")

    assert client.wallets["user-1"] == 100
    assert client.transaction_calls == 0


def test_a_hold_owned_by_another_student_is_never_touched() -> None:
    from services.student_credits.errors import StudentCreditLedgerConflictError

    client = FakeDynamoDBClient(wallets={"user-1": 100, "user-2": 100})
    runtime, reference = _held(client, "practice-owner")

    with pytest.raises(StudentCreditLedgerConflictError):
        runtime.release(
            user_id="user-2", reference_id=reference, feature="quick_practice", reason="x"
        )

    assert client.wallets == {"user-1": 75, "user-2": 100}
    assert client.ledger[reference]["authorizationState"]["S"] == "AUTHORIZED"


def test_a_failed_release_write_leaves_the_hold_intact() -> None:
    from botocore.exceptions import ClientError

    from services.student_credits.errors import StudentCreditRepositoryError

    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime, reference = _held(client, "practice-release-outage")

    def _unavailable(**_kwargs):
        raise ClientError({"Error": {"Code": "ServiceUnavailable"}}, "TransactWriteItems")

    client.transact_write_items = _unavailable

    with pytest.raises(StudentCreditRepositoryError):
        runtime.release(
            user_id="user-1", reference_id=reference, feature="quick_practice", reason="x"
        )

    assert client.wallets["user-1"] == 75
    assert client.ledger[reference]["authorizationState"]["S"] == "AUTHORIZED"


@pytest.mark.parametrize(
    "reason", ["PRACTICE_CREDIT_SETTLEMENT_FAILED", "PRACTICE_CANCELLED", "CANCELLED"]
)
def test_launcher_release_uses_the_assessment_owner_for_any_non_ready_end(
    reason: str,
) -> None:
    import json

    from features.practice_generation.agentcore_async import AgentCorePracticeAsyncLauncher

    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime, reference = _held(client, "practice-end")

    class _Assessments:
        @staticmethod
        def get(_test_id):
            return {
                "testId": "practice-end",
                "userId": "user-1",
                "meta": json.dumps({"practiceRequest": {"practiceType": "QUICK_PRACTICE"}}),
            }

    launcher = object.__new__(AgentCorePracticeAsyncLauncher)
    launcher._student_credits = runtime
    launcher._assessments = _Assessments()

    launcher._release_student_credits("practice-end", reason)
    launcher._release_student_credits("practice-end", reason)  # duplicate cleanup

    assert client.wallets["user-1"] == 100
    assert client.ledger[reference]["authorizationState"]["S"] == "RELEASED"


# --- client contract: technical failures are never "add credits" -------------


def test_a_store_outage_at_authorization_is_not_reported_as_insufficient() -> None:
    from botocore.exceptions import ClientError

    from services.student_credits.errors import (
        InsufficientStudentCreditsError,
        StudentCreditRepositoryError,
    )

    client = FakeDynamoDBClient(wallets={"user-1": 100})
    runtime = _file_policy_runtime(client)

    def _validation(**_kwargs):
        raise ClientError({"Error": {"Code": "ValidationException"}}, "TransactWriteItems")

    client.transact_write_items = _validation

    with pytest.raises(StudentCreditRepositoryError) as failure:
        runtime.authorize_practice(
            user_id="user-1",
            reference_id=practice_reference_id(user_id="user-1", test_id="practice-outage"),
            feature="quick_practice",
            effective_count=5,
        )

    assert not isinstance(failure.value, InsufficientStudentCreditsError)
    assert failure.value.reason_code == "STUDENT_CREDIT_STORE_UNAVAILABLE"
    assert client.wallets["user-1"] == 100


def test_insufficient_practice_hold_refuses_before_the_launch_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.doubt_solver.streaming_doubt_solver_service import (
        StreamDoubtSolverInput,
        stream_doubt_solver,
    )
    from tests.test_orchestrated_streaming import _REQUEST_ID, _make_adapter

    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    launches: list[object] = []

    class Launcher:
        def launch(self, request):
            launches.append(request)
            raise AssertionError("generation must not start without a hold")

    client = FakeDynamoDBClient(wallets={"user-1": 24})

    events = list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id=_REQUEST_ID,
                actor_id="user-1",
                conversation_id="conversation-1",
                turn_id="turn-1",
                query="Create 5 reasoning questions",
                original_query="Create 5 reasoning questions",
                language="english",
                exam_id="CAT",
                classification={
                    "subject": "reasoning",
                    "topic": "reasoning",
                    "intent": "practice",
                    "difficulty": "intermediate",
                    "retrieval_required": False,
                },
                classifier_confidence=0.99,
            ),
            adapter=_make_adapter(),
            practice_launcher=Launcher(),
            student_credits=_file_policy_runtime(client),
        )
    )

    assert launches == []
    assert events[-1].type == "error"
    assert events[-1].metadata == {
        "code": "INSUFFICIENT_CREDITS",
        "retryable": False,
        "action": "ADD_CREDITS",
        "requiredCredits": 25,
    }
    assert client.wallets["user-1"] == 24
    assert client.ledger == {}
