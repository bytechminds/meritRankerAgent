"""Doubt Solver streaming integration for student credit enforcement.

Covers the ordering contract (settlement precedes the terminal success event),
the plug-out guarantee, and that no non-accepted outcome ever debits.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import pytest

import config as cfg_module
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from schemas.billing import BillingConfig
from schemas.doubt_solver import DoubtSolverStreamEvent
from schemas.student_credits import StudentCreditPolicy
from services.doubt_solver.streaming_doubt_solver_service import (
    StreamDoubtSolverInput,
    stream_doubt_solver,
)
from services.student_credits.repository import StudentCreditRepository
from services.student_credits.runtime import StudentCreditRuntime, doubt_reference_id
from tests.test_student_credits import (
    CREDIT_LEDGER_TABLE,
    USER_CREDITS_TABLE,
    FakeDynamoDBClient,
    _policy,
)

_REQUEST_ID = "credit-stream-001"
_TURN_ID = "turn-credit-1"
_ACTOR_ID = "student-1"
_ANSWER = (
    "**Final Answer:**\n\\(20\\)\n\nUsing percentage = part per hundred, "
    "\\(20\\% \\times 100 = 20\\)."
)
_CLASSIFICATION = {
    "subject": "math",
    "intent": "solve",
    "difficulty": "basic",
    "need_web_search": False,
    "classifier_confidence": 0.5,
    "classification_source": "llm",
}


class _Adapter:
    """Answer generator that also reports one priced provider call."""

    def __init__(self, *, answer: str = _ANSWER, input_tokens: int = 1_000_000) -> None:
        self.answer = answer
        self.input_tokens = input_tokens
        self.calls = 0

    def _record(self) -> None:
        from observability.llm_usage import record_llm_call
        from schemas.llm_usage import ProviderTokenUsage

        record_llm_call(
            request_id=_REQUEST_ID,
            role="math.generator",
            provider="openai",
            model="gpt-test",
            deployment=None,
            attempt_type="primary",
            streaming=False,
            usage=ProviderTokenUsage(input_tokens=self.input_tokens, output_tokens=0),
            duration_ms=5,
            status="succeeded",
        )

    def generate(self, **_: object) -> str:
        self.calls += 1
        self._record()
        return self.answer

    def generate_stream(self, **_: object) -> Iterator[str]:
        self.calls += 1
        self._record()
        yield self.answer


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
                }
            ],
            "infra": {
                "doubt": {"fixed_cost": "0.001"},
                "quick_practice": {"fixed_cost": "0.003"},
                "mock_test": {"fixed_cost": "0.006"},
            },
            "credits": {"usd_per_credit": "0.001", "pricing_factor": "2"},
        }
    )


@pytest.fixture(autouse=True)
def _stream_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ANSWER_DELIVERY_POLICY", "always_verified")
    monkeypatch.setenv("ANSWER_VERIFIER_ENABLED", "false")
    monkeypatch.setattr(
        streaming_module,
        "_orchestrated_collect_context_node",
        lambda state, **_: {
            "context_text": "",
            "retrieval_context": {"mode": "fresh_solve", "confidence": 0.0},
        },
    )
    monkeypatch.setattr("services.llm.billing.load_billing_config", _billing_config)
    cfg_module._settings = None
    yield
    cfg_module._settings = None


def _runtime(client: FakeDynamoDBClient, *, dry_run: bool = False) -> StudentCreditRuntime:
    return StudentCreditRuntime(
        policy=StudentCreditPolicy(
            enforcement_enabled=True,
            dry_run=dry_run,
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


def _events(
    *,
    student_credits: StudentCreditRuntime | None,
    adapter: Any = None,
) -> list[DoubtSolverStreamEvent]:
    return list(
        stream_doubt_solver(
            StreamDoubtSolverInput(
                request_id=_REQUEST_ID,
                actor_id=_ACTOR_ID,
                turn_id=_TURN_ID,
                query="What is 20 percent of 100?",
                classification=dict(_CLASSIFICATION),
                classifier_confidence=0.5,
            ),
            adapter=adapter or _Adapter(),  # type: ignore[arg-type]
            student_credits=student_credits,
        )
    )


def test_enforcement_off_leaves_the_stream_untouched() -> None:
    """The plug-out guarantee: no wallet call, identical terminal contract."""
    events = _events(student_credits=None)

    assert events[-1].type == "complete"
    assert events[-1].response is not None
    assert events[-1].response.answer == _ANSWER


def test_successful_answer_settles_before_the_terminal_event() -> None:
    client = FakeDynamoDBClient(wallets={_ACTOR_ID: 500})

    events = _events(student_credits=_runtime(client))

    assert events[-1].type == "complete"
    assert events[-1].response is not None
    assert events[-1].response.answer == _ANSWER
    # The student authorized five credits; the measured overage is platform cost.
    assert client.wallets[_ACTOR_ID] == 495
    reference = doubt_reference_id(user_id=_ACTOR_ID, turn_id=_TURN_ID)
    assert client.ledger[reference]["amount"]["N"] == "5"
    assert client.ledger[reference]["authorizationState"]["S"] == "SETTLED"
    assert client.ledger[reference]["userId"]["S"] == _ACTOR_ID


def test_settlement_uses_the_turn_identity_not_the_request_id() -> None:
    client = FakeDynamoDBClient(wallets={_ACTOR_ID: 500})

    _events(student_credits=_runtime(client))

    assert list(client.ledger) == [
        doubt_reference_id(user_id=_ACTOR_ID, turn_id=_TURN_ID)
    ]
    assert _REQUEST_ID not in next(iter(client.ledger))


def test_retrying_a_completed_turn_cannot_start_paid_work_again() -> None:
    client = FakeDynamoDBClient(wallets={_ACTOR_ID: 500})
    runtime = _runtime(client)

    _events(student_credits=runtime)
    retry_adapter = _Adapter()
    events = _events(student_credits=runtime, adapter=retry_adapter)

    assert events[-1].type == "error"
    assert retry_adapter.calls == 0
    assert client.wallets[_ACTOR_ID] == 495
    assert len(client.ledger) == 1


def test_insufficient_balance_replaces_the_terminal_success_with_a_typed_error() -> None:
    client = FakeDynamoDBClient(wallets={_ACTOR_ID: 4})

    events = _events(student_credits=_runtime(client))

    assert events[-1].type == "error"
    assert events[-1].metadata == {
        "retryable": False,
        "code": "INSUFFICIENT_CREDITS",
        "action": "ADD_CREDITS",
        "requiredCredits": 5,
    }
    assert events[-1].label == "Not enough credits to continue. Add credits and try again."
    assert not any(event.type == "complete" for event in events)
    assert client.wallets[_ACTOR_ID] == 4
    assert client.ledger == {}


def test_unpriced_provider_call_never_debits_and_never_retracts_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnpricedAdapter(_Adapter):
        def _record(self) -> None:
            from observability.llm_usage import record_llm_call
            from schemas.llm_usage import ProviderTokenUsage

            record_llm_call(
                request_id=_REQUEST_ID,
                role="math.generator",
                provider="azure_openai",
                model="gpt-4.1-mini",
                deployment="gpt-4.1-mini",
                attempt_type="primary",
                streaming=False,
                usage=ProviderTokenUsage(input_tokens=1_000, output_tokens=500),
                duration_ms=5,
                status="succeeded",
            )

    client = FakeDynamoDBClient(wallets={_ACTOR_ID: 500})

    events = _events(student_credits=_runtime(client), adapter=UnpricedAdapter())

    assert events[-1].type == "complete"
    assert client.wallets[_ACTOR_ID] == 500
    reference = doubt_reference_id(user_id=_ACTOR_ID, turn_id=_TURN_ID)
    assert client.ledger[reference]["authorizationState"]["S"] == "RELEASED"
    assert client.transaction_calls == 2


def test_credit_store_outage_blocks_before_generation() -> None:
    class BrokenClient(FakeDynamoDBClient):
        def transact_write_items(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("dynamodb unavailable")

    client = BrokenClient(wallets={_ACTOR_ID: 500})

    events = _events(student_credits=_runtime(client))

    assert events[-1].type == "error"
    assert client.wallets[_ACTOR_ID] == 500


def test_dry_run_delivers_the_answer_without_moving_the_wallet() -> None:
    client = FakeDynamoDBClient(wallets={_ACTOR_ID: 500})

    events = _events(student_credits=_runtime(client, dry_run=True))

    assert events[-1].type == "complete"
    assert client.wallets[_ACTOR_ID] == 500
    assert client.ledger == {}
    assert client.transaction_calls == 0


def test_admission_refusal_emits_one_terminal_sse_error() -> None:
    """The out-of-credits path must still produce a valid single-terminal stream."""
    import asyncio

    import main

    response = main._student_credit_refusal(
        request_id=_REQUEST_ID,
        reason_code="INSUFFICIENT_CREDITS",
        stream=True,
    )

    async def drain() -> str:
        return b"".join([chunk async for chunk in response.body_iterator]).decode()

    body = asyncio.run(drain())
    payloads = [
        line.removeprefix("data: ")
        for line in body.splitlines()
        if line.startswith("data: ")
    ]

    assert response.media_type == "text/event-stream"
    assert len(payloads) == 1
    assert '"code": "INSUFFICIENT_CREDITS"' in payloads[0].replace('":', '": ')
    assert '"type": "error"' in payloads[0].replace('":', '": ')


def test_admission_refusal_non_stream_shape_matches_existing_errors() -> None:
    import main

    result = main._student_credit_refusal(
        request_id=_REQUEST_ID,
        reason_code="INSUFFICIENT_CREDITS",
        stream=False,
    )

    assert result == {
        "success": False,
        "request_id": _REQUEST_ID,
        "mode": "doubt_solver",
        "error": "INSUFFICIENT_CREDITS",
        "code": "INSUFFICIENT_CREDITS",
        "retryable": False,
    }


def test_below_minimum_wallet_is_refused_before_any_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission is the first chargeable gate: nothing paid may run once it refuses."""
    import main
    import observability.llm_usage as llm_usage

    provider_calls: list[str] = []
    real_record = llm_usage.record_llm_call

    def _spy(**kwargs: Any):
        provider_calls.append(str(kwargs.get("role")))
        return real_record(**kwargs)

    monkeypatch.setattr(llm_usage, "record_llm_call", _spy)
    client = FakeDynamoDBClient(wallets={_ACTOR_ID: 4})
    monkeypatch.setattr(
        main,
        "student_credit_runtime",
        StudentCreditRuntime(
            policy=_policy(),
            repository=StudentCreditRepository(
                user_credits_table=USER_CREDITS_TABLE,
                credit_ledger_table=CREDIT_LEDGER_TABLE,
                client=client,
            ),
        ),
    )

    result = main.invoke(
        {
            "mode": "doubt_solver",
            "query": "What is 20% of 50?",
            "user_id": _ACTOR_ID,
            "conversation_id": "credit-admission-conv",
            "turn_id": "credit-admission-turn",
            "stream": False,
        }
    )

    assert result["error"] == "INSUFFICIENT_CREDITS"
    assert provider_calls == []
    assert client.transaction_calls == 0
    assert client.wallets[_ACTOR_ID] == 4
