"""Focused tests for the IAM-authenticated AppSync progress transport."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import httpx
import pytest
from botocore.credentials import ReadOnlyCredentials

from features.practice_generation.appsync_progress_client import (
    AppSyncPracticeProgressClient,
    PracticeProgressError,
)
from features.practice_generation.config import PracticeConfigurationError, get_practice_config

_ENDPOINT = "https://example.appsync-api.ap-south-1.amazonaws.com/graphql"


class Credentials:
    def get_frozen_credentials(self):
        return ReadOnlyCredentials("temporary-key", "temporary-secret", "temporary-token")


class Session:
    def __init__(self, credentials=Credentials()) -> None:
        self.credentials = credentials
        self.calls = 0

    def get_credentials(self):
        self.calls += 1
        return self.credentials


def response_payload(*, status: str = "GENERATING") -> dict:
    return {
        "data": {
            "updatePracticeGenerationProgress": {
                "testId": "test-1",
                "userId": "user-1",
                "status": status,
                "totalQuestions": 5,
                "live": status == "READY",
                "meta": json.dumps(
                    {
                        "readyCount": 1,
                        "progressPercent": 20,
                        "playable": status == "READY",
                    }
                ),
                "updatedAt": "2026-08-02T00:00:00Z",
            }
        }
    }


def client(handler, *, session=None, max_retries=2):
    return AppSyncPracticeProgressClient(
        endpoint=_ENDPOINT,
        region="ap-south-1",
        session=session or Session(),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=max_retries,
        sleep=lambda _delay: _no_sleep(),
    )


async def _no_sleep() -> None:
    return None


def test_sigv4_uses_appsync_region_and_exact_graphql_variables() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["authorization"]
        captured["token"] = request.headers["x-amz-security-token"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=response_payload())

    session = Session()
    result = asyncio.run(
        client(handler, session=session).update_progress(
            test_id="test-1",
            user_id="user-1",
            status="GENERATING",
            meta={"readyCount": 1, "progressPercent": 20, "playable": False},
            live=False,
            expected_updated_at="before",
        )
    )

    assert "/ap-south-1/appsync/aws4_request" in captured["authorization"]
    assert captured["token"] == "temporary-token"
    assert captured["payload"]["variables"] == {
        "testId": "test-1",
        "userId": "user-1",
        "status": "GENERATING",
        "meta": '{"playable":false,"progressPercent":20,"readyCount":1}',
        "live": False,
        "expectedUpdatedAt": "before",
    }
    assert session.calls == 1
    assert result.test_id == "test-1"
    assert result.meta["readyCount"] == 1


def test_dynamodb_decimal_metadata_is_serialized_as_json_numbers() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=response_payload())

    asyncio.run(
        client(handler).update_progress(
            test_id="test-1",
            user_id="user-1",
            status="GENERATING",
            meta={
                "readyCount": Decimal("1"),
                "progressPercent": Decimal("20.5"),
                "practiceRequest": {"acceptedCount": Decimal("5")},
            },
            live=False,
            expected_updated_at="before",
        )
    )

    assert json.loads(captured["payload"]["variables"]["meta"]) == {
        "practiceRequest": {"acceptedCount": 5},
        "progressPercent": 20.5,
        "readyCount": 1,
    }


def test_unknown_meta_key_is_rejected_before_the_signed_transport() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=response_payload())

    with pytest.raises(PracticeProgressError, match="PRACTICE_PROGRESS_UNKNOWN_META_FIELD"):
        asyncio.run(
            client(handler).update_progress(
                test_id="test-1",
                user_id="user-1",
                status="GENERATING",
                meta={"plannerValidationReasonCode": "PLANNER_SCHEMA_INVALID"},
                live=False,
                expected_updated_at="before",
            )
        )

    assert calls == 0


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_http_status_is_bounded(status_code: int) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(status_code)
        return httpx.Response(200, json=response_payload())

    asyncio.run(
        client(handler).update_progress(
            test_id="test-1",
            user_id="user-1",
            status="GENERATING",
            meta={"readyCount": 1},
            live=False,
            expected_updated_at="before",
        )
    )
    assert calls == 3


def test_timeout_retries_the_exact_compare_and_swap_payload() -> None:
    calls = 0
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payloads.append(json.loads(request.content)["variables"]["meta"])
        if calls == 1:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, json=response_payload())

    asyncio.run(
        client(handler).update_progress(
            test_id="test-1",
            user_id="user-1",
            status="GENERATING",
            meta={"readyCount": 0},
            live=False,
            expected_updated_at="before",
        )
    )
    assert payloads == ['{"readyCount":0}', '{"readyCount":0}']


@pytest.mark.parametrize(
    "code",
    ["PRACTICE_PROGRESS_UNAUTHORIZED", "PRACTICE_PROGRESS_CONFLICT"],
)
def test_graphql_authorization_and_conflict_are_not_retried(code: str) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"errors": [{"message": code}]})

    with pytest.raises(PracticeProgressError, match=code) as raised:
        asyncio.run(
            client(handler).update_progress(
                test_id="test-1",
                user_id="user-1",
                status="GENERATING",
                meta={"readyCount": 0},
                live=False,
                expected_updated_at="before",
            )
        )
    assert raised.value.retryable is False
    assert calls == 1


def test_graphql_safe_unknown_field_detail_is_preserved_when_backend_supplies_it() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "errors": [
                    {
                        "message": "PRACTICE_PROGRESS_UNKNOWN_META_FIELD",
                        "extensions": {"unknownField": "examProfileId"},
                    }
                ]
            },
        )

    with pytest.raises(PracticeProgressError) as raised:
        asyncio.run(
            client(handler).update_progress(
                test_id="test-1",
                user_id="user-1",
                status="GENERATING",
                meta={"readyCount": 0},
                live=False,
                expected_updated_at="before",
            )
        )

    assert raised.value.code == "PRACTICE_PROGRESS_UNKNOWN_META_FIELD"
    assert raised.value.safe_detail == "examProfileId"


def test_missing_credentials_are_controlled_and_not_logged(caplog) -> None:
    secret = "must-not-appear"
    progress_client = client(
        lambda _request: httpx.Response(500),
        session=Session(credentials=None),
        max_retries=0,
    )
    with pytest.raises(
        PracticeProgressError,
        match="PRACTICE_PROGRESS_CREDENTIALS_UNAVAILABLE",
    ):
        asyncio.run(
            progress_client.update_progress(
                test_id="test-1",
                user_id="user-1",
                status="GENERATING",
                meta={"practiceRequest": {"querySummary": secret}},
                live=False,
                expected_updated_at="before",
            )
        )
    assert secret not in caplog.text


def test_enabled_practice_requires_appsync_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "true")
    monkeypatch.delenv("APPSYNC_GRAPHQL_ENDPOINT", raising=False)
    with pytest.raises(PracticeConfigurationError, match="APPSYNC_GRAPHQL_ENDPOINT"):
        get_practice_config(resource_contract=_contract()).validate_runtime()


def test_disabled_practice_does_not_require_appsync_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("PRACTICE_GENERATION_ENABLED", "false")
    monkeypatch.delenv("APPSYNC_GRAPHQL_ENDPOINT", raising=False)
    get_practice_config().validate_runtime()


def _contract():
    from features.practice_generation.resource_contract import PracticeResourceContract

    return PracticeResourceContract(
        region_name="ap-south-1",
        schema_version="1",
        reuse_key_contract_version="1",
        assessment_table="assessment",
        assessment_table_arn="arn:assessment",
        question_table="question",
        question_table_arn="arn:question",
        question_test_index="questionsByTest",
        question_bank_table="bank",
        question_bank_table_arn="arn:bank",
        question_bank_category_index="byCategory",
        question_bank_reuse_index="byReuse",
    )


@pytest.mark.parametrize(
    "error_body, expected_type",
    [
        ({"errorType": "DynamoDB:ConditionalCheckFailedException"},
         "DynamoDB:ConditionalCheckFailedException"),
        ({"errorType": "Unauthorized"}, "Unauthorized"),
        ({"errorType": "Lambda:Unhandled"}, "Lambda:Unhandled"),
        ({"errorType": "MappingTemplate",
          "extensions": {"code": "CONFLICT"}}, "MappingTemplate"),
        ({"errorType": "x", "extensions": {"errorType": "BadRequest"}}, "x"),
    ],
)
def test_graphql_rejection_exposes_the_sanitized_appsync_error_type(
    error_body: dict, expected_type: str
) -> None:
    """Q20-6a: the unclassified branch must name the AppSync error class."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errors": [error_body]})

    with pytest.raises(PracticeProgressError) as raised:
        asyncio.run(
            client(handler).update_progress(
                test_id="test-1",
                user_id="user-1",
                status="GENERATING",
                meta={"readyCount": 0},
                live=False,
                expected_updated_at="before",
            )
        )

    assert raised.value.code == "PRACTICE_PROGRESS_GRAPHQL_REJECTED"
    assert raised.value.error_type == expected_type
    assert raised.value.retryable is False


@pytest.mark.parametrize(
    "unsafe",
    [
        "student asked: what is 2+2",
        "Bearer abc.def.ghi",
        'DynamoDB {"question": "who won"}',
        "A" * 200,
        "",
        "9LeadingDigit",
    ],
)
def test_unsafe_graphql_error_type_is_never_exposed(unsafe: str) -> None:
    """Anything that is not a bare classification label is dropped, not logged."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"errors": [{"errorType": unsafe}]})

    with pytest.raises(PracticeProgressError) as raised:
        asyncio.run(
            client(handler).update_progress(
                test_id="test-1",
                user_id="user-1",
                status="GENERATING",
                meta={"readyCount": 0},
                live=False,
                expected_updated_at="before",
            )
        )

    assert raised.value.code == "PRACTICE_PROGRESS_GRAPHQL_REJECTED"
    assert raised.value.error_type is None


def test_graphql_message_is_never_used_as_the_error_type() -> None:
    """The GraphQL message may embed request data; it must not reach the field."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "errors": [
                    {
                        "message": "Conditional check failed for testId practice-abc",
                        "errorType": "DynamoDB:ConditionalCheckFailedException",
                    }
                ]
            },
        )

    with pytest.raises(PracticeProgressError) as raised:
        asyncio.run(
            client(handler).update_progress(
                test_id="test-1",
                user_id="user-1",
                status="GENERATING",
                meta={"readyCount": 0},
                live=False,
                expected_updated_at="before",
            )
        )

    assert raised.value.error_type == "DynamoDB:ConditionalCheckFailedException"
    assert "practice-abc" not in (raised.value.error_type or "")
    assert "Conditional check failed" not in (raised.value.error_type or "")


def test_graphql_scan_order_is_unchanged_by_the_diagnostic() -> None:
    """Behaviour guard: an unsanitizable first errorType still terminates the scan."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "errors": [
                    {"errorType": "has spaces so unsafe"},
                    {"message": "PRACTICE_PROGRESS_OWNER_MISMATCH"},
                ]
            },
        )

    with pytest.raises(PracticeProgressError) as raised:
        asyncio.run(
            client(handler).update_progress(
                test_id="test-1",
                user_id="user-1",
                status="GENERATING",
                meta={"readyCount": 0},
                live=False,
                expected_updated_at="before",
            )
        )

    assert raised.value.code == "PRACTICE_PROGRESS_GRAPHQL_REJECTED"
    assert raised.value.error_type is None
