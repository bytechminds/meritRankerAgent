"""QuestionBank promotion observability and the one-time reuse-key backfill."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

import features.practice_generation.repositories as repositories_module
from features.practice_generation.matching import build_slot_reuse_bucket_key
from features.practice_generation.repositories import PracticeRepositoryError, _item, _plain
from tests.practice_generation.test_practice_repository import (
    RecordingClient,
    _promotion_question,
    _promotion_repository,
    _promotion_slot,
)

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "backfill_question_bank_reuse_keys.py"
_spec = importlib.util.spec_from_file_location("backfill_question_bank_reuse_keys", _SCRIPT)
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)

_TRUST_AND_CONTENT = (
    "qbId", "question", "answers", "correctAnswer", "explanation", "difficulty",
    "category", "meta", "versionHash", "reuseSortKey", "createdAt", "updatedAt",
)


def _conditional_error() -> ClientError:
    return ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")


def _promoted_row() -> dict[str, Any]:
    client = RecordingClient()
    _promotion_repository(client).persist_verified_question(
        test_id="test-1", question=_promotion_question(), slot=_promotion_slot(), language="english"
    )
    return _plain(client.put_calls[0]["Item"])


def _old_key_row() -> dict[str, Any]:
    row = _promoted_row()
    row["reuseBucketKey"] = "v1#linear_equations#algebra#mcq#english"
    return row


# --- promotion outcomes are measurable ---------------------------------------


@pytest.fixture
def promotion_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        repositories_module,
        "emit_practice_event",
        lambda name, **kwargs: events.append({"name": name, **kwargs.get("details", {})})
        if name == "QUESTION_BANK_PROMOTION"
        else None,
    )
    return events


def test_promotion_reports_created_then_already_exists(promotion_events) -> None:  # noqa: ANN001
    class _Client(RecordingClient):
        def put_item(self, **kwargs):
            super().put_item(**kwargs)
            if len(self.put_calls) > 1:
                raise _conditional_error()
            return {}

    repository = _promotion_repository(_Client())
    for _ in range(2):
        repository.persist_verified_question(
            test_id="t", question=_promotion_question(), slot=_promotion_slot(), language="english"
        )

    assert [event["outcome"] for event in promotion_events] == ["CREATED", "ALREADY_EXISTS"]
    assert promotion_events[0]["qbId"] == promotion_events[1]["qbId"]
    assert promotion_events[0]["reuseBucketKey"] == "v1#math#algebra#mcq#english"


def test_promotion_failure_is_reported_and_still_raised(promotion_events) -> None:  # noqa: ANN001
    class _Client(RecordingClient):
        def put_item(self, **kwargs):
            raise ClientError({"Error": {"Code": "ProvisionedThroughputExceeded"}}, "PutItem")

    with pytest.raises(PracticeRepositoryError):
        _promotion_repository(_Client()).persist_verified_question(
            test_id="t", question=_promotion_question(), slot=_promotion_slot(), language="english"
        )

    assert [event["outcome"] for event in promotion_events] == ["FAILED"]


def test_promotion_without_a_derivable_key_is_reported_as_skipped(promotion_events) -> None:  # noqa: ANN001
    slot = _promotion_slot().model_copy(update={"subject_id": "unknown_family"})

    qb_id = _promotion_repository(RecordingClient()).persist_verified_question(
        test_id="t", question=_promotion_question(), slot=slot, language="english"
    )

    assert qb_id is None
    assert [event["outcome"] for event in promotion_events] == ["SKIPPED"]


# --- one-time reuse-key backfill ---------------------------------------------


def test_old_key_row_migrates_to_the_current_slot_key() -> None:
    decision, current = backfill.plan_row(_old_key_row())

    assert decision == "MIGRATE"
    assert current == build_slot_reuse_bucket_key(_promotion_slot(), language="english")


def test_current_row_is_left_alone() -> None:
    assert backfill.plan_row(_promoted_row())[0] == "ALREADY_CURRENT"


@pytest.mark.parametrize(
    ("mutate", "decision"),
    [
        (lambda row: row.update(explanation=""), "SKIP_NO_SOLUTION"),
        (lambda row: row["meta"].pop("topic"), "SKIP_UNDERIVABLE"),
        (lambda row: row.update(reuseSortKey="v1#hard#x#y"), "SKIP_SORT_KEY_INCONSISTENT"),
        (lambda row: row["meta"].update(status="INACTIVE"), "SKIP_NOT_REUSABLE"),
    ],
)
def test_rows_that_cannot_be_safely_reindexed_fail_closed(mutate, decision) -> None:  # noqa: ANN001
    row = _old_key_row()
    mutate(row)

    assert backfill.plan_row(row)[0] == decision


class _TableClient:
    """A one-table DynamoDB stand-in that enforces the backfill's condition."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = {row["qbId"]: deepcopy(row) for row in rows}
        self.updates: list[dict[str, Any]] = []

    def scan(self, **_kwargs):
        return {"Items": [_item(row) for row in self.rows.values()]}

    def update_item(self, **kwargs):
        self.updates.append(kwargs)
        row = self.rows[kwargs["Key"]["qbId"]["S"]]
        if row["reuseBucketKey"] != kwargs["ExpressionAttributeValues"][":old"]["S"]:
            raise _conditional_error()
        row["reuseBucketKey"] = kwargs["ExpressionAttributeValues"][":current"]["S"]
        return {}


def test_backfill_rewrites_only_the_bucket_key_and_is_idempotent() -> None:
    old = _old_key_row()
    table = _TableClient([old])

    first = backfill.run(table, table="qb", apply=True)
    second = backfill.run(table, table="qb", apply=True)

    assert first["MIGRATED"] == 1
    assert second == {"ALREADY_CURRENT": 1}
    assert len(table.updates) == 1
    update = table.updates[0]
    assert update["UpdateExpression"] == "SET #key = :current"
    assert update["ExpressionAttributeNames"] == {"#key": "reuseBucketKey"}
    migrated = table.rows[old["qbId"]]
    assert migrated["reuseBucketKey"] == "v1#math#algebra#mcq#english"
    for field in _TRUST_AND_CONTENT:
        assert migrated.get(field) == old.get(field)


def test_backfill_dry_run_writes_nothing() -> None:
    table = _TableClient([_old_key_row()])

    counts = backfill.run(table, table="qb", apply=False)

    assert counts == {"MIGRATE": 1}
    assert table.updates == []


def test_backfill_never_overwrites_a_concurrently_changed_key() -> None:
    old = _old_key_row()
    table = _TableClient([old])
    table.rows[old["qbId"]]["reuseBucketKey"] = "v1#something_else#algebra#mcq#english"
    stale = dict(old)

    assert backfill.migrate_row(
        table, table="qb", item=stale, current_key="v1#math#algebra#mcq#english"
    ) is False
    assert table.rows[old["qbId"]]["reuseBucketKey"] == "v1#something_else#algebra#mcq#english"


def test_successful_promotion_is_not_logged_as_a_warning() -> None:
    import logging

    from features.practice_generation.events import _default_practice_event_level

    assert _default_practice_event_level("QUESTION_BANK_PROMOTION") == logging.DEBUG
