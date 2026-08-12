"""Focused parent-state integration tests for the AppSync progress repository."""

from __future__ import annotations

from copy import deepcopy

import pytest

from features.practice_generation.appsync_progress_client import (
    PracticeProgressError,
    PracticeProgressResult,
)
from features.practice_generation.progress import AppSyncAssessmentProgressRepository
from features.practice_generation.progress_contract import (
    PRACTICE_PROGRESS_ALLOWED_META_KEYS,
    PRACTICE_PROGRESS_CONTRACT_VERSION,
)
from features.practice_generation.repositories import PracticeRepositoryError


class Assessments:
    def __init__(self) -> None:
        self.get_calls = 0
        self.item = {
            "testId": "test-1",
            "userId": "user-1",
            "status": "GENERATING",
            "live": False,
            "totalQuestions": 2,
            "updatedAt": "before",
            "meta": {
                "acceptedCount": 2,
                "examProfileId": "cat_management_pre",
                "failedCount": 0,
                "playable": False,
                "progressPercent": 0,
                "readyQuestionCount": 0,
                "readyQuestionIds": [],
                "readyCount": 0,
                "reusedCount": 0,
                "generatedCount": 0,
                "verifiedCount": 0,
                "bucketReadyCounts": {},
                "practiceRequest": {"examProfileId": "cat_management_pre"},
            },
        }
        self.failed_codes: list[str] = []

    def get(self, _test_id: str):
        self.get_calls += 1
        return deepcopy(self.item)

    def mark_failed(self, _test_id: str, error_code: str) -> None:
        self.failed_codes.append(error_code)


class Questions:
    def __init__(self) -> None:
        self.items = {
            "q-1": {
                "questionId": "q-1",
                "testId": "test-1",
                "_practiceMeta": {
                    "bucketId": "bucket-1",
                    "source": "GENERATED",
                },
            }
        }
        self.list_calls = 0
        self.batch_calls = 0

    def list_linked(self, _test_id: str):
        self.list_calls += 1
        return deepcopy(list(self.items.values()))

    def get_questions_by_ids(self, ids):
        self.batch_calls += 1
        return deepcopy([self.items[value] for value in ids if value in self.items])


class Client:
    def __init__(self, *, error: str | None = None) -> None:
        self.calls = []
        self.error = error

    async def update_progress(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if self.error:
            raise PracticeProgressError(self.error)
        return PracticeProgressResult.model_validate(
            {
                "testId": kwargs["test_id"],
                "userId": kwargs["user_id"],
                "status": kwargs["status"],
                "totalQuestions": 2,
                "live": kwargs["live"],
                "meta": {
                    "readyCount": kwargs["meta"]["readyCount"],
                    "progressPercent": kwargs["meta"]["progressPercent"],
                    "playable": kwargs["meta"]["playable"],
                },
                "updatedAt": "after",
            }
        )


def repository(*, error: str | None = None):
    assessments = Assessments()
    questions = Questions()
    client = Client(error=error)
    return (
        AppSyncAssessmentProgressRepository(
            assessments=assessments,
            questions=questions,
            client=client,
        ),
        assessments,
        questions,
        client,
    )


def test_progress_recalculates_persisted_questions_before_single_appsync_write() -> None:
    progress, _assessments, questions, client = repository()

    progress.update(
        "test-1",
        meta_updates={"phase": "GENERATING"},
        live=False,
        recalculate_manifest=True,
        authoritative_question_ids=("q-1",),
    )

    assert questions.list_calls == 1
    assert len(client.calls) == 1
    sent = client.calls[0]
    assert sent["status"] == "GENERATING"
    assert sent["live"] is False
    assert sent["expected_updated_at"] == "before"
    assert sent["meta"]["readyQuestionIds"] == ["q-1"]
    assert sent["meta"]["readyCount"] == 1
    assert sent["meta"]["progressPercent"] > 0
    assert "examProfileId" not in sent["meta"]


def test_legacy_exam_profile_is_filtered_without_losing_approved_recovery_metadata(caplog) -> None:
    progress, assessments, _questions, client = repository()
    secret = "cat_management_pre"

    progress.update(
        "test-1",
        meta_updates={"phase": "GENERATING", "lastCompletedStage": "SLOT_REUSE_MATCHED"},
        live=False,
    )

    sent = client.calls[0]["meta"]
    assert "examProfileId" not in sent
    assert assessments.item["meta"]["examProfileId"] == secret
    assert sent["practiceRequest"]["examProfileId"] == secret
    assert set(sent).issubset(PRACTICE_PROGRESS_ALLOWED_META_KEYS)
    assert "unknown_keys=('examProfileId',)" in caplog.text
    assert secret not in caplog.text
    assert PRACTICE_PROGRESS_CONTRACT_VERSION in caplog.text


def test_progress_strips_legacy_raw_query_summary_before_appsync_publish() -> None:
    progress, assessments, _questions, client = repository()
    raw_query = "My answer is secret and must not be published"
    assessments.item["meta"]["practiceRequest"] = {"querySummary": raw_query}

    progress.update(
        "test-1",
        meta_updates={"phase": "GENERATING"},
        live=False,
    )

    sent_request = client.calls[0]["meta"]["practiceRequest"]
    assert "querySummary" not in sent_request
    assert raw_query not in str(client.calls[0]["meta"])


def test_recovery_claim_is_single_cas_and_resets_only_running_groups() -> None:
    progress, assessments, _questions, client = repository()
    assessments.item["meta"].update(
        {
            "recoveryAttemptCount": 0,
            "generationGroups": {
                "running": {"state": "RUNNING"},
                "complete": {"state": "COMPLETED"},
            },
        }
    )

    assert progress.claim_recovery("test-1") is True
    sent = client.calls[0]
    assert sent["expected_updated_at"] == "before"
    assert sent["meta"]["recoveryAttemptCount"] == 1
    assert sent["meta"]["generationGroups"]["running"]["state"] == "PENDING"
    assert sent["meta"]["generationGroups"]["complete"]["state"] == "COMPLETED"


def test_recovery_claim_loser_does_not_execute_after_cas_conflict() -> None:
    progress, assessments, _questions, client = repository(
        error="PRACTICE_PROGRESS_CONFLICT"
    )
    assessments.item["meta"].update(
        {
            "recoveryAttemptCount": 0,
            "generationGroups": {"running": {"state": "RUNNING"}},
        }
    )

    assert progress.claim_recovery("test-1") is False
    assert len(client.calls) == 1


def test_generating_progress_cannot_decrease_or_become_playable() -> None:
    progress, assessments, _questions, client = repository()
    assessments.item["meta"].update(
        {
            "readyQuestionIds": ["q-1"],
            "readyCount": 1,
            "readyQuestionCount": 1,
            "progressPercent": 50,
        }
    )

    with pytest.raises(PracticeRepositoryError, match="PERCENT_DECREASED"):
        progress.update(
            "test-1",
            meta_updates={"progressPercent": 40},
            live=False,
        )
    with pytest.raises(PracticeRepositoryError, match="INVALID_GENERATING_STATE"):
        progress.update(
            "test-1",
            meta_updates={"progressPercent": 50, "playable": True},
            live=False,
        )
    assert client.calls == []


def test_conditional_conflict_recalculates_and_returns_controlled_failure() -> None:
    progress, assessments, questions, client = repository(error="PRACTICE_PROGRESS_CONFLICT")

    with pytest.raises(PracticeRepositoryError, match="ASSESSMENT_CONCURRENT_UPDATE"):
        progress.update(
            "test-1",
            meta_updates={"phase": "GENERATING"},
            live=False,
            recalculate_manifest=True,
        )

    assert len(client.calls) == 1
    assert assessments.get_calls >= 2
    assert questions.list_calls >= 2


def test_failed_progress_publication_uses_existing_conditional_failure_write() -> None:
    progress, assessments, _questions, client = repository(
        error="PRACTICE_PROGRESS_TRANSPORT_FAILED"
    )

    progress.mark_failed("test-1", "PRACTICE_GENERATION_FAILED")

    assert len(client.calls) == 1
    assert assessments.failed_codes == ["PRACTICE_GENERATION_FAILED"]


def test_invalid_progress_contract_is_fatal_and_never_uses_direct_failure_fallback() -> None:
    progress, assessments, _questions, client = repository()

    with pytest.raises(PracticeRepositoryError, match="PRACTICE_PROGRESS_UNKNOWN_META_FIELD"):
        progress.mark_failed(
            "test-1",
            "PRACTICE_GENERATION_FAILED",
            meta_updates={"providerResponse": "must-not-cross-the-contract"},
        )

    assert client.calls == []
    assert assessments.failed_codes == []


@pytest.mark.parametrize(
    ("case", "status", "live", "meta_updates"),
    [
        (
            "schema_v1_pattern_off_generating",
            "GENERATING",
            False,
            {"schemaVersion": "1", "phase": "MATCHING_EXISTING"},
        ),
        (
            "schema_v2_pattern_off_generating",
            "GENERATING",
            False,
            {
                "schemaVersion": "2",
                "phase": "GENERATING",
                "blueprint": {"schema_version": "2", "slots": []},
            },
        ),
        (
            "schema_v2_pattern_on_generating",
            "GENERATING",
            False,
            {
                "schemaVersion": "2",
                "phase": "GENERATING",
                "generationGroups": {
                    "group-1": {"state": "PENDING", "patternSelection": "guidance"}
                },
            },
        ),
        (
            "ready",
            "READY",
            True,
            {
                "acceptedCount": 2,
                "phase": "READY",
                "playable": True,
                "progressPercent": 100,
                "questionManifestVersion": 1,
                "readyCount": 2,
                "readyQuestionCount": 2,
                "readyQuestionIds": ["q-1", "q-2"],
                "verifiedCount": 2,
            },
        ),
        (
            "failed",
            "FAILED",
            False,
            {
                "phase": "FAILED",
                "playable": False,
                "errorCode": "PRACTICE_GENERATION_FAILED",
            },
        ),
    ],
)
def test_every_progress_publish_projects_to_the_backend_allowlist(
    case: str,
    status: str,
    live: bool,
    meta_updates: dict[str, object],
) -> None:
    del case
    progress, _assessments, _questions, client = repository()

    progress.update(
        "test-1",
        meta_updates=meta_updates,
        status=status,
        live=live,
    )

    assert set(client.calls[0]["meta"]).issubset(PRACTICE_PROGRESS_ALLOWED_META_KEYS)


def test_python_contract_matches_the_deployed_backend_allowlist_fixture() -> None:
    expected_backend_keys = {
        "schemaVersion",
        "idempotencyKey",
        "conversationId",
        "turnId",
        "practiceType",
        "requestedCount",
        "acceptedCount",
        "examStage",
        "requestedLanguage",
        "phase",
        "playable",
        "progressPercent",
        "readyQuestionCount",
        "questionManifestVersion",
        "readyQuestionIds",
        "readyCount",
        "reusedCount",
        "generatedCount",
        "verifiedCount",
        "failedCount",
        "practiceRequest",
        "resourceAliases",
        "blueprint",
        "plannerCalls",
        "plannerTier",
        "plannerRepaired",
        "plannerDeterministicFallback",
        "bucketReadyCounts",
        "deficits",
        "generationGroups",
        "finalizationAttempt",
        "finalizationReasonCode",
        "readyAt",
        "errorCode",
        "generationVersion",
        "startedAt",
        "lastProgressAt",
        "recoveryAttemptCount",
        "lastCompletedStage",
        "replacementWaveCount",
        "slotReadyCounts",
    }

    assert PRACTICE_PROGRESS_ALLOWED_META_KEYS == expected_backend_keys
