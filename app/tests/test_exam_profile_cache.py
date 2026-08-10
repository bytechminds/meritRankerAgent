from __future__ import annotations

import time

import pytest

from services.doubt_solver.exam_profile_cache import (
    DynamoExamProfileSource,
    ExamProfile,
    ExamProfileRuntime,
)


def _profile(index: int = 1, *, active: bool = True) -> ExamProfile:
    exam_id = f"TEST_{index}"
    return ExamProfile.model_validate(
        {
            "examProfileId": f"{exam_id}#DEFAULT",
            "examId": exam_id,
            "examName": f"Test {index}",
            "stage": "DEFAULT",
            "description": "Compact deterministic test exam context.",
            "sections": [
                {
                    "sectionId": "MATH",
                    "name": "Quantitative Aptitude",
                    "subject": "MATH",
                    "level": "INTERMEDIATE",
                    "questionCount": 10,
                    "marks": 10,
                    "timeMinutes": 10,
                    "questionStyles": ["mcq"],
                    "negativeMarks": 0.25,
                    "excludeTopics": ["calculus"],
                }
            ],
            "totalQuestions": 10,
            "totalMarks": 10,
            "totalTimeMinutes": 10,
            "active": active,
        }
    )


class _Source:
    def __init__(self, profiles: tuple[ExamProfile, ...] = (), fail: bool = False) -> None:
        self.profiles = profiles
        self.fail = fail
        self.calls = 0

    def load_active_profiles(self) -> tuple[ExamProfile, ...]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("DYNAMODB_UNAVAILABLE")
        return self.profiles


class _SsmClient:
    def get_parameters(self, **_: object) -> dict[str, object]:
        return {
            "Parameters": [
                {
                    "Name": "/meritranker/agent-runtime/v1/exam-profile/table-name",
                    "Value": "ExamProfileTable",
                },
                {
                    "Name": "/meritranker/agent-runtime/v1/exam-profile/table-arn",
                    "Value": "arn:aws:dynamodb:ap-south-1:123456789012:table/ExamProfileTable",
                },
            ]
        }


class _DynamoClient:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[dict[str, object]] = []

    def scan(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        self.requests.append(kwargs)
        if self.calls == 1:
            return {
                "Items": [
                    {
                        "examProfileId": {"S": "TEST_1#DEFAULT"},
                        "examId": {"S": "TEST_1"},
                        "examName": {"S": "Test 1"},
                        "stage": {"S": "DEFAULT"},
                        "description": {"S": "test"},
                        "sections": {
                            "L": [
                                {
                                    "M": {
                                        "sectionId": {"S": "MATH"},
                                        "name": {"S": "Math"},
                                        "subject": {"S": "MATH"},
                                        "level": {"S": "INTERMEDIATE"},
                                        "questionCount": {"N": "10"},
                                        "marks": {"N": "10"},
                                        "questionStyles": {"L": [{"S": "mcq"}]},
                                        "excludeTopics": {"L": []},
                                    }
                                }
                            ]
                        },
                        "totalQuestions": {"N": "10"},
                        "totalMarks": {"N": "10"},
                        "totalTimeMinutes": {"N": "10"},
                        "active": {"BOOL": True},
                    },
                    {"examProfileId": {"S": "malformed"}},
                ],
                "LastEvaluatedKey": {"examProfileId": {"S": "next"}},
            }
        return {"Items": []}


def test_startup_load_builds_exact_and_compatibility_indexes() -> None:
    source = _Source((_profile(),))
    runtime = ExamProfileRuntime(source)

    assert runtime.load() is True
    exact = runtime.resolve(
        exam_profile_id="TEST_1#DEFAULT",
        exam_id=None,
        exam_stage=None,
        subject="math",
    )
    compatibility = runtime.resolve(
        exam_profile_id=None,
        exam_id="test 1",
        exam_stage="default",
        subject="MATH",
    )

    assert exact.source == "dynamodb_cache"
    assert exact.context is not None
    assert exact.context.exam_profile_id == "TEST_1#DEFAULT"
    assert compatibility.cache_hit is True
    assert source.calls == 1


def test_dynamo_source_discovers_the_table_and_quarantines_malformed_items() -> None:
    dynamodb = _DynamoClient()
    source = DynamoExamProfileSource(
        region_name="ap-south-1",
        ssm_client=_SsmClient(),
        dynamodb_client=dynamodb,
    )

    profiles = source.load_active_profiles()

    assert [profile.exam_profile_id for profile in profiles] == ["TEST_1#DEFAULT"]
    assert dynamodb.calls == 2
    assert dynamodb.requests[0]["TableName"] == "ExamProfileTable"
    assert dynamodb.requests[0]["FilterExpression"] == "#active = :active"
    assert "previousCutoff" not in str(dynamodb.requests[0]["ProjectionExpression"])


def test_unknown_profile_uses_legacy_fallback_without_source_read() -> None:
    source = _Source((_profile(),))
    runtime = ExamProfileRuntime(source)
    assert runtime.load() is True

    resolution = runtime.resolve(
        exam_profile_id="MISSING#DEFAULT",
        exam_id=None,
        exam_stage=None,
    )

    assert resolution.context is None
    assert resolution.source == "legacy_fallback"
    assert resolution.reason == "PROFILE_NOT_FOUND"
    assert source.calls == 1


def test_failed_refresh_keeps_last_known_good_snapshot() -> None:
    source = _Source((_profile(),))
    runtime = ExamProfileRuntime(source)
    assert runtime.load() is True
    source.fail = True

    assert runtime.load() is False
    assert runtime.resolve(
        exam_profile_id="TEST_1#DEFAULT",
        exam_id=None,
        exam_stage=None,
    ).cache_hit is True


def test_empty_cache_and_malformed_candidate_do_not_crash_requests() -> None:
    runtime = ExamProfileRuntime(_Source(()))
    assert runtime.load() is True
    assert runtime.snapshot is not None
    assert runtime.snapshot.profile_count == 0
    resolution = runtime.resolve(
        exam_profile_id=None,
        exam_id="CAT",
        exam_stage="DEFAULT",
    )
    assert resolution.source == "legacy_fallback"


def test_inactive_profile_is_not_resolved_even_if_a_source_returns_it() -> None:
    runtime = ExamProfileRuntime(_Source((_profile(active=False),)))
    assert runtime.load() is True

    resolution = runtime.resolve(
        exam_profile_id="TEST_1#DEFAULT",
        exam_id=None,
        exam_stage=None,
    )

    assert resolution.source == "legacy_fallback"
    assert resolution.reason == "PROFILE_NOT_FOUND"


def test_normal_context_excludes_mock_totals_and_cutoffs() -> None:
    runtime = ExamProfileRuntime(_Source((_profile(),)))
    assert runtime.load() is True

    normal = runtime.resolve(
        exam_profile_id="TEST_1#DEFAULT",
        exam_id=None,
        exam_stage=None,
        subject="MATH",
    ).context
    mock = runtime.resolve(
        exam_profile_id="TEST_1#DEFAULT",
        exam_id=None,
        exam_stage=None,
        subject="MATH",
        full_mock=True,
    ).context

    assert normal is not None and mock is not None
    assert normal.totals is None
    assert "questionCount" not in normal.sections[0]
    assert mock.totals is not None
    assert "questionCount" in mock.sections[0]
    assert "Cutoff" not in normal.as_prompt_instruction()


@pytest.mark.parametrize(
    ("profile_count", "lookups"),
    ((10, 1_000), (25, 1_000), (50, 10_000), (100, 10_000)),
)
def test_cache_lookup_stress_has_no_source_calls(
    profile_count: int,
    lookups: int,
) -> None:
    profiles = tuple(_profile(index) for index in range(1, profile_count + 1))
    source = _Source(profiles)
    runtime = ExamProfileRuntime(source)
    assert runtime.load() is True

    started = time.perf_counter()
    for index in range(lookups):
        resolution = runtime.resolve(
            exam_profile_id=f"TEST_{(index % profile_count) + 1}#DEFAULT",
            exam_id=None,
            exam_stage=None,
            subject="MATH",
        )
        assert resolution.cache_hit is True
    duration_seconds = time.perf_counter() - started

    assert source.calls == 1
    assert duration_seconds < 2.0
