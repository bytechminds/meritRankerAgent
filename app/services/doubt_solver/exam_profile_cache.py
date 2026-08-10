"""Small, immutable startup cache for admin-managed exam profiles.

The request path intentionally performs no AWS I/O.  A failed load never
replaces a usable snapshot, allowing the existing bundled exam guidance to
remain the compatibility fallback during migration.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from boto3.dynamodb.types import TypeDeserializer
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from services.aws_client_factory import get_dynamodb_client, get_ssm_client

logger = logging.getLogger(__name__)

_ROOT = "/meritranker/agent-runtime/v1/exam-profile"
_TABLE_NAME_PATH = f"{_ROOT}/table-name"
_TABLE_ARN_PATH = f"{_ROOT}/table-arn"
_DESERIALIZER = TypeDeserializer()
_LEVELS = frozenset({"BASIC", "INTERMEDIATE", "ADVANCED", "MIXED"})


class ExamProfileSection(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    section_id: str = Field(alias="sectionId", min_length=1, max_length=96)
    name: str = Field(min_length=1, max_length=160)
    subject: str = Field(min_length=1, max_length=64)
    level: str
    question_count: int = Field(alias="questionCount", gt=0, le=500)
    marks: float = Field(ge=0)
    time_minutes: int | None = Field(default=None, alias="timeMinutes", ge=0)
    question_styles: tuple[str, ...] = Field(default_factory=tuple, alias="questionStyles")
    negative_marks: float | None = Field(default=None, alias="negativeMarks", ge=0)
    marks_per_question: float | None = Field(default=None, alias="marksPerQuestion", ge=0)
    exclude_topics: tuple[str, ...] = Field(default_factory=tuple, alias="excludeTopics")

    @field_validator("level")
    @classmethod
    def _valid_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in _LEVELS:
            raise ValueError("level must be BASIC, INTERMEDIATE, ADVANCED, or MIXED")
        return normalized

    @field_validator("subject")
    @classmethod
    def _canonical_subject(cls, value: str) -> str:
        return value.strip().upper().replace(" ", "_").replace("-", "_")


class ExamProfile(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    exam_profile_id: str = Field(alias="examProfileId", min_length=3, max_length=160)
    exam_id: str = Field(alias="examId", min_length=1, max_length=128)
    exam_name: str = Field(alias="examName", min_length=1, max_length=160)
    stage: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1000)
    sections: tuple[ExamProfileSection, ...] = Field(min_length=1, max_length=32)
    total_questions: int = Field(alias="totalQuestions", gt=0, le=1000)
    total_marks: float = Field(alias="totalMarks", ge=0)
    total_time_minutes: int = Field(alias="totalTimeMinutes", gt=0, le=2000)
    active: bool
    effective_year: int | None = Field(default=None, alias="effectiveYear", ge=1900, le=3000)

    @model_validator(mode="after")
    def _identity_and_sections_are_consistent(self) -> ExamProfile:
        expected = f"{self.exam_id.strip().upper()}#{self.stage.strip().upper()}"
        if self.exam_profile_id.strip().upper() != expected:
            raise ValueError("examProfileId must equal canonical examId#stage")
        section_ids = [section.section_id.casefold() for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("sectionId values must be unique within a profile")
        if sum(section.question_count for section in self.sections) != self.total_questions:
            raise ValueError("section questionCount total must equal totalQuestions")
        return self


@dataclass(frozen=True)
class AgentExamContext:
    """Prompt-safe projection; it deliberately excludes cutoffs and admin metadata."""

    exam_profile_id: str
    exam_id: str
    exam_name: str
    stage: str
    description: str
    sections: tuple[Mapping[str, object], ...]
    totals: Mapping[str, object] | None

    def as_prompt_instruction(self) -> str:
        payload: dict[str, object] = {
            "examProfileId": self.exam_profile_id,
            "examId": self.exam_id,
            "examName": self.exam_name,
            "stage": self.stage,
            "description": self.description,
            "sections": [dict(section) for section in self.sections],
        }
        if self.totals is not None:
            payload["mockFormat"] = dict(self.totals)
        return "Exam context (follow only when relevant):\n" + json.dumps(
            payload, separators=(",", ":"), ensure_ascii=False
        )

    def as_planner_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "examProfileId": self.exam_profile_id,
            "examId": self.exam_id,
            "examName": self.exam_name,
            "stage": self.stage,
            "description": self.description,
            "sections": [dict(section) for section in self.sections],
        }
        if self.totals is not None:
            payload["mockFormat"] = dict(self.totals)
        return payload


@dataclass(frozen=True)
class ExamProfileResolution:
    context: AgentExamContext | None
    source: str
    reason: str | None
    cache_hit: bool
    duration_us: int


@dataclass(frozen=True)
class ExamProfileSnapshot:
    by_id: Mapping[str, ExamProfile]
    by_exam_stage: Mapping[tuple[str, str], str]
    profile_count: int
    serialized_bytes: int
    load_duration_ms: int
    rss_before_bytes: int
    rss_after_bytes: int


class ExamProfileSource(Protocol):
    def load_active_profiles(self) -> tuple[ExamProfile, ...]: ...


class DynamoExamProfileSource:
    """Startup-only SSM discovery plus paginated DynamoDB active-profile scan."""

    def __init__(
        self,
        *,
        region_name: str | None = None,
        ssm_client: Any | None = None,
        dynamodb_client: Any | None = None,
    ) -> None:
        self._region_name = (
            region_name
            or os.getenv("AWS_REGION", "").strip()
            or os.getenv("AWS_DEFAULT_REGION", "").strip()
        )
        self._ssm_client = ssm_client
        self._dynamodb_client = dynamodb_client

    def load_active_profiles(self) -> tuple[ExamProfile, ...]:
        if not self._region_name:
            raise RuntimeError("EXAM_PROFILE_REGION_UNAVAILABLE")
        ssm = self._ssm_client or get_ssm_client(self._region_name)
        try:
            parameter_response = ssm.get_parameters(
                Names=[_TABLE_NAME_PATH, _TABLE_ARN_PATH], WithDecryption=False
            )
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError("EXAM_PROFILE_RESOURCE_LOOKUP_FAILED") from exc
        values = {
            str(item.get("Name")): str(item.get("Value", "")).strip()
            for item in parameter_response.get("Parameters", [])
        }
        table_name = values.get(_TABLE_NAME_PATH)
        if not table_name:
            raise RuntimeError("EXAM_PROFILE_TABLE_UNAVAILABLE")
        dynamodb = self._dynamodb_client or get_dynamodb_client(self._region_name)
        items: list[dict[str, Any]] = []
        start_key: dict[str, Any] | None = None
        while True:
            kwargs: dict[str, Any] = {
                "TableName": table_name,
                "FilterExpression": "#active = :active",
                "ExpressionAttributeNames": {
                    "#active": "active",
                    "#description": "description",
                    "#stage": "stage",
                },
                "ExpressionAttributeValues": {":active": {"BOOL": True}},
                "ProjectionExpression": (
                    "examProfileId, examId, examName, #stage, #description, sections, "
                    "totalQuestions, totalMarks, totalTimeMinutes, #active, effectiveYear"
                ),
            }
            if start_key is not None:
                kwargs["ExclusiveStartKey"] = start_key
            try:
                response = dynamodb.scan(**kwargs)
            except (BotoCoreError, ClientError) as exc:
                raise RuntimeError("EXAM_PROFILE_LOAD_FAILED") from exc
            items.extend(response.get("Items", []))
            start_key = response.get("LastEvaluatedKey")
            if not start_key:
                break
        profiles: list[ExamProfile] = []
        for item in items:
            try:
                decoded = {key: _DESERIALIZER.deserialize(value) for key, value in item.items()}
                profiles.append(ExamProfile.model_validate(decoded))
            except (AttributeError, TypeError, ValidationError) as exc:
                logger.warning("exam_profile_malformed_quarantined error=%s", type(exc).__name__)
        return tuple(profiles)


class ExamProfileRuntime:
    def __init__(self, source: ExamProfileSource | None = None) -> None:
        self._source = source or DynamoExamProfileSource()
        self._lock = threading.Lock()
        self._snapshot: ExamProfileSnapshot | None = None

    @property
    def snapshot(self) -> ExamProfileSnapshot | None:
        return self._snapshot

    def load(self) -> bool:
        started = time.perf_counter()
        rss_before = _rss_bytes()
        logger.info("exam_profiles_load_started source=dynamodb")
        try:
            profiles = self._source.load_active_profiles()
            candidate = _build_snapshot(profiles, started, rss_before)
        except Exception as exc:  # noqa: BLE001 - must preserve last-known-good cache
            logger.warning("exam_profiles_load_failed source=dynamodb error=%s", type(exc).__name__)
            return False
        with self._lock:
            self._snapshot = candidate
        logger.info(
            "exam_profiles_load_completed source=dynamodb profileCount=%d "
            "durationMs=%d serializedBytes=%d processRSSBefore=%d "
            "processRSSAfter=%d estimatedMemoryDelta=%d",
            candidate.profile_count,
            candidate.load_duration_ms,
            candidate.serialized_bytes,
            candidate.rss_before_bytes,
            candidate.rss_after_bytes,
            candidate.rss_after_bytes - candidate.rss_before_bytes,
        )
        return True

    def resolve(
        self,
        *,
        exam_profile_id: str | None,
        exam_id: str | None,
        exam_stage: str | None,
        subject: str | None = None,
        full_mock: bool = False,
    ) -> ExamProfileResolution:
        started = time.perf_counter_ns()
        snapshot = self._snapshot
        if snapshot is None:
            return _resolution(None, "legacy_fallback", "CACHE_UNAVAILABLE", False, started)
        profile: ExamProfile | None = None
        reason: str | None = None
        if exam_profile_id:
            profile = snapshot.by_id.get(_canonical_id(exam_profile_id))
            reason = None if profile else "PROFILE_NOT_FOUND"
        elif exam_id and exam_stage:
            canonical = snapshot.by_exam_stage.get(
                (_canonical_id(exam_id), _canonical_id(exam_stage))
            )
            profile = snapshot.by_id.get(canonical) if canonical else None
            reason = None if profile else "LEGACY_REQUEST_MAPPING"
        else:
            reason = "LEGACY_REQUEST_MAPPING"
        if profile is None:
            return _resolution(None, "legacy_fallback", reason, False, started)
        context = _project(profile, subject=subject, full_mock=full_mock)
        return _resolution(context, "dynamodb_cache", None, True, started)


def _resolution(
    context: AgentExamContext | None,
    source: str,
    reason: str | None,
    cache_hit: bool,
    started_ns: int,
) -> ExamProfileResolution:
    duration_us = (time.perf_counter_ns() - started_ns) // 1_000
    log = logger.debug if cache_hit else logger.info
    log(
        "exam_profile_resolved examProfileId=%s examId=%s stage=%s source=%s "
        "cacheHit=%s reason=%s durationUs=%d",
        context.exam_profile_id if context else "",
        context.exam_id if context else "",
        context.stage if context else "",
        source,
        cache_hit,
        reason or "",
        duration_us,
    )
    return ExamProfileResolution(
        context=context, source=source, reason=reason, cache_hit=cache_hit, duration_us=duration_us
    )


def _project(profile: ExamProfile, *, subject: str | None, full_mock: bool) -> AgentExamContext:
    requested_subject = (
        subject.strip().upper().replace(" ", "_").replace("-", "_") if subject else None
    )
    matched = tuple(
        section
        for section in profile.sections
        if requested_subject is None or section.subject == requested_subject
    )
    selected = matched or profile.sections if full_mock else matched
    sections = tuple(
        MappingProxyType(
            {
                "sectionId": section.section_id,
                "subject": section.subject,
                "level": section.level,
                "questionStyles": section.question_styles,
                "excludeTopics": section.exclude_topics,
                **(
                    {
                        "questionCount": section.question_count,
                        "marks": section.marks,
                        "timeMinutes": section.time_minutes,
                        "negativeMarks": section.negative_marks,
                        "marksPerQuestion": section.marks_per_question,
                    }
                    if full_mock
                    else {}
                ),
            }
        )
        for section in selected
    )
    totals: Mapping[str, object] | None = None
    if full_mock:
        totals = MappingProxyType(
            {
                "totalQuestions": profile.total_questions,
                "totalMarks": profile.total_marks,
                "totalTimeMinutes": profile.total_time_minutes,
            }
        )
    return AgentExamContext(
        profile.exam_profile_id,
        profile.exam_id,
        profile.exam_name,
        profile.stage,
        profile.description,
        sections,
        totals,
    )


def _build_snapshot(
    profiles: tuple[ExamProfile, ...], started: float, rss_before: int
) -> ExamProfileSnapshot:
    by_id: dict[str, ExamProfile] = {}
    by_exam_stage: dict[tuple[str, str], str] = {}
    for profile in profiles:
        if not profile.active:
            continue
        profile_id = _canonical_id(profile.exam_profile_id)
        key = (_canonical_id(profile.exam_id), _canonical_id(profile.stage))
        if profile_id in by_id or key in by_exam_stage:
            raise RuntimeError("EXAM_PROFILE_DUPLICATE_IDENTITY")
        by_id[profile_id] = profile
        by_exam_stage[key] = profile_id
    active_profiles = tuple(by_id.values())
    serialized = json.dumps(
        [profile.model_dump(by_alias=True, mode="json") for profile in active_profiles],
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return ExamProfileSnapshot(
        MappingProxyType(by_id),
        MappingProxyType(by_exam_stage),
        len(active_profiles),
        len(serialized),
        int((time.perf_counter() - started) * 1000),
        rss_before,
        _rss_bytes(),
    )


def _canonical_id(value: str) -> str:
    return value.strip().upper().replace(" ", "_").replace("-", "_")


def _rss_bytes() -> int:
    # Linux returns KiB; macOS returns bytes.  Local tests run on macOS.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if os.uname().sysname == "Darwin" else value * 1024


_runtime: ExamProfileRuntime | None = None
_runtime_lock = threading.Lock()


def get_exam_profile_runtime() -> ExamProfileRuntime:
    global _runtime  # noqa: PLW0603
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = ExamProfileRuntime()
    return _runtime


def reset_exam_profile_runtime_for_tests() -> None:
    global _runtime  # noqa: PLW0603
    with _runtime_lock:
        _runtime = None
