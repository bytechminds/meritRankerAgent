#!/usr/bin/env python3
"""Run the bounded Dev-only Average lifecycle gate.

This is an opt-in qualification harness. It uses the real Practice lifecycle,
including Dev DynamoDB persistence, AppSync progress publication, finalization,
and READY/FAILED terminal handling. It can use either an explicit candidate-model
override for pre-promotion qualification or the configured production route without
an override. It never enables student-credit enforcement. Public output and the
optional report contain only aggregate operational data; final question content may
be captured privately under ``/private/tmp`` for independent mathematical review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_RUN_FLAG = "RUN_PRACTICE_MATH_LIFECYCLE_QUALIFICATION"
_TARGET_FLAG = "PRACTICE_LIFECYCLE_QUALIFICATION_TARGET"
_TARGET_ACCOUNT_FLAG = "PRACTICE_LIFECYCLE_QUALIFICATION_AWS_ACCOUNT"
_PRIVATE_REVIEW_FLAG = "ALLOW_PRIVATE_MATH_QUALIFICATION_REVIEW"
_PRIVATE_ROOT = Path("/private/tmp").resolve()
_DEV_ACCOUNT_ID = "661012794182"
_REVIEW_CAPTURE_FORMAT = "practice_math_lifecycle_private_review_v1"
_DEFAULT_QUERY = "Create a mock test of 10 questions from average topic in math."
_MAX_EXPECTED_COUNT = 50


@dataclass(frozen=True)
class _LifecycleScenario:
    query: str
    subject: str
    topic: str
    difficulty: str
    expected_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-model",
        default="openai_gpt_5_6_terra",
        help="Existing registry alias to substitute for the Math generator only.",
    )
    parser.add_argument(
        "--use-production-route",
        action="store_true",
        help=(
            "Use the configured Practice Math route with no model override. "
            "Required to qualify an actual route promotion."
        ),
    )
    parser.add_argument(
        "--runs",
        default=1,
        type=int,
        help="Independent lifecycle runs (1..5).",
    )
    parser.add_argument(
        "--query",
        default=_DEFAULT_QUERY,
        help="Complete self-contained Practice request to replay.",
    )
    parser.add_argument(
        "--subject",
        default="math",
        help="Classified Practice subject for the request.",
    )
    parser.add_argument(
        "--topic",
        default="average",
        help="Classified Practice topic for the request.",
    )
    parser.add_argument(
        "--difficulty",
        choices=("basic", "intermediate", "advanced"),
        default="intermediate",
        help="Classified Practice difficulty for the request.",
    )
    parser.add_argument(
        "--expected-count",
        default=10,
        type=int,
        help="Expected count after the production request resolver runs (1..50).",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional new aggregate-only JSON path below /private/tmp.",
    )
    parser.add_argument(
        "--review-capture-path",
        default=None,
        help=(
            "Optional new private final-question capture below /private/tmp. Requires "
            "ALLOW_PRIVATE_MATH_QUALIFICATION_REVIEW=true and is never printed."
        ),
    )
    parser.add_argument(
        "--timeout-seconds",
        default=900,
        type=int,
        help="Maximum wait per lifecycle run (60..900 seconds).",
    )
    return parser.parse_args()


def _validate_private_path(raw_path: str, *, require_new: bool) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_relative_to(_PRIVATE_ROOT):
        raise ValueError("private qualification paths must be below /private/tmp.")
    if path == _PRIVATE_ROOT or not path.parent.is_dir():
        raise ValueError("private qualification path parent must already exist.")
    if require_new and path.exists():
        raise ValueError("private qualification output path must not already exist.")
    return path


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, separators=(",", ":"), sort_keys=True)
        output_file.write("\n")


def _scenario_from_args(args: argparse.Namespace) -> _LifecycleScenario:
    query = str(args.query or "").strip()
    subject = str(args.subject or "").strip().casefold()
    topic = str(args.topic or "").strip().casefold()
    if not query or not subject or not topic:
        raise ValueError("--query, --subject and --topic must be non-empty.")
    if not 1 <= args.expected_count <= _MAX_EXPECTED_COUNT:
        raise ValueError(
            f"--expected-count must be between 1 and {_MAX_EXPECTED_COUNT}."
        )
    return _LifecycleScenario(
        query=query,
        subject=subject,
        topic=topic,
        difficulty=args.difficulty,
        expected_count=args.expected_count,
    )


class _LocalTaskTracker:
    """Process-local AgentCore tracker substitute used only by this harness."""

    def __init__(self) -> None:
        self._next_task_id = 1
        self._events: dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    def add_async_task(self, name: str, metadata: dict | None = None) -> int:
        del name, metadata
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
            self._events[task_id] = threading.Event()
            return task_id

    def complete_async_task(self, task_id: int) -> bool:
        with self._lock:
            event = self._events.get(task_id)
        if event is None:
            return False
        event.set()
        return True

    def wait_for_completion(self, timeout_seconds: int) -> bool:
        with self._lock:
            if not self._events:
                raise RuntimeError("lifecycle run did not register a task.")
            event = self._events[max(self._events)]
        return event.wait(timeout=timeout_seconds)


@dataclass
class _EventCapture:
    test_id: str | None = None
    counts: Counter[str] = field(default_factory=Counter)
    reason_codes: Counter[str] = field(default_factory=Counter)
    runtime_event_counts: Counter[str] = field(default_factory=Counter)
    generator_routes: Counter[str] = field(default_factory=Counter)
    generator_model_aliases: Counter[str] = field(default_factory=Counter)
    generator_models: Counter[str] = field(default_factory=Counter)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(
        self,
        event: str,
        *,
        test_id: str,
        details: dict[str, object] | None,
    ) -> None:
        if self.test_id != test_id:
            return
        reason_code = (details or {}).get("reasonCode")
        with self._lock:
            self.counts[event] += 1
            if isinstance(reason_code, str) and reason_code:
                self.reason_codes[reason_code] += 1

    def observe_runtime(self, event: str, details: dict[str, object] | None) -> None:
        if event not in {
            "generator_model_attempt_started",
            "model_execution_completed",
            "llm_call_usage",
        }:
            return
        safe_details = details or {}
        role = str(safe_details.get("role") or "")
        route = str(safe_details.get("routeId") or safe_details.get("route") or "")
        model_alias = str(
            safe_details.get("modelAlias") or safe_details.get("model_alias") or ""
        )
        model = str(safe_details.get("model") or "")
        with self._lock:
            self.runtime_event_counts[event] += 1
            if role == "generator" or event == "generator_model_attempt_started":
                if route:
                    self.generator_routes[route] += 1
                if model_alias:
                    self.generator_model_aliases[model_alias] += 1
                if model:
                    self.generator_models[model] += 1

    def runtime_telemetry(self) -> dict[str, object]:
        with self._lock:
            return {
                "event_counts": dict(sorted(self.runtime_event_counts.items())),
                "generator_routes": dict(sorted(self.generator_routes.items())),
                "generator_model_aliases": dict(sorted(self.generator_model_aliases.items())),
                "generator_models": dict(sorted(self.generator_models.items())),
            }


def _install_event_capture(capture: _EventCapture) -> Callable[[], None]:
    from features.practice_generation import orchestration

    original = orchestration.emit_practice_event

    def wrapped(
        event: str, *, test_id: str, details: dict[str, object] | None = None, **kwargs: object
    ) -> None:
        capture.observe(event, test_id=test_id, details=details)
        original(event, test_id=test_id, details=details, **kwargs)

    orchestration.emit_practice_event = wrapped

    def restore() -> None:
        orchestration.emit_practice_event = original

    return restore


def _install_runtime_telemetry_capture(capture: _EventCapture) -> Callable[[], None]:
    """Capture route/model telemetry without retaining prompts or model output."""
    from observability import llm_usage
    from services.llm.orchestration import model_execution

    original_model_event = model_execution.log_event
    original_usage_event = llm_usage.log_event

    def model_event(event: str, **kwargs: object) -> None:
        details = kwargs.get("details")
        capture.observe_runtime(event, details if isinstance(details, dict) else None)
        original_model_event(event, **kwargs)

    def usage_event(event: str, **kwargs: object) -> None:
        details = kwargs.get("details")
        capture.observe_runtime(event, details if isinstance(details, dict) else None)
        original_usage_event(event, **kwargs)

    model_execution.log_event = model_event
    llm_usage.log_event = usage_event

    def restore() -> None:
        model_execution.log_event = original_model_event
        llm_usage.log_event = original_usage_event

    return restore


def _request(*, run_id: str, sequence: int, scenario: _LifecycleScenario):
    from features.practice_generation.planning import resolve_practice_request

    request_suffix = f"{run_id}-{sequence:02d}"
    request = resolve_practice_request(
        request_id=f"practice-lifecycle-{request_suffix}",
        user_id="qualification-runner",
        conversation_id=f"terra-lifecycle-{run_id}",
        turn_id=f"practice-lifecycle-{sequence:02d}",
        query=scenario.query,
        subject=scenario.subject,
        topic=scenario.topic,
        difficulty=scenario.difficulty,
        language="english",
        exam_id=None,
        exam_stage=None,
    )
    if request.accepted_count != scenario.expected_count:
        raise ValueError(
            "production request resolver count does not match --expected-count "
            f"({request.accepted_count} != {scenario.expected_count})."
        )
    return request


def _meta(assessment: dict[str, object]) -> dict[str, object]:
    value = assessment.get("meta")
    return value if isinstance(value, dict) else {}


def _run_report(
    *,
    assessment: dict[str, object] | None,
    test_id: str,
    duration_ms: int,
    completed: bool,
    events: _EventCapture,
    metrics: dict[str, object],
) -> dict[str, object]:
    meta = _meta(assessment or {})
    manifest = meta.get("readyQuestionIds")
    manifest_ids = [str(value) for value in manifest] if isinstance(manifest, list) else []
    requested_count = int((assessment or {}).get("totalQuestions") or 0)
    ready_count = int(meta.get("readyCount") or 0)
    return {
        "test_id": test_id,
        "completed_within_timeout": completed,
        "terminal_status": str((assessment or {}).get("status") or "UNKNOWN"),
        "terminal_reason": meta.get("errorCode")
        if isinstance(meta.get("errorCode"), str)
        else None,
        "requested_count": requested_count,
        "ready_count": ready_count,
        "playable": bool(meta.get("playable")),
        "manifest_count": len(manifest_ids),
        "unique_manifest_count": len(set(manifest_ids)),
        "reused_count": int(meta.get("reusedCount") or 0),
        "generated_count": int(meta.get("generatedCount") or 0),
        "verified_count": int(meta.get("verifiedCount") or 0),
        "failed_count": int(meta.get("failedCount") or 0),
        "replacement_wave_count": int(meta.get("replacementWaveCount") or 0),
        "key_correction_count": events.reason_codes["AUTHOR_AUTHORITY_KEY_CORRECTED"],
        "replacement_count": events.counts["question_replacement_completed"],
        "repair_count": events.counts["question_repair_completed"],
        "verification_approved_count": (
            events.reason_codes["VERIFIER_APPROVED"]
            + events.reason_codes["AUTHOR_AUTHORITY_KEY_CORRECTED"]
        ),
        "verification_event_count": events.counts["QUESTION_VERIFICATION_RESULT"],
        "event_counts": dict(sorted(events.counts.items())),
        "event_reason_counts": dict(sorted(events.reason_codes.items())),
        "runtime_telemetry": events.runtime_telemetry(),
        "duration_ms": duration_ms,
        "execution_metrics": metrics,
    }


def _private_review_records(
    *,
    assessment: dict[str, object],
    questions: list[dict[str, object]],
) -> list[dict[str, object]]:
    test_id = str(assessment.get("testId") or "")
    manifest = _meta(assessment).get("readyQuestionIds")
    manifest_ids = [str(value) for value in manifest] if isinstance(manifest, list) else []
    by_id = {str(question.get("questionId") or ""): question for question in questions}
    records: list[dict[str, object]] = []
    for question_id in manifest_ids:
        question = by_id.get(question_id)
        if question is None:
            raise ValueError("final manifest references an unreadable question.")
        records.append(
            {
                "review_id": f"{test_id}:{question_id}",
                "test_id": test_id,
                "question": question.get("question"),
                "options": question.get("options"),
                "correct_answer": question.get("correctAnswer"),
                "answer_contract": question.get("answers"),
                "meta": question.get("meta"),
            }
        )
    return records


def _run_lifecycle(
    *,
    launcher: Any,
    recorder: Any,
    sequence: int,
    run_id: str,
    scenario: _LifecycleScenario,
    timeout_seconds: int,
) -> tuple[dict[str, object], dict[str, object] | None, list[dict[str, object]] | None]:
    from qualify_practice_math import _summarize_metrics

    tracker = launcher._task_tracker
    if not isinstance(tracker, _LocalTaskTracker):
        raise RuntimeError("lifecycle launcher does not use the local qualification tracker.")
    capture = _EventCapture()
    restore_events = _install_event_capture(capture)
    restore_runtime_telemetry = _install_runtime_telemetry_capture(capture)
    before_metric_count = len(recorder.metrics)
    started_at = time.monotonic()
    try:
        launch = launcher.launch(_request(run_id=run_id, sequence=sequence, scenario=scenario))
        capture.test_id = launch.test_id
        if launch.status != "GENERATING" or not launcher.start(launch.test_id):
            assessment = launcher._assessments.get(launch.test_id)
            return (
                _run_report(
                    assessment=assessment,
                    test_id=launch.test_id,
                    duration_ms=int((time.monotonic() - started_at) * 1000),
                    completed=False,
                    events=capture,
                    metrics=_summarize_metrics(recorder.metrics[before_metric_count:]),
                ),
                assessment,
                None,
            )
        completed = tracker.wait_for_completion(timeout_seconds)
        if not completed:
            launcher.cancel(launch.test_id, "qualification-runner")
            completed = tracker.wait_for_completion(60)
        assessment = launcher._assessments.get(launch.test_id)
        report = _run_report(
            assessment=assessment,
            test_id=launch.test_id,
            duration_ms=int((time.monotonic() - started_at) * 1000),
            completed=completed,
            events=capture,
            metrics=_summarize_metrics(recorder.metrics[before_metric_count:]),
        )
        questions: list[dict[str, object]] | None = None
        if (
            assessment is not None
            and report["terminal_status"] == "READY"
            and isinstance(_meta(assessment).get("readyQuestionIds"), list)
        ):
            questions = launcher._progress._questions.get_questions_by_ids(
                [str(value) for value in _meta(assessment)["readyQuestionIds"]]
            )
        return report, assessment, questions
    finally:
        restore_events()
        restore_runtime_telemetry()


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    total = len(runs)
    ready = sum(item["terminal_status"] == "READY" for item in runs)
    failed = sum(item["terminal_status"] == "FAILED" for item in runs)
    completed = sum(bool(item["completed_within_timeout"]) for item in runs)
    requested = sum(int(item["requested_count"]) for item in runs)
    ready_questions = sum(int(item["ready_count"]) for item in runs)
    return {
        "run_count": total,
        "ready_count": ready,
        "failed_count": failed,
        "completed_within_timeout_count": completed,
        "ready_rate": round(ready / total, 4) if total else 0.0,
        "requested_question_count": requested,
        "ready_question_count": ready_questions,
        "key_correction_count": sum(int(item["key_correction_count"]) for item in runs),
        "replacement_count": sum(int(item["replacement_count"]) for item in runs),
        "repair_count": sum(int(item["repair_count"]) for item in runs),
        "duration_ms": {
            "mean": round(sum(int(item["duration_ms"]) for item in runs) / total, 2)
            if total
            else None,
            "max": max((int(item["duration_ms"]) for item in runs), default=None),
        },
    }


def main() -> int:
    args = _parse_args()
    if os.environ.get(_RUN_FLAG, "").strip().lower() != "true":
        print(f"{_RUN_FLAG}=true is required; no lifecycle qualification was run.")
        return 0
    if os.environ.get(_TARGET_FLAG, "").strip().lower() != "dev":
        print(f"{_TARGET_FLAG}=dev is required; no lifecycle qualification was run.")
        return 2
    if os.environ.get(_TARGET_ACCOUNT_FLAG, "").strip() != _DEV_ACCOUNT_ID:
        print(f"{_TARGET_ACCOUNT_FLAG} must name the designated Dev account.")
        return 2
    if os.environ.get("STUDENT_CREDIT_ENFORCEMENT_ENABLED", "").strip().lower() != "false":
        print("STUDENT_CREDIT_ENFORCEMENT_ENABLED=false is required.")
        return 2
    if not 1 <= args.runs <= 5:
        print("--runs must be between 1 and 5.")
        return 2
    if not 60 <= args.timeout_seconds <= 900:
        print("--timeout-seconds must be between 60 and 900.")
        return 2
    try:
        scenario = _scenario_from_args(args)
    except ValueError as exc:
        print(str(exc))
        return 2
    try:
        report_path = (
            _validate_private_path(args.report_path, require_new=True) if args.report_path else None
        )
        review_capture_path = (
            _validate_private_path(args.review_capture_path, require_new=True)
            if args.review_capture_path
            else None
        )
    except ValueError as exc:
        print(str(exc))
        return 2
    if review_capture_path is not None and (
        os.environ.get(_PRIVATE_REVIEW_FLAG, "").strip().lower() != "true"
    ):
        print(f"{_PRIVATE_REVIEW_FLAG}=true is required for --review-capture-path.")
        return 2

    app_root = str(Path(__file__).resolve().parents[1])
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    from qualify_practice_math import _RecordingGeneratorOverride

    from config import get_settings
    from features.practice_generation.agentcore_async import build_practice_async_launcher
    from services.llm.orchestration.config_registry import LlmConfigRegistry
    from services.llm.orchestration.orchestrator import LlmOrchestrator
    from services.llm.pricing import load_pricing_config
    from services.llm.runtime_factory import build_model_executor

    settings = get_settings()
    if settings.app_env == "production":
        print("Practice lifecycle qualification is not permitted in production.")
        return 2
    if not settings.enable_real_llm:
        print("ENABLE_REAL_LLM=true is required; no lifecycle qualification was run.")
        return 2
    if settings.student_credit_enforcement_enabled:
        print("student credit enforcement must remain disabled.")
        return 2
    import boto3

    identity = boto3.Session(region_name="ap-south-1").client("sts").get_caller_identity()
    if str(identity.get("Account") or "") != _DEV_ACCOUNT_ID:
        print("AWS identity is not the designated Dev account; no lifecycle qualification was run.")
        return 2

    registry = LlmConfigRegistry()
    if not args.use_production_route and args.candidate_model not in registry.model_map:
        print("--candidate-model must be an existing registry alias.")
        return 2
    from features.practice_generation.planning import practice_generator_route_subject
    from schemas.llm_routing import RouteRequest
    from services.llm.orchestration.route_resolver import resolve_route

    route_subject = practice_generator_route_subject(scenario.subject, scenario.difficulty)
    configured_route = resolve_route(
        RouteRequest(
            request_id="practice-math-lifecycle-route-audit",
            subject=route_subject,
            task_role="generator",
            difficulty=scenario.difficulty,
            intent="practice",
            language="english",
        )
    )
    if args.use_production_route and route_subject == "practice_math" and (
        configured_route.route_id != f"practice_math.generator.{scenario.difficulty}"
        or configured_route.model != "openai_gpt_5_6_terra"
    ):
        print("configured Practice Math route is not the qualified Terra route.")
        return 2
    recorder = _RecordingGeneratorOverride(
        build_model_executor(settings),
        alias=None if args.use_production_route else args.candidate_model,
        model_configs=registry.model_map,
        pricing_config=load_pricing_config(),
    )
    launcher = build_practice_async_launcher(
        task_tracker=_LocalTaskTracker(),
        llm_orchestrator=LlmOrchestrator(model_executor=recorder),
        student_credits=None,
    )
    if launcher is None:
        print("Practice runtime is not enabled; no lifecycle qualification was run.")
        return 2

    run_id = uuid.uuid4().hex[:12]
    reports: list[dict[str, object]] = []
    review_records: list[dict[str, object]] = []
    for sequence in range(1, args.runs + 1):
        report, assessment, questions = _run_lifecycle(
            launcher=launcher,
            recorder=recorder,
            sequence=sequence,
            run_id=run_id,
            scenario=scenario,
            timeout_seconds=args.timeout_seconds,
        )
        reports.append(report)
        if review_capture_path is not None and assessment is not None and questions is not None:
            review_records.extend(
                _private_review_records(assessment=assessment, questions=questions)
            )
        if report["terminal_status"] != "READY":
            break

    aggregate = {
        "run_kind": "dev_persistent_practice_lifecycle_qualification",
        "target": "designated_dev_account",
        "candidate_model_alias": (
            configured_route.model if args.use_production_route else args.candidate_model
        ),
        "route_binding_mode": "production_route" if args.use_production_route else "override",
        "generator_model_override": None if args.use_production_route else args.candidate_model,
        "configured_generator_route": {
            "route_id": configured_route.route_id,
            "model": configured_route.model,
            "provider_options": configured_route.provider_options,
        },
        "student_credit_enforcement": "disabled",
        "request_profile": {
            "subject": scenario.subject,
            "topic": scenario.topic,
            "difficulty": scenario.difficulty,
            "expected_count": scenario.expected_count,
        },
        "runs": reports,
        "summary": _aggregate_runs(reports),
        "private_review_capture": "created" if review_capture_path is not None else "not_requested",
    }
    if review_capture_path is not None:
        _write_private_json(
            review_capture_path,
            {
                "format": _REVIEW_CAPTURE_FORMAT,
                "candidate_model_alias": args.candidate_model,
                "records": review_records,
            },
        )
    if report_path is not None:
        _write_private_json(report_path, aggregate)
    print(json.dumps(aggregate, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
