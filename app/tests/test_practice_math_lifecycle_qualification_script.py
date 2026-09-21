from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from features.practice_generation.planning import select_planning_mode
from features.practice_generation.schemas import Difficulty, PracticeType


def _lifecycle_module():
    script = Path(__file__).parents[1] / "scripts" / "qualify_practice_math_lifecycle.py"
    spec = spec_from_file_location("practice_math_lifecycle_qualification_script", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lifecycle_harness_is_a_zero_cost_noop_without_its_run_flag() -> None:
    script = Path(__file__).parents[1] / "scripts" / "qualify_practice_math_lifecycle.py"
    environment = dict(os.environ)
    environment.pop("RUN_PRACTICE_MATH_LIFECYCLE_QUALIFICATION", None)

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == (
        "RUN_PRACTICE_MATH_LIFECYCLE_QUALIFICATION=true is required; "
        "no lifecycle qualification was run."
    )
    assert completed.stderr == ""


def test_lifecycle_harness_refuses_to_resolve_runtime_without_dev_target_guard() -> None:
    script = Path(__file__).parents[1] / "scripts" / "qualify_practice_math_lifecycle.py"
    environment = dict(os.environ)
    environment["RUN_PRACTICE_MATH_LIFECYCLE_QUALIFICATION"] = "true"
    environment.pop("PRACTICE_LIFECYCLE_QUALIFICATION_TARGET", None)

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert completed.stdout.strip() == (
        "PRACTICE_LIFECYCLE_QUALIFICATION_TARGET=dev is required; "
        "no lifecycle qualification was run."
    )
    assert completed.stderr == ""


def test_lifecycle_aggregate_preserves_terminal_and_count_evidence() -> None:
    module = _lifecycle_module()

    aggregate = module._aggregate_runs(
        [
            {
                "terminal_status": "READY",
                "completed_within_timeout": True,
                "requested_count": 10,
                "ready_count": 10,
                "key_correction_count": 1,
                "replacement_count": 0,
                "repair_count": 0,
                "duration_ms": 1200,
            },
            {
                "terminal_status": "FAILED",
                "completed_within_timeout": True,
                "requested_count": 10,
                "ready_count": 0,
                "key_correction_count": 0,
                "replacement_count": 1,
                "repair_count": 1,
                "duration_ms": 1800,
            },
        ]
    )

    assert aggregate == {
        "run_count": 2,
        "ready_count": 1,
        "failed_count": 1,
        "completed_within_timeout_count": 2,
        "ready_rate": 0.5,
        "requested_question_count": 20,
        "ready_question_count": 10,
        "key_correction_count": 1,
        "replacement_count": 1,
        "repair_count": 1,
        "duration_ms": {"mean": 1500.0, "max": 1800},
    }


def test_local_tracker_waits_for_the_most_recent_lifecycle_task() -> None:
    module = _lifecycle_module()
    tracker = module._LocalTaskTracker()

    first = tracker.add_async_task("practice_generation")
    assert tracker.complete_async_task(first) is True
    second = tracker.add_async_task("practice_generation")
    assert tracker.wait_for_completion(0) is False
    assert tracker.complete_async_task(second) is True
    assert tracker.wait_for_completion(0) is True


def test_lifecycle_event_capture_reports_aggregate_counts_only() -> None:
    module = _lifecycle_module()
    capture = module._EventCapture(test_id="assessment")

    capture.observe(
        "GENERATION_GROUP_CREATED",
        test_id="assessment",
        details={"reasonCode": "NO_VALID_OPTION", "slotId": "not-reported"},
    )

    assert capture.counts == {"GENERATION_GROUP_CREATED": 1}
    assert capture.reason_codes == {"NO_VALID_OPTION": 1}


def test_average_lifecycle_request_uses_the_production_resolver_route() -> None:
    module = _lifecycle_module()

    request = module._request(
        run_id="qualification",
        sequence=1,
        scenario=module._LifecycleScenario(
            query="Create a mock test of 10 questions from average topic in math.",
            subject="math",
            topic="average",
            difficulty="intermediate",
            expected_count=10,
        ),
    )

    assert request.practice_type is PracticeType.QUICK_PRACTICE
    assert request.accepted_count == 10
    assert request.subject == "math"
    assert request.topic == "average"
    assert request.difficulty is Difficulty.INTERMEDIATE
    assert select_planning_mode(request) == (
        "DETERMINISTIC",
        "EXISTING_DETERMINISTIC_RULE",
    )


def test_similar_source_lifecycle_request_preserves_intelligence_planning() -> None:
    module = _lifecycle_module()

    request = module._request(
        run_id="qualification",
        sequence=1,
        scenario=module._LifecycleScenario(
            query=(
                "create similar 20 questions like The ages of A and B are in the ratio 5:7. "
                "Five years ago, their ages were in the ratio 5:8. The respective present ages "
                "(in years) are: practice test"
            ),
            subject="math",
            topic="age_problems",
            difficulty="intermediate",
            expected_count=20,
        ),
    )

    assert request.practice_type is PracticeType.SIMILAR_QUESTION
    assert request.accepted_count == 20
    assert request.topic == "age_problems"
    assert select_planning_mode(request) == (
        "INTELLIGENCE",
        "EXISTING_PRACTICE_TYPE_RULE",
    )
