from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


def _qualification_module():
    script = Path(__file__).parents[1] / "scripts" / "qualify_practice_math.py"
    spec = spec_from_file_location("practice_math_qualification_script", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_math_qualification_script_is_a_zero_cost_noop_without_its_run_flag() -> None:
    script = Path(__file__).parents[1] / "scripts" / "qualify_practice_math.py"
    environment = dict(os.environ)
    environment.pop("RUN_PRACTICE_MATH_QUALIFICATION", None)

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == (
        "RUN_PRACTICE_MATH_QUALIFICATION=true is required; no qualification was run."
    )
    assert completed.stderr == ""


def test_authority_control_mode_is_also_a_zero_cost_noop_without_its_run_flag() -> None:
    script = Path(__file__).parents[1] / "scripts" / "qualify_practice_math.py"
    environment = dict(os.environ)
    environment.pop("RUN_PRACTICE_MATH_QUALIFICATION", None)

    completed = subprocess.run(
        [sys.executable, str(script), "--authority-controls"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == (
        "RUN_PRACTICE_MATH_QUALIFICATION=true is required; no qualification was run."
    )
    assert completed.stderr == ""


def test_authority_control_corpus_has_the_required_mixed_math_composition() -> None:
    module = _qualification_module()
    controls = module._AUTHORITY_CONTROLS

    assert len(controls) == 30
    assert sum(control.semantic_bucket == "VALID_CORRECT_KEY" for control in controls) == 21
    assert sum(control.semantic_bucket == "VALID_WRONG_KEY_ONLY" for control in controls) == 1
    assert sum(control.semantic_bucket == "CONTRADICTORY_DATA" for control in controls) == 5
    assert {control.semantic_bucket for control in controls if control.true_option_id is None} == {
        "AMBIGUOUS",
        "CONTRADICTORY_DATA",
        "MULTIPLE_VALID_OPTIONS",
        "NO_VALID_OPTION",
    }
    assert {control.topic for control in controls} == {
        "ages",
        "algebra",
        "average",
        "boats_and_streams",
        "geometry",
        "mensuration",
        "number_system",
        "percentage",
        "profit_loss",
        "ratio_proportion",
        "simple_compound_interest",
        "time_and_work",
        "time_speed_distance",
    }
    assert all(len(set(control.options)) == 4 for control in controls)


def test_authority_control_report_requires_zero_false_approvals_and_contradiction_rejection() -> (
    None
):
    module = _qualification_module()

    def outcome(control, *, accepted: bool = False):
        if control.true_option_id is not None:
            valid_ids = (control.true_option_id,)
            decision = "ACCEPT"
        elif accepted:
            valid_ids = ("0",)
            decision = "ACCEPT"
        else:
            valid_ids = ()
            decision = "REGENERATE"
        return module.CandidateOutcome(
            run_id="run",
            arm="authority_control",
            topic=control.topic,
            difficulty=control.difficulty,
            outcome="authority_agreement" if decision == "ACCEPT" else "authority_rejected",
            verifier_decision=decision,
            authority_valid_option_ids=valid_ids,
            verifier_reason_codes=(control.expected_reason_code,),
        )

    passing = [
        module.AuthorityControlEvaluation(control=control, outcome=outcome(control))
        for control in module._AUTHORITY_CONTROLS
    ]
    report = module._authority_control_report(evaluations=passing, metrics=[])

    assert report["launch_safety_decision"] == "MATH_AUTHORITY_LAUNCH_SAFETY_VALIDATED"
    assert report["authority_confusion"] == {
        "true_approvals": 22,
        "true_rejections": 8,
        "false_approvals": 0,
        "false_rejections": 0,
        "technical_unavailable": 0,
        "false_rejection_rate": 0.0,
        "valid_control_count": 22,
    }
    assert report["contradiction_controls"] == {
        "control_count": 5,
        "rejected_count": 5,
        "expected_reason_count": 5,
    }

    rejected = next(
        control
        for control in module._AUTHORITY_CONTROLS
        if control.semantic_bucket == "CONTRADICTORY_DATA"
    )
    failing = [
        module.AuthorityControlEvaluation(
            control=evaluation.control,
            outcome=outcome(evaluation.control, accepted=evaluation.control == rejected),
        )
        for evaluation in passing
    ]
    failed_report = module._authority_control_report(evaluations=failing, metrics=[])

    assert failed_report["authority_confusion"]["false_approvals"] == 1
    assert failed_report["launch_safety_decision"] == "NOT_ELIGIBLE_FALSE_APPROVAL"


def test_default_qualification_screen_covers_sixty_representative_math_items() -> None:
    module = _qualification_module()

    topics = module._selected_topics(None)

    assert len(topics) == 12
    assert len(topics) * 5 == 60
    assert {topic.difficulty for topic in topics} == {"basic", "intermediate", "advanced"}
    assert {topic.topic for topic in topics} == {
        "average",
        "percentage",
        "profit_loss",
        "ratio_proportion",
        "simple_compound_interest",
        "time_and_work",
        "time_speed_distance",
        "boats_and_streams",
        "number_system",
        "algebra",
        "geometry",
        "mensuration",
    }


def test_qualification_arm_selection_rejects_unknown_or_duplicate_arms() -> None:
    module = _qualification_module()

    assert module._selected_arms("candidate") == ("candidate",)
    assert module._selected_arms("current,candidate") == ("current", "candidate")

    try:
        module._selected_arms("candidate,candidate")
    except ValueError as exc:
        assert str(exc) == "Qualification arms must not repeat."
    else:
        raise AssertionError("duplicate arm selection must fail")


def test_manual_adjudication_report_keeps_semantic_and_authority_metrics_separate() -> None:
    module = _qualification_module()
    capture = {
        "format": "practice_math_private_review_v1",
        "records": [
            {
                "review_id": "current:average:basic:slot-001",
                "arm": "current",
                "topic": "average",
                "difficulty": "basic",
                "question": {"generator_correct_option_id": "0"},
                "math_authority": {"decision": "ACCEPT", "valid_option_ids": ["0"]},
            },
            {
                "review_id": "candidate:average:basic:slot-001",
                "arm": "candidate",
                "topic": "average",
                "difficulty": "basic",
                "question": {"generator_correct_option_id": "1"},
                "math_authority": {"decision": "ACCEPT", "valid_option_ids": ["0"]},
            },
            {
                "review_id": "candidate:algebra:advanced:slot-001",
                "arm": "candidate",
                "topic": "algebra",
                "difficulty": "advanced",
                "question": {"generator_correct_option_id": "2"},
                "math_authority": {"decision": "ACCEPT", "valid_option_ids": ["2"]},
            },
        ],
    }
    adjudication = {
        "format": "practice_math_manual_adjudication_v1",
        "records": [
            {
                "review_id": "current:average:basic:slot-001",
                "primary_bucket": "VALID_CORRECT_KEY",
                "true_option_id": "0",
            },
            {
                "review_id": "candidate:average:basic:slot-001",
                "primary_bucket": "VALID_WRONG_KEY_ONLY",
                "true_option_id": "0",
            },
            {
                "review_id": "candidate:algebra:advanced:slot-001",
                "primary_bucket": "NO_VALID_OPTION",
                "true_option_id": None,
            },
        ],
    }

    report = module._manual_adjudication_report(
        capture=capture,
        adjudication=adjudication,
    )

    candidate = report["arms"]["candidate"]
    assert candidate["semantic_valid_count"] == 1
    assert candidate["wrong_key_only_count"] == 1
    assert candidate["primary_buckets"] == {
        "NO_VALID_OPTION": 1,
        "VALID_WRONG_KEY_ONLY": 1,
    }
    assert candidate["authority_confusion"] == {
        "true_approvals": 1,
        "true_rejections": 0,
        "false_approvals": 1,
        "false_rejections": 0,
    }
    assert candidate["authority_technical"] == {"unavailable_count": 0}
    assert report["promotion_decision"] == "NOT_ELIGIBLE_PENDING_SEMANTIC_QUALITY_GATE"


def test_manual_adjudication_keeps_authority_unavailability_out_of_confusion_metrics() -> None:
    module = _qualification_module()
    capture = {
        "format": "practice_math_private_review_v1",
        "records": [
            {
                "review_id": "candidate:interest:intermediate:slot-001",
                "arm": "candidate",
                "topic": "interest",
                "difficulty": "intermediate",
                "question": {"generator_correct_option_id": "1"},
                "math_authority": {
                    "status": "unavailable",
                    "decision": None,
                    "valid_option_ids": [],
                },
            }
        ],
    }
    adjudication = {
        "format": "practice_math_manual_adjudication_v1",
        "records": [
            {
                "review_id": "candidate:interest:intermediate:slot-001",
                "primary_bucket": "VALID_CORRECT_KEY",
                "true_option_id": "1",
            }
        ],
    }

    report = module._manual_adjudication_report(
        capture=capture,
        adjudication=adjudication,
    )

    candidate = report["arms"]["candidate"]
    assert candidate["authority_confusion"] == {
        "true_approvals": 0,
        "true_rejections": 0,
        "false_approvals": 0,
        "false_rejections": 0,
    }
    assert candidate["authority_technical"] == {"unavailable_count": 1}
    assert report["promotion_decision"] == "NOT_ELIGIBLE_PENDING_AUTHORITY_COMPLETENESS"


def test_combined_review_capture_merges_restricted_capture_records(tmp_path: Path) -> None:
    module = _qualification_module()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps(
            {
                "format": "practice_math_private_review_v1",
                "records": [{"review_id": "first"}],
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "format": "practice_math_private_review_v1",
                "records": [{"review_id": "second"}],
            }
        ),
        encoding="utf-8",
    )

    combined = module._combined_review_capture((first, second))

    assert combined == {
        "format": "practice_math_private_review_v1",
        "records": [{"review_id": "first"}, {"review_id": "second"}],
    }


def test_reverification_rechecks_only_an_unavailable_authority_record() -> None:
    module = _qualification_module()

    class Verifier:
        def verify_slot(self, **kwargs: object) -> SimpleNamespace:
            assert kwargs["question"] is not None
            return SimpleNamespace(decision="ACCEPT", valid_option_ids=("1",))

    capture = {
        "format": "practice_math_private_review_v1",
        "records": [
            {
                "review_id": "run:candidate:simple_compound_interest:intermediate:slot-002",
                "arm": "candidate",
                "topic": "simple_compound_interest",
                "difficulty": "intermediate",
                "model": "gpt-5.6-terra",
                "question": {
                    "schema_version": "2",
                    "generation_item_id": "item-slot-002",
                    "slot_id": "slot-002",
                    "stem": "What is the simple interest on Rs. 100 at 10% for one year?",
                    "options": [
                        {"option_id": "0", "value": "Rs. 5"},
                        {"option_id": "1", "value": "Rs. 10"},
                        {"option_id": "2", "value": "Rs. 15"},
                        {"option_id": "3", "value": "Rs. 20"},
                    ],
                    "generator_correct_option_id": "1",
                },
                "math_authority": {
                    "status": "unavailable",
                    "decision": None,
                    "valid_option_ids": [],
                },
                "qualification_outcome": "verifier_error:ProviderExecutionError",
            }
        ],
    }

    refreshed, outcomes = module._reverify_unavailable_capture(
        capture=capture,
        verifier=Verifier(),
        run_id="recheck",
    )

    assert [outcome.outcome for outcome in outcomes] == ["authority_agreement"]
    assert refreshed["records"][0]["math_authority"] == {
        "status": "responded",
        "decision": "ACCEPT",
        "valid_option_ids": ["1"],
    }


def test_reverification_keeps_invalid_private_input_fail_closed() -> None:
    module = _qualification_module()

    class UnexpectedVerifier:
        def verify_slot(self, **kwargs: object) -> SimpleNamespace:
            raise AssertionError("invalid private input must not reach the Authority")

    capture = {
        "format": "practice_math_private_review_v1",
        "records": [
            {
                "review_id": "run:candidate:average:basic:slot-001",
                "arm": "candidate",
                "topic": "average",
                "difficulty": "basic",
                "model": "gpt-5.6-terra",
                "question": {
                    "schema_version": "2",
                    "generation_item_id": "item-slot-001",
                    "slot_id": "slot-001",
                    "stem": "Find the average of 2 and 4.",
                    "options": [],
                    "generator_correct_option_id": "1",
                },
                "math_authority": {
                    "status": "unavailable",
                    "decision": None,
                    "valid_option_ids": [],
                },
            }
        ],
    }

    refreshed, outcomes = module._reverify_unavailable_capture(
        capture=capture,
        verifier=UnexpectedVerifier(),
        run_id="recheck",
    )

    assert [outcome.outcome for outcome in outcomes] == ["verifier_error:InputValidationError"]
    assert refreshed["records"][0]["math_authority"]["status"] == "unavailable"
