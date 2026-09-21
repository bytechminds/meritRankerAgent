"""Schema-v2 authors omit prose; READY persistence requires authority-owned explanations."""

from __future__ import annotations

import json

import pytest

from features.practice_generation.generation import _is_persisted_schema_v2
from features.practice_generation.question_contract import (
    validate_persisted_playable_question,
)

OPTIONS = ["1", "2", "3", "4"]


def _item(schema_version: str, *, solution: str | None) -> dict[str, object]:
    """A persisted Question record in the shape the repository actually writes."""
    meta = {
        "slotId": "slot-001",
        "bucketId": "bucket-1",
        "verified": True,
        "schemaVersion": schema_version,
        "questionType": "mcq",
        "language": "english",
    }
    answers: dict[str, object]
    if schema_version == "2":
        answers = {
            "schemaVersion": "2",
            "optionIdentity": "INDEX_V1",
            "answerStatus": "VERIFIED",
            "answerVersion": 1,
            "options": [
                {"optionId": index, "value": value}
                for index, value in enumerate(OPTIONS)
            ],
            "correctOptionId": 3,
            "correctAnswer": "4",
            "answerExplanation": solution or "",
        }
    else:
        answers = {"options": list(OPTIONS), "correctAnswer": "4"}
    item: dict[str, object] = {
        "questionId": "q-1",
        "question": "Which option equals two plus two?",
        "options": list(OPTIONS),
        "correctAnswer": "4",
        "answers": answers,
        "meta": json.dumps(meta),
        "_practiceMeta": meta,
    }
    if solution is not None:
        item["explanation"] = solution
    return item


class TestPersistedSchemaVersionDetection:
    def test_v2_meta_is_detected(self) -> None:
        assert _is_persisted_schema_v2({"schemaVersion": "2"}) is True

    @pytest.mark.parametrize("meta", [{"schemaVersion": "1"}, {}, None, "2", 2])
    def test_anything_else_is_not_v2(self, meta: object) -> None:
        """Ambiguity must fail safe toward the stricter legacy requirement."""
        assert _is_persisted_schema_v2(meta) is False


class TestSolutionRequirementBySchemaVersion:
    def test_v2_without_both_explanation_snapshots_fails(self) -> None:
        contract = validate_persisted_playable_question(
            _item("2", solution=None),
            expected_question_type="mcq",
            expected_language="english",
            solution_required=(True and not _is_persisted_schema_v2({"schemaVersion": "2"})),
        )
        assert not contract.valid
        assert contract.reason_code == "ANSWER_CONTRACT_MISMATCH"

    def test_v2_with_both_explanation_snapshots_passes(self) -> None:
        contract = validate_persisted_playable_question(
            _item("2", solution="Two plus two is four."),
            expected_question_type="mcq",
            expected_language="english",
            solution_required=False,
        )
        assert contract.valid, contract.reason_code

    def test_v1_without_solution_still_fails(self) -> None:
        contract = validate_persisted_playable_question(
            _item("1", solution=None),
            expected_question_type="mcq",
            expected_language="english",
            solution_required=(True and not _is_persisted_schema_v2({"schemaVersion": "1"})),
        )
        assert not contract.valid
        assert contract.reason_code == "QUESTION_SOLUTION_REQUIRED"

    def test_v1_with_solution_passes(self) -> None:
        contract = validate_persisted_playable_question(
            _item("1", solution="Two plus two is four."),
            expected_question_type="mcq",
            expected_language="english",
            solution_required=True,
        )
        assert contract.valid, contract.reason_code

    def test_mixed_set_is_judged_per_item(self) -> None:
        """A v1 item in the same set must not be exempted by a v2 sibling."""
        results = []
        for version, solution in (("2", "Two plus two is four."), ("1", None)):
            meta = {"schemaVersion": version}
            results.append(
                validate_persisted_playable_question(
                    _item(version, solution=solution),
                    expected_question_type="mcq",
                    expected_language="english",
                    solution_required=(True and not _is_persisted_schema_v2(meta)),
                ).valid
            )
        assert results == [True, False]

    def test_shared_validator_still_requires_a_solution_when_requested(self) -> None:
        """The global validator still enforces the requirement when asked to."""
        contract = validate_persisted_playable_question(
            _item("2", solution="Two plus two is four."),
            expected_question_type="mcq",
            expected_language="english",
            solution_required=True,
        )
        assert contract.valid, contract.reason_code


class TestPlannerValidationReasonCodes:
    """Each blueprint contract invariant must be identifiable from logs.

    Every one of these previously collapsed into the catch-all PLANNER_SCHEMA_INVALID,
    which is why the production planner failures in 20Q run 1 could not be diagnosed
    from the logs at all. Logging only — no planner behaviour changes.
    """

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("planner slots must be intentionally distinct", "PLANNER_SLOTS_NOT_DISTINCT"),
            ("schema-v2 blueprint requires planner_family", "PLANNER_FAMILY_MISSING"),
            ("unsupported planner-slot subject", "PLANNER_UNSUPPORTED_SUBJECT"),
            ("unsupported demand-bucket subject", "PLANNER_UNSUPPORTED_SUBJECT"),
            ("planner-slot exam IDs must be unique", "PLANNER_DUPLICATE_EXAM_ID"),
            (
                "planner-slot variation constraints must be unique",
                "PLANNER_DUPLICATE_VARIATION_CONSTRAINT",
            ),
            (
                "schema-v1 blueprint cannot contain schema-v2 planner fields",
                "PLANNER_SCHEMA_VERSION_MISMATCH",
            ),
            ("blueprint requires compatibility buckets", "PLANNER_BUCKETS_MISSING"),
        ],
    )
    def test_contract_invariants_map_to_distinct_codes(
        self, message: str, expected: str
    ) -> None:
        from features.practice_generation.planning import _planner_validation_reason

        reason, _ = _planner_validation_reason(ValueError(message), raw="{}")
        assert reason == expected

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("schema-v2 slot count must equal accepted_count", "PLANNER_SLOT_COUNT_MISMATCH"),
            ("planner slot IDs must be unique", "PLANNER_DUPLICATE_SLOT"),
            (
                "planner slot metadata must use canonical identifiers",
                "PLANNER_SLOT_IDENTIFIER_INVALID",
            ),
        ],
    )
    def test_existing_codes_are_unchanged(self, message: str, expected: str) -> None:
        from features.practice_generation.planning import _planner_validation_reason

        reason, _ = _planner_validation_reason(ValueError(message), raw="{}")
        assert reason == expected

    def test_genuinely_unknown_message_still_falls_through(self) -> None:
        from features.practice_generation.planning import _planner_validation_reason

        reason, _ = _planner_validation_reason(ValueError("something new"), raw="{}")
        assert reason == "PLANNER_SCHEMA_INVALID"

    def test_every_blueprint_invariant_has_a_specific_code(self) -> None:
        """Guard against a new invariant silently rejoining the catch-all."""
        import inspect
        import re

        from features.practice_generation import schemas
        from features.practice_generation.planning import _planner_validation_reason

        source = inspect.getsource(schemas.PracticeBlueprint._validate_distribution)
        messages = re.findall(r'raise ValueError\("([^"]+)"\)', source)
        assert messages, "expected blueprint contract invariants"
        unmapped = [
            message
            for message in messages
            if _planner_validation_reason(ValueError(message), raw="{}")[0]
            == "PLANNER_SCHEMA_INVALID"
        ]
        assert not unmapped, f"invariants with no specific reason code: {unmapped}"


class TestOwnedErrorMessageDisclosure:
    """Only our own literal contract messages may be logged verbatim.

    The planner blocker survived one logging change because the diagnostic discards
    `str(error)` entirely. Disclosing it is safe only for messages this package raises
    as literals — never for anything that could carry model output or student text.
    """

    def test_every_owned_literal_is_disclosable(self) -> None:
        from features.practice_generation.planning import (
            _owned_error_message,
            _owned_value_error_messages,
        )

        owned = _owned_value_error_messages()
        assert owned, "expected the package to raise literal ValueErrors"
        for message in owned:
            assert _owned_error_message(ValueError(message)) == message

    def test_blueprint_contract_invariants_are_all_owned(self) -> None:
        """The invariants we are hunting must be disclosable when they fire."""
        import inspect
        import re as _re

        from features.practice_generation import schemas
        from features.practice_generation.planning import _owned_error_message

        source = inspect.getsource(schemas.PracticeBlueprint._validate_distribution)
        for message in _re.findall(r'raise ValueError\("([^"]+)"\)', source):
            assert _owned_error_message(ValueError(message)) == message

    @pytest.mark.parametrize(
        "message",
        [
            "Student asked: what is the capital of France?",
            "planner returned: {'slots': [...]}",
            "unexpected value: 42",
            "slot-001 conflicts with slot-002",
            "",
        ],
    )
    def test_unowned_or_dynamic_messages_stay_hidden(self, message: str) -> None:
        from features.practice_generation.planning import _owned_error_message

        assert _owned_error_message(ValueError(message)) == ""

    def test_interpolated_variants_of_owned_messages_stay_hidden(self) -> None:
        """Exact match only — a message that merely contains a literal is not owned."""
        from features.practice_generation.planning import _owned_error_message

        assert (
            _owned_error_message(
                ValueError("planner slots must be intentionally distinct: slot-002")
            )
            == ""
        )

    def test_diagnostic_carries_the_message_only_when_owned(self) -> None:
        from features.practice_generation.planning import _planner_validation_diagnostic

        owned = _planner_validation_diagnostic(
            ValueError("planner slots must be intentionally distinct"),
            raw="{}",
            attempt=0,
            phase="initial",
            duration_ms=1,
        )
        assert owned.owned_message == "planner slots must be intentionally distinct"

        foreign = _planner_validation_diagnostic(
            ValueError("some model produced this text"),
            raw="{}",
            attempt=0,
            phase="initial",
            duration_ms=1,
        )
        assert foreign.owned_message == ""
        assert foreign.reason_code == "PLANNER_SCHEMA_INVALID"


class TestOwnedErrorOriginDisclosure:
    """The raise site identifies a dynamic invariant without disclosing its message.

    The message allowlist cannot cover a ValueError raised by an enum coercion or a
    stdlib call, and those carry interpolated values we must never log. The code
    location is our own metadata and is safe.
    """

    def test_origin_points_at_application_code(self) -> None:
        from features.practice_generation.planning import (
            _owned_error_origin,
            apply_system_bucket_policy,
        )

        try:
            apply_system_bucket_policy(None, None)  # type: ignore[arg-type]
        except Exception as error:  # noqa: BLE001
            origin = _owned_error_origin(error)
        assert origin.endswith(tuple(f":{n}" for n in range(1, 100000)))
        assert origin.startswith("planning.py:")

    def test_origin_is_empty_without_a_traceback(self) -> None:
        from features.practice_generation.planning import _owned_error_origin

        assert _owned_error_origin(ValueError("never raised")) == ""

    def test_origin_never_contains_message_text(self) -> None:
        from features.practice_generation.planning import _owned_error_origin

        secret = "student asked about a sensitive topic"
        try:
            raise ValueError(secret)
        except ValueError as error:
            origin = _owned_error_origin(error)
        assert secret not in origin
        assert origin.count(":") == 1
