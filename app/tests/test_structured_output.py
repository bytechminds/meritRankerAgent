"""Contract tests for the shared LLM structured-output boundary.

Deliberately subject-independent: the layer under test must never know what an
academic subject, topic or question is, so nothing here references one.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

from services.llm.structured_output import (
    CanonicalScalarString,
    StructuredOutputError,
    canonical_number_text,
    parse_structured_output,
)


class _Sample(BaseModel):
    """One opted-in scalar beside fields that must stay strict."""

    model_config = {"extra": "forbid", "str_strip_whitespace": True}

    decision: str = Field(min_length=1)
    display: CanonicalScalarString = Field(min_length=1)
    strict_text: str = Field(default="ok", min_length=1)
    flag: bool = True


class _Contract(BaseModel):
    left: CanonicalScalarString
    right: CanonicalScalarString

    @model_validator(mode="after")
    def _must_agree(self) -> _Contract:
        if self.left != self.right:
            raise ValueError("left and right must agree")
        return self


def _payload(**overrides: object) -> str:
    import json

    body: dict[str, object] = {"decision": "ACCEPT", "display": "x"}
    body.update(overrides)
    return json.dumps(body)


# --------------------------------------------------------------------------
# Numeric canonicalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (0.0, "0"),
        (-0.0, "0"),
        (Decimal("-0"), "0"),
        (55, "55"),
        (55.0, "55"),
        (12.50, "12.5"),
        (-4, "-4"),
        (-4.0, "-4"),
        (1e3, "1000"),
        (Decimal("1E+3"), "1000"),
        (2.5e-8, "0.000000025"),
        (1e30, "1" + "0" * 30),
    ],
)
def test_canonical_number_text_is_deterministic(value: object, expected: str) -> None:
    assert canonical_number_text(value) == expected  # type: ignore[arg-type]


def test_equal_values_share_one_canonical_form() -> None:
    assert canonical_number_text(55) == canonical_number_text(55.0)
    assert canonical_number_text(0.0) == canonical_number_text(-0.0) == "0"


def test_precision_is_not_silently_widened_or_truncated() -> None:
    assert canonical_number_text(0.1 + 0.2) == "0.30000000000000004"
    assert canonical_number_text(1234567.0) == "1234567"


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), float("-inf")])
def test_canonical_number_text_rejects_non_scalars(value: object) -> None:
    with pytest.raises(ValueError):
        canonical_number_text(value)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Opted-in normalization: accepted representations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"55"', "55"),
        ("55", "55"),
        ("55.0", "55"),
        ("12.5", "12.5"),
        ("-4", "-4"),
        ("0", "0"),
        ('"50 km/h"', "50 km/h"),
        ('"Option B"', "Option B"),
        ('"Article 21"', "Article 21"),
        ('"photosynthesis"', "photosynthesis"),
        ('"  spaced  "', "spaced"),
    ],
)
def test_equivalent_representations_reach_one_canonical_dto(raw: str, expected: str) -> None:
    parsed = parse_structured_output(f'{{"decision":"ACCEPT","display":{raw}}}', _Sample)
    assert parsed.display == expected


def test_numeric_and_string_spellings_are_the_same_object() -> None:
    from_number = parse_structured_output('{"decision":"ACCEPT","display":55}', _Sample)
    from_string = parse_structured_output('{"decision":"ACCEPT","display":"55"}', _Sample)
    assert from_number == from_string


# --------------------------------------------------------------------------
# Opted-in normalization: rejected representations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["true", "false", "null", "[]", "{}", '["55"]', '{"value":55}', "[[1,2],[3]]"],
)
def test_opted_in_field_still_rejects_non_scalar_values(raw: str) -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output(f'{{"decision":"ACCEPT","display":{raw}}}', _Sample)
    assert excinfo.value.diagnostic.stage == "schema"
    assert excinfo.value.diagnostic.field_paths == ("display",)


def test_empty_string_is_still_rejected_by_the_fields_own_constraint() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output('{"decision":"ACCEPT","display":""}', _Sample)
    assert excinfo.value.diagnostic.error_types == ("string_too_short",)


# --------------------------------------------------------------------------
# Normalization is explicit, never global — the critical property
# --------------------------------------------------------------------------


def test_a_strict_string_field_still_rejects_a_number() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output(_payload(strict_text=55), _Sample)
    assert excinfo.value.diagnostic.field_paths == ("strict_text",)
    assert excinfo.value.diagnostic.error_types == ("string_type",)


def test_bool_is_never_coerced_into_a_strict_string_field() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output(_payload(strict_text=True), _Sample)
    assert excinfo.value.diagnostic.error_types == ("string_type",)


def test_bool_field_keeps_boolean_semantics() -> None:
    assert parse_structured_output(_payload(flag=False), _Sample).flag is False


def test_normalization_does_not_leak_to_other_models() -> None:
    class _Untouched(BaseModel):
        value: str

    with pytest.raises(ValidationError):
        _Untouched.model_validate({"value": 55})


# --------------------------------------------------------------------------
# Schema / contract / parse taxonomy
# --------------------------------------------------------------------------


def test_missing_required_field_is_a_schema_failure() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output('{"display":"55"}', _Sample)
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.stage == "schema"
    assert diagnostic.failure_kind == "structured_output_schema_failure"
    assert diagnostic.field_paths == ("decision",)


def test_extra_field_policy_is_unchanged() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output(_payload(surprise="x"), _Sample)
    assert excinfo.value.diagnostic.error_types == ("extra_forbidden",)


def test_cross_field_contradiction_is_a_contract_failure() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output('{"left":55,"right":"56"}', _Contract)
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.stage == "contract"
    assert diagnostic.failure_kind == "structured_output_contract_failure"


def test_normalization_happens_before_domain_validation() -> None:
    # 55 and "55" must agree once canonical, proving order of operations.
    assert parse_structured_output('{"left":55,"right":"55"}', _Contract).left == "55"


@pytest.mark.parametrize(
    ("content", "error_type"),
    [
        ("", "empty_output"),
        ("   ", "empty_output"),
        ("not json", "json_decode_error"),
        ('{"decision":"ACCEPT","display":"55"} trailing prose', "extra_trailing_text"),
        ('["not","an","object"]', "not_object"),
    ],
)
def test_malformed_content_is_a_typed_parse_failure(content: str, error_type: str) -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output(content, _Sample)
    diagnostic = excinfo.value.diagnostic
    assert diagnostic.stage == "parse"
    assert diagnostic.failure_kind == "structured_output_parse_failure"
    assert diagnostic.error_types == (error_type,)


def test_existing_fence_recovery_is_preserved() -> None:
    fenced = '```json\n{"decision":"ACCEPT","display":55}\n```'
    assert parse_structured_output(fenced, _Sample).display == "55"


# --------------------------------------------------------------------------
# Diagnostics carry metadata only
# --------------------------------------------------------------------------


def test_diagnostic_details_are_scalar_and_carry_no_model_content() -> None:
    with pytest.raises(StructuredOutputError) as excinfo:
        parse_structured_output('{"decision":"ACCEPT","display":[]}', _Sample)
    details = excinfo.value.diagnostic.as_event_details()
    assert details == {
        "failureKind": "structured_output_schema_failure",
        "validationStage": "schema",
        "schemaName": "_Sample",
        "validationErrorCount": 1,
        "fieldPaths": "display",
        "errorTypes": "string_type",
    }
    assert all(isinstance(value, (str, int)) for value in details.values())
