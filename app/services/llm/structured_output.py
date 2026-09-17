"""Shared reliability boundary for machine-readable LLM responses.

Every structured consumer resolves provider content through the same stages:

    provider content
      -> syntax parse            (shared strict JSON parser)
      -> representation          (opt-in, per field, never global)
      -> strict schema           (the caller's own Pydantic model)
      -> typed diagnostic        (safe metadata only, on failure)

The split exists because a representation difference and a semantic violation are
different failures with different correct responses, and collapsing them loses a
valid model decision. Representation tolerance is therefore opt-in at the field
that owns it: a schema states which equivalences it accepts, and every other field
stays exactly as strict as it was.

This layer knows nothing about subjects, topics or questions. Business meaning
belongs to the target schema and the domain logic behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, BeforeValidator, ValidationError

from services.doubt_solver.classifier_json import (
    ClassifierJsonError,
    parse_classifier_json_strict,
)

TModel = TypeVar("TModel", bound=BaseModel)

# Which layer rejected the response. Mirrors the Practice planner's vocabulary
# (PlannerValidationDiagnostic.validation_stage) so both features report failures
# in one language. "provider" is not here: it is owned by the caller, because it
# happens before any content exists to parse.
StructuredOutputStage = Literal["parse", "schema", "contract", "internal"]

PROVIDER_FAILURE = "provider_failure"
INTERNAL_FAILURE = "internal_structured_output_failure"

_FAILURE_KINDS: dict[StructuredOutputStage, str] = {
    "parse": "structured_output_parse_failure",
    "schema": "structured_output_schema_failure",
    "contract": "structured_output_contract_failure",
    "internal": INTERNAL_FAILURE,
}


@dataclass(frozen=True)
class StructuredOutputDiagnostic:
    """Allowlisted failure metadata. Never carries model or student content."""

    stage: StructuredOutputStage
    schema_name: str
    error_count: int = 1
    field_paths: tuple[str, ...] = ()
    error_types: tuple[str, ...] = ()

    @property
    def failure_kind(self) -> str:
        return _FAILURE_KINDS[self.stage]

    def as_event_details(self) -> dict[str, object]:
        """Render for the shared structured logger.

        Paths and types are joined rather than passed as lists: the event
        sanitizer replaces any non-scalar with its type name, which would log the
        literal string "list" instead of the fields that actually failed.
        """
        return {
            "failureKind": self.failure_kind,
            "validationStage": self.stage,
            "schemaName": self.schema_name,
            "validationErrorCount": self.error_count,
            "fieldPaths": ",".join(self.field_paths)[:512],
            "errorTypes": ",".join(self.error_types)[:256],
        }


class StructuredOutputError(ValueError):
    """One structured response could not be resolved to its canonical model."""

    def __init__(self, diagnostic: StructuredOutputDiagnostic) -> None:
        super().__init__(
            f"{diagnostic.failure_kind}: {diagnostic.schema_name} "
            f"({diagnostic.error_count} error(s))"
        )
        self.diagnostic = diagnostic


def canonical_number_text(value: int | float | Decimal) -> str:
    """Return the one textual form this layer uses for a numeric value.

    Equal values canonicalize identically, so 55 and 55.0 agree and 0.0, -0 and
    -0.0 all collapse onto "0". Uses the repr round-trip and ``Decimal`` rather
    than %g so no significant digit is dropped and no exponent leaks into a value
    a student could see. ``format(..., "f")`` on a normalized Decimal is the same
    idiom billing and student credits already use for money.

    Raises:
        ValueError: for booleans and non-finite numbers, which carry no scalar
            value even though Python models bool as an int.
    """
    if isinstance(value, bool):
        raise ValueError("A boolean is not a numeric scalar.")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("A non-finite number is not a scalar value.")
        # repr is the shortest string that round-trips the float exactly, so the
        # Decimal below inherits the float's real value and not a wider expansion.
        decimal_value = Decimal(repr(value))
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("A non-finite number is not a scalar value.")
        decimal_value = value
    else:
        decimal_value = Decimal(value)
    if decimal_value == 0:
        return "0"
    return format(decimal_value.normalize(), "f")


def _canonical_scalar_string(value: Any) -> Any:
    """Normalize only the representations a scalar display value may take.

    Anything else is returned untouched so the field's own strict validation
    rejects it with its normal error. This layer never invents a value, and it
    never converts a bool, null, list, object or non-finite number into text.
    """
    if isinstance(value, str):
        # Length, emptiness and stripping stay with the field that declared them.
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        try:
            return canonical_number_text(value)
        except (ValueError, InvalidOperation):
            return value
    return value


CanonicalScalarString = Annotated[str, BeforeValidator(_canonical_scalar_string)]
"""A string field that also accepts the numeric spelling of the same scalar.

Opt in per field. A plain ``str`` field stays strict and still rejects numbers.
"""


def _validation_diagnostic(
    error: ValidationError,
    *,
    schema_name: str,
) -> StructuredOutputDiagnostic:
    """Extract safe metadata from a Pydantic failure, dropping every value."""
    errors = error.errors(include_url=False)
    field_paths = tuple(
        sorted({".".join(str(part) for part in item.get("loc", ())) or "$" for item in errors})
    )
    error_types = tuple(sorted({str(item.get("type") or "validation_error") for item in errors}))
    # A model- or field-validator that raised surfaces as "value_error": the shape
    # was right and an invariant rejected it, which is a contract failure rather
    # than a schema failure.
    stage: StructuredOutputStage = (
        "contract" if error_types and set(error_types) == {"value_error"} else "schema"
    )
    return StructuredOutputDiagnostic(
        stage=stage,
        schema_name=schema_name,
        error_count=len(errors),
        field_paths=field_paths,
        error_types=error_types,
    )


def parse_structured_output(
    content: str,
    schema: type[TModel],
    *,
    schema_name: str | None = None,
) -> TModel:
    """Resolve provider content to ``schema``, or raise StructuredOutputError.

    Deterministic and local: no provider call, no retry, no network. A response
    that only differs in an opted-in representation resolves here instead of
    costing a second model call.
    """
    name = schema_name or schema.__name__
    try:
        payload, _recovered = parse_classifier_json_strict(content)
    except ClassifierJsonError as exc:
        raise StructuredOutputError(
            StructuredOutputDiagnostic(
                stage="parse",
                schema_name=name,
                field_paths=("$",),
                error_types=(exc.error_type,),
            )
        ) from exc
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        raise StructuredOutputError(_validation_diagnostic(exc, schema_name=name)) from exc
