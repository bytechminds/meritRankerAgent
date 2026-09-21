"""Shared, machine-readable Practice limitation contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from practice_limits import MAX_PRACTICE_QUESTIONS, MAX_REQUESTED_PRACTICE_QUESTIONS


class PracticeLimitation(BaseModel):
    """A compact product limitation applied to one Practice request."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    applied: Literal[True] = True
    type: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    requested_value: int = Field(
        ge=1,
        le=MAX_REQUESTED_PRACTICE_QUESTIONS,
        alias="requestedValue",
        serialization_alias="requestedValue",
    )
    effective_value: int = Field(
        ge=1,
        le=MAX_PRACTICE_QUESTIONS,
        alias="effectiveValue",
        serialization_alias="effectiveValue",
    )
    maximum_value: int = Field(
        ge=1,
        le=MAX_PRACTICE_QUESTIONS,
        alias="maximumValue",
        serialization_alias="maximumValue",
    )
    message_key: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        alias="messageKey",
        serialization_alias="messageKey",
    )

    @model_validator(mode="after")
    def _validate_effective_value(self) -> PracticeLimitation:
        if self.requested_value <= self.maximum_value:
            raise ValueError("limitation requires a requested value above its maximum")
        if self.effective_value != self.maximum_value:
            raise ValueError("limitation effective value must equal its maximum")
        return self
