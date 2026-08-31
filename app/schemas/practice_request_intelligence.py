"""Practice Request Intelligence contract — interpretation only, never planning.

The model interprets what the student explicitly asked for; deterministic code
validates the interpretation and every downstream distribution decision. The
static JSON Schema below is the exact grammar handed to the provider, so it is
versioned (``practice_request_intelligence_v4``) and never generated per request.

Topics are grounded by opaque token id, not by text the model retypes and not by
numeric range arithmetic. The model is given the query already split into deterministic
tokens, each labelled ``T0``, ``T1`` …, and selects the ids belonging to a topic; the
exact source text is reconstructed from the original query afterwards.

Both earlier grounding designs failed on the model side rather than ours. Under v2 a
model asked to copy "औसत" returned "ओसत". Under v3 a model asked for a numeric
``startToken``/``endToken`` range under-counted a four-token span and every later topic
silently shifted by one. Selecting labels it can see removes both: there is nothing to
transcribe and nothing to count.

Difficulty is a single mutually exclusive object rather than three flat fields.
The v1 contract allowed the grammar to emit combinations that carry no meaning —
a MIXED request with one stated level, an UNSPECIFIED request with one — and left
deterministic code to reject them after the fact. Here the grammar itself cannot
express them: each mode is its own closed ``anyOf`` branch pinned by a ``const``,
so a level exists only under SINGLE and a distribution only under CUSTOM.

Only JSON Schema features that Bedrock structured outputs support are used here.
Numeric and size bounds are intentionally absent — business bounds are enforced
deterministically in ``features.practice_generation.request_intelligence`` after
the response is deserialized.

This module is shared: the provider adapter reads the schema, and the Practice
feature reads the model. It therefore holds no provider and no feature imports.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PRACTICE_REQUEST_INTELLIGENCE_SCHEMA_NAME = "practice_request_intelligence_v4"

InterpretationStatus = Literal["RESOLVED", "BROAD", "AMBIGUOUS"]
DifficultyMode = Literal["UNSPECIFIED", "SINGLE", "MIXED", "CUSTOM"]
InterpretedDifficulty = Literal["BASIC", "INTERMEDIATE", "ADVANCED"]


class InterpretedTopic(BaseModel):
    """One explicitly requested topic, grounded by the token ids it covers.

    ``source_text`` is never supplied by the model. It is reconstructed from the
    original query once the selection has been validated, and is excluded from the
    wire schema entirely.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    token_ids: list[str] = Field(alias="tokenIds")
    normalized_name: str = Field(alias="normalizedName")
    source_text: str = Field(default="", exclude=True)


# The difficulty branches mirror the closed grammar exactly: extras are forbidden
# so a payload the schema cannot express is also one the deserializer refuses.
_CLOSED_BRANCH = ConfigDict(populate_by_name=True, extra="forbid")


class InterpretedDifficultyDistribution(BaseModel):
    """Explicit per-level counts. Arithmetic is validated deterministically."""

    model_config = _CLOSED_BRANCH

    basic: int
    intermediate: int
    advanced: int

    def total(self) -> int:
        return self.basic + self.intermediate + self.advanced


class UnspecifiedDifficulty(BaseModel):
    """The student stated no difficulty at all."""

    model_config = _CLOSED_BRANCH

    mode: Literal["UNSPECIFIED"]


class SingleDifficulty(BaseModel):
    """The student stated exactly one level, which is therefore required."""

    model_config = _CLOSED_BRANCH

    mode: Literal["SINGLE"]
    level: InterpretedDifficulty


class MixedDifficulty(BaseModel):
    """The student asked for a mix; the planner owns the split, not the model."""

    model_config = _CLOSED_BRANCH

    mode: Literal["MIXED"]


class CustomDifficulty(BaseModel):
    """The student stated per-level counts, which are therefore required."""

    model_config = _CLOSED_BRANCH

    mode: Literal["CUSTOM"]
    distribution: InterpretedDifficultyDistribution


InterpretedDifficultySpec = Annotated[
    UnspecifiedDifficulty | SingleDifficulty | MixedDifficulty | CustomDifficulty,
    Field(discriminator="mode"),
]


class PracticeRequestIntelligence(BaseModel):
    """Schema-constrained interpretation of one free-text Practice request."""

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    interpretation_status: InterpretationStatus = Field(alias="interpretationStatus")
    requested_count: int | None = Field(default=None, alias="requestedCount")
    topics: list[InterpretedTopic] = Field(default_factory=list)
    difficulty: InterpretedDifficultySpec


def _closed(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


_DISTRIBUTION_SCHEMA: dict[str, Any] = _closed(
    {
        "basic": {"type": "integer"},
        "intermediate": {"type": "integer"},
        "advanced": {"type": "integer"},
    },
    ["basic", "intermediate", "advanced"],
)

# Each branch is closed, so a level cannot appear outside SINGLE and a
# distribution cannot appear outside CUSTOM. The grammar, not a later check,
# is what makes the invalid combinations unrepresentable.
_DIFFICULTY_SCHEMA: dict[str, Any] = {
    "anyOf": [
        _closed({"mode": {"const": "UNSPECIFIED"}}, ["mode"]),
        _closed(
            {
                "mode": {"const": "SINGLE"},
                "level": {
                    "type": "string",
                    "enum": ["BASIC", "INTERMEDIATE", "ADVANCED"],
                },
            },
            ["mode", "level"],
        ),
        _closed({"mode": {"const": "MIXED"}}, ["mode"]),
        _closed(
            {"mode": {"const": "CUSTOM"}, "distribution": _DISTRIBUTION_SCHEMA},
            ["mode", "distribution"],
        ),
    ]
}

PRACTICE_REQUEST_INTELLIGENCE_SCHEMA: dict[str, Any] = _closed(
    {
        "interpretationStatus": {
            "type": "string",
            "enum": ["RESOLVED", "BROAD", "AMBIGUOUS"],
        },
        "requestedCount": {"type": ["integer", "null"]},
        "topics": {
            "type": "array",
            "items": _closed(
                {
                    "tokenIds": {"type": "array", "items": {"type": "string"}},
                    "normalizedName": {"type": "string"},
                },
                ["tokenIds", "normalizedName"],
            ),
        },
        "difficulty": _DIFFICULTY_SCHEMA,
    },
    ["interpretationStatus", "requestedCount", "topics", "difficulty"],
)


def practice_request_intelligence_schema_json() -> str:
    """Return the static schema exactly as the provider API expects it (a string)."""
    return json.dumps(PRACTICE_REQUEST_INTELLIGENCE_SCHEMA, separators=(",", ":"))
