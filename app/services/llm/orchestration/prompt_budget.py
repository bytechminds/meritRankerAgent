"""Conservative, provider-neutral prompt input measurement helpers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PromptInputBudget(BaseModel):
    """Component-level estimated input budget recorded before a model call."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    static_instruction_tokens: int = Field(alias="staticInstructionTokens", ge=0)
    current_question_tokens: int = Field(
        default=0,
        alias="currentQuestionTokens",
        ge=0,
    )
    exam_context_tokens: int = Field(alias="examContextTokens", ge=0)
    slot_context_tokens: int = Field(alias="slotContextTokens", ge=0)
    pattern_context_tokens: int = Field(alias="patternContextTokens", ge=0)
    reference_tokens: int = Field(alias="referenceTokens", ge=0)
    other_dynamic_tokens: int = Field(alias="otherDynamicTokens", ge=0)
    total_input_tokens: int = Field(alias="totalInputTokens", ge=0)
    max_input_tokens: int | None = Field(default=None, alias="maxInputTokens", ge=1)
    within_budget: bool = Field(alias="withinBudget")


def estimate_text_tokens(text: str | None) -> int:
    """Return a deliberately conservative token estimate without a tokenizer call."""
    if not text:
        return 0
    return (len(text.encode("utf-8")) + 2) // 3


def measure_prompt_input(
    *,
    static_instructions: str | None = None,
    current_question: str | None = None,
    exam_context: str | None = None,
    slot_context: str | None = None,
    pattern_context: str | None = None,
    references: str | None = None,
    other_dynamic_input: str | None = None,
    max_input_tokens: int | None = None,
) -> PromptInputBudget:
    """Measure independently assembled prompt components before model execution."""
    static_instruction_tokens = estimate_text_tokens(static_instructions)
    current_question_tokens = estimate_text_tokens(current_question)
    exam_context_tokens = estimate_text_tokens(exam_context)
    slot_context_tokens = estimate_text_tokens(slot_context)
    pattern_context_tokens = estimate_text_tokens(pattern_context)
    reference_tokens = estimate_text_tokens(references)
    other_dynamic_tokens = estimate_text_tokens(other_dynamic_input)
    total_input_tokens = (
        static_instruction_tokens
        + current_question_tokens
        + exam_context_tokens
        + slot_context_tokens
        + pattern_context_tokens
        + reference_tokens
        + other_dynamic_tokens
    )
    return PromptInputBudget(
        staticInstructionTokens=static_instruction_tokens,
        currentQuestionTokens=current_question_tokens,
        examContextTokens=exam_context_tokens,
        slotContextTokens=slot_context_tokens,
        patternContextTokens=pattern_context_tokens,
        referenceTokens=reference_tokens,
        otherDynamicTokens=other_dynamic_tokens,
        totalInputTokens=total_input_tokens,
        maxInputTokens=max_input_tokens,
        withinBudget=(
            max_input_tokens is None or total_input_tokens <= max_input_tokens
        ),
    )
