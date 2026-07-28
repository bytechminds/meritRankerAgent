"""Dedicated prompt composition for image-question classification."""

from __future__ import annotations

from services.prompt_loader import load_prompt

PROMPT_VERSION = "image-question-classifier-v5"
SCHEMA_VERSION = "image-question-classification-schema-v2"


def build_image_classifier_prompt() -> str:
    """Compose shared semantic rules with the minimal image overlay."""
    return "\n\n".join(
        (
            load_prompt("classification_semantics"),
            load_prompt("image_question_classifier"),
        )
    )
