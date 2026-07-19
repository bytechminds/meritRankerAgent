"""Dedicated prompt composition for image-question classification."""

from __future__ import annotations

import json

from schemas.doubt_solver import QueryClassification
from schemas.image_question_classification import ImageProviderOutput
from services.prompt_loader import load_prompt

PROMPT_VERSION = "image-question-classifier-v2"
SCHEMA_VERSION = "image-question-classification-schema-v2"


def build_image_classifier_prompt() -> str:
    """Compose the static instructions with the authoritative live schemas."""
    classifier_schema = QueryClassification.model_json_schema()
    output_schema = ImageProviderOutput.model_json_schema()
    return "\n\n".join(
        (
            load_prompt("image_question_classifier"),
            "Authoritative existing classification schema:\n"
            + json.dumps(classifier_schema, separators=(",", ":"), sort_keys=True),
            "Required image-classifier output schema:\n"
            + json.dumps(output_schema, separators=(",", ":"), sort_keys=True),
        )
    )
