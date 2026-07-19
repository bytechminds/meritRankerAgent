"""Optional live Gemini smoke test; excluded unless explicitly enabled."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import config as config_module
from schemas.image_input import ImageInput
from schemas.image_question_classification import ImageClassificationStatus
from services.image_question_classification.factory import build_image_question_classifier


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI_IMAGE_TEST", "false").lower() != "true",
    reason="live Gemini image test is opt-in",
)
def test_live_gemini_image_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    image_path = Path(os.environ["LIVE_GEMINI_IMAGE_PATH"])
    monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "true")
    config_module._settings = None
    settings = config_module.get_settings()
    classifier = build_image_question_classifier(settings)
    result = classifier.classify(
        image=ImageInput(
            source="upload",
            mimeType="image/png",
            bytes=image_path.read_bytes(),
        ),
        instruction=None,
        request_id="live-gemini-image-smoke",
    )
    assert result.status == ImageClassificationStatus.CLASSIFIED
