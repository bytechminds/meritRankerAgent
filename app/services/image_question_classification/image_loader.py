"""Safe image-payload loading boundary."""

from __future__ import annotations

import base64
import binascii
from typing import Protocol

from schemas.image_input import ImageInput
from services.image_question_classification.errors import (
    ImageReferenceUnavailableError,
    InvalidImageError,
)


class ImageReferenceLoader(Protocol):
    def load(self, image: ImageInput, *, timeout_seconds: float) -> bytes:
        """Resolve an approved storage reference without exposing its URL."""


class InlineImageLoader:
    """Loads inline bytes/base64 and refuses arbitrary URL or storage access."""

    def load(self, image: ImageInput, *, timeout_seconds: float) -> bytes:  # noqa: ARG002
        if image.image_bytes is not None:
            return image.image_bytes
        if image.base64 is not None:
            encoded = image.base64
            if encoded.startswith("data:"):
                _, separator, encoded = encoded.partition(",")
                if not separator:
                    raise InvalidImageError("Malformed image data URL.")
            try:
                return base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidImageError("Image base64 payload is invalid.") from exc
        raise ImageReferenceUnavailableError(
            "Image storage references require an approved server-side loader."
        )
