"""Image validation and detail-preserving normalization."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from schemas.image_input import ImageInput
from schemas.image_question_classification import ValidatedImage
from services.image_question_classification.errors import InvalidImageError
from services.image_question_classification.image_loader import ImageReferenceLoader

_FORMAT_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


@dataclass(frozen=True)
class ImageValidationConfig:
    max_image_bytes: int
    max_dimension: int
    min_dimension: int = 64


class ImageInputValidator:
    def __init__(
        self,
        *,
        loader: ImageReferenceLoader,
        config: ImageValidationConfig,
    ) -> None:
        self._loader = loader
        self._config = config

    def validate(self, image_input: ImageInput, *, timeout_seconds: float) -> ValidatedImage:
        supported_mimes = set(_FORMAT_MIME.values())
        declared_mime = image_input.mime_type.lower().split(";", maxsplit=1)[0].strip()
        if declared_mime not in supported_mimes:
            raise InvalidImageError("Unsupported image MIME type.")

        raw = self._loader.load(image_input, timeout_seconds=timeout_seconds)
        if not raw:
            raise InvalidImageError("Image payload is empty.")
        if len(raw) > self._config.max_image_bytes:
            raise InvalidImageError("Image payload exceeds the configured size limit.")

        try:
            with Image.open(io.BytesIO(raw)) as probe:
                actual_mime = _FORMAT_MIME.get(probe.format or "")
                if actual_mime is None or actual_mime != declared_mime:
                    raise InvalidImageError("Image content does not match its MIME type.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise InvalidImageError("Animated or multi-frame images are unsupported.")
                probe.verify()

            with Image.open(io.BytesIO(raw)) as decoded:
                decoded.load()
                normalized = ImageOps.exif_transpose(decoded)
                width, height = normalized.size
                if min(width, height) < self._config.min_dimension:
                    raise InvalidImageError("Image dimensions are too small to read reliably.")

                was_resized = max(width, height) > self._config.max_dimension
                if was_resized:
                    normalized.thumbnail(
                        (self._config.max_dimension, self._config.max_dimension),
                        Image.Resampling.LANCZOS,
                    )
                    width, height = normalized.size

                output = io.BytesIO()
                if actual_mime == "image/jpeg":
                    if normalized.mode not in {"RGB", "L"}:
                        normalized = normalized.convert("RGB")
                    normalized.save(output, format="JPEG", quality=94, optimize=True)
                elif actual_mime == "image/png":
                    normalized.save(output, format="PNG", optimize=True)
                else:
                    normalized.save(output, format="WEBP", quality=95, method=4)
        except InvalidImageError:
            raise
        except (Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError) as exc:
            raise InvalidImageError("Image payload is corrupt or unreadable.") from exc

        content = output.getvalue()
        return ValidatedImage(
            content=content,
            mime_type=actual_mime,
            width=width,
            height=height,
            content_hash=hashlib.sha256(content).hexdigest(),
            was_resized=was_resized,
        )
