"""Inbound image reference contract shared by request and classifier schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ImageInput(BaseModel):
    """Client-neutral image reference accepted by doubt-solver requests."""

    source: Literal["upload", "camera", "screenshot"]
    mime_type: str = Field(alias="mimeType", min_length=1, max_length=64)
    file_name: str | None = Field(default=None, alias="fileName", max_length=255)
    storage_key: str | None = Field(default=None, alias="storageKey", max_length=1024)
    signed_url: str | None = Field(default=None, alias="signedUrl", max_length=4096)
    image_bytes: bytes | None = Field(default=None, alias="bytes", repr=False)
    base64: str | None = Field(default=None, min_length=1, max_length=16_000_000, repr=False)
    width: int | None = Field(default=None, gt=0, le=100_000)
    height: int | None = Field(default=None, gt=0, le=100_000)

    model_config = {
        "populate_by_name": True,
        "str_strip_whitespace": True,
        "extra": "forbid",
    }

    @model_validator(mode="after")
    def _require_one_payload_source(self) -> ImageInput:
        sources = (self.image_bytes, self.base64, self.storage_key, self.signed_url)
        if sum(value is not None for value in sources) != 1:
            raise ValueError(
                "Image input must provide exactly one of bytes, base64, storageKey, or signedUrl."
            )
        return self
