"""Controlled errors for the image-question classification boundary."""

from __future__ import annotations


class ImageClassificationError(Exception):
    """Base class for controlled image-classification errors."""


class InvalidImageError(ImageClassificationError):
    """Raised when an image fails local validation before provider execution."""


class ImageReferenceUnavailableError(InvalidImageError):
    """Raised when no safe loader can resolve a storage reference."""


class ImageProviderError(ImageClassificationError):
    """Base class for provider failures."""


class ImageProviderTemporaryError(ImageProviderError):
    """Retryable timeout, throttling, connection, or upstream failure."""


class ImageProviderResponseError(ImageProviderError):
    """Provider returned malformed or schema-invalid output."""
