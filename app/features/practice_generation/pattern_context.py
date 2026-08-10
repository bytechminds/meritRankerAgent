"""Disabled-by-default PatternGraph plug-in boundary."""

from __future__ import annotations

from typing import Protocol

from features.practice_generation.schemas import PracticeGenerationRequest


class PatternContextProvider(Protocol):
    def discover_catalog(self, request: PracticeGenerationRequest) -> tuple[str, ...]: ...

    def hydrate_context(self, reference_ids: tuple[str, ...]) -> str: ...


class NoOpPatternContextProvider:
    def discover_catalog(self, request: PracticeGenerationRequest) -> tuple[str, ...]:
        return ()

    def hydrate_context(self, reference_ids: tuple[str, ...]) -> str:
        return ""


def build_pattern_context_provider(*, enabled: bool) -> PatternContextProvider:
    if enabled:
        raise RuntimeError("PRACTICE_PATTERN_CONTEXT_ENABLED=true is unavailable in this release.")
    return NoOpPatternContextProvider()
