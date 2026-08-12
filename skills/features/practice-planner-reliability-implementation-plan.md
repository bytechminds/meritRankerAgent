# Implementation Plan: Practice Planner Reliability

Keep the existing `BlueprintManager` boundary. Normalize supported multi-topic separators only
for deterministic fallback IDs, round-robin slots across requested topics, validate the resulting
`PracticeBlueprint`, and retain only schema/error metadata for its one repair attempt. Map an
invalid fallback to `BlueprintPlanningError` and terminalize it in the existing orchestrator.

No schema, model, provider-selection, persistence, or topology change is planned.
