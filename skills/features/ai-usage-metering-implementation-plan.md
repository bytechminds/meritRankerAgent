# Implementation Plan: Centralized AI Usage Metering V1

## Architecture Decision

The existing `record_llm_call()` observability sink remains the sole metering capture point. The
new operation accumulator lives in the existing execution context, outside LangGraph state and
academic services. `ProviderAdapterExecutor`, the legacy router, and image classifier already
converge on this sink.

## Files

- Add billing Pydantic contracts and a cached YAML loader/calculator.
- Extend usage records with the resolved model alias and append records to the active operation
  accumulator after existing safe call recording.
- Extend `ExecutionContext` with the operation descriptor/accumulator and preserve it in nested
  scopes.
- Extend request, streaming, and existing Practice background lifecycle boundaries only.
- Copy the existing context for each existing parallel Practice group; do not create a new worker.
- Update the existing model-pricing YAML with V1 feature/credit controls and only currently
  project-supplied rates.

## Failure Policy

Configuration structure errors fail startup. Missing usage or a profile is a shadow-only,
fail-open incomplete accounting result. Finalization and logging errors never affect academics.

## Compatibility

No public API or graph-state contract changes. Practice maps quick/similar/quiz to
`quick_practice`, and larger test artifacts to `mock_test`; ordinary doubt uses `doubt`.
