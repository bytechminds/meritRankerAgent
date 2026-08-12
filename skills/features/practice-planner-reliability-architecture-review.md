# Architecture Review: Practice Planner Reliability

## Recommendation

**APPROVE THE SCOPED DESIGN.**

The change remains inside existing Practice planning and observability boundaries. It does not add
a worker, graph edge, database field, public contract, provider, or model policy. Planner failure
is classified at the structured-output boundary, recovery remains bounded, and the orchestrator
continues to own assessment terminal state. Catching only `ProviderExecutionError` preserves the
existing fail-loud policy for configuration and programming failures.

The aggregate worktree is not releasable as one change because unrelated dirty routing, generator,
Pattern, and metering modifications overlap the repository. Isolate this scope before release.
