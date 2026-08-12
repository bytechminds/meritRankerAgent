# Performance-Cost Review: Practice Generation Recovery

## Recommendation

**PASS WITH NOTES** for the scoped implementation.

## Execution Type

Background per-request practice generation after immediate user acknowledgement.

## Critical Path

Planning, reuse lookup, bounded generation/verification waves, persistence, final validation, and
optional metadata-only repair.

## Network / I/O Call Inventory

Existing LLM and DynamoDB/AppSync calls remain. Recovery adds one Question UpdateItem per incorrect
slot and one authoritative parent recalculation, only on the proven first-pass mismatch.

## Model Call Analysis

No new model step. Maximum content attempts remain initial + one repair + one replacement for each
unresolved slot. Accepted siblings are not resent.

## Retry / Fallback Cost Analysis

Final metadata recovery is exactly once and zero-LLM. Provider fallbacks remain separate and
bounded by existing policy.

## Latency Risks

Conditional updates add minimal latency only on a recoverable final mismatch. No unbounded loop.

## Cost Risks

Local replay used 10 calls, 8,476 input, and 10,008 output tokens versus the failed run's 12 calls,
10,113 input, and 21,663 output. Azure cost remains unavailable because pricing is intentionally
not guessed.

## Scalability / Quota Risks

Unchanged. Recovery writes scale with failed slot count and are bounded by accepted count.

## Observability Requirements

Attempt, slot count, recoverability, lifecycle, and central provider usage events are present.

## Final Decision Reason

The change adds no LLM call or retry loop and materially reduced measured replay usage while
preserving academic verification.
