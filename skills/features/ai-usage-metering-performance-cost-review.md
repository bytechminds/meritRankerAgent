# Performance-Cost Review: Centralized AI Usage Metering V1

## Recommendation

**PASS WITH NOTES**

## Execution Type

**Type:** synchronous, streaming, and retained background Practice.  
**User-facing?** Summary computation is not user-visible.  
**Per-request or batch?** Per-operation; existing Practice waves may run two groups in parallel.

## Critical Path

| Step | Type | Blocking? |
|---|---|---|
| Existing provider call | Network | Existing behavior; unchanged. |
| Central record append | CPU + lock | Yes, bounded constant work. |
| Terminal calculation/log emission | CPU + logging | Existing terminal boundary; no I/O added. |

## Network / I/O Call Inventory

| Call | Service | Per-operation count | In a loop? | Timeout set? |
|---|---|---|---|---|
| New billing network call | None | 0 | No | N/A |
| YAML load | Local file, process-cached | At most once per process/path | No | N/A |

## Model Call Analysis

The feature adds zero model calls. It records all existing provider attempts, including fallbacks
and bounded Practice capacity paths.

## Retry / Fallback Cost Analysis

Retries and fallbacks are intentionally counted as actual provider attempts if usage is present.
Missing provider usage remains incomplete rather than fabricated.

## Latency Risks

| Risk | Affected path | Label | Mitigation |
|---|---|---|---|
| Shared accumulator lock under parallel Practice groups | Background Practice wave | [PERFORMANCE RISK] | Lock scope is one list append/snapshot; no remote I/O. |

## Cost Risks

| Risk | Trigger | Label | Mitigation |
|---|---|---|---|
| Unreviewed provider deployment price | Azure/DeepSeek/Gemini 2.5 route | [COST RISK] | Mark summary incomplete; do not infer a rate. |

## Scalability Risks

| Risk | Label | Deferred? |
|---|---|---|
| In-memory accumulator is lost on process termination | [SCALE BLOCKER] for financial accounting | [DEFER] — shadow observability only. |

## Observability Requirements

| Signal | Implementation | Status |
|---|---|---|
| Provider usage | Existing `llm_call_usage` and central usage records | Implemented. |
| Operation summary | `AI_USAGE_SUMMARY` with complete/incomplete flag | Implemented. |
| Missing rate | `COST_PROFILE_MISSING` safe diagnostic | Implemented. |

## Final Decision Reason

> The feature adds no remote call, worker, queue, or model attempt. It places bounded local work at
> an existing central sink and makes uncertain price data visible rather than creating a false cost.
