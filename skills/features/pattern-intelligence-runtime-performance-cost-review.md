# Performance-Cost Review: Pattern Intelligence Runtime

> Role: Performance-Cost Reviewer
> Date: 2026-08-10

## Recommendation

**BLOCK**

## Execution Type

Practice runs in the existing tracked background workflow. Doubt retrieval/generation is
user-facing synchronous or streaming. Pattern work is per request/retrieval group and optional.

## Critical Paths

| Path | Steps |
|---|---|
| Practice | Planner → QuestionBank reuse → Pattern candidate query → Pattern BatchGet → generation → verifier |
| Doubt | Classifier → embedding → S3 Vector query → Pattern BatchGet → optional linked GSI/ColBERT → existing generator |

No additional LLM call is added by Pattern Intelligence.

## Network / I/O Call Inventory

| Call | Per selected runtime call | Bound | Loop risk |
|---|---:|---:|---|
| Embedding | 1 | One query | Practice groups currently resolve serially |
| S3 Vector query | 1 | Max 24 candidates | Same |
| Pattern DynamoDB BatchGet | 1 | Max 24 IDs, one opt-in unprocessed retry | No per-candidate GetItem |
| PatternQuestion GSI | Doubt only, 0–1 | Max 5 projected records | No scan |
| ColBERT rerank | Doubt only, 0–1 | Linked set only | No whole-bank rerank |
| Generator LLM | Existing count | No added call; adaptive practice split can add batch calls | Bounded by slot groups |
| Verifier LLM | Existing per accepted practice question | Unchanged | Existing behavior |

## Prompt / Retrieval Bounds

| Component | Bound |
|---|---|
| Vector candidates | Config default 12; runtime max 24 |
| Doubt references | Output max 2; rerank pool max 5 |
| Pattern graph fields | Fixed field/item/character caps |
| SolveFlow | Max 6 strict steps |
| Prompt input target | Config default 3,800 tokens, valid 1,024–8,192 |
| Practice group size | Existing bounded group policy plus adaptive split |

References and optional guidance are removed before a logged baseline fallback; the current student
question is not truncated by this feature.

## Retry / Fallback Cost

| Service | Retry | Added latency |
|---|---:|---|
| Pattern BatchGet | 0 default; adapter opts into 1 | 50 ms bounded backoff plus AWS call |
| Vector / linked query | 0 local retry | Existing path fallback |
| ColBERT | Existing timeout policy | Deterministic linked-reference fallback |
| Model execution | Existing provider policy | No new Pattern-specific model retry |

## Latency Risks

| Risk | Label | Mitigation / requirement |
|---|---|---|
| Many distinct practice retrieval groups resolve serially | [PERFORMANCE RISK] | Measure first; design bounded concurrency before activation |
| Request-local memo does not span persisted generation waves | [PERFORMANCE RISK] | Exact re-entry avoids vector query but can repeat BatchGet |
| ColBERT cold start/model load | [COLD START RISK] | Optional, linked-only, feature fallback; measure Dev |
| Direct S3 Vector and IAM path is unverified | [PROD BLOCKER] | Live trace required |
| No current end-to-end latency output | [BLOCKER] | Controlled dry run |

## Cost Risks

| Risk | Label | Mitigation |
|---|---|---|
| Adaptive practice split increases existing generator calls | [COST RISK] | Log batch budgets/calls; benchmark before enablement |
| Pattern prompt tokens | [COST RISK] controlled | Component measurement and configurable cap |
| Repeated cross-wave hydration | [COST RISK] | Future assessment-local dedupe after evidence |
| No live token/cache metrics | [NOT VERIFIED] | Capture provider input/cached tokens by route |

## Scalability / Caching

No new process-wide cache is introduced. Static system prompt composition remains stable and dynamic
Pattern/question data stays in the user message, which is compatible with provider prefix caching.
Actual cache-hit and cached-token behavior is [NOT VERIFIED]. Assessment-level Pattern diversity,
bounded parallel retrieval groups, and persisted-wave memoization are deferred pending measurements.

## Observability Requirements

| Signal | Local status | Activation requirement |
|---|---|---|
| Candidate/group/decision counts | Implemented | Confirm emitted in Dev |
| Prompt component/total tokens | Implemented | Compare baseline vs Pattern |
| External call latency | Partial / [NOT VERIFIED] | Vector, BatchGet, GSI, ColBERT p50/p95 |
| Model/cache/token usage | Existing provider fields; [NOT VERIFIED] | Per route and Pattern tier |
| Quality/duplication/diversity | Not measured | Controlled evaluation set |

## Final Decision Reason

The access pattern is bounded and adds no model router call, but practice group serialization,
cross-wave rehydration, adaptive split costs, and all live latency/cache metrics are unverified.
Together with the missing live vector contract and absent current test gates, performance-cost
approval must remain BLOCK.

