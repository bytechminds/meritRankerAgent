# AI Architecture Plan: Practice Generation Recovery

## AI Workflow Plan

Retain initial generation, deterministic parsing, independent verification, one targeted repair,
and one fresh replacement. Final bookkeeping recovery uses no model.

## Graph Design

No node or edge changes. Existing `plan_and_fill`, `generate_group`, and `finalize` commands remain.

## Prompt / Model / Retrieval Boundaries

The existing generator/verifier models and routes remain. Repair receives one rejected candidate
plus allowlisted reason codes for the same immutable slot. No retrieval change.

## Tool Boundaries

No new tools.

## State / Schema Additions

No graph-state addition. Internal final validation diagnostics contain no model output.

## Hallucination Risks

Malformed generation is rejected before verifier; academic mismatch requires independent verifier
acceptance; repair cannot change the slot; final recovery cannot change question content or answer.

## Prompt-Injection Risks

The rejected candidate is untrusted user-role JSON under a static repair system prompt and the
existing structured output validator.

## Output Validation Strategy

GeneratedQuestion/VerificationResult Pydantic validation, slot binding, independent answer match,
playable contract, exact manifest, and full post-recovery revalidation remain mandatory.

## Evaluation / Test Strategy

Mock structured model payload inspection, bounded decision tests, invalid-output regressions,
32 deterministic manifests, and one local real-LLM replay.

## Model / Provider Requirements

Unchanged existing configured providers/models; no tool calling required.

## Latency / Cost Notes

At most the existing three content waves for a rejected slot; final recovery has zero model cost.

## Observability Plan

Safe repair/replacement/recovery lifecycle events with IDs, attempts, counts, and reason codes only.

## Clarification / Escalation Plan

Any request to add retries, change model/routing, or weaken verification requires new architecture
approval.

## Open Issues

Literal-query context-gate parity is separate and `[NOT VERIFIED]`; it did not block a semantically
equivalent self-contained practice replay.
