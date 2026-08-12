# Product Brief: Pattern Intelligence Runtime

> Role: Product Manager

## Product Stage

Foundation Demo / Dev integration hardening.

## User Problem

Students need practice questions and doubt explanations that preserve the right reasoning pattern without reusing an unsafe or mismatched question. Pattern retrieval must improve guidance without becoming a new source of incorrect answers or availability failures.

## Target User

Students preparing for supported exams through the existing practice-generation and doubt-solver flows.

## Pain Severity

High. A semantically similar but logically incompatible pattern can produce a wrong teaching path or weak assessment item.

## Why Now

The canonical PatternGraph/SolveFlow data model now exists in the sibling backend, while this runtime currently has only a legacy flattened consumer contract.

## User Flow

1. A student requests practice or submits a doubt.
2. The existing planner/classifier determines the student need.
3. The runtime may attach a compact, compatible PatternGraph method hint.
4. The existing verifier and answer-generation safeguards remain authoritative.

## MVP Scope

- Deterministic Pattern retrieval planning and authoritative Pattern hydration.
- Compact PatternGraph guidance for practice generation and doubt solving.
- Bounded, answer-redacted PatternQuestion references for doubt only.
- Safe fallback to the unchanged flow when Pattern data is unavailable or unsuitable.

## Non-Goals

- New ingestion, queues, caches, managed KBs, model agents, or LLM rerankers.
- PatternQuestion-to-playable-Question reuse without an explicit backend mapping.
- Production activation or infrastructure deployment.

## Success Metrics

| Metric | Target | Measurement Method |
|---|---|---|
| Unsafe Pattern promotion | 0 in deterministic fixtures | Unit tests for authoritative guards |
| Prompt context | Bounded before model invocation | Prompt-budget tests and metrics |
| Legacy fallback | Existing flow remains available | Failure-path tests |
| Live quality/cost/latency | [NOT VERIFIED] | Requires controlled Dev trace |

## User-Facing Acceptance Criteria

- [ ] Given a compatible PatternGraph, generation receives a compact method constraint and still uses the existing verifier.
- [ ] Given missing, malformed, inactive, or incompatible Pattern data, the student receives the existing safe flow.
- [ ] Given a linked PatternQuestion, its answer and solution are never sent as runtime reference context.

## Evidence Level

| Claim | Evidence Level |
|---|---|
| Canonical Pattern/PatternQuestion schema exists | Confirmed by sibling-backend read-only inspection |
| Direct S3 Vectors index is available to this runtime | [NOT VERIFIED] |
| A Pattern can safely map to a reusable QuestionBank item | [NOT VERIFIED] — current schema disproves it |

## Risks / Open Questions

| Risk / Question | Label | Status |
|---|---|---|
| Direct S3 Vector ingestion/metadata/IAM contract | [NOT VERIFIED] | Open |
| Pattern to playable QuestionBank relationship | [BLOCKER] for Pattern-driven reuse | Open |
| Live quality/cost/latency effect | [NOT VERIFIED] | Open |

## Dependencies

| Dependency | Status |
|---|---|
| Canonical Pattern table | Contract confirmed; runtime configuration [NOT VERIFIED] |
| PatternQuestion `questionsByPattern` GSI | Contract confirmed; runtime configuration [NOT VERIFIED] |
| Existing practice verifier and manifest gate | Available |
| Existing doubt solver and PromptResolver | Available |

## Release Recommendation

**Recommendation:** Proceed with local, feature-flagged implementation only.

**Reason:** The safe guidance path is implementable without changing public schemas, but live activation and Pattern-driven practice reuse remain blocked by missing verified contracts.

## Implementation Update — 2026-08-10

The local feature-gated guidance path is now implemented. It preserves QuestionBank reuse and
practice verification, sends no raw PatternQuestion to practice, uses only answer-redacted bounded
doubt references, and returns to existing flows on unavailable or unsuitable Pattern data. The
release recommendation remains local-only: the direct S3 Vectors source contract, live IAM, exact
practice metadata mapping, `REUSE_SAFE` mapping, and current post-edit validation remain
[NOT VERIFIED] or blocked.

