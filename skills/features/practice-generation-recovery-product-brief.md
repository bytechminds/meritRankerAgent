# Product Brief: Practice Generation Recovery

## Product Stage

Production hardening.

## User Problem

A student can pay the latency and model cost for verified questions but receive no playable test
because final bookkeeping does not match the planner manifest.

## Target User

Exam-preparation students requesting AI-generated practice.

## Pain Severity

Critical: the reported run spent 12 model calls and ended FAILED with no playable assessment.

## Why Now

The exact failure is persisted and reproducible, and it affects the core paid practice outcome.

## User Flow

1. Student requests practice.
2. The system preserves accepted questions and repairs only rejected slots.
3. The final manifest is validated and either becomes playable or fails with a typed reason.

## MVP Scope

- [x] Exact slot-to-bucket binding.
- [x] One targeted repair and one fresh replacement maximum.
- [x] One metadata-only final recovery followed by full revalidation.
- [x] Safe telemetry and deterministic reliability coverage.

## Non-Goals

- Model, routing, token, Pattern, billing, public schema, or infrastructure changes.
- Production deployment or weaker academic verification.

## Success Metrics

| Metric | Target | Measurement Method |
|---|---|---|
| Exact playable count | 100% in deterministic matrix | Final validator |
| Duplicate IDs/slots | 0 | Manifest assertions |
| Recovery LLM calls | 0 | Usage events |
| Repair waves | At most 1 repair + 1 replacement | Wave tests |

## User-Facing Acceptance Criteria

- [x] A valid three-question request reaches a playable three-question assessment.
- [x] One rejected question does not discard accepted siblings.
- [x] Unrecoverable corruption fails explicitly instead of publishing READY.

## Evidence Level

| Claim | Evidence Level |
|---|---|
| User has this problem | Confirmed by persisted failed assessment |
| User would use this feature | Confirmed by supplied replay |
| This is the right solution | Confirmed by source trace, regression, and local replay |

## Risks / Open Questions

| Risk / Question | Label | Status |
|---|---|---|
| Literal wording is diverted by context gate without source history | [NOT VERIFIED] | Open outside generation |
| Aggregate worktree contains unrelated changes | [PROD BLOCKER] | Open |

## Dependencies

| Dependency | Status |
|---|---|
| Existing planner/generator/verifier/repository path | Available |
| Existing local AWS dev resources | Available |

## Release Recommendation

**Recommendation:** Proceed for scoped code review; do not promote the aggregate worktree.

**Open items that must be resolved before release:**

- [ ] Isolate the scoped patch from unrelated dirty files.
- [ ] Decide whether the independent context-gate wording behavior needs a separate fix.
