---
name: meritranker-pattern-intelligence
description: Use for any change touching PatternGraph, SolveFlow, the Pattern Intelligence runtime contract, S3 Vector candidate retrieval, or Pattern-guided Practice/Doubt answers. Covers app/retrieval/pattern_intelligence/ and the compatibility-guard logic shared by Practice and Doubt Solver.
---

# MeritRanker Pattern Intelligence Runtime

Canonical Pattern guidance/reuse for Practice generation and Doubt Solver answers.

## Current status — read before assuming this is live

**Local implementation and validation complete; Dev activation remains
security-blocked.** `PATTERN_INTELLIGENCE_ENABLED`,
`PATTERN_INTELLIGENCE_REUSE_ENABLED`, and `PRACTICE_PATTERN_CONTEXT_ENABLED`
default to `false` — production is unchanged. Guidance/reuse must stay off in Dev
until the QuestionBank mutation boundary prevents authenticated owners from
forging authoritative Pattern linkage. Full detail:
`skills/features/pattern-intelligence-runtime.md`. Planning/review history:
`pattern-intelligence-runtime-*.md` under `skills/features/`.

## Runtime contract

```
S3 Vector candidate discovery
→ vector-score floor (negative eligibility only)
→ authoritative Pattern BatchGet
→ exact vector/DynamoDB version-hash parity
→ deterministic Pattern/slot compatibility
→ IGNORE | GUIDANCE_SAFE | REUSE_SAFE
```

DynamoDB Pattern records are PatternGraph/SolveFlow authority; QuestionBank is
playable-Question authority. Raw PatternQuestion records are never playable and
are never used for Doubt references.

## Owning code

`app/retrieval/pattern_intelligence/` — `interfaces.py`, `models.py`,
`service.py`, `colbert_adapter.py`.

## Invariants

- Compatibility guard fails closed on: status, graph shape, canonical
  subject/topic/category, numeric-complexity band, exam IDs, question type,
  Pattern family, required target/operation identities, slot exclusions, and
  `not_same_when`.
- Vector candidates without a score, without a version hash, below the
  retrieval floor, or stale against `kbCurrentVersionHash` are ignored, not
  fuzzy-matched.
- `REUSE_SAFE` requires an exact verified playable QuestionBank row with
  explicit Pattern ID/version and `VERIFIED_GENERATION`/`VERIFIED_INGESTION`
  evidence, full slot compatibility, checked student history, no
  activity/seen/exclusion conflict.
- `GUIDANCE_SAFE` supplies only compact answer-redacted target/givens/
  conditions/operations/traps and `not_same_when` — never a full solution.
- The independent verifier remains the only authority for generated questions,
  Pattern-guided or not.

## Feature flags — do not flip without explicit approval

`PATTERN_INTELLIGENCE_ENABLED`, `PATTERN_INTELLIGENCE_REUSE_ENABLED`,
`PRACTICE_PATTERN_CONTEXT_ENABLED` are security-gated off by design. Changing a
default is a production/security decision, not a routine code change — route
through Security Reviewer (`meritranker-tutor-role-review`).

## Tests

`app/tests/test_pattern_intelligence_runtime.py`,
`test_pattern_intelligence_runtime_adapters.py`,
`test_pattern_intelligence_doubt_integration.py`.

## Regression impact to check

Practice generation (guided-generation and reuse paths), Doubt Solver (linked
reference context), QuestionBank write path, S3 Vector/ColBERT retrieval.
