# QA Review: Pattern Intelligence Runtime

> Role: QA Reviewer
> Date: 2026-08-10

## Recommendation

**BLOCK**

## Evidence Checked

| Artefact | Checked? | Notes |
|---|---|---|
| BA requirements | Yes | AC-01 through AC-07 reviewed |
| Implementation report | Yes | Current validation explicitly unverified |
| Changed files | Yes | Static review across runtime, practice, doubt, prompts, tests |
| Current pytest output | [NOT VERIFIED] | Not run after final edits |
| Current Ruff output | [NOT VERIFIED] | Execution rejected by usage-limit approval gate |
| AgentCore validation | [NOT VERIFIED] | No current output |
| Feature context | Yes | Blockers and rollback documented |

## Requirements Coverage

| BA Criterion | Test evidence | Status |
|---|---|---|
| AC-01 deterministic plan | `test_pattern_intelligence_runtime.py` | Test added; execution [NOT VERIFIED] |
| AC-02 bounded hydration | Runtime + DynamoDB tests | Test added; execution [NOT VERIFIED] |
| AC-03 incompatible candidates excluded | Runtime/practice context tests | Test added; execution [NOT VERIFIED] |
| AC-04 practice guidance preserves reuse/verifier | Practice core/prompt/context tests | Test added; execution [NOT VERIFIED] |
| AC-05 linked-only redacted doubt references | Runtime/adapter/doubt tests | Test added; execution [NOT VERIFIED] |
| AC-06 fallback | Doubt integration + provider fallback paths | Test added; execution [NOT VERIFIED] |
| AC-07 input budget | Practice prompt and doubt integration tests | Test added; execution [NOT VERIFIED] |

## Checklist Result

| Check | Result | Notes |
|---|---|---|
| Happy paths represented | Pass (static) | Compatible practice/doubt fixtures exist |
| Validation failures represented | Pass (static) | Weak/stale/malformed/type/operation/exclusion cases |
| Service failures represented | Pass (static) | Vector, hydration, rerank, canonical-to-legacy fallback |
| No real infrastructure in focused tests | Pass (static) | Injected fakes/mocks |
| Full suite green | Fail | [NOT VERIFIED] |
| Ruff clean | Fail | [NOT VERIFIED] |
| AgentCore validation | Fail | [NOT VERIFIED] |

## Missing Tests / Evidence

| Scenario | Risk | Label |
|---|---|---|
| Full repository regression after final edits | High | [BLOCKER] |
| AgentCore package validation | High | [BLOCKER] |
| Live source metadata/version/score calibration | High | [PROD BLOCKER] |
| Assessment-level diversity and cross-wave dedupe | Medium | [DEFER] |
| Controlled 1/5/50 practice and doubt dry run | High | [PROD BLOCKER] |

## Regression Risks

| Component | Risk | Mitigation |
|---|---|---|
| Shared DynamoDB query wrapper | Projection option could affect callers | Keyword-only and default None; focused test added |
| Shared orchestrator normal/stream paths | Budget composition could alter message selection | Typed optional argument, default None, focused test added |
| Practice grouping | Strict source parity can reduce guidance availability | Intentional fail-closed fallback |
| S3 doubt selector | Canonical feature could suppress legacy retrieval | Explicit legacy fallback restored and tested |

## Blocking Findings

| # | Finding | Label | Resolution required |
|---|---|---|---|
| 1 | Current lint, pytest, make check, and AgentCore validation evidence is absent | [BLOCKER] | Run and attach current green outputs |
| 2 | Live direct S3 Vectors and IAM contract is unverified | [PROD BLOCKER] | Versioned producer/query/IAM evidence |
| 3 | No verified playable mapping supports REUSE_SAFE | [PROD BLOCKER] | Add authoritative mapping and separate acceptance tests |

## Documentation Status

`skills/features/pattern-intelligence-runtime.md` is current and explicitly blocks activation.

## Final Decision Reason

Static coverage is materially improved and fail-closed behavior is represented, but QA cannot pass a
large shared-runtime change without current executable gates. Live source and reuse contracts also
remain unresolved, so the only evidence-based recommendation is BLOCK.

