# QA Review: Centralized AI Usage Metering V1

## Recommendation

**BLOCK**

## Evidence Checked

| Artefact | Checked? | Notes |
|---|---|---|
| BA requirements | Yes | `ai-usage-metering-ba-requirements.md`. |
| Implementation report | Yes | Documents validation limitation. |
| Changed files | Yes | Central sink, operation lifecycle, config, and focused tests reviewed. |
| Test suite output | [NOT VERIFIED] | `pytest` unavailable in the active Python environment. |
| Lint output | [NOT VERIFIED] | `uv` cache sandbox prevented Ruff. |
| `agentcore validate` | Yes | PASS. |
| Local billing/hook smoke | Yes | PASS; synthetic provider usage only. |
| Feature context | Yes | Present and current. |

## Requirements Coverage

| BA Criterion | Test file | Test name | Status |
|---|---|---|---|
| AC-02/03 | `test_ai_usage_billing.py` | Decimal/cached/ceiling test | Written; [NOT VERIFIED] execution. |
| AC-04 | same | debit-rejection and no-debit summary tests | Written; [NOT VERIFIED] execution. |
| AC-05 | same | unknown/missing usage tests | Written; [NOT VERIFIED] execution. |
| AC-07 | same | nested/thread context tests | Written; [NOT VERIFIED] execution. |
| AC-01/06/08/09 | Existing and focused tests | central hook/config tests | [NOT VERIFIED] execution. |

## Checklist Result

| Check | Result | Notes |
|---|---|---|
| New behavior has tests | Pass | Focused tests added. |
| Tests use real AWS/network | Pass by inspection | New test has none. |
| `pytest` green | [NOT VERIFIED] | No execution evidence. |
| `ruff` zero errors | [NOT VERIFIED] | No execution evidence. |

## Missing Tests

| Scenario | Risk if untested | Label |
|---|---|---|
| Full async Practice launch → start → READY/FAILED handoff using the production launcher | Medium | [NOT VERIFIED] |
| Stream cancellation after Practice handoff | Medium | [NOT VERIFIED] |

## Regression Risks

| Component | Risk | Mitigated? |
|---|---|---|
| Streaming lifecycle | Incorrectly duplicated/omitted summary | Partially; focused review only. |
| Practice worker | Lost context in a thread | Yes in code via `copy_context`; execution [NOT VERIFIED]. |

## Blocking Findings

| # | Finding | Label | Resolution required |
|---|---|---|---|
| 1 | Required lint and pytest checks lack execution evidence. | [PROD BLOCKER] | Run `make check`/focused tests in CI or provisioned environment. |

## Non-Blocking Notes

- Unknown profiles correctly remain incomplete; this needs a controlled invoice reconciliation,
  not a guessed test fixture.

## Documentation Status

| Doc | Status |
|---|---|
| `skills/features/ai-usage-metering.md` | Up to date by code inspection. |

## Final Decision Reason

> Source-level checks and `agentcore validate` passed, but the required test and lint commands did
> not execute in this environment. QA cannot approve a production release without that evidence.
