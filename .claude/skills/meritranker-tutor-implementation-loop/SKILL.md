---
name: meritranker-tutor-implementation-loop
description: Use for every MeritRanker Tutor bug fix or feature change that will touch production code — enforces the required inspect-first, evidence-before-fix, smallest-safe-change, review, test, regression-check workflow from CLAUDE.md and AGENTS.md before any edit is made. Load immediately after meritranker-tutor-core-context and before writing or editing code.
---

# MeritRanker Implementation Loop

This encodes the Core Development Rule from [`CLAUDE.md`](../../../CLAUDE.md) and the
AI Development Role Workflow from [`AGENTS.md`](../../../AGENTS.md#ai-development-role-workflow).
This is the canonical, single definition of the production workflow for this repo —
feature skills (`meritranker-doubt-solver`, `meritranker-practice-generation`, etc.)
must not restate or fork this sequence; they only add owning-file and invariant
detail that this loop consumes at steps 2–3.

## The required chain, for any production-level code change

```
prompt
→ meritranker-tutor-core-context (repo architecture, hard rules, boundaries)
→ affected feature skill(s) (meritranker-doubt-solver / -practice-generation / ... )
→ relevant cross-cutting skills (meritranker-ai-usage-metering, -agent-observability,
   -pattern-intelligence, -conversation-history, -classification, -exam-profiles —
   whichever the change actually touches)
→ read the actual code/tests/config/logs (not the skill docs alone)
→ requirement + root-cause analysis
→ design the smallest safe change
→ meritranker-tutor-role-review, FIRST PASS: all 12 roles, on the proposed design
→ reconcile findings (fix or explicitly justify before proceeding — do not implement
   through an unresolved BLOCKED)
→ implementation (approved scope only)
→ focused tests
→ actual `git diff` review (your own, independent of the role pass)
→ meritranker-tutor-role-review, SECOND PASS: all 12 roles, on the actual diff
→ relevant regression tests (see checklist below)
→ real/live E2E when local tests cannot prove the behavior
→ release/readiness gate: PASS or NOT_READY
```

## Detailed steps

1. **Understand the requirement.** Restate it. State what is explicitly out of scope.
2. **Load context in order:** `meritranker-tutor-core-context`, then the feature
   skill(s) matching the change, then any cross-cutting skill the change actually
   touches. Do not skip straight to code.
3. **Inspect the actual execution path.** Read the real implementation
   (`app/graphs/`, `app/services/`, `app/features/practice_generation/`,
   `app/tools/`), the actual tests, config, and logs — never guess from symptoms
   alone, and never rely on the skill docs as a substitute for reading code.
4. **Reproduce or collect evidence.** Exact request/input, last successful stage,
   first failing stage, originating exception, owning component, whether a recent
   change is causal, whether the affected code is shared (check callers).
5. **Identify the first incorrect ownership boundary or invariant** — fix at the
   owning layer per `skills/core/architecture-principles.md`, not downstream.
6. **Reuse existing abstractions.** Search `app/services/`, `app/tools/`,
   `app/schemas/` for something that already does this before writing new code.
7. **Design the smallest safe change.** Prefer 1–3 focused modules. Do not
   reorganize directories, rename unrelated APIs, rewrite working components, fix
   unrelated lint/type errors, or touch unrelated dirty-worktree changes. Do not
   introduce a new abstraction, helper, constant, dependency, or pattern beyond
   what the literal request strictly requires — not even a small one. If the
   change appears to need touching something outside the literal request (a
   shared file, a related-but-distinct bug, a doc file, a config value), stop and
   ask the user for explicit permission before doing it; do not proceed and
   disclose it afterward. See `skills/roles/scope-guard.md`.
8. **First role-review pass — all 12 roles, on the design.** Invoke
   `meritranker-tutor-role-review`. Every role returns `APPROVED` /
   `NOT_APPLICABLE` / `BLOCKED` / `NOT_PROVEN` — none are silently omitted.
9. **Reconcile findings.** Any `BLOCKED` outcome must be resolved (fix the design,
   or get explicit user sign-off on the tradeoff) before implementation starts.

Only after 1–9 are satisfied:

10. Implement only the approved scope. Complete it fully — no TODOs, placeholders,
    or pseudo-code.
11. Add/update focused pytest tests in `app/tests/` (or `app/tests/practice_generation/`)
    for the changed behavior, plus unaffected-caller tests for shared code.
12. Review the actual `git diff` yourself, independent of the role pass.
13. **Second role-review pass — all 12 roles again, on the actual diff.** This is
    not optional and not the same pass as step 8: it checks what was actually
    built, including whether the diff matches the approved design.
14. Run the broader regression suite (`make check`) relevant to touched areas —
    see the regression-protection checklist below.
15. Run real/live verification only when local tests cannot prove the behavior
    (see `skills/core/testing-and-debugging.md`) — never claim live success from
    unit tests alone.
16. Report **PASS** or **NOT_READY** honestly, per the Completion Contract below.
    Never call something production-ready unless the required gates actually passed.

## Regression-protection checklist (ask before declaring done)

From `CLAUDE.md` — does this change affect: Practice, Doubt Solver, Pattern/retrieval,
persistence, routing/models/tokens, billing/cost metering, security/auth,
cancellation/resume, language, or API/schema/UI contracts? Test the areas actually
reachable from the changed code; do not skip shared callers.

## Completion Contract

A task is complete only when: root cause/requirement is understood; implementation
is fully integrated; focused tests pass; impact review is done; applicable
regression tests pass; the diff is clean; live validation ran when required; and
remaining limitations are stated explicitly. Use `NOT_READY` for anything not
actually proven — never fabricate a PASS.

## Related

- `meritranker-tutor-role-review` — which existing role guides to apply and when.
- `meritranker-tutor-documentation-sync` — which `skills/features/*.md` file to update.
- `skills/templates/` — required output format if producing a plan, review, or
  bugfix report (`implementation-plan-template.md`, `bugfix-report-template.md`,
  `engineer-implementation-report-template.md`).
