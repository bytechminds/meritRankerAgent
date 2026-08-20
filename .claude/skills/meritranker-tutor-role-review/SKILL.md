---
name: meritranker-tutor-role-review
description: Use to determine which existing MeritRanker Tutor expert-role guides (Product Manager, Business Analyst, Solution Architect, AI Solution Architect, Python Agent Engineer, QA Reviewer, Security Reviewer, Performance-Cost Reviewer, Documentation Maintainer, Scope Guard, Release Gatekeeper) must review a production change, both before implementation and on the actual diff afterward. Load before starting a non-trivial design and again before declaring a change done.
---

# MeritRanker Role Review

This project's AI development team is already defined in `skills/roles/`. Reuse those
role files; never invent new reviewer roles or fake an approval.

## Role sequence (from `AGENTS.md`)

| Phase | # | Role | File |
|---|---|---|---|
| Planning | 1 | Product Manager | `skills/roles/product-manager.md` |
| Planning | 2 | Business Analyst | `skills/roles/business-analyst.md` |
| Planning | 3 | Solution Architect | `skills/roles/solution-architect.md` |
| Planning | 4 | AI Solution Architect | `skills/roles/ai-solution-architect.md` |
| Implementation | 5 | Python Agent Engineer | `skills/roles/python-agent-engineer.md` |
| Review | 6 | QA Reviewer | `skills/roles/qa-reviewer.md` |
| Review | 7 | Security Reviewer | `skills/roles/security-reviewer.md` |
| Review | 8 | Performance-Cost Reviewer | `skills/roles/performance-cost-reviewer.md` |
| Review | 9 | Documentation Maintainer | `skills/roles/documentation-maintainer.md` |
| Guard | 10 | Scope Guard | `skills/roles/scope-guard.md` |
| Release | 11 | Release Gatekeeper | `skills/roles/release-gatekeeper.md` |

Also present: `skills/roles/prompt-engineer.md` — apply when a change touches
`app/prompts/*.md` or generation/verification prompt wording.

## Rules

1. Roles 1–4 must align before implementation begins — no code until planning is
   complete (scale this to the size of the change; a one-line bugfix does not need
   a full PM/BA pass, but a new feature or schema change does).
2. The Python Agent Engineer implements only the approved plan; scope changes go
   back to the relevant planning role.
3. After implementation, Solution Architect and AI Solution Architect review code
   boundaries and AI workflow before QA, Security, and Performance proceed.
4. QA, Security, Performance-Cost, and Documentation Maintainer review the actual
   diff — concurrently where possible.
5. Scope Guard reviews the plan before implementation and the actual diff after —
   every touched file, added abstraction, or "while I'm at it" change must trace
   back to the literal request or be explicitly approved by the user first. A
   `BLOCKED` from Scope Guard means stopping and asking, not proceeding and
   disclosing afterward. This runs regardless of change size.
6. Release Gatekeeper gives the final go/no-go from the collected evidence; it does
   not review code directly. Use `skills/templates/release-gate-template.md`.
7. Any role that cannot verify something must say "Not verified" — never guess,
   assume success, or fabricate a test result. See `skills/roles/README.md` "Core Rule".

## Applying this as Claude Code (single agent, multiple hats)

You are not spawning separate agents for each role by default. Instead, work
through each of the 12 roles listed above **explicitly, by name, in order**, in
your own review pass — never a single generic "looks good." Use the matching
template from `skills/templates/` when a role's output should be a formal
artifact (e.g. `qa-review-template.md`, `security-review-template.md`,
`performance-cost-review-template.md`).

### All roles means all roles — the gate

For a **production-level code change** (anything touching `app/`, prompts,
schemas, or runtime config — see `meritranker-tutor-implementation-loop`), every
one of the 12 roles must appear in the review output with exactly one of these
outcomes. A role must never simply be absent from the list:

| Outcome | Meaning |
|---|---|
| `APPROVED` | Role's checklist was applied to the actual diff/design and passed. |
| `NOT_APPLICABLE` | Role's concern genuinely does not apply to this change — state why in one sentence (e.g. "Security Reviewer = NOT_APPLICABLE — no auth, secrets, PII, or external-input surface touched"). |
| `BLOCKED` | Role's checklist found a real problem that must be fixed before release. |
| `NOT_PROVEN` | The role's concern applies but could not be verified in this session (no live environment, no credentials, etc.) — state exactly what remains unverified, per `skills/roles/README.md` "Core Rule". |

There is no repo-documented lighter-weight path for "trivial" changes (checked
`skills/roles/README.md`, `skills/core/`, `AGENTS.md`, `CLAUDE.md` — none define
one). Do not invent one. The correct mechanism for a genuinely trivial change
(e.g. a docs-only typo fix) is that most or all of the 12 roles legitimately
resolve to `NOT_APPLICABLE` quickly — not that they are skipped or omitted from
the list.

### When this runs

Per `meritranker-tutor-implementation-loop`, this full 11-role pass runs **twice**
for a production-level change: once on the proposed design before implementation,
and once on the actual `git diff` after implementation. The two passes are not
identical — the pre-implementation pass reviews the plan; the post-implementation
pass reviews what was actually built, including whether the diff matches the
approved plan.

## Related

- `meritranker-tutor-implementation-loop` — the full workflow this review slots into,
  including exactly where the two review passes happen.
- `meritranker-tutor-documentation-sync` — Documentation Maintainer's concrete obligation.
