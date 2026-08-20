---
name: meritranker-tutor-documentation-sync
description: Use whenever a MeritRanker Tutor change alters a feature's behavior, architecture, durable contract, or failure semantics — determines which skills/features/<name>.md file(s) must be read before implementation and updated after, so docs never drift from code. Load before finishing any feature change.
---

# MeritRanker Documentation Sync

`skills/features/` is living memory for AI coding agents, not archival documentation.
If docs and code disagree, the task is not complete — this is stated in both
`AGENTS.md` and `skills/features/README.md`.

## Required workflow

1. Before implementing, read the relevant `skills/features/<feature-name>.md` file
   (route via the feature-specific skill, e.g. `meritranker-practice-generation`).
2. After a behavior change, update that same file — do not create a second,
   competing doc for the same feature.
3. Update the `## Latest Changes` section and the status column in
   `skills/features/README.md` if status changed (Planned / In Progress /
   Partially Implemented / Implemented / Blocked / Deprecated / Removed).
4. State which docs were changed in your final report. If none were updated,
   justify why explicitly (e.g. "internal refactor, no contract change").
5. New feature? Copy `skills/templates/feature-context-template.md` to
   `skills/features/<feature-name>.md` — do not invent a different structure —
   and add a `meritranker-<feature-name>` skill under `.claude/skills/` mirroring
   the existing ones, once the feature is real enough to need routing.

## What counts as a doc-triggering change

Runtime logic, prompt wording (`app/prompts/*.md`), model/provider routing,
validation rules, schema/contract shape, log or event shape
(`app/observability/events.py`), fallback/retry behavior, persistence shape,
cancellation/resume semantics, or admin/approval gates.

## Labels used across all docs

`[ASSUMPTION]`, `[NOT VERIFIED]`, `[BLOCKER]`, `[PROD BLOCKER]`, `[AI RISK]`,
`[SECURITY RISK]`, `[PERFORMANCE RISK]`, `[AUTH TODO]`, `[DEFER]` — use these
instead of asserting something you haven't checked. Full definitions in
`skills/core/documentation-rules.md` and `skills/templates/README.md`.

## Do not

- Duplicate content already in `skills/core/*.md` into a feature file.
- Leave a mandatory template section blank — write `[NOT VERIFIED]` or `N/A`.
- Document trivial implementation details that add no decision-relevant context.
