# skills/features/ — Feature Context Files

> Feature context files are **living memory** for AI coding agents.
> Every feature has one file. Every feature change must update its file.
> If docs and code disagree, the task is not complete.

---

## Purpose

Each file in this directory describes one product feature in enough detail
for an AI coding agent to continue work safely without re-reading all the code.

Before implementing or modifying a feature:
1. Read the relevant feature file here.
2. After making changes, update the file before marking the task done.

**Creating a new feature?** Copy `skills/templates/feature-context-template.md`
to `skills/features/<feature-name>.md` and fill every section.

**No mandatory section should be empty.** Use `[NOT VERIFIED]` or `[DOCS TODO]`
if a fact is unknown.

---

## Current Feature Files

| File | Feature | Status |
|---|---|---|
| `demo-agent.md` | Local demo — AgentCore + LangGraph + Pydantic foundation | Local Demo |
| `doubt-solver.md` | Student doubt solving and tutoring workflow | Planned |
| `image-question-classification.md` | Image extraction and classification entry path | Partially Implemented |

---

## Status Meanings

| Status | Meaning |
|---|---|
| Planned | Designed or scoped but no code written |
| In Progress | Implementation has started but is incomplete |
| Local Demo | Running locally as a non-production demo |
| Partially Implemented | Some functionality implemented, not feature-complete |
| Implemented | Feature complete and in production |
| Blocked | Implementation blocked by an unresolved dependency or decision |
| Deprecated | Still in code but scheduled for removal |
| Removed | No longer in code |

---

## Governance Rules

- One file per feature — do not merge multiple features into one doc.
- Docs must reflect current code, not aspirational future state.
- Planned features are documented with `[NOT VERIFIED]` or "Not implemented" markers.
- `## Latest Changes` must be updated every time the feature changes.
- The `## Current Feature Files` table in this README must be updated when a file is added or removed.
