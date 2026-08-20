# skills/ — AI Coding Agent Guidance

This directory contains **repo-local development knowledge** for AI coding agents
(Claude Code, Codex, GitHub Copilot, and others).

> The files here are **never deployed**.  They exist only to give coding agents
> accurate, project-specific context so they make consistent, safe decisions.

---

## How to Use This Directory

1. **Always read `AGENTS.md` first** — it is the single authoritative instruction file.
2. Then read the relevant file(s) below before writing or reviewing code.
3. When you create or change a feature, update the matching `features/<name>.md`.

> **Claude Code specifically** auto-loads this knowledge via
> `.claude/skills/*/SKILL.md` — one project skill per core/workflow concern and
> one per feature (`meritranker-doubt-solver`, `meritranker-practice-generation`,
> etc.), each a thin pointer into this directory, not a duplicate. Other agents
> (Codex, Copilot) should keep reading this directory and `AGENTS.md` directly.

---

## Directory Map

| Path | Purpose |
|---|---|
| `skills/core/` | Permanent project-wide rules (coding style, architecture, testing, docs) |
| `skills/features/` | Per-feature context — one file per product feature |
| `skills/roles/` | Role-specific behaviour guides for specialised agent tasks |
| `skills/templates/` | Reusable document templates for plans, reviews, and bug reports |

---

## core/ — Permanent Rules

| File | Contents |
|---|---|
| `coding-standards.md` | Python style, type hints, module layout rules |
| `architecture-principles.md` | System boundaries, layering, replaceability |
| `agentcore-runtime.md` | AgentCore deployment details, local workflow |
| `langgraph-patterns.md` | LangGraph conventions, naming, graph structure |
| `pydantic-schemas.md` | Schema design and validation rules |
| `testing-and-debugging.md` | pytest strategy, debug tips, CI gate |
| `documentation-rules.md` | Mandatory doc maintenance rules |

## features/ — Feature Context

| File | Contents |
|---|---|
| `demo-agent.md` | Current demo/foundation feature |
| *(add more as features are built)* | |

## roles/ — Role Guides

| File | Role |
|---|---|
| `architect.md` | Architecture review behaviour |
| `python-agent-engineer.md` | Implementation behaviour |
| `qa-reviewer.md` | Test & quality review behaviour |
| `security-reviewer.md` | Security review behaviour |
| `documentation-maintainer.md` | Doc maintenance behaviour |

## templates/ — Reusable Templates

| File | Use |
|---|---|
| `feature-context-template.md` | Starting point for new `features/*.md` files |
| `implementation-plan-template.md` | Plan before coding a new feature |
| `review-checklist-template.md` | Pre-merge review checklist |
| `bugfix-report-template.md` | Structured bug report and fix record |
