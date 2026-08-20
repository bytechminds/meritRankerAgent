---
name: meritranker-tutor-core-context
description: Use for any MeritRanker Tutor (Python AgentCore + LangGraph) task in this repository — understanding repo layout, layering, coding standards, AgentCore runtime, LangGraph patterns, Pydantic schema rules, integration boundaries, security/privacy, or performance rules. Load this first for any non-trivial change before touching code.
---

# MeritRanker Core Context

Entry point for MeritRanker Tutor project knowledge. This is the Python AgentCore +
LangGraph runtime for student-facing tutoring (Practice generation, Doubt Solver,
classification, pattern intelligence). `AI-tutor-backend` (separate repo, Next.js +
Amplify/Lambda) owns DB/admin/frontend concerns and is out of scope for this skill.

## Read in this order

1. [`AGENTS.md`](../../../AGENTS.md) — authoritative repo-wide instruction file: repo
   boundaries, hard rules (no FastAPI, no infra without ask, no hardcoded secrets),
   dependency management (`uv`, never `pip`), local dev commands.
2. [`CLAUDE.md`](../../../CLAUDE.md) — engineering contract: priority order
   (correctness → reliability → cost → latency → maintainability), the 14-step Core
   Development Rule, change-surface rule, regression-protection checklist.
3. `skills/core/project-overview.md` — product purpose and growth phases.
4. `skills/core/architecture-principles.md` — layer boundaries
   (UI/API → auth → domain policy → orchestration → retrieval/providers →
   generation/verification → persistence → observability) and what must not leak
   across them.
5. `skills/core/coding-standards.md`, `skills/core/pydantic-schemas.md`,
   `skills/core/langgraph-patterns.md` — Python/typing/graph conventions.
6. `skills/core/integration-boundaries.md` — DynamoDB, Bedrock, model, cache, external
   call rules.
7. `skills/core/security-and-privacy.md`, `skills/core/performance-and-scalability.md`,
   `skills/core/testing-and-debugging.md` — as relevant to the change.

## Repository layout (verified against current tree)

```
agentcore/     AgentCore CLI/deployment config only — no Python app code
app/           all Python source: graphs/, services/, tools/, schemas/, prompts/,
               observability/, retrieval/, features/practice_generation/, tests/
skills/        this repo's own agent-guidance docs (source of truth, never deployed)
.claude/skills/  Claude Code project skills (this file and siblings) — thin pointers
               into skills/, not a duplicate copy
```

`agentcore/agentcore.json` declares `codeLocation: "app/"` — anything outside `app/`
is invisible to the deployed agent at runtime.

## Hard rules (do not violate)

- No FastAPI — AgentCore's `BedrockAgentCoreApp` provides the HTTP layer.
- No new infra (DynamoDB, Redis, queues, caches, S3, KB, vector search) unless the
  task explicitly requires it and existing architecture cannot satisfy it — see
  `CLAUDE.md` "Avoid Overengineering".
- No secrets hardcoded — env vars via `app/config.py`.
- Do not silently change public request/response schemas (`app/schemas/*.py`).
- Framework: LangGraph `StateGraph` for orchestration; Pydantic v2 for every
  request/response/state model; services in `app/services/` isolate external
  integrations; tools in `app/tools/` are self-contained LangGraph tool callables;
  prompts live in `app/prompts/*.md`, never inlined.

## Which skill to load next

Route by feature area — each has its own project skill that points to the matching
`skills/features/<name>.md`:

| Area | Skill |
|---|---|
| Doubt Solver / streaming answers | `meritranker-doubt-solver` |
| Practice / quiz / mock generation | `meritranker-practice-generation` |
| Pattern Intelligence / PatternGraph / retrieval | `meritranker-pattern-intelligence` |
| AI usage / credit / cost metering | `meritranker-ai-usage-metering` |
| Text/image academic classification | `meritranker-classification` |
| Conversation history / session memory | `meritranker-conversation-history` |
| Exam/stage profile cache | `meritranker-exam-profiles` |
| Structured events/logging/tracing | `meritranker-agent-observability` |
| Foundation demo graph | `meritranker-demo-agent` |

For the required end-to-end change workflow (inspect → plan → implement → review →
test → report), load `meritranker-tutor-implementation-loop`. For which reviewer roles to
apply, load `meritranker-tutor-role-review`. For doc-update obligations, load
`meritranker-tutor-documentation-sync`.

## Local dev commands

```bash
make env             # verify Python, uv, agentcore versions
make check            # ruff lint + pytest — run before every response with code changes
agentcore validate    # validate agentcore.json schema
```

Never claim tests passed without actually running `make check` (or the targeted
pytest invocation) in this session.
