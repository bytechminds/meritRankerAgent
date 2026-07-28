# AGENTS.md — MeritRanker Tutor

> **Read this file first.**  It is the primary instruction file for all AI coding agents
> (Claude Code, Codex, GitHub Copilot, or any other).  After reading this, consult the
> relevant files under `skills/` for deeper context.

## Project Purpose

**MeritRanker Tutor** is a Python AgentCore + LangGraph runtime for student-facing tutoring
workflows.  The project is structured so that workflows, integrations, and schemas are
independently replaceable as the product evolves from a local demo to a production system.

---

## Repository Boundaries (CRITICAL — never violate)

| Directory | Purpose |
|---|---|
| `agentcore/` | AgentCore CLI / deployment config **only**.  No Python application code. |
| `app/` | All Python source code, tests, prompts, and runtime config. |
| `skills/` | Repo-local development knowledge for AI coding agents. |
| Root | `Makefile`, `README.md`, `AGENTS.md`, `.gitignore` only. |

**Why this matters:** `agentcore/agentcore.json` declares `codeLocation: "app/"`.  The CLI
zips only the `app/` directory and deploys it.  Anything outside `app/` is invisible to the
deployed agent at runtime.

---

## Hard Rules for Every Coding Agent

1. **No FastAPI** — AgentCore provides the HTTP layer via `BedrockAgentCoreApp`.
2. **No infrastructure code** (DynamoDB, Redis, S3, Bedrock Knowledge Base, vector search)
   unless explicitly requested in a task.
3. **No real LLM calls** in the demo/foundation stage unless explicitly requested.
4. **No secrets hardcoded** anywhere in source.  Use env vars; load via `app/config.py`.
5. **Do not move `agentcore/`**, rename it, or put Python code inside it.
6. **Do not silently change public request/response schemas.**  Breaking schema changes need
   explicit discussion and updated tests.

---

## Python Application Rules

- **Framework:** LangGraph (`StateGraph`) for all workflow orchestration.
- **Schemas:** Pydantic v2 for every request / response / state model.  No raw dicts at
  API boundaries.
- **Config:** Environment variables only.  Load via `app/config.py` → `get_settings()`.
- **Logging:** Always call `configure_logging()` from `app/logging_config.py` before
  logging.  Never log secrets or full private payloads.
- **Services:** `app/services/` isolates every external integration (LLM, DB, API calls).
  Graph nodes call services; they do not call external APIs directly.
- **Tools:** `app/tools/` contains self-contained callables for LangGraph tool nodes only.
  No heavy infrastructure logic inside tool modules.
- **Prompts:** `app/prompts/` stores Markdown prompt templates.  Load at runtime; never
  inline large prompts in Python files.
- **Tests:** Every behavior change needs a pytest test.  Tests must run without AWS
  credentials or network access.

---

## Documentation Rules for Coding Agents

- When creating or modifying a feature, **update or create** `skills/features/<feature-name>.md`.
- Architectural decisions must be reflected in `skills/core/architecture-principles.md` or
  the relevant feature context file.
- Any new tool, service, schema, or graph file must be listed in the related feature context.
- **If docs and code disagree, the task is not complete.**

---

## Adding Features — Pre-flight Checklist

Before marking any task done:

- [ ] Code belongs in `app/`
- [ ] New request/response shapes have Pydantic schemas in `app/schemas/`
- [ ] New nodes / services have pytest tests
- [ ] No secrets hardcoded
- [ ] `make check` passes (ruff + pytest)
- [ ] `agentcore validate` passes
- [ ] Feature context file under `skills/features/` is created/updated
- [ ] Role output documents use the matching template from `skills/templates/`

---

## Dependency Management

All Python dependencies are managed with `uv` inside `app/`.

```bash
cd app && uv add <package>           # add runtime dep
cd app && uv add --group dev <pkg>   # add dev dep
cd app && uv sync --group dev        # install everything (first-time setup)
```

Never use `pip` directly.

---

## Local Development Workflow

```bash
make env        # verify Python, uv, agentcore versions
make check      # ruff lint + pytest (CI gate — run before every response)
agentcore validate   # validate agentcore.json schema
make dev        # start local AgentCore server with live logs
```

---

## AI Development Role Workflow

This project is developed by AI coding agents working in defined roles.
Every feature must pass through the role sequence below.

### Role Sequence

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
| Release | 10 | Release Gatekeeper | `skills/roles/release-gatekeeper.md` |

### Workflow Rules

1. **Roles 1–4 (PM, BA, Solution Architect, AI Solution Architect) must align before
   implementation begins.** No code is written until planning is complete.

2. **Python Agent Engineer implements only the approved plan.** Scope changes during
   implementation must be escalated back to the relevant planning role.

3. **After implementation, Solution Architect and AI Solution Architect review code
   boundaries and AI workflow** before QA, Security, and Performance proceed.

4. **QA, Security, Performance-Cost, and Documentation Maintainer review concurrently**
   where possible. All must complete before the Release Gatekeeper is invoked.

5. **Release Gatekeeper gives the final go/no-go** based on collected evidence from all
   roles. It does not review code directly.

6. **Every feature change must update `skills/features/<feature-name>.md`.**
   If docs and code disagree, the task is not done.

7. **If a role cannot verify something, it must state "Not verified" explicitly.**
   No role may guess, assume success, or claim 100% certainty.

See `skills/roles/README.md` for the full role boundary and permission table.

---

## Skills Index

> **`skills/core/` contains permanent project-wide engineering rules — read before implementing any feature.**

| File | Contents |
|---|---|
| `skills/core/README.md` | Core rules reading order and index |
| `skills/core/project-overview.md` | Project purpose, foundation stage, growth phases |
| `skills/core/coding-standards.md` | Python style, typing, structure rules, label table |
| `skills/core/architecture-principles.md` | System design boundaries, overengineering avoidance |
| `skills/core/agentcore-runtime.md` | AgentCore deployment & runtime details |
| `skills/core/langgraph-patterns.md` | LangGraph conventions, output validation [AI RISK] |
| `skills/core/pydantic-schemas.md` | Schema design rules, external output validation |
| `skills/core/integration-boundaries.md` | Service boundary rules for LLM, DB, retrieval, cache |
| `skills/core/security-and-privacy.md` | Secrets, auth, logging, PII, OWASP guidance |
| `skills/core/performance-and-scalability.md` | Cost profile, latency risk, premature infra avoidance |
| `skills/core/testing-and-debugging.md` | Test strategy, regression tests, [NOT VERIFIED] pattern |
| `skills/core/documentation-rules.md` | Docs maintenance rules, label usage guide |
| `skills/features/README.md` | Feature context index and status meanings |
| `skills/features/demo-agent.md` | Demo agent feature context (Local Demo) |
| `skills/features/doubt-solver.md` | Doubt Solver feature context (Planned) |
| `skills/features/classification-pipeline.md` | Text/image classification ownership, fallback, and validation |
| `skills/roles/README.md` | Role team overview and workflow rules |
| `skills/roles/*.md` | Role-specific agent behaviour guides |
| `skills/templates/README.md` | Template index and label reference |
| `skills/templates/*.md` | Required output formats for every role (plans, reviews, reports, release gate) |


## Mental Model

The project uses a **flat resource model**. Agents, memories, credentials, gateways, evaluators, and policies are
independent top-level arrays in `agentcore.json`. There is no binding between resources in the schema — each resource is
provisioned independently. Agents discover memories and credentials at runtime via environment variables or SDK calls.
Tags defined in `agentcore.json` flow through to deployed CloudFormation resources.

## Critical Invariants

1. **Schema-First Authority:** The `.json` files are the source of truth. Do not modify agent behavior by editing
   generated CDK code in `cdk/`.
2. **Resource Identity:** The `name` field determines the CloudFormation Logical ID.
   - **Renaming** a resource will **destroy and recreate** it.
   - **Modifying** other fields will update the resource **in-place**.
3. **Schema Validation:** If your JSON conforms to the types in `.llm-context/`, it will deploy successfully. Run
   `agentcore validate` to check.
4. **Resource Removal:** Use `agentcore remove` to remove resources. Run `agentcore deploy` after removal to tear down
   deployed infrastructure.

## Directory Structure

```
meritRankerTutor/
├── AGENTS.md               # This file — read first
├── Makefile                # Dev commands (test, lint, dev, deploy)
├── .gitignore
├── agentcore/              # AgentCore CLI / deployment config only
│   ├── agentcore.json      # Main project config (AgentCoreProjectSpec)
│   ├── aws-targets.json    # Deployment targets (account + region)
│   ├── .env.local          # Secrets — API keys (gitignored)
│   ├── .llm-context/       # TypeScript type definitions for AI assistants
│   └── cdk/                # AWS CDK project (@aws/agentcore-cdk L3 constructs)
├── app/                    # All Python agent application code
│   ├── main.py             # AgentCore entrypoint
│   ├── pyproject.toml      # Python deps (managed by uv)
│   ├── config.py           # Settings loaded from env
│   ├── logging_config.py   # Rich logging setup
│   ├── graphs/             # LangGraph StateGraph definitions
│   ├── schemas/            # Pydantic v2 request/response/state models
│   ├── services/           # External integrations (replaceable)
│   ├── tools/              # Agent-callable LangGraph tools
│   ├── prompts/            # Prompt template Markdown files
│   └── tests/              # pytest test suite
└── skills/                 # AI coding agent guidance (not deployed)
    ├── core/               # Permanent project rules
    ├── features/           # Per-feature context docs
    ├── roles/              # Role-specific agent behaviour guides
    └── templates/          # Reusable doc templates
```
