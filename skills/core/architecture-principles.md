# Architecture Principles — meritRankerTutor

> These principles govern how the system is structured.  Follow them when
> adding new features or reviewing existing ones.  Challenge any change that
> violates them unless there is a documented reason.

---

## Layer Map

```
HTTP Request
    │
    ▼
BedrockAgentCoreApp  (AgentCore runtime — HTTP layer)
    │
    ▼
main.py  (entrypoint: validate input → build state → invoke graph)
    │
    ▼
LangGraph StateGraph  (workflow control — graphs/)
    │         │
    ▼         ▼
 Nodes     Conditional edges
    │
    ▼
Services  (external integrations — services/)
    │
    ▼
External APIs / LLMs / Databases  (never accessed directly from nodes)
```

---

## Core Principles

### 1. AgentCore = Runtime Shell

AgentCore handles HTTP, deployment, versioning, and routing.
The Python application does not own any of these concerns.

- Do not replicate what AgentCore provides (HTTP server, packaging, env injection).
- Do not add FastAPI or any other web framework.
- `main.py` is the only file that interacts with `BedrockAgentCoreApp`.

### 2. LangGraph = Workflow Control

All multi-step agent logic lives in a `StateGraph`.

- One graph per feature/workflow.
- Nodes are small, focused functions.
- Conditional routing is added only when the control flow genuinely branches.
- Graphs are testable without starting the AgentCore runtime.

### 3. Pydantic = Contract Safety

Pydantic models are the formal contract between layers.

- `schemas/request.py` — what the caller sends.
- `schemas/response.py` — what the agent returns.
- `schemas/state.py` — what flows through the graph.
- Validate at the **entrypoint only**; inner layers trust the types.
- Never change a schema field name or type without a migration plan and tests.

### 4. Services = Replaceable Integrations

Every external dependency is hidden behind a service module.

- `services/mock_response_service.py` → future `services/bedrock_llm_service.py`
- Graph nodes call service functions; they never import `boto3`, `requests`,
  or any SDK directly.
- This makes testing easy (mock the service) and migration cheap (swap the file).

### 5. Tools = Agent-Callable Actions

LangGraph tool nodes use callables from `tools/`.

- Tools are thin wrappers that delegate to services.
- No infrastructure logic (DB calls, HTTP) directly inside tool functions.
- Tools are registered with the graph; they do not call the graph.

### 6. Prompts = Externalised Templates

Prompt text lives in `prompts/*.md`, not in Python files.

- Load prompts at service initialisation or node invocation.
- This keeps prompts editable without changing Python code.

### 7. Single AgentCore Runtime First, Multiple Graphs Inside It

Start with one `runtimes` entry in `agentcore.json` and multiple graphs inside it.
Do not create a second runtime to separate features when routing inside one runtime is sufficient.

Split into separate runtimes only when:
- Runtime boundaries differ (different memory, auth, network mode).
- Independent scaling or deployment cadence is required.
- A feature must be isolated for security or compliance reasons.

Split into separate agents only when the above conditions are met — not for convenience.

### 8. Avoid Overengineering

Do not add complexity ahead of need.

- Do not add a message queue before measuring throughput requirements.
- Do not add a cache before measuring cache hit value.
- Do not add a separate microservice before the current service shows scaling limits.
- Do not add multi-agent orchestration before a single agent proves insufficient.

When a concern is real but premature, label it `[DEFER]` in feature docs and return when justified.

### 9. Prefer Simple Replaceable Boundaries

Every external dependency should be swappable without changing the graph:
- Abstract behind a service with a typed function signature.
- Env vars select the implementation (e.g., `MODEL_PROVIDER=bedrock`).
- Tests use the mock implementation; production uses the real one.

If a design makes this swap harder, it needs justification.

### 10. Interfaces Before Implementation

Before adding a new service or integration:

1. Define the Pydantic request/response contract.
2. Write a mock service that satisfies the contract.
3. Wire the mock into the graph and write tests.
4. Swap the mock for a real implementation later.

### 11. Future Replaceability

Every integration point should be swappable without touching the graph:

| Today | Tomorrow |
|---|---|
| `mock_response_service.py` | `bedrock_llm_service.py` |
| In-memory state | DynamoDB / Redis state |
| No auth | JWT / Cognito validation middleware |
| Single-node graph | Multi-agent graph |

---

## What Belongs Where

| Concern | Location |
|---|---|
| HTTP handling | AgentCore (not our code) |
| Request validation | `main.py` (Pydantic) |
| Workflow logic | `graphs/` |
| External calls | `services/` |
| Agent tools | `tools/` |
| Prompt text | `prompts/` |
| Data contracts | `schemas/` |
| Settings | `config.py` |
| Logging setup | `logging_config.py` facade + `observability/` implementation |
| Tests | `tests/` |
| Deploy config | `agentcore/` |
| Agent guidance | `skills/` |

---

## Decision Record Placeholder

> TODO: Add dated ADR entries here when significant architectural decisions are made.
> Format: `YYYY-MM-DD — <decision title> — <brief rationale>`

- `2025-05-20` — Use TypedDict for LangGraph internal state; Pydantic only at API boundary.
  Rationale: LangGraph's reducer/checkpointer is most reliable with plain dicts; avoids
  serialisation surprises when adding persistence later.
- `2026-07-15` — Use S3 Vectors for student retrieval candidates and DynamoDB PatternGraph
  bundles as final authority. Rationale: vector metadata is stale-prone and cannot grant
  student final-answer authority without the current approved runtime bundle.
