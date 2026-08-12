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
- `2026-07-24` — Keep the existing query classifier as the only model-based academic
  classification authority. Text classification and existing image classification converge through
  a shared validated stage contract; Memory and DynamoDB provide bounded context but never classify.
  Conversation relation handling remains a separate non-model compatibility concern until its
  dedicated redesign.
- `2026-07-25` — Use a deterministic gate only to decide whether recent conversation is read, then
  let the existing query classifier decide academic fields plus relation, requested action, and at
  most one supplied turn ID. A deterministic selected-context builder converts that validated
  decision into one action-specific generation context. Rationale: this removes duplicate
  conversation interpretation without adding another model, summarizer, cache, or storage schema.
- `2026-07-27` — Resolve bounded external pronouns and indirect references inside the existing
  query-classifier authority. A focused deterministic validator distinguishes local from external
  references, rejects incompatible selections, and permits a one-candidate safe fallback; it never
  selects from Memory directly or adds persistent reference lineage. Rationale: semantic model
  selection remains primary while structural safeguards prevent unresolved standalone answers and
  recency-only cross-topic contamination.
- `2026-07-27` — Treat durable conversation transcripts and contextual candidate projections as
  separate responsibilities. Clarification, unresolved-reference, acknowledgement, technical, and
  failed-quality records remain stored but cannot enter classifier candidates; the same
  response-function policy blocks future meta responses from academic persistence. Reference
  compatibility requires a grounded antecedent and validated label before recency or same-entity
  grouping. Rationale: durable audit history must not become contextual authority by default.
- `2026-07-27` — Keep the validated frontend request as the sole response-language authority and
  apply conditional language guidance only at the existing `PromptResolver` generator boundary.
  English adds no prompt section; Hindi/Hinglish use private verified replay for SSE so language
  validation and the existing bounded repair occur before student-visible chunks. Rationale:
  central composition prevents route drift, while pre-emission verification avoids an
  unrepairable wrong-language partial stream without adding a classifier or translation model.
- `2026-07-28` — Keep correctness verification selective and outside graph topology. Deterministic
  recomputation runs first for supported forms; only intermediate/advanced Quant or Reasoning,
  correction/re-solve, and generated-practice cases may use the existing verifier task role.
  Primary generation, continuation, rewrite, and repair share a two-generator-call request cap.
  Rationale: formatting validation alone cannot protect numerical correctness, while a universal
  verifier or independent repair loops would add avoidable latency and cost.
- `2026-07-28` — Normalize all existing verifier infrastructure and parsing failures at the
  verifier boundary to `ANSWER_VERIFICATION_UNAVAILABLE`; preserve the original exception only in
  internal debug logging and never approve or persist an unavailable result. Keep one o4-mini call,
  no fallback verifier, no correctness-repair generation, and the shared two-generator-call cap.
  Rationale: fail-closed typed outcomes preserve public contracts and prevent exception-handling
  defects from becoming raw request failures.
- `2026-07-28` — Map durable practice generation onto the existing `MockTestQuiz` assessment and
  assessment-owned `Question` relationship without changing Amplify schemas. Keep detailed phases,
  counters, blueprint, and bounded group leases in `MockTestQuiz.meta`; use deterministic IDs and
  conditional writes, with DynamoDB as authority and SQS only as delivery. PatternGraph stays behind
  a disabled no-op provider. Rationale: this supplies resumability and exact-count readiness without
  adding a database, cache, Step Functions workflow, duplicate persistence service, or frontend
  contract.
- `2026-07-28` — Store practice job metadata as a compact DynamoDB document and mutate only atomic
  document paths. Question creation and authoritative aggregate/bucket counters share a
  transaction; the existing Question GSI is used only to prove final invariants, and a short
  projection remains `FINALIZING`. Rationale: prevent concurrent metadata loss and false failure
  from eventual consistency without adding a job table or orchestration service.
- `2026-07-28` — Require practice reuse and assessment reads to use configured GSIs, validating
  their live status/key schemas before feature enablement. Query only projected QuestionBank
  metadata per demand bucket, batch-read shortlisted records, bind generated-question transactions
  to unexpired group leases, and coordinate execution/lease/SQS visibility budgets. Rationale:
  preserve the existing tables and queue while preventing scans, stale-worker commits, stuck
  finalization, and unbounded candidate reads.
- `2026-08-01` — Run every explicit playable practice request as an AgentCore-tracked background
  task inside the existing runtime. The existing completion envelope returns the deterministic
  assessment ID immediately, while the retained PracticeGenerationGraph and DynamoDB manifest stay
  authoritative. Rationale: one runtime and one persistence path provide durable progress without
  reintroducing SQS, Lambda workers, containers, task tables, or frontend notification services.
- `2026-08-02` — Supersede direct ongoing `MockTestQuiz` document/transaction updates with the
  deployed IAM-authorized `updatePracticeGenerationProgress` mutation. Retain direct DynamoDB only
  for initial assessment create/recover/read and Question/QuestionBank access; recalculate the
  complete manifest from persisted Questions before one group-boundary or terminal publication.
  Register the AgentCore task before acknowledgement but start it only after conversation linkage
  succeeds. Rationale: one validated parent-update path preserves the existing AppSync subscription
  contract without adding infrastructure, credentials, schema fields, or a second protocol.
- `2026-08-04` — Treat an empty structured practice-generator response as a model-execution
  failure only when the configured completion budget is exhausted (`finish_reason=length` and the
  reasoning budget consumed it). Advance through the existing model-registry fallback chain before
  any practice repair branch. Rationale: fallback selection belongs to the shared ModelExecutor;
  a usable-but-invalid question remains the only input eligible for validation repair, preserving
  accepted questions and preventing unchanged primary-model retries.
- `2026-08-12` — Centralize shadow AI usage metering at the existing content-free
  `record_llm_call()` boundary, then aggregate only within request/task-local execution context.
  Practice transfers that accumulator to its existing durable test lifecycle and copied worker
  contexts. Rationale: this counts actual provider attempts without graph-node credit logic, a
  second provider path, financial persistence, new infrastructure, or balance mutation.
