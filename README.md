# MeritRanker Agent Python

Open-source Python agent toolkit for **education-focused AI workflows** and **exam-preparation use cases**.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and deployable via [Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html), this repository provides structured agent workflows—not a generic chat wrapper—for tutoring, doubt resolution, practice guidance, and answer validation.

> **Status:** Early-stage, actively developed. Suitable for local development, experimentation, and contribution. Not positioned as a finished commercial product.

---

## Demo

Screenshots from the MeritRanker education platform — the kind of student-facing workflows this agent toolkit is built to power. The UI lives in a separate frontend; **this repository is the Python agent backend** (doubt solver, routing, validation, streaming).

### Doubt Solver

Structured, subject-aware Q&A for exam prep — maths, reasoning, GK, science, and more.

![MeritRanker Doubt Solver — Ask Your Doubt](docs/images/doubt-solver-demo.png)

### Tests & Practice

Practice guidance, weak-area targeting, and progress-aware recommendations alongside agent workflows.

![MeritRanker Tests & Practice dashboard](docs/images/tests-practice-demo.png)

---

## Why this exists

Generic AI chat is not enough for reliable learning. Education applications need:

- **Structured reasoning** — step-by-step explanations aligned to subject and difficulty
- **Validation** — schema-checked outputs and answer-quality gates before students see responses
- **Context-aware retrieval** — direct S3 Vector candidate lookup with DynamoDB source-of-truth validation
- **Educator review support** — workflows designed for human oversight, not fully autonomous grading
- **Testable boundaries** — mock-first development, explicit schemas, and offline pytest coverage

MeritRanker Agent Python packages these concerns into replaceable services, LangGraph workflows, and configuration-driven LLM routing so teams can build education agents without ad-hoc prompt spaghetti.

---

## Features and current direction

| Area | Description |
|---|---|
| **Education AI agent workflows** | LangGraph graphs with explicit state (`request_id`, `query`, `classification`, `retrieval_context`, `context_text`, `answer`) |
| **Doubt-solving flow** | Classify student queries, select approved retrieval context when configured, generate subject-aware explanations |
| **Image-question entry** | Validate and normalize an uploaded question image, then extract and classify it through an isolated multimodal boundary |
| **Structured reasoning steps** | Prompt templates and generator contracts for math, reasoning, and general subjects |
| **Practice guidance** | Intent overlays (solve, explain, practice, visualize) and difficulty-aware routing |
| **Educator review support** | Architecture oriented toward reviewable outputs; full review UI is out of scope for this repo |
| **Answer validation / evaluation** | Answer-quality checks, empty-output guards, continuation logic, and safe failure messages |
| **Schemas, tests, examples** | Pydantic v2 models, 1,800+ offline tests, smoke scripts, and developer docs |

**LLM orchestration** includes YAML-based route and model registry configuration, multi-provider adapters (Azure OpenAI, OpenAI, mock, and others), fallback chains, and streaming support for generator routes.

### Student Retrieval

Student retrieval defaults to `RETRIEVAL_PROVIDER=s3_vector`. It embeds the normalized
student retrieval query with Bedrock Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`,
1024 normalized float dimensions), queries the S3 Vector runtime and pattern indexes, then
fetches the selected PatternGraph/SolveFlow bundle from DynamoDB before use. Vector metadata is
candidate-only; DynamoDB is authoritative.

Only fully approved runtime bundles may provide final-answer strategy. Pattern-assist bundles
provide a method hint only and require a fresh independent solve. When configuration, retrieval,
or graph compatibility checks fail, the flow uses `fresh_solve` without retrieval authority.

`ENABLE_COLBERT_RERANK=true` enables a bounded optional RAGatouille rerank over the S3 candidate
set only; timeout or dependency failures retain the original S3 order. Bedrock Knowledge Base
retrieval remains available only through `RETRIEVAL_PROVIDER=legacy_bedrock_kb` for legacy paths.

Required S3 Vector runtime configuration:

```bash
RETRIEVAL_PROVIDER=s3_vector
S3_VECTOR_BUCKET_NAME=<required>
DYNAMODB_PATTERN_TABLE=<required>
BEDROCK_EMBEDDING_MODEL_ID=amazon.titan-embed-text-v2:0
BEDROCK_EMBEDDING_DIMENSIONS=1024
BEDROCK_EMBEDDING_NORMALIZE=true
S3_VECTOR_DIMENSIONS=1024
S3_VECTOR_DISTANCE_METRIC=cosine
```

Live Bedrock, S3 Vector, and DynamoDB integration remains **[NOT VERIFIED]** in local CI;
unit tests use deterministic fakes and make no AWS calls.

### Image Questions

Image requests use an additional entry path without changing the text classifier or downstream
retrieval/generation nodes. The backend validates JPEG, PNG, or WebP content, normalizes EXIF
orientation, rejects unsafe references and unreadable inputs, and uses one structured multimodal
call to produce the existing `QueryClassification` contract. The default model is
`gemini-3.1-flash-lite`; the feature is disabled by default and requires
`GOOGLE_GEMINI_API_KEY` only when enabled.

```bash
IMAGE_CLASSIFIER_ENABLED=true
IMAGE_CLASSIFIER_PROVIDER=gemini
IMAGE_CLASSIFIER_MODEL=gemini-3.1-flash-lite
GOOGLE_GEMINI_API_KEY=<secret>
```

No image bytes, base64 payloads, signed URLs, prompts, or keys are logged. Default tests use fake
providers and make no Gemini calls. See
[docs/dev/image-question-classification.md](docs/dev/image-question-classification.md).

### Adaptive Verified Streaming

Orchestrated doubt-solver streams use a provider-neutral delivery policy. `adaptive`
is the default: only an explicitly low-risk request can forward provider chunks live;
all other requests are generated privately, pass the existing deterministic
answer-quality gate, receive at most one bounded regeneration, and replay the
approved Markdown answer. The stream keeps the existing `status`, `chunk`,
`complete`, and `error` event types. Its `complete` event now includes the
authoritative versioned Markdown response while `answer` remains the compatibility
projection.

```bash
ANSWER_DELIVERY_POLICY=adaptive
ANSWER_LIVE_STREAM_MIN_CLASSIFIER_CONFIDENCE=0.93
ANSWER_LIVE_STREAM_MIN_IMAGE_CONFIDENCE=0.90
ANSWER_LIVE_STREAM_MIN_PATTERN_CONFIDENCE=0.90
ANSWER_LIVE_STREAM_MAX_DIFFICULTY=basic
ANSWER_VERIFIER_ENABLED=true
ANSWER_VERIFIER_MAX_REPAIR_ATTEMPTS=1
ANSWER_REPLAY_MAX_CHUNK_CHARS=600
ANSWER_STREAM_HEARTBEAT_INTERVAL_SECONDS=10
```

`always_live` is blocked when `APP_ENV=production` unless
`ANSWER_ALLOW_ALWAYS_LIVE_IN_PRODUCTION=true` is deliberately approved. Live
streams never restart on a provider failure after visible output. Streaming responses
use one lifecycle boundary that guarantees `complete`, a controlled `error`, or a
confirmed disconnect. Raw SSE heartbeat comments keep long private generation and
verification requests active without creating application events. Disconnects cancel
future output and are logged as `client_disconnected`. An already in-flight synchronous
provider call may continue computing until the SDK call returns; its result is discarded
and never replayed after cancellation.

---

## Installation

### Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — Python package manager
- **Node.js 20+** — for the AgentCore CLI
- **AgentCore CLI** — `npm install -g @aws/agentcore`
- **AWS credentials** — only required for deployment or optional retrieval features

### Setup

```bash
git clone https://github.com/<your-org>/meritRankerTutor.git
cd meritRankerTutor

# Install Python dependencies
cd app && uv sync --group dev && cd ..

# Local environment (mock mode — no API keys required)
cp app/.env.local.example app/.env.local

# Verify tooling
make env
make check
agentcore validate
```

For the full environment variable reference, see [docs/dev/backend-env.md](docs/dev/backend-env.md).

---

## Quickstart

### 1. Run tests (offline, no credentials)

```bash
make check
```

### 2. Start the local agent

```bash
make dev
```

AgentCore starts a local HTTP server (default **http://localhost:8080**).

### 3. Send a doubt-solver request

In another terminal:

```bash
make smoke-doubt-solver
```

Or with curl directly:

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "doubt_solver",
    "query": "What is the formula for calculating percentage increase?",
    "user_id": "local-dev",
    "language": "en"
  }'
```

Mock mode (`ENABLE_REAL_LLM=false`, the default) returns deterministic placeholder answers without calling external providers.

### 4. Optional — real LLM providers

Set `ENABLE_REAL_LLM=true` and configure provider credentials in `app/.env.local`. See [docs/dev/backend-env.md](docs/dev/backend-env.md) for Azure OpenAI, OpenAI, and deployment mapping details.

Manual smoke targets (opt-in, not run in CI):

```bash
make smoke-llm-orchestration-mock      # orchestration dry-run, no network
make smoke-doubt-solver-real-llm       # requires running make dev + credentials
```

More examples: [examples/README.md](examples/README.md)

---

## Project structure

```
meritRankerTutor/
├── README.md                 # This file
├── LICENSE                   # MIT
├── CONTRIBUTING.md
├── SECURITY.md
├── ROADMAP.md
├── Makefile                  # test, lint, dev, smoke targets
├── AGENTS.md                 # AI coding-agent instructions
├── agentcore/                # AgentCore CLI / deployment config only (no Python app code)
│   ├── agentcore.json
│   └── cdk/
├── app/                      # All Python application code (deployed runtime)
│   ├── main.py               # AgentCore entrypoint
│   ├── config.py             # Settings from environment
│   ├── config/llm/           # Route and model registry YAML
│   ├── graphs/               # LangGraph workflow definitions
│   ├── schemas/              # Pydantic request/response/state models
│   ├── services/             # LLM, doubt solver, retrieval, secrets
│   ├── tools/                # LangGraph-callable tools (e.g. web search)
│   ├── prompts/              # Markdown prompt templates
│   ├── scripts/              # Manual smoke / dev scripts
│   └── tests/                # pytest suite (offline by default)
├── docs/dev/                 # Developer reference (env vars, orchestration flow)
├── examples/                 # Usage examples and smoke-test pointers
└── skills/                   # Repo-local engineering docs for contributors/agents
```

**Important boundary:** `agentcore/agentcore.json` declares `codeLocation: "app/"`. Only `app/` is packaged for deployment.

---

## Development commands

| Command | Description |
|---|---|
| `make env` | Print Python, uv, and AgentCore versions |
| `make test` | Run pytest |
| `make lint` | Ruff check |
| `make format` | Ruff format |
| `make check` | Lint + test (CI gate) |
| `make dev` | Start local AgentCore server |
| `make validate` | Validate `agentcore.json` |
| `make clean` | Remove cache and venv directories |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for phased plans: foundation, doubt-solver maturity, structured learning workflows, evaluation tooling, and optional production hardening.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, branch naming, testing, and PR expectations.

For security concerns, see [SECURITY.md](SECURITY.md).

---

## Responsible AI and accuracy

**AI outputs should be validated.** This toolkit generates explanatory content that may contain errors, omissions, or outdated information—especially for math, reasoning, and current-affairs topics.

- **Not a replacement for educators** — use for assistance and structured explanation, not as the sole authority for grading or high-stakes decisions.
- **Verify important facts** — downstream applications should show disclaimers and encourage cross-checking with textbooks, syllabi, or instructors.
- **Human review** — architect workflows so educators can review or override generated content where appropriate.
- **Privacy** — operators are responsible for how student queries are logged, stored, and sent to third-party model providers.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Related documentation

| Document | Purpose |
|---|---|
| [docs/dev/backend-env.md](docs/dev/backend-env.md) | Environment variables and local modes |
| [docs/dev/llm-orchestration-syntax-flow.md](docs/dev/llm-orchestration-syntax-flow.md) | LLM routing and orchestration flow |
| [docs/dev/image-question-classification.md](docs/dev/image-question-classification.md) | Image request, provider, cache, and testing contract |
| [app/tests/README.md](app/tests/README.md) | Test suite overview |
| [AGENTS.md](AGENTS.md) | Repository rules for coding agents |
