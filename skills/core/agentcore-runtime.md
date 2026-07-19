# AgentCore Runtime — meritRankerTutor

> Facts about the AgentCore CLI, deployment model, and local dev workflow.
> Read before touching `agentcore/` or `main.py`.

---

## How AgentCore Uses This Repository

```
agentcore/agentcore.json
  └── runtimes[0]
        ├── codeLocation: "app/"      ← AgentCore zips this directory
        ├── entrypoint:   "main.py"   ← entry file inside app/
        ├── build:        "CodeZip"
        └── runtimeVersion: "PYTHON_3_14"
```

**Implication:** Only files inside `app/` are visible to the deployed agent.
`skills/`, `agentcore/`, and root-level files are **not** deployed.

---

## Directory Rules

| Directory | Rule |
|---|---|
| `agentcore/` | AgentCore CLI config **only**.  Never put Python app code here. |
| `app/` | All Python source, deps (`pyproject.toml`, `uv.lock`), tests, prompts. |
| Root | `Makefile`, `README.md`, `AGENTS.md`, `.gitignore` only. |

**Do not:**
- Move or rename `agentcore/`.
- Add Python modules to `agentcore/`.
- Place `pyproject.toml` or `uv.lock` at the root — they belong in `app/`.
- Add FastAPI or any web framework to expose AgentCore endpoints — AgentCore provides the HTTP layer.
- Change `agentcore.json` without explicit task approval — name changes destroy and recreate CloudFormation resources.
- Add a new runtime entry in `agentcore.json` without architecture review.

---

## Local Development Workflow

### First-time setup

```bash
# Install Python deps (run from app/)
cd app && uv sync --group dev
```

### Daily commands (run from project root via Makefile)

```bash
make env        # print Python, uv, agentcore versions — confirm tools are working
make check      # ruff check + pytest — must pass before every response
agentcore validate   # validate agentcore.json against AgentCore schema
make dev        # start local AgentCore HTTP server with live logs
```

### Sending a test request

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "user_id": "local-user", "mode": "demo"}'
```

> **Note:** Confirm the exact local port/path from `agentcore dev` output.
> If it differs from `/invocations`, update README.md only — do not add FastAPI.

---

## `main.py` Entrypoint Contract

```python
from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload: dict) -> dict | Response:
    # 1. Validate payload with AgentRequest (Pydantic)
    # 2. Build graph state
    # 3. Invoke graph
    # 4. Return AgentResponse.model_dump()
    ...

if __name__ == "__main__":
    app.run()
```

- `main.py` must remain thin: logging setup, graph build, entrypoint decorator, done.
- All business logic belongs in `graphs/`, `services/`, or `tools/`.
- Orchestrated streaming returns a Starlette `StreamingResponse`, which AgentCore passes
  through directly. `stream_events_as_sse()` owns UTF-8 SSE framing, heartbeat comments,
  disconnect cancellation, and terminal enforcement. Application events remain
  `status|chunk|complete|error`.

---

## Environment Variables

Loaded in priority order (highest first):

1. Real environment variables set by the shell or AgentCore runtime.
2. `app/.env.local` — local developer secrets (gitignored).
3. Defaults in `app/config.py`.

Copy `app/.env.local.example` → `app/.env.local` and customise:

```bash
APP_ENV=local
LOG_LEVEL=INFO
MODEL_PROVIDER=mock
```

---

## agentcore.json Key Fields

| Field | Value | Notes |
|---|---|---|
| `name` | `meritRankerTutor` | CloudFormation logical ID — do not rename casually |
| `runtimes[0].codeLocation` | `app/` | Must always point to `app/` |
| `runtimes[0].entrypoint` | `main.py` | File inside `app/` |
| `runtimes[0].runtimeVersion` | `PYTHON_3_14` | Managed AgentCore runtime |
| `runtimes[0].build` | `CodeZip` | Zip-based deployment |
| `runtimes[0].protocol` | `HTTP` | Standard HTTP |

---

## Deployment

> TODO: Fill in deployment notes once staging/production accounts are configured.

```bash
# Deploy to AWS (requires configured aws-targets.json and credentials)
agentcore deploy

# Check status
agentcore status

# View logs
agentcore logs
```

---

## AgentCore CLI Reference

| Command | Description |
|---|---|
| `agentcore dev` | Start local server with hot-reload |
| `agentcore validate` | Validate agentcore.json schema |
| `agentcore deploy` | Deploy to AWS via CDK |
| `agentcore status` | Show deployment status |
| `agentcore invoke` | Invoke deployed agent |
| `agentcore logs` | View agent logs |
| `agentcore traces` | View agent traces |
