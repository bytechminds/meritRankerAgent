# Backend Environment Variables — Doubt Solver

> **File:** `docs/dev/backend-env.md`
> **Updated:** 2026-07-16
> **Related:** `app/.env.local.example`, `app/config.py`, `skills/features/doubt-solver.md`

This document is the canonical reference for every environment variable used by the
MeritRanker Tutor backend.  Read it before running Doubt Solver locally in any mode.

---

## Quick start (mock-only — no credentials needed)

```bash
cp app/.env.local.example app/.env.local
# All defaults work for mock mode — no changes required.
make dev       # start local AgentCore server (separate terminal)
make smoke-doubt-solver
```

All feature flags default to `false`.  The agent runs entirely in-process with the
mock LLM provider.  No AWS credentials, no API keys required.

---

## Safe local setup

1. Copy the example file — never edit the example file directly:
   ```bash
   cp app/.env.local.example app/.env.local
   ```
2. `app/.env.local` is listed in `.gitignore` — it will never be committed.
3. Fill in only the values you need for the mode you want to test.
4. `app/config.py` loads `.env.local` via `python-dotenv` at module load time.
   Real shell env vars always win (`override=False`).

**NEVER** put real secrets in `app/.env.local.example`.  That file is committed.

---

## Complete env var reference

### General

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `APP_ENV` | `local` | all modes | `local` / `staging` / `production` | Environment tag; `production` also forces JSON stdout and disables local log files |
| `LOG_LEVEL` | `INFO` | all modes | `DEBUG` / `INFO` / `WARNING` / `ERROR` | Backward-compatible Python logging level |
| `AGENT_LOG_LEVEL` | `LOG_LEVEL` | all modes | `DEBUG` / `INFO` / `WARNING` / `ERROR` | Preferred agent logging level |
| `AGENT_LOG_FORMAT` | local: `pretty_and_json_file`; production: `json` | all modes | `pretty` / `json` / `pretty_and_json_file` | Production always uses JSON |
| `AGENT_LOG_FILE_ENABLED` | local: `true`; production: `false` | local only | `true` / `false` | Ignored and disabled in production |
| `AGENT_DETAILED_LOGS` | `false` | local diagnostics | `true` / `false` | Adds bounded safe detail fields to pretty events |
| `AGENT_OBSERVABILITY_ENABLED` | `true` | all modes | `true` / `false` | Enables optional application spans; safe no-op without OpenTelemetry |
| `MODEL_PROVIDER` | `mock` | all modes | `mock` | Legacy field — not used for routing; kept for compatibility |

### LLM routing

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `ENABLE_REAL_LLM` | `false` | real LLM | `true` / `false` | Master switch. When `false` all LLM calls use the mock provider |
| `LLM_DEFAULT_PROVIDER` | `mock` | real LLM | `mock` / `azure_openai` / `openai` | Used as fallback provider name when a role has no config |
| `LLM_ROLE_CONFIG_JSON` | `{}` | real LLM | See format below | JSON map of role name → provider config |

**`LLM_ROLE_CONFIG_JSON` format** (single-line JSON, two roles):

```json
{
  "doubt_solver_classifier": {
    "provider": "azure_openai",
    "model_label": "gpt-4o-mini",
    "deployment": "YOUR_CLASSIFIER_DEPLOYMENT",
    "temperature": 0,
    "max_tokens": 500,
    "supports_streaming": false
  },
  "doubt_solver_generator": {
    "provider": "azure_openai",
    "model_label": "gpt-4o",
    "deployment": "YOUR_GENERATOR_DEPLOYMENT",
    "temperature": 0.2,
    "max_tokens": 1200,
    "supports_streaming": false
  }
}
```

For OpenAI native replace `"provider": "azure_openai"` with `"provider": "openai"` and add `"model": "gpt-4o"`.
Remove the `"deployment"` field (not used by the OpenAI provider).

### Azure OpenAI credentials

> Required when any role in `LLM_ROLE_CONFIG_JSON` sets `"provider": "azure_openai"`.

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | _(empty)_ | Azure OpenAI | `https://MY-RESOURCE.openai.azure.com/` | Must include trailing slash |
| `AZURE_OPENAI_API_KEY` | _(empty)_ | Azure OpenAI | `abc123...` (32+ char hex) | **Secret — never commit** |
| `AZURE_OPENAI_API_VERSION` | _(empty)_ | Azure OpenAI | `2024-02-01` | Azure API version string |

Missing any of these when `provider=azure_openai` → `LlmConfigurationError` at call time.

### OpenAI native credentials

> Required when any role in `LLM_ROLE_CONFIG_JSON` sets `"provider": "openai"`.

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `OPENAI_API_KEY` | _(empty)_ | OpenAI native | `sk-...` | **Secret — never commit** |
| `OPENAI_BASE_URL` | _(empty — uses OpenAI default)_ | optional | `https://api.openai.com/v1` | Override only for custom proxies |

Missing `OPENAI_API_KEY` when `provider=openai` → `LlmConfigurationError` at call time.

### Bedrock Knowledge Base retrieval

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `ENABLE_KB_RETRIEVAL` | `false` | Bedrock KB | `true` / `false` | When `false` → `retrieval_source="disabled"`, no AWS call |
| `BEDROCK_KB_ID` | _(empty)_ | Bedrock KB | `ABCDEF1234` (10-char alphanumeric) | **Required** when `ENABLE_KB_RETRIEVAL=true`. Missing → `KnowledgeBaseConfigurationError` |
| `BEDROCK_KB_REGION` | _(falls back to `AWS_REGION`)_ | Bedrock KB | `us-east-1` | AWS region for the Bedrock endpoint. Leave empty to use `AWS_REGION` |
| `BEDROCK_KB_MAX_RESULTS` | `5` | optional | `1`–`20` (integer) | Max KB results per query |
| `BEDROCK_KB_MIN_SCORE` | _(empty — no threshold)_ | optional | `0.7` (float 0–1) | Minimum similarity score filter. Leave empty to return all results |

### Image-question classification

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `IMAGE_CLASSIFIER_ENABLED` | `false` | image questions | `true` / `false` | Disabled image requests return a controlled unsupported state; text requests are unchanged |
| `IMAGE_CLASSIFIER_PROVIDER` | `gemini` | image questions | `gemini` | Provider selection behind the provider-neutral interface |
| `IMAGE_CLASSIFIER_MODEL` | `gemini-3.1-flash-lite` | image questions | stable model ID | No preview identifier |
| `GOOGLE_GEMINI_API_KEY` | _(empty)_ | enabled image questions | secret | Required at startup only when enabled; never expose to clients |
| `IMAGE_CLASSIFIER_TIMEOUT_MS` | `15000` | optional | positive integer | Provider request and single-flight wait budget |
| `IMAGE_CLASSIFIER_MAX_IMAGE_BYTES` | `8388608` | optional | positive integer | Pre-provider payload limit |
| `IMAGE_CLASSIFIER_MAX_OUTPUT_TOKENS` | `1400` | optional | positive integer | Structured extraction/classification output cap |
| `IMAGE_CLASSIFIER_CACHE_TTL_SECONDS` | `300` | optional | integer >= 0 | Bounded process-local deterministic-result cache; zero disables it |
| `IMAGE_CLASSIFIER_MIN_CONFIDENCE` | `0.65` | optional | float 0–1 | Minimum extraction/classification confidence |
| `IMAGE_CLASSIFIER_MAX_DIMENSION` | `2400` | optional | positive integer | Oversized images are resized proportionally with high-quality resampling |

### Adaptive verified streaming

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `ANSWER_DELIVERY_POLICY` | `adaptive` | orchestrated streams | `always_verified` / `adaptive` / `always_live` | `adaptive` streams live only for an approved low-risk profile; otherwise it verifies and replays |
| `ANSWER_LIVE_STREAM_MIN_CLASSIFIER_CONFIDENCE` | `0.93` | optional | float 0-1 | Minimum classifier confidence for adaptive live delivery |
| `ANSWER_LIVE_STREAM_MIN_IMAGE_CONFIDENCE` | `0.90` | optional | float 0-1 | Minimum image extraction/classification confidence for adaptive live delivery |
| `ANSWER_LIVE_STREAM_MIN_PATTERN_CONFIDENCE` | `0.90` | optional | float 0-1 | Minimum confidence when approved runtime retrieval is used |
| `ANSWER_LIVE_STREAM_MAX_DIFFICULTY` | `basic` | optional | supported route difficulty | Highest difficulty eligible for adaptive live delivery |
| `ANSWER_VERIFIER_ENABLED` | `true` | verified replay | `true` / `false` | Runs the existing deterministic answer-quality approval gate before replay |
| `ANSWER_VERIFIER_MAX_REPAIR_ATTEMPTS` | `1` | verified replay | `0` / `1` | Bounded private regeneration limit; draft output is never streamed |
| `ANSWER_REPLAY_MAX_CHUNK_CHARS` | `600` | verified replay | positive integer | Target size only; Markdown-safe blocks can be larger |
| `ANSWER_STREAM_HEARTBEAT_INTERVAL_SECONDS` | `10` | streaming transport | positive float | Emits private-content-free SSE comments while the application event iterator is idle |
| `ANSWER_ALLOW_ALWAYS_LIVE_IN_PRODUCTION` | `false` | production `always_live` | `true` / `false` | Explicit production approval guard; required when `APP_ENV=production` and policy is `always_live` |

### DynamoDB record fetch

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `ENABLE_DYNAMODB_FETCH` | `false` | DynamoDB | `true` / `false` | When `false` → returns `[]`/`None`, no AWS call |
| `DYNAMODB_QUESTION_TABLE` | _(empty)_ | DynamoDB | `meritranker-questions-dev` | **Required** when `ENABLE_DYNAMODB_FETCH=true`. Missing → `DynamoDbConfigurationError` |
| `DYNAMODB_PATTERN_TABLE` | _(empty)_ | DynamoDB | `meritranker-patterns-dev` | **Required** when fetching pattern records. Missing → `DynamoDbConfigurationError` |
| `DYNAMODB_DEFAULT_INDEX` | _(empty)_ | optional | `query_index` | Default GSI/LSI name for index-based queries |
| `DYNAMODB_REGION` | _(falls back to `AWS_REGION`)_ | optional | `ap-south-1` | AWS region for the DynamoDB client. Leave empty to use `AWS_REGION` |

### AWS credentials (boto3 default chain)

> These are consumed by boto3's built-in credential resolution chain — **not** read directly
> by `config.py`.  boto3 resolves credentials in this order:
>
> 1. Environment variables below
> 2. `~/.aws/credentials` (shared credentials file)
> 3. `~/.aws/config` (profile config or SSO)
> 4. IAM instance profile (EC2 / ECS / Lambda)
>
> For local development, **prefer** setting `AWS_PROFILE` and using `aws sso login`
> instead of long-term access keys.

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `AWS_REGION` | _(boto3 chain)_ | Bedrock KB, DynamoDB | `us-east-1` | Default region for all AWS clients. Can be overridden per-service |
| `AWS_ACCESS_KEY_ID` | _(boto3 chain)_ | Bedrock KB, DynamoDB | `AKIAIOSFODNN7EXAMPLE` | **Secret — never commit**. Use `~/.aws/credentials` or IAM role instead |
| `AWS_SECRET_ACCESS_KEY` | _(boto3 chain)_ | Bedrock KB, DynamoDB | `wJalrXUtnFEMI/...` | **Secret — never commit** |
| `AWS_SESSION_TOKEN` | _(boto3 chain)_ | Bedrock KB, DynamoDB | _(long token string)_ | Required only for temporary STS credentials |
| `AWS_PROFILE` | _(boto3 chain)_ | optional | `my-sso-profile` | Named profile from `~/.aws/config`. Recommended for SSO |

### Context builder

| Variable | Default | Required for | Format / example | Notes |
|---|---|---|---|---|
| `DOUBT_SOLVER_MAX_CONTEXT_CHARS` | `6000` | all modes | `6000` (integer > 0) | Hard cap on context string length passed to the answer generator. Lower = cheaper; higher = richer context |

---

## Missing config error behaviour

This table documents exactly what happens when a required variable is missing or wrong:

| Scenario | Result | Error type |
|---|---|---|
| `ENABLE_REAL_LLM=true` + `LLM_ROLE_CONFIG_JSON={}` | `LlmConfigurationError`: "no role config found for role=…" | Hard error at LLM call time |
| `LLM_ROLE_CONFIG_JSON=not-valid-json` | `LlmConfigurationError`: "not valid JSON" | Hard error at LLM call time |
| `provider=azure_openai` + missing `AZURE_OPENAI_ENDPOINT` | `LlmConfigurationError`: "AZURE_OPENAI_ENDPOINT is not set" | Hard error at LLM call time |
| `provider=azure_openai` + missing `AZURE_OPENAI_API_KEY` | `LlmConfigurationError`: "AZURE_OPENAI_API_KEY is not set" | Hard error at LLM call time |
| `provider=azure_openai` + missing `AZURE_OPENAI_API_VERSION` | `LlmConfigurationError`: "AZURE_OPENAI_API_VERSION is not set" | Hard error at LLM call time |
| `provider=openai` + missing `OPENAI_API_KEY` | `LlmConfigurationError`: "OPENAI_API_KEY is not set" | Hard error at LLM call time |
| `ENABLE_KB_RETRIEVAL=true` + missing `BEDROCK_KB_ID` | `KnowledgeBaseConfigurationError`: "BEDROCK_KB_ID" | Hard error at retrieval call time |
| `ENABLE_DYNAMODB_FETCH=true` + missing `DYNAMODB_QUESTION_TABLE` | `DynamoDbConfigurationError`: "DYNAMODB_QUESTION_TABLE" | Hard error at fetch call time |
| `ENABLE_DYNAMODB_FETCH=true` + missing `DYNAMODB_PATTERN_TABLE` | `DynamoDbConfigurationError`: "DYNAMODB_PATTERN_TABLE" | Hard error at pattern fetch call time |
| `DOUBT_SOLVER_MAX_CONTEXT_CHARS=not-a-number` | `ValueError` from `int()` at settings load | Fast-fail at startup |
| `BEDROCK_KB_MAX_RESULTS=not-a-number` | `ValueError` from `int()` at settings load | Fast-fail at startup |
| `IMAGE_CLASSIFIER_ENABLED=true` + missing `GOOGLE_GEMINI_API_KEY` | `ConfigurationError` naming the missing variable | Fast-fail at startup |
| `IMAGE_CLASSIFIER_ENABLED=true` + unsupported provider or invalid limits | `ConfigurationError` / `ValueError` | Fast-fail at startup |
| `APP_ENV=production` + `ANSWER_DELIVERY_POLICY=always_live` without approval | `ConfigurationError` | Fast-fail at startup |

**Graph behaviour on service errors:**
When `KnowledgeBaseConfigurationError`, `KnowledgeBaseServiceError`, `DynamoDbConfigurationError`,
or `DynamoDbServiceError` are raised inside graph nodes, the graph catches them, sets
`service_error=True`, and the final response has `success=True, needs_review=True`.
The agent does not crash — it degrades gracefully and flags the response for review.

---

## Mode-by-mode env var requirements

### Mode 1: Mock only (default, no credentials)

No changes to the default `.env.local.example` are needed.

```
ENABLE_REAL_LLM=false
ENABLE_KB_RETRIEVAL=false
ENABLE_DYNAMODB_FETCH=false
```

Smoke command: `make smoke-doubt-solver`

---

### Mode 2: Real LLM only (no retrieval)

#### Azure OpenAI

```bash
ENABLE_REAL_LLM=true
LLM_ROLE_CONFIG_JSON={"doubt_solver_classifier":{"provider":"azure_openai","model_label":"gpt-4o-mini","deployment":"YOUR_CLASSIFIER_DEPLOYMENT","temperature":0,"max_tokens":500,"supports_streaming":false},"doubt_solver_generator":{"provider":"azure_openai","model_label":"gpt-4o","deployment":"YOUR_GENERATOR_DEPLOYMENT","temperature":0.2,"max_tokens":1200,"supports_streaming":false}}
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-02-01
```

#### OpenAI native

```bash
ENABLE_REAL_LLM=true
LLM_ROLE_CONFIG_JSON={"doubt_solver_classifier":{"provider":"openai","model_label":"gpt-4o-mini","model":"gpt-4o-mini","temperature":0,"max_tokens":500,"supports_streaming":false},"doubt_solver_generator":{"provider":"openai","model_label":"gpt-4o","model":"gpt-4o","temperature":0.2,"max_tokens":1200,"supports_streaming":false}}
OPENAI_API_KEY=<your-key>
```

Smoke command: `make smoke-doubt-solver-real-llm`

---

### Mode 3: Bedrock KB only (mock LLM)

```bash
ENABLE_KB_RETRIEVAL=true
BEDROCK_KB_ID=YOUR_KB_ID
BEDROCK_KB_REGION=us-east-1
AWS_REGION=us-east-1
# Plus AWS credentials via env vars, ~/.aws/credentials, or IAM role
```

Smoke command: `make smoke-doubt-solver-with-retrieval`

---

### Mode 4: DynamoDB only (mock LLM, no KB)

```bash
ENABLE_DYNAMODB_FETCH=true
DYNAMODB_QUESTION_TABLE=meritranker-questions-dev
DYNAMODB_PATTERN_TABLE=meritranker-patterns-dev
DYNAMODB_REGION=ap-south-1
AWS_REGION=ap-south-1
# Plus AWS credentials
```

Smoke command: `make smoke-doubt-solver-with-retrieval`

---

### Mode 5: Combined — real LLM + KB + DynamoDB

All vars from modes 2, 3, and 4 combined.

Smoke command: `make smoke-doubt-solver-combined`

---

## Smoke test commands

| Command | What it tests | Credentials needed |
|---|---|---|
| `make smoke-doubt-solver` | Full graph, mock LLM, no retrieval | None |
| `make smoke-doubt-solver-real-llm` | Full graph, real LLM, no retrieval | Azure OpenAI or OpenAI |
| `make smoke-doubt-solver-with-retrieval` | Full graph, mock LLM, real KB + DynamoDB | AWS |
| `make smoke-doubt-solver-combined` | Full graph, real LLM + KB + DynamoDB | Azure/OpenAI + AWS |

All commands require `make dev` to be running in a separate terminal first.

---

## [NOT VERIFIED] items

The following have not been manually verified end-to-end:

- `[NOT VERIFIED]` Real Azure OpenAI response quality and latency
- `[NOT VERIFIED]` Real OpenAI native response quality and latency
- `[NOT VERIFIED]` Real Bedrock KB retrieval results shape (schema assumes Bedrock API v2024+)
- `[NOT VERIFIED]` Real DynamoDB table schema (`question_id` as primary key is assumed)
- `[NOT VERIFIED]` AWS IAM permissions needed for KB + DynamoDB read access
- AgentCore HTTP streaming is locally verified with the controlled mock executor for
  canonical completion and client-abort cancellation. Live-provider HTTP streaming
  remains `[NOT VERIFIED]`.
- `[NOT VERIFIED]` AWS SSO / named profile credential chain in `agentcore dev` context

---

## Security notes

1. `app/.env.local` is in `.gitignore` — confirmed in root `.gitignore` lines 26–28.
2. `app/.env.local.example` contains non-secret local defaults and empty secret placeholders.
3. The `azure_openai_provider.py` and `openai_provider.py` log `role` and `model_label`
   only — never API keys or endpoints.
4. `bedrock_kb_service.py` and `dynamodb_service.py` never log retrieved content.
5. `_sanitise_metadata()` in `streaming_adapter.py` enforces an allowlist
   (`_SAFE_METADATA_KEYS`) so secrets never enter `StreamEvent.metadata`.
6. boto3 uses the standard credential chain — no credentials are hardcoded anywhere.
`AGENT_LOCAL_LOG_CONTENT` accepts `off` or `preview`. Local development defaults to `preview`;
production always behaves as `off`. Preview content is bounded and written only to
`app/.logs/agent-runtime.log`.
