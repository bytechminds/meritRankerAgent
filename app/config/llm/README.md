# app/config/llm/README.md

## LLM Orchestration Config

This directory contains the non-secret YAML routing configuration for the LLM
orchestration layer.

---

### Part 9.2 — Azure API Mode Split (current)

Azure OpenAI resources can be accessed in two API styles.  The style is set
via `azure_api_mode` in `provider_profiles.yaml`.

#### Mode 1: `azure_deployment_chat_completions` (classic)

For endpoints of the form `https://<resource>.openai.azure.com` (no path suffix).
The SDK builds: `<endpoint>/openai/deployments/<deployment>/chat/completions?api-version=...`

```yaml
azure_primary:
  provider: azure_openai
  azure_api_mode: azure_deployment_chat_completions
  endpoint_env: AZURE_OPENAI_ENDPOINT   # must NOT end with /openai/v1
  api_key_env: AZURE_OPENAI_API_KEY
  api_version_env: AZURE_OPENAI_API_VERSION
```

#### Mode 2: `azure_openai_v1` (OpenAI-compatible — current)

For endpoints of the form `https://<resource>.openai.azure.com/openai/v1`.
The adapter uses `OpenAI(base_url=..., api_key=...)` — no deployment path.
`deployment` is passed as `model=` parameter.  `api_version` is NOT sent.

```yaml
azure_foundry_v1:
  provider: azure_openai
  azure_api_mode: azure_openai_v1
  endpoint_env: AZURE_OPENAI_ENDPOINT   # must end with /openai/v1
  api_key_env: AZURE_OPENAI_API_KEY
```

Currently `model_registry.yaml` uses `azure_foundry_v1` for all Azure models
because `AZURE_OPENAI_ENDPOINT` ends with `/openai/v1`.

#### Azure-hosted DeepSeek profile (`azure_deepseek`)

Separate from Azure GPT (`azure_foundry_v1`) and native DeepSeek (`deepseek_primary`):

```yaml
azure_deepseek:
  provider: azure_openai
  azure_api_mode: azure_deployment_chat_completions
  endpoint_env: AZURE_DEEPSEEK_ENDPOINT
  api_key_env: AZURE_DEEPSEEK_API_KEY
  api_version_env: AZURE_DEEPSEEK_API_VERSION
  optional_api_key: true
  optional_endpoint: true
  optional_api_version: true
```

Model aliases: `deepseek_azure_reasoning_generator`, `deepseek_azure_standard_generator`,
`deepseek_azure_advanced_generator`. Deployments from `AZURE_DEEPSEEK_*_DEPLOYMENT` env
vars. Inactive by default — optional test routes: `*.generator.deepseek_azure_test`.
For Foundry `/openai/v1` endpoints, switch profile `azure_api_mode` to `azure_openai_v1`.

**Validation:**
- `azure_deployment_chat_completions`: rejects endpoint ending with `/openai/v1`
  or containing `/api/projects/`. Requires `api_version`.
- `azure_openai_v1`: rejects endpoint not ending with `/openai/v1`. Does not
  require `api_version`.

---

### Part 9.1 — Azure-First Provider Strategy (current)

**Provider execution order:**

All primary model aliases (`math_basic_generator`, `math_reasoning_generator`,
`reasoning_standard_generator`, `english_fast_generator`, `general_fast_generator`)
are wired to Azure OpenAI as the **first provider**.

If Azure fails for a **controlled reason** (quota exhaustion, rate limit, timeout,
authentication failure, provider unavailability, unknown provider error), the
executor automatically tries the corresponding `_openai_native` fallback alias
which uses native OpenAI.

**`fallback_models` field (`model_registry.yaml`):**

```yaml
math_basic_generator:
  provider: azure_openai
  provider_profile: azure_foundry_v1
  deployment: gpt-4o
  fallback_models:
    - math_basic_generator_openai_native

math_basic_generator_openai_native:
  provider: openai
  provider_profile: openai_primary
  model_id: gpt-4o-mini
```

- `fallback_models` is a list of model **aliases** (not provider model IDs).
- Maximum 3 fallback aliases per primary model.
- Fallback aliases must exist in the registry; cycles and self-references are
  rejected at startup.
- Routes always reference **primary** aliases. Fallback aliases never appear in
  the routes section.

**Eligible failure kinds (trigger fallback):**

| `failure_kind` | Condition |
|---|---|
| `insufficient_quota` | 429 + `"insufficient_quota"` in error body |
| `rate_limited` | 429 generic rate-limit |
| `authentication_failed` | 401 / 403 / invalid API key |
| `model_not_found` | 404 / Azure deployment not found |
| `timeout` | Request timeout |
| `provider_unavailable` | Connection error / 5xx |
| `unknown_provider_error` | Unrecognized error type |

**Ineligible failure kinds (no fallback, fail immediately):**

| `failure_kind` | Reason |
|---|---|
| `invalid_request` | `BadRequestError` — programming error; fallback will fail too |

**`max_retries=0`:**

Both `OpenAIProviderAdapter` and `AzureOpenAIProviderAdapter` set `max_retries=0`
when building the SDK client. This disables SDK-level retry loops, which are
inappropriate for interactive tutoring sessions (bad latency). Fallback is handled
at the model-execution level, not the SDK level.

**Graph-level handling:**

If all primary + fallback aliases are exhausted, `RegistryBackedModelExecutor`
raises `ProviderExecutionError`. The `_generate_node` in the orchestrated doubt
solver graph catches this error and returns a safe user-facing message:

> "I couldn't generate the answer right now because the AI provider is unavailable
> or quota-limited. Please try again later."

Unexpected non-provider errors (e.g., programming bugs) propagate loudly and are
not silenced.

**Real Azure deployments** — deployment names are set to actual Azure deployment
names (`gpt-4o`, `gpt-4o-mini`). The preflight guard confirms no `YOUR_*`
placeholders remain at startup.

**`[NOT VERIFIED]`** Native OpenAI fallback success — requires active OpenAI billing quota.

**`[DEFER]`** Per-user-plan fallback policies (e.g., different fallback for free vs. paid).

**`[DEFER]`** Retry/backoff policy within a provider (currently `max_retries=0`).

**`[DEFER]`** Real provider streaming fallback.

---

### `llm_orchestration.yaml`

**What it contains:**

- `routes` — subject / task-role / difficulty routing table with model aliases,
  prompt file paths (relative), temperature, max_tokens, and fallback chains.
- `models` — model catalog mapping aliases to provider + capability metadata.
- `provider_profiles` — provider connection metadata (env var names only, never
  actual secret values).

**What it does NOT contain:**

- API keys, tokens, passwords, or any secret value.
- Provider profiles store only environment variable **names** (e.g.
  `GEMINI_API_KEY`). The actual values are loaded at runtime from the process
  environment or AWS Secrets Manager.

---

### Prompt paths

Routes reference prompt files such as `subjects/math_generator.md`.  
These paths are relative to `app/prompts/` and are **referenced but not loaded**
in Part 1 (config registry + route resolver).  
Prompt loading is implemented in Part 2 (PromptResolver).

---

### How config is loaded

The `LlmConfigRegistry` in
`app/services/llm_orchestration/config_registry.py`:

1. Reads this YAML **once** at startup (or on first registry access).
2. Validates it with strict Pydantic v2 models.
3. Resolves route inheritance (`inherits: default` etc.) at build time.
4. Compiles `route_map`, `model_map`, and `provider_profile_map` as in-memory
   Python dicts.

No YAML parsing happens at request time. Route lookups are pure dict lookups.

---

### Changing the config

Any change to this file requires an application **restart** (or redeploy) to take
effect, because the config is loaded once and compiled into memory.

Future: AgentCore config bundle source support will allow hot-reload without
redeploy. This is deferred (`[DEFER]`).

---

## Prompt files (Part 2)

Prompt files are local Markdown (`.md`) files stored in `app/prompts/`.

---

### Directory structure

```
app/prompts/
├── subjects/
│   ├── math_generator.md
│   ├── reasoning_generator.md
│   ├── english_generator.md
│   └── general_generator.md
├── levels/
│   ├── basic.md
│   ├── intermediate.md
│   └── advanced.md
└── intents/
    ├── solve.md
    ├── explain.md
    └── practice.md
```

Root-level files (`demo.md`, `query_classifier.md`, `answer_generator.md`) are used
by the existing demo and doubt-solver flows via `prompt_loader.py` and are separate
from the orchestration layer prompts above.

---

### How prompt paths work

- All paths in `llm_orchestration.yaml` under `routes[*][*][*].prompt` and
  `routes[*][*][*].overlays` are **relative to `app/prompts/`**.
- Example: `prompt: subjects/math_generator.md` resolves to
  `app/prompts/subjects/math_generator.md`.
- Paths must be `.md` files, must not contain `..`, must not be absolute, and must
  not be URLs.  The `PromptResolver` enforces these rules before loading.

---

### Caching

- Prompt file content is cached **in process memory** after first load, per
  `PromptResolver` instance.
- No disk I/O occurs for subsequent requests that use the same prompt path.
- Cache is per-instance — the module-level singleton (`get_prompt_resolver()`)
  persists for the lifetime of the process.

---

### Changing prompt files

Any change to a prompt file in `app/prompts/` requires an application **restart**
(or redeploy) to take effect, because content is cached after first load.

`[DEFER]` Prompt hot-reload (re-read on change) is not yet implemented.  
`[DEFER]` AgentCore config bundle prompt source is not yet implemented.  
`[DEFER]` Langfuse prompt management integration is not yet implemented.

---

### Security

- Prompt files must not contain secrets, API keys, tokens, or credentials.
- Retrieved student context is injected into the **user message only** — not into
  the system prompt.  The `PromptResolver` enforces this boundary.

---

## SecretResolver Foundation (Part 5)

Part 5 adds a reusable secret resolution layer under `app/services/secrets/`.

### Design

Provider profiles in `llm_orchestration.yaml` store **environment variable names
only** (e.g. `api_key_env: GEMINI_API_KEY`).  They never contain actual key values.

The `SecretResolver` protocol reads the actual values at runtime via an injected
backend.  In local development the `EnvSecretResolver` is used — it reads from
`os.environ` at resolve time only.

### How secrets are resolved (local development)

1. The provider profile in the YAML contains `api_key_env: GEMINI_API_KEY`.
2. `ProviderCredentialResolver.resolve(profile)` calls
   `secret_resolver.get_secret("GEMINI_API_KEY")`.
3. `EnvSecretResolver` reads `os.environ["GEMINI_API_KEY"]`.
4. Returns a `ProviderCredentials` object — never log this directly; use
   `creds.safe_metadata()` instead.

### What is NOT done yet

- `[DEFER]` AWS Secrets Manager resolver (`SecretsManagerSecretResolver`).
- `[DEFER]` AgentCore Identity resolver (`AgentCoreIdentitySecretResolver`).
- `[DEFER]` `credential_ref` runtime resolution — recognized but raises
  `SecretResolverUnsupportedError` in Part 5.
- `[DEFER]` Provider adapter wiring — `RegistryBackedModelExecutor` does not yet
  call `ProviderCredentialResolver`; that is deferred to Part 6.

### Rules

- Config bundles (`llm_orchestration.yaml`) must never contain actual secrets.
- `ProviderCredentials` must never be logged or included in metadata directly.
  Use `ProviderCredentials.safe_metadata()` for observability.
- `EnvSecretResolver` validates the env var name (must be `SCREAMING_SNAKE_CASE`)
  before reading.  Names that look like raw secrets (prefixed `sk-`, `AIza`,
  `AKIA`, `-----BEGIN`) are rejected.
- No env read occurs at import time or resolver construction time.
- The system prompt is composed from developer-controlled `.md` files only.


---

### Security invariants

- Do not add actual API keys to this file.
- Do not add actual model endpoints to this file.
- `provider_profiles` entries must contain only environment variable name strings
  in `SCREAMING_SNAKE_CASE` format.
- Secret-like values (`sk-...`, `AIza...`, `AKIA...`, `-----BEGIN ...`) are
  rejected at validation time by `ProviderProfile`.
- Model aliases and capability metadata are placeholders until live evaluation.
  `[NOT VERIFIED]`

---

## LlmOrchestrator (Part 3)

`LlmOrchestrator` in `app/services/llm_orchestration/orchestrator.py` is the
service-level coordinator that chains the three orchestration components:

```
RouteResolver  →  PromptResolver  →  ModelExecutor  →  OrchestrationResult
```

### What it does

1. **Accepts** a `RouteRequest`, a student query string, an optional
   classification object, and optional retrieved context.
2. **Validates** the query is non-empty and within `MAX_QUERY_CHARS` (4 000).
3. **Resolves** a `RouteDecision` via the injected `route_resolver_fn`.
4. **Builds** `list[LlmMessage]` via the injected `PromptResolver`.
5. **Calls** the injected `ModelExecutor.execute()` boundary.
6. **Returns** a safe, normalised `OrchestrationResult`.

### What it does NOT do (deferred)

- `[DEFER]` Real model provider calls — no Gemini, Azure OpenAI, or OpenAI SDK
  calls are made in Part 3.  A `MockModelExecutor` is provided for tests.
- `[DEFER]` `model_router.py` / `ModelRouterExecutor` adapter — the existing
  `model_router.py` uses `LlmRoleConfig` + role string and is not yet wired.
  Deferred to Part 5+.
- `[DEFER]` SecretResolver — no API key reading.
- `[DEFER]` Graph / `answer_generator_service.py` wiring — no graph changes.
- `[DEFER]` AgentCore config bundle prompt source.
- `[DEFER]` Langfuse prompt management integration.
- `[DEFER]` Provider-level fallback execution.

### OrchestrationResult safety

`OrchestrationResult` intentionally does **not** expose:
- The composed `messages` list (system prompt + user message).
- The student's original query string.
- The retrieved context string.
- Any classification data.
- Any API key, secret, or credential.

Tests that need to inspect the composed messages should use
`MockModelExecutor.last_messages` (available when using
`create_mock_orchestrator_for_tests()`).

### Dependencies

| Component | Part | Default |
|---|---|---|
| `route_resolver_fn` | Part 1 | `resolve_route` |
| `prompt_resolver` | Part 2 | `get_prompt_resolver()` singleton |
| `model_executor` | Part 3 | Required — no implicit default |

### Schemas

- `ModelExecutionResult` — `app/schemas/llm_orchestration.py`
- `OrchestrationResult` — `app/schemas/llm_orchestration.py`

Both schemas validate `metadata` fields and reject unsafe keys (`prompt`,
`query`, `context`, `api_key`, `secret`, `credential`, etc.) at construction
time.

## Model Execution Boundary (Part 4)

Part 4 adds registry-backed model config resolution and a clean execution
boundary without real provider calls.

```
RouteDecision  →  ModelConfigResolver  →  RegistryBackedModelExecutor
               →  ProviderExecutor (injected)  →  ModelExecutionResult
```

### What it does

1. Resolves `RouteDecision.model` to a `ModelConfig`.
2. Resolves `ModelConfig.provider_profile` to a `ProviderProfile`.
3. Validates provider/profile consistency.
4. Validates boundary-level provider options, including rejecting
   `thinking=true` when the model does not support thinking.
5. Builds an internal `ProviderExecutionRequest`.
6. Delegates execution to an injected `ProviderExecutor`.

### What it does NOT do (deferred)

- `[DEFER]` Real provider adapters.
- `[DEFER]` SecretResolver and runtime credential fetching.
- `[DEFER]` Actual fallback execution.
- `[DEFER]` Graph / `answer_generator_service.py` wiring.
- `[DEFER]` Provider SDK, AWS, boto3, or network calls.

`ResolvedModelConfig.safe_metadata` contains only `model_alias`, `provider`,
`supports_streaming`, `supports_thinking`, and `timeout_seconds`.  Provider
profile env-var references and credential refs are not copied into safe
metadata or logs.

## Provider Adapter Foundation (Part 6)

Part 6 adds concrete provider adapter implementations under
`app/services/llm_providers/`.  Each adapter wraps a real SDK client
(or fake client for tests) and produces a normalized `ModelExecutionResult`.

### New classes (coexist alongside legacy `BaseLlmProvider` hierarchy)

| Class | File | Purpose |
|---|---|---|
| `ProviderAdapter` | `base.py` | `@runtime_checkable` Protocol — duck-type interface |
| `sanitize_provider_metadata` | `base.py` | Strips unsafe keys from metadata dicts |
| `MockProviderAdapter` | `mock_provider.py` | In-process fake for tests; records `last_request`/`call_count` |
| `OpenAIProviderAdapter` | `openai_provider.py` | Wraps `openai.OpenAI`; requires `api_key` + `model_id` |
| `AzureOpenAIProviderAdapter` | `azure_openai_provider.py` | Supports `azure_deployment_chat_completions` (AzureOpenAI SDK) and `azure_openai_v1` (OpenAI SDK + base_url). Requires `api_key`, `endpoint`, `deployment`. `api_version` required only in classic mode. |
| `GeminiProviderAdapter` | `openai_compatible_adapter.py` | OpenAI-compatible Gemini endpoint; text + optional `generate_with_image()` (adapter-level). |
| `DeepSeekProviderAdapter` | `openai_compatible_adapter.py` | OpenAI-compatible DeepSeek chat/reasoner endpoint. |
| `ProviderAdapterFactory` | `provider_factory.py` | Maps provider names → adapter instances; supports custom injection |
| `ProviderAdapterExecutor` | `llm_orchestration/model_execution.py` | Resolves credentials → gets adapter → calls `generate()` |

### Error hierarchy (new — independent of legacy `LlmProviderError`)

```
LlmProviderAdapterError
├── LlmProviderConfigurationError  (missing credential / model_id / deployment)
├── LlmProviderExecutionError      (SDK exception during call)
├── LlmProviderResponseError       (empty/None content in response)
└── LlmProviderUnsupportedFeatureError
```

### Security invariants

- SDK imports are deferred (`from openai import OpenAI` inside `_build_client`).
  The `openai` package is not imported at module load time.
- `client_factory` injection replaces network construction in all tests.
  No env reads, no network calls in the test suite.
- Credential values (`api_key`, `endpoint`) are never included in error messages
  or `ModelExecutionResult.metadata`.
- `sanitize_provider_metadata` enforces `_UNSAFE_METADATA_KEYS` on every
  metadata dict that flows through the adapter layer.

### Wire-up example

```python
from services.llm_orchestration.model_execution import ProviderAdapterExecutor
from services.llm_providers.provider_factory import ProviderAdapterFactory
from services.secrets.env_secret_resolver import EnvSecretResolver
from services.secrets.provider_credentials import ProviderCredentialResolver

executor = ProviderAdapterExecutor(
    credential_resolver=ProviderCredentialResolver(secret_resolver=EnvSecretResolver()),
    provider_factory=ProviderAdapterFactory(),   # default: mock + openai + azure_openai + gemini + deepseek
)
result = executor.execute(request)
```

---

## Part 7 — Answer Generation Adapter + Orchestrated Graph Wiring

`AnswerGenerationAdapter` in `app/services/doubt_solver/answer_generation_adapter.py`
bridges the orchestrated LangGraph nodes and the `LlmOrchestrator`.  It translates
graph-state fields (subject, intent, difficulty, context_text) into a `RouteRequest`
and calls `orchestrator.generate()`.

The orchestrated 3-node graph (`build_orchestrated_doubt_solver_graph`) in
`app/graphs/doubt_solver_graph.py` uses this adapter in its `generate_answer` node.

---

## Part 8.2.1 — Production Mock-Mode Safety Guard

Guards against `MockModelExecutor` silently serving fake answers in production.

### Fail-fast rule

If **all three** conditions are true at startup:

| Condition | Value |
|---|---|
| `APP_ENV` | `production` |
| `ENABLE_ORCHESTRATED_DOUBT_SOLVER` | `true` |
| `ENABLE_REAL_LLM` | `false` |

→ `ConfigurationError` is raised **at module-import time** (before any request
is processed).  The process exits non-zero.  Operators must fix the config.

### How to fix in production

Set `ENABLE_REAL_LLM=true`.  The orchestrated path then builds the real
`RegistryBackedModelExecutor → ProviderAdapterFactory` chain.

### Escape hatch (internal testing only)

`ENABLE_ORCHESTRATED_MOCK_LLM=true` overrides the guard — the mock executor
is used even in `APP_ENV=production`.

**This must never be set in normal production deployments.**
It exists only for controlled internal canary/smoke testing.

### Allowed combinations

| `APP_ENV` | `ENABLE_ORCHESTRATED_DOUBT_SOLVER` | `ENABLE_REAL_LLM` | `ENABLE_ORCHESTRATED_MOCK_LLM` | Result |
|---|---|---|---|---|
| `production` | `true` | `false` | `false` (default) | ❌ `ConfigurationError` |
| `production` | `true` | `false` | `true` | ✅ Mock (explicit override) |
| `production` | `true` | `true` | any | ✅ Real provider chain |
| `local`/`dev`/`test` | `true` | `false` | any | ✅ Mock (non-production) |
| any | `false` | any | any | ✅ Legacy graph (guard not reached) |

### New config symbols

| Symbol | Type | Default | Description |
|---|---|---|---|
| `ConfigurationError` | `Exception` subclass in `config.py` | — | Raised at startup for unsafe config |
| `ENABLE_ORCHESTRATED_MOCK_LLM` | `bool` | `false` | Escape hatch (internal testing only) |
| `Settings.enable_orchestrated_mock_llm` | `bool` | `False` | Python-side representation |

### Tests

`app/tests/test_production_mock_guard.py` — 16 tests:
- Guard fires (4 tests): production + orchestrated + mock → non-zero exit, ConfigurationError, actionable message
- Guard passes (5 tests): production + real LLM, local, dev, test, escape hatch
- Legacy unaffected (2 tests): orchestrated=false, default config
- Config schema (5 tests): ConfigurationError importable, Settings field present, defaults False, static position guards

---

## Part 8.1 — Orchestrated Entrypoint Verified

`ENABLE_ORCHESTRATED_DOUBT_SOLVER=true` is verified to route through the
orchestrated 3-node graph at runtime.

### Key change — MockModelExecutor branch in `main.py`

When `ENABLE_ORCHESTRATED_DOUBT_SOLVER=true`, `main.py` now branches on
`ENABLE_REAL_LLM`:

| `ENABLE_REAL_LLM` | Executor used |
|---|---|
| `false` (default) | `MockModelExecutor` — no network/AWS calls |
| `true` | `RegistryBackedModelExecutor → ProviderAdapterFactory` — real provider chain |

This ensures tests and local development with `ENABLE_REAL_LLM=false` never
trigger real API calls, even when the orchestrated graph is active.

### Subprocess-based verification

`app/tests/test_main_orchestrated_entrypoint.py` contains **16 subprocess tests**
that verify the routing boundary.  Subprocess isolation is required because
`main.py` builds graphs at module-import time; the env var must be set BEFORE
the import.

Key assertions:
- `ENABLE_ORCHESTRATED_DOUBT_SOLVER=true` → answer contains `[orchestrated-mock]`
- `ENABLE_ORCHESTRATED_DOUBT_SOLVER=false` → legacy response has `needs_review`
- No sensitive fields (`prompt`, `messages`, `context_text`, `api_key`, …) in response
- `returncode == 0`, `success == True`
- `importlib.reload()` is not used anywhere in `main.py`
- `config.py` uses `load_dotenv(override=False)` — real env vars always win over `.env.local`

### Non-goals

- `[NOT VERIFIED]` AgentCore HTTP runtime end-to-end (POST /invocations) is not tested.
- No changes to `OrchestratedDoubtSolverState` fields.

---

## Part 8.2 — Orchestrated Student-Friendly Streaming (current)

Streaming for the orchestrated doubt solver emits student-friendly status labels
and real answer chunks through the same Azure-first provider chain as non-streaming
`generate()`.

### Event types (`DoubtSolverStreamEvent`)

| type | When | Fields |
|---|---|---|
| `status` | Progress labels | `stage`, `label` |
| `chunk` | Answer text delta | `content` |
| `complete` | Success terminus | `stage=complete`, `label=Done`, safe `metadata` |
| `error` | Safe failure | `stage=error`, student-facing `label` |

Labels are **deterministic UX text** from `get_stream_label()` — not chain-of-thought,
routing, model selection, or provider execution details.

### Flow (`stream_doubt_solver`)

1. `status` understanding → classify (same path as graph)
2. `status` thinking → collect context (same path as graph)
3. `status` generating (intent-specific label) → `AnswerGenerationAdapter.generate_stream()`
4. `chunk` events (provider text deltas)
5. `status` finalizing → `complete`

Graph state is **not** expanded. `stream=false` uses the existing invoke graph path.

### Provider streaming

- `LlmOrchestrator.generate_stream()` → `RegistryBackedModelExecutor.execute_stream()`
  → `ProviderAdapterExecutor.execute_stream()` → adapter `generate_stream()`
- Azure OpenAI v1: OpenAI SDK `stream=True`, deployment passed as `model=` (no
  deployment URL path appended)
- Mock: deterministic chunks for unit tests (no network)
- Native OpenAI fallback: same streaming pattern; buffered `generate()` if adapter
  lacks `generate_stream`

### AgentCore runtime

**VERIFIED** — `main.invoke()` returns a sync generator when `stream=true`;
`BedrockAgentCoreApp` serializes each yielded dict as SSE (`text/event-stream`).

`[NOT VERIFIED]` Live HTTP E2E with `agentcore dev` + `stream=true` in this session.

### Tests

`app/tests/test_orchestrated_streaming.py` — schema, labels, full flow, mock streaming,
Azure v1 fake-client streaming, error handling, non-stream regression.

### Security

- No prompt, messages, context_text, credentials, or raw provider response in events.
- Provider stream failures → safe `error` event at service level.
- Do not log chunk content at INFO.

---

## Part 8.2 (legacy) — Generator Streaming Foundation (superseded)

The previous word-split simulated streaming (`start` events, `generate_answer_stream()`)
has been superseded by the student-friendly streaming implementation above.

---

### Part 9.5 — Difficulty Classification and Difficulty-Based Routing (current)

**Status:** Complete — `make check` 1372 passed, `agentcore validate` ✓

#### Root cause fixed

Previously, `QueryClassification` had no `difficulty` field, so `_map_to_orchestrated_classification()` hardcoded `difficulty="default"` regardless of the query. Every advanced query routed to `math.generator.default` (800 tokens), causing truncation on advanced practice responses.

#### Difficulty values

| Value | When used |
|---|---|
| `advanced` | Query contains "advanced", "hard", "tough", "tricky", "high level", "ssc cgl level", "cat level", "upsc level" |
| `basic` | Query contains "basic", "simple", "beginner", "easy" |
| `intermediate` | Query contains "intermediate", "moderate" |
| `default` | No difficulty signal detected |

#### Classifier confidence fallback (Part 13.1)

| Route | Model alias | When used |
|---|---|---|
| `general.classifier.default` | `doubt_solver_classifier` | Primary classifier (always first) |
| When primary confidence < **0.92** (configurable via `DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD`) | `doubt_solver_classifier_strong` | One retry max |

Task role `classifier_strong` is a system task role — not a generator intent overlay.

#### Classification flow

```
student query
  → primary LLM classifier (doubt_solver_classifier)
  → if confidence < threshold (default 0.92): strong classifier (doubt_solver_classifier_strong)
  → _map_to_orchestrated_classification()
      passes raw.difficulty through to DoubtSolverClassification.difficulty
  → collect_context via ContextRetrievalService
  → generator
```

#### Route lookup order

```
1. (subject, "generator", difficulty)      → route_source="exact"
2. (subject, "generator", "default")       → route_source="subject_default"
3. ("general", "generator", "default")     → route_source="general_default"
```

Advanced queries with difficulty=`"advanced"` hit the `advanced` sub-route if it exists for the subject.

#### Separation of concerns

| Dimension | Controls |
|---|---|
| `difficulty` | Route selection → which model alias and max_tokens |
| `intent` | Prompt overlays → which `intents/*.md` overlay is appended |
| `task_role` | Always `"generator"` — unchanged |

Intent and difficulty are independent. An advanced practice query uses the advanced route AND the practice intent overlay.

#### Token budget update

`math.generator.advanced` max_tokens increased from 1000 to **1200** to prevent truncation on advanced practice responses (5 multi-step questions). Other routes unchanged.

---

### Part 9.4 — Intent-Aware Generator Prompt Overlays (current)

Routes can declare per-intent prompt overlays under the `intent_overlays` key.
When `PromptResolver` resolves a generation request, it appends the intent-specific
overlays **after** any route-level overlays, with deduplication.

#### Allowed intent keys

Only these four normalized intent keys are accepted in `intent_overlays`:

| Key | Student intent | Overlay file (production) |
|---|---|---|
| `solve` | Solve a problem / calculation | `intents/solve.md` |
| `explain` | Explain a concept or option | `intents/explain.md` |
| `practice` | Generate practice questions | `intents/practice.md` |
| `visualize` | Visual/diagram-style explanation | `intents/visualize.md` |

The raw `QueryClassification.intent` values (`solve_question`, `explain_concept`, etc.)
are normalized to these four keys by `_ORCHESTRATED_INTENT_MAP` in
`graphs/doubt_solver_graph.py` before being passed to the route resolver.

#### Prompt composition order

```
1. Base prompt   (route.prompt)
2. Route overlays (route.overlays, if any)
3. Intent overlays (intent_overlays[intent], if configured for this route + intent)
```

Deduplication: if the same path appears in both route overlays and intent overlays
it is only included once.

#### YAML schema

```yaml
math:
  generator:
    default:
      model: math_basic_generator
      prompt: subjects/math_generator.md
      temperature: 0.2
      max_tokens: 800
      intent_overlays:
        solve:
          - intents/solve.md
        explain:
          - intents/explain.md
        practice:
          - intents/practice.md
        visualize:
          - intents/visualize.md
      fallback:
        - general_default
        - safe_mock
```

Routes that use `inherits: default` inherit the parent's `intent_overlays` by default.
A child route may override individual intent keys; unspecified keys fall through from parent.

#### `visualize` intent note

The `visualize` overlay (`intents/visualize.md`) instructs the generator to use
text, Markdown, and Mermaid diagrams only. The generator must never claim to produce
an image or external graphic.

#### Non-goals (deferred)

- `output_mode` field (e.g. `"diagram"`, `"table"`) — deferred.
- Real provider streaming for visual responses — deferred.
- `task_role` is NOT affected by intent — it remains `"generator"` for all intents.

