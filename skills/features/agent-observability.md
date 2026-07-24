# Agent Observability and Clean Logging

## Status

Implemented locally. AgentCore runtime export, CloudWatch log delivery, custom span visibility, and
production volume/cost remain **[NOT VERIFIED]** until deployment.

## Architecture

All new logging and tracing code is isolated under `app/observability/`. The historical
`app/logging_config.py` remains a compatibility facade, so existing imports are unchanged.
Application paths call the shared `log_event()`, request-summary, context, and `stage_span()`
functions; they do not construct log handlers or provider-specific telemetry.

`ContextVar` carries request, trace, conversation, turn, and request-type identity through async
code. Streaming and persistence thread boundaries explicitly use `copy_context()`. Every scope
resets its token in `finally`, preventing sequential or concurrent request leakage.

`RequestExecutionSummary` is frozen. Updates replace the current value and accept only declared,
bounded fields. One `request_execution_summary` event is emitted per completed non-streaming
request or consumed stream. It contains decisions and status only, never question, prompt, answer,
retrieval text, recent conversation, image bytes, credentials, or provider payloads.

## Modes

| Environment | Console | Local file |
|---|---|---|
| local/development | concise Rich output | one rotating human-readable request log |
| production | one JSON object per stdout line | forcibly disabled |

Environment variables:

- `AGENT_LOG_LEVEL`
- `AGENT_LOG_FORMAT=pretty|json|pretty_and_json_file`
- `AGENT_LOG_FILE_ENABLED`
- `AGENT_DETAILED_LOGS`
- `AGENT_OBSERVABILITY_ENABLED`

The local file path is fixed at `.logs/agent-runtime.log` and is not environment-configurable. A
request-local collector stores at most 64
approved operational events and writes the complete request block once at terminal state while
holding the rotating handler lock. The file rotates at 20 MiB with three standard backups.
Concurrent requests cannot interleave their lines. File setup or write failures emit one bounded
warning and never fail a student request. Production does not depend on container-local files.

## Event Contract

Every structured event has `timestamp`, `level`, `service`, `environment`, `event`, `component`,
`request_id`, `trace_id`, `conversation_id`, `turn_id`, `stage`, `status`, `duration_ms`,
`error_code`, and bounded `details`.

The catalog covers request lifecycle, classification and fallback, follow-up context and
resolution, retrieval and fallback, generation, quality validation/repair, and conversation
persistence completion/skip/failure. Unknown event names are rejected.

Detail keys containing prompt, query, answer, content, message, image, payload, raw, secret, token,
credential, authorization, API key, or conversation context are removed before serialization.
String metadata is single-line and capped.

The readable local block includes the full bounded trace ID for CloudWatch/OTEL correlation.
Failures include only actionable safe evidence: stage, bounded error code, exception class,
model/route or retrieval source when already present, and typed persistence outcomes. Exception
messages, tracebacks, filenames, line numbers, request content, and provider payloads are excluded.
Generation events are sourced from the completed generator execution result. The readable model
label is the actual model alias, while safe runtime diagnostics retain route ID, task role,
provider, deployment, and fallback status. Classifier execution and generic path labels cannot
overwrite generator provenance.

Example diagnostic evidence:

```text
Trace: 1234567890abcdef1234567890abcdef
18:47:12  FAIL  Answer generation failed: PROVIDER_TIMEOUT, exception=TimeoutError, stage=generate
18:47:12  FAIL  Request failed: PROVIDER_FAILED, exception=ProviderExecutionError
```

## Local Inspection

Run from `app/`:

```bash
uv run python tools/inspect_agent_logs.py --latest
uv run python tools/inspect_agent_logs.py --request-id <request-id>
uv run python tools/inspect_agent_logs.py --conversation-id <conversation-id>
uv run python tools/inspect_agent_logs.py --failures
uv run python tools/inspect_agent_logs.py --persistence-failures
uv run python tools/inspect_agent_logs.py --follow-up-failures
uv run python tools/inspect_agent_logs.py --json --request-id <request-id>
```

Opening `.logs/agent-runtime.log` is sufficient for normal debugging. The optional inspector reads
the same request blocks and creates no latest file, daily file, request directory, separate request
file, or export file.

## Exact Local Log Example

```text
REQUEST 7f3a120b | 2026-07-23 18:44:40 | standalone | completed
Conversation: a12c90ef   Turn: 91bd8820   Duration: 12.00s
Trace: 0123456789abcdef0123456789abcdef
--------------------------------------------------------------------------------
18:44:40  OK    Request received
18:44:41  OK    Classified: math, solve, medium
18:44:51  OK    Answer generated: o4-mini
18:44:51  OK    Quality passed
18:44:52  OK    History saved
18:44:52  OK    Session updated
18:44:52  OK    Memory saved
--------------------------------------------------------------------------------
RESULT: Completed successfully.
================================================================================
REQUEST b9180e42 | 2026-07-23 18:45:10 | follow_up | completed
Conversation: a12c90ef   Turn: b72fe901   Duration: 9.40s
Trace: 123456789abcdef0123456789abcdef0
--------------------------------------------------------------------------------
18:45:10  OK    Request received
18:45:11  OK    Follow-up detected
18:45:11  WARN  Memory unavailable
18:45:11  OK    DynamoDB context loaded: 2 turns
18:45:12  OK    Follow-up resolved
18:45:18  OK    Answer generated: o4-mini
18:45:18  OK    Quality passed
18:45:19  OK    History saved
18:45:19  OK    Session updated
18:45:19  WARN  Memory write unavailable
--------------------------------------------------------------------------------
RESULT: Completed successfully.
================================================================================
REQUEST c024fe11 | 2026-07-23 18:46:00 | standalone | failed-quality
Conversation: e3301ac4   Turn: 2906ab73   Duration: 6.20s
Trace: 23456789abcdef0123456789abcdef01
--------------------------------------------------------------------------------
18:46:00  OK    Request received
18:46:01  OK    Classified: reasoning, solve, hard
18:46:05  OK    Answer generated: o4-mini
18:46:06  FAIL  Quality failed: QUALITY_GATE_FAILED, stage=validate_quality, repair_attempted=False
18:46:06  SKIP  History persistence skipped
18:46:06  SKIP  Session persistence skipped
18:46:06  SKIP  Memory persistence skipped
--------------------------------------------------------------------------------
RESULT: Quality validation failed; answer was not persisted.
================================================================================
```

## CloudWatch Logs Insights

Use the existing AgentCore Runtime application log group. Do not create an unrelated group.

Request timeline:

```text
fields @timestamp, event, stage, status, duration_ms, error_code, component
| filter request_id = "<request-id>"
| sort @timestamp asc
| limit 200
```

Recent failures:

```text
fields @timestamp, event, request_id, conversation_id, stage, error_code, details
| filter status = "failed"
| sort @timestamp desc
| limit 100
```

Persistence failures:

```text
fields @timestamp, request_id, conversation_id, turn_id, details
| filter event = "conversation_persistence_failed"
| sort @timestamp desc
| limit 100
```

Follow-up failures:

```text
fields @timestamp, event, request_id, conversation_id, stage, error_code, details
| filter event in ["follow_up_context_failed", "follow_up_resolution_failed"]
| sort @timestamp desc
| limit 100
```

Slow requests:

```text
fields @timestamp, request_id, conversation_id, duration_ms, details
| filter event = "request_execution_summary" and duration_ms >= 5000
| sort duration_ms desc
| limit 100
```

For high volume, an exact-match field index on `request_id` can reduce scanned data. Keep query
time windows narrow and set an explicit retention policy to control CloudWatch cost.

Failed request summaries:

```text
fields @timestamp, request_id, conversation_id, details.terminal_reason,
       details.quality_status, details.history_write_status,
       details.session_write_status, details.memory_write_status, duration_ms
| filter event = "request_execution_summary" and status != "completed"
| sort @timestamp desc
| limit 100
```

## OpenTelemetry and AgentCore

`agentcore/agentcore.json` explicitly enables the schema-supported
`instrumentation.enableOtel=true`. The application creates named request/classify/context/
follow-up/retrieve/generate/quality/persistence spans only when the OpenTelemetry API is available;
otherwise the context manager is a safe no-op.

The local environment does not currently contain `opentelemetry` or `aws-opentelemetry-distro`.
No ADOT version was guessed or added. AgentCore-hosted runtime instrumentation is the selected
deployment path. Custom span export remains **[NOT VERIFIED]** until:

1. CloudWatch Transaction Search is enabled in the target AWS account.
2. The runtime is deployed with instrumentation enabled.
3. Runtime application log delivery is active on the existing AgentCore log group.
4. A trace confirms the custom spans in CloudWatch Transaction Search.

## Required AWS Operator Actions

1. In the target account and Region, open **CloudWatch > Settings > Transaction Search** and
   complete the one-time enablement. This cannot be verified from local code.
2. Deploy the reviewed AgentCore runtime so `instrumentation.enableOtel=true` takes effect.
3. In **Bedrock AgentCore > Agent Runtime > runtime > Log delivery**, add or verify
   `APPLICATION_LOGS` delivery to the existing runtime CloudWatch log group and wait for
   **Delivery active**. Do not create a second application-owned runtime group.
4. If memory service traces are required, enable tracing on the existing AgentCore Memory resource
   from its **Tracing > Edit** control after Transaction Search is enabled.
5. Invoke one non-production standalone request and one follow-up request. Confirm JSON application
   events in the runtime log group and the expected custom spans under
   **CloudWatch > Transaction Search**.
6. Set reviewed retention, access control, and alarms for request failures, persistence partial
   failures, and latency. No retention or alarm was guessed in this change.

AWS references:

- [Configure AgentCore observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-configure.html)
- [Get started with AgentCore observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html)
- [CloudWatch Logs Insights filter syntax](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/CWL_QuerySyntax-Filter.html)

## Unchanged Boundaries

Public JSON/SSE schemas, graph topology, prompts, route selection, model/provider behavior,
retrieval, retry counts, fallback policies, quality decisions, persistence eligibility, and
frontend behavior are unchanged. No provider call, AWS call, extra model call, database write, or
new log group was added by observability.

## Local Diagnostic Content

`AGENT_LOCAL_LOG_CONTENT=off|preview` controls bounded content in the single local
`app/.logs/agent-runtime.log` file. Local defaults to `preview`; production forcibly uses `off`
regardless of the environment value. Preview limits are query 300 characters, resolved query 300,
previous turns 300 total, and response 600. Values are normalized to one line.

Previews are held only in the request-local readable-log buffer and never enter structured events
or CloudWatch. JWTs, authorization data, credentials, API keys, images/base64, provider payloads,
prompts, chain-of-thought, and retrieval documents remain excluded. Every terminal block includes a
`PERSISTENCE` section, including explicit `not_attempted` or `skipped` outcomes.
Secret-shaped values are redacted before preview truncation. The active log and rotated backups are
created with `0600` permissions.

Readable headers include the application version, current short commit when available, PID,
environment, Region, and independent history/session/Memory readiness. Runtime identity is computed
once per process; it causes no per-request Git, SSM, or provider calls.

## Validation

- Full test suite: 2,210 passed, 1 credential-gated test skipped, 2 pre-existing warnings.
- Readable-log and observability focused suites: 49 passed.
- Streaming, lifecycle, conversation, entrypoint, and observability integration suites: 267 passed.
- `agentcore validate`: valid.
- AgentCore CDK TypeScript build, format check, and Jest suite: passed.
- Atomic readable request-block emission, rotation, privacy, width, concurrency, and inspector
  tests: passed.
- Application import/startup smoke test: passed with external features disabled.
- Local `agentcore dev --logs` is verified on `127.0.0.1:8080`; runtime headers identify the
  active reload child, commit, environment, Region, and independent persistence readiness.
- Deployed CloudWatch delivery, Transaction Search, and custom span export remain
  **[NOT VERIFIED]** until runtime deployment and AWS console verification.
