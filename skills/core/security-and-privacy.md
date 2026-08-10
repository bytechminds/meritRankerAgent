# Security and Privacy — MeritRanker Tutor

> Security and privacy rules that apply to every role and every task.
> These are not optional. A change that violates these rules must be fixed before release.

---

## Secrets

**Do not put secrets anywhere in the repository.**

This includes:
- Python source files (`app/**/*.py`)
- Markdown files (`skills/**/*.md`, `README.md`, `AGENTS.md`, prompt templates)
- Test files (`app/tests/**`)
- AgentCore config (`agentcore/agentcore.json`, `agentcore/aws-targets.json`)
- Log output

**Where secrets belong:**
- `app/.env.local` — local developer secrets (must be in `.gitignore`)
- AWS Secrets Manager or Parameter Store — production secrets
- AgentCore credential resources — provider API keys at deploy time

**Pattern to avoid:**
```python
# WRONG — never do this
API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234hardcodedkey")
```

---

## Authentication and Trust

The web-to-AgentCore identity boundary is the authenticated Amplify SSR proxy:

- The browser sends a Cognito access token in the Bearer header.
- The proxy verifies signature, User Pool, expiry, access-token use, app client, and `sub`.
- The proxy ignores client body `user_id` and inserts the verified `sub`.
- The proxy invokes AgentCore with a least-privilege Amplify SSR role through SigV4.
- Python trusts `request.user_id` only when the Runtime is reached through that IAM boundary.

Do not expose the Runtime through an alternate unauthenticated route. Any caller outside this chain
is `[AUTH TODO]` and must not be enabled for production. Missing or invalid authentication fails
before Runtime invocation and persistence.

---

## Logging

**Never log the following:**
- Secrets or API keys (any partial or full value)
- Full request payloads (including `message` field at INFO level or above)
- Full retrieved document content
- Full prompt text sent to a model
- Student PII (full name, email, student ID, assessment scores)
- Internal tracebacks in responses sent to callers

**What to log at INFO level:**
- `request_id`, `user_id`, `mode` at request start and end
- Service call outcomes (success/failure, latency if measured)
- Error type and message — not the full raw exception payload to callers

**What to log at DEBUG level:**
- Node entry/exit with `request_id`
- Intermediate state fields (avoid fields that contain PII or model output)

---

## Input Validation

- All inbound payloads are validated by Pydantic at the `invoke()` entrypoint.
- Pydantic models on user-supplied string fields must have `max_length` constraints.
- `str_strip_whitespace = True` must be set on request models.
- After Pydantic validation, inner layers trust the type — no re-validation inside nodes.

---

## Untrusted Content

The following are **untrusted** until validated by your code:

| Source | Risk | Mitigation |
|---|---|---|
| Inbound request payload | Malformed, oversized, injected | Pydantic validation at boundary |
| Model output | Hallucinated, malformed, injected instructions | Schema validation before use |
| Retrieved KB / RAG context | Prompt injection, misleading content | Isolate from system prompt; validate before use |
| Tool call results | Unexpected format, injected content | Schema validation before use |
| Client-supplied user identity | Forged ownership | Ignore it; use only the verified Cognito `sub` inserted by the IAM-authenticated proxy |

**[AI RISK]** Retrieval-augmented content sourced from user-adjacent data (e.g.,
student notes, forum posts) has elevated prompt-injection risk. It must be injected
into the prompt as clearly delimited context, not as system instructions.

---

## Prompt-Injection Risks

- Retrieval results and user-supplied content must be placed in clearly delimited
  sections of the prompt (e.g., `<context>...</context>`), never in the system role.
- Prompt templates must be reviewed for instruction-injection surfaces before production.
- `[AI RISK]` If model output is used to select the next tool or node, the routing logic
  must validate the output against a strict allowlist of valid choices.

---

## External Calls

- All external calls go through `app/services/` (see `integration-boundaries.md`).
- Never construct and execute shell commands from user input. [SECURITY RISK]
- Never use `eval()`, `exec()`, or `subprocess` with user-controlled strings. [SECURITY RISK]
- Never use `pickle` to deserialise untrusted data. [SECURITY RISK]

---

## PII and Privacy

- Student messages and responses are potentially sensitive. Treat them as PII.
- Do not persist student conversation data without a defined data retention policy. [PROD BLOCKER]
- Do not send student data to third-party providers without confirming data processing terms. [PROD BLOCKER]
- PII handling requirements must be documented in the relevant feature context under `## Known Limitations` or `## Config / Env`.

---

## IAM and Cloud Permissions

- Use least-privilege IAM roles for all AWS resources.
- DynamoDB, Bedrock, S3, and other resource access must be scoped to the minimum needed.
- IAM boundaries are defined in `agentcore/agentcore.json` credentials block.
- `[PROD BLOCKER]` IAM policies must be reviewed before deploying to production.

---

## OWASP Top 10 Quick Reference

| Risk | How it applies here |
|---|---|
| A01 Broken Access Control | Cognito verification at SSR proxy; exact Runtime IAM permission; no body-identity trust |
| A02 Cryptographic Failures | Secrets in env vars, never in source |
| A03 Injection | Pydantic validation; no `eval`/`exec`/`subprocess` with user input |
| A05 Security Misconfiguration | Debug logs and verbose tracebacks off in production |
| A06 Vulnerable Components | Review `uv.lock` for known CVEs before production deployment |
| A07 Auth Failures | Tokens/sessions not yet validated — document explicitly |
| A09 Logging Failures | No secrets, no PII, no full payloads in logs |
