# ============================================================================
# meritRankerTutor — project Makefile
#
# All Python commands run inside app/ using uv so they use the correct venv.
# AgentCore commands (dev, validate) run from the project root.
#
# Usage:
#   make env                  — verify tool versions
#   make test                 — run pytest
#   make lint                 — ruff check (no auto-fix)
#   make format               — ruff format
#   make fix                  — ruff check --fix + ruff format
#   make check                — lint + test (CI gate)
#   make dev                  — start local AgentCore server with live logs
#   make validate             — validate agentcore.json schema
#   make smoke-doubt-solver   — curl local runtime (requires make dev running)
#   make clean                — remove all cache / venv dirs
# ============================================================================

AGENT_DIR := app

# Local AgentCore runtime endpoint — adjust if agentcore dev logs a different port.
AGENTCORE_LOCAL_URL ?= http://localhost:8080/invocations

.PHONY: dev validate test lint format fix check env clean \
        smoke-doubt-solver smoke-doubt-solver-real-llm smoke-doubt-solver-with-retrieval \
        smoke-doubt-solver-combined \
        smoke-llm-orchestration-mock smoke-llm-orchestration-real

# Start the local AgentCore runtime with structured log output.
dev:
	agentcore dev --logs

# Validate agentcore.json against the AgentCore schema (no AWS calls).
validate:
	agentcore validate

# Run all tests via pytest inside the app venv.
test:
	cd $(AGENT_DIR) && uv run pytest

# Lint — report issues without fixing them.  Use in CI.
lint:
	cd $(AGENT_DIR) && uv run ruff check .

# Format source files in-place.
format:
	cd $(AGENT_DIR) && uv run ruff format .

# Auto-fix fixable lint issues then format.
fix:
	cd $(AGENT_DIR) && uv run ruff check . --fix
	cd $(AGENT_DIR) && uv run ruff format .

# Full quality gate — lint then test.  Run before opening a PR.
check:
	cd $(AGENT_DIR) && uv run ruff check .
	cd $(AGENT_DIR) && uv run pytest

# Print tool versions to confirm the environment is healthy.
env:
	cd $(AGENT_DIR) && uv run python --version
	cd $(AGENT_DIR) && uv --version
	agentcore --version

# ---------------------------------------------------------------------------
# smoke-doubt-solver
#
# Manual end-to-end smoke test against the running local AgentCore runtime.
# Requires:  make dev   (in a separate terminal)
#
# [NOT VERIFIED] AgentCore HTTP runtime — run manually and verify output.
#
# Override endpoint:  AGENTCORE_LOCAL_URL=http://localhost:8080/invocations make smoke-doubt-solver
# ---------------------------------------------------------------------------
smoke-doubt-solver:
	@echo ">>> Smoke test: POST mode=doubt_solver to $(AGENTCORE_LOCAL_URL)"
	@echo ">>> Requires: make dev running in a separate terminal"
	@curl -s -X POST $(AGENTCORE_LOCAL_URL) \
	  -H "Content-Type: application/json" \
	  -d '{"mode":"doubt_solver","query":"A shopkeeper marks goods 40% above cost price and gives a 20% discount. Find the profit or loss percentage. Show step-by-step working.","user_id":"local-smoke","conversation_id":"local-smoke-conversation","turn_id":"local-smoke-turn-1","language":"en"}' \
	  | python3 -m json.tool || echo ">>> [FAILED] Is make dev running? Check the port in agentcore dev --logs output."

# ---------------------------------------------------------------------------
# smoke-doubt-solver-real-llm
#
# Smoke test using a real LLM provider.
# Requires the following env vars to be set before running:
#   ENABLE_REAL_LLM=true
#   LLM_ROLE_CONFIG_JSON=<json config for classifier and generator roles>
#   AZURE_OPENAI_ENDPOINT=<your endpoint>          (if using azure_openai provider)
#   AZURE_OPENAI_API_KEY=<your key>                (if using azure_openai provider)
#   AZURE_OPENAI_API_VERSION=<api version>         (if using azure_openai provider)
#   OPENAI_API_KEY=<your key>                      (if using openai provider)
#
# Do NOT hardcode any secrets. Load them via your shell environment or app/.env.local.
#
# [NOT VERIFIED] Real LLM response quality — verify manually.
# ---------------------------------------------------------------------------
smoke-doubt-solver-real-llm:
	@echo ">>> Smoke test (REAL LLM): POST mode=doubt_solver to $(AGENTCORE_LOCAL_URL)"
	@echo ">>> Requires: make dev running AND real LLM env vars set"
	@echo ">>> Required env vars: ENABLE_REAL_LLM, LLM_ROLE_CONFIG_JSON,"
	@echo ">>>   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION"
	@echo ">>>   (or OPENAI_API_KEY if using openai provider)"
	@curl -s -X POST $(AGENTCORE_LOCAL_URL) \
	  -H "Content-Type: application/json" \
	  -d '{"mode":"doubt_solver","query":"A train travels 240 km at a uniform speed. If the speed had been 8 km/h more, it would have taken 1 hour less. Find the speed of the train.","user_id":"local-smoke-llm","conversation_id":"local-smoke-llm-conversation","turn_id":"local-smoke-llm-turn-1","language":"en"}' \
	  | python3 -m json.tool || echo ">>> [FAILED] Is make dev running with ENABLE_REAL_LLM=true?"

# ---------------------------------------------------------------------------
# smoke-doubt-solver-with-retrieval
#
# Smoke test using real KB and/or DynamoDB retrieval.
# Requires the following env vars to be set before running:
#   ENABLE_KB_RETRIEVAL=true
#   BEDROCK_KB_ID=<your Bedrock Knowledge Base ID>
#   BEDROCK_KB_REGION=<AWS region>            (optional, defaults to AWS_REGION)
#   ENABLE_DYNAMODB_FETCH=true                (optional — for DynamoDB record fetch)
#   DYNAMODB_QUESTION_TABLE=<table name>      (required if ENABLE_DYNAMODB_FETCH=true)
#   DYNAMODB_PATTERN_TABLE=<table name>       (required if ENABLE_DYNAMODB_FETCH=true)
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (or IAM role)
#
# Do NOT hardcode any secrets. Load them via your shell environment or app/.env.local.
#
# [NOT VERIFIED] Real KB/DynamoDB retrieval quality — verify manually.
# ---------------------------------------------------------------------------
smoke-doubt-solver-with-retrieval:
	@echo ">>> Smoke test (KB + DynamoDB): POST mode=doubt_solver to $(AGENTCORE_LOCAL_URL)"
	@echo ">>> Requires: make dev running AND AWS retrieval env vars set"
	@echo ">>> Required env vars: ENABLE_KB_RETRIEVAL, BEDROCK_KB_ID"
	@echo ">>>   Optional: ENABLE_DYNAMODB_FETCH, DYNAMODB_QUESTION_TABLE"
	@curl -s -X POST $(AGENTCORE_LOCAL_URL) \
	  -H "Content-Type: application/json" \
	  -d '{"mode":"doubt_solver","query":"Explain the concept of percentage and how it relates to ratios.","user_id":"local-smoke-retrieval","conversation_id":"local-smoke-retrieval-conversation","turn_id":"local-smoke-retrieval-turn-1","language":"en"}' \
	  | python3 -m json.tool || echo ">>> [FAILED] Is make dev running with ENABLE_KB_RETRIEVAL=true?"

# ---------------------------------------------------------------------------
# smoke-doubt-solver-combined
#
# Full pipeline smoke test: real LLM + KB retrieval + DynamoDB fetch.
# All of the following env vars must be set before running.
#
# LLM (choose one provider):
#   ENABLE_REAL_LLM=true
#   LLM_ROLE_CONFIG_JSON=<json config for classifier and generator roles>
#   AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_API_VERSION
#   -or-  OPENAI_API_KEY
#
# AWS retrieval:
#   ENABLE_KB_RETRIEVAL=true
#   BEDROCK_KB_ID=<your KB ID>
#   ENABLE_DYNAMODB_FETCH=true
#   DYNAMODB_QUESTION_TABLE=<table name>
#   AWS_REGION + AWS credentials (env vars or ~/.aws/credentials or IAM role)
#
# Do NOT hardcode any secrets.
#
# [NOT VERIFIED] Combined real LLM + KB + DynamoDB path — verify manually.
# ---------------------------------------------------------------------------
smoke-doubt-solver-combined:
	@echo ">>> Smoke test (REAL LLM + KB + DynamoDB): POST mode=doubt_solver to $(AGENTCORE_LOCAL_URL)"
	@echo ">>> Requires: make dev running AND all LLM + AWS env vars set"
	@echo ">>> Required env vars: ENABLE_REAL_LLM, LLM_ROLE_CONFIG_JSON,"
	@echo ">>>   AZURE_OPENAI_ENDPOINT/API_KEY/API_VERSION (or OPENAI_API_KEY),"
	@echo ">>>   ENABLE_KB_RETRIEVAL, BEDROCK_KB_ID,"
	@echo ">>>   ENABLE_DYNAMODB_FETCH, DYNAMODB_QUESTION_TABLE, AWS_REGION"
	@curl -s -X POST $(AGENTCORE_LOCAL_URL) \
	  -H "Content-Type: application/json" \
	  -d '{"mode":"doubt_solver","query":"A student scored 72 out of 90 in mathematics. What is the percentage score and how does it compare to a passing score of 75%?","user_id":"local-smoke-combined","conversation_id":"local-smoke-combined-conversation","turn_id":"local-smoke-combined-turn-1","language":"en"}' \
	  | python3 -m json.tool || echo ">>> [FAILED] Is make dev running with all required env vars set?"

# ---------------------------------------------------------------------------
# smoke-llm-orchestration-mock
#
# Manual dry-run smoke test — exercises the full orchestration stack with
# MockProviderAdapter.  No credentials, no network, no real provider calls.
# Safe to run at any time without any env vars set.
#
# This target is NOT part of make check and is never run by pytest.
# ---------------------------------------------------------------------------
smoke-llm-orchestration-mock:
	@echo ">>> Running LLM orchestration dry-run (mock provider, no credentials)"
	cd $(AGENT_DIR) && uv run python -c "\
from services.llm_orchestration.dry_run import run_mock_orchestration_dry_run, LlmDryRunInput; \
r = run_mock_orchestration_dry_run(LlmDryRunInput(query='What is 10 percent of 200?')); \
print('content:     ', r.content); \
print('model:       ', r.model); \
print('provider:    ', r.provider); \
print('answer_src:  ', r.answer_source); \
print('route_id:    ', r.route_id); \
print('fallback:    ', r.fallback_used); \
"

# ---------------------------------------------------------------------------
# smoke-llm-orchestration-real
#
# Manual smoke test — exercises a REAL provider adapter with live credentials.
# Opt-in only: RUN_REAL_LLM_SMOKE=true must be set explicitly.
# NEVER run by make check or normal pytest collection.
#
# Usage:
#   RUN_REAL_LLM_SMOKE=true OPENAI_API_KEY=sk-... PROVIDER=openai \
#     make smoke-llm-orchestration-real
#
#   RUN_REAL_LLM_SMOKE=true \
#     AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_VERSION=... \
#     PROVIDER=azure_openai DEPLOYMENT=my-deployment \
#     make smoke-llm-orchestration-real
#
# Variables (must be set in shell):
#   RUN_REAL_LLM_SMOKE  — must be "true" to enable
#   PROVIDER            — openai | azure_openai
#   QUERY               — (optional) question to ask the model
#   DEPLOYMENT          — (required for azure_openai) Azure deployment name
#
# [NOT VERIFIED] Real provider response quality — verify manually.
# ---------------------------------------------------------------------------
PROVIDER    ?= openai
QUERY       ?= Explain percentage increase in one sentence.
DEPLOYMENT  ?=

smoke-llm-orchestration-real:
	@echo ">>> Running LLM orchestration real provider smoke (opt-in only)"
	@echo ">>> Requires: RUN_REAL_LLM_SMOKE=true and provider credentials in env"
	@echo ">>> Provider: $(PROVIDER)  Deployment: $(DEPLOYMENT)"
	cd $(AGENT_DIR) && RUN_REAL_LLM_SMOKE=$(RUN_REAL_LLM_SMOKE) \
	  uv run python scripts/smoke_llm_orchestration.py \
	  --provider $(PROVIDER) \
	  --query "$(QUERY)" \
	  $(if $(DEPLOYMENT),--deployment $(DEPLOYMENT),)

# Remove all generated / cache directories.
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".venv" -prune -exec rm -rf {} +
