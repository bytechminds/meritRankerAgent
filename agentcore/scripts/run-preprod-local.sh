#!/usr/bin/env bash
# PreProd-local: exact Prod behaviour (agentcore/prod-target.json, APP_ENV=production)
# and Prod provider credentials, served by the normal local runtime against Dev data.
#
#   run-preprod-local.sh           checks, summary, then serve on http://127.0.0.1:${PREPROD_PORT:-8080}
#   run-preprod-local.sh --check   checks and summary only
#
# Profiles (operator tooling only): MERITRANKER_PROD_PROFILE (default meritranker-prod) reads
# only the provider secret; MERITRANKER_DEV_PROFILE (default dev) runs the local Agent.
# Credentials are held in memory and passed only to the child process environment.
set -euo pipefail
CDK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../cdk" && pwd)"
(cd "$CDK_DIR" && npm run --silent build)
exec node "$CDK_DIR/dist/bin/preprod-local.js" "$@"
