#!/usr/bin/env bash
# Fail-closed Prod preflight for the MeritRanker AgentCore runtime.
#
#   deploy-prod.sh            preflight only
#   deploy-prod.sh --diff     preflight, then `agentcore deploy --target Prod --diff`
#   deploy-prod.sh --deploy   preflight, then `agentcore deploy --target Prod`
#
# Uses the operator's current AWS credentials (e.g. AWS_PROFILE); stores none.
# Never prints secret values.
set -euo pipefail

AGENTCORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDK_DIR="$AGENTCORE_DIR/cdk"
MODE="${1:-}"
case "$MODE" in
  ""|--diff|--deploy) ;;
  *) echo "usage: $0 [--diff|--deploy]" >&2; exit 2 ;;
esac

fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
pass() { echo "PREFLIGHT OK:   $*"; }

command -v aws >/dev/null || fail "aws CLI not found"
command -v node >/dev/null || fail "node not found"

# 1. Explicit target: read only the Prod entry.
read -r PROD_ACCOUNT PROD_REGION < <(node -e '
  const t = require(process.argv[1]).filter(x => x.name === "Prod");
  if (t.length !== 1) process.exit(3);
  console.log(t[0].account, t[0].region);
' "$AGENTCORE_DIR/aws-targets.json") || fail "aws-targets.json must define exactly one Prod target"
pass "target Prod ($PROD_ACCOUNT / $PROD_REGION)"

# 2. AWS account.
CALLER_ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
  || fail "sts:GetCallerIdentity failed; check AWS credentials"
[[ "$CALLER_ACCOUNT" == "$PROD_ACCOUNT" ]] \
  || fail "credentials are for account $CALLER_ACCOUNT, Prod is $PROD_ACCOUNT"
pass "account $CALLER_ACCOUNT"

# 3. AWS region.
EFFECTIVE_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region 2>/dev/null || true)}}"
[[ "$EFFECTIVE_REGION" == "$PROD_REGION" ]] \
  || fail "effective region '${EFFECTIVE_REGION:-unset}', Prod is $PROD_REGION"
pass "region $EFFECTIVE_REGION"

# 4. Critical production configuration (same validator the synth uses).
(cd "$CDK_DIR" && npm run --silent build) || fail "CDK build failed"
read -r SECRET_NAME REQUIRED_KEYS < <(node -e '
  const { loadProdTargetConfig } = require(process.argv[1]);
  const target = require(process.argv[2]).find(x => x.name === "Prod");
  const c = loadProdTargetConfig(process.argv[3], target);
  console.log(c.providerSecretName, c.requiredSecretKeys.join(","));
' "$CDK_DIR/dist/lib/prod-target.js" "$AGENTCORE_DIR/aws-targets.json" "$AGENTCORE_DIR") \
  || fail "prod-target.json failed validation (see error above)"
pass "production configuration"

# 5. Provider secret exists and carries the required key names (names only).
aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$PROD_REGION" \
  --query Name --output text >/dev/null 2>&1 || fail "secret $SECRET_NAME not found in $PROD_REGION"
MISSING_KEYS="$(aws secretsmanager get-secret-value --secret-id "$SECRET_NAME" --region "$PROD_REGION" \
  --query SecretString --output text 2>/dev/null | node -e '
    let raw = ""; process.stdin.on("data", d => (raw += d)).on("end", () => {
      let v; try { v = JSON.parse(raw); } catch { console.log("<secret is not a JSON object>"); return; }
      if (!v || typeof v !== "object" || Array.isArray(v)) { console.log("<secret is not a JSON object>"); return; }
      // Same shape the runtime loader accepts: ENV-style names mapped to strings.
      const malformed = Object.keys(v).filter(k => !/^[A-Z][A-Z0-9_]*$/.test(k) || typeof v[k] !== "string");
      if (malformed.length) { console.log(`<malformed entries: ${malformed.join(",")}>`); return; }
      const missing = process.argv[1].split(",").filter(k => typeof v[k] !== "string" || !v[k].trim());
      console.log(missing.join(","));
    });
  ' "$REQUIRED_KEYS")" || fail "could not read secret $SECRET_NAME"
[[ -z "$MISSING_KEYS" ]] || fail "secret $SECRET_NAME is missing: $MISSING_KEYS"
pass "secret $SECRET_NAME has required keys"

# 6. Synthesis of the Prod stack only.
export AGENTCORE_DEPLOY_TARGET=Prod
(cd "$CDK_DIR" && npx cdk synth --quiet >/dev/null) || fail "CDK synth failed"
pass "synth AgentCore-meritRankerTutor-Prod"

echo
node -e '
  const { loadProdTargetConfig } = require(process.argv[1]);
  const { formatConfigSummary } = require(process.argv[2]);
  const target = require(process.argv[3]).find(x => x.name === "Prod");
  const c = loadProdTargetConfig(process.argv[4], target);
  console.log(formatConfigSummary(c, [
    ["Target mode", "PROD"],
    ["AWS account / region", `${target.account} / ${target.region}`],
    ["Agent endpoint", "AgentCore qualifier prod"],
    ["Provider secret source", "Prod Secrets Manager (execution role)"],
    ["Required secret keys", "PRESENT"],
  ]));
' "$CDK_DIR/dist/lib/prod-target.js" "$CDK_DIR/dist/lib/preprod-local.js" "$AGENTCORE_DIR/aws-targets.json" "$AGENTCORE_DIR"
echo

# The agentcore CLI must run from the project root.
case "$MODE" in
  --diff) cd "$AGENTCORE_DIR/.." && exec agentcore deploy --target Prod --diff ;;
  --deploy) cd "$AGENTCORE_DIR/.." && exec agentcore deploy --target Prod ;;
  *) echo "Preflight passed. Re-run with --diff or --deploy." ;;
esac
