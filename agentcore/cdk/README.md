# AgentCore CDK Project

This CDK project is managed by the AgentCore CLI. It deploys your agent infrastructure into AWS using the `@aws/agentcore-cdk` L3 constructs.

## Structure

- `bin/cdk.ts` — Entry point. Reads project configuration from `agentcore/` and creates a stack per deployment target.
- `lib/cdk-stack.ts` — Defines `AgentCoreStack`, which wraps the `AgentCoreApplication` L3 construct.
- `test/cdk.test.ts` — Unit tests for stack synthesis.

## Practice generation

PracticeGenerationGraph runs as AgentCore-native tracked background execution inside the existing
runtime. The role reads only the exact practice SSM contract, directly creates/reads the initial
assessment, directly accesses Question and QuestionBank records through the named indexes, and
uses the separately deployed exact AppSync mutation-field permission for all ongoing assessment
progress. Runtime environment configuration contains only non-secret endpoint/feature values.
There is no second runtime, queue consumer, or practice-specific deployment path.

For the Dev target, `PRACTICE_GENERATION_ENABLED` defaults to `false`; set it at
synthesis/deployment time only after `APPSYNC_GRAPHQL_ENDPOINT` and the existing practice SSM
contract are available. The CDK forwards only non-secret runtime settings.

## Targets

One synthesis produces one target stack, chosen by `AGENTCORE_DEPLOY_TARGET`. Unset means every
non-Prod target (Dev), so Dev deploys are unchanged. Prod is synthesized only when selected.

| | Dev | Prod |
|---|---|---|
| Account/region | `aws-targets.json` | `aws-targets.json` (placeholder `000000000000` blocks deploy) |
| Runtime flags | operator shell (legacy) | committed `agentcore/prod-target.json` |
| Provider credentials | — | Secrets Manager `meritranker/agent-runtime/prod/providers`, read once at startup via `PROVIDER_CREDENTIALS_SECRET_ID` |
| Release pointer | `DEFAULT` endpoint | named `prod` endpoint pinned to `promotedVersion` |

The Prod secret is a JSON object keyed by the existing ENV names listed in
`requiredSecretKeys`. The execution role may call `secretsmanager:GetSecretValue` on that one
secret only (encrypt it with the default `aws/secretsmanager` key; a customer-managed key would
also need `kms:Decrypt`). The value is read once per runtime session, so a rotated key takes
effect on the next deployed version. Prod publishes its identity for the separate backend's server-side proxy:
`/meritranker/agent-runtime/v1/runtime/{arn,region,qualifier}` (`qualifier` exists once the
`prod` endpoint does). The backend's invoke policy should allow `bedrock-agentcore:InvokeAgentRuntime`
on the runtime ARN and `<runtime ARN>/runtime-endpoint/prod`.

## Modes

| Mode | Command | Behaviour config | Provider credentials | AWS application data |
|---|---|---|---|---|
| Dev | `agentcore dev` | `app/.env.local` | `app/.env.local` | Dev |
| PreProd-local | `agentcore/scripts/run-preprod-local.sh` | `prod-target.json` (`APP_ENV=production`) | Prod secret, in memory only | Dev |
| Prod | `agentcore/scripts/deploy-prod.sh --deploy` | `prod-target.json` | Prod secret via execution role | Prod |

PreProd-local reads the Prod provider secret with `MERITRANKER_PROD_PROFILE` (default
`meritranker-prod`, which must resolve to the Prod account) and runs the same local server
`agentcore dev` starts (`uvicorn main:app` from `app/.venv`) as `MERITRANKER_DEV_PROFILE`
(default `dev`, which must not be the Prod account). It serves `app/` filtered exactly as a Prod
deploy packages it (no `.env` files), staged in its own `.cache/preprod-runtime` so it never
touches deploy staging, with only the Prod-managed flags, the
secret's credentials and Dev resource identities in its environment (plus `PATH`/`HOME`-style
basics), so neither `app/.env.local` nor the shell changes behaviour: unmanaged settings stay at
code defaults exactly as in the deployed runtime. AppSync, region and short-term memory stay on
Dev, and tables resolve from Dev SSM. PreProd refuses a secret holding any Prod-managed,
resource, `AWS_*` or `MEMORY_*` key. Secret values
never touch disk, argv or output. Use `--check` to validate and print the summary without
starting; `PREPROD_PORT` overrides 8080.

PreProd-local proves the exact Prod feature configuration, model/deployment selection,
production credential values, configuration parsing, provider initialisation and startup, and
real Tutor/Practice behaviour when exercised manually. It does not prove Prod execution-role
IAM, backend-to-AgentCore IAM, or Prod DynamoDB/S3 access.

## Prod release and rollback

```bash
agentcore/scripts/deploy-prod.sh            # preflight only
agentcore/scripts/deploy-prod.sh --diff     # preflight + CDK diff
agentcore/scripts/deploy-prod.sh --deploy   # preflight + deploy
```

The preflight stops unless the caller account and region equal the Prod target, the committed
Prod configuration validates, the provider secret exists with its required key names, and the Prod
stack synthesizes.

1. Deploy code/config changes. AgentCore creates a new immutable runtime version; `prod` still
   points to `promotedVersion`, so students are unaffected.
2. Confirm `runtime_ready` for the new version and smoke-test it through `DEFAULT`.
3. Promote: set `promotedVersion` in `prod-target.json` to the new version, commit, and deploy.
   Only the endpoint changes; no new runtime version is created.
4. Roll back: set `promotedVersion` to the previous known-good version, commit, and deploy.
   Backend and clients need no change because they use the `prod` qualifier.

On the very first Prod deploy `promotedVersion` is `null`, so no `prod` endpoint exists yet;
set it to `1` after step 2 to create the endpoint.

## Useful commands

- `npm run build` compile TypeScript to JavaScript
- `npm run test` run unit tests
- `npx cdk synth` emit the synthesized CloudFormation template
- `npx cdk deploy` deploy this stack to your default AWS account/region
- `npx cdk diff` compare deployed stack with current state

## Usage

You typically don't need to interact with this directory directly. The AgentCore CLI handles synthesis and deployment:

```bash
agentcore deploy    # synthesizes and deploys via CDK
agentcore status    # checks deployment status
```
