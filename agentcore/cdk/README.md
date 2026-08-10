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

`PRACTICE_GENERATION_ENABLED` defaults to `false`. Set it at synthesis/deployment time only
after `APPSYNC_GRAPHQL_ENDPOINT` and the existing practice SSM contract are available. The CDK
forwards only the non-secret runtime settings consumed by the application; provider credentials
remain outside synthesized templates and keep the application's existing environment names. This
repository does not currently have an established safe deployment channel that supplies those
secret values to AgentCore Runtime, so enabled real-model deployment remains blocked until that
external configuration gap is resolved without placing secrets in CloudFormation artifacts.

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
