import * as cdk from 'aws-cdk-lib';
import { Match, Template } from 'aws-cdk-lib/assertions';
import { AgentCoreProjectSpecSchema } from '@aws/agentcore-cdk';
import { AgentCoreStack } from '../lib/cdk-stack';
import { prepareRuntimeSources } from '../lib/runtime-source';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

test('AgentCoreStack synthesizes with empty spec', () => {
  const app = new cdk.App();
  const stack = new AgentCoreStack(app, 'TestStack', {
    deploymentEnvironment: 'test',
    spec: {
      name: 'testproject',
      version: 1,
      managedBy: 'CDK' as const,
      runtimes: [],
      memories: [],
      credentials: [],
      evaluators: [],
      onlineEvalConfigs: [],
      policyEngines: [],
      agentCoreGateways: [],
      mcpRuntimeTools: [],
      unassignedTargets: [],
    },
  });
  const template = Template.fromStack(stack);
  template.hasOutput('StackNameOutput', {
    Description: 'Name of the CloudFormation Stack',
  });
});

test('runtime source staging excludes local secrets and development artifacts', () => {
  const repositoryRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'agentcore-source-'));
  const configRoot = path.join(repositoryRoot, 'agentcore');
  const appRoot = path.join(repositoryRoot, 'app');
  fs.mkdirSync(path.join(appRoot, 'tests'), { recursive: true });
  fs.mkdirSync(configRoot, { recursive: true });
  fs.writeFileSync(path.join(appRoot, 'pyproject.toml'), '[project]\nname = "test"\nversion = "1"\n');
  fs.writeFileSync(path.join(appRoot, 'main.py'), 'print("safe")\n');
  fs.writeFileSync(path.join(appRoot, '.env.local'), 'SECRET=unsafe\n');
  fs.writeFileSync(path.join(appRoot, '.env.production'), 'SECRET=unsafe\n');
  fs.writeFileSync(path.join(appRoot, 'runtime.log'), 'private\n');
  fs.writeFileSync(path.join(appRoot, 'tests', 'test_main.py'), 'assert True\n');
  const spec = AgentCoreProjectSpecSchema.parse({
    name: 'testproject',
    version: 1,
    managedBy: 'CDK',
    runtimes: [
      {
        name: 'runtime',
        build: 'CodeZip',
        entrypoint: 'main.py',
        codeLocation: 'app/',
        runtimeVersion: 'PYTHON_3_14',
      },
    ],
    memories: [],
    credentials: [],
    evaluators: [],
    onlineEvalConfigs: [],
    policyEngines: [],
    agentCoreGateways: [],
    mcpRuntimeTools: [],
    unassignedTargets: [],
  });

  try {
    const deploymentSpec = prepareRuntimeSources(spec, configRoot);
    const stagedRoot = path.resolve(repositoryRoot, deploymentSpec.runtimes[0].codeLocation);
    expect(fs.existsSync(path.join(stagedRoot, 'main.py'))).toBe(true);
    expect(fs.existsSync(path.join(stagedRoot, 'pyproject.toml'))).toBe(true);
    expect(fs.existsSync(path.join(stagedRoot, '.env.local'))).toBe(false);
    expect(fs.existsSync(path.join(stagedRoot, '.env.production'))).toBe(false);
    expect(fs.existsSync(path.join(stagedRoot, 'runtime.log'))).toBe(false);
    expect(fs.existsSync(path.join(stagedRoot, 'tests'))).toBe(false);
  } finally {
    fs.rmSync(repositoryRoot, { recursive: true, force: true });
  }
});

test('runtime conversation policies are scoped to configured tables and SSM paths', () => {
  const app = new cdk.App();
  const spec = AgentCoreProjectSpecSchema.parse({
    name: 'testproject',
    version: 1,
    managedBy: 'CDK',
    runtimes: [
      {
        name: 'runtime',
        build: 'CodeZip',
        entrypoint: 'main.py',
        codeLocation: 'app/',
        runtimeVersion: 'PYTHON_3_14',
      },
    ],
    memories: [
      {
        name: 'meritranker_short_term_memory',
        eventExpiryDuration: 30,
        strategies: [],
      },
    ],
    credentials: [],
    evaluators: [],
    onlineEvalConfigs: [],
    policyEngines: [],
    agentCoreGateways: [],
    mcpRuntimeTools: [],
    unassignedTargets: [],
  });
  const stack = new AgentCoreStack(app, 'ConversationStack', {
    spec,
    deploymentEnvironment: 'dev',
  });
  const template = Template.fromStack(stack);
  const rendered = JSON.stringify(template.toJSON());
  template.hasResourceProperties('AWS::BedrockAgentCore::Memory', {
    Name: 'meritranker_short_term_memory_dev',
    EventExpiryDuration: 30,
  });
  const memoryResources = template.findResources('AWS::BedrockAgentCore::Memory');
  expect(Object.values(memoryResources)).toHaveLength(1);
  expect(Object.values(memoryResources)[0].Properties).not.toHaveProperty('MemoryStrategies');
  const memoryLogicalId = Object.keys(memoryResources)[0];
  template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
    EnvironmentVariables: {
      MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID: {
        'Fn::GetAtt': [memoryLogicalId, 'MemoryId'],
      },
    },
  });
  template.hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: Match.arrayWith([
        Match.objectLike({
          Action: ['bedrock-agentcore:CreateEvent', 'bedrock-agentcore:ListEvents'],
          Effect: 'Allow',
          Resource: {
            'Fn::GetAtt': [memoryLogicalId, 'MemoryArn'],
          },
        }),
      ]),
    },
  });
  expect(rendered).toContain('dynamodb:GetItem');
  expect(rendered).toContain('dynamodb:PutItem');
  expect(rendered).toContain('dynamodb:Query');
  expect(rendered).toContain('dynamodb:UpdateItem');
  expect(rendered).toContain('bedrock-agentcore:CreateEvent');
  expect(rendered).toContain('bedrock-agentcore:ListEvents');
  expect(rendered).not.toContain('bedrock-agentcore:DeleteEvent');
  expect(rendered).not.toContain('bedrock-agentcore:ListMemoryRecords');
  expect(rendered).toContain('conversation-history');
  expect(rendered).toContain('conversation-session');
  expect(rendered).toContain('ConversationHistoryByConversation');
  expect(rendered).not.toContain('/index/*');
  expect(rendered).not.toContain('dynamodb:*');
  expect(rendered).not.toContain('ssm:*');
});
