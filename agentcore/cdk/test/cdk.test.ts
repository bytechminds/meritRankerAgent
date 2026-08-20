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
    runtimeEnvironment: {
      AWS_REGION: 'ap-south-1',
      APPSYNC_GRAPHQL_ENDPOINT: 'https://e7rfdkgqczag5bz34ov67locl4.appsync-api.ap-south-1.amazonaws.com/graphql',
      PRACTICE_GENERATION_ENABLED: 'false',
      ENABLE_ORCHESTRATED_DOUBT_SOLVER: 'true',
      ENABLE_REAL_LLM: 'true',
      PRACTICE_RESOURCE_PARAMETER_ROOT: '/meritranker/agent-runtime/v1/practice',
    },
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
    EnvironmentVariables: Match.objectLike({
      MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID: {
        'Fn::GetAtt': [memoryLogicalId, 'MemoryId'],
      },
      AWS_REGION: 'ap-south-1',
      APPSYNC_GRAPHQL_ENDPOINT: 'https://e7rfdkgqczag5bz34ov67locl4.appsync-api.ap-south-1.amazonaws.com/graphql',
      PRACTICE_GENERATION_ENABLED: 'false',
      ENABLE_ORCHESTRATED_DOUBT_SOLVER: 'true',
      ENABLE_REAL_LLM: 'true',
      PRACTICE_RESOURCE_PARAMETER_ROOT: '/meritranker/agent-runtime/v1/practice',
    }),
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
  expect(rendered).toContain('dynamodb:Scan');
  expect(rendered).toContain('bedrock-agentcore:CreateEvent');
  expect(rendered).toContain('bedrock-agentcore:ListEvents');
  expect(rendered).not.toContain('bedrock-agentcore:DeleteEvent');
  expect(rendered).not.toContain('bedrock-agentcore:ListMemoryRecords');
  expect(rendered).toContain('conversation-history');
  expect(rendered).toContain('conversation-session');
  expect(rendered).toContain('exam-profile');
  expect(rendered).toContain('ConversationHistoryByConversation');
  expect(rendered).not.toContain('/index/*');
  expect(rendered).not.toContain('dynamodb:*');
  expect(rendered).not.toContain('ssm:*');
  template.resourceCountIs('AWS::Lambda::Function', 0);
  template.resourceCountIs('AWS::Lambda::EventSourceMapping', 0);
  template.resourceCountIs('AWS::Logs::LogGroup', 0);
  expect(rendered).not.toContain('PracticeGenerationWorker');
  expect(rendered).not.toContain('PRACTICE_WORKER_TIMEOUT_SECONDS');
  expect(rendered).not.toContain('PRACTICE_PROVIDER_SECRET_ARN');
  expect(rendered).not.toContain('secretsmanager:GetSecretValue');
  expect(rendered).toContain('dynamodb:BatchGetItem');
  expect(rendered).not.toContain('appsync:*');
  expect(rendered).toContain('dynamodb:TransactWriteItems');
  expect(rendered).toContain('dynamodb:DescribeTable');
  expect(rendered).toContain('dynamodb:DeleteItem');
  expect(rendered).not.toContain('sqs:ReceiveMessage');
  expect(rendered).not.toContain('sqs:DeleteMessage');
  expect(rendered).not.toContain('sqs:ChangeMessageVisibility');
  expect(rendered).not.toContain('PracticeGenerationQueue');
  expect(rendered).not.toContain('PracticeGenerationDLQ');
  expect(rendered).toContain('/practice/mock-test-quiz/table-arn');
  expect(rendered).toContain('/practice/question/table-arn');
  expect(rendered).toContain('/practice/question-bank/table-arn');
  expect(rendered).toContain('/practice/question/test-id-index-name');
  expect(rendered).toContain('/practice/question-bank/category-index-name');
  expect(rendered).toContain('/practice/question-bank/reuse-index-name');
  expect(rendered).not.toContain('/pattern-intelligence/');
  expect(rendered).not.toContain('/practice/question-bank/pattern-index-name');
  expect(rendered).not.toContain('/practice/attempt/');
  expect(rendered).not.toContain('s3vectors:QueryVectors');
  expect(rendered).not.toContain('sqs:SendMessage');
});

test('enabled practice grants only its existing AppSync progress mutation', () => {
  const app = new cdk.App();
  const stack = new AgentCoreStack(app, 'PracticeProgressStack', {
    spec: AgentCoreProjectSpecSchema.parse({
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
    }),
    deploymentEnvironment: 'dev',
    runtimeEnvironment: {
      AWS_REGION: 'ap-south-1',
      APPSYNC_GRAPHQL_ENDPOINT: 'https://t37helcceraznaejlcm27uhmwi.appsync-api.ap-south-1.amazonaws.com/graphql',
      PRACTICE_GENERATION_ENABLED: 'true',
    },
  });

  const rendered = JSON.stringify(Template.fromStack(stack).toJSON());

  expect(rendered).toContain('appsync:GraphQL');
  expect(rendered).toContain('apis/t37helcceraznaejlcm27uhmwi/types/Mutation/fields/updatePracticeGenerationProgress');
  expect(rendered).toContain('dynamodb:UpdateItem');
  expect(rendered).not.toContain('appsync:*');
  expect(rendered).not.toContain('/types/Mutation/fields/*');
});

function practiceStack(
  runtimeEnvironment: Record<string, string>,
  memoryCount = 0
): cdk.Stack {
  return new AgentCoreStack(new cdk.App(), 'QuestionBankPromotionStack', {
    spec: AgentCoreProjectSpecSchema.parse({
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
      memories: Array.from({ length: memoryCount }, (_unused, index) => ({
        name: `meritranker_memory_${index}`,
        eventExpiryDuration: 30,
        strategies: [],
      })),
      credentials: [],
      evaluators: [],
      onlineEvalConfigs: [],
      policyEngines: [],
      agentCoreGateways: [],
      mcpRuntimeTools: [],
      unassignedTargets: [],
    }),
    deploymentEnvironment: 'dev',
    runtimeEnvironment: {
      AWS_REGION: 'ap-south-1',
      APPSYNC_GRAPHQL_ENDPOINT:
        'https://t37helcceraznaejlcm27uhmwi.appsync-api.ap-south-1.amazonaws.com/graphql',
      PRACTICE_GENERATION_ENABLED: 'true',
      ...runtimeEnvironment,
    },
  });
}

function questionBankPutStatements(stack: cdk.Stack): unknown[] {
  const template = Template.fromStack(stack).toJSON();
  const statements: unknown[] = [];
  for (const resource of Object.values(template.Resources ?? {}) as any[]) {
    if (resource.Type !== 'AWS::IAM::Policy') continue;
    for (const statement of resource.Properties?.PolicyDocument?.Statement ?? []) {
      const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
      const mutates = actions.some(
        (action: string) => action === 'dynamodb:PutItem' || action === 'dynamodb:UpdateItem'
      );
      if (!mutates) continue;
      // CloudFormation strips hyphens from the SSM parameter logical id.
      if (JSON.stringify(statement.Resource ?? '').includes('questionbanktablearn')) {
        statements.push(statement);
      }
    }
  }
  return statements;
}

test('QuestionBank write is granted exactly once when promotion is enabled', () => {
  const statements = questionBankPutStatements(
    practiceStack({ PRACTICE_QUESTION_BANK_PROMOTION_ENABLED: 'true' })
  );

  expect(statements).toHaveLength(1);
  const [statement] = statements as any[];
  // Least privilege: create the row, plus Pattern enrichment. Nothing else.
  expect(statement.Action).toEqual(['dynamodb:PutItem', 'dynamodb:UpdateItem']);
  const rendered = JSON.stringify(statement);
  for (const forbidden of [
    'dynamodb:DeleteItem',
    'dynamodb:Scan',
    'dynamodb:BatchWriteItem',
    'dynamodb:TransactWriteItems',
    'dynamodb:*',
  ]) {
    expect(rendered).not.toContain(forbidden);
  }
});

test('QuestionBank write is absent while promotion is disabled', () => {
  expect(questionBankPutStatements(practiceStack({}))).toHaveLength(0);
});

test('Pattern reuse alone no longer grants QuestionBank write', () => {
  const statements = questionBankPutStatements(
    practiceStack({
      PATTERN_INTELLIGENCE_ENABLED: 'true',
      PATTERN_INTELLIGENCE_REUSE_ENABLED: 'true',
    })
  );

  expect(statements).toHaveLength(0);
});

test.each(['true', 'TRUE', 'True'])(
  'promotion value %s grants QuestionBank write exactly as the Python runtime reads it',
  (value) => {
    // config.py accepts any casing via .lower(); the grant must not disagree, or the
    // runtime would attempt PutItem without permission.
    expect(
      questionBankPutStatements(
        practiceStack({ PRACTICE_QUESTION_BANK_PROMOTION_ENABLED: value })
      )
    ).toHaveLength(1);
  }
);

test.each(['false', 'FALSE', '0', 'yes', ''])(
  'promotion value %s grants no QuestionBank write',
  (value) => {
    expect(
      questionBankPutStatements(
        practiceStack({ PRACTICE_QUESTION_BANK_PROMOTION_ENABLED: value })
      )
    ).toHaveLength(0);
  }
);

test.each([0, 1, 2])(
  'QuestionBank write grant is independent of memory count (%i memories)',
  (memoryCount) => {
    // The grant previously sat inside the per-memory loop, so its presence and
    // count tracked how many memories the spec declared.
    expect(
      questionBankPutStatements(
        practiceStack({ PRACTICE_QUESTION_BANK_PROMOTION_ENABLED: 'true' }, memoryCount)
      )
    ).toHaveLength(1);
    expect(questionBankPutStatements(practiceStack({}, memoryCount))).toHaveLength(0);
  }
);

test('no broad QuestionBank mutation action is ever granted', () => {
  const template = Template.fromStack(
    practiceStack({ PRACTICE_QUESTION_BANK_PROMOTION_ENABLED: 'true' })
  ).toJSON();
  const forbidden = [
    'dynamodb:DeleteItem',
    'dynamodb:Scan',
    'dynamodb:BatchWriteItem',
    'dynamodb:TransactWriteItems',
    'dynamodb:*',
  ];
  for (const resource of Object.values(template.Resources ?? {}) as any[]) {
    if (resource.Type !== 'AWS::IAM::Policy') continue;
    for (const statement of resource.Properties?.PolicyDocument?.Statement ?? []) {
      const rendered = JSON.stringify(statement);
      if (!rendered.includes('questionbanktablearn')) continue;
      for (const action of forbidden) {
        expect(rendered).not.toContain(action);
      }
    }
  }
});

function semanticStack(mode: string): cdk.Stack {
  return practiceStack({ PRACTICE_QUESTION_SEMANTIC_REUSE_MODE: mode });
}

function questionVectorStatements(stack: cdk.Stack): any[] {
  const template = Template.fromStack(stack).toJSON();
  const found: any[] = [];
  for (const resource of Object.values(template.Resources ?? {}) as any[]) {
    if (resource.Type !== 'AWS::IAM::Policy') continue;
    for (const statement of resource.Properties?.PolicyDocument?.Statement ?? []) {
      const actions = Array.isArray(statement.Action) ? statement.Action : [statement.Action];
      if (actions.some((a: string) => String(a).startsWith('s3vectors:'))) found.push(statement);
    }
  }
  return found;
}

test.each(['shadow', 'on'])(
  'semantic mode %s grants filtered questions-v1 discovery only',
  (mode) => {
    const statements = questionVectorStatements(semanticStack(mode));
    expect(statements).toHaveLength(1);
    // GetVectors is the documented dependent permission for filtered queries with
    // returnMetadata; nothing else may appear.
    expect(statements[0].Action).toEqual(['s3vectors:QueryVectors', 's3vectors:GetVectors']);
    const rendered = JSON.stringify(Template.fromStack(semanticStack(mode)).toJSON());
    expect(rendered).toContain('S3_VECTOR_QUESTION_INDEX_ARN');
    expect(rendered).toContain('question-bank/vector/index-arn');
    for (const forbidden of [
      's3vectors:PutVectors',
      's3vectors:DeleteVectors',
      's3vectors:ListVectors',
      's3vectors:*',
      'pattern-intelligence/vector/index-arn',
    ]) {
      expect(rendered).not.toContain(forbidden);
    }
  }
);

test('semantic mode off grants no vector access and injects no ARN', () => {
  expect(questionVectorStatements(semanticStack('off'))).toHaveLength(0);
  const rendered = JSON.stringify(Template.fromStack(semanticStack('off')).toJSON());
  expect(rendered).not.toContain('S3_VECTOR_QUESTION_INDEX_ARN');
});
