import * as cdk from 'aws-cdk-lib';
import { Template } from 'aws-cdk-lib/assertions';
import { AgentCoreProjectSpecSchema, type AwsDeploymentTarget } from '@aws/agentcore-cdk';
import * as fs from 'fs';
import * as path from 'path';
import { AgentCoreStack } from '../lib/cdk-stack';
import {
  PROD_QUALIFIER,
  REQUIRED_PROD_RUNTIME_ENV,
  selectDeploymentTargets,
  validateProdTargetConfig,
  withProdEndpoint,
} from '../lib/prod-target';

const AGENTCORE_DIR = path.resolve(__dirname, '..', '..');
const DEV: AwsDeploymentTarget = { name: 'Dev', account: '661012794182', region: 'ap-south-1' };
const PROD: AwsDeploymentTarget = { name: 'Prod', account: '123456789012', region: 'ap-south-1' };
const PROD_APPSYNC = 'https://prodapi0123456789.appsync-api.ap-south-1.amazonaws.com/graphql';

function committedProdConfig(): Record<string, any> {
  return JSON.parse(fs.readFileSync(path.join(AGENTCORE_DIR, 'prod-target.json'), 'utf8'));
}

/** The committed file with only its two documented placeholders filled in. */
function filledProdConfig(overrides: Record<string, unknown> = {}): Record<string, any> {
  const config = committedProdConfig();
  config.runtimeEnvironment.APPSYNC_GRAPHQL_ENDPOINT = PROD_APPSYNC;
  return { ...config, ...overrides };
}

function spec() {
  return AgentCoreProjectSpecSchema.parse({
    name: 'meritRankerTutor',
    version: 1,
    managedBy: 'CDK',
    runtimes: [
      { name: 'runtime', build: 'CodeZip', entrypoint: 'main.py', codeLocation: 'app/', runtimeVersion: 'PYTHON_3_14' },
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
}

function prodStack(promotedVersion: number | null) {
  const config = validateProdTargetConfig(PROD, filledProdConfig({ promotedVersion }));
  const stack = new AgentCoreStack(new cdk.App(), 'AgentCore-meritRankerTutor-Prod', {
    spec: withProdEndpoint(spec(), config.promotedVersion),
    deploymentEnvironment: 'Prod',
    runtimeEnvironment: config.runtimeEnvironment,
    productionRuntime: {
      runtimeName: config.runtimeName,
      providerSecretName: config.providerSecretName,
      ...(config.promotedVersion !== null ? { qualifier: PROD_QUALIFIER } : {}),
    },
    env: { account: PROD.account, region: PROD.region },
  });
  return Template.fromStack(stack);
}

function policyStatements(template: Template): any[] {
  return Object.values(template.findResources('AWS::IAM::Policy')).flatMap(
    (policy: any) => policy.Properties.PolicyDocument.Statement
  );
}

describe('target selection', () => {
  test('unset selector synthesizes only Dev, exactly as before Prod existed', () => {
    expect(selectDeploymentTargets([DEV, PROD], undefined)).toEqual([DEV]);
    expect(selectDeploymentTargets([DEV, PROD], '  ')).toEqual([DEV]);
  });

  test('Prod is synthesized only when explicitly selected, and alone', () => {
    expect(selectDeploymentTargets([DEV, PROD], 'Prod')).toEqual([PROD]);
    expect(selectDeploymentTargets([DEV, PROD], 'Dev')).toEqual([DEV]);
    expect(() => selectDeploymentTargets([DEV, PROD], 'prod')).toThrow(/exactly one target/);
  });
});

describe('Prod configuration contract', () => {
  test('committed placeholder account blocks deployment', () => {
    const placeholder = { ...PROD, account: '000000000000' };
    expect(() => validateProdTargetConfig(placeholder, filledProdConfig())).toThrow(/production account/);
  });

  test('committed placeholder AppSync endpoint blocks deployment', () => {
    expect(() => validateProdTargetConfig(PROD, committedProdConfig())).toThrow(/REPLACE_ME/);
  });

  test('filled committed config resolves the deterministic Prod environment', () => {
    const config = validateProdTargetConfig(PROD, filledProdConfig());
    expect(config.runtimeName).toBe('meritRankerTutor_runtime_prod');
    expect(config.providerSecretName).toBe('meritranker/agent-runtime/prod/providers');
    expect(config.runtimeEnvironment).toMatchObject({
      APP_ENV: 'production',
      ENABLE_REAL_LLM: 'true',
      ENABLE_ORCHESTRATED_DOUBT_SOLVER: 'true',
      PRACTICE_GENERATION_ENABLED: 'true',
      STUDENT_CREDIT_ENFORCEMENT_ENABLED: 'true',
      STUDENT_CREDIT_DRY_RUN: 'true',
      AZURE_OPENAI_DEPLOYMENT_GPT_4_1: 'gpt-4.1',
      AZURE_OPENAI_DEPLOYMENT_GPT_4_1_MINI: 'gpt-4.1-mini',
      AZURE_OPENAI_DEPLOYMENT_GPT_5_4_MINI: 'gpt-5.4-mini',
      AZURE_OPENAI_DEPLOYMENT_GPT_6_SOL: 'gpt-6-sol',
      AWS_REGION: 'ap-south-1',
      PROVIDER_CREDENTIALS_SECRET_ID: 'meritranker/agent-runtime/prod/providers',
    });
    for (const name of REQUIRED_PROD_RUNTIME_ENV) expect(config.runtimeEnvironment[name]).toBeTruthy();
  });

  test.each(REQUIRED_PROD_RUNTIME_ENV)('missing %s blocks deployment', name => {
    const config = filledProdConfig();
    delete config.runtimeEnvironment[name];
    expect(() => validateProdTargetConfig(PROD, config)).toThrow(new RegExp(name));
  });

  test('non-production APP_ENV blocks deployment', () => {
    const config = filledProdConfig();
    config.runtimeEnvironment.APP_ENV = 'local';
    expect(() => validateProdTargetConfig(PROD, config)).toThrow(/APP_ENV/);
  });

  test('missing provider secret reference blocks deployment', () => {
    const config = filledProdConfig();
    delete config.providerSecretName;
    expect(() => validateProdTargetConfig(PROD, config)).toThrow(/providerSecretName/);
  });

  test.each(['AZURE_OPENAI_API_KEY', 'TAVILY_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'AZURE_OPENAI_ENDPOINT'])(
    'credential %s is rejected from runtime environment',
    name => {
      const config = filledProdConfig();
      config.runtimeEnvironment[name] = 'not-a-real-value';
      expect(() => validateProdTargetConfig(PROD, config)).toThrow(/credential/);
    }
  );
});

describe('Prod stack', () => {
  test('first deploy: no endpoint yet, runtime identity published, secret scoped', () => {
    const template = prodStack(null);
    template.resourceCountIs('AWS::BedrockAgentCore::RuntimeEndpoint', 0);
    template.hasResourceProperties('AWS::BedrockAgentCore::Runtime', {
      AgentRuntimeName: 'meritRankerTutor_runtime_prod',
    });
    template.hasResourceProperties('AWS::SSM::Parameter', { Name: '/meritranker/agent-runtime/v1/runtime/arn' });
    template.hasResourceProperties('AWS::SSM::Parameter', {
      Name: '/meritranker/agent-runtime/v1/runtime/region',
      Value: 'ap-south-1',
    });
    const names = Object.values(template.findResources('AWS::SSM::Parameter')).map((p: any) => p.Properties.Name);
    expect(names).not.toContain('/meritranker/agent-runtime/v1/runtime/qualifier');

    const secretStatements = policyStatements(template).filter(statement =>
      JSON.stringify(statement.Action).includes('secretsmanager')
    );
    expect(secretStatements).toEqual([
      {
        Action: 'secretsmanager:GetSecretValue',
        Effect: 'Allow',
        Resource: {
          'Fn::Join': [
            '',
            [
              'arn:',
              { Ref: 'AWS::Partition' },
              ':secretsmanager:ap-south-1:123456789012:secret:meritranker/agent-runtime/prod/providers-??????',
            ],
          ],
        },
      },
    ]);
  });

  test('promoted version creates the stable prod endpoint and publishes the qualifier', () => {
    const template = prodStack(7);
    template.hasResourceProperties('AWS::BedrockAgentCore::RuntimeEndpoint', {
      Name: 'prod',
      AgentRuntimeVersion: '7',
    });
    template.hasResourceProperties('AWS::SSM::Parameter', {
      Name: '/meritranker/agent-runtime/v1/runtime/qualifier',
      Value: 'prod',
    });
  });

  test('runtime environment carries the secret reference but no credential values', () => {
    const template = prodStack(null);
    const runtime: any = Object.values(template.findResources('AWS::BedrockAgentCore::Runtime'))[0];
    const environment = runtime.Properties.EnvironmentVariables;
    expect(environment.APP_ENV).toBe('production');
    expect(environment.PROVIDER_CREDENTIALS_SECRET_ID).toBe('meritranker/agent-runtime/prod/providers');
    for (const key of committedProdConfig().requiredSecretKeys) expect(environment).not.toHaveProperty(key);
    expect(JSON.stringify(template.toJSON())).not.toMatch(/AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY/);
  });

  test('qualified QuestionBank promotion keeps its existing scoped write grant', () => {
    const actions = policyStatements(prodStack(null)).map(statement => JSON.stringify(statement.Action));
    expect(actions).toContain(JSON.stringify(['dynamodb:PutItem', 'dynamodb:UpdateItem']));
  });
});

test('Dev stack gains no Prod-only resources or grants', () => {
  const template = Template.fromStack(
    new AgentCoreStack(new cdk.App(), 'AgentCore-meritRankerTutor-Dev', {
      spec: spec(),
      deploymentEnvironment: 'Dev',
      runtimeEnvironment: { AWS_REGION: 'ap-south-1', PRACTICE_GENERATION_ENABLED: 'false' },
    })
  );
  const rendered = JSON.stringify(template.toJSON());
  expect(rendered).not.toContain('secretsmanager');
  expect(rendered).not.toContain('/meritranker/agent-runtime/v1/runtime/');
  expect(rendered).not.toContain('AgentRuntimeName":"meritRankerTutor_runtime_prod');
  template.resourceCountIs('AWS::BedrockAgentCore::RuntimeEndpoint', 0);
});
