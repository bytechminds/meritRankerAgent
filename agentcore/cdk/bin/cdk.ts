#!/usr/bin/env node
import { AgentCoreStack } from '../lib/cdk-stack';
import { prepareRuntimeSources } from '../lib/runtime-source';
import { ConfigIO, type AwsDeploymentTarget } from '@aws/agentcore-cdk';
import { App, type Environment } from 'aws-cdk-lib';
import * as path from 'path';
import * as fs from 'fs';

function toEnvironment(target: AwsDeploymentTarget): Environment {
  return {
    account: target.account,
    region: target.region,
  };
}

function sanitize(name: string): string {
  return name.replace(/_/g, '-');
}

function toStackName(projectName: string, targetName: string): string {
  return `AgentCore-${sanitize(projectName)}-${sanitize(targetName)}`;
}

function booleanEnvironment(name: string, defaultValue: boolean): string {
  const raw = process.env[name];
  if (raw === undefined || raw === '') return String(defaultValue);
  const normalized = raw.trim().toLowerCase();
  if (normalized !== 'true' && normalized !== 'false') {
    throw new Error(`${name} must be true or false`);
  }
  return normalized;
}

function runtimeEnvironment(target: AwsDeploymentTarget): Record<string, string> {
  const practiceEnabled = booleanEnvironment('PRACTICE_GENERATION_ENABLED', false);
  const patternIntelligenceEnabled = booleanEnvironment(
    'PATTERN_INTELLIGENCE_ENABLED',
    false
  );
  const endpoint = process.env.APPSYNC_GRAPHQL_ENDPOINT?.trim();
  if (practiceEnabled === 'true' && !endpoint) {
    throw new Error('APPSYNC_GRAPHQL_ENDPOINT is required when practice generation is enabled');
  }
  return {
    AWS_REGION: target.region,
    ...(endpoint ? { APPSYNC_GRAPHQL_ENDPOINT: endpoint } : {}),
    PRACTICE_GENERATION_ENABLED: practiceEnabled,
    PATTERN_INTELLIGENCE_ENABLED: patternIntelligenceEnabled,
    PATTERN_INTELLIGENCE_REUSE_ENABLED: booleanEnvironment(
      'PATTERN_INTELLIGENCE_REUSE_ENABLED',
      false
    ),
    ENABLE_ORCHESTRATED_DOUBT_SOLVER: booleanEnvironment(
      'ENABLE_ORCHESTRATED_DOUBT_SOLVER',
      true
    ),
    ENABLE_REAL_LLM: booleanEnvironment('ENABLE_REAL_LLM', true),
    PRACTICE_RESOURCE_PARAMETER_ROOT:
      process.env.PRACTICE_RESOURCE_PARAMETER_ROOT?.trim() ||
      '/meritranker/agent-runtime/v1/practice',
  };
}

async function main() {
  // Config root is parent of cdk/ directory. The CLI sets process.cwd() to agentcore/cdk/.
  const configRoot = path.resolve(process.cwd(), '..');
  const configIO = new ConfigIO({ baseDir: configRoot });

  const spec = await configIO.readProjectSpec();
  const deploymentSpec = prepareRuntimeSources(spec, configRoot);
  const targets = await configIO.readAWSDeploymentTargets();

  // Extract MCP configuration from project spec.
  // Gateway fields are stored in agentcore.json but may not yet be on the
  // AgentCoreProjectSpec type from @aws/agentcore-cdk, so we read them
  // dynamically and cast the resulting object.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const specAny = spec as any;
  const mcpSpec = specAny.agentCoreGateways?.length
    ? {
        agentCoreGateways: specAny.agentCoreGateways,
        mcpRuntimeTools: specAny.mcpRuntimeTools,
        unassignedTargets: specAny.unassignedTargets,
      }
    : undefined;

  // Read deployed state for credential ARNs (populated by pre-deploy identity setup)
  let deployedState: Record<string, unknown> | undefined;
  try {
    deployedState = JSON.parse(fs.readFileSync(path.join(configRoot, '.cli', 'deployed-state.json'), 'utf8'));
  } catch {
    // Deployed state may not exist on first deploy
  }

  if (targets.length === 0) {
    throw new Error('No deployment targets configured. Please define targets in agentcore/aws-targets.json');
  }

  const app = new App();

  for (const target of targets) {
    const env = toEnvironment(target);
    const stackName = toStackName(deploymentSpec.name, target.name);

    // Extract credentials from deployed state for this target
    const targetState = (deployedState as Record<string, unknown>)?.targets as
      | Record<string, Record<string, unknown>>
      | undefined;
    const targetResources = targetState?.[target.name]?.resources as Record<string, unknown> | undefined;
    const credentials = targetResources?.credentials as
      | Record<string, { credentialProviderArn: string; clientSecretArn?: string }>
      | undefined;

    new AgentCoreStack(app, stackName, {
      spec: deploymentSpec,
      deploymentEnvironment: target.name,
      mcpSpec,
      credentials,
      runtimeEnvironment: runtimeEnvironment(target),
      env,
      description: `AgentCore stack for ${spec.name} deployed to ${target.name} (${target.region})`,
      tags: {
        'agentcore:project-name': deploymentSpec.name,
        'agentcore:target-name': target.name,
      },
    });
  }

  app.synth();
}

main().catch((error: unknown) => {
  console.error('AgentCore CDK synthesis failed:', error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
