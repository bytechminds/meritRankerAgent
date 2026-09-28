import type { AgentCoreProjectSpec, AwsDeploymentTarget } from '@aws/agentcore-cdk';
import * as fs from 'fs';
import * as path from 'path';

export const PROD_TARGET_NAME = 'Prod';
export const PROD_QUALIFIER = 'prod';
/** Deploy-tooling selector: which single target this synthesis produces. */
export const DEPLOY_TARGET_SELECTOR_ENV = 'AGENTCORE_DEPLOY_TARGET';
export const PROVIDER_SECRET_ENV = 'PROVIDER_CREDENTIALS_SECRET_ID';

const PLACEHOLDER_ACCOUNT = '000000000000';
const PLACEHOLDER_MARKER = 'REPLACE_ME';

/** Existing ENV names whose Prod value must be committed, never inherited from a shell. */
export const REQUIRED_PROD_RUNTIME_ENV = [
  'APP_ENV',
  'ENABLE_REAL_LLM',
  'ENABLE_ORCHESTRATED_DOUBT_SOLVER',
  'PRACTICE_GENERATION_ENABLED',
  'PRACTICE_QUESTION_BANK_PROMOTION_ENABLED',
  'PATTERN_INTELLIGENCE_ENABLED',
  'PATTERN_INTELLIGENCE_REUSE_ENABLED',
  'PRACTICE_PATTERN_CONTEXT_ENABLED',
  'STUDENT_CREDIT_ENFORCEMENT_ENABLED',
  'STUDENT_CREDIT_DRY_RUN',
  'WEB_SEARCH_ENABLED',
  'IMAGE_CLASSIFIER_ENABLED',
  'ANSWER_RECOVERY_ENABLED',
  'AZURE_OPENAI_DEPLOYMENT_GPT_4_1',
  'AZURE_OPENAI_DEPLOYMENT_GPT_4_1_MINI',
  'AZURE_OPENAI_DEPLOYMENT_GPT_5_4_MINI',
  'AZURE_OPENAI_DEPLOYMENT_GPT_6_SOL',
  'APPSYNC_GRAPHQL_ENDPOINT',
  'PRACTICE_RESOURCE_PARAMETER_ROOT',
] as const;

// Credentials belong in the provider secret or the execution role, never in
// synthesized runtime environment variables.
const FORBIDDEN_RUNTIME_ENV = /(^AWS_ACCESS_KEY_ID$|^AWS_SECRET_ACCESS_KEY$|^AWS_SESSION_TOKEN$|_API_KEY$|_SECRET$|_TOKEN$|^AZURE_OPENAI_ENDPOINT$)/;

export interface ProdTargetConfig {
  target: AwsDeploymentTarget;
  runtimeName: string;
  providerSecretName: string;
  requiredSecretKeys: string[];
  /** Runtime version the `prod` endpoint points to; null until the first version exists. */
  promotedVersion: number | null;
  runtimeEnvironment: Record<string, string>;
}

function fail(message: string): never {
  throw new Error(`Prod target is not deployable: ${message}`);
}

export function validateProdTargetConfig(target: AwsDeploymentTarget, raw: unknown): ProdTargetConfig {
  if (!/^\d{12}$/.test(target.account) || target.account === PLACEHOLDER_ACCOUNT) {
    fail(`aws-targets.json Prod account must be the real 12-digit production account (found ${target.account}).`);
  }
  if (!target.region) fail('aws-targets.json Prod region is required.');
  if (!raw || typeof raw !== 'object') fail('prod-target.json must be a JSON object.');
  const config = raw as Record<string, unknown>;
  const runtimeName = config.runtimeName;
  if (typeof runtimeName !== 'string' || !/^[a-zA-Z][a-zA-Z0-9_]{0,47}$/.test(runtimeName)) {
    fail('runtimeName must match ^[a-zA-Z][a-zA-Z0-9_]{0,47}$.');
  }
  const providerSecretName = config.providerSecretName;
  if (typeof providerSecretName !== 'string' || !/^[A-Za-z0-9/_+=.@-]{1,512}$/.test(providerSecretName)) {
    fail('providerSecretName must be a Secrets Manager secret name.');
  }
  const requiredSecretKeys = config.requiredSecretKeys;
  if (
    !Array.isArray(requiredSecretKeys) ||
    requiredSecretKeys.length === 0 ||
    !requiredSecretKeys.every(key => typeof key === 'string' && /^[A-Z][A-Z0-9_]*$/.test(key))
  ) {
    fail('requiredSecretKeys must be a non-empty list of ENV names.');
  }
  const promotedVersion = config.promotedVersion ?? null;
  if (promotedVersion !== null && !(Number.isInteger(promotedVersion) && (promotedVersion as number) > 0)) {
    fail('promotedVersion must be null or a positive integer runtime version.');
  }
  const environment = config.runtimeEnvironment;
  if (!environment || typeof environment !== 'object' || Array.isArray(environment)) {
    fail('runtimeEnvironment must be an object of existing ENV names.');
  }
  const runtimeEnvironment: Record<string, string> = {};
  for (const [name, value] of Object.entries(environment as Record<string, unknown>)) {
    if (typeof value !== 'string' || !value.trim()) fail(`runtimeEnvironment.${name} must be a non-empty string.`);
    if (value.includes(PLACEHOLDER_MARKER)) fail(`runtimeEnvironment.${name} still contains ${PLACEHOLDER_MARKER}.`);
    if (FORBIDDEN_RUNTIME_ENV.test(name) || (requiredSecretKeys as string[]).includes(name)) {
      fail(`runtimeEnvironment.${name} is a credential; it belongs in the provider secret.`);
    }
    runtimeEnvironment[name] = value;
  }
  for (const name of REQUIRED_PROD_RUNTIME_ENV) {
    if (!(name in runtimeEnvironment)) fail(`runtimeEnvironment.${name} must be set explicitly.`);
  }
  if (runtimeEnvironment.APP_ENV !== 'production') fail('runtimeEnvironment.APP_ENV must be "production".');
  if (runtimeEnvironment.ENABLE_REAL_LLM !== 'true') fail('runtimeEnvironment.ENABLE_REAL_LLM must be "true".');
  if (PROVIDER_SECRET_ENV in runtimeEnvironment) fail(`${PROVIDER_SECRET_ENV} is derived from providerSecretName.`);
  if (!/^https:\/\/[a-z0-9]+\.appsync-api\.[a-z0-9-]+\.amazonaws\.com\/graphql$/.test(runtimeEnvironment.APPSYNC_GRAPHQL_ENDPOINT)) {
    fail('runtimeEnvironment.APPSYNC_GRAPHQL_ENDPOINT must be the production AppSync GraphQL URL.');
  }
  return {
    target,
    runtimeName: runtimeName as string,
    providerSecretName: providerSecretName as string,
    requiredSecretKeys: requiredSecretKeys as string[],
    promotedVersion: promotedVersion as number | null,
    runtimeEnvironment: {
      ...runtimeEnvironment,
      AWS_REGION: target.region,
      [PROVIDER_SECRET_ENV]: providerSecretName as string,
    },
  };
}

export function loadProdTargetConfig(configRoot: string, target: AwsDeploymentTarget): ProdTargetConfig {
  const file = path.join(configRoot, 'prod-target.json');
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    fail(`${file} is missing or not valid JSON.`);
  }
  return validateProdTargetConfig(target, raw);
}

/**
 * One synthesis produces exactly one target stack. Unset selects every non-Prod
 * target (today: Dev only), so existing Dev deploys synthesize exactly what they
 * did before; Prod is synthesized only when explicitly selected.
 */
export function selectDeploymentTargets(
  targets: AwsDeploymentTarget[],
  selector: string | undefined
): AwsDeploymentTarget[] {
  const requested = selector?.trim();
  if (!requested) return targets.filter(target => target.name !== PROD_TARGET_NAME);
  const selected = targets.filter(target => target.name === requested);
  if (selected.length !== 1) {
    throw new Error(`${DEPLOY_TARGET_SELECTOR_ENV}=${requested} does not name exactly one target in aws-targets.json.`);
  }
  return selected;
}

/** Adds the stable `prod` endpoint through the existing L3 runtime `endpoints` mechanism. */
export function withProdEndpoint(spec: AgentCoreProjectSpec, promotedVersion: number | null): AgentCoreProjectSpec {
  if (promotedVersion === null) return spec;
  return {
    ...spec,
    runtimes: spec.runtimes.map(runtime => ({
      ...runtime,
      endpoints: {
        [PROD_QUALIFIER]: {
          version: promotedVersion,
          description: 'MeritRanker production release pointer',
        },
      },
    })) as AgentCoreProjectSpec['runtimes'],
  };
}
