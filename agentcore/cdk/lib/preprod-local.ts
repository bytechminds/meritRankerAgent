import type { AwsDeploymentTarget } from '@aws/agentcore-cdk';
import * as fs from 'fs';
import * as path from 'path';
import { PROD_TARGET_NAME, PROVIDER_SECRET_ENV, loadProdTargetConfig, type ProdTargetConfig } from './prod-target';
import { isDeployableSource } from './runtime-source';

/**
 * PreProd-local = Prod behaviour + Prod provider credentials + local runtime + Dev data.
 *
 * Prod-managed values in prod-target.json that identify AWS resources stay on the
 * local/Dev side so a local run can never write Prod application data.
 */
export const LOCAL_RESOURCE_ENV = ['APPSYNC_GRAPHQL_ENDPOINT', 'AWS_REGION'] as const;

// The only shell variables the child inherits; everything else starts at code defaults,
// as in the deployed runtime. Static AWS keys and experimental flags never pass through.
export const INHERITED_PARENT_ENV = [
  'PATH', 'HOME', 'USER', 'LOGNAME', 'SHELL', 'TERM', 'TMPDIR', 'LANG', 'LC_ALL', 'LC_CTYPE',
  'AWS_CONFIG_FILE', 'AWS_SHARED_CREDENTIALS_FILE',
];

const DEV_MEMORY_ENV = 'MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID';

export type Env = Record<string, string | undefined>;

/** Same shape the runtime SecretsManagerSecretResolver accepts. Returns only key names on error. */
export function parseProviderSecret(secretString: string, requiredKeys: string[]): Record<string, string> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(secretString);
  } catch {
    throw new Error('Provider secret is not a JSON object.');
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Provider secret is not a JSON object.');
  }
  const entries = Object.entries(parsed as Record<string, unknown>);
  const malformed = entries.filter(([k, v]) => !/^[A-Z][A-Z0-9_]*$/.test(k) || typeof v !== 'string').map(([k]) => k);
  if (malformed.length) throw new Error(`Provider secret has malformed entries: ${malformed.join(', ')}`);
  const values = Object.fromEntries(entries) as Record<string, string>;
  const missing = requiredKeys.filter(k => !values[k]?.trim());
  if (missing.length) throw new Error(`Provider secret is missing: ${missing.join(', ')}`);
  return values;
}

/** The secret may only carry credentials: never a flag, resource identity, or AWS identity. */
export function assertCredentialOnlySecret(values: Record<string, string>, prod: ProdTargetConfig): void {
  const reserved = new Set([...Object.keys(prod.runtimeEnvironment), ...LOCAL_RESOURCE_ENV, PROVIDER_SECRET_ENV]);
  const colliding = Object.keys(values).filter(k => reserved.has(k) || k.startsWith('AWS_') || /^MEMORY_.*_ID$/.test(k));
  if (colliding.length) throw new Error(`Provider secret holds non-credential keys: ${colliding.join(', ')}`);
}

function sameEndpoint(a: string, b: string): boolean {
  const normalize = (url: string) => url.trim().toLowerCase().replace(/\/+$/, '');
  return normalize(a) === normalize(b);
}

/** Reads one KEY=value line from a dotenv file without evaluating anything. */
export function readDotenvValue(file: string, name: string): string | undefined {
  if (!fs.existsSync(file)) return undefined;
  let value: string | undefined;
  for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
    const match = /^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(line);
    if (match?.[1] === name) value = match[2].replace(/^(['"])(.*)\1$/, '$2').trim();
  }
  return value;
}

export interface PreprodEnvironmentInput {
  prod: ProdTargetConfig;
  secretValues: Record<string, string>;
  parentEnv: Env;
  devProfile: string;
  localResources: Record<(typeof LOCAL_RESOURCE_ENV)[number], string>;
  devMemoryId?: string;
  port: string;
}

/** Child environment: Prod-managed flags, credentials, Dev resources; nothing else from the shell. */
export function buildPreprodEnvironment(input: PreprodEnvironmentInput): Env {
  const env: Env = {};
  for (const name of INHERITED_PARENT_ENV) if (input.parentEnv[name] !== undefined) env[name] = input.parentEnv[name];
  for (const [name, value] of Object.entries(input.prod.runtimeEnvironment)) {
    if (name === PROVIDER_SECRET_ENV || (LOCAL_RESOURCE_ENV as readonly string[]).includes(name)) continue;
    env[name] = value;
  }
  Object.assign(env, input.localResources, input.secretValues);
  env.AWS_PROFILE = input.devProfile;
  if (input.devMemoryId) env[DEV_MEMORY_ENV] = input.devMemoryId;
  env.PORT = input.port;
  env.LOCAL_DEV = '1';
  return env;
}

function onOff(value: string | undefined): string {
  return value === 'true' ? 'ON' : 'OFF';
}

/** Operator summary shared by PreProd-local and the Prod preflight. Names and flags only. */
export function formatConfigSummary(prod: ProdTargetConfig, header: [string, string][]): string {
  const e = prod.runtimeEnvironment;
  const creditMode =
    e.STUDENT_CREDIT_ENFORCEMENT_ENABLED !== 'true'
      ? 'disabled'
      : e.STUDENT_CREDIT_DRY_RUN === 'true'
        ? 'enforce + dry-run (no real debits)'
        : 'enforcing (real debits)';
  const rows: [string, string][] = [
    ...header,
    ['Config source', 'agentcore/prod-target.json'],
    ['APP_ENV', e.APP_ENV],
    ['Tutor (orchestrated, real LLM)', onOff(e.ENABLE_ORCHESTRATED_DOUBT_SOLVER === 'true' && e.ENABLE_REAL_LLM === 'true' ? 'true' : 'false')],
    ['Practice', onOff(e.PRACTICE_GENERATION_ENABLED)],
    ['Web search', onOff(e.WEB_SEARCH_ENABLED)],
    ['Image classifier', onOff(e.IMAGE_CLASSIFIER_ENABLED)],
    ['Answer recovery', onOff(e.ANSWER_RECOVERY_ENABLED)],
    ['QuestionBank promotion', onOff(e.PRACTICE_QUESTION_BANK_PROMOTION_ENABLED)],
    ['QuestionBank semantic reuse', e.PRACTICE_QUESTION_SEMANTIC_REUSE_MODE ?? 'off (code default)'],
    ['Pattern Intelligence', onOff(e.PATTERN_INTELLIGENCE_ENABLED)],
    ['Credit mode', creditMode],
    ['Provider secret', prod.providerSecretName],
    ['Prod endpoint version', prod.promotedVersion === null ? 'none yet (first deploy)' : String(prod.promotedVersion)],
  ];
  const deployments = Object.keys(e)
    .filter(name => name.startsWith('AZURE_OPENAI_DEPLOYMENT_'))
    .sort()
    .map(name => `  ${name}=${e[name]}`);
  const width = Math.max(...rows.map(([label]) => label.length));
  return [...rows.map(([label, value]) => `${label.padEnd(width)} : ${value}`), 'Azure deployment names:', ...deployments].join('\n');
}

export interface PreprodDependencies {
  /** Runs the AWS CLI and returns stdout; must never echo stdout on failure. */
  aws(args: string[], profile: string): string;
  spawn(command: string, args: string[], options: { cwd: string; env: Env }): Promise<number>;
  log(line: string): void;
}

export interface PreprodOptions {
  repoRoot: string;
  prodProfile: string;
  devProfile: string;
  port: string;
  checkOnly: boolean;
  parentEnv: Env;
}

function readProdTarget(agentcoreDir: string): AwsDeploymentTarget {
  const targets = JSON.parse(fs.readFileSync(path.join(agentcoreDir, 'aws-targets.json'), 'utf8')) as AwsDeploymentTarget[];
  const prod = targets.filter(target => target.name === PROD_TARGET_NAME);
  if (prod.length !== 1) throw new Error('aws-targets.json must define exactly one Prod target.');
  return prod[0];
}

function callerAccount(deps: PreprodDependencies, profile: string): string {
  return deps.aws(['sts', 'get-caller-identity', '--query', 'Account', '--output', 'text'], profile).trim();
}

export async function runPreprodLocal(options: PreprodOptions, deps: PreprodDependencies): Promise<number> {
  const agentcoreDir = path.join(options.repoRoot, 'agentcore');
  const appDir = path.join(options.repoRoot, 'app');
  const target = readProdTarget(agentcoreDir);

  // 1-3. Prod configuration loads and validates (APP_ENV=production, deployment names, ...).
  const prod = loadProdTargetConfig(agentcoreDir, target);

  // 4-5. Prod provider secret, read with the Prod profile into memory only.
  const prodAccount = callerAccount(deps, options.prodProfile);
  if (prodAccount !== target.account) {
    throw new Error(`Profile ${options.prodProfile} is account ${prodAccount}, Prod is ${target.account}.`);
  }
  const secretString = deps.aws(
    ['secretsmanager', 'get-secret-value', '--secret-id', prod.providerSecretName, '--region', target.region,
      '--query', 'SecretString', '--output', 'text'],
    options.prodProfile
  );
  const secretValues = parseProviderSecret(secretString.replace(/\n$/, ''), prod.requiredSecretKeys);
  assertCredentialOnlySecret(secretValues, prod);

  // 6. Local application data stays on the Dev account and Dev resources.
  const devAccount = callerAccount(deps, options.devProfile);
  if (devAccount === target.account) {
    throw new Error(`Profile ${options.devProfile} resolves to the Prod account; PreProd-local must use Dev data.`);
  }
  const localEnvFile = path.join(appDir, '.env.local');
  const appsync = options.parentEnv.APPSYNC_GRAPHQL_ENDPOINT ?? readDotenvValue(localEnvFile, 'APPSYNC_GRAPHQL_ENDPOINT');
  if (!appsync) throw new Error('APPSYNC_GRAPHQL_ENDPOINT (Dev) must be set in app/.env.local or the shell.');
  if (sameEndpoint(appsync, prod.runtimeEnvironment.APPSYNC_GRAPHQL_ENDPOINT)) {
    throw new Error('Local APPSYNC_GRAPHQL_ENDPOINT is the Prod endpoint; PreProd-local must use Dev.');
  }
  const region = options.parentEnv.AWS_REGION ?? readDotenvValue(localEnvFile, 'AWS_REGION') ?? target.region;
  let devMemoryId: string | undefined;
  try {
    const state = JSON.parse(fs.readFileSync(path.join(agentcoreDir, '.cli', 'deployed-state.json'), 'utf8'));
    devMemoryId = state?.targets?.Dev?.resources?.memories?.meritranker_short_term_memory?.memoryId;
  } catch {
    devMemoryId = undefined;
  }
  const uvicorn = path.join(appDir, '.venv', 'bin', 'uvicorn');
  if (!fs.existsSync(uvicorn)) throw new Error(`Local runtime not installed: ${uvicorn} (run agentcore dev once).`);
  // Serve source filtered exactly as a Prod deploy packages it (no .env files), so
  // app/.env.local cannot change any setting. PreProd has its own staging directory so
  // it never races the deploy staging in .cache/runtime-source.
  const runtimeDir = path.join(agentcoreDir, '.cache', 'preprod-runtime');
  fs.rmSync(runtimeDir, { recursive: true, force: true });
  fs.cpSync(appDir, runtimeDir, { recursive: true, filter: source => isDeployableSource(appDir, source) });

  deps.log(
    formatConfigSummary(prod, [
      ['Target mode', 'PREPROD-LOCAL'],
      ['Agent endpoint', `local http://127.0.0.1:${options.port}/invocations`],
      ['AWS application resources', `Dev/local (profile ${options.devProfile}, account ${devAccount})`],
      ['Runtime source', 'staged like Prod (no .env files); unmanaged settings at code defaults'],
      ['AppSync progress', 'Dev (from app/.env.local or shell)'],
      ['Short-term memory', devMemoryId ? 'Dev memory' : 'none (Dev memory id not deployed)'],
      ['Provider secret source', `Prod Secrets Manager (profile ${options.prodProfile})`],
      ['Provider secret', 'FOUND'],
      ['Required secret keys', 'PRESENT'],
    ])
  );
  if (options.checkOnly) return 0;

  const env = buildPreprodEnvironment({
    prod,
    secretValues,
    parentEnv: options.parentEnv,
    devProfile: options.devProfile,
    localResources: { APPSYNC_GRAPHQL_ENDPOINT: appsync, AWS_REGION: region },
    devMemoryId,
    port: options.port,
  });
  // Same server `agentcore dev` starts for this HTTP runtime, without CLI-injected env.
  return deps.spawn(uvicorn, ['main:app', '--host', '127.0.0.1', '--port', options.port], { cwd: runtimeDir, env });
}
