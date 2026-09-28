import type { AwsDeploymentTarget } from '@aws/agentcore-cdk';
import { spawnSync } from 'child_process';
import * as crypto from 'crypto';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import {
  assertCredentialOnlySecret,
  buildPreprodEnvironment,
  formatConfigSummary,
  parseProviderSecret,
  runPreprodLocal,
  type Env,
  type PreprodDependencies,
} from '../lib/preprod-local';
import { validateProdTargetConfig } from '../lib/prod-target';
import { isDeployableSource } from '../lib/runtime-source';

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const PROD: AwsDeploymentTarget = { name: 'Prod', account: '123456789012', region: 'ap-south-1' };
const DEV_ACCOUNT = '661012794182';
const PROD_APPSYNC = 'https://prodapi0123456789.appsync-api.ap-south-1.amazonaws.com/graphql';
const DEV_APPSYNC = 'https://devapi0123456789.appsync-api.ap-south-1.amazonaws.com/graphql';
const SECRET = {
  AZURE_OPENAI_ENDPOINT: 'https://secret-endpoint-value.invalid/openai/v1',
  AZURE_OPENAI_API_KEY: 'AZURE-SECRET-VALUE-1',
  GEMINI_API_KEY: 'GEMINI-SECRET-VALUE-2',
  GOOGLE_GEMINI_API_KEY: 'IMAGE-SECRET-VALUE-3',
  TAVILY_API_KEY: 'TAVILY-SECRET-VALUE-4',
};
const SECRET_VALUES = Object.values(SECRET);

function filledProdFile(): Record<string, any> {
  const config = JSON.parse(fs.readFileSync(path.join(REPO_ROOT, 'agentcore', 'prod-target.json'), 'utf8'));
  config.runtimeEnvironment.APPSYNC_GRAPHQL_ENDPOINT = PROD_APPSYNC;
  return config;
}

const prod = validateProdTargetConfig(PROD, filledProdFile());

/** Hostile parent shell: experimental flags, static keys, a secret id, a stray memory id. */
const HOSTILE_PARENT: Env = {
  PATH: process.env.PATH,
  HOME: process.env.HOME,
  APP_ENV: 'local',
  PRACTICE_GENERATION_ENABLED: 'false',
  WEB_SEARCH_ENABLED: 'false',
  STUDENT_CREDIT_DRY_RUN: 'false',
  AZURE_OPENAI_DEPLOYMENT_GPT_6_SOL: 'experimental-deployment',
  AWS_ACCESS_KEY_ID: 'AKIASHELLKEY',
  AWS_SECRET_ACCESS_KEY: 'shell-secret',
  AWS_SESSION_TOKEN: 'shell-token',
  PROVIDER_CREDENTIALS_SECRET_ID: 'some/other/secret',
  MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID: 'prod-memory-id',
  ENABLE_KB_RETRIEVAL: 'true',
  DEEPSEEK_API_KEY: 'shell-deepseek-key',
};

function preprodEnv(parentEnv: Env = HOSTILE_PARENT): Env {
  return buildPreprodEnvironment({
    prod,
    secretValues: SECRET,
    parentEnv,
    devProfile: 'dev',
    localResources: { APPSYNC_GRAPHQL_ENDPOINT: DEV_APPSYNC, AWS_REGION: 'ap-south-1' },
    devMemoryId: 'dev-memory-id',
    port: '8080',
  });
}

describe('PreProd child environment', () => {
  test('every Prod-managed flag comes from prod-target.json, overriding the shell', () => {
    const env = preprodEnv();
    for (const [name, value] of Object.entries(prod.runtimeEnvironment)) {
      if (['APPSYNC_GRAPHQL_ENDPOINT', 'AWS_REGION', 'PROVIDER_CREDENTIALS_SECRET_ID'].includes(name)) continue;
      expect(env[name]).toBe(value);
    }
    expect(env.APP_ENV).toBe('production');
  });

  test('application AWS context is Dev: Dev profile, Dev AppSync, Dev memory, no static or Prod identity', () => {
    const env = preprodEnv();
    expect(env.AWS_PROFILE).toBe('dev');
    expect(env.APPSYNC_GRAPHQL_ENDPOINT).toBe(DEV_APPSYNC);
    expect(env.MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID).toBe('dev-memory-id');
    for (const name of [
      'AWS_ACCESS_KEY_ID',
      'AWS_SECRET_ACCESS_KEY',
      'AWS_SESSION_TOKEN',
      'PROVIDER_CREDENTIALS_SECRET_ID',
    ]) {
      expect(env).not.toHaveProperty(name);
    }
  });

  test('unmanaged shell settings do not reach the child; they stay at code defaults as in Prod', () => {
    const env = preprodEnv();
    expect(env).not.toHaveProperty('ENABLE_KB_RETRIEVAL');
    expect(env).not.toHaveProperty('DEEPSEEK_API_KEY');
    expect(env.PATH).toBe(process.env.PATH);
  });

  test('credentials reach the child under their existing ENV names only', () => {
    const env = preprodEnv();
    for (const [name, value] of Object.entries(SECRET)) expect(env[name]).toBe(value);
  });

  test('staged runtime ignores app/.env.local entirely: Prod flags win, unmanaged settings are code defaults', () => {
    const python = path.join(REPO_ROOT, 'app', '.venv', 'bin', 'python');
    // Stage with the deploy filter into a test-only directory (never the shared deploy staging).
    const appDir = path.join(REPO_ROOT, 'app');
    const staged = fs.mkdtempSync(path.join(os.tmpdir(), 'preprod-staged-'));
    try {
      fs.cpSync(appDir, staged, {
        recursive: true,
        filter: source => isDeployableSource(appDir, source),
      });
      expect(fs.existsSync(path.join(staged, '.env.local'))).toBe(false);
      const env = preprodEnv({
        ...HOSTILE_PARENT,
        // Local resource identities that normally come from Dev SSM; given so no AWS call is made.
        DYNAMODB_USER_CREDITS_TABLE: 'dev-user-credits',
        DYNAMODB_CREDIT_LEDGER_TABLE: 'dev-credit-ledger',
      });
      const script = [
        'import json, os, config',
        's = config.get_settings()',
        'print(json.dumps({"app_env": s.app_env, "log_level": s.log_level, "real": s.enable_real_llm,',
        ' "orchestrated": s.enable_orchestrated_doubt_solver, "web": s.web_search_enabled,',
        ' "image": s.image_classifier_enabled, "recovery": s.answer_recovery_enabled,',
        ' "enforce": s.student_credit_enforcement_enabled, "dry_run": s.student_credit_dry_run,',
        ' "practice": os.environ["PRACTICE_GENERATION_ENABLED"], "promotion": os.environ["PRACTICE_QUESTION_BANK_PROMOTION_ENABLED"],',
        ' "sol": os.environ["AZURE_OPENAI_DEPLOYMENT_GPT_6_SOL"], "tavily_is_secret": s.tavily_api_key == os.environ["TAVILY_API_KEY"],',
        ' "kb": s.enable_kb_retrieval, "role_json": s.llm_role_config_json, "deepseek_key": s.deepseek_api_key}))',
      ].join('\n');
      const result = spawnSync(python, ['-c', script], {
        cwd: staged,
        env: env as NodeJS.ProcessEnv,
        encoding: 'utf8',
      });
      expect(result.status).toBe(0);
      const settings = JSON.parse(result.stdout.trim().split('\n').pop() as string);
      expect(settings).toEqual({
        app_env: 'production',
        log_level: 'INFO',
        real: true,
        orchestrated: true,
        web: true,
        image: true,
        recovery: true,
        enforce: true,
        dry_run: true,
        practice: 'true',
        promotion: 'true',
        sol: 'gpt-6-sol',
        tavily_is_secret: true,
        kb: false,
        role_json: '{}',
        deepseek_key: '',
      });
      for (const value of SECRET_VALUES) {
        expect(result.stdout).not.toContain(value);
        expect(result.stderr).not.toContain(value);
      }
    } finally {
      fs.rmSync(staged, { recursive: true, force: true });
    }
  }, 120_000);
});

test('secret may carry credentials only, never flags, resources, or AWS identity', () => {
  expect(() => assertCredentialOnlySecret(SECRET, prod)).not.toThrow();
  expect(() => assertCredentialOnlySecret({ ...SECRET, DEEPSEEK_API_KEY: 'x' }, prod)).not.toThrow();
  for (const key of [
    'APP_ENV',
    'APPSYNC_GRAPHQL_ENDPOINT',
    'AWS_ACCESS_KEY_ID',
    'MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID',
    'PROVIDER_CREDENTIALS_SECRET_ID',
  ]) {
    expect(() => assertCredentialOnlySecret({ ...SECRET, [key]: 'SENSITIVE-VALUE' }, prod)).toThrow(
      new RegExp(`non-credential keys: ${key}$`)
    );
  }
});

describe('provider secret parsing', () => {
  test('accepts the runtime shape and reports problems by key name only', () => {
    expect(parseProviderSecret(JSON.stringify(SECRET), prod.requiredSecretKeys)).toEqual(SECRET);
    const missing = { ...SECRET, TAVILY_API_KEY: ' ' };
    expect(() => parseProviderSecret(JSON.stringify(missing), prod.requiredSecretKeys)).toThrow(
      /missing: TAVILY_API_KEY$/
    );
    expect(() => parseProviderSecret(JSON.stringify({ ...SECRET, notes: 'x' }), prod.requiredSecretKeys)).toThrow(
      /malformed entries: notes$/
    );
    for (const bad of ['not json AZURE-SECRET-VALUE-1', '["AZURE-SECRET-VALUE-1"]']) {
      try {
        parseProviderSecret(bad, prod.requiredSecretKeys);
        throw new Error('expected failure');
      } catch (error) {
        expect(String(error)).not.toContain('AZURE-SECRET-VALUE-1');
      }
    }
  });
});

test('safe summary shows flags and deployment names, never credentials', () => {
  const summary = formatConfigSummary(prod, [['Target mode', 'PREPROD-LOCAL']]);
  expect(summary).toContain('APP_ENV');
  expect(summary).toMatch(/Practice\s+: ON/);
  expect(summary).toMatch(/Credit mode\s+: enforce \+ dry-run/);
  expect(summary).toContain('AZURE_OPENAI_DEPLOYMENT_GPT_6_SOL=gpt-6-sol');
  for (const value of SECRET_VALUES) expect(summary).not.toContain(value);
});

describe('runPreprodLocal', () => {
  let root: string;

  function write(relative: string, content: string) {
    fs.mkdirSync(path.dirname(path.join(root, relative)), { recursive: true });
    fs.writeFileSync(path.join(root, relative), content);
  }

  /** Hashes of every file except the sanitized source staging (code copies only). */
  function snapshot(): Record<string, string> {
    const files: Record<string, string> = {};
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (path.relative(root, full) === path.join('agentcore', '.cache', 'preprod-runtime')) continue;
        if (entry.isDirectory()) walk(full);
        else files[path.relative(root, full)] = crypto.createHash('sha256').update(fs.readFileSync(full)).digest('hex');
      }
    };
    walk(root);
    return files;
  }

  beforeEach(() => {
    root = fs.mkdtempSync(path.join(os.tmpdir(), 'preprod-'));
    write('agentcore/aws-targets.json', JSON.stringify([{ ...PROD, name: 'Dev', account: DEV_ACCOUNT }, PROD]));
    write('agentcore/prod-target.json', JSON.stringify(filledProdFile()));
    write(
      'agentcore/.cli/deployed-state.json',
      JSON.stringify({
        targets: {
          Dev: { resources: { memories: { meritranker_short_term_memory: { memoryId: 'dev-memory-id' } } } },
          Prod: { resources: { memories: { meritranker_short_term_memory: { memoryId: 'prod-memory-id' } } } },
        },
      })
    );
    write(
      'app/.env.local',
      `STUDENT_CREDIT_DRY_RUN=false\nAPPSYNC_GRAPHQL_ENDPOINT=${DEV_APPSYNC}\nAWS_REGION=ap-south-1\n`
    );
    write('app/.venv/bin/uvicorn', '');
    write('app/pyproject.toml', '[project]\nname = "t"\nversion = "1"\n');
    write('app/main.py', 'app = None\n');
    write('app/.logs/agent-events.jsonl', 'local event\n');
    fs.copyFileSync(
      path.join(REPO_ROOT, 'agentcore', 'agentcore.json'),
      path.join(root, 'agentcore', 'agentcore.json')
    );
  });

  function allFileContents(): string {
    const chunks: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else chunks.push(fs.readFileSync(full, 'utf8'));
      }
    };
    walk(root);
    return chunks.join('\n');
  }

  afterEach(() => fs.rmSync(root, { recursive: true, force: true }));

  function deps(accounts: Record<string, string>, secret: string = JSON.stringify(SECRET)) {
    const logs: string[] = [];
    const spawned: { command: string; args: string[]; cwd: string; env: Env }[] = [];
    const awsCalls: { args: string[]; profile: string }[] = [];
    const d: PreprodDependencies = {
      aws(args, profile) {
        awsCalls.push({ args, profile });
        if (args[0] === 'sts') return `${accounts[profile]}\n`;
        return `${secret}\n`;
      },
      async spawn(command, args, options) {
        spawned.push({ command, args, ...options });
        return 0;
      },
      log: line => logs.push(line),
    };
    return { d, logs, spawned, awsCalls };
  }

  const options = (checkOnly = false) => ({
    repoRoot: root,
    prodProfile: 'meritranker-prod',
    devProfile: 'dev',
    port: '8080',
    checkOnly,
    parentEnv: HOSTILE_PARENT,
  });

  test('starts the existing local server with Prod behaviour, Dev data, and nothing written or printed', async () => {
    const before = snapshot();
    const { d, logs, spawned, awsCalls } = deps({ 'meritranker-prod': PROD.account, dev: DEV_ACCOUNT });
    await expect(runPreprodLocal(options(), d)).resolves.toBe(0);

    expect(snapshot()).toEqual(before);
    const onDisk = allFileContents();
    for (const value of SECRET_VALUES) expect(onDisk).not.toContain(value);
    const output = logs.join('\n');
    for (const value of SECRET_VALUES) expect(output).not.toContain(value);
    expect(output).toMatch(/Target mode\s+: PREPROD-LOCAL/);
    expect(output).toMatch(/Provider secret\s+: FOUND/);

    // Prod profile is used only for identity and the one secret; Dev only for identity.
    expect(awsCalls.map(c => `${c.profile}:${c.args[0]} ${c.args[1]}`)).toEqual([
      'meritranker-prod:sts get-caller-identity',
      'meritranker-prod:secretsmanager get-secret-value',
      'dev:sts get-caller-identity',
    ]);
    expect(awsCalls[1].args).toContain('meritranker/agent-runtime/prod/providers');

    expect(spawned).toHaveLength(1);
    const [{ command, args, cwd, env }] = spawned;
    expect(command).toBe(path.join(root, 'app', '.venv', 'bin', 'uvicorn'));
    expect(args).toEqual(['main:app', '--host', '127.0.0.1', '--port', '8080']);
    expect(cwd).toBe(path.join(root, 'agentcore', '.cache', 'preprod-runtime'));
    expect(fs.existsSync(path.join(root, 'agentcore', '.cache', 'runtime-source'))).toBe(false);
    expect(fs.existsSync(path.join(cwd, 'main.py'))).toBe(true);
    expect(fs.existsSync(path.join(cwd, '.env.local'))).toBe(false);
    expect(fs.existsSync(path.join(cwd, '.logs'))).toBe(false);
    expect(env).not.toHaveProperty('ENABLE_KB_RETRIEVAL');
    expect(env.AWS_PROFILE).toBe('dev');
    expect(env.APPSYNC_GRAPHQL_ENDPOINT).toBe(DEV_APPSYNC);
    expect(env.MEMORY_MERITRANKER_SHORT_TERM_MEMORY_ID).toBe('dev-memory-id');
    expect(env.STUDENT_CREDIT_DRY_RUN).toBe('true');
    expect(env.APP_ENV).toBe('production');
    for (const arg of args) for (const value of SECRET_VALUES) expect(arg).not.toContain(value);
  });

  test('--check validates and summarizes without starting the runtime', async () => {
    const { d, spawned } = deps({ 'meritranker-prod': PROD.account, dev: DEV_ACCOUNT });
    await expect(runPreprodLocal(options(true), d)).resolves.toBe(0);
    expect(spawned).toHaveLength(0);
  });

  test.each([
    ['prod profile is not the Prod account', { 'meritranker-prod': DEV_ACCOUNT, dev: DEV_ACCOUNT }, /is account/],
    ['dev profile is the Prod account', { 'meritranker-prod': PROD.account, dev: PROD.account }, /must use Dev data/],
  ])('stops when %s', async (_label, accounts, message) => {
    const { d, spawned } = deps(accounts);
    await expect(runPreprodLocal(options(), d)).rejects.toThrow(message);
    expect(spawned).toHaveLength(0);
  });

  test('stops when local AppSync points at Prod', async () => {
    write('app/.env.local', `APPSYNC_GRAPHQL_ENDPOINT=${PROD_APPSYNC}\n`);
    const { d, spawned } = deps({ 'meritranker-prod': PROD.account, dev: DEV_ACCOUNT });
    await expect(runPreprodLocal(options(), d)).rejects.toThrow(/Prod endpoint/);
    expect(spawned).toHaveLength(0);
  });

  test('stops when the secret carries a non-credential key', async () => {
    const { d, spawned } = deps(
      { 'meritranker-prod': PROD.account, dev: DEV_ACCOUNT },
      JSON.stringify({ ...SECRET, APPSYNC_GRAPHQL_ENDPOINT: PROD_APPSYNC })
    );
    await expect(runPreprodLocal(options(), d)).rejects.toThrow(/non-credential keys: APPSYNC_GRAPHQL_ENDPOINT/);
    expect(spawned).toHaveLength(0);
  });

  test('stops when local AppSync is the Prod endpoint in another spelling', async () => {
    write('app/.env.local', `APPSYNC_GRAPHQL_ENDPOINT=${PROD_APPSYNC.toUpperCase()}/\n`);
    const { d } = deps({ 'meritranker-prod': PROD.account, dev: DEV_ACCOUNT });
    await expect(runPreprodLocal(options(), d)).rejects.toThrow(/Prod endpoint/);
  });

  test('stops on a missing required secret key without revealing values', async () => {
    const { d, spawned } = deps(
      { 'meritranker-prod': PROD.account, dev: DEV_ACCOUNT },
      JSON.stringify({ ...SECRET, GEMINI_API_KEY: '' })
    );
    const error = await runPreprodLocal(options(), d).catch(e => e);
    expect(String(error)).toMatch(/missing: GEMINI_API_KEY/);
    for (const value of SECRET_VALUES) expect(String(error)).not.toContain(value);
    expect(spawned).toHaveLength(0);
  });

  test('stops while prod-target.json still has placeholders', async () => {
    write('agentcore/prod-target.json', fs.readFileSync(path.join(REPO_ROOT, 'agentcore', 'prod-target.json'), 'utf8'));
    const { d } = deps({ 'meritranker-prod': PROD.account, dev: DEV_ACCOUNT });
    await expect(runPreprodLocal(options(), d)).rejects.toThrow(/REPLACE_ME/);
  });
});

test('Prod synthesis through bin/cdk.ts ignores exported shell feature flags', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'prod-synth-'));
  try {
    fs.mkdirSync(path.join(root, 'agentcore', 'cdk'), { recursive: true });
    fs.mkdirSync(path.join(root, 'app'));
    fs.writeFileSync(path.join(root, 'app', 'pyproject.toml'), '[project]\nname = "t"\nversion = "1"\n');
    fs.writeFileSync(path.join(root, 'app', 'main.py'), 'app = None\n');
    fs.copyFileSync(
      path.join(REPO_ROOT, 'agentcore', 'agentcore.json'),
      path.join(root, 'agentcore', 'agentcore.json')
    );
    fs.writeFileSync(
      path.join(root, 'agentcore', 'aws-targets.json'),
      JSON.stringify([{ ...PROD, name: 'Dev', account: DEV_ACCOUNT }, PROD])
    );
    fs.writeFileSync(path.join(root, 'agentcore', 'prod-target.json'), JSON.stringify(filledProdFile()));
    const out = path.join(root, 'out');
    const result = spawnSync('node', [path.join(__dirname, '..', 'dist', 'bin', 'cdk.js')], {
      cwd: path.join(root, 'agentcore', 'cdk'),
      env: {
        ...process.env,
        ...HOSTILE_PARENT,
        ENABLE_REAL_LLM: 'false',
        PATTERN_INTELLIGENCE_ENABLED: 'true',
        AGENTCORE_DEPLOY_TARGET: 'Prod',
        CDK_OUTDIR: out,
      } as NodeJS.ProcessEnv,
      encoding: 'utf8',
    });
    expect(result.stderr).toBe('');
    expect(fs.readdirSync(out).filter(name => name.endsWith('.template.json'))).toEqual([
      'AgentCore-meritRankerTutor-Prod.template.json',
    ]);
    const template = JSON.parse(
      fs.readFileSync(path.join(out, 'AgentCore-meritRankerTutor-Prod.template.json'), 'utf8')
    );
    const runtime: any = Object.values(template.Resources).find(
      (r: any) => r.Type === 'AWS::BedrockAgentCore::Runtime'
    );
    const deployed = runtime.Properties.EnvironmentVariables;
    for (const [name, value] of Object.entries(prod.runtimeEnvironment)) expect(deployed[name]).toBe(value);
    expect(JSON.stringify(template)).not.toMatch(
      /AKIASHELLKEY|shell-secret|some\/other\/secret|experimental-deployment/
    );
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}, 180_000);
