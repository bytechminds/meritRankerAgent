#!/usr/bin/env node
import { spawn, spawnSync } from 'child_process';
import * as path from 'path';
import { INHERITED_PARENT_ENV, runPreprodLocal, type Env } from '../lib/preprod-local';

// Operator-tooling defaults only; never part of application configuration.
const prodProfile = process.env.MERITRANKER_PROD_PROFILE || 'meritranker-prod';
const devProfile = process.env.MERITRANKER_DEV_PROFILE || 'dev';

function aws(args: string[], profile: string): string {
  // Same allowlisted environment as the child, so the checks see what the child will use.
  const env: NodeJS.ProcessEnv = { AWS_PROFILE: profile };
  for (const name of INHERITED_PARENT_ENV) if (process.env[name] !== undefined) env[name] = process.env[name];
  const result = spawnSync('aws', args, { env, encoding: 'utf8', maxBuffer: 1024 * 1024 });
  if (result.status !== 0) {
    // stdout is never echoed: on get-secret-value it would be the secret.
    throw new Error(`aws ${args.slice(0, 2).join(' ')} failed for profile ${profile}.`);
  }
  return result.stdout;
}

function run(command: string, args: string[], options: { cwd: string; env: Env }): Promise<number> {
  return new Promise(resolve => {
    const child = spawn(command, args, { cwd: options.cwd, env: options.env as NodeJS.ProcessEnv, stdio: 'inherit' });
    for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => child.kill(signal));
    child.on('exit', code => resolve(code ?? 1));
  });
}

runPreprodLocal(
  {
    repoRoot: path.resolve(__dirname, '..', '..', '..', '..'),
    prodProfile,
    devProfile,
    port: process.env.PREPROD_PORT || '8080',
    checkOnly: process.argv.includes('--check'),
    parentEnv: { ...process.env },
  },
  { aws, spawn: run, log: line => console.log(line) }
)
  .then(code => {
    process.exitCode = code;
  })
  .catch((error: unknown) => {
    console.error(`PREPROD FAIL: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  });
