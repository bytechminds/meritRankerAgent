import type { AgentCoreProjectSpec } from '@aws/agentcore-cdk';
import * as fs from 'fs';
import * as path from 'path';

const EXCLUDED_DIRECTORIES = new Set([
  '.git',
  '.mypy_cache',
  '.pytest_cache',
  '.ruff_cache',
  '.venv',
  '__pycache__',
  'tests',
]);

function isDeployableSource(sourceRoot: string, sourcePath: string): boolean {
  const relative = path.relative(sourceRoot, sourcePath);
  if (!relative) return true;
  const segments = relative.split(path.sep);
  if (segments.some(segment => EXCLUDED_DIRECTORIES.has(segment))) return false;
  const name = path.basename(sourcePath);
  if (name === '.env' || name.startsWith('.env.')) return false;
  if (name.endsWith('.log') || name.endsWith('.pyc')) return false;
  return true;
}

export function prepareRuntimeSources(spec: AgentCoreProjectSpec, configRoot: string): AgentCoreProjectSpec {
  const repositoryRoot = path.dirname(configRoot);
  const stagingRoot = path.join(configRoot, '.cache', 'runtime-source');
  fs.rmSync(stagingRoot, { recursive: true, force: true });

  return {
    ...spec,
    runtimes: spec.runtimes.map(runtime => {
      const sourceRoot = path.resolve(repositoryRoot, runtime.codeLocation);
      const destination = path.join(stagingRoot, runtime.name);
      fs.cpSync(sourceRoot, destination, {
        recursive: true,
        filter: sourcePath => isDeployableSource(sourceRoot, sourcePath),
      });
      if (!fs.existsSync(path.join(destination, 'pyproject.toml'))) {
        throw new Error(`Sanitized runtime source is missing pyproject.toml: ${runtime.name}`);
      }
      return {
        ...runtime,
        codeLocation: path.relative(repositoryRoot, destination) as typeof runtime.codeLocation,
      };
    }),
  };
}
