import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { isDeployableSource } from '../lib/runtime-source';

function stage(files: Record<string, string>): string[] {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'runtime-source-'));
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'runtime-staged-'));
  try {
    for (const [relative, content] of Object.entries(files)) {
      fs.mkdirSync(path.dirname(path.join(root, relative)), { recursive: true });
      fs.writeFileSync(path.join(root, relative), content);
    }
    fs.cpSync(root, out, { recursive: true, filter: source => isDeployableSource(root, source) });
    const staged: string[] = [];
    const walk = (dir: string) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(full);
        else staged.push(path.relative(out, full));
      }
    };
    walk(out);
    return staged.sort();
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
    fs.rmSync(out, { recursive: true, force: true });
  }
}

test('local diagnostic .logs never ship; runtime sources and data assets still do', () => {
  const staged = stage({
    'main.py': 'app = None\n',
    'config/llm/model_registry.yaml': 'models: {}\n',
    'prompts/subjects/math_generator.md': '# prompt\n',
    'data/fixtures.jsonl': '{}\n',
    '.logs/agent-events.jsonl': 'local event\n',
    '.logs/agent-events.jsonl.1': 'rotated event\n',
    '.logs/agent-runtime.log': 'local log\n',
    '.env.local': 'SECRET=unsafe\n',
  });
  expect(staged).toEqual([
    path.join('config', 'llm', 'model_registry.yaml'),
    path.join('data', 'fixtures.jsonl'),
    'main.py',
    path.join('prompts', 'subjects', 'math_generator.md'),
  ]);
});

test('the filter rejects the .logs directory itself and anything below it', () => {
  const root = path.join(os.tmpdir(), 'app');
  expect(isDeployableSource(root, path.join(root, '.logs'))).toBe(false);
  expect(isDeployableSource(root, path.join(root, '.logs', 'agent-events.jsonl.3'))).toBe(false);
  expect(isDeployableSource(root, path.join(root, 'observability', 'logging.py'))).toBe(true);
});
