import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

import dashboard from '../.vercel/output/functions/api/v1/dashboard.func/index.js';
import experiments from '../.vercel/output/functions/api/v1/experiments.func/index.js';
import health from '../.vercel/output/functions/api/v1/health.func/index.js';
import snapshot from '../publicSnapshot.js';

function invoke(handler, url) {
  const response = {
    code: 200,
    status(code) { this.code = code; return this; },
    setHeader() { return this; },
    json(body) { this.body = body; return this; },
  };
  handler({ url, method: 'GET' }, response);
  return response;
}

test('prebuilt output serves the measured dashboard and paginated experiments', async () => {
  const config = JSON.parse(await readFile(new URL('../.vercel/output/config.json', import.meta.url)));
  assert.equal(config.version, 3);
  assert.match(await readFile(new URL('../.vercel/output/static/index.html', import.meta.url), 'utf8'), /assets\/index-/);
  const result = invoke(dashboard, 'https://demo.example/api/v1/dashboard');
  assert.equal(result.code, 200);
  assert.deepEqual(result.body, snapshot);
  const page = invoke(experiments, 'https://demo.example/api/v1/experiments?limit=6&offset=6');
  assert.equal(page.code, 200);
  assert.deepEqual(page.body.experiments, snapshot.experiments.slice(6, 12));
  const invalid = invoke(experiments, 'https://demo.example/api/v1/experiments?limit=0');
  assert.equal(invalid.code, 422);
  const check = invoke(health, 'https://demo.example/api/v1/health');
  assert.equal(check.body.report_sha256, snapshot.report_sha256);
});
