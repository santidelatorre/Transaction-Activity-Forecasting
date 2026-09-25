import assert from 'node:assert/strict';
import test from 'node:test';

import dashboard from '../api/v1/dashboard.js';
import experiments from '../api/v1/experiments.js';
import health from '../api/v1/health.js';
import snapshot from '../publicSnapshot.js';

function invoke(handler, url, method = 'GET') {
  const response = {
    code: 200,
    status(code) { this.code = code; return this; },
    setHeader() { return this; },
    json(body) { this.body = body; return this; },
  };
  handler({ url, method }, response);
  return response;
}

test('public evidence matches the selected recorded result and contains no client rows', () => {
  const result = invoke(dashboard, '/api/v1/dashboard');
  assert.equal(result.code, 200);
  assert.equal(result.body, snapshot);
  assert.equal(snapshot.publication_mode, 'read_only_aggregate_snapshot');
  assert.equal(snapshot.base_sha, '051ce64a7cf0ab999f8aacb81fa405d5fa0257cf');
  assert.equal(snapshot.metrics.validation_clients, 1000);
  assert.equal(snapshot.metrics.macro_f1, snapshot.experiments.find((row) => row.candidate === 'A').metrics.macro_f1);
  assert.match(snapshot.report_sha256, /^[a-f0-9]{64}$/);
  assert.doesNotMatch(JSON.stringify(snapshot), /"client_id"|"transactions"|"submission"/);
  assert.equal(invoke(dashboard, '/api/v1/dashboard', 'POST').code, 405);
});

test('public experiment API paginates the verified report arms', () => {
  const first = invoke(experiments, '/api/v1/experiments?limit=6&offset=0');
  const second = invoke(experiments, '/api/v1/experiments?limit=6&offset=6');
  assert.equal(first.code, 200);
  assert.deepEqual(first.body.experiments, snapshot.experiments.slice(0, 6));
  assert.deepEqual(second.body.experiments, snapshot.experiments.slice(6, 12));
  assert.equal(first.body.comparison_group, snapshot.comparison_group);
  assert.equal(invoke(experiments, '/api/v1/experiments?limit=0').code, 422);
  assert.equal(invoke(experiments, '/api/v1/experiments?offset=-1').code, 422);
  assert.equal(invoke(experiments, '/api/v1/experiments', 'POST').code, 405);
});

test('public healthcheck reports snapshot provenance', () => {
  const result = invoke(health, '/api/v1/health');
  assert.equal(result.code, 200);
  assert.equal(result.body.status, 'ready');
  assert.equal(result.body.report_sha256, snapshot.report_sha256);
});
