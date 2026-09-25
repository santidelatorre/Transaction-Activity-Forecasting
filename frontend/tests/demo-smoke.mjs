// Integration test in a DOM emulator against the live API. No screenshots or
// browser-layout claims: jsdom does not implement a visual layout engine.
import assert from 'node:assert/strict';
import { setTimeout as delay } from 'node:timers/promises';
import { JSDOM } from 'jsdom';
import React, { act } from 'react';
import { createRoot } from 'react-dom/client';
import { createServer } from 'vite';

const origin = process.env.DEMO_URL || 'http://127.0.0.1:8000';
const dom = new JSDOM('<div id="root"></div>', { url: origin });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const nativeFetch = globalThis.fetch;
globalThis.fetch = (path, options) => nativeFetch(new URL(path, origin), options);
const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
let root;

async function until(predicate, description) {
  for (let attempt = 0; attempt < 300; attempt += 1) {
    if (predicate()) return;
    await act(async () => delay(100));
  }
  throw new Error(`Timed out: ${description}\n${document.body.textContent}`);
}

async function clickButton(text) {
  const button = [...document.querySelectorAll('button')].find(node => node.textContent.includes(text));
  assert.ok(button, `Button exists: ${text}`);
  await act(async () => button.click());
}

try {
  const { default: App } = await server.ssrLoadModule('/src/App.jsx');
  root = createRoot(document.getElementById('root'));
  await act(async () => root.render(React.createElement(App)));
  await until(() => document.querySelector('h1'), 'clear case renders');
  const cases = await (await nativeFetch(`${origin}/api/v1/cases`)).json();
  const clear = cases.find(item => item.kind === 'clear');
  const ambiguous = cases.find(item => item.kind === 'ambiguous');
  const clearPrediction = await (await nativeFetch(`${origin}/api/v1/clients/${clear.client_id}/prediction`)).json();
  assert.equal(document.querySelector('h1').textContent, clearPrediction.predicted_family);
  assert.ok(document.body.textContent.includes(clearPrediction.base_sha));
  assert.ok(document.body.textContent.includes('90-day horizon'));
  assert.ok(document.body.textContent.includes('Uncalibrated score'));
  await clickButton('Investigate this prediction');
  await until(() => document.body.textContent.includes('Stopped:'), 'clear investigation');
  assert.ok(document.body.textContent.includes('3/8 tools'));

  await clickButton('Ambiguous signal');
  await until(() => document.querySelector('h1') && document.body.textContent.includes(`Client overview · ${ambiguous.client_id}`), 'ambiguous case renders');
  assert.ok(!document.body.textContent.includes('Stopped:'), 'Previous investigation cleared');
  const ambiguousPrediction = await (await nativeFetch(`${origin}/api/v1/clients/${ambiguous.client_id}/prediction`)).json();
  assert.equal(document.querySelector('h1').textContent, ambiguousPrediction.predicted_family);
  await clickButton('Investigate this prediction');
  await until(() => document.body.textContent.includes('Stopped:'), 'ambiguous investigation');
  assert.ok(document.body.textContent.includes('Compare alternatives'));
  assert.ok(document.body.textContent.includes('6/8 tools'));
  assert.ok(document.body.textContent.includes('submission unchanged'));

  await act(async () => root.unmount());
  window.history.replaceState(null, '', '/?client_id=UNKNOWN');
  root = createRoot(document.getElementById('root'));
  await act(async () => root.render(React.createElement(App)));
  await until(() => document.querySelector('[role="alert"]'), 'unknown client error');
  assert.ok(document.body.textContent.includes('Unknown TEST client'));
  assert.equal(document.querySelector('h1'), null);
  console.log('PASS: real API → React → two conditional investigations → unknown-client state');
} finally {
  if (root) await act(async () => root.unmount());
  await server.close();
  dom.window.close();
  globalThis.fetch = nativeFetch;
}
