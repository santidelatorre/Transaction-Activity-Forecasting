import { cp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import snapshot from './publicSnapshot.js';

const output = fileURLToPath(new URL('./.vercel/output/', import.meta.url));
const dist = fileURLToPath(new URL('./dist/', import.meta.url));

if (!snapshot.available || snapshot.publication_mode !== 'read_only_aggregate_snapshot') {
  throw new Error('A measured, aggregate-only snapshot is required');
}
if (!snapshot.report_sha256 || snapshot.metrics?.validation_clients !== 1000) {
  throw new Error('The public snapshot lacks verified VALID provenance');
}
await readFile(new URL('./dist/index.html', import.meta.url));
await mkdir(output, { recursive: true });
// A failed CLI build leaves this non-spec file behind; prebuilt deploy reads it.
await rm(new URL('./.vercel/output/builds.json', import.meta.url), { force: true });
await cp(dist, fileURLToPath(new URL('./.vercel/output/static/', import.meta.url)), {
  recursive: true,
  force: true,
});

for (const name of ['dashboard', 'experiments', 'health']) {
  const directory = new URL(`./.vercel/output/functions/api/v1/${name}.func/`, import.meta.url);
  await mkdir(directory, { recursive: true });
  await rm(new URL('handler.js', directory), { force: true });
  await rm(new URL('snapshot.js', directory), { force: true });
  const source = await readFile(new URL(`./api/v1/${name}.js`, import.meta.url), 'utf8');
  const imported = "import snapshot from '../../publicSnapshot.js';";
  const exported = 'export default function handler';
  if (!source.startsWith(imported) || !source.includes(exported)) {
    throw new Error(`Unexpected handler structure in ${name}`);
  }
  const bundled = source.replace(imported, `const snapshot = ${JSON.stringify(snapshot)};`)
    .replace(exported, 'module.exports = function handler');
  await writeFile(new URL('index.js', directory), bundled);
  await writeFile(new URL('package.json', directory), '{"type":"commonjs"}');
  await writeFile(new URL('.vc-config.json', directory), JSON.stringify({
    runtime: 'nodejs22.x',
    handler: 'index.js',
    launcherType: 'Nodejs',
    shouldAddHelpers: true,
  }));
}

await writeFile(new URL('./.vercel/output/config.json', import.meta.url),
  JSON.stringify({ version: 3, routes: [{ src: '^/$', dest: '/index.html' }] }));
console.log(`Prepared Vercel Build Output API bundle at ${output}`);
