import snapshot from '../../publicSnapshot.js';

function integerParameter(raw, fallback, minimum, maximum) {
  const value = raw ?? fallback;
  if (!/^\d+$/.test(String(value))) return null;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= minimum && parsed <= maximum ? parsed : null;
}

export default function handler(request, response) {
  if (request.method && request.method !== 'GET') {
    return response.status(405).json({ detail: 'Method not allowed' });
  }
  const url = new URL(request.url || '/api/v1/experiments', 'http://localhost');
  const limit = integerParameter(request.query?.limit ?? url.searchParams.get('limit'), 100, 1, 100);
  const offset = integerParameter(request.query?.offset ?? url.searchParams.get('offset'), 0, 0, Number.MAX_SAFE_INTEGER);
  if (limit === null || offset === null) {
    return response.status(422).json({ detail: 'Invalid experiment page' });
  }
  response.setHeader?.('Cache-Control', 'public, max-age=0, s-maxage=300');
  return response.status(200).json({
    available: snapshot.experiments.length > 0,
    experiments: snapshot.experiments.slice(offset, offset + limit),
    limit,
    offset,
    comparison_group: snapshot.comparison_group,
  });
}
