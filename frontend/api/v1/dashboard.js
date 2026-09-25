import snapshot from '../../publicSnapshot.js';

export default function handler(request, response) {
  if (request.method && request.method !== 'GET') {
    return response.status(405).json({ detail: 'Method not allowed' });
  }
  response.setHeader?.('Cache-Control', 'public, max-age=0, s-maxage=300');
  return response.status(200).json(snapshot);
}
