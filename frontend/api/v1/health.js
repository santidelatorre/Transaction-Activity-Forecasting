import snapshot from '../../publicSnapshot.js';

export default function handler(request, response) {
  if (request.method && request.method !== 'GET') {
    return response.status(405).json({ detail: 'Method not allowed' });
  }
  return response.status(200).json({
    status: snapshot.available ? 'ready' : 'waiting_for_outputs',
    dashboard_available: snapshot.available,
    dashboard_model_version: snapshot.model_version,
    publication_mode: snapshot.publication_mode,
    report_sha256: snapshot.report_sha256,
  });
}
