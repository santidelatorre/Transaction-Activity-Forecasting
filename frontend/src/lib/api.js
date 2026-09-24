async function request(path) {
  const response = await fetch(path, { headers: { Accept: 'application/json' } });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (body.detail) message = body.detail;
    } catch {
      // Keep the HTTP status message if the response is not JSON.
    }
    throw new Error(message);
  }
  return response.json();
}

export function getHealth() {
  return request('/api/v1/health');
}

export function getOverview({ page = 1, pageSize = 15, query = '' } = {}) {
  const params = new URLSearchParams({ page, page_size: pageSize, q: query });
  return request(`/api/v1/overview?${params}`);
}

export function getClient(clientId) {
  return request(`/api/v1/clients/${encodeURIComponent(clientId)}`);
}

export function getResults() {
  return request('/api/v1/results');
}
