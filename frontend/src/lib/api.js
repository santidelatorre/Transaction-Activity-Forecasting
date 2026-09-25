export async function request(path, { signal, method = 'GET' } = {}) {
  const response = await fetch(`/api/v1${path}`, {
    signal, method, headers: { Accept: 'application/json' },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).detail || message; } catch { /* HTTP status remains */ }
    throw new Error(message);
  }
  return response.json();
}
