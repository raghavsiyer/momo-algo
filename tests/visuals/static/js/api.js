// Thin fetch wrappers for the chart server's JSON endpoints.
async function getJson(path, params) {
  const query = params ? `?${new URLSearchParams(params).toString()}` : '';
  const response = await fetch(`${path}${query}`);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

export function fetchTickers(params) {
  return getJson('/api/tickers', params);
}

export function fetchChart(params) {
  return getJson('/api/chart', params);
}

export function fetchMeta() {
  return getJson('/api/meta');
}
