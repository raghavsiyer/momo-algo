// URL query-string state for deep links. Every loaded view mirrors its inputs
// into the address bar via history.replaceState so any state can be shared.
export function readUrlState() {
  const params = new URLSearchParams(window.location.search);
  const state = {};
  for (const [key, value] of params.entries()) state[key] = value;
  return state;
}

export function writeUrlState(state) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(state)) {
    if (value === null || value === undefined || value === '' || value === false) continue;
    params.set(key, value === true ? '1' : String(value));
  }
  const query = params.toString();
  const url = query ? `${window.location.pathname}?${query}` : window.location.pathname;
  window.history.replaceState(null, '', url);
}
