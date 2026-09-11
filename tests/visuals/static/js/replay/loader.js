// Bundle loader: kicks off (or finds) a server-side build, then streams the
// chunk files sequentially into an in-memory record store the player and
// components query. Chunks are 60s of replay time; playback can begin as soon
// as chunk 0 arrives, while later chunks are still being computed.

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).error || detail; } catch (error) {}
    throw new Error(detail);
  }
  return response.json();
}

export class BundleLoader {
  constructor() {
    this.meta = null;
    this.events = null;
    this.status = { state: 'new', chunks_ready: 0 };
    this.bundleId = null;
    // Record store — flat arrays sorted by time (chunks arrive in order).
    this.ticks = [];
    this.books = [];
    this.bars = [];   // {t, tf, bar}
    this.states = [];
    this.fills = [];
    this.loadedChunks = 0;
    this.totalChunks = null; // known once status.state === 'ready'
    this.onProgress = null;  // ({loadedChunks, status})
    this.onRecords = null;   // (newRecordsByKind) — fired per chunk
    this._stopped = false;
  }

  loadedThroughMs() {
    if (!this.meta) return -Infinity;
    return this.meta.window.start_ms + this.loadedChunks * this.meta.chunk_seconds * 1000;
  }

  buildDone() {
    return this.status.state === 'ready' && this.totalChunks !== null && this.loadedChunks >= this.totalChunks;
  }

  stop() { this._stopped = true; }

  async start(params) {
    const query = new URLSearchParams(params).toString();
    const build = await getJson(`/api/replay/build?${query}`);
    this.bundleId = build.bundle_id;
    await this._waitForMeta();
    this._pumpChunks();   // background; do not await
    this._pumpEvents();   // background; resolves when build is ready
    return this.meta;
  }

  async _waitForMeta() {
    for (let attempt = 0; attempt < 600 && !this._stopped; attempt += 1) {
      await this._refreshStatus();
      if (this.status.state === 'error') throw new Error(this.status.error || 'replay build failed');
      try {
        this.meta = await getJson(`/api/replay/${this.bundleId}/meta`);
        return;
      } catch (error) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }
    if (!this.meta) throw new Error('replay build timed out waiting for meta');
  }

  async _refreshStatus() {
    try {
      this.status = await getJson(`/api/replay/${this.bundleId}/status`);
      if (this.status.state === 'ready') this.totalChunks = this.status.chunks_ready;
      if (this.onProgress) this.onProgress(this);
    } catch (error) { /* status file may not exist yet */ }
  }

  async _pumpEvents() {
    while (!this._stopped && this.status.state !== 'ready' && this.status.state !== 'error') {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      await this._refreshStatus();
    }
    if (this._stopped || this.status.state !== 'ready') return;
    try {
      this.events = await getJson(`/api/replay/${this.bundleId}/events`);
      if (this.onProgress) this.onProgress(this);
    } catch (error) { /* optional */ }
  }

  async _pumpChunks() {
    while (!this._stopped) {
      const next = this.loadedChunks;
      if (this.totalChunks !== null && next >= this.totalChunks) return; // done
      if (this.status.state !== 'ready' && next >= (this.status.chunks_ready || 0)) {
        await new Promise((resolve) => setTimeout(resolve, 700));
        await this._refreshStatus();
        continue;
      }
      try {
        const response = await fetch(`/api/replay/${this.bundleId}/chunks/${next}`);
        if (!response.ok) {
          if (this.status.state === 'ready') {
            // Older bundles may lack leading chunks (tape started mid-window);
            // treat a missing chunk inside the known range as empty.
            if (this.totalChunks !== null && next < this.totalChunks) {
              this.loadedChunks = next + 1;
              if (this.onProgress) this.onProgress(this);
              continue;
            }
            return; // past the end
          }
          await new Promise((resolve) => setTimeout(resolve, 700));
          await this._refreshStatus();
          continue;
        }
        const text = await response.text();
        this._ingest(text);
        this.loadedChunks = next + 1;
        if (this.onProgress) this.onProgress(this);
      } catch (error) {
        console.error('replay chunk pump:', error);
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }
  }

  _ingest(ndjson) {
    const fresh = { ticks: [], books: [], bars: [], states: [], fills: [] };
    for (const line of ndjson.split('\n')) {
      if (!line) continue;
      let record;
      try { record = JSON.parse(line); } catch (error) { continue; }
      switch (record.k) {
        case 'tick': fresh.ticks.push(record); break;
        case 'book': fresh.books.push(record); break;
        case 'bar': fresh.bars.push(record); break;
        case 'state': fresh.states.push(record); break;
        case 'fill': fresh.fills.push(record); break;
        default: break; // anchors are redundant with the full record store
      }
    }
    this.ticks.push(...fresh.ticks);
    this.books.push(...fresh.books);
    this.bars.push(...fresh.bars);
    this.states.push(...fresh.states);
    this.fills.push(...fresh.fills);
    if (this.onRecords) this.onRecords(fresh);
  }
}

// Binary search: index of last element with t <= target, or -1.
export function lastIndexAtOrBefore(records, targetMs) {
  let lo = 0;
  let hi = records.length - 1;
  let best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (records[mid].t <= targetMs) { best = mid; lo = mid + 1; }
    else hi = mid - 1;
  }
  return best;
}
