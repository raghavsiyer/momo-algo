// App shell: topbar tabs (Leaderboard / Charts / Replay), sidebar with the
// session ticker list, and per-view content. URL query state wins over server
// defaults so every view deep-links.
import { fetchChart, fetchMeta, fetchTickers } from './api.js';
import { createChartPanel } from './chartPanel.js';
import { createTickerTable } from './tickerTable.js';
import { createSidebar } from './sidebar.js';
import { attachAllpassRibbon } from './ribbon.js';
import { readUrlState, writeUrlState } from './state.js';
import { currentTheme, setTheme } from './theme.js';

const VIEWS = ['leaderboard', 'charts', 'replay'];

const statusEl = document.getElementById('status');
const chartsRoot = document.getElementById('charts');
const leaderboardRoot = document.getElementById('view-leaderboard');

let SESSION_WINDOWS = {};
let TZ_LABEL = '';
let TZ_OFFSET_MINUTES = 0;
let activeView = 'leaderboard';
let lastChartMeta = null;
let replayModule = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle('error', Boolean(isError));
}

function el(id) {
  return document.getElementById(id);
}

// ---------- busy / loading affordances ----------

let busyCount = 0;

function setBusy(active) {
  busyCount = Math.max(0, busyCount + (active ? 1 : -1));
  el('busyBar').hidden = busyCount === 0;
}

function loadingPanel(label) {
  return `<div class="loading-panel"><span class="spinner"></span><span class="loading-label">${label}</span></div>`;
}

const sidebar = createSidebar({
  getSelectedSymbol: () => el('symbol').value.trim().toUpperCase(),
  onOpen: (symbol) => { el('symbol').value = symbol; switchView('charts'); loadChart(); },
});

const tickerTable = createTickerTable({
  root: leaderboardRoot,
  getSelectedSymbol: () => el('symbol').value.trim().toUpperCase(),
  onPick: (symbol) => { el('symbol').value = symbol; sidebar.refreshSelection(); },
  onOpen: (symbol) => { el('symbol').value = symbol; switchView('charts'); loadChart(); },
  setStatus,
});

// ---------- view switching ----------

function switchView(view, options = {}) {
  if (!VIEWS.includes(view)) view = 'leaderboard';
  activeView = view;
  for (const name of VIEWS) {
    el(`view-${name}`).hidden = name !== view;
  }
  for (const button of document.querySelectorAll('#tabs button')) {
    button.classList.toggle('active', button.dataset.view === view);
  }
  if (!options.skipUrl) syncUrl();
  if (view === 'replay' && replayModule) replayModule.onEnter();
}

function updateContext() {
  const symbol = el('symbol').value.trim().toUpperCase();
  const sessionEl = el('session');
  const sessionLabel = sessionEl.options[sessionEl.selectedIndex]?.textContent || '';
  const range = `${el('start').value}–${el('end').value}`;
  const parts = [];
  if (symbol) parts.push(symbol);
  parts.push(el('date').value, `${sessionLabel} ${range} ${TZ_LABEL}`.trim());
  el('context').textContent = parts.join(' · ');
}

function syncUrl() {
  writeUrlState({
    view: activeView !== 'leaderboard' ? activeView : null,
    symbol: el('symbol').value.trim().toUpperCase() || null,
    date: el('date').value,
    session: el('session').value,
    tf: el('timeframe').value !== 'all' ? el('timeframe').value : null,
    start: el('start').value,
    end: el('end').value,
    fetch: el('allowFetch').checked,
    move: el('moveStart').checked,
  });
  updateContext();
}

// ---------- data loading ----------

function applySessionWindow() {
  const windowSpec = SESSION_WINDOWS[el('session').value] || SESSION_WINDOWS.premarket;
  if (!windowSpec) return;
  el('start').value = windowSpec.start;
  el('end').value = windowSpec.end;
}

async function loadTickers() {
  syncUrl();
  setBusy(true);
  setStatus(`Loading ${el('date').value} tickers…`);
  if (!leaderboardRoot.querySelector('.ticker-browser')) {
    leaderboardRoot.innerHTML = loadingPanel(`Loading ${el('date').value} tickers…`);
  }
  try {
    const payload = await fetchTickers({
      date: el('date').value,
      session: el('session').value,
      start: el('start').value,
      end: el('end').value,
    });
    tickerTable.render(payload);
    sidebar.setRows(payload.rows || []);
  } catch (error) {
    leaderboardRoot.innerHTML = `<div class="empty">${error.message}</div>`;
    sidebar.setRows([]);
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

function renderChart(DATA) {
  document.title = `${DATA.symbol} · Momo Visuals`;
  chartsRoot.innerHTML = '';
  const panels = [];
  for (const frame of DATA.frames) {
    const panel = createChartPanel(chartsRoot, DATA, frame, { onReplayFrom: replayFromTime });
    if (panel) panels.push(panel);
  }
  setStatus(`${DATA.windowLabel} | ${DATA.sourceLabel}`);
  lastChartMeta = { symbol: DATA.symbol, windowLabel: DATA.windowLabel };
  loadAllpassRibbon(DATA.symbol, panels);
}

// Click-to-replay from the charts view: the replay window starts at the
// clicked minute and playback seeks to the exact clicked second.
function replayFromTime(localHHMMSS) {
  const symbol = el('symbol').value.trim().toUpperCase();
  const date = el('date').value;
  if (!symbol || !date || !replayModule) return;
  const [hh, mm, ss] = localHHMMSS.split(':').map(Number);
  const start = `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
  let end = el('end').value;
  if (!end || end <= start) {
    const endMinutes = Math.min(23 * 60 + 59, hh * 60 + mm + 30);
    end = `${String(Math.floor(endMinutes / 60)).padStart(2, '0')}:${String(endMinutes % 60).padStart(2, '0')}`;
  }
  const [y, mo, d] = date.split('-').map(Number);
  const seekMs = Date.UTC(y, mo - 1, d, hh, mm, ss) - TZ_OFFSET_MINUTES * 60 * 1000;
  switchView('replay');
  replayModule.startFrom({ symbol, date, start, end, seekMs });
}

async function loadAllpassRibbon(symbol, panels) {
  if (!panels.length) return;
  try {
    const findResponse = await fetch(`/api/replay/find?symbol=${encodeURIComponent(symbol)}&date=${encodeURIComponent(el('date').value)}`);
    if (!findResponse.ok) return;
    const found = await findResponse.json();
    if (!found.bundles || !found.bundles.length) return;
    const allSegments = [];
    for (const bundle of found.bundles) {
      const eventsResponse = await fetch(`/api/replay/${bundle.bundle_id}/events`);
      if (!eventsResponse.ok) continue;
      const events = await eventsResponse.json();
      if (events.allpass) allSegments.push(...events.allpass);
    }
    if (!allSegments.length) return;
    for (const panel of panels) attachAllpassRibbon(panel, allSegments, TZ_OFFSET_MINUTES);
  } catch (error) { /* ribbon is best-effort */ }
}

async function loadChart(event) {
  if (event) event.preventDefault();
  const symbol = el('symbol').value.trim().toUpperCase();
  if (!symbol) { setStatus('Ticker required', true); return; }
  if (activeView !== 'charts') switchView('charts', { skipUrl: true });
  syncUrl();
  sidebar.refreshSelection();
  const params = {
    symbol,
    date: el('date').value,
    session: el('session').value,
    timeframe: el('timeframe').value,
    start: el('start').value,
    end: el('end').value,
  };
  if (el('allowFetch').checked) params.allow_fetch = '1';
  if (!el('moveStart').checked) params.no_move_start = '1';
  setBusy(true);
  setStatus(`Loading ${symbol} ${el('date').value}…`);
  if (!chartsRoot.childElementCount) {
    chartsRoot.innerHTML = loadingPanel(`Loading ${symbol} · ${el('date').value}…`);
  }
  try {
    const payload = await fetchChart(params);
    renderChart(payload);
  } catch (error) {
    chartsRoot.innerHTML = `<div class="empty">${error.message}</div>`;
    setStatus(error.message, true);
  } finally {
    setBusy(false);
  }
}

function reloadActiveData() {
  loadTickers();
  if (el('symbol').value.trim() && activeView === 'charts') loadChart();
}

// ---------- boot ----------

function hydrateControls(meta, urlState) {
  SESSION_WINDOWS = meta.sessionWindows || {};
  TZ_LABEL = meta.tzLabel || '';
  TZ_OFFSET_MINUTES = meta.tzOffsetMinutes || 0;
  for (const chip of document.querySelectorAll('[data-tz]')) chip.textContent = TZ_LABEL;
  const defaults = meta.defaults || {};
  el('symbol').value = urlState.symbol || defaults.symbol || '';
  el('date').value = urlState.date || defaults.date || '';
  if (meta.dateLimits) {
    if (meta.dateLimits.min) el('date').min = meta.dateLimits.min;
    if (meta.dateLimits.max) el('date').max = meta.dateLimits.max;
  }
  el('session').value = urlState.session || defaults.session || 'premarket';
  el('timeframe').value = urlState.tf || defaults.timeframe || 'all';
  applySessionWindow();
  if (urlState.start || defaults.start) el('start').value = urlState.start || defaults.start;
  if (urlState.end || defaults.end) el('end').value = urlState.end || defaults.end;
  el('allowFetch').checked = urlState.fetch === '1' || (!('fetch' in urlState) && Boolean(defaults.allowFetch));
  el('moveStart').checked = urlState.move === '1' || (!('move' in urlState) && Boolean(defaults.moveStart));
  el('defaultTickerSort').value = tickerTable.getDefaultSort();
  sidebar.setSort(tickerTable.getDefaultSort());
}

function wireEvents() {
  for (const button of document.querySelectorAll('#tabs button')) {
    button.addEventListener('click', () => {
      switchView(button.dataset.view);
      if (button.dataset.view === 'charts' && el('symbol').value.trim() && !chartsRoot.childElementCount) loadChart();
    });
  }
  el('collapseSide').addEventListener('click', () => {
    document.getElementById('app').classList.toggle('side-collapsed');
  });
  updateThemeToggleGlyph();
  el('themeToggle').addEventListener('click', () => {
    setTheme(currentTheme() === 'light' ? 'dark' : 'light');
    updateThemeToggleGlyph();
    // DOM/CSS retheme automatically via tokens; canvas + Lightweight Charts
    // bake colors at creation, so re-render what's chart-based.
    if (el('symbol').value.trim() && !el('view-charts').hidden) loadChart();
    else if (el('symbol').value.trim()) chartsRoot.innerHTML = '';
    if (replayModule && replayModule.retheme) replayModule.retheme();
  });
  const form = el('controls');
  form.addEventListener('submit', loadChart);
  form.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || event.shiftKey) return;
    event.preventDefault();
    loadChart(event);
  });
  el('defaultTickerSort').addEventListener('change', (event) => {
    tickerTable.setDefaultSort(event.target.value);
    sidebar.setSort(event.target.value);
  });
  el('symbol').addEventListener('input', () => {
    tickerTable.fillTickerQuickPick();
    tickerTable.updateSelectedTickerRows();
    sidebar.refreshSelection();
  });
  el('session').addEventListener('change', () => { applySessionWindow(); reloadActiveData(); });
  el('date').addEventListener('change', () => reloadActiveData());
  for (const id of ['start', 'end']) el(id).addEventListener('change', () => reloadActiveData());
  for (const id of ['timeframe', 'allowFetch', 'moveStart']) el(id).addEventListener('change', () => {
    if (el('symbol').value.trim()) loadChart();
  });
  el('replayWindowBtn').addEventListener('click', () => {
    switchView('replay');
    if (replayModule) replayModule.prefill(currentReplayContext());
  });
}

function updateThemeToggleGlyph() {
  // Shows the mode a click switches TO.
  el('themeToggle').innerHTML = currentTheme() === 'light' ? '&#9789;' : '&#9788;';
}

function currentReplayContext() {
  return {
    symbol: el('symbol').value.trim().toUpperCase(),
    date: el('date').value,
    start: el('start').value,
    end: el('end').value,
  };
}

async function boot() {
  let meta = { defaults: {}, sessionWindows: {}, dateLimits: null };
  try {
    meta = await fetchMeta();
  } catch (error) {
    setStatus(`Server meta unavailable: ${error.message}`, true);
  }
  const urlState = readUrlState();
  hydrateControls(meta, urlState);
  wireEvents();

  try {
    const module = await import('./replay/index.js');
    replayModule = module.createReplayView({
      root: document.getElementById('replayRoot'),
      setStatus,
      getContext: currentReplayContext,
      tzLabel: () => TZ_LABEL,
    });
  } catch (error) {
    replayModule = null; // replay UI ships in a later milestone
  }

  const view = urlState.view || (el('symbol').value.trim() ? 'charts' : 'leaderboard');
  switchView(view, { skipUrl: true });
  loadTickers();
  if (el('symbol').value.trim() && view === 'charts') loadChart();
  if (view === 'replay' && replayModule) replayModule.restoreFromUrl(urlState);
  updateContext();
}

boot();
