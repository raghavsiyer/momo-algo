// Replay view: setup form → build/load bundle → live playback session with
// chart, order book, tape, criteria panel, timeline, and transport controls.
import { BundleLoader } from './loader.js';
import { Player, SPEEDS } from './player.js';
import { makeResizer, savedSize } from '../resize.js';
import { createReplayChart } from './replayChart.js';
import { createOrderBook } from './orderbook.js';
import { createTape } from './tape.js';
import { createOfiPane } from './ofiPane.js';
import { createCriteriaPanel } from './criteria.js';
import { createTimeline } from './timeline.js';
import { fmtChartTime } from '../util.js';
import { writeUrlState, readUrlState } from '../state.js';

export function createReplayView(hooks) {
  const root = hooks.root;
  let loader = null;
  let player = null;
  let session = null; // {chart, book, tape, criteria, timeline}
  let current = null; // {symbol, date, start, end}

  // ---------------- setup form ----------------

  function renderSetup(prefill = {}) {
    destroySession();
    root.innerHTML = `
      <div class="replay-setup">
        <div class="replay-setup-title">Live Replay</div>
        <p class="meta">Plays a cached window back through the real scoring stack — tick tape, L2 ladder,
        criteria gates, and simulated entries exactly as the algo would have seen them.</p>
        <form id="replaySetupForm" class="toolbar">
          <label>Ticker <input id="rSymbol" value="${prefill.symbol || ''}" placeholder="CELZ" autocomplete="off" /></label>
          <label>Date <input id="rDate" type="date" value="${prefill.date || ''}" /></label>
          <label>Start <input id="rStart" size="5" value="${prefill.start || ''}" /><span class="tz-chip">${hooks.tzLabel()}</span></label>
          <label>End <input id="rEnd" size="5" value="${prefill.end || ''}" /><span class="tz-chip">${hooks.tzLabel()}</span></label>
          <label class="check"><input id="rForce" type="checkbox" /> rebuild</label>
          <button type="submit" class="primary">&#9654; Start replay</button>
        </form>
        <div id="replayPresets" class="replay-presets"></div>
        <div class="build-card" id="replayBuildStatus" hidden>
          <span class="spinner"></span>
          <div class="build-info">
            <div class="build-title" id="buildTitle"></div>
            <div class="build-bar"><span id="buildBarFill" style="width:0%"></span></div>
            <div class="meta" id="buildMeta"></div>
          </div>
        </div>
      </div>`;
    root.querySelector('#replaySetupForm').addEventListener('submit', (event) => {
      event.preventDefault();
      startSession({
        symbol: root.querySelector('#rSymbol').value.trim().toUpperCase(),
        date: root.querySelector('#rDate').value,
        start: root.querySelector('#rStart').value,
        end: root.querySelector('#rEnd').value,
        force: root.querySelector('#rForce').checked ? '1' : '',
      });
    });
    const refreshPresets = () => loadPresets();
    root.querySelector('#rSymbol').addEventListener('change', refreshPresets);
    root.querySelector('#rDate').addEventListener('change', refreshPresets);
    loadPresets();
  }

  async function loadPresets() {
    const presetsEl = root.querySelector('#replayPresets');
    const symbol = root.querySelector('#rSymbol')?.value.trim().toUpperCase();
    const date = root.querySelector('#rDate')?.value;
    if (!presetsEl || !symbol || !date) return;
    presetsEl.innerHTML = '';
    try {
      const response = await fetch(`/api/replay/windows?symbol=${encodeURIComponent(symbol)}&date=${encodeURIComponent(date)}`);
      if (!response.ok) return;
      const payload = await response.json();
      if (!payload.windows || !payload.windows.length) return;
      presetsEl.innerHTML = '<span class="meta">Suggested windows:</span>';
      for (const window of payload.windows) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = `preset-chip ${window.kind}`;
        chip.textContent = `${window.label} (${window.start}–${window.end})`;
        chip.addEventListener('click', () => {
          root.querySelector('#rStart').value = window.start;
          root.querySelector('#rEnd').value = window.end;
        });
        presetsEl.appendChild(chip);
      }
    } catch (error) { /* presets are optional */ }
  }

  function destroySession() {
    if (player) player.pause();
    if (loader) loader.stop();
    if (session && session.chart) session.chart.destroy();
    loader = null;
    player = null;
    session = null;
  }

  // ---------------- live session ----------------

  async function startSession(params, resumeState = {}) {
    if (!params.symbol || !params.date) {
      hooks.setStatus('Replay needs a ticker and date', true);
      return;
    }
    destroySession();
    current = params;
    syncUrl();
    updateBuildCard(`Preparing replay — ${params.symbol} · ${params.date}`, 0, 'requesting build…');
    hooks.setStatus('Preparing replay…');

    loader = new BundleLoader();
    loader.onProgress = () => {
      if (!session) {
        const s = loader.status;
        if (s.state === 'building') {
          updateBuildCard(
            `Building replay — ${params.symbol} · ${params.date}`,
            s.progress_pct || 0,
            `${s.progress_pct || 0}% · ${s.chunks_ready || 0} chunks · scoring cached tape through the live stack`,
          );
        }
      }
      if (session && player) {
        player.checkPendingSeek();
        session.timeline.refresh();
        updateClock(player.t, playerFlags());
      }
    };

    let meta;
    try {
      meta = await loader.start({
        symbol: params.symbol, date: params.date,
        start: params.start || '', end: params.end || '',
        force: params.force || '',
      });
    } catch (error) {
      hooks.setStatus(`Replay build failed: ${error.message}`, true);
      updateBuildCard('Replay build failed', 0, error.message);
      return;
    }

    renderSession(meta);
    // Begin playback as soon as the first chunk lands.
    const waitForFirstChunk = () => {
      if (!loader) return;
      if (loader.loadedChunks > 0) {
        if (resumeState.t) player.seek(Number(resumeState.t));
        if (resumeState.speed && SPEEDS.includes(Number(resumeState.speed))) player.setSpeed(Number(resumeState.speed));
        player.play();
        hooks.setStatus(`Replaying ${meta.symbol} ${meta.day}`);
      } else {
        setTimeout(waitForFirstChunk, 300);
      }
    };
    waitForFirstChunk();
  }

  function playerFlags() {
    return player
      ? { playing: player.playing, buffering: player.buffering, finished: player.finished, speed: player.speed }
      : { playing: false, buffering: false, finished: false, speed: 1 };
  }

  function updateBuildCard(title, pct, metaText) {
    const card = root.querySelector('#replayBuildStatus');
    if (!card) return;
    card.hidden = false;
    card.querySelector('#buildTitle').textContent = title;
    card.querySelector('#buildBarFill').style.width = `${Math.max(0, Math.min(100, pct))}%`;
    card.querySelector('#buildMeta').textContent = metaText || '';
  }

  let clockEl = null;
  let playBtn = null;

  function updateClock(t, flags) {
    if (!clockEl || !loader || !loader.meta) return;
    const shift = (loader.meta.tz_offset_minutes || 0) * 60;
    const label = fmtChartTime(Math.floor(t / 1000) + shift, true);
    const stateTag = flags.buffering ? ' · buffering…'
      : flags.finished ? ' · end'
      : loader.status.state === 'building' ? ` · building ${loader.status.progress_pct || 0}%`
      : '';
    clockEl.textContent = `${label} ${loader.meta.tz_label}${stateTag}`;
    if (playBtn) playBtn.innerHTML = flags.playing ? '&#10074;&#10074;' : '&#9654;';
  }

  function renderSession(meta) {
    root.innerHTML = `
      <div class="replay-session">
        <div class="replay-controls">
          <button type="button" id="rpBack" title="New replay">&#8617;</button>
          <button type="button" id="rpPlay" class="primary play-btn">&#9654;</button>
          <button type="button" id="rpStepBack" title="Back 10s">-10s</button>
          <button type="button" id="rpStepFwd" title="Forward 10s">+10s</button>
          <span class="speed-group" id="rpSpeeds"></span>
          <button type="button" id="rpPrevEvent" title="Previous fill/signal">&#8676;</button>
          <button type="button" id="rpNextEvent" title="Next fill/signal">&#8677;</button>
          <span class="tf-group" id="rpTfs"></span>
          <label class="check live-toggle" title="Fills the live session actually logged that day (hollow rings) vs this replay's fills (solid triangles)"><input type="checkbox" id="rpLiveFills" checked /> live fills</label>
          <span class="replay-clock" id="rpClock"></span>
          <span class="replay-title">${meta.symbol} · ${meta.day}</span>
        </div>
        <div class="replay-grid">
          <div class="replay-main" id="rpChart"><div class="rp-divider-h" id="rpDividerH" title="Drag to resize chart"></div></div>
          <aside class="replay-side"><div class="rp-divider-v" id="rpDividerV" title="Drag to resize panel"></div>
            <div id="rpBook"></div>
            <div id="rpOfi"></div>
            <div id="rpTape"></div>
          </aside>
          <div class="replay-criteria" id="rpCriteria"></div>
          <div class="replay-timeline" id="rpTimeline"></div>
        </div>
      </div>`;

    clockEl = root.querySelector('#rpClock');
    playBtn = root.querySelector('#rpPlay');

    player = new Player(loader);
    const chart = createReplayChart(root.querySelector('#rpChart'), meta, loader, (t) => player.seek(t));
    const book = createOrderBook(root.querySelector('#rpBook'), meta);
    const ofi = createOfiPane(root.querySelector('#rpOfi'), meta, loader, (t) => player.seek(t));
    const tape = createTape(root.querySelector('#rpTape'), meta, loader);
    const criteria = createCriteriaPanel(root.querySelector('#rpCriteria'), meta);
    const timeline = createTimeline(root.querySelector('#rpTimeline'), meta, loader, (t) => player.seek(t));
    session = { chart, book, ofi, tape, criteria, timeline };

    player.subscribe(chart);
    player.subscribe(book);
    player.subscribe(ofi);
    player.subscribe(tape);
    player.subscribe(criteria);
    player.subscribe(timeline);
    player.onClock = (t, flags) => {
      timeline.setPlayhead(t);
      ofi.setPlayhead(t);
      chart.syncLiveFills(t);
      updateClock(t, flags);
    };
    root.querySelector('#rpLiveFills').addEventListener('change', (event) => {
      chart.setLiveFillsVisible(event.target.checked);
    });

    // Transport wiring.
    playBtn.addEventListener('click', () => player.toggle());
    root.querySelector('#rpBack').addEventListener('click', () => { syncUrl(true); renderSetup(current || {}); });
    root.querySelector('#rpStepBack').addEventListener('click', () => player.stepSeconds(-10));
    root.querySelector('#rpStepFwd').addEventListener('click', () => player.stepSeconds(10));

    const speedsEl = root.querySelector('#rpSpeeds');
    for (const speed of SPEEDS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = `${speed}x`;
      btn.classList.toggle('active', speed === player.speed);
      btn.addEventListener('click', () => {
        player.setSpeed(speed);
        for (const other of speedsEl.children) other.classList.toggle('active', other === btn);
      });
      speedsEl.appendChild(btn);
    }

    const tfsEl = root.querySelector('#rpTfs');
    for (const tf of ['10s', '1m', '2m', '5m']) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = tf;
      btn.classList.toggle('active', tf === chart.getTimeframe());
      btn.addEventListener('click', () => {
        chart.setTimeframe(tf);
        for (const other of tfsEl.children) other.classList.toggle('active', other === btn);
      });
      tfsEl.appendChild(btn);
    }

    function eventTimes() {
      const times = [];
      const fills = (loader.events && loader.events.fills) || loader.fills;
      for (const fill of fills) times.push(fill.t);
      if (loader.events && loader.events.signals) for (const s of loader.events.signals) times.push(s.t);
      return times.sort((a, b) => a - b);
    }
    root.querySelector('#rpPrevEvent').addEventListener('click', () => {
      const times = eventTimes().filter((t) => t < player.t - 500);
      if (times.length) player.seek(times[times.length - 1]);
    });
    root.querySelector('#rpNextEvent').addEventListener('click', () => {
      const time = eventTimes().find((t) => t > player.t + 500);
      if (time) player.seek(time);
    });

    // Panel resizing: chart/criteria split (vertical) and side column width
    // (horizontal). Grid rows/columns reflow, so nothing overlaps.
    const grid = root.querySelector('.replay-grid');
    const mainEl = root.querySelector('.replay-main');
    const sideEl = root.querySelector('.replay-side');
    const storedChartH = savedSize('rpChartH', 0);
    if (storedChartH) grid.style.setProperty('--rp-chart-h', `${storedChartH}px`);
    const storedSideW = savedSize('rpSideW', 0);
    if (storedSideW) grid.style.setProperty('--rp-side-w', `${storedSideW}px`);
    makeResizer(root.querySelector('#rpDividerH'), {
      axis: 'y', key: 'rpChartH', min: 220, max: 900,
      get: () => mainEl.getBoundingClientRect().height,
      set: (px) => grid.style.setProperty('--rp-chart-h', `${px}px`),
    });
    makeResizer(root.querySelector('#rpDividerV'), {
      axis: 'x', key: 'rpSideW', min: 220, max: 540, invert: true,
      get: () => sideEl.getBoundingClientRect().width,
      set: (px) => grid.style.setProperty('--rp-side-w', `${px}px`),
    });

    // Keyboard transport: space = play/pause, arrows = step.
    root.tabIndex = -1;
    root.addEventListener('keydown', (event) => {
      if (event.key === ' ') { event.preventDefault(); player.toggle(); }
      else if (event.key === 'ArrowLeft') player.stepSeconds(-10);
      else if (event.key === 'ArrowRight') player.stepSeconds(10);
    });
    root.focus();
  }

  function syncUrl(dropTime = false) {
    if (!current) return;
    const existing = readUrlState();
    writeUrlState({
      ...existing,
      view: 'replay',
      symbol: current.symbol,
      date: current.date,
      from: current.start || null,
      to: current.end || null,
      t: dropTime || !player ? null : Math.round(player.t),
      speed: player ? player.speed : null,
    });
  }

  // ---------------- public API ----------------

  return {
    onEnter() {
      if (!session && !root.querySelector('#replaySetupForm')) renderSetup(hooks.getContext());
    },
    prefill(context) {
      if (session && current && current.symbol === context.symbol && current.date === context.date) return;
      if (context.symbol && context.date) {
        startSession({ symbol: context.symbol, date: context.date, start: context.start, end: context.end });
      } else {
        renderSetup(context);
      }
    },
    // Chart click-to-replay: always starts a fresh session (even for the same
    // symbol/day) and seeks to the exact clicked instant once chunks land.
    startFrom(context) {
      if (!context.symbol || !context.date) {
        renderSetup(context);
        return;
      }
      startSession(
        { symbol: context.symbol, date: context.date, start: context.start, end: context.end },
        { t: context.seekMs },
      );
    },
    restoreFromUrl(urlState) {
      if (urlState.symbol && urlState.date) {
        startSession(
          { symbol: urlState.symbol, date: urlState.date, start: urlState.from || '', end: urlState.to || '' },
          { t: urlState.t, speed: urlState.speed },
        );
      } else {
        renderSetup(hooks.getContext());
      }
    },
    // Light/dark toggle: DOM panes retheme via CSS tokens; the chart and
    // timeline canvas re-apply the refreshed palette here.
    retheme() {
      if (!session) return;
      session.chart.retheme();
      session.timeline.refresh();
      if (session.ofi && session.ofi.refresh) session.ofi.refresh();
    },
  };
}
