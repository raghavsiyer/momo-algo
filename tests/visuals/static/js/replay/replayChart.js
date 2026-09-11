// Live-forming candlestick chart for replay. Closed bars (with server-side
// VWAP/EMA values) come from `bar` records; the in-flight bar is accumulated
// from `tick` records — the only client-side math is o/h/l/c/v accumulation.
import { COLORS } from '../theme.js';
import { fmtPrice, fmtValue, fmtVolume, lineStyle } from '../util.js';
import { lastIndexAtOrBefore } from './loader.js';

const TF_SECONDS = { '10s': 10, '1m': 60, '2m': 120, '5m': 300 };

export function createReplayChart(container, meta, loader, onSeekRequest) {
  const shiftSecs = (meta.tz_offset_minutes || 0) * 60;
  const toChartTime = (epochSecs) => epochSecs + shiftSecs;
  const msToChartTime = (ms) => Math.floor(ms / 1000) + shiftSecs;

  const wrap = document.createElement('div');
  wrap.className = 'replay-chart-wrap';
  const readout = document.createElement('div');
  readout.className = 'value-readout replay-readout';
  const chartEl = document.createElement('div');
  chartEl.className = 'replay-chart';
  const followBtn = document.createElement('button');
  followBtn.type = 'button';
  followBtn.className = 'follow-btn';
  followBtn.textContent = 'Follow';
  followBtn.hidden = true;
  wrap.appendChild(readout);
  wrap.appendChild(chartEl);
  wrap.appendChild(followBtn);
  container.appendChild(wrap);

  const chart = LightweightCharts.createChart(chartEl, {
    autoSize: true,
    layout: { background: { color: COLORS.panel }, textColor: COLORS.text },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: COLORS.grid, scaleMargins: { top: 0.08, bottom: 0.24 } },
    timeScale: { borderColor: COLORS.grid, timeVisible: true, secondsVisible: true, rightOffset: 6, barSpacing: 9 },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    localization: { priceFormatter: fmtPrice },
  });
  const candles = chart.addCandlestickSeries({
    upColor: COLORS.up, downColor: COLORS.down, borderUpColor: COLORS.up, borderDownColor: COLORS.down,
    wickUpColor: COLORS.up, wickDownColor: COLORS.down, priceLineVisible: true,
  });
  const volume = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'volume', priceLineVisible: false });
  volume.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
  const vwapSeries = chart.addLineSeries({ color: cssVar('--vwap'), lineWidth: 2, priceLineVisible: false, title: 'VWAP' });
  const ema9Series = chart.addLineSeries({ color: cssVar('--ema9'), lineWidth: 2, priceLineVisible: false, title: 'EMA 9' });
  const ema20Series = chart.addLineSeries({ color: cssVar('--ema20'), lineWidth: 2, priceLineVisible: false, title: 'EMA 20' });

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  for (const level of meta.resistance_levels || []) {
    if (level.price == null) continue;
    candles.createPriceLine({
      price: level.price, color: 'rgba(248, 113, 113, 0.55)', lineWidth: 1,
      lineStyle: lineStyle.dashed, axisLabelVisible: true,
      title: level.kind ? `R ${level.kind}` : 'R',
    });
  }

  // ---- per-TF state ----
  let activeTf = '10s';
  const closedBars = {};   // tf -> array of chart-format bars
  const overlays = {};     // tf -> {vwap: [], ema9: [], ema20: []}
  const forming = {};      // tf -> {time, open, high, low, close, volume} | null
  for (const tf of Object.keys(TF_SECONDS)) {
    closedBars[tf] = [];
    overlays[tf] = { vwap: [], ema9: [], ema20: [] };
    forming[tf] = null;
    for (const bar of (meta.seed_candles[tf] || [])) {
      const time = toChartTime(bar.t);
      closedBars[tf].push({ time, open: bar.o, high: bar.h, low: bar.l, close: bar.c, volume: bar.v });
      if (bar.vwap != null) overlays[tf].vwap.push({ time, value: bar.vwap });
      if (bar.ema9 != null) overlays[tf].ema9.push({ time, value: bar.ema9 });
      if (bar.ema20 != null) overlays[tf].ema20.push({ time, value: bar.ema20 });
    }
  }

  // ---- fill markers ----
  // Replay fills: solid triangles (same look as the charts view).
  // Logged live-session fills: outlined rings, toggleable — what the algo
  // actually did that day, vs. what this replay's engine does.
  const markerLayer = document.createElement('div');
  markerLayer.className = 'trade-marker-layer';
  wrap.appendChild(markerLayer);
  let markers = [];             // {tMs, price, el, live}
  let liveFillsVisible = true;
  let liveFillCursor = 0;       // count of loader.events.live_fills already added

  function repositionMarkers() {
    const secs = TF_SECONDS[activeTf];
    for (const item of markers) {
      if (item.live && !liveFillsVisible) { item.el.hidden = true; continue; }
      // Anchor to the bar bucket of the CURRENT timeframe so markers stay
      // on their candle when the timeframe changes.
      const time = toChartTime(Math.floor(item.tMs / 1000 / secs) * secs);
      const x = chart.timeScale().timeToCoordinate(time);
      const y = candles.priceToCoordinate(item.price);
      if (x == null || y == null || Number.isNaN(x) || Number.isNaN(y)) { item.el.hidden = true; continue; }
      item.el.hidden = false;
      item.el.style.left = `${Math.round(x)}px`;
      item.el.style.top = `${Math.round(y)}px`;
    }
  }

  function addFillMarker(fill, live = false) {
    const isBuy = fill.side === 'BUY';
    const el = document.createElement('div');
    el.className = live ? `live-marker ${isBuy ? 'buy' : 'sell'}` : `trade-marker ${isBuy ? 'buy' : 'sell'}`;
    const pnl = fill.pnl != null && fill.side === 'SELL' ? ` | pnl ${fill.pnl >= 0 ? '+' : ''}${fill.pnl}` : '';
    const prefix = live ? 'LIVE session: ' : '';
    el.setAttribute('data-tooltip', `${prefix}${fill.side} ${fill.shares} @ ${fmtValue(fill.price)} | ${fill.reason || ''}${pnl}`);
    if (!live) {
      if (isBuy) el.style.borderBottomColor = cssVar('--buy-marker');
      else el.style.borderTopColor = cssVar('--sell-marker');
    }
    markerLayer.appendChild(el);
    markers.push({ tMs: fill.t, price: fill.price, el, live });
    repositionMarkers();
  }

  function setMarkersFromFills(fills) {
    markerLayer.innerHTML = '';
    markers = [];
    liveFillCursor = 0;
    for (const fill of fills) addFillMarker(fill);
  }

  function syncLiveFills(tMs) {
    const liveFills = loader.events && loader.events.live_fills;
    if (!liveFills) return;
    while (liveFillCursor < liveFills.length && liveFills[liveFillCursor].t <= tMs) {
      addFillMarker(liveFills[liveFillCursor], true);
      liveFillCursor += 1;
    }
  }

  chart.timeScale().subscribeVisibleLogicalRangeChange(() => requestAnimationFrame(repositionMarkers));

  // ---- follow / manual pan ----
  let follow = true;
  function setFollow(value) {
    follow = value;
    followBtn.hidden = value;
  }
  chartEl.addEventListener('wheel', () => setFollow(false), { passive: true });
  chartEl.addEventListener('pointerdown', () => setFollow(false));
  followBtn.addEventListener('click', () => { setFollow(true); chart.timeScale().scrollToRealTime(); });

  // Clicking a bar jumps playback to that instant (drag-pans don't fire
  // Lightweight Charts click events, so panning stays safe).
  if (onSeekRequest) {
    chart.subscribeClick((param) => {
      if (!param || param.time == null) return;
      onSeekRequest((Number(param.time) - shiftSecs) * 1000);
    });
  }

  // ---- rendering ----
  function volumeRow(bar) {
    const color = bar.close >= bar.open ? COLORS.up + '99' : COLORS.down + '99';
    return { time: bar.time, value: bar.volume, color };
  }

  function renderAll() {
    const bars = closedBars[activeTf].slice();
    if (forming[activeTf]) bars.push(forming[activeTf]);
    candles.setData(bars.map(({ time, open, high, low, close }) => ({ time, open, high, low, close })));
    volume.setData(bars.map(volumeRow));
    vwapSeries.setData(overlays[activeTf].vwap);
    ema9Series.setData(overlays[activeTf].ema9);
    ema20Series.setData(overlays[activeTf].ema20);
    updateReadout();
    repositionMarkers();
  }

  function updateForming(tf, tick) {
    const secs = TF_SECONDS[tf];
    const bucketSecs = Math.floor(tick.t / 1000 / secs) * secs;
    const time = toChartTime(bucketSecs);
    const cur = forming[tf];
    if (!cur || cur.time !== time) {
      forming[tf] = { time, open: tick.p, high: tick.p, low: tick.p, close: tick.p, volume: tick.s };
    } else {
      cur.high = Math.max(cur.high, tick.p);
      cur.low = Math.min(cur.low, tick.p);
      cur.close = tick.p;
      cur.volume += tick.s;
    }
  }

  function commitBar(record) {
    const tf = record.tf;
    const bar = record.bar;
    const time = toChartTime(Math.floor(bar.t / 1000));
    const row = { time, open: bar.o, high: bar.h, low: bar.l, close: bar.c, volume: bar.v };
    closedBars[tf].push(row);
    if (bar.vwap != null) overlays[tf].vwap.push({ time, value: bar.vwap });
    if (bar.ema9 != null) overlays[tf].ema9.push({ time, value: bar.ema9 });
    if (bar.ema20 != null) overlays[tf].ema20.push({ time, value: bar.ema20 });
    if (forming[tf] && forming[tf].time === time) forming[tf] = null;
    if (tf !== activeTf) return;
    candles.update({ time, open: bar.o, high: bar.h, low: bar.l, close: bar.c });
    volume.update(volumeRow(row));
    if (bar.vwap != null) vwapSeries.update({ time, value: bar.vwap });
    if (bar.ema9 != null) ema9Series.update({ time, value: bar.ema9 });
    if (bar.ema20 != null) ema20Series.update({ time, value: bar.ema20 });
  }

  let lastPrice = null;
  let hoveredTime = null; // crosshair-pinned bar time, null = follow latest

  // Last overlay value at or before a bar time (arrays are time-sorted).
  function overlayValueAt(points, time) {
    let lo = 0;
    let hi = points.length - 1;
    let best = null;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (points[mid].time <= time) { best = points[mid].value; lo = mid + 1; }
      else hi = mid - 1;
    }
    return best;
  }

  function barAtTime(tf, time) {
    if (forming[tf] && forming[tf].time === time) return forming[tf];
    const bars = closedBars[tf];
    let lo = 0;
    let hi = bars.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (bars[mid].time === time) return bars[mid];
      if (bars[mid].time < time) lo = mid + 1;
      else hi = mid - 1;
    }
    return null;
  }

  function updateReadout() {
    const tf = activeTf;
    const bar = (hoveredTime !== null ? barAtTime(tf, hoveredTime) : null)
      || forming[tf]
      || closedBars[tf][closedBars[tf].length - 1];
    if (!bar) { readout.textContent = ''; return; }
    const vwap = overlayValueAt(overlays[tf].vwap, bar.time);
    const ema9 = overlayValueAt(overlays[tf].ema9, bar.time);
    const ema20 = overlayValueAt(overlays[tf].ema20, bar.time);
    readout.innerHTML = `<div><span class="muted">O</span> ${fmtValue(bar.open)} <span class="muted">H</span> ${fmtValue(bar.high)} <span class="muted">L</span> ${fmtValue(bar.low)} <span class="muted">C</span> ${fmtValue(bar.close)} <span class="muted">V</span> ${fmtVolume(bar.volume)}</div>`
      + `<div><span class="ema9">EMA 9</span> ${fmtValue(ema9)} <span class="ema20">EMA 20</span> ${fmtValue(ema20)}</div>`
      + `<div><span class="vwap">VWAP</span> ${fmtValue(vwap)}</div>`;
  }

  chart.subscribeCrosshairMove((param) => {
    hoveredTime = param && param.time != null ? Number(param.time) : null;
    updateReadout();
  });

  renderAll();
  chart.timeScale().fitContent();

  return {
    setTimeframe(tf) {
      if (!(tf in TF_SECONDS)) return;
      activeTf = tf;
      renderAll();
      if (follow) chart.timeScale().scrollToRealTime();
    },
    getTimeframe() { return activeTf; },

    // Re-apply palette after a light/dark toggle (COLORS was refreshed).
    retheme() {
      chart.applyOptions({
        layout: { background: { color: COLORS.panel }, textColor: COLORS.text },
        grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
        rightPriceScale: { borderColor: COLORS.grid },
        timeScale: { borderColor: COLORS.grid },
      });
      candles.applyOptions({
        upColor: COLORS.up, downColor: COLORS.down, borderUpColor: COLORS.up,
        borderDownColor: COLORS.down, wickUpColor: COLORS.up, wickDownColor: COLORS.down,
      });
      vwapSeries.applyOptions({ color: cssVar('--vwap') });
      ema9Series.applyOptions({ color: cssVar('--ema9') });
      ema20Series.applyOptions({ color: cssVar('--ema20') });
      renderAll(); // rebuilds per-bar volume colors
    },

    onBatch(batch) {
      for (const record of batch.bars) commitBar(record);
      for (const tick of batch.ticks) {
        lastPrice = tick.p;
        for (const tf of Object.keys(TF_SECONDS)) updateForming(tf, tick);
      }
      if (batch.ticks.length && forming[activeTf]) {
        const bar = forming[activeTf];
        candles.update({ time: bar.time, open: bar.open, high: bar.high, low: bar.low, close: bar.close });
        volume.update(volumeRow(bar));
      }
      for (const fill of batch.fills) addFillMarker(fill);
      syncLiveFills(batch.t);
      if (batch.ticks.length) updateReadout();
      if (follow && (batch.bars.length || batch.ticks.length)) chart.timeScale().scrollToRealTime();
    },

    // Called from the player clock so live fills appear on time even during
    // quiet stretches, and once events.json lands after the build finishes.
    syncLiveFills,

    setLiveFillsVisible(visible) {
      liveFillsVisible = Boolean(visible);
      repositionMarkers();
    },

    onSeek(snapshot) {
      // Rebuild every TF's closed bars + forming state from the record store.
      for (const tf of Object.keys(TF_SECONDS)) {
        closedBars[tf] = [];
        overlays[tf] = { vwap: [], ema9: [], ema20: [] };
        forming[tf] = null;
        for (const bar of (meta.seed_candles[tf] || [])) {
          const time = toChartTime(bar.t);
          closedBars[tf].push({ time, open: bar.o, high: bar.h, low: bar.l, close: bar.c, volume: bar.v });
          if (bar.vwap != null) overlays[tf].vwap.push({ time, value: bar.vwap });
          if (bar.ema9 != null) overlays[tf].ema9.push({ time, value: bar.ema9 });
          if (bar.ema20 != null) overlays[tf].ema20.push({ time, value: bar.ema20 });
        }
      }
      for (let index = 0; index < snapshot.barsBefore; index += 1) {
        const record = loader.bars[index];
        const bar = record.bar;
        const time = toChartTime(Math.floor(bar.t / 1000));
        closedBars[record.tf].push({ time, open: bar.o, high: bar.h, low: bar.l, close: bar.c, volume: bar.v });
        if (bar.vwap != null) overlays[record.tf].vwap.push({ time, value: bar.vwap });
        if (bar.ema9 != null) overlays[record.tf].ema9.push({ time, value: bar.ema9 });
        if (bar.ema20 != null) overlays[record.tf].ema20.push({ time, value: bar.ema20 });
      }
      for (const tf of Object.keys(TF_SECONDS)) {
        const secs = TF_SECONDS[tf];
        const bucketStart = Math.floor(snapshot.t / 1000 / secs) * secs * 1000;
        forming[tf] = null;
        // Forming bar = ticks inside the bucket that is open at snapshot.t.
        const startIdx = lastIndexAtOrBefore(loader.ticks, bucketStart - 1) + 1;
        for (let index = startIdx; index < snapshot.ticksBefore; index += 1) {
          updateForming(tf, loader.ticks[index]);
        }
      }
      setMarkersFromFills(loader.fills.slice(0, snapshot.fillsBefore));
      syncLiveFills(snapshot.t);
      renderAll();
      if (follow) chart.timeScale().scrollToRealTime();
    },

    destroy() { chart.remove(); wrap.remove(); },
  };
}
