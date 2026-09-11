// One chart panel per timeframe: candles + volume + overlays + indicator
// sub-panes + trade markers, all computed in Python and rendered verbatim.
import { COLORS } from './theme.js';
import { fmtPrice, fmtValue, fmtVolume, lineStyle } from './util.js';
import { createTapeCanvasPane } from './tapePane.js';
import { createIndicatorPane } from './indicatorPane.js';
import { createTradeMarkerLayer } from './markers.js';
import { syncCharts } from './sync.js';
import { makeResizer, savedSize } from './resize.js';

// hooks (optional): { onReplayFrom(localHHMMSS) } — enables click-to-replay:
// clicking a bar shows a confirm popover; confirming starts a replay session
// from that wall-clock time.
export function createChartPanel(root, DATA, frame, hooks = {}) {
  const panel = document.createElement('section');
  panel.className = 'panel';
  const head = document.createElement('div');
  head.className = 'panel-head';
  const title = document.createElement('div');
  const fillLabel = frame.counts.fills ? ` | fills=${frame.counts.fills}` : '';
  const sourceCounts = frame.is10s
    ? `tick-built=${frame.counts.tick || 0} | fetched=${frame.counts.polled || 0}`
    : `candles=${frame.counts.total}`;
  title.innerHTML = `<div class="tf-title">${DATA.symbol} ${frame.label}</div><div class="counts">${sourceCounts}${fillLabel}</div>`;
  const legend = document.createElement('div');
  legend.className = 'legend';
  head.appendChild(title);
  head.appendChild(legend);
  panel.appendChild(head);
  if (frame.empty) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = frame.noDataReason || 'No chartable data found for this period.';
    panel.appendChild(empty);
    root.appendChild(panel);
    return;
  }
  const chartWrap = document.createElement('div');
  chartWrap.className = 'chart-wrap';
  const readout = document.createElement('div');
  readout.className = 'value-readout';
  const chartEl = document.createElement('div');
  chartEl.className = 'chart';
  const storedHeight = savedSize(`chartH.${frame.timeframe}`, 0);
  if (storedHeight) chartEl.style.height = `${storedHeight}px`;
  chartWrap.appendChild(readout);
  chartWrap.appendChild(chartEl);
  panel.appendChild(chartWrap);
  // Bottom drag handle: resizes this chart's height in document flow, so
  // indicator panes and the next panel are pushed down, never overlapped.
  const resizeHandle = document.createElement('div');
  resizeHandle.className = 'resize-h';
  resizeHandle.title = 'Drag to resize chart';
  panel.appendChild(resizeHandle);
  makeResizer(resizeHandle, {
    axis: 'y', key: `chartH.${frame.timeframe}`, min: 240, max: 900,
    get: () => chartEl.getBoundingClientRect().height,
    set: (px) => { chartEl.style.height = `${px}px`; },
  });
  root.appendChild(panel);

  const readoutByTime = new Map(frame.readout.map((row) => [row.time, row]));
  const latestReadout = frame.readout.length ? frame.readout[frame.readout.length - 1] : null;
  function setReadout(row) {
    const item = row || latestReadout;
    if (!item) { readout.textContent = ''; return; }
    readout.innerHTML = `<div><span class="muted">O</span> ${fmtValue(item.open)} <span class="muted">H</span> ${fmtValue(item.high)} <span class="muted">L</span> ${fmtValue(item.low)} <span class="muted">C</span> ${fmtValue(item.close)} <span class="muted">V</span> ${fmtVolume(item.volume)}</div><div><span class="ema9">EMA 9</span> ${fmtValue(item.ema_9)} <span class="ema20">EMA 20</span> ${fmtValue(item.ema_20)}</div><div><span class="vwap">VWAP</span> ${fmtValue(item.vwap)}</div><div><span class="ma50">MA (50)</span> ${fmtValue(item.ma_50)}</div>`;
  }
  setReadout(latestReadout);

  const chart = LightweightCharts.createChart(chartEl, {
    autoSize: true,
    layout: { background: { color: COLORS.panel }, textColor: COLORS.text },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: COLORS.grid, scaleMargins: { top: 0.08, bottom: 0.24 } },
    timeScale: { borderColor: COLORS.grid, timeVisible: true, secondsVisible: frame.timeframe === '10s', rightOffset: 4, barSpacing: frame.timeframe === '10s' ? 8 : 10 },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    localization: { priceFormatter: fmtPrice },
  });
  const candles = chart.addCandlestickSeries({ upColor: COLORS.up, downColor: COLORS.down, borderUpColor: COLORS.up, borderDownColor: COLORS.down, wickUpColor: COLORS.up, wickDownColor: COLORS.down, priceLineVisible: false });
  candles.setData(frame.candles);
  const updateTradeMarkers = createTradeMarkerLayer(chartWrap, chart, candles, frame.markers || []);
  const volume = chart.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: 'volume', priceLineVisible: false });
  volume.priceScale().applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } });
  volume.setData(frame.volume);
  const indicatorPanes = [];
  const overlaySeries = {};
  for (const [key, overlay] of Object.entries(frame.overlays)) {
    const series = chart.addLineSeries({ color: overlay.color, lineWidth: overlay.width, lineStyle: lineStyle[overlay.style] ?? 0, priceLineVisible: false, title: overlay.label });
    series.setData(overlay.data);
    overlaySeries[key] = series;
    const item = document.createElement('label');
    item.innerHTML = `<input type="checkbox" checked data-key="${key}" /> <span style="color:${overlay.color}">${overlay.label}</span>`;
    legend.appendChild(item);
  }
  for (const pane of frame.indicators || []) {
    const indicator = (pane.timeMultiplier || 1) !== 1
      ? createTapeCanvasPane(panel, pane, chart, frame)
      : createIndicatorPane(panel, pane);
    indicatorPanes.push(indicator);
    const item = document.createElement('label');
    item.innerHTML = `<input type="checkbox" checked data-indicator="${pane.kind}" /> ${pane.title}`;
    legend.appendChild(item);
  }
  const chartIndicatorPanes = indicatorPanes.filter((indicator) => indicator.chart);
  const syncPanesToMain = syncCharts(chart, chartIndicatorPanes);
  legend.addEventListener('change', (event) => {
    const key = event.target.dataset.key;
    if (key && overlaySeries[key]) overlaySeries[key].applyOptions({ visible: event.target.checked });
    const indicatorKind = event.target.dataset.indicator;
    if (indicatorKind) {
      for (const indicator of indicatorPanes.filter((item) => item.pane.kind === indicatorKind)) {
        indicator.wrap.hidden = !event.target.checked;
        if (event.target.checked) {
          requestAnimationFrame(indicator.redraw || syncPanesToMain);
        }
      }
    }
  });
  chart.subscribeCrosshairMove((param) => {
    const row = param && param.time != null ? readoutByTime.get(param.time) : null;
    setReadout(row || latestReadout);
  });

  if (hooks.onReplayFrom) {
    const popover = document.createElement('div');
    popover.className = 'replay-from-popover';
    popover.hidden = true;
    chartWrap.appendChild(popover);
    const hidePopover = () => { popover.hidden = true; };
    chart.subscribeClick((param) => {
      if (!param || param.time == null || !param.point) { hidePopover(); return; }
      // Chart times are local-shifted epoch seconds, so UTC accessors read
      // as wall-clock; replay windows are minute-granular.
      const clicked = new Date(Number(param.time) * 1000);
      const hh = String(clicked.getUTCHours()).padStart(2, '0');
      const mm = String(clicked.getUTCMinutes()).padStart(2, '0');
      const ss = String(clicked.getUTCSeconds()).padStart(2, '0');
      popover.innerHTML = '';
      const confirmBtn = document.createElement('button');
      confirmBtn.type = 'button';
      confirmBtn.className = 'primary';
      confirmBtn.innerHTML = `&#9654; Replay from ${hh}:${mm}:${ss}`;
      confirmBtn.addEventListener('click', () => {
        hidePopover();
        hooks.onReplayFrom(`${hh}:${mm}:${ss}`);
      });
      const cancelBtn = document.createElement('button');
      cancelBtn.type = 'button';
      cancelBtn.className = 'popover-cancel';
      cancelBtn.textContent = '×';
      cancelBtn.title = 'Cancel';
      cancelBtn.addEventListener('click', hidePopover);
      popover.appendChild(confirmBtn);
      popover.appendChild(cancelBtn);
      const wrapRect = chartWrap.getBoundingClientRect();
      popover.hidden = false;
      const left = Math.max(6, Math.min(param.point.x + 10, wrapRect.width - 190));
      const top = Math.max(6, Math.min(param.point.y - 40, wrapRect.height - 46));
      popover.style.left = `${Math.round(left)}px`;
      popover.style.top = `${Math.round(top)}px`;
    });
    // A stale popover over a panned chart points at the wrong bar — drop it.
    chart.timeScale().subscribeVisibleLogicalRangeChange(hidePopover);
  }

  chart.timeScale().fitContent();
  updateTradeMarkers();
  syncPanesToMain();
  return { chart, wrap: chartWrap, frame };
}
