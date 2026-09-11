// Time & sales stream: newest print on top, side-colored, block prints
// emphasized. Rows are pruned so the DOM stays bounded during long replays.
import { fmtValue, fmtNumber, fmtChartTime } from '../util.js';

const MAX_ROWS = 160;
const BATCH_APPEND_CAP = 40; // at high speed only the latest prints matter

export function createTape(container, meta, loader) {
  const shiftSecs = (meta.tz_offset_minutes || 0) * 60;
  const wrap = document.createElement('div');
  wrap.className = 'tape';
  wrap.innerHTML = '<div class="tape-head">Time &amp; Sales</div><div class="tape-rows"></div>';
  container.appendChild(wrap);
  const rowsEl = wrap.querySelector('.tape-rows');

  let sizeEma = 0;

  function rowHtml(tick) {
    sizeEma = sizeEma === 0 ? tick.s : sizeEma + 0.02 * (tick.s - sizeEma);
    const big = sizeEma > 0 && tick.s >= Math.max(sizeEma * 4, 500) ? ' big' : '';
    const sideClass = tick.side === 'B' ? 'buy' : tick.side === 'S' ? 'sell' : 'mid';
    const timeLabel = fmtChartTime(Math.floor(tick.t / 1000) + shiftSecs, true);
    return `<div class="tape-row ${sideClass}${big}">`
      + `<span class="tape-time">${timeLabel}</span>`
      + `<span class="tape-price">${fmtValue(tick.p)}</span>`
      + `<span class="tape-size">${fmtNumber(tick.s)}</span>`
      + '</div>';
  }

  function appendTicks(ticks) {
    if (!ticks.length) return;
    const slice = ticks.length > BATCH_APPEND_CAP ? ticks.slice(-BATCH_APPEND_CAP) : ticks;
    const html = slice.map(rowHtml).reverse().join('');
    rowsEl.insertAdjacentHTML('afterbegin', html);
    while (rowsEl.childElementCount > MAX_ROWS) rowsEl.lastElementChild.remove();
  }

  let pending = [];
  let rafQueued = false;

  function paint() {
    rafQueued = false;
    appendTicks(pending);
    pending = [];
  }

  return {
    onBatch(batch) {
      if (!batch.ticks.length) return;
      pending.push(...batch.ticks);
      if (!rafQueued) { rafQueued = true; requestAnimationFrame(paint); }
    },
    onSeek(snapshot) {
      pending = [];
      sizeEma = 0;
      rowsEl.innerHTML = '';
      const start = Math.max(0, snapshot.ticksBefore - MAX_ROWS);
      appendTicks(loader.ticks.slice(start, snapshot.ticksBefore));
    },
  };
}
