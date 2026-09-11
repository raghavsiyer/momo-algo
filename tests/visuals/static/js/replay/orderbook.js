// L2 ladder: asks stacked above the spread, bids below, depth bars sized to
// the largest level in view, per-level flash on size change. DOM rows — the
// ladder is not time-axis-aligned, so the canvas constraint doesn't apply.
import { fmtValue, fmtNumber } from '../util.js';

export function createOrderBook(container, meta) {
  const wrap = document.createElement('div');
  wrap.className = 'ladder';
  wrap.innerHTML = `
    <div class="ladder-head">Order Book ${meta.has_book ? '' : '<span class="ladder-nodata">no L2 captured</span>'}</div>
    <div class="ladder-rows">
      <div class="ladder-asks"></div>
      <div class="ladder-spread"><span class="ladder-last">--</span><span class="ladder-spread-val"></span></div>
      <div class="ladder-bids"></div>
    </div>`;
  container.appendChild(wrap);
  const asksEl = wrap.querySelector('.ladder-asks');
  const bidsEl = wrap.querySelector('.ladder-bids');
  const lastEl = wrap.querySelector('.ladder-last');
  const spreadEl = wrap.querySelector('.ladder-spread-val');

  let pendingBook = null;
  let pendingTick = null;
  let rafQueued = false;
  const prevSizes = { a: new Map(), b: new Map() };

  function renderSide(root, levels, side, maxSize) {
    // Asks render worst→best so the best ask sits just above the spread row.
    const ordered = side === 'a' ? [...levels].reverse() : levels;
    const previous = prevSizes[side];
    const next = new Map();
    const rows = [];
    for (const [price, size] of ordered) {
      const pct = maxSize > 0 ? Math.min(100, (size / maxSize) * 100) : 0;
      const old = previous.get(price);
      const flash = old !== undefined && old !== size ? ' flash' : '';
      next.set(price, size);
      rows.push(
        `<div class="ladder-row ${side === 'a' ? 'ask' : 'bid'}${flash}">`
        + `<span class="ladder-bar" style="width:${pct}%"></span>`
        + `<span class="ladder-price">${fmtValue(price)}</span>`
        + `<span class="ladder-size">${fmtNumber(size)}</span>`
        + '</div>',
      );
    }
    prevSizes[side] = next;
    root.innerHTML = rows.join('');
  }

  function paint() {
    rafQueued = false;
    if (pendingBook) {
      const { b, a } = pendingBook;
      const maxSize = Math.max(1, ...b.map(([, s]) => s), ...a.map(([, s]) => s));
      renderSide(asksEl, a, 'a', maxSize);
      renderSide(bidsEl, b, 'b', maxSize);
      const bestBid = b.length ? b[0][0] : null;
      const bestAsk = a.length ? a[0][0] : null;
      if (bestBid != null && bestAsk != null) {
        spreadEl.textContent = `spread ${fmtValue(bestAsk - bestBid)}`;
      }
      pendingBook = null;
    }
    if (pendingTick) {
      lastEl.textContent = fmtValue(pendingTick.p);
      lastEl.className = `ladder-last ${pendingTick.side === 'B' ? 'buy' : pendingTick.side === 'S' ? 'sell' : ''}`;
      pendingTick = null;
    }
  }

  function queue() {
    if (!rafQueued) { rafQueued = true; requestAnimationFrame(paint); }
  }

  return {
    onBatch(batch) {
      if (batch.books.length) pendingBook = batch.books[batch.books.length - 1];
      if (batch.ticks.length) pendingTick = batch.ticks[batch.ticks.length - 1];
      if (pendingBook || pendingTick) queue();
    },
    onSeek(snapshot) {
      prevSizes.a = new Map();
      prevSizes.b = new Map();
      pendingBook = snapshot.book;
      pendingTick = snapshot.lastTick;
      if (!snapshot.book) {
        asksEl.innerHTML = '';
        bidsEl.innerHTML = '';
        spreadEl.textContent = '';
      }
      queue();
    },
    setLast(tick) {
      pendingTick = tick;
      queue();
    },
  };
}
