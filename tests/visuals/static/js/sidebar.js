// Compact ticker list in the sidebar: symbol, % gain, volume, and cached-data
// badges (T = tick tape, B = L2 book) so it's obvious before opening replay
// whether the ladder will populate.
import { fmtCompact, fmtPct } from './util.js';

export function createSidebar(hooks) {
  const listEl = document.getElementById('sideTickers');
  let rows = [];
  let sortKey = 'pctGain';

  function render() {
    listEl.innerHTML = '';
    if (!rows.length) {
      listEl.innerHTML = '<div class="side-hint">No tickers for this window.</div>';
      return;
    }
    const sorted = [...rows].sort((a, b) => (Number(b[sortKey] || 0)) - (Number(a[sortKey] || 0)));
    const selected = hooks.getSelectedSymbol();
    for (const item of sorted) {
      const row = document.createElement('div');
      row.className = 'side-row' + (item.symbol === selected ? ' selected' : '');
      row.dataset.symbol = item.symbol;
      const gain = Number(item.pctGain || 0);
      const data = item.data || {};
      row.innerHTML = `
        <span class="sym">${item.symbol}</span>
        <span class="gain ${gain >= 0 ? 'gain-positive' : 'gain-negative'}">${fmtPct(item.pctGain)}</span>
        <span class="badges">
          <i class="dot t ${data.tape ? 'on' : ''}" title="tick tape ${data.tape ? 'cached' : 'missing'}"></i>
          <i class="dot b ${data.book ? 'on' : ''}" title="L2 book ${data.book ? 'cached' : 'missing'}"></i>
        </span>
        <span class="sub">vol ${fmtCompact(item.volume)}</span>`;
      row.addEventListener('click', () => hooks.onOpen(item.symbol));
      listEl.appendChild(row);
    }
  }

  return {
    setRows(newRows) { rows = newRows || []; render(); },
    setSort(key) { sortKey = key === 'volume' ? 'volume' : 'pctGain'; render(); },
    refreshSelection() { render(); },
    getRows() { return rows; },
  };
}
