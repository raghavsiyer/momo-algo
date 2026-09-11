// Session ticker leaderboard: sortable table + top-10 quick-pick datalist.
import { fmtNumber, fmtPct, fmtValue } from './util.js';

const DEFAULT_SORT_STORAGE_KEY = 'momoAlgoCharts.defaultTickerSort';

export function loadDefaultTickerSort() {
  try {
    const stored = window.localStorage.getItem(DEFAULT_SORT_STORAGE_KEY);
    if (stored === 'volume' || stored === 'pctGain') return stored;
  } catch (error) {}
  return 'pctGain';
}

function fmtTablePrice(value) {
  if (value === null || value === undefined || value === '') return '-';
  return fmtValue(Number(value));
}

function tickerSortValue(row, key) {
  if (key === 'symbol') return row.symbol || '';
  if (row[key] === null || row[key] === undefined || row[key] === '') return null;
  return Number(row[key]);
}

function sortTickerRows(rows, sort) {
  const direction = sort.dir || 'desc';
  const sortedRows = [...rows];
  sortedRows.sort((left, right) => {
    const leftValue = tickerSortValue(left, sort.key);
    const rightValue = tickerSortValue(right, sort.key);
    let result = 0;
    if (typeof leftValue === 'string') result = leftValue.localeCompare(rightValue);
    else if (leftValue === null && rightValue === null) result = (left.symbol || '').localeCompare(right.symbol || '');
    else if (leftValue === null) return 1;
    else if (rightValue === null) return -1;
    else result = leftValue === rightValue ? (left.symbol || '').localeCompare(right.symbol || '') : leftValue - rightValue;
    return direction === 'asc' ? result : -result;
  });
  return sortedRows;
}

// hooks: { root, getSelectedSymbol(), onPick(symbol), onOpen(symbol), setStatus(msg, isError) }
export function createTickerTable(hooks) {
  let tickerRows = [];
  let defaultTickerSort = loadDefaultTickerSort();
  let tickerSort = { key: defaultTickerSort, dir: 'desc' };

  function sortedTickerRows() {
    return sortTickerRows(tickerRows, tickerSort);
  }

  function defaultSortedTickerRows() {
    return sortTickerRows(tickerRows, { key: defaultTickerSort, dir: 'desc' });
  }

  function fillTickerQuickPick() {
    const list = document.getElementById('tickerQuickPick');
    if (!list) return;
    list.innerHTML = '';
    for (const item of defaultSortedTickerRows().slice(0, 10)) {
      const option = document.createElement('option');
      option.value = item.symbol;
      list.appendChild(option);
    }
  }

  function updateSelectedTickerRows() {
    const selected = hooks.getSelectedSymbol();
    for (const row of hooks.root.querySelectorAll('tr[data-symbol]')) {
      row.classList.toggle('selected', Boolean(selected) && row.dataset.symbol === selected);
    }
  }

  function fillTickerRows() {
    const body = document.getElementById('tickerRows');
    if (!body) return;
    body.innerHTML = '';
    for (const item of sortedTickerRows()) {
      const row = document.createElement('tr');
      row.dataset.symbol = item.symbol;
      const gainClass = Number(item.pctGain || 0) >= 0 ? 'gain-positive' : 'gain-negative';
      const data = item.data || {};
      const badges = '<span class="badges">'
        + `<i class="dot t ${data.tape ? 'on' : ''}" title="tick tape ${data.tape ? 'cached' : 'missing'}"></i>`
        + `<i class="dot b ${data.book ? 'on' : ''}" title="L2 book ${data.book ? 'cached' : 'missing'}"></i>`
        + '</span>';
      row.innerHTML = '<td><span class="ticker-symbol">' + item.symbol + '</span>' + badges + '</td>'
        + '<td>' + fmtNumber(item.volume) + '</td>'
        + '<td class="' + gainClass + '">' + fmtPct(item.pctGain) + '</td>'
        + '<td>' + fmtTablePrice(item.first) + '</td>'
        + '<td>' + fmtTablePrice(item.last) + '</td>'
        + '<td>' + fmtTablePrice(item.high) + '</td>'
        + '<td>' + fmtNumber(item.prints) + '</td>'
        + '<td>' + item.firstTime + '</td>'
        + '<td>' + item.lastTime + '</td>';
      body.appendChild(row);
    }
    updateSelectedTickerRows();
  }

  function render(payload) {
    tickerRows = payload.rows || [];
    hooks.root.innerHTML = '<section class="ticker-browser">'
      + '<div class="ticker-browser-head"><div><div class="ticker-browser-title">' + payload.sessionLabel + ' Tickers</div><div class="meta" id="tickerHint">Click a ticker to fill the symbol box. Press Load to chart it.</div></div><div class="ticker-browser-meta" id="tickerMeta"></div></div>'
      + '<table class="ticker-table"><thead><tr>'
      + '<th><button type="button" data-sort="symbol">Ticker</button></th>'
      + '<th><button type="button" data-sort="volume">' + payload.volumeLabel + '</button></th>'
      + '<th><button type="button" data-sort="pctGain">' + payload.gainLabel + '</button></th>'
      + '<th>Open</th><th>Last</th><th>High</th><th>Prints</th><th>First</th><th>Last Tick</th>'
      + '</tr></thead><tbody id="tickerRows"></tbody></table>'
      + '</section>';
    document.getElementById('tickerMeta').textContent = payload.windowLabel + ' | ' + fmtNumber(payload.counts.symbols) + ' symbols | ' + fmtNumber(payload.counts.points) + ' points';
    for (const button of hooks.root.querySelectorAll('[data-sort]')) {
      button.addEventListener('click', () => {
        const key = button.dataset.sort;
        if (tickerSort.key === key) tickerSort.dir = tickerSort.dir === 'asc' ? 'desc' : 'asc';
        else tickerSort = { key, dir: key === 'symbol' ? 'asc' : 'desc' };
        fillTickerRows();
      });
    }
    hooks.root.querySelector('#tickerRows').addEventListener('click', (event) => {
      const row = event.target.closest('tr[data-symbol]');
      if (!row) return;
      hooks.onPick(row.dataset.symbol);
      fillTickerQuickPick();
      updateSelectedTickerRows();
      hooks.setStatus(row.dataset.symbol + ' selected. Press Load to chart it.');
    });
    hooks.root.querySelector('#tickerRows').addEventListener('dblclick', (event) => {
      const row = event.target.closest('tr[data-symbol]');
      if (!row) return;
      hooks.onPick(row.dataset.symbol);
      fillTickerQuickPick();
      updateSelectedTickerRows();
      hooks.onOpen(row.dataset.symbol);
    });
    fillTickerRows();
    fillTickerQuickPick();
    if (!tickerRows.length) hooks.setStatus('No ' + payload.sessionLabel.toLowerCase() + ' tickers found for that window.', true);
    else hooks.setStatus(payload.sessionLabel + ' tickers loaded.');
  }

  function setDefaultSort(value) {
    const key = value === 'volume' ? 'volume' : 'pctGain';
    defaultTickerSort = key;
    try { window.localStorage.setItem(DEFAULT_SORT_STORAGE_KEY, key); } catch (error) {}
    tickerSort = { key, dir: 'desc' };
    fillTickerRows();
    fillTickerQuickPick();
  }

  return { render, setDefaultSort, fillTickerQuickPick, updateSelectedTickerRows, getDefaultSort: () => defaultTickerSort };
}
