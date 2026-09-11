// The execution-criteria panel: every ATTENTION and EXECUTION criterion as a
// meter row (live value vs threshold), the entry chain as a left-to-right
// pipeline in evaluate_entry order, both confidences vs their 50 floor, and
// the master ENTRY ALLOWED lamp with current block reasons.
import { fmtValue } from '../util.js';

function fmtGateValue(value, unit) {
  if (value === null || value === undefined) return '--';
  if (unit === '%') return `${Number(value).toFixed(1)}%`;
  if (Math.abs(value) >= 10000) return `${(value / 1000).toFixed(0)}k`;
  if (Math.abs(value) >= 100) return Number(value).toFixed(0);
  return Number(value).toFixed(2);
}

// Meter range for >=-gates runs from the sell level (when negative) up to the
// entry threshold, so "how far from the exit trigger" is visible, not just
// "how far from zero". A sell level above zero keeps the bottom bound at 0.
function meterRange(def) {
  const hi = Number(def.threshold);
  const lo = def.sell !== null && def.sell !== undefined && Number(def.sell) < 0 ? Number(def.sell) : 0;
  return { lo, hi };
}

function meterFill(value, def) {
  if (value === null || value === undefined || def.threshold === null || def.threshold === undefined) return 0;
  const v = Number(value);
  if (def.direction === '<=') {
    const t = Number(def.threshold);
    if (v <= 0) return 1;
    if (t === 0) return v <= 0 ? 1 : 0;
    return Math.max(0, Math.min(1, t / v));
  }
  // '>' / '>=' : position within [sell-or-zero, entry threshold].
  const { lo, hi } = meterRange(def);
  if (hi === lo) return v > hi ? 1 : 0;
  return Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
}

// Three-state coloring: green = entry criterion passing; red = the
// evaluate_exit sell level is breached; yellow = failing entry but not in
// sell territory (or no sell rule exists for this variable).
function gateStateClass(passing, value, def) {
  if (passing) return 'pass';
  if (def.sell !== null && def.sell !== undefined && value !== null && value !== undefined) {
    const v = Number(value);
    const sell = Number(def.sell);
    const breached = def.direction === '<=' ? v >= sell : v <= sell;
    if (breached) return 'sell';
  }
  return 'warn';
}

export function createCriteriaPanel(container, meta) {
  const defs = meta.criteria_defs;
  const wrap = document.createElement('div');
  wrap.className = 'criteria';
  container.appendChild(wrap);

  if (!meta.criteria_available) {
    wrap.innerHTML = '<div class="criteria-unavailable">This symbol-day is outside the cached evaluation universe — '
      + 'chart, order book, and tape replay only. Criteria states can\'t be reproduced authentically.</div>';
    return { onBatch() {}, onSeek() {} };
  }

  const gateRows = new Map();   // name -> {rowEl, fillEl, valueEl, def}
  const chainNodes = new Map(); // name -> {nodeEl, def}

  function buildGroup(title, groupDefs, keyPrefix) {
    const section = document.createElement('div');
    section.className = 'criteria-group';
    section.innerHTML = `<div class="criteria-group-title">${title}</div>`;
    for (const def of groupDefs) {
      const row = document.createElement('div');
      row.className = 'gate-row' + (def.required === false ? ' advisory' : '');
      row.title = def.description || '';
      row.innerHTML = `
        <span class="gate-dot"></span>
        <span class="gate-name">${def.name.replace(/_/g, ' ')}</span>
        <span class="gate-meter"><span class="gate-fill"></span><span class="gate-sell-tick"></span></span>
        <span class="gate-value">--</span>`;
      const sellTick = row.querySelector('.gate-sell-tick');
      if (def.sell !== null && def.sell !== undefined && def.direction !== '<=' && def.threshold) {
        const { lo, hi } = meterRange(def);
        const pct = hi === lo ? 0 : Math.max(0, Math.min(100, ((Number(def.sell) - lo) / (hi - lo)) * 100));
        sellTick.style.left = `${pct}%`;
        sellTick.title = `sell at ${def.sell}`;
      } else {
        sellTick.hidden = true;
      }
      section.appendChild(row);
      gateRows.set(keyPrefix + def.name, {
        rowEl: row,
        fillEl: row.querySelector('.gate-fill'),
        valueEl: row.querySelector('.gate-value'),
        def,
      });
    }
    return section;
  }

  // Header first: the master lamp + confidences must never scroll out of view.
  const header = document.createElement('div');
  header.className = 'criteria-header';
  header.innerHTML = `
    <div class="allowed-block">
      <div class="allowed-lamp" id="allowedLamp"><span class="lamp-dot"></span> ENTRY ALLOWED</div>
      <div class="block-reasons" id="blockReasons"></div>
    </div>
    <div class="conf-block">
      <div class="conf-row"><span class="conf-label">eval</span><span class="conf-meter"><span class="conf-fill" id="confEvalFill"></span><span class="conf-thr" style="left:${defs.confidence.eval_min}%"></span></span><span class="conf-value" id="confEvalVal">--</span></div>
      <div class="conf-row"><span class="conf-label">live</span><span class="conf-meter"><span class="conf-fill" id="confLiveFill"></span><span class="conf-thr" style="left:${defs.confidence.live_min}%"></span></span><span class="conf-value" id="confLiveVal">--</span></div>
      <div class="conf-row"><span class="conf-label">setup</span><span class="conf-meter"><span class="conf-fill" id="confSetupFill"></span><span class="conf-thr" style="left:40%"></span></span><span class="conf-value" id="confSetupVal">--</span></div>
    </div>`;
  wrap.appendChild(header);

  const columns = document.createElement('div');
  columns.className = 'criteria-columns';
  columns.appendChild(buildGroup('Attention', defs.attention, 'att:'));
  columns.appendChild(buildGroup('Execution', defs.execution, 'exe:'));
  wrap.appendChild(columns);

  // ---- entry chain pipeline ----
  const chainWrap = document.createElement('div');
  chainWrap.className = 'chain';
  chainWrap.innerHTML = '<div class="criteria-group-title">Entry chain <span class="chain-note">evaluate_entry order</span></div>';
  const chainRow = document.createElement('div');
  chainRow.className = 'chain-row';
  for (const def of defs.chain) {
    const node = document.createElement('div');
    node.className = 'chain-node';
    node.innerHTML = `<span class="chain-dot"></span><span class="chain-label">${def.label}</span><span class="chain-value"></span>`;
    chainRow.appendChild(node);
    chainNodes.set(def.name, { nodeEl: node, def });
  }
  chainWrap.appendChild(chainRow);
  wrap.appendChild(chainWrap);

  const confEvalFill = header.querySelector('#confEvalFill');
  const confLiveFill = header.querySelector('#confLiveFill');
  const confSetupFill = header.querySelector('#confSetupFill');
  const confEvalVal = header.querySelector('#confEvalVal');
  const confLiveVal = header.querySelector('#confLiveVal');
  const confSetupVal = header.querySelector('#confSetupVal');
  const allowedLamp = header.querySelector('#allowedLamp');
  const blockReasons = header.querySelector('#blockReasons');

  function applyGateRow(key, passing, value, active = true) {
    const entry = gateRows.get(key);
    if (!entry) return;
    const { rowEl, fillEl, valueEl, def } = entry;
    const state = active ? gateStateClass(Boolean(passing), value, def) : null;
    rowEl.classList.toggle('pass', state === 'pass');
    rowEl.classList.toggle('warn', state === 'warn');
    rowEl.classList.toggle('sell', state === 'sell');
    rowEl.classList.toggle('inactive', !active);
    fillEl.style.width = `${Math.round(meterFill(value, def) * 100)}%`;
    const threshold = def.threshold !== null && def.threshold !== undefined
      ? ` / ${fmtGateValue(def.threshold, def.unit)}${def.unit && def.unit !== '%' ? def.unit : ''}`
      : '';
    valueEl.textContent = `${fmtGateValue(value, def.unit)}${threshold}`;
  }

  function applyState(state) {
    if (!state) return;
    for (const [name, passing, value] of state.att || []) applyGateRow(`att:${name}`, passing === 1, value);
    for (const [name, passing, value] of state.exe || []) applyGateRow(`exe:${name}`, passing === 1, value);
    for (const row of state.chain || []) {
      const [name, passing, value, active, detail] = row;
      const node = chainNodes.get(name);
      if (!node) continue;
      node.nodeEl.classList.toggle('pass', passing === 1 && active === 1);
      node.nodeEl.classList.toggle('fail', passing !== 1 && active === 1);
      node.nodeEl.classList.toggle('inactive', active !== 1);
      const valueEl = node.nodeEl.querySelector('.chain-value');
      valueEl.textContent = active !== 1
        ? (detail || 'n/a')
        : (node.def.direction === 'bool' ? (passing === 1 ? 'yes' : 'no') : fmtGateValue(value, node.def.unit));
      node.nodeEl.title = detail || node.def.label;
    }
    const evalConf = state.eval ?? 0;
    const liveConf = state.live ?? 0;
    const setupScore = state.setup ?? 0;
    const exitMin = defs.confidence.exit_min ?? 40;
    confEvalFill.style.width = `${Math.min(100, evalConf)}%`;
    confLiveFill.style.width = `${Math.min(100, liveConf)}%`;
    confSetupFill.style.width = `${Math.min(100, setupScore)}%`;
    const confState = (value, entryMin, hasExit) => {
      if (value >= entryMin) return 'ok';
      if (hasExit && value < exitMin) return 'sell';
      return 'warn';
    };
    for (const [el, value, entryMin, hasExit] of [
      [confEvalFill, evalConf, defs.confidence.eval_min, true],
      [confLiveFill, liveConf, defs.confidence.live_min, true],
      [confSetupFill, setupScore, 40, false],
    ]) {
      const cls = confState(value, entryMin, hasExit);
      el.classList.toggle('ok', cls === 'ok');
      el.classList.toggle('warn', cls === 'warn');
      el.classList.toggle('sell', cls === 'sell');
    }
    confEvalVal.textContent = evalConf.toFixed(1);
    confLiveVal.textContent = liveConf.toFixed(1);
    confSetupVal.textContent = state.setupTf ? `${setupScore.toFixed(0)} (${state.setupTf})` : setupScore.toFixed(0);
    allowedLamp.classList.toggle('on', state.allowed === 1);
    allowedLamp.classList.toggle('armed', state.chainPass === 1);
    blockReasons.textContent = state.allowed === 1
      ? (state.chainPass === 1 ? 'all gates aligned' : '')
      : (state.block || []).join(' · ');
  }

  let pendingState = null;
  let rafQueued = false;

  function paint() {
    rafQueued = false;
    applyState(pendingState);
    pendingState = null;
  }

  return {
    onBatch(batch) {
      if (!batch.states.length) return;
      pendingState = batch.states[batch.states.length - 1];
      if (!rafQueued) { rafQueued = true; requestAnimationFrame(paint); }
    },
    onSeek(snapshot) {
      pendingState = null;
      rafQueued = false;
      applyState(snapshot.state);
    },
  };
}
