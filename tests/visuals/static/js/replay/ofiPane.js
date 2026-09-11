// Live OFI flow pane: the depth-normalized OFI gate read (ofi_norm_gate —
// the exact value evaluate_entry/evaluate_exit enforce) as a rolling
// sparkline with the entry floor and the adverse-flow exit level drawn as
// guides, plus a live value chip and the exit's ratio-lapse arming state.
//
// Data comes from the bundle's state records (chain row "ofi_norm" carries
// the gate value; "tape_imbalance" carries the ratio_gate the exit
// conditions on). Depth-inactive stretches render as gaps — the gate is
// vacuous there, so drawing a line would be a lie.
import { COLORS } from '../theme.js';
import { fmtChartTime } from '../util.js';
import { lastIndexAtOrBefore } from './loader.js';

const WINDOW_MS = 120_000;   // sparkline lookback
const PAD_TOP = 5;
const PAD_BOTTOM = 5;
const GUTTER_RIGHT = 34;     // threshold labels

function chainValues(state) {
  let v = null;
  let pass = false;
  let ratio = null;
  for (const row of state.chain || []) {
    if (row[0] === 'ofi_norm') {
      if (row[3] === 1) { v = row[2]; pass = row[1] === 1; }
    } else if (row[0] === 'tape_imbalance') {
      ratio = row[2];
    }
  }
  return { t: state.t, v, pass, ratio };
}

export function createOfiPane(container, meta, loader, onSeekRequest) {
  const wrap = document.createElement('div');
  wrap.className = 'ofi-pane';
  container.appendChild(wrap);

  const chainDefs = (meta.criteria_defs && meta.criteria_defs.chain) || [];
  const ofiDef = chainDefs.find((d) => d.name === 'ofi_norm') || {};
  const exeDefs = (meta.criteria_defs && meta.criteria_defs.execution) || [];
  const ratioDef = exeDefs.find((d) => d.name === 'tape_imbalance') || {};
  const entryThr = ofiDef.threshold ?? null;
  const sellThr = ofiDef.sell ?? null;
  const ratioMax = ratioDef.sell ?? 0.5;

  if (!meta.criteria_available) {
    wrap.innerHTML = `
      <div class="ofi-head">OFI Flow</div>
      <div class="ofi-unavailable">needs cached evaluation — see criteria panel</div>`;
    return { onBatch() {}, onSeek() {}, setPlayhead() {} };
  }

  const legend = `entry ≥ ${entryThr ?? '?'} · exit < ${sellThr ?? '?'} & ratio ≤ ${Math.round(ratioMax * 100)}%`;
  wrap.innerHTML = `
    <div class="ofi-head">OFI Flow
      <span class="ofi-chip" id="ofiChip">--</span>
    </div>
    <div class="ofi-spark-wrap">
      <canvas class="ofi-spark"></canvas>
      <div class="ofi-readout" hidden></div>
    </div>
    <div class="ofi-sub" title="Depth-normalized order flow: net book flow / displayed depth (EMA kernel). The entry chain requires it at or above the floor while L2 depth is live; the adverse-flow exit needs it under the sell level AND buy dominance lapsed.">${legend}</div>`;

  const chipEl = wrap.querySelector('#ofiChip');
  const sparkWrap = wrap.querySelector('.ofi-spark-wrap');
  const canvas = wrap.querySelector('.ofi-spark');
  const readoutEl = wrap.querySelector('.ofi-readout');
  const subEl = wrap.querySelector('.ofi-sub');
  const shiftSecs = (meta.tz_offset_minutes || 0) * 60;

  let points = [];          // {t, v(null = depth inactive), pass, ratio}
  let rightT = meta.window.start_ms;
  let hoverX = null;

  function latest() {
    return points.length ? points[points.length - 1] : null;
  }

  function exitArmed(point) {
    return point && point.v !== null && sellThr !== null && point.v < sellThr
      && point.ratio !== null && point.ratio <= ratioMax;
  }

  function prune() {
    const cutoff = rightT - WINDOW_MS - 5000;
    let drop = 0;
    while (drop < points.length && points[drop].t < cutoff) drop += 1;
    if (drop) points = points.slice(drop);
  }

  function applyHeader() {
    const point = latest();
    const stale = point && (rightT - point.t) > 15_000;
    if (!point || point.v === null || stale) {
      chipEl.textContent = 'no L2';
      chipEl.className = 'ofi-chip inactive';
    } else {
      chipEl.textContent = (point.v >= 0 ? '+' : '') + point.v.toFixed(2);
      chipEl.className = 'ofi-chip ' + (point.pass ? 'pass' : exitArmed(point) ? 'sell' : '');
    }
    if (!point || point.v === null || stale) {
      subEl.textContent = legend;
      subEl.className = 'ofi-sub';
    } else if (exitArmed(point)) {
      subEl.textContent = `EXIT ARMED · flow ${point.v.toFixed(2)} < ${sellThr} & ratio ${Math.round(point.ratio * 100)}% ≤ ${Math.round(ratioMax * 100)}%`;
      subEl.className = 'ofi-sub sell';
    } else if (point.v !== null && sellThr !== null && point.v < sellThr) {
      subEl.textContent = `flow adverse — exit held, buy ratio ${point.ratio !== null ? Math.round(point.ratio * 100) : '--'}% > ${Math.round(ratioMax * 100)}%`;
      subEl.className = 'ofi-sub warn';
    } else {
      subEl.textContent = legend;
      subEl.className = 'ofi-sub';
    }
  }

  function scaleMax() {
    // Symmetric range: keep both guides in frame, grow with the data.
    let m = Math.max(Math.abs(entryThr ?? 0.7), Math.abs(sellThr ?? 0.7)) * 1.6;
    for (const point of points) {
      if (point.v !== null && point.t >= rightT - WINDOW_MS) {
        m = Math.max(m, Math.abs(point.v) * 1.15);
      }
    }
    return m;
  }

  function draw() {
    const rect = sparkWrap.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const plotW = width - GUTTER_RIGHT;
    const plotH = height - PAD_TOP - PAD_BOTTOM;
    const vMax = scaleMax();
    const yFor = (v) => PAD_TOP + (1 - (v + vMax) / (2 * vMax)) * plotH;
    const xFor = (t) => ((t - (rightT - WINDOW_MS)) / WINDOW_MS) * plotW;
    const zeroY = yFor(0);

    // Guides: zero line solid, entry/sell dashed, labels in the gutter.
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.grid;
    ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(plotW, zeroY); ctx.stroke();
    ctx.font = '9px ui-sans-serif, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.setLineDash([3, 3]);
    if (entryThr !== null) {
      const y = yFor(entryThr);
      ctx.strokeStyle = COLORS.up;
      ctx.globalAlpha = 0.55;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(plotW, y); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = COLORS.up;
      ctx.fillText(String(entryThr), plotW + 4, y);
    }
    if (sellThr !== null) {
      const y = yFor(sellThr);
      ctx.strokeStyle = COLORS.down;
      ctx.globalAlpha = 0.55;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(plotW, y); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = COLORS.down;
      ctx.fillText(String(sellThr), plotW + 4, y);
    }
    ctx.setLineDash([]);
    // Zero label yields when a threshold label would overprint it.
    const zeroClear = (entryThr === null || Math.abs(zeroY - yFor(entryThr)) > 9)
      && (sellThr === null || Math.abs(zeroY - yFor(sellThr)) > 9);
    if (zeroClear) {
      ctx.fillStyle = COLORS.muted;
      ctx.fillText('0', plotW + 4, zeroY);
    }

    // Series: contiguous depth-active runs; area tinted by sign via clip.
    const visible = points.filter((p) => p.t >= rightT - WINDOW_MS - 2000 && p.t <= rightT);
    const runs = [];
    let run = [];
    for (const point of visible) {
      if (point.v === null) {
        if (run.length) { runs.push(run); run = []; }
      } else {
        run.push(point);
      }
    }
    if (run.length) runs.push(run);

    for (const r of runs) {
      if (r.length < 2) continue;
      const areaPath = () => {
        ctx.beginPath();
        ctx.moveTo(xFor(r[0].t), zeroY);
        for (const point of r) ctx.lineTo(xFor(point.t), yFor(point.v));
        ctx.lineTo(xFor(r[r.length - 1].t), zeroY);
        ctx.closePath();
      };
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, plotW, zeroY); ctx.clip();
      ctx.globalAlpha = 0.12; ctx.fillStyle = COLORS.up; areaPath(); ctx.fill();
      ctx.restore();
      ctx.save();
      ctx.beginPath(); ctx.rect(0, zeroY, plotW, height - zeroY); ctx.clip();
      ctx.globalAlpha = 0.12; ctx.fillStyle = COLORS.down; areaPath(); ctx.fill();
      ctx.restore();
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      r.forEach((point, i) => {
        const x = xFor(point.t);
        const y = yFor(point.v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // "Now" dot on the newest fresh value, colored by gate state.
    const point = latest();
    if (point && point.v !== null && rightT - point.t <= 15_000) {
      ctx.fillStyle = point.pass ? COLORS.up : exitArmed(point) ? COLORS.down : COLORS.text;
      ctx.beginPath();
      ctx.arc(Math.min(xFor(point.t), plotW - 2), yFor(point.v), 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Hover crosshair.
    if (hoverX !== null && hoverX <= plotW) {
      ctx.strokeStyle = COLORS.crosshair;
      ctx.globalAlpha = 0.6;
      ctx.beginPath(); ctx.moveTo(hoverX, 0); ctx.lineTo(hoverX, height); ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  let rafQueued = false;
  function queueDraw() {
    if (!rafQueued) {
      rafQueued = true;
      requestAnimationFrame(() => { rafQueued = false; draw(); applyHeader(); });
    }
  }

  // Hover readout + click/drag to seek (same transport semantics as the
  // timeline, scoped to the pane's 120s window).
  function timeAt(clientX) {
    const rect = sparkWrap.getBoundingClientRect();
    const plotW = Math.max(1, rect.width - GUTTER_RIGHT);
    const fraction = Math.max(0, Math.min(1, (clientX - rect.left) / plotW));
    return rightT - WINDOW_MS + fraction * WINDOW_MS;
  }
  let dragging = false;
  sparkWrap.addEventListener('pointermove', (event) => {
    const rect = sparkWrap.getBoundingClientRect();
    hoverX = event.clientX - rect.left;
    const t = timeAt(event.clientX);
    const idx = lastIndexAtOrBefore(points, t);
    const point = idx >= 0 ? points[idx] : null;
    if (point && point.v !== null) {
      readoutEl.hidden = false;
      readoutEl.textContent = `${fmtChartTime(Math.floor(point.t / 1000) + shiftSecs, true)} · ${(point.v >= 0 ? '+' : '') + point.v.toFixed(2)}`;
      readoutEl.style.left = `${Math.min(hoverX + 8, rect.width - 90)}px`;
    } else {
      readoutEl.hidden = true;
    }
    if (dragging) onSeekRequest(t);
    queueDraw();
  });
  sparkWrap.addEventListener('pointerleave', () => {
    hoverX = null;
    readoutEl.hidden = true;
    queueDraw();
  });
  sparkWrap.addEventListener('pointerdown', (event) => {
    dragging = true;
    sparkWrap.setPointerCapture(event.pointerId);
    onSeekRequest(timeAt(event.clientX));
  });
  sparkWrap.addEventListener('pointerup', (event) => {
    dragging = false;
    sparkWrap.releasePointerCapture(event.pointerId);
  });
  if (window.ResizeObserver) new ResizeObserver(queueDraw).observe(sparkWrap);

  return {
    refresh: queueDraw,
    onBatch(batch) {
      for (const state of batch.states) points.push(chainValues(state));
      rightT = batch.t;
      prune();
      queueDraw();
    },
    onSeek(snapshot) {
      rightT = snapshot.t;
      points = [];
      const states = loader.states;
      const end = lastIndexAtOrBefore(states, snapshot.t);
      let start = end;
      while (start >= 0 && states[start].t >= snapshot.t - WINDOW_MS - 5000) start -= 1;
      for (let i = start + 1; i <= end; i += 1) points.push(chainValues(states[i]));
      queueDraw();
    },
    setPlayhead(t) {
      if (t > rightT) {
        rightT = t;
        prune();
        queueDraw();
      }
    },
  };
}
