// Timeline scrubber: canvas price sparkline over the whole window, loaded /
// buffered extent, fill + signal pins, all-gates-pass bands, and a draggable
// playhead.
import { COLORS } from '../theme.js';
import { fmtChartTime } from '../util.js';

export function createTimeline(container, meta, loader, onSeekRequest) {
  const startMs = meta.window.start_ms;
  const endMs = meta.window.end_ms;
  const spanMs = Math.max(1, endMs - startMs);
  const shiftSecs = (meta.tz_offset_minutes || 0) * 60;

  const wrap = document.createElement('div');
  wrap.className = 'timeline';
  const canvas = document.createElement('canvas');
  canvas.className = 'timeline-canvas';
  wrap.appendChild(canvas);
  container.appendChild(wrap);

  let playheadMs = startMs;
  let sparkline = [];         // decimated [t, price]
  let sparkMin = Infinity;
  let sparkMax = -Infinity;
  let sparkCursor = 0;        // ticks consumed into the sparkline
  const SPARK_BUCKET_MS = Math.max(1000, Math.floor(spanMs / 1200));
  let lastBucket = -1;

  function absorbTicks() {
    const ticks = loader.ticks;
    while (sparkCursor < ticks.length) {
      const tick = ticks[sparkCursor];
      sparkCursor += 1;
      const bucket = Math.floor((tick.t - startMs) / SPARK_BUCKET_MS);
      if (bucket !== lastBucket) {
        sparkline.push([tick.t, tick.p]);
        lastBucket = bucket;
      } else {
        sparkline[sparkline.length - 1] = [tick.t, tick.p];
      }
      if (tick.p < sparkMin) sparkMin = tick.p;
      if (tick.p > sparkMax) sparkMax = tick.p;
    }
  }

  function xFor(t, width) {
    return ((t - startMs) / spanMs) * width;
  }

  function draw() {
    const rect = wrap.getBoundingClientRect();
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

    // Loaded extent.
    const loadedX = xFor(Math.min(endMs, loader.loadedThroughMs()), width);
    ctx.fillStyle = 'rgba(139, 148, 158, 0.12)';
    ctx.fillRect(0, 0, loadedX, height);

    // All-pass bands (from events.json once the build is done).
    if (loader.events && loader.events.allpass) {
      ctx.fillStyle = 'rgba(34, 197, 94, 0.22)';
      for (const [t0, t1] of loader.events.allpass) {
        const x0 = xFor(t0, width);
        ctx.fillRect(x0, 0, Math.max(2, xFor(t1, width) - x0), height);
      }
    }

    // Price sparkline.
    absorbTicks();
    if (sparkline.length > 1 && sparkMax > sparkMin) {
      ctx.strokeStyle = 'rgba(201, 209, 217, 0.75)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      let started = false;
      for (const [t, price] of sparkline) {
        const x = xFor(t, width);
        const y = height - 5 - ((price - sparkMin) / (sparkMax - sparkMin)) * (height - 12);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Fill pins (loaded fills stream in; events.json completes them).
    // Replay fills solid; logged live-session fills hollow.
    function fillPinPath(fill) {
      const x = xFor(fill.t, width);
      ctx.beginPath();
      if (fill.side === 'BUY') {
        ctx.moveTo(x, height - 2); ctx.lineTo(x - 4, height + 4 - 12); ctx.lineTo(x + 4, height + 4 - 12);
      } else {
        ctx.moveTo(x, 2); ctx.lineTo(x - 4, 10); ctx.lineTo(x + 4, 10);
      }
      ctx.closePath();
    }
    const fills = (loader.events && loader.events.fills) || loader.fills;
    for (const fill of fills) {
      ctx.fillStyle = fill.side === 'BUY' ? COLORS.up : COLORS.down;
      fillPinPath(fill);
      ctx.fill();
    }
    const liveFills = (loader.events && loader.events.live_fills) || [];
    ctx.lineWidth = 1.5;
    for (const fill of liveFills) {
      ctx.strokeStyle = fill.side === 'BUY' ? COLORS.up : COLORS.down;
      fillPinPath(fill);
      ctx.stroke();
    }

    // Signal pins.
    if (loader.events && loader.events.signals) {
      ctx.fillStyle = COLORS.accent;
      for (const signal of loader.events.signals) {
        const x = xFor(signal.t, width);
        ctx.fillRect(x - 1, 0, 2, 6);
      }
    }

    // Playhead.
    const px = xFor(playheadMs, width);
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();

    // Time labels.
    ctx.fillStyle = COLORS.muted;
    ctx.font = '10px ui-sans-serif, sans-serif';
    ctx.textBaseline = 'bottom';
    ctx.textAlign = 'left';
    ctx.fillText(fmtChartTime(Math.floor(startMs / 1000) + shiftSecs, true), 4, height - 2);
    ctx.textAlign = 'right';
    ctx.fillText(fmtChartTime(Math.floor(endMs / 1000) + shiftSecs, true), width - 4, height - 2);
  }

  let rafQueued = false;
  function queueDraw() {
    if (!rafQueued) { rafQueued = true; requestAnimationFrame(() => { rafQueued = false; draw(); }); }
  }

  function seekFromPointer(event) {
    const rect = wrap.getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    onSeekRequest(startMs + fraction * spanMs);
  }

  let dragging = false;
  wrap.addEventListener('pointerdown', (event) => { dragging = true; wrap.setPointerCapture(event.pointerId); seekFromPointer(event); });
  wrap.addEventListener('pointermove', (event) => { if (dragging) seekFromPointer(event); });
  wrap.addEventListener('pointerup', (event) => { dragging = false; wrap.releasePointerCapture(event.pointerId); });
  if (window.ResizeObserver) new ResizeObserver(queueDraw).observe(wrap);

  return {
    setPlayhead(t) { playheadMs = t; queueDraw(); },
    refresh: queueDraw,
    onBatch() { queueDraw(); },
    onSeek(snapshot) { playheadMs = snapshot.t; queueDraw(); },
  };
}
