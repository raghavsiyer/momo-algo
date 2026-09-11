// Quarter-second tape indicator pane drawn on a raw canvas.
//
// DO NOT rewrite this as a separate Lightweight Charts instance with its own
// time scale: sub-second series at a different time multiplier leave residual
// jitter/misalignment when panning, and a dense time spine on the main chart
// shrinks its candles under fitContent(). The pane borrows the main chart's
// x-axis via timeScale().timeToCoordinate() and draws its own grid, y-axis
// labels, crosshair, and hit-testing. Major ticks anchor to globally stable
// candle indices (steps 1, 2, 4, 8, ...), never "every Nth visible candle",
// so grid lines don't slide during pan/zoom.
import { COLORS } from './theme.js';
import { fmtChartTime } from './util.js';

export function createTapeCanvasPane(panel, pane, mainChart, frame) {
  const timeMultiplier = pane.timeMultiplier || 1;
  const axisHeight = 22;
  const minTickSpacing = 92;
  const minSubdivisionSpacing = 18;
  const subdivisionSteps = [40, 16, 8, 4, 2, 1];
  const wrap = document.createElement('div');
  wrap.className = 'indicator-wrap';
  wrap.dataset.indicator = pane.kind || '';
  const label = document.createElement('div');
  label.className = 'indicator-label';
  label.textContent = pane.title;
  const legend = document.createElement('div');
  legend.className = 'indicator-legend';
  const canvas = document.createElement('canvas');
  canvas.className = 'indicator-chart';
  canvas.style.height = `${pane.height || 150}px`;
  canvas.style.cursor = 'crosshair';
  wrap.appendChild(label);
  wrap.appendChild(legend);
  wrap.appendChild(canvas);
  panel.appendChild(wrap);

  const candleTimes = (frame.candles || []).map((item) => item.time).sort((left, right) => left - right);
  const candleIntervals = [];
  for (let index = 1; index < candleTimes.length; index += 1) {
    const delta = candleTimes[index] - candleTimes[index - 1];
    if (delta > 0 && delta < 3600) candleIntervals.push(delta);
  }
  candleIntervals.sort((left, right) => left - right);
  const candleInterval = candleIntervals.length ? candleIntervals[Math.floor(candleIntervals.length / 2)] : 10;
  const seriesList = [];
  if (pane.histogram && pane.histogram.data && pane.histogram.data.length) {
    seriesList.push({ kind: 'histogram', label: pane.histogram.label, data: pane.histogram.data, color: null });
    const item = document.createElement('span');
    item.textContent = pane.histogram.label;
    legend.appendChild(item);
  }
  for (const line of pane.lines || []) {
    if (!line.data || !line.data.length) continue;
    seriesList.push({ kind: 'line', label: line.label, data: line.data, color: line.color, width: line.width || 1 });
    const item = document.createElement('span');
    item.style.color = line.color;
    item.textContent = line.label;
    legend.appendChild(item);
  }

  const allValues = [];
  for (const series of seriesList) {
    for (const point of series.data) {
      if (typeof point.value === 'number' && Number.isFinite(point.value)) allValues.push(point.value);
    }
  }
  const globalMin = allValues.length ? Math.min(...allValues) : 0;
  const globalMax = allValues.length ? Math.max(...allValues) : 1;

  let minValue = globalMin;
  let maxValue = globalMax;

  function computeVisibleRange(plotRight) {
    const visible = [];
    for (const series of seriesList) {
      for (const point of series.data) {
        if (typeof point.value !== 'number') continue;
        const x = coordinateForSecond(Number(point.time) / timeMultiplier);
        if (x == null || x < -4 || x > plotRight + 4) continue;
        visible.push(point.value);
      }
    }
    if (typeof pane.baseline === 'number') visible.push(pane.baseline);
    if (typeof pane.threshold === 'number') visible.push(pane.threshold);
    if (!visible.length) { minValue = globalMin; maxValue = globalMax; return; }
    let lo = Math.min(...visible);
    let hi = Math.max(...visible);
    if (lo === hi) {
      const pad = Math.max(1, Math.abs(lo) * 0.1);
      lo -= pad;
      hi += pad;
    } else {
      const pad = (hi - lo) * 0.08;
      lo -= pad;
      hi += pad;
    }
    minValue = lo;
    maxValue = hi;
  }

  let crosshairX = null;
  let crosshairY = null;

  function getPlotRight() {
    return mainChart.timeScale().width() || Math.max(1, (canvas.getBoundingClientRect().width || 400) - 56);
  }

  function coordinateForSecond(second) {
    if (!candleTimes.length) return null;
    if (second <= candleTimes[0]) return mainChart.timeScale().timeToCoordinate(candleTimes[0]);
    const lastTime = candleTimes[candleTimes.length - 1];
    if (second >= lastTime) return mainChart.timeScale().timeToCoordinate(lastTime);
    let low = 0;
    let high = candleTimes.length - 1;
    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      const midTime = candleTimes[mid];
      if (midTime === second) return mainChart.timeScale().timeToCoordinate(midTime);
      if (midTime < second) low = mid + 1;
      else high = mid - 1;
    }
    const leftTime = candleTimes[Math.max(0, low - 1)];
    const rightTime = candleTimes[Math.min(candleTimes.length - 1, low)];
    const leftX = mainChart.timeScale().timeToCoordinate(leftTime);
    const rightX = mainChart.timeScale().timeToCoordinate(rightTime);
    if (leftX == null || rightX == null || rightTime === leftTime) return null;
    return leftX + ((second - leftTime) / (rightTime - leftTime)) * (rightX - leftX);
  }

  function secondForCoordinate(px) {
    if (!candleTimes.length) return null;
    let bestLeft = null;
    let bestRight = null;
    for (const time of candleTimes) {
      const x = mainChart.timeScale().timeToCoordinate(time);
      if (x == null || Number.isNaN(x)) continue;
      if (x <= px) { if (!bestLeft || x > bestLeft.x) bestLeft = { time, x }; }
      if (x >= px) { if (!bestRight || x < bestRight.x) bestRight = { time, x }; break; }
    }
    if (!bestLeft && !bestRight) return null;
    if (!bestLeft) return bestRight.time;
    if (!bestRight) return bestLeft.time;
    if (bestRight.x === bestLeft.x) return bestLeft.time;
    return bestLeft.time + (px - bestLeft.x) / (bestRight.x - bestLeft.x) * (bestRight.time - bestLeft.time);
  }

  function yForValue(value, plotTop, plotBottom) {
    return plotTop + (maxValue - value) / (maxValue - minValue) * Math.max(1, plotBottom - plotTop);
  }

  function valueForY(y, plotTop, plotBottom) {
    return maxValue - (y - plotTop) / Math.max(1, plotBottom - plotTop) * (maxValue - minValue);
  }

  function formatAxisValue(value) {
    const precision = pane.priceFormat && typeof pane.priceFormat.precision === 'number' ? pane.priceFormat.precision : 2;
    if (Math.abs(value) >= 1000) return value.toFixed(0);
    return value.toFixed(precision);
  }

  function yAxisTicks(plotTop, plotBottom) {
    const range = maxValue - minValue;
    const ticks = [];
    const steps = [0, 0.25, 0.5, 0.75, 1];
    for (const fraction of steps) {
      const value = maxValue - range * fraction;
      const y = yForValue(value, plotTop, plotBottom);
      if (y >= plotTop && y <= plotBottom) ticks.push(value);
    }
    return ticks;
  }

  function visibleAxisTicks(plotRight) {
    const visible = [];
    for (let index = 0; index < candleTimes.length; index += 1) {
      const time = candleTimes[index];
      const x = mainChart.timeScale().timeToCoordinate(time);
      if (x == null || Number.isNaN(x)) continue;
      if (x >= -minTickSpacing && x <= plotRight + minTickSpacing) visible.push({ time, x, index });
    }
    if (!visible.length) return { major: [], minor: [] };
    const candleSpacings = [];
    for (let index = 1; index < visible.length; index += 1) {
      const left = visible[index - 1];
      const right = visible[index];
      if (right.index === left.index + 1) candleSpacings.push(Math.abs(right.x - left.x));
    }
    candleSpacings.sort((left, right) => left - right);
    const candleSpacing = candleSpacings.length ? candleSpacings[Math.floor(candleSpacings.length / 2)] : plotRight / Math.max(1, visible.length);
    const majorSteps = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
    const majorStep = majorSteps.find((candidate) => candleSpacing * candidate >= minTickSpacing) || majorSteps[majorSteps.length - 1];
    const major = [];
    const firstMajorIndex = Math.ceil(visible[0].index / majorStep) * majorStep;
    const lastVisibleIndex = visible[visible.length - 1].index;
    for (let index = firstMajorIndex; index <= lastVisibleIndex; index += majorStep) {
      const time = candleTimes[index];
      const x = mainChart.timeScale().timeToCoordinate(time);
      if (x == null || Number.isNaN(x) || x < -1 || x > plotRight + 1) continue;
      major.push({ time, x, index });
    }

    const minor = [];
    for (let index = 1; index < visible.length; index += 1) {
      const left = visible[index - 1];
      const right = visible[index];
      const timeSpan = right.time - left.time;
      const pixelSpan = right.x - left.x;
      if (timeSpan <= 0 || timeSpan > candleInterval * 1.5 || pixelSpan <= minSubdivisionSpacing * 2) continue;
      const maxDivisions = Math.floor(pixelSpan / minSubdivisionSpacing);
      const divisions = subdivisionSteps.find((candidate) => candidate <= maxDivisions && timeSpan / candidate >= 0.25) || 1;
      for (let division = 1; division < divisions; division += 1) {
        const fraction = division / divisions;
        const x = left.x + pixelSpan * fraction;
        if (x >= -1 && x <= plotRight + 1) minor.push({ time: left.time + timeSpan * fraction, x });
      }
    }
    return { major, minor };
  }

  function draw() {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLORS.panel;
    ctx.fillRect(0, 0, width, height);
    const plotRight = getPlotRight();
    computeVisibleRange(plotRight);
    const plotTop = 8;
    const axisTop = Math.max(plotTop + 1, height - axisHeight);
    const plotBottom = axisTop;
    const axisTicks = visibleAxisTicks(plotRight);

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    for (const tick of axisTicks.minor) {
      const x = Math.round(tick.x) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, axisTop);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.beginPath();
    ctx.moveTo(0, 0.5);
    ctx.lineTo(plotRight, 0.5);
    ctx.moveTo(0, axisTop + 0.5);
    ctx.lineTo(width, axisTop + 0.5);
    ctx.moveTo(plotRight + 0.5, 0);
    ctx.lineTo(plotRight + 0.5, axisTop);
    for (const tick of axisTicks.major) {
      const x = Math.round(tick.x) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, axisTop);
    }
    for (const value of yAxisTicks(plotTop, plotBottom)) {
      const y = Math.round(yForValue(value, plotTop, plotBottom)) + 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(plotRight, y);
    }
    ctx.stroke();

    ctx.fillStyle = COLORS.muted;
    ctx.font = '11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const tick of axisTicks.major) {
      if (tick.x < 28 || tick.x > plotRight - 28) continue;
      const label = fmtChartTime(tick.time, frame.timeframe === '10s');
      ctx.fillText(label, tick.x, axisTop + axisHeight / 2);
    }
    ctx.textAlign = 'left';
    for (const value of yAxisTicks(plotTop, plotBottom)) {
      const y = yForValue(value, plotTop, plotBottom);
      const labelY = Math.min(axisTop - 6, Math.max(plotTop + 6, y));
      ctx.fillText(formatAxisValue(value), plotRight + 6, labelY);
    }

    if (typeof pane.baseline === 'number') {
      const baselineY = yForValue(pane.baseline, plotTop, plotBottom);
      ctx.strokeStyle = COLORS.baseline;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, baselineY);
      ctx.lineTo(plotRight, baselineY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (typeof pane.threshold === 'number') {
      const thresholdY = yForValue(pane.threshold, plotTop, plotBottom);
      ctx.strokeStyle = COLORS.criteria;
      ctx.setLineDash([1, 4]);
      ctx.beginPath();
      ctx.moveTo(0, thresholdY);
      ctx.lineTo(plotRight, thresholdY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, plotRight, axisTop);
    ctx.clip();

    for (const series of seriesList) {
      if (series.kind === 'histogram') {
        const baselineY = yForValue(typeof pane.baseline === 'number' ? pane.baseline : 0, plotTop, plotBottom);
        const points = [];
        const timeGaps = [];
        for (const point of series.data) {
          if (typeof point.value !== 'number') continue;
          const x = coordinateForSecond(Number(point.time) / timeMultiplier);
          if (x == null || x < -2 || x > plotRight + 2) continue;
          points.push({ x, y: yForValue(point.value, plotTop, plotBottom), color: point.color, time: point.time });
        }
        for (let index = 1; index < points.length; index += 1) {
          const dt = points[index].time - points[index - 1].time;
          if (dt > 0) timeGaps.push(dt);
        }
        timeGaps.sort((left, right) => left - right);
        const medianGap = timeGaps.length ? timeGaps[Math.floor(timeGaps.length / 2)] : Infinity;
        const gapThreshold = medianGap * 3;
        if (points.length > 0) {
          let segStart = 0;
          while (segStart < points.length) {
            const segColor = points[segStart].color || COLORS.accent;
            let segEnd = segStart;
            while (segEnd + 1 < points.length && (points[segEnd + 1].color || COLORS.accent) === segColor && (points[segEnd + 1].time - points[segEnd].time) <= gapThreshold) segEnd += 1;
            ctx.fillStyle = segColor;
            ctx.globalAlpha = 0.7;
            ctx.beginPath();
            ctx.moveTo(points[segStart].x, baselineY);
            for (let index = segStart; index <= segEnd; index += 1) ctx.lineTo(points[index].x, points[index].y);
            ctx.lineTo(points[segEnd].x, baselineY);
            ctx.closePath();
            ctx.fill();
            ctx.globalAlpha = 1;
            segStart = segEnd + 1;
          }
        }
      } else {
        ctx.strokeStyle = series.color || COLORS.accent;
        ctx.lineWidth = series.width || 1;
        const intervals = [];
        for (let index = 1; index < series.data.length; index += 1) {
          if (typeof series.data[index].value !== 'number' || typeof series.data[index - 1].value !== 'number') continue;
          const dt = series.data[index].time - series.data[index - 1].time;
          if (dt > 0) intervals.push(dt);
        }
        intervals.sort((left, right) => left - right);
        const medianInterval = intervals.length ? intervals[Math.floor(intervals.length / 2)] : Infinity;
        const gapThreshold = medianInterval * 3;
        ctx.beginPath();
        let prevTime = null;
        let started = false;
        for (const point of series.data) {
          if (typeof point.value !== 'number') { prevTime = null; started = false; continue; }
          const x = coordinateForSecond(Number(point.time) / timeMultiplier);
          if (x == null) continue;
          const y = yForValue(point.value, plotTop, plotBottom);
          const isGap = prevTime != null && (point.time - prevTime) > gapThreshold;
          if (!started || isGap) {
            if (started) ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
          prevTime = point.time;
        }
        if (started) ctx.stroke();
      }
    }

    ctx.restore();

    if (crosshairX != null && crosshairY != null && crosshairX >= 0 && crosshairX <= plotRight) {
      const cx = Math.round(crosshairX) + 0.5;
      const cy = Math.min(Math.max(plotTop, crosshairY), plotBottom);
      ctx.strokeStyle = COLORS.crosshair;
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, axisTop);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, Math.round(cy) + 0.5);
      ctx.lineTo(plotRight, Math.round(cy) + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);

      const cursorValue = valueForY(cy, plotTop, plotBottom);
      const priceLabel = formatAxisValue(cursorValue);
      ctx.font = '11px ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
      const priceLabelWidth = ctx.measureText(priceLabel).width + 10;
      const priceLabelHeight = 18;
      const priceLabelX = plotRight + 1;
      const priceLabelY = Math.round(cy) - priceLabelHeight / 2;
      ctx.fillStyle = COLORS.crosshairLabelBg;
      ctx.beginPath();
      ctx.roundRect(priceLabelX, priceLabelY, priceLabelWidth, priceLabelHeight, 3);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(priceLabel, priceLabelX + 5, Math.round(cy));

      const cursorSecond = secondForCoordinate(crosshairX);
      if (cursorSecond != null) {
        const timeLabel = fmtChartTime(Math.round(cursorSecond), true);
        const timeLabelWidth = ctx.measureText(timeLabel).width + 10;
        const timeLabelHeight = 18;
        const timeLabelX = Math.round(crosshairX) - timeLabelWidth / 2;
        const timeLabelY = axisTop + (axisHeight - timeLabelHeight) / 2;
        ctx.fillStyle = COLORS.crosshairLabelBg;
        ctx.beginPath();
        ctx.roundRect(timeLabelX, timeLabelY, timeLabelWidth, timeLabelHeight, 3);
        ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(timeLabel, Math.round(crosshairX), axisTop + axisHeight / 2);
      }
    }
  }

  canvas.addEventListener('mousemove', (event) => {
    const rect = canvas.getBoundingClientRect();
    crosshairX = event.clientX - rect.left;
    crosshairY = event.clientY - rect.top;
    requestAnimationFrame(draw);
  });
  canvas.addEventListener('mouseleave', () => {
    crosshairX = null;
    crosshairY = null;
    requestAnimationFrame(draw);
  });

  mainChart.timeScale().subscribeVisibleLogicalRangeChange(() => requestAnimationFrame(draw));
  mainChart.timeScale().subscribeVisibleTimeRangeChange(() => requestAnimationFrame(draw));
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(() => requestAnimationFrame(draw));
    observer.observe(canvas);
  }
  requestAnimationFrame(draw);
  return { chart: null, wrap, pane, redraw: draw };
}
