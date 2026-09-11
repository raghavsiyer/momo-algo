// "All gates aligned" ribbon for the Charts view: when a ready replay bundle
// covers the day, its all-pass segments are painted as green bands across the
// bottom of the price chart (DOM overlay repositioned on range changes — the
// same proven pattern as the trade-marker layer).

function makeInterpolator(candleTimes, chart) {
  return function xFor(chartTime) {
    if (!candleTimes.length) return null;
    if (chartTime <= candleTimes[0]) return chart.timeScale().timeToCoordinate(candleTimes[0]);
    const lastTime = candleTimes[candleTimes.length - 1];
    if (chartTime >= lastTime) return chart.timeScale().timeToCoordinate(lastTime);
    let lo = 0;
    let hi = candleTimes.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (candleTimes[mid] === chartTime) return chart.timeScale().timeToCoordinate(chartTime);
      if (candleTimes[mid] < chartTime) lo = mid + 1;
      else hi = mid - 1;
    }
    const leftTime = candleTimes[Math.max(0, lo - 1)];
    const rightTime = candleTimes[Math.min(candleTimes.length - 1, lo)];
    const leftX = chart.timeScale().timeToCoordinate(leftTime);
    const rightX = chart.timeScale().timeToCoordinate(rightTime);
    if (leftX == null || rightX == null || rightTime === leftTime) return leftX;
    return leftX + ((chartTime - leftTime) / (rightTime - leftTime)) * (rightX - leftX);
  };
}

export function attachAllpassRibbon(panel, segments, tzOffsetMinutes) {
  if (!segments.length || !panel.frame.candles.length) return;
  const shiftSecs = tzOffsetMinutes * 60;
  const layer = document.createElement('div');
  layer.className = 'ribbon-layer';
  layer.title = 'green: every entry-chain gate passing (from replay bundle)';
  panel.wrap.appendChild(layer);
  const candleTimes = panel.frame.candles.map((bar) => bar.time);
  const xFor = makeInterpolator(candleTimes, panel.chart);
  const bands = segments.map(([t0, t1]) => {
    const el = document.createElement('div');
    el.className = 'ribbon-band';
    layer.appendChild(el);
    return { el, c0: Math.floor(t0 / 1000) + shiftSecs, c1: Math.floor(t1 / 1000) + shiftSecs };
  });

  function reposition() {
    for (const band of bands) {
      const x0 = xFor(band.c0);
      const x1 = xFor(band.c1);
      if (x0 == null || x1 == null) { band.el.hidden = true; continue; }
      band.el.hidden = false;
      band.el.style.left = `${Math.round(x0)}px`;
      band.el.style.width = `${Math.max(2, Math.round(x1 - x0))}px`;
    }
  }

  panel.chart.timeScale().subscribeVisibleLogicalRangeChange(() => requestAnimationFrame(reposition));
  if (window.ResizeObserver) new ResizeObserver(() => reposition()).observe(panel.wrap);
  requestAnimationFrame(reposition);
}
