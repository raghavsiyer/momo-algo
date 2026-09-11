// Standard Lightweight Charts indicator sub-pane (MACD, confidence, RVOL...).
// Quarter-second tape panes use tapePane.js instead — never give sub-second
// data its own Lightweight Charts time scale.
import { COLORS } from './theme.js';
import { fmtPrice, lineStyle } from './util.js';

export function createIndicatorPane(panel, pane) {
  const wrap = document.createElement('div');
  wrap.className = 'indicator-wrap';
  wrap.dataset.indicator = pane.kind || '';
  const label = document.createElement('div');
  label.className = 'indicator-label';
  label.textContent = pane.title;
  const legend = document.createElement('div');
  legend.className = 'indicator-legend';
  const chartEl = document.createElement('div');
  chartEl.className = 'indicator-chart';
  chartEl.style.height = `${pane.height || 150}px`;
  wrap.appendChild(label);
  wrap.appendChild(legend);
  wrap.appendChild(chartEl);
  panel.appendChild(wrap);

  const chart = LightweightCharts.createChart(chartEl, {
    autoSize: true,
    layout: { background: { color: COLORS.panel }, textColor: COLORS.text },
    grid: { vertLines: { color: COLORS.grid }, horzLines: { color: COLORS.grid } },
    rightPriceScale: { borderColor: COLORS.grid, scaleMargins: { top: 0.12, bottom: 0.16 } },
    timeScale: { borderColor: COLORS.grid, timeVisible: true, secondsVisible: pane.secondsVisible || false, rightOffset: 4, barSpacing: pane.secondsVisible ? 8 : undefined },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    localization: { priceFormatter: fmtPrice },
  });
  let baselineSeries = null;
  if (pane.histogram && pane.histogram.data && pane.histogram.data.length) {
    const histogram = chart.addHistogramSeries({ priceFormat: pane.priceFormat, priceLineVisible: false });
    histogram.setData(pane.histogram.data);
    baselineSeries = histogram;
    const item = document.createElement('span');
    item.textContent = pane.histogram.label;
    legend.appendChild(item);
  }
  for (const line of pane.lines || []) {
    if (!line.data || !line.data.length) continue;
    const series = chart.addLineSeries({ color: line.color, lineWidth: line.width || 1, priceFormat: pane.priceFormat, priceLineVisible: false, title: line.label });
    series.setData(line.data);
    baselineSeries = baselineSeries || series;
    const item = document.createElement('span');
    item.style.color = line.color;
    item.textContent = line.label;
    legend.appendChild(item);
  }
  if (baselineSeries && typeof pane.baseline === 'number') {
    baselineSeries.createPriceLine({ price: pane.baseline, color: COLORS.baseline, lineWidth: 1, lineStyle: lineStyle.dashed, axisLabelVisible: true, title: pane.baselineLabel || '' });
  }
  if (baselineSeries && typeof pane.threshold === 'number') {
    baselineSeries.createPriceLine({ price: pane.threshold, color: COLORS.criteria, lineWidth: 1, lineStyle: lineStyle.dashed, axisLabelVisible: true, title: pane.thresholdLabel || 'criteria' });
  }
  chart.timeScale().fitContent();
  return { chart, wrap, pane };
}
