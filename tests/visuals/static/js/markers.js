// Trade fill markers as absolutely-positioned DOM triangles with hover
// tooltips, repositioned on every visible-range change. (Lightweight Charts'
// setMarkers can't carry rich tooltips.)
import { fmtValue } from './util.js';

export function createTradeMarkerLayer(chartWrap, chart, priceSeries, markers) {
  const layer = document.createElement('div');
  layer.className = 'trade-marker-layer';
  chartWrap.appendChild(layer);
  const markerItems = (markers || []).map((marker) => {
    const isBuy = marker.side === 'BUY';
    const el = document.createElement('div');
    el.className = `trade-marker ${isBuy ? 'buy' : 'sell'}`;
    el.setAttribute('data-tooltip', marker.text || `${marker.side || 'FILL'} @ ${fmtValue(marker.price)}`);
    if (isBuy) el.style.borderBottomColor = marker.color;
    else el.style.borderTopColor = marker.color;
    layer.appendChild(el);
    return { marker, el, isBuy };
  });

  function updateTradeMarkers() {
    for (const item of markerItems) {
      const x = chart.timeScale().timeToCoordinate(item.marker.time);
      const y = priceSeries.priceToCoordinate(item.marker.price);
      if (x == null || y == null || Number.isNaN(x) || Number.isNaN(y)) {
        item.el.hidden = true;
        continue;
      }
      item.el.hidden = false;
      item.el.style.left = `${Math.round(x)}px`;
      item.el.style.top = `${Math.round(y)}px`;
    }
  }

  chart.timeScale().subscribeVisibleLogicalRangeChange(() => requestAnimationFrame(updateTradeMarkers));
  if (window.ResizeObserver) {
    const observer = new ResizeObserver(() => updateTradeMarkers());
    observer.observe(chartWrap);
  }
  requestAnimationFrame(updateTradeMarkers);
  return updateTradeMarkers;
}
