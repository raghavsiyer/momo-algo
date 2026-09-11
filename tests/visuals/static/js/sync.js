// Keeps the main chart and its Lightweight Charts indicator panes on the same
// visible logical range. Canvas tape panes are excluded (they redraw off the
// main chart's time scale directly).
export function syncCharts(mainChart, indicators) {
  if (!indicators.length) return () => {};
  const subCharts = indicators.map((indicator) => indicator.chart);
  let syncing = false;
  function setRange(targets, range) {
    if (!range || syncing) return;
    syncing = true;
    for (const chart of targets) chart.timeScale().setVisibleLogicalRange(range);
    syncing = false;
  }
  function syncToMain() {
    setRange(subCharts, mainChart.timeScale().getVisibleLogicalRange());
  }
  mainChart.timeScale().subscribeVisibleLogicalRangeChange((range) => setRange(subCharts, range));
  for (const subChart of subCharts) {
    subChart.timeScale().subscribeVisibleLogicalRangeChange((range) => setRange([mainChart, ...subCharts.filter((chart) => chart !== subChart)], range));
  }
  return syncToMain;
}
