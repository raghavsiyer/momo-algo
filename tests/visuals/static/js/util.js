export const lineStyle = { solid: 0, dotted: 1, dashed: 2 };

export function fmtPrice(price) {
  if (price == null || Number.isNaN(price)) return '';
  return Math.abs(price) >= 10 ? price.toFixed(2) : price.toFixed(3);
}

export function fmtValue(value) {
  return value == null || Number.isNaN(value) ? '--' : fmtPrice(value);
}

export function fmtVolume(value) {
  if (value == null || Number.isNaN(value)) return '--';
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
}

export function fmtNumber(value) {
  if (value === null || value === undefined || value === '') return '-';
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(Number(value || 0));
}

export function fmtPct(value) {
  if (value === null || value === undefined || value === '') return '-';
  return Number(value || 0).toFixed(2) + '%';
}

export function fmtCompact(value) {
  if (value === null || value === undefined || value === '') return '-';
  const number = Number(value || 0);
  if (Math.abs(number) >= 1e9) return (number / 1e9).toFixed(1) + 'B';
  if (Math.abs(number) >= 1e6) return (number / 1e6).toFixed(1) + 'M';
  if (Math.abs(number) >= 1e3) return (number / 1e3).toFixed(1) + 'K';
  return String(Math.round(number));
}

// Chart times are shifted-to-local epoch seconds (the standard Lightweight
// Charts trick), so formatting uses the UTC accessors.
export function fmtChartTime(time, withSeconds = false) {
  if (time == null) return '';
  const date = new Date(Number(time) * 1000);
  const hh = String(date.getUTCHours()).padStart(2, '0');
  const mm = String(date.getUTCMinutes()).padStart(2, '0');
  const ss = String(date.getUTCSeconds()).padStart(2, '0');
  return withSeconds ? `${hh}:${mm}:${ss}` : `${hh}:${mm}`;
}
