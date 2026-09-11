// Reads the color tokens defined in css/theme.css so canvas drawing and
// Lightweight Charts options always match the stylesheet.
//
// COLORS is mutated in place by refreshTheme() so every module that imported
// it sees the new palette after a light/dark toggle; components that bake
// colors into chart options re-apply them via their own retheme paths.

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const TOKENS = {
  bg: '--bg',
  panel: '--panel',
  grid: '--grid',
  text: '--text',
  muted: '--muted',
  accent: '--accent',
  up: '--up',
  down: '--down',
  baseline: '--baseline',
  criteria: '--criteria',
  crosshair: '--crosshair',
  crosshairLabelBg: '--crosshair-label-bg',
};

export const COLORS = {};

export function refreshTheme() {
  for (const [key, token] of Object.entries(TOKENS)) COLORS[key] = cssVar(token);
}

export function currentTheme() {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

export function setTheme(theme) {
  if (theme === 'light') document.documentElement.dataset.theme = 'light';
  else delete document.documentElement.dataset.theme;
  try { window.localStorage.setItem('momoVisuals.theme', theme); } catch (error) {}
  refreshTheme();
}

refreshTheme();
