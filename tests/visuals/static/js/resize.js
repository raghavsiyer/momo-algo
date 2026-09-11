// Pointer-drag resize handles. Panels resize in document/grid flow, so
// adjacent content is pushed — never overlapped. Sizes persist per key.

export function savedSize(key, fallback) {
  try {
    const stored = Number(window.localStorage.getItem(`momoVisuals.size.${key}`));
    if (Number.isFinite(stored) && stored > 0) return stored;
  } catch (error) {}
  return fallback;
}

function persist(key, value) {
  try { window.localStorage.setItem(`momoVisuals.size.${key}`, String(Math.round(value))); } catch (error) {}
}

// makeResizer(handleEl, {axis: 'y'|'x', key, get, set, min, max, invert})
//   get()  -> current size in px
//   set(px) -> apply size
//   invert: drag toward negative axis grows the target (left-edge handles)
export function makeResizer(handle, options) {
  const axis = options.axis || 'y';
  let startPos = 0;
  let startSize = 0;

  function clamp(value) {
    return Math.max(options.min ?? 120, Math.min(options.max ?? 2000, value));
  }

  handle.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    handle.setPointerCapture(event.pointerId);
    handle.classList.add('dragging');
    startPos = axis === 'y' ? event.clientY : event.clientX;
    startSize = options.get();
  });
  handle.addEventListener('pointermove', (event) => {
    if (!handle.classList.contains('dragging')) return;
    const pos = axis === 'y' ? event.clientY : event.clientX;
    let delta = pos - startPos;
    if (options.invert) delta = -delta;
    options.set(clamp(startSize + delta));
  });
  const stop = (event) => {
    if (!handle.classList.contains('dragging')) return;
    handle.classList.remove('dragging');
    try { handle.releasePointerCapture(event.pointerId); } catch (error) {}
    if (options.key) persist(options.key, options.get());
  };
  handle.addEventListener('pointerup', stop);
  handle.addEventListener('pointercancel', stop);
}
