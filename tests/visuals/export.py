"""Static HTML export for the chart view.

Bundles the ES modules from static/js by concatenating them in dependency
order with import/export lines stripped (modules keep unique top-level names
to make this safe — enforced by check_bundle), inlining the vendored
lightweight-charts build, both stylesheets, and the chart payload. The result
is a single self-contained file that works over file:// with no server and no
network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

VISUALS_DIR = Path(__file__).resolve().parent
STATIC_DIR = VISUALS_DIR / "static"
VENDOR_JS = STATIC_DIR / "vendor" / "lightweight-charts-4.2.3.standalone.production.js"

# Chart-view modules in dependency order. Replay modules are deliberately
# excluded — replay needs the server.
MODULE_ORDER = (
    "theme.js",
    "util.js",
    "resize.js",
    "sync.js",
    "tapePane.js",
    "indicatorPane.js",
    "markers.js",
    "chartPanel.js",
)

_IMPORT_LINE = re.compile(r"^import\s.*$", re.MULTILINE)
_EXPORT_PREFIX = re.compile(r"^export\s+(?=(?:async\s+)?function|const|let|class)", re.MULTILINE)
_TOP_LEVEL_DECL = re.compile(r"^(?:async\s+)?(?:function|const|let|class)\s+([A-Za-z_$][\w$]*)", re.MULTILINE)


def _strip_module(source: str) -> str:
    source = _IMPORT_LINE.sub("", source)
    return _EXPORT_PREFIX.sub("", source)


def build_bundle_js() -> str:
    parts = []
    for name in MODULE_ORDER:
        source = (STATIC_DIR / "js" / name).read_text(encoding="utf-8")
        parts.append(f"// ---- {name} ----\n{_strip_module(source)}")
    bundle = "\n".join(parts)
    check_bundle(bundle)
    return bundle


def check_bundle(bundle: str) -> None:
    """Concatenation is only safe while top-level names stay unique."""
    seen: dict[str, int] = {}
    for match in _TOP_LEVEL_DECL.finditer(bundle):
        name = match.group(1)
        seen[name] = seen.get(name, 0) + 1
    duplicates = sorted(name for name, count in seen.items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate top-level declarations in static bundle: {', '.join(duplicates)}")


def write_static_html(payload: dict, output: Path) -> None:
    payload_json = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    title = payload.get("title", "Algo Charts").replace("<", "&lt;").replace(">", "&gt;")
    css = "\n".join(
        (STATIC_DIR / "css" / name).read_text(encoding="utf-8")
        for name in ("theme.css", "layout.css")
    )
    vendor = VENDOR_JS.read_text(encoding="utf-8").replace("</", "<\\/")
    bundle = build_bundle_js().replace("</", "<\\/")
    window_label = payload.get("windowLabel", "")
    source_label = payload.get("sourceLabel", "")
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
{css}
  </style>
  <script>{vendor}</script>
</head>
<body>
  <header>
    <div><h1>{title}</h1><div class="meta">Mouse wheel zooms, drag pans, crosshair inspects values.</div></div>
    <div class="meta" style="text-align:right">{window_label}<br />{source_label}</div>
  </header>
  <main id="charts"></main>
  <script>
    const DATA = {payload_json};
    const root = document.getElementById('charts');
{bundle}
    for (const frame of DATA.frames) createChartPanel(root, DATA, frame);
  </script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")
