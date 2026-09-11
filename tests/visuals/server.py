"""HTTP server for the visuals site.

Serves the static ES-module frontend from tests/visuals/static/ and exposes
JSON endpoints backed by the payload builders in charts.py. All financial
computation stays in Python — the browser is only a viewer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from collections import OrderedDict
from datetime import date, tzinfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC_ROOT = Path(__file__).resolve().parent / "static"

CONTENT_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".map": "application/json",
}

# Chart payloads re-parse the day's CSVs from scratch, which costs seconds on
# busy days. Cache them keyed by request params, invalidated by the mtimes of
# every source file that can feed the payload.
_PAYLOAD_CACHE: OrderedDict[tuple, tuple[tuple, dict]] = OrderedDict()
_PAYLOAD_CACHE_MAX = 16
_PAYLOAD_CACHE_LOCK = threading.Lock()

# One replay build thread per bundle id. The builder itself also holds a
# process-global lock, so concurrent requests just queue.
_REPLAY_BUILDS: dict[str, threading.Thread] = {}
_REPLAY_BUILDS_LOCK = threading.Lock()
_BUNDLE_ID_RE = re.compile(r"^[A-Z0-9._-]+$", re.IGNORECASE)


def _send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _send_file(handler: BaseHTTPRequestHandler, path: Path, cache_control: str) -> None:
    stat = path.stat()
    etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
    if handler.headers.get("If-None-Match") == etag:
        handler.send_response(304)
        handler.send_header("ETag", etag)
        handler.end_headers()
        return
    data = path.read_bytes()
    content_type = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
    handler.send_response(200)
    handler.send_header("Content-Type", f"{content_type}; charset=utf-8" if content_type.startswith("text/") else content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("ETag", etag)
    handler.send_header("Cache-Control", cache_control)
    handler.end_headers()
    handler.wfile.write(data)


def _serve_static(handler: BaseHTTPRequestHandler, rel_path: str) -> None:
    target = (STATIC_ROOT / rel_path).resolve()
    static_root = STATIC_ROOT.resolve()
    if static_root not in target.parents and target != static_root:
        _send_json(handler, 404, {"error": "Not found"})
        return
    if not target.is_file():
        _send_json(handler, 404, {"error": "Not found"})
        return
    # Vendor files carry a version in the filename, so they can cache hard;
    # app files revalidate on every load via ETag.
    cache_control = "max-age=86400" if "vendor" in target.parts else "no-cache"
    _send_file(handler, target, cache_control)


def _stat_key(path: Path) -> tuple | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _chart_source_fingerprint(charts, log_dir: Path, symbol: str, ymds: list[str]) -> tuple:
    """Every file whose contents can influence a chart payload for symbol+day."""
    keys: list[tuple] = []
    cache_dir = log_dir / charts.CHART_CACHE_DIR
    for ymd in ymds:
        for name in (f"tape_{ymd}.csv", f"scoring_{ymd}.csv"):
            key = _stat_key(log_dir / name)
            if key:
                keys.append(key)
        key = _stat_key(log_dir / "sim" / f"sim_trades_{ymd}.csv")
        if key:
            keys.append(key)
        key = _stat_key(log_dir / f"sim_trades_{ymd}.csv")
        if key:
            keys.append(key)
        for path in sorted(cache_dir.glob(f"*_{symbol}_{ymd}.csv")):
            key = _stat_key(path)
            if key:
                keys.append(key)
    return tuple(keys)


def _availability(charts, chart_cache, log_dir: Path, symbol: str, ymd: str) -> dict:
    return {
        "tape": chart_cache.timesales_cache_path(log_dir, symbol, ymd).exists(),
        "book": chart_cache.book_cache_path(log_dir, symbol, ymd).exists(),
        "candles": any(
            chart_cache.candle_cache_path(log_dir, symbol, tf, ymd).exists()
            for tf in ("10s", "1m", "2m", "5m")
        ),
    }


def _meta_payload(charts, args: argparse.Namespace, local_tz: tzinfo) -> dict:
    day = charts._parse_day(args.date, local_tz, args.log_dir)
    available_days = charts._available_data_days(args.log_dir)
    bound_days = [*available_days, day]
    session_windows = {
        key: {"start": spec.start.strftime("%H:%M"), "end": spec.end.strftime("%H:%M"), "label": spec.label}
        for key, spec in charts.SESSION_SPECS.items()
    }
    tz = charts.MarketClock.PREMARKET_WINDOW_TZ
    tz_label = "local"
    tz_offset_minutes = 0
    if tz is not None:
        try:
            from datetime import datetime

            now = datetime.now().astimezone(tz)
            tz_label = tz.tzname(now) or "local"
            offset = now.utcoffset()
            tz_offset_minutes = int(offset.total_seconds() // 60) if offset else 0
        except Exception:
            tz_label = str(tz)
    return {
        "defaults": {
            "symbol": (args.symbol or "").strip().upper(),
            "date": day.isoformat(),
            "timeframe": args.timeframe or "all",
            "session": getattr(args, "session", charts.DEFAULT_SESSION),
            "start": charts.DEFAULT_START_LOCAL.strftime("%H:%M"),
            "end": charts.DEFAULT_END_LOCAL.strftime("%H:%M"),
            "allowFetch": bool(args.allow_fetch),
            "moveStart": bool(args.move_start),
        },
        "sessionWindows": session_windows,
        "dateLimits": {
            "min": min(bound_days).isoformat() if bound_days else "",
            "max": max(bound_days).isoformat() if bound_days else "",
        },
        "availableDays": [item.isoformat() for item in available_days],
        "tzLabel": tz_label,
        "tzOffsetMinutes": tz_offset_minutes,
    }


def _cached_chart_payload(charts, args, local_tz, symbol, timeframe, chart_day, start_local, end_local, allow_fetch, use_move_start) -> dict:
    start_utc, end_utc = charts._window_utc(chart_day, start_local, end_local, local_tz)
    ymds = charts._candidate_ymds(chart_day, start_utc, end_utc, local_tz)
    cache_key = (symbol, timeframe, chart_day.isoformat(), str(start_local), str(end_local), allow_fetch, use_move_start)
    fingerprint = _chart_source_fingerprint(charts, args.log_dir, symbol, ymds)
    with _PAYLOAD_CACHE_LOCK:
        hit = _PAYLOAD_CACHE.get(cache_key)
        if hit and hit[0] == fingerprint:
            _PAYLOAD_CACHE.move_to_end(cache_key)
            return hit[1]
    payload, _, _, _ = charts._build_chart_payload(
        symbol,
        timeframe,
        chart_day,
        start_local,
        end_local,
        args.log_dir,
        allow_fetch=allow_fetch,
        use_move_start=use_move_start,
        local_tz=local_tz,
    )
    # The build may itself write back 10s candles, so re-stat after building.
    fingerprint = _chart_source_fingerprint(charts, args.log_dir, symbol, ymds)
    with _PAYLOAD_CACHE_LOCK:
        _PAYLOAD_CACHE[cache_key] = (fingerprint, payload)
        _PAYLOAD_CACHE.move_to_end(cache_key)
        while len(_PAYLOAD_CACHE) > _PAYLOAD_CACHE_MAX:
            _PAYLOAD_CACHE.popitem(last=False)
    return payload


def _replay_cache_dir(log_dir: Path) -> Path:
    return log_dir / "replay_cache"


def _start_replay_build(charts, args, local_tz, query) -> dict:
    """Start (or find) a replay bundle build; returns {bundle_id, state}."""
    from datetime import datetime as _datetime, timezone as _timezone
    from exec.replay_session import ReplaySessionBuilder, bundle_name

    symbol = query.get("symbol", [""])[0].strip().upper()
    if not symbol:
        raise ValueError("symbol required")
    chart_day = charts._parse_day(query.get("date", [None])[0] or args.date, local_tz, args.log_dir)
    start_raw = query.get("start", [""])[0]
    end_raw = query.get("end", [""])[0]
    start_local = charts._parse_hhmm(start_raw) if start_raw else charts.DEFAULT_START_LOCAL
    end_local = charts._parse_hhmm(end_raw) if end_raw else charts.DEFAULT_END_LOCAL
    start_utc = _datetime.combine(chart_day, start_local, tzinfo=local_tz).astimezone(_timezone.utc)
    end_utc = _datetime.combine(chart_day, end_local, tzinfo=local_tz).astimezone(_timezone.utc)
    force = charts._truthy(query.get("force", [""])[0])

    bundle_id = bundle_name(symbol, chart_day, start_utc, end_utc)
    bundle_dir = _replay_cache_dir(args.log_dir) / bundle_id
    status_path = bundle_dir / "status.json"
    state = "new"
    if status_path.exists() and not force:
        try:
            state = json.loads(status_path.read_text(encoding="utf-8")).get("state", "new")
        except (OSError, ValueError):
            state = "new"

    with _REPLAY_BUILDS_LOCK:
        running = _REPLAY_BUILDS.get(bundle_id)
        if running is not None and running.is_alive():
            return {"bundle_id": bundle_id, "state": "building"}
        if state == "ready" and not force:
            return {"bundle_id": bundle_id, "state": "ready"}

        builder = ReplaySessionBuilder(
            symbol, chart_day, start_utc, end_utc,
            log_dir=args.log_dir, force=force,
        )

        def run() -> None:
            try:
                builder.build()
            except Exception:
                pass  # status.json already carries state=error

        thread = threading.Thread(target=run, name=f"replay-build-{bundle_id}", daemon=True)
        _REPLAY_BUILDS[bundle_id] = thread
        thread.start()
    return {"bundle_id": bundle_id, "state": "building"}


def _replay_windows_payload(charts, chart_cache, args, local_tz, query) -> dict:
    """Suggested replay windows: cached near-HOD windows + sim-trade times."""
    from datetime import datetime as _datetime, time as _dtime, timedelta as _timedelta, timezone as _timezone

    symbol = query.get("symbol", [""])[0].strip().upper()
    if not symbol:
        raise ValueError("symbol required")
    chart_day = charts._parse_day(query.get("date", [None])[0] or args.date, local_tz, args.log_dir)
    day_start = _datetime.combine(chart_day, _dtime(0, 0), tzinfo=local_tz).astimezone(_timezone.utc)
    day_end = day_start + _timedelta(days=1)
    ymds = charts._candidate_ymds(chart_day, day_start, day_end, local_tz)

    def hhmm(ts_utc) -> str:
        return ts_utc.astimezone(local_tz).strftime("%H:%M")

    windows: list[dict] = []
    try:
        df = chart_cache.load_near_hod_window_cache(symbol, day_start, day_end, args.log_dir, ymds=ymds)
    except Exception:
        df = None
    if df is not None and not df.empty:
        for row in df.itertuples(index=False):
            start_ts = row.start_ts.to_pydatetime()
            end_ts = row.end_ts.to_pydatetime()
            # Pad so the run-up and resolution are visible around the window.
            start_ts -= _timedelta(minutes=2)
            end_ts += _timedelta(minutes=5)
            windows.append({
                "start": hhmm(start_ts),
                "end": hhmm(end_ts),
                "label": f"near HOD {hhmm(start_ts)}",
                "kind": "near_hod",
            })

    try:
        fills = charts._load_trade_fills(symbol, day_start, day_end, args.log_dir, ymds)
    except Exception:
        fills = []
    for fill in fills:
        if fill.get("side") != "BUY":
            continue
        ts = fill["ts"]
        windows.append({
            "start": hhmm(ts - _timedelta(minutes=5)),
            "end": hhmm(ts + _timedelta(minutes=10)),
            "label": f"entry {hhmm(ts)}",
            "kind": "fill",
        })

    # De-dupe identical windows, keep at most 8, fills first.
    seen: set[tuple] = set()
    unique: list[dict] = []
    for window in sorted(windows, key=lambda item: (item["kind"] != "fill", item["start"])):
        key = (window["start"], window["end"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(window)
    return {"symbol": symbol, "date": chart_day.isoformat(), "windows": unique[:8]}


def _serve_bundle_file(handler, log_dir: Path, bundle_id: str, rel: str) -> None:
    if not _BUNDLE_ID_RE.match(bundle_id):
        _send_json(handler, 404, {"error": "Not found"})
        return
    path = _replay_cache_dir(log_dir) / bundle_id / rel
    if not path.is_file():
        _send_json(handler, 404, {"error": "Not found"})
        return
    if path.suffix == ".gz":
        stat = path.stat()
        etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
        if handler.headers.get("If-None-Match") == etag:
            handler.send_response(304)
            handler.send_header("ETag", etag)
            handler.end_headers()
            return
        data = path.read_bytes()
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-ndjson")
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("ETag", etag)
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.wfile.write(data)
        return
    _send_file(handler, path, "no-cache")


def make_handler(args: argparse.Namespace, local_tz: tzinfo) -> type[BaseHTTPRequestHandler]:
    import charts
    import chart_cache

    class VisualsHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                if path in {"/", "/index.html"}:
                    _send_file(self, STATIC_ROOT / "index.html", "no-cache")
                    return
                if path.startswith("/static/"):
                    _serve_static(self, path[len("/static/"):])
                    return
                if path == "/api/meta":
                    _send_json(self, 200, _meta_payload(charts, args, local_tz))
                    return
                if path == "/api/tickers":
                    query = parse_qs(parsed.query)
                    chart_day = charts._parse_day(query.get("date", [None])[0] or args.date, local_tz, args.log_dir)
                    session, start_local, end_local = charts._session_window_from_query(query, args)
                    payload = charts._session_ticker_payload(chart_day, session, start_local, end_local, args.log_dir, local_tz)
                    ymd = chart_day.strftime("%Y%m%d")
                    for row in payload.get("rows", []):
                        row["data"] = _availability(charts, chart_cache, args.log_dir, row["symbol"], ymd)
                    _send_json(self, 200, payload)
                    return
                if path == "/api/chart":
                    query = parse_qs(parsed.query)
                    symbol = query.get("symbol", [""])[0].strip().upper()
                    timeframe = charts._normalize_timeframe(query.get("timeframe", [None])[0])
                    chart_day = charts._parse_day(query.get("date", [None])[0] or args.date, local_tz, args.log_dir)
                    _, start_local, end_local = charts._session_window_from_query(query, args)
                    payload = _cached_chart_payload(
                        charts, args, local_tz, symbol, timeframe, chart_day, start_local, end_local,
                        charts._truthy(query.get("allow_fetch", [""])[0]),
                        not charts._truthy(query.get("no_move_start", [""])[0]),
                    )
                    _send_json(self, 200, payload)
                    return
                if path == "/api/replay/build":
                    query = parse_qs(parsed.query)
                    _send_json(self, 200, _start_replay_build(charts, args, local_tz, query))
                    return
                if path == "/api/replay/windows":
                    query = parse_qs(parsed.query)
                    _send_json(self, 200, _replay_windows_payload(charts, chart_cache, args, local_tz, query))
                    return
                if path == "/api/replay/find":
                    query = parse_qs(parsed.query)
                    symbol = query.get("symbol", [""])[0].strip().upper()
                    chart_day = charts._parse_day(query.get("date", [None])[0] or args.date, local_tz, args.log_dir)
                    prefix = f"{symbol}_{chart_day.strftime('%Y%m%d')}_"
                    bundles = []
                    cache_root = _replay_cache_dir(args.log_dir)
                    if symbol and cache_root.is_dir():
                        for item in sorted(cache_root.iterdir()):
                            if not item.name.startswith(prefix) or not item.is_dir():
                                continue
                            try:
                                status = json.loads((item / "status.json").read_text(encoding="utf-8"))
                                if status.get("state") != "ready":
                                    continue
                                meta = json.loads((item / "meta.json").read_text(encoding="utf-8"))
                                bundles.append({
                                    "bundle_id": item.name,
                                    "start_ms": meta["window"]["start_ms"],
                                    "end_ms": meta["window"]["end_ms"],
                                })
                            except (OSError, ValueError, KeyError):
                                continue
                    _send_json(self, 200, {"bundles": bundles})
                    return
                replay_match = re.match(r"^/api/replay/([^/]+)/(status|meta|events)$", path)
                if replay_match:
                    bundle_id, kind = replay_match.groups()
                    _serve_bundle_file(self, args.log_dir, bundle_id, f"{kind}.json")
                    return
                chunk_match = re.match(r"^/api/replay/([^/]+)/chunks/(\d+)$", path)
                if chunk_match:
                    bundle_id, index = chunk_match.groups()
                    _serve_bundle_file(self, args.log_dir, bundle_id, f"chunks/{int(index):06d}.ndjson.gz")
                    return
                _send_json(self, 404, {"error": "Not found"})
            except (argparse.ArgumentTypeError, ValueError) as exc:
                _send_json(self, 400, {"error": str(exc)})
            except BrokenPipeError:
                pass
            except Exception as exc:
                _send_json(self, 500, {"error": f"Request failed: {exc}"})

    return VisualsHandler


def serve(args: argparse.Namespace, local_tz: tzinfo, open_browser) -> None:
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args, local_tz))
    host, port = server.server_address[:2]
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{browser_host}:{port}/"
    print(f"Serving chart UI at {url}")
    if not args.no_open:
        open_browser(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping chart server.")
    finally:
        server.server_close()
