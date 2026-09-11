#!/usr/bin/env python3
"""
charts.py - Render interactive algo-style intraday charts for a ticker.

Default window is local premarket 03:45-07:00. Before rendering, the full
premarket 1-minute chart is checked with the same move-start detector used by
the live pipeline; if the detected start of move is later than 03:45, charts
are clipped to that move start through 07:00.

The browser is only a viewer. Candles and overlays are computed in Python from
this repo's own data and indicator functions, then serialized into HTML/JSON
powered by TradingView Lightweight Charts.

Usage:
    python tests/visuals/charts.py --serve      # browser UI for ticker/date changes
    python tests/visuals/charts.py AZI          # static 10s, 1m, 5m HTML
    python tests/visuals/charts.py AZI 2m       # static 2m HTML
    make visuals                               # same as --serve
    make visuals AZI 2m
"""

from __future__ import annotations

import argparse
import csv
import re
import webbrowser
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from chart_cache import (
    CHART_CACHE_DIR,
    SOURCE_POLLED,
    SOURCE_STREAM,
    SOURCE_TICK,
    load_candle_cache,
    write_candle_cache,
)
from config import (
    MarketClock,
    ENTRY_CONFIDENCE_MIN,
    SIM_OFI_NORM_MIN,
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_SPEED_MIN,
)
from market_data.provider import get_historical_client
from technical_analysis.indicators import add_bollinger_bands, add_ema, add_ma, add_macd, add_vwap
from technical_analysis.patterns import detect_move_start


LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "tests" / "visuals"
DEFAULT_TIMEFRAMES = ("10s", "1m", "5m")
DEFAULT_START_LOCAL = MarketClock.PREMARKET_WINDOW_START_LOCAL
DEFAULT_END_LOCAL = MarketClock.PREMARKET_WINDOW_END_LOCAL
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8765
CRITERIA_LINE_COLOR = "#22c55e"


def _open_in_browser(target: str | Path) -> None:
    try:
        url = target.resolve().as_uri() if isinstance(target, Path) else target
        webbrowser.open(url)
    except Exception:
        pass


@dataclass(frozen=True)
class TimeframeSpec:
    key: str
    label: str
    rule: str
    seconds: int


@dataclass(frozen=True)
class SessionSpec:
    key: str
    label: str
    start: dtime
    end: dtime
    volume_label: str
    gain_label: str


TIMEFRAMES: dict[str, TimeframeSpec] = {
    "10s": TimeframeSpec("10s", "10 sec", "10s", 10),
    "1m": TimeframeSpec("1m", "1 min", "1min", 60),
    "2m": TimeframeSpec("2m", "2 min", "2min", 120),
    "5m": TimeframeSpec("5m", "5 min", "5min", 300),
}
TF_ALIASES = {
    "10": "10s", "10s": "10s", "10sec": "10s", "10second": "10s",
    "1": "1m", "1m": "1m", "1min": "1m", "1minute": "1m",
    "2": "2m", "2m": "2m", "2min": "2m", "2minute": "2m",
    "5": "5m", "5m": "5m", "5min": "5m", "5minute": "5m",
}
DEFAULT_SESSION = "premarket"
SESSION_SPECS: dict[str, SessionSpec] = {
    "premarket": SessionSpec("premarket", "Premarket", MarketClock.PREMARKET_WINDOW_START_LOCAL, MarketClock.PREMARKET_WINDOW_END_LOCAL, "Premarket Vol", "Premarket % Gainer"),
    "regular": SessionSpec("regular", "Regular Session", dtime(7, 0), dtime(13, 0), "Regular Vol", "Regular % Gainer"),
    "after_hours": SessionSpec("after_hours", "After Hours", dtime(13, 0), dtime(17, 0), "After Hours Vol", "After Hours % Gainer"),
    "1day": SessionSpec("1day", "1 Day", MarketClock.PREMARKET_WINDOW_START_LOCAL, dtime(17, 0), "1 Day Vol", "1 Day % Gainer"),
}
SESSION_ALIASES = {
    "premarket": "premarket",
    "pre_market": "premarket",
    "pre": "premarket",
    "regular": "regular",
    "regular_session": "regular",
    "rth": "regular",
    "after_hours": "after_hours",
    "afterhours": "after_hours",
    "ah": "after_hours",
    "1day": "1day",
    "1_day": "1day",
    "1d": "1day",
    "day": "1day",
}

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"
FLAT_COLOR = "#b0bec5"
POLLED_COLOR = "#7d8590"
BG_COLOR = "#0f1117"
PANEL_COLOR = "#131722"
GRID_COLOR = "#2a2e39"
TEXT_COLOR = "#c9d1d9"
MUTED_COLOR = "#8b949e"
VWAP_COLOR = "#f5c542"
EMA9_COLOR = "#2d7ff9"
EMA20_COLOR = "#ff8f00"
MA50_COLOR = "#b388ff"
BB_COLOR = "#90a4ae"
BUY_MARKER_COLOR = "#22c55e"
SELL_MARKER_COLOR = "#f87171"
MACD_LINE_COLOR = "#4f8cff"
MACD_SIGNAL_COLOR = "#ffb020"
MACD_HIST_UP_COLOR = "#26a69a"
MACD_HIST_DOWN_COLOR = "#ef5350"
RVOL_BAR_COLOR = "#8bd450"
RVOL_BASELINE_COLOR = "#f5c542"
ROLLING_RVOL_5M_WINDOW = 20
CONFIDENCE_PA_COLOR = "#34d399"
TAPE_ACCELERATION_COLOR = "#f59e0b"
TAPE_IMBALANCE_UP_COLOR = "#38bdf8"
TAPE_IMBALANCE_DOWN_COLOR = "#fb7185"
TAPE_SPEED_COLOR = "#c084fc"
TAPE_INDICATOR_BUCKET_SECONDS = 0.25
TAPE_INDICATOR_TIME_MULTIPLIER = int(round(1 / TAPE_INDICATOR_BUCKET_SECONDS))

OVERLAY_COLUMNS: tuple[tuple[str, str, str, int, str], ...] = (
    ("vwap", "VWAP", VWAP_COLOR, 2, "solid"),
    ("ema_9", "EMA 9", EMA9_COLOR, 2, "solid"),
    ("ema_20", "EMA 20", EMA20_COLOR, 2, "solid"),
    ("ma_50", "MA 50", MA50_COLOR, 1, "solid"),
    ("bb_upper", "BB upper", BB_COLOR, 1, "dashed"),
    ("bb_lower", "BB lower", BB_COLOR, 1, "dashed"),
)


def _local_tz() -> tzinfo:
    return MarketClock.PREMARKET_WINDOW_TZ or timezone.utc


def _parse_hhmm(value: str) -> dtime:
    try:
        hour_s, minute_s = value.split(":", 1)
        return dtime(int(hour_s), int(minute_s))
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected HH:MM, got {value!r}") from exc


def _parse_day(value: str | None, local_tz: tzinfo, log_dir: Path | None = None) -> date:
    if value:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc
    if log_dir is not None:
        return _default_data_day(log_dir, local_tz)
    return datetime.now(tz=local_tz).date()


def _dates_from_filename(path: Path) -> set[date]:
    days: set[date] = set()
    for match in re.finditer(r"20\d{6}", path.name):
        try:
            parsed = datetime.strptime(match.group(0), "%Y%m%d").date()
        except ValueError:
            continue
        if parsed.weekday() < 5:
            days.add(parsed)
    return days


def _available_data_days(log_dir: Path) -> list[date]:
    paths = _glob_logs(log_dir, ["tape_*.csv", "scoring_*.csv"])
    paths.extend(_glob_logs(log_dir / CHART_CACHE_DIR, ["*.csv"]))
    days: set[date] = set()
    for path in paths:
        days.update(_dates_from_filename(path))
    return sorted(days)


def _default_data_day(log_dir: Path, local_tz: tzinfo) -> date:
    """Default the chart to today's local date.

    Cache filenames are stamped in UTC, so a session running past 17:00 PT
    spills a few ticks into the next UTC day. Picking the newest filename
    date would open the chart on that empty rollover day, so use the local
    calendar date instead and fall back to the most recent weekday on
    weekends.
    """
    today = datetime.now(tz=local_tz).date()
    while today.weekday() >= 5:
        today -= timedelta(days=1)
    return today


def _window_utc(day: date, start_local: dtime, end_local: dtime, local_tz: tzinfo) -> tuple[datetime, datetime]:
    start = datetime.combine(day, start_local, tzinfo=local_tz)
    end = datetime.combine(day, end_local, tzinfo=local_tz)
    if end <= start:
        end += timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _normalize_timeframe(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower()
    if key in {"", "all"}:
        return None
    if key not in TF_ALIASES:
        allowed = ", ".join(sorted(TIMEFRAMES))
        raise argparse.ArgumentTypeError(f"unknown timeframe {value!r}; expected one of {allowed}")
    return TF_ALIASES[key]


def _normalize_session(value: str | None) -> str:
    if value is None:
        return DEFAULT_SESSION
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"", "default"}:
        return DEFAULT_SESSION
    if key not in SESSION_ALIASES:
        allowed = ", ".join(spec.label for spec in SESSION_SPECS.values())
        raise argparse.ArgumentTypeError(f"unknown session {value!r}; expected one of {allowed}")
    return SESSION_ALIASES[key]


def _session_window_from_query(query: dict[str, list[str]], args: argparse.Namespace) -> tuple[str, dtime, dtime]:
    session = _normalize_session(query.get("session", [getattr(args, "session", DEFAULT_SESSION)])[0])
    spec = SESSION_SPECS[session]
    start_local = _parse_hhmm(query.get("start", [spec.start.strftime("%H:%M")])[0])
    end_local = _parse_hhmm(query.get("end", [spec.end.strftime("%H:%M")])[0])
    return session, start_local, end_local


def _to_utc(value) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _float(row: dict[str, str], *names: str, default: float = 0.0) -> float:
    raw = _first(row, *names)
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(row: dict[str, str], *names: str, default: int = 0) -> int:
    raw = _first(row, *names)
    if raw == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _bool(row: dict[str, str], *names: str) -> bool:
    raw = _first(row, *names).strip().lower()
    return raw in {"1", "true", "yes", "y"}


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _candidate_ymds(day: date, start_utc: datetime, end_utc: datetime, local_tz: tzinfo) -> list[str]:
    days: set[date] = {day, start_utc.date(), end_utc.date()}
    days.add(start_utc.astimezone(local_tz).date())
    days.add(end_utc.astimezone(local_tz).date())
    expanded = set(days)
    for item in days:
        expanded.add(item - timedelta(days=1))
        expanded.add(item + timedelta(days=1))
    return [d.strftime("%Y%m%d") for d in sorted(expanded)]


def _glob_logs(log_dir: Path, patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in log_dir.glob(pattern):
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return sorted(paths)


def _cached_timesale_paths(log_dir: Path, ymds: list[str]) -> list[Path]:
    patterns: list[str] = []
    for ymd in ymds:
        patterns.extend([
            f"*timesale*{ymd}*.csv",
            f"*timesales*{ymd}*.csv",
            f"*tick*{ymd}*.csv",
            f"*raw*tape*{ymd}*.csv",
        ])
    paths = []
    for path in _glob_logs(log_dir, patterns) + _glob_logs(log_dir / CHART_CACHE_DIR, [f"timesales_*_{ymd}.csv" for ymd in ymds]):
        name = path.name.lower()
        if name.startswith("tape_") or name.startswith("scoring_"):
            continue
        if "sim_trades" in name or "setup_signals" in name:
            continue
        paths.append(path)
    return paths


def _load_cached_timesales(symbol: str, start_utc: datetime, end_utc: datetime, log_dir: Path, ymds: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for path in _cached_timesale_paths(log_dir, ymds):
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if _first(row, "symbol", "sym", "ticker").upper() != symbol:
                        continue
                    event_type = _first(row, "event_type", "type").lower()
                    if event_type and event_type not in {"timesale", "trade", "tick"}:
                        continue
                    if _bool(row, "cancel", "cancelled") or _bool(row, "correction", "corrected"):
                        continue
                    ts = _to_utc(_first(row, "ts", "timestamp", "time", "date"))
                    if ts is None or ts < start_utc or ts > end_utc:
                        continue
                    price = _float(row, "last", "price", "last_price")
                    if price <= 0:
                        continue
                    rows.append({
                        "ts": ts,
                        "price": price,
                        "volume": max(1, _int(row, "size", "volume", "last_size", default=1)),
                        "source": "tick",
                    })
        except OSError:
            continue
    return _point_df(rows)


def _load_polled_points(symbol: str, start_utc: datetime, end_utc: datetime, log_dir: Path, ymds: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    patterns = [f"tape_{ymd}.csv" for ymd in ymds] + [f"scoring_{ymd}.csv" for ymd in ymds]
    for path in _glob_logs(log_dir, patterns):
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if _first(row, "symbol", "sym", "ticker").upper() != symbol:
                        continue
                    ts = _to_utc(_first(row, "ts", "timestamp", "time", "date"))
                    if ts is None or ts < start_utc or ts > end_utc:
                        continue
                    price = _float(row, "price", "last", "price_raw", "last_price")
                    if price <= 0:
                        continue
                    volume = _int(row, "buy_vol", default=0) + _int(row, "sell_vol", default=0)
                    if volume <= 0:
                        volume = _int(row, "buying_pressure", default=0) + _int(row, "selling_pressure", default=0)
                    rows.append({"ts": ts, "price": price, "volume": max(1, volume), "source": "snapshot"})
        except OSError:
            continue
    return _point_df(rows)


def _load_premarket_points(start_utc: datetime, end_utc: datetime, log_dir: Path, ymds: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    patterns = [f"tape_{ymd}.csv" for ymd in ymds] + [f"scoring_{ymd}.csv" for ymd in ymds]
    for path in _glob_logs(log_dir, patterns):
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    symbol = _first(row, "symbol", "sym", "ticker").upper()
                    if not symbol:
                        continue
                    ts = _to_utc(_first(row, "ts", "timestamp", "time", "date"))
                    if ts is None or ts < start_utc or ts > end_utc:
                        continue
                    price = _float(row, "price", "last", "price_raw", "last_price")
                    if price <= 0:
                        continue
                    volume = _int(row, "buy_vol", default=0) + _int(row, "sell_vol", default=0)
                    if volume <= 0:
                        volume = _int(row, "buying_pressure", default=0) + _int(row, "selling_pressure", default=0)
                    if volume <= 0:
                        volume = _int(row, "size", "volume", "last_size", default=0)
                    rows.append({"symbol": symbol, "ts": ts, "price": price, "volume": max(0, volume)})
        except OSError:
            continue
    if not rows:
        return pd.DataFrame(columns=["symbol", "ts", "price", "volume"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["ts", "price"]).sort_values(["symbol", "ts"]).reset_index(drop=True)


def _local_time_label(ts, local_tz: tzinfo) -> str:
    return pd.Timestamp(ts).tz_convert(local_tz).strftime("%H:%M:%S")


def _session_ticker_payload(
    chart_day: date,
    session_key: str,
    start_local: dtime,
    end_local: dtime,
    log_dir: Path,
    local_tz: tzinfo,
) -> dict:
    session_key = _normalize_session(session_key)
    session = SESSION_SPECS[session_key]
    start_utc, end_utc = _window_utc(chart_day, start_local, end_local, local_tz)
    day_session = SESSION_SPECS["1day"]
    day_start_utc, day_end_utc = _window_utc(chart_day, day_session.start, day_session.end, local_tz)
    ymds = _candidate_ymds(chart_day, day_start_utc, day_end_utc, local_tz)
    day_points = _load_premarket_points(day_start_utc, day_end_utc, log_dir, ymds)
    points = _load_premarket_points(start_utc, end_utc, log_dir, ymds)
    symbols = set(day_points["symbol"].unique()) if not day_points.empty else set()
    if not points.empty:
        symbols.update(points["symbol"].unique())
    rows: list[dict] = []
    session_groups = {symbol: group.sort_values("ts") for symbol, group in points.groupby("symbol", sort=True)} if not points.empty else {}
    for symbol in sorted(symbols):
        group = session_groups.get(symbol)
        if group is None or group.empty:
            rows.append({
                "symbol": symbol,
                "volume": None,
                "pctGain": None,
                "first": None,
                "last": None,
                "high": None,
                "prints": 0,
                "firstTime": "-",
                "lastTime": "-",
            })
            continue
        prices = group["price"].dropna()
        if prices.empty:
            continue
        first_price = float(prices.iloc[0])
        last_price = float(prices.iloc[-1])
        pct_gain = ((last_price - first_price) / first_price * 100.0) if first_price > 0 else 0.0
        total_volume = int(group["volume"].clip(lower=0).sum())
        rows.append({
            "symbol": symbol,
            "volume": total_volume,
            "pctGain": round(pct_gain, 2),
            "first": round(first_price, 4),
            "last": round(last_price, 4),
            "high": round(float(prices.max()), 4),
            "prints": int(len(group)),
            "firstTime": _local_time_label(group["ts"].iloc[0], local_tz),
            "lastTime": _local_time_label(group["ts"].iloc[-1], local_tz),
        })
    rows.sort(
        key=lambda item: (
            item["pctGain"] is not None,
            item["pctGain"] if item["pctGain"] is not None else float("-inf"),
            item["volume"] is not None,
            item["volume"] if item["volume"] is not None else float("-inf"),
            item["symbol"],
        ),
        reverse=True,
    )
    return {
        "date": chart_day.isoformat(),
        "session": session.key,
        "sessionLabel": session.label,
        "volumeLabel": session.volume_label,
        "gainLabel": session.gain_label,
        "windowLabel": f"{session.label} {start_utc.astimezone(local_tz):%H:%M} -> {end_utc.astimezone(local_tz):%H:%M %Z}",
        "rows": rows,
        "counts": {"symbols": len(rows), "points": int(len(points))},
    }


def _sim_trade_paths(log_dir: Path, ymds: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for base in (log_dir / "sim", log_dir):
        for ymd in ymds:
            path = base / f"sim_trades_{ymd}.csv"
            if path.exists() and path not in seen:
                seen.add(path)
                paths.append(path)
    return sorted(paths)


def _trade_marker_text(row: dict[str, str], side: str, price: float, shares: int) -> str:
    reason = _first(row, "entry_reason" if side == "BUY" else "exit_reason", "reason")
    setup_tf = _first(row, "setup_timeframe")
    confidence = _float(row, "live_confidence", "confidence", default=0.0)
    parts = [f"{side} {shares} @ {price:.2f}"]
    if side == "BUY":
        if setup_tf:
            parts.append(f"pullback_{setup_tf}")
        elif reason:
            parts.append(reason)
    else:
        if reason:
            parts.append(reason)
    if confidence > 0:
        parts.append(f"live {confidence:.0f}")
    return " | ".join(parts)


def _load_trade_fills(symbol: str, start_utc: datetime, end_utc: datetime, log_dir: Path, ymds: list[str]) -> list[dict]:
    fills: list[dict] = []
    for path in _sim_trade_paths(log_dir, ymds):
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if _first(row, "symbol", "sym", "ticker").upper() != symbol:
                        continue
                    ts = _to_utc(_first(row, "ts", "timestamp", "time", "date"))
                    if ts is None or ts < start_utc or ts > end_utc:
                        continue
                    side = _first(row, "side", "action").upper()
                    if side not in {"BUY", "SELL"}:
                        continue
                    price = _float(row, "fill_price", "price", "last", "last_price")
                    if price <= 0:
                        continue
                    shares = max(0, _int(row, "shares", "size", "qty", "quantity", default=0))
                    fills.append({
                        "ts": ts,
                        "side": side,
                        "price": price,
                        "shares": shares,
                        "text": _trade_marker_text(row, side, price, shares),
                    })
        except OSError:
            continue
    return sorted(fills, key=lambda fill: fill["ts"])


def _load_confidence_series(
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    log_dir: Path,
    ymds: list[str],
) -> pd.DataFrame:
    """Load confidence, PA, and tape indicator series from tape logs for a symbol."""
    rows: list[dict] = []
    patterns = [f"tape_{ymd}.csv" for ymd in ymds]
    for path in _glob_logs(log_dir, patterns):
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if _first(row, "symbol", "sym", "ticker").upper() != symbol:
                        continue
                    ts = _to_utc(_first(row, "ts", "timestamp", "time", "date"))
                    if ts is None or ts < start_utc or ts > end_utc:
                        continue
                    acceleration_ready_raw = _first(row, "pressure_acceleration_ema_ready", "pressure_acceleration_ready")
                    acceleration_ready = _bool(row, "pressure_acceleration_ema_ready", "pressure_acceleration_ready") if acceleration_ready_raw else True
                    depth_active_raw = _first(row, "depth_active")
                    depth_active = _bool(row, "depth_active") if depth_active_raw else False
                    rows.append({
                        "ts": ts,
                        "live_confidence": _float(row, "live_confidence", default=0.0),
                        "pa_10s": _float(row, "pa_10s", default=0.0),
                        "pa_1m": _float(row, "pa_1m", default=0.0),
                        "pa_2m": _float(row, "pa_2m", default=0.0),
                        "pa_5m": _float(row, "pa_5m", default=0.0),
                        "tape_acceleration": _float(
                            row,
                            "pressure_acceleration_ema_z",
                            "pressure_acceleration_z",
                            "pressure_acceleration",
                            default=0.0,
                        ) if acceleration_ready else None,
                        "tape_imbalance": _float(row, "trade_imbalance", default=0.0),
                        "tape_speed": _float(row, "tape_pps", "entry_speed", default=0.0),
                        # Gate-source read first (EMA kernel — matches the
                        # SIM_OFI_NORM_MIN threshold drawn on the pane);
                        # older logs only carry the boxcar column.
                        "ofi_normalized": _float(row, "ofi_norm_ema", "ofi_normalized", default=0.0) if depth_active else None,
                    })
        except OSError:
            continue
    if not rows:
        return pd.DataFrame(columns=["ts"] + _OVERLAY_SERIES_COLS)
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # Scale PA scores from 0-1 to 0-100 to share axis with live_confidence
    for col in ("pa_10s", "pa_1m", "pa_2m", "pa_5m"):
        df[col] = df[col] * 100.0
    return df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)


_CONFIDENCE_COLS = ["live_confidence", "pa_10s", "pa_1m", "pa_2m", "pa_5m"]
_TAPE_INDICATOR_COLS = ["tape_acceleration", "tape_imbalance", "tape_speed", "ofi_normalized"]
_OVERLAY_SERIES_COLS = [*_CONFIDENCE_COLS, *_TAPE_INDICATOR_COLS]


def _apply_confidence_overlay(bars: pd.DataFrame, confidence: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Forward-fill confidence and tape series onto bar timestamps for any timeframe."""
    if bars.empty or confidence.empty:
        return bars
    out = bars.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    spec = TIMEFRAMES[timeframe]
    # Snap tape timestamps to the timeframe's bar boundary
    conf = confidence[["ts"] + _OVERLAY_SERIES_COLS].copy()
    conf["ts"] = pd.to_datetime(conf["ts"], utc=True).dt.floor(f"{spec.seconds}s")
    conf = conf.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    merged = pd.merge_asof(
        out.sort_values("ts"),
        conf,
        on="ts",
        direction="backward",
    )
    return merged.sort_values("ts").reset_index(drop=True)


def _write_10s_candle_cache(symbol: str, bars: pd.DataFrame, log_dir: Path) -> None:
    write_candle_cache(symbol, "10s", bars, log_dir, source=SOURCE_STREAM)


def _point_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "price", "volume", "source"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(1).astype(int)
    return df.dropna(subset=["ts", "price"]).sort_values("ts").reset_index(drop=True)


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume", "source"])


def _normalize_candle_df(df: pd.DataFrame | None, symbol: str, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_bars()
    out = df.copy()
    required = {"ts", "open", "high", "low", "close", "volume"}
    if not required.issubset(out.columns):
        return _empty_bars()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out["symbol"] = out.get("symbol", symbol)
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ts", "open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0).astype(int)
    out["source"] = source
    return out[["symbol", "ts", "open", "high", "low", "close", "volume", "source"]].sort_values("ts")


def _fetch_1m_candles(symbol: str, start_utc: datetime, end_utc: datetime, allow_fetch: bool) -> pd.DataFrame:
    if not allow_fetch:
        return _empty_bars()
    try:
        hist = get_historical_client()
        fetch_window = getattr(hist, "get_candles_1min_window", None)
        if fetch_window is not None:
            raw = fetch_window(symbol, start_utc, end_utc)
        else:
            raw = hist.get_candles_1min(symbol)
    except Exception as exc:
        print(f"[{symbol}] 1m historical candle fetch skipped: {exc}")
        return _empty_bars()
    return _clip_bars(_normalize_candle_df(raw, symbol, "polled"), start_utc, end_utc)


def _fetch_10s_candles(symbol: str, start_utc: datetime, end_utc: datetime, allow_fetch: bool) -> pd.DataFrame:
    if not allow_fetch:
        return _empty_bars()
    try:
        hist = get_historical_client()
        fetch_window = getattr(hist, "get_candles_10sec_window", None)
        if fetch_window is None:
            return _empty_bars()
        raw = fetch_window(symbol, start_utc, end_utc)
    except Exception as exc:
        print(f"[{symbol}] 10s historical tick fetch skipped: {exc}")
        return _empty_bars()
    return _clip_bars(_normalize_candle_df(raw, symbol, "polled"), start_utc, end_utc)


def _clip_bars(df: pd.DataFrame, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.to_datetime(df["ts"], utc=True)
    return df.loc[(ts >= pd.Timestamp(start_utc)) & (ts <= pd.Timestamp(end_utc))].copy().reset_index(drop=True)


def _clip_points(df: pd.DataFrame, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    ts = pd.to_datetime(df["ts"], utc=True)
    return df.loc[(ts >= pd.Timestamp(start_utc)) & (ts <= pd.Timestamp(end_utc))].copy().reset_index(drop=True)


def _bars_from_points(symbol: str, points: pd.DataFrame, timeframe: str, source: str) -> pd.DataFrame:
    if points.empty:
        return _empty_bars()
    spec = TIMEFRAMES[timeframe]
    df = points.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    ohlc = df["price"].resample(spec.rule).ohlc().dropna()
    volume = df["volume"].resample(spec.rule).sum()
    bars = ohlc.join(volume.rename("volume"), how="left").dropna(subset=["open"])
    if bars.empty:
        return _empty_bars()
    bars = bars.reset_index()
    bars["symbol"] = symbol
    bars["volume"] = bars["volume"].fillna(0).astype(int)
    bars["source"] = source
    return bars[["symbol", "ts", "open", "high", "low", "close", "volume", "source"]]


def _resample_candles(symbol: str, candles: pd.DataFrame, timeframe: str, source: str = "polled") -> pd.DataFrame:
    if candles.empty:
        return _empty_bars()
    spec = TIMEFRAMES[timeframe]
    df = candles.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    rows = (
        df.set_index("ts")
        .sort_index()
        .resample(spec.rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open"])
        .reset_index()
    )
    if rows.empty:
        return _empty_bars()
    rows["symbol"] = symbol
    rows["source"] = source
    rows["volume"] = rows["volume"].fillna(0).astype(int)
    return rows[["symbol", "ts", "open", "high", "low", "close", "volume", "source"]]


def _merge_prefer_ticks(polled: pd.DataFrame, tick: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not polled.empty:
        p = polled.copy()
        p["_priority"] = 0
        frames.append(p)
    if not tick.empty:
        t = tick.copy()
        t["_priority"] = 1
        frames.append(t)
    if not frames:
        return _empty_bars()
    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.sort_values(["ts", "_priority"]).drop_duplicates(subset=["ts"], keep="last")
    return out.drop(columns=["_priority"]).sort_values("ts").reset_index(drop=True)


def _detect_active_start(full_1m: pd.DataFrame, start_utc: datetime, end_utc: datetime) -> datetime:
    if full_1m.empty:
        return start_utc
    detect_df = full_1m[["ts", "open", "high", "low", "close", "volume"]].copy().reset_index(drop=True)
    move = detect_move_start(detect_df)
    if move is None:
        return start_utc
    try:
        move_start = pd.Timestamp(detect_df.iloc[move.start_idx]["ts"]).to_pydatetime()
    except (IndexError, KeyError):
        return start_utc
    if move_start.tzinfo is None:
        move_start = move_start.replace(tzinfo=timezone.utc)
    else:
        move_start = move_start.astimezone(timezone.utc)
    if start_utc < move_start < end_utc:
        return move_start
    return start_utc


def _with_overlays(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if bars.empty:
        return bars
    df = bars.copy().sort_values("ts").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_vwap(df)
    df = add_ema(df, 9)
    df = add_ema(df, 20)
    df = add_ma(df, 50)
    if timeframe == "1m":
        df = add_macd(df)
    if timeframe == "5m":
        df = add_bollinger_bands(df)
        df = _add_rolling_rvol_5m(df)
    return df


def _point_vwap_by_10s(points: pd.DataFrame) -> pd.Series:
    if points.empty:
        return pd.Series(dtype="float64")
    df = points.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["ts", "price"])
    df = df.loc[df["volume"] > 0]
    if df.empty:
        return pd.Series(dtype="float64")
    tick_df = pd.DataFrame({
        "ts": df["ts"],
        "high": df["price"],
        "low": df["price"],
        "close": df["price"],
        "volume": df["volume"],
    })
    tick_df = add_vwap(tick_df)
    if tick_df.empty or "vwap" not in tick_df.columns:
        return pd.Series(dtype="float64")
    series = tick_df.set_index(pd.to_datetime(tick_df["ts"], utc=True))["vwap"].resample(TIMEFRAMES["10s"].rule).last().ffill()
    return series.dropna()


def _apply_10s_point_vwap(bars: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    if bars.empty or points.empty:
        return bars
    point_vwap = _point_vwap_by_10s(points)
    if point_vwap.empty:
        return bars
    out = bars.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    mapped = out["ts"].map(point_vwap)
    if mapped.notna().any():
        out["vwap"] = mapped.combine_first(out.get("vwap"))
    return out


def _with_10s_overlays(bars: pd.DataFrame, vwap_points: pd.DataFrame) -> pd.DataFrame:
    cached_vwap = pd.Series(dtype="float64")
    if not bars.empty and "vwap" in bars.columns:
        cached_vwap = pd.Series(pd.to_numeric(bars["vwap"], errors="coerce").to_numpy(), index=pd.to_datetime(bars["ts"], utc=True))
    df = _with_overlays(bars, "10s")
    if not df.empty and not cached_vwap.empty and cached_vwap.notna().any():
        mapped = pd.to_datetime(df["ts"], utc=True).map(cached_vwap)
        if mapped.notna().any():
            df["vwap"] = mapped.combine_first(df.get("vwap"))
    return _apply_10s_point_vwap(df, vwap_points)


def _vwap_points_for_10s(cached_ticks: pd.DataFrame, polled_points: pd.DataFrame) -> pd.DataFrame:
    return cached_ticks


def _add_rolling_rvol_5m(df: pd.DataFrame, window: int = ROLLING_RVOL_5M_WINDOW) -> pd.DataFrame:
    out = df.copy()
    volume = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    prior_avg = volume.shift(1).rolling(window=window, min_periods=1).mean()
    rvol = volume / prior_avg
    rvol = rvol.where(prior_avg > 0, 0.0).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    out["rvol_5m_rolling"] = rvol
    return out


def _build_chart_frames(
    symbol: str,
    timeframes: tuple[str, ...],
    end_utc: datetime,
    active_start_utc: datetime,
    cached_ticks: pd.DataFrame,
    cached_candles: dict[str, pd.DataFrame],
    tick_10s_bars: pd.DataFrame,
    polled_points: pd.DataFrame,
    polled_1m: pd.DataFrame,
    fetched_10s: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    active_ticks = _clip_points(cached_ticks, active_start_utc, end_utc)
    active_tick_10s = _clip_bars(tick_10s_bars, active_start_utc, end_utc)
    active_points = _clip_points(polled_points, active_start_utc, end_utc)
    active_cached_1m = _clip_bars(cached_candles.get("1m", _empty_bars()), active_start_utc, end_utc)
    active_1m = _merge_prefer_ticks(_clip_bars(polled_1m, active_start_utc, end_utc), active_cached_1m)
    active_10s = _clip_bars(fetched_10s, active_start_utc, end_utc)
    vwap_points_10s = _vwap_points_for_10s(cached_ticks, polled_points)

    for timeframe in timeframes:
        active_cached = _clip_bars(cached_candles.get(timeframe, _empty_bars()), active_start_utc, end_utc)
        if timeframe == "10s":
            tick_bars = active_tick_10s if not active_tick_10s.empty else _merge_prefer_ticks(active_cached, _bars_from_points(symbol, active_ticks, timeframe, SOURCE_TICK))
            polled_bars = active_10s
            frames[timeframe] = _with_10s_overlays(_merge_prefer_ticks(polled_bars, tick_bars), vwap_points_10s)
            continue
        elif timeframe == "1m":
            tick_bars = _bars_from_points(symbol, active_ticks, timeframe, SOURCE_TICK)
            polled_bars = active_1m
            if polled_bars.empty and not active_tick_10s.empty:
                polled_bars = _resample_candles(symbol, active_tick_10s, timeframe, SOURCE_STREAM)
            if polled_bars.empty:
                polled_bars = _bars_from_points(symbol, active_points, timeframe, "snapshot")
        else:
            tick_bars = _bars_from_points(symbol, active_ticks, timeframe, SOURCE_TICK)
            resampled = _resample_candles(symbol, active_1m, timeframe, SOURCE_POLLED)
            polled_bars = resampled
            if polled_bars.empty and not active_tick_10s.empty:
                polled_bars = _resample_candles(symbol, active_tick_10s, timeframe, SOURCE_STREAM)
            if polled_bars.empty:
                polled_bars = _bars_from_points(symbol, active_points, timeframe, "snapshot")
        frames[timeframe] = _with_overlays(_merge_prefer_ticks(polled_bars, tick_bars), timeframe)
    return frames


def _chart_time_units(ts, local_tz: tzinfo, multiplier: int = 1) -> int:
    local = pd.Timestamp(ts).tz_convert(local_tz).to_pydatetime()
    return int(round(local.replace(tzinfo=timezone.utc).timestamp() * multiplier))


def _clean_float(value) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _payload_float(value) -> float | None:
    clean = _clean_float(value)
    return None if clean is None else round(clean, 6)


def _indicator_line_data(df: pd.DataFrame, column: str, local_tz: tzinfo) -> list[dict]:
    if column not in df.columns:
        return []
    values: list[dict] = []
    for ts, raw in zip(df["ts"], df[column]):
        value = _clean_float(raw)
        if value is None:
            continue
        values.append({"time": _chart_time_units(ts, local_tz), "value": round(value, 6)})
    return values


def _indicator_line_data_padded(df: pd.DataFrame, column: str, local_tz: tzinfo) -> list[dict]:
    """Like _indicator_line_data but emits whitespace entries for NaN values.

    This ensures the sub-chart time axis covers the same range as the parent
    chart, so logical-range sync stays aligned.
    """
    if column not in df.columns:
        return []
    values: list[dict] = []
    for ts, raw in zip(df["ts"], df[column]):
        t = _chart_time_units(ts, local_tz)
        value = _clean_float(raw)
        if value is None:
            values.append({"time": t})  # whitespace — reserves time axis space
        else:
            values.append({"time": t, "value": round(value, 6)})
    return values


def _indicator_histogram_data(df: pd.DataFrame, column: str, local_tz: tzinfo, up_color: str, down_color: str) -> list[dict]:
    if column not in df.columns:
        return []
    values: list[dict] = []
    for ts, raw in zip(df["ts"], df[column]):
        value = _clean_float(raw)
        if value is None:
            continue
        values.append({"time": _chart_time_units(ts, local_tz), "value": round(value, 6), "color": up_color if value >= 0 else down_color})
    return values


def _tape_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    tape = df.attrs.get("tape_series")
    return tape if isinstance(tape, pd.DataFrame) and not tape.empty else df


def _quarter_second_chart_time(ts, local_tz: tzinfo) -> int:
    return _chart_time_units(ts, local_tz, TAPE_INDICATOR_TIME_MULTIPLIER)


def _quarter_second_indicator_line_data(df: pd.DataFrame, column: str, local_tz: tzinfo) -> list[dict]:
    if column not in df.columns:
        return []
    rows: list[dict] = []
    for ts, raw in zip(df["ts"], df[column]):
        value = _clean_float(raw)
        if value is None:
            continue
        rows.append({"time": _quarter_second_chart_time(ts, local_tz), "value": round(value, 6)})
    if not rows:
        return []
    return list({row["time"]: row for row in rows}.values())


def _quarter_second_indicator_histogram_data(df: pd.DataFrame, column: str, local_tz: tzinfo, up_color: str, down_color: str) -> list[dict]:
    if column not in df.columns:
        return []
    rows: list[dict] = []
    for ts, raw in zip(df["ts"], df[column]):
        value = _clean_float(raw)
        if value is None:
            continue
        rows.append({
            "time": _quarter_second_chart_time(ts, local_tz),
            "value": round(value, 6),
            "color": up_color if value >= 0 else down_color,
        })
    if not rows:
        return []
    return list({row["time"]: row for row in rows}.values())


def _series_times(items: Iterable[dict]) -> set[int]:
    times: set[int] = set()
    for item in items:
        raw = item.get("time")
        if isinstance(raw, int):
            times.add(raw)
    return times


def _indicator_times(panes: Iterable[dict]) -> set[int]:
    times: set[int] = set()
    for pane in panes:
        histogram = pane.get("histogram")
        if histogram:
            times.update(_series_times(histogram.get("data") or []))
        for line in pane.get("lines") or []:
            times.update(_series_times(line.get("data") or []))
    return times


def _indicator_panes_payload(df: pd.DataFrame, timeframe: str, local_tz: tzinfo) -> list[dict]:
    panes: list[dict] = []
    if timeframe == "1m" and {"macd_line", "macd_signal", "macd_hist"}.issubset(df.columns):
        panes.append({
            "kind": "macd",
            "title": "MACD",
            "height": 170,
            "priceFormat": {"type": "price", "precision": 4, "minMove": 0.0001},
            "baseline": 0.0,
            "baselineLabel": "0",
            "histogram": {"label": "Hist", "data": _indicator_histogram_data(df, "macd_hist", local_tz, MACD_HIST_UP_COLOR, MACD_HIST_DOWN_COLOR)},
            "lines": [
                {"label": "MACD", "color": MACD_LINE_COLOR, "width": 2, "data": _indicator_line_data_padded(df, "macd_line", local_tz)},
                {"label": "Signal", "color": MACD_SIGNAL_COLOR, "width": 2, "data": _indicator_line_data_padded(df, "macd_signal", local_tz)},
            ],
        })
    if timeframe == "10s" and "live_confidence" in df.columns:
        conf_data = _indicator_line_data_padded(df, "pa_10s", local_tz)
        if conf_data:
            panes.append({
                "kind": "confidence",
                "title": "Live Confidence (10s)",
                "height": 150,
                "secondsVisible": True,
                "priceFormat": {"type": "price", "precision": 1, "minMove": 0.1},
                "baseline": 50.0,
                "baselineLabel": "50",
                "threshold": ENTRY_CONFIDENCE_MIN,
                "thresholdLabel": f"Criteria {ENTRY_CONFIDENCE_MIN:.0f}",
                "histogram": None,
                "lines": [{"label": "PA 10s", "color": CONFIDENCE_PA_COLOR, "width": 2, "data": conf_data}],
            })
    tape_df = _tape_indicator_frame(df)
    if timeframe == "10s" and "tape_acceleration" in tape_df.columns:
        acceleration_data = _quarter_second_indicator_line_data(tape_df, "tape_acceleration", local_tz)
        if acceleration_data:
            panes.append({
                "kind": "tape_acceleration",
                "title": "Tape Acceleration Z",
                "timeMultiplier": TAPE_INDICATOR_TIME_MULTIPLIER,
                "height": 130,
                "secondsVisible": True,
                "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                "baseline": 0.0,
                "baselineLabel": "0",
                "threshold": SIM_PRESSURE_ACCEL_Z,
                "thresholdLabel": f"Criteria {SIM_PRESSURE_ACCEL_Z:.2f}",
                "histogram": None,
                "lines": [{"label": "EMA Z", "color": TAPE_ACCELERATION_COLOR, "width": 2, "data": acceleration_data}],
            })
    if timeframe == "10s" and "tape_imbalance" in tape_df.columns:
        imbalance_data = _quarter_second_indicator_histogram_data(tape_df, "tape_imbalance", local_tz, TAPE_IMBALANCE_UP_COLOR, TAPE_IMBALANCE_DOWN_COLOR)
        if imbalance_data:
            panes.append({
                "kind": "tape_imbalance",
                "title": "Tape Imbalance",
                "timeMultiplier": TAPE_INDICATOR_TIME_MULTIPLIER,
                "height": 130,
                "secondsVisible": True,
                "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                "baseline": 0.0,
                "baselineLabel": "0",
                "threshold": SIM_PRESSURE_BUY_RATIO_MIN,
                "thresholdLabel": f"Criteria {SIM_PRESSURE_BUY_RATIO_MIN:.2f}",
                "histogram": {"label": "Imbalance", "data": imbalance_data},
                "lines": [],
            })
    if timeframe == "10s" and "tape_speed" in tape_df.columns:
        speed_data = _quarter_second_indicator_line_data(tape_df, "tape_speed", local_tz)
        if speed_data:
            panes.append({
                "kind": "tape_speed",
                "title": "Tape Speed",
                "timeMultiplier": TAPE_INDICATOR_TIME_MULTIPLIER,
                "height": 130,
                "secondsVisible": True,
                "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                "baseline": 0.0,
                "baselineLabel": "0",
                "threshold": SIM_PRESSURE_SPEED_MIN,
                "thresholdLabel": f"Criteria {SIM_PRESSURE_SPEED_MIN:.0f}",
                "histogram": None,
                "lines": [{"label": "Prints/sec", "color": TAPE_SPEED_COLOR, "width": 2, "data": speed_data}],
            })
    if timeframe == "10s" and "ofi_normalized" in tape_df.columns:
        ofi_data = _quarter_second_indicator_line_data(tape_df, "ofi_normalized", local_tz)
        if ofi_data:
            panes.append({
                "kind": "ofi_normalized",
                "title": "OFI (Normalized)",
                "timeMultiplier": TAPE_INDICATOR_TIME_MULTIPLIER,
                "height": 130,
                "secondsVisible": True,
                "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                "baseline": 0.0,
                "baselineLabel": "0",
                "threshold": SIM_OFI_NORM_MIN,
                "thresholdLabel": f"Criteria {SIM_OFI_NORM_MIN:.1f}",
                "histogram": None,
                "lines": [{"label": "OFI", "color": TAPE_ACCELERATION_COLOR, "width": 2, "data": ofi_data}],
            })
    if timeframe == "1m" and "pa_1m" in df.columns:
        conf_data = _indicator_line_data_padded(df, "pa_1m", local_tz)
        if conf_data:
            panes.append({
                "kind": "confidence",
                "title": "Live Confidence (1m)",
                "height": 150,
                "priceFormat": {"type": "price", "precision": 1, "minMove": 0.1},
                "baseline": 50.0,
                "baselineLabel": "50",
                "histogram": None,
                "lines": [{"label": "PA 1m", "color": CONFIDENCE_PA_COLOR, "width": 2, "data": conf_data}],
            })
    if timeframe == "2m" and "pa_2m" in df.columns:
        conf_data = _indicator_line_data_padded(df, "pa_2m", local_tz)
        if conf_data:
            panes.append({
                "kind": "confidence",
                "title": "Live Confidence (2m)",
                "height": 150,
                "priceFormat": {"type": "price", "precision": 1, "minMove": 0.1},
                "baseline": 50.0,
                "baselineLabel": "50",
                "histogram": None,
                "lines": [{"label": "PA 2m", "color": CONFIDENCE_PA_COLOR, "width": 2, "data": conf_data}],
            })
    if timeframe == "5m" and "pa_5m" in df.columns:
        conf_data = _indicator_line_data_padded(df, "pa_5m", local_tz)
        if conf_data:
            panes.append({
                "kind": "confidence",
                "title": "Live Confidence (5m)",
                "height": 150,
                "priceFormat": {"type": "price", "precision": 1, "minMove": 0.1},
                "baseline": 50.0,
                "baselineLabel": "50",
                "histogram": None,
                "lines": [{"label": "PA 5m", "color": CONFIDENCE_PA_COLOR, "width": 2, "data": conf_data}],
            })
    if timeframe == "5m" and "rvol_5m_rolling" in df.columns:
        panes.append({
            "kind": "rvol",
            "title": f"5m Rolling RVOL ({ROLLING_RVOL_5M_WINDOW})",
            "height": 150,
            "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
            "baseline": 1.0,
            "baselineLabel": "1.0",
            "histogram": {"label": "RVOL", "data": _indicator_histogram_data(df, "rvol_5m_rolling", local_tz, RVOL_BAR_COLOR, RVOL_BAR_COLOR)},
            "lines": [],
        })
    return panes


def _trade_marker_time(ts: datetime, timeframe: str, local_tz: tzinfo) -> int:
    spec = TIMEFRAMES[timeframe]
    local = pd.Timestamp(ts).tz_convert(local_tz).floor(f"{spec.seconds}s").to_pydatetime()
    return int(round(local.replace(tzinfo=timezone.utc).timestamp()))


def _align_marker_time(raw_time: int, candle_times: list[int]) -> int | None:
    if not candle_times:
        return None
    idx = bisect_left(candle_times, raw_time)
    if idx < len(candle_times):
        return candle_times[idx]
    return candle_times[-1]


def _trade_markers_for_frame(trades: list[dict], df: pd.DataFrame, timeframe: str, local_tz: tzinfo) -> list[dict]:
    if df.empty or not trades:
        return []
    candle_times = [_chart_time_units(ts, local_tz) for ts in df["ts"]]
    markers: list[dict] = []
    for fill in trades:
        raw_time = _trade_marker_time(fill["ts"], timeframe, local_tz)
        marker_time = _align_marker_time(raw_time, candle_times)
        if marker_time is None:
            continue
        is_buy = fill["side"] == "BUY"
        markers.append({
            "time": marker_time,
            "side": fill["side"],
            "price": round(float(fill["price"]), 6),
            "color": BUY_MARKER_COLOR if is_buy else SELL_MARKER_COLOR,
            "text": fill["text"],
        })
    return markers


def _series_payload(df: pd.DataFrame, timeframe: str, local_tz: tzinfo, trades: list[dict]) -> dict:
    candles: list[dict] = []
    volume: list[dict] = []
    readout: list[dict] = []
    for row in df.itertuples(index=False):
        open_ = float(row.open)
        high = float(row.high)
        low = float(row.low)
        close = float(row.close)
        is_polled_10s = timeframe == "10s" and getattr(row, "source", "") == "polled"
        if is_polled_10s:
            candle_color = POLLED_COLOR
        elif close > open_:
            candle_color = UP_COLOR
        elif close < open_:
            candle_color = DOWN_COLOR
        else:
            candle_color = FLAT_COLOR
        timestamp = _chart_time_units(row.ts, local_tz)
        row_volume = int(max(0, float(row.volume)))
        candles.append({
            "time": timestamp,
            "open": round(open_, 6),
            "high": round(high, 6),
            "low": round(low, 6),
            "close": round(close, 6),
            "volume": row_volume,
            "color": candle_color,
            "borderColor": candle_color,
            "wickColor": candle_color,
        })
        volume.append({"time": timestamp, "value": row_volume, "color": candle_color + "99"})
        readout.append({
            "time": timestamp,
            "open": round(open_, 6),
            "high": round(high, 6),
            "low": round(low, 6),
            "close": round(close, 6),
            "volume": row_volume,
            "ema_9": _payload_float(getattr(row, "ema_9", None)),
            "ema_20": _payload_float(getattr(row, "ema_20", None)),
            "vwap": _payload_float(getattr(row, "vwap", None)),
            "ma_50": _payload_float(getattr(row, "ma_50", None)),
        })

    overlays: dict[str, dict] = {}
    for column, label, color, width, style in OVERLAY_COLUMNS:
        if column not in df.columns:
            continue
        values = []
        for ts, raw in zip(df["ts"], df[column]):
            value = _clean_float(raw)
            if value is None:
                continue
            values.append({"time": _chart_time_units(ts, local_tz), "value": round(value, 6)})
        if values:
            overlays[column] = {"label": label, "color": color, "width": width, "style": style, "data": values}

    markers = _trade_markers_for_frame(trades, df, timeframe, local_tz)
    indicators = _indicator_panes_payload(df, timeframe, local_tz)
    stream_sources = {SOURCE_TICK, SOURCE_STREAM}
    tick_count = int(df["source"].astype(str).isin(stream_sources).sum()) if timeframe == "10s" and "source" in df.columns else 0
    polled_count = int((df["source"].astype(str) == SOURCE_POLLED).sum()) if timeframe == "10s" and "source" in df.columns else len(df)
    return {
        "timeframe": timeframe,
        "label": TIMEFRAMES[timeframe].label,
        "is10s": timeframe == "10s",
        "candles": candles,
        "volume": volume,
        "readout": readout,
        "overlays": overlays,
        "indicators": indicators,
        "markers": markers,
        "counts": {"tick": tick_count, "polled": polled_count, "fills": len(markers), "total": len(df)},
    }


def _empty_series_payload(timeframe: str, reason: str) -> dict:
    return {
        "timeframe": timeframe,
        "label": TIMEFRAMES[timeframe].label,
        "is10s": timeframe == "10s",
        "empty": True,
        "noDataReason": reason,
        "candles": [],
        "volume": [],
        "readout": [],
        "overlays": {},
        "indicators": [],
        "markers": [],
        "counts": {"tick": 0, "polled": 0, "fills": 0, "total": 0},
    }


def _default_output(symbol: str, day: date, timeframes: tuple[str, ...]) -> Path:
    tf_label = "_".join(timeframes)
    return OUT_DIR / f"{symbol}_{day.strftime('%Y%m%d')}_{tf_label}.html"


def _timeframes_for(timeframe: str | None) -> tuple[str, ...]:
    return (timeframe,) if timeframe else DEFAULT_TIMEFRAMES


def _payload_from_frames(
    symbol: str,
    frames: dict[str, pd.DataFrame],
    trades: list[dict],
    local_tz: tzinfo,
    start_utc: datetime,
    end_utc: datetime,
    full_start_utc: datetime,
) -> dict:
    if not frames:
        raise ValueError(f"No chartable data found for {symbol} in the selected window.")
    move_label = ""
    if start_utc > full_start_utc:
        move_label = f" | clipped to move start {start_utc.astimezone(local_tz):%H:%M:%S %Z}"
    return {
        "symbol": symbol,
        "title": f"{symbol} algo charts",
        "windowLabel": (
            f"{start_utc.astimezone(local_tz):%Y-%m-%d %H:%M:%S %Z}"
            f" -> {end_utc.astimezone(local_tz):%H:%M:%S %Z}{move_label}"
        ),
        "sourceLabel": "Default uses cached live polled/streamed data only; allow new polling enables historical API fetches for all timeframes",
        "frames": [
            _series_payload(df, tf, local_tz, trades) if not df.empty else _empty_series_payload(
                tf,
                "No cached stream/tick data found for this 10s period." if tf == "10s" else "No cached chart data found for this period.",
            )
            for tf, df in frames.items()
        ],
    }


def _build_chart_payload(
    symbol: str,
    timeframe: str | None,
    chart_day: date,
    start_local: dtime,
    end_local: dtime,
    log_dir: Path,
    allow_fetch: bool,
    use_move_start: bool,
    local_tz: tzinfo,
) -> tuple[dict, datetime, datetime, tuple[str, ...]]:
    symbol = symbol.strip().upper()
    if not symbol:
        raise ValueError("Ticker is required.")
    full_start_utc, end_utc = _window_utc(chart_day, start_local, end_local, local_tz)
    timeframes = _timeframes_for(timeframe)
    ymds = _candidate_ymds(chart_day, full_start_utc, end_utc, local_tz)

    cached_ticks = _load_cached_timesales(symbol, full_start_utc, end_utc, log_dir, ymds)
    polled_points = _load_polled_points(symbol, full_start_utc, end_utc, log_dir, ymds)
    trade_fills = _load_trade_fills(symbol, full_start_utc, end_utc, log_dir, ymds)
    cache_timeframes = tuple(dict.fromkeys(tf for tf in (*timeframes, "1m", "10s") if tf in {"1m", "10s"}))
    cached_candles = {tf: load_candle_cache(symbol, tf, full_start_utc, end_utc, log_dir, ymds=ymds) for tf in cache_timeframes}
    cached_tick_10s = cached_candles.get("10s", _empty_bars())
    raw_tick_10s = _bars_from_points(symbol, cached_ticks, "10s", "tick") if "10s" in timeframes and not cached_ticks.empty else _empty_bars()
    tick_10s_bars = _merge_prefer_ticks(cached_tick_10s, raw_tick_10s) if not raw_tick_10s.empty else cached_tick_10s
    if not raw_tick_10s.empty:
        _write_10s_candle_cache(symbol, tick_10s_bars, log_dir)
    fetched_1m = _fetch_1m_candles(symbol, full_start_utc, end_utc, allow_fetch=allow_fetch)
    cached_1m = cached_candles.get("1m", _empty_bars())
    polled_1m = _merge_prefer_ticks(fetched_1m, cached_1m) if not cached_1m.empty else fetched_1m
    fetched_10s = _fetch_10s_candles(symbol, full_start_utc, end_utc, allow_fetch=allow_fetch and "10s" in timeframes)
    full_1m = _merge_prefer_ticks(
        polled_1m if not polled_1m.empty else _bars_from_points(symbol, polled_points, "1m", "polled"),
        _bars_from_points(symbol, cached_ticks, "1m", "tick"),
    )
    active_start_utc = full_start_utc
    if use_move_start:
        active_start_utc = _detect_active_start(full_1m, full_start_utc, end_utc)

    frames = _build_chart_frames(
        symbol,
        timeframes,
        end_utc,
        active_start_utc,
        cached_ticks,
        cached_candles,
        tick_10s_bars,
        polled_points,
        polled_1m,
        fetched_10s,
    )
    # Overlay live confidence / per-TF PA scores onto all frames from tape logs
    confidence_df = _load_confidence_series(symbol, active_start_utc, end_utc, log_dir, ymds)
    if not confidence_df.empty:
        for tf in list(frames.keys()):
            if not frames[tf].empty:
                frames[tf] = _apply_confidence_overlay(frames[tf], confidence_df, tf)
                if tf == "10s":
                    frames[tf].attrs["tape_series"] = confidence_df[["ts"] + _TAPE_INDICATOR_COLS].copy()
    active_trades = [fill for fill in trade_fills if active_start_utc <= fill["ts"] <= end_utc]
    payload = _payload_from_frames(symbol, frames, active_trades, local_tz, active_start_utc, end_utc, full_start_utc)
    return payload, active_start_utc, end_utc, timeframes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render cached/polled intraday charts for a ticker")
    parser.add_argument("symbol", nargs="?", help="Ticker symbol, e.g. AZI. Omit to launch the local browser UI")
    parser.add_argument("timeframe", nargs="?", type=_normalize_timeframe, help="Optional timeframe: 10s, 1m, 2m, or 5m")
    parser.add_argument("--date", help="Local chart date, YYYY-MM-DD. Defaults to the most recent weekday with cached data")
    parser.add_argument("--session", type=_normalize_session, default=DEFAULT_SESSION, help="Session window for server defaults: premarket, regular, after_hours, or 1day")
    parser.add_argument("--start", type=_parse_hhmm, default=DEFAULT_START_LOCAL, help=f"Local start time, default {DEFAULT_START_LOCAL:%H:%M}")
    parser.add_argument("--end", type=_parse_hhmm, default=DEFAULT_END_LOCAL, help=f"Local end time, default {DEFAULT_END_LOCAL:%H:%M}")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="Directory containing tape/timesale logs")
    parser.add_argument("--save", type=Path, help="Output HTML path. Defaults to tests/visuals/<symbol>_<date>_<tf>.html")
    parser.add_argument("--serve", action="store_true", help="Start a local browser UI for changing ticker/date/timeframe")
    parser.add_argument("--host", default=DEFAULT_SERVER_HOST, help=f"Server host for --serve, default {DEFAULT_SERVER_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help=f"Server port for --serve, default {DEFAULT_SERVER_PORT}")
    parser.add_argument("--allow-fetch", action="store_true", help="Allow new historical/API polling for all timeframes. Disabled by default")
    parser.add_argument("--no-open", action="store_true", help="Save/open without launching a browser")
    parser.add_argument("--move-start", action="store_true", help="Clip chart window to detected move start (off by default)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    local_tz = _local_tz()
    if args.serve or not args.symbol:
        import server

        server.serve(args, local_tz, _open_in_browser)
        return

    symbol = args.symbol.strip().upper()
    chart_day = _parse_day(args.date, local_tz, args.log_dir)
    try:
        payload, active_start_utc, end_utc, timeframes = _build_chart_payload(
            symbol,
            args.timeframe,
            chart_day,
            args.start,
            args.end,
            args.log_dir,
            allow_fetch=args.allow_fetch,
            use_move_start=args.move_start,
            local_tz=local_tz,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    import export

    output = args.save or _default_output(symbol, chart_day, timeframes)
    export.write_static_html(payload, output)
    print(f"Saved {output}")
    print(
        f"Window: {active_start_utc.astimezone(local_tz):%Y-%m-%d %H:%M:%S %Z}"
        f" -> {end_utc.astimezone(local_tz):%H:%M:%S %Z}"
    )
    if not args.no_open:
        _open_in_browser(output)


if __name__ == "__main__":
    main()
