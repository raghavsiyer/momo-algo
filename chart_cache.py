from __future__ import annotations

import csv
import json
import logging
import queue
import threading
import time
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from market_data.tradier_rest import OHLCVCandle
from technical_analysis.indicators import add_bollinger_bands, add_ema, add_ma, add_macd, add_vwap

logger = logging.getLogger(__name__)


LOG_DIR = Path(__file__).resolve().parent / "logs"
CHART_CACHE_DIR = "chart_cache"

SOURCE_STREAM = "stream"
SOURCE_POLLED = "polled"
SOURCE_TICK = "tick"
SOURCE_LEGACY_TICK_10S = "algo_tick_cache_v2"

TIMEFRAME_ALIASES = {
    "10": "10s",
    "10s": "10s",
    "10sec": "10s",
    "10second": "10s",
    "1": "1m",
    "1m": "1m",
    "1min": "1m",
    "1minute": "1m",
    "2": "2m",
    "2m": "2m",
    "2min": "2m",
    "2minute": "2m",
    "5": "5m",
    "5m": "5m",
    "5min": "5m",
    "5minute": "5m",
}

RAW_TIMESALES_COLUMNS = (
    "symbol",
    "ts",
    "price",
    "size",
    "bid",
    "ask",
    "exchange",
    "seq",
    "flag",
    "session",
    "source",
    # Merged-tape prints carry their side and feed so replays use what the
    # live tape decided (empty for prints from before, or from IBKR alone).
    "side",
    "origin",
)

# L2 order book snapshots (IBKR depth). Ladders are JSON-encoded
# [[price, size], ...] best-first, so the schema is independent of the
# configured number of depth rows (IBKR_L2_ROWS).
BOOK_CACHE_COLUMNS = (
    "symbol",
    "ts",
    "bids",
    "asks",
    "source",
)

CANDLE_CACHE_COLUMNS = (
    "symbol",
    "timeframe",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr_14",
    "vwap",
    "bar_vwap",
    "ema_9",
    "ema_20",
    "ma_50",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "rvol_5m_rolling",
    "source",
)

LEGACY_TICK_10S_COLUMNS = ("symbol", "ts", "open", "high", "low", "close", "volume", "source")

NEAR_HOD_WINDOW_CACHE_COLUMNS = (
    "symbol",
    "replay_start_ts",
    "replay_end_ts",
    "start_ts",
    "end_ts",
    "start_price",
    "start_hod",
    "start_atr_14",
    "start_volume",
    "threshold_atrs",
    "source",
)


def canonical_timeframe(timeframe: str) -> str:
    key = str(timeframe).strip().lower()
    try:
        return TIMEFRAME_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported chart cache timeframe: {timeframe}") from exc


def cache_dir(log_dir: Path = LOG_DIR) -> Path:
    return Path(log_dir) / CHART_CACHE_DIR


def timesales_cache_path(log_dir: Path, symbol: str, ymd: str) -> Path:
    return cache_dir(log_dir) / f"timesales_{symbol.upper()}_{ymd}.csv"


def book_cache_path(log_dir: Path, symbol: str, ymd: str) -> Path:
    return cache_dir(log_dir) / f"book_{symbol.upper()}_{ymd}.csv"


def candle_cache_path(log_dir: Path, symbol: str, timeframe: str, ymd: str) -> Path:
    tf = canonical_timeframe(timeframe)
    return cache_dir(log_dir) / f"ohlcv_{tf}_{symbol.upper()}_{ymd}.csv"


def legacy_tick_10s_cache_path(log_dir: Path, symbol: str, ymd: str) -> Path:
    return cache_dir(log_dir) / f"tick_10s_ohlcv_{symbol.upper()}_{ymd}.csv"


def near_hod_window_cache_path(log_dir: Path, symbol: str, ymd: str) -> Path:
    return cache_dir(log_dir) / f"near_hod_windows_{symbol.upper()}_{ymd}.csv"


def _empty_candles() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDLE_CACHE_COLUMNS)


def _empty_timesales() -> pd.DataFrame:
    return pd.DataFrame(columns=RAW_TIMESALES_COLUMNS)


def _empty_book() -> pd.DataFrame:
    return pd.DataFrame(columns=BOOK_CACHE_COLUMNS)


def _empty_near_hod_windows() -> pd.DataFrame:
    return pd.DataFrame(columns=NEAR_HOD_WINDOW_CACHE_COLUMNS)


def _ymds_between(start_utc: datetime, end_utc: datetime) -> list[str]:
    start = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
    end = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
    if end < start:
        start, end = end, start
    return [ts.strftime("%Y%m%d") for ts in pd.date_range(start.normalize(), end.normalize(), freq="D")]


def _candles_to_frame(candles: Iterable[OHLCVCandle]) -> pd.DataFrame:
    rows = []
    for candle in candles:
        rows.append({
            "symbol": candle.symbol,
            "ts": candle.ts,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "bar_vwap": candle.vwap,
        })
    return pd.DataFrame(rows)


def _rolling_rvol_5m(df: pd.DataFrame, window: int = 20) -> pd.Series:
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    prior_avg = volume.shift(1).rolling(window=window, min_periods=1).mean()
    rvol = volume / prior_avg
    return rvol.where(prior_avg > 0, 0.0).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


def _rolling_atr_14(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=1).mean().fillna(0.0)
    if not atr.empty:
        atr.iloc[0] = 0.0
    return atr


def prepare_candle_cache_frame(
    bars: pd.DataFrame | Iterable[OHLCVCandle] | None,
    symbol: str,
    timeframe: str,
    source: str | None = None,
) -> pd.DataFrame:
    tf = canonical_timeframe(timeframe)
    if bars is None:
        return _empty_candles()
    raw = _candles_to_frame(bars) if not isinstance(bars, pd.DataFrame) else bars.copy()
    if raw.empty:
        return _empty_candles()
    required = {"ts", "open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        return _empty_candles()

    out = raw.copy()
    out["symbol"] = out.get("symbol", symbol)
    out["symbol"] = out["symbol"].astype(str).str.upper().where(out["symbol"].notna(), symbol.upper())
    out = out.loc[out["symbol"] == symbol.upper()].copy()
    if out.empty:
        return _empty_candles()
    out["timeframe"] = tf
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ts", "open", "high", "low", "close"])
    if out.empty:
        return _empty_candles()
    out["volume"] = out["volume"].fillna(0).astype(int)
    # Stable: rows sharing a timestamp keep submission order so the
    # writer's "last submitted wins" dedup is deterministic.
    out = out.sort_values("ts", kind="stable").reset_index(drop=True)
    out["atr_14"] = _rolling_atr_14(out)

    if "bar_vwap" not in out.columns:
        out["bar_vwap"] = out.get("vwap")
    out["bar_vwap"] = pd.to_numeric(out["bar_vwap"], errors="coerce")
    typical = (out["high"] + out["low"] + out["close"]) / 3.0
    vwap_price = out["bar_vwap"].combine_first(typical)
    vwap_input = out[["ts", "open", "high", "low", "close", "volume"]].copy()
    vwap_input["high"] = vwap_price
    vwap_input["low"] = vwap_price
    vwap_input["close"] = vwap_price
    out["vwap"] = add_vwap(vwap_input)["vwap"]

    out = add_ema(out, 9)
    out = add_ema(out, 20)
    out = add_ma(out, 50)
    if tf == "1m":
        out = add_macd(out)
    if tf == "5m":
        out = add_bollinger_bands(out)
        out["rvol_5m_rolling"] = _rolling_rvol_5m(out)

    if source is not None:
        out["source"] = source
    elif "source" not in out.columns:
        out["source"] = SOURCE_POLLED
    else:
        out["source"] = out["source"].fillna(SOURCE_POLLED).astype(str)

    for col in CANDLE_CACHE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.loc[:, CANDLE_CACHE_COLUMNS].sort_values("ts", kind="stable").reset_index(drop=True)
    return out


def prepare_near_hod_window_cache_frame(
    windows: pd.DataFrame | Iterable[dict[str, Any]] | None,
    symbol: str,
) -> pd.DataFrame:
    if windows is None:
        return _empty_near_hod_windows()
    raw = windows.copy() if isinstance(windows, pd.DataFrame) else pd.DataFrame(list(windows))
    if raw.empty:
        return _empty_near_hod_windows()

    out = raw.copy()
    for col in NEAR_HOD_WINDOW_CACHE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.loc[:, NEAR_HOD_WINDOW_CACHE_COLUMNS].copy()
    out["symbol"] = out["symbol"].fillna(symbol).astype(str).str.upper()
    out = out.loc[out["symbol"] == symbol.upper()].copy()
    if out.empty:
        return _empty_near_hod_windows()

    for col in ("replay_start_ts", "replay_end_ts", "start_ts", "end_ts"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    out["replay_start_ts"] = out["replay_start_ts"].combine_first(out["start_ts"])
    out["replay_end_ts"] = out["replay_end_ts"].combine_first(out["end_ts"])
    out = out.dropna(subset=["replay_start_ts", "replay_end_ts", "start_ts", "end_ts"])
    if out.empty:
        return _empty_near_hod_windows()

    for col in ("start_price", "start_hod", "start_atr_14", "threshold_atrs"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)
    out["start_volume"] = pd.to_numeric(out["start_volume"], errors="coerce").fillna(0).astype(int)
    out["source"] = out["source"].fillna("replay").astype(str).where(out["source"].notna(), "replay")
    out["source"] = out["source"].where(out["source"].astype(bool), "replay")

    return out.sort_values(["start_ts", "end_ts"]).reset_index(drop=True)


def _read_candle_file(path: Path, symbol: str, timeframe: str, legacy_source: str | None = None) -> pd.DataFrame:
    try:
        cached = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return _empty_candles()
    if legacy_source is not None:
        cached["source"] = legacy_source
    return prepare_candle_cache_frame(cached, symbol, timeframe, source=None)


def _read_raw_candle_file(path: Path) -> pd.DataFrame:
    """Rows of a canonical candle file with NO indicator recompute.

    The writer merges these with the incoming bars and runs
    ``prepare_candle_cache_frame`` once on the result, so preparing the
    existing rows first (as ``_read_candle_file`` does) only doubled the
    per-write cost — indicators were recomputed over the whole day twice
    per closed bar.
    """
    try:
        raw = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return _empty_candles()
    if raw.empty:
        return _empty_candles()
    for col in CANDLE_CACHE_COLUMNS:
        if col not in raw.columns:
            raw[col] = pd.NA
    raw["source"] = raw["source"].fillna(SOURCE_POLLED).astype(str)
    return raw.loc[:, list(CANDLE_CACHE_COLUMNS)]


def load_candle_cache(
    symbol: str,
    timeframe: str,
    start_utc: datetime,
    end_utc: datetime,
    log_dir: Path = LOG_DIR,
    ymds: list[str] | None = None,
) -> pd.DataFrame:
    tf = canonical_timeframe(timeframe)
    frames: list[pd.DataFrame] = []
    for ymd in ymds or _ymds_between(start_utc, end_utc):
        canonical = _read_candle_file(candle_cache_path(log_dir, symbol, tf, ymd), symbol, tf)
        if not canonical.empty:
            frames.append(canonical)
        if tf == "10s":
            legacy = _read_candle_file(legacy_tick_10s_cache_path(log_dir, symbol, ymd), symbol, tf, legacy_source=SOURCE_STREAM)
            if not legacy.empty:
                frames.append(legacy)
    if not frames:
        return _empty_candles()
    merged = pd.concat(frames, ignore_index=True)
    merged["ts"] = pd.to_datetime(merged["ts"], utc=True)
    start_ts = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
    end_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
    merged = merged.loc[(merged["ts"] >= start_ts) & (merged["ts"] <= end_ts)].copy()
    source_priority = merged["source"].map({SOURCE_POLLED: 0, SOURCE_STREAM: 1, SOURCE_TICK: 1, SOURCE_LEGACY_TICK_10S: 1}).fillna(0)
    merged["_priority"] = source_priority
    merged = merged.sort_values(["ts", "_priority"]).drop_duplicates(subset=["ts"], keep="last")
    return merged.drop(columns=["_priority"]).sort_values("ts").reset_index(drop=True)


def load_timesales_cache(
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    log_dir: Path = LOG_DIR,
    ymds: list[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ymd in ymds or _ymds_between(start_utc, end_utc):
        path = timesales_cache_path(log_dir, symbol, ymd)
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path, dtype={"flag": str})
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if not raw.empty:
            frames.append(raw)
    if not frames:
        return _empty_timesales()

    merged = pd.concat(frames, ignore_index=True)
    for column in RAW_TIMESALES_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
    merged = merged.loc[:, RAW_TIMESALES_COLUMNS].copy()

    wanted_symbol = symbol.upper()
    merged["symbol"] = merged["symbol"].fillna("").astype(str).str.upper()
    merged = merged.loc[merged["symbol"] == wanted_symbol].copy()
    if merged.empty:
        return _empty_timesales()

    merged["ts"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce")
    for column in ("price", "size", "bid", "ask", "seq"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged = merged.dropna(subset=["ts", "price", "size"]).copy()
    if merged.empty:
        return _empty_timesales()

    merged["price"] = merged["price"].astype(float)
    merged["size"] = merged["size"].fillna(0).astype(int)
    merged["bid"] = merged["bid"].fillna(0.0).astype(float)
    merged["ask"] = merged["ask"].fillna(0.0).astype(float)
    merged["seq"] = merged["seq"].fillna(0).astype(int)
    merged = merged.loc[(merged["price"] > 0) & (merged["size"] > 0)].copy()
    if merged.empty:
        return _empty_timesales()

    start_ts = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
    end_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
    merged = merged.loc[(merged["ts"] >= start_ts) & (merged["ts"] <= end_ts)].copy()
    if merged.empty:
        return _empty_timesales()

    for column in ("exchange", "flag", "session", "source", "side", "origin"):
        merged[column] = merged[column].fillna("").astype(str)
    merged["source"] = merged["source"].where(merged["source"].astype(bool), SOURCE_STREAM)

    merged = (
        merged.sort_values(["ts", "seq"])
        .drop_duplicates(subset=["ts", "seq", "price", "size"], keep="last")
        .reset_index(drop=True)
    )
    return merged.loc[:, RAW_TIMESALES_COLUMNS]


def load_near_hod_window_cache(
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    log_dir: Path = LOG_DIR,
    ymds: list[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ymd in ymds or _ymds_between(start_utc, end_utc):
        path = near_hod_window_cache_path(log_dir, symbol, ymd)
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        prepared = prepare_near_hod_window_cache_frame(raw, symbol)
        if not prepared.empty:
            frames.append(prepared)
    if not frames:
        return _empty_near_hod_windows()

    merged = pd.concat(frames, ignore_index=True)
    start_ts = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
    end_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
    merged = merged.loc[(merged["end_ts"] >= start_ts) & (merged["start_ts"] <= end_ts)].copy()
    if merged.empty:
        return _empty_near_hod_windows()
    merged = merged.drop_duplicates(
        subset=["replay_start_ts", "replay_end_ts", "start_ts", "end_ts"],
        keep="last",
    )
    return merged.loc[:, NEAR_HOD_WINDOW_CACHE_COLUMNS].sort_values("start_ts").reset_index(drop=True)


def write_near_hod_window_cache(
    symbol: str,
    windows: pd.DataFrame | Iterable[dict[str, Any]] | None,
    log_dir: Path = LOG_DIR,
) -> None:
    clean = prepare_near_hod_window_cache_frame(windows, symbol)
    if clean.empty:
        return
    target_dir = cache_dir(log_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    for ymd, group in clean.groupby(clean["start_ts"].dt.strftime("%Y%m%d")):
        path = near_hod_window_cache_path(log_dir, symbol, ymd)
        existing = _empty_near_hod_windows()
        if path.exists():
            try:
                existing = prepare_near_hod_window_cache_frame(pd.read_csv(path), symbol)
            except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
                existing = _empty_near_hod_windows()
        merged = pd.concat([existing, group], ignore_index=True) if not existing.empty else group.copy()
        merged = merged.drop_duplicates(
            subset=["replay_start_ts", "replay_end_ts", "start_ts", "end_ts"],
            keep="last",
        ).sort_values("start_ts")
        output = prepare_near_hod_window_cache_frame(merged, symbol)
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            output.to_csv(tmp_path, index=False)
            tmp_path.replace(path)
        except OSError:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def write_candle_cache(
    symbol: str,
    timeframe: str,
    bars: pd.DataFrame | Iterable[OHLCVCandle] | None,
    log_dir: Path = LOG_DIR,
    source: str = SOURCE_STREAM,
) -> None:
    clean = prepare_candle_cache_frame(bars, symbol, timeframe, source=source)
    if clean.empty:
        return
    target_dir = cache_dir(log_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    tf = canonical_timeframe(timeframe)
    for ymd, group in clean.groupby(clean["ts"].dt.strftime("%Y%m%d")):
        path = candle_cache_path(log_dir, symbol, tf, ymd)
        existing = _read_raw_candle_file(path) if path.exists() else _empty_candles()
        merged = pd.concat([existing, group], ignore_index=True) if not existing.empty else group.copy()
        merged["ts"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce")
        merged = merged.dropna(subset=["ts"])
        source_priority = merged["source"].map({SOURCE_POLLED: 0, SOURCE_STREAM: 1, SOURCE_TICK: 1, SOURCE_LEGACY_TICK_10S: 1}).fillna(0)
        merged["_priority"] = source_priority
        # Stable sort so that, among equal (ts, priority), the row written
        # LAST wins — the writer coalesces several submissions into one call
        # and the newest bar for a timestamp must be the one kept.
        merged = merged.sort_values(["ts", "_priority"], kind="stable").drop_duplicates(subset=["ts"], keep="last")
        output = prepare_candle_cache_frame(merged.drop(columns=["_priority"]), symbol, tf, source=None)
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            output.to_csv(tmp_path, index=False)
            tmp_path.replace(path)
        except OSError:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


def timesale_row_from_event(event: Any, source: str = SOURCE_STREAM) -> dict[str, Any]:
    ts = event.ts
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return {
        "symbol": str(event.symbol).upper(),
        "ts": ts.isoformat(),
        "price": float(event.last),
        "size": int(event.size),
        "bid": float(getattr(event, "bid", 0.0)),
        "ask": float(getattr(event, "ask", 0.0)),
        "exchange": str(getattr(event, "exchange", "")),
        "seq": int(getattr(event, "seq", 0)),
        "flag": str(getattr(event, "flag", "")),
        "session": str(getattr(event, "session", "")),
        "source": source,
        "side": str(getattr(event, "aggressor", None) or ""),
        "origin": str(getattr(event, "origin", "") or ""),
    }


def append_timesales(rows: Iterable[dict[str, Any]], log_dir: Path = LOG_DIR) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        ts = pd.Timestamp(row.get("ts"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        clean = {col: row.get(col, "") for col in RAW_TIMESALES_COLUMNS}
        clean["symbol"] = symbol
        clean["ts"] = ts.isoformat()
        clean["source"] = clean.get("source") or SOURCE_STREAM
        grouped.setdefault((symbol, ts.strftime("%Y%m%d")), []).append(clean)
    if not grouped:
        return
    target_dir = cache_dir(log_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    for (symbol, ymd), group in grouped.items():
        if not group:
            continue
        path = timesales_cache_path(log_dir, symbol, ymd)
        exists = path.exists() and path.stat().st_size > 0
        # A day file started under an older column set keeps its own header
        # (new columns are dropped from its rows) so its CSV stays aligned.
        fieldnames = _existing_header(path) if exists else None
        try:
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=fieldnames or RAW_TIMESALES_COLUMNS, extrasaction="ignore",
                )
                if not exists:
                    writer.writeheader()
                writer.writerows(group)
        except OSError:
            continue


def _existing_header(path: Path) -> list[str] | None:
    try:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), None)
    except OSError:
        return None
    return [col for col in header if col] if header else None


def book_row_from_snapshot(snapshot: Any, source: str = SOURCE_STREAM) -> dict[str, Any]:
    """Flatten an OrderBookSnapshot into a book-cache CSV row.

    Ladders are JSON [[price, size], ...] best-first, already aggregated
    per price by the stream layer.
    """
    ts = getattr(snapshot, "ts", None) or datetime.now(tz=timezone.utc)
    return {
        "symbol": str(getattr(snapshot, "symbol", "")).upper(),
        "ts": pd.Timestamp(ts).isoformat(),
        "bids": json.dumps([[lvl.price, lvl.size] for lvl in getattr(snapshot, "bids", [])]),
        "asks": json.dumps([[lvl.price, lvl.size] for lvl in getattr(snapshot, "asks", [])]),
        "source": source,
    }


def append_book_rows(rows: Iterable[dict[str, Any]], log_dir: Path = LOG_DIR) -> None:
    """Append book-cache rows, grouped per (symbol, day) file."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        ts = pd.Timestamp(row.get("ts"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        clean = {col: row.get(col, "") for col in BOOK_CACHE_COLUMNS}
        clean["symbol"] = symbol
        clean["ts"] = ts.isoformat()
        clean["source"] = clean.get("source") or SOURCE_STREAM
        grouped.setdefault((symbol, ts.strftime("%Y%m%d")), []).append(clean)
    if not grouped:
        return
    target_dir = cache_dir(log_dir)
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    for (symbol, ymd), group in grouped.items():
        if not group:
            continue
        path = book_cache_path(log_dir, symbol, ymd)
        exists = path.exists() and path.stat().st_size > 0
        try:
            with path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=BOOK_CACHE_COLUMNS)
                if not exists:
                    writer.writeheader()
                writer.writerows(group)
        except OSError:
            continue


def load_book_cache(
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
    log_dir: Path = LOG_DIR,
    ymds: list[str] | None = None,
) -> pd.DataFrame:
    """Load cached L2 book snapshots for *symbol* within [start_utc, end_utc].

    Returns a DataFrame with columns (symbol, ts, bids, asks, source) sorted
    by ts; ``bids``/``asks`` are decoded to [[price, size], ...] lists.
    Consumers (backtest) rebuild OrderBookSnapshot objects from these rows.
    """
    frames: list[pd.DataFrame] = []
    for ymd in ymds or _ymds_between(start_utc, end_utc):
        path = book_cache_path(log_dir, symbol, ymd)
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if not raw.empty:
            frames.append(raw)
    if not frames:
        return _empty_book()

    merged = pd.concat(frames, ignore_index=True)
    for column in BOOK_CACHE_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
    merged = merged.loc[:, BOOK_CACHE_COLUMNS].copy()

    wanted_symbol = symbol.upper()
    merged["symbol"] = merged["symbol"].fillna("").astype(str).str.upper()
    merged = merged.loc[merged["symbol"] == wanted_symbol].copy()
    if merged.empty:
        return _empty_book()

    merged["ts"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce", format="ISO8601")
    merged = merged.dropna(subset=["ts"]).copy()
    merged = merged.loc[(merged["ts"] >= start_utc) & (merged["ts"] <= end_utc)].copy()
    if merged.empty:
        return _empty_book()

    def _decode(val: Any) -> list:
        try:
            out = json.loads(val) if isinstance(val, str) and val else []
        except (ValueError, TypeError):
            return []
        return out if isinstance(out, list) else []

    merged["bids"] = merged["bids"].map(_decode)
    merged["asks"] = merged["asks"].map(_decode)
    merged = merged.sort_values("ts", kind="stable").reset_index(drop=True)
    return merged


class ChartCacheWriter:
    """Single background thread that performs all chart-cache disk writes.

    Why: ``write_candle_cache`` reads, merges and rewrites a whole day file
    — 20-70 ms per closed bar — and the live scorers used to call it from
    the shared IBKR stream thread, whose callbacks run synchronously for
    every symbol. Measured 2026-09-03 at ~12% of that thread's CPU, and a
    20-70 ms latency spike for every symbol on each bar close. Handing the
    work to one writer thread takes it off the hot path and also
    serialises writes to a given file (the poll thread and the stream
    thread previously raced on the same ``.tmp`` path for 1m bars).

    Submissions are FIFO. Consecutive candle submissions for the same
    (symbol, timeframe, dir, source) are coalesced into one merge/rewrite,
    which is semantically identical to writing them one after another
    because the writer dedups by timestamp with a stable sort (last wins).
    ``drain`` blocks until everything submitted so far is on disk.
    """

    def __init__(self, name: str = "chart-cache-writer") -> None:
        self._q: queue.Queue = queue.Queue()
        self._cv = threading.Condition()
        self._pending = 0
        self._name = name
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()

    # -- submission --------------------------------------------------------

    def submit_rows(self, fn: Callable[[list, Path], None], rows: list[dict[str, Any]], log_dir: Path) -> None:
        self._submit(("rows", (fn, rows, Path(log_dir))))

    def submit_candles(
        self,
        symbol: str,
        timeframe: str,
        bars: pd.DataFrame | Iterable[OHLCVCandle] | None,
        log_dir: Path,
        source: str,
    ) -> None:
        if bars is None:
            return
        if isinstance(bars, pd.DataFrame):
            payload: Any = bars.copy()
        else:
            # Closed OHLCVCandle objects are never mutated after close, so a
            # shallow list copy pins the set of bars without deep-copying.
            payload = list(bars)
            if not payload:
                return
        self._submit(("candles", (symbol.upper(), canonical_timeframe(timeframe), payload, Path(log_dir), source)))

    def _submit(self, item: tuple) -> None:
        with self._cv:
            self._pending += 1
        self._q.put(item)
        self._ensure_started()

    def _ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, daemon=True, name=self._name)
            self._thread.start()

    # -- completion --------------------------------------------------------

    @property
    def pending(self) -> int:
        with self._cv:
            return self._pending

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until every submission made so far has been written.

        Returns False if *timeout* elapsed first (the writer keeps going).
        """
        with self._cv:
            return self._cv.wait_for(lambda: self._pending == 0, timeout)

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        while True:
            batch = [self._q.get()]
            while True:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                self._process(batch)
            except Exception:  # pragma: no cover - defensive; per-item errors are caught below
                logger.debug("chart-cache writer batch failed", exc_info=True)
            finally:
                with self._cv:
                    self._pending = max(0, self._pending - len(batch))
                    if self._pending == 0:
                        self._cv.notify_all()

    @staticmethod
    def _process(batch: list[tuple]) -> None:
        # Coalesce list-of-candle submissions per key at the position of the
        # first occurrence; DataFrame submissions and row appends are kept
        # as individual steps. Different keys are different files, so the
        # reordering this implies never changes any file's final content.
        candle_lists: dict[tuple, list] = {}
        steps: list[tuple] = []
        for kind, payload in batch:
            if kind == "candles":
                symbol, tf, bars, log_dir, source = payload
                if isinstance(bars, list):
                    key = (symbol, tf, str(log_dir), source)
                    if key not in candle_lists:
                        candle_lists[key] = []
                        steps.append(("candles-list", (symbol, tf, log_dir, source, key)))
                    candle_lists[key].extend(bars)
                else:
                    steps.append(("candles-frame", payload))
            else:
                steps.append((kind, payload))
        for kind, payload in steps:
            try:
                if kind == "rows":
                    fn, rows, log_dir = payload
                    fn(rows, log_dir)
                elif kind == "candles-list":
                    symbol, tf, log_dir, source, key = payload
                    write_candle_cache(symbol, tf, candle_lists[key], log_dir, source=source)
                else:
                    symbol, tf, bars, log_dir, source = payload
                    write_candle_cache(symbol, tf, bars, log_dir, source=source)
            except Exception:
                logger.debug("chart-cache write failed (%s)", kind, exc_info=True)


_writer: ChartCacheWriter | None = None
_writer_lock = threading.Lock()


def get_chart_cache_writer() -> ChartCacheWriter:
    """Process-wide writer shared by every live scorer."""
    global _writer
    if _writer is None:
        with _writer_lock:
            if _writer is None:
                _writer = ChartCacheWriter()
    return _writer


class LiveChartCache:
    """Per-symbol buffer in front of the chart-cache files.

    With ``writer`` set (the live scorers pass the process-wide
    :class:`ChartCacheWriter`) every disk write is handed to the writer
    thread and ``drain`` waits for it. Without a writer — the default, used
    by tests and offline tools — writes happen inline on the calling
    thread exactly as before.
    """

    def __init__(
        self,
        symbol: str,
        log_dir: Path = LOG_DIR,
        raw_flush_size: int = 50,
        raw_flush_seconds: float = 1.0,
        writer: ChartCacheWriter | None = None,
    ) -> None:
        self._symbol = symbol.upper()
        self._log_dir = Path(log_dir)
        self._raw_flush_size = max(1, raw_flush_size)
        self._raw_flush_seconds = max(0.0, raw_flush_seconds)
        self._raw_buffer: list[dict[str, Any]] = []
        self._last_raw_flush = time.monotonic()
        self._book_buffer: list[dict[str, Any]] = []
        self._last_book_flush = time.monotonic()
        self._writer = writer

    def record_timesale(self, event: Any) -> None:
        self._raw_buffer.append(timesale_row_from_event(event, source=SOURCE_STREAM))
        now = time.monotonic()
        if len(self._raw_buffer) >= self._raw_flush_size or now - self._last_raw_flush >= self._raw_flush_seconds:
            self.flush_timesales()

    def flush_timesales(self) -> None:
        if not self._raw_buffer:
            return
        pending = self._raw_buffer
        self._raw_buffer = []
        if self._writer is not None:
            self._writer.submit_rows(append_timesales, pending, self._log_dir)
        else:
            append_timesales(pending, self._log_dir)
        self._last_raw_flush = time.monotonic()

    def record_book(self, snapshot: Any) -> None:
        """Buffer an L2 OrderBookSnapshot (already deduped upstream)."""
        self._book_buffer.append(book_row_from_snapshot(snapshot, source=SOURCE_STREAM))
        now = time.monotonic()
        if len(self._book_buffer) >= self._raw_flush_size or now - self._last_book_flush >= self._raw_flush_seconds:
            self.flush_books()

    def flush_books(self) -> None:
        if not self._book_buffer:
            return
        pending = self._book_buffer
        self._book_buffer = []
        if self._writer is not None:
            self._writer.submit_rows(append_book_rows, pending, self._log_dir)
        else:
            append_book_rows(pending, self._log_dir)
        self._last_book_flush = time.monotonic()

    def record_candles(self, timeframe: str, bars: pd.DataFrame | Iterable[OHLCVCandle] | None, source: str) -> None:
        if self._writer is not None:
            self._writer.submit_candles(self._symbol, timeframe, bars, self._log_dir, source)
        else:
            write_candle_cache(self._symbol, timeframe, bars, self._log_dir, source=source)

    def drain(self, timeout: float | None = None) -> bool:
        """Block until queued writes are on disk (no-op without a writer)."""
        if self._writer is None:
            return True
        return self._writer.drain(timeout)