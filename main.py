"""
main.py  —  Algorithmic Trader Entry Point

Runs the full pipeline for a manually entered ticker:

    Input  →  Stage 1 (price/vol/rvol)
           →  Stage 2 (float/news/static confidence → watchlist)
           →  Indicators built (all timeframes)
           →  Resistance map built
           →  Dynamic confidence scorer started (background thread)
           →  Eval summary + strengths/warnings printed

Type 'quit' or 'exit' to stop. 'list' shows watchlist.
"""


import json
import logging
import os
import re
import shutil
import sys
import threading
import time as _time
from datetime import datetime, timedelta, timezone
from typing import Optional

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI
from strategy.evaluator import StockEvaluator
from config import (
    TRADIER_ENABLED, TRADIER_API_KEY, IBKR_ENABLED, MARKET_DATA_ENABLED,
    AV_API_KEY, AV_ENABLED,
    FINNHUB_API_KEY, FINNHUB_ENABLED,
    FINBERT_ENABLED,
    BRAVE_SEARCH_API_KEY, BRAVE_ENABLED,
    BOT_NAME, BOT_SUBTITLE,
    _g, _y, _r, _c, _dim, _b,
    SCANNER_MOMO_ENABLED, SCANNER_MOMO_INTERVAL,
    SCANNER_MOMO_INTERVAL_FAST, SCANNER_MOMO_INTERVAL_DEF, SCANNER_MOMO_INTERVAL_SLOW,
    SCANNER_IBKR_ENABLED, SCANNER_IBKR_INTERVAL,
    SCANNER_IBKR_INTERVAL_FAST, SCANNER_IBKR_INTERVAL_DEF, SCANNER_IBKR_INTERVAL_SLOW,
    SCANNER_IBKR_DROP_SECS,
    SCANNER_YT_ENABLED, SCANNER_YT_CHANNEL, SCANNER_YT_URL,
    SCANNER_YT_RESOLVE_RETRY_S,
    SCANNER_YT_INTERVAL_FAST, SCANNER_YT_INTERVAL_DEF, SCANNER_YT_INTERVAL_SLOW,
    SCANNER_YT_MODERNBERT_ENABLED, SCANNER_YT_MODERNBERT_MODE,
    SCANNER_YT_CHAT_LOG_ENABLED,
    CACHE_TTL_DAYS,
    news_freshness_cutoffs,
    MarketClock,
    SIM_ACCOUNT_SIZE,
    SIM_BACKTEST_LOG_DIR,
)
from strategy.confidence_scorer import DynamicScorer
from market_data.provider import (
    get_historical_client, get_shared_stream, shutdown_market_data, stream_enabled,
)
from market_data.tradier_stream import wake_shared_stream
from market_data.momo_client import MomoClient
from market_data.finnhub_client import FinnhubNewsArticle
from market_data.youtube_client import YouTubeLiveChatClient
from market_data.modernbert_client import ModernBERTNewsClassifier
from market_data.chat_logger import ChatShadowLogger
from chart_cache import load_candle_cache, load_timesales_cache
from cache import TickerCache, apply_cached_news
from exec.sim_logger import SimFill, SimTradeLogger, summarize_closed_trades
from exec.setup_logger import SetupSignal, get_setup_signal_logger
from technical_analysis.indicators import build_daily
from technical_analysis.patterns import latest_extended_session_start

from pipeline import (
    run_pipeline, validate_ticker, _detect_move_start_ts,
    _apply_stage1_snapshot, _current_cached_fail_reasons,
    _refresh_cached_stage1_snapshot,
)
from display import _print_stock_summary, print_watchlist, watch_status_line, format_live_weights, format_levels, format_top_gainers, format_tape, format_order_book, set_app, print_levels, print_live_detail
from scanners import _momo_scanner, _ibkr_scanner, _youtube_scanner, _scanner_consumer, _inject_yt_news, _is_soft_failure, _failures_are_soft, RateLimitedSymbolResolver

import queue

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

_STATE_FILE = "logs/state.json"


def _parse_news_time(spec: str) -> Optional[datetime]:
    """Parse a time specification for news injection.

    Supports:
      - Relative: 5m, 2h, 1d, 1d5h, 30M (case-insensitive)
      - Time-of-day: 9:30, 9:30:05 (assumes today ET)
      - Date: 6/3, 6/3/26, 6/3/2026, 6-3, 6-3-26, 6-3-2026
      - Date + time: 6/3 9:30

    Returns a UTC datetime, or None if unparseable.
    """
    spec = spec.strip()
    if not spec:
        return None

    now_utc = datetime.now(tz=timezone.utc)

    # --- Relative time: digits followed by m/h/d units, possibly chained ---
    _rel_pattern = re.compile(r'^(\d+[mhdMHD])+$')
    if _rel_pattern.match(spec):
        total_seconds = 0
        for val, unit in re.findall(r'(\d+)([mhdMHD])', spec):
            n = int(val)
            u = unit.lower()
            if u == 'm':
                total_seconds += n * 60
            elif u == 'h':
                total_seconds += n * 3600
            elif u == 'd':
                total_seconds += n * 86400
        return now_utc - timedelta(seconds=total_seconds)

    # --- Time-of-day: H:MM or H:MM:SS (assumes today in ET) ---
    _time_match = re.match(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?$', spec)
    if _time_match:
        h, m = int(_time_match.group(1)), int(_time_match.group(2))
        s = int(_time_match.group(3)) if _time_match.group(3) else 0
        today_et = MarketClock.now_et().date()
        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        dt_et = datetime(today_et.year, today_et.month, today_et.day, h, m, s, tzinfo=et)
        return dt_et.astimezone(timezone.utc)

    # --- Date with optional time: M/D, M/D/YY, M-D-YYYY, etc. ---
    # Split on space to separate date from optional time
    parts = spec.split(None, 1)
    date_str = parts[0]
    time_str = parts[1] if len(parts) > 1 else None

    _date_match = re.match(r'^(\d{1,2})[/\-](\d{1,2})(?:[/\-](\d{2,4}))?$', date_str)
    if _date_match:
        month = int(_date_match.group(1))
        day = int(_date_match.group(2))
        year_str = _date_match.group(3)
        if year_str:
            year = int(year_str)
            if year < 100:
                year += 2000
        else:
            year = MarketClock.now_et().year

        h, m, s = 9, 30, 0  # default to market open
        if time_str:
            _tm = re.match(r'^(\d{1,2}):(\d{2})(?::(\d{2}))?$', time_str)
            if _tm:
                h, m = int(_tm.group(1)), int(_tm.group(2))
                s = int(_tm.group(3)) if _tm.group(3) else 0

        from zoneinfo import ZoneInfo
        et = ZoneInfo("America/New_York")
        try:
            dt_et = datetime(year, month, day, h, m, s, tzinfo=et)
            return dt_et.astimezone(timezone.utc)
        except ValueError:
            return None

    return None


def _parse_live_time_range(spec: str) -> Optional[tuple[datetime, datetime, str]]:
    """Parse an optional Pacific-time range for the live candle detail view."""
    spec = spec.strip()
    if not spec:
        return None
    spec = re.sub(r"\s*(?:pt|pst|pdt)\s*$", "", spec, flags=re.IGNORECASE).strip()

    time_re = r"(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?"
    match = re.match(rf"^\s*{time_re}\s*(?:-\s*|\s+){time_re}\s*$", spec)
    if not match:
        return None

    def _parts_to_hms(groups: tuple[str | None, str | None, str | None]) -> tuple[int, int, int] | None:
        h = int(groups[0] or 0)
        m = int(groups[1] or 0)
        s = int(groups[2] or 0)
        if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
            return None
        return h, m, s

    start_hms = _parts_to_hms(match.group(1, 2, 3))
    end_hms = _parts_to_hms(match.group(4, 5, 6))
    if start_hms is None or end_hms is None:
        return None

    from zoneinfo import ZoneInfo
    pt = ZoneInfo("America/Los_Angeles")
    today_pt = datetime.now(tz=pt).date()
    start_pt = datetime(today_pt.year, today_pt.month, today_pt.day, *start_hms, tzinfo=pt)
    end_pt = datetime(today_pt.year, today_pt.month, today_pt.day, *end_hms, tzinfo=pt)
    if end_pt <= start_pt:
        end_pt += timedelta(days=1)

    def _label(hms: tuple[int, int, int]) -> str:
        h, m, s = hms
        if s:
            return f"{h:02d}:{m:02d}:{s:02d}"
        if m:
            return f"{h:02d}:{m:02d}"
        return f"{h:02d}:00"

    return (
        start_pt.astimezone(timezone.utc),
        end_pt.astimezone(timezone.utc),
        f"{_label(start_hms)}-{_label(end_hms)} PT",
    )


class _HistoricalLiveSnapshot:
    def __init__(self, candle_map: dict[str, list]):
        self._candle_map = candle_map

    def get_candle_snapshot(self) -> dict[str, list]:
        return {tf_label: list(candles) for tf_label, candles in self._candle_map.items()}


def _extended_session_window_utc() -> tuple[datetime, datetime]:
    now_et = MarketClock.now_et()
    session_date = now_et.date()
    if now_et.time() < MarketClock.EXT_OPEN:
        session_date -= timedelta(days=1)
    start_et = datetime(
        session_date.year,
        session_date.month,
        session_date.day,
        MarketClock.EXT_OPEN.hour,
        MarketClock.EXT_OPEN.minute,
        tzinfo=MarketClock.ET,
    )
    return start_et.astimezone(timezone.utc), datetime.now(tz=timezone.utc)


def _premarket_window_utc() -> tuple[datetime, datetime]:
    return MarketClock.premarket_window_utc()


def _build_historical_live_snapshot(
    symbol: str,
    hist,
    time_range: tuple[datetime, datetime] | None = None,
    full_premarket: bool = False,
    include_10s: bool = True,
) -> Optional[_HistoricalLiveSnapshot]:
    if hist is None:
        return None

    start_utc, end_utc = time_range or (_premarket_window_utc() if full_premarket else _extended_session_window_utc())
    candle_map: dict[str, list] = {}

    try:
        fetch_1m_window = getattr(hist, "get_candles_1min_window", None)
        if fetch_1m_window is not None:
            df_1m = fetch_1m_window(symbol, start_utc, end_utc)
        else:
            df_1m = hist.get_candles_1min(symbol)
    except Exception as exc:
        logger.debug("[%s] historical live 1m fallback failed: %s", symbol, exc)
        df_1m = None

    df_1m = DynamicScorer._normalize_candle_df(df_1m)
    if full_premarket and not df_1m.empty:
        from technical_analysis.patterns import detect_move_start

        move = detect_move_start(df_1m)
        if move is not None:
            try:
                move_start = DynamicScorer._coerce_utc_dt(df_1m.iloc[move.start_idx]["ts"])
            except (IndexError, KeyError):
                move_start = None
            if move_start is not None:
                start_utc = max(start_utc, move_start)
        df_1m = DynamicScorer._clip_df_since(df_1m, start_utc)

    if not df_1m.empty:
        candles_1m = DynamicScorer._df_to_candles(df_1m, "1min")
        if candles_1m:
            candle_map["1m"] = candles_1m

        df_1m_indexed = df_1m.set_index("ts")
        candles_2m = DynamicScorer._resample_indexed(df_1m_indexed, symbol, "2min")
        if candles_2m:
            candle_map["2m"] = candles_2m
        candles_5m = DynamicScorer._resample_indexed(df_1m_indexed, symbol, "5min")
        if candles_5m:
            candle_map["5m"] = candles_5m

    df_10s = None
    if include_10s:
        try:
            if time_range is not None and hasattr(hist, "get_candles_10sec_window"):
                df_10s = hist.get_candles_10sec_window(symbol, start_utc, end_utc)
            elif hasattr(hist, "get_candles_10sec"):
                df_10s = hist.get_candles_10sec(symbol)
        except Exception as exc:
            logger.debug("[%s] historical live 10s fallback failed: %s", symbol, exc)

    df_10s = DynamicScorer._normalize_candle_df(df_10s)
    if not df_10s.empty:
        candles_10s = DynamicScorer._df_to_candles(df_10s, "10sec")
        if candles_10s:
            candle_map["10s"] = candles_10s

    try:
        if hasattr(hist, "get_candles_daily"):
            df_daily = hist.get_candles_daily(symbol, days_back=60)
        else:
            df_daily = None
    except Exception as exc:
        logger.debug("[%s] historical live daily fallback failed: %s", symbol, exc)
        df_daily = None

    df_daily = DynamicScorer._normalize_candle_df(df_daily)
    if not df_daily.empty:
        candles_daily = DynamicScorer._df_to_candles(df_daily, "daily")
        if candles_daily:
            candle_map["1d"] = candles_daily

    if not candle_map:
        return None
    return _HistoricalLiveSnapshot(candle_map)


def _remember_pipeline_failure(rejected: set[str], ticker: str, reasons: list[str], fallback_reason: str = "") -> None:
    if _failures_are_soft(reasons) or _is_soft_failure(fallback_reason or ""):
        rejected.discard(ticker)
    else:
        rejected.add(ticker)


def _load_state() -> dict:
    try:
        with open(_STATE_FILE) as _f:
            return json.load(_f)
    except (OSError, json.JSONDecodeError):
        return {}

def _save_state(updates: dict) -> None:
    state = _load_state()
    state.update(updates)
    tmp_path = _STATE_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as _f:
            json.dump(state, _f)
        os.replace(tmp_path, _STATE_FILE)  # atomic — no torn state.json on crash
    except OSError as exc:
        logger.debug("State save failed: %s", exc)


# ---------------------------------------------------------------------------
# Logging setup — INFO to console, DEBUG available via --debug flag
# ---------------------------------------------------------------------------

class _DynamicStdoutHandler(logging.StreamHandler):
    """Always writes to the *current* sys.stdout so prompt_toolkit's
    patch_stdout context manager can intercept log output and render it
    above the sticky input prompt without interleaving.
    """
    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


class _DedupFilter(logging.Filter):
    """Suppress repeated WARNING+ messages within a cooldown window.

    When the network drops, dozens of identical warnings flood the
    terminal every few seconds.  This filter groups them by a short
    message prefix (stripping verbose HTTP details) and only lets one
    through per *_COOLDOWN* seconds.  When the cooldown expires the
    next message includes a suppressed count.
    """

    _COOLDOWN = 60.0  # seconds between identical warnings

    def __init__(self) -> None:
        super().__init__()
        # key → (last_emitted_monotonic, suppressed_count)
        self._seen: dict[str, tuple[float, int]] = {}

    @staticmethod
    def _key(msg: str) -> str:
        """Derive a short grouping key from the full message."""
        # Strip verbose HTTP pool / URL details that vary per request
        for marker in ("HTTPSConnectionPool", "HTTPConnectionPool"):
            idx = msg.find(marker)
            if idx > 0:
                return msg[:idx].rstrip(": ")
        return msg[:80]

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True

        key = self._key(record.getMessage())
        now = _time.monotonic()

        prev = self._seen.get(key)
        if prev is None:
            # First occurrence — allow and track
            self._seen[key] = (now, 0)
            return True

        last, count = prev
        if now - last < self._COOLDOWN:
            # Inside cooldown — suppress
            self._seen[key] = (last, count + 1)
            return False

        # Cooldown expired — allow and annotate
        self._seen[key] = (now, 0)
        if count > 0:
            # Pre-format to avoid breaking %-style arg substitution
            record.msg = f"{record.getMessage()}  (+{count} suppressed)"
            record.args = None
        return True


_log_handler = _DynamicStdoutHandler()
_log_handler.setFormatter(logging.Formatter(
    fmt=f"  \033[2m%(asctime)s  %(levelname)-7s\033[0m  %(message)s",
    datefmt="%H:%M:%S",
))
_log_handler.addFilter(_DedupFilter())
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_log_handler)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live weights display — rendered inside the prompt_toolkit bottom toolbar
# ---------------------------------------------------------------------------

# Set of ticker symbols whose weights are currently displayed in the toolbar.
_weights_active: set[str] = set()

# Set of ticker symbols whose levels are currently displayed in the toolbar.
_levels_active: set[str] = set()

# Live rank display — rendered in bottom toolbar when active.
_rank_active: bool = False

# Live IBKR top-gainers display — rendered in bottom toolbar when active.
_top_active: bool = False
_TOP_LIMIT_DEFAULT: int = 10
_top_limit: int = _TOP_LIMIT_DEFAULT  # `top N` overrides for the current view

# Live order-flow displays, symbol -> how many rows/levels to render.
# Both read straight off the symbol's DynamicScorer, so they only exist for
# tickers already being streamed — neither opens a subscription.
# 0 = "everything streamed": tape shows the full rolling print window, ofi
# shows the full ladder the depth line delivers. An explicit N caps rows.
_TAPE_ROWS_DEFAULT: int = 0
_OFI_LEVELS_DEFAULT: int = 0
_tape_active: dict[str, int] = {}   # `<TICKER> tape [rows]`
_ofi_active: dict[str, int] = {}    # `<TICKER> ofi [levels]`
# Last successfully rendered flow block per (kind, symbol). try_*_view()
# returns None whenever the stream thread holds the scorer lock — which on
# a busy tape is a large fraction of frames, since prints both trigger the
# redraw and hold that lock. Reusing the previous frame keeps the block's
# height stable; collapsing to a one-line "refreshing..." placeholder and
# back is what read as flicker.
_flow_frame_cache: dict[tuple[str, str], str] = {}

# Toolbar invalidation — scorers push redraws instead of polling.
_toolbar_app = None                # set once PromptSession starts
_toolbar_last_invalidate: float = 0.0
_TOOLBAR_MIN_INTERVAL: float = 0.1  # max 10 redraws/sec


def _invalidate_toolbar() -> None:
    """Thread-safe throttled toolbar redraw trigger (called from scorer threads)."""
    global _toolbar_last_invalidate
    import time
    now = time.monotonic()
    if now - _toolbar_last_invalidate < _TOOLBAR_MIN_INTERVAL:
        return
    _toolbar_last_invalidate = now
    app = _toolbar_app
    if app is not None:
        try:
            app.invalidate()
        except Exception:
            pass


def _on_ibkr_scan() -> None:
    """Repaint the toolbar when a new IBKR scan lands (called from the
    scanner thread).  Gated on the `top` display so a closed view never
    costs a redraw at the scanner's own cadence."""
    if _top_active:
        _invalidate_toolbar()


# ---------------------------------------------------------------------------
# Scanner command handler
# ---------------------------------------------------------------------------

_SCANNER_ACTIONS = frozenset({"stop", "go", "start", "fast", "def", "slow", "interval", "poll"})


def _scanner_cmd(
    action: str,
    name: str,
    paused: threading.Event,
    force_poll: threading.Event,
    interval: list,
    speeds: dict[str, int],
    state_key: str,
    *,
    on_stop=None,
    on_go=None,
    has_auto: bool = False,
    auto_hint: str = "",
    log_name: str | None = None,
) -> None:
    """Handle stop/go/poll/interval/fast/def/slow commands for any scanner."""
    _log = log_name or name
    if action == "stop":
        if paused.is_set():
            print(f"  {_dim(f'{name} scanner already paused')}")
        else:
            paused.set()
            if on_stop:
                on_stop()
            logger.info("[scanner:%s] Paused", _log)
    elif action in ("go", "start"):
        if not paused.is_set():
            print(f"  {_dim(f'{name} scanner already running')}")
        else:
            if on_go:
                on_go()
            paused.clear()
            logger.info("[scanner:%s] Resumed", _log)
    elif action == "poll":
        if paused.is_set():
            print(f"  {_y('⚠')}  scanner is paused — use '{name} go' first")
        else:
            force_poll.set()
            print(f"  {_dim('forcing poll...')}")
    elif action == "interval":
        cur = interval[0]
        if has_auto and cur is None:
            print(f"  {name} poll interval  auto  ({auto_hint})")
        else:
            labels = {v: k for k, v in speeds.items()}
            suffix = f"  ({labels[cur]})" if cur in labels else ""
            print(f"  {name} poll interval  {cur}s{suffix}")
    elif action == "auto" and has_auto:
        interval[0] = None
        _save_state({state_key: None})
        print(f"  {_dim(f'{name} poll interval → auto (server hint)')}")
    elif action.startswith("interval "):
        try:
            new_val = int(action.split(maxsplit=1)[1])
            if new_val < 1:
                raise ValueError
        except (IndexError, ValueError):
            print(f"  {_y('⚠')}  Usage: {name} interval <seconds>")
            return
        interval[0] = new_val
        _save_state({state_key: new_val})
        print(f"  {_dim(f'{name} poll interval → {new_val}s ({new_val}s)')}")
    elif action in speeds:
        interval[0] = speeds[action]
        _save_state({state_key: speeds[action]})
        print(f"  {_dim(f'{name} poll interval → {speeds[action]}s ({action})')}")


def _fmt_sim_ts(ts: Optional[datetime]) -> str:
    if ts is None:
        return "--"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(MarketClock.local_tz()).strftime("%H:%M:%S")


def _fmt_pnl(value: float) -> str:
    return (_g if value >= 0 else _r)(f"${value:+.2f}")


def _fmt_pnl_pct(value: float) -> str:
    return (_g if value >= 0 else _r)(f"{value:+.2%}")


def _fmt_setup_timeframe(timeframe: str) -> str:
    if not timeframe:
        return "--"
    return {
        "10sec": "10s",
        "1min": "1m",
        "2min": "2m",
        "5min": "5m",
    }.get(timeframe, timeframe)


def _day_setup_signals(date_str: str) -> list[SetupSignal]:
    return get_setup_signal_logger().day_signals(date_str)


def _print_setup_signals(day_str: str, signals: list[SetupSignal], ticker: str | None = None) -> None:
    if ticker is not None:
        signals = [s for s in signals if s.symbol == ticker]

    if not signals:
        if ticker is None:
            print(f"  {_dim(f'No setup signals logged for {day_str}.')}")
        else:
            print(f"  {_dim(f'No setup signals logged for {ticker} on {day_str}.')}")
        print()
        return

    approved = sum(1 for s in signals if s.signal_approved)
    entered = sum(1 for s in signals if s.entered)
    title = "Setup Signals" if ticker is None else f"Setup Signals {ticker}"
    print(
        f"\n  {_b(title)} {_dim(day_str)} ({len(signals)})  "
        f"{_dim(f'approved={approved} entered={entered} not_approved={len(signals) - approved}') }"
    )
    for signal in signals:
        if signal.entered:
            status = _g("ENTERED")
        elif signal.signal_approved:
            status = _g("APPROVED")
        elif signal.status == "FOUND":
            status = _y("FOUND")
        else:
            status = _r("NOT APPROVED")
        setup_tf = f"/{signal.setup_timeframe}" if signal.setup_timeframe else ""
        range_text = "--"
        if signal.approved_min_price > 0 and signal.approved_max_price > 0:
            range_text = f"${signal.approved_min_price:.2f}-${signal.approved_max_price:.2f}"
        approval_text = ""
        if signal.signal_approved:
            approval_text = f"  approval={_fmt_sim_ts(signal.approval_ts)} ${signal.approval_price:.2f}"
        entry_text = ""
        if signal.entered:
            entry_text = f"  entry={_fmt_sim_ts(signal.entry_ts)} ${signal.entry_price:.2f} x{signal.entry_shares}"
        print(
            f"    {(_g if signal.signal_approved else _r)(signal.symbol)}  {status}  "
            f"signal={_fmt_sim_ts(signal.ts)} ${signal.signal_price:.2f}  "
            f"setup={signal.setup_name}{setup_tf}@{signal.setup_score:.0f}  "
            f"range={range_text}  eval={signal.eval_confidence:.1f} "
            f"live={signal.live_confidence:.1f}{approval_text}{entry_text}"
        )
        print(
            f"      best={_fmt_sim_ts(signal.best_ts)} ${signal.best_price:.2f}  "
            f"speed={signal.highest_trade_speed:.1f}/s  "
            f"buy={signal.highest_buy_ratio:.0%}  "
            f"accel={signal.highest_pressure_acceleration:+.2f}  "
            f"imb={signal.highest_trade_imbalance:+.2f}  "
            f"ofi={signal.highest_ofi_rolling}  "
            f"net={signal.highest_net_pressure}  "
            f"B/S={signal.highest_buying_pressure}/{signal.highest_selling_pressure}"
        )
        if signal.status_reason or signal.block_reasons:
            detail = signal.status_reason or signal.block_reasons
            print(f"      {_dim(detail)}")
    print()


def _matching_setup_signal(
    signals: list[SetupSignal],
    symbol: str,
    entry_ts: Optional[datetime],
) -> Optional[SetupSignal]:
    if entry_ts is None:
        return None
    if entry_ts.tzinfo is None:
        entry_ts = entry_ts.replace(tzinfo=timezone.utc)

    best: Optional[SetupSignal] = None
    best_delta = 5.0
    for signal in signals:
        if signal.symbol != symbol or not signal.entered or signal.entry_ts is None:
            continue
        signal_ts = signal.entry_ts
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.replace(tzinfo=timezone.utc)
        delta = abs((signal_ts - entry_ts).total_seconds())
        if delta <= best_delta:
            best = signal
            best_delta = delta
    return best


def _day_sim_fills(date_str: str) -> list[SimFill]:
    from exec.engine import SimulationEngine

    sim_logger = SimulationEngine.get_logger()
    if sim_logger is None:
        sim_logger = SimTradeLogger()
    return sim_logger.day_fills(date_str)


def _lifetime_sim_realized() -> tuple[float, Optional[str]]:
    """Return (total realized P&L across all sim days, first-trade date str).

    Sums SELL-fill P&L from every ``sim_trades_*.csv`` ever logged. The date
    is the earliest day with any sim trade, or None if none exist.
    """
    from exec.engine import SimulationEngine

    sim_logger = SimulationEngine.get_logger()
    if sim_logger is None:
        sim_logger = SimTradeLogger()
    dates = sim_logger.trade_dates()
    if not dates:
        return 0.0, None
    total = 0.0
    for date_str in dates:
        for fill in sim_logger.day_fills(date_str):
            if fill.side.upper() == "SELL":
                total += fill.pnl_dollars
    return total, dates[0]


def _closed_sim_trades(fills: list[SimFill]) -> list[tuple[SimFill | None, SimFill]]:
    open_buys: dict[str, list[SimFill]] = {}
    closed: list[tuple[SimFill | None, SimFill]] = []

    for fill in sorted(fills, key=lambda f: f.ts):
        side = fill.side.upper()
        if side == "BUY":
            open_buys.setdefault(fill.symbol, []).append(fill)
        elif side == "SELL":
            buys = open_buys.get(fill.symbol)
            buy = buys.pop(0) if buys else None
            closed.append((buy, fill))
    return closed

def _load_cached_seed_market_data(symbol: str):
    end_utc = datetime.now(tz=timezone.utc)
    start_utc = latest_extended_session_start(end_utc)
    return (
        load_candle_cache(symbol, "1m", start_utc, end_utc),
        load_candle_cache(symbol, "10s", start_utc, end_utc),
        load_timesales_cache(symbol, start_utc, end_utc),
    )

def _ensure_cached_scorer(
    ticker: str,
    entry,
    rmap,
    evaluator: StockEvaluator,
    tradier,
    scorers: dict[str, DynamicScorer],
    on_update: Optional[callable] = None,
) -> bool:
    if not stream_enabled():
        return False

    if ticker not in evaluator.watchlist_symbols:
        # Build-then-rebind (never mutate in place): the toolbar render
        # thread iterates evaluator.watchlist concurrently.
        _new_wl = [e for e in evaluator.watchlist if e.symbol != ticker]
        _new_wl.append(entry)
        _new_wl.sort(key=lambda e: e.watchlist_sort_score, reverse=True)
        evaluator.watchlist = _new_wl
        evaluator.watchlist_symbols.add(ticker)

    if ticker in scorers:
        return True

    try:
        df_1m_seed, df_10s_seed, df_timesales_seed = _load_cached_seed_market_data(ticker)

        # Daily candle fetch is a network call that can timeout independently.
        # The scorer works fine without daily candles, so don't let a timeout
        # here prevent the scorer from starting entirely.
        daily_candles = []
        try:
            df_daily = tradier.get_candles_daily(ticker, days_back=60)
            df_daily = build_daily(df_daily) if not df_daily.empty else df_daily
            daily_candles = DynamicScorer._df_to_candles(df_daily, "daily") if not df_daily.empty else []
        except Exception as exc:
            logger.warning("[%s] Daily candle fetch failed (scorer starting without): %s", ticker, exc)

        move_start_ts = _detect_move_start_ts(df_1m_seed) if not df_1m_seed.empty else None

        scorer = DynamicScorer(
            entry,
            rmap,
            avg_daily_volume=entry.avg_volume,
            daily_candles=daily_candles,
            seed_1min_session=df_1m_seed,
            seed_10sec=df_10s_seed,
            seed_timesales=df_timesales_seed,
            move_start_ts=move_start_ts,
            on_update=on_update,
        )
        scorer.prime_from_seed_candles()
        scorer.start()
        scorers[ticker] = scorer
        logger.debug(
            "[%s] Cached scorer started: 1m=%d 10s=%d t&s=%d",
            ticker, len(df_1m_seed), len(df_10s_seed), len(df_timesales_seed),
        )
        return True
    except Exception as exc:
        logger.warning("[%s] Cached scorer start failed: %s", ticker, exc)
        return False
# ---------------------------------------------------------------------------
# Main input loop
# ---------------------------------------------------------------------------

def main() -> None:
    global _rank_active, _top_active, _top_limit
    evaluator = StockEvaluator()
    tradier   = get_historical_client()
    scorers: dict[str, DynamicScorer] = {}
    faded: set[str] = set()  # tickers whose scorer was stopped due to HOD fade
    news_refreshed: set[str] = set()  # tickers whose news has been re-queried this session

    # Load FinBERT model up front so progress bar doesn't interfere with prompt
    evaluator._finbert._ensure_model()

    # -- Start background scanners --
    ticker_q      = queue.Queue()
    scanner_stop  = threading.Event()
    cache = TickerCache()
    queued: set[str] = set()  # tickers already in the queue (prevents duplicates)
    rejected: set[str] = set()  # tickers hard-rejected by Stage 1 (manual REPL only)
    pending_momo_hits: dict[str, tuple[float, float]] = {}  # deprecated — kept for consumer API compat
    pipeline_done: set[str] = set()  # symbols whose pipeline has finished (pass or hard-fail)
    scanner_lock = threading.Lock()  # protects queued, pipeline_done
    scanner_threads: list[threading.Thread] = []
    muted_tickers: set[str] = set()  # tickers whose scanner output is suppressed

    momo_client: Optional[MomoClient] = None
    momo_paused = threading.Event()     # set = scanner paused
    momo_force_poll = threading.Event()  # set = wake scanner immediately
    _state = _load_state()
    momo_interval: list[int] = [_state.get("momo_interval", SCANNER_MOMO_INTERVAL)]  # mutable so fast/def/slow can change it live
    scanner_ready_events: list[threading.Event] = []
    ibkr_client = None
    ibkr_paused = threading.Event()      # set = scanner paused
    ibkr_force_poll = threading.Event()  # set = wake scanner immediately
    ibkr_interval: list[int] = [_state.get("ibkr_interval", SCANNER_IBKR_INTERVAL)]
    yt_paused = threading.Event()       # set = scanner paused
    yt_force_poll = threading.Event()    # set = wake scanner immediately
    yt_interval: list = [_state.get("yt_interval", None)]   # int = fixed; None (default, or set via "yt auto") = server hint
    yt_client: Optional[YouTubeLiveChatClient] = None
    yt_url = SCANNER_YT_URL

    def _ensure_consumer() -> None:
        """Start the pipeline consumer exactly once. Scanners can come up at
        different times (YouTube waits for its stream to go live), so every
        start path calls this instead of assuming it went first."""
        if not scanner_threads or any(th.name == "scanner-consumer" for th in scanner_threads):
            return
        _consumer_ready = threading.Event()
        scanner_ready_events.append(_consumer_ready)
        t = threading.Thread(
            target=_scanner_consumer,
            args=(ticker_q, scanner_stop, evaluator, tradier, scorers, cache,
                  queued, _consumer_ready, pending_momo_hits, pipeline_done, scanner_lock, news_refreshed, muted_tickers, faded, _invalidate_toolbar),
            daemon=True, name="scanner-consumer",
        )
        t.start()
        scanner_threads.append(t)

    def _start_yt_scanner(quiet: bool = False) -> bool:
        """Resolve the live stream (if needed) and start the YouTube scanner.

        Returns True when the scanner is running or permanently not wanted
        (disabled / nothing configured to follow); False when the only
        blocker is that no stream is live yet — the caller retries, because
        the tracked stream opens well after the 04:00 ET session start.
        """
        nonlocal yt_client, yt_url
        if not SCANNER_YT_ENABLED or yt_client is not None:
            return True
        if not yt_url:
            if not SCANNER_YT_CHANNEL:
                return True
            if not quiet:
                logger.info("[youtube] Resolving live stream for @%s ...", SCANNER_YT_CHANNEL)
            yt_url = YouTubeLiveChatClient.resolve_live_stream(SCANNER_YT_CHANNEL) or ""
            if not yt_url:
                return False
        yt_client = YouTubeLiveChatClient(yt_url)
        # ModernBERT news detection + raw-chat logging.  Built here rather
        # than alongside FinBERT so a ~600 MB model is never loaded for a
        # scanner that isn't going to run; preload() is a daemon thread, so
        # startup stays non-blocking either way.
        yt_news_clf = None
        yt_symbol_resolver = None
        if SCANNER_YT_MODERNBERT_ENABLED:
            yt_news_clf = ModernBERTNewsClassifier()
            yt_news_clf.preload()
            yt_symbol_resolver = RateLimitedSymbolResolver(ibkr_client)
            logger.info(
                "[youtube] ModernBERT news detection: %s mode (model=%s, threshold=%.2f)",
                SCANNER_YT_MODERNBERT_MODE, yt_news_clf.model, yt_news_clf.threshold,
            )
        yt_chat_log = ChatShadowLogger() if SCANNER_YT_CHAT_LOG_ENABLED else None
        _yt_ready = threading.Event()
        scanner_ready_events.append(_yt_ready)
        t = threading.Thread(
            target=_youtube_scanner,
            args=(ticker_q, scanner_stop, evaluator, cache, queued,
                  yt_paused, yt_client, yt_force_poll,
                  evaluator._finbert, tradier, _yt_ready, yt_interval, scanner_lock, muted_tickers),
            kwargs={
                "news_clf": yt_news_clf,
                "chat_log": yt_chat_log,
                "symbol_resolver": yt_symbol_resolver,
            },
            daemon=True, name="scanner-youtube",
        )
        t.start()
        scanner_threads.append(t)
        _ensure_consumer()
        return True

    def _start_scanners() -> None:
        """Create clients and start every enabled scanner thread (+ consumer).

        Runs synchronously at launch when the extended session is open; an
        overnight/weekend launch instead arms the scanner-autostart watcher
        below, which calls this as the session opens. Ready events land in
        *scanner_ready_events* so the synchronous path can wait for the
        startup log lines before the banner prints.
        """
        nonlocal momo_client, ibkr_client
        if SCANNER_MOMO_ENABLED:
            momo_client = MomoClient()
            _momo_ready = threading.Event()
            scanner_ready_events.append(_momo_ready)
            t = threading.Thread(
                target=_momo_scanner,
                args=(ticker_q, scanner_stop, evaluator, cache, queued, momo_client, momo_paused, momo_interval, _momo_ready, momo_force_poll, pipeline_done, scanner_lock, muted_tickers, scorers),
                daemon=True, name="scanner-momo",
            )
            t.start()
            scanner_threads.append(t)

        if SCANNER_IBKR_ENABLED and IBKR_ENABLED:
            # Dedicated client, not get_historical_client() — that singleton can
            # hand back the Tradier client, which has no scanner API.
            from market_data.ibkr_rest import IBKRHistoricalClient
            ibkr_client = IBKRHistoricalClient()
            _ibkr_ready = threading.Event()
            scanner_ready_events.append(_ibkr_ready)

            def _on_ibkr_drop(symbols: list[str]) -> None:
                """Stop watching names that lost their seat on the ranked list.

                Same end state as the fade-from-HOD path: scorer stopped,
                entry moved to the FADED section, and dropped from *queued*
                so the scanners can re-add it if it climbs back on its own.
                """
                for sym in symbols:
                    if sym in faded:
                        continue
                    faded.add(sym)
                    _sc = scorers.pop(sym, None)
                    if _sc:
                        _sc.stop()
                    with scanner_lock:
                        queued.discard(sym)
                    logger.debug(
                        "[%s] Scorer stopped — off the ranked list for %ds",
                        sym, SCANNER_IBKR_DROP_SECS,
                    )
                if symbols:
                    _invalidate_toolbar()

            t = threading.Thread(
                target=_ibkr_scanner,
                args=(ticker_q, scanner_stop, evaluator, cache, queued, ibkr_client, ibkr_paused, ibkr_interval, _ibkr_ready, ibkr_force_poll, pipeline_done, scanner_lock, muted_tickers, scorers),
                kwargs={"on_scan": _on_ibkr_scan, "on_drop": _on_ibkr_drop},
                daemon=True, name="scanner-ibkr",
            )
            t.start()
            scanner_threads.append(t)

        if not _start_yt_scanner():
            # Stream not live yet (it opens ~06:29 ET, well after the 04:00
            # session start) — keep checking until it appears or the
            # extended session ends.
            logger.info(
                "[youtube] No live stream for @%s yet — retrying every %ds until it opens",
                SCANNER_YT_CHANNEL, SCANNER_YT_RESOLVE_RETRY_S,
            )

            def _yt_retry() -> None:
                while not scanner_stop.is_set() and MarketClock.scanners_active():
                    scanner_stop.wait(SCANNER_YT_RESOLVE_RETRY_S)
                    if scanner_stop.is_set():
                        return
                    if _start_yt_scanner(quiet=True):
                        return
                logger.debug("[youtube] Session over — stopped waiting for a live stream")

            threading.Thread(target=_yt_retry, daemon=True,
                             name="scanner-yt-autostart").start()

        _ensure_consumer()

    if MarketClock.scanners_active():
        _start_scanners()
        # Wait for all scanners to log their startup lines before printing
        # the banner so log messages don't interleave with box-drawing chars.
        for ev in scanner_ready_events:
            ev.wait(timeout=10)
    else:
        # Launched while the market is closed: instead of never starting,
        # sleep straight to the extended-session open (04:00 ET) and bring
        # every scanner up then. The wait is computed, not polled, so the
        # start lands *at* the open; the loop absorbs an early wake. The
        # deferred path skips the banner wait — nothing prints around it
        # by then.
        def _deferred_scanner_start() -> None:
            while not scanner_stop.is_set() and not MarketClock.scanners_active():
                scanner_stop.wait(max(1.0, MarketClock.seconds_to_session_open()))
            if scanner_stop.is_set():
                return
            logger.info("[scanners] Extended session open — starting scanners")
            _start_scanners()
        threading.Thread(target=_deferred_scanner_start, daemon=True,
                         name="scanner-autostart").start()

    # ---- scanners row ----
    scan_parts: list[str] = []
    scan_parts_plain: list[str] = []
    if SCANNER_MOMO_ENABLED:
        ok = momo_client is not None
        plain = f"momo {'✓' if ok else '✗'}"
        scan_parts.append(_g(plain) if ok else _r(plain))
        scan_parts_plain.append(plain)
    if SCANNER_IBKR_ENABLED and IBKR_ENABLED:
        ok = ibkr_client is not None
        plain = f"ibkr {'✓' if ok else '✗'}"
        scan_parts.append(_g(plain) if ok else _r(plain))
        scan_parts_plain.append(plain)
    if SCANNER_YT_ENABLED and (yt_url or SCANNER_YT_CHANNEL):
        ok = yt_client is not None
        plain = f"zendoo {'✓' if ok else '✗'}"
        scan_parts.append(_g(plain) if ok else _r(plain))
        scan_parts_plain.append(plain)
    if not scan_parts:
        scan_text = _dim("none")
        scan_text_plain = "none"
    else:
        scan_text = "  ".join(scan_parts)
        scan_text_plain = "  ".join(scan_parts_plain)
    # If scanners are waiting on market hours, note it
    closed_note = f" {_dim('(closed — auto-start 04:00 ET)')}" if not MarketClock.scanners_active() and scan_parts else ""
    closed_note_len = len(" (closed — auto-start 04:00 ET)") if closed_note else 0
    cache_note = f" │ {_dim(f'{len(cache)} cached')}" if len(cache) else ""
    cache_note_len = len(f" │ {len(cache)} cached") if len(cache) else 0

    # ---- apis row ----
    def _key_ok(val: str) -> bool:
        """True only if val looks like a real key: alphanumeric/hyphen/dot, contains a digit, ≥8 chars."""
        return bool(re.fullmatch(r'(?=[A-Za-z0-9\-.]*[0-9])[A-Za-z0-9\-.]{8,}', val))

    def _mk(ok: bool, label: str) -> tuple[str, str]:
        plain = f"{label} {'✓' if ok else '✗'}"
        return (_g(plain) if ok else _r(plain)), plain

    api_parts: list[str] = []
    api_parts_plain: list[str] = []
    for _ok, _lbl in [
        (TRADIER_ENABLED and _key_ok(TRADIER_API_KEY),  "tradier"),
        (IBKR_ENABLED,                                      "ibkr"),
        (AV_ENABLED and _key_ok(AV_API_KEY),              "av"),
        (FINNHUB_ENABLED and _key_ok(FINNHUB_API_KEY),       "finnhub"),
        (BRAVE_ENABLED and _key_ok(BRAVE_SEARCH_API_KEY),    "brave"),
        (FINBERT_ENABLED,                                     "finbert"),
    ]:
        _c2, _p = _mk(_ok, _lbl)
        api_parts.append(_c2)
        api_parts_plain.append(_p)
    api_text = "  ".join(api_parts)
    api_text_plain = "  ".join(api_parts_plain)

    # (label, coloured content, printable length of that content)
    _rows = [
        ("scanners", f"{scan_text}{closed_note}{cache_note}",
         len(scan_text_plain) + closed_note_len + cache_note_len),
        ("APIs", api_text, len(api_text_plain)),
        ("commands", "list · help · q", len("list · help · q")),
    ]
    # Size the box to its widest row. The scanners and APIs rows grow every
    # time a feed or provider is added; a fixed width just pushed the right
    # border out of alignment once a row outgrew it, with no other signal.
    # 13 = 3 leading spaces + the 8-wide label + 2 separator spaces.
    _W = max(64, 3 + len(BOT_NAME), 3 + len(BOT_SUBTITLE),
             *(13 + _plen for _lbl, _content, _plen in _rows))

    def _row(label: str, content: str, plain_len: int) -> str:
        return f"  {_c('║')}   {_dim(f'{label:<8}')}  {content}{' ' * (_W - 13 - plain_len)}{_c('║')}"

    _name_pad = _W - 3 - len(BOT_NAME)
    _sub_pad  = _W - 3 - len(BOT_SUBTITLE)
    print()
    print(f"  {_c('╔' + '═'*_W + '╗')}")
    print(f"  {_c('║')}{' '*_W}{_c('║')}")
    print(f"  {_c('║')}   {_b(_c(BOT_NAME))}{' '* _name_pad}{_c('║')}")
    print(f"  {_c('║')}   {_dim(BOT_SUBTITLE)}{' '* _sub_pad}{_c('║')}")
    print(f"  {_c('║')}{' '*_W}{_c('║')}")
    print(f"  {_c('╠' + '═'*_W + '╣')}")
    for _lbl, _content, _plen in _rows:
        print(_row(_lbl, _content, _plen))
    print(f"  {_c('╚' + '═'*_W + '╝')}")
    print()

    # ------------------------------------------------------------------
    # Seed top gainers into the pipeline on startup
    # ------------------------------------------------------------------
    if MarketClock.scanners_active():
        _seed_client = momo_client or MomoClient()
        # Ensure a consumer thread is running to process the seeded tickers
        if not scanner_threads:
            _consumer_ready = threading.Event()
            t = threading.Thread(
                target=_scanner_consumer,
                args=(ticker_q, scanner_stop, evaluator, tradier, scorers, cache,
                      queued, _consumer_ready, pending_momo_hits, pipeline_done, scanner_lock, news_refreshed, muted_tickers, faded, _invalidate_toolbar),
                daemon=True, name="scanner-consumer",
            )
            t.start()
            scanner_threads.append(t)
            _consumer_ready.wait(timeout=5)

        def _seed_recent_cached_symbols() -> int:
            """Queue cache symbols updated during the current market phase."""
            _now = MarketClock.now_et()
            _today = _now.date()
            # Use the start of the current phase as cutoff:
            #   RTH or post-market → 9:30 (RTH open)
            #   Pre-market or before → 4:00 (EXT open)
            if _now.time() >= MarketClock.RTH_OPEN:
                _phase_start = MarketClock.RTH_OPEN
            else:
                _phase_start = MarketClock.EXT_OPEN
            _session_start = datetime(
                _today.year, _today.month, _today.day,
                _phase_start.hour, _phase_start.minute,
                tzinfo=MarketClock.ET,
            )
            _cutoff_ts = _session_start.timestamp()
            _seeded_recent = 0
            for _sym in cache.symbols:
                _ts = cache.get_updated_ts(_sym)
                if _ts is None:
                    continue
                if _ts < _cutoff_ts:
                    continue
                if validate_ticker(_sym) is None:
                    continue
                with scanner_lock:
                    if _sym in queued:
                        continue
                    queued.add(_sym)
                ticker_q.put(("seed", _sym))
                _seeded_recent += 1
            return _seeded_recent

        try:
            _gainers = _seed_client.get_top_gainers()[:10]
            _stale_gainers = False
            # Staleness guard: skip if the freshest timestamp is from
            # before today's premarket open (screener serving old data).
            if _gainers:
                from datetime import datetime as _dt_cls
                _freshest_ts = max(
                    (s.updated_at for s in _gainers if s.updated_at),
                    default=None,
                )
                _today_date = MarketClock.now_et().date()
                _session_start = _dt_cls(
                    _today_date.year, _today_date.month, _today_date.day,
                    MarketClock.EXT_OPEN.hour, MarketClock.EXT_OPEN.minute,
                    tzinfo=MarketClock.ET,
                )
                if _freshest_ts is not None and _freshest_ts < _session_start:
                    logger.info(
                        "[startup] Skipping stale top gainers (freshest %s < session %s)",
                        _freshest_ts, _session_start,
                    )
                    _stale_gainers = True
                    _gainers = []
            _seeded = 0
            for _stock in _gainers:
                _sym = _stock.symbol.upper()
                if validate_ticker(_sym) is None:
                    continue
                if _sym in queued:
                    continue
                with scanner_lock:
                    queued.add(_sym)
                ticker_q.put(("seed", _sym))
                _seeded += 1
            if _seeded:
                logger.info("[startup] Seeded %d top gainers into pipeline", _seeded)
            if _stale_gainers:
                _seeded_recent = _seed_recent_cached_symbols()
                if _seeded_recent:
                    logger.info("[startup] Seeded %d cache symbols from current session", _seeded_recent)
        except Exception as exc:
            logger.warning("[startup] Failed to seed top gainers: %s", exc)

        # Also seed latest momentum screener hits (5-min movers, 2+ hits only)
        try:
            _momos_raw = _seed_client.get_momos_raw()
            if _momos_raw:
                from datetime import datetime as _dt_cls2
                _freshest_ts_m = max(
                    (s.updated_at for s in _momos_raw if s.updated_at),
                    default=None,
                )
                _today_date_m = MarketClock.now_et().date()
                _session_start_m = _dt_cls2(
                    _today_date_m.year, _today_date_m.month, _today_date_m.day,
                    MarketClock.EXT_OPEN.hour, MarketClock.EXT_OPEN.minute,
                    tzinfo=MarketClock.ET,
                )
                if _freshest_ts_m is not None and _freshest_ts_m < _session_start_m:
                    logger.info(
                        "[startup] Skipping stale momentum hits (freshest %s < session %s)",
                        _freshest_ts_m, _session_start_m,
                    )
                    _momos_raw = []
            # Count occurrences per symbol — only seed those with 2+ hits
            from collections import Counter as _Counter
            _momo_counts = _Counter(s.symbol.upper() for s in _momos_raw)
            _seeded_m = 0
            for _sym, _count in _momo_counts.most_common(5):
                if _count < 2:
                    continue
                if validate_ticker(_sym) is None:
                    continue
                if _sym in queued:
                    continue
                with scanner_lock:
                    queued.add(_sym)
                ticker_q.put(("seed", _sym))
                _seeded_m += 1
            if _seeded_m:
                logger.info("[startup] Seeded %d momentum hits into pipeline (2+ occurrences)", _seeded_m)
        except Exception as exc:
            logger.warning("[startup] Failed to seed momentum hits: %s", exc)

    def _shutdown() -> None:
        scanner_stop.set()
        if scorers:
            _snapshot = list(scorers.values())
            for _sc in _snapshot:
                _sc.stop()
            logger.info("[shutdown] DynamicScorer stopped (%d tickers closed)", len(_snapshot))
        # Join workers before tearing down connections — a scanner still
        # mid-fetch would otherwise reconnect and leak the slot we just freed.
        for _t in scanner_threads:
            _t.join(timeout=8)
        try:
            shutdown_market_data()
        except Exception:
            logger.debug("[shutdown] market data shutdown failed", exc_info=True)
        cache.flush()

    def _format_weights_for_toolbar(sym: str, entry) -> str:
        scorer = scorers.get(sym)
        pa = None
        snap = None
        if scorer is not None:
            try:
                snap = scorer.snapshot()
                pa = snap.tf_pa_scores or None
            except Exception:
                pass
        found_ts = cache.get_session_found_ts(sym)
        return format_live_weights(entry, price_action_tf_scores=pa, scorer_snapshot=snap, found_ts=found_ts)

    def _bottom_toolbar():
        parts = [watch_status_line(evaluator, faded=faded, scorers=scorers)]
        if _rank_active and scorers:
            ranked = []
            # Snapshot: the consumer thread pops/assigns scorers concurrently.
            for sym, sc in list(scorers.items()):
                snap_fn = getattr(sc, "try_snapshot", sc.snapshot)
                snap = snap_fn()
                if snap is None:
                    continue
                ranked.append((sym, snap.prints_seen, snap.buying_pressure, snap.selling_pressure, snap.trade_speed))
            ranked.sort(key=lambda x: x[1], reverse=True)
            lines = [f"  {_b('T&S Activity Ranking')}", f"  {'─' * 40}"]
            for i, (sym, total, bp, sp, speed) in enumerate(ranked, 1):
                net = bp - sp
                net_color = _g if net > 0 else (_r if net < 0 else _dim)
                lines.append(
                    f"  {_dim(f'{i}.')} {_b(f'{sym:<6}')} "
                    f"{total:>5,} prints  {speed:.0f}/s  "
                    f"{net_color(f'{net:+d}')} net "
                    f"({_g(f'{bp}↑')}/{_r(f'{sp}↓')})"
                )
            parts.append("\n".join(lines))
        if _top_active:
            # Unsliced: format_top_gainers drops hard-rejected names first,
            # then takes `limit`, so the view fills to the count asked for.
            _scan_rows = list(getattr(ibkr_client, "latest_scan", []) or [])
            _scan_ts = float(getattr(ibkr_client, "latest_scan_ts", 0.0) or 0.0)
            # Snapshot under the lock the scanner threads mutate it with.
            with scanner_lock:
                _rejected_syms = frozenset(pipeline_done)
            parts.append("\n" + format_top_gainers(
                _scan_rows,
                age_secs=(_time.monotonic() - _scan_ts) if _scan_ts else None,
                cache=cache,
                scorers=scorers,
                watchlist_symbols=evaluator.watchlist_symbols,
                paused=ibkr_paused.is_set(),
                interval=ibkr_interval[0],
                rejected=_rejected_syms,
                limit=_top_limit,
            ))
        # Order-flow views. Each renders from a plain-data copy the scorer
        # assembles under its own lock; a None view means the stream thread
        # held that lock this frame, so the block re-shows its previous
        # frame (_flow_frame_cache) rather than stalling the prompt or
        # collapsing to a placeholder.
        for sym, _rows in sorted(_tape_active.items()):
            _sc_tp = scorers.get(sym)
            if _sc_tp is None:
                parts.append("\n  " + _dim(f"{sym} tape — scorer stopped, no live stream"))
                continue
            _tp_view = _sc_tp.try_tape_view(_rows)
            if _tp_view is None:
                _tp_blk = _flow_frame_cache.get(("tape", sym)) or format_tape(None, limit=_rows)
            else:
                _tp_blk = format_tape(_tp_view, limit=_rows)
                _flow_frame_cache[("tape", sym)] = _tp_blk
            parts.append("\n" + _tp_blk)
        for sym, _levels in sorted(_ofi_active.items()):
            _sc_of = scorers.get(sym)
            if _sc_of is None:
                parts.append("\n  " + _dim(f"{sym} book — scorer stopped, no live stream"))
                continue
            _bk_view = _sc_of.try_book_view()
            if _bk_view is None:
                _bk_blk = _flow_frame_cache.get(("book", sym)) or format_order_book(None, limit=_levels)
            else:
                _bk_blk = format_order_book(_bk_view, limit=_levels)
                _flow_frame_cache[("book", sym)] = _bk_blk
            parts.append("\n" + _bk_blk)
        for sym in sorted(_weights_active):
            entry = next((e for e in evaluator.watchlist if e.symbol == sym), None)
            if entry is None:
                cached = cache.get(sym)
                if cached:
                    entry = cached[0]
            if entry is not None and entry.evaluation is not None:
                weight_text = _format_weights_for_toolbar(sym, entry)
                parts.append("\n" + weight_text)
        for sym in sorted(_levels_active):
            _cached_lv = cache.get(sym)
            if _cached_lv:
                _entry_lv, _rmap_lv = _cached_lv
                _lv_price = None
                _sc_lv = scorers.get(sym)
                if _sc_lv is not None:
                    try:
                        _snap_fn_lv = getattr(_sc_lv, "try_snapshot", _sc_lv.snapshot)
                        _snap_lv = _snap_fn_lv()
                        if _snap_lv and getattr(_snap_lv, "last_price", 0) > 0:
                            _lv_price = _snap_lv.last_price
                    except Exception:
                        pass
                parts.append("\n" + format_levels(sym, _entry_lv, _rmap_lv, current_price=_lv_price))
        return ANSI("\n".join(parts))

    from prompt_toolkit.styles import Style

    _toolbar_style = Style.from_dict({"bottom-toolbar": "noreverse"})
    session = PromptSession(bottom_toolbar=_bottom_toolbar, style=_toolbar_style)

    global _toolbar_app
    _toolbar_app = session.app
    set_app(session.app)

    with patch_stdout(raw=True):
        while True:
            try:
                raw = session.prompt("\n  Ticker > ").strip()
            except (KeyboardInterrupt, EOFError):
                _shutdown()
                print()
                break

            if not raw:
                continue

            cmd = raw.lower()

            if cmd == "exit":
                _shutdown()
                print()
                break

            if cmd in ("q", "quit"):
                if _rank_active or _top_active or _weights_active or _levels_active or _tape_active or _ofi_active:
                    if _rank_active:
                        _rank_active = False
                        print(f"  {_dim('Rank display closed')}")
                    if _top_active:
                        _top_active = False
                        _top_limit = _TOP_LIMIT_DEFAULT
                        print(f"  {_dim('Top gainers display closed')}")
                    if _weights_active:
                        n = len(_weights_active)
                        _weights_active.clear()
                        print(f"  {_dim(f'Closed {n} weights display(s)')}")
                    if _levels_active:
                        n = len(_levels_active)
                        _levels_active.clear()
                        print(f"  {_dim(f'Closed {n} levels display(s)')}")
                    if _tape_active:
                        n = len(_tape_active)
                        _tape_active.clear()
                        print(f"  {_dim(f'Closed {n} tape display(s)')}")
                    if _ofi_active:
                        n = len(_ofi_active)
                        _ofi_active.clear()
                        print(f"  {_dim(f'Closed {n} book display(s)')}")
                    _flow_frame_cache.clear()
                    continue
                _shutdown()
                print()
                break

            if cmd in ("list", "watch"):
                print_watchlist(evaluator, faded=faded, scorers=scorers)
                continue

            if cmd in ("setup clear", "setups clear"):
                day_str = MarketClock.today().strftime("%Y%m%d")
                signals = _day_setup_signals(day_str)
                n = len(signals)
                if n == 0:
                    print(f"  {_dim('No setup signals to clear for today.')}")
                    continue
                print(f"  {_y('!')}  Clear {n} setup signal{'s' if n != 1 else ''} for {day_str}?")
                confirm = pt_prompt("      Confirm? [y/N] ").strip().lower()
                if confirm == "y":
                    removed = get_setup_signal_logger().clear_session(day_str)
                    print(f"  {_dim(f'Cleared {removed} setup signal(s) for today.')}")
                else:
                    print(f"  {_dim('Cancelled.')}")
                continue

            if cmd in ("setup", "setups"):
                day_str = MarketClock.today().strftime("%Y%m%d")
                signals = _day_setup_signals(day_str)
                _print_setup_signals(day_str, signals)
                continue

            if cmd in ("pos clear", "positions clear"):
                from exec.engine import SimulationEngine

                day_str = MarketClock.today().strftime("%Y%m%d")
                positions = SimulationEngine.active_positions()
                fills = _day_sim_fills(day_str)
                n_pos = len(positions)
                n_fills = len(fills)
                if n_pos == 0 and n_fills == 0:
                    print(f"  {_dim('No positions or fills to clear for today.')}")
                    continue
                parts = []
                if n_pos:
                    parts.append(f"{n_pos} open position{'s' if n_pos != 1 else ''}")
                if n_fills:
                    parts.append(f"{n_fills} fill{'s' if n_fills != 1 else ''}")
                print(f"  {_y('!')}  Clear {' + '.join(parts)} for {day_str}?")
                confirm = pt_prompt("      Confirm? [y/N] ").strip().lower()
                if confirm == "y":
                    sim_logger = SimulationEngine.get_logger() or SimTradeLogger()
                    removed = sim_logger.clear_session(day_str)
                    SimulationEngine.reset_session()
                    print(f"  {_dim(f'Cleared {n_pos} position(s) and {removed} fill(s) for today.')}")
                else:
                    print(f"  {_dim('Cancelled.')}")
                continue

            if cmd in ("pos full", "positions full"):
                from exec.engine import SimulationEngine

                wrap_width = max(150, shutil.get_terminal_size((200, 24)).columns)
                sim_logger = SimulationEngine.get_logger() or SimTradeLogger()
                bt_logger = SimTradeLogger(log_dir=SIM_BACKTEST_LOG_DIR)

                live_dates = set(sim_logger.trade_dates())
                bt_dates = set(bt_logger.trade_dates())
                all_dates = sorted(live_dates | bt_dates)

                lifetime_pnl, first_trade_date = _lifetime_sim_realized()
                current_account = SIM_ACCOUNT_SIZE + lifetime_pnl
                lifetime_pct = lifetime_pnl / SIM_ACCOUNT_SIZE if SIM_ACCOUNT_SIZE else 0.0
                if first_trade_date:
                    start_fmt = f"{first_trade_date[:4]}-{first_trade_date[4:6]}-{first_trade_date[6:]}"
                    account_suffix = f" ({_fmt_pnl_pct(lifetime_pct)} since {start_fmt})"
                else:
                    account_suffix = ""
                print(f"\n  {_b('Account')} ${current_account:,.2f}{account_suffix}")

                live_total_closed = 0
                live_total_winners = 0
                live_total_losers = 0
                live_total_pnl = 0.0
                bt_total_closed = 0
                bt_total_winners = 0
                bt_total_losers = 0
                bt_total_pnl = 0.0
                rendered_any = False

                for date_str in all_dates:
                    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                    # Gather live and backtest closed trades for this date
                    live_closed = []
                    bt_closed = []
                    if date_str in live_dates:
                        live_closed = _closed_sim_trades(sim_logger.day_fills(date_str))
                    if date_str in bt_dates:
                        bt_closed = _closed_sim_trades(bt_logger.day_fills(date_str))
                    if not live_closed and not bt_closed:
                        continue
                    rendered_any = True

                    # Date header with combined count
                    total_day = len(live_closed) + len(bt_closed)
                    print(f"\n  {_dim('─' * wrap_width)}")
                    print(f"  {_b(date_fmt)} ({total_day} trades)")

                    # Live sub-section
                    if live_closed:
                        live_summary = summarize_closed_trades(live_closed)
                        live_total_closed += len(live_closed)
                        live_total_winners += live_summary.winners
                        live_total_losers += live_summary.losers
                        live_total_pnl += live_summary.total_pnl
                        day_setup_signals = _day_setup_signals(date_str)
                        print(
                            f"    {_b('LIVE')} ({len(live_closed)})  "
                            f"{_fmt_pnl(live_summary.total_pnl)} ({_fmt_pnl_pct(live_summary.realized_pct)})  "
                            f"{_dim(f'W/L={live_summary.winners}/{live_summary.losers}')}"
                        )
                        for buy, sell in live_closed:
                            setup_signal = _matching_setup_signal(day_setup_signals, sell.symbol, buy.ts if buy else sell.ts)
                            setup_tf = sell.setup_timeframe or (buy.setup_timeframe if buy else "") or (setup_signal.setup_timeframe if setup_signal else "")
                            eval_conf = buy.eval_confidence if buy else sell.eval_confidence
                            ta_conf = buy.live_confidence if buy else sell.live_confidence
                            entry_price = buy.fill_price if buy else sell.fill_price - (sell.pnl_dollars / sell.shares if sell.shares else 0.0)
                            entry_ts = _fmt_sim_ts(buy.ts) if buy else "--"
                            prefix = (
                                f"      {(_g if sell.pnl_dollars >= 0 else _r)(sell.symbol)}  "
                                f"entry={entry_ts} ${entry_price:.2f}  "
                                f"exit={_fmt_sim_ts(sell.ts)} ${sell.fill_price:.2f}  "
                                f"shares={sell.shares}  held={sell.hold_secs:.0f}s  "
                                f"P&L={_fmt_pnl(sell.pnl_dollars)} ({sell.pnl_pct:+.2%})  "
                                f"conf=eval:{eval_conf:.1f} TA:{ta_conf:.1f}  "
                                f"pullback={_fmt_setup_timeframe(setup_tf)}  EXIT "
                            )
                            print(f"{prefix}{sell.exit_reason or ''}")

                    # Backtest sub-section
                    if bt_closed:
                        bt_summary = summarize_closed_trades(bt_closed)
                        bt_total_closed += len(bt_closed)
                        bt_total_winners += bt_summary.winners
                        bt_total_losers += bt_summary.losers
                        bt_total_pnl += bt_summary.total_pnl
                        print(
                            f"    {_dim('[BT]')} ({len(bt_closed)})  "
                            f"{_fmt_pnl(bt_summary.total_pnl)} ({_fmt_pnl_pct(bt_summary.realized_pct)})  "
                            f"{_dim(f'W/L={bt_summary.winners}/{bt_summary.losers}')}"
                        )
                        for buy, sell in bt_closed:
                            eval_conf = buy.eval_confidence if buy else sell.eval_confidence
                            ta_conf = buy.live_confidence if buy else sell.live_confidence
                            entry_price = buy.fill_price if buy else sell.fill_price - (sell.pnl_dollars / sell.shares if sell.shares else 0.0)
                            entry_ts = _fmt_sim_ts(buy.ts) if buy else "--"
                            setup_tf = sell.setup_timeframe or (buy.setup_timeframe if buy else "")
                            prefix = (
                                f"      {(_g if sell.pnl_dollars >= 0 else _r)(sell.symbol)}  "
                                f"entry={entry_ts} ${entry_price:.2f}  "
                                f"exit={_fmt_sim_ts(sell.ts)} ${sell.fill_price:.2f}  "
                                f"shares={sell.shares}  held={sell.hold_secs:.0f}s  "
                                f"P&L={_fmt_pnl(sell.pnl_dollars)} ({sell.pnl_pct:+.2%})  "
                                f"conf=eval:{eval_conf:.1f} TA:{ta_conf:.1f}  "
                                f"pullback={_fmt_setup_timeframe(setup_tf)}  EXIT "
                            )
                            print(f"{prefix}{sell.exit_reason or ''}")

                if not rendered_any:
                    print(f"  {_dim('No sim trades recorded.')}")
                else:
                    # Separated P&L footers
                    if live_total_closed:
                        print(
                            f"\n  {_b('Live P&L')} {_fmt_pnl(live_total_pnl)} ({_fmt_pnl_pct(lifetime_pct)})  "
                            f"{_dim(f'days={len(live_dates)} closed={live_total_closed} W/L={live_total_winners}/{live_total_losers}')}"
                        )
                    if bt_total_closed:
                        bt_pct = bt_total_pnl / SIM_ACCOUNT_SIZE if SIM_ACCOUNT_SIZE else 0.0
                        print(
                            f"  {_dim('[BT]')} {_b('Backtest P&L')} {_fmt_pnl(bt_total_pnl)} ({_fmt_pnl_pct(bt_pct)})  "
                            f"{_dim(f'days={len(bt_dates)} closed={bt_total_closed} W/L={bt_total_winners}/{bt_total_losers}')}"
                        )

                print()
                continue

            if cmd in ("pos", "positions"):
                from exec.engine import SimulationEngine

                positions = SimulationEngine.active_positions()
                day_str = MarketClock.today().strftime("%Y%m%d")
                fills = _day_sim_fills(day_str)
                closed = _closed_sim_trades(fills)
                closed_summary = summarize_closed_trades(closed)
                setup_signals = _day_setup_signals(day_str)

                lifetime_pnl, first_trade_date = _lifetime_sim_realized()
                current_account = SIM_ACCOUNT_SIZE + lifetime_pnl
                lifetime_pct = lifetime_pnl / SIM_ACCOUNT_SIZE if SIM_ACCOUNT_SIZE else 0.0
                if first_trade_date:
                    start_fmt = f"{first_trade_date[:4]}-{first_trade_date[4:6]}-{first_trade_date[6:]}"
                    account_suffix = f" ({_fmt_pnl_pct(lifetime_pct)} since {start_fmt})"
                else:
                    account_suffix = ""
                print(f"\n  {_b('Account')} ${current_account:,.2f}{account_suffix}")

                if positions:
                    print(f"\n  {_b('Open Positions')} ({len(positions)})")
                    for sym, pos in positions.items():
                        held = (datetime.now(tz=timezone.utc) - pos.entry_time).total_seconds()
                        setup_signal = _matching_setup_signal(setup_signals, sym, pos.entry_time)
                        setup_tf = pos.setup_timeframe or (setup_signal.setup_timeframe if setup_signal else "")
                        eval_conf = pos.entry_signal.eval_confidence
                        ta_conf = pos.entry_signal.live_confidence
                        print(
                            f"    {_g(sym)}  "
                            f"entry={_fmt_sim_ts(pos.entry_time)} ${pos.entry_price:.2f}  "
                            f"shares={pos.shares}  held={held:.0f}s  "
                            f"conf=eval:{eval_conf:.1f} TA:{ta_conf:.1f}  "
                            f"pullback={_fmt_setup_timeframe(setup_tf)}"
                        )
                elif not closed:
                    print(f"  {_dim('No  sim positions')}")

                if closed:
                    print(f"\n  {_b('Closed Positions')} {_dim(day_str)} ({len(closed)})")
                    for buy, sell in closed:
                        setup_signal = _matching_setup_signal(setup_signals, sell.symbol, buy.ts if buy else sell.ts)
                        setup_tf = sell.setup_timeframe or (buy.setup_timeframe if buy else "") or (setup_signal.setup_timeframe if setup_signal else "")
                        eval_conf = buy.eval_confidence if buy else sell.eval_confidence
                        ta_conf = buy.live_confidence if buy else sell.live_confidence
                        entry_price = buy.fill_price if buy else sell.fill_price - (sell.pnl_dollars / sell.shares if sell.shares else 0.0)
                        entry_ts = _fmt_sim_ts(buy.ts) if buy else "--"
                        prefix = (
                            f"    {(_g if sell.pnl_dollars >= 0 else _r)(sell.symbol)}  "
                            f"entry={entry_ts} ${entry_price:.2f}  "
                            f"exit={_fmt_sim_ts(sell.ts)} ${sell.fill_price:.2f}  "
                            f"shares={sell.shares}  held={sell.hold_secs:.0f}s  "
                            f"P&L={_fmt_pnl(sell.pnl_dollars)} ({sell.pnl_pct:+.2%})  "
                            f"conf=eval:{eval_conf:.1f} TA:{ta_conf:.1f}  "
                            f"pullback={_fmt_setup_timeframe(setup_tf)}  EXIT "
                        )
                        print(f"{prefix}{sell.exit_reason or ''}")

                    by_symbol = {}
                    for buy, sell in closed:
                        entry_notional = buy.notional if buy and buy.notional > 0 else max(sell.notional - sell.pnl_dollars, 0.0)
                        stats = by_symbol.setdefault(sell.symbol, {"pnl": 0.0, "entry_notional": 0.0, "trades": 0})
                        stats["pnl"] += sell.pnl_dollars
                        stats["entry_notional"] += entry_notional
                        stats["trades"] += 1
                    if len(by_symbol) > 1:
                        print(f"\n  {_b('Realized P&L By Stock')}")
                        for sym, stats in sorted(by_symbol.items()):
                            stock_pct = stats["pnl"] / stats["entry_notional"] if stats["entry_notional"] else 0.0
                            print(
                                f"    {(_g if stats['pnl'] >= 0 else _r)(sym)}  "
                                f"P&L={_fmt_pnl(stats['pnl'])} ({_fmt_pnl_pct(stock_pct)})  "
                                f"trades={int(stats['trades'])}"
                            )
                    print(
                        f"\n  {_b('Realized P&L')} {_fmt_pnl(closed_summary.total_pnl)} ({_fmt_pnl_pct(closed_summary.realized_pct)})  "
                        f"sum%={_fmt_pnl_pct(closed_summary.summed_trade_pct)}  "
                        f"{_dim(f'day={day_str} closed={len(closed)} W/L={closed_summary.winners}/{closed_summary.losers} open={len(positions)}')}"
                    )
                print()
                continue

            if cmd == "cache":
                syms = cache.symbols
                if not syms:
                    print("  Cache is empty.")
                else:
                    print(f"\n  {_dim(f'Cached tickers ({len(syms)}):')}")
                    print(f"  {', '.join(_b(s) for s in sorted(syms))}")
                continue

            if cmd == "rank":
                if _rank_active:
                    _rank_active = False
                    print(f"  {_dim('Rank display closed')}")
                elif not scorers:
                    print(f"  {_dim('No active scorers.')}")
                else:
                    _rank_active = True
                    print(f"  {_dim('Rank display started — q to close')}")
                continue

            if cmd == "top" or cmd.startswith("top "):
                _top_arg = cmd[3:].strip()
                if _top_arg:
                    # `top N` always (re)opens at N — an explicit count is a
                    # request to see that many, not a toggle.
                    if not _top_arg.isdigit() or int(_top_arg) < 1:
                        print(f"  {_y('⚠')}  Usage: top | top <count>  (e.g. top 20)")
                    elif ibkr_client is None:
                        print(f"  {_y('⚠')}  {'IBKR scanner starts at 04:00 ET (market closed)' if SCANNER_IBKR_ENABLED and IBKR_ENABLED else 'IBKR scanner not enabled'}")
                    else:
                        _top_limit = int(_top_arg)
                        _top_active = True
                        print(f"  {_dim(f'Top {_top_limit} gainers display started — q to close')}")
                elif _top_active:
                    _top_active = False
                    _top_limit = _TOP_LIMIT_DEFAULT
                    print(f"  {_dim('Top gainers display closed')}")
                elif ibkr_client is None:
                    print(f"  {_y('⚠')}  {'IBKR scanner starts at 04:00 ET (market closed)' if SCANNER_IBKR_ENABLED and IBKR_ENABLED else 'IBKR scanner not enabled'}")
                else:
                    _top_limit = _TOP_LIMIT_DEFAULT
                    _top_active = True
                    print(f"  {_dim(f'Top {_top_limit} gainers display started — q to close')}")
                continue

            if cmd == "clear":
                n = len(cache)
                print(f"  {_y('!')}  Cache already auto-expires after {CACHE_TTL_DAYS}d. Do you want to manually clear all {n} entr{'ies' if n != 1 else 'y'} now?")
                confirm = pt_prompt("      Confirm? [y/N] ").strip().lower()
                if confirm == "y":
                    cache.clear()
                    print(f"  {_dim(f'Cache cleared ({n} entries removed).')}")
                else:
                    print(f"  {_dim('Cancelled.')}")
                continue

            if cmd == "help":
                try:
                    with open("logs/help.txt", encoding="utf-8") as _f:
                        print()
                        print(_f.read())
                except OSError:
                    print(f"  {_y('⚠')}  help file not found (logs/help.txt)")
                continue

            if cmd == "weights" or cmd == "weights group":
                from strategy.scoring import print_weight_breakdown
                print()
                print_weight_breakdown(sort="group" if cmd.endswith("group") else "effective")
                print()
                continue

            if cmd == "conf" or cmd.startswith("conf "):
                import config as _cfg
                _conf_action = cmd[5:].strip() if cmd.startswith("conf ") else ""
                _conf_scope = ""
                if _conf_action == "eval" or _conf_action.startswith("eval "):
                    _conf_scope, _conf_action = "eval", _conf_action[4:].strip()
                elif _conf_action == "live" or _conf_action.startswith("live "):
                    _conf_scope, _conf_action = "live", _conf_action[4:].strip()
                if _conf_action in ("stop", "off"):
                    if _conf_scope == "eval":
                        _cfg.CONFIDENCE_BYPASS_EVAL = True
                        logger.info("[conf] Eval-confidence bypass ENABLED — entries/exits ignore eval_confidence")
                    elif _conf_scope == "live":
                        _cfg.CONFIDENCE_BYPASS_LIVE = True
                        logger.info("[conf] Live-confidence bypass ENABLED — entries/exits ignore live_confidence")
                    else:
                        _cfg.CONFIDENCE_BYPASS = True
                        logger.info("[conf] Confidence bypass ENABLED — trades ignore confidence gates")
                elif _conf_action in ("go", "start", "on"):
                    if _conf_scope == "eval":
                        _cfg.CONFIDENCE_BYPASS_EVAL = False
                        logger.info("[conf] Eval-confidence gate ACTIVE")
                        if _cfg.CONFIDENCE_BYPASS:
                            print(f"  {_y('⚠')}  global bypass still on — 'conf on' to enforce all gates")
                    elif _conf_scope == "live":
                        _cfg.CONFIDENCE_BYPASS_LIVE = False
                        logger.info("[conf] Live-confidence gate ACTIVE")
                        if _cfg.CONFIDENCE_BYPASS:
                            print(f"  {_y('⚠')}  global bypass still on — 'conf on' to enforce all gates")
                    else:
                        _cfg.CONFIDENCE_BYPASS = False
                        _cfg.CONFIDENCE_BYPASS_EVAL = False
                        _cfg.CONFIDENCE_BYPASS_LIVE = False
                        logger.info("[conf] Confidence bypass DISABLED — confidence gates active")
                else:
                    def _conf_state(bypassed: bool) -> str:
                        return _g("OFF (bypassed)") if bypassed else _dim("ON (active)")
                    print(f"  Eval confidence gate: {_conf_state(_cfg.confidence_bypass_eval())}")
                    print(f"  Live confidence gate: {_conf_state(_cfg.confidence_bypass_live())}")
                    print(f"  {_dim('conf stop/off = bypass both | conf go/on = enforce both')}")
                    print(f"  {_dim('conf eval|live off/on = bypass/enforce one gate')}")
                continue

            if cmd.startswith("momo "):
                _action = cmd[5:]
                if _action in _SCANNER_ACTIONS or _action.startswith("interval "):
                    if momo_client is None:
                        print(f"  {_y('⚠')}  Momo scanner not enabled")
                    else:
                        _scanner_cmd(
                            _action, "momo", momo_paused, momo_force_poll, momo_interval,
                            {"fast": SCANNER_MOMO_INTERVAL_FAST, "def": SCANNER_MOMO_INTERVAL_DEF, "slow": SCANNER_MOMO_INTERVAL_SLOW},
                            "momo_interval",
                            on_stop=lambda: setattr(evaluator._momo, 'paused', True),
                            on_go=lambda: setattr(evaluator._momo, 'paused', False),
                        )
                    continue

            if cmd.startswith("ibkr "):
                _action = cmd[5:]
                if _action in _SCANNER_ACTIONS or _action.startswith("interval "):
                    if ibkr_client is None:
                        print(f"  {_y('⚠')}  IBKR scanner not enabled")
                    else:
                        _scanner_cmd(
                            _action, "ibkr", ibkr_paused, ibkr_force_poll, ibkr_interval,
                            {"fast": SCANNER_IBKR_INTERVAL_FAST, "def": SCANNER_IBKR_INTERVAL_DEF, "slow": SCANNER_IBKR_INTERVAL_SLOW},
                            "ibkr_interval",
                        )
                    continue

            if cmd == "tradier" or cmd.startswith("tradier "):
                _action = cmd[8:] if cmd.startswith("tradier ") else ""
                import config as _cfg
                if not TRADIER_ENABLED:
                    print(f"  {_y('⚠')}  Tradier not enabled")
                elif _action == "stop":
                    if _cfg.TRADIER_PAUSED:
                        print(f"  {_dim('tradier already paused')}")
                    else:
                        _cfg.TRADIER_PAUSED = True
                        wake_shared_stream()
                        logger.info("[tradier] Paused — REST polls + live streaming suspended")
                elif _action in ("go", "start"):
                    if not _cfg.TRADIER_PAUSED:
                        print(f"  {_dim('tradier already running')}")
                    else:
                        _cfg.TRADIER_PAUSED = False
                        wake_shared_stream()
                        logger.info("[tradier] Resumed — REST polls + live streaming re-enabled")
                elif _action == "":
                    _tr_state = _r("PAUSED") if _cfg.TRADIER_PAUSED else _g("running")
                    print(f"  Tradier: {_tr_state}")
                    print(f"  {_dim('tradier stop = pause polls + stream | tradier go/start = resume')}")
                else:
                    print(f"  {_y('⚠')}  Usage: tradier stop | go | start")
                continue

            if cmd.startswith("yt "):
                _action = cmd[3:]
                if _action in _SCANNER_ACTIONS or _action == "auto" or _action.startswith("interval "):
                    if not (SCANNER_YT_ENABLED and yt_url and yt_client is not None):
                        print(f"  {_y('⚠')}  YouTube scanner not enabled")
                    else:
                        _auto_hint = f"server hint {yt_client.poll_timeout_ms / 1000:.1f}s" if yt_client.poll_timeout_ms > 0 else "?"
                        _scanner_cmd(
                            _action, "yt", yt_paused, yt_force_poll, yt_interval,
                            {"fast": SCANNER_YT_INTERVAL_FAST, "def": SCANNER_YT_INTERVAL_DEF, "slow": SCANNER_YT_INTERVAL_SLOW},
                            "yt_interval",
                            has_auto=True, auto_hint=_auto_hint, log_name="youtube",
                        )
                    continue

            # Live order-flow views: "<TICKER> tape [rows]" / "<TICKER> ofi
            # [levels]" (command-first order also accepted). Both read the
            # scorer's existing order-flow state — they never subscribe, so a
            # ticker that isn't already streaming is refused rather than
            # started.
            _inv = raw.split()
            if len(_inv) in (2, 3) and _inv[1].lower() in ("tape", "ofi"):
                # Ticker-first form — normalise to command-first and fall
                # through to the shared handler below.
                raw = " ".join([_inv[1], _inv[0], *_inv[2:]])
                cmd = raw.lower()
            if cmd in ("tape", "ofi") or cmd.startswith("tape ") or cmd.startswith("ofi "):
                _is_tape = cmd.startswith("tape")
                _flow_state = _tape_active if _is_tape else _ofi_active
                _flow_name = "tape" if _is_tape else "book"
                _flow_unit = "rows" if _is_tape else "levels"
                _flow_default = _TAPE_ROWS_DEFAULT if _is_tape else _OFI_LEVELS_DEFAULT
                _flow_usage = (
                    f"  {_y('⚠')}  Usage: <TICKER> {'tape' if _is_tape else 'ofi'} "
                    f"[{_flow_unit}]  (e.g. {'AAPL tape 20' if _is_tape else 'AAPL ofi 15'}; "
                    f"omit the count to show everything streamed)"
                )
                _flow_args = raw.split()[1:]
                if not _flow_args or len(_flow_args) > 2:
                    print(_flow_usage)
                    continue
                _flow_sym = validate_ticker(_flow_args[0])
                if _flow_sym is None:
                    print(f"  {_y('⚠')}  Invalid ticker: '{_flow_args[0]}'")
                    continue
                _flow_n = _flow_default
                _flow_explicit = len(_flow_args) == 2
                if _flow_explicit:
                    if not _flow_args[1].isdigit() or int(_flow_args[1]) < 1:
                        print(_flow_usage)
                        continue
                    _flow_n = int(_flow_args[1])
                # Bare command toggles; an explicit count always (re)opens at it.
                if _flow_sym in _flow_state and not _flow_explicit:
                    del _flow_state[_flow_sym]
                    _flow_frame_cache.pop((_flow_name, _flow_sym), None)
                    print(f"  {_dim(f'{_flow_sym} {_flow_name} display closed')}")
                    continue
                if scorers.get(_flow_sym) is None:
                    print(f"  {_y('⚠')}  {_flow_sym} is not streaming — run the ticker first")
                    continue
                _flow_state[_flow_sym] = _flow_n
                _flow_amount = f"{_flow_n} {_flow_unit}" if _flow_n > 0 else f"all streamed {_flow_unit}"
                print(f"  {_dim(f'{_flow_sym} {_flow_name} display started ({_flow_amount}) — q to close')}")
                continue

            # Optional suffixes: "[ticker] refresh|find|setups|noise|stop|go|float <val>|news "<headline>""
            _parts = raw.split()
            _suffix = _parts[1].lower() if len(_parts) >= 2 else ""
            # float override: TICKER float __M/K
            set_float: float | None = None
            if _suffix in ("float", "-f") and len(_parts) == 3:
                _fval = _parts[2].strip()
                try:
                    if _fval[-1:].lower() == "m":
                        set_float = float(_fval[:-1])
                    elif _fval[-1:].lower() == "k":
                        set_float = float(_fval[:-1]) / 1000
                    else:
                        print(f"  Invalid float value: '{_fval}' — use e.g. 5.2M or 123.4k")
                        continue
                    raw = _parts[0]
                except ValueError:
                    print(f"  Invalid float value: '{_fval}'")
                    continue
            # news override: TICKER news "headline text here" [time] OR TICKER news <float>
            set_news: str | None = None
            set_news_time: datetime | None = None   # optional published_at override
            set_sentiment: float | None = None      # direct sentiment override
            if _suffix in ("news", "-n"):
                _news_rest = " ".join(_parts[2:]) if len(_parts) >= 3 else ""
                # Strip surrounding quotes if present
                if len(_news_rest) >= 2 and _news_rest[0] == '"' and _news_rest[-1] == '"':
                    _news_rest = _news_rest[1:-1]
                elif '"' in _news_rest:
                    # Quoted headline with time spec on either side:
                    #   "headline" 5m   OR   5m "headline"
                    _q_start = _news_rest.find('"')
                    _q_end = _news_rest.rfind('"')
                    if _q_start < _q_end:
                        _before = _news_rest[:_q_start].strip()
                        _after = _news_rest[_q_end + 1:].strip()
                        _news_rest = _news_rest[_q_start + 1:_q_end]
                        _time_part = _after or _before
                        if _time_part:
                            set_news_time = _parse_news_time(_time_part)
                            if set_news_time is None:
                                print(f"  Invalid time: '{_time_part}' — use e.g. 5m, 2h, 1d, 9:30, 6/3")
                                continue
                if _news_rest:
                    # If the value is a bare number in [-1, 1], treat as sentiment override
                    try:
                        _sval = float(_news_rest)
                        if -1 <= _sval <= 1:
                            set_sentiment = _sval
                        else:
                            print(f"  Sentiment must be between -1 and 1 (got {_sval})")
                            continue
                    except ValueError:
                        # Check if last token is a time spec (unquoted headline)
                        if set_news_time is None:
                            _words = _news_rest.rsplit(None, 1)
                            if len(_words) == 2:
                                _maybe_time = _parse_news_time(_words[1])
                                if _maybe_time is not None:
                                    _news_rest = _words[0]
                                    set_news_time = _maybe_time
                        set_news = _news_rest
                    raw = _parts[0]
                else:
                    print(f"  Usage: {_parts[0]} news \"headline text\" [time] or {_parts[0]} news <-1..1>")
                    continue
            if _suffix in ("refresh", "-r", "find", "start", "setup", "setups", "noise", "stop", "go", "cache", "clear", "weights", "-w", "split", "-s", "levels", "-l", "live", "-v") or set_float is not None or set_news is not None or set_sentiment is not None:
                if set_float is None and set_news is None and set_sentiment is None:
                    raw = _parts[0]
            force_refresh = _suffix in ("refresh", "-r")
            show_found_at = _suffix == "find"
            show_start = _suffix == "start"
            mark_noise = _suffix == "noise"
            mute_ticker = _suffix == "stop"
            unmute_ticker = _suffix == "go"
            show_cache = _suffix == "cache"
            clear_cache = _suffix == "clear"
            show_setups = _suffix in ("setup", "setups")
            show_weights = _suffix in ("weights", "-w")
            show_split = _suffix in ("split", "-s")
            show_levels = _suffix in ("levels", "-l")
            show_live = _suffix in ("live", "-v")
            live_full = False
            live_time_range = None
            live_time_range_label = ""
            live_default_full_range = False
            _live_args = _parts[2:] if show_live else []
            if _live_args and _live_args[0].lower() == "full":
                live_full = True
                _live_args = _live_args[1:]
            if show_live and _live_args:
                _live_range_spec = " ".join(_live_args)
                _live_range = _parse_live_time_range(_live_range_spec)
                if _live_range is None:
                    print(f"  Invalid live option/range: '{_live_range_spec}' — use e.g. full, 12-16, 12:10-16:05, or 12:10:05 16:10:05 (PT)")
                    continue
                live_time_range = (_live_range[0], _live_range[1])
                live_time_range_label = _live_range[2]
            elif live_full:
                live_time_range = _premarket_window_utc()
                live_time_range_label = "full premarket"
                live_default_full_range = True

            ticker = validate_ticker(raw)
            if ticker is None:
                print(f"  Invalid ticker: '{raw}'")
                continue

            if show_setups:
                day_str = MarketClock.today().strftime("%Y%m%d")
                _print_setup_signals(day_str, _day_setup_signals(day_str), ticker=ticker)
                continue

            def _ensure_cached_for_injection(sym: str) -> Optional[tuple]:
                """Ensure symbol has a cached entry before applying manual overrides."""
                _cached_local = cache.get(sym)
                if _cached_local:
                    return _cached_local

                print(f"  {_dim(f'{sym} not in cache — evaluating first...')}")
                _pending_yt_local = cache.get_pending_yt_news(sym)
                _result_local = run_pipeline(
                    sym, evaluator, tradier,
                    yt_headlines=_pending_yt_local,
                    force=True,
                    cached_news=cache.get_news(sym),
                    on_scorer_update=_invalidate_toolbar,
                )
                if _result_local is None:
                    print(f"  {_y('⚠')}  {sym} could not be evaluated for injection")
                    return None

                cache.pop_pending_yt_news(sym)
                cache.put(sym, _result_local.entry, _result_local.rmap)
                cache.flush()

                if _result_local.passed:
                    rejected.discard(sym)
                    faded.discard(sym)
                    if _result_local.scorer is not None:
                        _old_sc = scorers.pop(sym, None)
                        if _old_sc:
                            _old_sc.stop()
                        scorers[sym] = _result_local.scorer
                else:
                    evaluator.watchlist = [e for e in evaluator.watchlist if e.symbol != sym]
                    evaluator.watchlist_symbols.discard(sym)
                    _remember_pipeline_failure(rejected, sym, _result_local.fail_reasons, evaluator.last_fail_reason or "")

                return (_result_local.entry, _result_local.rmap)

            if mute_ticker:
                if ticker in muted_tickers:
                    print(f"  {_dim(f'{ticker} output already muted')}")
                else:
                    muted_tickers.add(ticker)
                    logger.info("[display] %s output muted", ticker)
                continue

            if unmute_ticker:
                if ticker not in muted_tickers:
                    print(f"  {_dim(f'{ticker} output is not muted')}")
                else:
                    muted_tickers.discard(ticker)
                    logger.info("[display] %s output unmuted", ticker)
                continue

            if clear_cache:
                if cache.remove(ticker):
                    cache.flush()
                    print(f"  {_dim(f'{ticker} removed from cache.')}")
                else:
                    print(f"  {_y('⚠')}  {ticker} not in cache")
                continue

            if show_weights:
                entry = next((e for e in evaluator.watchlist if e.symbol == ticker), None)
                if entry is None:
                    cached = cache.get(ticker)
                    if cached:
                        entry = cached[0]
                if entry is None or entry.evaluation is None:
                    print(f"  {_y('⚠')}  {ticker} not evaluated — run it first")
                    continue
                if ticker in _weights_active:
                    _weights_active.discard(ticker)
                    print(f"  {_dim(f'{ticker} weights display closed')}")
                else:
                    _weights_active.add(ticker)
                    print(f"  {_dim(f'{ticker} weights display started — q to close')}")
                continue

            if show_found_at:
                first_seen_ts = cache.get_ts(ticker)
                if first_seen_ts is not None:
                    print(f"  {_b(ticker)}  first found  {_dim(MarketClock.fmt_local_ts(first_seen_ts))}")
                    continue

                entry = next((e for e in evaluator.watchlist if e.symbol == ticker), None)
                if entry is not None:
                    print(f"  {_b(ticker)}  first found  {_dim(MarketClock.fmt_local(entry.evaluated_at))}")
                else:
                    print(f"  {_y('⚠')}  {ticker} has not been found yet")
                continue

            # Show move start timestamp
            if show_start:
                sc = scorers.get(ticker)
                if sc is not None and sc._move_start_ts is not None:
                    _ms_local = sc._move_start_ts.astimezone(MarketClock.local_tz())
                    print(f"  {_b(ticker)}  move start  {_dim(_ms_local.strftime('%H:%M:%S'))}")
                else:
                    # Compute from 1min candles on the fly
                    from technical_analysis.patterns import detect_move_start, latest_extended_session_start
                    import pandas as pd
                    _start_utc = latest_extended_session_start()
                    _df_1m = tradier.get_candles_1min_window(ticker, _start_utc)
                    if _df_1m.empty:
                        print(f"  {_y('⚠')}  {ticker}: no 1-min candle data available")
                    else:
                        _move = detect_move_start(_df_1m)
                        if _move is None:
                            print(f"  {_y('⚠')}  {ticker}: no move detected")
                        else:
                            try:
                                _ms_ts = pd.Timestamp(_df_1m.iloc[_move.start_idx]["ts"])
                                if _ms_ts.tzinfo is None:
                                    _ms_ts = _ms_ts.tz_localize("UTC")
                                _ms_local = _ms_ts.to_pydatetime().astimezone(MarketClock.local_tz())
                                print(
                                    f"  {_b(ticker)}  move start  {_dim(_ms_local.strftime('%H:%M:%S'))}"
                                    f"  {_dim(f'({_move.direction} {_move.price_change_pct:+.1f}% from ${_move.start_price:.2f})')}"
                                )
                            except (IndexError, KeyError):
                                print(f"  {_y('⚠')}  {ticker}: could not resolve move start timestamp")
                continue

            # Mark current headline as noise and find another
            if mark_noise:
                from cache import NoiseStore
                cached = _ensure_cached_for_injection(ticker)
                if not cached:
                    continue
                entry, rmap = cached
                _excluded = entry.news_articles[0].title if entry.news_articles else None
                if not _excluded:
                    print(f"  {_y('⚠')}  {ticker} has no headline to mark as noise")
                    continue
                # Persist to both the entry field and the global noise store
                entry.noise_titles.add(_excluded)
                NoiseStore.get().add(ticker, _excluded)
                print(f"  {_dim(f'Marked as noise: {chr(34)}{_excluded}{chr(34)}')}")
                news_refreshed.discard(ticker)
                _all_noise = entry.noise_titles | NoiseStore.get().titles(ticker)
                _updated = evaluator.refresh_news(entry, exclude_titles=_all_noise, allow_older=True)
                if _updated:
                    cache.put(ticker, entry, rmap)
                    cache.flush()
                    print(f"  {_g('✓')}  {ticker} replacement news: {entry.news_source}")
                else:
                    entry.news_articles = []
                    entry.news_count_last_hour = 0
                    entry.avg_sentiment = None
                    entry.news_source = ""
                    entry.news_is_fresh = False
                    entry.news_is_breaking = False
                    if entry.evaluation:
                        entry.evaluation.update_news(None, 0)
                    cache.put(ticker, entry, rmap)
                    cache.flush()
                    print(f"  {_y('⚠')}  {ticker} no replacement news found")
                _print_stock_summary(entry, rmap)
                continue

            # Split info: TICKER split (show) or TICKER split 5/1 [date] (set)
            if show_split:
                _split_args = _parts[2:] if len(_parts) >= 3 else []
                cached = cache.get(ticker)
                if not cached:
                    # Only auto-evaluate for mutating split-set mode.
                    if _split_args:
                        cached = _ensure_cached_for_injection(ticker)
                        if not cached:
                            continue
                    else:
                        print(f"  {_y('⚠')}  {ticker} not in cache — run it first")
                        continue
                entry, rmap = cached
                # If extra args provided → set mode
                if _split_args:
                    # Parse: colon = ratio (5:1), slash/dash = date (5/1, 2026-05-01)
                    import re as _re_split
                    _ratio_val: float | None = None
                    _date_val: str | None = None
                    _year = MarketClock.today().year
                    for _arg in _split_args:
                        # Ratio: digits:digits
                        _m_ratio = _re_split.match(r"^(\d+):(\d+)$", _arg)
                        if _m_ratio:
                            _num, _den = int(_m_ratio.group(1)), int(_m_ratio.group(2))
                            _ratio_val = max(_num, _den) / min(_num, _den) if min(_num, _den) > 0 else None
                            continue
                        # Date: YYYY-MM-DD or YYYY/MM/DD
                        _m_full = _re_split.match(r"^(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})$", _arg)
                        if _m_full:
                            _date_val = f"{_m_full.group(1)}-{int(_m_full.group(2)):02d}-{int(_m_full.group(3)):02d}"
                            continue
                        # Date: M/D or MM/DD or M-D or MM-DD (no year → current year)
                        _m_short = _re_split.match(r"^(\d{1,2})[/\-](\d{1,2})$", _arg)
                        if _m_short:
                            _date_val = f"{_year}-{int(_m_short.group(1)):02d}-{int(_m_short.group(2)):02d}"
                            continue
                    if _date_val is None:
                        _date_val = MarketClock.now_et().strftime("%Y-%m-%d")
                    entry.has_recent_reverse_split = True
                    entry.reverse_split_date = _date_val
                    if _ratio_val is not None:
                        entry.reverse_split_ratio = _ratio_val
                    if entry.evaluation:
                        pass  # reverse_split no longer a scoring variable
                    cache.put(ticker, entry, rmap)
                    cache.flush()
                    _ratio_str = f" ({entry.reverse_split_ratio:.0f}-for-1)" if entry.reverse_split_ratio else ""
                    logger.info("[%s] Split manually set: %s%s", ticker, _date_val, _ratio_str)
                    print(f"  {_g('✓')}  {ticker} split updated: {_date_val}{_ratio_str}")
                else:
                    # Show mode
                    if entry.has_recent_reverse_split:
                        _ratio_str = f" ({entry.reverse_split_ratio:.0f}-for-1)" if entry.reverse_split_ratio else ""
                        print(f"  {_b(ticker)}  reverse split: {entry.reverse_split_date or '?'}{_ratio_str}")
                    else:
                        print(f"  {_b(ticker)}  no recent reverse split")
                continue

            # Levels: TICKER levels — toggle live levels display in toolbar
            if show_levels:
                cached = cache.get(ticker)
                if not cached:
                    print(f"  {_y('⚠')}  {ticker} not in cache — run it first")
                    continue
                if ticker in _levels_active:
                    _levels_active.discard(ticker)
                    print(f"  {_dim(f'{ticker} levels display closed')}")
                else:
                    _levels_active.add(ticker)
                    print(f"  {_dim(f'{ticker} levels display started — q to close')}")
                continue

            # Live candle detail: TICKER live — per-candle quality breakdown
            if show_live:
                sc = scorers.get(ticker)
                if live_full and sc is not None:
                    if live_default_full_range:
                        _active_start = sc._active_price_action_start(datetime.now(tz=timezone.utc))
                        live_time_range = (_active_start, datetime.now(tz=timezone.utc))
                        _active_start_local = _active_start.astimezone(MarketClock.local_tz())
                        live_time_range_label = f"from move start {_active_start_local.strftime('%H:%M')}"
                    print(f"  {_dim(f'{ticker} full live — showing active scorer candles')}")
                elif live_full:
                    _hist_sc = _build_historical_live_snapshot(
                        ticker,
                        tradier,
                        time_range=live_time_range,
                        full_premarket=True,
                        include_10s=False,
                    )
                    if _hist_sc is not None:
                        sc = _hist_sc
                        print(f"  {_dim(f'{ticker} full live — showing historical premarket candles (1m/2m/5m/1d)')}")
                if sc is None:
                    sc = _build_historical_live_snapshot(ticker, tradier, time_range=live_time_range)
                    if sc is None:
                        print(f"  {_y('⚠')}  {ticker} has no active scorer and no historical live candles yet — run it first")
                        continue
                    print(f"  {_dim(f'{ticker} has no active scorer — showing historical candles')}")
                print_live_detail(
                    ticker,
                    sc,
                    time_range=live_time_range,
                    time_range_label=live_time_range_label,
                    **({
                        "max_rows": None,
                        "sort_by_severity": False,
                    } if live_full else {}),
                )
                continue

            # Manual float override: TICKER float 2.5
            if set_float is not None:
                cached = _ensure_cached_for_injection(ticker)
                if not cached:
                    continue
                entry, rmap = cached
                _old = entry.float_shares_m
                entry.float_shares_m = set_float
                if entry.evaluation:
                    sup = entry.evaluation.stock_eval
                    v = sup.get("float_shares")
                    if v:
                        v.evaluate(set_float)
                # Clear "float" from missing_data since we now have a value
                if hasattr(entry, 'missing_data') and "float" in entry.missing_data:
                    entry.missing_data.remove("float")
                entry.static_confidence = entry.eval_confidence
                cache.put(ticker, entry, rmap)
                cache.flush()
                logger.info("[%s] Float manually set: %.2fM → %.2fM", ticker, _old, set_float)
                print(f"  {_g('✓')}  {ticker} float updated: {_old:.2f}M → {set_float:.2f}M")
                _print_stock_summary(entry, rmap)
                continue

            # Manual news override: TICKER news "headline"
            if set_news is not None or set_sentiment is not None:
                cached = _ensure_cached_for_injection(ticker)
                if not cached:
                    continue
                entry, rmap = cached
                if set_sentiment is not None:
                    # Direct sentiment override — no headline change
                    _sent = set_sentiment
                    _old_sent = entry.avg_sentiment
                    entry.avg_sentiment = _sent
                    if entry.evaluation:
                        entry.evaluation.update_news(_sent, entry.news_count_last_hour or 1, entry.news_source or "manual")
                    entry.static_confidence = entry.eval_confidence
                    cache.put(ticker, entry, rmap)
                    cache.flush()
                    _old_str = f"{_old_sent:.2f}" if _old_sent is not None else "N/A"
                    logger.info("[%s] Sentiment manually set: %s → %.2f", ticker, _old_str, _sent)
                    print(f"  {_g('✓')}  {ticker} sentiment: {_old_str} → {_sent:.2f}")
                    _print_stock_summary(entry, rmap)
                    continue
                # Score headline with FinBERT if available
                _sent: float | None = None
                if FINBERT_ENABLED:
                    _sent = evaluator._finbert.score_headline(ticker, set_news)
                _now = set_news_time if set_news_time is not None else datetime.now(tz=timezone.utc)
                _article = FinnhubNewsArticle(
                    symbol=ticker, title=set_news, source="manual",
                    url="", published_at=_now, summary="",
                )
                session_cutoff, breaking_cutoff, _ = news_freshness_cutoffs()
                entry.news_articles = [_article]
                entry.news_count_last_hour = 1
                entry.avg_sentiment = _sent
                entry.news_source = "manual"
                entry.news_is_fresh = _now >= session_cutoff
                entry.news_is_breaking = _now >= breaking_cutoff
                if entry.evaluation:
                    entry.evaluation.update_news(_sent, 1, "manual")
                entry.static_confidence = entry.eval_confidence
                cache.put(ticker, entry, rmap)
                cache.flush()
                _sent_str = f"{_sent:.2f}" if _sent is not None else "N/A"
                logger.info("[%s] News manually set: %r (sentiment=%s)", ticker, set_news, _sent_str)
                print(f"  {_g('✓')}  {ticker} news set: \"{set_news}\" (sentiment={_sent_str})")
                _print_stock_summary(entry, rmap)
                continue

            # Check cache first — all inputs consult cache before running pipeline
            cached = None if force_refresh else cache.get(ticker)
            if cached:
                entry, rmap = cached
                _s1_snapshot = None
                _current_fail_reasons: list[str] = []
                # Inject any pending YouTube news into the cached entry
                _pending_yt = cache.pop_pending_yt_news(ticker)
                if _pending_yt:
                    _inject_yt_news(ticker, _pending_yt, evaluator, cache)
                    cached = cache.get(ticker)
                    if cached:
                        entry, rmap = cached
                if apply_cached_news(entry, cache.get_news(ticker)):
                    cache.put(ticker, entry, rmap)
                    cache.flush()
                # Refresh the Stage 1 snapshot fields — price/vol/rvol are stale
                # the moment they're cached. One fast quote call keeps the summary honest.
                if MARKET_DATA_ENABLED:
                    _s1_snapshot, _current_fail_reasons = _refresh_cached_stage1_snapshot(ticker, entry, evaluator)
                else:
                    _current_fail_reasons = _current_cached_fail_reasons(entry)
                # Fade-from-HOD removed: it only ever fired while you were
                # looking at the ticker, so it reclaimed no slots from the
                # names nobody queried. The IBKR scanner's drop check
                # (SCANNER_IBKR_DROP_SECS) now releases dead names on its
                # own cadence instead. `faded` remains — the drop path and
                # the FADED display section both still use it.
                # Refresh news freshness tags (display only) and re-score.
                # If cached news is 1+ day old, re-query sources for fresher articles.
                _news_stale = False
                if entry.news_articles:
                    _pub = entry.news_articles[0].published_at
                    if _pub.tzinfo is None:
                        _pub = _pub.replace(tzinfo=timezone.utc)
                    _, _, _stale_cut = news_freshness_cutoffs()
                    _news_stale = _pub < _stale_cut
                else:
                    _news_stale = True

                if _news_stale and ticker not in news_refreshed:
                    news_refreshed.add(ticker)
                    from cache import NoiseStore
                    _all_noise = (entry.noise_titles or set()) | NoiseStore.get().titles(ticker)
                    _updated = evaluator.refresh_news(entry, exclude_titles=_all_noise or None)
                    if _updated:
                        cache.put(ticker, entry, rmap)
                        cache.flush()

                if entry.news_articles:
                    session_cutoff, breaking_cutoff, _ = news_freshness_cutoffs()
                    pub = entry.news_articles[0].published_at
                    entry.news_is_fresh = pub >= session_cutoff
                    entry.news_is_breaking = pub >= breaking_cutoff
                    if entry.evaluation:
                        entry.evaluation.update_news(entry.avg_sentiment, 1, entry.news_source)

                _current_fail_reasons = _current_cached_fail_reasons(entry, _s1_snapshot)
                if MARKET_DATA_ENABLED and _failures_are_soft(_current_fail_reasons):
                    _s1_snapshot, _current_fail_reasons = _refresh_cached_stage1_snapshot(ticker, entry, evaluator)

                if ticker not in faded and not _current_fail_reasons:
                    rejected.discard(ticker)
                    _ensure_cached_scorer(
                        ticker, entry, rmap, evaluator, tradier, scorers,
                        on_update=_invalidate_toolbar,
                    )
                elif _failures_are_soft(_current_fail_reasons):
                    rejected.discard(ticker)

                _print_stock_summary(entry, rmap, fail_reasons=_current_fail_reasons or None)
                continue

            # "TICKER cache" — only show cached data, don't run the pipeline
            if show_cache:
                print(f"  {_y('⚠')}  {ticker} not in cache")
                continue

            # Hard-rejected tickers are skipped unless user forces a refresh
            if ticker in rejected and not force_refresh:
                print(f"  {_r('✗')}  {ticker} — previously rejected")
                continue

            # Collect any pending YouTube news headlines for this ticker
            _pending_yt = cache.get_pending_yt_news(ticker)

            # Preserve cached news during refresh; run_pipeline applies it
            # before printing if live sources do not return anything newer.
            _cached_news = cache.get_news(ticker)

            if force_refresh:
                old = scorers.pop(ticker, None)
                if old:
                    old.stop(wait=False)

            logger.info("[%s] running pipeline%s", ticker,
                        " (refresh)" if force_refresh else "")
            result = run_pipeline(
                ticker, evaluator, tradier,
                yt_headlines=_pending_yt,
                force=True,
                cached_news=_cached_news,
                on_scorer_update=_invalidate_toolbar,
            )

            if result is not None:
                apply_cached_news(result.entry, _cached_news)

            if result is not None and result.passed:
                cache.pop_pending_yt_news(ticker)
                rejected.discard(ticker)  # passed — clear any stale rejection
                faded.discard(ticker)     # refresh clears fade status
                cache.put(ticker, result.entry, result.rmap)
                cache.flush()
                if result.scorer is not None:
                    scorers[ticker] = result.scorer
            elif result is not None:
                cache.pop_pending_yt_news(ticker)
                # Did not fully pass but force-evaluated — still cache so
                # late-arriving YouTube news injection can find the entry.
                cache.put(ticker, result.entry, result.rmap)
                cache.flush()
                # Remove from live watchlist (force mode adds it for display only)
                evaluator.watchlist = [e for e in evaluator.watchlist if e.symbol != ticker]
                evaluator.watchlist_symbols.discard(ticker)
                _remember_pipeline_failure(rejected, ticker, result.fail_reasons, evaluator.last_fail_reason or "")
            else:
                # Pipeline returned None (e.g. float API completely unavailable)
                _remember_pipeline_failure(rejected, ticker, [], evaluator.last_fail_reason or "")


if __name__ == "__main__":
    main()
