"""
pipeline.py  —  Core evaluation pipeline for a single ticker.

Orchestrates the full flow:
    Stage 1 (Tradier hard filter)
  → Stage 2 (enrichment + static confidence)
  → Indicator builds (all timeframes, in parallel)
  → Resistance map construction
  → Dynamic confidence scorer (background thread)
"""

import logging
import re
import time as _time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from market_data.tradier_rest import HistoricalClient

from strategy.evaluator import StockEvaluator
from config import (
    MARKET_DATA_ENABLED,
    CANDLES_DAILY_LOOKBACK,
    FLOAT_MAX_M,
    RVOL_1DAY_MIN,
    CHANGE_PCT_MIN,
    ATTENTION_VOLUME_MIN,
    ATTENTION_RVOL_1DAY_MIN,
    _r, _y, _b, _dim,
)
from strategy.confidence_scorer import DynamicScorer
from chart_cache import load_candle_cache
from technical_analysis.indicators import (
    build_1min, build_5min, build_daily,
)
from technical_analysis.patterns import detect_move_start, latest_extended_session_start
from technical_analysis.resistance import ResistanceMap
from market_data.provider import (
    get_shared_stream, release_thread_connection, stream_enabled,
)
from market_data.tradier_rest import TradierClientError
from market_data.tradier_stream import LiveTimesale
from display import _print_stock_summary, safe_print
from cache import CachedNews, apply_cached_news

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Za-z]{1,5}$")


def validate_ticker(raw: str) -> str | None:
    """
    Validate and normalise a ticker input.
    Returns uppercased ticker string on success, None on failure.
    Rules: 1–5 characters, letters only, no numbers or spaces.
    """
    cleaned = raw.strip()
    if _TICKER_RE.match(cleaned):
        return cleaned.upper()
    return None


# ---------------------------------------------------------------------------
# Cached-entry Stage 1 refresh (shared by main REPL + scanner consumer)
# ---------------------------------------------------------------------------

def _apply_stage1_snapshot(entry, s1) -> bool:
    """Refresh cached volatile quote fields from a fresh Stage 1 snapshot."""
    if getattr(s1, "fail_reason", "") == "symbol not found":
        return False

    price = float(getattr(s1, "price", 0.0) or 0.0)
    if price <= 0:
        return False

    entry.price = price
    entry.change_pct = float(getattr(s1, "change_pct", 0.0) or 0.0)
    entry.volume = int(getattr(s1, "volume", 0) or 0)
    # IBKR quotes carry no avVolume (clamped to 1 in Stage 1) — never let
    # that overwrite the daily-bar average derived at evaluation time.
    _s1_avg = int(getattr(s1, "avg_volume", 0) or 0)
    if _s1_avg > 1:
        entry.avg_volume = _s1_avg
    else:
        entry.avg_volume = max(int(getattr(entry, "avg_volume", 1) or 1), 1)
    entry.rvol_1day = float(getattr(s1, "rvol_1day", 0.0) or 0.0)
    entry.rvol_5min = float(getattr(s1, "rvol_5min", 0.0) or 0.0)

    daily_high = float(getattr(s1, "daily_high", 0.0) or 0.0)
    if daily_high > 0:
        entry.daily_high = max(float(getattr(entry, "daily_high", 0.0) or 0.0), daily_high)

    if entry.evaluation:
        for var_name, value in (
            ("price", entry.price),
            ("rvol_1day", entry.rvol_1day),
        ):
            for group in entry.evaluation.groups:
                var = group.get(var_name)
                if var is not None:
                    var.evaluate(value)
                    break
        entry.static_confidence = entry.eval_confidence

    for group in getattr(entry, "criteria_groups", []) or []:
        if getattr(group, "name", "") != "ATTENTION":
            continue
        group.update("change_pct_min", entry.change_pct >= CHANGE_PCT_MIN, entry.change_pct)
        group.update("volume_min", entry.volume >= ATTENTION_VOLUME_MIN, float(entry.volume))
        group.update("rvol_1day_min", entry.rvol_1day >= ATTENTION_RVOL_1DAY_MIN, entry.rvol_1day)
        break

    return True


def _current_cached_fail_reasons(entry, s1=None) -> list[str]:
    reasons: list[str] = []
    if s1 is not None and not getattr(s1, "passed", False):
        reason = getattr(s1, "fail_reason", "")
        if reason:
            reasons.append(reason)
    float_m = float(getattr(entry, "float_shares_m", 0.0) or 0.0)
    if float_m > FLOAT_MAX_M:
        reasons.append(f"float: {float_m:.2f}M (> {FLOAT_MAX_M:.0f}M)")
    rvol_1day = float(getattr(entry, "rvol_1day", 0.0) or 0.0)
    if 0 < rvol_1day < RVOL_1DAY_MIN:
        reasons.append(f"rvol_1day: {rvol_1day:.1f}x (< {RVOL_1DAY_MIN:.0f}x)")
    return reasons


def _refresh_cached_stage1_snapshot(ticker: str, entry, evaluator) -> tuple[object | None, list[str]]:
    """Fetch a fresh Stage 1 quote (via the active market-data provider)
    and apply it to a cached entry. Returns (s1, current fail reasons).

    Never raises — on any fetch/apply failure the cached fields are left
    untouched and the failure is logged at debug level.
    """
    s1 = None
    try:
        s1 = evaluator.stage1(ticker)
        _apply_stage1_snapshot(entry, s1)
    except Exception as exc:
        logger.debug(f"[{ticker}] Cache hit quote refresh failed: {exc}")
    return s1, _current_cached_fail_reasons(entry, s1)


def _normalize_candle_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    normalized = df.copy()
    if "ts" in normalized.columns:
        normalized["ts"] = pd.to_datetime(normalized["ts"], utc=True)
        normalized = (
            normalized.drop_duplicates(subset=["ts"], keep="last")
            .sort_values("ts")
            .reset_index(drop=True)
        )
    return normalized


def _clip_since(df: pd.DataFrame, start_ts) -> pd.DataFrame:
    if df.empty or start_ts is None or "ts" not in df.columns:
        return df.copy()
    clipped = df[pd.to_datetime(df["ts"], utc=True) >= pd.Timestamp(start_ts)]
    return clipped.reset_index(drop=True)


def _resample_ohlcv(df: pd.DataFrame, symbol: str, rule: str, interval: str) -> pd.DataFrame:
    if df.empty or "ts" not in df.columns:
        return pd.DataFrame()
    indexed = df.set_index("ts")
    resampled = indexed.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna().reset_index()
    if resampled.empty:
        return resampled
    resampled["symbol"] = symbol
    resampled["interval"] = interval
    resampled["vwap"] = None
    return resampled


def _detect_move_start_ts(df_1min: pd.DataFrame):
    if df_1min.empty or "ts" not in df_1min.columns:
        return None
    move = detect_move_start(df_1min)
    if move is None:
        return None
    try:
        return pd.Timestamp(df_1min.iloc[move.start_idx]["ts"]).to_pydatetime()
    except (IndexError, KeyError):
        return None


def _fetch_extended_1min(tradier, symbol: str) -> pd.DataFrame:
    start = latest_extended_session_start()
    fetch_window = getattr(tradier, "get_candles_1min_window", None)
    if fetch_window is not None:
        return _normalize_candle_df(fetch_window(symbol, start))
    # Fallback for alternate providers that do not yet support precise windows.
    return _normalize_candle_df(tradier.get_candles_1min(symbol, n=2000))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EarlyStream:
    """Live T&S candle buffer started once the float gate passes.

    Started in parallel with news/splits so data is already accumulating
    by the time the pipeline completes.  Call ``stop()`` to tear down.
    """
    candles: list = field(default_factory=list)
    stop_event: Optional[threading.Event] = None  # kept for API compat; unused
    active: bool = False
    _symbol: str = field(default="", init=False, repr=False)

    def start(self, symbol: str) -> None:
        if not stream_enabled():
            return
        try:
            self._symbol = symbol
            candles: list = []
            self.candles = candles
            lock = threading.Lock()
            interval_seconds = 10

            def _bucket_ts(ts: "datetime") -> "datetime":  # type: ignore[name-defined]
                from datetime import datetime, timezone
                epoch = ts.timestamp()
                floored = (epoch // interval_seconds) * interval_seconds
                return datetime.fromtimestamp(floored, tz=timezone.utc)

            def on_ts(event: LiveTimesale) -> None:
                if event.cancel or event.correction:
                    return
                bar_ts = _bucket_ts(event.ts)
                with lock:
                    if candles and candles[-1].ts == bar_ts:
                        c = candles[-1]
                        c.high = max(c.high, event.last)
                        c.low = min(c.low, event.last)
                        c.close = event.last
                        c.volume += event.size
                    else:
                        from market_data.tradier_rest import OHLCVCandle
                        candles.append(OHLCVCandle(
                            symbol=symbol,
                            ts=bar_ts,
                            open=event.last,
                            high=event.last,
                            low=event.last,
                            close=event.last,
                            volume=event.size,
                            interval="10sec",
                        ))

            get_shared_stream().subscribe(
                symbol, on_ts, lambda e: None, sub_id=f"early:{symbol}"
            )
            self.active = True
            logger.debug(f"[{symbol}] Early T&S stream started (float passed)")
        except Exception as exc:
            logger.debug(f"[{symbol}] Early stream failed to start: {exc}")

    def stop(self) -> None:
        if self._symbol:
            try:
                get_shared_stream().unsubscribe(self._symbol, sub_id=f"early:{self._symbol}")
            except Exception:
                pass
        self.active = False


@dataclass
class PipelineResult:
    """Structured return from run_pipeline on success."""
    entry: object               # WatchlistEntry
    rmap: object                # ResistanceMap
    scorer: Optional[object]    # DynamicScorer or None
    early_stream: EarlyStream   # live T&S candle buffer
    passed: bool = True         # False when force=True but stages failed
    fail_reasons: list = field(default_factory=list)  # Stage 1/2 failure reasons   # live T&S candle buffer


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    symbol: str,
    evaluator: StockEvaluator,
    tradier: "HistoricalClient",
    yt_headlines: Optional[list] = None,
    quiet: bool = False,
    force: bool = False,
    allow_manual_rating: bool = True,
    cached_float: Optional[float] = None,
    cached_news: CachedNews | None = None,
    on_scorer_update: Optional[callable] = None,
) -> Optional[PipelineResult]:
    """
    Run the full pipeline for a single ticker.

    Stage 1 — Tradier hard filter (price / vol / rvol)
    Stage 2 + candle fetches — fired in parallel after Stage 1 passes
    Indicators built from pre-fetched candles.
    Resistance map built.
    Dynamic confidence scorer started in background.

    Returns PipelineResult on success, None on fail.

    *force* — if True, continue evaluation even when Stage 1 or Stage 2
    gates fail so the full summary is always displayed for manual queries.
    The returned PipelineResult will have ``passed=False`` and the gate
    failures listed in ``fail_reasons``.
    """

    fail_reasons: list[str] = []
    # The evaluator's fail reason is process-wide state; clear the previous
    # symbol's reason so a Stage 2 failure that sets nothing specific can't
    # be reported with a stale Stage 1 message (a float-lookup failure used
    # to print the prior ticker's "price $81.60 not in range").
    evaluator.last_fail_reason = ""

    # ------------------------------------------------------------------
    # Stage 1  (quick Tradier quote check)
    # ------------------------------------------------------------------
    _t_s1 = _time.perf_counter()
    s1 = evaluator.stage1(symbol)
    evaluator._stage1_ms = (_time.perf_counter() - _t_s1) * 1000
    if not s1.passed:
        # "symbol not found" is irrecoverable — bail even under force=True,
        # otherwise the parallel candle/Momo fetches all hammer a bogus
        # symbol (timesales 400, history 10s timeout, livequote 400) before
        # we finally report ✗.
        #
        # "market data unavailable" also bails early, for the opposite
        # reason: the symbol may be perfectly valid, but the feed is down,
        # so the downstream fetches would fail too.  Reported distinctly so
        # a provider outage is never mistaken for a bad ticker — retrying
        # once the feed recovers is the correct response.
        if s1.fail_reason in ("symbol not found", "market data unavailable"):
            if not quiet:
                safe_print(f"  {_r('✗')}  {symbol} — {s1.fail_reason}")
            evaluator.last_fail_reason = s1.fail_reason
            return None
        if not force:
            if not quiet:
                safe_print(f"  {_r('✗')}  {symbol} — {s1.fail_reason}")
            evaluator.last_fail_reason = s1.fail_reason
            return None
        # force=True: record the failure but continue evaluation
        fail_reasons.append(s1.fail_reason)
        evaluator.last_fail_reason = s1.fail_reason

    # ------------------------------------------------------------------
    # Stage 2 + candle fetches in parallel
    # ------------------------------------------------------------------

    df_daily_pre = pd.DataFrame()
    df_1min_seed = pd.DataFrame()
    df_10sec_seed = pd.DataFrame()
    move_start_ts = None

    # --- Early stream: T&S candle buffer started once float passes ---
    early_stream = EarlyStream()

    with ThreadPoolExecutor(max_workers=6) as pool:
        def _fetch_daily():
            _t0 = _time.perf_counter()
            try:
                df = tradier.get_candles_daily(
                    symbol,
                    days_back=CANDLES_DAILY_LOOKBACK,
                    include_current_day=False,
                )
                result = build_daily(df) if not df.empty else df
            finally:
                # Pool workers are discarded when the `with` block exits;
                # release the IBKR slot they bound rather than leaking one
                # connection per evaluated ticker.
                release_thread_connection()
            evaluator._daily_candles_ms = (_time.perf_counter() - _t0) * 1000
            return result

        def _fetch_1min():
            try:
                return _fetch_extended_1min(tradier, symbol)
            finally:
                release_thread_connection()

        daily_fut = pool.submit(_fetch_daily)
        intraday_1m_fut = pool.submit(_fetch_1min) if MARKET_DATA_ENABLED else None

        try:
            df_daily_pre = daily_fut.result()
        except Exception as exc:
            logger.debug(f"[{symbol}] Daily candle fetch failed: {exc}")

        entry = evaluator.stage2(
            s1, df_daily=df_daily_pre if not df_daily_pre.empty else None,
            on_float_pass=lambda: early_stream.start(symbol),
            yt_headlines=yt_headlines,
            skip_float_gate=force,
            cached_float=cached_float)

    if entry is None:
        early_stream.stop()
        if not force:
            if not quiet:
                safe_print(f"  {_r('✗')}  {symbol} — {evaluator.last_fail_reason}")
            return None
        # force=True but stage2 still returned None (e.g. float API completely unavailable)
        if not quiet:
            safe_print(f"  {_r('✗')}  {symbol} — {evaluator.last_fail_reason}")
        return None

    apply_cached_news(entry, cached_news)

    # --- Static confidence gate ---
    from config import STATIC_CONFIDENCE_MIN
    if not force and entry.static_confidence < STATIC_CONFIDENCE_MIN:
        early_stream.stop()
        rmap_lite = ResistanceMap(symbol=symbol, current_price=entry.price)
        rmap_lite.build_static()
        if not quiet:
            _print_stock_summary(
                entry, rmap_lite,
                fail_reasons=[f"static confidence {entry.static_confidence:.0f}% < {STATIC_CONFIDENCE_MIN:.0f}%"],
            )
        return PipelineResult(
            entry=entry, rmap=rmap_lite, scorer=None,
            early_stream=early_stream, passed=False,
            fail_reasons=[f"static confidence {entry.static_confidence:.0f}% < {STATIC_CONFIDENCE_MIN:.0f}%"],
        )

    # --- Manual news rating (fallback — only when sentiment scoring failed) ---
    if (
        allow_manual_rating
        and not quiet
        and entry.news_count_last_hour > 0
        and entry.avg_sentiment is None
    ):
        # Map 1-5 rating to a [-1, 1] sentiment score → then to [0, 1] quality
        _RATING_MAP = {
            1: -0.8,    # very bearish
            2: -0.4,    # bearish
            3:  0.0,    # neutral
            4:  0.4,    # bullish
            5:  0.8,    # very bullish
        }
        print(f"  Rate news for {_b(symbol)}:  {_dim('1=very bearish  2=bearish  3=neutral  4=bullish  5=very bullish  Enter=skip')}")
        try:
            raw_rating = input("  Rating (1-5) > ").strip()
            if raw_rating in ("1", "2", "3", "4", "5"):
                sent = _RATING_MAP[int(raw_rating)]
                entry.avg_sentiment = sent
                # Update the breaking_news quality variable
                if entry.evaluation:
                    entry.evaluation.update_news(sent, 1)
        except (KeyboardInterrupt, EOFError):
            pass

    # ------------------------------------------------------------------
    # Build indicators + resistance (silent)
    # ------------------------------------------------------------------

    df_1min = df_5min = df_daily = pd.DataFrame()
    rmap = ResistanceMap(symbol=symbol, current_price=entry.price)

    if not MARKET_DATA_ENABLED:
        rmap.build_static()
    else:
        avg_vol = entry.avg_volume
        try:
            frames = {"Daily": df_daily_pre if not df_daily_pre.empty else pd.DataFrame()}
            if intraday_1m_fut is not None:
                df_1min_seed = _normalize_candle_df(intraday_1m_fut.result())

            move_start_ts = _detect_move_start_ts(df_1min_seed)
            df_1min_move = _clip_since(df_1min_seed, move_start_ts)

            frames["1-min"] = build_1min(df_1min_move, avg_vol) if not df_1min_move.empty else df_1min_move

            df_5min_raw = _resample_ohlcv(df_1min_move, symbol, "5min", "5min")
            frames["5-min"] = build_5min(df_5min_raw, avg_vol) if not df_5min_raw.empty else df_5min_raw

        except TradierClientError as exc:
            early_stream.stop()
            if not quiet:
                safe_print(f"  {_r('✗')}  {symbol} — rejected by Tradier ({exc})")
            return None

        try:
            df_1min  = frames["1-min"]
            df_daily = frames["Daily"]
            df_5min  = frames["5-min"]

        except Exception as exc:
            logger.warning(f"[{symbol}] Indicator build error: {exc}")
            safe_print(f"  {_y('⚠')}  {symbol} — indicator build partially failed: {exc}")

        try:
            rmap.build_static()
            rmap.build_structure(df_daily, df_5min)
        except Exception as exc:
            logger.warning(f"[{symbol}] Resistance map error: {exc}")

    # ------------------------------------------------------------------
    # Detect float gate failure (force mode only — gate was skipped)
    # ------------------------------------------------------------------
    if force and entry.float_shares_m > FLOAT_MAX_M:
        fail_reasons.append(f"float: {entry.float_shares_m:.2f}M (> {FLOAT_MAX_M:.0f}M)")
    if entry.rvol_1day > 0 and entry.rvol_1day < RVOL_1DAY_MIN:
        fail_reasons.append(f"rvol_1day: {entry.rvol_1day:.1f}x (< {RVOL_1DAY_MIN:.0f}x)")

    # Update daily_high from actual intraday candles (Stage 1 quote may be stale/None)
    if not df_1min_seed.empty and "high" in df_1min_seed.columns:
        candle_hod = df_1min_seed["high"].max()
        if candle_hod > 0:
            entry.daily_high = max(entry.daily_high, candle_hod)

    pipeline_passed = len(fail_reasons) == 0

    # ------------------------------------------------------------------
    # Start dynamic confidence scorer (active market-data stream)
    # ------------------------------------------------------------------
    # Early stream served its purpose — unsubscribe before scorer starts its own
    early_stream.stop()

    scorer = None
    if stream_enabled() and pipeline_passed:
        daily_candles = DynamicScorer._df_to_candles(df_daily, "daily") if not df_daily.empty else []
        try:
            end_utc = datetime.now(tz=timezone.utc)
            start_utc = move_start_ts or latest_extended_session_start(end_utc)
            df_10sec_seed = load_candle_cache(symbol, "10s", start_utc, end_utc)
        except Exception as exc:
            logger.debug(f"[{symbol}] 10s cache seed load failed: {exc}")
        scorer = DynamicScorer(entry, rmap, avg_daily_volume=entry.avg_volume,
                               daily_candles=daily_candles,
                               seed_1min_session=df_1min_seed,
                               seed_10sec=df_10sec_seed,
                               move_start_ts=move_start_ts,
                               on_update=on_scorer_update)
        scorer.prime_from_seed_candles()
        scorer.start()

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    if not quiet:
        _print_stock_summary(entry, rmap, fail_reasons=fail_reasons or None)

    return PipelineResult(
        entry=entry, rmap=rmap, scorer=scorer, early_stream=early_stream,
        passed=pipeline_passed, fail_reasons=fail_reasons,
    )
