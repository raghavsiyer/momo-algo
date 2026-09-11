"""Visual replay session builder.

Replays a cached symbol window through the REAL production stack
(DynamicScorer + OrderFlowTracker + SimulationEngine) — reusing the
BacktestRunner's loading/seeding primitives — and captures a time-indexed
"replay bundle" the visuals site plays back like a live session: every print,
decimated L2 ladders, forming/closed display bars, full criteria/gate state
snapshots, and simulated fills.

Differences from exec.backtest:
  - No near-HOD window gating: every tick in the user-chosen window is scored,
    so the criteria panel is continuous.
  - No fade-from-HOD break: the user chose the window; replay all of it.
  - Output is an NDJSON chunk stream under logs/replay_cache/, never
    logs/sim/ (sim + setup loggers are re-pointed into the bundle directory
    for the duration of the build).

Bundle layout:
  logs/replay_cache/<SYM>_<YYYYMMDD>_<HHMMSS>-<HHMMSS>_<hash8>/
    meta.json            symbol/window/criteria defs/seed candles/levels
    status.json          {state: building|ready|error, chunks_ready, ...}
    events.json          (on finalize) fills + approved signals + all-pass segments
    chunks/NNNNNN.ndjson.gz   60s of replay time each; first record is an anchor
    sim/                 sim_trades/setup_signals CSVs captured during the build

Chunks are gzip'd with mtime=0 so identical replays produce byte-identical
bundles (tested).

CLI (for testing without the UI):
  python -m exec.replay_session CELZ 2026-08-06 --start 06:30 --end 07:30
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import logging
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import config
from config import (
    ATTENTION_DIST_FROM_HOD_ATR_MAX,
    ATTENTION_RVOL_1DAY_MIN,
    ATTENTION_VOLUME_MIN,
    ATTENTION_VWAP_SLOPE_MIN,
    CHANGE_PCT_MIN,
    ENTRY_CONFIDENCE_MIN,
    EXEC_DIST_TO_RESISTANCE_MULT,
    EXEC_HIDDEN_SELLER_MAX,
    EXEC_SPREAD_PCT_MAX,
    IBKR_ENABLED,
    MarketClock,
    ORDER_BOOK_MIN_DEPTH_SHARES,
    SIM_BACKTEST_LOG_DIR,
    SIM_EXIT_ACCEL_BUY_RATIO_MAX,
    SIM_EXIT_ACCEL_MAX,
    SIM_ENTRY_DWELL_SECS,
    SIM_ENTRY_DWELL_UPMOVE,
    SIM_EXIT_CONFIDENCE_MIN,
    SIM_EXIT_DWELL_SECS,
    SIM_EXIT_OFI_NORM_MAX,
    SIM_EXIT_RESISTANCE_ATR,
    SIM_FIRST_MACD_CYCLE_ONLY,
    SIM_MAX_SETUP_EXTENSION_PCT,
    SIM_OFI_NORM_MIN,
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_PROGRESS_MIN,
    SIM_PRESSURE_SPEED_MIN,
    SIM_PRESSURE_VOL_IMBALANCE_MIN,
    SIM_SETUP_ANTICIPATION_PCT,
    SIM_SETUP_SCORE_MIN,
)

logger = logging.getLogger(__name__)

BUNDLE_FORMAT_VERSION = 2  # v2: events.json carries logged live-session fills
CHUNK_SECONDS = 60
STATE_INTERVAL_MS = 500
BOOK_DISPLAY_INTERVAL_MS = 200
BOOK_DISPLAY_DEPTH = 10
DISPLAY_TF_SECONDS = {"10s": 10, "1m": 60, "2m": 120, "5m": 300}

# SimulationEngine + the setup logger are process-global; one build at a time.
_BUILD_LOCK = threading.Lock()


class ReplayBuildError(Exception):
    pass


# ---------------------------------------------------------------------------
# Bundle identity
# ---------------------------------------------------------------------------

def _hashed_constants() -> tuple:
    return (
    ("format", BUNDLE_FORMAT_VERSION),
    ("change_pct_min", CHANGE_PCT_MIN),
    ("attention_volume_min", ATTENTION_VOLUME_MIN),
    ("attention_rvol_min", ATTENTION_RVOL_1DAY_MIN),
    ("attention_hod_atr_max", ATTENTION_DIST_FROM_HOD_ATR_MAX),
    ("book_min_depth", ORDER_BOOK_MIN_DEPTH_SHARES),
    ("speed_min", SIM_PRESSURE_SPEED_MIN),
    ("activity_min", SIM_PRESSURE_ACTIVITY_MIN),
    ("buy_ratio_min", SIM_PRESSURE_BUY_RATIO_MIN),
    ("vol_imb_min", SIM_PRESSURE_VOL_IMBALANCE_MIN),
    ("accel_z", SIM_PRESSURE_ACCEL_Z),
    ("progress_min", SIM_PRESSURE_PROGRESS_MIN),
    ("ofi_norm_min", SIM_OFI_NORM_MIN),
    ("ofi_gate_source", config.OFI_GATE_SOURCE),
    ("ofi_ema_tau", config.OFI_EMA_TAU_SECS),
    ("setup_score_min", SIM_SETUP_SCORE_MIN),
    ("anticipation_pct", SIM_SETUP_ANTICIPATION_PCT),
    ("extension_pct", SIM_MAX_SETUP_EXTENSION_PCT),
    ("spread_max", EXEC_SPREAD_PCT_MAX),
    ("resistance_mult", EXEC_DIST_TO_RESISTANCE_MULT),
    ("hidden_seller_max", EXEC_HIDDEN_SELLER_MAX),
    ("entry_confidence_min", ENTRY_CONFIDENCE_MIN),
    ("ibkr_enabled", IBKR_ENABLED),
    # Exit thresholds surfaced as "sell" levels on the criteria meters.
    ("exit_accel_max", SIM_EXIT_ACCEL_MAX),
    ("exit_accel_buy_ratio_max", SIM_EXIT_ACCEL_BUY_RATIO_MAX),
    ("exit_resistance_atr", SIM_EXIT_RESISTANCE_ATR),
    ("exit_confidence_min", SIM_EXIT_CONFIDENCE_MIN),
    ("exit_ofi_norm_max", SIM_EXIT_OFI_NORM_MAX),
    # Dwell shifts fill timestamps, so it must bust the bundle cache.
    ("entry_dwell_secs", SIM_ENTRY_DWELL_SECS),
    ("entry_dwell_upmove", SIM_ENTRY_DWELL_UPMOVE),
    ("exit_dwell_secs", SIM_EXIT_DWELL_SECS),
    # Armed-band entry regime — every constant that moves a fill or a gate
    # state must bust the bundle cache. Read as config.X (live attribute, not
    # from-import) so a runtime override mutating config before the build —
    # e.g. the backtest --entry-regime flag — hashes the value that runs.
    ("entry_regime", config.SIM_ENTRY_REGIME),
    ("anticipation_max_pct", config.SIM_ANTICIPATION_MAX_PCT),
    ("level_cross_tol_pct", config.SIM_LEVEL_CROSS_TOLERANCE_PCT),
    ("arm_evidence_ttl", config.SIM_ARM_EVIDENCE_TTL_SECS),
    ("arm_strong_ttl", config.SIM_ARM_STRONG_TTL_SECS),
    ("arm_strong_requires_wall", config.SIM_ARM_STRONG_REQUIRES_WALL),
    ("trigger_ttl", config.SIM_TRIGGER_TTL_SECS),
    ("trigger_trail_window", config.SIM_TRIGGER_TRAIL_WINDOW_SECS),
    ("trigger_trail_min_span", config.SIM_TRIGGER_TRAIL_MIN_SPAN_SECS),
    ("wall_band_pct", config.SIM_WALL_BAND_PCT),
    ("wall_evap_min", config.SIM_WALL_EVAP_MIN),
    ("wall_replenish_veto", config.SIM_WALL_REPLENISH_VETO),
    ("exit_regime", config.SIM_EXIT_REGIME),
    ("struct_stop_buffer", config.SIM_STRUCT_STOP_BUFFER_PCT),
    ("prove_break_secs", config.SIM_PROVE_BREAK_SECS),
    ("prove_progress_secs", config.SIM_PROVE_PROGRESS_SECS),
    ("prove_progress_pct", config.SIM_PROVE_PROGRESS_PCT),
    ("exit_lapse_sustain", config.SIM_EXIT_LAPSE_SUSTAIN_SECS),
    ("trail_peak_arm_pct", config.SIM_TRAIL_PEAK_ARM_PCT),
    ("trail_peak_giveback_pct", config.SIM_TRAIL_PEAK_GIVEBACK_PCT),
    )


def config_hash() -> str:
    """Bundle-cache key over every threshold the bundle surfaces.

    Built at CALL time (not import time) so runtime regime overrides — the
    backtest --entry-regime flag mutates config before replaying — hash the
    values that will actually run.

    Honest limitation: code changes outside these constants do not bust the
    cache — rebuild with force=True after changing scorer/engine logic.
    """
    blob = json.dumps(_hashed_constants(), separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def bundle_name(symbol: str, day: date, start_utc: datetime, end_utc: datetime) -> str:
    return (
        f"{symbol.upper()}_{day.strftime('%Y%m%d')}_"
        f"{start_utc.strftime('%H%M%S')}-{end_utc.strftime('%H%M%S')}_{config_hash()}"
    )


# ---------------------------------------------------------------------------
# Display bar accumulator (display only — scoring uses the production stack)
# ---------------------------------------------------------------------------

class _DisplayBars:
    """Builds OHLCV display bars + VWAP/EMA continuation from replay ticks.

    Session VWAP is one cumulative accumulator shared by every timeframe,
    seeded from cached candles so it continues the session anchor rather than
    restarting at the window start. EMAs continue from the cached candles'
    last values per timeframe.
    """

    def __init__(self) -> None:
        self.cum_pv = 0.0
        self.cum_v = 0.0
        self._bars: dict[str, dict | None] = {tf: None for tf in DISPLAY_TF_SECONDS}
        self._ema: dict[str, dict[int, float | None]] = {tf: {9: None, 20: None} for tf in DISPLAY_TF_SECONDS}

    def seed(self, seed_candles: dict[str, list[dict]]) -> None:
        finest = next((tf for tf in ("10s", "1m", "2m", "5m") if seed_candles.get(tf)), None)
        if finest:
            for bar in seed_candles[finest]:
                volume = float(bar.get("v") or 0.0)
                price = bar.get("bar_vwap")
                if price is None:
                    price = (float(bar["h"]) + float(bar["l"]) + float(bar["c"])) / 3.0
                self.cum_pv += float(price) * volume
                self.cum_v += volume
        for tf, bars in seed_candles.items():
            if tf not in self._ema or not bars:
                continue
            last = bars[-1]
            self._ema[tf][9] = last.get("ema9")
            self._ema[tf][20] = last.get("ema20")

    def _close_bar(self, tf: str, bar: dict) -> dict:
        for span in (9, 20):
            prev = self._ema[tf][span]
            k = 2.0 / (span + 1.0)
            self._ema[tf][span] = bar["c"] if prev is None else prev + k * (bar["c"] - prev)
        vwap = self.cum_pv / self.cum_v if self.cum_v > 0 else None
        return {
            "t": bar["t"],
            "o": bar["o"], "h": bar["h"], "l": bar["l"], "c": bar["c"], "v": bar["v"],
            "vwap": round(vwap, 6) if vwap is not None else None,
            "ema9": round(self._ema[tf][9], 6) if self._ema[tf][9] is not None else None,
            "ema20": round(self._ema[tf][20], 6) if self._ema[tf][20] is not None else None,
        }

    def on_tick(self, ts_ms: int, price: float, size: int) -> list[tuple[str, dict]]:
        """Returns [(tf, closed_bar_payload)] for every bar this tick closes."""
        self.cum_pv += price * size
        self.cum_v += size
        closed: list[tuple[str, dict]] = []
        for tf, secs in DISPLAY_TF_SECONDS.items():
            bucket = (ts_ms // (secs * 1000)) * (secs * 1000)
            cur = self._bars[tf]
            if cur is not None and bucket > cur["t"]:
                closed.append((tf, self._close_bar(tf, cur)))
                cur = None
            if cur is None:
                self._bars[tf] = {"t": bucket, "o": price, "h": price, "l": price, "c": price, "v": size}
            else:
                cur["h"] = max(cur["h"], price)
                cur["l"] = min(cur["l"], price)
                cur["c"] = price
                cur["v"] += size
        return closed

    def forming(self) -> dict[str, dict]:
        return {tf: dict(bar) for tf, bar in self._bars.items() if bar is not None}


# ---------------------------------------------------------------------------
# Chunk writer
# ---------------------------------------------------------------------------

class _ChunkWriter:
    """Time-ordered NDJSON records → 60s gzip chunks with anchor records.

    gzip mtime is pinned to 0 so re-running the same replay produces
    byte-identical chunk files (the determinism test relies on this).
    """

    def __init__(self, bundle_dir: Path, start_ms: int, anchor_fn: Callable[[int], dict], on_roll: Callable[[int], None]) -> None:
        self._dir = bundle_dir / "chunks"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._start_ms = start_ms
        self._anchor_fn = anchor_fn
        self._on_roll = on_roll
        self._index: int | None = None
        self._lines: list[str] = []

    def _chunk_index(self, t_ms: int) -> int:
        return max(0, (t_ms - self._start_ms) // (CHUNK_SECONDS * 1000))

    def _flush_current(self) -> None:
        if self._index is None:
            return
        path = self._dir / f"{self._index:06d}.ndjson.gz"
        raw = ("\n".join(self._lines) + "\n").encode("utf-8")
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as gz:
                gz.write(raw)
        tmp.replace(path)  # atomic: the server never sees a partial chunk
        self._on_roll(self._index + 1)

    def _open_chunk(self, index: int) -> None:
        self._index = index
        chunk_start_ms = self._start_ms + index * CHUNK_SECONDS * 1000
        anchor = self._anchor_fn(chunk_start_ms)
        self._lines = [json.dumps(anchor, separators=(",", ":"), allow_nan=False)]

    def write(self, record: dict) -> None:
        index = self._chunk_index(int(record["t"]))
        if self._index is None:
            # Always start at chunk 0: if the first record lands minutes into
            # the window (tape starts late), the leading chunks must still
            # exist as anchor-only files or the client's sequential fetch 404s.
            self._open_chunk(0)
        while index > self._index:
            self._flush_current()
            self._open_chunk(self._index + 1)
        self._lines.append(json.dumps(record, separators=(",", ":"), allow_nan=False))

    def finalize(self, end_ms: int) -> int:
        """Flush through the chunk containing end_ms; returns chunk count."""
        if self._index is None:
            self._open_chunk(0)
        final_index = self._chunk_index(max(end_ms - 1, self._start_ms))
        while self._index < final_index:
            self._flush_current()
            self._open_chunk(self._index + 1)
        self._flush_current()
        return self._index + 1


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _ms(ts: datetime) -> int:
    return int(ts.timestamp() * 1000)


def _round(value, digits=4):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(number, digits)


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _ladder(levels: list, depth: int) -> list[list[float]]:
    out = []
    for level in levels[:depth]:
        price = getattr(level, "price", None)
        size = getattr(level, "size", None)
        if price is None or size is None:
            continue
        out.append([round(float(price), 6), int(size)])
    return out


# name: (entry_threshold, direction, unit, sell_threshold)
# sell_threshold is the evaluate_exit level that fires a SELL on this variable
# (None when no exit rule reads it). The UI paints the meter red only at/past
# the sell level, yellow between sell and entry, green when the entry
# criterion passes. Note: tape_acceleration displays the z estimator while the
# exit reads the raw EMA slope — the sell line is the raw constant, a
# sign-faithful approximation on the z scale.
CRITERIA_DISPLAY = {
    "change_pct_min": (CHANGE_PCT_MIN, ">=", "%", None),
    "volume_min": (float(ATTENTION_VOLUME_MIN), ">=", "", None),
    "rvol_1day_min": (ATTENTION_RVOL_1DAY_MIN, ">=", "x", None),
    "dist_from_hod_atr_max": (ATTENTION_DIST_FROM_HOD_ATR_MAX, "<=", "ATR", None),
    "order_book_density_ok": (float(ORDER_BOOK_MIN_DEPTH_SHARES), ">=", "sh", None),
    "vwap_slope_positive": (ATTENTION_VWAP_SLOPE_MIN, ">=", "", None),
    "close_above_vwap": (0.0, ">", "", None),
    "macd_positive": (0.0, ">", "", None),
    "tape_speed": (SIM_PRESSURE_SPEED_MIN, ">=", "/s", None),
    "tape_imbalance": (SIM_PRESSURE_BUY_RATIO_MIN, ">=", "", SIM_EXIT_ACCEL_BUY_RATIO_MAX),
    "tape_vol_flow": (SIM_PRESSURE_VOL_IMBALANCE_MIN, ">", "", None),
    "tape_acceleration": (SIM_PRESSURE_ACCEL_Z, ">=", "z", SIM_EXIT_ACCEL_MAX),
    "spread_ok": (EXEC_SPREAD_PCT_MAX, "<=", "", None),
    "dist_to_resistance_ok": (EXEC_DIST_TO_RESISTANCE_MULT, ">=", "ATR", SIM_EXIT_RESISTANCE_ATR),
    "no_hidden_seller": (EXEC_HIDDEN_SELLER_MAX, "<=", "", None),
}


def _criteria_defs() -> dict:
    from exec.posterior import entry_threshold

    def group_defs(group_config: list[dict]) -> list[dict]:
        defs = []
        for group in group_config:
            for item in group.get("criteria", []):
                threshold, direction, unit, sell = CRITERIA_DISPLAY.get(item["name"], (None, ">=", "", None))
                defs.append({
                    "name": item["name"],
                    "description": item.get("description", ""),
                    "required": bool(item.get("required", True)),
                    "threshold": threshold,
                    "direction": direction,
                    "unit": unit,
                    "sell": sell,
                })
        return defs

    return {
        "attention": group_defs(config.CRITERIA_ATTENTION),
        "execution": group_defs(config.CRITERIA_EXECUTION),
        "chain": [
            {"name": "entry_allowed", "label": "Entry allowed", "direction": "bool"},
            {"name": "setup_score", "label": "Setup score", "threshold": SIM_SETUP_SCORE_MIN, "direction": ">="},
            {"name": "price_band", "label": "Price in band", "direction": "band"},
            # Present only while enforced — entry_gate_states omits the gate
            # when SIM_FIRST_MACD_CYCLE_ONLY is off, and an always-on def
            # would render a permanently dead node.
            *([{"name": "macd_cycle", "label": "First MACD cycle", "direction": "bool"}]
              if SIM_FIRST_MACD_CYCLE_ONLY else []),
            {"name": "tape_speed", "label": "Tape speed", "threshold": SIM_PRESSURE_SPEED_MIN, "direction": ">=", "unit": "/s"},
            {"name": "tape_activity", "label": "Tape activity", "threshold": float(SIM_PRESSURE_ACTIVITY_MIN), "direction": ">="},
            {"name": "tape_imbalance", "label": "Buy ratio", "threshold": SIM_PRESSURE_BUY_RATIO_MIN, "direction": ">="},
            {"name": "tape_vol_flow", "label": "Volume flow", "threshold": SIM_PRESSURE_VOL_IMBALANCE_MIN, "direction": ">"},
            {"name": "tape_acceleration", "label": "Acceleration z", "threshold": SIM_PRESSURE_ACCEL_Z, "direction": ">=", "unit": "z"},
            # Displayed in PERCENT (value scaled x100 in entry_gate_states —
            # the config constant stays a fraction): 0.005 would render as a
            # misleading "0.01" under the 2dp formatter.
            {"name": "tape_progress", "label": "Price progress", "threshold": round(SIM_PRESSURE_PROGRESS_MIN * 100, 3), "direction": ">", "unit": "%"},
            {"name": "ofi_norm", "label": "OFI normalized", "threshold": SIM_OFI_NORM_MIN, "direction": ">=", "sell": SIM_EXIT_OFI_NORM_MAX},
            # Armed-band chain nodes (emitted by entry_gate_states only in
            # the armed_band regime's below/cross zones — harmless extra
            # defs otherwise; the pane renders only nodes present in state
            # records).
            *([
                {"name": "arm_evidence", "label": "Armed evidence", "threshold": config.SIM_ARM_EVIDENCE_TTL_SECS, "direction": "<=", "unit": "s"},
                {"name": "ratio_now", "label": "Buy ratio (now)", "threshold": SIM_PRESSURE_BUY_RATIO_MIN, "direction": ">="},
                {"name": "wall", "label": "Ask-wall evaporation", "threshold": config.SIM_WALL_EVAP_MIN, "direction": ">="},
                {"name": "trigger", "label": "Entry trigger", "threshold": config.SIM_TRIGGER_TTL_SECS, "direction": "<=", "unit": "s"},
                {"name": "wall_veto", "label": "No fresh wall", "threshold": config.SIM_WALL_REPLENISH_VETO, "direction": ">"},
            ] if config.SIM_ENTRY_REGIME == "armed_band" else []),
            # Posterior chain nodes (Upgrade 1): the AND-stack nodes above
            # simply never appear in state records under this regime.
            *([
                {"name": "scored_zone", "label": "Scored zone", "direction": "<="},
                {"name": "trigger", "label": "Entry trigger", "threshold": config.SIM_TRIGGER_TTL_SECS, "direction": "<=", "unit": "s"},
                {"name": "posterior", "label": "Entry posterior",
                 "threshold": entry_threshold(), "direction": ">="},
            ] if config.SIM_ENTRY_REGIME == "posterior" else []),
            {"name": "headroom", "label": "Resistance headroom", "threshold": EXEC_DIST_TO_RESISTANCE_MULT, "direction": ">=", "unit": "ATR", "sell": SIM_EXIT_RESISTANCE_ATR},
        ],
        "confidence": {
            "eval_min": ENTRY_CONFIDENCE_MIN,
            "live_min": ENTRY_CONFIDENCE_MIN,
            "exit_min": SIM_EXIT_CONFIDENCE_MIN,
        },
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

@dataclass
class ReplayBuildResult:
    bundle_dir: Path
    chunks: int
    ticks: int
    fills: int
    criteria_available: bool
    from_cache: bool = False


class ReplaySessionBuilder:
    def __init__(
        self,
        symbol: str,
        day: date,
        window_start_utc: datetime,
        window_end_utc: datetime,
        log_dir: Path | None = None,
        out_dir: Path | None = None,
        on_progress: Optional[Callable[[float, int], None]] = None,
        force: bool = False,
    ) -> None:
        self.symbol = symbol.upper()
        self.day = day
        self.start_utc = window_start_utc.astimezone(timezone.utc)
        self.end_utc = window_end_utc.astimezone(timezone.utc)
        if self.end_utc <= self.start_utc:
            raise ReplayBuildError("window end must be after window start")
        default_log_dir = Path(__file__).resolve().parents[1] / "logs"
        self.log_dir = Path(log_dir) if log_dir else default_log_dir
        self.out_dir = Path(out_dir) if out_dir else self.log_dir / "replay_cache"
        self.on_progress = on_progress
        self.force = force
        self.bundle_dir = self.out_dir / bundle_name(self.symbol, day, self.start_utc, self.end_utc)

    # -- public ------------------------------------------------------------

    def build(self) -> ReplayBuildResult:
        status_path = self.bundle_dir / "status.json"
        if not self.force and status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                status = {}
            if status.get("state") == "ready":
                return ReplayBuildResult(
                    bundle_dir=self.bundle_dir,
                    chunks=int(status.get("chunks_ready", 0)),
                    ticks=int(status.get("ticks", 0)),
                    fills=int(status.get("fills", 0)),
                    criteria_available=bool(status.get("criteria_available", False)),
                    from_cache=True,
                )
        with _BUILD_LOCK:
            return self._build_locked()

    # -- internals ---------------------------------------------------------

    def _status(self, **payload) -> None:
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.bundle_dir / "status.json", payload)

    def _build_locked(self) -> ReplayBuildResult:
        # Heavy imports deferred so importing this module stays cheap.
        from chart_cache import load_candle_cache
        from exec.backtest import (
            BacktestRunner,
            _approved_setup_signals,
            _reset_replay_criteria,
            _reset_replay_setup_state,
            _retime_replay_static_evaluation,
            _seeded_replay_volume,
        )
        from cache import _refresh_scoring_thresholds, _reset_live_state
        from exec.engine import SimulationEngine
        from exec.setup_logger import configure_setup_signal_logger
        from exec.signals import entry_gate_states
        from strategy.confidence_scorer import DynamicScorer
        from technical_analysis.patterns import latest_extended_session_start

        started = _time.monotonic()
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self._status(state="building", chunks_ready=0, progress_pct=0.0,
                     criteria_available=False, symbol=self.symbol)

        # A replay build must never touch live market infrastructure. The
        # scorer's slot-priority push calls get_shared_stream() when a setup
        # arms, which would lazily construct a REAL IBKRSharedStream and dial
        # the Gateway mid-build — stub it for the duration of the build.
        import strategy.confidence_scorer as _scorer_module

        class _NullStream:
            pass

        original_get_stream = _scorer_module.get_shared_stream
        _scorer_module.get_shared_stream = lambda: _NullStream()

        runner = BacktestRunner(no_tradier=True, log_dir=self.log_dir)

        # Route sim + setup output INTO the bundle before any scorer exists —
        # the scorer captures the setup logger at construction time.
        SimulationEngine.reset_session()
        sim_dir = str(self.bundle_dir / "sim")
        SimulationEngine.configure_logger(sim_dir)
        configure_setup_signal_logger(sim_dir)

        try:
            entry = rmap = scorer = None
            cache_item = runner._load_cache_entries(self.day).get(self.symbol)
            criteria_available = bool(cache_item) and len(cache_item) >= 2
            seed_candles = self._load_seed_candles(load_candle_cache, latest_extended_session_start)

            if criteria_available:
                entry = copy.deepcopy(cache_item[0])
                rmap = copy.deepcopy(cache_item[1])
                runner._migrate_evaluation(entry)
                ev = getattr(entry, "evaluation", None)
                if ev is not None:
                    _refresh_scoring_thresholds(ev)
                    _reset_live_state(ev)
                _reset_replay_criteria(entry)
                _reset_replay_setup_state(entry)
                if hasattr(entry, "daily_high"):
                    entry.daily_high = 0.0

                df_1m_seed, df_10s_seed, df_ts_seed = runner._load_cached_seed_market_data(self.symbol, self.start_utc)
                entry.volume = _seeded_replay_volume(df_1m_seed, df_ts_seed)
                avg_volume = float(getattr(entry, "avg_volume", 0.0) or 0.0)
                entry.rvol_1day = entry.volume / avg_volume if avg_volume > 0 else 0.0
                _retime_replay_static_evaluation(entry, self.start_utc)
                scorer_kwargs: dict[str, Any] = {
                    "avg_daily_volume": entry.avg_volume,
                    "daily_candles": [],
                    "cache_live_data": False,
                }
                if not df_1m_seed.empty:
                    scorer_kwargs["seed_1min_session"] = df_1m_seed
                if not df_10s_seed.empty:
                    scorer_kwargs["seed_10sec"] = df_10s_seed
                if not df_ts_seed.empty:
                    scorer_kwargs["seed_timesales"] = df_ts_seed
                scorer = DynamicScorer(entry, rmap, **scorer_kwargs)
                if hasattr(scorer, "_tf_buf"):
                    scorer._tf_buf._reconcile = True
                runner._seed_scorer_from_cached_market_data(scorer, df_1m_seed, df_10s_seed, df_ts_seed)

            book_snapshots = runner._load_book_snapshots(self.symbol, self.start_utc, self.end_utc)

            display = _DisplayBars()
            display.seed(seed_candles)

            self._write_meta(seed_candles, rmap, criteria_available, bool(book_snapshots))
            self._status(state="building", chunks_ready=0, progress_pct=0.0,
                         criteria_available=criteria_available, symbol=self.symbol)

            result = self._run_loop(
                runner=runner,
                scorer=scorer,
                entry=entry,
                rmap=rmap,
                book_snapshots=book_snapshots,
                display=display,
                entry_gate_states=entry_gate_states,
                retime=_retime_replay_static_evaluation,
                sim_engine=SimulationEngine,
                approved_setup_signals=_approved_setup_signals,
                criteria_available=criteria_available,
            )
            self._status(state="ready", chunks_ready=result.chunks, progress_pct=100.0,
                         criteria_available=criteria_available, symbol=self.symbol,
                         ticks=result.ticks, fills=result.fills,
                         build_secs=round(_time.monotonic() - started, 1))
            return result
        except Exception as exc:
            logger.exception("replay build failed for %s %s", self.symbol, self.day)
            self._status(state="error", error=str(exc), symbol=self.symbol, chunks_ready=0)
            raise
        finally:
            _scorer_module.get_shared_stream = original_get_stream
            SimulationEngine.reset_session()
            SimulationEngine.configure_logger(SIM_BACKTEST_LOG_DIR)
            configure_setup_signal_logger(SIM_BACKTEST_LOG_DIR)

    def _load_logged_live_fills(self) -> list[dict]:
        """Fills the LIVE session's sim engine actually logged inside this
        window — shown alongside the replay's fresh fills so "what it did
        then" and "what it would do now" can be compared directly. Reads only
        the canonical sim_logger schema (ts/symbol/side/fill_price/shares)."""
        ymds = sorted({self.start_utc.strftime("%Y%m%d"), self.end_utc.strftime("%Y%m%d")})
        records: list[dict] = []
        seen: set[tuple] = set()
        for ymd in ymds:
            for base in (self.log_dir / "sim", self.log_dir):
                path = base / f"sim_trades_{ymd}.csv"
                if not path.exists():
                    continue
                try:
                    with path.open(newline="", encoding="utf-8") as handle:
                        for row in csv.DictReader(handle):
                            if (row.get("symbol") or "").strip().upper() != self.symbol:
                                continue
                            raw_ts = (row.get("ts") or "").strip()
                            try:
                                ts = datetime.fromisoformat(raw_ts)
                            except ValueError:
                                continue
                            if ts.tzinfo is None:
                                ts = ts.replace(tzinfo=timezone.utc)
                            ts = ts.astimezone(timezone.utc)
                            if ts < self.start_utc or ts > self.end_utc:
                                continue
                            side = (row.get("side") or "").strip().upper()
                            if side not in {"BUY", "SELL"}:
                                continue
                            try:
                                price = float(row.get("fill_price") or 0.0)
                            except (TypeError, ValueError):
                                continue
                            if price <= 0:
                                continue
                            key = (raw_ts, side, row.get("fill_price"), row.get("shares"))
                            if key in seen:
                                continue
                            seen.add(key)
                            reason = row.get("entry_reason") if side == "BUY" else row.get("exit_reason")
                            try:
                                shares = int(float(row.get("shares") or 0))
                            except (TypeError, ValueError):
                                shares = 0
                            records.append({
                                "t": _ms(ts),
                                "side": side,
                                "price": _round(price, 6),
                                "shares": shares,
                                "reason": (reason or "").strip()[:160],
                            })
                except OSError:
                    continue
        records.sort(key=lambda item: item["t"])
        return records

    def _load_seed_candles(self, load_candle_cache, latest_extended_session_start) -> dict[str, list[dict]]:
        session_start = latest_extended_session_start(self.start_utc)
        seed: dict[str, list[dict]] = {}
        for tf, secs in DISPLAY_TF_SECONDS.items():
            try:
                df = load_candle_cache(self.symbol, tf, session_start, self.start_utc, self.log_dir)
            except Exception:
                seed[tf] = []
                continue
            bars: list[dict] = []
            if df is not None and not df.empty:
                for row in df.itertuples(index=False):
                    ts = row.ts.to_pydatetime() if hasattr(row.ts, "to_pydatetime") else row.ts
                    if ts + timedelta(seconds=secs) > self.start_utc:
                        continue  # causality: only fully closed bars
                    bars.append({
                        "t": int(ts.timestamp()),
                        "o": _round(row.open, 6), "h": _round(row.high, 6),
                        "l": _round(row.low, 6), "c": _round(row.close, 6),
                        "v": int(float(row.volume or 0)),
                        "vwap": _round(getattr(row, "vwap", None), 6),
                        "bar_vwap": _round(getattr(row, "bar_vwap", None), 6),
                        "ema9": _round(getattr(row, "ema_9", None), 6),
                        "ema20": _round(getattr(row, "ema_20", None), 6),
                        "ma50": _round(getattr(row, "ma_50", None), 6),
                    })
            seed[tf] = bars
        return seed

    def _write_meta(self, seed_candles: dict, rmap, criteria_available: bool, has_book: bool) -> None:
        local_tz = MarketClock.PREMARKET_WINDOW_TZ or timezone.utc
        offset = self.start_utc.astimezone(local_tz).utcoffset()
        tz_offset_minutes = int(offset.total_seconds() // 60) if offset else 0
        try:
            tz_label = local_tz.tzname(self.start_utc.astimezone(local_tz)) or "local"
        except Exception:
            tz_label = "local"
        levels = []
        if rmap is not None:
            try:
                for level in rmap.all_levels(active_only=True):
                    levels.append({
                        "price": _round(getattr(level, "price", None), 6),
                        "strength": _round(getattr(level, "strength", None), 3),
                        "kind": str(getattr(level, "source", "") or getattr(level, "kind", "")),
                    })
            except Exception:
                levels = []
        # Strip the seed-only bar_vwap helper from the serialized candles.
        clean_seed = {
            tf: [{key: value for key, value in bar.items() if key != "bar_vwap"} for bar in bars]
            for tf, bars in seed_candles.items()
        }
        _write_json(self.bundle_dir / "meta.json", {
            "format": BUNDLE_FORMAT_VERSION,
            "symbol": self.symbol,
            "day": self.day.isoformat(),
            "window": {"start_ms": _ms(self.start_utc), "end_ms": _ms(self.end_utc)},
            "tz_label": tz_label,
            "tz_offset_minutes": tz_offset_minutes,
            "criteria_available": criteria_available,
            "has_book": has_book,
            "chunk_seconds": CHUNK_SECONDS,
            "seed_candles": clean_seed,
            "criteria_defs": _criteria_defs(),
            "resistance_levels": levels,
        })

    # -- the replay loop ---------------------------------------------------

    def _run_loop(
        self,
        runner,
        scorer,
        entry,
        rmap,
        book_snapshots,
        display: _DisplayBars,
        entry_gate_states,
        retime,
        sim_engine,
        approved_setup_signals,
        criteria_available: bool,
    ) -> ReplayBuildResult:
        symbol = self.symbol
        start_ms = _ms(self.start_utc)
        end_ms = _ms(self.end_utc)
        span_ms = max(1, end_ms - start_ms)

        current_book: dict | None = None
        last_book_emit_ms = -10**18
        last_state_ms = -10**18
        last_state_sig: tuple | None = None
        last_state_record: dict | None = None
        ticks_processed = 0
        fills_emitted: list[dict] = []
        allpass_segments: list[list[int]] = []
        allpass_open: int | None = None
        last_price = 0.0
        last_bid = 0.0
        last_ask = 0.0
        session_volume = 0

        sim_logger = sim_engine.get_logger()
        fill_idx = len(sim_logger.fills) if sim_logger is not None else 0

        def anchor_fn(chunk_start_ms: int) -> dict:
            return {
                "t": chunk_start_ms,
                "k": "anchor",
                "p": _round(last_price, 6),
                "bid": _round(last_bid, 6),
                "ask": _round(last_ask, 6),
                "vol": session_volume,
                "book": current_book,
                "bars": display.forming(),
                "state": last_state_record,
            }

        def on_roll(chunks_ready: int) -> None:
            progress = min(99.0, 100.0 * (chunks_ready * CHUNK_SECONDS * 1000) / span_ms)
            self._status(state="building", chunks_ready=chunks_ready, progress_pct=round(progress, 1),
                         criteria_available=criteria_available, symbol=symbol)
            if self.on_progress:
                self.on_progress(progress, ticks_processed)

        writer = _ChunkWriter(self.bundle_dir, start_ms, anchor_fn, on_roll)

        def sample_state(t_ms: int, force: bool, tick_price: float) -> None:
            nonlocal last_state_ms, last_state_sig, last_state_record, allpass_open
            if scorer is None or entry is None:
                return
            att_group = entry.get_criteria_group("ATTENTION")
            exe_group = entry.get_criteria_group("EXECUTION")
            chain = entry_gate_states(entry, scorer._ofi_tracker, tick_price, scorer._last_atr, rmap)
            att = [[c.name, 1 if c.passing else 0, _round(c.value)] for c in (att_group.criteria if att_group else [])]
            exe = [[c.name, 1 if c.passing else 0, _round(c.value)] for c in (exe_group.criteria if exe_group else [])]
            chain_rows = [[g.name, 1 if g.passing else 0, _round(g.value), 1 if g.active else 0, g.detail] for g in chain]
            sig = (
                tuple(row[1] for row in att),
                tuple(row[1] for row in exe),
                tuple(row[1] for row in chain_rows),
            )
            chain_all_pass = all(g.passing for g in chain)
            if chain_all_pass and allpass_open is None:
                allpass_open = t_ms
            elif not chain_all_pass and allpass_open is not None:
                allpass_segments.append([allpass_open, t_ms])
                allpass_open = None
            if not force and sig == last_state_sig and (t_ms - last_state_ms) < STATE_INTERVAL_MS:
                return
            tracker = scorer._ofi_tracker
            record = {
                "t": t_ms,
                "k": "state",
                "allowed": 1 if entry.entry_allowed else 0,
                "eval": _round(entry.eval_confidence, 1),
                "live": _round(entry.live_confidence, 1),
                "setup": _round(entry.setup_pattern_score, 1),
                "setupTf": getattr(entry, "setup_pattern_timeframe", "") or "",
                "att": att,
                "exe": exe,
                "chain": chain_rows,
                "block": entry.entry_block_reasons[:6],
                "depth": 1 if tracker.depth_active else 0,
                "chainPass": 1 if chain_all_pass else 0,
            }
            writer.write(record)
            last_state_ms = t_ms
            last_state_sig = sig
            last_state_record = record

        book_idx = 0
        tick_rows: list[tuple[datetime, float, int]] = []
        ta_last_minute: Optional[datetime] = None
        static_eval_retimed = False

        tick_iter = runner._iter_ticks(symbol, self.start_utc, self.end_utc)
        for _call_count, tick in tick_iter:
            ticks_processed += 1
            t_ms = _ms(tick.ts)

            # Books ≤ tick.ts feed the scorer BEFORE the print (live arrival
            # order — same rule as exec/backtest.py's replay loop).
            drained_book = None
            while book_idx < len(book_snapshots) and book_snapshots[book_idx].ts <= tick.ts:
                snapshot = book_snapshots[book_idx]
                if scorer is not None:
                    scorer._on_book(snapshot)
                drained_book = snapshot
                book_idx += 1
            if drained_book is not None:
                current_book = {
                    "b": _ladder(drained_book.bids, BOOK_DISPLAY_DEPTH),
                    "a": _ladder(drained_book.asks, BOOK_DISPLAY_DEPTH),
                }
                book_ms = _ms(drained_book.ts)
                if book_ms - last_book_emit_ms >= BOOK_DISPLAY_INTERVAL_MS:
                    writer.write({"t": book_ms, "k": "book", **current_book})
                    last_book_emit_ms = book_ms

            if entry is not None:
                entry.price = tick.last
                if not static_eval_retimed:
                    retime(entry, self.start_utc)
                    static_eval_retimed = True

            tick_rows.append((tick.ts, tick.last, tick.size))
            if scorer is not None:
                minute_key = tick.ts.replace(second=0, microsecond=0)
                if ta_last_minute != minute_key:
                    runner._refresh_ta_from_ticks(scorer, tick_rows)
                    ta_last_minute = minute_key
                cached_snapshot = getattr(tick, "_cached_tape_snapshot", None)
                if cached_snapshot is not None:
                    runner._apply_cached_tape_snapshot(scorer, cached_snapshot)
                scorer._on_quote(runner._quote_from_tick(tick))
                scorer._on_timesale(tick)

            last_price = tick.last
            last_bid = float(tick.bid or 0.0)
            last_ask = float(tick.ask or 0.0)
            session_volume += int(tick.size or 0)

            side = "N"
            if last_ask > 0 and tick.last >= last_ask:
                side = "B"
            elif last_bid > 0 and tick.last <= last_bid:
                side = "S"
            writer.write({
                "t": t_ms, "k": "tick",
                "p": _round(tick.last, 6), "s": int(tick.size or 0),
                "side": side,
                "bid": _round(last_bid, 6), "ask": _round(last_ask, 6),
            })

            for tf, bar in display.on_tick(t_ms, float(tick.last), int(tick.size or 0)):
                writer.write({"t": t_ms, "k": "bar", "tf": tf, "bar": bar})

            new_fill = False
            if sim_logger is not None:
                fills = sim_logger.fills
                while fill_idx < len(fills):
                    fill = fills[fill_idx]
                    fill_idx += 1
                    if fill.symbol != symbol:
                        continue
                    new_fill = True
                    record = {
                        "t": _ms(fill.ts),
                        "k": "fill",
                        "side": fill.side,
                        "price": _round(fill.fill_price, 6),
                        "shares": int(fill.shares),
                        "reason": fill.entry_reason if fill.side == "BUY" else fill.exit_reason,
                        "setup": fill.setup_name,
                        "setupTf": fill.setup_timeframe,
                        "eval": _round(fill.eval_confidence, 1),
                        "live": _round(fill.live_confidence, 1),
                        "pnl": _round(fill.pnl_dollars, 2),
                        "pnlPct": _round(fill.pnl_pct, 3),
                        "holdSecs": _round(fill.hold_secs, 1),
                    }
                    writer.write(record)
                    fills_emitted.append(record)

            sample_state(t_ms, force=new_fill, tick_price=float(tick.last))

        if scorer is not None and tick_rows:
            runner._refresh_ta_from_ticks(scorer, tick_rows)
        if allpass_open is not None:
            allpass_segments.append([allpass_open, end_ms])

        chunks = writer.finalize(end_ms)

        signals = []
        if scorer is not None:
            for signal in approved_setup_signals(scorer, symbol, self.start_utc, self.end_utc):
                ts = getattr(signal, "approval_ts", None) or getattr(signal, "ts", None)
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                signals.append({
                    "t": _ms(ts.astimezone(timezone.utc)),
                    "name": getattr(signal, "setup_name", ""),
                    "tf": getattr(signal, "setup_timeframe", ""),
                    "level": _round(getattr(signal, "setup_level", None), 6),
                    "score": _round(getattr(signal, "setup_score", None), 1),
                })
        _write_json(self.bundle_dir / "events.json", {
            "fills": fills_emitted,
            "live_fills": self._load_logged_live_fills(),
            "signals": signals,
            "allpass": allpass_segments,
        })

        return ReplayBuildResult(
            bundle_dir=self.bundle_dir,
            chunks=chunks,
            ticks=ticks_processed,
            fills=len(fills_emitted),
            criteria_available=criteria_available,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_hhmm(value: str) -> dtime:
    hour_s, minute_s = value.split(":", 1)
    return dtime(int(hour_s), int(minute_s))


def window_utc_for(day: date, start_local: dtime | None, end_local: dtime | None) -> tuple[datetime, datetime]:
    """Resolve a local-time window (display tz) to UTC; defaults to premarket."""
    default_start, default_end = MarketClock.premarket_window_utc(day)
    tz = MarketClock.PREMARKET_WINDOW_TZ or timezone.utc
    start = datetime.combine(day, start_local, tzinfo=tz).astimezone(timezone.utc) if start_local else default_start
    end = datetime.combine(day, end_local, tzinfo=tz).astimezone(timezone.utc) if end_local else default_end
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual replay bundle from cached market data")
    parser.add_argument("symbol")
    parser.add_argument("day", help="YYYY-MM-DD (local trading day)")
    parser.add_argument("--start", type=_parse_hhmm, default=None, help="local HH:MM (default: premarket open)")
    parser.add_argument("--end", type=_parse_hhmm, default=None, help="local HH:MM (default: premarket close)")
    parser.add_argument("--force", action="store_true", help="rebuild even if a cached bundle exists")
    args = parser.parse_args()

    day = date.fromisoformat(args.day)
    start_utc, end_utc = window_utc_for(day, args.start, args.end)
    started = _time.monotonic()
    builder = ReplaySessionBuilder(args.symbol, day, start_utc, end_utc, force=args.force)

    def progress(pct: float, ticks: int) -> None:
        print(f"\r  {pct:5.1f}%  ticks={ticks}", end="", flush=True)

    builder.on_progress = progress
    result = builder.build()
    elapsed = _time.monotonic() - started
    print()
    print(f"bundle: {result.bundle_dir}")
    print(
        f"chunks={result.chunks} ticks={result.ticks} fills={result.fills} "
        f"criteria={'yes' if result.criteria_available else 'no'} "
        f"{'(cached)' if result.from_cache else f'built in {elapsed:.1f}s'}"
    )


if __name__ == "__main__":
    main()
