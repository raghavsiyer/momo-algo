"""
Simulation Engine — monitors live stream data and simulates trade execution.

This engine hooks into the DynamicScorer's existing stream (T&S + quotes)
and evaluates entry/exit signals on every tick. When conditions are met,
it simulates fills at the live bid/ask and logs everything.

Key principles:
  - No real orders are placed — all fills are simulated
  - Fill prices use bid (sell) / ask (buy) from the live NBBO
  - Core decision: buying pressure vs selling pressure + OFI
  - Everything else (Stage 4, setup pattern, confidence) is filtering
  - Thread-safe: called from the DynamicScorer's stream callbacks
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from exec.order_flow import OrderFlowTracker
    from strategy.evaluator import WatchlistEntry
    from technical_analysis.resistance import ResistanceMap

from config import (
    IBKR_ENABLED,
    SIM_ACCOUNT_SIZE,
    SIM_COOLDOWN_SECS,
    SIM_ENABLED,
    SIM_ENTRY_DWELL_SECS,
    SIM_ENTRY_DWELL_UPMOVE,
    SIM_EXIT_DWELL_SECS,
    SIM_EXIT_RESISTANCE_ATR,
    SIM_FILL_LATENCY_MS,
    SIM_FILL_OFFSET_CENTS,
    SIM_FILL_SLIP_BPS,
    SIM_LATENCY_FILL,
    SIM_MAX_CONCURRENT,
    SIM_MAX_POSITION_PCT,
    SIM_MAX_TRADES_PER_SYMBOL,
    SIM_USE_ASK_FOR_BUY,
    SIM_USE_BID_FOR_SELL,
    _g, _r, _b, _dim,
)
from exec.signals import (
    EntrySignal,
    ExitSignal,
    entry_band_floor,
    evaluate_entry,
    evaluate_exit,
)
from exec.sim_logger import SimFill, SimTradeLogger
from exec.setup_logger import SetupSignalLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Latency-based fill: pending order state
# ---------------------------------------------------------------------------

_FILL_LATENCY = timedelta(milliseconds=SIM_FILL_LATENCY_MS) if SIM_LATENCY_FILL else timedelta(0)

# Signal dwell: a triggered signal must hold continuously for this long before
# it fills. Distinct from _FILL_LATENCY, which models the post-decision round
# trip; dwell delays the decision itself. See config.SIM_ENTRY_DWELL_SECS.
_ENTRY_DWELL = timedelta(seconds=max(0.0, SIM_ENTRY_DWELL_SECS))
_EXIT_DWELL = timedelta(seconds=max(0.0, SIM_EXIT_DWELL_SECS))


@dataclass(slots=True)
class _PendingBuy:
    """Deferred buy fill — waiting for latency window to elapse."""
    signal_ts: datetime
    signal: EntrySignal
    signal_key: Optional[datetime]


@dataclass(slots=True)
class _PendingSell:
    """Deferred sell fill — waiting for latency window to elapse."""
    signal_ts: datetime
    reason: str


# ---------------------------------------------------------------------------
# Position tracking
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SimPosition:
    """A simulated open position."""
    symbol: str
    entry_price: float
    shares: int
    entry_time: datetime
    entry_signal: EntrySignal
    bid_at_entry: float
    ask_at_entry: float
    setup_name: str = ""
    setup_timeframe: str = ""
    setup_score: float = 0.0
    setup_pullback_ts: Optional[datetime] = None
    setup_breakout_ts: Optional[datetime] = None
    setup_breakout_price: Optional[float] = None
    # Setup level at fill time — the ceiling this trade is designed to break.
    # evaluate_exit's resistance-approach rule exempts it (and the entry band
    # under it) exactly like the entry headroom gate; captured here because
    # the live entry's setup_price_level can change/clear during the hold.
    setup_level: Optional[float] = None
    # Structural floor at fill time (armed-band): the entry band floor —
    # max(pullback low, level*(1-SIM_ANTICIPATION_MAX_PCT)) — frozen like
    # setup_level. The thesis-lifecycle exit's structural stop reads this;
    # a break of the structure that justified the entry is thesis death
    # regardless of what the flow meters say at that instant.
    setup_pullback_low: Optional[float] = None
    setup_band_floor: Optional[float] = None
    # Thesis-lifecycle phase (SIM_EXIT_REGIME=thesis_lifecycle). break_ts =
    # first event-time the setup level traded at/after fill (fill time for
    # at/above-level fills); proven latches once price reaches the prove
    # target. peak_price ratchets on every tick — the peak trail (a price
    # FACT, exits instantly) reads it. Event-time only — replay-deterministic.
    break_ts: Optional[datetime] = None
    proven: bool = False
    peak_price: float = 0.0
    # F8: depth-regime tracking over the hold. last_depth_active is seeded at
    # entry; depth_flips increments on every depth_active transition (LRU
    # eviction / 30s book staleness), so exit-rule regime changes mid-position
    # are reconstructable from the SELL row.
    last_depth_active: bool = False
    depth_flips: int = 0


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Live simulation engine that evaluates entry/exit on every tick.

    Lifecycle:
      1. Created by DynamicScorer.__init__() (one per ticker)
      2. on_tick() called from _on_timesale (under scorer's lock)
      3. on_quote() called from _on_quote (under scorer's lock)
      4. stop() called when scorer stops

    The engine maintains its own lock for position state. It does NOT
    take the scorer's lock — it receives pre-computed values.
    """

    # Class-level shared state (across all tickers)
    # Lock-scope note (F10, measured 2026-07-26): worst-case on_tick critical
    # section — the full entry gate cascade incl. rmap.nearest_above — is
    # ~6 µs, i.e. <1% lock utilization at 1,000 aggregate prints/s. The fill
    # path does synchronous CSV write+flush under the lock but fires only
    # ~10×/session. Not worth per-symbol lock splitting until a profile of a
    # live multi-symbol session says otherwise.
    _global_lock = threading.Lock()
    _positions: dict[str, SimPosition] = {}
    _trade_counts: dict[str, int] = defaultdict(int)  # per-symbol session counts
    _last_trade_ts: dict[str, datetime] = {}
    # Signal keys (setup_pullback_ts) already acted on, per symbol. Enforces
    # "one buy per signal" — a signal cannot be re-entered even after the
    # position is closed; a fresh signal (new pullback) is required.
    _traded_signal_keys: dict[str, set] = defaultdict(set)
    _logger: Optional[SimTradeLogger] = None
    # When set (e.g. by the backtest runner), new loggers are created against
    # this directory instead of the default SIM_LOG_DIR. Keeps backtest fills
    # physically separate from live trades.
    _log_dir_override: Optional[str] = None

    def __init__(
        self,
        symbol: str,
        entry: "WatchlistEntry",
        ofi_tracker: "OrderFlowTracker",
        rmap: Optional["ResistanceMap"] = None,
        setup_logger: Optional[SetupSignalLogger] = None,
    ) -> None:
        self._symbol = symbol
        self._entry = entry
        self._ofi = ofi_tracker
        self._rmap = rmap
        self._setup_logger = setup_logger
        self._enabled = SIM_ENABLED

        # Local state (updated by caller)
        self._last_price: float = entry.price
        self._bid: float = 0.0
        self._ask: float = 0.0
        self._last_atr: float = 0.0
        self._clock_now: Optional[datetime] = None

        # Latency-based fill: pending orders awaiting fill tick
        self._pending_buy: Optional[_PendingBuy] = None
        self._pending_sell: Optional[_PendingSell] = None

        # Signal dwell: timestamp of the first print in the CURRENT unbroken
        # run of triggered evaluations. None = signal is not currently armed.
        # Any evaluation that fails to trigger clears it, so the hold measures
        # continuous confirmation rather than elapsed time since first sighting.
        # _entry_armed_mid anchors the mid-quote at arming for the up-move
        # requirement (SIM_ENTRY_DWELL_UPMOVE); it lives and dies with
        # _entry_armed_ts.
        self._entry_armed_ts: Optional[datetime] = None
        self._entry_armed_mid: Optional[float] = None
        self._exit_armed_ts: Optional[datetime] = None

        # Initialize shared logger once
        with SimulationEngine._global_lock:
            if SimulationEngine._logger is None:
                SimulationEngine._logger = SimTradeLogger(log_dir=SimulationEngine._log_dir_override)

    # ------------------------------------------------------------------
    # Public API (called from DynamicScorer callbacks)
    # ------------------------------------------------------------------

    def on_tick(
        self,
        price: float,
        bid: float,
        ask: float,
        last_atr: float,
        event_ts: Optional[datetime] = None,
    ) -> None:
        """
        Evaluate entry/exit signals on each T&S print.

        Called from DynamicScorer._on_timesale (already under scorer lock).
        This method must be fast — no blocking I/O.
        """
        if not self._enabled:
            return

        self._last_price = price
        self._bid = bid
        self._ask = ask
        if last_atr > 0:
            self._last_atr = last_atr

        # Replay mode can provide historical event time.
        self._clock_now = event_ts

        # Need valid quotes for fill simulation
        if bid <= 0 or ask <= 0:
            return

        now = self._now()

        with SimulationEngine._global_lock:
            if SIM_LATENCY_FILL:
                # --- Resolve pending latency fills ---
                if self._pending_buy is not None:
                    if now >= self._pending_buy.signal_ts + _FILL_LATENCY:
                        self._execute_pending_buy()
                    else:
                        self._clock_now = None
                        return

                if self._pending_sell is not None:
                    if now >= self._pending_sell.signal_ts + _FILL_LATENCY:
                        self._execute_pending_sell()
                    else:
                        self._clock_now = None
                        return

            # --- Normal signal evaluation ---
            if self._symbol in SimulationEngine._positions:
                self._evaluate_exit()
            else:
                self._evaluate_entry()
        self._clock_now = None

    def on_quote(self, bid: float, ask: float) -> None:
        """Update bid/ask from quote events (lighter than on_tick)."""
        if bid > 0:
            self._bid = bid
        if ask > 0:
            self._ask = ask

    def stop(self) -> None:
        """Clean up when scorer stops. Force-exit any open position."""
        if not self._enabled:
            return
        with SimulationEngine._global_lock:
            self._pending_buy = None
            self._pending_sell = None
            self._entry_armed_ts = None
            self._entry_armed_mid = None
            self._exit_armed_ts = None
            if self._symbol in SimulationEngine._positions:
                self._force_exit("scorer_stopped")

    @classmethod
    def configure_logger(cls, log_dir: Optional[str]) -> None:
        """Point the shared sim-trade logger at *log_dir* (None = default).

        Used by the backtest runner to keep replay fills separate from live
        `pos` trades. Recreates the logger so the change takes effect even if
        one was already lazily created.
        """
        with cls._global_lock:
            cls._log_dir_override = log_dir
            if cls._logger is not None:
                cls._logger.close()
            cls._logger = SimTradeLogger(log_dir=log_dir)

    @classmethod
    def get_logger(cls) -> Optional[SimTradeLogger]:
        return cls._logger

    @classmethod
    def active_positions(cls) -> dict[str, SimPosition]:
        with cls._global_lock:
            return dict(cls._positions)

    @classmethod
    def reset_session(cls) -> None:
        """Reset all session state (for new trading day)."""
        with cls._global_lock:
            cls._positions.clear()
            cls._trade_counts.clear()
            cls._last_trade_ts.clear()
            cls._traded_signal_keys.clear()
            if cls._logger:
                cls._logger.close()
                cls._logger = SimTradeLogger(log_dir=cls._log_dir_override)

    # ------------------------------------------------------------------
    # Internal (must be called under _global_lock)
    # ------------------------------------------------------------------

    def _evaluate_entry(self) -> None:
        """Check entry conditions and simulate a buy if all pass."""
        # --- Guard: max concurrent positions ---
        if len(SimulationEngine._positions) >= SIM_MAX_CONCURRENT:
            return

        # --- Guard: per-symbol trade limit (0 = unlimited) ---
        if SIM_MAX_TRADES_PER_SYMBOL > 0 and SimulationEngine._trade_counts[self._symbol] >= SIM_MAX_TRADES_PER_SYMBOL:
            return

        # --- Guard: cooldown ---
        last_ts = SimulationEngine._last_trade_ts.get(self._symbol)
        now = self._now()
        if last_ts is not None:
            elapsed = (now - last_ts).total_seconds()
            if elapsed < SIM_COOLDOWN_SECS:
                return

        # --- Guard: one buy per signal ---
        # A signal is identified by its setup pullback timestamp. Once a
        # signal has been traded it cannot be re-entered (even after the
        # position is closed) — a fresh signal is required to buy again.
        signal_key = self._signal_key()
        if signal_key is not None and signal_key in SimulationEngine._traded_signal_keys[self._symbol]:
            return

        # --- Evaluate entry signal ---
        sig = evaluate_entry(
            entry=self._entry,
            ofi_tracker=self._ofi,
            last_price=self._last_price,
            last_atr=self._last_atr,
            rmap=self._rmap,
        )
        if not sig.triggered:
            self._entry_armed_ts = None
            self._entry_armed_mid = None
            return

        # --- Dwell: require the signal to hold continuously ---
        # Arm on the first triggered print, execute on the first print at or
        # past the hold. Deliberately re-checked per print: a spike that
        # satisfies the chain once and decays never reaches this branch again.
        # With SIM_ENTRY_DWELL_UPMOVE, price must also advance over the hold:
        # mid below the arm anchor restarts the clock at the new lower mid,
        # and the fill needs mid strictly above the anchor — a stalled mid
        # extends the hold until it either up-ticks (fill) or dips (re-arm).
        if _ENTRY_DWELL:
            mid = (self._bid + self._ask) / 2.0  # on_tick guarantees bid/ask > 0
            if self._entry_armed_ts is None:
                self._entry_armed_ts = now
                self._entry_armed_mid = mid
                return
            if SIM_ENTRY_DWELL_UPMOVE and mid < self._entry_armed_mid:
                self._entry_armed_ts = now
                self._entry_armed_mid = mid
                return
            if now - self._entry_armed_ts < _ENTRY_DWELL:
                return
            if SIM_ENTRY_DWELL_UPMOVE and mid <= self._entry_armed_mid:
                return
        self._entry_armed_ts = None
        self._entry_armed_mid = None

        # --- Defer fill by latency window (when enabled) ---
        if SIM_LATENCY_FILL:
            self._pending_buy = _PendingBuy(
                signal_ts=now,
                signal=sig,
                signal_key=signal_key,
            )
            return

        # --- Immediate fill ---
        self._execute_buy(sig, signal_key)

    def _execute_pending_buy(self) -> None:
        """Resolve a pending buy — latency window has elapsed, fill at current NBBO."""
        pb = self._pending_buy
        self._pending_buy = None
        if pb is None:
            return
        self._execute_buy(pb.signal, pb.signal_key)

    def _execute_buy(self, sig: EntrySignal, signal_key: Optional[datetime]) -> None:
        """Simulate a buy fill and open a position."""
        now = self._now()
        fill_price = self._compute_buy_fill()
        shares = self._compute_position_size(fill_price)
        if shares <= 0:
            return

        pos = SimPosition(
            symbol=self._symbol,
            entry_price=fill_price,
            shares=shares,
            entry_time=now,
            entry_signal=sig,
            bid_at_entry=self._bid,
            ask_at_entry=self._ask,
            setup_name=self._entry.setup_pattern_name if self._entry.setup_pattern_name not in ("pullback_pending", "pullback_detected") else "pullback_new_high",
            setup_timeframe=self._entry.setup_pattern_timeframe,
            setup_score=self._entry.setup_pattern_score,
            setup_pullback_ts=self._entry.setup_pullback_ts,
            setup_breakout_ts=self._entry.setup_breakout_ts,
            setup_breakout_price=self._entry.setup_breakout_price,
            setup_level=getattr(self._entry, "setup_price_level", None),
            setup_pullback_low=getattr(self._entry, "setup_pullback_low", None),
            setup_band_floor=(
                entry_band_floor(
                    self._entry.setup_price_level,
                    getattr(self._entry, "setup_pullback_low", None),
                )
                if getattr(self._entry, "setup_price_level", None) else None
            ),
            last_depth_active=self._ofi.depth_active,
        )
        # Thesis lifecycle: a fill at/above the level starts its prove clock
        # immediately; a below-level (anticipation) fill first owes a break.
        if pos.setup_level is None or fill_price >= pos.setup_level:
            pos.break_ts = now
        pos.peak_price = fill_price
        SimulationEngine._positions[self._symbol] = pos
        # Opening a position hands control to the exit path — start its dwell
        # clean so arming observed before the fill cannot carry into the hold.
        self._exit_armed_ts = None
        SimulationEngine._trade_counts[self._symbol] += 1
        SimulationEngine._last_trade_ts[self._symbol] = now
        if signal_key is not None:
            SimulationEngine._traded_signal_keys[self._symbol].add(signal_key)

        # --- Log the fill ---
        spread = self._ask - self._bid
        sim_fill = SimFill(
            ts=now,
            symbol=self._symbol,
            side="BUY",
            fill_price=fill_price,
            shares=shares,
            notional=fill_price * shares,
            bid=self._bid,
            ask=self._ask,
            spread=spread,
            entry_reason=" | ".join(sig.reasons),
            eval_confidence=sig.eval_confidence,
            live_confidence=sig.live_confidence,
            setup_name=pos.setup_name,
            setup_timeframe=pos.setup_timeframe,
            setup_score=pos.setup_score,
            setup_pullback_ts=pos.setup_pullback_ts,
            setup_breakout_ts=pos.setup_breakout_ts,
            setup_breakout_price=pos.setup_breakout_price,
            buying_pressure=self._ofi.buying_pressure,
            selling_pressure=self._ofi.selling_pressure,
            net_pressure=sig.net_pressure,
            entry_speed=self._ofi.trade_speed,
            entry_buy_ratio=self._ofi.buy_ratio,
            pressure_acceleration=self._ofi.pressure_acceleration,
            pressure_acceleration_ready=self._ofi.pressure_acceleration_ready,
            trade_imbalance=self._ofi.trade_imbalance,
            ofi_rolling=self._ofi.ofi_rolling,
            depth_active=self._ofi.depth_active,
            ofi_normalized=self._ofi.ofi_normalized,
            book_depth_avg=self._ofi.book_depth_avg,
        )
        SimulationEngine._logger.log_fill(sim_fill)
        if self._setup_logger is not None:
            self._setup_logger.mark_entered(
                symbol=self._symbol,
                ts=now,
                entry_price=fill_price,
                shares=shares,
            )

        t_local = now.astimezone().strftime("%H:%M:%S")
        logger.info(
            _g(f"[{self._symbol}] ▲ BUY  {shares} sh @ ${fill_price:.2f}  {t_local}") +
            f"  {_dim('|')}  {' · '.join(sig.reasons)}"
        )

    def _evaluate_exit(self) -> None:
        """Check exit conditions and simulate a sell if triggered."""
        pos = SimulationEngine._positions.get(self._symbol)
        if pos is None:
            return

        # F8: count depth-regime transitions during the hold — exit 2 arming
        # or disarming mid-position (LRU eviction / 30s book staleness).
        cur_depth = self._ofi.depth_active
        if cur_depth != pos.last_depth_active:
            pos.depth_flips += 1
            pos.last_depth_active = cur_depth

        import config as _config
        now = self._now()
        thesis = _config.SIM_EXIT_REGIME == "thesis_lifecycle"

        sig = evaluate_exit(
            entry=self._entry,
            ofi_tracker=self._ofi,
            last_price=self._last_price,
            entry_price=pos.entry_price,
            last_atr=self._last_atr,
            rmap=self._rmap,
            entry_setup_level=pos.setup_level,
        )
        flow_lapse_now = sig.triggered and sig.reason.split(":", 1)[0] in (
            "pressure_reversal", "accel_collapse", "ofi_negative",
        )

        if thesis:
            # Phase advance + structural stops + prove-it timeout. These are
            # thesis-level exits: they fire immediately (no dwell — the
            # structure breaking or the clock expiring is not a flow blip),
            # but still honor the latency fill model like any signal exit.
            reason = self._thesis_pre_exit(pos, now, flow_lapse_now)
            if reason is not None:
                self._exit_armed_ts = None
                if SIM_LATENCY_FILL:
                    self._pending_sell = _PendingSell(signal_ts=now, reason=reason)
                    return
                self._execute_sell(pos, reason)
                return

        if not sig.triggered:
            self._exit_armed_ts = None
            return

        # --- Dwell: require the exit signal to hold continuously ---
        # Same arm/reset discipline as the entry side. Note the forced-exit
        # paths (_force_exit: EOD, scorer stop) deliberately bypass this —
        # they are not signal-driven and must not be delayed or reset.
        # Thesis lifecycle, PROVEN phase: flow-lapse exits must SUSTAIN —
        # a trade that already demonstrated follow-through is not exited on
        # a single-instant meter blip (the flow reads breathe mid-squeeze);
        # the structural trail above is the fast backstop.
        required_dwell = _EXIT_DWELL
        if thesis and pos.proven and sig.reason.split(":", 1)[0] in (
            "pressure_reversal", "accel_collapse", "ofi_negative",
        ):
            required_dwell = max(
                _EXIT_DWELL, timedelta(seconds=_config.SIM_EXIT_LAPSE_SUSTAIN_SECS)
            )
        if required_dwell:
            if self._exit_armed_ts is None:
                self._exit_armed_ts = now
                return
            if now - self._exit_armed_ts < required_dwell:
                return
        self._exit_armed_ts = None

        if SIM_LATENCY_FILL:
            self._pending_sell = _PendingSell(signal_ts=now, reason=sig.reason)
            return

        self._execute_sell(pos, sig.reason)

    def _thesis_pre_exit(
        self, pos: SimPosition, now: datetime, flow_lapse_now: bool = False
    ) -> Optional[str]:
        """Thesis-lifecycle phase machine. Returns an exit reason, or None.

        Encodes the user's timing principle directly: the entry is a
        hypothesis about immediate follow-through.
          1. Phase advance: below-level fills owe a level break; any fill
             owes progress to max(fill, level)*(1+SIM_PROVE_PROGRESS_PCT).
             Reaching it latches PROVEN.
          2. Structural stop (flow-independent, first priority): UNPROVEN
             trades die on a band-floor break; PROVEN trades trail the
             broken level's anticipation band — a reclaimed level failing
             to hold is thesis death no matter what the meters read.
          3. Prove-it timeout: no break / no progress within the horizons
             means the timing was wrong — exit and let a NEW signal re-enter
             (GLE 08-11 +9.5% three minutes later is a re-entry, not a hold).
        """
        import config as _config
        price = self._last_price
        if price <= 0:
            return None
        level = pos.setup_level

        # --- 1. Phase advance (event-time) ---
        pos.peak_price = max(pos.peak_price, price)
        if pos.break_ts is None and level is not None and price >= level:
            pos.break_ts = now
        prove_ref = max(pos.entry_price, level or pos.entry_price)
        if not pos.proven and pos.break_ts is not None:
            if price >= prove_ref * (1.0 + _config.SIM_PROVE_PROGRESS_PCT):
                pos.proven = True

        buffer = 1.0 - _config.SIM_STRUCT_STOP_BUFFER_PCT

        # --- 2. Structural stops (price FACTS — instant, no sustain) ---
        # Peak reversal: two INSTANT facts together — price has given back
        # SIM_TRAIL_PEAK_GIVEBACK_PCT off the spike high AND flow is red at
        # this very tick (single-instant read, NOT the 3s sustained form).
        # A violent blow-off reversal shows both within a tick or two, so
        # the fast-squeeze collapse exits near the top without waiting out
        # the sustain; a healthy mid-squeeze wiggle (price dips, flow still
        # green) holds. Price-off-peak ALONE is deliberately not a stop —
        # measured on the 9-day replay it decapitated the +22% winner at
        # +3% (3% wiggles are normal breathing on these names). Armed only
        # once the peak extends SIM_TRAIL_PEAK_ARM_PCT past the prove
        # reference, so retest wiggle near the level never arms it.
        if pos.peak_price >= prove_ref * (1.0 + _config.SIM_TRAIL_PEAK_ARM_PCT):
            peak_trail = pos.peak_price * (1.0 - _config.SIM_TRAIL_PEAK_GIVEBACK_PCT)
            if price <= peak_trail and flow_lapse_now:
                return (
                    f"peak_reversal: {price:.4f} <= {peak_trail:.4f} "
                    f"({_config.SIM_TRAIL_PEAK_GIVEBACK_PCT:.0%} off peak "
                    f"{pos.peak_price:.4f}) with flow lapsed"
                )
        if pos.proven and level is not None:
            trail = level * (1.0 - _config.SIM_SETUP_ANTICIPATION_PCT) * buffer
            if price <= trail:
                return (
                    f"structural_stop: {price:.4f} <= {trail:.4f} "
                    f"(proven trail under level {level:.4f})"
                )
        elif pos.setup_band_floor is not None:
            stop = pos.setup_band_floor * buffer
            if price <= stop:
                return (
                    f"structural_stop: {price:.4f} <= {stop:.4f} "
                    f"(band floor {pos.setup_band_floor:.4f})"
                )

        # --- 3. Prove-it timeout ---
        if not pos.proven:
            if pos.break_ts is None:
                waited = (now - pos.entry_time).total_seconds()
                if waited >= _config.SIM_PROVE_BREAK_SECS:
                    return (
                        f"prove_timeout: level {level:.4f} never traded in "
                        f"{waited:.0f}s after sub-level fill"
                    )
            else:
                waited = (now - pos.break_ts).total_seconds()
                if waited >= _config.SIM_PROVE_PROGRESS_SECS:
                    return (
                        f"prove_timeout: no progress to "
                        f"+{_config.SIM_PROVE_PROGRESS_PCT:.1%} within {waited:.0f}s "
                        f"of the break"
                    )
        return None

    def _execute_pending_sell(self) -> None:
        """Resolve a pending sell — latency window has elapsed, fill at current NBBO."""
        ps = self._pending_sell
        self._pending_sell = None
        if ps is None:
            return
        pos = SimulationEngine._positions.get(self._symbol)
        if pos is None:
            return
        self._execute_sell(pos, ps.reason)

    def _force_exit(self, reason: str) -> None:
        """Force-exit an open position (e.g. end of day, scorer stop)."""
        pos = SimulationEngine._positions.get(self._symbol)
        if pos is None:
            return
        self._execute_sell(pos, f"forced:{reason}")

    def _execute_sell(self, pos: SimPosition, reason: str) -> None:
        """Simulate a sell fill and log P&L."""
        now = self._now()
        # Closing hands control back to the entry path (and covers _force_exit,
        # which never touches the dwell state itself).
        self._entry_armed_ts = None
        self._entry_armed_mid = None
        self._exit_armed_ts = None
        fill_price = self._compute_sell_fill()
        pnl_dollars = (fill_price - pos.entry_price) * pos.shares
        pnl_pct = (fill_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
        hold_secs = (now - pos.entry_time).total_seconds()

        spread = self._ask - self._bid if self._ask > 0 and self._bid > 0 else 0.0
        sim_fill = SimFill(
            ts=now,
            symbol=self._symbol,
            side="SELL",
            fill_price=fill_price,
            shares=pos.shares,
            notional=fill_price * pos.shares,
            bid=self._bid,
            ask=self._ask,
            spread=spread,
            exit_reason=reason,
            eval_confidence=self._entry.eval_confidence,
            live_confidence=self._entry.live_confidence,
            setup_name=pos.setup_name,
            setup_timeframe=pos.setup_timeframe,
            setup_score=pos.setup_score,
            setup_pullback_ts=pos.setup_pullback_ts,
            setup_breakout_ts=pos.setup_breakout_ts,
            setup_breakout_price=pos.setup_breakout_price,
            buying_pressure=self._ofi.buying_pressure,
            selling_pressure=self._ofi.selling_pressure,
            net_pressure=self._ofi.buying_pressure - self._ofi.selling_pressure,
            entry_speed=self._ofi.trade_speed,
            entry_buy_ratio=self._ofi.buy_ratio,
            pressure_acceleration=self._ofi.pressure_acceleration,
            pressure_acceleration_ready=self._ofi.pressure_acceleration_ready,
            trade_imbalance=self._ofi.trade_imbalance,
            ofi_rolling=self._ofi.ofi_rolling,
            depth_active=self._ofi.depth_active,
            ofi_normalized=self._ofi.ofi_normalized,
            book_depth_avg=self._ofi.book_depth_avg,
            depth_flips=pos.depth_flips,
            pnl_dollars=pnl_dollars,
            pnl_pct=pnl_pct,
            hold_secs=hold_secs,
        )
        SimulationEngine._logger.log_fill(sim_fill)

        # Remove position
        SimulationEngine._positions.pop(self._symbol, None)

        pnl_str = f"${pnl_dollars:+.2f} ({pnl_pct:+.2%})"
        pnl_color = _g if pnl_dollars >= 0 else _r
        t_local = now.astimezone().strftime("%H:%M:%S")
        logger.info(
            _r(f"[{self._symbol}] ▼ SELL {pos.shares} sh @ ${fill_price:.2f}  {t_local}") +
            f"  {_dim('|')}  {reason}  {_dim('|')}  P&L={pnl_color(pnl_str)}  {_dim(f'held {hold_secs:.0f}s')}"
        )

    # ------------------------------------------------------------------
    # Fill price computation
    # ------------------------------------------------------------------

    def _compute_buy_fill(self) -> float:
        """Simulate buy fill price from live NBBO."""
        if SIM_USE_ASK_FOR_BUY:
            base = self._ask
        else:
            base = (self._bid + self._ask) / 2.0
        if not SIM_LATENCY_FILL:
            slip = base * (SIM_FILL_SLIP_BPS / 10_000.0)
            return round(base + slip, 4)
        return round(base, 4)

    def _compute_sell_fill(self) -> float:
        """Simulate sell fill price from live NBBO."""
        if SIM_USE_BID_FOR_SELL:
            base = self._bid
        else:
            base = (self._bid + self._ask) / 2.0
        if not SIM_LATENCY_FILL:
            slip = base * (SIM_FILL_SLIP_BPS / 10_000.0)
            return round(base - slip, 4)
        return round(base, 4)

    def _limit_price_buy(self) -> float:
        """IOC limit price for buy orders: ask + offset."""
        return round(self._ask + SIM_FILL_OFFSET_CENTS / 100.0, 4)

    def _limit_price_sell(self) -> float:
        """IOC limit price for sell orders: bid - offset."""
        return round(self._bid - SIM_FILL_OFFSET_CENTS / 100.0, 4)

    def _compute_position_size(self, fill_price: float) -> int:
        """Compute number of shares based on account size and risk limits."""
        max_notional = SIM_ACCOUNT_SIZE * SIM_MAX_POSITION_PCT
        if fill_price <= 0:
            return 0
        shares = int(max_notional / fill_price)
        return max(1, shares)

    def _now(self) -> datetime:
        """Return replay event time when provided, else wall-clock UTC."""
        return self._clock_now or datetime.now(tz=timezone.utc)

    def _signal_key(self) -> Optional[datetime]:
        """Identity of the current setup signal (its pullback timestamp).

        Used to enforce one buy per signal. Returns None when no setup
        pullback timestamp is available, in which case the per-signal
        guard is skipped (other guards still apply).
        """
        return self._entry.setup_pullback_ts
