"""
exec/order_flow.py

Order Flow Imbalance (OFI) tracker for the execution layer.

Consumes LiveQuote (L1 NBBO updates) and LiveTimesale (T&S prints) from
the Tradier stream and computes three real-time signals:

  1. BOOK LEAN  (volume imbalance ρ)
     ρ = (bid_size - ask_size) / (bid_size + ask_size)
     Range: [-1, +1].  +1 = all weight on bid (bullish).
     Source: Cartea et al. (2018), Lipton et al. (2013).

  2. ORDER FLOW IMBALANCE  (L1 OFI / multi-level depth OFI)
     Tracks Δ bid volume and Δ ask volume across consecutive quote
     updates using the three-case formula from Cont, Kukanov & Stoikov
     (2014) / Xu, Gould & Howison (2019).  Positive OFI = net buying
     pressure arriving at the book.
     When an L2 depth feed is active (IBKR market depth via on_book),
     the same three-case rule is applied per book level with geometric
     level weights (OFI_L2_LEVEL_DECAY^(m-1)) and supersedes the L1
     samples — all consumers transparently read the deeper signal.
     Source: Cont et al. (2014), Kolm, Turiel & Westray (2021).

  3. TRADE FLOW IMBALANCE  (aggressor ratio)
     Classifies each T&S print as buyer- or seller-initiated by
     comparing execution price to the NBBO at time of print:
       price >= ask  →  buyer lifted offer  (buy volume)
       price <= bid  →  seller hit bid      (sell volume)
       otherwise     →  midpoint / crossed  (neutral)
     A print from the merged tastytrade+IBKR tape arrives already sided
     (``event.aggressor``, judged the same way against the NBBO and the last
     800 ms of quotes) and that side is used as is.
     Rolling ratio over a configurable window.

All three signals are exposed as properties and can be read by the
DynamicScorer (for flag evaluation) or by the execution engine (for
entry/exit confirmation).

Thread-safety: callers must hold a lock around on_quote / on_timesale,
or route all calls through a single thread (as DynamicScorer does).

USAGE:
    from exec.order_flow import OrderFlowTracker

    ofi = OrderFlowTracker()

    # Feed from stream callbacks (inside DynamicScorer._on_quote / _on_timesale)
    ofi.on_quote(event)
    ofi.on_timesale(event)

    # Read current state
    ofi.book_lean        # float in [-1, +1]
    ofi.ofi_rolling      # float, cumulative OFI over window
    ofi.trade_imbalance   # float in [-1, +1]
    ofi.buying_pressure   # int, orders at ask in pressure window
    ofi.selling_pressure  # int, orders at bid in pressure window
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import islice
from typing import TYPE_CHECKING, Optional

from config import (
    OFI_WINDOW_SECS,
    OFI_L2_LEVEL_DECAY,
    OFI_DEPTH_STALE_SECS,
    OFI_EMA_TAU_SECS,
    OFI_GATE_SOURCE,
    PRESSURE_WINDOW_SECS,
    PRESSURE_ACTIVITY_WINDOW_SECS,
    BOOK_LEAN_EMA_ALPHA,
    HIDDEN_WINDOW_SECS,
    HIDDEN_KEEP_SECS,
    HIDDEN_BOOK_SAMPLE_SECS,
    HIDDEN_PRICE_TOL,
    HIDDEN_RATIO_FLOOR,
    HIDDEN_RATIO_CEILING,
    HIDDEN_MIN_SAMPLES,
    HIDDEN_MIN_HALF_WINDOW_SECS,
    TAPE_BASELINE_WINDOW_SECS,
    TAPE_BASELINE_MIN_PRINTS,
    TAPE_ACCEL_ESTIMATOR,
    TAPE_EMA_FAST_TAU_SECS,
    TAPE_EMA_SLOW_TAU_SECS,
    TAPE_ACCEL_EMA_MIN_NEFF,
    TAPE_PHI_CLIP_HI,
    TAPE_PHI_MIN_WEIGHT,
    TAPE_SPEED_GATE_SOURCE,
    TAPE_RATIO_GATE_SOURCE,
    PROGRESS_WINDOW_SECS,
    PROGRESS_MIN_SPAN_SECS,
    SIM_PRESSURE_SPEED_MIN,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_VOL_IMBALANCE_MIN,
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_PROGRESS_MIN,
    SIM_OFI_NORM_MIN,
    SIM_MAX_SETUP_EXTENSION_PCT,
    SIM_TRIGGER_TRAIL_WINDOW_SECS,
    SIM_TRIGGER_TRAIL_MIN_SPAN_SECS,
    SIM_WALL_BAND_PCT,
    SIM_WALL_EVAP_MIN,
    IBKR_L2_ROWS,
)
from market_data.tradier_stream import LiveQuote, LiveTimesale

if TYPE_CHECKING:  # avoid importing ib_insync-dependent module at runtime
    from market_data.ibkr_stream import OrderBookSnapshot

logger = logging.getLogger(__name__)

# Full gate-predicate sweeps for the armed-band latches decay-read several
# EMA kernels; on burst tape (100s of prints/sec) running them per print
# would double ingestion cost for no benefit against multi-second TTLs.
_LATCH_EVAL_MIN_GAP_SECS = 0.25


# ---------------------------------------------------------------------------
# L1 Book State — immutable snapshot of one side at a point in time
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class L1BookState:
    """Snapshot of L1 best bid or ask at a single point in time."""
    price: float
    size: int
    ts: datetime


# ---------------------------------------------------------------------------
# OFI event — one computed sample (produced on each quote update)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OFIEvent:
    """Single order flow imbalance sample from one quote update."""
    ts: datetime
    delta_bid: int          # ΔV^b — net volume change at best bid
    delta_ask: int          # ΔV^a — net volume change at best ask
    ofi: int                # delta_bid - delta_ask

    @property
    def is_buy_pressure(self) -> bool:
        return self.ofi > 0

    @property
    def is_sell_pressure(self) -> bool:
        return self.ofi < 0


# ---------------------------------------------------------------------------
# Trade classification — one per T&S print
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ClassifiedTrade:
    """A T&S print classified by aggressor side."""
    ts: datetime
    price: float
    size: int
    side: str               # "buy" | "sell" | "mid"


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------


class OrderFlowTracker:
    """
    Stateful tracker that consumes L1 quotes and T&S prints to produce
    three real-time order flow signals.

    Feed it from stream callbacks. Read signals via properties.
    """

    def __init__(
        self,
        ofi_window_secs: int = OFI_WINDOW_SECS,
        pressure_window_secs: int = PRESSURE_WINDOW_SECS,
        book_lean_ema_alpha: float = BOOK_LEAN_EMA_ALPHA,
        activity_window_secs: int = PRESSURE_ACTIVITY_WINDOW_SECS,
    ) -> None:
        # --- L1 state (current quote, for OFI delta computation) ---
        self._cur_bid: Optional[L1BookState] = None
        self._cur_ask: Optional[L1BookState] = None

        # --- OFI rolling window ---
        self._ofi_window = ofi_window_secs
        self._ofi_events: deque[OFIEvent] = deque()
        self._ofi_sum: int = 0          # running sum of e.ofi (hot-path read)

        # --- L2 depth state (multi-level OFI from IBKR market depth) ---
        # Ladders are rank-ordered (price, size) tuples, best level first,
        # already aggregated by price by the stream layer.
        self._prev_book_bids: list[tuple[float, int]] = []
        self._prev_book_asks: list[tuple[float, int]] = []
        self._book_event_count: int = 0
        self._last_book_ts: Optional[datetime] = None
        self._depth_samples: deque[tuple[datetime, int]] = deque()  # (ts, total displayed shares)
        self._depth_sum: int = 0

        # --- Latest event timestamp (trade/quote/book) ---
        # Used as "now" for staleness + EMA decay so replays behave
        # identically to live (never wall clock).
        self._last_event_ts: Optional[datetime] = None

        # --- EMA-kernel OFI (exponentially forgotten flow / depth) ---
        # Each OFI sample adds its signed flow to a sum that decays with
        # time constant OFI_EMA_TAU_SECS; depth snapshots feed a decayed
        # mean of displayed depth with the same kernel. The ratio is the
        # EMA analogue of ofi_normalized. Reads decay to the latest event
        # timestamp WITHOUT mutating state (replay-safe), so stale flow
        # ages out between events instead of squatting in a boxcar window.
        self._ofi_ema_sum: float = 0.0
        self._ofi_ema_dep_pv: float = 0.0   # decayed sum of depth samples
        self._ofi_ema_dep_w: float = 0.0    # decayed count of depth samples
        self._ofi_ema_last_ts: Optional[datetime] = None

        # --- EMA tape intensity (dual exponential kernel accel estimator) ---
        # c_fast/c_slow are exponentially-decayed counts of directional
        # prints; rate = c/tau. accel = (r_fast - r_slow)/(r_fast + r_slow).
        # c_slow_sq decays at tau_slow/2 and yields Kish effective N:
        # n_eff = c_slow^2 / c_slow_sq, for the significance z = accel*sqrt(n_eff).
        self._ema_c_fast: float = 0.0
        self._ema_c_slow: float = 0.0
        self._ema_c_slow_sq: float = 0.0
        # Buy-side share of the fast kernel (ratio gate). Decays identically
        # to c_fast, so the ratio b_fast/c_fast is decay-invariant on read.
        self._ema_b_fast: float = 0.0
        self._ema_last_ts: Optional[datetime] = None
        # Dispersion (phi-hat) accumulators: decayed moments of directional
        # inter-arrival gaps at tau_slow. phi = clip(CV^2, 1, TAPE_PHI_CLIP_HI)
        # corrects the Kish z for print clustering (batched arrivals violate
        # the independence n_eff assumes). _phi_cv2 caches the last trusted
        # raw CV^2 so quiet spells carry the estimate instead of resetting it.
        self._phi_w: float = 0.0
        self._phi_dt: float = 0.0
        self._phi_dt_sq: float = 0.0
        self._phi_cv2: float = 1.0

        # --- Book lean (EMA-smoothed) ---
        self._book_lean_ema: float = 0.0
        self._book_lean_raw: float = 0.0
        self._ema_alpha = book_lean_ema_alpha
        self._ema_initialised: bool = False

        # --- Trade flow (aggressor classification) ---
        # Trade/volume retention must cover every consumer that reads
        # self._trades: density telemetry, buy_vol/sell_vol and
        # hidden-liquidity detection (which looks back HIDDEN_WINDOW_SECS).
        self._trade_window = max(pressure_window_secs, activity_window_secs, HIDDEN_WINDOW_SECS)
        self._trades: deque[ClassifiedTrade] = deque()
        self._buy_vol: int = 0      # running sum inside window
        self._sell_vol: int = 0

        # --- Buying / selling pressure (order counts, short window) ---
        self._pressure_window = pressure_window_secs
        self._buy_prints: deque[datetime] = deque()   # timestamps of buy trades
        self._sell_prints: deque[datetime] = deque()   # timestamps of sell trades
        self._pressure_accel_cache_key: tuple[Optional[datetime], Optional[datetime], Optional[datetime], int, int] | None = None
        self._pressure_accel_cache: tuple[int, int, int, int] = (0, 0, 0, 0)

        # --- T&S baseline telemetry ---
        # Stores timestamps of ALL classified prints (buy + sell) over a
        # window of baseline + signal seconds.  The baseline speed is
        # computed from the older portion [now - total, now - signal], and
        # the signal speed from the recent [now - signal, now].
        self._baseline_window = TAPE_BASELINE_WINDOW_SECS
        # Retain prints long enough to serve both the baseline reference and
        # the (wider) activity-count window, whichever is longer, so
        # activity_count never silently undercounts.
        self._activity_window = activity_window_secs
        self._baseline_total_window = max(
            TAPE_BASELINE_WINDOW_SECS + pressure_window_secs,
            activity_window_secs,
        )
        self._baseline_prints: deque[datetime] = deque()
        # Activity-window suffix of the baseline prints, maintained
        # incrementally so activity_count is O(1) per read instead of
        # re-walking the window (O(occupancy)) on every tick-criteria read —
        # quadratic on burst tape (>1k prints/s).
        self._activity_prints: deque[datetime] = deque()

        # --- Hidden liquidity detection ---
        # Rolling samples of ask-side book state.  Used to detect
        #   hidden_seller: ask price not advancing while buy-side aggressor
        #     volume hits/exceeds displayed ask size (iceberg refilling at ask)
        # Sourced from real L1 quotes while no depth feed flows, and from the
        # book's best ask (decimated to HIDDEN_BOOK_SAMPLE_SECS) while depth
        # is active — replay/backtest quotes are synthetic (placeholder
        # sizes) and never feed this.
        self._ask_samples: deque[L1BookState] = deque()
        self._last_ask_sample_ts: Optional[datetime] = None

        # --- Trailing price-progress samples ---
        # (ts, price) of directional prints, decimated to >= 1s apart so the
        # progress read scans <= ~window+2 slots instead of the raw tape
        # (O(prints) would be quadratic on burst tape). Retained slightly
        # past the lookback window.
        self._progress_samples: deque[tuple[datetime, float]] = deque()

        # --- Armed-band state (level-anchored latches + trigger primitives) ---
        # The scorer pushes the active setup's entry band each event
        # (set_entry_band). While a band is set and the last trade price sits
        # inside it, every gate predicate that evaluates True stamps its
        # last-true event-ts — "evidence was true recently while in band" —
        # and the two trigger primitives stamp edges: accel rising-edge and
        # trailing-high micro-break. All event-time (replay-deterministic).
        self._band_level: Optional[float] = None
        self._band_floor: Optional[float] = None
        self._band_top: Optional[float] = None
        self._band_pullback_ts: Optional[datetime] = None
        self._gate_last_true: dict[str, datetime] = {}
        self._accel_prev_ok: bool = False
        self._accel_edge_ts: Optional[datetime] = None
        self._last_latch_eval_ts: Optional[datetime] = None
        self._last_trade_price: Optional[float] = None
        # Trailing-high micro-break trigger. Ratchet runs on RAW prints —
        # a decimated series misses verticals that clear the high between
        # samples (the exact chase this trigger exists to beat). _trail_max
        # is a monotonic (price-decreasing) deque: front = trailing high.
        self._trail_prices: deque[tuple[datetime, float]] = deque()
        self._trail_max: deque[tuple[datetime, float]] = deque()
        self._microbreak_ts: Optional[datetime] = None
        self._microbreak_price: Optional[float] = None
        # Level-anchored ask-wall (armed-band arm evidence): EMA baseline of
        # displayed ask shares within ±SIM_WALL_BAND_PCT of the setup level,
        # and the current sample's evaporation trend vs that baseline.
        # Weighted-mean decay pair (pv/w) at OFI_EMA_TAU_SECS; samples
        # decimated to HIDDEN_BOOK_SAMPLE_SECS like the hidden-liquidity
        # read (undecimated book bursts would multiply the cost for no
        # information at multi-second arm TTLs).
        self._wall_pv: float = 0.0
        self._wall_w: float = 0.0
        self._wall_ema_last_ts: Optional[datetime] = None
        self._wall_last_sample_ts: Optional[datetime] = None
        self._wall_trend: Optional[float] = None
        self._wall_covered: bool = False

        # --- Counters ---
        self._quote_count: int = 0
        self._trade_count: int = 0

    # -----------------------------------------------------------------------
    # Quote handler — L1 book lean + OFI
    # -----------------------------------------------------------------------

    def on_quote(self, event: LiveQuote) -> Optional[OFIEvent]:
        """
        Process an L1 NBBO update.  Computes book lean and OFI delta.

        Returns the OFIEvent if an OFI sample was produced (requires a
        previous quote to compute deltas), or None on the first call.

        While the L2 depth feed is active, the L1 OFI sample is NOT
        appended (the depth feed supersedes it — appending both would
        double-count the inside level). Book lean and hidden-liquidity
        sampling always run.
        """
        self._quote_count += 1

        bid = L1BookState(price=event.bid, size=event.bid_size, ts=event.bid_ts)
        ask = L1BookState(price=event.ask, size=event.ask_size, ts=event.ask_ts)
        # Replay/backtest quotes carry real NBBO prices but placeholder
        # sizes — every size-derived read must skip them or it reads noise
        # (a 1-share "displayed ask" made hidden_seller trip on any
        # non-advancing ask, spuriously blocking replay entries).
        synthetic = getattr(event, "synthetic", False)

        # --- Book lean (instantaneous + EMA) ---
        total_size = bid.size + ask.size
        if total_size > 0 and not synthetic:
            raw = (bid.size - ask.size) / total_size
            self._book_lean_raw = raw
            if not self._ema_initialised:
                self._book_lean_ema = raw
                self._ema_initialised = True
            else:
                self._book_lean_ema = (
                    self._ema_alpha * raw
                    + (1 - self._ema_alpha) * self._book_lean_ema
                )

        # --- OFI: Cont et al. (2014) three-case delta ---
        ofi_event: Optional[OFIEvent] = None
        now = max(event.bid_ts, event.ask_ts)
        self._touch_event_ts(now)
        if (
            self._cur_bid is not None
            and self._cur_ask is not None
            and not self.depth_active
        ):
            delta_bid = self._compute_delta_bid(self._cur_bid, bid)
            delta_ask = self._compute_delta_ask(self._cur_ask, ask)
            ofi_event = OFIEvent(
                ts=now,
                delta_bid=delta_bid,
                delta_ask=delta_ask,
                ofi=delta_bid - delta_ask,
            )
            self._ofi_events.append(ofi_event)
            self._ofi_sum += ofi_event.ofi
            self._ofi_ema_add(now, float(ofi_event.ofi))
            self._trim_ofi(now)

        # --- Hidden-liquidity samples: record current displayed ask size ---
        # Skipped for synthetic quotes (placeholder sizes) and while the
        # depth feed is active (the book's best ask supersedes — same
        # precedent as the L1 OFI suppression above).
        if ask.size > 0 and ask.price > 0 and not synthetic and not self.depth_active:
            self._ask_samples.append(ask)
            self._last_ask_sample_ts = now
            self._trim_one_sided_samples(self._ask_samples, now)

        # Update state: current becomes the new quote
        self._cur_bid = bid
        self._cur_ask = ask

        return ofi_event

    # -----------------------------------------------------------------------
    # L2 depth handler — multi-level OFI
    # -----------------------------------------------------------------------

    def on_book(self, snapshot: "OrderBookSnapshot") -> Optional[OFIEvent]:
        """
        Process an L2 depth snapshot (IBKR market depth).

        Computes the multi-level OFI of Cont/Kukanov/Stoikov extended to the
        top M levels: the three-case delta is evaluated per level rank, and
        each level's contribution is weighted by OFI_L2_LEVEL_DECAY^(m-1) so
        the inside quote dominates. The weighted flow feeds the SAME rolling
        OFI window as L1 events, so every downstream consumer (ofi_rolling,
        ofi_rate, chop, intensity) transparently upgrades to deep OFI.

        Returns the OFIEvent produced, or None on the first snapshot
        (no previous book to diff against).
        """
        bids = [(lvl.price, lvl.size) for lvl in snapshot.bids if lvl.price > 0]
        asks = [(lvl.price, lvl.size) for lvl in snapshot.asks if lvl.price > 0]
        now = snapshot.ts
        # Treat a resumed feed (gap beyond staleness) as a fresh book: diffing
        # against a stale ladder would produce one large spurious delta.
        first = (
            self._last_book_ts is None
            or (now - self._last_book_ts).total_seconds() > OFI_DEPTH_STALE_SECS
        )
        self._touch_event_ts(now)
        self._book_event_count += 1
        self._last_book_ts = now

        # Depth-density telemetry (total displayed shares, both sides)
        total_depth = sum(s for _, s in bids) + sum(s for _, s in asks)
        self._depth_samples.append((now, total_depth))
        self._depth_sum += total_depth
        cutoff = now - timedelta(seconds=self._ofi_window)
        while self._depth_samples and self._depth_samples[0][0] < cutoff:
            self._depth_sum -= self._depth_samples[0][1]
            self._depth_samples.popleft()

        ofi_event: Optional[OFIEvent] = None
        if not first:
            delta_bid = self._compute_depth_delta(self._prev_book_bids, bids, side="bid")
            delta_ask = self._compute_depth_delta(self._prev_book_asks, asks, side="ask")
            ofi_event = OFIEvent(
                ts=now,
                delta_bid=delta_bid,
                delta_ask=delta_ask,
                ofi=delta_bid - delta_ask,
            )
            self._ofi_events.append(ofi_event)
            self._ofi_sum += ofi_event.ofi
            self._trim_ofi(now)
        # Every snapshot contributes a depth sample to the EMA kernel (a
        # fresh book with no diffable predecessor still describes standing
        # liquidity); flow is added only when a delta exists.
        self._ofi_ema_add(
            now,
            float(ofi_event.ofi) if ofi_event is not None else 0.0,
            float(total_depth),
        )

        # --- Hidden-liquidity samples from the book's best ask ---
        # While depth flows this is the sample source (real displayed size;
        # L1 sampling is suppressed in on_quote). Decimated to the ~4/s L1
        # quote cadence the detector was tuned on — hidden_seller walks the
        # whole sample window on every print, so undecimated book bursts
        # would blow up that read's cost.
        if asks:
            best_price, best_size = asks[0]
            if best_size > 0 and (
                self._last_ask_sample_ts is None
                or (now - self._last_ask_sample_ts).total_seconds() >= HIDDEN_BOOK_SAMPLE_SECS
            ):
                self._ask_samples.append(L1BookState(price=best_price, size=best_size, ts=now))
                self._last_ask_sample_ts = now
                self._trim_one_sided_samples(self._ask_samples, now)

        # --- Level-anchored ask-wall sample (armed-band arm evidence) ---
        self._sample_wall(now, asks, fresh=first)

        self._prev_book_bids = bids
        self._prev_book_asks = asks
        self._update_armed_latches(now)
        return ofi_event

    def _sample_wall(self, now: datetime, asks: list[tuple[float, int]], fresh: bool) -> None:
        """Fold one decimated ask-mass sample at the setup level.

        The trend read is current-mass vs the EMA baseline BEFORE folding, so
        a collapsing wall reads as evaporation instead of dragging its own
        baseline down with it. A resumed feed (fresh=True) restarts the
        baseline — diffing against a pre-gap wall is meaningless.

        Coverage guard: with a full ladder (IBKR_L2_ROWS rows) whose deepest
        visible ask sits below the wall band's top, part of the band has
        scrolled off the truncated view — the sample is skipped and the read
        goes vacuous rather than reporting false evaporation.
        """
        if self._band_level is None:
            return
        if fresh:
            self._wall_pv = 0.0
            self._wall_w = 0.0
            self._wall_ema_last_ts = None
            self._wall_trend = None
            self._wall_covered = False
        if self._wall_last_sample_ts is not None and (
            (now - self._wall_last_sample_ts).total_seconds() < HIDDEN_BOOK_SAMPLE_SECS
        ):
            return
        self._wall_last_sample_ts = now
        level = self._band_level
        lo = level * (1.0 - SIM_WALL_BAND_PCT)
        hi = level * (1.0 + SIM_WALL_BAND_PCT)
        if not asks:
            self._wall_covered = False
            self._wall_trend = None
            return
        if len(asks) >= IBKR_L2_ROWS and asks[-1][0] < hi:
            self._wall_covered = False
            self._wall_trend = None
            return
        self._wall_covered = True
        mass = float(sum(s for p, s in asks if lo <= p <= hi))
        # Trend vs baseline (needs ~2s of effective history to mean anything)
        baseline = self._wall_pv / self._wall_w if self._wall_w >= 8.0 else None
        if baseline is None or baseline <= 0:
            self._wall_trend = None
        else:
            self._wall_trend = round(1.0 - mass / baseline, 4)
        # Fold (decay-then-add, event-time; negative gaps fold without rewind)
        if self._wall_ema_last_ts is not None:
            dt = (now - self._wall_ema_last_ts).total_seconds()
            if dt > 0:
                k = math.exp(-dt / OFI_EMA_TAU_SECS)
                self._wall_pv *= k
                self._wall_w *= k
            elif dt < 0:
                now = self._wall_ema_last_ts
        self._wall_pv += mass
        self._wall_w += 1.0
        self._wall_ema_last_ts = now

    @staticmethod
    def _compute_depth_delta(
        prev: list[tuple[float, int]],
        cur: list[tuple[float, int]],
        side: str,
    ) -> int:
        """
        Level-weighted net volume change on one side of the book.

        Applies the Cont et al. three-case rule at each level rank m
        (comparing the m-th best price across consecutive snapshots) and
        weights the contribution by OFI_L2_LEVEL_DECAY^(m-1). Improvement
        for the bid side is price UP; for the ask side it is price DOWN.
        """
        depth = max(len(prev), len(cur))
        if depth == 0:
            return 0
        total = 0.0
        weight = 1.0
        for m in range(depth):
            p_prev, s_prev = prev[m] if m < len(prev) else (0.0, 0)
            p_cur, s_cur = cur[m] if m < len(cur) else (0.0, 0)
            if p_prev <= 0 and p_cur <= 0:
                delta = 0
            elif p_prev <= 0:
                delta = s_cur          # level newly populated
            elif p_cur <= 0:
                delta = -s_prev        # level vanished
            elif side == "bid":
                if p_cur > p_prev:
                    delta = s_cur
                elif p_cur == p_prev:
                    delta = s_cur - s_prev
                else:
                    delta = -s_prev
            else:  # ask: signs inverted (price up = consumed)
                if p_cur > p_prev:
                    delta = -s_prev
                elif p_cur == p_prev:
                    delta = s_cur - s_prev
                else:
                    delta = s_cur
            total += weight * delta
            weight *= OFI_L2_LEVEL_DECAY
        return int(round(total))

    @property
    def depth_active(self) -> bool:
        """True when the L2 depth feed is flowing (fresh within OFI_DEPTH_STALE_SECS).

        Staleness is measured against the latest event timestamp seen by this
        tracker (trade/quote/book), never wall clock, so backtests replaying
        historical timestamps behave identically to live.
        """
        if self._last_book_ts is None or self._last_event_ts is None:
            return False
        age = (self._last_event_ts - self._last_book_ts).total_seconds()
        return age <= OFI_DEPTH_STALE_SECS

    @property
    def book_depth_avg(self) -> float:
        """Average total displayed shares (both sides) per depth snapshot in the window."""
        if not self._depth_samples:
            return 0.0
        return self._depth_sum / len(self._depth_samples)

    @property
    def ofi_normalized(self) -> float:
        """Rolling (boxcar) OFI normalised by average displayed depth.

        Expresses net order flow as a fraction of the book's standing
        liquidity, making the magnitude comparable across thick/thin names.
        0.0 until depth samples exist. Remains the logged telemetry quantity
        regardless of OFI_GATE_SOURCE (same convention as trade_speed /
        buy_ratio vs their EMA gate sources).
        """
        avg_depth = self.book_depth_avg
        if avg_depth <= 0:
            return 0.0
        return round(self._ofi_sum / avg_depth, 4)

    def _ofi_ema_add(self, ts: datetime, ofi: float, depth: Optional[float] = None) -> None:
        """Decay the EMA-OFI accumulators to ``ts`` and fold in one sample."""
        if self._ofi_ema_last_ts is not None:
            dt = (ts - self._ofi_ema_last_ts).total_seconds()
            if dt > 0:
                k = math.exp(-dt / OFI_EMA_TAU_SECS)
                self._ofi_ema_sum *= k
                self._ofi_ema_dep_pv *= k
                self._ofi_ema_dep_w *= k
            elif dt < 0:
                # Out-of-order sample (rare): fold in without rewinding decay.
                ts = self._ofi_ema_last_ts
        self._ofi_ema_sum += ofi
        if depth is not None:
            self._ofi_ema_dep_pv += depth
            self._ofi_ema_dep_w += 1.0
        self._ofi_ema_last_ts = ts

    @property
    def ofi_normalized_ema(self) -> float:
        """EMA-kernel normalized OFI: decayed net flow / decayed mean depth.

        Same units and sign convention as ofi_normalized, but with an
        exponential memory (tau = OFI_EMA_TAU_SECS) instead of a boxcar:
        a burst registers immediately and ages out smoothly, so the read
        cannot squat on a 30s-old spike after the flow has died (measured
        vs the boxcar on 30 replayed sessions: stale-ON on dead tape 6.8%
        vs 14.9% at matched noise rejection). Decays to the latest event
        timestamp on read without mutating state (replay-safe). 0.0 until
        depth samples exist.
        """
        if self._ofi_ema_last_ts is None or self._ofi_ema_dep_w <= 0:
            return 0.0
        total = self._ofi_ema_sum
        now = self._last_event_ts
        if now is not None:
            dt = (now - self._ofi_ema_last_ts).total_seconds()
            if dt > 0:
                total *= math.exp(-dt / OFI_EMA_TAU_SECS)
        # The depth mean pv/w is decay-invariant (both scale by the same k).
        avg_depth = self._ofi_ema_dep_pv / self._ofi_ema_dep_w
        if avg_depth <= 0:
            return 0.0
        return round(total / avg_depth, 4)

    def _touch_event_ts(self, ts: datetime) -> None:
        if self._last_event_ts is None or ts > self._last_event_ts:
            self._last_event_ts = ts

    # -----------------------------------------------------------------------
    # T&S handler — trade flow (aggressor) classification
    # -----------------------------------------------------------------------

    def on_timesale(self, event: LiveTimesale) -> ClassifiedTrade:
        """
        Classify a T&S print by aggressor side and add to the rolling window.
        Uses the NBBO embedded in the timesale event for classification.
        """
        self._trade_count += 1
        self._touch_event_ts(event.ts)
        # The merged tape (market_data/tape_merge.py) has already sided the
        # print against the NBBO, allowing for a late quote; anything else is
        # classified here from the bid/ask it carries.
        side = getattr(event, "aggressor", None)
        if side not in ("buy", "sell", "mid"):
            side = self._classify(event.last, event.bid, event.ask)

        ct = ClassifiedTrade(
            ts=event.ts,
            price=event.last,
            size=event.size,
            side=side,
        )

        # --- Trailing-high micro-break trigger (all prints, raw cadence) ---
        self._last_trade_price = event.last
        tp = self._trail_prices
        tm = self._trail_max
        trail_cutoff = event.ts - timedelta(seconds=SIM_TRIGGER_TRAIL_WINDOW_SECS)
        while tp and tp[0][0] < trail_cutoff:
            tp.popleft()
        while tm and tm[0][0] < trail_cutoff:
            tm.popleft()
        if (
            tm
            and event.last > tm[0][1]
            and tp
            and (event.ts - tp[0][0]).total_seconds() >= SIM_TRIGGER_TRAIL_MIN_SPAN_SECS
        ):
            self._microbreak_ts = event.ts
            self._microbreak_price = event.last
        tp.append((event.ts, event.last))
        while tm and tm[-1][1] <= event.last:
            tm.pop()
        tm.append((event.ts, event.last))

        # Update running sums
        if side == "buy":
            self._buy_vol += event.size
            self._buy_prints.append(event.ts)
        elif side == "sell":
            self._sell_vol += event.size
            self._sell_prints.append(event.ts)

        # Baseline tracker: record all directional prints (buy + sell)
        if side != "mid":
            self._baseline_prints.append(event.ts)
            self._trim_baseline(event.ts)
            self._activity_prints.append(event.ts)
            self._trim_activity(event.ts)
            self._trades.append(ct)
            self._ema_update(event.ts, side == "buy")
            # Progress samples: decimate to 1s, keep window + slack. The
            # newest slot is always refreshed so the "now" price is current.
            ps = self._progress_samples
            if ps and (event.ts - ps[-1][0]).total_seconds() < 1.0:
                ps[-1] = (ps[-1][0], event.last)
            else:
                ps.append((event.ts, event.last))
            cutoff = event.ts - timedelta(seconds=PROGRESS_WINDOW_SECS + 5)
            while ps and ps[0][0] < cutoff:
                ps.popleft()

        self._trim_trades(event.ts)
        self._trim_pressure(event.ts)
        self._update_armed_latches(event.ts)

        return ct

    # -----------------------------------------------------------------------
    # Public read-only properties
    # -----------------------------------------------------------------------

    @property
    def book_lean(self) -> float:
        """EMA-smoothed volume imbalance ρ ∈ [-1, +1]. 0 = neutral."""
        return round(self._book_lean_ema, 4)

    @property
    def book_lean_raw(self) -> float:
        """Instantaneous (unsmoothed) volume imbalance."""
        return round(self._book_lean_raw, 4)

    @property
    def ofi_rolling(self) -> int:
        """Cumulative OFI over the rolling window (shares). + = buy pressure.

        Level-weighted deep OFI while the L2 depth feed is active; L1 OFI
        otherwise. Maintained incrementally (O(1) read).
        """
        return self._ofi_sum

    @property
    def ofi_rate(self) -> float:
        """OFI per second over the rolling window. Normalises for variable tick rate."""
        if not self._ofi_events:
            return 0.0
        span = (self._ofi_events[-1].ts - self._ofi_events[0].ts).total_seconds()
        if span <= 0:
            return 0.0
        return self.ofi_rolling / span

    @property
    def trade_imbalance(self) -> float:
        """
        Net trade flow imbalance ∈ [-1, +1].
        (buy_vol - sell_vol) / (buy_vol + sell_vol).
        0 = neutral.  +1 = all aggressive buying.
        """
        total = self._buy_vol + self._sell_vol
        if total == 0:
            return 0.0
        return round((self._buy_vol - self._sell_vol) / total, 4)

    @property
    def buying_pressure(self) -> int:
        """Count of trades at/above the ask in the pressure rolling window."""
        return len(self._buy_prints)

    @property
    def selling_pressure(self) -> int:
        """Count of trades at/below the bid in the pressure rolling window."""
        return len(self._sell_prints)

    @property
    def activity_count(self) -> int:
        """Total directional prints over the (wider) activity window.

        Decoupled from the pressure/signal window: trade_speed, buy_ratio and
        pressure_acceleration read the short PRESSURE_WINDOW_SECS window, but
        the "enough recent tape to trust the read" floor counts prints over
        the wider PRESSURE_ACTIVITY_WINDOW_SECS so a short signal window
        cannot make the floor unsatisfiable on low-rate tape.

        O(1): the activity-window suffix is maintained incrementally on
        ingestion (append + amortized popleft per directional print) instead
        of re-walking the window on every read, which is quadratic on burst
        tape and this is read once per print in the live hot path.
        """
        return len(self._activity_prints)

    @property
    def buy_ratio(self) -> float:
        """Fraction of prints that are buy-initiated in the pressure window. [0, 1]."""
        total = len(self._buy_prints) + len(self._sell_prints)
        if total == 0:
            return 0.5
        return len(self._buy_prints) / total

    @property
    def trade_speed(self) -> float:
        """Total prints per second in the pressure window."""
        total = len(self._buy_prints) + len(self._sell_prints)
        if total == 0:
            return 0.0
        return total / self._pressure_window

    @property
    def baseline_speed(self) -> float:
        """Prints per second in the baseline window (excludes signal window).

        The baseline window is [now - (baseline + signal), now - signal].
        This keeps the current signal window separate from the baseline.
        Returns 0.0 if fewer than TAPE_BASELINE_MIN_PRINTS in the window.
        """
        if not self._baseline_prints:
            return 0.0
        now = self._baseline_prints[-1]
        signal_cutoff = now - timedelta(seconds=self._pressure_window)
        count = 0
        for ts in self._baseline_prints:
            if ts < signal_cutoff:
                count += 1
        if count < TAPE_BASELINE_MIN_PRINTS:
            return 0.0
        return count / self._baseline_window

    @property
    def speed_ratio(self) -> float:
        """Signal speed / baseline speed for telemetry.

        Returns 0.0 when baseline is not yet established (warm-up period).
        A value of 2.0 means current activity is 2× the recent baseline.
        """
        base = self.baseline_speed
        if base <= 0:
            return 0.0
        return self.trade_speed / base

    def _pressure_acceleration_halves(self) -> tuple[int, int, int, int]:
        buys = self._buy_prints
        sells = self._sell_prints
        if not buys and not sells:
            cache_key = (None, None, None, 0, 0)
            if self._pressure_accel_cache_key == cache_key:
                return self._pressure_accel_cache
            self._pressure_accel_cache = (0, 0, 0, 0)
            self._pressure_accel_cache_key = cache_key
            return self._pressure_accel_cache

        latest = buys[-1] if buys else sells[-1]
        if sells and sells[-1] > latest:
            latest = sells[-1]
        cache_key = (
            latest,
            buys[0] if buys else None,
            sells[0] if sells else None,
            len(buys),
            len(sells),
        )
        if self._pressure_accel_cache_key == cache_key:
            return self._pressure_accel_cache

        midpoint = latest - timedelta(seconds=self._pressure_window / 2)
        recent_buys = sum(1 for ts in buys if ts >= midpoint)
        older_buys = len(buys) - recent_buys
        recent_sells = sum(1 for ts in sells if ts >= midpoint)
        older_sells = len(sells) - recent_sells
        self._pressure_accel_cache = (recent_buys, recent_sells, older_buys, older_sells)
        self._pressure_accel_cache_key = cache_key
        return self._pressure_accel_cache

    @property
    def pressure_acceleration_ready(self) -> bool:
        """True when both halves of the pressure window have enough prints."""
        recent_buys, recent_sells, older_buys, older_sells = self._pressure_acceleration_halves()
        return (recent_buys + recent_sells) >= 5 and (older_buys + older_sells) >= 5

    @property
    def pressure_acceleration(self) -> float:
        """
        Rate of change of total tape activity, bounded to [-1, 1]:
        (recent_total - older_total) / (recent_total + older_total).
        Positive = tape speeding up. Negative = tape slowing down.
        Returns 0.0 when pressure_acceleration_ready is False.
        """
        recent_buys, recent_sells, older_buys, older_sells = self._pressure_acceleration_halves()
        recent_total = recent_buys + recent_sells
        older_total = older_buys + older_sells
        if recent_total < 5 or older_total < 5:
            return 0.0

        return (recent_total - older_total) / (recent_total + older_total)

    @property
    def pressure_acceleration_z(self) -> float:
        """
        Significance of the acceleration: accel * sqrt(N), where N is the total
        directional prints in the pressure window (buy + sell). Under a
        constant-rate tape the raw acceleration has noise std ~ 1/sqrt(N), so
        this is an approximate z-score: how many sigma the tape is accelerating
        above random jitter. Returns 0.0 when pressure_acceleration_ready is False.
        """
        accel = self.pressure_acceleration
        n = len(self._buy_prints) + len(self._sell_prints)
        return accel * (n ** 0.5)

    # -----------------------------------------------------------------------
    # EMA tape intensity — dual exponential kernel acceleration estimator
    # -----------------------------------------------------------------------
    # Each directional print contributes an exponentially-decaying weight to
    # a fast and a slow intensity estimate (rate = decayed count / tau).
    # accel_ema = (r_fast - r_slow) / (r_fast + r_slow)   ∈ (-1, 1)
    # Significance uses the Kish effective sample size of the slow kernel:
    # n_eff = c_slow² / c_slow_sq,  z = accel * sqrt(n_eff).
    # Per-print O(1) updates; reads decay to the latest event timestamp
    # WITHOUT mutating state (replay-safe, idempotent).

    def _ema_update(self, ts: datetime, is_buy: bool = False) -> None:
        """Decay accumulators to ``ts`` and add one directional print."""
        if self._ema_last_ts is not None:
            dt = (ts - self._ema_last_ts).total_seconds()
            gap = 0.0
            if dt > 0:
                decay_fast = math.exp(-dt / TAPE_EMA_FAST_TAU_SECS)
                self._ema_c_fast *= decay_fast
                self._ema_b_fast *= decay_fast
                decay_slow = math.exp(-dt / TAPE_EMA_SLOW_TAU_SECS)
                self._ema_c_slow *= decay_slow
                self._ema_c_slow_sq *= decay_slow * decay_slow
                # Dispersion moments decay at the same slow tau.
                self._phi_w *= decay_slow
                self._phi_dt *= decay_slow
                self._phi_dt_sq *= decay_slow
                gap = dt
            elif dt < 0:
                # Out-of-order print (rare): fold it in without rewinding decay.
                ts = self._ema_last_ts
            # One observed inter-arrival gap (clamped >= 0) per directional print.
            self._phi_w += 1.0
            self._phi_dt += gap
            self._phi_dt_sq += gap * gap
            if self._phi_w >= TAPE_PHI_MIN_WEIGHT:
                mu = self._phi_dt / self._phi_w
                if mu > 1e-9:
                    var = max(self._phi_dt_sq / self._phi_w - mu * mu, 0.0)
                    self._phi_cv2 = var / (mu * mu)
                else:
                    self._phi_cv2 = math.inf  # all-zero gaps: fully batched tape
        self._ema_c_fast += 1.0
        if is_buy:
            self._ema_b_fast += 1.0
        self._ema_c_slow += 1.0
        self._ema_c_slow_sq += 1.0
        self._ema_last_ts = ts

    def _ema_decayed(self) -> tuple[float, float, float]:
        """Accumulators decayed to the latest event ts (read-only, no mutation)."""
        if self._ema_last_ts is None:
            return 0.0, 0.0, 0.0
        c_fast, c_slow, c_slow_sq = self._ema_c_fast, self._ema_c_slow, self._ema_c_slow_sq
        now = self._last_event_ts
        if now is not None:
            dt = (now - self._ema_last_ts).total_seconds()
            if dt > 0:
                c_fast *= math.exp(-dt / TAPE_EMA_FAST_TAU_SECS)
                decay_slow = math.exp(-dt / TAPE_EMA_SLOW_TAU_SECS)
                c_slow *= decay_slow
                c_slow_sq *= decay_slow * decay_slow
        return c_fast, c_slow, c_slow_sq

    def reseed_ema_from_prints(self, now: Optional[datetime] = None) -> None:
        """Rebuild the EMA intensity accumulators from the print deques.

        For reseed paths (backtest cache snapshots, test fixtures) that
        construct ``_buy_prints``/``_sell_prints``/``_baseline_prints``
        directly instead of feeding ``on_timesale``.  Replays every known
        directional print timestamp through the decay recurrence so the
        accumulators hold exactly what live ingestion would have produced.
        Uses ``_baseline_prints`` when populated (it retains the longest
        directional history), else the pressure deques. Timestamps are
        sorted before the walk — reseed callers (tests, cache hydrate) may
        construct deques out of chronological order, and the decay
        recurrence silently skips negative gaps.
        """
        prints = (
            sorted(self._baseline_prints)
            if self._baseline_prints
            else sorted([*self._buy_prints, *self._sell_prints])
        )
        self._ema_c_fast = 0.0
        self._ema_c_slow = 0.0
        self._ema_c_slow_sq = 0.0
        self._ema_b_fast = 0.0
        self._ema_last_ts = None
        self._phi_w = 0.0
        self._phi_dt = 0.0
        self._phi_dt_sq = 0.0
        self._phi_cv2 = 1.0
        # Armed-band print-driven state: the reseed deques carry timestamps
        # only (no prices), so the trailing-high ratchet cannot be rebuilt —
        # start it clean along with the latches/edges (they re-stamp within
        # one TTL of live ingestion).
        self._trail_prices.clear()
        self._trail_max.clear()
        self._last_trade_price = None
        self._reset_armed_state()
        for ts in prints:
            self._ema_update(ts)
        # Rebuild the buy-side fast kernel from the pressure deque — side
        # information lives only there. Its 10s span fully covers the fast
        # kernel's ~2s memory (weight of a 10s-old print is e^-5 ≈ 0.7%).
        b_fast = 0.0
        prev_b: Optional[datetime] = None
        for ts in sorted(self._buy_prints):
            if prev_b is not None:
                dt_b = (ts - prev_b).total_seconds()
                if dt_b > 0:
                    b_fast *= math.exp(-dt_b / TAPE_EMA_FAST_TAU_SECS)
            b_fast += 1.0
            prev_b = ts
        if prev_b is not None and self._ema_last_ts is not None:
            dt_b = (self._ema_last_ts - prev_b).total_seconds()
            if dt_b > 0:
                b_fast *= math.exp(-dt_b / TAPE_EMA_FAST_TAU_SECS)
        self._ema_b_fast = b_fast
        # Rebuild the activity-window suffix for the same reseed paths
        # (direct deque construction bypasses on_timesale's maintenance).
        self._activity_prints.clear()
        if prints:
            cutoff = prints[-1] - timedelta(seconds=self._activity_window)
            self._activity_prints.extend(ts for ts in prints if ts >= cutoff)
        if now is not None:
            self._touch_event_ts(now)

    @property
    def tape_speed_ema(self) -> float:
        """Instantaneous directional prints/sec from the fast exponential kernel."""
        c_fast, _, _ = self._ema_decayed()
        return c_fast / TAPE_EMA_FAST_TAU_SECS

    @property
    def buy_ratio_ema(self) -> float:
        """Buy share of the fast exponential kernel. [0, 1]; 0.5 when empty.

        b_fast and c_fast decay by the same factor, so their ratio needs no
        decay-to-read. Returns the neutral 0.5 (same convention as the empty
        boxcar window) when the decayed kernel holds less than half an
        effective print of evidence.
        """
        c_fast, _, _ = self._ema_decayed()
        if c_fast < 0.5:
            return 0.5
        return self._ema_b_fast / self._ema_c_fast

    @property
    def pressure_acceleration_ema_n_eff(self) -> float:
        """Kish effective sample size of the slow kernel."""
        _, c_slow, c_slow_sq = self._ema_decayed()
        if c_slow_sq <= 0:
            return 0.0
        return (c_slow * c_slow) / c_slow_sq

    @property
    def pressure_acceleration_ema_ready(self) -> bool:
        """True once the slow kernel holds enough effective prints to trust."""
        return self.pressure_acceleration_ema_n_eff >= TAPE_ACCEL_EMA_MIN_NEFF

    @property
    def pressure_acceleration_ema(self) -> float:
        """Bounded intensity ratio (r_fast - r_slow)/(r_fast + r_slow) ∈ (-1, 1).

        Positive = tape arriving faster than its recent norm (igniting);
        negative = decaying. 0.0 until pressure_acceleration_ema_ready.
        """
        if not self.pressure_acceleration_ema_ready:
            return 0.0
        c_fast, c_slow, _ = self._ema_decayed()
        r_fast = c_fast / TAPE_EMA_FAST_TAU_SECS
        r_slow = c_slow / TAPE_EMA_SLOW_TAU_SECS
        total = r_fast + r_slow
        if total <= 0:
            return 0.0
        return (r_fast - r_slow) / total

    @property
    def tape_dispersion_phi(self) -> float:
        """Index-of-dispersion estimate for directional print arrivals.

        phi-hat = clip(decayed CV^2 of inter-arrival gaps, 1, TAPE_PHI_CLIP_HI).
        Kish n_eff assumes independent arrivals; batched/clustered tape
        (many prints per timestamp, burst arrivals) inflates n_eff and hence
        z by ~sqrt(phi). 1.0 (no correction) until the gap mass first warms;
        afterwards quiet spells carry the last trusted estimate.
        """
        return min(max(self._phi_cv2, 1.0), TAPE_PHI_CLIP_HI)

    @property
    def pressure_acceleration_ema_z(self) -> float:
        """Dispersion-corrected significance: accel * sqrt(n_eff / phi-hat)."""
        accel = self.pressure_acceleration_ema
        if accel == 0.0:
            return 0.0
        return accel * math.sqrt(
            self.pressure_acceleration_ema_n_eff / self.tape_dispersion_phi
        )

    # -----------------------------------------------------------------------
    # Acceleration gate — single source of truth for all consumers
    # -----------------------------------------------------------------------
    # Entry/exit gates, execution criteria, readiness scoring and logging all
    # read these three properties, switched by config TAPE_ACCEL_ESTIMATOR,
    # so the estimator can be swapped without touching any consumer.

    @property
    def accel_gate_ready(self) -> bool:
        """Selected estimator has enough data to be trusted."""
        if TAPE_ACCEL_ESTIMATOR == "ema":
            return self.pressure_acceleration_ema_ready
        return self.pressure_acceleration_ready

    @property
    def accel_gate_raw(self) -> float:
        """Bounded raw acceleration from the selected estimator ∈ (-1, 1)."""
        if TAPE_ACCEL_ESTIMATOR == "ema":
            return self.pressure_acceleration_ema
        return self.pressure_acceleration

    @property
    def accel_gate_z(self) -> float:
        """Significance (z) of the acceleration from the selected estimator."""
        if TAPE_ACCEL_ESTIMATOR == "ema":
            return self.pressure_acceleration_ema_z
        return self.pressure_acceleration_z

    @property
    def speed_gate(self) -> float:
        """Prints/sec read used by the entry speed floor (single source).

        "ema" (default): tape_speed_ema — instantaneous directional rate from
        the fast kernel (tau=2s). Same units and steady-state value as the
        boxcar, but arms at surge onset in ~0.4s median vs 2.3s (the boxcar
        never armed during 57% of replayed surges — they end before its 10s
        window fills) and halves stale passes on dead tape.
        "boxcar": legacy trade_speed (count / fixed 10s window), which remains
        the logged telemetry quantity regardless of this switch.
        """
        if TAPE_SPEED_GATE_SOURCE == "ema":
            return self.tape_speed_ema
        return self.trade_speed

    @property
    def ratio_gate(self) -> float:
        """Buy-ratio read used by the entry imbalance gate and the
        accel-collapse exit band (single source — enter >= 0.60, collapse
        exit <= 0.50 on the same estimator, so entry and exit views cannot
        contradict).

        "ema" (default): buy_ratio_ema — fast-kernel (tau=2s) count ratio.
        Measured vs the 10s boxcar at the same 0.60 floor: half the noise
        passes (gate ON on locally balanced tape), a third of the flip-stale
        passes, and +12% surge coverage at near-equal strictness. A
        Wilson-LB significance gate was also measured and rejected — phi
        spikes during bursts (clustering IS the surge signature) crush its
        n_eff exactly at onset.
        "boxcar": legacy buy_ratio (counts / fixed 10s window), which remains
        the logged telemetry quantity regardless of this switch.
        """
        if TAPE_RATIO_GATE_SOURCE == "ema":
            return self.buy_ratio_ema
        return self.buy_ratio

    @property
    def price_progress(self) -> Optional[float]:
        """Net trade-price progress over the trailing PROGRESS_WINDOW_SECS.

        (last directional print) / (print ~window ago) - 1. The only tape
        read that measures whether PRICE responded to the flow the other
        gates measure — flow without progress is absorption/churn, the
        measured failure shape of climax-day entries.

        Returns None (not ready) when directional-tape history spans less
        than PROGRESS_MIN_SPAN_SECS — a seconds-old awakening carries no
        progress information, and the entry gate deliberately fails closed
        on it rather than chase a cold burst. When history covers at least
        the minimum span but not the full window, the oldest sample serves
        as the reference (partial window, conservative).
        """
        ps = self._progress_samples
        if len(ps) < 2:
            return None
        now = self._last_event_ts or ps[-1][0]
        cutoff = now - timedelta(seconds=PROGRESS_WINDOW_SECS)
        ref: Optional[float] = None
        for ts, price in ps:  # oldest -> newest; decimated, <= ~window slots
            if ts <= cutoff:
                ref = price
            else:
                break
        if ref is None:
            oldest_ts, oldest_price = ps[0]
            if (now - oldest_ts).total_seconds() < PROGRESS_MIN_SPAN_SECS:
                return None
            ref = oldest_price
        if ref <= 0:
            return None
        return round(ps[-1][1] / ref - 1.0, 5)

    @property
    def ofi_norm_gate(self) -> float:
        """Depth-normalized OFI read used by the entry floor and the
        adverse-flow exit (single source — both sides of the position read
        the same estimator, so entry and exit views cannot contradict).

        "ema" (default): ofi_normalized_ema — exponential-kernel flow at
        tau=OFI_EMA_TAU_SECS. Measured vs the 30s boxcar over 30 replayed
        L2 sessions (772 squeeze legs): +14 legs coverage at matched noise
        episodes, added-delay p90 36.6s vs 44.3s, and stale-ON on dead
        tape halved — the boxcar's failure mode is a 30s-old flow spike
        holding the gate open after the tape dies.
        "boxcar": legacy ofi_normalized (30s window), which remains the
        logged telemetry quantity regardless of this switch.
        """
        if OFI_GATE_SOURCE == "ema":
            return self.ofi_normalized_ema
        return self.ofi_normalized

    # -----------------------------------------------------------------------
    # Gate predicates — the ONLY place a gate value meets its config floor
    # -----------------------------------------------------------------------
    # exec.signals.tape_entry_gates (F10 single decision surface) and the
    # tracker-internal latch stamps (armed-band entry) both read these, so
    # the latch condition can never drift from the live gate condition.
    # Floors come from config verbatim; re-timing WHEN a gate is evaluated
    # is allowed, changing its floor here is not (2026-08-05 directive).

    def speed_gate_ok(self) -> bool:
        """Tape active enough: speed_gate >= SIM_PRESSURE_SPEED_MIN."""
        return self.speed_gate >= SIM_PRESSURE_SPEED_MIN

    def activity_ok(self) -> bool:
        """Enough prints in the activity window for ratio/accel to mean anything."""
        return self.activity_count >= SIM_PRESSURE_ACTIVITY_MIN

    def ratio_gate_ok(self) -> bool:
        """Buyers dominate: ratio_gate >= SIM_PRESSURE_BUY_RATIO_MIN."""
        return self.ratio_gate >= SIM_PRESSURE_BUY_RATIO_MIN

    def vol_flow_ok(self) -> bool:
        """30s dollar flow not sell-dominated: trade_imbalance > floor."""
        return self.trade_imbalance > SIM_PRESSURE_VOL_IMBALANCE_MIN

    def accel_gate_ok(self) -> bool:
        """Buying accelerating and statistically significant."""
        return self.accel_gate_ready and self.accel_gate_z >= SIM_PRESSURE_ACCEL_Z

    def progress_gate_ok(self) -> bool:
        """Price made net trailing progress (fails closed on thin history)."""
        progress = self.price_progress
        return progress is not None and progress > SIM_PRESSURE_PROGRESS_MIN

    def ofi_gate_ok(self) -> bool:
        """Depth-normalized OFI floor; vacuous pass while no live depth feed."""
        if not self.depth_active:
            return True
        return self.ofi_norm_gate >= SIM_OFI_NORM_MIN

    # -----------------------------------------------------------------------
    # Armed-band: entry band, evidence latches, trigger primitives
    # -----------------------------------------------------------------------

    def set_entry_band(
        self,
        level: Optional[float],
        floor: Optional[float],
        pullback_ts: Optional[datetime],
    ) -> None:
        """Anchor the tracker to the active setup's entry band.

        Idempotent per event — the scorer pushes before every ingestion.
        A CHANGED (level, pullback_ts) identity means a different setup
        instance: all level-anchored state (latches, edges, wall trend)
        resets so evidence can never carry across setups. ``None`` clears.
        The trailing-high price history is market fact, not level-anchored,
        and survives band changes.
        """
        if level is None or level <= 0:
            if self._band_level is not None:
                self._reset_armed_state()
            self._band_level = None
            self._band_floor = None
            self._band_top = None
            self._band_pullback_ts = None
            return
        changed = (
            self._band_level != float(level)
            or self._band_pullback_ts != pullback_ts
        )
        self._band_level = float(level)
        self._band_floor = float(floor) if floor else float(level)
        self._band_top = float(level) * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT)
        self._band_pullback_ts = pullback_ts
        if changed:
            self._reset_armed_state()

    def _reset_armed_state(self) -> None:
        self._gate_last_true.clear()
        self._accel_prev_ok = False
        self._accel_edge_ts = None
        self._microbreak_ts = None
        self._microbreak_price = None
        self._last_latch_eval_ts = None
        # Wall state is anchored to the (now different) level
        self._wall_pv = 0.0
        self._wall_w = 0.0
        self._wall_ema_last_ts = None
        self._wall_last_sample_ts = None
        self._wall_trend = None
        self._wall_covered = False

    def _update_armed_latches(self, now: datetime) -> None:
        """Stamp per-gate last-true times while price sits in the armed band.

        Called at the end of every trade/book ingestion. A dip UNDER the
        band floor invalidates the armed state (stale-latch guard: evidence
        gathered before a floor break must not fire a later trigger); above
        the band top the extended zone applies the legacy instantaneous
        stack, so nothing stamps. The predicate sweep is decimated to
        _LATCH_EVAL_MIN_GAP_SECS — TTLs are seconds, EMA reads cost exp()s.
        """
        if self._band_level is None or self._last_trade_price is None:
            return
        price = self._last_trade_price
        if self._band_floor is not None and price < self._band_floor:
            if self._gate_last_true or self._accel_edge_ts or self._microbreak_ts:
                self._gate_last_true.clear()
                self._accel_edge_ts = None
                self._microbreak_ts = None
                self._microbreak_price = None
            self._accel_prev_ok = False
            self._last_latch_eval_ts = now
            return
        if self._band_top is not None and price > self._band_top:
            return
        if (
            self._last_latch_eval_ts is not None
            and (now - self._last_latch_eval_ts).total_seconds() < _LATCH_EVAL_MIN_GAP_SECS
        ):
            return
        self._last_latch_eval_ts = now
        stamp = self._gate_last_true
        if self.speed_gate_ok():
            stamp["speed"] = now
        if self.activity_ok():
            stamp["activity"] = now
        if self.ratio_gate_ok():
            stamp["ratio"] = now
        if self.vol_flow_ok():
            stamp["vol_flow"] = now
        if self.progress_gate_ok():
            stamp["progress"] = now
        # OFI stamps only on real depth evidence; the no-depth vacuous pass
        # is applied by the consumer (depth-inactive == not applicable).
        if self.depth_active and self.ofi_gate_ok():
            stamp["ofi"] = now
        accel_ok = self.accel_gate_ok()
        if accel_ok and not self._accel_prev_ok:
            self._accel_edge_ts = now
        self._accel_prev_ok = accel_ok

    @property
    def entry_band(self) -> Optional[tuple[float, float, float]]:
        """(floor, level, top) of the armed band, or None when no setup."""
        if self._band_level is None:
            return None
        return (self._band_floor, self._band_level, self._band_top)

    def gate_last_true_age(self, gate: str) -> Optional[float]:
        """Seconds since ``gate`` last evaluated True in-band; None = never."""
        ts = self._gate_last_true.get(gate)
        if ts is None or self._last_event_ts is None:
            return None
        return max(0.0, (self._last_event_ts - ts).total_seconds())

    def accel_edge_age(self) -> Optional[float]:
        """Seconds since the accel gate's last False→True edge in-band."""
        if self._accel_edge_ts is None or self._last_event_ts is None:
            return None
        return max(0.0, (self._last_event_ts - self._accel_edge_ts).total_seconds())

    def microbreak_age(self) -> Optional[float]:
        """Seconds since a print last broke the trailing high; None = never."""
        if self._microbreak_ts is None or self._last_event_ts is None:
            return None
        return max(0.0, (self._last_event_ts - self._microbreak_ts).total_seconds())

    @property
    def trailing_high(self) -> Optional[float]:
        """Highest print in the trailing trigger window (display/telemetry)."""
        return self._trail_max[0][1] if self._trail_max else None

    @property
    def wall_trend(self) -> Optional[float]:
        """Evaporation of ask mass at the setup level vs its EMA baseline.

        1 - current/baseline: positive = supply leaving (evaporation/
        consumption — both bullish), negative = replenishment. None while
        unavailable (no band, no depth, thin history, truncated coverage).
        """
        return self._wall_trend

    def wall_evidence_ok(self) -> bool:
        """Ask-wall arm evidence — ARM EVIDENCE ONLY, never a hard gate.

        Vacuous True whenever the read cannot exist (no depth feed, no band,
        insufficient baseline, coverage-guarded truncated ladder); a real
        read passes on evaporation >= SIM_WALL_EVAP_MIN. Consulted only
        when SIM_ARM_STRONG_REQUIRES_WALL is enabled.
        """
        if not self.depth_active or self._band_level is None:
            return True
        if not self._wall_covered or self._wall_trend is None:
            return True
        return self._wall_trend >= SIM_WALL_EVAP_MIN

    @property
    def buy_volume(self) -> int:
        """Aggressive buy volume in the current rolling window."""
        return self._buy_vol

    @property
    def sell_volume(self) -> int:
        """Aggressive sell volume in the current rolling window."""
        return self._sell_vol

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def quote_count(self) -> int:
        return self._quote_count

    @property
    def ofi_chop_ratio(self) -> float:
        """
        Fraction of consecutive OFI samples with a sign change.
        Range [0, 1].  High values = directional oscillation = chop.
        Requires ≥ 3 events; returns 0.0 otherwise.
        """
        events = self._ofi_events
        if len(events) < 3:
            return 0.0
        changes = 0
        prev_sign: Optional[bool] = None
        for e in events:
            if e.ofi == 0:
                continue
            cur_sign = e.ofi > 0
            if prev_sign is not None and cur_sign != prev_sign:
                changes += 1
            prev_sign = cur_sign
        denom = len(events) - 1
        return changes / denom if denom > 0 else 0.0

    @property
    def ofi_intensity(self) -> float:
        """
        Average |OFI| per event in the rolling window (shares).
        Low intensity = exhaustion (nobody is aggressing either side).
        """
        if not self._ofi_events:
            return 0.0
        return sum(abs(e.ofi) for e in self._ofi_events) / len(self._ofi_events)

    # -----------------------------------------------------------------------
    # Raw-state reads — for the `tape` / `ofi` REPL displays
    # -----------------------------------------------------------------------
    # Every one of these returns state the tracker already holds; none of them
    # trigger a fetch or a subscription.  Copies are returned so a display
    # caller cannot mutate tracker state.

    def recent_prints(self, limit: int) -> list[ClassifiedTrade]:
        """The newest ``limit`` classified prints, newest first.

        Only directional prints are retained — ``on_timesale`` drops
        mid/neutral ones — so this is the tape exactly as every tape var
        counts it.  Bounded by the rolling trade window, so a dead tape
        returns fewer than ``limit`` rows rather than stale ones.
        """
        if limit <= 0:
            return []
        return list(islice(reversed(self._trades), limit))

    @property
    def retained_prints(self) -> int:
        """Directional prints currently held in the rolling trade window."""
        return len(self._trades)

    @property
    def trade_window_secs(self) -> int:
        """Span of the rolling trade window that :meth:`recent_prints` draws from."""
        return self._trade_window

    @property
    def book_ladder(self) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
        """Last L2 ladder fed to :meth:`on_book` as ``(bids, asks)``, best first.

        Empty until a depth snapshot arrives; :attr:`depth_active` says
        whether what's here is still fresh.
        """
        return list(self._prev_book_bids), list(self._prev_book_asks)

    @property
    def l1_quote(self) -> tuple[Optional[L1BookState], Optional[L1BookState]]:
        """Current best ``(bid, ask)`` from the last L1 quote, or ``(None, None)``.

        The states are frozen dataclasses, so this needs no defensive copy.
        """
        return self._cur_bid, self._cur_ask

    @property
    def last_book_ts(self) -> Optional[datetime]:
        """Timestamp of the last depth snapshot; ``None`` if none ever arrived."""
        return self._last_book_ts

    @property
    def last_event_ts(self) -> Optional[datetime]:
        """Latest trade/quote/book timestamp — the tracker's notion of "now"."""
        return self._last_event_ts

    @property
    def ofi_sample_count(self) -> int:
        """OFI samples inside the rolling window (L2 book events, or L1 quotes)."""
        return len(self._ofi_events)

    @property
    def ofi_window_secs(self) -> int:
        """Span of the rolling OFI window."""
        return self._ofi_window

    # -----------------------------------------------------------------------
    # Hidden liquidity (iceberg / refresh) detection
    # -----------------------------------------------------------------------

    @property
    def hidden_seller(self) -> float:
        """
        Hidden seller intensity ∈ [0, 1].

        Detects iceberg supply at the ask: aggressive buy volume keeps
        hitting the ask, but the ask price is NOT advancing — meaning
        a hidden seller keeps refilling the displayed offer.

        intensity = clip( (buy_vol_in_window / avg_ask_size_in_window - 1) / 4, 0, 1)
        gated to 0 when the ask price has advanced during the window.
        Returns 0.0 when there isn't enough data.

        Ask samples come from real L1 quotes while no depth feed flows and
        from the book's best ask (decimated) while depth is active;
        synthetic replay quotes never feed them, so replays without book
        capture read 0.0 (vacuous pass) instead of tripping on placeholder
        1-share sizes.

        (A mirror-image hidden_buyer read existed but had no consumers and
        cost a bid-side sample deque maintained on every quote — removed
        in the F10 cleanup.)
        """
        samples = self._ask_samples
        if len(samples) < HIDDEN_MIN_SAMPLES:
            return 0.0

        oldest = samples[0]
        newest = samples[-1]
        if (newest.ts - oldest.ts).total_seconds() < HIDDEN_MIN_HALF_WINDOW_SECS:
            return 0.0

        cutoff = newest.ts - timedelta(seconds=HIDDEN_WINDOW_SECS)
        in_window = [s for s in samples if s.ts >= cutoff]
        if len(in_window) < HIDDEN_MIN_SAMPLES:
            return 0.0

        prices = [s.price for s in in_window]
        avg_size = sum(s.size for s in in_window) / len(in_window)
        if avg_size <= 0:
            return 0.0

        # Gate: ask must NOT have advanced during the window
        if max(prices) > prices[0] + HIDDEN_PRICE_TOL:
            return 0.0
        aggressor_vol = sum(
            t.size for t in self._trades
            if t.side == "buy" and t.ts >= cutoff
        )

        if aggressor_vol <= 0:
            return 0.0

        ratio = aggressor_vol / avg_size
        span = max(HIDDEN_RATIO_CEILING - HIDDEN_RATIO_FLOOR, 1e-9)
        intensity = max(0.0, min(1.0, (ratio - HIDDEN_RATIO_FLOOR) / span))
        return round(intensity, 4)

    @staticmethod
    def _trim_one_sided_samples(
        samples: deque,
        now: datetime,
        keep_secs: int = HIDDEN_KEEP_SECS,
    ) -> None:
        """Trim displayed-size samples older than ``keep_secs``."""
        cutoff = now - timedelta(seconds=keep_secs)
        while samples and samples[0].ts < cutoff:
            samples.popleft()

    # -----------------------------------------------------------------------
    # T&S density / print-size metrics  (telemetry — no scoring vars yet)
    # -----------------------------------------------------------------------

    @property
    def avg_print_size(self) -> float:
        """Average trade size (shares) in the rolling window.
        Small = retail noise; large = institutional interest."""
        if not self._trades:
            return 0.0
        return sum(t.size for t in self._trades) / len(self._trades)

    @property
    def tape_prints_per_sec(self) -> float:
        """Total T&S prints per second in the rolling window."""
        return self._density_pps(None)

    @property
    def buy_prints_per_sec(self) -> float:
        """Buy-side T&S prints per second."""
        return self._density_pps("buy")

    @property
    def sell_prints_per_sec(self) -> float:
        """Sell-side T&S prints per second."""
        return self._density_pps("sell")

    @property
    def density_imbalance(self) -> float:
        """(buy_pps - sell_pps) / (buy_pps + sell_pps).  Count-based, not volume-based.
        Captures whether buyers are *more frequent* not just *bigger*."""
        bpps = self.buy_prints_per_sec
        spps = self.sell_prints_per_sec
        total = bpps + spps
        if total == 0:
            return 0.0
        return round((bpps - spps) / total, 4)

    def _density_pps(self, side: Optional[str]) -> float:
        """Prints/sec for a given side (None = all sides)."""
        if len(self._trades) < 2:
            return 0.0
        span = (self._trades[-1].ts - self._trades[0].ts).total_seconds()
        if span <= 0:
            return 0.0
        if side is None:
            return round(len(self._trades) / span, 2)
        count = sum(1 for t in self._trades if t.side == side)
        return round(count / span, 2)

    # -----------------------------------------------------------------------
    # OFI delta computation (Cont et al. 2014, equations 2-3)
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_delta_bid(prev: L1BookState, cur: L1BookState) -> int:
        """
        ΔV^b: net volume change at the best bid.

        Three cases:
          bid_price went UP    → all current volume is new      → +V_cur
          bid_price unchanged  → delta is the size difference   → V_cur - V_prev
          bid_price went DOWN  → all previous volume was filled → -V_prev
        """
        if cur.price > prev.price:
            return cur.size
        if cur.price == prev.price:
            return cur.size - prev.size
        return -prev.size

    @staticmethod
    def _compute_delta_ask(prev: L1BookState, cur: L1BookState) -> int:
        """
        ΔV^a: net volume change at the best ask.

        Signs are inverted vs bid (ask going UP means previous was consumed):
          ask_price went UP    → previous volume was filled → -V_prev
          ask_price unchanged  → delta is the size difference → V_cur - V_prev
          ask_price went DOWN  → all current volume is new  → +V_cur
        """
        if cur.price > prev.price:
            return -prev.size
        if cur.price == prev.price:
            return cur.size - prev.size
        return cur.size

    # -----------------------------------------------------------------------
    # Aggressor classification
    # -----------------------------------------------------------------------

    @staticmethod
    def _classify(price: float, bid: float, ask: float) -> str:
        """
        Classify a trade by comparing execution price to NBBO.

          price >= ask  → buyer lifted the offer   → "buy"
          price <= bid  → seller hit the bid        → "sell"
          otherwise     → crossed mid / indeterminate → "mid"
        """
        if ask > 0 and price >= ask:
            return "buy"
        if bid > 0 and price <= bid:
            return "sell"
        return "mid"

    # -----------------------------------------------------------------------
    # Window trimming
    # -----------------------------------------------------------------------

    def _trim_ofi(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._ofi_window)
        while self._ofi_events and self._ofi_events[0].ts < cutoff:
            self._ofi_sum -= self._ofi_events[0].ofi
            self._ofi_events.popleft()

    def _trim_trades(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._trade_window)
        while self._trades and self._trades[0].ts < cutoff:
            old = self._trades.popleft()
            if old.side == "buy":
                self._buy_vol -= old.size
            elif old.side == "sell":
                self._sell_vol -= old.size

    def _trim_pressure(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._pressure_window)
        while self._buy_prints and self._buy_prints[0] < cutoff:
            self._buy_prints.popleft()
        while self._sell_prints and self._sell_prints[0] < cutoff:
            self._sell_prints.popleft()

    def _trim_baseline(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._baseline_total_window)
        while self._baseline_prints and self._baseline_prints[0] < cutoff:
            self._baseline_prints.popleft()

    def _trim_activity(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self._activity_window)
        while self._activity_prints and self._activity_prints[0] < cutoff:
            self._activity_prints.popleft()
