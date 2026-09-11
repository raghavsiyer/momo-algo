"""OFI tuning — phase 1 (build).

Replays raw cached L2 book snapshots + T&S prints through the PRODUCTION
OrderFlowTracker (exec/order_flow.py) and samples, on a 250ms-decimated
event grid:

  - price (last trade)
  - the five tape entry gates' inputs (speed_gate, ratio_gate,
    trade_imbalance, accel_gate_z/ready, activity_count)
  - depth_active
  - ofi_normalized under several window variants:
      boxcar w in {8, 15, 30}s  (30 = current production, identity-checked
      against the tracker's own ofi_normalized read)
      EMA-kernel tau in {5, 10}s (exponentially forgotten OFI sum over
      exponentially weighted mean depth — candidate faster estimator)

Writes one npz per symbol-day into <out>/SYM_YMD.npz.

Run:  python ofi_tune_build.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

from exec.order_flow import OrderFlowTracker  # noqa: E402

CACHE = os.path.join(REPO, "logs", "chart_cache")
# v2: adds supply-side decomposition (ask/bid near-touch loss/add rates,
# buy/sell fill-volume rates, short notional imbalance) for the ARM/TRIGGER
# entry-structure study. New output dir so v1 results stay reproducible.
OUT = os.environ.get("OFI_TUNE_OUT", os.path.join(REPO, "logs", "research", "ofi_tune_v2"))
os.makedirs(OUT, exist_ok=True)

BOX_WINDOWS = (8.0, 15.0, 30.0)
EMA_TAUS = (5.0, 10.0)
SAMPLE_MS = 250
SUPPLY_TAU = 4.0        # EMA tau for depth loss/add + fill-volume rates.
                        # Tape is batch-flushed ~500ms behind the event-driven
                        # book (measured); a 4s kernel washes out that skew so
                        # cancel-vs-fill attribution stays honest.
NIMB_TAU = 5.0          # short notional (dollar-flow) imbalance kernel
NEAR_TOUCH_PCT = 0.005  # near-touch zone: levels within 0.5% of the side's best

# book>=15k rows, timesales>=10k rows (from coverage survey)
DAYS = [
    ("CLRO", "20260806"), ("CELZ", "20260806"), ("BJDX", "20260805"),
    ("AMIX", "20260804"), ("WYHG", "20260806"), ("AMIX", "20260805"),
    ("PAVS", "20260806"), ("AZI", "20260806"), ("YXT", "20260805"),
    ("ADGM", "20260804"), ("PN", "20260806"), ("NAMI", "20260806"),
    ("LVWR", "20260727"), ("DSY", "20260806"), ("SURG", "20260806"),
    ("XHLD", "20260806"), ("ENTX", "20260727"), ("TNMG", "20260804"),
    ("BYAH", "20260806"), ("AMIX", "20260729"), ("GAUZ", "20260806"),
    ("ATPC", "20260804"), ("MOVE", "20260804"), ("HYFM", "20260803"),
    ("VMAR", "20260805"), ("HYFM", "20260804"), ("MGRX", "20260806"),
    ("DXST", "20260804"), ("DFNS", "20260727"), ("KIDZ", "20260728"),
]


class _Level:
    __slots__ = ("price", "size")

    def __init__(self, price: float, size: int) -> None:
        self.price = price
        self.size = size


class _Snap:
    __slots__ = ("ts", "bids", "asks")

    def __init__(self, ts, bids, asks) -> None:
        self.ts = ts
        self.bids = bids
        self.asks = asks


class _TS:
    __slots__ = ("ts", "last", "bid", "ask", "size")

    def __init__(self, ts, last, bid, ask, size) -> None:
        self.ts = ts
        self.last = last
        self.bid = bid
        self.ask = ask
        self.size = size


def _load_trades(symbol: str, ymd: str) -> list[_TS]:
    path = os.path.join(CACHE, f"timesales_{symbol}_{ymd}.csv")
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    for c in ("price", "size", "bid", "ask"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ts", "price", "size"])
    df = df[(df["price"] > 0) & (df["size"] > 0)]
    df = df.sort_values(["ts", "seq"], kind="stable")
    out: list[_TS] = []
    for ts, price, size, bid, ask in zip(
        df["ts"].dt.to_pydatetime(), df["price"].to_numpy(),
        df["size"].to_numpy(), df["bid"].to_numpy(), df["ask"].to_numpy(),
    ):
        p = float(price)
        b, a = float(bid) if bid == bid else 0.0, float(ask) if ask == ask else 0.0
        if b <= 0 or a <= 0:  # same synthesis as backtest._timesale_event_from_cache_row
            half = max(p * 0.002, 0.0001) / 2.0
            b, a = p - half, p + half
        out.append(_TS(ts, p, b, a, int(size)))
    return out


def _load_books(symbol: str, ymd: str) -> list[_Snap]:
    path = os.path.join(CACHE, f"book_{symbol}_{ymd}.csv")
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts", kind="stable")
    out: list[_Snap] = []
    for ts, bids_s, asks_s in zip(
        df["ts"].dt.to_pydatetime(), df["bids"].to_numpy(), df["asks"].to_numpy()
    ):
        def levels(raw) -> list[_Level]:
            if not isinstance(raw, str) or not raw:
                return []
            try:
                pairs = json.loads(raw)
            except ValueError:
                return []
            lv = []
            for pair in pairs:
                try:
                    price, size = float(pair[0]), int(pair[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if price > 0 and size > 0:  # backtest._load_book_snapshots rule
                    lv.append(_Level(price, size))
            return lv
        out.append(_Snap(ts, levels(bids_s), levels(asks_s)))
    return out


def build_day(symbol: str, ymd: str) -> None:
    out_path = os.path.join(OUT, f"{symbol}_{ymd}.npz")
    if os.path.exists(out_path):
        print(f"[skip] {symbol} {ymd} (exists)")
        return
    t0 = time.time()
    trades = _load_trades(symbol, ymd)
    books = _load_books(symbol, ymd)
    if not trades or not books:
        print(f"[skip] {symbol} {ymd}: trades={len(trades)} books={len(books)}")
        return

    tracker = OrderFlowTracker()  # production config (30s OFI window)

    # Harness-side window/EMA replicas fed from the tracker's own OFIEvents
    # (identical deltas for any window; only the aggregation differs).
    box_ofi = {w: deque() for w in BOX_WINDOWS}   # (ts_s, ofi)
    box_ofi_sum = {w: 0.0 for w in BOX_WINDOWS}
    box_dep = {w: deque() for w in BOX_WINDOWS}   # (ts_s, depth)
    box_dep_sum = {w: 0.0 for w in BOX_WINDOWS}
    ema_ofi = {tau: 0.0 for tau in EMA_TAUS}
    ema_dep_pv = {tau: 0.0 for tau in EMA_TAUS}
    ema_dep_w = {tau: 0.0 for tau in EMA_TAUS}
    ema_last: dict[float, float | None] = {tau: None for tau in EMA_TAUS}

    # --- Supply-side decomposition state (price-keyed ladder diffs) ---
    sup = {
        "ask_loss": 0.0, "ask_add": 0.0, "bid_loss": 0.0, "bid_add": 0.0,
        "buy_fill": 0.0, "sell_fill": 0.0, "last": None,
        "nb": 0.0, "ns": 0.0, "nimb_last": None,
        "prev_asks": None, "prev_bids": None, "prev_book_ts": None,
    }

    def sup_decay(ts_s: float) -> None:
        last = sup["last"]
        if last is not None and ts_s > last:
            k = math.exp(-(ts_s - last) / SUPPLY_TAU)
            for key in ("ask_loss", "ask_add", "bid_loss", "bid_add", "buy_fill", "sell_fill"):
                sup[key] *= k
        sup["last"] = ts_s

    def nimb_decay(ts_s: float) -> None:
        last = sup["nimb_last"]
        if last is not None and ts_s > last:
            k = math.exp(-(ts_s - last) / NIMB_TAU)
            sup["nb"] *= k
            sup["ns"] *= k
        sup["nimb_last"] = ts_s

    def sup_on_book(ts_s: float, snap) -> None:
        """Near-touch depth loss/add per side, price-keyed across snapshots."""
        asks = {l.price: l.size for l in snap.asks}
        bids = {l.price: l.size for l in snap.bids}
        sup_decay(ts_s)
        prev_ts = sup["prev_book_ts"]
        fresh = prev_ts is None or (ts_s - prev_ts) > 30.0  # stale gap = reseed
        if not fresh and sup["prev_asks"] is not None:
            pa, pb = sup["prev_asks"], sup["prev_bids"]
            if pa:
                best_pa = min(pa)
                zone_pa = {p for p in pa if p <= best_pa * (1 + NEAR_TOUCH_PCT)}
                sup["ask_loss"] += sum(max(0, pa[p] - asks.get(p, 0)) for p in zone_pa)
            if asks:
                best_ca = min(asks)
                zone_ca = {p for p in asks if p <= best_ca * (1 + NEAR_TOUCH_PCT)}
                sup["ask_add"] += sum(max(0, asks[p] - pa.get(p, 0)) for p in zone_ca)
            if pb:
                best_pb = max(pb)
                zone_pb = {p for p in pb if p >= best_pb * (1 - NEAR_TOUCH_PCT)}
                sup["bid_loss"] += sum(max(0, pb[p] - bids.get(p, 0)) for p in zone_pb)
            if bids:
                best_cb = max(bids)
                zone_cb = {p for p in bids if p >= best_cb * (1 - NEAR_TOUCH_PCT)}
                sup["bid_add"] += sum(max(0, bids[p] - sup["prev_bids"].get(p, 0)) for p in zone_cb)
        sup["prev_asks"] = asks
        sup["prev_bids"] = bids
        sup["prev_book_ts"] = ts_s

    def sup_on_trade(ts_s: float, price: float, size: int, bid: float, ask: float) -> None:
        sup_decay(ts_s)
        nimb_decay(ts_s)
        if ask > 0 and price >= ask:
            sup["buy_fill"] += size
            sup["nb"] += price * size
        elif bid > 0 and price <= bid:
            sup["sell_fill"] += size
            sup["ns"] += price * size

    def on_ofi_event(ts_s: float, ofi: float, depth: float | None) -> None:
        for w in BOX_WINDOWS:
            dq, cutoff = box_ofi[w], ts_s - w
            dq.append((ts_s, ofi))
            box_ofi_sum[w] += ofi
            while dq and dq[0][0] < cutoff:
                box_ofi_sum[w] -= dq.popleft()[1]
            if depth is not None:
                dd = box_dep[w]
                dd.append((ts_s, depth))
                box_dep_sum[w] += depth
                while dd and dd[0][0] < cutoff:
                    box_dep_sum[w] -= dd.popleft()[1]
        for tau in EMA_TAUS:
            last = ema_last[tau]
            if last is not None and ts_s > last:
                k = math.exp(-(ts_s - last) / tau)
                ema_ofi[tau] *= k
                ema_dep_pv[tau] *= k
                ema_dep_w[tau] *= k
            ema_ofi[tau] += ofi
            if depth is not None:
                ema_dep_pv[tau] += depth
                ema_dep_w[tau] += 1.0
            ema_last[tau] = ts_s

    def trim_only(ts_s: float) -> None:
        """Advance window trims on trade time so stale flow ages out even
        with no book event (matches tracker's trim-on-any-event behavior
        closely enough for sampling)."""
        for w in BOX_WINDOWS:
            dq, cutoff = box_ofi[w], ts_s - w
            while dq and dq[0][0] < cutoff:
                box_ofi_sum[w] -= dq.popleft()[1]
            dd = box_dep[w]
            while dd and dd[0][0] < cutoff:
                box_dep_sum[w] -= dd.popleft()[1]

    # Sample buffers
    S: dict[str, list] = {k: [] for k in (
        "t", "price", "depth_active", "speed", "ratio", "volimb",
        "accel_ready", "accelz", "activity",
        "norm8", "norm15", "norm30", "norm30_tracker",
        "ema5", "ema10", "depth30",
        "ask_loss_r", "ask_add_r", "bid_loss_r", "bid_add_r",
        "buy_fill_r", "sell_fill_r", "nimb5",
        # v3 (2026-08-23): production EXIT-stack reads, for relabeling
        # entries by what evaluate_exit would have realized.
        "accel_raw", "buy_p", "sell_p",
    )}
    last_sample_ms = -1 << 62
    last_price = 0.0

    def sample(ts) -> None:
        nonlocal last_sample_ms
        t_ms = int(ts.timestamp() * 1000)
        if t_ms - last_sample_ms < SAMPLE_MS or last_price <= 0:
            return
        last_sample_ms = t_ms

        def norm(w: float) -> float:
            dd = box_dep[w]
            if not dd:
                return 0.0
            avg = box_dep_sum[w] / len(dd)
            return box_ofi_sum[w] / avg if avg > 0 else 0.0

        def enorm(tau: float) -> float:
            if ema_dep_w[tau] <= 0:
                return 0.0
            avg = ema_dep_pv[tau] / ema_dep_w[tau]
            return ema_ofi[tau] / avg if avg > 0 else 0.0

        S["t"].append(t_ms)
        S["price"].append(last_price)
        S["depth_active"].append(1 if tracker.depth_active else 0)
        S["speed"].append(tracker.speed_gate)
        S["ratio"].append(tracker.ratio_gate)
        S["volimb"].append(tracker.trade_imbalance)
        S["accel_ready"].append(1 if tracker.accel_gate_ready else 0)
        S["accelz"].append(tracker.accel_gate_z)
        S["activity"].append(tracker.activity_count)
        S["accel_raw"].append(tracker.accel_gate_raw)
        S["buy_p"].append(tracker.buying_pressure)
        S["sell_p"].append(tracker.selling_pressure)
        S["norm8"].append(norm(8.0))
        S["norm15"].append(norm(15.0))
        S["norm30"].append(norm(30.0))
        S["norm30_tracker"].append(tracker.ofi_normalized)
        S["ema5"].append(enorm(5.0))
        S["ema10"].append(enorm(10.0))
        S["depth30"].append(tracker.book_depth_avg)
        S["ask_loss_r"].append(sup["ask_loss"] / SUPPLY_TAU)
        S["ask_add_r"].append(sup["ask_add"] / SUPPLY_TAU)
        S["bid_loss_r"].append(sup["bid_loss"] / SUPPLY_TAU)
        S["bid_add_r"].append(sup["bid_add"] / SUPPLY_TAU)
        S["buy_fill_r"].append(sup["buy_fill"] / SUPPLY_TAU)
        S["sell_fill_r"].append(sup["sell_fill"] / SUPPLY_TAU)
        ntot = sup["nb"] + sup["ns"]
        S["nimb5"].append((sup["nb"] - sup["ns"]) / ntot if ntot > 0 else 0.0)

    bi, n_books = 0, len(books)
    for tr in trades:
        while bi < n_books and books[bi].ts <= tr.ts:
            snap = books[bi]
            ev = tracker.on_book(snap)
            depth = sum(l.size for l in snap.bids) + sum(l.size for l in snap.asks)
            if ev is not None:
                on_ofi_event(snap.ts.timestamp(), float(ev.ofi), float(depth))
            sup_on_book(snap.ts.timestamp(), snap)
            bi += 1
            sample(snap.ts)
        tracker.on_timesale(tr)
        sup_on_trade(tr.ts.timestamp(), tr.last, tr.size, tr.bid, tr.ask)
        trim_only(tr.ts.timestamp())
        last_price = tr.last
        sample(tr.ts)
    # drain remaining books after last trade
    while bi < n_books:
        snap = books[bi]
        ev = tracker.on_book(snap)
        depth = sum(l.size for l in snap.bids) + sum(l.size for l in snap.asks)
        if ev is not None:
            on_ofi_event(snap.ts.timestamp(), float(ev.ofi), float(depth))
        sup_on_book(snap.ts.timestamp(), snap)
        bi += 1
        sample(snap.ts)

    arrays = {
        "t": np.array(S["t"], dtype=np.int64),
        "price": np.array(S["price"], dtype=np.float32),
        "depth_active": np.array(S["depth_active"], dtype=np.uint8),
        "speed": np.array(S["speed"], dtype=np.float32),
        "ratio": np.array(S["ratio"], dtype=np.float32),
        "volimb": np.array(S["volimb"], dtype=np.float32),
        "accel_ready": np.array(S["accel_ready"], dtype=np.uint8),
        "accelz": np.array(S["accelz"], dtype=np.float32),
        "activity": np.array(S["activity"], dtype=np.int32),
        "norm8": np.array(S["norm8"], dtype=np.float32),
        "norm15": np.array(S["norm15"], dtype=np.float32),
        "norm30": np.array(S["norm30"], dtype=np.float32),
        "norm30_tracker": np.array(S["norm30_tracker"], dtype=np.float32),
        "ema5": np.array(S["ema5"], dtype=np.float32),
        "ema10": np.array(S["ema10"], dtype=np.float32),
        "depth30": np.array(S["depth30"], dtype=np.float32),
        "ask_loss_r": np.array(S["ask_loss_r"], dtype=np.float32),
        "ask_add_r": np.array(S["ask_add_r"], dtype=np.float32),
        "bid_loss_r": np.array(S["bid_loss_r"], dtype=np.float32),
        "bid_add_r": np.array(S["bid_add_r"], dtype=np.float32),
        "buy_fill_r": np.array(S["buy_fill_r"], dtype=np.float32),
        "sell_fill_r": np.array(S["sell_fill_r"], dtype=np.float32),
        "nimb5": np.array(S["nimb5"], dtype=np.float32),
        "accel_raw": np.array(S["accel_raw"], dtype=np.float32),
        "buy_p": np.array(S["buy_p"], dtype=np.int32),
        "sell_p": np.array(S["sell_p"], dtype=np.int32),
    }
    np.savez_compressed(out_path, **arrays)

    # identity check: harness 30s boxcar vs tracker read, on depth-active samples
    m = arrays["depth_active"] == 1
    if m.sum() > 100:
        d = np.abs(arrays["norm30"][m] - arrays["norm30_tracker"][m])
        print(f"[ok] {symbol} {ymd}: {len(arrays['t'])} samples, "
              f"depth-active {100*m.mean():.0f}%, w30 identity p99 |d|={np.percentile(d, 99):.4f}, "
              f"{time.time()-t0:.0f}s")
    else:
        print(f"[ok] {symbol} {ymd}: {len(arrays['t'])} samples (little depth), {time.time()-t0:.0f}s")


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for sym, ymd in DAYS:
        if only and f"{sym}_{ymd}" not in only:
            continue
        try:
            build_day(sym, ymd)
        except Exception as exc:  # keep the sweep going
            print(f"[err] {sym} {ymd}: {exc!r}")
