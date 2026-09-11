"""Scratch harness: 2-bin vs EMA tape-acceleration signal-timing comparison.

Replays cached stream tapes (logs/chart_cache/timesales_SYM_YYYYMMDD.csv)
through the REAL OrderFlowTracker, recording BOTH estimators per print on
identical input. Ground-truth "tape surges" are detected independently of
either estimator (1s-binned directional print rate vs trailing median), so
detection latency and false fires can be compared objectively.

Usage (via Pylance snippet or terminal):
    from _tape_accel_compare_scratch import replay_tape, TAPES
    replay_tape("ZYBT", "20260720")   # -> writes /tmp/tape_accel_compare/ZYBT_20260720.json

Outputs per tape:
  surges:   [{onset, end, peak_pps, base_pps, px_onset, fwd_ret_30s, fwd_ret_60s}]
  fires_2bin / fires_ema: [{ts, z, accel, speed, buy_ratio, price, full_gate}]
  matching: per-surge first fire within [-2s, +10s] of onset -> latency
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from exec.order_flow import OrderFlowTracker
from market_data.tradier_stream import LiveTimesale
from config import (
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_SPEED_MIN,
)

CACHE_DIR = "/Users/ahariharan/Desktop/momo_algo/logs/chart_cache"
OUT_DIR = "/tmp/tape_accel_compare"

# Ground-truth surge detection (independent of both estimators)
SURGE_ABS_FLOOR_PPS = 20.0      # bin must reach at least this many prints/s
SURGE_MULT = 3.0                # ... and >= this multiple of trailing median
SURGE_BASELINE_SECS = 30        # trailing median window (1s bins)
SURGE_END_MULT = 1.5            # surge ends when rate drops below this multiple
SURGE_MERGE_GAP_SECS = 5        # merge surges closer than this

# Gate edge-trigger hysteresis
Z_FIRE = SIM_PRESSURE_ACCEL_Z   # fire when z crosses >= this
Z_REARM = 1.0                   # re-arm when z falls back below this

MATCH_PRE_SECS = 2.0            # fire may lead onset by up to this (kernel lead)
MATCH_POST_SECS = 10.0          # fire within this after onset counts as detection


def _load_tape(symbol: str, day: str) -> pd.DataFrame:
    path = os.path.join(CACHE_DIR, f"timesales_{symbol}_{day}.csv")
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return df


def replay_tape(symbol: str, day: str, out_dir: str = OUT_DIR, save_traces: bool = False) -> dict:
    df = _load_tape(symbol, day)
    n = len(df)
    trk = OrderFlowTracker()

    ts_arr = df["ts"].to_numpy()                       # datetime64[ns, UTC]
    epoch = (ts_arr - np.datetime64("1970-01-01T00:00:00", "ns").astype("datetime64[ns]")).astype("int64") / 1e9 \
        if ts_arr.dtype.kind == "M" else \
        ((df["ts"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    price_arr = df["price"].to_numpy(dtype=np.float64)
    size_arr = df["size"].fillna(0).to_numpy(dtype=np.int64)
    bid_arr = df["bid"].fillna(0.0).to_numpy(dtype=np.float64)
    ask_arr = df["ask"].fillna(0.0).to_numpy(dtype=np.float64)
    py_ts = df["ts"].dt.to_pydatetime()

    # Per-print traces
    z2 = np.zeros(n); a2 = np.zeros(n); r2 = np.zeros(n, dtype=bool)
    ze = np.zeros(n); ae = np.zeros(n); re_ = np.zeros(n, dtype=bool)
    spd = np.zeros(n); spd_ema = np.zeros(n); bratio = np.zeros(n)
    act = np.zeros(n, dtype=np.int64)
    is_dir = np.zeros(n, dtype=bool)

    for i in range(n):
        ev = LiveTimesale(
            event_type="timesale", symbol=symbol, exchange="",
            last=price_arr[i], bid=bid_arr[i], ask=ask_arr[i],
            size=int(size_arr[i]), ts=py_ts[i], seq=i, flag="",
            cancel=False, correction=False, session="normal",
        )
        ct = trk.on_timesale(ev)
        is_dir[i] = ct.side != "mid"
        r2[i] = trk.pressure_acceleration_ready
        a2[i] = trk.pressure_acceleration
        z2[i] = trk.pressure_acceleration_z
        re_[i] = trk.pressure_acceleration_ema_ready
        ae[i] = trk.pressure_acceleration_ema
        ze[i] = trk.pressure_acceleration_ema_z
        spd[i] = trk.trade_speed
        spd_ema[i] = trk.tape_speed_ema
        bratio[i] = trk.buy_ratio
        act[i] = trk.activity_count

    surges = _detect_surges(epoch, is_dir, price_arr)
    fires_2bin = _edge_fires(epoch, price_arr, r2, z2, spd, bratio, act)
    fires_ema = _edge_fires(epoch, price_arr, re_, ze, spd, bratio, act)
    matches = {
        "2bin": _match(surges, fires_2bin),
        "ema": _match(surges, fires_ema),
    }

    session_secs = float(epoch[-1] - epoch[0]) if n > 1 else 0.0
    result = {
        "symbol": symbol, "day": day, "prints": int(n),
        "directional_prints": int(is_dir.sum()),
        "session_secs": session_secs,
        "surges": surges,
        "fires_2bin": fires_2bin,
        "fires_ema": fires_ema,
        "matches": matches,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{symbol}_{day}.json"), "w") as f:
        json.dump(result, f)
    if save_traces:
        np.savez_compressed(
            os.path.join(out_dir, f"{symbol}_{day}_trace.npz"),
            epoch=epoch, price=price_arr, is_dir=is_dir,
            z2=z2, a2=a2, r2=r2, ze=ze, ae=ae, re=re_,
            spd=spd, spd_ema=spd_ema, bratio=bratio, act=act,
        )
    return _summary(result)


def _detect_surges(epoch: np.ndarray, is_dir: np.ndarray, price: np.ndarray) -> list[dict]:
    """1s-binned directional print rate vs trailing median -> surge windows."""
    if len(epoch) == 0:
        return []
    t0 = math.floor(epoch[0])
    bins = np.floor(epoch - t0).astype(np.int64)
    n_bins = int(bins[-1]) + 1
    rate = np.bincount(bins[is_dir], minlength=n_bins).astype(np.float64)

    # price per bin (last print price in bin, ffilled)
    px = np.full(n_bins, np.nan)
    px[bins] = price  # later prints in same bin overwrite = last
    for i in range(1, n_bins):
        if np.isnan(px[i]):
            px[i] = px[i - 1]

    surges: list[dict] = []
    window: deque[float] = deque(maxlen=SURGE_BASELINE_SECS)
    in_surge = False
    onset = end = 0
    peak = base_at_onset = 0.0
    for i in range(n_bins):
        base = float(np.median(window)) if len(window) >= 10 else float("nan")
        r = rate[i]
        if not in_surge:
            if not math.isnan(base) and r >= max(SURGE_ABS_FLOOR_PPS, SURGE_MULT * max(base, 1.0)):
                in_surge, onset, peak, base_at_onset = True, i, r, base
                calm = 0
        else:
            peak = max(peak, r)
            if r < SURGE_END_MULT * max(base_at_onset, 1.0):
                calm += 1
                if calm >= 3:
                    in_surge = False
                    surges.append({"onset": onset, "end": i, "peak_pps": peak, "base_pps": base_at_onset})
            else:
                calm = 0
        if not in_surge:
            window.append(r)
    if in_surge:
        surges.append({"onset": onset, "end": n_bins - 1, "peak_pps": peak, "base_pps": base_at_onset})

    # merge close surges
    merged: list[dict] = []
    for s in surges:
        if merged and s["onset"] - merged[-1]["end"] <= SURGE_MERGE_GAP_SECS:
            merged[-1]["end"] = s["end"]
            merged[-1]["peak_pps"] = max(merged[-1]["peak_pps"], s["peak_pps"])
        else:
            merged.append(s)

    # annotate absolute time + forward returns
    for s in merged:
        s["onset_epoch"] = t0 + s["onset"]
        s["end_epoch"] = t0 + s["end"]
        p0 = px[s["onset"]]
        s["px_onset"] = float(p0) if not math.isnan(p0) else None
        for horizon in (30, 60):
            j = min(s["onset"] + horizon, n_bins - 1)
            seg = px[s["onset"]: j + 1]
            seg = seg[~np.isnan(seg)]
            key = f"fwd_ret_{horizon}s"
            s[key] = float(np.nanmax(seg) / p0 - 1.0) if len(seg) and p0 and not math.isnan(p0) else None
    return merged


def _edge_fires(epoch, price, ready, z, spd, bratio, act) -> list[dict]:
    """Edge-triggered gate crossings with hysteresis re-arm."""
    fires: list[dict] = []
    armed = True
    for i in range(len(epoch)):
        if not ready[i]:
            continue
        if armed and z[i] >= Z_FIRE:
            full = (
                spd[i] >= SIM_PRESSURE_SPEED_MIN
                and act[i] >= SIM_PRESSURE_ACTIVITY_MIN
                and bratio[i] >= SIM_PRESSURE_BUY_RATIO_MIN
            )
            fires.append({
                "epoch": float(epoch[i]), "z": float(z[i]),
                "price": float(price[i]), "speed": float(spd[i]),
                "buy_ratio": float(bratio[i]), "full_gate": bool(full),
            })
            armed = False
        elif not armed and z[i] < Z_REARM:
            armed = True
    return fires


def _match(surges: list[dict], fires: list[dict]) -> dict:
    """Per-surge detection latency + false-fire count."""
    fire_ts = np.array([f["epoch"] for f in fires]) if fires else np.array([])
    latencies: list = []
    matched_fire_idx: set[int] = set()
    for s in surges:
        lo, hi = s["onset_epoch"] - MATCH_PRE_SECS, s["onset_epoch"] + MATCH_POST_SECS
        idx = np.where((fire_ts >= lo) & (fire_ts <= hi))[0]
        if len(idx):
            latencies.append(float(fire_ts[idx[0]] - s["onset_epoch"]))
            matched_fire_idx.update(int(k) for k in idx)
        else:
            latencies.append(None)
    # a fire also counts as "covered" if inside any surge span (post-onset continuation)
    for k, f in enumerate(fires):
        if k in matched_fire_idx:
            continue
        for s in surges:
            if s["onset_epoch"] - MATCH_PRE_SECS <= f["epoch"] <= s["end_epoch"]:
                matched_fire_idx.add(k)
                break
    false_fires = [k for k in range(len(fires)) if k not in matched_fire_idx]
    return {
        "latencies": latencies,
        "n_detected": sum(1 for l in latencies if l is not None),
        "n_surges": len(surges),
        "n_fires": len(fires),
        "n_false_fires": len(false_fires),
        "false_fire_idx": false_fires,
    }


def _summary(result: dict) -> dict:
    out = {"symbol": result["symbol"], "day": result["day"], "prints": result["prints"],
           "n_surges": len(result["surges"])}
    for name in ("2bin", "ema"):
        m = result["matches"][name]
        lats = [l for l in m["latencies"] if l is not None]
        out[name] = {
            "detected": f"{m['n_detected']}/{m['n_surges']}",
            "median_latency_s": round(float(np.median(lats)), 2) if lats else None,
            "fires": m["n_fires"],
            "false_fires": m["n_false_fires"],
        }
    return out


TAPES = [
    ("SBEV", "20260706"), ("LHSW", "20260706"),
    ("CCHH", "20260707"), ("BJDX", "20260707"),
    ("ONFO", "20260708"), ("VTAK", "20260708"),
    ("GMM", "20260710"), ("ZBAO", "20260710"),
    ("AGEN", "20260713"), ("VEEE", "20260713"),
    ("VIVS", "20260715"), ("SOBR", "20260715"),
    ("TGHL", "20260716"), ("IQST", "20260716"),
    ("ZYBT", "20260720"), ("BIYA", "20260720"),
    ("VIVK", "20260721"), ("SLGB", "20260721"),
    ("EHGO", "20260723"), ("WLDS", "20260723"),
]

if __name__ == "__main__":
    import sys
    picks = TAPES
    if len(sys.argv) > 1:
        picks = [tuple(a.split("_")) for a in sys.argv[1:]]
    for sym, d in picks:
        print(json.dumps(replay_tape(sym, d)))
