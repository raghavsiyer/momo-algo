"""
One-off offline sweep: EMA vs hard-window tape estimators.

Compares, at near-HOD breakout decision points across cached stream tapes:
  - flat speed (10s window)      vs  speed_ema (exponential kernel)
  - 2-bin acceleration (+z)      vs  acceleration_ema (+z)
  - flat imbalance (window)      vs  imbalance_ema

Decision population = pokes to a fresh 30s high AT/NEAR the running HOD
(includes failures), 30s refractory. Labels over the next 120s:
  - softmfe : price reaches +1.5% at any point
  - cleango : +1.5% target hit BEFORE -1.0% stop
AUC is rank-based (no sklearn). Higher = more predictive of continuation.

NOT committed — throwaway research harness.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from chart_cache import cache_dir, load_timesales_cache
from _squeeze_analysis_scratch import compute_ts_metrics

# ── config ────────────────────────────────────────────────────────────────
N_FILES = 30            # richest tapes to use
MIN_ROWS = 800          # skip thin tapes
NEAR_HOD = 0.995        # poke must be >= this * running HOD
LOOKBACK_S = 30.0       # "fresh high" reference window
REFRACTORY_S = 30.0     # min spacing between decisions
WARMUP_S = 60.0         # need this much tape before a decision
FWD_S = 120.0           # forward horizon for labelling
MIN_SPEED = 3.0         # tape-alive floor (broad pop, keeps failures)
TARGET = 0.015          # +1.5%
STOP = 0.010            # -1.0%
WIDE_START = datetime(2020, 1, 1, tzinfo=timezone.utc)
WIDE_END = datetime(2030, 1, 1, tzinfo=timezone.utc)


def tsec_of(ts_series: pd.Series) -> np.ndarray:
    """UTC epoch seconds — tz-safe (avoids the astype(int64) collapse bug)."""
    tt = pd.to_datetime(ts_series, utc=True)
    return ((tt - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks_sorted = np.empty(len(scores), dtype=float)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def pick_files() -> list[tuple[str, str, Path]]:
    d = cache_dir()
    files = sorted(d.glob("timesales_*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    out: list[tuple[str, str, Path]] = []
    for p in files:
        stem = p.stem  # timesales_<SYM>_<YMD>
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        _, sym, ymd = parts
        out.append((sym, ymd, p))
        if len(out) >= N_FILES:
            break
    return out


def process(sym: str, ymd: str) -> list[dict]:
    df = load_timesales_cache(sym, WIDE_START, WIDE_END, ymds=[ymd])
    if df.empty:
        return []
    df = df[df["source"] == "stream"].reset_index(drop=True)
    if len(df) < MIN_ROWS:
        return []

    m = compute_ts_metrics(df)  # default tau: speed=3, fast=3, slow=10, imb=10
    price = m["price"].to_numpy(dtype=float)
    tsec = tsec_of(m["ts"])
    n = len(price)
    if n < MIN_ROWS:
        return []

    run_hi = np.maximum.accumulate(price)
    back_idx = np.searchsorted(tsec, tsec - LOOKBACK_S, side="left")
    fwd_end = np.searchsorted(tsec, tsec + FWD_S, side="right")

    # directional-print cumulative count for the 2-bin accel z (N in last 10s)
    is_dir = np.isin(m["side"].to_numpy(), ("buy", "sell")).astype(float)
    cum_dir = np.cumsum(is_dir)
    win10 = np.searchsorted(tsec, tsec - 10.0, side="left")

    flat_speed = m["speed"].to_numpy(float)
    speed_ema = m["speed_ema"].to_numpy(float)
    accel2 = m["acceleration"].to_numpy(float)
    accel_ema = m["acceleration_ema"].to_numpy(float)
    accel_ema_z = m["acceleration_ema_z"].to_numpy(float)
    flat_imb = m["imbalance"].to_numpy(float)
    imb_ema = m["imbalance_ema"].to_numpy(float)

    rows: list[dict] = []
    last_dec = -1e9
    for i in range(n):
        if tsec[i] - tsec[0] < WARMUP_S:
            continue
        if tsec[-1] - tsec[i] < FWD_S:
            break
        if flat_speed[i] < MIN_SPEED:
            continue
        if price[i] < NEAR_HOD * run_hi[i]:
            continue
        # fresh 30s high
        b = back_idx[i]
        if b >= i:
            continue
        if price[i] <= price[b:i].max():
            continue
        if tsec[i] - last_dec < REFRACTORY_S:
            continue
        # forward label
        e = fwd_end[i]
        fwd = price[i + 1:e]
        if len(fwd) < 5:
            continue
        entry = price[i]
        softmfe = int((fwd.max() / entry - 1.0) >= TARGET)
        # path-aware clean-go
        cleango = 0
        for px in fwd:
            r = px / entry - 1.0
            if r >= TARGET:
                cleango = 1
                break
            if r <= -STOP:
                cleango = 0
                break
        last_dec = tsec[i]

        n_win = cum_dir[i] - (cum_dir[win10[i] - 1] if win10[i] > 0 else 0.0)
        rows.append({
            "flat_speed": flat_speed[i],
            "speed_ema": speed_ema[i],
            "accel2": accel2[i],
            "accel2_z": accel2[i] * np.sqrt(max(n_win, 0.0)),
            "accel_ema": accel_ema[i],
            "accel_ema_z": accel_ema_z[i],
            "flat_imb": flat_imb[i],
            "imb_ema": imb_ema[i],
            "softmfe": softmfe,
            "cleango": cleango,
        })
    return rows


def main() -> None:
    files = pick_files()
    all_rows: list[dict] = []
    for sym, ymd, _ in files:
        try:
            r = process(sym, ymd)
        except Exception as exc:  # noqa: BLE001 - research script
            print(f"  ! {sym} {ymd}: {exc}")
            continue
        if r:
            all_rows.extend(r)
            print(f"  {sym} {ymd}: {len(r)} decisions")
    if not all_rows:
        print("no decisions collected")
        return
    d = pd.DataFrame(all_rows)
    print(f"\nPooled decisions: {len(d)}")
    for lab in ("softmfe", "cleango"):
        print(f"  base rate {lab}: {d[lab].mean():.3f}")

    estimators = [
        ("SPEED  flat (10s)", "flat_speed"),
        ("SPEED  ema (tau=3)", "speed_ema"),
        ("ACCEL  2-bin raw", "accel2"),
        ("ACCEL  2-bin z", "accel2_z"),
        ("ACCEL  ema raw", "accel_ema"),
        ("ACCEL  ema z", "accel_ema_z"),
        ("IMBAL  flat (30s)", "flat_imb"),
        ("IMBAL  ema (tau=10)", "imb_ema"),
    ]
    print(f"\n{'estimator':<22s} {'AUC softmfe':>12s} {'AUC cleango':>12s}")
    print("-" * 48)
    for name, col in estimators:
        a1 = rank_auc(d[col].to_numpy(), d["softmfe"].to_numpy())
        a2 = rank_auc(d[col].to_numpy(), d["cleango"].to_numpy())
        print(f"{name:<22s} {a1:>12.3f} {a2:>12.3f}")


if __name__ == "__main__":
    main()
