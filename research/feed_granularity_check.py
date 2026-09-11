"""Feed granularity check: can IBKR's batched tape support print-sequence
(absorption) reading?

The tape is batch-flushed (~1.5 flushes/sec, measured 2026-08). The question
is what survives inside a flush. An absorption detector needs, per print:
its order (sequence), price, size, and a side read (bid/ask context) — and
enough flushes across a multi-second dip to see the pattern develop.

Measures, on the N largest cached symbol-days:
  1. Clumping: % prints sharing a timestamp, clump sizes, flush rate in the
     busiest 60s window.
  2. Within-clump survival: distinct prices/sizes inside clumps, seq order.
  3. Side classifiability: % prints at/inside bid or ask.
  4. Dip coverage: median flushes per 10s during the busiest windows.

Run:  python feed_granularity_check.py [N_days]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(REPO, "logs", "chart_cache")


def analyze(path: str) -> dict | None:
    name = os.path.basename(path).replace("timesales_", "").replace(".csv", "")
    df = pd.read_csv(path)
    if len(df) < 5000:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "price", "size"]).sort_values(["ts", "seq"], kind="stable")
    t_ms = (df["ts"].astype("int64") // 1_000_000).to_numpy()
    price = pd.to_numeric(df["price"], errors="coerce").to_numpy()
    size = pd.to_numeric(df["size"], errors="coerce").to_numpy()
    bid = pd.to_numeric(df["bid"], errors="coerce").to_numpy()
    ask = pd.to_numeric(df["ask"], errors="coerce").to_numpy()
    seq = pd.to_numeric(df["seq"], errors="coerce").to_numpy()
    n = len(df)

    # --- clumps: runs of identical timestamps ---
    new_clump = np.r_[True, t_ms[1:] != t_ms[:-1]]
    clump_id = np.cumsum(new_clump) - 1
    n_clumps = clump_id[-1] + 1
    clump_sizes = np.bincount(clump_id)
    dup_share = 1.0 - n_clumps / n

    # busiest 60s window by print count
    sec = t_ms // 1000
    counts = pd.Series(1, index=sec).groupby(level=0).sum()
    roll = counts.rolling(60, min_periods=1).sum()
    end_s = int(roll.idxmax())
    win = (sec > end_s - 60) & (sec <= end_s)
    win_t = t_ms[win]
    flushes_busy60 = len(np.unique(win_t))
    prints_busy60 = int(win.sum())

    # flushes per 10s across the busiest 10 minutes (dip-coverage view)
    busy_mask = (sec > end_s - 600) & (sec <= end_s)
    bt = np.unique(t_ms[busy_mask])
    if len(bt) > 1:
        per10 = pd.Series(1, index=bt // 10_000).groupby(level=0).sum()
        med_flush_10s = float(per10.median())
    else:
        med_flush_10s = 0.0

    # --- within-clump survival (clumps with >= 2 prints) ---
    multi = clump_sizes[clump_id] >= 2
    dfm = pd.DataFrame({
        "cid": clump_id[multi], "price": price[multi],
        "size": size[multi], "seq": seq[multi],
    })
    if len(dfm):
        g = dfm.groupby("cid")
        multi_price = float((g["price"].nunique() > 1).mean())
        multi_size = float((g["size"].nunique() > 1).mean())
        seq_ordered = float(g["seq"].apply(lambda s: bool(s.is_monotonic_increasing)).mean())
    else:
        multi_price = multi_size = seq_ordered = float("nan")

    # --- side classifiability ---
    has_quote = (bid > 0) & (ask > 0) & (ask > bid)
    at_ask = has_quote & (price >= ask)
    at_bid = has_quote & (price <= bid)
    classifiable = float((at_ask | at_bid).mean())

    return {
        "name": name, "prints": n, "dup_share": dup_share,
        "clump_p50": float(np.percentile(clump_sizes, 50)),
        "clump_p90": float(np.percentile(clump_sizes, 90)),
        "clump_max": int(clump_sizes.max()),
        "flushes_busy60": flushes_busy60, "prints_busy60": prints_busy60,
        "med_flush_10s": med_flush_10s,
        "multi_price": multi_price, "multi_size": multi_size,
        "seq_ordered": seq_ordered, "classifiable": classifiable,
        "quote_share": float(has_quote.mean()),
    }


def main() -> None:
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    paths = sorted(glob.glob(os.path.join(CACHE, "timesales_*.csv")),
                   key=os.path.getsize, reverse=True)[:n_days]
    rows = [r for r in (analyze(p) for p in paths) if r]
    print(f"{'day':16s} {'prints':>7} {'dup%':>6} {'clmp p50/p90/max':>16} "
          f"{'flushes/60s':>11} {'prints/60s':>10} {'fl/10s':>6} "
          f"{'>1px%':>6} {'>1sz%':>6} {'seqOK%':>6} {'side%':>6}")
    for r in rows:
        print(f"{r['name']:16s} {r['prints']:7d} {100*r['dup_share']:5.1f}% "
              f"{r['clump_p50']:5.0f}/{r['clump_p90']:4.0f}/{r['clump_max']:5d} "
              f"{r['flushes_busy60']:11d} {r['prints_busy60']:10d} {r['med_flush_10s']:6.0f} "
              f"{100*r['multi_price']:5.1f}% {100*r['multi_size']:5.1f}% "
              f"{100*r['seq_ordered']:5.1f}% {100*r['classifiable']:5.1f}%")

    print("\nverdict inputs:")
    print("  - seqOK%  ~100 => print ORDER survives batching")
    print("  - >1px%/>1sz% high => per-print price/size survive (not aggregated)")
    print("  - side%   high => bid/ask context attached, side classifiable")
    print("  - fl/10s  => tape checkpoints per 10s of a dip (>=10 = a multi-"
          "second dip is seen at sub-second granularity)")


if __name__ == "__main__":
    main()
