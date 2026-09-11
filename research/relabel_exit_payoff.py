"""Relabel level-approach events with PRODUCTION-EXIT payoffs.

The 08-14 corpus labels events with a +2/-3/120s triple barrier — not what
the system earns. This walks each event forward through the same exit rules
evaluate_exit enforces (the three flow exits that produced every observed
replay exit), on the v3 corpus's exact production tracker reads:

  exit 1  pressure reversal: sell share of 10s counts >= SIM_EXIT_SELL_RATIO_MIN
  exit 1b accel collapse:    accel_raw <= SIM_EXIT_ACCEL_MAX and ratio lapsed
  exit 2  ofi negative:      ema10 < SIM_EXIT_OFI_NORM_MAX and ratio lapsed
          (depth-active only)

Confidence (exit 3) and resistance (exit 4) exits need scorer/rmap state
that does not exist offline; neither fired in any replayed trade. Entries
fill at the event price; holds are capped at HOLD_CAP_SECS.

Env:  OFI_TUNE_DATA must point at the v3 corpus (exit reads present).
Out:  logs/research/level_approach_labels_v3.csv  (original columns +
      pnl_exit / hold_secs / exit_reason)
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from config import (  # noqa: E402
    SIM_EXIT_ACCEL_BUY_RATIO_MAX,
    SIM_EXIT_ACCEL_MAX,
    SIM_EXIT_OFI_NORM_MAX,
    SIM_EXIT_SELL_RATIO_MIN,
    SIM_PRESSURE_ACTIVITY_MIN,
)
from label_level_approaches import _events_for_day, _sample_index_for_grid  # noqa: E402
from ofi_tune_study import DATA, load_days  # noqa: E402

OUT_CSV = os.path.join(REPO, "logs", "research", "level_approach_labels_v3.csv")
HOLD_CAP_SECS = 300.0


def exit_walk(z: dict, sp: np.ndarray, t_ms: np.ndarray, si: int,
              entry_price: float) -> tuple[float, float, str]:
    """(pnl, hold_secs, reason) from the first sample after entry."""
    n = len(t_ms)
    t0 = t_ms[si]
    for j in range(si + 1, n):
        hold = (t_ms[j] - t0) / 1000.0
        if hold > HOLD_CAP_SECS:
            return sp[j] / entry_price - 1.0, hold, "hold_cap"
        buy_p, sell_p = int(z["buy_p"][j]), int(z["sell_p"][j])
        tot = buy_p + sell_p
        if tot >= SIM_PRESSURE_ACTIVITY_MIN and sell_p / tot >= SIM_EXIT_SELL_RATIO_MIN:
            return sp[j] / entry_price - 1.0, hold, "pressure_reversal"
        ratio = float(z["ratio"][j])
        if (
            z["accel_ready"][j]
            and float(z["accel_raw"][j]) <= SIM_EXIT_ACCEL_MAX
            and ratio <= SIM_EXIT_ACCEL_BUY_RATIO_MAX
        ):
            return sp[j] / entry_price - 1.0, hold, "accel_collapse"
        if (
            z["depth_active"][j]
            and float(z["ema10"][j]) < SIM_EXIT_OFI_NORM_MAX
            and ratio <= SIM_EXIT_ACCEL_BUY_RATIO_MAX
        ):
            return sp[j] / entry_price - 1.0, hold, "ofi_negative"
    return sp[n - 1] / entry_price - 1.0, (t_ms[n - 1] - t0) / 1000.0, "day_end"


def main() -> None:
    days = load_days()
    paths = {os.path.basename(p).replace(".npz", ""): p
             for p in glob.glob(os.path.join(DATA, "*.npz"))}
    all_rows: list[dict] = []
    for day in days:
        raw = np.load(paths[day.name])
        z = {k: raw[k] for k in raw.files}
        if "accel_raw" not in z:
            raise SystemExit(f"{day.name}: no exit reads — point OFI_TUNE_DATA at the v3 corpus")
        events = _events_for_day(day, z)
        grid_to_sample = _sample_index_for_grid(day)
        sp = z["price"].astype(np.float64)
        t_ms = z["t"]
        for row in events:
            ts = datetime.fromisoformat(row["event_ts"])
            i = int(ts.timestamp()) - day.grid_t0
            si = int(grid_to_sample[i])
            pnl, hold, reason = exit_walk(z, sp, t_ms, si, row["event_price"])
            row["pnl_exit"] = round(pnl, 5)
            row["hold_secs"] = round(hold, 1)
            row["exit_reason"] = reason
        all_rows.extend(events)

    fields = list(all_rows[0].keys())
    for row in all_rows:
        for f in row:
            if f not in fields:
                fields.append(f)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    pnl = np.array([r["pnl_exit"] for r in all_rows])
    tb = np.array([r["tb_ret"] for r in all_rows])
    hold = np.array([r["hold_secs"] for r in all_rows])
    print(f"events={len(all_rows)} -> {OUT_CSV}")
    print(f"pnl_exit: mean={pnl.mean():+.4f} median={np.median(pnl):+.4f} "
          f"P(>0)={np.mean(pnl > 0):.3f} P(>=0.5%)={np.mean(pnl >= 0.005):.3f} "
          f"p5={np.percentile(pnl, 5):+.4f} p95={np.percentile(pnl, 95):+.4f}")
    print(f"hold_secs: median={np.median(hold):.1f} p90={np.percentile(hold, 90):.1f}")
    print(f"corr(pnl_exit, tb_ret)={np.corrcoef(pnl, tb)[0, 1]:.3f}")
    reasons = {}
    for r in all_rows:
        reasons[r["exit_reason"]] = reasons.get(r["exit_reason"], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        m = np.array([r["pnl_exit"] for r in all_rows if r["exit_reason"] == k])
        print(f"  {k:18s} n={v:5d}  mean={m.mean():+.4f}")


if __name__ == "__main__":
    main()
