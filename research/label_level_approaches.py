"""Triple-barrier labels at LEVEL-APPROACH instants (Stage-C prep).

The 2026-08-10 study conditioned only at gate-crossing instants — adversely
selected moments mid-burst — and found the ~48-51% ceiling. This harness
produces the dataset that study could not: every APPROACH of a prior swing
high (the armed-band's structural level), labeled by what happened next,
with the armed-band evidence state sampled AT the approach.

Event definition (per zigzag pivot high L, from ofi_tune_study.Day):
  1. price dips >= APPROACH_PULL_MIN below L after the pivot,
  2. then re-enters the armed band [L*(1-BAND_DEPTH), L*(1+BAND_TOL)];
  the re-entry second is the event. One event per band visit (price must
  leave the band before the same level can fire again); the level expires
  LEVEL_TTL_SECS after its pivot.

Label: triple barrier from the event price — UP +TB_UP before DN -TB_DN
within TB_TIMEOUT_SECS (+1), DN first (-1), timeout (0). fwdmax60/120
retained for threshold-free reads.

Features: instantaneous tape/OFI state, the ARMED-RECENT proxy (each
evidence gate true at ANY 250ms sample in the trailing ARM_TTL_SECS —
mirrors the tracker latches), and the v2 supply-side rates (ask_loss_r
etc.) when present.

Run:  python label_level_approaches.py          # writes CSV + prints summary
Env:  OFI_TUNE_DATA (npz dir; defaults to logs/research/ofi_tune — point at
      ofi_tune_v2 for supply-side columns)

Output: logs/research/level_approach_labels.csv
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import numpy as np

from ofi_tune_study import (  # noqa: F401 — shares DATA resolution + floors
    ACTIVITY_MIN,
    DATA,
    RATIO_MIN,
    SPEED_MIN,
    VOLIMB_MIN,
    Day,
    load_days,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(REPO, "logs", "research", "level_approach_labels.csv")

# Armed band geometry (mirror config defaults; kept literal so the dataset
# is reproducible independent of config churn — note any change here in the
# summary when comparing runs)
BAND_DEPTH = 0.06        # SIM_ANTICIPATION_MAX_PCT
BAND_TOL = 0.01          # SIM_LEVEL_CROSS_TOLERANCE_PCT
ARM_TTL_SECS = 10.0      # SIM_ARM_EVIDENCE_TTL_SECS
OFI_EVIDENCE_MIN = 0.70  # SIM_OFI_NORM_MIN (ema10 is the matching estimator)

APPROACH_PULL_MIN = 0.02   # dip below the level required before an approach
LEVEL_TTL_SECS = 900       # level relevance horizon after its pivot
TB_UP = 0.02               # upper barrier (fraction of event price)
TB_DN = 0.03               # lower barrier
TB_TIMEOUT_SECS = 120

SUPPLY_KEYS = ("ask_loss_r", "ask_add_r", "bid_loss_r", "bid_add_r",
               "buy_fill_r", "sell_fill_r", "nimb5")

# Progress feature (posterior fit, 2026-08-23): mirrors the production
# price_progress read — last price vs the last sample at/before 15s ago
# (PROGRESS_WINDOW_SECS), oldest-sample fallback when history spans >= 10s
# (PROGRESS_MIN_SPAN_SECS), not-ready otherwise.
PROG_WINDOW_MS = 15_000
PROG_MIN_SPAN_MS = 10_000


def _progress_at(day: Day, si: int) -> tuple[float, int]:
    """(progress fraction, ready flag) at sample index ``si``."""
    t_now = int(day.t[si])
    jp = int(np.searchsorted(day.t, t_now - PROG_WINDOW_MS, side="right")) - 1
    if jp >= 0:
        ref = float(day.price[jp])
    elif int(day.t[0]) <= t_now - PROG_MIN_SPAN_MS:
        ref = float(day.price[0])
    else:
        return 0.0, 0
    if ref <= 0:
        return 0.0, 0
    return float(day.price[si]) / ref - 1.0, 1


def _sample_index_for_grid(day: Day) -> np.ndarray:
    """For each grid second, the last 250ms-sample index at/before it."""
    t_s = (day.t // 1000).astype(np.int64)
    seconds = day.grid_t0 + np.arange(day.grid_n)
    return np.clip(np.searchsorted(t_s, seconds, side="right") - 1, 0, len(t_s) - 1)


def _events_for_day(day: Day, extras: dict[str, np.ndarray]) -> list[dict]:
    piv_highs = [(i, p) for i, p, kind in day._zigzag() if kind == "H"]
    gp = day.grid_price
    n = day.grid_n
    grid_to_sample = _sample_index_for_grid(day)
    z_speed, z_ratio = day.speed, day.ratio
    z = extras  # raw arrays incl. activity/volimb/accel/norm + supply

    events: list[dict] = []
    for i_pivot, level in piv_highs:
        lo = level * (1.0 - BAND_DEPTH)
        hi = level * (1.0 + BAND_TOL)
        end = min(n, i_pivot + LEVEL_TTL_SECS)
        dipped = False
        in_band = True  # at the pivot price is at the level — require a dip first
        for i in range(i_pivot + 1, end):
            px = gp[i]
            if px <= level * (1.0 - APPROACH_PULL_MIN):
                dipped = True
                in_band = False
                continue
            inside = lo <= px <= hi
            if inside and not in_band and dipped:
                # --- approach event at grid second i ---
                si = grid_to_sample[i]
                lookback = max(0, si - int(ARM_TTL_SECS * 4))  # 250ms grid
                sl = slice(lookback, si + 1)
                depth_now = bool(day.depth[si])
                armed_speed = bool(np.any(z_speed[sl] >= SPEED_MIN))
                armed_ratio = bool(np.any(z_ratio[sl] >= RATIO_MIN))
                armed_volimb = bool(np.any(z["volimb"][sl] > VOLIMB_MIN))
                armed_ofi = (
                    bool(np.any(z["ema10"][sl] >= OFI_EVIDENCE_MIN))
                    if depth_now else True  # vacuous, as in production
                )
                armed_recent = armed_speed and armed_ratio and armed_volimb and armed_ofi

                # Triple barrier from event price on the 1s grid
                up = px * (1.0 + TB_UP)
                dn = px * (1.0 - TB_DN)
                horizon = min(n, i + TB_TIMEOUT_SECS)
                fwd = gp[i + 1:horizon]
                label, t_out = 0, TB_TIMEOUT_SECS
                if len(fwd):
                    hit_up = np.argmax(fwd >= up) if np.any(fwd >= up) else -1
                    hit_dn = np.argmax(fwd <= dn) if np.any(fwd <= dn) else -1
                    if hit_up >= 0 and (hit_dn < 0 or hit_up <= hit_dn):
                        label, t_out = 1, hit_up + 1
                    elif hit_dn >= 0:
                        label, t_out = -1, hit_dn + 1
                f60 = gp[i + 1:min(n, i + 61)]
                f120 = gp[i + 1:min(n, i + 121)]
                # Realized return AT the outcome (barrier hit or timeout) —
                # includes gap-through beyond the barrier; the EV threshold
                # study needs actual payoffs, not barrier nominals.
                tb_ret = gp[min(n - 1, i + t_out)] / px - 1.0
                prog, prog_ready = _progress_at(day, si)
                row = {
                    "day": day.name,
                    "event_ts": datetime.fromtimestamp(
                        day.grid_t0 + i, tz=timezone.utc
                    ).isoformat(),
                    "level": round(level, 4),
                    "event_price": round(px, 4),
                    "rel_to_level": round(px / level - 1.0, 5),
                    "secs_since_pivot": i - i_pivot,
                    "depth_active": int(depth_now),
                    "speed": round(float(z_speed[si]), 2),
                    "ratio": round(float(z_ratio[si]), 4),
                    "volimb": round(float(z["volimb"][si]), 4),
                    "accelz": round(float(z["accelz"][si]), 2),
                    "accel_ready": int(z["accel_ready"][si]),
                    "activity": int(z["activity"][si]),
                    "ofi_ema10": round(float(z["ema10"][si]), 4),
                    "progress15": round(prog, 5),
                    "progress_ready": prog_ready,
                    "armed_speed": int(armed_speed),
                    "armed_ratio": int(armed_ratio),
                    "armed_volimb": int(armed_volimb),
                    "armed_ofi": int(armed_ofi),
                    "armed_recent": int(armed_recent),
                    "label": label,
                    "t_to_outcome_s": t_out,
                    "fwdmax60": round(float(f60.max() / px - 1.0), 5) if len(f60) else 0.0,
                    "fwdmax120": round(float(f120.max() / px - 1.0), 5) if len(f120) else 0.0,
                    "tb_ret": round(float(tb_ret), 5),
                    "fwd60_close": round(float(gp[min(n - 1, i + 60)] / px - 1.0), 5),
                    "fwd120_close": round(float(gp[min(n - 1, i + 120)] / px - 1.0), 5),
                }
                for key in SUPPLY_KEYS:
                    if key in z:
                        row[key] = round(float(z[key][si]), 5)
                events.append(row)
                in_band = True
                dipped = False
            elif not inside:
                in_band = False
    return events


def main() -> None:
    days = load_days()
    all_rows: list[dict] = []
    import glob as _glob
    paths = {os.path.basename(p).replace(".npz", ""): p
             for p in _glob.glob(os.path.join(DATA, "*.npz"))}
    for day in days:
        raw = np.load(paths[day.name])
        extras = {k: raw[k] for k in raw.files}
        all_rows.extend(_events_for_day(day, extras))

    if not all_rows:
        print("no events")
        return
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = list(all_rows[0].keys())
    for row in all_rows:  # supply cols may be absent on v1 data
        for f in row:
            if f not in fields:
                fields.append(f)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    lab = np.array([r["label"] for r in all_rows])
    fm120 = np.array([r["fwdmax120"] for r in all_rows])
    armed = np.array([r["armed_recent"] for r in all_rows], dtype=bool)
    print(f"days={len(days)} approach-events={len(all_rows)} -> {OUT_CSV}")

    def _line(name: str, mask: np.ndarray) -> None:
        k = int(mask.sum())
        if k == 0:
            print(f"  {name:24s} n=0")
            return
        print(
            f"  {name:24s} n={k:5d}  P(up-first)={np.mean(lab[mask] == 1):.3f}  "
            f"P(fwdmax120>=2%)={np.mean(fm120[mask] >= 0.02):.3f}  "
            f"P(down-first)={np.mean(lab[mask] == -1):.3f}"
        )

    _line("ALL approaches", np.ones(len(lab), dtype=bool))
    _line("armed_recent=1", armed)
    _line("armed_recent=0", ~armed)
    depth = np.array([r["depth_active"] for r in all_rows], dtype=bool)
    _line("armed=1 & depth", armed & depth)


if __name__ == "__main__":
    main()
