"""OFI tuning — phase 2 (study).

Reads the per-symbol-day npz built by ofi_tune_build.py and measures, for
each OFI estimator variant x threshold:

ENTRY (gate = tape_pass & depth_active & ofi >= theta):
  - squeeze coverage: legs (zigzag low->high, gain >= 4%) with a fire while
    >= 2% of the leg is still ahead
  - added delay vs the tape-only gate (the OFI gate's marginal latency)
  - remaining gain at first fire (are we entering with meat left?)
  - noise: fire episodes outside legs, expectancy (fwd60/fwd120) of fires,
    stale-ON rate (ofi >= theta while tape is dead)

EXIT (fire = depth_active & ofi < theta_x):
  - premature-exit rate: exit fires between entry and peak while >= 1.5%
    upside remains
  - post-peak reaction: give-back % from peak at first exit fire

Run:  python ofi_tune_study.py            # full sweep tables
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# v1 npz by default (reproduces the 2026-08-10 study); point at
# ofi_tune_v2 via OFI_TUNE_DATA to study the supply-side arrays.
DATA = os.environ.get("OFI_TUNE_DATA", os.path.join(REPO, "logs", "research", "ofi_tune"))

# Tape gate floors (config.py SIM_PRESSURE_*)
SPEED_MIN = 10.0
ACTIVITY_MIN = 50
RATIO_MIN = 0.60
VOLIMB_MIN = -0.20
ACCEL_Z = 1.07

LEG_GAIN_MIN = 0.04       # squeeze leg = zigzag leg gaining >= 4%
ZZ_REV = 0.025            # zigzag reversal threshold
LEG_MAX_SECS = 900
ENTRY_MEAT_MIN = 0.02     # entry zone: >= 2% of leg still ahead
EXIT_MEAT_MIN = 0.015     # premature exit if >= 1.5% still ahead
DEPTH_COVER_MIN = 0.50    # leg counts for OFI eval if >=50% entry zone depth-active

VARIANTS = ("norm8", "norm15", "norm30_tracker", "ema5", "ema10")
ENTRY_THRESHOLDS = {
    "norm8": [0.15, 0.25, 0.4, 0.6, 0.8],
    "norm15": [0.25, 0.4, 0.6, 0.8, 0.9, 1.0],
    "norm30_tracker": [0.5, 0.75, 1.0, 1.5, 2.0],
    "ema5": [0.1, 0.2, 0.3, 0.45, 0.5, 0.6],
    "ema10": [0.2, 0.3, 0.45, 0.5, 0.6, 0.7, 0.8],
}
EXIT_THRESHOLDS = {
    "norm8": [-0.6, -0.4, -0.25, -0.15],
    "norm15": [-0.8, -0.6, -0.4, -0.25],
    "norm30_tracker": [-1.2, -1.0, -0.9, -0.6, -0.4],
    "ema5": [-0.45, -0.3, -0.2, -0.1],
    "ema10": [-0.75, -0.6, -0.5, -0.45, -0.3, -0.2],
}
# Composite exits: OFI adverse AND buy dominance lapsed (ratio_gate <= 0.50,
# same band the accel-collapse exit uses via SIM_EXIT_ACCEL_BUY_RATIO_MAX).
EXIT_RATIO_MAX = 0.50


class Day:
    __slots__ = ("name", "t", "price", "grid_t0", "grid_price", "grid_n",
                 "sample_grid_idx", "tape_pass", "depth", "vals", "speed",
                 "ratio", "legs")

    def __init__(self, path: str) -> None:
        z = np.load(path)
        self.name = os.path.basename(path).replace(".npz", "")
        self.t = z["t"]  # ms
        self.price = z["price"].astype(np.float64)
        self.depth = z["depth_active"].astype(bool)
        self.speed = z["speed"]
        self.ratio = z["ratio"]
        self.tape_pass = (
            (z["speed"] >= SPEED_MIN)
            & (z["activity"] >= ACTIVITY_MIN)
            & (z["ratio"] >= RATIO_MIN)
            & (z["volimb"] > VOLIMB_MIN)
            & (z["accel_ready"] == 1)
            & (z["accelz"] >= ACCEL_Z)
        )
        self.vals = {v: z[v].astype(np.float64) for v in VARIANTS}

        # 1s forward-filled price grid for labeling + forward returns
        t_s = (self.t // 1000).astype(np.int64)
        self.grid_t0 = int(t_s[0])
        self.grid_n = int(t_s[-1]) - self.grid_t0 + 1
        gp = np.empty(self.grid_n, dtype=np.float64)
        idx = t_s - self.grid_t0
        gp[:] = np.nan
        gp[idx] = self.price  # last sample in each second wins
        # forward fill
        mask = np.isnan(gp)
        valid = np.where(~mask, np.arange(self.grid_n), 0)
        np.maximum.accumulate(valid, out=valid)
        gp = gp[valid]
        self.grid_price = gp
        self.sample_grid_idx = idx
        self.legs = self._label_legs()

    def _zigzag(self) -> list[tuple[int, float, str]]:
        p = self.grid_price
        piv: list[tuple[int, float, str]] = []
        ext_hi_i = ext_lo_i = 0
        ext_hi = ext_lo = p[0]
        trend = 0  # 0 unknown, +1 up (tracking high), -1 down (tracking low)
        for i in range(1, len(p)):
            x = p[i]
            if trend >= 0:
                if x > ext_hi:
                    ext_hi, ext_hi_i = x, i
                if trend == 0 and x < ext_lo:
                    ext_lo, ext_lo_i = x, i
                if x <= ext_hi * (1 - ZZ_REV):
                    piv.append((ext_hi_i, ext_hi, "H"))
                    trend = -1
                    ext_lo, ext_lo_i = x, i
            else:
                if x < ext_lo:
                    ext_lo, ext_lo_i = x, i
                if x >= ext_lo * (1 + ZZ_REV):
                    piv.append((ext_lo_i, ext_lo, "L"))
                    trend = 1
                    ext_hi, ext_hi_i = x, i
            if trend == 0 and x >= ext_lo * (1 + ZZ_REV):
                piv.append((ext_lo_i, ext_lo, "L"))
                trend = 1
                ext_hi, ext_hi_i = x, i
        return piv

    def _label_legs(self) -> list[dict]:
        piv = self._zigzag()
        legs = []
        for a, b in zip(piv, piv[1:]):
            if a[2] == "L" and b[2] == "H":
                i0, p0 = a[0], a[1]
                i1, p1 = b[0], b[1]
                gain = p1 / p0 - 1.0
                if gain >= LEG_GAIN_MIN and (i1 - i0) <= LEG_MAX_SECS:
                    legs.append({"i0": i0, "i1": i1, "p0": p0, "p1": p1, "gain": gain})
        return legs


def load_days() -> list[Day]:
    days = []
    for path in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
        try:
            days.append(Day(path))
        except Exception as exc:
            print(f"[skip] {path}: {exc!r}")
    return days


def fwd_return(day: Day, gi: int, horizon: int) -> tuple[float, float]:
    """(signed return at horizon, max favorable within horizon) from grid idx."""
    p0 = day.grid_price[gi]
    hi = min(gi + horizon, day.grid_n - 1)
    if hi <= gi or p0 <= 0:
        return 0.0, 0.0
    seg = day.grid_price[gi:hi + 1]
    return seg[-1] / p0 - 1.0, seg.max() / p0 - 1.0


def entry_zone_mask(day: Day, leg: dict) -> np.ndarray:
    """Boolean over samples: inside [i0,i1] with >= ENTRY_MEAT_MIN remaining."""
    gi = day.sample_grid_idx
    in_leg = (gi >= leg["i0"]) & (gi <= leg["i1"])
    remaining = leg["p1"] / np.maximum(day.price, 1e-9) - 1.0
    return in_leg & (remaining >= ENTRY_MEAT_MIN)


def study_entry(days: list[Day]) -> None:
    # Pre-compute per-leg sample masks + depth coverage
    all_legs = []  # (day, leg, zone_mask, eligible)
    for day in days:
        for leg in day.legs:
            zone = entry_zone_mask(day, leg)
            n = zone.sum()
            eligible = n >= 4 and day.depth[zone].mean() >= DEPTH_COVER_MIN
            all_legs.append((day, leg, zone, eligible))

    n_elig = sum(1 for *_x, e in all_legs if e)
    print(f"days={len(days)} legs(gain>=4%)={len(all_legs)} depth-eligible={n_elig}")

    # tape-only baseline
    rows = []
    variants = [("tape-only", None, None)]
    for v in VARIANTS:
        for th in ENTRY_THRESHOLDS[v]:
            variants.append((v, v, th))

    for label, v, th in variants:
        covered = 0
        delays, meats, added = [], [], []
        fire_fwd60, fire_fwd120 = [], []
        noise_ep = 0
        fire_total = 0
        stale_on = 0
        stale_denom = 0
        for day in days:
            fires = day.tape_pass & day.depth
            if v is not None:
                fires = fires & (day.vals[v] >= th)
            # legs
            leg_windows = []
            for d2, leg, zone, eligible in all_legs:
                if d2 is not day:
                    continue
                leg_windows.append((leg["i0"], leg["i1"]))
                if not eligible:
                    continue
                zf = fires & zone
                if zf.any():
                    covered += 1
                    i_first = np.argmax(zf)
                    t_first = day.t[i_first]
                    delays.append((day.sample_grid_idx[i_first] - leg["i0"]))
                    meats.append(leg["p1"] / day.price[i_first] - 1.0)
                    # added delay vs tape-only inside the same zone
                    ztape = day.tape_pass & day.depth & zone
                    if ztape.any():
                        added.append((t_first - day.t[np.argmax(ztape)]) / 1000.0)
            # noise episodes: fires whose grid idx not inside any leg [i0, i1+60]
            fi = np.where(fires)[0]
            fire_total += len(fi)
            if len(fi):
                gi = day.sample_grid_idx[fi]
                in_any = np.zeros(len(fi), dtype=bool)
                for (i0, i1) in leg_windows:
                    in_any |= (gi >= i0 - 5) & (gi <= i1 + 60)
                out = fi[~in_any]
                # group into episodes (>5s gap starts new)
                if len(out):
                    tt = day.t[out]
                    noise_ep += 1 + int(np.sum(np.diff(tt) > 5000))
                for k in fi:
                    f60 = fwd_return(day, day.sample_grid_idx[k], 60)[0]
                    f120 = fwd_return(day, day.sample_grid_idx[k], 120)[0]
                    fire_fwd60.append(f60)
                    fire_fwd120.append(f120)
            # stale-ON: ofi still >= th while tape is dead (speed < 3)
            if v is not None:
                dead = day.depth & (day.speed < 3.0)
                stale_denom += int(dead.sum())
                stale_on += int((dead & (day.vals[v] >= th)).sum())

        med = lambda a: float(np.median(a)) if len(a) else float("nan")
        f60 = np.array(fire_fwd60) * 100 if fire_fwd60 else np.array([np.nan])
        rows.append((
            label if v is None else f"{v}>={th}",
            covered, n_elig,
            med(delays), med(added), med(meats) * 100 if meats else float("nan"),
            noise_ep, fire_total,
            float(np.mean(f60)), float(np.percentile(f60, 10)),
            (100.0 * stale_on / stale_denom) if stale_denom else float("nan"),
        ))

    print(f"\n{'variant':>22} {'cov':>7} {'dly_s':>6} {'+dly':>6} {'meat%':>6} "
          f"{'noiseEp':>7} {'fires':>7} {'fwd60%':>7} {'p10':>6} {'staleON%':>8}")
    for r in rows:
        print(f"{r[0]:>22} {r[1]:>3}/{r[2]:<3} {r[3]:>6.1f} {r[4]:>6.1f} {r[5]:>6.2f} "
              f"{r[6]:>7} {r[7]:>7} {r[8]:>7.2f} {r[9]:>6.2f} {r[10]:>8.2f}")


def study_exit(days: list[Day], entry_variant: str = "norm30_tracker",
               entry_th: float = 1.0) -> None:
    # Entry per leg: first tape+OFI fire in zone (fixed entry policy so
    # the exit comparison is apples-to-apples across exit variants).
    print(f"\nEXIT sweep (entry fixed: tape_pass & {entry_variant}>={entry_th}):")
    entries = []  # (day, leg, sample idx of entry)
    for day in days:
        fires = day.tape_pass & day.depth & (day.vals[entry_variant] >= entry_th)
        for leg in day.legs:
            zone = entry_zone_mask(day, leg)
            if zone.sum() < 4 or day.depth[zone].mean() < DEPTH_COVER_MIN:
                continue
            zf = fires & zone
            if zf.any():
                entries.append((day, leg, int(np.argmax(zf))))
    print(f"entered legs: {len(entries)}")

    # (label, variant, threshold, require_ratio_lapse)
    exit_defs = []
    for v in VARIANTS:
        for th in EXIT_THRESHOLDS[v]:
            exit_defs.append((f"{v}<{th}", v, th, False))
            exit_defs.append((f"{v}<{th}&r<=.5", v, th, True))

    header = (f"{'variant':>24} {'premat%':>8} {'foregone%':>9} {'giveback%':>9} "
              f"{'react_s':>8} {'miss%':>6} {'capt%':>7}")
    print(header)
    for label, v, th, need_lapse in exit_defs:
        prem = 0
        foregone, giveback, react, captured = [], [], [], []
        missed = 0
        for day, leg, ei in entries:
            exit_fire = day.depth & (day.vals[v] < th)
            if need_lapse:
                exit_fire &= day.ratio <= EXIT_RATIO_MAX
            gi = day.sample_grid_idx
            peak_i = leg["i1"]
            p_entry = day.price[ei]
            # premature: exit fire strictly after entry, before peak, meat >= 1.5%
            remaining = leg["p1"] / np.maximum(day.price, 1e-9) - 1.0
            pre = exit_fire.copy()
            pre[:ei + 1] = False
            pre &= (gi < peak_i) & (remaining >= EXIT_MEAT_MIN)
            if pre.any():
                prem += 1
                k = int(np.argmax(pre))
                foregone.append(remaining[k] * 100)
            # reaction after peak (within 180s)
            post = exit_fire.copy()
            post &= (gi > peak_i) & (gi <= peak_i + 180)
            if post.any():
                k = int(np.argmax(post))
                giveback.append((1.0 - day.price[k] / leg["p1"]) * 100)
                react.append(gi[k] - peak_i)
            else:
                missed += 1
                hi = min(peak_i + 180, day.grid_n - 1)
                giveback.append((1.0 - day.grid_price[hi] / leg["p1"]) * 100)
            # captured: OFI exit as the ONLY exit — first fire after entry,
            # else forced flat at peak+180s
            any_exit = exit_fire.copy()
            any_exit[:ei + 1] = False
            any_exit &= gi <= peak_i + 180
            if any_exit.any():
                k = int(np.argmax(any_exit))
                captured.append((day.price[k] / p_entry - 1.0) * 100)
            else:
                hi = min(peak_i + 180, day.grid_n - 1)
                captured.append((day.grid_price[hi] / p_entry - 1.0) * 100)
        n = max(len(entries), 1)
        med = lambda a: float(np.median(a)) if len(a) else float("nan")
        print(f"{label:>24} {100.0*prem/n:>8.1f} {med(foregone):>9.2f} "
              f"{med(giveback):>9.2f} {med(react):>8.1f} {100.0*missed/n:>6.1f} "
              f"{med(captured):>7.2f}")


if __name__ == "__main__":
    days = load_days()
    study_entry(days)
    study_exit(days)
