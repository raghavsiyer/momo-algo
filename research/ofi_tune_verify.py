"""OFI tuning — phase 3 (robustness verification of the chosen candidates)."""
from __future__ import annotations

import numpy as np

from ofi_tune_study import (
    VARIANTS, DEPTH_COVER_MIN, EXIT_MEAT_MIN, EXIT_RATIO_MAX,
    entry_zone_mask, load_days,
)

ENTRY_CANDS = [
    ("norm30>=1.0 (current)", "norm30_tracker", 1.0),
    ("norm15>=0.9", "norm15", 0.9),
    ("ema10>=0.7", "ema10", 0.7),
    ("ema5>=0.5", "ema5", 0.5),
]
EXIT_CANDS = [
    ("norm30<-0.6 (current)", "norm30_tracker", -0.6, False),
    ("norm30<-1.0", "norm30_tracker", -1.0, False),
    ("ema10<-0.75", "ema10", -0.75, False),
    ("ema10<-0.75&r<=.5", "ema10", -0.75, True),
    ("ema10<-0.6&r<=.5", "ema10", -0.6, True),
]


def main() -> None:
    days = load_days()

    legs = []
    for day in days:
        for leg in day.legs:
            zone = entry_zone_mask(day, leg)
            eligible = zone.sum() >= 4 and day.depth[zone].mean() >= DEPTH_COVER_MIN
            if eligible:
                legs.append((day, leg, zone))

    # gain tiers
    tiers = [(0.04, 0.06, "4-6%"), (0.06, 0.10, "6-10%"), (0.10, 9.9, ">10%")]

    print("=== ENTRY: coverage by leg-gain tier + added-delay tail ===")
    for label, v, th in ENTRY_CANDS:
        tier_cov = {t[2]: [0, 0] for t in tiers}
        added = []
        for day, leg, zone in legs:
            fires = day.tape_pass & day.depth & (day.vals[v] >= th)
            tname = next(t[2] for t in tiers if t[0] <= leg["gain"] < t[1])
            tier_cov[tname][1] += 1
            zf = fires & zone
            ztape = day.tape_pass & day.depth & zone
            if zf.any():
                tier_cov[tname][0] += 1
                if ztape.any():
                    added.append((day.t[np.argmax(zf)] - day.t[np.argmax(ztape)]) / 1000.0)
        a = np.array(added) if added else np.array([np.nan])
        cov_s = "  ".join(f"{k}:{c}/{n}" for k, (c, n) in tier_cov.items())
        print(f"{label:>24}  {cov_s}  +dly p50/p75/p90 = "
              f"{np.percentile(a,50):.1f}/{np.percentile(a,75):.1f}/{np.percentile(a,90):.1f}s")

    print("\n=== ENTRY: per-day paired coverage & noise vs current ===")
    cur_v, cur_th = "norm30_tracker", 1.0
    for label, v, th in ENTRY_CANDS[1:]:
        dcov, dnoise = [], []
        for day in days:
            day_legs = [(lg, zn) for d, lg, zn in legs if d is day]
            f_cur = day.tape_pass & day.depth & (day.vals[cur_v] >= cur_th)
            f_new = day.tape_pass & day.depth & (day.vals[v] >= th)
            c_cur = sum(1 for lg, zn in day_legs if (f_cur & zn).any())
            c_new = sum(1 for lg, zn in day_legs if (f_new & zn).any())
            dcov.append(c_new - c_cur)

            def noise_eps(f):
                fi = np.where(f)[0]
                if not len(fi):
                    return 0
                gi = day.sample_grid_idx[fi]
                in_any = np.zeros(len(fi), dtype=bool)
                for lg in day.legs:
                    in_any |= (gi >= lg["i0"] - 5) & (gi <= lg["i1"] + 60)
                out = fi[~in_any]
                if not len(out):
                    return 0
                return 1 + int(np.sum(np.diff(day.t[out]) > 5000))
            dnoise.append(noise_eps(f_new) - noise_eps(f_cur))
        dcov, dnoise = np.array(dcov), np.array(dnoise)
        print(f"{label:>24}  cov delta: sum {dcov.sum():+d} "
              f"(days better {int((dcov > 0).sum())}, worse {int((dcov < 0).sum())})  "
              f"noise delta: sum {dnoise.sum():+d} "
              f"(days better {int((dnoise < 0).sum())}, worse {int((dnoise > 0).sum())})")

    print("\n=== EXIT: premature & capture on big legs (gain >= 6%) ===")
    # entry fixed at current gate so exit deltas are attributable to the exit
    entries = []
    for day, leg, zone in legs:
        fires = day.tape_pass & day.depth & (day.vals["norm30_tracker"] >= 1.0)
        zf = fires & zone
        if zf.any() and leg["gain"] >= 0.06:
            entries.append((day, leg, int(np.argmax(zf))))
    print(f"entered big legs: {len(entries)}")
    for label, v, th, need_lapse in EXIT_CANDS:
        prem = 0
        captured = []
        for day, leg, ei in entries:
            exit_fire = day.depth & (day.vals[v] < th)
            if need_lapse:
                exit_fire &= day.ratio <= EXIT_RATIO_MAX
            gi = day.sample_grid_idx
            peak_i = leg["i1"]
            remaining = leg["p1"] / np.maximum(day.price, 1e-9) - 1.0
            pre = exit_fire.copy()
            pre[:ei + 1] = False
            pre &= (gi < peak_i) & (remaining >= EXIT_MEAT_MIN)
            if pre.any():
                prem += 1
            any_exit = exit_fire.copy()
            any_exit[:ei + 1] = False
            any_exit &= gi <= peak_i + 180
            p_entry = day.price[ei]
            if any_exit.any():
                k = int(np.argmax(any_exit))
                captured.append((day.price[k] / p_entry - 1.0) * 100)
            else:
                hi = min(peak_i + 180, day.grid_n - 1)
                captured.append((day.grid_price[hi] / p_entry - 1.0) * 100)
        n = max(len(entries), 1)
        cap = np.array(captured)
        print(f"{label:>24}  premat {100.0*prem/n:5.1f}%  capt med {np.median(cap):5.2f}%  "
              f"mean {cap.mean():5.2f}%  p10 {np.percentile(cap,10):6.2f}%")


if __name__ == "__main__":
    main()
