"""Scratch harness: entry/exit knife-edge study (F6). SHIPPED: V2 (fast<=0.50).

Simulates entry->exit episodes on the 40 cached tapes, varying ONLY the
exit-1b (accel-collapse) ratio term, to measure the cost of the shared
0.60 boundary between the entry ratio gate (buy_ratio_ema >= 0.60) and
exit 1b (boxcar buy_ratio <= 0.60).

Entry (level semantics, matches live evaluate_entry tape stack):
  ready & ze >= 1.07 & spd_ema >= 10 & p_fast >= 0.60 & vol_ok & act >= 50
Exits evaluated per print while in position (tape-only; OFI/confidence/
resistance exits are not simulatable from tape and are held fixed anyway):
  exit 1  (all variants): n10 >= 50 and boxcar sell share >= 0.60
  exit 1b (variant):      ready & accel_raw <= -0.07 & <ratio term>
    V0 current:   boxcar buy ratio <= 0.60      [knife edge]
    V1 band:      boxcar buy ratio <= 0.50
    V2 fast-band: fast-EMA buy share <= 0.50
    V3 slow-band: slow-EMA (tau=10s) buy share <= 0.50
    V4 wilson:    Wilson-LB(sell share, Kish n_eff fast, z=1.96) >= 0.50

Usage:
    from _tape_exit_scratch import phase1b_all, study
    phase1b_all()   # builds /tmp/tape_exit/{SYM}_{DAY}.npz  (accel/p_slow/...)
    study()         # episode sim + comparison table
"""
from __future__ import annotations

import math
import os

import numpy as np

import _tape_ratio_scratch as trs
from config import (
    SIM_EXIT_ACCEL_MAX,
    SIM_EXIT_SELL_RATIO_MIN,
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_SPEED_MIN,
    TAPE_ACCEL_EMA_MIN_NEFF,
    TAPE_EMA_FAST_TAU_SECS,
    TAPE_EMA_SLOW_TAU_SECS,
)

RATIO_DIR = "/tmp/tape_ratio"
OUT_DIR = "/tmp/tape_exit"

TAU_F = TAPE_EMA_FAST_TAU_SECS      # 2.0
TAU_S = TAPE_EMA_SLOW_TAU_SECS      # 10.0
VOL_MIN_SHARE = 0.40                # volr30 > 0.40 <=> trade_imbalance > -0.20


def phase1b(symbol: str, day: str) -> str:
    """Extend the F5 trace with exit-side quantities (read-after-update)."""
    z = np.load(os.path.join(RATIO_DIR, f"{symbol}_{day}.npz"))
    epoch = z["epoch"].astype(np.float64)
    n = len(epoch)
    side, _size = trs._side_size(symbol, day, n)

    accel_raw = np.zeros(n, dtype=np.float32)
    p_slow = np.full(n, 0.5, dtype=np.float32)
    neff_fast = np.zeros(n, dtype=np.float32)

    c_fast = c_slow = c_slow_sq = 0.0     # side-agnostic directional kernels
    s2_fast = 0.0                          # fast kernel sum of squared weights
    wb_f = ws_f = 0.0                      # fast kernel by side (for neff join)
    wb_s = ws_s = 0.0                      # slow kernel by side
    prev = epoch[0]
    for i in range(n):
        dt = epoch[i] - prev
        if dt > 0:
            df = math.exp(-dt / TAU_F)
            ds = math.exp(-dt / TAU_S)
            c_fast *= df
            s2_fast *= df * df
            wb_f *= df
            ws_f *= df
            c_slow *= ds
            c_slow_sq *= ds * ds
            wb_s *= ds
            ws_s *= ds
            prev = epoch[i]
        s = side[i]
        if s != 0:
            c_fast += 1.0
            s2_fast += 1.0
            c_slow += 1.0
            c_slow_sq += 1.0
            if s == 1:
                wb_f += 1.0
                wb_s += 1.0
            else:
                ws_f += 1.0
                ws_s += 1.0
        # reads (production property semantics)
        if c_slow_sq > 0:
            neff_slow = c_slow * c_slow / c_slow_sq
            if neff_slow >= TAPE_ACCEL_EMA_MIN_NEFF:
                r_f = c_fast / TAU_F
                r_s = c_slow / TAU_S
                tot = r_f + r_s
                if tot > 0:
                    accel_raw[i] = (r_f - r_s) / tot
        tot_s = wb_s + ws_s
        if tot_s > 1e-9:
            p_slow[i] = wb_s / tot_s
        if s2_fast > 0:
            neff_fast[i] = (wb_f + ws_f) ** 2 / s2_fast

    # trailing 10s directional print count (exit-1 activity floor)
    dirs = (side != 0).astype(np.float64)
    n10 = trs._window_sums(epoch, dirs, -10.0, 0.0).astype(np.int32)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{symbol}_{day}.npz")
    np.savez_compressed(out, accel_raw=accel_raw, p_slow=p_slow,
                        neff_fast=neff_fast, n10=n10)
    return out


def phase1b_all(force: bool = False) -> None:
    for sym, day in trs._tapes():
        out = os.path.join(OUT_DIR, f"{sym}_{day}.npz")
        if not force and os.path.exists(out):
            continue
        phase1b(sym, day)
        print("done", sym, day, flush=True)


# ---------------------------------------------------------------------------
# Episode simulation
# ---------------------------------------------------------------------------

def _load_joined() -> list[dict]:
    tapes = []
    for sym, day in trs._tapes():
        key = f"{sym}_{day}"
        z1 = np.load(os.path.join(RATIO_DIR, f"{key}.npz"))
        z2 = np.load(os.path.join(OUT_DIR, f"{key}.npz"))
        d = {k: z1[k] for k in z1.files}
        d.update({k: z2[k] for k in z2.files})
        d["key"] = key
        # price is not stored in the ratio npz; rebuild from dispersion trace
        zd = np.load(os.path.join(trs.DISP_DIR, f"{key}.npz"))
        d["price"] = zd["price"].astype(np.float64)
        tapes.append(d)
    return tapes


def _episodes(epoch, price, entry_mask, exit_mask):
    """Walk entry->exit episodes. Returns list of (i, j, forced_end)."""
    e_idx = np.flatnonzero(entry_mask)
    x_idx = np.flatnonzero(exit_mask)
    out = []
    pos = 0                       # next candidate entry pointer into e_idx
    n = len(epoch)
    while pos < len(e_idx):
        i = e_idx[pos]
        k = np.searchsorted(x_idx, i + 1)
        if k >= len(x_idx):
            out.append((i, n - 1, True))
            break
        j = x_idx[k]
        out.append((i, j, False))
        pos = np.searchsorted(e_idx, j + 1)
    return out


def _entry_points(epoch, entry_mask, refractory_secs: float = 60.0) -> np.ndarray:
    """First full-stack pass per opportunity: rising edges with a refractory.

    Live enforces one-buy-per-signal (a traded setup is never re-entered;
    exec/engine.py _evaluate_entry), so exit-policy churn loops are
    impossible. Each entry here = one opportunity; entries less than
    refractory_secs after the previous accepted entry are the same move.
    """
    idx = np.flatnonzero(entry_mask)
    if len(idx) == 0:
        return idx
    out = [idx[0]]
    last_t = epoch[idx[0]]
    for i in idx[1:]:
        if epoch[i] - last_t >= refractory_secs:
            out.append(i)
            last_t = epoch[i]
    return np.asarray(out)


def study(refractory_secs: float = 60.0) -> None:
    """One-shot trades on an IDENTICAL entry set; only the exit varies."""
    tapes = _load_joined()

    def variants(d):
        acc_ok = d["ready"] & (d["accel_raw"] <= SIM_EXIT_ACCEL_MAX)
        sell_box = 1.0 - d["bratio"]
        exit1 = (d["n10"] >= SIM_PRESSURE_ACTIVITY_MIN) & (sell_box >= SIM_EXIT_SELL_RATIO_MIN)
        wl_sell = trs.wilson_lb(1.0 - d["p_fast"], d["neff_fast"], 1.96)
        return {
            "V0 box<=0.60 (cur)": exit1 | (acc_ok & (d["bratio"] <= 0.60)),
            "V1 box<=0.50":       exit1 | (acc_ok & (d["bratio"] <= 0.50)),
            "V2 fast<=0.50":      exit1 | (acc_ok & (d["p_fast"] <= 0.50)),
            "V3 slow<=0.50":      exit1 | (acc_ok & (d["p_slow"] <= 0.50)),
            "V4 wilsonLB>=0.50":  exit1 | (acc_ok & (wl_sell >= 0.50)),
            "V5 exit1-only":      exit1,
        }

    names = None
    agg: dict[str, dict] = {}
    n_entries = 0
    for d in tapes:
        ep, px = d["epoch"], d["price"]
        vol_ok = ~(np.nan_to_num(d["volr30"], nan=1.0) <= VOL_MIN_SHARE)
        entry = (d["ready"] & (d["ze"] >= SIM_PRESSURE_ACCEL_Z)
                 & (d["spd_ema"] >= SIM_PRESSURE_SPEED_MIN)
                 & (d["p_fast"] >= SIM_PRESSURE_BUY_RATIO_MIN)
                 & vol_ok & (d["act"] >= SIM_PRESSURE_ACTIVITY_MIN))
        e_pts = _entry_points(ep, entry, refractory_secs)
        n_entries += len(e_pts)
        vmasks = variants(d)
        names = list(vmasks)
        for name, xm in vmasks.items():
            a = agg.setdefault(name, {
                "ret": [], "hold": [], "fast_exit": 0, "n": 0, "forced": 0,
                "giveup": [], "capture": [], "exit1b": 0,
            })
            x_idx = np.flatnonzero(xm)
            for i in e_pts:
                k = np.searchsorted(x_idx, i + 1)
                forced = k >= len(x_idx)
                j = len(ep) - 1 if forced else x_idx[k]
                a["n"] += 1
                a["ret"].append(px[j] / px[i] - 1.0)
                a["hold"].append(ep[j] - ep[i])
                if forced:
                    a["forced"] += 1
                if ep[j] - ep[i] <= 5.0:
                    a["fast_exit"] += 1
                exit1_here = (d["n10"][j] >= SIM_PRESSURE_ACTIVITY_MIN
                              and (1.0 - d["bratio"][j]) >= SIM_EXIT_SELL_RATIO_MIN)
                if not forced and not exit1_here:
                    a["exit1b"] += 1
                kk = np.searchsorted(ep, ep[j] + 30.0, side="right") - 1
                if ep[kk] >= ep[j] + 25.0:
                    a["giveup"].append(px[kk] / px[j] - 1.0)
                hi = px[i:j + 1].max()
                if hi > px[i] * 1.001:
                    a["capture"].append((px[j] - px[i]) / (hi - px[i]))

    print(f"one-shot trades, identical entries (n={n_entries}, refractory={refractory_secs:.0f}s), "
          f"{len(tapes)} tapes; tape-only exits, no spread")
    hdr = (f"{'variant':<20}{'<=5s%':>7}{'1b%':>6}"
           f"{'ret_med':>9}{'ret_mean':>9}{'p10':>8}{'p90':>8}{'win%':>6}"
           f"{'hold_med':>9}{'hold_p90':>9}{'giveup30':>9}{'capture':>8}{'forced':>7}")
    print(hdr)
    for name in names:
        a = agg[name]
        ret = np.array(a["ret"]) * 100
        hold = np.array(a["hold"])
        gu = np.array(a["giveup"]) * 100
        cap = np.array(a["capture"])
        print(f"{name:<20}{a['fast_exit'] / max(a['n'], 1) * 100:>7.1f}"
              f"{a['exit1b'] / max(a['n'], 1) * 100:>6.1f}"
              f"{np.median(ret):>9.3f}{ret.mean():>9.3f}"
              f"{np.percentile(ret, 10):>8.3f}{np.percentile(ret, 90):>8.3f}"
              f"{(ret > 0).mean() * 100:>6.1f}"
              f"{np.median(hold):>9.2f}{np.percentile(hold, 90):>9.2f}"
              f"{gu.mean():>9.3f}{cap.mean():>8.3f}{a['forced']:>7d}")


if __name__ == "__main__":
    phase1b_all()
    study()
