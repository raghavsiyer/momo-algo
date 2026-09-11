"""Scratch harness: buy-ratio gate study (F5).

Compares three ratio-gate formulations on identical tape input, using the
existing /tmp/tape_dispersion npz traces (per-print, 1:1 with CSV rows)
joined with per-print side/size re-derived vectorized from the tape CSVs
(same _classify rule as OrderFlowTracker: >=ask buy, <=bid sell, else mid).

Variants:
  A. boxcar:  buy_ratio (10s boxcar count ratio) >= 0.60      [current prod]
  B. fast-pt: fast-EMA (tau=2s) count ratio point read >= thr  [straw man]
  C. wilson:  Wilson lower bound of fast-EMA ratio, with Kish n_eff of the
              fast kernel divided by phi (clipped cv2 from the dispersion
              study), gate LB_z >= 0.50                        [proposal]

Metrics (level-gate, evaluated at directional prints in "context" =
speed_ema >= SPEED_MIN and activity >= ACTIVITY_MIN, i.e. where the ratio
gate is the deciding vote):
  - pooled pass rate (B and C calibrated to match A's pooled pass rate)
  - noise-arm: gate ON while centered +/-3s true ratio <= 0.50 (>=15 prints)
  - flip-stale: gate ON while centered true ratio <= 0.40
  - surge coverage + arming latency vs price surges (_detect_surges)
  - full-gate fires (speed_ema & act & ratio & edge-triggered accel z)
Divergence study: among count-gate-ON context prints, trailing-30s VOLUME
ratio buckets vs forward 30s/60s returns.

Usage:
    from _tape_ratio_scratch import phase1_all, aggregate
    phase1_all()          # builds /tmp/tape_ratio/{SYM}_{DAY}.npz
    aggregate()           # calibrates + prints comparison tables
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from _tape_accel_compare_scratch import _detect_surges
from config import (
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_SPEED_MIN,
    TAPE_EMA_FAST_TAU_SECS,
    TAPE_PHI_CLIP_HI,
)

CACHE_DIR = "/Users/ahariharan/Desktop/momo_algo/logs/chart_cache"
DISP_DIR = "/tmp/tape_dispersion"
OUT_DIR = "/tmp/tape_ratio"

TAU = TAPE_EMA_FAST_TAU_SECS            # 2.0s fast kernel
Z_FIRE = SIM_PRESSURE_ACCEL_Z           # 1.07 (ema)
Z_REARM = 1.0
TRUTH_HALF_SECS = 3.0
TRUTH_MIN_PRINTS = 15
VOL_WINDOW_SECS = 30.0


def _tapes() -> list[tuple[str, str]]:
    out = []
    for f in sorted(os.listdir(DISP_DIR)):
        if f.endswith(".npz"):
            sym, day = f[:-4].rsplit("_", 1)
            out.append((sym, day))
    return out


def _side_size(symbol: str, day: str, n_expect: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-print side (+1 buy / -1 sell / 0 mid) and size."""
    df = pd.read_csv(os.path.join(CACHE_DIR, f"timesales_{symbol}_{day}.csv"))
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    assert len(df) == n_expect, f"{symbol}_{day}: csv {len(df)} != npz {n_expect}"
    price = df["price"].to_numpy(dtype=np.float64)
    bid = df["bid"].fillna(0.0).to_numpy(dtype=np.float64)
    ask = df["ask"].fillna(0.0).to_numpy(dtype=np.float64)
    size = df["size"].fillna(0).to_numpy(dtype=np.int64)
    side = np.zeros(n_expect, dtype=np.int8)
    side[(ask > 0) & (price >= ask)] = 1
    side[(side == 0) & (bid > 0) & (price <= bid)] = -1
    return side, size


def _ratio_traces(epoch: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast-kernel EMA buy-ratio, Kish n_eff (fast kernel), and speed_ema.

    Read-after-update at each print: decay by dt since previous print, add
    the print if directional (matches production decayed-read semantics).
    """
    n = len(epoch)
    p_fast = np.full(n, 0.5, dtype=np.float32)
    neff = np.zeros(n, dtype=np.float32)
    spd_ema = np.zeros(n, dtype=np.float32)
    wb = 0.0
    ws = 0.0
    s2 = 0.0
    prev = epoch[0]
    for i in range(n):
        dt = epoch[i] - prev
        if dt > 0:
            d = math.exp(-dt / TAU)
            wb *= d
            ws *= d
            s2 *= d * d
            prev = epoch[i]
        s = side[i]
        if s == 1:
            wb += 1.0
            s2 += 1.0
        elif s == -1:
            ws += 1.0
            s2 += 1.0
        tot = wb + ws
        if tot > 1e-9:
            p_fast[i] = wb / tot
            neff[i] = tot * tot / s2
        spd_ema[i] = tot / TAU
    return p_fast, neff, spd_ema


def _window_sums(epoch: np.ndarray, vals: np.ndarray, lo_off: float, hi_off: float) -> np.ndarray:
    """Sum of vals over epoch in [t+lo_off, t+hi_off] per print (inclusive-ish)."""
    c = np.concatenate([[0.0], np.cumsum(vals, dtype=np.float64)])
    lo = np.searchsorted(epoch, epoch + lo_off, side="left")
    hi = np.searchsorted(epoch, epoch + hi_off, side="right")
    return c[hi] - c[lo]


def phase1(symbol: str, day: str) -> str:
    z = np.load(os.path.join(DISP_DIR, f"{symbol}_{day}.npz"))
    epoch = z["epoch"].astype(np.float64)
    price = z["price"].astype(np.float64)
    is_dir = z["is_dir"]
    n = len(epoch)

    side, size = _side_size(symbol, day, n)
    # sanity: side classification must reproduce the tracker's is_dir
    mismatch = int(((side != 0) != is_dir).sum())
    assert mismatch == 0, f"{symbol}_{day}: side/is_dir mismatch on {mismatch} prints"

    p_fast, neff_fast, spd_ema = _ratio_traces(epoch, side)
    # production semantics: phi defaults to 1.0 until the dispersion
    # accumulators are warm (raw cv2 is NaN there), inf clips to the ceiling
    cv2 = np.nan_to_num(z["cv2"].astype(np.float32), nan=1.0)
    phi = np.clip(cv2, 1.0, TAPE_PHI_CLIP_HI)
    neff_phi = neff_fast / phi

    # centered +/-3s truth (counts)
    dirs = (side != 0).astype(np.float64)
    buys = (side == 1).astype(np.float64)
    t_dir = _window_sums(epoch, dirs, -TRUTH_HALF_SECS, TRUTH_HALF_SECS)
    t_buy = _window_sums(epoch, buys, -TRUTH_HALF_SECS, TRUTH_HALF_SECS)
    truth = np.where(t_dir >= TRUTH_MIN_PRINTS, t_buy / np.maximum(t_dir, 1.0), np.nan).astype(np.float32)

    # trailing 30s volume ratio
    bvol = np.where(side == 1, size, 0).astype(np.float64)
    svol = np.where(side == -1, size, 0).astype(np.float64)
    w_bv = _window_sums(epoch, bvol, -VOL_WINDOW_SECS, 0.0)
    w_sv = _window_sums(epoch, svol, -VOL_WINDOW_SECS, 0.0)
    vtot = w_bv + w_sv
    volr30 = np.where(vtot > 0, w_bv / np.maximum(vtot, 1.0), np.nan).astype(np.float32)

    # forward returns (nan within 60s of tape end)
    def fwd(secs: float) -> np.ndarray:
        idx = np.searchsorted(epoch, epoch + secs, side="right") - 1
        r = (price[idx] / np.maximum(price, 1e-9) - 1.0).astype(np.float32)
        r[epoch + secs > epoch[-1]] = np.nan
        return r

    fwd30, fwd60 = fwd(30.0), fwd(60.0)

    surges = _detect_surges(epoch, is_dir, price)
    t0 = math.floor(epoch[0])
    surge_on = np.array([t0 + s["onset"] for s in surges], dtype=np.float64)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{symbol}_{day}.npz")
    np.savez_compressed(
        out,
        epoch=epoch, is_dir=(side != 0),
        p_fast=p_fast, neff_phi=neff_phi.astype(np.float32),
        spd_ema=spd_ema,
        bratio=z["bratio"], act=z["act"], ze=z["ze"], ready=z["ready"],
        truth=truth, volr30=volr30, fwd30=fwd30, fwd60=fwd60,
        surge_on=surge_on,
    )
    return out


def phase1_all(force: bool = False) -> None:
    for sym, day in _tapes():
        out = os.path.join(OUT_DIR, f"{sym}_{day}.npz")
        if not force and os.path.exists(out):
            continue
        phase1(sym, day)
        print("done", sym, day, flush=True)


# ---------------------------------------------------------------------------
# Phase 2: aggregation
# ---------------------------------------------------------------------------

def wilson_lb(p: np.ndarray, n: np.ndarray, zc: float) -> np.ndarray:
    n = np.maximum(n, 1e-9)
    z2 = zc * zc
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    rad = zc * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (center - rad) / denom


def _edge_fires_np(epoch, ready, ze, gates_on) -> np.ndarray:
    """Edge-triggered accel-z fires ANDed with level gates. Returns epochs."""
    armed = True
    out = []
    for i in range(len(epoch)):
        zi = ze[i]
        if not ready[i]:
            continue
        if armed and zi >= Z_FIRE:
            if gates_on[i]:
                out.append(epoch[i])
            armed = False
        elif not armed and zi < Z_REARM:
            armed = True
    return np.asarray(out)


def _load_all() -> list[dict]:
    out = []
    for sym, day in _tapes():
        p = os.path.join(OUT_DIR, f"{sym}_{day}.npz")
        z = np.load(p)
        d = {k: z[k] for k in z.files}
        d["key"] = f"{sym}_{day}"
        out.append(d)
    return out


def aggregate(z_grid: np.ndarray | None = None) -> None:
    tapes = _load_all()
    thr = SIM_PRESSURE_BUY_RATIO_MIN                    # 0.60

    # context: directional prints where the other level gates pass
    for d in tapes:
        d["ctx"] = (
            d["is_dir"]
            & (d["spd_ema"] >= SIM_PRESSURE_SPEED_MIN)
            & (d["act"] >= SIM_PRESSURE_ACTIVITY_MIN)
        )

    def pooled_pass(gate_key_fn) -> float:
        num = den = 0
        for d in tapes:
            g = gate_key_fn(d)[d["ctx"]]
            num += int(g.sum())
            den += len(g)
        return num / max(den, 1)

    target = pooled_pass(lambda d: d["bratio"] >= thr)

    # calibrate fast point threshold and wilson z to match pooled pass rate
    def match_scalar(fn, grid):
        best, best_err = None, 1e9
        for v in grid:
            r = pooled_pass(lambda d, v=v: fn(d, v))
            err = abs(r - target)
            if err < best_err:
                best, best_err = v, err
        return best

    pt_thr = match_scalar(lambda d, v: d["p_fast"] >= v, np.arange(0.50, 0.80, 0.005))
    if z_grid is None:
        z_grid = np.arange(0.0, 3.0, 0.05)
    wl_z = match_scalar(lambda d, v: wilson_lb(d["p_fast"], d["neff_phi"], v) >= 0.50, z_grid)

    variants = {
        f"boxcar>= {thr:.2f}": lambda d: d["bratio"] >= thr,
        f"fast-pt>={pt_thr:.3f}": lambda d, v=pt_thr: d["p_fast"] >= v,
        f"wilson z={wl_z:.2f}": lambda d, v=wl_z: wilson_lb(d["p_fast"], d["neff_phi"], v) >= 0.50,
    }

    print(f"context prints pooled pass target (boxcar) = {target:.4f}")
    hdr = f"{'variant':<18}{'pass':>7}{'noise<=.50':>11}{'flip<=.40':>10}{'srg cov':>9}{'lat med':>9}{'lat p90':>9}{'fires':>7}{'srg-hit':>8}"
    print(hdr)
    for name, fn in variants.items():
        pas = pooled_pass(fn)
        noise_n = noise_d = flip_n = 0
        cov_n = cov_d = 0
        lats = []
        fires_tot = 0
        hit_tot = 0
        for d in tapes:
            g = fn(d)
            ctx = d["ctx"]
            tr = d["truth"]
            m = ctx & g & ~np.isnan(tr)
            noise_d += int(m.sum())
            noise_n += int((m & (tr <= 0.50)).sum())
            flip_n += int((m & (tr <= 0.40)).sum())
            # surge coverage: armed within [-2, +10]s of onset
            ep = d["epoch"]
            gate_ep = ep[d["is_dir"] & g]
            for on in d["surge_on"]:
                cov_d += 1
                j = np.searchsorted(gate_ep, on - 2.0)
                if j < len(gate_ep) and gate_ep[j] <= on + 10.0:
                    cov_n += 1
                    lats.append(max(gate_ep[j] - on, 0.0))
            # full-gate fires
            lvl = ctx & g
            f_ep = _edge_fires_np(ep, d["ready"], d["ze"], lvl)
            fires_tot += len(f_ep)
            for on in d["surge_on"]:
                if len(f_ep) and ((f_ep >= on - 2.0) & (f_ep <= on + 10.0)).any():
                    hit_tot += 1
        lats = np.asarray(lats)
        lm = f"{np.median(lats):.2f}" if len(lats) else "-"
        lp = f"{np.percentile(lats, 90):.2f}" if len(lats) else "-"
        print(f"{name:<18}{pas:>7.4f}{noise_n / max(noise_d, 1):>11.4f}{flip_n / max(noise_d, 1):>10.4f}"
              f"{cov_n / max(cov_d, 1):>9.3f}{lm:>9}{lp:>9}{fires_tot:>7d}{hit_tot:>8d}")
    print(f"(surges total={sum(len(d['surge_on']) for d in tapes)})")

    # ---- divergence study: count-gate ON, volume ratio buckets ----
    print("\ndivergence (context & boxcar-ratio ON, subsampled >=5s apart):")
    rows = {"vol>0.50": ([], []), "vol<=0.50": ([], []), "vol<=0.40": ([], [])}
    for d in tapes:
        m = d["ctx"] & (d["bratio"] >= thr) & ~np.isnan(d["volr30"]) & ~np.isnan(d["fwd60"])
        idx = np.flatnonzero(m)
        last = -1e18
        for i in idx:
            if d["epoch"][i] - last < 5.0:
                continue
            last = d["epoch"][i]
            v = d["volr30"][i]
            for name, cond in (("vol>0.50", v > 0.50), ("vol<=0.50", v <= 0.50), ("vol<=0.40", v <= 0.40)):
                if cond:
                    rows[name][0].append(d["fwd30"][i])
                    rows[name][1].append(d["fwd60"][i])
    for name, (a30, a60) in rows.items():
        a30, a60 = np.asarray(a30, dtype=np.float64), np.asarray(a60, dtype=np.float64)
        if len(a30) == 0:
            print(f"  {name:<10} n=0")
            continue
        print(f"  {name:<10} n={len(a30):>6d}  fwd30 med={np.nanmedian(a30) * 100:+.3f}% mean={np.nanmean(a30) * 100:+.3f}%"
              f"  fwd60 med={np.nanmedian(a60) * 100:+.3f}% mean={np.nanmean(a60) * 100:+.3f}%")


if __name__ == "__main__":
    phase1_all()
    aggregate()
