"""Scratch: dispersion-corrected EMA accel z — measurement before implementation.

Question: does dividing the EMA accel z by sqrt(phi_hat) — phi_hat = decayed
CV^2 of directional inter-arrival times (index-of-dispersion estimate, tau =
TAPE_EMA_SLOW_TAU_SECS) — make the z-gate's null calibration consistent
ACROSS tapes? Today SIM_PRESSURE_ACCEL_Z=2.5 is percentile-matched to the
AVERAGE clustering of the calibration tapes; per-symbol dispersion variation
is uncorrected.

Per tape (real OrderFlowTracker replay, identical to production):
  ze[i]   = pressure_acceleration_ema_z per print
  cv2[i]  = RAW (unclipped) decayed CV^2 of directional inter-arrivals
  z_corr  = ze / sqrt(clip(cv2, 1, cap))       (computed at aggregation)

Comparison at pooled percentile-matched thresholds (equal overall strictness):
  A. per-tape P(fire | ready) level rate      — cross-tape CV / IQR/median
  B. per-tape edge fires per active minute    — cross-tape CV / IQR/median
  C. surge detection %, latency, false fires  — harness ground truth
Decision: implement iff dispersion tightens materially (>=~25% CV reduction)
without material detection/false-fire regression.

Usage:
    from _tape_dispersion_scratch import replay, run_chunk, select_extra_tapes, aggregate
    run_chunk(TAPES[:5])          # replays + writes /tmp/tape_dispersion/*.npz
    aggregate()                   # loads all traces, prints report
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

sys.path.insert(0, "/Users/ahariharan/Desktop/momo_algo")

from _tape_accel_compare_scratch import (  # noqa: E402
    TAPES, _detect_surges, _load_tape, _match,
)
from exec.order_flow import OrderFlowTracker  # noqa: E402
from market_data.tradier_stream import LiveTimesale  # noqa: E402
from config import (  # noqa: E402
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_SPEED_MIN,
    TAPE_EMA_SLOW_TAU_SECS,
)

CACHE_DIR = "/Users/ahariharan/Desktop/momo_algo/logs/chart_cache"
OUT_DIR = "/tmp/tape_dispersion"

Z_FIRE = SIM_PRESSURE_ACCEL_Z          # 2.5 — production gate
REARM_RATIO = 1.0 / Z_FIRE             # production Z_REARM=1.0 → ratio 0.4
ACTIVE_GAP_CAP_S = 5.0                 # cap per-gap contribution to active time
MIN_READY_FOR_RATES = 500              # tapes with fewer ready prints excluded from dispersion stats
PHI_MIN_W = 8.0                        # min decayed gap mass before trusting cv2


def replay(symbol: str, day: str, out_dir: str = OUT_DIR) -> dict:
    """Replay one tape through the REAL tracker + phi accumulators. Writes npz."""
    df = _load_tape(symbol, day)
    n = len(df)
    trk = OrderFlowTracker()

    # _load_tape parses ts with utc=True (tz-aware) — use the safe conversion.
    import pandas as pd
    epoch = ((df["ts"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    price_arr = df["price"].to_numpy(dtype=np.float64)
    size_arr = df["size"].fillna(0).to_numpy(dtype=np.int64)
    bid_arr = df["bid"].fillna(0.0).to_numpy(dtype=np.float64)
    ask_arr = df["ask"].fillna(0.0).to_numpy(dtype=np.float64)
    py_ts = df["ts"].dt.to_pydatetime()

    is_dir = np.zeros(n, dtype=bool)
    ready = np.zeros(n, dtype=bool)
    ze = np.zeros(n, dtype=np.float32)
    cv2 = np.full(n, np.nan, dtype=np.float32)   # NaN = phi not trusted yet
    neff = np.zeros(n, dtype=np.float32)
    spd = np.zeros(n, dtype=np.float32)
    bratio = np.zeros(n, dtype=np.float32)
    act = np.zeros(n, dtype=np.int32)

    # phi accumulators: decayed moments of directional inter-arrival dt
    tau = TAPE_EMA_SLOW_TAU_SECS
    w_sum = 0.0
    dt_sum = 0.0
    dt_sq = 0.0
    last_dir_epoch: float | None = None

    # Incremental activity_count replica (trk.activity_count walks the whole
    # 30s window per read -> O(n * window occupancy), pathological on dense
    # tape). Semantics: # directional prints with ts >= newest_dir_ts - 30s.
    # Retention in the tracker is max(baseline+signal, activity) >= 30s, so
    # this deque replica is exactly equivalent.
    act_win = float(trk._activity_window)
    act_dq: deque[float] = deque()

    for i in range(n):
        ev = LiveTimesale(
            event_type="timesale", symbol=symbol, exchange="",
            last=price_arr[i], bid=bid_arr[i], ask=ask_arr[i],
            size=int(size_arr[i]), ts=py_ts[i], seq=i, flag="",
            cancel=False, correction=False, session="normal",
        )
        ct = trk.on_timesale(ev)
        d = ct.side != "mid"
        is_dir[i] = d
        if d:
            if last_dir_epoch is not None:
                dt = max(epoch[i] - last_dir_epoch, 0.0)
                decay = math.exp(-dt / tau)
                w_sum *= decay
                dt_sum *= decay
                dt_sq *= decay
                w_sum += 1.0
                dt_sum += dt
                dt_sq += dt * dt
            last_dir_epoch = epoch[i]
        if w_sum >= PHI_MIN_W:
            mu = dt_sum / w_sum
            m2 = dt_sq / w_sum
            var = max(m2 - mu * mu, 0.0)
            cv2[i] = var / (mu * mu) if mu > 1e-9 else np.inf
        elif i > 0:
            cv2[i] = cv2[i - 1]  # carry (ratios invariant under uniform decay)

        if d:
            act_dq.append(epoch[i])
            cutoff = epoch[i] - act_win
            while act_dq[0] < cutoff:
                act_dq.popleft()

        ready[i] = trk.pressure_acceleration_ema_ready
        ze[i] = trk.pressure_acceleration_ema_z
        neff[i] = trk.pressure_acceleration_ema_n_eff
        spd[i] = trk.trade_speed
        bratio[i] = trk.buy_ratio
        act[i] = len(act_dq)

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"{symbol}_{day}.npz"),
        epoch=epoch, price=price_arr, is_dir=is_dir, ready=ready,
        ze=ze, cv2=cv2, neff=neff, spd=spd, bratio=bratio, act=act,
    )
    return {
        "symbol": symbol, "day": day, "prints": n,
        "directional": int(is_dir.sum()), "ready": int(ready.sum()),
        "median_cv2_ready": float(np.nanmedian(cv2[ready])) if ready.any() else None,
    }


def run_chunk(picks: list[tuple[str, str]], skip_done: bool = True) -> list[dict]:
    out = []
    total = len(picks)
    for k, (sym, day) in enumerate(picks, 1):
        path = os.path.join(OUT_DIR, f"{sym}_{day}.npz")
        if skip_done and os.path.exists(path):
            print(f"[{k}/{total}] {sym}_{day} — already done, skipped", flush=True)
            out.append({"symbol": sym, "day": day, "skipped": True})
            continue
        print(f"[{k}/{total}] {sym}_{day} — replaying…", flush=True)
        t0 = time.perf_counter()
        try:
            res = replay(sym, day)
            print(
                f"[{k}/{total}] {sym}_{day} — done in {time.perf_counter() - t0:.0f}s "
                f"({res['prints']} prints, ready={res['ready']}, "
                f"cv2_med={res['median_cv2_ready']:.2f})",
                flush=True,
            )
            out.append(res)
        except Exception as exc:  # noqa: BLE001 — scratch: record and continue
            print(f"[{k}/{total}] {sym}_{day} — ERROR {type(exc).__name__}: {exc}", flush=True)
            out.append({"symbol": sym, "day": day, "error": f"{type(exc).__name__}: {exc}"})
    return out


def select_extra_tapes(n_extra: int = 20) -> list[tuple[str, str]]:
    """Stratified-by-size sample of cached tapes, excluding the harness TAPES."""
    files = []
    for f in os.listdir(CACHE_DIR):
        if not (f.startswith("timesales_") and f.endswith(".csv")):
            continue
        parts = f[len("timesales_"):-len(".csv")].rsplit("_", 1)
        if len(parts) != 2:
            continue
        sym, day = parts
        if (sym, day) in TAPES:
            continue
        sz = os.path.getsize(os.path.join(CACHE_DIR, f))
        if sz < 400_000:  # need enough prints for the gate to arm meaningfully
            continue
        files.append((sz, sym, day))
    files.sort()
    if len(files) <= n_extra:
        return [(s, d) for _, s, d in files]
    idx = np.linspace(0, len(files) - 1, n_extra).astype(int)
    return [(files[i][1], files[i][2]) for i in idx]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _edge_fires_np(epoch, ready, z, thr, spd, bratio, act):
    """Edge-triggered fires with hysteresis (rearm at REARM_RATIO*thr)."""
    fires = []
    armed = True
    rearm = REARM_RATIO * thr
    for i in range(len(epoch)):
        if not ready[i]:
            continue
        zi = z[i]
        if armed and zi >= thr:
            full = (
                spd[i] >= SIM_PRESSURE_SPEED_MIN
                and act[i] >= SIM_PRESSURE_ACTIVITY_MIN
                and bratio[i] >= SIM_PRESSURE_BUY_RATIO_MIN
            )
            fires.append({"epoch": float(epoch[i]), "z": float(zi), "full_gate": bool(full)})
            armed = False
        elif not armed and zi < rearm:
            armed = True
    return fires


def _active_secs(epoch, ready):
    e = epoch[ready]
    if len(e) < 2:
        return 0.0
    gaps = np.diff(e)
    return float(np.minimum(gaps, ACTIVE_GAP_CAP_S).sum())


def _disp(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return {}
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    mean = float(v.mean())
    return {
        "n": int(len(v)), "mean": round(mean, 4), "median": round(med, 4),
        "cv": round(float(v.std() / mean), 3) if mean > 0 else None,
        "iqr_over_median": round(float((q3 - q1) / med), 3) if med > 0 else None,
        "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
    }


def aggregate(clip_hi: float = 8.0, out_dir: str = OUT_DIR) -> dict:
    traces = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".npz"):
            d = dict(np.load(os.path.join(out_dir, f), allow_pickle=False))
            d["name"] = f[:-4]
            traces.append(d)
    if not traces:
        return {"error": "no traces"}

    # --- pooled percentile matching (equal overall strictness) ---
    pooled_ze, pooled_zc = [], []
    for t in traces:
        r = t["ready"].astype(bool)
        phi = np.clip(np.nan_to_num(t["cv2"], nan=1.0, posinf=clip_hi), 1.0, clip_hi)
        t["zc"] = t["ze"] / np.sqrt(phi)
        t["phi"] = phi
        pooled_ze.append(t["ze"][r])
        pooled_zc.append(t["zc"][r])
    pooled_ze = np.concatenate(pooled_ze)
    pooled_zc = np.concatenate(pooled_zc)
    target_rate = float((pooled_ze >= Z_FIRE).mean())
    t_star = float(np.quantile(pooled_zc, 1.0 - target_rate))

    # --- per-tape metrics at both gates ---
    per_tape = []
    lat_u, lat_c = [], []
    pooled = {"surges": 0, "det_u": 0, "det_c": 0, "fires_u": 0, "fires_c": 0,
              "ff_u": 0, "ff_c": 0, "full_u": 0, "full_c": 0}
    for t in traces:
        r = t["ready"].astype(bool)
        n_ready = int(r.sum())
        row = {"name": t["name"], "ready": n_ready,
               "phi_med": round(float(np.median(t["phi"][r])), 3) if n_ready else None}
        if n_ready >= MIN_READY_FOR_RATES:
            row["rate_u"] = float((t["ze"][r] >= Z_FIRE).mean())
            row["rate_c"] = float((t["zc"][r] >= t_star).mean())
            fu = _edge_fires_np(t["epoch"], r, t["ze"], Z_FIRE, t["spd"], t["bratio"], t["act"])
            fc = _edge_fires_np(t["epoch"], r, t["zc"], t_star, t["spd"], t["bratio"], t["act"])
            asecs = _active_secs(t["epoch"], r)
            row["fpm_u"] = 60.0 * len(fu) / asecs if asecs > 0 else None
            row["fpm_c"] = 60.0 * len(fc) / asecs if asecs > 0 else None
            surges = _detect_surges(t["epoch"], t["is_dir"].astype(bool), t["price"])
            mu = _match(surges, fu)
            mc = _match(surges, fc)
            pooled["surges"] += mu["n_surges"]
            pooled["det_u"] += mu["n_detected"]
            pooled["det_c"] += mc["n_detected"]
            pooled["fires_u"] += mu["n_fires"]
            pooled["fires_c"] += mc["n_fires"]
            pooled["ff_u"] += mu["n_false_fires"]
            pooled["ff_c"] += mc["n_false_fires"]
            pooled["full_u"] += sum(1 for f in fu if f["full_gate"])
            pooled["full_c"] += sum(1 for f in fc if f["full_gate"])
            lat_u.extend([l for l in mu["latencies"] if l is not None])
            lat_c.extend([l for l in mc["latencies"] if l is not None])
        per_tape.append(row)

    rated = [r for r in per_tape if "rate_u" in r]
    report = {
        "n_tapes": len(traces), "n_rated": len(rated),
        "clip_hi": clip_hi, "target_rate": round(target_rate, 4), "t_star": round(t_star, 3),
        "phi_med_across_tapes": _disp([r["phi_med"] for r in per_tape]),
        "level_rate_uncorrected": _disp([r["rate_u"] for r in rated]),
        "level_rate_corrected": _disp([r["rate_c"] for r in rated]),
        "fires_per_min_uncorrected": _disp([r["fpm_u"] for r in rated]),
        "fires_per_min_corrected": _disp([r["fpm_c"] for r in rated]),
        "surge_pooled": {
            "surges": pooled["surges"],
            "detected_u": pooled["det_u"], "detected_c": pooled["det_c"],
            "fires_u": pooled["fires_u"], "fires_c": pooled["fires_c"],
            "false_fires_u": pooled["ff_u"], "false_fires_c": pooled["ff_c"],
            "full_gate_fires_u": pooled["full_u"], "full_gate_fires_c": pooled["full_c"],
            "lat_med_u": round(float(np.median(lat_u)), 2) if lat_u else None,
            "lat_med_c": round(float(np.median(lat_c)), 2) if lat_c else None,
            "lat_p90_u": round(float(np.percentile(lat_u, 90)), 2) if lat_u else None,
            "lat_p90_c": round(float(np.percentile(lat_c, 90)), 2) if lat_c else None,
        },
        "per_tape": per_tape,
    }
    return report


if __name__ == "__main__":
    picks = TAPES
    if len(sys.argv) > 1:
        picks = [tuple(a.rsplit("_", 1)) for a in sys.argv[1:]]
    for res in run_chunk(picks):
        print(json.dumps(res))
