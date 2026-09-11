"""Scratch: JOINT tape-gate activation study — how often do the five tape
criteria (speed, activity, buy-ratio, volume-flow, accel-z) pass TOGETHER,
and is the joint signal meaningful?

Questions (user, 2026-07-26):
  1. How often does the joint gate activate on a stock? (episodes/min, spacing,
     duration, per-tape spread)
  2. Are joint fires meaningful or noise? (alignment with surge ground truth,
     forward returns vs baseline and vs vetoed z-fires)
  3. Is the joint gate missing a lot? (surge detection: joint vs accel-z-only,
     veto attribution for surges z caught but the joint gate rejected)
  4. Misaligned or perfect? (false-episode share, latency from surge onset)

Replays the SAME 40 tapes as the dispersion study (/tmp/tape_dispersion names)
through the REAL OrderFlowTracker + exec.signals.tape_entry_gates — the exact
production gate definition, zero reimplementation drift. Per print we record
side/size plus every gate read, so all offline sensitivity work is free later.

Usage:
    from _tape_joint_gate_scratch import tapes_all, run_chunk, aggregate
    run_chunk(tapes_all()[:10])
    aggregate()
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/ahariharan/Desktop/momo_algo")

from _tape_accel_compare_scratch import _detect_surges, _load_tape  # noqa: E402
from exec.order_flow import OrderFlowTracker  # noqa: E402
from exec.signals import tape_entry_gates  # noqa: E402
from market_data.tradier_stream import LiveTimesale  # noqa: E402
from config import (  # noqa: E402
    SIM_PRESSURE_ACCEL_Z,
    SIM_PRESSURE_ACTIVITY_MIN,
    SIM_PRESSURE_BUY_RATIO_MIN,
    SIM_PRESSURE_SPEED_MIN,
    SIM_PRESSURE_VOL_IMBALANCE_MIN,
)

OUT_DIR = "/tmp/tape_joint_gate"
DISP_DIR = "/tmp/tape_dispersion"      # defines the 40-tape sample
MATCH_PRE_SECS = 2.0                   # same surge-match window as prior studies
MATCH_POST_SECS = 10.0
ACTIVE_GAP_CAP_S = 5.0                 # cap per-gap contribution to active time
MIN_READY_FOR_RATES = 500
FWD_HORIZON_S = 60.0
BASELINE_SAMPLES = 2000                # per-tape control sample of ready dir prints

GATES = ("speed", "act", "ratio", "vimb", "accel")


def tapes_all() -> list[tuple[str, str]]:
    """The exact 40-tape sample of the dispersion study."""
    out = []
    for f in sorted(os.listdir(DISP_DIR)):
        if f.endswith(".npz"):
            sym, day = f[:-4].rsplit("_", 1)
            out.append((sym, day))
    return out


def replay(symbol: str, day: str, out_dir: str = OUT_DIR) -> dict:
    """Replay one tape; record production tape_entry_gates reads per print."""
    import pandas as pd

    df = _load_tape(symbol, day)
    n = len(df)
    trk = OrderFlowTracker()

    epoch = ((df["ts"] - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy()
    price_arr = df["price"].to_numpy(dtype=np.float64)
    size_arr = df["size"].fillna(0).to_numpy(dtype=np.int64)
    bid_arr = df["bid"].fillna(0.0).to_numpy(dtype=np.float64)
    ask_arr = df["ask"].fillna(0.0).to_numpy(dtype=np.float64)
    py_ts = df["ts"].dt.to_pydatetime()

    side = np.zeros(n, dtype=np.int8)          # +1 buy / -1 sell / 0 mid
    speed = np.zeros(n, dtype=np.float32)      # speed_gate (EMA source)
    act = np.zeros(n, dtype=np.int32)          # activity_count (30s)
    ratio = np.zeros(n, dtype=np.float32)      # ratio_gate (EMA source)
    vimb = np.zeros(n, dtype=np.float32)       # trade_imbalance (10s volume flow)
    ready = np.zeros(n, dtype=bool)            # accel_gate_ready
    zc = np.zeros(n, dtype=np.float32)         # accel_gate_z (dispersion-corrected)

    for i in range(n):
        ev = LiveTimesale(
            event_type="timesale", symbol=symbol, exchange="",
            last=price_arr[i], bid=bid_arr[i], ask=ask_arr[i],
            size=int(size_arr[i]), ts=py_ts[i], seq=i, flag="",
            cancel=False, correction=False, session="normal",
        )
        ct = trk.on_timesale(ev)
        side[i] = 1 if ct.side == "buy" else (-1 if ct.side == "sell" else 0)
        g = tape_entry_gates(trk)
        speed[i] = g.speed
        act[i] = g.activity
        ratio[i] = g.buy_ratio
        vimb[i] = g.vol_imbalance
        ready[i] = g.accel_ready
        zc[i] = g.accel_z

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f"{symbol}_{day}.npz"),
        epoch=epoch, price=price_arr, size=size_arr, side=side,
        speed=speed, act=act, ratio=ratio, vimb=vimb, ready=ready, zc=zc,
    )
    return {"symbol": symbol, "day": day, "prints": n,
            "directional": int((side != 0).sum()), "ready": int(ready.sum())}


def run_chunk(picks: list[tuple[str, str]], skip_done: bool = True) -> list[dict]:
    out = []
    total = len(picks)
    for k, (sym, day) in enumerate(picks, 1):
        path = os.path.join(OUT_DIR, f"{sym}_{day}.npz")
        if skip_done and os.path.exists(path):
            print(f"[{k}/{total}] {sym}_{day} — already done, skipped", flush=True)
            continue
        print(f"[{k}/{total}] {sym}_{day} — replaying…", flush=True)
        t0 = time.perf_counter()
        try:
            res = replay(sym, day)
            print(f"[{k}/{total}] {sym}_{day} — done in {time.perf_counter() - t0:.0f}s "
                  f"({res['prints']} prints, ready={res['ready']})", flush=True)
            out.append(res)
        except Exception as exc:  # noqa: BLE001 — scratch: record and continue
            print(f"[{k}/{total}] {sym}_{day} — ERROR {type(exc).__name__}: {exc}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _ok_bits(t: dict) -> dict[str, np.ndarray]:
    return {
        "speed": t["speed"] >= SIM_PRESSURE_SPEED_MIN,
        "act": t["act"] >= SIM_PRESSURE_ACTIVITY_MIN,
        "ratio": t["ratio"] >= SIM_PRESSURE_BUY_RATIO_MIN,
        "vimb": t["vimb"] > SIM_PRESSURE_VOL_IMBALANCE_MIN,
        "accel": t["ready"] & (t["zc"] >= SIM_PRESSURE_ACCEL_Z),
    }


def _episodes(epoch: np.ndarray, all_pass: np.ndarray) -> list[dict]:
    """Maximal runs of consecutive all_pass prints -> episode dicts."""
    eps = []
    i, n = 0, len(all_pass)
    while i < n:
        if all_pass[i]:
            j = i
            while j + 1 < n and all_pass[j + 1]:
                j += 1
            eps.append({"i0": i, "i1": j, "t0": float(epoch[i]), "t1": float(epoch[j]),
                        "prints": j - i + 1, "span": float(epoch[j] - epoch[i])})
            i = j + 1
        else:
            i += 1
    return eps


def _active_secs(epoch: np.ndarray, mask: np.ndarray) -> float:
    e = epoch[mask]
    if len(e) < 2:
        return 0.0
    return float(np.minimum(np.diff(e), ACTIVE_GAP_CAP_S).sum())


def _fwd_returns(epoch: np.ndarray, price: np.ndarray, t0: float) -> tuple[float, float] | None:
    """(MFE, net) over FWD_HORIZON_S after t0; None if censored by tape end."""
    if t0 + FWD_HORIZON_S > epoch[-1]:
        return None
    i0 = int(np.searchsorted(epoch, t0, side="right"))
    i1 = int(np.searchsorted(epoch, t0 + FWD_HORIZON_S, side="right"))
    if i1 <= i0:
        return None
    p0 = price[max(i0 - 1, 0)]
    if p0 <= 0:
        return None
    seg = price[i0:i1]
    return float(seg.max() / p0 - 1.0), float(seg[-1] / p0 - 1.0)


def _disp(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return {}
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    return {"n": int(len(v)), "median": round(med, 4),
            "p25": round(float(q1), 4), "p75": round(float(q3), 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


def aggregate(out_dir: str = OUT_DIR) -> dict:
    traces = []
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".npz"):
            d = dict(np.load(os.path.join(out_dir, f), allow_pickle=False))
            d["name"] = f[:-4]
            traces.append(d)
    if not traces:
        return {"error": "no traces"}

    per_tape = []
    pooled = {
        "ready_prints": 0, "pass_counts": {g: 0 for g in GATES}, "joint": 0,
        "sole_blocker": {g: 0 for g in GATES}, "sole_total": 0,
        "surges": 0, "det_joint": 0, "det_accel": 0,
        "veto_counts": {g: 0 for g in GATES}, "vetoed_surges": 0,
        "n_episodes": 0, "aligned_eps": 0,
    }
    lat_joint, lat_accel = [], []
    ep_spans, ep_prints, ep_gaps = [], [], []
    fwd = {"aligned": [], "false": [], "vetoed_z": [], "baseline": []}

    for t in traces:
        ok = _ok_bits(t)
        r = t["ready"].astype(bool)
        n_ready = int(r.sum())
        all_pass = ok["speed"] & ok["act"] & ok["ratio"] & ok["vimb"] & ok["accel"]
        is_dir = t["side"] != 0

        row = {"name": t["name"], "prints": len(r), "ready": n_ready}
        if n_ready < MIN_READY_FOR_RATES:
            row["excluded"] = True
            per_tape.append(row)
            continue

        # A. marginal + joint pass rates on ready prints
        for g in GATES:
            row[f"p_{g}"] = float(ok[g][r].mean())
            pooled["pass_counts"][g] += int(ok[g][r].sum())
        row["p_joint"] = float(all_pass[r].mean())
        indep = float(np.prod([row[f"p_{g}"] for g in GATES]))
        row["joint_over_indep"] = row["p_joint"] / indep if indep > 0 else None
        pooled["ready_prints"] += n_ready
        pooled["joint"] += int(all_pass[r].sum())

        # time-weighted joint share of active tape time
        dt = np.minimum(np.diff(t["epoch"]), ACTIVE_GAP_CAP_S)
        rr = r[:-1]
        row["joint_time_share"] = (
            float(dt[rr & all_pass[:-1]].sum() / dt[rr].sum()) if dt[rr].sum() > 0 else None
        )

        # B. sole-blocker attribution (exactly one gate failing, on ready prints)
        n_fail = np.zeros(len(r), dtype=np.int8)
        for g in GATES:
            n_fail += (~ok[g]).astype(np.int8)
        sole = r & (n_fail == 1)
        pooled["sole_total"] += int(sole.sum())
        for g in GATES:
            pooled["sole_blocker"][g] += int((sole & ~ok[g]).sum())

        # C. episodes
        eps = _episodes(t["epoch"], all_pass)
        row["episodes"] = len(eps)
        asecs = _active_secs(t["epoch"], r)
        row["eps_per_active_min"] = 60.0 * len(eps) / asecs if asecs > 0 else None
        ep_spans.extend(e["span"] for e in eps)
        ep_prints.extend(e["prints"] for e in eps)
        ep_gaps.extend(eps[k + 1]["t0"] - eps[k]["t1"] for k in range(len(eps) - 1))
        pooled["n_episodes"] += len(eps)

        # D. surge alignment
        surges = _detect_surges(t["epoch"], is_dir, t["price"])
        pooled["surges"] += len(surges)
        ep_starts = np.array([e["t0"] for e in eps]) if eps else np.array([])
        accel_t = t["epoch"][r & ok["accel"]]
        pass_t = t["epoch"][all_pass]
        aligned_idx: set[int] = set()
        for s in surges:
            lo, hi = s["onset_epoch"] - MATCH_PRE_SECS, s["onset_epoch"] + MATCH_POST_SECS
            hits = pass_t[(pass_t >= lo) & (pass_t <= hi)]
            zhits = accel_t[(accel_t >= lo) & (accel_t <= hi)]
            if len(zhits):
                pooled["det_accel"] += 1
                lat_accel.append(float(zhits[0] - s["onset_epoch"]))
            if len(hits):
                pooled["det_joint"] += 1
                lat_joint.append(float(hits[0] - s["onset_epoch"]))
            elif len(zhits):
                # z caught it, joint gate vetoed: attribute failing gates at z prints
                pooled["vetoed_surges"] += 1
                w = r & ok["accel"] & (t["epoch"] >= lo) & (t["epoch"] <= hi)
                for g in ("speed", "act", "ratio", "vimb"):
                    if bool((~ok[g][w]).all()):
                        pooled["veto_counts"][g] += 1
            # mark episodes aligned to this surge (start in match window or inside span)
            if len(ep_starts):
                m = ((ep_starts >= lo) & (ep_starts <= hi)) | (
                    (ep_starts >= s["onset_epoch"]) & (ep_starts <= s["end_epoch"]))
                aligned_idx.update(np.where(m)[0].tolist())

        row["surges"] = len(surges)
        row["aligned_eps"] = len(aligned_idx)
        pooled["aligned_eps"] += len(aligned_idx)

        # E. forward returns: aligned vs false episodes, vetoed z-fires, baseline
        for k, e in enumerate(eps):
            fr = _fwd_returns(t["epoch"], t["price"], e["t0"])
            if fr is not None:
                fwd["aligned" if k in aligned_idx else "false"].append(fr)
        # vetoed z-fires: accel ok but joint fails (edge onsets of accel-ok runs)
        vz = r & ok["accel"] & ~all_pass
        onset = vz & ~np.roll(vz, 1)
        onset[0] = vz[0]
        vt = t["epoch"][onset]
        if len(vt) > 400:  # cap per tape so one tape can't dominate the pool
            vt = vt[np.linspace(0, len(vt) - 1, 400).astype(int)]
        for tt in vt:
            fr = _fwd_returns(t["epoch"], t["price"], float(tt))
            if fr is not None:
                fwd["vetoed_z"].append(fr)
        # baseline: evenly sampled ready directional prints
        cand = np.where(r & is_dir)[0]
        if len(cand):
            samp = cand[np.linspace(0, len(cand) - 1, min(BASELINE_SAMPLES, len(cand))).astype(int)]
            for i in samp:
                fr = _fwd_returns(t["epoch"], t["price"], float(t["epoch"][i]))
                if fr is not None:
                    fwd["baseline"].append(fr)

        per_tape.append(row)

    rated = [x for x in per_tape if "p_joint" in x]
    report = {
        "n_tapes": len(traces), "n_rated": len(rated),
        "thresholds": {"speed": SIM_PRESSURE_SPEED_MIN, "act": SIM_PRESSURE_ACTIVITY_MIN,
                       "ratio": SIM_PRESSURE_BUY_RATIO_MIN, "vimb": SIM_PRESSURE_VOL_IMBALANCE_MIN,
                       "accel_z": SIM_PRESSURE_ACCEL_Z},
        "pooled_pass_rates": {g: round(pooled["pass_counts"][g] / pooled["ready_prints"], 4)
                              for g in GATES},
        "pooled_joint_rate": round(pooled["joint"] / pooled["ready_prints"], 4),
        "per_tape_joint_rate": _disp([x["p_joint"] for x in rated]),
        "per_tape_joint_time_share": _disp([x["joint_time_share"] for x in rated]),
        "per_tape_joint_over_indep": _disp([x["joint_over_indep"] for x in rated]),
        "sole_blocker_shares": {g: round(pooled["sole_blocker"][g] / pooled["sole_total"], 3)
                                for g in GATES} if pooled["sole_total"] else {},
        "episodes": {
            "total": pooled["n_episodes"],
            "per_active_min": _disp([x["eps_per_active_min"] for x in rated]),
            "span_s": _disp(ep_spans), "prints": _disp(ep_prints), "gap_s": _disp(ep_gaps),
        },
        "surge_alignment": {
            "surges": pooled["surges"],
            "det_joint": pooled["det_joint"],
            "det_joint_pct": round(100.0 * pooled["det_joint"] / pooled["surges"], 1)
            if pooled["surges"] else None,
            "det_accel_only": pooled["det_accel"],
            "det_accel_pct": round(100.0 * pooled["det_accel"] / pooled["surges"], 1)
            if pooled["surges"] else None,
            "latency_joint_s": _disp(lat_joint), "latency_accel_s": _disp(lat_accel),
            "vetoed_surges": pooled["vetoed_surges"],
            "veto_gate_always_failing": pooled["veto_counts"],
            "aligned_episodes": pooled["aligned_eps"],
            "false_episodes": pooled["n_episodes"] - pooled["aligned_eps"],
            "aligned_share": round(pooled["aligned_eps"] / pooled["n_episodes"], 3)
            if pooled["n_episodes"] else None,
        },
        "fwd_returns_60s": {
            k: {"n": len(v),
                "mfe_med": round(float(np.median([x[0] for x in v])), 4) if v else None,
                "net_med": round(float(np.median([x[1] for x in v])), 4) if v else None,
                "net_mean": round(float(np.mean([x[1] for x in v])), 4) if v else None,
                "net_p25": round(float(np.percentile([x[1] for x in v], 25)), 4) if v else None,
                "net_p75": round(float(np.percentile([x[1] for x in v], 75)), 4) if v else None}
            for k, v in fwd.items()
        },
        "per_tape": per_tape,
    }
    return report


# ---------------------------------------------------------------------------
# Direction-leg relaxation sweep (follow-up to the activation study):
# can a lower ratio floor or an OR-with-volume-flow leg claw back part of the
# 28% vetoed up-surges without admitting too many down-surge dip-buy fires?
# ---------------------------------------------------------------------------

CLUSTER_GAP_S = 5.0

# (label, ratio_min, vimb_or): dir_ok = ratio >= ratio_min  [| vimb >= vimb_or]
# The vimb > -0.20 distribution guard stays on in ALL variants.
DIR_VARIANTS: list[tuple[str, float, float | None]] = [
    ("r0.60 base", 0.60, None),
    ("r0.58", 0.58, None),
    ("r0.55", 0.55, None),
    ("r0.52", 0.52, None),
    ("r0.50", 0.50, None),
    ("r0.60|v0.20", 0.60, 0.20),
    ("r0.60|v0.30", 0.60, 0.30),
    ("r0.60|v0.40", 0.60, 0.40),
    ("r0.55|v0.30", 0.55, 0.30),
]


def _clusters_from_mask(epoch: np.ndarray, ap: np.ndarray, gap: float = CLUSTER_GAP_S) -> np.ndarray:
    """(n_clusters, 2) [t0, t1] of all_pass runs merged at <=gap seconds."""
    if not ap.any():
        return np.empty((0, 2))
    d = np.diff(ap.astype(np.int8))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0]
    if ap[0]:
        starts = np.r_[0, starts]
    if ap[-1]:
        ends = np.r_[ends, len(ap) - 1]
    t0, t1 = epoch[starts], epoch[ends]
    new = np.r_[True, (t0[1:] - t1[:-1]) > gap]
    c0 = t0[new]
    c1 = t1[np.r_[new[1:], True]]
    return np.stack([c0, c1], axis=1)


def sweep_direction_variants(out_dir: str = OUT_DIR) -> None:
    # --- precompute per tape: gate bits, surges (direction + fwd), active time
    tapes = []
    surs = {"up": [], "fr_mfe": [], "fr_net": [], "lo": [], "hi": [],
            "onset": [], "end": [], "tape": []}
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".npz"):
            continue
        t = dict(np.load(os.path.join(out_dir, f), allow_pickle=False))
        r = t["ready"].astype(bool)
        if int(r.sum()) < MIN_READY_FOR_RATES:
            continue
        ok = _ok_bits(t)
        t["fixed"] = r & ok["speed"] & ok["act"] & ok["vimb"] & ok["accel"]
        t["r"] = r
        k = len(tapes)
        for s in _detect_surges(t["epoch"], t["side"] != 0, t["price"]):
            i0 = int(np.searchsorted(t["epoch"], s["onset_epoch"], side="left"))
            i1 = int(np.searchsorted(t["epoch"], s["end_epoch"], side="right")) - 1
            if i1 <= i0 or t["price"][i0] <= 0:
                continue
            fr = _fwd_returns(t["epoch"], t["price"], s["onset_epoch"])
            surs["up"].append(t["price"][i1] / t["price"][i0] - 1.0 >= 0)
            surs["fr_mfe"].append(fr[0] if fr else np.nan)
            surs["fr_net"].append(fr[1] if fr else np.nan)
            surs["lo"].append(s["onset_epoch"] - MATCH_PRE_SECS)
            surs["hi"].append(s["onset_epoch"] + MATCH_POST_SECS)
            surs["onset"].append(s["onset_epoch"])
            surs["end"].append(s["end_epoch"])
            surs["tape"].append(k)
        t["asecs"] = _active_secs(t["epoch"], r)
        tapes.append(t)
    up = np.array(surs["up"])
    net = np.array(surs["fr_net"])
    tape_of = np.array(surs["tape"])
    lo, hi = np.array(surs["lo"]), np.array(surs["hi"])
    sp0, sp1 = np.array(surs["onset"]), np.array(surs["end"])
    print(f"{len(tapes)} tapes, {len(up)} surges ({int(up.sum())} up)\n")

    det_by_variant: dict[str, np.ndarray] = {}
    rows = []
    for label, r_min, v_or in DIR_VARIANTS:
        det = np.zeros(len(up), dtype=bool)
        n_cl, aligned, fpm = 0, 0, []
        for k, t in enumerate(tapes):
            dir_ok = t["ratio"] >= r_min
            if v_or is not None:
                dir_ok |= t["vimb"] >= v_or
            ap = t["fixed"] & dir_ok
            pass_t = t["epoch"][ap]
            m = tape_of == k
            iA = np.searchsorted(pass_t, lo[m])
            iB = np.searchsorted(pass_t, hi[m], side="right")
            det[m] = iB > iA
            cl = _clusters_from_mask(t["epoch"], ap)
            n_cl += len(cl)
            if len(cl):
                c0 = cl[:, 0]
                a = ((c0[:, None] >= lo[None, m]) & (c0[:, None] <= hi[None, m])) | (
                    (c0[:, None] >= sp0[None, m]) & (c0[:, None] <= sp1[None, m]))
                aligned += int(a.any(axis=1).sum())
            if t["asecs"] > 0:
                fpm.append(60.0 * len(cl) / t["asecs"])
        det_by_variant[label] = det
        du, dd = det & up, det & ~up
        rows.append({
            "label": label,
            "fpm_med": round(float(np.median(fpm)), 2),
            "clusters": n_cl,
            "aligned_pct": round(100.0 * aligned / n_cl, 1) if n_cl else None,
            "det_up_pct": round(100.0 * du.sum() / up.sum(), 1),
            "det_dn_pct": round(100.0 * dd.sum() / (~up).sum(), 1),
            "net_up_mean": round(float(np.nanmean(net[du])), 4),
            "net_dn_mean": round(float(np.nanmean(net[dd])), 4),
            "ev_per_det": round(float(np.nanmean(net[det])), 4),
            "total_net": round(float(np.nansum(net[det])), 2),
        })

    base = det_by_variant["r0.60 base"]
    hdr = (f"{'variant':>13} {'fpm':>5} {'clus':>6} {'algn%':>6} {'up%':>6} {'dn%':>6} "
           f"{'netUp':>7} {'netDn':>7} {'EV/det':>7} {'ΣnetR':>7} "
           f"{'+up':>4} {'+upNet':>7} {'+dn':>4} {'+dnNet':>7}")
    print(hdr)
    for row in rows:
        det = det_by_variant[row["label"]]
        add = det & ~base
        au, ad = add & up, add & ~up
        print(f"{row['label']:>13} {row['fpm_med']:>5.2f} {row['clusters']:>6} "
              f"{row['aligned_pct']:>6.1f} {row['det_up_pct']:>6.1f} {row['det_dn_pct']:>6.1f} "
              f"{row['net_up_mean']:>7.4f} {row['net_dn_mean']:>7.4f} {row['ev_per_det']:>7.4f} "
              f"{row['total_net']:>7.2f} "
              f"{int(au.sum()):>4} {float(np.nanmean(net[au])) if au.any() else 0.0:>7.4f} "
              f"{int(ad.sum()):>4} {float(np.nanmean(net[ad])) if ad.any() else 0.0:>7.4f}")

    # per-tape consistency of the headline relaxations vs baseline
    for label in ("r0.55", "r0.60|v0.30", "r0.55|v0.30"):
        det = det_by_variant[label]
        gain_u = gain_d = tapes_u = tapes_d = 0
        for k in range(len(tapes)):
            m = tape_of == k
            du = int(((det & ~base) & up & m).sum())
            dd = int(((det & ~base) & ~up & m).sum())
            gain_u += du
            gain_d += dd
            tapes_u += du > 0
            tapes_d += dd > 0
        print(f"\n{label}: +{gain_u} up-surges across {tapes_u} tapes, "
              f"+{gain_d} down-surges across {tapes_d} tapes")


if __name__ == "__main__":
    picks = tapes_all()
    if len(sys.argv) > 1:
        if sys.argv[1] == "aggregate":
            rep = aggregate()
            rep.pop("per_tape", None)
            print(json.dumps(rep, indent=2))
            sys.exit(0)
        if sys.argv[1] == "sweep":
            sweep_direction_variants()
            sys.exit(0)
        picks = [tuple(a.split("_")) for a in sys.argv[1:]]
    run_chunk(picks)
