"""Floor-jitter sensitivity: are the AND-stack thresholds over-precise?

For each gate floor, move it -20/-10/+10/+20% (others at production values)
and measure, over the 30-symbol-day corpus:

  - fire episodes: distinct all-gates-pass spells (>=5s apart) — the unit
    that approximates "entry opportunities the stack would act on"
  - Jaccard overlap of pass-samples vs the baseline pass set

Reading: a robust rule barely moves under small jitter (high Jaccard,
episode counts stable). Thresholds fit to corpus noise churn the pass set.
Also reports each gate's SOLE-BINDING rate — how often it is the only
failing gate (the marginal constraint) — since jitter only matters where a
gate actually binds.

Run:  python gate_jitter_study.py     (reads logs/research/ofi_tune_v2)
"""
from __future__ import annotations

import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.environ.get("OFI_TUNE_DATA", os.path.join(REPO, "logs", "research", "ofi_tune_v2"))

# Production floors (config SIM_PRESSURE_* / SIM_OFI_NORM_MIN)
FLOORS = {
    "speed": 10.0,
    "activity": 50.0,
    "ratio": 0.60,
    "volimb": -0.20,
    "accelz": 1.07,
    "ofi": 0.70,
}
JITTERS = (-0.20, -0.10, 0.0, 0.10, 0.20)
EPISODE_GAP_MS = 5_000


def load_days() -> list[dict]:
    days = []
    for path in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
        z = np.load(path)
        days.append({
            "name": os.path.basename(path).replace(".npz", ""),
            "t": z["t"],
            "speed": z["speed"].astype(np.float64),
            "activity": z["activity"].astype(np.float64),
            "ratio": z["ratio"].astype(np.float64),
            "volimb": z["volimb"].astype(np.float64),
            "accelz": z["accelz"].astype(np.float64),
            "accel_ready": z["accel_ready"].astype(bool),
            "ofi": z["ema10"].astype(np.float64),
            "depth": z["depth_active"].astype(bool),
        })
    return days


def gate_masks(day: dict, floors: dict) -> dict[str, np.ndarray]:
    return {
        "speed": day["speed"] >= floors["speed"],
        "activity": day["activity"] >= floors["activity"],
        "ratio": day["ratio"] >= floors["ratio"],
        "volimb": day["volimb"] > floors["volimb"],
        "accelz": day["accel_ready"] & (day["accelz"] >= floors["accelz"]),
        # OFI vacuous without depth, as in production
        "ofi": ~day["depth"] | (day["ofi"] >= floors["ofi"]),
    }


def stack_pass(day: dict, floors: dict) -> np.ndarray:
    m = gate_masks(day, floors)
    out = np.ones(len(day["t"]), dtype=bool)
    for v in m.values():
        out &= v
    return out


def episodes(day: dict, mask: np.ndarray) -> int:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0
    gaps = np.diff(day["t"][idx])
    return 1 + int((gaps > EPISODE_GAP_MS).sum())


def main() -> None:
    days = load_days()
    total = sum(len(d["t"]) for d in days)
    base_pass = {d["name"]: stack_pass(d, FLOORS) for d in days}
    base_eps = sum(episodes(d, base_pass[d["name"]]) for d in days)
    base_samples = sum(int(m.sum()) for m in base_pass.values())
    print(f"days={len(days)} samples={total} baseline: pass-samples={base_samples} "
          f"({100*base_samples/total:.2f}%) episodes={base_eps}")

    # Sole-binding rate per gate
    print("\nsole-binding (only failing gate) share of all samples:")
    sole = {g: 0 for g in FLOORS}
    for d in days:
        m = gate_masks(d, FLOORS)
        fails = {g: ~v for g, v in m.items()}
        n_fail = np.zeros(len(d["t"]), dtype=np.int32)
        for v in fails.values():
            n_fail += v
        for g in FLOORS:
            sole[g] += int((fails[g] & (n_fail == 1)).sum())
    for g in FLOORS:
        print(f"  {g:9s} {100*sole[g]/total:6.2f}%")

    print(f"\njitter sweep (episodes | Jaccard vs baseline, {len(days)} days pooled):")
    header = "  gate      " + " ".join(f"{int(j*100):+4d}%          " for j in JITTERS)
    print(header)
    for g in FLOORS:
        cells = []
        for j in JITTERS:
            floors = dict(FLOORS)
            floors[g] = FLOORS[g] * (1.0 + j)
            eps, inter, union = 0, 0, 0
            for d in days:
                m = stack_pass(d, floors)
                b = base_pass[d["name"]]
                eps += episodes(d, m)
                inter += int((m & b).sum())
                union += int((m | b).sum())
            jac = inter / union if union else 1.0
            cells.append(f"{eps:5d} J={jac:.2f}  ")
        print(f"  {g:9s} " + " ".join(cells))


if __name__ == "__main__":
    main()
