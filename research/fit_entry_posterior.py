"""Fit the entry posterior (Upgrade 1) on the level-approach corpus.

Model: L2-regularized logistic regression, P(label == +1) — the approach
resolves up-first (+2% before -3% within 120s). Features are encoded by
exec.posterior.encode_features (single source with the live path; this
script cannot drift from what production computes).

Honesty constraints:
  - Validation is leave-one-DATE-out (all symbol-days sharing a calendar
    date held out together): events within a date share market regime, and
    symbols repeat across the corpus — pooled or per-row CV would leak.
  - The EV threshold is chosen on pooled OUT-OF-FOLD predictions over the
    DEPLOYED population (activity >= ACTIVITY_MIN — live, thinner tape is
    hard-blocked as a validity gate), with realized outcome returns
    (tb_ret: actual price at barrier hit / timeout, gap-through included).
  - lambda is picked by pooled OOF log-loss over a 4-point grid; the same
    OOF set then feeds the EV read (tiny selection optimism, accepted).

Run:  python fit_entry_posterior.py [--write]
      --write  refit on ALL data and write exec/entry_posterior.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from exec.posterior import FEATURES, encode_features  # noqa: E402

CSV = os.path.join(REPO, "logs", "research", "level_approach_labels.csv")
CSV_V3 = os.path.join(REPO, "logs", "research", "level_approach_labels_v3.csv")
ARTIFACT = os.path.join(REPO, "exec", "entry_posterior.json")

# --label exit (2026-08-23): target/payoff from the production-exit relabel
# (relabel_exit_payoff.py) instead of the +2/-3 barrier — win means the
# EXIT STACK realized >= EXIT_WIN_MIN from that entry, and EV curves use
# the realized exit pnl. This is the PnL-shaped objective.
EXIT_WIN_MIN = 0.005

ACTIVITY_MIN = 50          # config.SIM_PRESSURE_ACTIVITY_MIN (validity gate)
LAMBDAS = (0.3, 1.0, 3.0, 10.0)
COSTS = (0.002, 0.005, 0.010)   # round-trip cost assumptions (fraction)
COST_PICK = 0.005               # threshold chosen at this cost


def load_matrix(label: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """(df, X, y, payoff) for the chosen label scheme."""
    df = pd.read_csv(CSV_V3 if label == "exit" else CSV)
    raws = df.to_dict("records")
    X = np.array(
        [encode_features({**r, "rel_to_level": r["rel_to_level"]}) for r in raws],
        dtype=np.float64,
    )
    if label == "exit":
        payoff = df["pnl_exit"].to_numpy(dtype=np.float64)
        y = (payoff >= EXIT_WIN_MIN).astype(np.float64)
    else:
        payoff = df["tb_ret"].to_numpy(dtype=np.float64)
        y = (df["label"].to_numpy() == 1).astype(np.float64)
    return df, X, y, payoff


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """IRLS Newton fit; returns [intercept, coef...]; intercept unpenalized."""
    n, k = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    pen = lam * np.diag(np.r_[0.0, np.ones(k)])
    w = np.zeros(k + 1)
    for _ in range(60):
        z = np.clip(Xb @ w, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (y - p) - pen @ w
        H = (Xb * np.clip(p * (1 - p), 1e-9, None)[:, None]).T @ Xb + pen
        step = np.linalg.solve(H, g)
        w += step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    z = np.clip(w[0] + X @ w[1:], -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p, kind="stable")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    # midranks for ties
    s = pd.Series(p)
    ranks = s.rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return (X - mean) / scale, mean, scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the artifact")
    ap.add_argument("--label", choices=("barrier", "exit"), default="barrier",
                    help="barrier: +2/-3 triple barrier; exit: production-exit pnl")
    args = ap.parse_args()

    df, X, y, tb_ret = load_matrix(args.label)
    dates = df["day"].str.split("_").str[1].to_numpy()
    deployed = df["activity"].to_numpy() >= ACTIVITY_MIN
    n = len(y)
    print(f"corpus: {n} events, {len(np.unique(dates))} dates, "
          f"base P(up-first)={y.mean():.3f}, deployed(activity>={ACTIVITY_MIN}) "
          f"n={int(deployed.sum())} base={y[deployed].mean():.3f}")

    # ---- Leave-one-date-out CV over the lambda grid ----
    oof = {lam: np.full(n, np.nan) for lam in LAMBDAS}
    for date in np.unique(dates):
        te = dates == date
        tr = ~te
        Xtr_s, mean, scale = standardize(X[tr])
        Xte_s = (X[te] - mean) / scale
        for lam in LAMBDAS:
            w = fit_logistic(Xtr_s, y[tr], lam)
            oof[lam][te] = predict(w, Xte_s)

    print("\nlambda selection (pooled OOF):")
    base_ll = logloss(y, np.full(n, y.mean()))
    best_lam, best_ll = None, np.inf
    for lam in LAMBDAS:
        ll = logloss(y, oof[lam])
        a = auc(y, oof[lam])
        print(f"  lam={lam:5.1f}  logloss={ll:.4f} (base {base_ll:.4f})  AUC={a:.4f}")
        if ll < best_ll:
            best_lam, best_ll = lam, ll
    p_oof = oof[best_lam]
    print(f"  -> lambda={best_lam}")

    # ---- Calibration by decile (all events) ----
    print("\ncalibration (OOF deciles):")
    q = np.quantile(p_oof, np.arange(0, 1.01, 0.1))
    for i in range(10):
        m = (p_oof >= q[i]) & (p_oof <= q[i + 1] if i == 9 else p_oof < q[i + 1])
        if m.sum():
            print(f"  p in [{q[i]:.3f},{q[i+1]:.3f})  n={int(m.sum()):5d}  "
                  f"pred={p_oof[m].mean():.3f}  actual={y[m].mean():.3f}  "
                  f"mean tb_ret={tb_ret[m].mean():+.4f}")

    # ---- EV threshold on the deployed population ----
    print("\nEV curve (deployed population, OOF, payoff=tb_ret-cost):")
    print(f"  {'theta':>6} {'n':>6} {'take%':>6} {'P(up)':>6} {'mean_ret':>9} "
          + " ".join(f"totEV@{c:.1%}" for c in COSTS))
    pd_, yd, rd = p_oof[deployed], y[deployed], tb_ret[deployed]
    nd = len(pd_)
    # Threshold grid from the DEPLOYED score distribution (take-rates
    # 50%..0.5%) — an absolute grid misses the distribution entirely when
    # the base rate moves (the exit label's scores live under 0.30).
    theta_grid = np.unique(np.round(
        np.quantile(pd_, np.linspace(0.50, 0.995, 40)), 4
    ))
    best = {c: (None, -np.inf) for c in COSTS}
    for th in theta_grid:
        m = pd_ >= th
        k = int(m.sum())
        if k == 0:
            continue
        mean_ret = rd[m].mean()
        tot = {c: (rd[m] - c).sum() for c in COSTS}
        for c in COSTS:
            if tot[c] > best[c][1]:
                best[c] = (float(th), tot[c])
        print(f"  {th:6.2f} {k:6d} {100*k/nd:5.1f}% {yd[m].mean():6.3f} "
              f"{mean_ret:+9.4f} " + " ".join(f"{tot[c]:+11.2f}" for c in COSTS))
    for c in COSTS:
        print(f"  best theta @cost={c:.1%}: {best[c][0]} (total EV {best[c][1]:+.2f})")

    # ---- Threshold selection ----
    # Primary rule: max total OOF EV at COST_PICK. Guard: the pick must
    # select a non-trivial population (>= MIN_TAKE of deployed events) with
    # positive EV — a handful of tail events clearing the bar is noise, not
    # an operating point. When the EV rule is degenerate (2026-08-23 fit:
    # every theta with real coverage has negative barrier EV — approaches
    # are base-negative against +2/-3 and production exits, not the entry
    # instant, carry the realized edge), fall back to the calibration
    # plateau: the theta maximizing OOF ACTUAL P(up) among thetas keeping
    # >= MIN_TAKE coverage — the top of the calibrated score range, below
    # the overconfident tail where actual outcomes degrade.
    MIN_TAKE = 0.05
    ev_th, ev_tot = best[COST_PICK]
    ev_n = int((pd_ >= ev_th).sum()) if ev_th is not None else 0
    ev_degenerate = ev_th is None or ev_tot <= 0 or ev_n < MIN_TAKE * nd
    if not ev_degenerate:
        theta_star = ev_th
        criterion = f"max total EV @cost={COST_PICK:.1%}"
    else:
        theta_star, best_pup = None, -1.0
        for th in theta_grid:
            m = pd_ >= th
            if m.sum() < MIN_TAKE * nd:
                continue
            pup = float(yd[m].mean())
            if pup > best_pup:
                theta_star, best_pup = float(th), pup
        if theta_star is None:  # can't happen with a quantile grid; belt+braces
            theta_star = float(np.quantile(pd_, 0.9))
        criterion = (
            f"calibration plateau (EV rule degenerate): max OOF P(up) "
            f"with take >= {MIN_TAKE:.0%}"
        )
    print(f"\n  theta* = {theta_star}  [{criterion}]")

    # per-date robustness at theta_star
    print(f"\nper-date EV at theta={theta_star} cost={COST_PICK:.1%}:")
    pos = 0
    date_d = dates[deployed]
    for date in np.unique(date_d):
        m = (date_d == date) & (pd_ >= theta_star)
        if m.sum() == 0:
            print(f"  {date}: no entries")
            continue
        ev = (rd[m] - COST_PICK).sum()
        pos += ev > 0
        print(f"  {date}: n={int(m.sum()):4d}  totEV={ev:+8.2f}  meanret={rd[m].mean():+.4f}")
    print(f"  positive dates: {pos}/{len(np.unique(date_d))}")

    # ---- Final refit on ALL data ----
    Xs, mean, scale = standardize(X)
    w = fit_logistic(Xs, y, best_lam)
    print("\ncoefficients (standardized, full refit):")
    for name, c in sorted(zip(FEATURES, w[1:]), key=lambda t: -abs(t[1])):
        print(f"  {name:16s} {c:+.4f}")
    print(f"  {'(intercept)':16s} {w[0]:+.4f}")

    if not args.write:
        print("\n(dry run — pass --write to emit the artifact)")
        return

    blob = {
        "version": 1,
        "features": list(FEATURES),
        "mean": [float(v) for v in mean],
        "scale": [float(v) for v in scale],
        "coef": [float(v) for v in w[1:]],
        "intercept": float(w[0]),
        "threshold": float(theta_star),
        "meta": {
            "created": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "dataset": os.path.relpath(CSV, REPO),
            "corpus_dates": [str(np.min(dates)), str(np.max(dates))],
            "n_events": int(n),
            "n_deployed": int(deployed.sum()),
            "base_rate": round(float(y.mean()), 4),
            "lambda": best_lam,
            "cv": {
                "scheme": "leave-one-date-out",
                "folds": int(len(np.unique(dates))),
                "oof_auc": round(auc(y, p_oof), 4),
                "oof_logloss": round(best_ll, 4),
                "base_logloss": round(base_ll, 4),
            },
            "ev": {
                "cost_roundtrip": COST_PICK,
                "payoff": (
                    "pnl_exit (production exit-stack walk, relabel_exit_payoff.py)"
                    if args.label == "exit"
                    else "tb_ret (realized at +2%/-3%/120s barrier outcome)"
                ),
                "best_theta_by_cost": {f"{c}": best[c][0] for c in COSTS},
                "ev_degenerate": bool(ev_degenerate),
                "threshold_criterion": criterion,
            },
            "label": (
                f"exit-stack pnl >= {EXIT_WIN_MIN:.1%} (production exits)"
                if args.label == "exit"
                else "up-first: +2% before -3% within 120s (triple barrier)"
            ),
            "notes": (
                "Fitted at band re-entry instants; live scoring reads every "
                "in-band tick — PnL truth comes from replay A/B, not OOF EV."
            ),
        },
    }
    with open(ARTIFACT, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=1)
    print(f"\nwrote {ARTIFACT}")


if __name__ == "__main__":
    main()
