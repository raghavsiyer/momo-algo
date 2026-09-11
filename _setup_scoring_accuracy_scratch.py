"""Accuracy study for the migrated volume-response candle & setup scoring.

Part A — setup scoring: walk-forward `detect_pullback_new_high` over cached
10s sessions.  For every detected breakout, record the NEW volume-response
topping_tail/jackknife risks (from the signal) and the OLD raw-ATR threshold
metrics (reconstructed from git HEAD), then measure forward outcomes:
  - race: first touch of close+2*ATR (win) vs close-2*ATR (loss) within 90 bars
  - fwd return at +18 bars (3 min) in ATR units
Compare discrimination (AUC, Spearman, tercile win rates) of old vs new
setup_pattern_score = 1 - (clamp01(tt) + clamp01(jk)) / 2.

Part B — candle scoring: per closed 10s candle, compute the key_candles
volume-response risk (same machinery as `_update_key_candles`) and check it
against forward returns at +6 and +18 bars (ATR units), overall and near-HOD.

Run: python _setup_scoring_accuracy_scratch.py
"""
from __future__ import annotations

import glob
import math

import numpy as np
import pandas as pd

import config
from technical_analysis import patterns as P

ATR_PERIOD = config.TOPPING_TAIL_ATR_PERIOD
WINDOW = 160            # trailing bars fed to the detector (live-buffer proxy)
FWD_RACE_BARS = 90      # 15 min on 10s TF
FWD_RET_BARS = 18       # 3 min
RACE_ATR_MULT = 2.0


# ---------------------------------------------------------------- old math
def old_setup_rejection(opens, highs, lows, closes, *, leg_high_idx,
                        pb_start, pb_end, end_idx, leg_high):
    """Pre-migration `_setup_rejection_metrics` (raw ATR ratios, hard gate)."""
    atr = P._atr_from_arrays(highs, lows, closes, end_idx, ATR_PERIOD)
    if atr <= 0:
        return 0.0, 0.0
    min_high = leg_high * config.PATTERN_HOD_PROXIMITY
    tt = jk = 0.0
    for idx in [leg_high_idx, *range(pb_start, pb_end + 1)]:
        if idx < 0 or idx >= len(highs):
            continue
        high = float(highs[idx])
        if high < min_high:
            continue
        o = float(opens[idx])
        c = float(closes[idx])
        upper_wick = max(0.0, high - max(o, c))
        ratio = upper_wick / atr
        if ratio >= config.PATTERN_MIN_RAW:
            tt = max(tt, ratio)
        if c < o:
            body = (o - c) / atr
            if body >= config.PATTERN_MIN_RAW:
                jk = max(jk, body)
    return tt, jk


# ---------------------------------------------------------------- helpers
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def setup_score(tt: float, jk: float) -> float:
    """SETUP_PATTERN group score: equal-weight LINEAR_MIN(0,1) qualities."""
    return 1.0 - (clamp01(tt) + clamp01(jk)) / 2.0


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_win > score_loss) with tie credit."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    all_v = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(all_v):
        j = i
        while j + 1 < len(all_v) and all_v[j + 1] == all_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def tercile_table(scores: np.ndarray, wins: np.ndarray, losses: np.ndarray,
                  fwd: np.ndarray, label: str) -> str:
    qs = np.quantile(scores, [1 / 3, 2 / 3])
    rows = []
    for name, mask in (
        ("low ", scores <= qs[0]),
        ("mid ", (scores > qs[0]) & (scores <= qs[1])),
        ("high", scores > qs[1]),
    ):
        n = int(mask.sum())
        w = int(wins[mask].sum())
        l = int(losses[mask].sum())
        wr = w / max(1, w + l)
        rows.append(
            f"    {name} n={n:4d}  score[{scores[mask].min():.2f},{scores[mask].max():.2f}]"
            f"  win_rate={wr:.3f} ({w}W/{l}L)  mean_fwd3m={fwd[mask].mean():+.3f} ATR"
        )
    return f"  {label} terciles:\n" + "\n".join(rows)


# ---------------------------------------------------------------- Part A
def run_part_a(files: list[str]):
    recs = []
    for si, f in enumerate(files):
        try:
            df = pd.read_csv(f, usecols=["ts", "open", "high", "low", "close", "volume"])
        except Exception:
            continue
        n = len(df)
        if n < 60:
            continue
        O = df["open"].astype(float).values
        H = df["high"].astype(float).values
        L = df["low"].astype(float).values
        C = df["close"].astype(float).values
        V = df["volume"].astype(float).values
        TS = pd.to_datetime(df["ts"]).values

        last_sig = (-10, 0.0)  # (bar, leg_high) dedupe
        for i in range(20, n - FWD_RET_BARS):
            if C[i] <= H[i - 1]:  # breakout precondition
                continue
            s = max(0, i - WINDOW + 1)
            win = pd.DataFrame({
                "ts": TS[s:i + 1], "open": O[s:i + 1], "high": H[s:i + 1],
                "low": L[s:i + 1], "close": C[s:i + 1], "volume": V[s:i + 1],
            })
            sigs = P.detect_pullback_new_high(win, "10sec")
            if not sigs:
                continue
            sig = sigs[0]
            bidx = len(win) - 1
            wo, wh, wl, wc = (win["open"].values, win["high"].values,
                              win["low"].values, win["close"].values)
            window_res = P._resolved_pullback_window(wo, wh, wc, bidx)
            if window_res is None:
                continue
            pb_start, pb_end, _lvl = window_res
            leg = P._setup_green_leg(wo, wh, wl, wc, pb_start - 1)
            if leg is None:
                continue
            _ls, leg_high_idx, _lo, leg_high, _sz = leg
            if abs(leg_high - last_sig[1]) < 1e-9 and i - last_sig[0] <= 6:
                continue  # duplicate fire off the same leg
            last_sig = (i, leg_high)

            tt_old, jk_old = old_setup_rejection(
                wo, wh, wl, wc, leg_high_idx=leg_high_idx, pb_start=pb_start,
                pb_end=pb_end, end_idx=bidx, leg_high=leg_high)

            atr_sig = P._atr_from_arrays(H, L, C, i, ATR_PERIOD)
            if atr_sig <= 0:
                continue
            # pullback volume character (quiet vs loud) via window baseline
            base = P._volume_response_baseline(win["volume"].values)
            pb_vol_ratio = float("nan")
            if base is not None and base[0] > 0:
                pb_vols = V[s + pb_start: s + pb_end + 1]
                pos = pb_vols[pb_vols > 0]
                if len(pos):
                    pb_vol_ratio = float(pos.max() / base[0])

            up = C[i] + RACE_ATR_MULT * atr_sig
            dn = C[i] - RACE_ATR_MULT * atr_sig
            outcome = -1  # neutral
            for j in range(i + 1, min(n, i + 1 + FWD_RACE_BARS)):
                hit_up = H[j] >= up
                hit_dn = L[j] <= dn
                if hit_dn:      # conservative: same-bar both -> loss
                    outcome = 0
                    break
                if hit_up:
                    outcome = 1
                    break
            j_end = min(n - 1, i + FWD_RET_BARS)
            fwd3m = (C[j_end] - C[i]) / atr_sig
            recs.append((tt_old, jk_old, sig.topping_tail, sig.jackknife,
                         outcome, fwd3m, pb_vol_ratio))
        if (si + 1) % 150 == 0:
            print(f"  ... {si + 1}/{len(files)} sessions, {len(recs)} signals")

    if not recs:
        print("PART A: no signals found")
        return
    a = np.array(recs, dtype=float)
    tt_old, jk_old, tt_new, jk_new, outcome, fwd3m, pbvr = a.T
    score_old = np.array([setup_score(t, j) for t, j in zip(tt_old, jk_old)])
    score_new = np.array([setup_score(t, j) for t, j in zip(tt_new, jk_new)])
    wins = (outcome == 1).astype(int)
    losses = (outcome == 0).astype(int)
    decided = outcome >= 0

    print(f"\nPART A — setup scoring ({len(a)} signals, "
          f"{wins.sum()}W / {losses.sum()}L / {int((outcome < 0).sum())} neutral)")
    print(f"  base win rate (decided): {wins.sum() / max(1, decided.sum()):.3f}")
    print(f"  score distribution  old: mean={score_old.mean():.3f} med={np.median(score_old):.3f}"
          f"   new: mean={score_new.mean():.3f} med={np.median(score_new):.3f}")
    print(f"  |new-old| > 0.10 on {(np.abs(score_new - score_old) > 0.10).mean():.1%} of signals; "
          f"corr(old,new)={np.corrcoef(score_old, score_new)[0, 1]:.3f}")
    print(f"  AUC (win vs loss)   old: {auc(score_old[decided], outcome[decided]):.4f}"
          f"   new: {auc(score_new[decided], outcome[decided]):.4f}")
    print(f"  Spearman vs fwd3m   old: {spearman(score_old, fwd3m):+.4f}"
          f"   new: {spearman(score_new, fwd3m):+.4f}")
    print(tercile_table(score_old, wins, losses, fwd3m, "OLD score"))
    print(tercile_table(score_new, wins, losses, fwd3m, "NEW score"))

    # doctrine check: quiet vs loud pullbacks
    has_vr = np.isfinite(pbvr)
    quiet = has_vr & (pbvr < 1.0)
    loud = has_vr & (pbvr >= config.VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN)
    for name, m in (("quiet pullback (max vol < baseline)", quiet),
                    ("loud pullback  (max vol >= 1.5x baseline)", loud)):
        md = m & decided
        if md.sum() == 0:
            continue
        wr = wins[md].sum() / max(1, wins[md].sum() + losses[md].sum())
        print(f"  {name}: n={int(m.sum())} win_rate={wr:.3f} "
              f"mean_fwd3m={fwd3m[m].mean():+.3f} "
              f"old_score={score_old[m].mean():.3f} new_score={score_new[m].mean():.3f}")


# ---------------------------------------------------------------- Part B
def run_part_b(files: list[str]):
    floor_c = clamp01(config.SETUP_VOLUME_SEVERITY_FLOOR)  # noqa: F841 (parity note)
    rej_all, fwd1_all, fwd3_all, nearhod_all = [], [], [], []
    for si, f in enumerate(files):
        try:
            df = pd.read_csv(f, usecols=["open", "high", "low", "close", "volume"])
        except Exception:
            continue
        n = len(df)
        if n < 60:
            continue
        O = df["open"].astype(float).values
        H = df["high"].astype(float).values
        L = df["low"].astype(float).values
        C = df["close"].astype(float).values
        V = df["volume"].astype(float).values
        run_hod = np.maximum.accumulate(H)

        baseline = None
        next_refresh = 0
        for i in range(20, n - FWD_RET_BARS):
            if i >= next_refresh:
                baseline = P._volume_response_baseline(V[: i + 1])
                next_refresh = i + 30
            if baseline is None:
                continue
            vol = V[i]
            if not (math.isfinite(vol) and vol > 0):
                continue
            atr = P._atr_from_arrays(H, L, C, i, ATR_PERIOD)
            if atr <= 0:
                continue
            res = P._candle_volume_response(
                O[i], H[i], L[i], C[i], vol, atr, baseline[0], baseline[1])
            if res is None:
                continue
            vw, structural, red = res
            if vw <= 0:
                continue  # eligibility gate, mirrors _update_key_candles
            rejection = min(1.0, 1.0 - (1.0 - structural) * (1.0 - red))
            rej_all.append(rejection)
            fwd1_all.append((C[i + 6] - C[i]) / atr)
            fwd3_all.append((C[i + FWD_RET_BARS] - C[i]) / atr)
            nearhod_all.append(H[i] >= 0.995 * run_hod[i])
        if (si + 1) % 150 == 0:
            print(f"  ... {si + 1}/{len(files)} sessions, {len(rej_all)} candles")

    rej = np.array(rej_all)
    f1 = np.array(fwd1_all)
    f3 = np.array(fwd3_all)
    nh = np.array(nearhod_all, dtype=bool)
    print(f"\nPART B — candle scoring ({len(rej)} volume-eligible candles, "
          f"{int(nh.sum())} near-HOD)")
    for label, mask in (("all candles", np.ones(len(rej), dtype=bool)),
                        ("near-HOD   ", nh)):
        if mask.sum() < 100:
            continue
        print(f"  {label}: Spearman(rejection, fwd1m)={spearman(rej[mask], f1[mask]):+.4f}"
              f"  (rejection, fwd3m)={spearman(rej[mask], f3[mask]):+.4f}")
    print("  near-HOD forward returns by rejection-risk bucket:")
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    for lo, hi in bins:
        m = nh & (rej >= lo) & (rej < hi)
        if m.sum() < 30:
            continue
        print(f"    rej [{lo:.1f},{hi:.1f}): n={int(m.sum()):6d} "
              f"mean_fwd1m={f1[m].mean():+.3f} mean_fwd3m={f3[m].mean():+.3f} "
              f"pct_neg_3m={(f3[m] < 0).mean():.1%}")


if __name__ == "__main__":
    files = sorted(f for f in glob.glob("logs/chart_cache/ohlcv_10s_*.csv")
                   if "_TEST_" not in f)
    print(f"{len(files)} cached 10s sessions")
    run_part_a(files)
    run_part_b(files)
