"""A/B report: legacy vs armed_band entry regime over replayed days.

Reads the per-arm backtest outputs (setup_signals + sim_trades CSVs written
under logs/sim/backtest/<tag>/) and pairs setups by setup_id — the id embeds
(symbol, setup name, timeframe, level, pullback_ts) and setup DETECTION is
regime-invariant, so the same id in both arms is the same trade opportunity.

Per paired setup: entry premium vs level, fill-time delta, forward max/min
from fill (from logs/chart_cache timesales), realized PnL/hold/exit reason
(from sim_trades). Unpaired rows are reported as added/lost entries — never
silently dropped.

Named regressions (--day 20260811 --strict): the five 08-11 live-sim trades
this redesign was measured against.

Run:
  python ab_entry_report.py --days 20260811 [--legacy-tag legacy]
                            [--armed-tag armed] [--strict]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from bisect import bisect_left
from datetime import datetime, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKTEST_DIR = os.path.join(REPO, "logs", "sim", "backtest")
CHART_CACHE = os.path.join(REPO, "logs", "chart_cache")


def _read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class TapeIndex:
    """Forward price lookups from a cached timesales CSV."""

    def __init__(self, symbol: str, day: str) -> None:
        rows = _read_csv(os.path.join(CHART_CACHE, f"timesales_{symbol}_{day}.csv"))
        pairs = []
        for r in rows:
            ts = _parse_ts(r.get("ts", ""))
            try:
                px = float(r.get("price", "") or 0)
            except ValueError:
                continue
            if ts is not None and px > 0:
                pairs.append((ts, px))
        pairs.sort(key=lambda x: x[0])
        self.ts = [p[0] for p in pairs]
        self.px = [p[1] for p in pairs]

    def fwd(self, start: datetime, secs: float) -> tuple[float | None, float | None]:
        if not self.ts:
            return None, None
        i0 = bisect_left(self.ts, start)
        i1 = bisect_left(self.ts, start + timedelta(seconds=secs))
        window = self.px[i0:i1]
        if not window:
            return None, None
        return max(window), min(window)


def _load_arm(tag: str, day: str) -> tuple[dict[str, dict], list[dict]]:
    base = os.path.join(BACKTEST_DIR, tag) if tag else BACKTEST_DIR
    setups = {
        r["setup_id"]: r
        for r in _read_csv(os.path.join(base, f"setup_signals_{day}.csv"))
        if r.get("setup_id")
    }
    trades = _read_csv(os.path.join(base, f"sim_trades_{day}.csv"))
    return setups, trades


def _entered(setups: dict[str, dict]) -> dict[str, dict]:
    return {
        sid: r for sid, r in setups.items()
        if str(r.get("entered", "")).strip().lower() in ("true", "1", "yes")
    }


def _sell_for(trades: list[dict], symbol: str, entry_ts: datetime | None) -> dict | None:
    """First SELL row for symbol at/after entry_ts."""
    best = None
    for r in trades:
        if r.get("symbol") != symbol or r.get("side") != "SELL":
            continue
        ts = _parse_ts(r.get("ts", ""))
        if ts is None or (entry_ts is not None and ts < entry_ts):
            continue
        if best is None or ts < _parse_ts(best["ts"]):
            best = r
    return best


def _fmt_pct(x: float | None) -> str:
    return "--" if x is None else f"{x:+.2%}"


def run_day(day: str, legacy_tag: str, armed_tag: str) -> dict:
    leg_setups, leg_trades = _load_arm(legacy_tag, day)
    arm_setups, arm_trades = _load_arm(armed_tag, day)
    leg_entered, arm_entered = _entered(leg_setups), _entered(arm_setups)
    tapes: dict[str, TapeIndex] = {}

    def tape(symbol: str) -> TapeIndex:
        if symbol not in tapes:
            tapes[symbol] = TapeIndex(symbol, day)
        return tapes[symbol]

    def describe(sid: str, row: dict, trades: list[dict]) -> dict:
        symbol = row.get("symbol", "")
        level = float(row.get("setup_level", 0) or 0)
        entry_price = float(row.get("entry_price", 0) or 0)
        entry_ts = _parse_ts(row.get("entry_ts", ""))
        fwd60 = tape(symbol).fwd(entry_ts, 60) if entry_ts else (None, None)
        fwd120 = tape(symbol).fwd(entry_ts, 120) if entry_ts else (None, None)
        sell = _sell_for(trades, symbol, entry_ts)
        return {
            "setup_id": sid,
            "symbol": symbol,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "level": level,
            "premium": (entry_price / level - 1.0) if level > 0 else None,
            "fwdmax60": (fwd60[0] / entry_price - 1.0) if (fwd60[0] and entry_price) else None,
            "fwdmin120": (fwd120[1] / entry_price - 1.0) if (fwd120[1] and entry_price) else None,
            "pnl_pct": float(sell.get("pnl_pct", 0) or 0) if sell else None,
            "hold_secs": float(sell.get("hold_secs", 0) or 0) if sell else None,
            "exit_reason": (sell or {}).get("exit_reason", ""),
            "arm_state": row.get("arm_state", ""),
        }

    both = sorted(set(leg_entered) & set(arm_entered))
    only_leg = sorted(set(leg_entered) - set(arm_entered))
    only_arm = sorted(set(arm_entered) - set(leg_entered))

    print(f"\n=== {day}  legacy[{legacy_tag}] vs armed[{armed_tag}] ===")
    print(
        f"  setups: legacy {len(leg_setups)} ({len(leg_entered)} entered) | "
        f"armed {len(arm_setups)} ({len(arm_entered)} entered) | "
        f"paired {len(both)}, lost {len(only_leg)}, added {len(only_arm)}"
    )

    paired_rows = []
    for sid in both:
        a = describe(sid, leg_entered[sid], leg_trades)
        b = describe(sid, arm_entered[sid], arm_trades)
        delta_t = (
            (b["entry_ts"] - a["entry_ts"]).total_seconds()
            if a["entry_ts"] and b["entry_ts"] else None
        )
        paired_rows.append((a, b, delta_t))
        print(
            f"  PAIR {a['symbol']:6s} {sid.split(':')[-1][-12:]:>12s}  "
            f"fill {a['entry_price']:.4f}->{b['entry_price']:.4f}  "
            f"premium {_fmt_pct(a['premium'])}->{_fmt_pct(b['premium'])}  "
            f"dt={'--' if delta_t is None else f'{delta_t:+.1f}s'}  "
            f"pnl {_fmt_pct(a['pnl_pct'])}"
            f"->{_fmt_pct(b['pnl_pct'])}  "
            f"[{b['arm_state']}]"
        )
    for sid in only_leg:
        a = describe(sid, leg_entered[sid], leg_trades)
        blocked = arm_setups.get(sid, {})
        print(
            f"  LOST {a['symbol']:6s} fill {a['entry_price']:.4f} "
            f"premium {_fmt_pct(a['premium'])} pnl "
            f"{_fmt_pct(a['pnl_pct'])}  "
            f"armed-state[{blocked.get('arm_state', 'setup absent')}] "
            f"block[{blocked.get('block_reasons', '')[:60]}]"
        )
    for sid in only_arm:
        b = describe(sid, arm_entered[sid], arm_trades)
        print(
            f"  ADDED {b['symbol']:6s} fill {b['entry_price']:.4f} "
            f"premium {_fmt_pct(b['premium'])} "
            f"fwdmax60 {_fmt_pct(b['fwdmax60'])} fwdmin120 {_fmt_pct(b['fwdmin120'])} "
            f"pnl {_fmt_pct(b['pnl_pct'])}  "
            f"[{b['arm_state']}]"
        )

    prem_deltas = [
        b["premium"] - a["premium"]
        for a, b, _ in paired_rows
        if a["premium"] is not None and b["premium"] is not None
    ]
    if prem_deltas:
        prem_deltas.sort()
        med = prem_deltas[len(prem_deltas) // 2]
        print(f"  paired premium delta: median {med:+.2%} over {len(prem_deltas)} pairs")

    return {
        "day": day,
        "paired": paired_rows,
        "lost": [describe(s, leg_entered[s], leg_trades) for s in only_leg],
        "added": [describe(s, arm_entered[s], arm_trades) for s in only_arm],
        "leg_entered": leg_entered,
        "arm_entered": arm_entered,
        "arm_trades": arm_trades,
    }


def named_regressions_0811(result: dict) -> list[str]:
    """The five 08-11 live-sim trades, as machine-checked expectations."""
    failures: list[str] = []
    arm = list(result["arm_entered"].values())

    def fills(symbol: str) -> list[float]:
        return [
            float(r.get("entry_price", 0) or 0)
            for r in arm if r.get("symbol") == symbol
        ]

    aihs = fills("AIHS")
    if aihs and not any(3.50 <= f <= 3.78 for f in aihs):
        failures.append(f"AIHS armed fill(s) {aihs} not in the 3.50-3.78 evidence window")
    if not aihs:
        failures.append("AIHS: no armed entry (live-sim entered; armed arm must too)")

    plag = fills("PLAG")
    if not plag:
        failures.append("PLAG: no armed entries (winner setup must still enter)")
    if plag and min(plag) >= 1.3907:
        failures.append(f"PLAG armed fills {plag} show no premium reduction vs 1.3907/1.0605 legacy-style fills")

    sells = [r for r in result["arm_trades"] if r.get("side") == "SELL" and r.get("symbol") == "GLE"]
    for s in sells:
        reason = s.get("exit_reason", "")
        hold = float(s.get("hold_secs", 0) or 0)
        if hold > 60 and not reason.startswith(("structural_stop", "prove_timeout")):
            failures.append(f"GLE hold {hold:.0f}s with exit '{reason}' — unproven holds must die fast")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", required=True, help="comma-separated YYYYMMDD list")
    ap.add_argument("--legacy-tag", default="legacy")
    ap.add_argument("--armed-tag", default="armed")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if 08-11 named regressions fail")
    args = ap.parse_args()

    failures: list[str] = []
    for day in [d.strip() for d in args.days.split(",") if d.strip()]:
        result = run_day(day, args.legacy_tag, args.armed_tag)
        if day == "20260811":
            regs = named_regressions_0811(result)
            if regs:
                print("  NAMED REGRESSIONS FAILED:")
                for r in regs:
                    print(f"    - {r}")
                failures.extend(regs)
            else:
                print("  named regressions: PASS")

    if failures and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
