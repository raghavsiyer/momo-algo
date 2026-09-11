"""Named 08-11 regressions: replay the five live-sim trade windows under
both regimes and compare fills.

exec.replay_session replays a user-chosen window continuously (no near-HOD
gating), through the full production stack, with sim fills captured inside
the bundle — the right instrument for "what would the armed band have done
at 11:02:55", which the day-level backtest's replay windows do not cover.

Regimes are injected via SIM_ENTRY_REGIME / SIM_EXIT_REGIME env vars
(config reads them at import in the child process); the bundle name embeds
config_hash(), which now includes both regimes, so arms never collide.

Run:  python regression_0811.py            (10 replays: 5 windows x 2 arms)
"""
from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")

# Live-sim trades 2026-08-11 (UTC) -> replay windows in LOCAL (PT = UTC-7)
# start/end as exec.replay_session expects.
WINDOWS = [
    # name, symbol, live fill, setup level, PT start, PT end
    ("AIHS 11:02:55", "AIHS", 3.8119, 3.83, "03:50", "04:10"),
    ("PLAG 11:39:09", "PLAG", 1.0605, 1.0302, "04:30", "04:45"),
    ("PLAG 12:11:23", "PLAG", 1.3907, 1.39, "05:02", "05:15"),
    ("PLAG 12:43:56", "PLAG", 1.4007, 1.3799, "05:35", "05:50"),
    ("GLE 13:12:58", "GLE", 0.7173, 0.7105, "06:03", "06:18"),
]

ARMS = {
    "legacy": {"SIM_ENTRY_REGIME": "legacy", "SIM_EXIT_REGIME": "legacy"},
    "armed": {"SIM_ENTRY_REGIME": "armed_band", "SIM_EXIT_REGIME": "thesis_lifecycle"},
}


def run_window(symbol: str, start: str, end: str, arm_env: dict) -> str | None:
    env = dict(os.environ)
    env.update(arm_env)
    proc = subprocess.run(
        [PY, "-m", "exec.replay_session", symbol, "2026-08-11",
         "--start", start, "--end", end, "--force"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=1800,
    )
    out = proc.stdout + proc.stderr
    m = re.search(r"bundle:\s*(\S+)", out)
    if proc.returncode != 0 or not m:
        print(f"    replay FAILED rc={proc.returncode}: {out.strip().splitlines()[-3:]}")
        return None
    return m.group(1)


def bundle_trades(bundle_dir: str) -> list[dict]:
    rows: list[dict] = []
    for path in glob.glob(os.path.join(bundle_dir, "sim", "sim_trades_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def main() -> None:
    results = []
    for name, symbol, live_fill, level, start, end in WINDOWS:
        print(f"\n=== {name}  (live fill {live_fill}, level {level}) ===")
        row = {"name": name, "live_fill": live_fill, "level": level}
        for arm, env in ARMS.items():
            bundle = run_window(symbol, start, end, env)
            if bundle is None:
                row[arm] = None
                continue
            trades = bundle_trades(bundle)
            buys = [t for t in trades if t.get("side") == "BUY"]
            sells = [t for t in trades if t.get("side") == "SELL"]
            if buys:
                fill = float(buys[0]["fill_price"])
                prem = fill / level - 1.0
                pnl = float(sells[0]["pnl_pct"]) if sells else None
                hold = sells[0]["hold_secs"] if sells else "--"
                reason = (sells[0].get("exit_reason", "") if sells else "")[:60]
                entry_reason = buys[0].get("entry_reason", "")[:80]
                print(
                    f"  {arm:6s} fill {fill:.4f} (prem {prem:+.2%})  "
                    f"pnl {pnl if pnl is None else f'{pnl:+.2%}'} hold {hold}s"
                )
                print(f"         entry[{entry_reason}]")
                if reason:
                    print(f"         exit [{reason}]")
                row[arm] = {"fill": fill, "prem": prem, "pnl": pnl,
                            "n_buys": len(buys)}
            else:
                print(f"  {arm:6s} NO ENTRY in window")
                row[arm] = {"fill": None, "prem": None, "pnl": None, "n_buys": 0}
        results.append(row)

    print("\n=== summary (armed vs legacy vs live) ===")
    for r in results:
        leg, arm = r.get("legacy"), r.get("armed")
        leg_f = leg and leg.get("fill")
        arm_f = arm and arm.get("fill")
        print(
            f"  {r['name']:15s} live {r['live_fill']:.4f} | "
            f"legacy {'--' if not leg_f else f'{leg_f:.4f}'} | "
            f"armed {'--' if not arm_f else f'{arm_f:.4f}'}"
            + (
                f"  (armed vs live {arm_f / r['live_fill'] - 1.0:+.2%})"
                if arm_f else ""
            )
        )


if __name__ == "__main__":
    main()
