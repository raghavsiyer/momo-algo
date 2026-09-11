#!/usr/bin/env python3
"""
visualize_scoring.py — Plot every quality function curve defined in SCORING_GROUPS.

Reads the live config, builds each ScoringVariable, sweeps the input domain,
and renders quality vs. raw value. One subplot per variable, grouped by
VariableGroup. No hardcoded function shapes — everything comes from config.

Usage:
    python visualize_scoring.py            # interactive matplotlib window
    python visualize_scoring.py --save     # save to scoring_curves.png
"""

import argparse
import math
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from config import SCORING_GROUPS, SETUP_PATTERN_GROUP
from strategy.scoring import QualityFn, ScoringVariable, build_groups

# ── Unit labels — keyed by variable name from SCORING_GROUPS ──────────
# If a variable isn't listed here it falls back to its name.
_VAR_UNITS: dict[str, str] = {
    "breaking_news":   "sentiment (0=bearish, 1=bullish)",
    "ipo_recency":     "days since IPO",
    "reverse_split":   "flag (0=no, 1=yes)",
    "sector":          "tier quality (0–1)",
    "float_shares":    "float (millions)",
    "price":           "price ($)",
    "volume":          "volume (shares)",
    "rvol_1day":       "1-day RVOL (×)",
    "rvol_5min":       "5-min RVOL (×)",
    "tape_density":    "prints / sec",
    "buying_pressure": "buy orders at ask (10s)",
    "selling_pressure": "sell orders at bid (10s)",
    "big_sellers":     "ask size (shares at best ask)",
    "hidden_seller":   "iceberg intensity (0–1)",
    "ofi_chop":        "chop ratio (0–1)",
    "daily_trend":     "total chart range (%)",
    "above_200ema":    "flag (0=below, 1=above both)",
    "ath_proximity":   "price / ATH ratio",
    "vwap_slope":      "VWAP slope ($/candle)",
    "roc":             "rate of change (%)",
    "wick_quality":    "signed wick quality [-,+]",      # legacy
    "body_impulse":    "signed candle body / ATR",       # legacy
    "leg_smoothness":  "signed R² of last N closes [-1,+1]",  # legacy
    "setup_outcome":   "per-TF setup batting average [0,1]",
    # SETUP_PATTERN
    "pullback_depth":       "adjusted depth (0=shallow, 0.45=deep)",
    "pullback_candles":     "candles in pullback",
    "pullback_structure":   "structure quality (0–1)",
    "breakout_extension":   "extension above level (%)",
    "leg_candles":          "consecutive green candles",
}


# ── Colour palette per group ──────────────────────────────────────────
_GROUP_COLORS = {
    "CATALYST":      "#e74c3c",
    "SUPPLY":        "#e67e22",
    "DEMAND":        "#2ecc71",
    "ORDER_FLOW":    "#3498db",
    "TREND":         "#9b59b6",
    "PRICE_ACTION":  "#1abc9c",
    "SETUP_PATTERN": "#f39c12",
}
_DEFAULT_COLOR = "#7f8c8d"


def _sweep_domain(v: ScoringVariable, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Generate (raw_values, quality_values) arrays for a variable's full domain."""
    lo, hi = v.lo, v.hi

    # Determine a sensible sweep range based on function semantics
    if v.fn == QualityFn.SATURATING:
        # Threshold is lo; sweep 0 → 5× threshold
        x_lo, x_hi = 0.0, max(lo * 5, 1.0)
    elif v.fn == QualityFn.LINEAR_MIN:
        # quality=1 at 0, quality=0 at hi.  Show full 0→hi with margin.
        x_lo = 0.0
        x_hi = hi * 1.15 if hi > 0 else 1.0
    elif v.fn == QualityFn.LINEAR_MAX:
        # quality=0 at 0, quality=1 at lo.  Show full 0→lo with margin.
        x_lo = 0.0
        x_hi = lo * 1.15 if lo > 0 else 1.0
    elif v.fn in (QualityFn.BELL, QualityFn.BELL_SKEW, QualityFn.LINEAR_RANGE):
        margin = abs(hi - lo) * 0.15 if hi != lo else 1.0
        x_lo = max(lo, 0.0)
        x_hi = hi + margin
    else:
        x_lo, x_hi = lo, hi

    # For log-scale variables, sweep in raw space (pre-log)
    if v.log_scale:
        x_lo = max(x_lo, 0.0)

    # If variable has a cutoff, end the x-axis slightly past it to show the drop
    if v.cutoff is not None:
        x_hi = v.cutoff * 1.1
    if v.cutoff_lo is not None:
        x_lo = v.cutoff_lo * 0.9
    # For BELL_SKEW with no upper cutoff, end at hi (the bell bound)
    if v.fn == QualityFn.BELL_SKEW and v.cutoff is None:
        x_hi = v.hi * 1.05

    xs = np.linspace(x_lo, x_hi, n)
    ys = np.array([v.evaluate(float(x)) for x in xs])
    return xs, ys


def _format_num(val: float) -> str:
    """Compact number formatting for axis labels."""
    if abs(val) >= 1_000_000:
        return f"{val/1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{val/1_000:.0f}K"
    if val == int(val):
        return str(int(val))
    return f"{val:.2g}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize scoring quality curves")
    parser.add_argument("--save", action="store_true", help="Save to scoring_curves.png instead of showing")
    args = parser.parse_args()

    groups = build_groups(SCORING_GROUPS) + build_groups([SETUP_PATTERN_GROUP])

    # Count total variables for layout
    all_vars: list[tuple[str, str, ScoringVariable]] = []
    for g in groups:
        for v in g.variables:
            all_vars.append((g.name, _GROUP_COLORS.get(g.name, _DEFAULT_COLOR), v))

    n = len(all_vars)
    cols = 3
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.5 * rows))
    fig.suptitle("Quality Function Curves — All Scoring Variables", fontsize=16, fontweight="bold", y=0.995)
    axes = np.array(axes).flatten()

    for i, (group_name, color, var) in enumerate(all_vars):
        ax = axes[i]
        xs, ys = _sweep_domain(var)

        ax.fill_between(xs, ys, alpha=0.15, color=color)
        ax.plot(xs, ys, color=color, linewidth=2)

        # Mark key thresholds
        if var.fn == QualityFn.SATURATING and var.lo > 0:
            ax.axvline(var.lo, color=color, linestyle="--", alpha=0.5, linewidth=1)
            ax.annotate(f"thresh={_format_num(var.lo)}", xy=(var.lo, 0.5),
                        fontsize=7, color=color, alpha=0.8, ha="left",
                        xytext=(5, 5), textcoords="offset points")
        elif var.skew_peak is not None:
            ax.axvline(var.skew_peak, color=color, linestyle="--", alpha=0.5, linewidth=1)
            ax.annotate(f"peak={_format_num(var.skew_peak)}", xy=(var.skew_peak, 1.0),
                        fontsize=7, color=color, alpha=0.8, ha="center",
                        xytext=(0, -12), textcoords="offset points")



        # 0.5 neutral line
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

        # Labels
        ax.set_title(f"{group_name}: {var.name}", fontsize=10, fontweight="bold", color=color)
        unit = _VAR_UNITS.get(var.name, var.name)
        ax.set_xlabel(unit, fontsize=8)
        ax.set_ylabel("quality", fontsize=8)
        ax.set_ylim(-0.05, 1.08)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        # Extra x-axis subdivisions for float and price
        if var.name == "float_shares":
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
            ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21], minor=True)
            ax.grid(True, which="minor", alpha=0.1)
        elif var.name == "price":
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
            ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], minor=True)
            ax.grid(True, which="minor", alpha=0.1)

        # Annotate domain bounds and function type
        fn_label = var.fn.name
        flags = []
        if var.log_scale:
            flags.append("log")
        if var.optional:
            flags.append("opt")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        ax.annotate(f"{fn_label}{flag_str}", xy=(0.02, 0.92), xycoords="axes fraction",
                    fontsize=6.5, color="gray", fontstyle="italic")
        ax.annotate(f"lo={_format_num(var.lo)}", xy=(0.02, 0.02), xycoords="axes fraction",
                    fontsize=6.5, color="gray")
        ax.annotate(f"hi={_format_num(var.hi)}", xy=(0.98, 0.02), xycoords="axes fraction",
                    fontsize=6.5, color="gray", ha="right")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        out = "scoring_curves.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        # Render to temp file and open in Preview (supports scrolling)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
        tmp.close()
        subprocess.run(["open", tmp.name])


if __name__ == "__main__":
    main()
