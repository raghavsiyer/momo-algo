#!/usr/bin/env python3
"""
visualize_candles.py - Visualize candle-response scoring for top-volume candles.

Draws a synthetic move with many low/normal-volume context candles and a
curated set of high-volume candles. The main panel shows only the candles that
make the top-volume cut; a compact strip below keeps the nearby unscored
candles visible for scale/context.

Usage:
    python tests/visuals/visualize_candles.py
    python tests/visuals/visualize_candles.py --save
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    PATTERN_MIN_RAW,
    VOLUME_RESPONSE_RANGE_ATR_FULL,
    VOLUME_RESPONSE_RANGE_ATR_MIN,
    VOLUME_RESPONSE_TOP_VOLUME_PCT,
    VOLUME_RESPONSE_WICK_ATR_FULL,
    VOLUME_RESPONSE_WICK_ATR_MIN,
    VOLUME_RESPONSE_WICK_RANGE_FULL,
    VOLUME_RESPONSE_WICK_RANGE_MIN,
    VOLUME_RESPONSE_RED_VOLUME_MAX_RISK,
    VOLUME_RESPONSE_RED_VOLUME_RATIO_FULL,
    VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN,
)
from strategy.scoring import QualityFn, compute_quality


@dataclass(frozen=True)
class CandleSpec:
    label: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    @property
    def is_green(self) -> bool:
        return self.close >= self.open


@dataclass(frozen=True)
class CandleScore:
    upper_wick_atr: float
    red_body_atr: float
    risk: float
    quality: float
    selected: bool


def _sample_candles() -> list[CandleSpec]:
    """Representative move: many context bars, varied high-volume bars."""
    important: dict[int, CandleSpec] = {
        # These high-volume examples are intentionally shaped as a visible
        # risk/quality ramp at ATR=0.20 while still using the real scoring rule.
        3:  CandleSpec("clean close high", 5.06, 5.34, 5.04, 5.32, 62_000),     # Q100: upper wick < threshold
        8:  CandleSpec("minor upper wick", 5.32, 5.62, 5.30, 5.58, 59_000),    # Q100: below wick significance gate
        13: CandleSpec("mild upper wick", 5.58, 5.82, 5.55, 5.76, 56_000),     # Q70:  0.30 ATR upper wick
        18: CandleSpec("modest red body", 5.76, 5.80, 5.64, 5.68, 54_000),     # Q60:  0.40 ATR red body
        23: CandleSpec("medium upper wick", 5.68, 5.95, 5.66, 5.85, 52_000),   # Q50:  0.50 ATR upper wick
        28: CandleSpec("heavier red body", 5.85, 5.90, 5.70, 5.73, 50_000),    # Q40:  0.60 ATR red body
        33: CandleSpec("heavy upper wick", 5.73, 6.04, 5.71, 5.90, 48_000),    # Q30:  0.70 ATR upper wick
        38: CandleSpec("deep red body", 5.90, 5.96, 5.70, 5.74, 46_000),       # Q20:  0.80 ATR red body
        43: CandleSpec("large upper wick", 5.74, 6.12, 5.71, 5.94, 44_000),    # Q10:  0.90 ATR upper wick
        48: CandleSpec("capped rejection", 5.94, 6.18, 5.70, 5.74, 42_000),    # Q0:   >=1.0 ATR rejection
        53: CandleSpec("lower wick save", 5.74, 5.98, 5.50, 5.95, 40_000),     # Q100: lower wick ignored
        58: CandleSpec("red close low", 5.95, 6.04, 5.68, 5.72, 38_000),       # Q0:   >=1.0 ATR red body
    }

    candles: list[CandleSpec] = []
    for idx in range(60):
        if idx in important:
            candles.append(important[idx])
            continue
        base = 5.02 + 0.010 * idx + 0.06 * math.sin(idx / 3.2)
        green = idx % 4 != 0
        open_ = base
        close = base + (0.035 + 0.006 * (idx % 3) if green else -(0.025 + 0.004 * (idx % 4)))
        high = max(open_, close) + 0.035 + 0.006 * (idx % 5)
        low = min(open_, close) - 0.030 - 0.005 * (idx % 4)
        volume = 3_500 + ((idx * 1_550) % 24_000)
        candles.append(CandleSpec(f"context {idx + 1:02d}", open_, high, low, close, volume))
    return candles


def _selected_indices(candles: list[CandleSpec], top_pct: float) -> set[int]:
    candidates = [idx for idx, candle in enumerate(candles) if candle.volume > 0]
    if not candidates:
        return set()
    pct = min(1.0, max(0.0, top_pct))
    top_count = max(1, math.ceil(len(candidates) * pct)) if pct > 0.0 else 1
    volume_cutoff = sorted((candles[idx].volume for idx in candidates), reverse=True)[top_count - 1]
    return {idx for idx in candidates if candles[idx].volume >= volume_cutoff}


def _score_candle(candle: CandleSpec, atr: float, selected: bool, volume_floor: float) -> CandleScore:
    if atr <= 0.0:
        return CandleScore(0.0, 0.0, 0.0, 1.0, selected)

    def _smoothstep(raw: float) -> float:
        x = max(0.0, min(1.0, raw))
        return x * x * (3.0 - (2.0 * x))

    candle_range = max(0.0, candle.high - candle.low)
    if candle_range <= 0.0:
        return CandleScore(0.0, 0.0, 0.0, 1.0, selected)
    volume_ratio = candle.volume / max(volume_floor, 1.0)
    range_atr = candle_range / atr
    range_weight = _smoothstep(
        (range_atr - VOLUME_RESPONSE_RANGE_ATR_MIN)
        / max(1e-9, VOLUME_RESPONSE_RANGE_ATR_FULL - VOLUME_RESPONSE_RANGE_ATR_MIN)
    )
    upper_wick = max(0.0, candle.high - max(candle.open, candle.close))
    upper_wick_atr = upper_wick / atr
    upper_wick_share = upper_wick / candle_range
    tail_size = _smoothstep(
        (upper_wick_atr - VOLUME_RESPONSE_WICK_ATR_MIN)
        / max(1e-9, VOLUME_RESPONSE_WICK_ATR_FULL - VOLUME_RESPONSE_WICK_ATR_MIN)
    )
    tail_shape = _smoothstep(
        (upper_wick_share - VOLUME_RESPONSE_WICK_RANGE_MIN)
        / max(1e-9, VOLUME_RESPONSE_WICK_RANGE_FULL - VOLUME_RESPONSE_WICK_RANGE_MIN)
    )
    rejection = tail_size * tail_shape

    red_body_atr = 0.0
    if candle.close < candle.open:
        red_body_atr = (candle.open - candle.close) / atr
        if red_body_atr >= PATTERN_MIN_RAW:
            rejection = max(rejection, red_body_atr)

    structural_risk = min(1.0, rejection * range_weight)
    red_volume_risk = 0.0
    if candle.close < candle.open:
        red_volume_intensity = _smoothstep(
            (volume_ratio - VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN)
            / max(1e-9, VOLUME_RESPONSE_RED_VOLUME_RATIO_FULL - VOLUME_RESPONSE_RED_VOLUME_RATIO_MIN)
        )
        red_volume_risk = red_volume_intensity * max(0.0, min(1.0, VOLUME_RESPONSE_RED_VOLUME_MAX_RISK))

    risk = min(1.0, 1.0 - ((1.0 - structural_risk) * (1.0 - red_volume_risk)))
    quality = compute_quality(risk, QualityFn.LINEAR_MIN, 0.0, 1.0)
    return CandleScore(upper_wick_atr, red_body_atr, risk, quality, selected)


def _score_set(candles: list[CandleSpec], atr: float, top_pct: float) -> tuple[list[CandleScore], float, float]:
    selected = _selected_indices(candles, top_pct)
    volumes = sorted(candle.volume for candle in candles if candle.volume > 0)
    if volumes:
        mid = len(volumes) // 2
        volume_floor = volumes[mid] if len(volumes) % 2 else (volumes[mid - 1] + volumes[mid]) / 2.0
    else:
        volume_floor = 1.0
    scores = [_score_candle(candle, atr, idx in selected, volume_floor) for idx, candle in enumerate(candles)]

    weighted_risk = 0.0
    selected_volume = 0.0
    for candle, score in zip(candles, scores):
        if not score.selected:
            continue
        weighted_risk += candle.volume * score.risk
        selected_volume += candle.volume

    aggregate_risk = weighted_risk / selected_volume if selected_volume > 0.0 else 0.0
    aggregate_quality = compute_quality(aggregate_risk, QualityFn.LINEAR_MIN, 0.0, 1.0)
    return scores, aggregate_risk, aggregate_quality


def _format_volume(volume: int) -> str:
    if volume >= 1_000_000:
        return f"{volume / 1_000_000:.1f}M"
    if volume >= 1_000:
        return f"{volume / 1_000:.0f}K"
    return str(volume)


def _axis_label(label: str) -> str:
    labels = {
        "clean close high": "clean high",
        "threshold wick": "threshold",
        "mild upper wick": "mild wick",
        "lower wick save": "wick save",
        "modest red body": "red body",
        "medium upper wick": "medium wick",
        "heavier red body": "heavier red",
        "heavy upper wick": "heavy wick",
        "deep red body": "deep red",
        "large upper wick": "large wick",
        "capped rejection": "capped",
        "red close low": "close low",
    }
    return labels.get(label, label)


def _selected_rows(candles: list[CandleSpec], scores: list[CandleScore]) -> list[tuple[int, CandleSpec, CandleScore]]:
    rows = [(idx, candle, score) for idx, (candle, score) in enumerate(zip(candles, scores)) if score.selected]
    return sorted(rows, key=lambda row: (-row[2].quality, row[2].risk, row[0]))


def _draw_scored_candles(candle_ax, volume_ax, candles: list[CandleSpec], scores: list[CandleScore]) -> None:
    body_width = 0.58
    wick_color = "#2f3437"
    green = "#1f9d55"
    red = "#c0392b"
    selected_color = "#f4b000"
    selected = _selected_rows(candles, scores)

    for plot_idx, (timeline_idx, candle, score) in enumerate(selected):
        color = green if candle.is_green else red
        candle_ax.axvspan(plot_idx - 0.45, plot_idx + 0.45, color=selected_color, alpha=0.10, zorder=0)

        candle_ax.vlines(plot_idx, candle.low, candle.high, color=wick_color, linewidth=1.9, zorder=2)
        body_low = min(candle.open, candle.close)
        body_height = max(abs(candle.close - candle.open), 0.005)
        candle_ax.add_patch(
            Rectangle(
                (plot_idx - body_width / 2, body_low),
                body_width,
                body_height,
                facecolor=color,
                edgecolor=selected_color,
                linewidth=2.0,
                alpha=1.0,
                zorder=3,
            )
        )

        quality_label = f"Q {score.quality * 100:.0f}\nR {score.risk:.2f}"
        candle_ax.text(
            plot_idx,
            candle.high + 0.04,
            quality_label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=selected_color,
            fontweight="bold",
        )
        candle_ax.text(
            plot_idx,
            candle.low - 0.04,
            f"#{timeline_idx + 1}",
            ha="center",
            va="top",
            fontsize=7,
            color="#6c757d",
        )

        volume_ax.bar(
            plot_idx,
            candle.volume,
            color=color,
            edgecolor=selected_color,
            linewidth=1.5,
            alpha=0.90,
        )


def _draw_context_strip(context_ax, candles: list[CandleSpec], scores: list[CandleScore]) -> None:
    green = "#1f9d55"
    red = "#c0392b"
    selected_color = "#f4b000"
    body_width = 0.22
    for idx, (candle, score) in enumerate(zip(candles, scores)):
        color = green if candle.is_green else red
        alpha = 0.82 if score.selected else 0.18
        linewidth = 1.6 if score.selected else 0.5
        if score.selected:
            context_ax.axvspan(idx - 0.42, idx + 0.42, color=selected_color, alpha=0.12, zorder=0)
        context_ax.vlines(idx, candle.low, candle.high, color="#2f3437", linewidth=0.9, alpha=alpha, zorder=2)
        body_low = min(candle.open, candle.close)
        body_height = max(abs(candle.close - candle.open), 0.004)
        context_ax.add_patch(
            Rectangle(
                (idx - body_width / 2, body_low),
                body_width,
                body_height,
                facecolor=color,
                edgecolor=selected_color if score.selected else color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=3,
            )
        )


def _draw_table(table_ax, candles: list[CandleSpec], scores: list[CandleScore]) -> None:
    table_ax.axis("off")
    rows = []
    for timeline_idx, candle, score in _selected_rows(candles, scores):
        rows.append([
            f"#{timeline_idx + 1}",
            candle.label,
            _format_volume(candle.volume),
            f"{score.upper_wick_atr:.2f}",
            f"{score.red_body_atr:.2f}",
            f"{score.risk:.2f}",
            f"{score.quality * 100:.0f}",
        ])
    table = table_ax.table(
        cellText=rows,
        colLabels=["#", "scored candle", "volume", "upper\n/ATR", "red body\n/ATR", "risk", "quality"],
        colWidths=[0.06, 0.27, 0.12, 0.13, 0.14, 0.10, 0.10],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.32)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#23313a")
            cell.set_text_props(color="white", weight="bold")
            continue
        cell.set_facecolor("#fff7dc")
        if col_idx in (5, 6):
            cell.set_text_props(weight="bold")


def _plot(atr: float) -> plt.Figure:
    candles = _sample_candles()
    top_pct = VOLUME_RESPONSE_TOP_VOLUME_PCT
    scores, aggregate_risk, aggregate_quality = _score_set(candles, atr, top_pct)
    selected_count = sum(score.selected for score in scores)
    selected_volume = sum(candle.volume for candle, score in zip(candles, scores) if score.selected)
    selected = _selected_rows(candles, scores)

    fig = plt.figure(figsize=(18, 9.5))
    grid = fig.add_gridspec(3, 2, width_ratios=[2.45, 1.75], height_ratios=[3.0, 0.95, 0.75], hspace=0.34, wspace=0.18)
    candle_ax = fig.add_subplot(grid[0, 0])
    volume_ax = fig.add_subplot(grid[1, 0], sharex=candle_ax)
    context_ax = fig.add_subplot(grid[2, 0])
    table_ax = fig.add_subplot(grid[:, 1])

    _draw_scored_candles(candle_ax, volume_ax, candles, scores)
    _draw_context_strip(context_ax, candles, scores)
    _draw_table(table_ax, candles, scores)

    fig.suptitle(
        "Scored top-volume candle response rating",
        x=0.06,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    subtitle = (
        f"top-volume cut={top_pct:.0%} from config | "
        f"ATR=${atr:.2f} | selected={selected_count}/{len(candles)} | "
        f"selected volume={_format_volume(int(selected_volume))} | "
        f"aggregate risk={aggregate_risk:.2f} -> quality={aggregate_quality * 100:.0f}/100"
    )
    fig.text(0.06, 0.928, subtitle, ha="left", fontsize=10, color="#4f5b62")

    candle_ax.set_ylabel("scored candle price ($)")
    candle_ax.grid(True, axis="y", alpha=0.25)
    candle_ax.set_xlim(-0.7, max(selected_count - 0.3, 0.7))
    candle_ax.set_xticks(range(selected_count))
    candle_ax.set_xticklabels([_axis_label(candle.label) for _, candle, _ in selected], fontsize=8)
    scored_candles = [candle for _, candle, _ in selected] or candles
    price_min = min(candle.low for candle in scored_candles) - 0.12
    price_max = max(candle.high for candle in scored_candles) + 0.22
    candle_ax.set_ylim(price_min, price_max)
    candle_ax.legend(
        handles=[
            Patch(facecolor="#1f9d55", label="green close"),
            Patch(facecolor="#c0392b", label="red close"),
            Patch(facecolor="#f4b000", label="scored top-volume candle"),
        ],
        loc="upper left",
        fontsize=8,
        frameon=False,
    )

    volume_ax.set_ylabel("scored\nvolume")
    volume_ax.set_xticks(range(selected_count))
    volume_ax.tick_params(axis="x", labelbottom=False)
    volume_ax.grid(True, axis="y", alpha=0.25)
    volume_ax.yaxis.set_major_formatter(lambda value, _: _format_volume(int(value)))

    context_ax.set_title("full synthetic move context (muted candles are not scored)", loc="left", fontsize=9, color="#5f6a6a", pad=3)
    context_ax.set_ylabel("context\nprice")
    context_ax.set_xlim(-0.7, len(candles) - 0.3)
    context_ax.set_ylim(min(candle.low for candle in candles) - 0.08, max(candle.high for candle in candles) + 0.08)
    context_ax.set_xticks(range(0, len(candles), 5))
    context_ax.set_xticklabels([str(idx + 1) for idx in range(0, len(candles), 5)], fontsize=7)
    context_ax.tick_params(axis="y", labelsize=7)
    context_ax.grid(True, axis="y", alpha=0.18)
    context_ax.set_xlabel("timeline candle #")

    fig.text(
        0.012,
        0.012,
        "Risk rule: candles score after VOLUME_RESPONSE_RANGE_* size gates; high-volume red closes add VOLUME_RESPONSE_RED_* pressure. "
        "Main panel and table show only scored top-volume candles; context strip shows nearby unscored candles for scale.",
        fontsize=8,
        color="#5f6a6a",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.12, wspace=0.18, hspace=0.34)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize top-volume candle-response scoring")
    parser.add_argument("--atr", type=float, default=0.20, help="ATR dollars used to normalize wick/body size")
    parser.add_argument("--save", action="store_true", help="Save to PNG instead of opening Preview")
    parser.add_argument("--out", default="candle_rating_demo.png", help="Output PNG path when --save is used")
    args = parser.parse_args()

    figure = _plot(args.atr)
    if args.save:
        output_path = Path(args.out)
        figure.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        figure.savefig(temp_file.name, dpi=150, bbox_inches="tight")
        temp_file.close()
        subprocess.run(["open", temp_file.name], check=False)


if __name__ == "__main__":
    main()