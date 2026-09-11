"""
strategy/scoring.py  —  Quality-function scoring engine.

Maps raw variable values to 0-1 quality via configurable functions
(linear, bell, saturating), then combines them into a weighted composite.

All heavy math uses only stdlib + math; no numpy/pandas dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional

from config import (
    PRICE_ACTION_TF_2M_ENABLED,
    PRICE_ACTION_TF_WEIGHTS,
    PRICE_ACTION_TF_ALIGNMENT_BONUS,
    PRICE_ACTION_TF_CONFLICT_PENALTY,
    LIVE_DAMP_START,
    LIVE_DAMP_FLOOR,
    LIVE_DAMP_HALFLIFE,
)

__all__ = [
    "QualityFn",
    "ScoringVariable",
    "VariableGroup",
    "StaticInputs",
    "LiveInputs",
    "compute_quality",
    "build_groups",
    "effective_weights",
    "format_weight_breakdown",
    "print_weight_breakdown",
]


# ── Quality function types ────────────────────────────────────────────

class QualityFn(IntEnum):
    LINEAR_MIN   = auto()  # lower is better  (float, ipo recency)
    LINEAR_MAX   = auto()  # higher is better  (above 200ema – binary-ish)
    LINEAR_RANGE = auto()  # full-range linear  (daily trend, trade imbalance)
    BELL         = auto()  # gaussian, peak at center  (tape density, spread)
    BELL_SKEW    = auto()  # asymmetric gaussian  (price, volume)
    SATURATING   = auto()  # 1-2^(-v/t), threshold=0.5 neutral  (RVOL, vwap slope)


# ── Core quality computation ──────────────────────────────────────────

_LN2 = math.log(2)
_LOG1P = math.log1p  # log(1+x), avoids log(0)


def compute_quality(
    value: float,
    fn: QualityFn,
    lo: float,
    hi: float,
    *,
    skew_peak: Optional[float] = None,
    log_scale: bool = False,
) -> float:
    """Map *value* to [0, 1] quality using the specified function.

    Parameters
    ----------
    value : raw input value
    fn    : quality function type
    lo, hi: function domain / thresholds (interpretation varies by fn)
    skew_peak : peak position for BELL_SKEW
    log_scale : pre-transform value & thresholds to log(1+x) space
    """
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0

    # ── log pre-transform (power-law vars: float, volume) ─────────
    if log_scale:
        value = _LOG1P(max(value, 0.0))
        lo = _LOG1P(max(lo, 0.0))
        hi = _LOG1P(max(hi, 0.0))
        if skew_peak is not None:
            skew_peak = _LOG1P(max(skew_peak, 0.0))

    # ── dispatch ──────────────────────────────────────────────────
    if fn == QualityFn.LINEAR_MIN:
        return _linear_min(value, hi)
    if fn == QualityFn.LINEAR_MAX:
        return _linear_max(value, lo)
    if fn == QualityFn.LINEAR_RANGE:
        return _linear_range(value, lo, hi)
    if fn == QualityFn.BELL:
        return _bell(value, lo, hi)
    if fn == QualityFn.BELL_SKEW:
        return _bell_skew(value, lo, hi, skew_peak)
    if fn == QualityFn.SATURATING:
        return _saturating(value, lo)
    return 0.0


# ── Individual quality functions ──────────────────────────────────────

def _linear_min(value: float, max_thresh: float) -> float:
    """Lower is better.  quality=1 at 0, quality=0 at max_thresh."""
    if max_thresh <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - value / max_thresh))


def _linear_max(value: float, min_thresh: float) -> float:
    """Higher is better.  quality=0 at 0, quality=1 at min_thresh+."""
    if min_thresh <= 0:
        return 1.0 if value > 0 else 0.0
    return max(0.0, min(1.0, value / min_thresh))


def _linear_range(value: float, lo: float, hi: float) -> float:
    """Full-range linear map.  lo→0, hi→1."""
    span = hi - lo
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / span))


def _bell(value: float, lo: float, hi: float) -> float:
    """Gaussian centered at midpoint, σ=(hi-lo)/4.  0 outside [lo, hi]."""
    if value < lo or value > hi:
        return 0.0
    span = hi - lo
    if span <= 0:
        return 0.0
    mu = (lo + hi) * 0.5
    sigma = span * 0.25
    return math.exp(-0.5 * ((value - mu) / sigma) ** 2)


def _bell_skew(
    value: float,
    lo: float,
    hi: float,
    peak: Optional[float] = None,
) -> float:
    """Symmetric Gaussian in linear space, peak at skew_peak.

    lo/hi are the full bell extensions (lo may be negative for symmetry).
    Hard cutoffs are applied separately in ScoringVariable.evaluate().
    σ calibrated so quality ≈ 0.05 at both bounds.
    """
    if value < lo or value > hi:
        return 0.0
    span = hi - lo
    if span <= 0:
        return 0.0
    if peak is None:
        peak = (lo + hi) * 0.5
    peak = max(lo, min(hi, peak))
    # σ from peak to hi (symmetric so same to lo)
    sigma = (hi - peak) / 2.45 if hi > peak else 1e-9
    return math.exp(-0.5 * ((value - peak) / sigma) ** 2)


def _saturating(value: float, threshold: float) -> float:
    """Saturating curve: quality = 1 - 2^(-value/threshold).

    threshold maps to 0.5 (neutral midpoint).
    Negative values → 0.
    """
    if threshold <= 0:
        return 1.0 if value > 0 else 0.0
    if value <= 0:
        return 0.0
    return 1.0 - math.pow(2.0, -value / threshold)


# ── ScoringVariable ───────────────────────────────────────────────────

@dataclass(slots=True)
class ScoringVariable:
    """A single scored input with weight, quality function, and thresholds."""

    name: str
    weight: float
    fn: QualityFn
    lo: float           # min threshold / domain bound
    hi: float           # max threshold / domain bound

    skew_peak: Optional[float] = None
    cutoff: Optional[float] = None     # hard upper bound (value > cutoff → quality 0)
    cutoff_lo: Optional[float] = None  # hard lower bound (value < cutoff_lo → quality 0)
    log_scale: bool = False
    optional: bool = False      # if True & quality=0 → excluded from group avg
    dampened: bool = False       # if True, live updates use EMA smoothing

    value: Optional[float] = field(default=None, repr=False)
    quality: float = 0.0
    _n_obs: int = field(default=0, repr=False)  # live observation count (for dampening)

    def evaluate(self, raw: float) -> float:
        """Compute quality from *raw* value.  Returns quality (0-1)."""
        # Hard cutoffs (use getattr so instances unpickled before these
        # slots existed don't raise AttributeError)
        cutoff = getattr(self, "cutoff", None)
        if cutoff is not None and raw > cutoff:
            self.quality = 0.0
            self.value = raw
            return 0.0
        cutoff_lo = getattr(self, "cutoff_lo", None)
        if cutoff_lo is not None and raw < cutoff_lo:
            self.quality = 0.0
            self.value = raw
            return 0.0
        new_q = compute_quality(
            raw, self.fn, self.lo, self.hi,
            skew_peak=self.skew_peak,
            log_scale=self.log_scale,
        )
        if self.dampened and self.value is not None:
            # EMA with decaying alpha: α = max(floor, start / (1 + n/halflife))
            alpha = max(LIVE_DAMP_FLOOR, LIVE_DAMP_START / (1.0 + self._n_obs / LIVE_DAMP_HALFLIFE))
            self.quality = alpha * new_q + (1.0 - alpha) * self.quality
            self._n_obs += 1
        else:
            self.quality = new_q
            if self.dampened:
                self._n_obs = 1
        self.value = raw
        return self.quality

    def reset(self, neutral: float = 0.5) -> None:
        """Reset to neutral (for live vars before data arrives)."""
        self.value = None
        self.quality = neutral
        self._n_obs = 0


# ── VariableGroup ─────────────────────────────────────────────────────

@dataclass(slots=True)
class VariableGroup:
    """A weighted collection of scoring variables measuring one concept."""

    name: str
    weight: float
    variables: list[ScoringVariable] = field(default_factory=list)
    category: str = "QUALITY"  # spec bucket: QUALITY | STRENGTH_BUY | STRENGTH_SELL | SETUP
    # Optional override for `score` — used by per-TF combiners (e.g.
    # TECHNICAL_ANALYSIS) where the group's true score is computed from
    # sub-scores rather than the weighted avg of its variables' qualities.
    score_override: Optional[float] = None

    @property
    def score(self) -> float:
        """Weighted average of active (non-excluded) variables."""
        # getattr guards against unpickled instances that predate score_override.
        override = getattr(self, "score_override", None)
        if override is not None:
            return override
        w_sum = 0.0
        wq_sum = 0.0
        for v in self.variables:
            if v.value is None and (v.optional or v.dampened):
                continue  # no data yet → excluded from denominator
            # Unevaluated non-optional/non-dampened vars → neutral 0.5 (data not found)
            q = v.quality if v.value is not None else 0.5
            w_sum += v.weight
            wq_sum += v.weight * q
        return wq_sum / w_sum if w_sum > 0.0 else 0.0

    @property
    def has_active_vars(self) -> bool:
        """True if at least one variable has been evaluated (value is not None)."""
        return any(v.value is not None for v in self.variables)

    def evaluate_dict(self, values: dict[str, float]) -> float:
        """Evaluate all variables from a {name: raw_value} dict.

        Keys missing from *values* are skipped (left at current quality).
        Returns the group score.
        """
        for v in self.variables:
            if v.name in values:
                v.evaluate(values[v.name])
        return self.score

    def get(self, name: str) -> Optional[ScoringVariable]:
        """Lookup a variable by name, or None."""
        for v in self.variables:
            if v.name == name:
                return v
        return None

    def reset_live(self, neutral: float = 0.5) -> None:
        """Reset all live (value=None) variables to neutral quality."""
        for v in self.variables:
            if v.value is None:
                v.quality = neutral


# ── Input dataclasses ─────────────────────────────────────────────────

@dataclass(slots=True)
class StaticInputs:
    """Pre-processed raw values for one-time static evaluation.

    Each field name matches the corresponding ``ScoringVariable.name``
    in ``SCORING_GROUPS``.  ``None`` means "skip / leave at default"
    (useful for optional variables like ``breaking_news``).
    """

    # STOCK_EVAL
    float_shares: Optional[float] = None   # float in millions (None = not found)
    breaking_news: Optional[float] = None   # (sentiment+1)/2 ∈ [0,1], None = no news
    rvol_1day: Optional[float] = None      # 1-day RVOL (None = not found)
    price: float = 0.0
    daily_trend: float = 0.0           # daily base risk (%) -- lower = cleaner consolidation


@dataclass(slots=True)
class LiveInputs:
    """Raw values updated in real-time by the dynamic scorer.

    Each field name matches the corresponding ``ScoringVariable.name``
    in ``SCORING_GROUPS``.  ``None`` means "don't update this cycle"
    so partial updates work naturally.
    """

    # TECHNICAL_ANALYSIS (per-TF setup outcome, combined via combine_tf_scores)
    setup_outcome: Optional[float] = None

    # Poll-derived (candle refresh)
    vwap_slope: Optional[float] = None
    above_200ema: Optional[float] = None

    # Order flow (L2-gated)
    ofi_chop: Optional[float] = None

    # High-volume candle response, computed on candle close by
    # DynamicScorer._update_key_candles. Higher = more rejection risk = worse;
    # lower = cleaner high-volume acceptance = better.
    key_candles: Optional[float] = None

    # ATR as a percent of price (14-period 1-min ATR / price * 100) and candle
    # choppiness (red/green alternation ratio across timeframes, succeeded
    # pullbacks excused). Both computed on candle close by
    # DynamicScorer._update_atrp_and_choppiness.
    atrp: Optional[float] = None
    candle_choppiness: Optional[float] = None


# ── Per-timeframe price action with multi-TF combination ─────────────


# A per-TF evidence mass at or below this is negligible: the TF drops out of
# combine_tf_scores() entirely (average and alignment/conflict alike), and the
# scorer zeroes the setup_outcome group weight when no TF clears it.
MIN_TF_EVIDENCE_MASS: float = 0.05


def combine_tf_scores(
    tf_scores: dict[str, float],
    tf_masses: dict[str, float] | None = None,
) -> float:
    """Combine per-TF price-action scores into a single 0-1 score.

    *tf_scores*: ``{"10s": 0.6, "1m": 0.7, "2m": 0.4, "5m": 0.8}``
    *tf_masses*: optional per-TF evidence mass (0-1) multiplied into each TF's
    blend weight. TFs at or below MIN_TF_EVIDENCE_MASS are ignored entirely.
    None (default) treats every TF as full-mass — identical to the legacy blend.
    """
    if not tf_scores:
        return 0.5  # neutral when no data

    if tf_masses is None:
        masses = {tf: 1.0 for tf in tf_scores}
    else:
        masses = {tf: max(0.0, min(1.0, tf_masses.get(tf, 1.0))) for tf in tf_scores}
        tf_scores = {tf: s for tf, s in tf_scores.items() if masses[tf] > MIN_TF_EVIDENCE_MASS}
        if not tf_scores:
            return 0.5  # nothing with usable evidence

    # weighted average
    w_sum = 0.0
    ws_sum = 0.0
    for tf, sc in tf_scores.items():
        if tf == "2m" and not PRICE_ACTION_TF_2M_ENABLED:
            continue
        w = PRICE_ACTION_TF_WEIGHTS.get(tf, 0.0) * masses[tf]
        w_sum += w
        ws_sum += w * sc
    base = ws_sum / w_sum if w_sum > 0.0 else 0.5

    # alignment / conflict detection
    active_scores = {tf: s for tf, s in tf_scores.items() if tf != "2m" or PRICE_ACTION_TF_2M_ENABLED}
    all_bullish = all(s >= 0.5 for s in active_scores.values())
    all_bearish = all(s < 0.5 for s in active_scores.values())

    if all_bullish or all_bearish:
        return min(1.0, base * PRICE_ACTION_TF_ALIGNMENT_BONUS)

    # conflict: higher TF bearish, lower TF bullish
    higher_tfs = ("5m",) if not PRICE_ACTION_TF_2M_ENABLED else ("5m", "2m")
    lower_tfs = ("1m", "10s")
    higher_bearish = any(tf_scores.get(t, 0.5) < 0.5 for t in higher_tfs)
    lower_bullish = any(tf_scores.get(t, 0.5) >= 0.5 for t in lower_tfs)
    if higher_bearish and lower_bullish:
        return base * PRICE_ACTION_TF_CONFLICT_PENALTY

    return base


# ── Factory: build groups from config dicts ───────────────────────────

def build_groups(
    group_defs: list[dict],
) -> list[VariableGroup]:
    """Construct ``VariableGroup`` instances from config-style dicts.

    Each dict in *group_defs*::

        {
            "name": "CATALYST",
            "weight": 1.0,
            "category": "QUALITY",   # optional, defaults to "QUALITY"
            "vars": [
                {
                    "name": "breaking_news",
                    "weight": 1.0,
                    "fn": "SATURATING",
                    "lo": 0, "hi": 0.5,
                    "optional": True,
                },
                ...
            ],
        }
    """
    groups: list[VariableGroup] = []
    for gd in group_defs:
        variables: list[ScoringVariable] = []
        for vd in gd.get("vars", []):
            variables.append(ScoringVariable(
                name=vd["name"],
                weight=vd.get("weight", 1.0),
                fn=QualityFn[vd["fn"]] if isinstance(vd["fn"], str) else vd["fn"],
                lo=vd.get("lo", 0.0),
                hi=vd.get("hi", 0.0),
                skew_peak=vd.get("skew_peak"),
                cutoff=vd.get("cutoff"),
                cutoff_lo=vd.get("cutoff_lo"),
                log_scale=vd.get("log_scale", False),
                optional=vd.get("optional", False),
                dampened=vd.get("dampened", False),
            ))
        groups.append(VariableGroup(
            name=gd["name"],
            weight=gd.get("weight", 1.0),
            variables=variables,
            category=gd.get("category", "QUALITY"),
        ))
    return groups


# ── Effective-weight breakdown ────────────────────────────────────────


def effective_weights(
    groups: list[VariableGroup],
) -> list[dict]:
    """Compute each variable's *effective* contribution to the composite.

    For variable ``v`` in group ``g``::

        w_var_norm   = v.weight / Σ(weights of vars in g)
        w_group_norm = g.weight / Σ(weights of all groups)
        w_effective  = w_var_norm × w_group_norm

    Σ(w_effective) over all variables ≈ 1.0 (subject to optional vars).
    Returns a list of dicts sorted by descending effective weight,
    suitable for tabular display or DataFrame conversion.
    """
    total_group_w = sum(g.weight for g in groups) or 1.0
    rows: list[dict] = []
    for g in groups:
        group_var_total = sum(v.weight for v in g.variables) or 1.0
        group_norm = g.weight / total_group_w
        for v in g.variables:
            var_norm = v.weight / group_var_total
            rows.append({
                "group":         g.name,
                "category":      g.category,
                "variable":      v.name,
                "fn":            v.fn.name,
                "var_weight":    round(v.weight, 3),
                "var_norm":      round(var_norm, 4),
                "group_weight":  round(g.weight, 3),
                "group_norm":    round(group_norm, 4),
                "effective":     round(var_norm * group_norm, 4),
                "effective_pct": round(var_norm * group_norm * 100.0, 2),
                "lo":            v.lo,
                "hi":            v.hi,
                "optional":      v.optional,
            })
    rows.sort(key=lambda r: r["effective"], reverse=True)
    return rows


def format_weight_breakdown(
    groups: list[VariableGroup],
    *,
    sort: str = "effective",
) -> str:
    """Return a printable table of effective weights.

    *sort*: ``"effective"`` (default) or ``"group"``.
    """
    rows = effective_weights(groups)
    if sort == "group":
        rows.sort(key=lambda r: (r["group"], -r["effective"]))

    headers = ["group", "variable", "fn", "var_w", "grp_w", "effective", "%"]
    widths = [12, 18, 12, 7, 7, 10, 7]
    sep = "  "

    def fmt_row(cells: list[str]) -> str:
        return sep.join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    cur_group: Optional[str] = None
    for r in rows:
        # Group separator line (only meaningful when sorting by group)
        if sort == "group" and r["group"] != cur_group:
            cur_group = r["group"]
        lines.append(fmt_row([
            r["group"],
            r["variable"],
            r["fn"],
            f"{r['var_weight']:.2f}",
            f"{r['group_weight']:.2f}",
            f"{r['effective']:.4f}",
            f"{r['effective_pct']:.2f}",
        ]))
    total = sum(r["effective"] for r in rows)
    lines.append(fmt_row(["-" * w for w in widths]))
    lines.append(f"Σ effective = {total:.4f}  (target ≈ 1.0)")
    return "\n".join(lines)


def print_weight_breakdown(
    groups: Optional[list[VariableGroup]] = None,
    *,
    sort: str = "effective",
) -> None:
    """Print the effective-weight table to stdout.

    With no argument, builds groups from ``config.SCORING_GROUPS``.
    """
    if groups is None:
        from config import SCORING_GROUPS
        groups = build_groups(SCORING_GROUPS)
    print(format_weight_breakdown(groups, sort=sort))
