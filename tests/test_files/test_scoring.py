"""
tests/test_scoring.py — Unit tests for the quality-function scoring engine.
"""

import math
import pytest

from strategy.scoring import (
    QualityFn,
    ScoringVariable,
    VariableGroup,
    compute_quality,
    build_groups,
    combine_tf_scores,
)


# ── Quality function tests ────────────────────────────────────────────

class TestComputeQuality:
    """Verify each quality function produces correct 0-1 outputs."""

    def test_linear_min_at_zero(self):
        assert compute_quality(0.0, QualityFn.LINEAR_MIN, 0, 10) == 1.0

    def test_linear_min_at_max(self):
        assert compute_quality(10.0, QualityFn.LINEAR_MIN, 0, 10) == 0.0

    def test_linear_min_midpoint(self):
        assert compute_quality(5.0, QualityFn.LINEAR_MIN, 0, 10) == pytest.approx(0.5)

    def test_linear_min_beyond_max(self):
        assert compute_quality(20.0, QualityFn.LINEAR_MIN, 0, 10) == 0.0

    def test_linear_max_at_zero(self):
        assert compute_quality(0.0, QualityFn.LINEAR_MAX, 1, 0) == 0.0

    def test_linear_max_at_threshold(self):
        assert compute_quality(1.0, QualityFn.LINEAR_MAX, 1, 0) == 1.0

    def test_linear_max_above_threshold(self):
        assert compute_quality(5.0, QualityFn.LINEAR_MAX, 1, 0) == 1.0

    def test_linear_range_lo(self):
        assert compute_quality(-10.0, QualityFn.LINEAR_RANGE, -10, 10) == 0.0

    def test_linear_range_hi(self):
        assert compute_quality(10.0, QualityFn.LINEAR_RANGE, -10, 10) == 1.0

    def test_linear_range_mid(self):
        assert compute_quality(0.0, QualityFn.LINEAR_RANGE, -10, 10) == pytest.approx(0.5)

    def test_bell_at_center(self):
        q = compute_quality(5.0, QualityFn.BELL, 0, 10)
        assert q == pytest.approx(1.0, abs=0.001)

    def test_bell_at_edge(self):
        q = compute_quality(0.0, QualityFn.BELL, 0, 10)
        assert q < 0.15  # far from center

    def test_bell_outside_range(self):
        assert compute_quality(-1.0, QualityFn.BELL, 0, 10) == 0.0
        assert compute_quality(11.0, QualityFn.BELL, 0, 10) == 0.0

    def test_bell_skew_at_peak(self):
        q = compute_quality(3.0, QualityFn.BELL_SKEW, 0.5, 20, skew_peak=3.0)
        assert q == pytest.approx(1.0, abs=0.001)

    def test_bell_skew_outside_range(self):
        assert compute_quality(0.0, QualityFn.BELL_SKEW, 0.5, 20, skew_peak=3.0) == 0.0
        assert compute_quality(25.0, QualityFn.BELL_SKEW, 0.5, 20, skew_peak=3.0) == 0.0

    def test_saturating_at_zero(self):
        assert compute_quality(0.0, QualityFn.SATURATING, 5, 0) == 0.0

    def test_saturating_at_threshold(self):
        q = compute_quality(5.0, QualityFn.SATURATING, 5, 0)
        assert q == pytest.approx(0.5, abs=0.01)

    def test_saturating_double_threshold(self):
        q = compute_quality(10.0, QualityFn.SATURATING, 5, 0)
        assert q == pytest.approx(0.75, abs=0.01)

    def test_saturating_large_value(self):
        q = compute_quality(50.0, QualityFn.SATURATING, 5, 0)
        assert q > 0.95

    def test_log_scale_linear_min(self):
        # With log_scale, steep gradient near 0, flat near max
        q_low = compute_quality(1.0, QualityFn.LINEAR_MIN, 0, 20, log_scale=True)
        q_mid = compute_quality(10.0, QualityFn.LINEAR_MIN, 0, 20, log_scale=True)
        assert q_low > q_mid  # 1M float >> 10M float in quality

    def test_none_value(self):
        assert compute_quality(None, QualityFn.LINEAR_MIN, 0, 10) == 0.0

    def test_nan_value(self):
        assert compute_quality(float('nan'), QualityFn.LINEAR_MIN, 0, 10) == 0.0


# ── ScoringVariable tests ────────────────────────────────────────────

class TestScoringVariable:
    def test_evaluate(self):
        v = ScoringVariable("test", 1.0, QualityFn.LINEAR_RANGE, 0, 10)
        q = v.evaluate(5.0)
        assert q == pytest.approx(0.5)
        assert v.quality == pytest.approx(0.5)
        assert v.value == 5.0

    def test_reset(self):
        v = ScoringVariable("test", 1.0, QualityFn.LINEAR_RANGE, 0, 10)
        v.evaluate(5.0)
        v.reset(0.5)
        assert v.quality == 0.5
        assert v.value is None


# ── VariableGroup tests ───────────────────────────────────────────────

class TestVariableGroup:
    def test_score_simple(self):
        g = VariableGroup("test", 1.0, [
            ScoringVariable("a", 1.0, QualityFn.LINEAR_RANGE, 0, 10),
            ScoringVariable("b", 1.0, QualityFn.LINEAR_RANGE, 0, 10),
        ])
        g.variables[0].evaluate(10.0)  # quality=1.0
        g.variables[1].evaluate(0.0)   # quality=0.0
        assert g.score == pytest.approx(0.5)

    def test_score_weighted(self):
        g = VariableGroup("test", 1.0, [
            ScoringVariable("a", 3.0, QualityFn.LINEAR_RANGE, 0, 10),
            ScoringVariable("b", 1.0, QualityFn.LINEAR_RANGE, 0, 10),
        ])
        g.variables[0].evaluate(10.0)  # quality=1.0, weight=3
        g.variables[1].evaluate(0.0)   # quality=0.0, weight=1
        assert g.score == pytest.approx(0.75)

    def test_optional_excluded_when_no_data(self):
        g = VariableGroup("test", 1.0, [
            ScoringVariable("a", 1.0, QualityFn.LINEAR_RANGE, 0, 10, optional=True),
            ScoringVariable("b", 1.0, QualityFn.LINEAR_RANGE, 0, 10),
        ])
        # a: optional, never evaluated → value=None → excluded
        g.variables[1].evaluate(10.0)
        assert g.score == pytest.approx(1.0)

    def test_optional_included_when_has_data(self):
        g = VariableGroup("test", 1.0, [
            ScoringVariable("a", 1.0, QualityFn.LINEAR_RANGE, 0, 10, optional=True),
            ScoringVariable("b", 1.0, QualityFn.LINEAR_RANGE, 0, 10),
        ])
        g.variables[0].evaluate(5.0)  # quality=0.5, optional but has data
        g.variables[1].evaluate(10.0)
        assert g.score == pytest.approx(0.75)

    def test_evaluate_dict(self):
        g = VariableGroup("test", 1.0, [
            ScoringVariable("x", 1.0, QualityFn.LINEAR_RANGE, 0, 10),
            ScoringVariable("y", 1.0, QualityFn.LINEAR_RANGE, 0, 10),
        ])
        sc = g.evaluate_dict({"x": 10.0, "y": 0.0})
        assert sc == pytest.approx(0.5)


# ── Multi-TF combination tests ───────────────────────────────────────

class TestCombineTfScores:
    def test_all_bullish_alignment_bonus(self):
        scores = {"5m": 0.8, "2m": 0.7, "1m": 0.6, "10s": 0.5}
        result = combine_tf_scores(scores)
        # All >= 0.5 → alignment bonus
        base = 0.40 * 0.8 + 0.25 * 0.7 + 0.20 * 0.6 + 0.15 * 0.5
        assert result == pytest.approx(min(1.0, base * 1.15), abs=0.01)

    def test_conflict_penalty(self):
        scores = {"5m": 0.3, "2m": 0.4, "1m": 0.7, "10s": 0.8}
        result = combine_tf_scores(scores)
        # With PRICE_ACTION_TF_2M_ENABLED=False the 2m score is skipped;
        # active weights (5m=0.40, 1m=0.20, 10s=0.15) are renormalised.
        w_sum = 0.40 + 0.20 + 0.15
        base = (0.40 * 0.3 + 0.20 * 0.7 + 0.15 * 0.8) / w_sum
        assert result == pytest.approx(base * 0.7, abs=0.01)

    def test_empty_returns_neutral(self):
        assert combine_tf_scores({}) == 0.5

    def test_masses_none_matches_legacy(self):
        scores = {"5m": 0.3, "2m": 0.4, "1m": 0.7, "10s": 0.8}
        assert combine_tf_scores(scores, None) == combine_tf_scores(scores)
        full = {tf: 1.0 for tf in scores}
        assert combine_tf_scores(scores, full) == pytest.approx(combine_tf_scores(scores))

    def test_negligible_mass_tf_is_ignored(self):
        # A near-zero-mass bearish 5m must neither drag the average nor
        # register as a higher-TF conflict against the bullish lower TFs.
        scores = {"5m": 0.3, "1m": 0.7, "10s": 0.8}
        masses = {"5m": 0.01, "1m": 1.0, "10s": 1.0}
        result = combine_tf_scores(scores, masses)
        w_sum = 0.20 + 0.15
        base = (0.20 * 0.7 + 0.15 * 0.8) / w_sum
        # 5m dropped → remaining TFs all bullish → alignment bonus applies
        assert result == pytest.approx(min(1.0, base * 1.15), abs=0.01)

    def test_partial_mass_scales_blend_weight(self):
        scores = {"1m": 0.9, "10s": 0.1}
        masses = {"1m": 1.0, "10s": 0.25}
        result = combine_tf_scores(scores, masses)
        w_sum = 0.20 * 1.0 + 0.15 * 0.25
        base = (0.20 * 1.0 * 0.9 + 0.15 * 0.25 * 0.1) / w_sum
        # mixed bullish/bearish, no higher-TF present → no bonus, no penalty
        assert result == pytest.approx(base, abs=0.01)

    def test_all_masses_negligible_returns_neutral(self):
        assert combine_tf_scores({"10s": 0.9}, {"10s": 0.0}) == 0.5


# ── build_groups tests ───────────────────────────────────────────────

class TestBuildGroups:
    def test_build_from_config(self):
        defs = [
            {"name": "TEST", "weight": 1.0, "vars": [
                {"name": "x", "weight": 1.0, "fn": "LINEAR_MAX", "lo": 1.0, "hi": 0.0},
            ]},
        ]
        groups = build_groups(defs)
        assert len(groups) == 1
        assert groups[0].name == "TEST"
        assert len(groups[0].variables) == 1
        assert groups[0].variables[0].fn == QualityFn.LINEAR_MAX
