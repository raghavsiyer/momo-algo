"""tests/test_weight_breakdown.py — Effective-weight breakdown utility."""

import pytest

from config import SCORING_GROUPS
from strategy.scoring import (
    build_groups,
    effective_weights,
    format_weight_breakdown,
)


class TestEffectiveWeights:
    def test_sums_to_one(self):
        groups = build_groups(SCORING_GROUPS)
        rows = effective_weights(groups)
        total = sum(r["effective"] for r in rows)
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_sorted_descending(self):
        groups = build_groups(SCORING_GROUPS)
        rows = effective_weights(groups)
        effs = [r["effective"] for r in rows]
        assert effs == sorted(effs, reverse=True)

    def test_row_schema(self):
        groups = build_groups(SCORING_GROUPS)
        rows = effective_weights(groups)
        expected = {
            "group", "category", "variable", "fn",
            "var_weight", "var_norm",
            "group_weight", "group_norm",
            "effective", "effective_pct",
            "lo", "hi", "optional",
        }
        assert expected.issubset(rows[0].keys())

    def test_synthetic_two_groups_two_vars(self):
        # 2 groups, equal group weights, equal var weights → all 4 effective = 0.25
        from strategy.scoring import VariableGroup, ScoringVariable, QualityFn
        g1 = VariableGroup(name="A", weight=1.0, variables=[
            ScoringVariable(name="a1", weight=1.0, fn=QualityFn.LINEAR_MAX, lo=0, hi=1),
            ScoringVariable(name="a2", weight=1.0, fn=QualityFn.LINEAR_MAX, lo=0, hi=1),
        ])
        g2 = VariableGroup(name="B", weight=1.0, variables=[
            ScoringVariable(name="b1", weight=1.0, fn=QualityFn.LINEAR_MAX, lo=0, hi=1),
            ScoringVariable(name="b2", weight=1.0, fn=QualityFn.LINEAR_MAX, lo=0, hi=1),
        ])
        rows = effective_weights([g1, g2])
        assert all(r["effective"] == pytest.approx(0.25) for r in rows)

    def test_double_weight_doubles_effective(self):
        from strategy.scoring import VariableGroup, ScoringVariable, QualityFn
        g = VariableGroup(name="G", weight=1.0, variables=[
            ScoringVariable(name="big",   weight=2.0, fn=QualityFn.LINEAR_MAX, lo=0, hi=1),
            ScoringVariable(name="small", weight=1.0, fn=QualityFn.LINEAR_MAX, lo=0, hi=1),
        ])
        rows = effective_weights([g])
        big = next(r for r in rows if r["variable"] == "big")
        small = next(r for r in rows if r["variable"] == "small")
        assert big["effective"] == pytest.approx(2.0 * small["effective"], abs=1e-3)


class TestFormatBreakdown:
    def test_format_returns_string(self):
        groups = build_groups(SCORING_GROUPS)
        out = format_weight_breakdown(groups)
        assert isinstance(out, str)
        # Sanity: header present, every group represented
        assert "effective" in out
        for g in groups:
            assert g.name in out

    def test_format_sort_by_group(self):
        groups = build_groups(SCORING_GROUPS)
        out = format_weight_breakdown(groups, sort="group")
        assert isinstance(out, str)
        assert "Σ effective" in out
