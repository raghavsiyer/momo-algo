"""tests/test_entry_gate.py — Stage-4 entry gate behaviour on WatchlistEntry."""

from datetime import datetime, timezone

import pytest

from config import (
    SCORING_GROUPS,
    CRITERIA_ATTENTION,
    CRITERIA_EXECUTION,
    ENTRY_CONFIDENCE_MIN,
)
from strategy.evaluator import CompositeEvaluation, WatchlistEntry
from strategy.scoring import build_groups
from strategy.criteria import build_criteria_groups


def _make_entry(*, all_qualities: float = 0.8, exec_all_pass: bool = True) -> WatchlistEntry:
    """Build a minimal WatchlistEntry with controllable evaluation + criteria."""
    groups = build_groups(SCORING_GROUPS)
    gmap = {g.name: g for g in groups}
    for g in groups:
        for v in g.variables:
            v.quality = all_qualities
            v.value = all_qualities
    gmap = {g.name: g for g in groups}
    stock_eval = gmap["STOCK_EVAL"]
    stock_eval.get("float_shares").value = 5.0
    stock_eval.get("rvol_1day").value = 10.0
    stock_eval.get("price").value = 5.0
    stock_eval.get("daily_trend").value = 25.0

    ev = CompositeEvaluation(
        stock_eval=stock_eval,
        technical_analysis=gmap["TECHNICAL_ANALYSIS"],
    )

    cgs = build_criteria_groups(CRITERIA_ATTENTION + CRITERIA_EXECUTION)
    if exec_all_pass:
        for g in cgs:
            if g.name == "EXECUTION":
                for c in g.criteria:
                    c.update(True, 1.0)
    # Seed ATTENTION criteria as passing (stock is "in play")
    for g in cgs:
        if g.name == "ATTENTION":
            for c in g.criteria:
                c.update(True, 1.0)

    entry = WatchlistEntry(
        symbol="TEST", evaluated_at=datetime.now(tz=timezone.utc),
        price=5.00, change_pct=15.0, volume=1_000_000, avg_volume=200_000,
        rvol_1day=10.0, rvol_5min=5.0, daily_high=5.10,
        float_shares_m=5.0, free_float_pct=0.5, sector="energy", industry="oil",
        country="US",
        ipo_date=None, days_since_ipo=200, has_recent_reverse_split=False,
        reverse_split_ratio=None, reverse_split_date=None,
        news_count_last_hour=1, news_is_fresh=True, news_is_breaking=True,
        avg_sentiment=0.7,
        evaluation=ev,
        criteria_groups=cgs,
    )
    return entry


class TestEntryAllowed:
    def test_blocked_when_confidence_too_low(self):
        # all qualities 0.1 → eval/live confidence = 10 → below ENTRY_CONFIDENCE_MIN
        entry = _make_entry(all_qualities=0.1, exec_all_pass=True)
        assert entry.eval_confidence < ENTRY_CONFIDENCE_MIN
        assert entry.live_confidence < ENTRY_CONFIDENCE_MIN
        assert entry.entry_allowed is False
        assert any("confidence" in r for r in entry.entry_block_reasons)

    def test_blocked_when_any_required_exec_failing(self):
        entry = _make_entry(all_qualities=0.9, exec_all_pass=True)
        assert entry.eval_confidence >= ENTRY_CONFIDENCE_MIN
        assert entry.live_confidence >= ENTRY_CONFIDENCE_MIN
        # Flip one required gate
        execg = entry.get_criteria_group("EXECUTION")
        execg.update("tape_speed", False)
        assert entry.entry_allowed is False
        assert "execution:tape_speed" in entry.entry_block_reasons

    def test_confidence_bypass_skips_eval_and_live_confidence(self, monkeypatch):
        monkeypatch.setattr("config.CONFIDENCE_BYPASS", True)
        entry = _make_entry(all_qualities=0.1, exec_all_pass=True)

        assert entry.eval_confidence < ENTRY_CONFIDENCE_MIN
        assert entry.live_confidence < ENTRY_CONFIDENCE_MIN
        assert entry.entry_allowed is True
        assert not any("confidence" in r for r in entry.entry_block_reasons)

    def test_eval_bypass_still_enforces_live_confidence(self, monkeypatch):
        monkeypatch.setattr("config.CONFIDENCE_BYPASS_EVAL", True)
        entry = _make_entry(all_qualities=0.1, exec_all_pass=True)

        assert entry.has_live_data is True
        assert entry.entry_allowed is False
        reasons = entry.entry_block_reasons
        assert not any("eval_confidence" in r for r in reasons)
        assert any("live_confidence" in r for r in reasons)

    def test_live_bypass_still_enforces_eval_confidence(self, monkeypatch):
        monkeypatch.setattr("config.CONFIDENCE_BYPASS_LIVE", True)
        entry = _make_entry(all_qualities=0.1, exec_all_pass=True)

        assert entry.entry_allowed is False
        reasons = entry.entry_block_reasons
        assert any("eval_confidence" in r for r in reasons)
        assert not any("live_confidence" in r for r in reasons)

    def test_both_scoped_bypasses_skip_all_confidence_gates(self, monkeypatch):
        monkeypatch.setattr("config.CONFIDENCE_BYPASS_EVAL", True)
        monkeypatch.setattr("config.CONFIDENCE_BYPASS_LIVE", True)
        entry = _make_entry(all_qualities=0.1, exec_all_pass=True)

        assert entry.entry_allowed is True
        assert not any("confidence" in r for r in entry.entry_block_reasons)

    def test_allowed_when_confidence_high_and_all_exec_pass(self):
        entry = _make_entry(all_qualities=0.9, exec_all_pass=True)
        assert entry.entry_allowed is True
        assert entry.entry_block_reasons == []

    def test_hard_cutoff_blocks_entry_even_when_eval_confidence_passes(self):
        entry = _make_entry(all_qualities=0.9, exec_all_pass=True)
        float_var = entry.evaluation.stock_eval.get("float_shares")
        float_var.evaluate(22.5)

        assert entry.eval_confidence >= ENTRY_CONFIDENCE_MIN
        assert entry.entry_allowed is False
        assert any("stock_eval:float_shares" in r for r in entry.entry_block_reasons)

    def test_attention_required_blocks_entry(self):
        # ATTENTION:dist_from_hod_atr_max is required — failing it
        # must block entry.
        entry = _make_entry(all_qualities=0.9, exec_all_pass=True)
        attn = entry.get_criteria_group("ATTENTION")
        attn.update("dist_from_hod_atr_max", False)
        assert entry.entry_allowed is False
        assert "attention:dist_from_hod_atr_max" in entry.entry_block_reasons
