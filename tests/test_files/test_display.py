from datetime import datetime, timedelta, timezone
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import SCORING_GROUPS
from display import (
    _print_stock_summary, format_live_weights, format_top_gainers,
    format_tape, print_live_detail,
)
from strategy.scoring import build_groups


def _make_tape_view(n_prints: int) -> SimpleNamespace:
    """Duck-typed TapeView with *n_prints* directional prints."""
    ts = datetime(2026, 8, 4, 14, 30, 0, tzinfo=timezone.utc)
    gates = SimpleNamespace(
        all_pass=False, speed=1.0, speed_ok=False, activity=3, activity_ok=False,
        buy_ratio=0.5, ratio_ok=False, vol_imbalance=0.0, vol_flow_ok=False,
        accel_z=0.0, accel_ready=False, accel_ok=False,
    )
    prints = tuple(
        SimpleNamespace(side="buy" if i % 2 == 0 else "sell",
                        ts=ts + timedelta(seconds=i), price=5.0 + i * 0.01, size=100)
        for i in range(n_prints)
    )
    return SimpleNamespace(
        symbol="TEST", prints=prints, retained=n_prints, window_secs=30,
        prints_seen=n_prints, last_price=5.05, bid=5.04, ask=5.06, gates=gates,
        tape_speed_ema=1.0, trade_speed=1.0, baseline_speed=1.0, speed_ratio=1.0,
        buy_ratio_ema=0.5, buy_ratio=0.5, buying_pressure=1, selling_pressure=1,
        density_imbalance=0.0, trade_imbalance=0.0, buy_volume=100, sell_volume=100,
        avg_print_size=100.0, accel_raw=0.0, accel_ready=False, accel_z=0.0,
        accel_n_eff=0.0, dispersion_phi=1.0, hidden_seller=0.0,
    )


class TestFormatTapeFixedHeight(unittest.TestCase):
    """The tape block must not change height as the rolling window grows and
    trims — height oscillation in the toolbar reads as flicker."""

    def test_height_is_constant_across_print_counts(self):
        limit = 10
        heights = {
            n: len(format_tape(_make_tape_view(n), limit=limit).splitlines())
            for n in (1, 4, 10)
        }
        self.assertEqual(len(set(heights.values())), 1, heights)

    def test_overfull_window_is_clamped_to_limit(self):
        limit = 5
        block = format_tape(_make_tape_view(12), limit=limit)
        clamped = format_tape(_make_tape_view(5), limit=limit)
        self.assertEqual(len(block.splitlines()), len(clamped.splitlines()))

    def test_limit_zero_follows_the_window(self):
        """limit <= 0 = 'everything streamed': all prints, no padding."""
        few = format_tape(_make_tape_view(3), limit=0)
        many = format_tape(_make_tape_view(17), limit=0)
        self.assertEqual(
            len(many.splitlines()) - len(few.splitlines()), 17 - 3
        )
        self.assertNotIn("\n\n", many)   # no pad rows in follow mode


class _EvalStub:
    def __init__(self, groups) -> None:
        self.groups = groups

    def group_by_name(self, name: str):
        return next((group for group in self.groups if group.name == name), None)


class _DummyScorer:
    def __init__(self, outcomes=None, active_legs=None) -> None:
        self._completed_outcomes = outcomes or []
        self._active_legs = active_legs or {}


class TestPrintLiveDetail(unittest.TestCase):

    def test_summary_infers_did_not_pass_for_cached_high_float_entry(self):
        groups = build_groups(SCORING_GROUPS)
        stock_eval = next(g for g in groups if g.name == "STOCK_EVAL")
        stock_eval.get("float_shares").evaluate(77.6)
        stock_eval.get("price").evaluate(13.46)
        stock_eval.get("rvol_1day").evaluate(1000.0)

        entry = SimpleNamespace(
            symbol="UTZ",
            evaluation=_EvalStub(groups),
            news_articles=[],
            news_source="",
            news_is_breaking=False,
            news_is_fresh=False,
            news_count_last_hour=0,
            avg_sentiment=None,
            days_since_ipo=None,
            has_recent_reverse_split=False,
            sector="Food Products",
            industry="Food Products",
            country="US",
            float_shares_m=77.6,
            float_source="AV",
            price=13.46,
            change_pct=80.7,
            volume=360_000,
            rvol_1day=1000.0,
            avg_volume=590_000,
            eval_confidence=59.4,
            live_confidence=0.0,
            has_live_data=False,
            missing_data=[],
        )
        rmap = SimpleNamespace(all_levels=lambda: [])

        with patch("display.safe_print") as mock_print:
            _print_stock_summary(entry, rmap)

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
        self.assertIn("UTZ  EVALUATION SUMMARY", plain)
        self.assertIn("(DID NOT PASS)", plain)

    def test_summary_infers_did_not_pass_for_cached_low_rvol_entry(self):
        groups = build_groups(SCORING_GROUPS)
        stock_eval = next(g for g in groups if g.name == "STOCK_EVAL")
        stock_eval.get("float_shares").evaluate(1.9)
        stock_eval.get("price").evaluate(0.39)
        stock_eval.get("rvol_1day").evaluate(3.52)

        entry = SimpleNamespace(
            symbol="TNMG",
            evaluation=_EvalStub(groups),
            news_articles=[],
            news_source="",
            news_is_breaking=False,
            news_is_fresh=False,
            news_count_last_hour=0,
            avg_sentiment=None,
            days_since_ipo=None,
            has_recent_reverse_split=False,
            sector="Media",
            industry="Media",
            country="TW",
            float_shares_m=1.9,
            float_source="AV",
            price=0.39,
            change_pct=19.3,
            volume=253_000,
            rvol_1day=3.52,
            avg_volume=72_000,
            eval_confidence=40.6,
            live_confidence=0.0,
            has_live_data=False,
            missing_data=[],
        )
        rmap = SimpleNamespace(all_levels=lambda: [])

        with patch("display.safe_print") as mock_print:
            _print_stock_summary(entry, rmap)

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
        self.assertIn("TNMG  EVALUATION SUMMARY", plain)
        self.assertIn("(DID NOT PASS)", plain)

    def test_no_outcomes_shows_warning(self):
        with patch("display.safe_print") as mock_print:
            print_live_detail("TEST", _DummyScorer())

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("no setup outcomes recorded yet", output)

    def test_completed_outcomes_displayed_per_tf(self):
        outcome = SimpleNamespace(
            timeframe="1m",
            quality=0.80,
            extension_pct=0.12,
            success=True,
            mono_ts=0.0,
        )
        with patch("display.safe_print") as mock_print:
            print_live_detail("TEST", _DummyScorer(outcomes=[outcome]))

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("1M", output)
        self.assertIn("SUCCESS", output)
        self.assertIn("12.0%", output)

    def test_failed_outcome_shows_fail(self):
        outcome = SimpleNamespace(
            timeframe="5m",
            quality=0.60,
            extension_pct=0.03,
            success=False,
            mono_ts=0.0,
        )
        with patch("display.safe_print") as mock_print:
            print_live_detail("TEST", _DummyScorer(outcomes=[outcome]))

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("5M", output)
        self.assertIn("FAIL", output)

    def test_active_leg_displayed(self):
        leg = SimpleNamespace(
            quality=0.90,
            extension_pct=0.05,
            candle_count=3,
        )
        with patch("display.safe_print") as mock_print:
            print_live_detail("TEST", _DummyScorer(active_legs={"1m": leg}))

        output = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn("ACTIVE", output)
        self.assertIn("bars=3", output)

    def test_weights_display_shows_all_live_ta_vars(self):
        groups = build_groups(SCORING_GROUPS)
        ev = SimpleNamespace(groups=groups)
        ta = next(g for g in groups if g.name == "TECHNICAL_ANALYSIS")
        for v in ta.variables:
            v.value = 0.5
            v.quality = 0.5

        entry = SimpleNamespace(
            symbol="TEST",
            evaluation=ev,
            eval_confidence=50.0,
            live_confidence=50.0,
            has_live_data=True,
        )

        output = format_live_weights(
            entry,
            scorer_snapshot=SimpleNamespace(
                tf_pa_var_qualities={
                    "1m": {
                        "setup_outcome": 0.5,
                    }
                },
                tf_pa_scores={"1m": 0.5},
                buying_pressure=0.0,
                selling_pressure=0.0,
                trade_speed=0.0,
                pressure_acceleration=0.0,
                pressure_acceleration_ready=False,
                baseline_speed=0.0,
                speed_ratio=0.0,
                trade_imbalance=0.0,
                ofi_rolling=0.0,
                book_lean=0.0,
                buy_volume=0,
                sell_volume=0,
            ),
        )

        for text in (
            "setup outcome",
            "Key candles",
            "atrp",
            "candle chop",
        ):
            self.assertIn(text, output)

    def test_weights_display_does_not_mark_advisory_eval_quality_as_failed_gates(self):
        groups = build_groups(SCORING_GROUPS)
        ev = SimpleNamespace(groups=groups)
        stock_eval = next(g for g in groups if g.name == "STOCK_EVAL")
        stock_eval.get("breaking_news").evaluate(0.02)
        stock_eval.get("daily_trend").evaluate(612.0)

        entry = SimpleNamespace(
            symbol="TEST",
            evaluation=ev,
            eval_confidence=50.0,
            live_confidence=0.0,
            has_live_data=False,
        )

        output = format_live_weights(entry)
        plain = re.sub(r"\x1b\[[0-9;]*m", "", output)
        news_line = next(line for line in plain.splitlines() if "breaking_news" in line)
        daily_line = next(line for line in plain.splitlines() if "daily_trend" in line)

        self.assertNotIn("✗", news_line)
        self.assertNotIn("✗", daily_line)
        self.assertIn("quality 0 → 1", news_line)
        self.assertIn("lower cleaner", daily_line)
        self.assertNotIn("worst", daily_line)


def _plain(text: str) -> str:
    """Strip ANSI colour so assertions can look at the rendered columns."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class _CacheStub:
    """Minimal TickerCache stand-in — get() returns (entry, rmap) or None."""

    def __init__(self, entries: dict) -> None:
        self._entries = entries

    def get(self, sym: str):
        entry = self._entries.get(sym)
        return (entry, None) if entry is not None else None


def _entry(price: float, change_pct: float):
    return SimpleNamespace(price=price, change_pct=change_pct)


class _ScorerStub:
    def __init__(self, last_price: float | None) -> None:
        self._last_price = last_price

    def try_snapshot(self):
        if self._last_price is None:
            return None  # lock was held — the toolbar must skip, not block
        return SimpleNamespace(last_price=self._last_price)


class _Row:
    def __init__(self, symbol: str, rank: int) -> None:
        self.symbol = symbol
        self.rank = rank


class TestFormatTopGainers(unittest.TestCase):
    """The `top` display — a read-only view of the latest IBKR scan."""

    def test_no_snapshot_yet_says_waiting(self):
        out = _plain(format_top_gainers([], age_secs=None))
        self.assertIn("waiting for first scan", out)

    def test_empty_scan_is_reported_as_quiet_not_missing(self):
        """Nothing clearing the filters is a valid result, not an outage."""
        out = _plain(format_top_gainers([], age_secs=3.0))
        self.assertIn("no symbols currently clearing", out)
        self.assertNotIn("waiting for first scan", out)

    def test_rows_render_in_rank_order_with_1_based_numbering(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0), _Row("KIDZ", 1), _Row("IONQ", 2)], age_secs=2.0,
        ))
        body = [l for l in out.splitlines() if "." in l and "─" not in l]
        self.assertEqual(len(body), 3)
        self.assertIn("1. ACMR", body[0])
        self.assertIn("2. KIDZ", body[1])
        self.assertIn("3. IONQ", body[2])

    def test_header_carries_count_age_interval_and_pause(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0)], age_secs=4.0, interval=30, paused=True,
        ))
        head = out.splitlines()[0]
        self.assertIn("IBKR Top Gainers", head)
        self.assertIn("1", head)
        self.assertIn("4s ago", head)
        self.assertIn("30s poll", head)
        self.assertIn("paused", head)

    def test_hard_rejected_symbols_are_hidden(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0), _Row("KIDZ", 1), _Row("IONQ", 2)],
            age_secs=2.0, rejected={"KIDZ"},
        ))
        self.assertIn("ACMR", out)
        self.assertIn("IONQ", out)
        self.assertNotIn("KIDZ", out)

    def test_hidden_rejects_are_renumbered_contiguously(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0), _Row("KIDZ", 1), _Row("IONQ", 2)],
            age_secs=2.0, rejected={"KIDZ"},
        ))
        body = [l for l in out.splitlines() if "." in l and "─" not in l]
        self.assertIn("1. ACMR", body[0])
        self.assertIn("2. IONQ", body[1])

    def test_header_discloses_how_many_were_hidden(self):
        """A filtered view must never read as a quiet scan."""
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0), _Row("KIDZ", 1)],
            age_secs=2.0, rejected={"KIDZ"},
        ))
        self.assertIn("1 rejected hidden", out.splitlines()[0])

    def test_no_disclosure_when_nothing_was_hidden(self):
        out = _plain(format_top_gainers([_Row("ACMR", 0)], age_secs=2.0))
        self.assertNotIn("rejected hidden", out)

    def test_all_rejected_is_distinct_from_an_empty_scan(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0), _Row("KIDZ", 1)],
            age_secs=2.0, rejected={"ACMR", "KIDZ"},
        ))
        self.assertIn("all 2 scanned names hard-rejected", out)
        self.assertNotIn("no symbols currently clearing", out)

    def test_limit_applies_after_filtering(self):
        """`top 2` yields 2 live names, not 2 minus the rejects."""
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0), _Row("KIDZ", 1), _Row("IONQ", 2)],
            age_secs=2.0, rejected={"ACMR"}, limit=2,
        ))
        body = [l for l in out.splitlines() if "." in l and "─" not in l]
        self.assertEqual(len(body), 2)
        self.assertIn("1. KIDZ", body[0])
        self.assertIn("2. IONQ", body[1])

    def test_unknown_symbol_shows_dashes_and_no_tag(self):
        out = _plain(format_top_gainers([_Row("ACMR", 0)], age_secs=1.0))
        row = out.splitlines()[-1]
        self.assertIn("ACMR", row)
        self.assertIn("—", row)
        self.assertNotIn("cached", row)
        self.assertNotIn("live", row)

    def test_cached_entry_supplies_price_and_change(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0)], age_secs=1.0,
            cache=_CacheStub({"ACMR": _entry(10.0, 25.0)}),
        ))
        row = out.splitlines()[-1]
        self.assertIn("$10.00", row)
        self.assertIn("+25.0%", row)
        self.assertIn("cached", row)

    def test_live_price_rebases_change_off_the_implied_prev_close(self):
        """A fresh tape price paired with the stale evaluation-time % would
        be wrong; entry.price/change_pct pin the previous close exactly."""
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0)], age_secs=1.0,
            cache=_CacheStub({"ACMR": _entry(10.0, 25.0)}),   # prev close = 8.00
            scorers={"ACMR": _ScorerStub(12.0)},
        ))
        row = out.splitlines()[-1]
        self.assertIn("$12.00", row)
        self.assertIn("+50.0%", row)

    def test_negative_change_still_renders(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0)], age_secs=1.0,
            cache=_CacheStub({"ACMR": _entry(9.0, -4.0)}),
        ))
        self.assertIn("-4.0%", out.splitlines()[-1])

    def test_scorer_without_cache_entry_shows_live_price_and_no_change(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0)], age_secs=1.0, scorers={"ACMR": _ScorerStub(7.5)},
        ))
        row = out.splitlines()[-1]
        self.assertIn("$7.50", row)
        self.assertIn("live", row)

    def test_contended_scorer_snapshot_falls_back_instead_of_blocking(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0)], age_secs=1.0,
            cache=_CacheStub({"ACMR": _entry(10.0, 25.0)}),
            scorers={"ACMR": _ScorerStub(None)},
        ))
        row = out.splitlines()[-1]
        self.assertIn("$10.00", row)
        self.assertIn("+25.0%", row)

    def test_watchlist_tag_wins_over_live_and_cached(self):
        out = _plain(format_top_gainers(
            [_Row("ACMR", 0)], age_secs=1.0,
            cache=_CacheStub({"ACMR": _entry(10.0, 25.0)}),
            scorers={"ACMR": _ScorerStub(12.0)},
            watchlist_symbols={"ACMR"},
        ))
        row = out.splitlines()[-1]
        self.assertIn("watching", row)
        self.assertNotIn("cached", row)

    def test_raising_scorer_does_not_break_the_render(self):
        class _Boom:
            def try_snapshot(self):
                raise RuntimeError("scorer died")

        out = _plain(format_top_gainers(
            [_Row("ACMR", 0)], age_secs=1.0,
            cache=_CacheStub({"ACMR": _entry(10.0, 25.0)}),
            scorers={"ACMR": _Boom()},
        ))
        self.assertIn("$10.00", out.splitlines()[-1])

    def test_minute_scale_age_is_abbreviated(self):
        out = _plain(format_top_gainers([_Row("ACMR", 0)], age_secs=185.0))
        self.assertIn("3m ago", out.splitlines()[0])


if __name__ == "__main__":
    unittest.main()