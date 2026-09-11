"""Tests for setup signal logging separate from simulated fills."""

import csv
import json
from datetime import datetime, timedelta, timezone

from config import MarketClock
from exec.setup_logger import SetupSignal, SetupSignalLogger


def _make_signal(
    *,
    setup_id: str = "TEST:pullback_new_high:1min:1.2000:2026",
    symbol: str = "TEST",
    ts: datetime | None = None,
    signal_approved: bool = False,
    status: str = "FOUND",
) -> SetupSignal:
    now = ts or datetime(2026, 6, 4, 14, 30, tzinfo=timezone.utc)
    return SetupSignal(
        setup_id=setup_id,
        ts=now,
        last_updated_ts=now,
        symbol=symbol,
        setup_name="pullback_new_high",
        setup_timeframe="1min",
        setup_score=72.5,
        eval_confidence=66.0,
        live_confidence=64.0,
        signal_price=1.23,
        setup_level=1.20,
        approved_min_price=1.20,
        approved_max_price=1.248,
        signal_approved=signal_approved,
        status=status,
        best_ts=now,
        best_price=1.23,
        highest_trade_speed=52.0,
        highest_buy_ratio=0.68,
        highest_pressure_acceleration=0.12,
        highest_trade_imbalance=0.36,
        highest_ofi_rolling=110,
        highest_net_pressure=85,
        highest_buying_pressure=140,
        highest_selling_pressure=55,
    )


class TestSetupSignalLogger:
    def test_upsert_round_trips_setup_signal(self, tmp_path):
        logger = SetupSignalLogger(log_dir=str(tmp_path))
        signal = _make_signal(signal_approved=True, status="APPROVED")
        signal.approval_ts = signal.ts + timedelta(seconds=1)
        signal.setup_failure_detail = "tape_speed=3.0/s < 10.0/s; spread=4.00% > 2.00%"

        logger.upsert_signal(signal)

        rows = logger.day_signals("20260604")
        assert len(rows) == 1
        assert rows[0].symbol == "TEST"
        assert rows[0].setup_timeframe == "1min"
        assert rows[0].signal_approved is True
        assert rows[0].highest_trade_speed == 52.0
        assert rows[0].approved_min_price == 1.20
        assert rows[0].approved_max_price == 1.248
        assert rows[0].setup_failure_detail == signal.setup_failure_detail

        expected_ts = signal.ts.astimezone(MarketClock.local_tz()).isoformat(timespec="milliseconds")
        with (tmp_path / "setup_signals_20260604.csv").open(newline="", encoding="utf-8") as f:
            csv_row = next(csv.DictReader(f))
        json_row = json.loads((tmp_path / "setup_signals_20260604.json").read_text(encoding="utf-8"))[0]
        assert csv_row["ts"] == expected_ts
        assert csv_row["best_ts"] == expected_ts
        assert csv_row["setup_failure_detail"] == signal.setup_failure_detail
        assert json_row["ts"] == expected_ts
        assert json_row["setup_failure_detail"] == signal.setup_failure_detail

    def test_upsert_keeps_latest_update_for_same_setup(self, tmp_path):
        logger = SetupSignalLogger(log_dir=str(tmp_path))
        base = _make_signal(status="FOUND")
        later = _make_signal(signal_approved=True, status="APPROVED")
        later.last_updated_ts = base.last_updated_ts + timedelta(seconds=5)
        later.approval_ts = later.last_updated_ts
        later.approval_price = 1.24

        logger.upsert_signal(base)
        logger.upsert_signal(later)

        rows = logger.day_signals("20260604")
        assert len(rows) == 1
        assert rows[0].status == "APPROVED"
        assert rows[0].approval_price == 1.24

    def test_mark_entered_updates_newest_symbol_setup(self, tmp_path):
        logger = SetupSignalLogger(log_dir=str(tmp_path))
        older = _make_signal(setup_id="OLD", symbol="TEST")
        newer = _make_signal(
            setup_id="NEW",
            symbol="TEST",
            ts=older.ts + timedelta(seconds=10),
        )
        logger.upsert_signal(older)
        logger.upsert_signal(newer)

        entry_ts = newer.ts + timedelta(seconds=1)
        logger.mark_entered("TEST", entry_ts, entry_price=1.25, shares=800)

        rows = {row.setup_id: row for row in logger.day_signals("20260604")}
        assert rows["OLD"].entered is False
        assert rows["NEW"].entered is True
        assert rows["NEW"].signal_approved is True
        assert rows["NEW"].entry_price == 1.25
        assert rows["NEW"].entry_shares == 800

    def test_day_signals_normalizes_legacy_layer_reason(self, tmp_path):
        logger = SetupSignalLogger(log_dir=str(tmp_path))
        legacy = _make_signal(status="NOT_APPROVED")
        legacy.eval_confidence = 0.0
        legacy.live_confidence = 20.1
        reason = (
            "confidence 20.1 < 60.0, layer:TECHNICAL_ANALYSIS 0.25 < 0.4, "
            "attention:dist_from_hod_atr_max, execution:tape_speed"
        )
        legacy.status_reason = reason
        legacy.block_reasons = reason
        logger.upsert_signal(legacy)

        rows = SetupSignalLogger(log_dir=str(tmp_path)).day_signals("20260604")

        assert len(rows) == 1
        assert rows[0].signal_approved is True
        assert rows[0].status == "APPROVED"
        assert "layer:TECHNICAL_ANALYSIS" not in rows[0].status_reason
        assert "live_confidence 20.1 < 60.0" in rows[0].status_reason
        assert "attention:dist_from_hod_atr_max" in rows[0].status_reason
