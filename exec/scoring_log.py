"""
exec/scoring_log.py — CSV logger for scoring variable telemetry.

Two output files per session day:

``logs/scoring_YYYYMMDD.csv`` — one row per snapshot, one column per variable.
    Columns: ts, symbol, {var}_raw, {var}_q for each variable,
    plus {group}_score for each group, eval_confidence, live_confidence.

``logs/tape_YYYYMMDD.csv`` — one row per snapshot, order flow + density metrics.
    Columns: ts, symbol, price, avg_print_size, tape_pps, buy/sell_pps,
    density_imbalance, buying_pressure, selling_pressure, buy/sell_vol, book_lean, ofi_rolling,
    pa_10s/1m/2m/5m, pa_combined, eval_confidence, live_confidence.

Enable/disable via ``config.SCORING_LOG_ENABLED``.
"""

from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exec.order_flow import OrderFlowTracker
    from strategy.evaluator import WatchlistEntry

from config import SCORING_GROUPS, SCORING_LOG_DIR
from exec.signals import tape_entry_gates
from strategy.scoring import build_groups

logger = logging.getLogger(__name__)

_LOG_DIR = Path(SCORING_LOG_DIR)

# Build column names from config so they stay in sync automatically
_TEMPLATE_GROUPS = build_groups(SCORING_GROUPS)

_SCORING_FIELDS: list[str] = ["ts", "symbol"]
for _g in _TEMPLATE_GROUPS:
    for _v in _g.variables:
        _SCORING_FIELDS.append(f"{_v.name}_raw")
        _SCORING_FIELDS.append(f"{_v.name}_q")
    _SCORING_FIELDS.append(f"{_g.name}_score")
_SCORING_FIELDS += ["eval_confidence", "live_confidence"]

_TAPE_FIELDS = [
    "ts", "symbol", "price",
    "avg_print_size", "tape_pps", "buy_pps", "sell_pps",
    "entry_speed", "entry_buy_ratio", "pressure_acceleration",
    "pressure_acceleration_ready", "trade_imbalance", "entry_tape_gate",
    "density_imbalance", "buying_pressure", "selling_pressure",
    "buy_vol", "sell_vol", "book_lean", "ofi_rolling",
    "pa_10s", "pa_1m", "pa_2m", "pa_5m", "pa_1d", "pa_combined",
    "eval_confidence", "live_confidence",
    # EMA accel estimator + L2 depth telemetry (appended for CSV compat)
    "speed_ema", "pressure_acceleration_ema", "pressure_acceleration_ema_z",
    "pressure_acceleration_ema_ready", "ofi_normalized", "book_depth_avg",
    "buy_ratio_ema",
    # F8: explicit depth regime flag — ofi_normalized == 0.0 alone is
    # ambiguous between "no book" and "flat book".
    "depth_active",
    # EMA-kernel OFI gate read (ofi_normalized stays the boxcar telemetry).
    "ofi_norm_ema",
    # Trailing price-progress gate read (blank while warming up).
    "price_progress",
]


def _upgrade_tape_csv_header(path: Path) -> None:
    """Rewrite an existing tape CSV to the current schema (missing cols blank).

    Same pattern as exec.sim_logger._upgrade_csv_header — keeps same-day
    restarts appending aligned rows after new columns are introduced.
    """
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_TAPE_FIELDS).writeheader()
        return
    header = rows[0]
    if header == _TAPE_FIELDS:
        return
    normalized = []
    for values in rows[1:]:
        row = dict(zip(header, values))
        normalized.append({field: row.get(field, "") for field in _TAPE_FIELDS})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TAPE_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)


class ScoringLogger:
    """Thread-safe CSV writer. One per session, shared across all scorers."""

    _instance: ScoringLogger | None = None
    _lock_cls = threading.Lock()

    def __init__(self) -> None:
        today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        self._lock = threading.Lock()
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Scoring variables CSV
        self._sc_path = _LOG_DIR / f"scoring_{today}.csv"
        self._sc_file, self._sc_writer = self._open_csv(self._sc_path, _SCORING_FIELDS)

        # Tape / order flow CSV
        self._tp_path = _LOG_DIR / f"tape_{today}.csv"
        _upgrade_tape_csv_header(self._tp_path)
        self._tp_file, self._tp_writer = self._open_csv(self._tp_path, _TAPE_FIELDS)

        logger.debug(f"Scoring log → {self._sc_path}")
        logger.debug(f"Tape log    → {self._tp_path}")

    @staticmethod
    def _open_csv(path: Path, fields: list[str]):
        write_header = not path.exists()
        f = open(path, "a", newline="", buffering=1)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        return f, w

    @classmethod
    def get(cls) -> ScoringLogger:
        """Return the singleton instance (lazy-init)."""
        if cls._instance is None:
            with cls._lock_cls:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def snapshot(self, entry: WatchlistEntry) -> None:
        """Write one flat row: ts, symbol, vars, group scores, eval/live confidence."""
        ev = entry.evaluation
        if ev is None:
            return

        now = datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")
        row: dict = {"ts": now, "symbol": entry.symbol}

        for g in ev.groups:
            for v in g.variables:
                row[f"{v.name}_raw"] = round(v.value, 6) if v.value is not None else ""
                row[f"{v.name}_q"] = round(v.quality, 4)
            row[f"{g.name}_score"] = round(g.score, 4)

        row["eval_confidence"] = round(entry.eval_confidence, 2)
        row["live_confidence"] = round(entry.live_confidence, 2)

        with self._lock:
            if self._sc_writer:
                self._sc_writer.writerow(row)

    def tape_snapshot(
        self,
        entry: WatchlistEntry,
        ofi: OrderFlowTracker,
        price: float,
        tf_pa_scores: dict[str, float] | None = None,
    ) -> None:
        """Write one row of T&S density / order flow telemetry."""
        now = datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")
        pa = tf_pa_scores or {}
        entry_speed = ofi.trade_speed
        entry_buy_ratio = ofi.buy_ratio
        pressure_acceleration = ofi.pressure_acceleration
        pressure_acceleration_ready = ofi.pressure_acceleration_ready
        # Gate mirrors the live entry gate exactly (single source:
        # exec.signals.tape_entry_gates — F10). Note: the activity term now
        # reads the 30s activity_count like the live gate; before F10 this
        # column used the 10s pressure-window count, which had drifted from
        # the live gate when F3 decoupled the windows. entry_speed /
        # entry_buy_ratio stay the boxcar telemetry/backtest-parity columns.
        entry_tape_gate = tape_entry_gates(ofi).all_pass
        row = {
            "ts": now,
            "symbol": entry.symbol,
            "price": round(price, 4),
            "avg_print_size": round(ofi.avg_print_size, 1),
            "tape_pps": ofi.tape_prints_per_sec,
            "buy_pps": ofi.buy_prints_per_sec,
            "sell_pps": ofi.sell_prints_per_sec,
            "entry_speed": round(entry_speed, 2),
            "entry_buy_ratio": round(entry_buy_ratio, 4),
            "pressure_acceleration": round(pressure_acceleration, 4),
            "pressure_acceleration_ready": pressure_acceleration_ready,
            "trade_imbalance": ofi.trade_imbalance,
            "entry_tape_gate": entry_tape_gate,
            "density_imbalance": ofi.density_imbalance,
            "buying_pressure": ofi.buying_pressure,
            "selling_pressure": ofi.selling_pressure,
            "buy_vol": ofi.buy_volume,
            "sell_vol": ofi.sell_volume,
            "book_lean": ofi.book_lean,
            "ofi_rolling": ofi.ofi_rolling,
            "pa_10s": round(pa["10s"], 4) if "10s" in pa else "",
            "pa_1m":  round(pa["1m"], 4)  if "1m"  in pa else "",
            "pa_2m":  round(pa["2m"], 4)  if "2m"  in pa else "",
            "pa_5m":  round(pa["5m"], 4)  if "5m"  in pa else "",
            "pa_1d":  round(pa["1d"], 4)  if "1d"  in pa else "",
            "pa_combined": round(
                entry.evaluation.technical_analysis.score, 4
            ) if entry.evaluation else "",
            "eval_confidence": round(entry.eval_confidence, 2),
            "live_confidence": round(entry.live_confidence, 2),
            "speed_ema": round(ofi.tape_speed_ema, 2),
            "buy_ratio_ema": round(ofi.buy_ratio_ema, 4),
            "pressure_acceleration_ema": round(ofi.pressure_acceleration_ema, 4),
            "pressure_acceleration_ema_z": round(ofi.pressure_acceleration_ema_z, 2),
            "pressure_acceleration_ema_ready": ofi.pressure_acceleration_ema_ready,
            "ofi_normalized": ofi.ofi_normalized,
            "book_depth_avg": round(ofi.book_depth_avg, 1),
            "depth_active": ofi.depth_active,
            "ofi_norm_ema": ofi.ofi_normalized_ema,
            "price_progress": "" if ofi.price_progress is None else ofi.price_progress,
        }

        with self._lock:
            if self._tp_writer:
                self._tp_writer.writerow(row)

    def close(self) -> None:
        with self._lock:
            for f in (self._sc_file, self._tp_file):
                if f:
                    f.close()
            self._sc_file = self._sc_writer = None
            self._tp_file = self._tp_writer = None
