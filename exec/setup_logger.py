"""
Setup signal logger — writes detected trade setups separately from fills.

Each session produces:
  - setup_signals_YYYYMMDD.csv
  - setup_signals_YYYYMMDD.json

The records are upserted by setup_id so a setup can be logged when found,
then enriched later with approval, entry, and peak T&S readings.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import MarketClock, SIM_LOG_DIR

logger = logging.getLogger(__name__)


def _signal_date_str(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(MarketClock.ET).strftime("%Y%m%d")


def _local_iso(ts: Optional[datetime]) -> str:
    if ts is None:
        return ""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(MarketClock.local_tz()).isoformat(timespec="milliseconds")


def _parse_dt(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_float(value: object, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _parse_int(value: object, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _normalize_reason_text(text: str, eval_confidence: float, live_confidence: float) -> str:
    if not text:
        return ""
    text = text.replace("setup cleared before entry approval", "setup cleared before execution entry")
    parts: list[str] = []
    for raw_part in text.split(","):
        part = raw_part.strip()
        if not part or (part.startswith("layer:") and part.endswith(" < 0.4")):
            continue
        if part.startswith("confidence "):
            value_text, separator, threshold = part.removeprefix("confidence ").partition(" < ")
            if separator:
                value = _parse_float(value_text)
                if abs(value - live_confidence) < 0.05:
                    part = f"live_confidence {value:.1f} < {threshold}"
                elif abs(value - eval_confidence) < 0.05:
                    part = f"eval_confidence {value:.1f} < {threshold}"
        parts.append(part)
    return ", ".join(parts)


_SETUP_CSV_FIELDS = [
    "setup_id", "ts", "last_updated_ts", "symbol",
    "setup_name", "setup_timeframe", "setup_score", "eval_confidence", "live_confidence",
    "signal_price", "setup_level", "approved_min_price", "approved_max_price",
    "signal_approved", "approval_ts", "approval_price", "entered",
    "entry_ts", "entry_price", "entry_shares",
    "entry_allowed", "status", "status_reason", "block_reasons", "setup_failure_detail",
    "arm_state",
    "best_ts", "best_price", "best_score", "best_distance_to_range",
    "highest_trade_speed", "highest_buy_ratio", "highest_pressure_acceleration",
    "highest_trade_imbalance", "highest_ofi_rolling", "highest_net_pressure",
    "highest_buying_pressure", "highest_selling_pressure",
]


@dataclass(slots=True)
class SetupSignal:
    """A setup found by the live scorer, separate from simulated fills."""
    setup_id: str
    ts: datetime
    last_updated_ts: datetime
    symbol: str
    setup_name: str
    setup_timeframe: str
    setup_score: float
    eval_confidence: float
    live_confidence: float
    signal_price: float
    setup_level: float
    approved_min_price: float
    approved_max_price: float
    signal_approved: bool = False
    approval_ts: Optional[datetime] = None
    approval_price: float = 0.0
    entered: bool = False
    entry_ts: Optional[datetime] = None
    entry_price: float = 0.0
    entry_shares: int = 0
    entry_allowed: bool = False
    status: str = "FOUND"
    status_reason: str = ""
    block_reasons: str = ""
    setup_failure_detail: str = ""
    # Armed-band telemetry: last-evaluated arm/trigger summary, e.g.
    # "cross armed trig=break" or "below stale:ratio,ofi". Empty in the
    # legacy regime. Mutates in place like block_reasons (last state wins).
    arm_state: str = ""
    best_ts: Optional[datetime] = None
    best_price: float = 0.0
    best_score: float = 0.0
    best_distance_to_range: float = 0.0
    highest_trade_speed: float = 0.0
    highest_buy_ratio: float = 0.0
    highest_pressure_acceleration: float = 0.0
    highest_trade_imbalance: float = 0.0
    highest_ofi_rolling: int = 0
    highest_net_pressure: int = 0
    highest_buying_pressure: int = 0
    highest_selling_pressure: int = 0


class SetupSignalLogger:
    """CSV/JSON logger for setup signals, keyed by setup_id."""

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._signals: dict[str, SetupSignal] = {}
        self._log_dir: str = log_dir if log_dir is not None else SIM_LOG_DIR
        Path(self._log_dir).mkdir(parents=True, exist_ok=True)

    def upsert_signal(self, signal: SetupSignal) -> None:
        """Insert or replace one setup signal record."""
        with self._lock:
            self._signals[signal.setup_id] = signal
            self._write_day_files(_signal_date_str(signal.ts))

    def mark_entered(
        self,
        symbol: str,
        ts: datetime,
        entry_price: float,
        shares: int,
    ) -> None:
        """Mark the newest non-entered setup for a symbol as entered."""
        with self._lock:
            candidates = [
                signal for signal in self._signals.values()
                if signal.symbol == symbol and not signal.entered
            ]
            if not candidates:
                return
            signal = max(candidates, key=lambda s: s.ts)
            signal.entered = True
            signal.signal_approved = True
            signal.entry_ts = ts
            signal.entry_price = entry_price
            signal.entry_shares = shares
            signal.last_updated_ts = ts
            signal.status = "ENTERED"
            if signal.approval_ts is None:
                signal.approval_ts = ts
                signal.approval_price = entry_price
            self._write_day_files(_signal_date_str(signal.ts))

    def day_signals(self, date_str: str) -> list[SetupSignal]:
        """Return setup signals for a full market day, including prior runs."""
        with self._lock:
            signals = self._load_csv(date_str)
            signals.extend(
                signal for signal in self._signals.values()
                if _signal_date_str(signal.ts) == date_str
            )
        return self._unique_sorted(signals)

    def clear_session(self, date_str: str) -> int:
        """Clear in-memory signals for *date_str* and remove day files on disk.

        Returns the number of signals removed.
        """
        with self._lock:
            before = len(self._signals)
            self._signals = {
                sid: sig for sid, sig in self._signals.items()
                if _signal_date_str(sig.ts) != date_str
            }
            removed = before - len(self._signals)

            csv_path = Path(self._log_dir) / f"setup_signals_{date_str}.csv"
            json_path = Path(self._log_dir) / f"setup_signals_{date_str}.json"
            if csv_path.exists():
                csv_path.unlink()
            if json_path.exists():
                json_path.unlink()

            return removed

    def close(self) -> None:
        """Compatibility no-op; setup logs are opened per write."""

    def _write_day_files(self, date_str: str) -> None:
        signals = self._unique_sorted(
            self._load_csv(date_str) + [
                signal for signal in self._signals.values()
                if _signal_date_str(signal.ts) == date_str
            ]
        )
        csv_path = Path(self._log_dir) / f"setup_signals_{date_str}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = csv_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            with tmp_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_SETUP_CSV_FIELDS)
                writer.writeheader()
                for signal in signals:
                    writer.writerow(self._row(signal))
            tmp_path.replace(csv_path)
        except OSError:
            # Clean up stale tmp on failure; non-critical — next upsert retries.
            tmp_path.unlink(missing_ok=True)

        json_path = Path(self._log_dir) / f"setup_signals_{date_str}.json"
        json_path.write_text(
            json.dumps([self._json_record(signal) for signal in signals], indent=2),
            encoding="utf-8",
        )

    def _load_csv(self, date_str: str) -> list[SetupSignal]:
        csv_path = Path(self._log_dir) / f"setup_signals_{date_str}.csv"
        if not csv_path.exists():
            return []
        signals: list[SetupSignal] = []
        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                signal = self._from_row(row)
                if signal is not None:
                    signals.append(signal)
        return signals

    @staticmethod
    def _unique_sorted(signals: list[SetupSignal]) -> list[SetupSignal]:
        by_id: dict[str, SetupSignal] = {}
        for signal in signals:
            current = by_id.get(signal.setup_id)
            if current is None or signal.last_updated_ts >= current.last_updated_ts:
                by_id[signal.setup_id] = signal
        return sorted(by_id.values(), key=lambda s: (s.ts, s.symbol, s.setup_id))

    @staticmethod
    def _row(signal: SetupSignal) -> dict[str, object]:
        return {
            "setup_id": signal.setup_id,
            "ts": _local_iso(signal.ts),
            "last_updated_ts": _local_iso(signal.last_updated_ts),
            "symbol": signal.symbol,
            "setup_name": signal.setup_name,
            "setup_timeframe": signal.setup_timeframe,
            "setup_score": round(signal.setup_score, 1),
            "eval_confidence": round(signal.eval_confidence, 1),
            "live_confidence": round(signal.live_confidence, 1),
            "signal_price": round(signal.signal_price, 4),
            "setup_level": round(signal.setup_level, 4),
            "approved_min_price": round(signal.approved_min_price, 4),
            "approved_max_price": round(signal.approved_max_price, 4),
            "signal_approved": signal.signal_approved,
            "approval_ts": _local_iso(signal.approval_ts),
            "approval_price": round(signal.approval_price, 4),
            "entered": signal.entered,
            "entry_ts": _local_iso(signal.entry_ts),
            "entry_price": round(signal.entry_price, 4),
            "entry_shares": signal.entry_shares,
            "entry_allowed": signal.entry_allowed,
            "status": signal.status,
            "status_reason": signal.status_reason,
            "block_reasons": signal.block_reasons,
            "setup_failure_detail": signal.setup_failure_detail,
            "arm_state": signal.arm_state,
            "best_ts": _local_iso(signal.best_ts),
            "best_price": round(signal.best_price, 4),
            "best_score": round(signal.best_score, 4),
            "best_distance_to_range": round(signal.best_distance_to_range, 4),
            "highest_trade_speed": round(signal.highest_trade_speed, 2),
            "highest_buy_ratio": round(signal.highest_buy_ratio, 4),
            "highest_pressure_acceleration": round(signal.highest_pressure_acceleration, 4),
            "highest_trade_imbalance": round(signal.highest_trade_imbalance, 4),
            "highest_ofi_rolling": signal.highest_ofi_rolling,
            "highest_net_pressure": signal.highest_net_pressure,
            "highest_buying_pressure": signal.highest_buying_pressure,
            "highest_selling_pressure": signal.highest_selling_pressure,
        }

    @staticmethod
    def _json_record(signal: SetupSignal) -> dict[str, object]:
        row = SetupSignalLogger._row(signal)
        return row

    @staticmethod
    def _from_row(row: dict[str, str]) -> Optional[SetupSignal]:
        ts = _parse_dt(row.get("ts"))
        if ts is None:
            return None
        last_updated = _parse_dt(row.get("last_updated_ts")) or ts
        setup_name = str(row.get("setup_name", "") or "")
        eval_confidence = _parse_float(row.get("eval_confidence"))
        live_confidence = _parse_float(row.get("live_confidence") or row.get("confidence"))
        signal_price = _parse_float(row.get("signal_price"))
        signal_approved = _parse_bool(row.get("signal_approved"))
        if not signal_approved and setup_name == "pullback_new_high":
            signal_approved = True
        approval_ts = _parse_dt(row.get("approval_ts"))
        approval_price = _parse_float(row.get("approval_price"))
        if signal_approved:
            approval_ts = approval_ts or ts
            approval_price = approval_price or signal_price
        status = str(row.get("status", "") or "")
        if signal_approved and status in {"", "FOUND", "NOT_APPROVED"}:
            status = "APPROVED"
        status_reason = _normalize_reason_text(
            str(row.get("status_reason", "") or ""),
            eval_confidence,
            live_confidence,
        )
        block_reasons = _normalize_reason_text(
            str(row.get("block_reasons", "") or ""),
            eval_confidence,
            live_confidence,
        )
        setup_failure_detail = _normalize_reason_text(
            str(row.get("setup_failure_detail", "") or ""),
            eval_confidence,
            live_confidence,
        )
        return SetupSignal(
            setup_id=str(row.get("setup_id", "") or ""),
            ts=ts,
            last_updated_ts=last_updated,
            symbol=str(row.get("symbol", "") or ""),
            setup_name=setup_name,
            setup_timeframe=str(row.get("setup_timeframe", "") or ""),
            setup_score=_parse_float(row.get("setup_score")),
            eval_confidence=eval_confidence,
            live_confidence=live_confidence,
            signal_price=signal_price,
            setup_level=_parse_float(row.get("setup_level")),
            approved_min_price=_parse_float(row.get("approved_min_price")),
            approved_max_price=_parse_float(row.get("approved_max_price")),
            signal_approved=signal_approved,
            approval_ts=approval_ts,
            approval_price=approval_price,
            entered=_parse_bool(row.get("entered")),
            entry_ts=_parse_dt(row.get("entry_ts")),
            entry_price=_parse_float(row.get("entry_price")),
            entry_shares=_parse_int(row.get("entry_shares")),
            entry_allowed=_parse_bool(row.get("entry_allowed")),
            status=status,
            status_reason=status_reason,
            block_reasons=block_reasons,
            setup_failure_detail=setup_failure_detail,
            arm_state=str(row.get("arm_state", "") or ""),
            best_ts=_parse_dt(row.get("best_ts")),
            best_price=_parse_float(row.get("best_price")),
            best_score=_parse_float(row.get("best_score")),
            best_distance_to_range=_parse_float(row.get("best_distance_to_range")),
            highest_trade_speed=_parse_float(row.get("highest_trade_speed")),
            highest_buy_ratio=_parse_float(row.get("highest_buy_ratio")),
            highest_pressure_acceleration=_parse_float(row.get("highest_pressure_acceleration")),
            highest_trade_imbalance=_parse_float(row.get("highest_trade_imbalance")),
            highest_ofi_rolling=_parse_int(row.get("highest_ofi_rolling")),
            highest_net_pressure=_parse_int(row.get("highest_net_pressure")),
            highest_buying_pressure=_parse_int(row.get("highest_buying_pressure")),
            highest_selling_pressure=_parse_int(row.get("highest_selling_pressure")),
        )


_default_logger: Optional[SetupSignalLogger] = None
_default_logger_lock = threading.Lock()


def get_setup_signal_logger() -> SetupSignalLogger:
    """Return the shared setup logger for the current process."""
    global _default_logger
    with _default_logger_lock:
        if _default_logger is None:
            _default_logger = SetupSignalLogger()
        return _default_logger


def configure_setup_signal_logger(log_dir: Optional[str]) -> SetupSignalLogger:
    """Repoint the shared setup logger at *log_dir* (None = default).

    Used by the backtest runner so replay setup signals are written to a
    separate directory from live trading.
    """
    global _default_logger
    with _default_logger_lock:
        _default_logger = SetupSignalLogger(log_dir=log_dir)
        return _default_logger