"""
Simulation trade logger — writes simulated fills to CSV and JSON.

Each session produces:
  - sim_trades_YYYYMMDD.csv  (one row per fill)
  - sim_trades_YYYYMMDD.json (structured, for replay/analysis)

Logs include: timestamp, symbol, side (BUY/SELL), fill price, size,
bid/ask at fill, rationale, P&L (for sells), confidence at entry/exit,
pressure/OFI readings, and setup that triggered entry.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import MarketClock, SIM_LOG_DIR

logger = logging.getLogger(__name__)


def _fill_date_str(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(MarketClock.ET).strftime("%Y%m%d")


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
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except (TypeError, ValueError):
        return default

_TRADE_CSV_FIELDS = [
    "ts", "symbol", "side", "fill_price", "shares", "notional",
    "bid", "ask", "spread",
    "entry_reason", "exit_reason",
    "eval_confidence", "live_confidence",
    "setup_name", "setup_timeframe", "setup_score", "setup_pullback_ts",
    "setup_breakout_ts", "setup_breakout_price",
    "buying_pressure", "selling_pressure", "net_pressure",
    "entry_speed", "entry_buy_ratio", "pressure_acceleration",
    "pressure_acceleration_ready", "trade_imbalance",
    "ofi_rolling",
    "depth_active", "ofi_normalized", "book_depth_avg", "depth_flips",
    "pnl_dollars", "pnl_pct", "hold_secs",
]


def _upgrade_csv_header(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        return

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_TRADE_CSV_FIELDS).writeheader()
        return

    header = rows[0]
    if header == _TRADE_CSV_FIELDS:
        return

    normalized = []
    for values in rows[1:]:
        if len(values) == len(_TRADE_CSV_FIELDS):
            row = dict(zip(_TRADE_CSV_FIELDS, values))
        else:
            row = dict(zip(header, values))
            extras = values[len(header):]
            missing_fields = [field for field in _TRADE_CSV_FIELDS if field not in header]
            row.update(dict(zip(missing_fields, extras)))
        if "confidence" in row:
            row.setdefault("live_confidence", row.get("confidence", ""))
        normalized.append({field: row.get(field, "") for field in _TRADE_CSV_FIELDS})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_TRADE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(normalized)


@dataclass(slots=True)
class SimFill:
    """A single simulated fill event."""
    ts: datetime
    symbol: str
    side: str             # "BUY" or "SELL"
    fill_price: float
    shares: int
    notional: float
    bid: float
    ask: float
    spread: float
    entry_reason: str = ""
    exit_reason: str = ""
    eval_confidence: float = 0.0
    live_confidence: float = 0.0
    setup_name: str = ""
    setup_timeframe: str = ""
    setup_score: float = 0.0
    setup_pullback_ts: Optional[datetime] = None
    setup_breakout_ts: Optional[datetime] = None
    setup_breakout_price: Optional[float] = None
    buying_pressure: int = 0
    selling_pressure: int = 0
    net_pressure: int = 0
    entry_speed: float = 0.0
    entry_buy_ratio: float = 0.0
    pressure_acceleration: float = 0.0
    pressure_acceleration_ready: bool = False
    trade_imbalance: float = 0.0
    ofi_rolling: int = 0
    # F8 regime telemetry: whether the L2 book was live (depth_active) when
    # this fill executed, plus the book-derived OFI values at that instant.
    # depth_flips counts depth_active transitions during the hold (SELL only).
    depth_active: bool = False
    ofi_normalized: float = 0.0
    book_depth_avg: float = 0.0
    depth_flips: int = 0
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    hold_secs: float = 0.0


@dataclass(slots=True)
class ClosedTradeSummary:
    """Aggregate realized P&L for paired simulated trades."""
    total_pnl: float = 0.0
    realized_pct: float = 0.0
    summed_trade_pct: float = 0.0
    winners: int = 0
    losers: int = 0


def summarize_closed_trades(
    closed_trades: list[tuple[Optional[SimFill], SimFill]],
) -> ClosedTradeSummary:
    """Summarize paired BUY/SELL fills into realized P&L stats."""
    total_pnl = 0.0
    total_entry_notional = 0.0
    summed_trade_pct = 0.0
    winners = 0
    losers = 0

    for buy, sell in closed_trades:
        total_pnl += sell.pnl_dollars
        summed_trade_pct += sell.pnl_pct
        if sell.pnl_dollars > 0:
            winners += 1
        else:
            losers += 1

        if buy is not None and buy.notional > 0:
            entry_notional = buy.notional
        else:
            entry_notional = max(sell.notional - sell.pnl_dollars, 0.0)
        total_entry_notional += entry_notional

    realized_pct = total_pnl / total_entry_notional if total_entry_notional > 0 else 0.0
    return ClosedTradeSummary(
        total_pnl=total_pnl,
        realized_pct=realized_pct,
        summed_trade_pct=summed_trade_pct,
        winners=winners,
        losers=losers,
    )


class SimTradeLogger:
    """Thread-safe logger for simulated trade fills."""

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_file = None
        self._json_path: Optional[Path] = None
        self._trades: list[SimFill] = []
        self._session_date: str = ""
        self._log_dir: str = log_dir if log_dir is not None else SIM_LOG_DIR
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        os.makedirs(self._log_dir, exist_ok=True)

    def _open_csv(self, date_str: str) -> None:
        """Open (or reopen) the CSV writer for the given date."""
        if self._session_date == date_str and self._csv_writer is not None:
            return

        if self._csv_file is not None:
            self._csv_file.close()

        self._session_date = date_str
        csv_path = os.path.join(self._log_dir, f"sim_trades_{date_str}.csv")
        self._json_path = Path(self._log_dir) / f"sim_trades_{date_str}.json"

        file_exists = os.path.exists(csv_path)
        if file_exists:
            _upgrade_csv_header(csv_path)
        self._csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=_TRADE_CSV_FIELDS)
        if not file_exists:
            self._csv_writer.writeheader()

    def log_fill(self, fill: SimFill) -> None:
        """Write a fill to both CSV and JSON."""
        with self._lock:
            date_str = _fill_date_str(fill.ts)
            self._open_csv(date_str)

            row = {
                "ts": fill.ts.isoformat(timespec="milliseconds"),
                "symbol": fill.symbol,
                "side": fill.side,
                "fill_price": round(fill.fill_price, 4),
                "shares": fill.shares,
                "notional": round(fill.notional, 2),
                "bid": round(fill.bid, 4),
                "ask": round(fill.ask, 4),
                "spread": round(fill.spread, 4),
                "entry_reason": fill.entry_reason,
                "exit_reason": fill.exit_reason,
                "eval_confidence": round(fill.eval_confidence, 1),
                "live_confidence": round(fill.live_confidence, 1),
                "setup_name": fill.setup_name,
                "setup_timeframe": fill.setup_timeframe,
                "setup_score": round(fill.setup_score, 1),
                "setup_pullback_ts": fill.setup_pullback_ts.isoformat(timespec="milliseconds") if fill.setup_pullback_ts else "",
                "setup_breakout_ts": fill.setup_breakout_ts.isoformat(timespec="milliseconds") if fill.setup_breakout_ts else "",
                "setup_breakout_price": round(fill.setup_breakout_price, 4) if fill.setup_breakout_price else "",
                "buying_pressure": fill.buying_pressure,
                "selling_pressure": fill.selling_pressure,
                "net_pressure": fill.net_pressure,
                "entry_speed": round(fill.entry_speed, 2),
                "entry_buy_ratio": round(fill.entry_buy_ratio, 4),
                "pressure_acceleration": round(fill.pressure_acceleration, 4),
                "pressure_acceleration_ready": fill.pressure_acceleration_ready,
                "trade_imbalance": round(fill.trade_imbalance, 4),
                "ofi_rolling": fill.ofi_rolling,
                "depth_active": fill.depth_active,
                "ofi_normalized": round(fill.ofi_normalized, 3),
                "book_depth_avg": round(fill.book_depth_avg, 1),
                "depth_flips": fill.depth_flips,
                "pnl_dollars": round(fill.pnl_dollars, 2),
                "pnl_pct": round(fill.pnl_pct, 4),
                "hold_secs": round(fill.hold_secs, 1),
            }
            self._csv_writer.writerow(row)
            self._csv_file.flush()

            self._trades.append(fill)
            self._write_json()

    def _write_json(self) -> None:
        """Rewrite the full JSON log (append-friendly alternative: use jsonlines)."""
        if self._json_path is None:
            return
        records = []
        for f in self._trades:
            d = {
                "ts": f.ts.isoformat(timespec="milliseconds"),
                "symbol": f.symbol,
                "side": f.side,
                "fill_price": f.fill_price,
                "shares": f.shares,
                "notional": f.notional,
                "bid": f.bid,
                "ask": f.ask,
                "spread": f.spread,
                "entry_reason": f.entry_reason,
                "exit_reason": f.exit_reason,
                "eval_confidence": f.eval_confidence,
                "live_confidence": f.live_confidence,
                "setup_name": f.setup_name,
                "setup_timeframe": f.setup_timeframe,
                "setup_score": f.setup_score,
                "setup_pullback_ts": f.setup_pullback_ts.isoformat(timespec="milliseconds") if f.setup_pullback_ts else "",
                "setup_breakout_ts": f.setup_breakout_ts.isoformat(timespec="milliseconds") if f.setup_breakout_ts else "",
                "setup_breakout_price": f.setup_breakout_price if f.setup_breakout_price else "",
                "buying_pressure": f.buying_pressure,
                "selling_pressure": f.selling_pressure,
                "net_pressure": f.net_pressure,
                "entry_speed": f.entry_speed,
                "entry_buy_ratio": f.entry_buy_ratio,
                "pressure_acceleration": f.pressure_acceleration,
                "pressure_acceleration_ready": f.pressure_acceleration_ready,
                "trade_imbalance": f.trade_imbalance,
                "ofi_rolling": f.ofi_rolling,
                "depth_active": f.depth_active,
                "ofi_normalized": f.ofi_normalized,
                "book_depth_avg": f.book_depth_avg,
                "depth_flips": f.depth_flips,
                "pnl_dollars": f.pnl_dollars,
                "pnl_pct": f.pnl_pct,
                "hold_secs": f.hold_secs,
            }
            records.append(d)
        self._json_path.write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        """Flush and close file handles."""
        with self._lock:
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def fills(self) -> list[SimFill]:
        with self._lock:
            return list(self._trades)

    def trade_dates(self) -> list[str]:
        """Return sorted YYYYMMDD date strings that have logged sim trades.

        Combines on-disk ``sim_trades_*.csv`` files with any in-memory fills
        from the current session.
        """
        dates: set[str] = set()
        for path in Path(self._log_dir).glob("sim_trades_*.csv"):
            ds = path.stem.replace("sim_trades_", "")
            if len(ds) == 8 and ds.isdigit():
                dates.add(ds)
        with self._lock:
            for fill in self._trades:
                dates.add(_fill_date_str(fill.ts))
        return sorted(dates)

    def day_fills(self, date_str: str) -> list[SimFill]:
        """Return fills for a full market day, including earlier process runs."""
        with self._lock:
            if self._csv_file is not None:
                self._csv_file.flush()

        csv_path = Path(self._log_dir) / f"sim_trades_{date_str}.csv"
        fills: list[SimFill] = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    ts = _parse_dt(row.get("ts"))
                    if ts is None:
                        continue
                    fills.append(SimFill(
                        ts=ts,
                        symbol=str(row.get("symbol", "") or ""),
                        side=str(row.get("side", "") or ""),
                        fill_price=_parse_float(row.get("fill_price")),
                        shares=_parse_int(row.get("shares")),
                        notional=_parse_float(row.get("notional")),
                        bid=_parse_float(row.get("bid")),
                        ask=_parse_float(row.get("ask")),
                        spread=_parse_float(row.get("spread")),
                        entry_reason=str(row.get("entry_reason", "") or ""),
                        exit_reason=str(row.get("exit_reason", "") or ""),
                        eval_confidence=_parse_float(row.get("eval_confidence")),
                        live_confidence=_parse_float(row.get("live_confidence") or row.get("confidence")),
                        setup_name=str(row.get("setup_name", "") or ""),
                        setup_timeframe=str(row.get("setup_timeframe", "") or ""),
                        setup_score=_parse_float(row.get("setup_score")),
                        setup_pullback_ts=_parse_dt(row.get("setup_pullback_ts")),
                        setup_breakout_ts=_parse_dt(row.get("setup_breakout_ts")),
                        setup_breakout_price=_parse_float(row.get("setup_breakout_price")) or None,
                        buying_pressure=_parse_int(row.get("buying_pressure")),
                        selling_pressure=_parse_int(row.get("selling_pressure")),
                        net_pressure=_parse_int(row.get("net_pressure")),
                        entry_speed=_parse_float(row.get("entry_speed")),
                        entry_buy_ratio=_parse_float(row.get("entry_buy_ratio")),
                        pressure_acceleration=_parse_float(row.get("pressure_acceleration")),
                        pressure_acceleration_ready=_parse_bool(row.get("pressure_acceleration_ready")),
                        trade_imbalance=_parse_float(row.get("trade_imbalance")),
                        ofi_rolling=_parse_int(row.get("ofi_rolling")),
                        depth_active=_parse_bool(row.get("depth_active")),
                        ofi_normalized=_parse_float(row.get("ofi_normalized")),
                        book_depth_avg=_parse_float(row.get("book_depth_avg")),
                        depth_flips=_parse_int(row.get("depth_flips")),
                        pnl_dollars=_parse_float(row.get("pnl_dollars")),
                        pnl_pct=_parse_float(row.get("pnl_pct")),
                        hold_secs=_parse_float(row.get("hold_secs")),
                    ))

        fills.extend(f for f in self.fills if _fill_date_str(f.ts) == date_str)
        seen = set()
        unique: list[SimFill] = []
        for fill in sorted(fills, key=lambda f: f.ts):
            key = (
                fill.ts.isoformat(timespec="milliseconds"), fill.symbol,
                fill.side, fill.fill_price, fill.shares,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(fill)
        return unique

    def clear_session(self, date_str: str) -> int:
        """Clear in-memory fills for *date_str* and remove day files on disk.

        Returns the number of fills removed.
        """
        with self._lock:
            before = len(self._trades)
            self._trades = [f for f in self._trades if _fill_date_str(f.ts) != date_str]
            removed = before - len(self._trades)

            # Close current CSV handle if it belongs to this date
            if self._session_date == date_str:
                if self._csv_file is not None:
                    self._csv_file.close()
                    self._csv_file = None
                    self._csv_writer = None
                self._session_date = ""

            csv_path = Path(self._log_dir) / f"sim_trades_{date_str}.csv"
            json_path = Path(self._log_dir) / f"sim_trades_{date_str}.json"
            if csv_path.exists():
                csv_path.unlink()
            if json_path.exists():
                json_path.unlink()

            return removed

    @property
    def session_pnl(self) -> float:
        """Total P&L for the current session."""
        return sum(f.pnl_dollars for f in self._trades if f.side == "SELL")

    def summary(self) -> str:
        """One-line session summary."""
        sells = [f for f in self._trades if f.side == "SELL"]
        if not sells:
            buys = [f for f in self._trades if f.side == "BUY"]
            return f"Sim: {len(buys)} open position(s), 0 closed"
        wins = sum(1 for f in sells if f.pnl_dollars > 0)
        total_pnl = sum(f.pnl_dollars for f in sells)
        return (
            f"Sim: {len(sells)} trades, "
            f"{wins}W/{len(sells)-wins}L, "
            f"P&L=${total_pnl:+.2f}"
        )
