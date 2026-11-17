from __future__ import annotations

import argparse
import csv
import copy
import inspect
import logging
import os
import pickle
import re
import select
import sys
import textwrap
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from cache import TickerCache, _refresh_scoring_thresholds, _reset_live_state
from chart_cache import (
	SOURCE_POLLED,
	append_timesales,
	load_book_cache,
	load_candle_cache,
	load_near_hod_window_cache,
	load_timesales_cache,
	prepare_candle_cache_frame,
	write_candle_cache,
	write_near_hod_window_cache,
)
from config import (
	ATTENTION_DIST_FROM_HOD_ATR_MAX,
	ATTENTION_RVOL_1DAY_MIN,
	ATTENTION_VOLUME_MIN,
	ATTENTION_VWAP_SLOPE_MIN,
	CHANGE_PCT_MIN,
	BACKTEST_PULLBACK_WINDOW_TAIL_BARS,
	BACKTEST_REPLAY_WINDOW_BRIDGE_SECS,
	DAILY_HIGH_FADE_PCT,
	ENTRY_CONFIDENCE_MIN,
	EXEC_DIST_TO_RESISTANCE_MULT,
	EXEC_HIDDEN_SELLER_MAX,
	EXEC_SPREAD_PCT_MAX,
	IBKR_ENABLED,
	NEWS_BREAKING_MIN,
	NEWS_NOISE_THRESHOLD,
	PRESSURE_WINDOW_SECS,
	PRICE_MAX,
	PRICE_MIN,
	SCORING_GROUPS,
	SIM_BACKTEST_LOG_DIR,
	SIM_FIRST_MACD_CYCLE_ONLY,
	SIM_MAX_SETUP_EXTENSION_PCT,
	SIM_OFI_NORM_MIN,
	SIM_PRESSURE_ACCEL_Z,
	SIM_PRESSURE_ACTIVITY_MIN,
	SIM_PRESSURE_BUY_RATIO_MIN,
	SIM_PRESSURE_PROGRESS_MIN,
	SIM_PRESSURE_SPEED_MIN,
	SIM_SETUP_SCORE_MIN,
	TAPE_BASELINE_MIN_PRINTS,
	TAPE_BASELINE_WINDOW_SECS,
	TS_MIN_NOTIONAL,
	MarketClock,
	_g, _r, _dim,
)
from exec.sim_logger import SimFill, summarize_closed_trades
from exec.engine import SimulationEngine
from exec.setup_logger import get_setup_signal_logger, configure_setup_signal_logger
from market_data.provider import get_historical_client
from market_data.ibkr_stream import OrderBookLevel, OrderBookSnapshot
from market_data.tradier_rest import OHLCVCandle
from market_data.tradier_stream import LiveQuote, LiveTimesale
from strategy.confidence_scorer import DynamicScorer, _atr_from_candles
from strategy.evaluator import CompositeEvaluation
from strategy.scoring import build_groups
from technical_analysis.patterns import (
	detect_move_start,
	detect_pullback_new_high,
	detect_pullback_pending,
	latest_extended_session_start,
)

logger = logging.getLogger(__name__)

_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[1] / "logs" / "ticker_cache.pkl"
_SNAPSHOT_DIR = Path(__file__).resolve().parents[1] / "logs" / "cache_snapshots"
_DEFAULT_EVAL_STATS = Path(__file__).resolve().parents[1] / "logs" / "eval_stats.log"
_CACHE_TIMESALE_HOLE_SECONDS = 60

_SKIP_CHECK_INTERVAL = 200  # check stdin every N ticks


@contextmanager
def _no_live_stream(*, active: bool):
	"""Neutralize the scorer's shared-stream handle for the duration of a replay.

	Blocking provider *fetches* is not enough to make a backtest offline:
	``DynamicScorer`` reaches for ``get_shared_stream()`` on the slot-priority
	push (whenever a setup arms) and again in ``stop()``'s unsubscribe, and that
	getter lazily constructs a REAL stream and dials the Gateway. Under
	``--no-api`` every such call is absorbed by a no-op stub instead.
	"""
	if not active:
		yield
		return

	import strategy.confidence_scorer as scorer_module

	class _NullStream:
		"""Absorbs subscribe/unsubscribe/set_priority without touching a socket."""

		def __getattr__(self, _name: str):
			return lambda *args, **kwargs: None

	original = scorer_module.get_shared_stream
	scorer_module.get_shared_stream = lambda: _NullStream()
	try:
		yield
	finally:
		scorer_module.get_shared_stream = original


def _check_skip() -> bool:
	"""Non-blocking check if Enter was pressed (stdin has data)."""
	try:
		if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
			sys.stdin.readline()
			return True
	except (OSError, ValueError):
		pass
	return False


@dataclass
class BacktestEvent:
	symbol: str
	stream_start_utc: datetime
	first_seen_utc: datetime
	updated_utc: Optional[datetime]
	entry: Any
	rmap: Any
	window_start_utc: Optional[datetime] = None


@dataclass(frozen=True)
class CachedTapeSnapshot:
	symbol: str
	ts: datetime
	price: float
	entry_speed: float
	entry_buy_ratio: float
	pressure_acceleration: float
	pressure_acceleration_ready: bool
	buying_pressure: int
	selling_pressure: int
	buy_vol: int
	sell_vol: int
	avg_print_size: float
	source_path: Path


@dataclass(frozen=True)
class CachedNearHodWindow:
	start_ts: datetime
	end_ts: datetime
	start_price: float
	start_hod: float
	start_atr_14: float
	start_volume: int = 0


@dataclass
class SymbolBacktestResult:
	symbol: str
	stream_start_utc: datetime
	stream_end_utc: datetime
	premarket_change_pct: float
	ticks_processed: int
	tradier_calls: int
	approved_setups: int
	failure_detail: str
	faded_from_hod: bool
	final_eval: float
	final_live: float
	sim_trades: int = 0
	sim_pnl_dollars: float = 0.0
	sim_realized_pct: float = 0.0
	sim_winners: int = 0
	sim_losers: int = 0


def _to_utc(ts: datetime) -> datetime:
	if ts.tzinfo is None:
		return ts.replace(tzinfo=timezone.utc)
	return ts.astimezone(timezone.utc)


def _safe_iso(dt: Optional[datetime]) -> str:
	if dt is None:
		return "-"
	return dt.astimezone(MarketClock.local_tz()).strftime("%Y-%m-%dT%H:%M:%S %Z")


def _local_window_label(start_utc: datetime, end_utc: datetime) -> str:
	local_tz = MarketClock.local_tz()
	start_local = start_utc.astimezone(local_tz)
	end_local = end_utc.astimezone(local_tz)
	if start_local.date() == end_local.date():
		return f"{start_local:%Y-%m-%d %H:%M} -> {end_local:%H:%M %Z}"
	return f"{start_local:%Y-%m-%d %H:%M %Z} -> {end_local:%Y-%m-%d %H:%M %Z}"


def _csv_first(row: dict, *names: str) -> str:
	for name in names:
		value = row.get(name)
		if value not in (None, ""):
			return str(value)
	return ""


def _csv_float(row: dict, *names: str, default: float = 0.0) -> float:
	value = _csv_first(row, *names)
	if not value:
		return default
	try:
		return float(value)
	except ValueError:
		return default


def _csv_int(row: dict, *names: str, default: int = 0) -> int:
	value = _csv_first(row, *names)
	if not value:
		return default
	try:
		return int(float(value))
	except ValueError:
		return default


def _csv_bool(row: dict, *names: str, default: bool = False) -> bool:
	value = _csv_first(row, *names).strip().lower()
	if not value:
		return default
	return value in {"1", "true", "t", "yes", "y"}


def _csv_dt(row: dict, *names: str) -> Optional[datetime]:
	value = _csv_first(row, *names).strip()
	if not value:
		return None
	try:
		if re.fullmatch(r"\d+(\.\d+)?", value):
			epoch = float(value)
			if epoch > 1_000_000_000_000:
				epoch /= 1000.0
			return datetime.fromtimestamp(epoch, tz=timezone.utc)
		return _to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
	except ValueError:
		return None


def _fmt_number(value: Optional[float], places: int = 2) -> str:
	if value is None:
		return "n/a"
	return f"{value:.{places}f}"


def _replay_change_pct(start_price: Optional[float], end_price: float) -> float:
	start = float(start_price or 0.0)
	end = float(end_price or 0.0)
	if start <= 0 or end <= 0:
		return 0.0
	return (end - start) / start * 100.0


def _criterion_value(criterion) -> Optional[float]:
	value = getattr(criterion, "value", None)
	return float(value) if value is not None else None


def _criterion_failure_text(group_name: str, criterion) -> str:
	name = getattr(criterion, "name", "")
	value = _criterion_value(criterion)
	if name == "change_pct_min":
		return f"change={_fmt_number(value, 1)}% < {CHANGE_PCT_MIN:.1f}%"
	if name == "volume_min":
		return f"volume={int(value or 0):,} < {ATTENTION_VOLUME_MIN:,}"
	if name == "rvol_1day_min":
		return f"rvol_1d={_fmt_number(value, 2)} < {ATTENTION_RVOL_1DAY_MIN:.2f}"
	if name == "dist_from_hod_atr_max":
		return f"dist_hod_atr={_fmt_number(value, 2)} > {ATTENTION_DIST_FROM_HOD_ATR_MAX:.2f}"
	if name == "order_book_density_ok":
		return "order_book_density failed"
	if name == "vwap_slope_positive":
		return f"vwap_slope={_fmt_number(value, 4)} < {ATTENTION_VWAP_SLOPE_MIN:g}"
	if name == "close_above_vwap":
		return f"close_minus_vwap={_fmt_number(value, 4)} <= 0"
	if name == "macd_positive":
		return f"macd_hist={_fmt_number(value, 4)} <= 0"
	if name == "tape_speed":
		return f"tape_speed={_fmt_number(value, 1)}/s < {SIM_PRESSURE_SPEED_MIN:.1f}/s"
	if name == "tape_imbalance":
		return f"buy_ratio={(value or 0.0):.0%} < {SIM_PRESSURE_BUY_RATIO_MIN:.0%}"
	if name == "tape_acceleration":
		return f"tape_accel_z={_fmt_number(value, 2)} < {SIM_PRESSURE_ACCEL_Z:.2f}"
	if name == "spread_ok":
		return f"spread={(value or 0.0):.2%} > {EXEC_SPREAD_PCT_MAX:.2%}"
	if name == "dist_to_resistance_ok":
		return f"res_headroom={_fmt_number(value, 2)} < {EXEC_DIST_TO_RESISTANCE_MULT:.1f} ATR"
	if name == "no_hidden_seller":
		return f"hidden_seller={_fmt_number(value, 2)} > {EXEC_HIDDEN_SELLER_MAX:.2f}"
	return f"{group_name.lower()}:{name} failed"


def _first_required_failure(entry, group_name: str) -> str:
	get_group = getattr(entry, "get_criteria_group", None)
	group = get_group(group_name) if callable(get_group) else None
	if group is None:
		return ""
	for criterion in group.criteria:
		if criterion.required and not criterion.passing:
			return _criterion_failure_text(group.name, criterion)
	return ""


def _backtest_failure_detail(entry, scorer: Optional[DynamicScorer] = None) -> str:
	price = float(getattr(entry, "price", 0.0) or 0.0)
	if price > PRICE_MAX:
		return f"price={price:.2f} > {PRICE_MAX:.2f}"
	if price > 0 and price < PRICE_MIN:
		return f"price={price:.2f} < {PRICE_MIN:.2f}"

	import config as _config
	if not _config.confidence_bypass_eval():
		eval_confidence = float(getattr(entry, "eval_confidence", 0.0) or 0.0)
		if eval_confidence < ENTRY_CONFIDENCE_MIN:
			return f"eval_confidence={eval_confidence:.1f} < {ENTRY_CONFIDENCE_MIN:.1f}"
	if not _config.confidence_bypass_live():
		has_live_data = bool(getattr(entry, "has_live_data", False))
		live_confidence = float(getattr(entry, "live_confidence", 0.0) or 0.0)
		if has_live_data and live_confidence < ENTRY_CONFIDENCE_MIN:
			return f"live_confidence={live_confidence:.1f} < {ENTRY_CONFIDENCE_MIN:.1f}"

	attention_failure = _first_required_failure(entry, "ATTENTION")
	if attention_failure:
		return attention_failure

	setup_score = float(getattr(entry, "setup_pattern_score", 0.0) or 0.0)
	if setup_score < SIM_SETUP_SCORE_MIN:
		return f"setup_score={setup_score:.0f} < {SIM_SETUP_SCORE_MIN:.0f}"

	setup_level = getattr(entry, "setup_price_level", None)
	if setup_level is not None and setup_level > 0 and price > 0:
		max_price = setup_level * (1.0 + SIM_MAX_SETUP_EXTENSION_PCT)
		if price > max_price:
			return f"price={price:.2f} > setup_max={max_price:.2f}"

	if SIM_FIRST_MACD_CYCLE_ONLY and bool(getattr(entry, "macd_cycle_done", False)):
		return "first_macd_cycle_done (1m hist flipped negative)"

	execution_failure = _first_required_failure(entry, "EXECUTION")
	if execution_failure:
		return execution_failure

	if scorer is not None:
		ofi = getattr(scorer, "_ofi_tracker", None)
		if ofi is not None:
			total_activity = int(getattr(ofi, "activity_count", 0) or 0)
			if total_activity < SIM_PRESSURE_ACTIVITY_MIN:
				return f"tape_activity={total_activity} < {SIM_PRESSURE_ACTIVITY_MIN}"
			progress = getattr(ofi, "price_progress", None)
			if progress is None:
				return "price_progress warming up"
			if progress <= SIM_PRESSURE_PROGRESS_MIN:
				return f"price_progress={progress:.3%} <= {SIM_PRESSURE_PROGRESS_MIN:.1%}"
			if bool(getattr(ofi, "depth_active", False)):
				ofi_norm = float(getattr(ofi, "ofi_norm_gate", 0.0) or 0.0)
				if ofi_norm < SIM_OFI_NORM_MIN:
					return f"ofi_norm={ofi_norm:.2f} < {SIM_OFI_NORM_MIN}"
	return ""


def _approved_setup_signals(scorer: DynamicScorer, symbol: str, start_utc: datetime, end_utc: datetime) -> list:
	setup_logger = getattr(scorer, "_setup_logger", None)
	if setup_logger is None:
		return []
	signals_by_id = getattr(setup_logger, "_signals", {})
	setup_lock = getattr(setup_logger, "_lock", None)
	if setup_lock is None:
		signals = list(signals_by_id.values())
	else:
		with setup_lock:
			signals = list(signals_by_id.values())

	approved = []
	for signal in signals:
		if signal.symbol != symbol or not signal.signal_approved:
			continue
		signal_ts = _to_utc(signal.ts)
		if start_utc <= signal_ts <= end_utc:
			approved.append(signal)
	return approved


def _approved_setup_count(scorer: DynamicScorer, symbol: str, start_utc: datetime, end_utc: datetime) -> int:
	return len(_approved_setup_signals(scorer, symbol, start_utc, end_utc))


def _setup_failure_detail(signals: list) -> str:
	if not signals:
		return ""

	parts: list[str] = []
	for signal in sorted(signals, key=lambda signal: _to_utc(signal.ts)):
		detail = str(getattr(signal, "setup_failure_detail", "") or "").strip()
		if not detail:
			detail = str(getattr(signal, "block_reasons", "") or "").strip()
		if not detail:
			detail = "all setup-time hard gates passed"
		setup_ts = _to_utc(signal.ts).astimezone().strftime("%H:%M:%S")
		tf = getattr(signal, "setup_timeframe", "") or ""
		tf_label = f"/{tf}" if tf else ""
		parts.append(f"setup@{setup_ts}{tf_label} {detail}")
	return " | ".join(parts)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _wrapped_summary_lines(prefix: str, text: str, *, width: int) -> list[str]:
	visible_prefix = _ANSI_RE.sub("", prefix)
	available = max(24, width - len(visible_prefix))
	wrapped = textwrap.wrap(
		text,
		width=available,
		break_long_words=False,
		break_on_hyphens=False,
	)
	if not wrapped:
		return [prefix.rstrip()]
	lines = [f"{prefix}{wrapped[0]}"]
	indent = " " * len(visible_prefix)
	lines.extend(f"{indent}{line}" for line in wrapped[1:])
	return lines


def _setup_failure_items(detail: str) -> list[tuple[str, str]]:
	items: list[tuple[str, str]] = []
	for part in str(detail or "").split(" | "):
		part = part.strip()
		if not part:
			continue
		label, _, rest = part.partition(" ")
		if label.startswith("setup@") and rest:
			items.append((label.removeprefix("setup@"), rest.strip()))
		else:
			items.append(("-", part))
	return items


def _closed_sim_trades(fills: list[SimFill]) -> list[tuple[SimFill | None, SimFill]]:
	open_buys: dict[str, list[SimFill]] = {}
	closed: list[tuple[SimFill | None, SimFill]] = []

	for fill in sorted(fills, key=lambda f: f.ts):
		side = fill.side.upper()
		if side == "BUY":
			open_buys.setdefault(fill.symbol, []).append(fill)
		elif side == "SELL":
			buys = open_buys.get(fill.symbol)
			buy = buys.pop(0) if buys else None
			closed.append((buy, fill))
	return closed


def _fill_time_label(ts: datetime | None) -> str:
	if ts is None:
		return "-"
	return ts.astimezone(MarketClock.local_tz()).strftime("%m-%d %H:%M:%S")


def _fill_time_label_short(ts: datetime | None) -> str:
	if ts is None:
		return "-"
	return ts.astimezone(MarketClock.local_tz()).strftime("%H:%M:%S")


def _fill_setup_field(buy: SimFill | None, sell: SimFill | None, attr: str) -> object:
	for fill in (buy, sell):
		if fill is None:
			continue
		value = getattr(fill, attr, None)
		if value not in (None, ""):
			return value
	return ""


def _print_trade_results(fills: list[SimFill]) -> None:
	buys = [fill for fill in fills if fill.side.upper() == "BUY"]
	sells = [fill for fill in fills if fill.side.upper() == "SELL"]
	closed_trades = _closed_sim_trades(fills)
	trade_width = 150

	print("\n" + "=" * trade_width)
	print("TRADES")
	print("=" * trade_width)
	print(f"Entries: {len(buys):,}    Exits: {len(sells):,}")

	if not buys:
		print("No simulated entries triggered.")
		print("=" * trade_width)
		return

	total_pnl = 0.0
	winners = 0
	if closed_trades:
		print()
		print("Closed Trades")
		print("-" * trade_width)
		print(
			f"{'SYM':<6} {'BUY':<9} {'SELL':<9} {'TF':<6} {'BREAKOUT':<17} "
			f"{'ENTRY':>8} {'EXIT':>8} {'P&L $':>10} {'P&L %':>8} {'HELD':>7}  EXIT"
		)
		print("-" * trade_width)
		for buy, sell in closed_trades:
			total_pnl += sell.pnl_dollars
			if sell.pnl_dollars > 0:
				winners += 1
			held = f"{sell.hold_secs:.0f}s"
			setup_tf = str(_fill_setup_field(buy, sell, "setup_timeframe") or "-")
			breakout_ts = _fill_setup_field(buy, sell, "setup_breakout_ts")
			breakout_price = _fill_setup_field(buy, sell, "setup_breakout_price")
			if isinstance(breakout_ts, datetime) and breakout_price:
				breakout_label = f"{_fill_time_label_short(breakout_ts)}@{float(breakout_price):.2f}"
			elif isinstance(breakout_ts, datetime):
				breakout_label = _fill_time_label_short(breakout_ts)
			else:
				breakout_label = "-"
			entry_price = f"{buy.fill_price:.2f}" if buy else "-"
			exit_price = f"{sell.fill_price:.2f}"
			pnl_color = _g if sell.pnl_dollars >= 0 else _r
			pnl_dollars_str = pnl_color(f"{sell.pnl_dollars:>+10.2f}")
			pnl_pct_str = pnl_color(f"{sell.pnl_pct:>+8.2%}")
			prefix = (
				f"{sell.symbol:<6} {_g(_fill_time_label_short(buy.ts if buy else None)):<20} "
				f"{_r(_fill_time_label_short(sell.ts)):<20} {setup_tf:<6} {breakout_label:<17} "
				f"{entry_price:>8} {exit_price:>8} {pnl_dollars_str} "
				f"{pnl_pct_str} {held:>7}  "
			)
			for line in _wrapped_summary_lines(prefix, str(sell.exit_reason or ""), width=trade_width):
				print(line)
	else:
		print("Closed Trades: none")

	# Open positions (bought but not sold by end of replay)
	open_pos = [b for b in buys if not any(s.symbol == b.symbol and s.ts > b.ts for s in sells)]
	if open_pos:
		print()
		print(f"Open Positions (not exited): {len(open_pos)}")
		print("-" * trade_width)
		for buy in open_pos:
			setup_tf = str(buy.setup_timeframe or "-")
			breakout_label = _fill_time_label(buy.setup_breakout_ts)
			if buy.setup_breakout_price:
				breakout_label += f"@${buy.setup_breakout_price:.2f}"
			prefix = (
				f"  {buy.symbol:<6} buy={_fill_time_label(buy.ts)} "
				f"tf={setup_tf} breakout={breakout_label} entry @ ${buy.fill_price:.2f}  "
			)
			for line in _wrapped_summary_lines(prefix, str(buy.entry_reason or ""), width=trade_width):
				print(line)

	if closed_trades:
		win_rate = winners / len(closed_trades) * 100
		print()
		print(f"Realized P&L: ${total_pnl:+.2f}    Win rate: {win_rate:.0f}% ({winners}/{len(closed_trades)})")
	print("=" * trade_width)


def _sim_fills_since(start_index: int, symbol: str) -> list[SimFill]:
	sim_logger = SimulationEngine.get_logger()
	if sim_logger is None:
		return []
	return [fill for fill in sim_logger.fills[start_index:] if fill.symbol == symbol]


def _reset_replay_criteria(entry: Any) -> None:
	for group in getattr(entry, "criteria_groups", []) or []:
		for criterion in getattr(group, "criteria", []) or []:
			reset = getattr(criterion, "reset", None)
			if callable(reset):
				reset()
	# Clear persisted execution snapshot so backtest recomputes from ticks
	if not hasattr(entry, "execution_snapshot"):
		entry.execution_snapshot = {}
	else:
		entry.execution_snapshot = {}


def _reset_replay_setup_state(entry: Any) -> None:
	group = getattr(entry, "setup_pattern", None)
	for variable in getattr(group, "variables", []) or []:
		variable.value = None
		variable.quality = 0.0
	entry.setup_pattern_name = ""
	entry.setup_pattern_timeframe = ""
	entry.setup_pullback_depth = None
	entry.setup_price_level = None
	entry.setup_pullback_low = None
	entry.setup_pullback_ts = None
	entry.setup_resolved_at = None
	entry.setup_breakout_ts = None
	entry.setup_breakout_price = None
	entry.setup_failed_at = None
	# A cached live-session entry may carry macd_cycle_done=True; the replay
	# scorer recomputes it from replayed bars, but entries evaluated before
	# the first criteria refresh must not inherit the stale block.
	entry.macd_cycle_done = False
	entry.macd_cycle_done_ts = None


def _seeded_replay_volume(df_1m_seed: pd.DataFrame, df_timesales_seed: pd.DataFrame) -> int:
	candle_volume = 0
	if not df_1m_seed.empty and "volume" in df_1m_seed.columns:
		candle_volume = int(pd.to_numeric(df_1m_seed["volume"], errors="coerce").fillna(0).clip(lower=0).sum())
	timesale_volume = 0
	if not df_timesales_seed.empty and "size" in df_timesales_seed.columns:
		timesale_volume = int(pd.to_numeric(df_timesales_seed["size"], errors="coerce").fillna(0).clip(lower=0).sum())
	return max(candle_volume, timesale_volume)


def _retime_replay_static_evaluation(entry: Any, as_of_utc: datetime) -> None:
	evaluation = getattr(entry, "evaluation", None)
	stock_eval = getattr(evaluation, "stock_eval", None)
	get_var = getattr(stock_eval, "get", None)
	if not callable(get_var):
		return

	price_var = get_var("price")
	price = float(getattr(entry, "price", 0.0) or 0.0)
	if price_var is not None and price > 0:
		price_var.evaluate(price)

	rvol_var = get_var("rvol_1day")
	rvol = float(getattr(entry, "rvol_1day", 0.0) or 0.0)
	if rvol_var is not None:
		if rvol > 0:
			rvol_var.evaluate(rvol)
		else:
			rvol_var.value = None
			rvol_var.quality = 0.0
			if hasattr(rvol_var, "_n_obs"):
				rvol_var._n_obs = 0

	articles = []
	for article in getattr(entry, "news_articles", []) or []:
		published = getattr(article, "published_at", None)
		if published is None:
			continue
		published_utc = _to_utc(published)
		if published_utc <= as_of_utc:
			articles.append(article)
	articles.sort(key=lambda article: _to_utc(article.published_at), reverse=True)
	entry.news_articles = articles[:1]
	entry.news_count_last_hour = len(articles)
	if articles:
		pub = _to_utc(articles[0].published_at)
		entry.news_is_fresh = pub.astimezone(MarketClock.ET).date() == as_of_utc.astimezone(MarketClock.ET).date()
		entry.news_is_breaking = pub >= (as_of_utc - timedelta(minutes=NEWS_BREAKING_MIN))
		entity_scores = [float(a.entity_sentiment) for a in articles if getattr(a, "entity_sentiment", None) is not None]
		entry.avg_sentiment = sum(entity_scores) / len(entity_scores) if entity_scores else None
	else:
		entry.news_is_fresh = False
		entry.news_is_breaking = False
		entry.avg_sentiment = None
	news_var = get_var("breaking_news")
	if news_var is not None:
		if entry.avg_sentiment is None or entry.news_count_last_hour <= 0:
			news_var.value = None
			news_var.quality = 0.0
		elif -NEWS_NOISE_THRESHOLD <= entry.avg_sentiment <= NEWS_NOISE_THRESHOLD:
			news_var.value = None
			news_var.quality = 0.0
		else:
			news_var.evaluate((entry.avg_sentiment + 1.0) / 2.0)


def _print_backtest_summary(results: list[SymbolBacktestResult]) -> None:
	width = 126
	total_ticks = sum(r.ticks_processed for r in results)
	total_calls = sum(r.tradier_calls for r in results)
	faded = sum(1 for r in results if r.faded_from_hod)
	total_sim_trades = sum(r.sim_trades for r in results)
	total_sim_pnl = sum(r.sim_pnl_dollars for r in results)
	sorted_results = sorted(results, key=lambda r: r.premarket_change_pct, reverse=True)

	print("\n" + "=" * width)
	print("BACKTEST SUMMARY")
	print("=" * width)
	print(
		f"Symbols: {len(results):,}    Ticks: {total_ticks:,}    "
		f"API calls: {total_calls:,}    Faded: {faded:,}    "
		f"Sim trades: {total_sim_trades:,}    Sim P&L: ${total_sim_pnl:+.2f}"
	)
	print()
	print("Replay Results")
	print("-" * width)
	print(
		f"{'SYM':<6} {'START':<8} {'END':<8} {'CHG%':>8} {'TICKS':>10} "
		f"{'EVAL':>6} {'LIVE':>6} {'SETUPS':>6} {'TRADES':>6} "
		f"{'P&L':>11} {'P&L%':>7} {'W/L':>5} {'FADED':>5}"
	)
	print("-" * width)
	for result in sorted_results:
		start_str = result.stream_start_utc.astimezone().strftime("%H:%M:%S")
		end_str = result.stream_end_utc.astimezone().strftime("%H:%M:%S")
		wl_str = f"{result.sim_winners}/{result.sim_losers}" if result.sim_trades else "-"
		print(
			f"{result.symbol:<6} {start_str:<8} {end_str:<8} {result.premarket_change_pct:>+7.1f}% "
			f"{result.ticks_processed:>10,} {result.final_eval:>6.1f} "
			f"{result.final_live:>6.1f} {result.approved_setups:>6} {result.sim_trades:>6} "
			f"${result.sim_pnl_dollars:>+10.2f} {result.sim_realized_pct:>+6.1%} {wl_str:>5} "
			f"{'YES' if result.faded_from_hod else 'no':>5}"
		)

	diagnostic_results = [r for r in sorted_results[:10] if str(r.failure_detail or "").strip()]
	if diagnostic_results:
		print()
		print("Setup Diagnostics (top 10 symbols by CHG%, setups oldest to newest)")
		print("-" * width)
		for result in diagnostic_results:
			start_str = result.stream_start_utc.astimezone().strftime("%H:%M:%S")
			end_str = result.stream_end_utc.astimezone().strftime("%H:%M:%S")
			print(f"  {result.symbol}  {start_str}-{end_str}  setups={result.approved_setups}")
			for label, detail in _setup_failure_items(result.failure_detail):
				prefix = f"    {label:<8} " if label != "-" else "    -        "
				for line in _wrapped_summary_lines(prefix, detail, width=width):
					print(line)
			print()


class BacktestRunner:
	"""
	Event-time replay backtest.

	Workflow per day:
	1. Build ticker timeline from cache first_seen/updated timestamps inside the premarket window.
	2. For each ticker (sequentially), poll Tradier tick data from stream start to premarket close.
	3. Replay ticks through DynamicScorer callbacks in timestamp order.
	4. Apply the same fade-from-HOD drop rule used in main loop.

	Design choices for no-lookahead:
	- Only symbols first seen or updated inside the selected day window are replayed.
	- Tick stream is processed strictly in chronological order.
	- Rewatch times can be sourced from eval_stats.log when present (same-day window).
	"""

	def __init__(
		self,
		cache_path: Path = _DEFAULT_CACHE_PATH,
		eval_stats_path: Path = _DEFAULT_EVAL_STATS,
		window_start_et: Optional[dtime] = None,
		window_end_et: Optional[dtime] = None,
		log_dir: Path = _LOG_DIR,
		no_tradier: bool = False,
		no_api: bool = False,
		move_start: bool = False,
		output_tag: str = "",
	) -> None:
		self._cache_path = cache_path
		self._eval_stats_path = eval_stats_path
		self._window_start_et = window_start_et
		self._window_end_et = window_end_et
		self._log_dir = log_dir
		# ``no_api`` is the superset of ``no_tradier``: it blocks the same
		# provider fetches AND the live-stream dial-out the scorer makes when a
		# setup arms, so the whole replay runs off cached data with no socket.
		self._no_api = bool(no_api)
		self._no_tradier = bool(no_tradier or no_api)
		self._move_start = move_start
		# Merely constructing the provider client can fail on a machine with no
		# credentials (Tradier raises on a missing token), so --no-api skips it.
		# Every call site is gated by _no_tradier, so None is never dereferenced.
		self._hist = None if self._no_api else get_historical_client()
		self._tradier = self._hist
		self._peak_change_cache: dict[tuple[str, float, float], float] = {}
		# Sticky per-symbol skip: set the moment Enter is seen (checked between
		# fetch chunks too, so skips are responsive during long IBKR fetch
		# phases); cleared at the start of each symbol replay.
		self._skip_requested: bool = False
		# Route all simulated fills + setup signals to a dedicated backtest
		# directory so replay output stays separate from live `pos` trades.
		# An output tag (A/B arm name, e.g. "legacy"/"armed") gets its own
		# subdirectory so both arms of a comparison never overwrite each
		# other's day files.
		backtest_log_dir = (
			str(Path(SIM_BACKTEST_LOG_DIR) / output_tag)
			if output_tag else SIM_BACKTEST_LOG_DIR
		)
		SimulationEngine.configure_logger(backtest_log_dir)
		configure_setup_signal_logger(backtest_log_dir)

	def run_day(self, day: date, symbols: set[str] | None = None) -> list[SymbolBacktestResult]:
		return self.run_range(day, day, symbols=symbols)

	def run_range(self, start_day: date, end_day: date, symbols: set[str] | None = None) -> list[SymbolBacktestResult]:
		"""Replay all cached events in [start_day, end_day], ordered by the
		peak %-change each ticker reached inside its replay window (highest
		first, across ALL days — not chronologically)."""
		prepared: list[tuple[date, datetime, list[BacktestEvent]]] = []
		cur = start_day
		while cur <= end_day:
			events = self._build_day_events(cur)
			if symbols:
				events = [e for e in events if e.symbol in symbols]
			if not events:
				window_start_utc, window_end_utc = self._window_bounds_utc(cur)
				logger.info(
					"[backtest] No cache events found for %s in local window %s",
					cur.isoformat(),
					_local_window_label(window_start_utc, window_end_utc),
				)
			else:
				prepared.append((cur, self._window_end_utc(cur), events))
			cur += timedelta(days=1)
		if not prepared:
			return []

		SimulationEngine.reset_session()
		# A repeated backtest of the same day must replace prior output, not
		# append to it. Clear every day's sim-trade and setup-signal output
		# up front — replay order interleaves days, so lazy per-day clearing
		# would wipe results already written this run.
		sim_logger = SimulationEngine.get_logger()
		for day, _, _ in prepared:
			day_str = day.strftime("%Y%m%d")
			if sim_logger is not None:
				sim_logger.clear_session(day_str)
			get_setup_signal_logger().clear_session(day_str)

		keyed: list[tuple[float, date, datetime, BacktestEvent]] = []
		for day, end_utc, events in prepared:
			for event in events:
				keyed.append((self._peak_change_pct(event, end_utc), day, end_utc, event))
		keyed.sort(key=lambda item: (-item[0], item[1], item[3].symbol))

		total = len(keyed)
		logger.info(
			"[backtest] Replaying %d event(s) across %d day(s), ordered by peak %%change (desc)",
			total, len(prepared),
		)
		results: list[SymbolBacktestResult] = []
		with _no_live_stream(active=self._no_api):
			for idx, (peak, day, end_utc, event) in enumerate(keyed, start=1):
				peak_label = f"{peak:+.1f}%" if peak != float("-inf") else "n/a"
				logger.info(
					"[backtest] [%d/%d] %s %s peak=%s start=%s first_seen=%s updated=%s",
					idx,
					total,
					event.symbol,
					day.isoformat(),
					peak_label,
					_safe_iso(event.stream_start_utc),
					_safe_iso(event.first_seen_utc),
					_safe_iso(event.updated_utc),
				)
				results.append(self._replay_symbol(event, end_utc))
		return results

	def _peak_change_pct(self, event: BacktestEvent, end_utc: datetime) -> float:
		"""Highest %-change vs previous close reached inside the event's
		replay window (even if price faded afterwards). Sort key only."""
		key = (event.symbol, event.stream_start_utc.timestamp(), end_utc.timestamp())
		cached = self._peak_change_cache.get(key)
		if cached is not None:
			return cached
		peak = self._compute_peak_change_pct(event, end_utc)
		self._peak_change_cache[key] = peak
		return peak

	def _compute_peak_change_pct(self, event: BacktestEvent, end_utc: datetime) -> float:
		entry = event.entry
		price = float(getattr(entry, "price", 0.0) or 0.0)
		change_pct = float(getattr(entry, "change_pct", 0.0) or 0.0)
		# Prefer the eval-time stored prev_close: entry.change_pct is mutated
		# live while entry.price is not, so deriving from a live-tracked
		# snapshot mixes epochs. Derivation is the legacy-snapshot fallback.
		prev_close = float(getattr(entry, "prev_close", 0.0) or 0.0)
		if prev_close <= 0:
			prev_close = price / (1.0 + change_pct / 100.0) if price > 0 and change_pct > -100.0 else 0.0

		candidates: list[float] = []
		# Live-tracked session peak persisted by DynamicScorer._raise_hod.
		stored_peak = float(getattr(entry, "peak_change_pct", 0.0) or 0.0)
		if stored_peak > 0:
			candidates.append(stored_peak)
		# Snapshot HOD (live-updated) against prev_close.
		daily_high = float(getattr(entry, "daily_high", 0.0) or 0.0)
		if prev_close > 0 and daily_high > 0:
			candidates.append((daily_high - prev_close) / prev_close * 100.0)
		if prev_close > 0:
			candidates.append(change_pct)

		start_utc = event.stream_start_utc  # already move-start adjusted
		peak_end = end_utc
		# If the algo purposefully stopped watching (faded from HOD), cached
		# near-HOD windows end at that point. A new peak (fresh HOD) can only
		# print inside a near-HOD window, so bound the candle scan there.
		try:
			windows = self._load_cached_near_hod_windows(event.symbol, start_utc, end_utc)
		except Exception:
			windows = []
		if windows:
			last_end = max(w.end_ts for w in windows)
			if last_end > start_utc:
				peak_end = min(end_utc, last_end)

		# Sort key is best-effort from CACHED data only — never fetch here.
		# Partial coverage keys on whatever max it does have; the replay's own
		# backfills persist candles, so the key self-corrects on later runs.
		candles = load_candle_cache(event.symbol, "1m", start_utc, peak_end, self._log_dir)
		if not candles.empty and "high" in candles.columns and prev_close > 0:
			highs = pd.to_numeric(candles["high"], errors="coerce").dropna()
			if not highs.empty and float(highs.max()) > 0:
				candidates.append((float(highs.max()) - prev_close) / prev_close * 100.0)

		return max(candidates) if candidates else float("-inf")

	# ------------------------------------------------------------------
	# Manual ticker injection — evaluate fresh + write into day snapshot
	# ------------------------------------------------------------------
	def add_tickers_to_day(
		self,
		symbols: list[str],
		day: date,
		at_et: Optional[dtime] = None,
	) -> list[str]:
		"""Evaluate *symbols* via the live pipeline and inject them into
		``logs/cache_snapshots/<day>.pkl`` so they participate in
		``run_day(day)``.

		Useful for backtesting days where the live scanner was offline.
		Caveats:
		  * News, float and profile come from the **current** APIs, so
		    they reflect today, not *day*.  The tick replay itself is
		    still historical and accurate.
		  * Stage 1 / Stage 2 gates are bypassed (``force=True``) so the
		    ticker is always cached; the historical replay will then
		    decide what actually trades.

		``at_et`` is the synthetic first-seen time stamped into the
		snapshot; defaults to the configured premarket window start.
		"""
		if self._no_api:
			raise RuntimeError(
				"add_tickers_to_day evaluates symbols through the live pipeline — "
				"incompatible with no_api=True"
			)

		from pipeline import run_pipeline
		from strategy.evaluator import StockEvaluator

		if at_et is not None:
			stamp_dt_et = datetime.combine(day, at_et, tzinfo=MarketClock.ET)
			stamp_utc = stamp_dt_et.astimezone(timezone.utc)
		else:
			stamp_utc, _ = self._window_bounds_utc(day)
		stamp_ts = stamp_utc.timestamp()

		# Load existing snapshot (if any) so we merge instead of overwrite.
		existing = TickerCache.load_snapshot(day) or {}
		merged: dict[str, tuple] = dict(existing)

		evaluator = StockEvaluator()
		# Warm FinBERT once so per-symbol news scoring doesn't pay the
		# model-load cost repeatedly.
		evaluator._finbert._ensure_model()

		added: list[str] = []
		for raw in symbols:
			sym = raw.strip().upper()
			if not sym:
				continue
			logger.info("[backtest] add: evaluating %s for %s", sym, day.isoformat())
			try:
				result = run_pipeline(
					sym,
					evaluator,
					self._tradier,
					quiet=True,
					force=True,
					allow_manual_rating=False,
				)
			except Exception:
				logger.exception("[backtest] add: pipeline failed for %s", sym)
				continue

			if result is None:
				logger.warning("[backtest] add: %s — pipeline returned no result", sym)
				continue

			# The snapshot is pickled below; wait for the MA levels so they are
			# on the map when it is. (They reflect bars as of now, not *day* —
			# the same caveat as float and profile above.)
			ma_levels = getattr(result, "ma_levels", None)
			if ma_levels is not None:
				try:
					ma_levels.result(timeout=120)
				except Exception as exc:
					logger.warning("[backtest] add: %s — MA levels not marked: %s", sym, exc)

			# We never want the live scorer to keep streaming — we only
			# need the static evaluation + resistance map.
			if result.scorer is not None:
				try:
					result.scorer.stop()
				except Exception:
					logger.debug("[backtest] add: scorer.stop() failed for %s", sym, exc_info=True)
			try:
				result.early_stream.stop()
			except Exception:
				pass

			# Replace the live-fetched news with news bounded to *day*.
			# AV/Finnhub both accept arbitrary date ranges, so this is
			# the only field of the entry that we can correctly back-date.
			self._apply_historical_news(evaluator, result.entry, stamp_utc)

			merged[sym] = (result.entry, result.rmap, stamp_ts, stamp_ts)
			added.append(sym)
			logger.info(
				"[backtest] add: %s evaluated  eval=%.1f live=%.1f  first_seen=%s",
				sym, result.entry.eval_confidence, result.entry.live_confidence,
				_safe_iso(stamp_utc),
			)

		if not added:
			logger.warning("[backtest] add: no tickers were successfully evaluated")
			return added

		# Persist merged snapshot back to disk.
		_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
		snap_path = _SNAPSHOT_DIR / f"{day.isoformat()}.pkl"
		with open(snap_path, "wb") as f:
			pickle.dump({"entries": merged}, f, protocol=pickle.HIGHEST_PROTOCOL)
		logger.info(
			"[backtest] add: snapshot updated %s (+%d, total=%d)",
			snap_path, len(added), len(merged),
		)
		return added

	def _window_bounds_utc(self, day: date) -> tuple[datetime, datetime]:
		default_start_utc, default_end_utc = MarketClock.premarket_window_utc(day)
		start_utc = default_start_utc
		end_utc = default_end_utc
		if self._window_start_et is not None:
			start_utc = datetime.combine(day, self._window_start_et, tzinfo=MarketClock.ET).astimezone(timezone.utc)
		if self._window_end_et is not None:
			end_utc = datetime.combine(day, self._window_end_et, tzinfo=MarketClock.ET).astimezone(timezone.utc)
		return start_utc, end_utc

	def _window_end_utc(self, day: date) -> datetime:
		_, end_utc = self._window_bounds_utc(day)
		return end_utc

	def _load_cache_entries(self, day: date) -> dict[str, tuple]:
		"""Load cache entries: daily snapshot first, then fall back to live cache."""
		# For today's replay, keep the daily snapshot synchronized with the
		# live cache so late same-day re-evaluations are not missed.
		# Merge live cache entries INTO the existing snapshot so that manually
		# injected tickers (--add / --momo-gainers) are preserved.
		if day == MarketClock.now_et().date() and self._cache_path.exists():
			try:
				existing_snap = TickerCache.load_snapshot(day) or {}
				live_cache = TickerCache(path=self._cache_path)
				live_cache.snapshot_day(day)
				live_entries = TickerCache.load_snapshot(day) or {}
				# Re-merge: injected entries take precedence (they were
				# explicitly added), live entries fill in the rest.
				if existing_snap:
					merged = dict(live_entries)
					merged.update(existing_snap)
					if merged != live_entries:
						snap_path = _SNAPSHOT_DIR / f"{day.isoformat()}.pkl"
						with open(snap_path, "wb") as f:
							pickle.dump({"entries": merged}, f, protocol=pickle.HIGHEST_PROTOCOL)
			except Exception as exc:
				logger.debug("[backtest] Snapshot refresh failed for %s: %s", day.isoformat(), exc)

		# Try daily snapshot
		entries = TickerCache.load_snapshot(day)
		if entries:
			logger.info("[backtest] Loaded snapshot for %s (%d symbols)", day.isoformat(), len(entries))
			return entries

		# Fall back to live cache (only useful for today)
		if not self._cache_path.exists():
			return {}
		with open(self._cache_path, "rb") as f:
			raw = pickle.load(f)
		if isinstance(raw, dict) and "entries" in raw:
			return raw.get("entries", {})
		if isinstance(raw, dict):
			return raw
		return {}

	def _eval_stats_hits(self, day: date) -> dict[str, list[datetime]]:
		if not self._eval_stats_path.exists():
			return {}

		start_utc, end_utc = self._window_bounds_utc(day)
		pat = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \|\s*([A-Z.]+)\s*\|")
		hits: dict[str, list[datetime]] = {}

		for line in self._eval_stats_path.read_text().splitlines():
			m = pat.match(line)
			if m is None:
				continue
			ts_utc = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
			if not (start_utc <= ts_utc <= end_utc):
				continue
			sym = m.group(2).upper()
			hits.setdefault(sym, []).append(ts_utc)

		for sym in hits:
			hits[sym].sort()
		return hits

	def _build_day_events(self, day: date) -> list[BacktestEvent]:
		start_utc, end_utc = self._window_bounds_utc(day)
		cache_entries = self._load_cache_entries(day)
		stats_hits = self._eval_stats_hits(day)

		events: list[BacktestEvent] = []
		for sym, item in cache_entries.items():
			if not isinstance(item, tuple) or len(item) < 3:
				continue
			entry, rmap = item[0], item[1]
			first_seen = datetime.fromtimestamp(float(item[2]), tz=timezone.utc)
			updated = None
			if len(item) >= 4:
				updated = datetime.fromtimestamp(float(item[3]), tz=timezone.utc)


			# Cache is source-of-truth for symbol inclusion in the day replay.
			# If a symbol was cached on a prior day but re-evaluated today,
			# replay from today's update time rather than dropping it.
			cache_start = None
			if start_utc <= first_seen <= end_utc:
				cache_start = first_seen
			elif first_seen.astimezone(MarketClock.ET).date() == day and first_seen < start_utc:
				cache_start = start_utc
			elif updated is not None and start_utc <= updated <= end_utc:
				cache_start = updated
			elif updated is not None and updated.astimezone(MarketClock.ET).date() == day:
				# Re-evaluated later in the same day (e.g. manual -r after
				# the window closed) — still include from window start.
				cache_start = start_utc
			if cache_start is None:
				continue

			candidate_starts = stats_hits.get(sym.upper(), []) or [cache_start]
			for st in candidate_starts:
				stream_start = max(start_utc, st)
				if stream_start >= end_utc:
					continue
				events.append(
					BacktestEvent(
						symbol=sym.upper(),
						stream_start_utc=stream_start,
						first_seen_utc=first_seen,
						updated_utc=updated,
						entry=entry,
						rmap=rmap,
					)
				)

		if self._move_start:
			events = self._override_with_move_start(events, start_utc, end_utc)
		else:
			events = self._apply_default_move_start(events, start_utc, end_utc)

		events.sort(key=lambda e: (e.stream_start_utc, e.symbol))
		return events

	def _override_with_move_start(
		self,
		events: list[BacktestEvent],
		start_utc: datetime,
		end_utc: datetime,
	) -> list[BacktestEvent]:
		"""Replace stream_start_utc with detected move start from cached 1m candles."""
		for event in events:
			df_1m = load_candle_cache(event.symbol, "1m", start_utc, end_utc, self._log_dir)
			if df_1m.empty:
				continue
			move = detect_move_start(df_1m)
			if move is None:
				continue
			move_ts = df_1m.iloc[move.start_idx]["ts"]
			if hasattr(move_ts, "to_pydatetime"):
				move_ts = move_ts.to_pydatetime()
			if move_ts.tzinfo is None:
				move_ts = move_ts.replace(tzinfo=timezone.utc)
			if start_utc <= move_ts < event.stream_start_utc:
				logger.info(
					"[backtest] %s move-start override: %s → %s",
					event.symbol,
					event.stream_start_utc.astimezone().strftime("%H:%M:%S"),
					move_ts.astimezone().strftime("%H:%M:%S"),
				)
				event.stream_start_utc = move_ts
		return events

	def _apply_default_move_start(
		self,
		events: list[BacktestEvent],
		start_utc: datetime,
		end_utc: datetime,
	) -> list[BacktestEvent]:
		"""Advance stream_start to first available candle when it sits on the window boundary.

		When a ticker's stream_start equals the window start (i.e. no
		precise first_seen within the window), advance it to the first
		candle in the chart cache — there's no data to replay before that.
		"""
		for event in events:
			if event.stream_start_utc != start_utc:
				continue
			df_1m = load_candle_cache(event.symbol, "1m", start_utc, end_utc, self._log_dir)
			if df_1m.empty:
				continue
			first_ts = df_1m.iloc[0]["ts"]
			if hasattr(first_ts, "to_pydatetime"):
				first_ts = first_ts.to_pydatetime()
			if first_ts.tzinfo is None:
				first_ts = first_ts.replace(tzinfo=timezone.utc)
			if start_utc < first_ts < end_utc:
				event.window_start_utc = start_utc
				event.stream_start_utc = first_ts
		return events

	def _skip_pending(self) -> bool:
		"""True once the user pressed Enter during the current symbol.

		Sticky: fetch helpers consult this between chunks and short-circuit
		to empty results, so one keypress aborts the whole fetch cascade and
		the replay loop exits on its next iteration."""
		if not self._skip_requested and _check_skip():
			self._skip_requested = True
		return self._skip_requested

	def _replay_symbol(self, event: BacktestEvent, end_utc: datetime) -> SymbolBacktestResult:
		# Drain any buffered Enter presses so they don't skip this ticker
		_check_skip()
		self._skip_requested = False

		entry = copy.deepcopy(event.entry)
		rmap = copy.deepcopy(event.rmap)

		# Migrate stale cache entries that have the old 6-group evaluation schema.
		self._migrate_evaluation(entry)
		ev = getattr(entry, "evaluation", None)
		if ev is not None:
			_refresh_scoring_thresholds(ev)
			_reset_live_state(ev)
		_reset_replay_criteria(entry)
		_reset_replay_setup_state(entry)
		if hasattr(entry, "daily_high"):
			entry.daily_high = 0.0

		df_1m_seed, df_10s_seed, df_timesales_seed = self._load_cached_seed_market_data(
			event.symbol,
			event.stream_start_utc,
		)
		entry.volume = _seeded_replay_volume(df_1m_seed, df_timesales_seed)
		avg_volume = float(getattr(entry, "avg_volume", 0.0) or 0.0)
		entry.rvol_1day = entry.volume / avg_volume if avg_volume > 0 else 0.0
		_retime_replay_static_evaluation(entry, event.stream_start_utc)
		scorer_kwargs: dict[str, Any] = {
			"avg_daily_volume": entry.avg_volume,
			"daily_candles": [],
			"cache_live_data": False,
		}
		if not df_1m_seed.empty:
			scorer_kwargs["seed_1min_session"] = df_1m_seed
		if not df_10s_seed.empty:
			scorer_kwargs["seed_10sec"] = df_10s_seed
		if not df_timesales_seed.empty:
			scorer_kwargs["seed_timesales"] = df_timesales_seed

		scorer = DynamicScorer(entry, rmap, **scorer_kwargs)
		# Replay re-seeds whole-session candle context on every near-HOD window
		# entry; keep the pattern buffer chronological + ts-deduped so a stale
		# bar can't be dropped into the middle of the live sequence.
		if hasattr(scorer, "_tf_buf"):
			scorer._tf_buf._reconcile = True
		sim_logger = SimulationEngine.get_logger()
		sim_fill_start = len(sim_logger.fills) if sim_logger is not None else 0
		self._seed_scorer_from_cached_market_data(scorer, df_1m_seed, df_10s_seed, df_timesales_seed)

		# L2 book replay: captured depth snapshots (if any) are interleaved
		# with the tick stream by timestamp and fed through the SAME live
		# handler (scorer._on_book), so replayed deep OFI is computed by
		# production code. Sessions without captured depth degrade naturally
		# (depth_active stays False → L1 OFI semantics).
		book_snapshots = self._load_book_snapshots(
			event.symbol, event.stream_start_utc, end_utc,
		)
		book_idx = 0
		if book_snapshots:
			logger.info(
				"[backtest] %s: replaying %d cached L2 book snapshots",
				event.symbol, len(book_snapshots),
			)

		ticks_processed = 0
		tradier_calls = 0
		faded = False
		fade_resume_logged = False
		tick_rows: list[tuple[datetime, float, int]] = []
		ta_last_minute: Optional[datetime] = None
		last_live_conf: float = entry.live_confidence
		high_price: float = 0.0
		low_price: float = float("inf")
		start_price: Optional[float] = None
		static_eval_retimed = False
		# Use the original window boundary for near-HOD/pullback window loading
		# so that advancing stream_start (first-candle move-start) doesn't change
		# which cached windows match or how pullback intersection works.
		hod_window_start = event.window_start_utc or event.stream_start_utc
		near_hod_windows = self._load_cached_near_hod_windows(event.symbol, hod_window_start, end_utc)
		if not near_hod_windows:
			near_hod_windows = self._derive_near_hod_windows_from_candle_cache(event.symbol, hod_window_start, end_utc)
		pullback_replay_windows, pullback_windows_available = self._derive_pullback_replay_windows_from_candle_cache(
			event.symbol,
			hod_window_start,
			end_utc,
			near_hod_windows,
		)
		if pullback_windows_available and pullback_replay_windows:
			replay_windows = pullback_replay_windows
			replay_window_source = "pullback"
		elif near_hod_windows:
			replay_windows = near_hod_windows
			replay_window_source = "near_hod_fallback" if pullback_windows_available else "near_hod"
		else:
			replay_windows = []
			replay_window_source = "pullback_empty" if pullback_windows_available else "none"
		use_cached_near_hod_windows = pullback_windows_available or bool(replay_windows)
		near_hod_window_idx = 0
		active_near_hod_window_idx: int | None = None
		near_hod_handler_ticks = 0
		generated_near_hod_windows: list[dict[str, Any]] = []
		open_near_hod_window: dict[str, Any] | None = None
		last_near_hod_ts: datetime | None = None
		if replay_window_source == "pullback":
			logger.info(
				"[backtest] %s: using %d candle pullback replay windows inside 2ATR-HOD",
				event.symbol, len(replay_windows),
			)
		elif replay_window_source == "near_hod_fallback":
			logger.info(
				"[backtest] %s: no candle pullback replay windows; using %d cached 2ATR-HOD replay windows",
				event.symbol, len(replay_windows),
			)
		elif replay_window_source == "pullback_empty":
			logger.info(
				"[backtest] %s: using 0 candle pullback replay windows inside 2ATR-HOD",
				event.symbol,
			)
		elif use_cached_near_hod_windows:
			logger.info(
				"[backtest] %s: using %d cached 2ATR-HOD replay windows",
				event.symbol, len(replay_windows),
			)
		use_full_tick_stream = False
		# Keep cached-window replays windowed.  _iter_ticks() fills holes inside
		# the selected replay windows directly; widening to the full stream can
		# drop the actual window when a large Tradier chunk overflows.

		def _fade_should_break(tick) -> bool:
			"""Fade-from-HOD ends the replay ONLY when no replay window remains.

			Live, a >DAILY_HIGH_FADE_PCT fade drops the ticker from the
			pipeline — but the scanners re-add it as soon as it prints near
			the rolling HOD again, and a later replay window IS that
			re-discovery evidence (a pullback structure near HOD). Breaking
			unconditionally amputates every post-fade recovery (YXT
			2026-08-05: faded 28% off the pre-move spike at 04:58, was
			re-discovered minutes later and ran +450%).
			"""
			nonlocal faded, fade_resume_logged
			if not self._is_faded_from_hod(entry):
				return False
			if any(w.start_ts > tick.ts for w in replay_windows):
				if not fade_resume_logged:
					logger.info(
						"[backtest] %s faded from HOD @ %s px=%.2f — later replay window exists, resuming (re-discovery)",
						event.symbol, tick.ts.astimezone().strftime("%H:%M:%S"), tick.last,
					)
					fade_resume_logged = True
				return False
			faded = True
			logger.info(
				"[backtest] %s FADED from HOD @ %s  px=%.2f  high=%.2f",
				event.symbol, tick.ts.astimezone().strftime("%H:%M:%S"), tick.last, high_price,
			)
			return True

		tick_iter = (
			self._iter_ticks(event.symbol, event.stream_start_utc, end_utc, replay_windows)
			if use_cached_near_hod_windows and not use_full_tick_stream
			else self._iter_ticks(event.symbol, event.stream_start_utc, end_utc)
		)
		for call_count, tick in tick_iter:
			tradier_calls = max(tradier_calls, call_count)
			ticks_processed += 1

			# Feed captured book snapshots up to this tick's timestamp —
			# mirrors live arrival order (book state precedes the print).
			while book_idx < len(book_snapshots) and book_snapshots[book_idx].ts <= tick.ts:
				scorer._on_book(book_snapshots[book_idx])
				book_idx += 1

			# Periodic non-blocking check: skip ticker on Enter press. Also
			# honors a skip raised during a fetch inside the tick generator.
			if self._skip_requested or (ticks_processed % _SKIP_CHECK_INTERVAL == 0 and _check_skip()):
				logger.info("[backtest] %s SKIPPED by user (Enter pressed)", event.symbol)
				break

			entry.price = tick.last
			if not static_eval_retimed:
				_retime_replay_static_evaluation(entry, event.stream_start_utc)
				static_eval_retimed = True

			if start_price is None:
				start_price = tick.last
			high_price = max(high_price, tick.last)
			low_price = min(low_price, tick.last)

			tick_rows.append((tick.ts, tick.last, tick.size))
			minute_key = tick.ts.replace(second=0, microsecond=0)
			if use_cached_near_hod_windows:
				in_window, near_hod_window_idx = self._tick_in_near_hod_windows(
					tick.ts,
					replay_windows,
					near_hod_window_idx,
				)
				entered_window = in_window and active_near_hod_window_idx != near_hod_window_idx
				if entered_window:
					window = replay_windows[near_hod_window_idx]
					self._seed_replay_window_candle_context(scorer, event.symbol, window.start_ts)
					entry.daily_high = max(float(getattr(entry, "daily_high", 0.0) or 0.0), window.start_hod)
					if window.start_atr_14 > 0:
						scorer._last_atr = window.start_atr_14
					if window.start_volume > int(getattr(entry, "volume", 0) or 0):
						entry.volume = window.start_volume
						avg_vol = float(getattr(scorer, "_avg_vol", 0.0) or 0.0)
						if avg_vol > 0:
							entry.rvol_1day = entry.volume / avg_vol
					active_near_hod_window_idx = near_hod_window_idx
				if in_window and (entered_window or ta_last_minute != minute_key):
					self._refresh_ta_from_ticks(scorer, tick_rows)
					ta_last_minute = minute_key
				process_tick = in_window
			else:
				new_minute = ta_last_minute != minute_key
				if new_minute:
					self._refresh_ta_from_ticks(scorer, tick_rows)
					ta_last_minute = minute_key
				near_hod = self._replay_near_hod(scorer, tick.last)
				if near_hod:
					if open_near_hod_window is None:
						open_near_hod_window = {
							"symbol": event.symbol,
							"replay_start_ts": event.stream_start_utc,
							"replay_end_ts": end_utc,
							"start_ts": tick.ts,
							"end_ts": tick.ts,
							"start_price": tick.last,
							"start_hod": float(getattr(entry, "daily_high", 0.0) or 0.0),
							"start_atr_14": float(getattr(scorer, "_last_atr", 0.0) or 0.0),
							"start_volume": int(getattr(entry, "volume", 0) or 0),
							"threshold_atrs": ATTENTION_DIST_FROM_HOD_ATR_MAX,
							"source": "replay",
						}
					last_near_hod_ts = tick.ts
				else:
					if open_near_hod_window is not None:
						open_near_hod_window["end_ts"] = last_near_hod_ts or open_near_hod_window["start_ts"]
						generated_near_hod_windows.append(open_near_hod_window)
						open_near_hod_window = None
						last_near_hod_ts = None
				process_tick = near_hod

			if not process_tick:
				self._advance_skipped_replay_tick(scorer, tick)
				if _fade_should_break(tick):
					break
				continue

			cached_snapshot = getattr(tick, "_cached_tape_snapshot", None)
			if cached_snapshot is not None:
				self._apply_cached_tape_snapshot(scorer, cached_snapshot)

			near_hod_handler_ticks += 1
			quote = self._quote_from_tick(tick)
			scorer._on_quote(quote)
			scorer._on_timesale(tick)

			# Log live confidence changes > 1 point
			cur_live_conf = entry.live_confidence
			if abs(cur_live_conf - last_live_conf) >= 1.0:
				logger.debug(
					"[backtest] %s @ %s  live %.1f → %.1f  px=%.2f",
					event.symbol, tick.ts.astimezone().strftime("%H:%M:%S"),
					last_live_conf, cur_live_conf, tick.last,
				)
				last_live_conf = cur_live_conf

			if _fade_should_break(tick):
				break

		if open_near_hod_window is not None:
			open_near_hod_window["end_ts"] = last_near_hod_ts or open_near_hod_window["start_ts"]
			generated_near_hod_windows.append(open_near_hod_window)
		if generated_near_hod_windows and not use_cached_near_hod_windows:
			write_near_hod_window_cache(event.symbol, generated_near_hod_windows, self._log_dir)

		# One final TA refresh at stream end.
		if tick_rows:
			self._refresh_ta_from_ticks(scorer, tick_rows)

		premarket_change_pct = _replay_change_pct(start_price, entry.price)
		final_eval = entry.eval_confidence
		final_live = entry.live_confidence
		approved_setup_signals = _approved_setup_signals(scorer, event.symbol, event.stream_start_utc, end_utc)
		approved_setups = len(approved_setup_signals)
		failure_detail = _setup_failure_detail(approved_setup_signals) or _backtest_failure_detail(entry, scorer)
		logger.info(
			"[backtest] %s DONE  ticks=%d  handled=%d  eval=%.1f  live=%.1f  px: %.2f→%.2f (hi=%.2f lo=%.2f)%s%s\n",
			event.symbol, ticks_processed, near_hod_handler_ticks, final_eval, final_live,
			start_price or 0, entry.price, high_price, low_price if low_price != float("inf") else 0.0,
			"  [FADED]" if faded else "",
			"  [SKIPPED]" if self._skip_requested else "",
		)

		scorer.stop()
		sim_fills = _sim_fills_since(sim_fill_start, event.symbol)
		closed_sim_trades = _closed_sim_trades(sim_fills)
		sim_summary = summarize_closed_trades(closed_sim_trades)
		return SymbolBacktestResult(
			symbol=event.symbol,
			stream_start_utc=event.stream_start_utc,
			stream_end_utc=end_utc,
			premarket_change_pct=premarket_change_pct,
			ticks_processed=ticks_processed,
			tradier_calls=tradier_calls,
			approved_setups=approved_setups,
			failure_detail=failure_detail,
			faded_from_hod=faded,
			final_eval=final_eval,
			final_live=final_live,
			sim_trades=len(closed_sim_trades),
			sim_pnl_dollars=sim_summary.total_pnl,
			sim_realized_pct=sim_summary.realized_pct,
			sim_winners=sim_summary.winners,
			sim_losers=sim_summary.losers,
		)

	def _iter_ticks(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
		windows: list[CachedNearHodWindow] | None = None,
	) -> Iterable[tuple[int, LiveTimesale]]:
		if windows is not None:
			merged_windows = self._merge_near_hod_windows(windows)
			if not merged_windows:
				return
			load_cached_timesales = self._load_cached_timesales
			if "windows" in inspect.signature(load_cached_timesales).parameters:
				cached_timesales = load_cached_timesales(symbol, start_utc, end_utc, windows=merged_windows)
			else:
				cached_timesales = self._filter_events_to_windows(
					load_cached_timesales(symbol, start_utc, end_utc),
					merged_windows,
				)
			cached_timesales = self._filter_events_to_windows(cached_timesales, merged_windows)
			if cached_timesales:
				holes = self._timesale_cache_holes(cached_timesales, merged_windows)
				logger.info(
					"[backtest] %s: using %d cached live timesale ticks across %d replay windows (%s → %s)%s",
					symbol, len(cached_timesales), len(merged_windows), _safe_iso(start_utc), _safe_iso(end_utc),
					f"; will fill {len(holes)} holes from {'IBKR' if IBKR_ENABLED else 'Tradier'} as reached" if holes else "",
				)
				for call_count, event in self._iter_cached_with_hole_fetch(symbol, cached_timesales, holes):
					yield call_count, event
				return

			cached_chart_ticks = self._load_cached_chart_ticks(symbol, start_utc, end_utc)
			if cached_chart_ticks and not IBKR_ENABLED:
				filtered = self._filter_events_to_windows(cached_chart_ticks, merged_windows)
				logger.info(
					"[backtest] %s: using %d cached chart ticks across %d replay windows (%s → %s)",
					symbol, len(filtered), len(merged_windows), _safe_iso(start_utc), _safe_iso(end_utc),
				)
				for event in filtered:
					yield 0, event
				return

			for window in merged_windows:
				window_start = max(window.start_ts, start_utc)
				window_end = min(window.end_ts, end_utc)
				if window_end < window_start:
					continue
				yield from self._iter_ticks(symbol, window_start, window_end, windows=None)
			return

		cached_timesales = self._load_cached_timesales(symbol, start_utc, end_utc)
		if cached_timesales:
			holes = self._timesale_cache_holes(
				cached_timesales,
				[CachedNearHodWindow(start_utc, end_utc, 0.0, 0.0, 0.0)],
			)
			logger.info(
				"[backtest] %s: using %d cached live timesale ticks (%s → %s)%s",
				symbol, len(cached_timesales), _safe_iso(start_utc), _safe_iso(end_utc),
				f"; will fill {len(holes)} holes from {'IBKR' if IBKR_ENABLED else 'Tradier'} as reached" if holes else "",
			)
			for call_count, event in self._iter_cached_with_hole_fetch(symbol, cached_timesales, holes):
				yield call_count, event
			return

		cached_chart_ticks = self._load_cached_chart_ticks(symbol, start_utc, end_utc)
		if cached_chart_ticks and not IBKR_ENABLED:
			logger.info(
				"[backtest] %s: using %d cached chart ticks (%s → %s)",
				symbol, len(cached_chart_ticks), _safe_iso(start_utc), _safe_iso(end_utc),
			)
			for event in cached_chart_ticks:
				yield 0, event
			return

		rows: list[dict] = []
		try:
			rows = self._fetch_ticks(symbol, start_utc, end_utc)
		except Exception as e:
			logger.warning("[backtest] %s: historical fetch failed — %s", symbol, e)

		cached_snapshots = self._load_cached_tape_snapshots(symbol, start_utc, end_utc)
		if rows:
			source_label = "IBKR ticks" if IBKR_ENABLED else "ticks"
			logger.info(
				"[backtest] %s: fetched %d %s (%s → %s)",
				symbol, len(rows), source_label, _safe_iso(start_utc), _safe_iso(end_utc),
			)
		if cached_snapshots:
			logger.info(
				"[backtest] %s: merging %d cached live tape-gate snapshots",
				symbol, len(cached_snapshots),
			)
		if not rows and not cached_snapshots:
			return

		events: list[tuple[datetime, int, LiveTimesale]] = [
			(event.ts, 1, event)
			for event in self._events_from_tick_rows(symbol, rows, start_utc, end_utc)
		]

		seq = len(events)
		for snapshot in cached_snapshots:
			seq += 1
			event = self._synthetic_tick_from_snapshot(snapshot, seq)
			events.append((snapshot.ts, 0, event))

		events.sort(key=lambda item: (item[0], item[1], item[2].seq))
		for _, call_count, event in events:
			yield call_count, event

	@staticmethod
	def _timesale_cache_holes(
		events: list[LiveTimesale],
		windows: list[CachedNearHodWindow],
	) -> list[tuple[datetime, datetime]]:
		if not windows:
			return []
		min_gap = timedelta(seconds=_CACHE_TIMESALE_HOLE_SECONDS)
		ordered_events = sorted(events, key=lambda event: (event.ts, event.seq))
		holes: list[tuple[datetime, datetime]] = []
		for window in windows:
			window_start = _to_utc(window.start_ts)
			window_end = _to_utc(window.end_ts)
			if window_end <= window_start:
				continue
			window_events = [event for event in ordered_events if window_start <= event.ts <= window_end]
			if not window_events:
				holes.append((window_start, window_end))
				continue

			first_ts = window_events[0].ts
			if first_ts - window_start >= min_gap:
				holes.append((window_start, first_ts))
			prev_ts = first_ts
			for event in window_events[1:]:
				if event.ts - prev_ts >= min_gap:
					holes.append((prev_ts, event.ts))
				prev_ts = event.ts
			if window_end - prev_ts >= min_gap:
				holes.append((prev_ts, window_end))
		return holes

	def _cached_timesale_frame_holes(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
	) -> list[tuple[datetime, datetime]]:
		cached = load_timesales_cache(symbol, start_utc, end_utc, self._log_dir)
		if cached.empty or "ts" not in cached.columns:
			return []
		ordered = pd.to_datetime(cached["ts"], utc=True, errors="coerce").dropna().sort_values()
		if ordered.empty:
			return []

		min_gap = timedelta(seconds=_CACHE_TIMESALE_HOLE_SECONDS)
		start = _to_utc(start_utc)
		end = _to_utc(end_utc)
		holes: list[tuple[datetime, datetime]] = []
		first_ts = ordered.iloc[0].to_pydatetime()
		if first_ts - start >= min_gap:
			holes.append((start, first_ts))
		prev_ts = first_ts
		for ts_value in ordered.iloc[1:]:
			ts = ts_value.to_pydatetime()
			if ts - prev_ts >= min_gap:
				holes.append((prev_ts, ts))
			prev_ts = ts
		if end - prev_ts >= min_gap:
			holes.append((prev_ts, end))
		return holes

	def _iter_cached_with_hole_fetch(
		self,
		symbol: str,
		cached_events: list[LiveTimesale],
		holes: list[tuple[datetime, datetime]],
	) -> Iterable[tuple[int, LiveTimesale]]:
		ordered_cached = sorted(cached_events, key=lambda event: (event.ts, event.seq))
		ordered_holes = sorted(holes, key=lambda hole: (hole[0], hole[1]))
		seen: set[tuple[datetime, float, int]] = set()
		idx = 0

		def _yield_cached_until(cutoff: datetime, include_cutoff: bool) -> Iterable[tuple[int, LiveTimesale]]:
			nonlocal idx
			while idx < len(ordered_cached):
				event = ordered_cached[idx]
				if event.ts > cutoff or (event.ts == cutoff and not include_cutoff):
					break
				idx += 1
				key = (event.ts, event.last, event.size)
				seen.add(key)
				yield 0, event

		for hole_start, hole_end in ordered_holes:
			yield from _yield_cached_until(hole_start, include_cutoff=True)
			fetched = self._fetch_events_for_holes(symbol, [(hole_start, hole_end)])
			if fetched:
				source_label = "IBKR ticks" if IBKR_ENABLED else "Tradier ticks"
				logger.info(
					"[backtest] %s: filled cached live timesale hole %s → %s with %d %s",
					symbol, _safe_iso(hole_start), _safe_iso(hole_end), len(fetched), source_label,
				)
			for event in sorted(fetched, key=lambda item: (item.ts, item.seq)):
				key = (event.ts, event.last, event.size)
				if key in seen:
					continue
				seen.add(key)
				yield 1, event
			yield from _yield_cached_until(hole_end, include_cutoff=True)

		while idx < len(ordered_cached):
			event = ordered_cached[idx]
			idx += 1
			key = (event.ts, event.last, event.size)
			seen.add(key)
			yield 0, event

	def _fetch_events_for_holes(
		self,
		symbol: str,
		holes: list[tuple[datetime, datetime]],
	) -> list[LiveTimesale]:
		events: list[LiveTimesale] = []
		for hole_start, hole_end in holes:
			if hole_end <= hole_start:
				continue
			try:
				rows = self._fetch_ticks(symbol, hole_start, hole_end)
			except Exception as exc:
				logger.warning(
					"[backtest] %s: historical fetch failed for cache hole %s → %s — %s",
					symbol, _safe_iso(hole_start), _safe_iso(hole_end), exc,
				)
				continue
			events.extend(
				self._events_from_tick_rows(symbol, rows, hole_start, hole_end, seq_start=len(events))
			)
		return events

	@staticmethod
	def _events_from_tick_rows(
		symbol: str,
		rows: list[dict],
		start_utc: datetime,
		end_utc: datetime,
		seq_start: int = 0,
	) -> list[LiveTimesale]:
		events: list[LiveTimesale] = []
		seq = seq_start
		last_ts: Optional[datetime] = None
		for row in rows:
			ts = _to_utc(row["ts"])
			if ts < start_utc or ts > end_utc:
				continue
			if last_ts is not None and ts < last_ts:
				continue
			last_ts = ts
			seq += 1

			px = float(row.get("price", 0.0) or 0.0)
			if px <= 0:
				continue
			bid = float(row.get("bid") or 0.0)
			ask = float(row.get("ask") or 0.0)
			if bid <= 0 or ask <= 0:
				spread = max(px * 0.002, 0.0001)
				bid = px - spread / 2.0
				ask = px + spread / 2.0

			events.append(LiveTimesale(
				event_type="timesale",
				symbol=symbol,
				exchange=str(row.get("exchange", "") or ""),
				last=px,
				bid=bid,
				ask=ask,
				size=max(1, int(row.get("size", 1) or 1)),
				ts=ts,
				seq=seq,
				flag=str(row.get("flag", "") or ""),
				cancel=bool(row.get("cancel", False)),
				correction=bool(row.get("correction", False)),
				session=str(row.get("session", "") or ""),
			))
		return events

	@staticmethod
	def _merge_replay_events(
		cached_events: list[LiveTimesale],
		fetched_events: list[LiveTimesale],
	) -> list[tuple[int, LiveTimesale]]:
		merged: list[tuple[int, LiveTimesale]] = []
		seen: set[tuple[datetime, float, int]] = set()
		for event in sorted(cached_events, key=lambda item: (item.ts, item.seq)):
			key = (event.ts, event.last, event.size)
			seen.add(key)
			merged.append((0, event))
		for event in sorted(fetched_events, key=lambda item: (item.ts, item.seq)):
			key = (event.ts, event.last, event.size)
			if key in seen:
				continue
			seen.add(key)
			merged.append((1, event))
		merged.sort(key=lambda item: (item[1].ts, item[0], item[1].seq))
		return merged

	@staticmethod
	def _merge_near_hod_windows(windows: list[CachedNearHodWindow]) -> list[CachedNearHodWindow]:
		if not windows:
			return []
		ordered = sorted(windows, key=lambda window: (window.start_ts, window.end_ts))
		merged: list[CachedNearHodWindow] = []
		for window in ordered:
			if window.end_ts < window.start_ts:
				continue
			if not merged:
				merged.append(window)
				continue
			prev = merged[-1]
			# Bridge sub-minute holes: one bar marginally outside the near-HOD
			# gate must not split replay coverage around an entry moment.
			if window.start_ts <= prev.end_ts + timedelta(seconds=max(1, BACKTEST_REPLAY_WINDOW_BRIDGE_SECS)):
				merged[-1] = CachedNearHodWindow(
					start_ts=prev.start_ts,
					end_ts=max(prev.end_ts, window.end_ts),
					start_price=prev.start_price,
					start_hod=prev.start_hod,
					start_atr_14=prev.start_atr_14,
					start_volume=prev.start_volume,
				)
			else:
				merged.append(window)
		return merged

	@staticmethod
	def _filter_events_to_windows(
		events: list[LiveTimesale],
		windows: list[CachedNearHodWindow],
	) -> list[LiveTimesale]:
		if not events or not windows:
			return []
		filtered: list[LiveTimesale] = []
		idx = 0
		for event in sorted(events, key=lambda item: (item.ts, item.seq)):
			while idx < len(windows) and event.ts > windows[idx].end_ts:
				idx += 1
			if idx >= len(windows):
				break
			if windows[idx].start_ts <= event.ts <= windows[idx].end_ts:
				filtered.append(event)
		return filtered

	def _cached_timesale_paths(self, day: date) -> list[Path]:
		ymd = day.strftime("%Y%m%d")
		patterns = (
			f"*timesale*{ymd}*.csv",
			f"*timesales*{ymd}*.csv",
			f"*tick*{ymd}*.csv",
			f"*raw*tape*{ymd}*.csv",
		)
		paths: list[Path] = []
		seen: set[Path] = set()
		for pattern in patterns:
			for path in self._log_dir.glob(pattern):
				name = path.name.lower()
				if name.startswith("tape_") or name.startswith("scoring_"):
					continue
				if "sim_trades" in name or "setup_signals" in name:
					continue
				if path not in seen:
					seen.add(path)
					paths.append(path)
		return sorted(paths)

	def _timesale_event_from_cache_row(self, symbol: str, row: pd.Series, seq: int) -> Optional[LiveTimesale]:
		ts_raw = row.get("ts")
		if hasattr(ts_raw, "to_pydatetime"):
			ts = ts_raw.to_pydatetime()
		elif isinstance(ts_raw, datetime):
			ts = ts_raw
		else:
			parsed = pd.to_datetime(ts_raw, utc=True, errors="coerce")
			if pd.isna(parsed):
				return None
			ts = parsed.to_pydatetime()
		ts = _to_utc(ts)

		price = float(row.get("price", 0.0) or 0.0)
		size = int(row.get("size", 0) or 0)
		if price <= 0 or size <= 0:
			return None
		bid = float(row.get("bid", 0.0) or 0.0)
		ask = float(row.get("ask", 0.0) or 0.0)
		if bid <= 0 or ask <= 0:
			spread = max(price * 0.002, 0.0001)
			bid = price - spread / 2.0
			ask = price + spread / 2.0

		return LiveTimesale(
			event_type="timesale",
			symbol=symbol,
			exchange=str(row.get("exchange", "") or ""),
			last=price,
			bid=bid,
			ask=ask,
			size=size,
			ts=ts,
			seq=int(row.get("seq", 0) or seq),
			flag=str(row.get("flag", "") or ""),
			cancel=False,
			correction=False,
			session=str(row.get("session", "") or ""),
		)

	def _load_cached_timesales(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
		windows: list[CachedNearHodWindow] | None = None,
	) -> list[LiveTimesale]:
		canonical = load_timesales_cache(symbol, start_utc, end_utc, self._log_dir)
		if not canonical.empty:
			if windows:
				ts_col = canonical["ts"]
				mask = pd.Series(False, index=canonical.index)
				for window in windows:
					window_start = pd.Timestamp(max(window.start_ts, start_utc))
					window_end = pd.Timestamp(min(window.end_ts, end_utc))
					if window_start.tzinfo is None:
						window_start = window_start.tz_localize("UTC")
					else:
						window_start = window_start.tz_convert("UTC")
					if window_end.tzinfo is None:
						window_end = window_end.tz_localize("UTC")
					else:
						window_end = window_end.tz_convert("UTC")
					mask |= (ts_col >= window_start) & (ts_col <= window_end)
				canonical = canonical.loc[mask].copy()
				if canonical.empty:
					return []
			events: list[LiveTimesale] = []
			for seq, (_, row) in enumerate(canonical.iterrows(), start=1):
				event = self._timesale_event_from_cache_row(symbol, row, seq)
				if event is not None:
					events.append(event)
			events.sort(key=lambda event: (event.ts, event.seq))
			return events

		day = start_utc.astimezone(MarketClock.ET).date()
		events: list[LiveTimesale] = []
		seq = 0
		for path in self._cached_timesale_paths(day):
			try:
				with path.open(newline="") as handle:
					reader = csv.DictReader(handle)
					for row in reader:
						row_symbol = _csv_first(row, "symbol", "sym", "ticker").upper()
						if row_symbol != symbol.upper():
							continue
						event_type = _csv_first(row, "event_type", "type").lower()
						if event_type and event_type not in {"timesale", "trade", "tick"}:
							continue
						ts = _csv_dt(row, "ts", "timestamp", "time", "date")
						if ts is None or ts < start_utc or ts > end_utc:
							continue
						price = _csv_float(row, "last", "price", "last_price")
						if price <= 0:
							continue
						bid = _csv_float(row, "bid", "bid_price")
						ask = _csv_float(row, "ask", "ask_price")
						side = _csv_first(row, "side", "aggressor", "aggressor_side").lower()
						if bid <= 0 or ask <= 0:
							spread = max(price * 0.002, 0.0001)
							if side.startswith("b"):
								bid = price - spread
								ask = price
							elif side.startswith("s"):
								bid = price
								ask = price + spread
							else:
								continue
						seq += 1
						events.append(LiveTimesale(
							event_type="timesale",
							symbol=symbol,
							exchange=_csv_first(row, "exchange", "exch"),
							last=price,
							bid=bid,
							ask=ask,
							size=max(1, _csv_int(row, "size", "volume", "last_size", default=1)),
							ts=ts,
							seq=_csv_int(row, "seq", "sequence", default=seq),
							flag=_csv_first(row, "flag"),
							cancel=_csv_bool(row, "cancel", "cancelled"),
							correction=_csv_bool(row, "correction", "corrected"),
							session=_csv_first(row, "session"),
						))
			except OSError as exc:
				logger.warning("[backtest] %s: cached timesale read failed (%s) — %s", symbol, path, exc)
		events.sort(key=lambda event: (event.ts, event.seq))
		return events

	def _load_cached_chart_ticks(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
	) -> list[LiveTimesale]:
		for timeframe, seconds in (("10s", 10), ("1m", 60), ("2m", 120), ("5m", 300)):
			cached = load_candle_cache(symbol, timeframe, start_utc, end_utc, self._log_dir)
			cached = self._closed_cached_candles(cached, seconds, end_utc)
			if cached.empty:
				continue
			events: list[LiveTimesale] = []
			for seq, (_, row) in enumerate(cached.iterrows(), start=1):
				ts_raw = row.get("ts")
				if hasattr(ts_raw, "to_pydatetime"):
					bar_ts = ts_raw.to_pydatetime()
				elif isinstance(ts_raw, datetime):
					bar_ts = ts_raw
				else:
					parsed = pd.to_datetime(ts_raw, utc=True, errors="coerce")
					if pd.isna(parsed):
						continue
					bar_ts = parsed.to_pydatetime()
				bar_ts = _to_utc(bar_ts)
				event_ts = bar_ts + timedelta(seconds=seconds) - timedelta(microseconds=1)
				if event_ts < start_utc or event_ts > end_utc:
					continue

				price = float(row.get("close", 0.0) or 0.0)
				if price <= 0:
					continue
				size = max(1, int(row.get("volume", 0) or 1))
				spread = max(price * 0.002, 0.0001)
				events.append(LiveTimesale(
					event_type="timesale",
					symbol=symbol,
					exchange="chart_cache",
					last=price,
					bid=price - spread / 2.0,
					ask=price + spread / 2.0,
					size=size,
					ts=event_ts,
					seq=seq,
					flag=f"chart_cache:{timeframe}",
					cancel=False,
					correction=False,
					session=str(row.get("source", "") or ""),
				))
			if events:
				return events
		return []

	@staticmethod
	def _closed_cached_candles(candles: pd.DataFrame, seconds: int, end_utc: datetime) -> pd.DataFrame:
		if candles.empty or "ts" not in candles.columns:
			return candles
		out = candles.copy()
		ts = pd.to_datetime(out["ts"], utc=True, errors="coerce")
		bar_end = ts + pd.to_timedelta(seconds, unit="s") - pd.Timedelta(microseconds=1)
		end_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
		out = out.loc[ts.notna() & (bar_end <= end_ts)].copy()
		return out.reset_index(drop=True)

	def _load_book_snapshots(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
	) -> list[OrderBookSnapshot]:
		"""Rebuild OrderBookSnapshot objects from the captured book cache."""
		df = load_book_cache(symbol, start_utc, end_utc, self._log_dir)
		if df.empty:
			return []
		snapshots: list[OrderBookSnapshot] = []
		for row in df.itertuples(index=False):
			ts = row.ts.to_pydatetime()
			def _levels(pairs: list, side: str) -> list[OrderBookLevel]:
				out: list[OrderBookLevel] = []
				for pos, pair in enumerate(pairs):
					try:
						price, size = float(pair[0]), int(pair[1])
					except (TypeError, ValueError, IndexError):
						continue
					if price <= 0 or size <= 0:
						continue
					out.append(OrderBookLevel(
						symbol=symbol, side=side, position=pos,
						price=price, size=size, market_maker="", ts=ts,
					))
				return out
			snapshots.append(OrderBookSnapshot(
				symbol=symbol,
				bids=_levels(row.bids, "bid"),
				asks=_levels(row.asks, "ask"),
				ts=ts,
			))
		return snapshots

	def _load_cached_seed_market_data(
		self,
		symbol: str,
		stream_start_utc: datetime,
	) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
		seed_end = stream_start_utc - timedelta(microseconds=1)
		seed_start = latest_extended_session_start(stream_start_utc)
		if seed_end < seed_start:
			return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		df_1m = load_candle_cache(symbol, "1m", seed_start, seed_end, self._log_dir)
		df_10s = load_candle_cache(symbol, "10s", seed_start, seed_end, self._log_dir)
		return (
			self._closed_cached_candles(df_1m, 60, seed_end),
			self._closed_cached_candles(df_10s, 10, seed_end),
			load_timesales_cache(symbol, seed_start, seed_end, self._log_dir),
		)

	@staticmethod
	def _seed_scorer_from_cached_market_data(
		scorer: DynamicScorer,
		df_1m_seed: pd.DataFrame,
		df_10s_seed: pd.DataFrame,
		df_timesales_seed: pd.DataFrame,
	) -> None:
		if not hasattr(scorer, "_tf_buf"):
			return
		latest_seed_price = 0.0

		if not df_1m_seed.empty:
			candles_1m = DynamicScorer._df_to_candles(df_1m_seed, "1min")
			if candles_1m:
				cached_atr = BacktestRunner._latest_cached_atr(df_1m_seed)
				scorer._candles = candles_1m
				scorer._candles_session = candles_1m
				BacktestRunner._seed_replay_stream_vwap(scorer, candles_1m)
				latest_seed_price = float(candles_1m[-1].close)
				BacktestRunner._update_replay_hod_atr(scorer, candles_1m, latest_seed_price, update_atr=cached_atr <= 0)
				if cached_atr > 0:
					scorer._last_atr = cached_atr
				if len(candles_1m) >= 3 and BacktestRunner._replay_near_hod(scorer, latest_seed_price):
					scorer._update_price_action_quality("1m", candles_1m)

		if not df_10s_seed.empty:
			candles_10s = DynamicScorer._df_to_candles(df_10s_seed, "10sec")
			if candles_10s:
				scorer._tf_buf.seed("10sec", candles_10s)
				latest_seed_price = float(candles_10s[-1].close)
				BacktestRunner._update_replay_hod_atr(scorer, candles_10s, latest_seed_price, update_atr=False)
				if len(candles_10s) >= 3 and BacktestRunner._replay_near_hod(scorer, latest_seed_price):
					scorer._update_price_action_quality("10s", candles_10s)

		if not df_timesales_seed.empty and hasattr(scorer, "_seed_timesales_from_cache_locked"):
			lock = getattr(scorer, "_lock", None)
			if lock is None:
				scorer._seed_timesales_from_cache_locked()
			else:
				with lock:
					scorer._seed_timesales_from_cache_locked()

	@staticmethod
	def _seed_replay_stream_vwap(scorer: DynamicScorer, candles: list[OHLCVCandle]) -> None:
		if not candles:
			return
		current_session = DynamicScorer._session_key_from_utc(candles[-1].ts)
		cum_pv = 0.0
		cum_vol = 0
		history: list[float] = []
		for candle in candles:
			if DynamicScorer._session_key_from_utc(candle.ts) != current_session:
				continue
			volume = int(candle.volume)
			if volume <= 0:
				continue
			bar_price = DynamicScorer._bar_vwap_price(candle)
			cum_pv += bar_price * volume
			cum_vol += volume
			if cum_vol > 0:
				history.append(cum_pv / cum_vol)
		scorer._vwap_session_key = current_session
		scorer._vwap_cum_pv_closed = cum_pv
		scorer._vwap_cum_vol_closed = cum_vol
		scorer._vwap_closed_history = deque(history[-32:], maxlen=32)
		if cum_vol > 0:
			scorer._stream_vwap = cum_pv / cum_vol
			scorer._last_vwap = scorer._stream_vwap

	@staticmethod
	def _latest_cached_atr(df: pd.DataFrame) -> float:
		if df.empty or "atr_14" not in df.columns:
			return 0.0
		values = pd.to_numeric(df["atr_14"], errors="coerce").dropna()
		if values.empty:
			return 0.0
		value = float(values.iloc[-1])
		return value if value > 0 else 0.0

	@staticmethod
	def _update_replay_hod_atr(
		scorer: DynamicScorer,
		candles: list[OHLCVCandle],
		price: float,
		*,
		update_atr: bool = True,
	) -> None:
		entry = getattr(scorer, "_entry", None)
		if entry is not None:
			rolling_high = max((float(c.high) for c in candles), default=0.0)
			if rolling_high > float(getattr(entry, "daily_high", 0.0) or 0.0):
				entry.daily_high = rolling_high
		if price > 0:
			scorer._last_price = price
		if update_atr:
			atr = _atr_from_candles(candles, 14)
			if atr > 0:
				scorer._last_atr = atr

	@staticmethod
	def _replay_near_hod(scorer: DynamicScorer, price: float) -> bool:
		entry = getattr(scorer, "_entry", None)
		atr = float(getattr(scorer, "_last_atr", 0.0) or 0.0)
		hod = float(getattr(entry, "daily_high", 0.0) or 0.0) if entry is not None else 0.0
		if atr <= 0 or hod <= 0 or price <= 0:
			return False
		return (hod - price) / atr <= ATTENTION_DIST_FROM_HOD_ATR_MAX

	def _load_cached_near_hod_windows(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
	) -> list[CachedNearHodWindow]:
		cached = load_near_hod_window_cache(symbol, start_utc, end_utc, self._log_dir)
		if cached.empty:
			return []
		start_ts = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
		end_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
		covered = cached.loc[(cached["replay_start_ts"] <= start_ts) & (cached["replay_end_ts"] >= end_ts)].copy()
		if covered.empty:
			return []

		windows: list[CachedNearHodWindow] = []
		for _, row in covered.sort_values("start_ts").iterrows():
			window_start = _to_utc(pd.Timestamp(row["start_ts"]).to_pydatetime())
			window_end = _to_utc(pd.Timestamp(row["end_ts"]).to_pydatetime())
			source = str(row.get("source", "") or "")
			if source == "candle_cache":
				window_start = min(window_end, window_start + timedelta(minutes=1) - timedelta(microseconds=1))
				window_end = window_end + timedelta(minutes=1)
			elif source == "candle_cache_close":
				window_end = window_end + timedelta(minutes=1)
			if window_end < start_utc or window_start > end_utc:
				continue
			windows.append(CachedNearHodWindow(
				start_ts=max(window_start, start_utc),
				end_ts=min(window_end, end_utc),
				start_price=float(row.get("start_price", 0.0) or 0.0),
				start_hod=float(row.get("start_hod", 0.0) or 0.0),
				start_atr_14=float(row.get("start_atr_14", 0.0) or 0.0),
				start_volume=int(row.get("start_volume", 0) or 0),
			))
		return windows

	@staticmethod
	def _frame_covers_window(
		df: pd.DataFrame,
		start_utc: datetime,
		end_utc: datetime,
		bar_seconds: int = 60,
	) -> bool:
		"""True if *df* has candles spanning the replay window.

		Tolerance scales with the bar size: the last CLOSED bar of a 5m frame
		legitimately sits up to ~2 bar-widths before the window end, and thin
		premarket tapes open late — so demanding a fixed 60s at both edges
		would refetch identical data forever."""
		if df is None or df.empty or "ts" not in df.columns:
			return False
		ts = pd.to_datetime(df["ts"], utc=True, errors="coerce").dropna()
		if ts.empty:
			return False
		tol = pd.Timedelta(seconds=max(60, 2 * bar_seconds))
		start_ts = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
		end_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
		return ts.min() <= start_ts + tol and ts.max() >= end_ts - tol

	@staticmethod
	def _resample_1min_candles(df_1m: pd.DataFrame, seconds: int) -> pd.DataFrame:
		"""Resample 1-minute OHLCV candles up to a coarser *seconds* timeframe."""
		if df_1m is None or df_1m.empty:
			return pd.DataFrame()
		df = df_1m.copy()
		df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
		df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
		for col in ("open", "high", "low", "close", "volume"):
			if col not in df.columns:
				return pd.DataFrame()
			df[col] = pd.to_numeric(df[col], errors="coerce")
		agg = (
			df.resample(f"{seconds}s", label="left", closed="left")
			.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
			.dropna(subset=["open", "high", "low", "close"])
		)
		return agg.reset_index()

	def _fetch_candle_frame_from_provider(
		self,
		symbol: str,
		cache_tf: str,
		seconds: int,
		start_utc: datetime,
		end_utc: datetime,
	) -> pd.DataFrame:
		"""Fetch candles from the market-data provider (IBKR when enabled,
		else Tradier) for the replay window and prepare a candle-cache frame
		(with ``atr_14``).  10s bars are fetched natively (IBKR only — Tradier
		has no sub-minute bars, so the frame stays empty there); 2m/5m are
		resampled from fetched 1-minute bars.  Used as a fallback when the
		on-disk candle cache lacks premarket coverage.  Returns an empty frame
		on failure."""
		if self._no_tradier or self._skip_pending():
			return pd.DataFrame()
		if seconds == 10:
			bars = self._fetch_native_10sec_window(symbol, start_utc, end_utc)
		else:
			fetch = getattr(self._tradier, "get_candles_1min_window", None)
			if not callable(fetch):
				return pd.DataFrame()
			try:
				df_1m = fetch(symbol, start_utc, end_utc)
			except Exception as exc:
				logger.warning("[backtest] %s: 1m candle fetch failed for %s window — %s", symbol, cache_tf, exc)
				return pd.DataFrame()
			if df_1m is None or df_1m.empty:
				return pd.DataFrame()
			bars = df_1m if seconds <= 60 else self._resample_1min_candles(df_1m, seconds)
		if bars is None or bars.empty:
			return pd.DataFrame()
		prepared = prepare_candle_cache_frame(bars, symbol, cache_tf)
		if prepared.empty:
			return pd.DataFrame()
		closed = self._closed_cached_candles(prepared, seconds, end_utc)
		if not closed.empty:
			# Persist the backfill so subsequent runs load it from the candle
			# cache instead of refetching. SOURCE_POLLED loses ts-dedup to
			# stream/tick bars, so cached live data is never overwritten.
			write_candle_cache(symbol, cache_tf, closed, self._log_dir, source=SOURCE_POLLED)
		return closed

	def _fetch_native_10sec_window(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
	) -> pd.DataFrame:
		"""Native 10-second bars from the provider, chunked to respect IBKR's
		~4h max duration per 10s-bar request.  Tradier's client has no
		``get_candles_10sec_window`` → empty frame (sub-minute unsupported)."""
		fetch = getattr(self._tradier, "get_candles_10sec_window", None)
		if not callable(fetch):
			return pd.DataFrame()
		chunk = timedelta(hours=4)
		frames: list[pd.DataFrame] = []
		cursor = start_utc
		while cursor < end_utc:
			if self._skip_pending():
				return pd.DataFrame()
			chunk_end = min(cursor + chunk, end_utc)
			try:
				df = fetch(symbol, cursor, chunk_end)
			except Exception as exc:
				logger.warning("[backtest] %s: 10s candle fetch failed (%s → %s) — %s", symbol, cursor, chunk_end, exc)
				return pd.DataFrame()
			if df is not None and not df.empty:
				frames.append(df)
			cursor = chunk_end
		if not frames:
			return pd.DataFrame()
		merged = pd.concat(frames, ignore_index=True)
		merged["ts"] = pd.to_datetime(merged["ts"], utc=True, errors="coerce")
		merged = (
			merged.dropna(subset=["ts"])
			.drop_duplicates(subset=["ts"], keep="last")
			.sort_values("ts", kind="stable")
			.reset_index(drop=True)
		)
		return merged

	def _derive_near_hod_windows_from_candle_cache(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
	) -> list[CachedNearHodWindow]:
		candles = load_candle_cache(symbol, "1m", start_utc, end_utc, self._log_dir)
		candles = self._closed_cached_candles(candles, 60, end_utc)
		if not self._frame_covers_window(candles, start_utc, end_utc):
			fetched = self._fetch_candle_frame_from_provider(symbol, "1m", 60, start_utc, end_utc)
			if self._frame_covers_window(fetched, start_utc, end_utc):
				logger.info(
					"[backtest] %s: 1m candle cache missing premarket coverage — fetched from %s",
					symbol, "IBKR" if IBKR_ENABLED else "Tradier",
				)
				candles = fetched
		if candles.empty or "atr_14" not in candles.columns:
			return []
		candles = candles.copy()
		candles["ts"] = pd.to_datetime(candles["ts"], utc=True, errors="coerce")
		candles = candles.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
		if candles.empty:
			return []

		start_ts = pd.Timestamp(start_utc).tz_convert("UTC") if pd.Timestamp(start_utc).tzinfo else pd.Timestamp(start_utc, tz="UTC")
		end_ts = pd.Timestamp(end_utc).tz_convert("UTC") if pd.Timestamp(end_utc).tzinfo else pd.Timestamp(end_utc, tz="UTC")
		first_ts = candles.iloc[0]["ts"]
		last_ts = candles.iloc[-1]["ts"]
		if first_ts > start_ts + pd.Timedelta(seconds=60) or last_ts < end_ts - pd.Timedelta(seconds=60):
			return []

		candles["high"] = pd.to_numeric(candles["high"], errors="coerce").fillna(0.0)
		candles["close"] = pd.to_numeric(candles["close"], errors="coerce").fillna(0.0)
		candles["volume"] = pd.to_numeric(candles["volume"], errors="coerce").fillna(0).astype(int)
		candles["atr_14"] = pd.to_numeric(candles["atr_14"], errors="coerce").fillna(0.0)
		candles["rolling_hod"] = candles["high"].cummax()
		candles["cum_volume"] = candles["volume"].cumsum().astype(int)
		candles["near_hod"] = (
			(candles["atr_14"] > 0)
			& (candles["close"] > 0)
			& ((candles["rolling_hod"] - candles["close"]) / candles["atr_14"] <= ATTENTION_DIST_FROM_HOD_ATR_MAX)
		)

		windows: list[dict[str, Any]] = []
		open_window: dict[str, Any] | None = None
		for _, row in candles.iterrows():
			bar_ts = row["ts"]
			bar_end = min(bar_ts + pd.Timedelta(minutes=1) - pd.Timedelta(microseconds=1), end_ts)
			if open_window is not None:
				open_window["end_ts"] = bar_end.to_pydatetime()
			if bool(row["near_hod"]):
				if open_window is None:
					open_window = {
						"symbol": symbol,
						"replay_start_ts": start_utc,
						"replay_end_ts": end_utc,
						"start_ts": max(bar_end.to_pydatetime(), start_utc),
						"end_ts": bar_end.to_pydatetime(),
						"start_price": float(row["close"]),
						"start_hod": float(row["rolling_hod"]),
						"start_atr_14": float(row["atr_14"]),
						"start_volume": int(row["cum_volume"]),
						"threshold_atrs": ATTENTION_DIST_FROM_HOD_ATR_MAX,
						"source": "candle_cache_next_close",
					}
			elif open_window is not None:
				windows.append(open_window)
				open_window = None
		if open_window is not None:
			windows.append(open_window)

		if windows:
			write_near_hod_window_cache(symbol, windows, self._log_dir)
		return self._load_cached_near_hod_windows(symbol, start_utc, end_utc)

	def _derive_pullback_replay_windows_from_candle_cache(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
		near_hod_windows: list[CachedNearHodWindow],
	) -> tuple[list[CachedNearHodWindow], bool]:
		all_windows: list[CachedNearHodWindow] = []
		found_candle_cache = False

		# Any provider FETCH below is clipped to the detected move start —
		# there is nothing to replay before the move, so pre-move bars are
		# only ever needed by the 1m near-HOD/move detection itself (which
		# has its own full-window path). Cached loads stay full-window (free).
		fetch_start = start_utc
		df_1m_cached = load_candle_cache(symbol, "1m", start_utc, end_utc, self._log_dir)
		if df_1m_cached.empty:
			# Bootstrap: 1m is the cheapest frame and everything else (move
			# start, 2m/5m resampling) derives from it — fetch it first so
			# the 10s fetch below can clip to the move instead of spanning
			# the whole premarket window.
			df_1m_cached = self._fetch_candle_frame_from_provider(symbol, "1m", 60, start_utc, end_utc)
		if not df_1m_cached.empty:
			move = detect_move_start(df_1m_cached)
			if move is not None:
				move_ts = pd.Timestamp(df_1m_cached.iloc[move.start_idx]["ts"]).to_pydatetime()
				if move_ts.tzinfo is None:
					move_ts = move_ts.replace(tzinfo=timezone.utc)
				if start_utc < move_ts < end_utc:
					fetch_start = move_ts

		for cache_tf, detector_tf, seconds in (
			("10s", "10sec", 10),
			("1m", "1min", 60),
			("2m", "2min", 120),
			("5m", "5min", 300),
		):
			cached = load_candle_cache(symbol, cache_tf, start_utc, end_utc, self._log_dir)
			cached = self._closed_cached_candles(cached, seconds, end_utc)
			df = self._prepare_pullback_candle_frame(cached)
			covers_move = self._frame_covers_window(df, fetch_start, end_utc, bar_seconds=seconds)
			if not covers_move:
				fetched_frame = pd.DataFrame()
				if cache_tf in ("2m", "5m") and not df_1m_cached.empty:
					# Resample locally from already-cached 1m — no provider call.
					resampled = self._resample_1min_candles(df_1m_cached, seconds)
					if resampled is not None and not resampled.empty:
						prepared = prepare_candle_cache_frame(resampled, symbol, cache_tf)
						if not prepared.empty:
							fetched_frame = self._closed_cached_candles(prepared, seconds, end_utc)
							if not fetched_frame.empty:
								write_candle_cache(symbol, cache_tf, fetched_frame, self._log_dir, source=SOURCE_POLLED)
								logger.info(
									"[backtest] %s: %s candles resampled from cached 1m (%d bars, no fetch)",
									symbol, cache_tf, len(fetched_frame),
								)
				if fetched_frame.empty:
					fetched_frame = self._fetch_candle_frame_from_provider(
						symbol, cache_tf, seconds, fetch_start, end_utc,
					)
					if not fetched_frame.empty:
						logger.info(
							"[backtest] %s: %s candle cache missing move coverage — fetched %d bars from %s",
							symbol, cache_tf, len(fetched_frame), "IBKR" if IBKR_ENABLED else "Tradier",
						)
				fetched_df = self._prepare_pullback_candle_frame(fetched_frame)
				if not fetched_df.empty:
					df = fetched_df
			if df.empty:
				continue
			found_candle_cache = True
			all_windows.extend(
				self._derive_pullback_windows_for_timeframe(
					symbol,
					df,
					detector_tf,
					seconds,
					start_utc,
					end_utc,
				)
			)

		if not found_candle_cache:
			return [], False
		merged = self._merge_near_hod_windows(all_windows)
		if near_hod_windows:
			merged = self._intersect_replay_windows(merged, self._merge_near_hod_windows(near_hod_windows))
		return self._merge_near_hod_windows(merged), True

	@staticmethod
	def _prepare_pullback_candle_frame(candles: pd.DataFrame) -> pd.DataFrame:
		if candles.empty:
			return pd.DataFrame()
		required = {"ts", "open", "high", "low", "close", "volume"}
		if not required.issubset(candles.columns):
			return pd.DataFrame()
		df = candles.copy()
		df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
		for col in ("open", "high", "low", "close", "volume"):
			df[col] = pd.to_numeric(df[col], errors="coerce")
		if "atr_14" not in df.columns:
			df["atr_14"] = 0.0
		df["atr_14"] = pd.to_numeric(df["atr_14"], errors="coerce").fillna(0.0)
		df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).sort_values("ts").reset_index(drop=True)
		if df.empty:
			return pd.DataFrame()
		df["volume"] = df["volume"].fillna(0).astype(int)
		df["rolling_hod"] = df["high"].cummax()
		df["cum_volume"] = df["volume"].cumsum().astype(int)
		return df

	@staticmethod
	def _bar_end(ts: pd.Timestamp | datetime, seconds: int) -> datetime:
		base = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
		return _to_utc(base) + timedelta(seconds=seconds) - timedelta(microseconds=1)

	@staticmethod
	def _pullback_signal_start_ts(signal, fallback_bar_ts: pd.Timestamp | datetime, seconds: int) -> datetime:
		raw = getattr(signal, "pullback_ts", None) or (fallback_bar_ts.to_pydatetime() if hasattr(fallback_bar_ts, "to_pydatetime") else fallback_bar_ts)
		# Start the window at the pullback bar's OPEN, not its close: the
		# window-entry candle seed only loads bars CLOSED strictly before
		# the window start, so a close-anchored start dropped the pullback
		# candle into a microsecond-wide crack between seed and tick stream
		# — the scorer then never saw the red candle and the setup was
		# undetectable in replay (YXT 2026-08-05 12:11 10s pullback).
		# Replaying the bar's own ticks rebuilds it exactly.
		return _to_utc(raw)

	def _derive_pullback_windows_for_timeframe(
		self,
		symbol: str,
		df: pd.DataFrame,
		detector_tf: str,
		seconds: int,
		start_utc: datetime,
		end_utc: datetime,
	) -> list[CachedNearHodWindow]:
		windows: list[CachedNearHodWindow] = []
		open_window: CachedNearHodWindow | None = None
		for idx in range(3, len(df)):
			prefix = df.iloc[:idx + 1].copy()
			bar = df.iloc[idx]
			bar_end = min(self._bar_end(bar["ts"], seconds), end_utc)
			pending = detect_pullback_pending(prefix, detector_tf)
			resolved = detect_pullback_new_high(prefix, detector_tf)

			if pending or resolved:
				signal = resolved[0] if resolved else pending[0]
				if open_window is None:
					window_start = max(
						self._pullback_signal_start_ts(signal, bar["ts"], seconds),
						start_utc,
					)
					if window_start > end_utc:
						continue
					open_window = CachedNearHodWindow(
						start_ts=window_start,
						end_ts=max(window_start, bar_end),
						start_price=float(bar["close"]),
						start_hod=float(bar["rolling_hod"]),
						start_atr_14=float(bar["atr_14"]),
						start_volume=int(bar["cum_volume"]),
					)
				else:
					open_window = CachedNearHodWindow(
						start_ts=open_window.start_ts,
						end_ts=max(open_window.end_ts, bar_end),
						start_price=open_window.start_price,
						start_hod=open_window.start_hod,
						start_atr_14=open_window.start_atr_14,
						start_volume=open_window.start_volume,
					)
				if resolved:
					# Tail bars past the resolution: the breakout bar closes the
					# pattern, but the entry evaluation runs on the bars right
					# after it — they must stay inside replay coverage.
					tail = timedelta(seconds=seconds * max(1, BACKTEST_PULLBACK_WINDOW_TAIL_BARS))
					open_window = CachedNearHodWindow(
						start_ts=open_window.start_ts,
						end_ts=min(open_window.end_ts + tail, end_utc),
						start_price=open_window.start_price,
						start_hod=open_window.start_hod,
						start_atr_14=open_window.start_atr_14,
						start_volume=open_window.start_volume,
					)
					windows.append(open_window)
					open_window = None
			elif open_window is not None:
				# Same tail on unresolved closes — a late resumption (the
				# pattern expiring one bar before the rip) is exactly the bar
				# the replay must not skip.
				tail = timedelta(seconds=seconds * max(1, BACKTEST_PULLBACK_WINDOW_TAIL_BARS))
				windows.append(CachedNearHodWindow(
					start_ts=open_window.start_ts,
					end_ts=min(max(open_window.end_ts, bar_end) + tail, end_utc),
					start_price=open_window.start_price,
					start_hod=open_window.start_hod,
					start_atr_14=open_window.start_atr_14,
					start_volume=open_window.start_volume,
				))
				open_window = None
		if open_window is not None:
			tail = timedelta(seconds=seconds * max(1, BACKTEST_PULLBACK_WINDOW_TAIL_BARS))
			windows.append(CachedNearHodWindow(
				start_ts=open_window.start_ts,
				end_ts=min(open_window.end_ts + tail, end_utc),
				start_price=open_window.start_price,
				start_hod=open_window.start_hod,
				start_atr_14=open_window.start_atr_14,
				start_volume=open_window.start_volume,
			))
		return windows

	@staticmethod
	def _intersect_replay_windows(
		windows: list[CachedNearHodWindow],
		gates: list[CachedNearHodWindow],
	) -> list[CachedNearHodWindow]:
		if not windows or not gates:
			return []
		out: list[CachedNearHodWindow] = []
		gate_idx = 0
		for window in windows:
			while gate_idx < len(gates) and gates[gate_idx].end_ts < window.start_ts:
				gate_idx += 1
			idx = gate_idx
			while idx < len(gates) and gates[idx].start_ts <= window.end_ts:
				gate = gates[idx]
				start = max(window.start_ts, gate.start_ts)
				end = min(window.end_ts, gate.end_ts)
				if start <= end:
					out.append(CachedNearHodWindow(
						start_ts=start,
						end_ts=end,
						start_price=gate.start_price if gate.start_price > 0 else window.start_price,
						start_hod=gate.start_hod if gate.start_hod > 0 else window.start_hod,
						start_atr_14=gate.start_atr_14 if gate.start_atr_14 > 0 else window.start_atr_14,
						start_volume=gate.start_volume if gate.start_volume > 0 else window.start_volume,
					))
				idx += 1
		return out

	def _seed_replay_window_candle_context(
		self,
		scorer: DynamicScorer,
		symbol: str,
		window_start_utc: datetime,
	) -> None:
		seed_end = window_start_utc - timedelta(microseconds=1)
		seed_start = latest_extended_session_start(window_start_utc)
		if seed_end < seed_start:
			return

		for cache_tf, detector_tf, seconds in (
			("10s", "10sec", 10),
			("1m", "1min", 60),
			("2m", "2min", 120),
			("5m", "5min", 300),
		):
			cached = load_candle_cache(symbol, cache_tf, seed_start, seed_end, self._log_dir)
			cached = self._closed_cached_candles(cached, seconds, seed_end)
			if cached.empty:
				continue
			candles = DynamicScorer._df_to_candles(cached, detector_tf)
			if not candles:
				continue
			scorer._tf_buf.seed(detector_tf, candles)
			if detector_tf == "1min":
				scorer._candles = candles
				cached_atr = self._latest_cached_atr(cached)
				self._update_replay_hod_atr(scorer, candles, float(candles[-1].close), update_atr=cached_atr <= 0)
				if cached_atr > 0:
					scorer._last_atr = cached_atr
			elif detector_tf == "2min":
				scorer._candles_2min = candles
			elif detector_tf == "5min":
				scorer._candles_5min = candles

	@staticmethod
	def _tick_in_near_hod_windows(
		ts: datetime,
		windows: list[CachedNearHodWindow],
		idx: int,
	) -> tuple[bool, int]:
		while idx < len(windows) and ts > windows[idx].end_ts:
			idx += 1
		if idx < len(windows) and windows[idx].start_ts <= ts <= windows[idx].end_ts:
			return True, idx
		return False, idx

	@staticmethod
	def _advance_skipped_replay_tick(scorer: DynamicScorer, tick: LiveTimesale) -> None:
		entry = getattr(scorer, "_entry", None)
		if entry is None:
			return
		price = float(tick.last)
		size = max(0, int(tick.size or 0))
		scorer._last_event_ts = tick.ts
		scorer._last_price = price
		if tick.bid > 0:
			scorer._bid = tick.bid
		if tick.ask > 0:
			scorer._ask = tick.ask
		if price > float(getattr(entry, "daily_high", 0.0) or 0.0):
			entry.daily_high = price

		if size <= 0:
			return
		entry.volume = int(getattr(entry, "volume", 0) or 0) + size
		avg_vol = float(getattr(scorer, "_avg_vol", 0.0) or 0.0)
		if avg_vol > 0:
			entry.rvol_1day = entry.volume / avg_vol

		get_group = getattr(entry, "get_criteria_group", None)
		attn = get_group("ATTENTION") if callable(get_group) else None
		if attn is None:
			return
		prev_close = float(getattr(scorer, "_prev_close", 0.0) or 0.0)
		if prev_close > 0 and price > 0:
			entry.change_pct = (price - prev_close) / prev_close * 100.0
			attn.update("change_pct_min", entry.change_pct >= CHANGE_PCT_MIN, entry.change_pct)
		attn.update("volume_min", entry.volume >= ATTENTION_VOLUME_MIN, float(entry.volume))
		if avg_vol > 0:
			attn.update("rvol_1day_min", entry.rvol_1day >= ATTENTION_RVOL_1DAY_MIN, entry.rvol_1day)

	def _load_cached_tape_snapshots(
		self,
		symbol: str,
		start_utc: datetime,
		end_utc: datetime,
	) -> list[CachedTapeSnapshot]:
		day = start_utc.astimezone(MarketClock.ET).date()
		path = self._log_dir / f"tape_{day.strftime('%Y%m%d')}.csv"
		if not path.exists():
			return []
		snapshots: list[CachedTapeSnapshot] = []
		try:
			with path.open(newline="") as handle:
				reader = csv.DictReader(handle)
				for row in reader:
					if _csv_first(row, "symbol").upper() != symbol.upper():
						continue
					if not _csv_bool(row, "entry_tape_gate"):
						continue
					ts = _csv_dt(row, "ts", "timestamp")
					if ts is None or ts < start_utc or ts > end_utc:
						continue
					price = _csv_float(row, "price")
					if price <= 0:
						continue
					entry_speed = _csv_float(row, "entry_speed")
					buy_ratio = _csv_float(row, "entry_buy_ratio", default=0.5)
					buying_pressure = _csv_int(row, "buying_pressure")
					selling_pressure = _csv_int(row, "selling_pressure")
					if buying_pressure + selling_pressure <= 0 and entry_speed > 0:
						activity = max(1, int(round(entry_speed * PRESSURE_WINDOW_SECS)))
						buying_pressure = int(round(activity * buy_ratio))
						selling_pressure = max(0, activity - buying_pressure)
					snapshots.append(CachedTapeSnapshot(
						symbol=symbol,
						ts=ts,
						price=price,
						entry_speed=entry_speed,
						entry_buy_ratio=buy_ratio,
						pressure_acceleration=_csv_float(row, "pressure_acceleration"),
						pressure_acceleration_ready=_csv_bool(row, "pressure_acceleration_ready"),
						buying_pressure=buying_pressure,
						selling_pressure=selling_pressure,
						buy_vol=_csv_int(row, "buy_vol"),
						sell_vol=_csv_int(row, "sell_vol"),
						avg_print_size=_csv_float(row, "avg_print_size", default=1.0),
						source_path=path,
					))
		except OSError as exc:
			logger.warning("[backtest] %s: cached tape read failed (%s) — %s", symbol, path, exc)
		snapshots.sort(key=lambda snapshot: snapshot.ts)
		return snapshots

	@staticmethod
	def _synthetic_tick_from_snapshot(snapshot: CachedTapeSnapshot, seq: int) -> LiveTimesale:
		price = snapshot.price
		spread = max(price * 0.002, 0.0001)
		size = max(1, int(round(snapshot.avg_print_size or 1.0)))
		if price * size < TS_MIN_NOTIONAL:
			size = max(size, int((TS_MIN_NOTIONAL / max(price, 0.0001)) + 1))
		event = LiveTimesale(
			event_type="timesale",
			symbol=snapshot.symbol,
			exchange="cached-tape",
			last=price,
			bid=price - spread,
			ask=price,
			size=size,
			ts=snapshot.ts,
			seq=seq,
			flag="cached_tape_gate",
			cancel=False,
			correction=False,
			session="cached",
		)
		setattr(event, "_cached_tape_snapshot", snapshot)
		return event

	@staticmethod
	def _spread_timestamps(start: datetime, end: datetime, count: int) -> list[datetime]:
		if count <= 0:
			return []
		if count == 1:
			return [end]
		span = max((end - start).total_seconds(), 0.001)
		step = span / (count - 1)
		return [start + timedelta(seconds=step * i) for i in range(count)]

	@staticmethod
	def _split_cached_pressure(
		snapshot: CachedTapeSnapshot,
		total_override: int | None = None,
	) -> tuple[int, int, int, int]:
		logged_total = max(0, snapshot.buying_pressure + snapshot.selling_pressure)
		total = logged_total if total_override is None else max(0, int(total_override))
		if total <= 0:
			return 0, 0, 0, 0
		# Preserve the logged buy_ratio even when scaling to a different total.
		buy_ratio = (snapshot.buying_pressure / logged_total) if logged_total > 0 else 0.5

		# Use acceleration to determine the older/recent total split.
		# acceleration = (recent_total - older_total) / (recent_total + older_total)
		# with total = recent_total + older_total:
		# → recent_total = total * (1 + accel) / 2
		# → older_total  = total * (1 - accel) / 2
		accel = snapshot.pressure_acceleration if snapshot.pressure_acceleration_ready else 0.0
		accel = max(-1.0, min(1.0, accel))
		older_total = max(1, int(round(total * (1.0 - accel) / 2.0)))
		recent_total = max(0, total - older_total)
		if older_total <= 0:
			buy_total = max(0, min(total, int(round(total * buy_ratio))))
			return 0, 0, buy_total, total - buy_total

		# Distribute buys uniformly by buy_ratio across both halves (we no
		# longer have per-half buy-ratio information in the acceleration metric).
		older_buy = max(0, min(older_total, int(round(older_total * buy_ratio))))
		recent_buy = max(0, min(recent_total, int(round(recent_total * buy_ratio))))

		# Reconcile rounding so total buys matches expected buy_total.
		buy_total = max(0, min(total, int(round(total * buy_ratio))))
		delta = buy_total - (older_buy + recent_buy)
		if delta > 0:
			add_recent = min(delta, recent_total - recent_buy)
			recent_buy += add_recent
			delta -= add_recent
			older_buy += min(delta, older_total - older_buy)
		elif delta < 0:
			remove_recent = min(-delta, recent_buy)
			recent_buy -= remove_recent
			delta += remove_recent
			older_buy -= min(-delta, older_buy)

		older_sell = max(0, older_total - older_buy)
		recent_sell = max(0, recent_total - recent_buy)
		return older_buy, older_sell, recent_buy, recent_sell

	@staticmethod
	def _apply_cached_tape_snapshot(scorer: DynamicScorer, snapshot: CachedTapeSnapshot) -> None:
		ofi = scorer._ofi_tracker
		window = getattr(ofi, "_pressure_window", PRESSURE_WINDOW_SECS)
		activity_window = max(getattr(ofi, "_activity_window", window), window)

		# Rebuild the pressure (signal) window from the logged RATE rather than the
		# absolute logged counts, so trade_speed == entry_speed for whatever signal
		# window is configured at replay time (the counts may have been captured
		# under a different window). The older/recent split still recreates the
		# logged buy_ratio and pressure_acceleration.
		n_press = max(0, int(round(snapshot.entry_speed * window)))
		older_buy, older_sell, recent_buy, recent_sell = BacktestRunner._split_cached_pressure(
			snapshot, total_override=n_press
		)
		midpoint = snapshot.ts - timedelta(seconds=window / 2)
		older_start = snapshot.ts - timedelta(seconds=window) + timedelta(milliseconds=1)
		older_end = midpoint - timedelta(milliseconds=1)
		recent_start = midpoint
		recent_end = snapshot.ts

		buy_times = BacktestRunner._spread_timestamps(older_start, older_end, older_buy)
		buy_times.extend(BacktestRunner._spread_timestamps(recent_start, recent_end, recent_buy))
		sell_times = BacktestRunner._spread_timestamps(older_start, older_end, older_sell)
		sell_times.extend(BacktestRunner._spread_timestamps(recent_start, recent_end, recent_sell))

		# Fill the remainder of the (wider) activity window so activity_count
		# reflects the logged total directional prints — the gate's print-count
		# floor reads this window, decoupled from the short signal window.
		activity_total = max(
			snapshot.buying_pressure + snapshot.selling_pressure,
			int(round(snapshot.entry_speed * activity_window)),
		)
		extra_activity = max(0, activity_total - (len(buy_times) + len(sell_times)))
		activity_start = snapshot.ts - timedelta(seconds=activity_window) + timedelta(milliseconds=1)
		activity_end = older_start - timedelta(milliseconds=1)
		activity_times = BacktestRunner._spread_timestamps(activity_start, activity_end, extra_activity)

		# Baseline reference sits strictly older than the activity window.
		baseline_start = snapshot.ts - timedelta(seconds=activity_window + TAPE_BASELINE_WINDOW_SECS) + timedelta(milliseconds=1)
		baseline_end = snapshot.ts - timedelta(seconds=activity_window) - timedelta(milliseconds=1)
		baseline_times = BacktestRunner._spread_timestamps(
			baseline_start,
			baseline_end,
			max(TAPE_BASELINE_MIN_PRINTS, 1),
		)

		ofi._buy_prints = deque(sorted(buy_times))
		ofi._sell_prints = deque(sorted(sell_times))
		all_directional = sorted(baseline_times + activity_times + buy_times + sell_times)
		ofi._baseline_prints = deque(all_directional)
		ofi._buy_vol = snapshot.buy_vol or int(snapshot.buying_pressure * max(1.0, snapshot.avg_print_size))
		ofi._sell_vol = snapshot.sell_vol or int(snapshot.selling_pressure * max(1.0, snapshot.avg_print_size))

		# Rebuild the EMA accel accumulators from the reconstructed prints
		# (shared helper — reproduces what live ingestion would hold).
		ofi.reseed_ema_from_prints(now=snapshot.ts)

	def _fetch_tick_chunk(self, symbol: str, start_utc: datetime, end_utc: datetime) -> list[dict]:
		if self._no_tradier:
			return []
		rows = self._tradier.get_timesales_tick(symbol, start_utc, end_utc, session_filter="all")
		if rows:
			return rows
		if end_utc - start_utc <= timedelta(minutes=2):
			return []

		mid_utc = start_utc + ((end_utc - start_utc) / 2)
		if mid_utc <= start_utc or mid_utc >= end_utc:
			return []
		return self._fetch_tick_chunk(symbol, start_utc, mid_utc) + self._fetch_tick_chunk(symbol, mid_utc, end_utc)

	def _persist_fetched_ticks(self, symbol: str, rows: list[dict]) -> None:
		"""Persist IBKR historical ticks to the canonical timesales cache so
		subsequent runs replay from disk instead of refetching (tick requests
		are paced and slow). Only IBKR rows are cached — they carry real
		attached NBBO; Tradier historical ticks lack bid/ask fidelity.
		``source="ibkr_hist"`` keeps provenance distinct from live stream capture.
		"""
		if not rows:
			return
		try:
			append_timesales(
				(
					{
						"symbol": symbol,
						"ts": row["ts"],
						"price": row.get("price", 0.0),
						"size": row.get("size", 0),
						"bid": row.get("bid", 0.0),
						"ask": row.get("ask", 0.0),
						"exchange": row.get("exchange", ""),
						"seq": 0,
						"flag": row.get("flag", ""),
						"session": row.get("session", "ibkr"),
						"source": "ibkr_hist",
					}
					for row in rows
					if float(row.get("price", 0.0) or 0.0) > 0 and int(row.get("size", 0) or 0) > 0
				),
				self._log_dir,
			)
		except Exception as exc:
			logger.warning("[backtest] %s: failed to persist fetched ticks — %s", symbol, exc)

	def _fetch_ticks(self, symbol: str, start_utc: datetime, end_utc: datetime) -> list[dict]:
		if self._no_tradier or self._skip_pending():
			return []
		if IBKR_ENABLED:
			fetch = getattr(self._hist, "get_timesales_tick", None)
			if not callable(fetch):
				return []
			# Tick requests are heavily paced by IBKR and paginate at 1000
			# ticks/request — dense premarket windows take many sequential
			# calls. Log BEFORE fetching so long waits are attributable.
			logger.info(
				"[backtest] %s: fetching IBKR historical ticks %s → %s (paced; dense tape takes a while)",
				symbol, _safe_iso(start_utc), _safe_iso(end_utc),
			)
			rows = fetch(symbol, start_utc, end_utc, session_filter="all")
			self._persist_fetched_ticks(symbol, rows)
			return rows
		rows = self._tradier.get_timesales_tick(symbol, start_utc, end_utc, session_filter="all")
		if rows:
			return rows

		# 502 body overflow — split into 30-minute chunks
		logger.info("[backtest] %s: chunked fallback (body overflow)", symbol)
		all_rows: list[dict] = []
		cursor = start_utc
		while cursor < end_utc:
			chunk_end = min(cursor + timedelta(minutes=30), end_utc)
			chunk = self._fetch_tick_chunk(symbol, cursor, chunk_end)
			all_rows.extend(chunk)
			cursor = chunk_end
		return all_rows

	@staticmethod
	def _quote_from_tick(event: LiveTimesale) -> LiveQuote:
		return LiveQuote(
			event_type="quote",
			symbol=event.symbol,
			bid=event.bid,
			bid_size=1,
			bid_exchange=event.exchange,
			bid_ts=event.ts,
			ask=event.ask,
			ask_size=1,
			ask_exchange=event.exchange,
			ask_ts=event.ts,
			synthetic=True,  # placeholder sizes — size-derived reads must skip
		)

	@staticmethod
	def _is_faded_from_hod(entry) -> bool:
		if entry.daily_high <= 0:
			return False
		return entry.price < entry.daily_high * (1.0 - DAILY_HIGH_FADE_PCT)

	# ------------------------------------------------------------------
	# Historical news (called from add_tickers_to_day)
	# ------------------------------------------------------------------
	@staticmethod
	def _apply_historical_news(evaluator, entry, as_of_utc: datetime) -> None:
		"""Replace ``entry``'s news fields with articles fetched bounded
		to *as_of_utc* (the synthetic first-seen time of the backtest).

		Pipeline already filled the entry with *today's* news (because
		the live cascade doesn't know about historical dates).  Only AV
		and Finnhub accept arbitrary date windows, so we re-query those
		two and refit the catalyst variable.  Momo (live feed) and Brave
		(realtime web search) are skipped because they cannot return
		date-bounded historical results.
		"""
		from config import (
			AV_ENABLED, FINNHUB_ENABLED, NEWS_NOISE_THRESHOLD,
			NEWS_BREAKING_MIN, _MarketCalendar,
		)

		symbol = entry.symbol
		lookback = _MarketCalendar.lookback(2)
		_fetch_since = as_of_utc - lookback
		minutes_back = int(lookback.total_seconds() / 60)

		# Try to get company name for relevance filtering (live profile —
		# the field changes rarely, so today's value is acceptable).
		co_name = ""
		us_company = True
		try:
			if FINNHUB_ENABLED:
				profile = evaluator._finnhub.get_profile(symbol)
				if profile:
					co_name = profile.name or ""
					us_company = (profile.country or "").upper() in ("US", "USA", "UNITED STATES")
		except Exception:
			pass

		articles: list = []
		news_count = 0
		avg_sentiment: Optional[float] = None
		news_source = ""

		# 1) AV (preferred — has ticker-scoped entity sentiment)
		if AV_ENABLED:
			try:
				raw = evaluator._av.get_recent_news(symbol, minutes_back=minutes_back, as_of=as_of_utc)
				raw = [a for a in raw if _fetch_since <= a.published_at <= as_of_utc]
				arts, cnt, sent = evaluator._filter_and_score(
					raw, symbol, co_name, use_entity=True, us_company=us_company,
				)
				if cnt > 0:
					articles, news_count, avg_sentiment, news_source = arts, cnt, sent, "alpha_vantage"
			except Exception:
				logger.debug("[backtest] add: AV historical news failed for %s", symbol, exc_info=True)

		# 2) Finnhub fallback
		if not articles and FINNHUB_ENABLED:
			try:
				raw = evaluator._finnhub.get_recent_news(symbol, minutes_back=minutes_back, as_of=as_of_utc)
				arts, cnt, sent = evaluator._filter_and_score(
					raw, symbol, co_name, us_company=us_company,
				)
				if cnt > 0:
					articles, news_count, avg_sentiment, news_source = arts, cnt, sent, "finnhub"
			except Exception:
				logger.debug("[backtest] add: Finnhub historical news failed for %s", symbol, exc_info=True)

		# Stamp results onto the entry.  Even an empty result is
		# meaningful — it tells us the catalyst was absent that day.
		entry.news_articles = articles[:1]
		entry.news_count_last_hour = news_count
		entry.avg_sentiment = avg_sentiment
		entry.news_source = news_source

		# Freshness flags relative to *as_of_utc* — fresh = same trading
		# session, breaking = within NEWS_BREAKING_MIN of the anchor.
		if articles:
			pub = articles[0].published_at
			# Same trading day in ET as the anchor → fresh.
			same_session = pub.astimezone(MarketClock.ET).date() == as_of_utc.astimezone(MarketClock.ET).date()
			entry.news_is_fresh = same_session
			entry.news_is_breaking = pub >= (as_of_utc - timedelta(minutes=NEWS_BREAKING_MIN))
		else:
			entry.news_is_fresh = False
			entry.news_is_breaking = False

		# Keep the catalyst variable in sync.
		if entry.evaluation is not None:
			noise_gate = (
				avg_sentiment is not None
				and -NEWS_NOISE_THRESHOLD <= avg_sentiment <= NEWS_NOISE_THRESHOLD
			)
			entry.evaluation.update_news(avg_sentiment, news_count, news_source, noise_gate)
		logger.info(
			"[backtest] add: %s historical news → src=%s count=%d sent=%s",
			symbol, news_source or "none", news_count,
			f"{avg_sentiment:.2f}" if avg_sentiment is not None else "N/A",
		)

	@staticmethod
	def _migrate_evaluation(entry) -> None:
		"""Rebuild evaluation when cache has the old 6-group schema."""
		ev = getattr(entry, "evaluation", None)
		if ev is None:
			return
		# Current schema has 'stock_eval'; old schema has 'catalyst'/'supply'/etc.
		if hasattr(ev, "stock_eval"):
			return

		# Collect quality values from old groups to seed the new evaluation.
		old_values: dict[str, float] = {}
		for attr in ("catalyst", "supply", "demand", "order_flow", "trend", "price_action"):
			g = getattr(ev, attr, None)
			if g is None:
				continue
			for v in g.variables:
				if v.value is not None:
					old_values[v.name] = v.quality

		# Build fresh 2-group evaluation.
		groups = build_groups(SCORING_GROUPS)
		gmap = {g.name: g for g in groups}
		for g in groups:
			for v in g.variables:
				if v.name in old_values:
					v.quality = old_values[v.name]
					v.value = old_values[v.name]
				else:
					v.quality = 0.5
					v.value = 0.5

		entry.evaluation = CompositeEvaluation(
			stock_eval=gmap["STOCK_EVAL"],
			technical_analysis=gmap["TECHNICAL_ANALYSIS"],
		)

	@staticmethod
	def _merge_session_1min_frame(scorer: DynamicScorer, df_1m: pd.DataFrame) -> None:
		"""Keep ``scorer._df_1min_session`` current during replay.

		Live, the REST poll loop (``_refresh_candles``) continually refreshes
		the full-session 1m frame; the replay has no polling loop, so without
		this merge the frame stays frozen at its pre-stream seed rows and
		every consumer — the MACD / close-vs-VWAP attention gates (which
		prefer ``_df_1min_session`` over the df they are handed), the session
		VWAP context, ``_df_to_candles`` fallbacks — reads stale bars.
		Observed 2026-08-05 (YXT): macd_hist frozen at the seed value across
		the whole session and close-vs-VWAP drifting to -$2.25 while the
		stream VWAP advanced, corrupting every setup-diagnostics line.
		"""
		if df_1m is None or df_1m.empty or not hasattr(scorer, "_df_1min_session"):
			return
		try:
			fresh = DynamicScorer._normalize_candle_df(df_1m)
			existing = scorer._df_1min_session
			if existing is None or existing.empty:
				scorer._df_1min_session = fresh
				return
			merged = pd.concat([existing, fresh], ignore_index=True)
			merged["ts"] = pd.to_datetime(merged["ts"], utc=True)
			scorer._df_1min_session = (
				merged.drop_duplicates(subset=["ts"], keep="last")
				.sort_values("ts")
				.reset_index(drop=True)
			)
		except Exception:
			logger.exception("[backtest] failed to merge session 1m frame for %s", scorer._sym)

	@staticmethod
	def _refresh_ta_from_ticks(scorer: DynamicScorer, ticks: list[tuple[datetime, float, int]]) -> None:
		if len(ticks) < 3:
			return

		df = pd.DataFrame(ticks, columns=["ts", "price", "volume"])
		if df.empty:
			return
		df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
		df = df.set_index("ts")

		def _bars(rule: str, interval: str) -> list[OHLCVCandle]:
			ohlc = df["price"].resample(rule).ohlc().dropna()
			vol = df["volume"].resample(rule).sum()
			rows = ohlc.join(vol.rename("volume"), how="left").dropna()
			out: list[OHLCVCandle] = []
			for ts, row in rows.iterrows():
				out.append(
					OHLCVCandle(
						symbol=scorer._sym,
						ts=_to_utc(ts.to_pydatetime()),
						open=float(row["open"]),
						high=float(row["high"]),
						low=float(row["low"]),
						close=float(row["close"]),
						volume=int(row["volume"]),
						interval=interval,
						vwap=None,
					)
				)
			return out

		def _candles_df(candles: list[OHLCVCandle]) -> pd.DataFrame:
			return pd.DataFrame([
				{
					"symbol": candle.symbol,
					"ts": candle.ts,
					"open": candle.open,
					"high": candle.high,
					"low": candle.low,
					"close": candle.close,
					"volume": candle.volume,
					"vwap": candle.vwap,
				}
				for candle in candles
			])

		try:
			c10s = _bars("10s", "10sec")
			c1m = _bars("1min", "1min")
			c2m = _bars("2min", "2min")
			c5m = _bars("5min", "5min")

			# Mirror the live poll path without lookahead: candle state, ATR,
			# and HOD are derived only from replay ticks seen so far. Technical
			# analysis confidence updates only when the current replay price is
			# within the strict ATR-distance HOD window.
			if c1m:
				scorer._candles = c1m
				scorer._candles_session = c1m
				# Keep the session 1m frame current — the attention criteria
				# (MACD, close-vs-VWAP) read it in preference to the df they
				# are handed, and the replay has no poll loop to refresh it.
				BacktestRunner._merge_session_1min_frame(scorer, _candles_df(c1m))
			if c2m:
				scorer._candles_2min = c2m
			if c5m:
				scorer._candles_5min = c5m
			latest_price = float(ticks[-1][1]) if ticks else 0.0
			if c1m:
				BacktestRunner._update_replay_hod_atr(scorer, c1m, latest_price)
				update_criteria = getattr(scorer, "_update_criteria_and_hidden_seller", None)
				if callable(update_criteria):
					update_criteria(_candles_df(c1m), include_hidden_seller=False)
			if not BacktestRunner._replay_near_hod(scorer, latest_price):
				return

			if len(c10s) >= 3:
				scorer._update_price_action_quality("10s", c10s)
			if len(c1m) >= 3:
				scorer._update_price_action_quality("1m", c1m)
			if len(c2m) >= 3:
				scorer._update_price_action_quality("2m", c2m)
			if len(c5m) >= 3:
				scorer._update_price_action_quality("5m", c5m)
		except Exception:
			logger.exception("[backtest] failed TA refresh from replay ticks for %s", scorer._sym)


def _parse_day(value: str) -> date:
	"""Parse flexible date: 'today', MM-DD, MM/DD, M-D, YYYY-MM-DD, YY-MM-DD, etc."""
	if value.lower() in ("today", "t"):
		return date.today()
	value = value.replace("/", "-")
	parts = value.split("-")
	if len(parts) == 2:
		month, day = int(parts[0]), int(parts[1])
		return date(date.today().year, month, day)
	elif len(parts) == 3:
		year, month, day = parts
		year = int(year)
		if year < 100:
			year += 2000
		return date(year, int(month), int(day))
	raise argparse.ArgumentTypeError(f"Cannot parse date: {value!r}")


def _parse_dates(raw: list[str]) -> tuple[Optional[date], Optional[date], list[str]]:
	"""
	Parse positional date args into (day, None, tickers) or (start, end, tickers).

	Tokens that look like ticker symbols (all-alpha, not a date keyword) are
	extracted and returned separately.

	Accepts:
	  today | t           → single day
	  5/27 | 5-27         → single day
	  5/19 5/27           → range (two tokens)
	  5/19-5/27           → range (single token with slash-dash-slash pattern)
	  today VSME          → single day + ticker filter
	"""
	if not raw:
		return None, None, []

	# Separate ticker tokens from date tokens.
	# A token is a ticker if it's all-alpha and not a date keyword.
	_DATE_KEYWORDS = {"today", "t"}
	date_tokens: list[str] = []
	ticker_tokens: list[str] = []
	for token in raw:
		low = token.lower()
		if low in _DATE_KEYWORDS:
			date_tokens.append(token)
		elif re.fullmatch(r'[A-Za-z]+', token) and len(token) <= 5:
			ticker_tokens.append(token.upper())
		else:
			date_tokens.append(token)

	if not date_tokens:
		return None, None, ticker_tokens
	if len(date_tokens) == 1:
		token = date_tokens[0]
		# Check for range: "5/19-5/27" → split on the dash between two slash-dates
		# Pattern: digits/digits-digits/digits
		m = re.match(r'^(\d{1,2}/\d{1,2})-(\d{1,2}/\d{1,2})$', token)
		if m:
			return _parse_day(m.group(1)), _parse_day(m.group(2)), ticker_tokens
		return _parse_day(token), None, ticker_tokens
	if len(date_tokens) == 2:
		return _parse_day(date_tokens[0]), _parse_day(date_tokens[1]), ticker_tokens
	raise argparse.ArgumentTypeError(f"Too many positional arguments: {raw}")


def _build_arg_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Replay day-by-day scanner backtest using cache + Tradier tick polling.")
	p.add_argument("dates", nargs="*", help="Day or range (e.g. t, 5/27, 5/19 5/27, 5/19-5/27). Ticker symbols can also be appended (e.g. today VSME).")
	p.add_argument("-t", "--ticker", type=str, default="", help="Comma-separated ticker(s) to filter replay (e.g. --ticker VSME,ACMR)")
	p.add_argument("--start", type=_parse_day, help="Start day (inclusive)")
	p.add_argument("--end", type=_parse_day, help="End day (inclusive)")
	p.add_argument("--cache", type=Path, default=_DEFAULT_CACHE_PATH, help="Path to ticker_cache.pkl")
	p.add_argument("--snapshot", action="store_true", help="Save a daily snapshot from the live cache and exit")
	p.add_argument("--no-conf", action="store_true", help="Bypass confidence gates (still calculated/displayed, but not used for trade decisions)")
	p.add_argument(
		"--add",
		type=str,
		default="",
		help=(
			"Comma-separated tickers to evaluate (live APIs) and inject "
			"into the day's snapshot before replaying. Useful for days "
			"the scanner was offline. Example: --add ACMR,QIW,RGTI"
		),
	)
	p.add_argument(
		"--add-at",
		type=str,
		default="",
		help="Time (HH:MM) stamped as first_seen for --add tickers (default: premarket window start).",
	)
	p.add_argument(
		"--momo-gainers",
		action="store_true",
		help="Fetch top gainers from Momo AI and inject any that aren't already in the day's snapshot.",
	)
	p.add_argument(
		"--no-api",
		action="store_true",
		help=(
			"Backtest purely from cached data: no provider fetches (ticks, "
			"candles) and no live-stream connection. Cannot be combined with "
			"--add/--momo-gainers, which need live APIs."
		),
	)
	p.add_argument(
		"--no-tradier",
		action="store_true",
		help="Skip all provider fetches; use only cached data. (--no-api also blocks the live stream.)",
	)
	p.add_argument(
		"--move-start",
		action="store_true",
		help="Replay from detected start-of-move (from cached 1m candles) instead of live discovery time.",
	)
	p.add_argument(
		"--entry-regime",
		choices=("legacy", "armed_band", "posterior"),
		default=None,
		help="Override config.SIM_ENTRY_REGIME for this replay (A/B arms).",
	)
	p.add_argument(
		"--exit-regime",
		choices=("legacy", "thesis_lifecycle"),
		default=None,
		help="Override config.SIM_EXIT_REGIME for this replay (A/B arms).",
	)
	p.add_argument(
		"--tag",
		default="",
		help="Output subdirectory under the backtest log dir (e.g. 'legacy'/'armed') so A/B arms don't overwrite each other.",
	)
	return p


def main() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s %(levelname)s %(message)s",
		datefmt="%H:%M:%S",
	)

	parser = _build_arg_parser()
	args = parser.parse_args()

	# Resolve positional dates and ticker tokens
	pos_day, pos_end, pos_tickers = _parse_dates(args.dates)

	# Merge positional tickers with --ticker flag
	ticker_filter: set[str] | None = None
	all_tickers = pos_tickers + [s for s in re.split(r"[,\s]+", args.ticker) if s]
	if all_tickers:
		ticker_filter = {t.upper() for t in all_tickers}

	# --start/--end flags override positional range
	start = args.start or pos_end and pos_day
	end = args.end or pos_end
	day = pos_day if pos_end is None else None

	# Snapshot mode: save today's cache snapshot and exit
	if args.snapshot:
		cache = TickerCache(path=args.cache)
		snap_day = day or datetime.now(tz=MarketClock.ET).date()
		path = cache.snapshot_day(snap_day)
		print(f"Snapshot saved: {path}")
		return

	if day is None and (start is None or end is None):
		parser.error("Provide a day (e.g. 'make backtest today') or range (e.g. 'make backtest 5/19 5/27')")

	# Apply confidence bypass if requested
	if args.no_conf:
		import config
		config.CONFIDENCE_BYPASS = True
		logger.info("[backtest] Confidence bypass ENABLED — trades ignore confidence gates")

	# Regime overrides — mutate config BEFORE any runner/scorer construction
	# (and before any bundle hash could be computed) so the whole replay,
	# including replay_session.config_hash(), sees the override.
	if args.entry_regime:
		import config
		config.SIM_ENTRY_REGIME = args.entry_regime
		logger.info(f"[backtest] Entry regime override: {args.entry_regime}")
	if args.exit_regime:
		import config
		config.SIM_EXIT_REGIME = args.exit_regime
		logger.info(f"[backtest] Exit regime override: {args.exit_regime}")

	if args.no_api and (args.add or args.momo_gainers):
		parser.error("--no-api cannot be combined with --add/--momo-gainers (both evaluate via live APIs)")
	if args.no_api:
		logger.info("[backtest] --no-api: cached data only — no provider fetches, no stream connection")

	runner = BacktestRunner(
		cache_path=args.cache,
		no_tradier=args.no_tradier,
		no_api=args.no_api,
		move_start=args.move_start,
		output_tag=args.tag,
	)

	# --- Manual ticker injection (--add) ---------------------------------
	if args.add:
		symbols = [s for s in re.split(r"[,\s]+", args.add) if s]
		# Inject into either the single day or each day in the range.
		at_et: Optional[dtime] = None
		if args.add_at:
			try:
				hh, mm = args.add_at.split(":")
				at_et = dtime(int(hh), int(mm))
			except Exception:
				parser.error(f"--add-at must be HH:MM, got: {args.add_at!r}")
		target_days: list[date] = []
		if day is not None:
			target_days = [day]
		elif start is not None and end is not None:
			cur = start
			while cur <= end:
				target_days.append(cur)
				cur += timedelta(days=1)
		else:
			parser.error("--add requires a day or range")
		for d in target_days:
			added = runner.add_tickers_to_day(symbols, d, at_et=at_et)
			logger.info("[backtest] add: %s → %d tickers cached", d.isoformat(), len(added))

	# --- Momo top-gainers injection (--momo-gainers) ---------------------
	if args.momo_gainers:
		from market_data.momo_client import MomoClient

		at_et_momo: Optional[dtime] = None
		if args.add_at:
			try:
				hh, mm = args.add_at.split(":")
				at_et_momo = dtime(int(hh), int(mm))
			except Exception:
				pass  # already validated above if --add was also used

		momo = MomoClient()
		gainers = momo.get_top_gainers()
		gainer_symbols = [g.symbol for g in gainers if g.symbol]
		logger.info("[backtest] momo-gainers: fetched %d top gainers", len(gainer_symbols))

		target_days_momo: list[date] = []
		if day is not None:
			target_days_momo = [day]
		elif start is not None and end is not None:
			cur = start
			while cur <= end:
				target_days_momo.append(cur)
				cur += timedelta(days=1)
		else:
			parser.error("--momo-gainers requires a day or range")

		for d in target_days_momo:
			existing = TickerCache.load_snapshot(d) or {}
			existing_syms = {s.upper() for s in existing}
			new_syms = [s for s in gainer_symbols if s.upper() not in existing_syms]
			if not new_syms:
				logger.info("[backtest] momo-gainers: all gainers already in %s snapshot", d.isoformat())
				continue
			logger.info(
				"[backtest] momo-gainers: injecting %d new tickers for %s (skipped %d existing)",
				len(new_syms), d.isoformat(), len(gainer_symbols) - len(new_syms),
			)
			added = runner.add_tickers_to_day(new_syms, d, at_et=at_et_momo)
			logger.info("[backtest] momo-gainers: %s → %d tickers cached", d.isoformat(), len(added))

	if day is not None:
		results = runner.run_day(day, symbols=ticker_filter)
	else:
		if start > end:
			parser.error("Start must be <= end")
		results = runner.run_range(start, end, symbols=ticker_filter)

	if not results:
		print("No symbols replayed.")
		return

	_print_backtest_summary(results)

	# --- Trade results ---
	trade_logger = SimulationEngine.get_logger()
	_print_trade_results(trade_logger.fills if trade_logger else [])


if __name__ == "__main__":
	main()
# TODO: Backtesting engine for strategy validation
# Planned: historical replay, performance metrics, strategy parameter optimization
