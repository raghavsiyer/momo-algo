"""
display.py  —  Terminal output formatting for the trading system.

Contains all presentation logic:
  - Stock evaluation summary (strengths / warnings)
  - Watchlist table
  - Scanner hit summary lines
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from prompt_toolkit import ANSI as _ANSI_class

from strategy.evaluator import StockEvaluator
from strategy.scoring import StaticInputs, LiveInputs, build_groups
from technical_analysis.resistance import LevelType
from cache import TickerCache
from config import (
    IPO_WINDOW_DAYS,
    FLOAT_MAX_M, FLOAT_DISPLAY_ULTRA_LOW, FLOAT_DISPLAY_LOW,
    RVOL_1DAY_MIN,
    CONFIDENCE_GREEN_COLOR, CONFIDENCE_YELLOW_COLOR,
    sentiment_label, sentiment_is_bullish, sentiment_is_bearish,
    news_freshness_cutoffs,
    NEWS_OLD_DAYS_MAX,
    RESISTANCE_APPROACH_DIST,
    SPLIT_SEVERE_RATIO,
    SECTORS_TIER1, SECTORS_TIER2, SECTORS_TIER3,
    IBKR_L2_MAX_DEPTH_SUBS,
    TAPE_ACCEL_ESTIMATOR,
    _g, _y, _r, _c, _dim, _b, _w,
    MarketClock,
    scanner_ibkr_scan_spec,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-safe print that coordinates with prompt_toolkit's renderer.
# When a prompt_toolkit Application is running, raw print() from background
# threads corrupts the screen.  safe_print() routes output through the app's
# renderer so the toolbar is hidden/restored atomically.
# ---------------------------------------------------------------------------

_pt_app = None  # set by set_app() from main.py once PromptSession starts


def set_app(app) -> None:
    """Store a reference to the running prompt_toolkit Application."""
    global _pt_app
    _pt_app = app


def safe_print(*args, **kwargs) -> None:
    """Print that coordinates with prompt_toolkit when active.

    Uses plain ``print()`` so output flows through the same
    ``StdoutProxy`` channel as logging.  ``patch_stdout(raw=True)``
    already intercepts writes and renders them above the sticky prompt,
    so calling ``app.print_text()`` here would create a second,
    uncoordinated output path that races with the proxy.
    """
    text = " ".join(str(a) for a in args)
    print(text, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_ago(dt) -> str:
    """Return a human-readable 'x seconds/minutes/hours/days ago' string."""
    delta = datetime.now(tz=timezone.utc) - dt
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s} second{'s' if s != 1 else ''} ago"
    m = s // 60
    if m < 60:
        return f"{m} minute{'s' if m != 1 else ''} ago"
    h = m // 60
    if h < 24:
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = h // 24
    return f"{d} day{'s' if d != 1 else ''} ago"


def _clean_source(source: str) -> str:
    """Normalize scanner source for display: 'youtube-news' → 'youtube'."""
    return source.replace("-news", "").replace("-chat", "")


def _color_headline(text: str, sentiment: float | None, stale: bool = False) -> str:
    """Color a headline green/red by sentiment; dim if stale/old."""
    if not text:
        return text
    if stale:
        return _dim(text)
    if sentiment is not None:
        if sentiment_is_bullish(sentiment):
            return _g(text)
        if sentiment_is_bearish(sentiment):
            return _r(text)
    return text


def _inferred_summary_fail_reasons(entry) -> list[str]:
    """Hard eval failures that should still mark cached/manual summaries as failed."""
    failures: list[str] = []
    float_m = float(getattr(entry, "float_shares_m", 0.0) or 0.0)
    if float_m > FLOAT_MAX_M:
        failures.append(f"float: {float_m:.2f}M (> {FLOAT_MAX_M:.0f}M)")
    rvol_1day = float(getattr(entry, "rvol_1day", 0.0) or 0.0)
    if 0 < rvol_1day < RVOL_1DAY_MIN:
        failures.append(f"rvol_1day: {rvol_1day:.1f}x (< {RVOL_1DAY_MIN:.0f}x)")
    return failures


# ---------------------------------------------------------------------------
# Stock summary printer
# ---------------------------------------------------------------------------


def _print_stock_summary(entry, rmap, fail_reasons: list[str] | None = None) -> None:
    """Print evaluation summary with curated strengths / warning signs."""
    if fail_reasons is None:
        fail_reasons = _inferred_summary_fail_reasons(entry)

    ev = entry.evaluation

    # Helper: quality-based lookup
    def _qvar(group_name: str, var_name: str):
        g = ev.group_by_name(group_name) if ev else None
        if g is None:
            return None, None
        v = g.get(var_name)
        if v is None or v.value is None:
            return None, None
        return v.quality, v.value

    strengths: list[tuple[str, str]] = []
    warnings:  list[tuple[str, str]] = []

    # ── News / Catalyst ───────────────────────────────────────────────
    _art = entry.news_articles[0] if entry.news_articles else None
    _ago = _time_ago(_art.published_at) if _art else "recently"
    _headline = f"\"{_art.title}\"" if _art else ""
    _src_tag = f"  ({_clean_source(entry.news_source)})" if entry.news_source else ""

    _session_cut, _breaking_cut, _ = news_freshness_cutoffs()
    _news_is_breaking = entry.news_is_breaking
    _news_is_fresh = entry.news_is_fresh
    if entry.news_articles and getattr(entry.news_articles[0], 'published_at', None):
        _pub = entry.news_articles[0].published_at
        _news_is_breaking = _pub >= _breaking_cut
        _news_is_fresh = _pub >= _session_cut
    _is_old_news = not _news_is_breaking and not _news_is_fresh
    _freshness = (_b("BREAKING") if _news_is_breaking
                  else "FRESH" if _news_is_fresh else _dim("OLD"))

    if entry.avg_sentiment is not None:
        _label = sentiment_label(entry.avg_sentiment)
        _sc = (_g if sentiment_is_bullish(entry.avg_sentiment)
               else _y if not sentiment_is_bearish(entry.avg_sentiment)
               else _r)
        _sent_tag = f"{_label} {_dim('(')}{_sc(_b(f'{entry.avg_sentiment:+.2f}'))}{_dim(')')}"
    else:
        _sent_tag = "Neutral"

    _styled_headline = _color_headline(_headline, entry.avg_sentiment, stale=_is_old_news) if _headline else ""

    nq, _ = _qvar("STOCK_EVAL", "breaking_news")
    if entry.news_count_last_hour == 0:
        warnings.append(("[NO NEWS]  No catalyst in last 2 trading days", ""))
    elif nq is not None and nq >= 0.6:
        strengths.append((f"[{_freshness}]  {_sent_tag} · {_ago}{_src_tag}", _styled_headline))
    else:
        warnings.append((f"[{_freshness}]  {_sent_tag} · {_ago}{_src_tag}", _styled_headline))

    # IPO / reverse split — only when noteworthy
    if entry.days_since_ipo is not None and entry.days_since_ipo <= IPO_WINDOW_DAYS:
        strengths.append((f"Recent IPO — {entry.days_since_ipo} days ago", ""))
    if entry.has_recent_reverse_split:
        if entry.reverse_split_ratio and entry.reverse_split_ratio > SPLIT_SEVERE_RATIO:
            strengths.append((f"Reverse split catalyst {entry.reverse_split_date} "
                              f"({entry.reverse_split_ratio:.0f}-for-1)", ""))
        else:
            strengths.append((f"Reverse split catalyst {entry.reverse_split_date}", ""))

    # ── Technical analysis — setup outcome quality ──────────────────────
    soq, _ = _qvar("TECHNICAL_ANALYSIS", "setup_outcome")
    if soq is not None:
        if soq < 0.3:
            warnings.append(("Setups failing — poor follow-through", ""))
        elif soq >= 0.7:
            strengths.append(("Setups working — clean continuation legs", ""))

    # ── Daily chart ───────────────────────────────────────────────────
    tq, _ = _qvar("STOCK_EVAL", "daily_trend")
    if tq is not None:
        if tq >= 0.7:
            strengths.append(("Tight daily base — clean consolidation", ""))
        elif tq >= 0.5:
            strengths.append(("Daily chart relatively tight", ""))
        elif tq < 0.2:
            warnings.append(("Daily chart wide / volatile — no clean base", ""))
        elif tq < 0.35:
            warnings.append(("Daily bars choppy — weak setup", ""))
        elif tq < 0.5:
            warnings.append(("Daily chart weak — no clean base", ""))

    # ── Resistance ────────────────────────────────────────────────────
    # ── 200 EMA/MA resistance via resistance map ───────────────────────
    ma200_above = [
        l for l in rmap.all_levels()
        if l.active and l.price > entry.price
        and ("200 EMA" in l.source or "200 MA" in l.source)
    ]
    nearest_ma200 = min(ma200_above, key=lambda l: l.price) if ma200_above else None
    if nearest_ma200:
        dist = nearest_ma200.price - entry.price
        pct = dist / entry.price * 100
        if dist <= RESISTANCE_APPROACH_DIST:
            warnings.append((f"Approaching {nearest_ma200.source} at "
                             f"${nearest_ma200.price:.2f} (+{pct:.1f}%)", ""))

    nearest_structural = next(
        (l for l in sorted(rmap.all_levels(), key=lambda l: l.price)
         if l.active and l.price > entry.price
         and l.level_type in (LevelType.STRUCTURE, LevelType.DYNAMIC)
         and "200 EMA" not in l.source and "200 MA" not in l.source),
        None,
    )
    if nearest_structural:
        dist = nearest_structural.price - entry.price
        pct = dist / entry.price * 100
        if dist <= RESISTANCE_APPROACH_DIST:
            warnings.append((f"Near resistance at ${nearest_structural.price:.2f} "
                             f"(+{pct:.1f}%) — {nearest_structural.source}", ""))

    # ── Missing data ────────────────────────────────────────────────
    _missing = getattr(entry, 'missing_data', [])
    if _missing:
        warnings.insert(0, (f"Data not found: {', '.join(_missing)}", "scoring these as neutral (50%)"))


    # ── Build output as a single block ──────────────────────────────
    from config import STATIC_CONFIDENCE_MIN
    _did_not_pass = bool(fail_reasons)
    lines: list[str] = []
    lines.append(f"\n{_c('═' * 60)}")
    _title_suffix = f"  {_r('(DID NOT PASS)')}" if _did_not_pass else ""
    lines.append(f"  {_b(_c(f'{entry.symbol}  EVALUATION SUMMARY'))}{_title_suffix}")
    lines.append(f"{_c('═' * 60)}")

    _group_line = ""
    if ev:
        _group_line = "  ".join(f"{g.name[:3]}={g.score:.0%}" for g in ev.groups)
    if _group_line:
        lines.append(f"  {_dim(_group_line)}")

    # --- Sector / Country line ---
    _sector_raw = entry.sector or ""
    _industry_raw = entry.industry or ""
    if _sector_raw and _industry_raw and _sector_raw.lower() != _industry_raw.lower():
        _sector_str = f"{_sector_raw} / {_industry_raw}"
    else:
        _sector_str = _sector_raw or _industry_raw or "—"
    # Color sector green if it matches our ranked sector keywords
    _sector_lower = _sector_str.lower()
    if any(kw in _sector_lower for kw in SECTORS_TIER1 + SECTORS_TIER2 + SECTORS_TIER3):
        _sector_str = _g(_sector_str)
    _country_str = getattr(entry, 'country', '') or "—"
    if _country_str.upper() in ("CN", "HK", "TW"):
        _country_str = _y(_b(_country_str))
    lines.append(f"  {_dim('Sector:')}     {_sector_str}     {_dim('Country:')} {_country_str}")

    # --- Price + Float line (quality-colored values) ---
    def _conf_color(quality: float | None, text: str) -> str:
        """Color text by quality using confidence thresholds."""
        if quality is None:
            return text
        pct = quality * 100.0
        if pct >= CONFIDENCE_GREEN_COLOR:
            return _g(text)
        if pct >= CONFIDENCE_YELLOW_COLOR:
            return _y(text)
        return _r(text)

    _price_q, _ = _qvar("STOCK_EVAL", "price")
    _float_q, _ = _qvar("STOCK_EVAL", "float_shares")
    _rvol_q, _ = _qvar("STOCK_EVAL", "rvol_1day")

    _float_src = f" ({entry.float_source})" if getattr(entry, 'float_source', '') else ""
    if entry.float_shares_m > 0:
        _float_str = (f"{entry.float_shares_m:.2f}M" if entry.float_shares_m >= 1.0
                      else f"{entry.float_shares_m * 1_000:.0f}K")
    else:
        _float_str = "—"
    _price_display = _conf_color(_price_q, f"${entry.price:.2f}")
    _float_display = _conf_color(_float_q, _float_str)
    lines.append(f"  {_dim('Price:')}      {_b(_price_display)}  ({entry.change_pct:+.1f}%)  "
                 f"{_dim('Float:')} {_float_display}{_dim(_float_src)}")

    # --- Volume + RVOL line (quality-colored values) ---
    if entry.volume > 0:
        _vol_str = (f"{entry.volume / 1_000_000:.2f}M" if entry.volume >= 1_000_000
                    else f"{entry.volume / 1_000:.0f}K")
    else:
        _vol_str = "—"
    _rvol_tod_num = _fmt_num(entry.rvol_1day) if entry.rvol_1day > 0 else ""
    _rvol_tod = (_rvol_tod_num + (" x" if _rvol_tod_num[-1:] in ("K", "M") else "x")) if _rvol_tod_num else "—"
    _rvol_simple_num = _fmt_num(entry.volume / entry.avg_volume) if entry.avg_volume > 0 else ""
    _rvol_simple = (_rvol_simple_num + (" x" if _rvol_simple_num[-1:] in ("K", "M") else "x")) if _rvol_simple_num else "—"
    if _rvol_tod == _rvol_simple:
        _rvol_display = _rvol_tod
    else:
        _rvol_display = f"{_rvol_tod} / {_rvol_simple}"
    # Volume doesn't have its own scoring var; color by rvol quality as proxy
    _vol_display = _conf_color(_rvol_q, _vol_str)
    _rvol_display = _conf_color(_rvol_q, _rvol_display)
    lines.append(f"  {_dim('Vol:')}        {_b(_vol_display)}  "
                 f"{_dim('RVOL:')} {_rvol_display}")

    if strengths:
        lines.append(f"\n  {_g(_b('STRENGTHS'))}")
        for label, sub in strengths:
            lines.append(f"    {_g('✓')}  {label}")
            if sub:
                lines.append(f"        {_dim('·')} {sub}")

    if warnings:
        lines.append(f"\n  {_y(_b('WARNING SIGNS'))}")
        for label, sub in warnings:
            lines.append(f"    {_y('⚠')}  {label}")
            if sub:
                lines.append(f"        {_dim('·')} {sub}")

    if not strengths and not warnings:
        lines.append(f"\n  {_dim('(No significant signals detected)')}")

    _eval = entry.eval_confidence
    _live = entry.live_confidence
    score_color = (_g if _eval >= CONFIDENCE_GREEN_COLOR
                   else _y if _eval >= CONFIDENCE_YELLOW_COLOR
                   else _r)
    lines.append(f"\n  {_dim('Eval Confidence:')}  {score_color(_b(f'{_eval:.1f} / 100'))}")
    if entry.has_live_data:
        live_color = (_g if _live >= CONFIDENCE_GREEN_COLOR
                      else _y if _live >= CONFIDENCE_YELLOW_COLOR
                      else _r)
        lines.append(f"  {_dim('Live Confidence:')}  {live_color(_b(f'{_live:.1f} / 100'))}")
    else:
        lines.append(f"  {_dim('Live Confidence:')}  {_dim('—')}")
    lines.append(f"{_c('═' * 60)}")
    safe_print("\n".join(lines))


# ---------------------------------------------------------------------------
# Watchlist display helper
# ---------------------------------------------------------------------------

def print_watchlist(evaluator: StockEvaluator, faded: set[str] | None = None,
                    scorers: dict | None = None) -> None:
    """Print three-tier watchlist: FOCUSED → WATCHING."""
    _faded = faded or set()
    _scorers = scorers or {}

    # Partition watchlist entries
    focused: list = []
    watching: list = []
    for e in evaluator.watchlist:
        if e.symbol in _faded:
            continue  # faded tickers go to a separate section below
        if e.entry_allowed:
            focused.append(e)
        else:
            watching.append(e)
    faded_entries = [e for e in evaluator.watchlist if e.symbol in _faded]

    total = len(focused) + len(watching) + len(faded_entries)
    if total == 0:
        safe_print("\n  Watchlist is empty.")
        return

    def _setup_tag(e) -> str:
        if e.setup_pattern_name:
            tf = f" ({e.setup_pattern_timeframe})" if e.setup_pattern_timeframe else ""
            return f"{e.setup_pattern_name}{tf}"
        return ""

    def _block_reasons(e) -> str:
        reasons = e.entry_block_reasons
        if not reasons:
            return ""
        return ", ".join(reasons)

    def _confidence_pair(e, color) -> str:
        eval_part = f"E{e.eval_confidence:.0f}"
        live_part = f"L{e.live_confidence:.0f}" if e.has_live_data else "L--"
        return color(f"{eval_part}/{live_part}")

    lines: list[str] = []

    # ── FOCUSED ──
    lines.append(f"\n{'─'*60}")
    lines.append(f"  {_g('▶')} FOCUSED  ({len(focused)} ticker{'s' if len(focused) != 1 else ''})")
    lines.append(f"{'─'*60}")
    if focused:
        for e in sorted(focused, key=lambda x: x.watchlist_sort_score, reverse=True):
            setup = _setup_tag(e)
            setup_str = f"  SETUP: {setup}" if setup else ""
            _fs = (f"{e.float_shares_m:.1f}M" if e.float_shares_m >= 1.0 else f"{e.float_shares_m * 1_000:.0f}K") if e.float_shares_m > 0 else "\u2014"
            lines.append(
                f"  {_b(e.symbol):<6} {_confidence_pair(e, _g):>9}  "
                f"${e.price:>6.2f}  {e.change_pct:>+.1f}%  "
                f"{_dim(_fs)}"
                f"{_g(setup_str) if setup_str else ''}"
            )
    else:
        lines.append(f"  {_dim('(none)')}")

    # ── WATCHING ──
    lines.append(f"\n{'─'*60}")
    lines.append(f"  {_y('◉')} WATCHING  ({len(watching)} ticker{'s' if len(watching) != 1 else ''})")
    lines.append(f"{'─'*60}")
    if watching:
        for e in sorted(watching, key=lambda x: x.watchlist_sort_score, reverse=True):
            setup = _setup_tag(e)
            setup_str = f"  SETUP: {setup}" if setup else ""
            block = _block_reasons(e)
            block_str = f"  {_dim('blocked:')} {_dim(block)}" if block else ""
            _fs = (f"{e.float_shares_m:.1f}M" if e.float_shares_m >= 1.0 else f"{e.float_shares_m * 1_000:.0f}K") if e.float_shares_m > 0 else "\u2014"
            lines.append(
                f"  {_b(e.symbol):<6} {_confidence_pair(e, _y):>9}  "
                f"${e.price:>6.2f}  {e.change_pct:>+.1f}%  "
                f"{_dim(_fs)}"
                f"{setup_str}{block_str}"
            )
    else:
        lines.append(f"  {_dim('(none)')}")

    # ── FADED ──
    if faded_entries:
        lines.append(f"\n{'─'*60}")
        lines.append(f"  {_dim('✗')} FADED  ({len(faded_entries)} ticker{'s' if len(faded_entries) != 1 else ''})  {_dim('— scorer stopped, use refresh to re-evaluate')}")
        lines.append(f"{'─'*60}")
        for e in faded_entries:
            lines.append(
                f"  {_dim(e.symbol):<6} {_confidence_pair(e, _dim):>9}  "
                f"${e.price:>6.2f}  {e.change_pct:>+.1f}%"
            )

    lines.append("")
    safe_print("\n".join(lines))


# ---------------------------------------------------------------------------
# Live weights display
# ---------------------------------------------------------------------------

def _fmt_num(val: float) -> str:
    """Compact number formatting for weight display values."""
    if abs(val) >= 1_000_000:
        return f"{val / 1_000_000:.1f}M"
    if abs(val) >= 1_000:
        return f"{val / 1_000:.0f}K"
    if val == 0:
        return "0"
    if abs(val) < 0.01:
        return f"{val:.4f}"
    if abs(val) < 10:
        return f"{val:.2f}"
    return f"{val:.1f}"


def _quality_color(q: float, text: str) -> str:
    """Color text by quality: green ≥0.75, red <0.25, yellow between."""
    if q >= 0.75:
        return _g(text)
    if q < 0.25:
        return _r(text)
    return _y(text)


def _threshold_str(v) -> str:
    """Build a human-readable threshold hint for a scoring variable."""
    from strategy.scoring import QualityFn
    fn = v.fn
    has_cutoff = _has_hard_cutoff(v)
    if fn == QualityFn.LINEAR_MIN:
        if not has_cutoff:
            return f"lower cleaner; fades by {_fmt_num(v.hi)}+"
        return f"best→0  worst→{_fmt_num(v.hi)}"
    if fn == QualityFn.LINEAR_MAX:
        if not has_cutoff:
            return f"higher stronger; rises near {_fmt_num(v.lo)}+"
        return f"best→{_fmt_num(v.lo)}+"
    if fn == QualityFn.LINEAR_RANGE:
        label = "quality" if not has_cutoff else "range"
        return f"{label} {_fmt_num(v.lo)} → {_fmt_num(v.hi)}"
    if fn in (QualityFn.BELL, QualityFn.BELL_SKEW):
        peak = v.skew_peak if v.skew_peak is not None else (v.lo + v.hi) / 2
        return f"peak {_fmt_num(peak)}  ({_fmt_num(v.lo)}–{_fmt_num(v.hi)})"
    if fn == QualityFn.SATURATING:
        if not has_cutoff:
            return f"quality rises near {_fmt_num(v.lo)}+"
        return f"50%@{_fmt_num(v.lo)}"
    return ""


def _has_hard_cutoff(v) -> bool:
    return getattr(v, "cutoff", None) is not None or getattr(v, "cutoff_lo", None) is not None


def _hard_cutoff_failed(v) -> bool:
    if v.value is None:
        return False
    cutoff = getattr(v, "cutoff", None)
    if cutoff is not None and v.value > cutoff:
        return True
    cutoff_lo = getattr(v, "cutoff_lo", None)
    if cutoff_lo is not None and v.value < cutoff_lo:
        return True
    return False


def _compare_sign(v) -> str:
    """Return a colored </>/≈ indicator only for hard cutoff-style vars."""
    from strategy.scoring import QualityFn
    if v.value is None:
        return _dim("—")
    if not _has_hard_cutoff(v):
        return _dim("·")
    val = v.value
    fn = v.fn
    if fn == QualityFn.LINEAR_MIN:
        return _g("<") if val < v.hi else _r(">")
    if fn == QualityFn.LINEAR_MAX:
        return _g(">") if val >= v.lo else _r("<")
    if fn == QualityFn.SATURATING:
        return _g(">") if val >= v.lo else _r("<")
    if fn in (QualityFn.BELL, QualityFn.BELL_SKEW):
        peak = v.skew_peak if v.skew_peak is not None else (v.lo + v.hi) / 2
        # Close to peak = good
        span = (v.hi - v.lo) or 1.0
        if abs(val - peak) / span < 0.25:
            return _g("≈")
        return _y("~") if v.lo <= val <= v.hi else _r("✗")
    if fn == QualityFn.LINEAR_RANGE:
        mid = (v.lo + v.hi) / 2
        return _g(">") if val >= mid else _r("<")
    return " "


_TA_TF_ORDER = ("10s", "1m", "2m", "5m")
_TA_PER_TF_VARS = ("setup_outcome", "key_candles", "candle_choppiness")
_TA_VAR_LABELS = {
    "setup_outcome":  "setup outcome",
    "key_candles": "Key candles",
    "candle_choppiness": "candle chop",
}

# Box-drawing characters for the TA table.
_BX_TL, _BX_TR, _BX_BL, _BX_BR = "┌", "┐", "└", "┘"
_BX_H, _BX_V = "─", "│"
_BX_T_DOWN, _BX_T_UP, _BX_T_LEFT, _BX_T_RIGHT, _BX_CROSS = "┬", "┴", "┤", "├", "┼"


def _render_ta_table(
    g,
    tf_pa_var_qualities: dict[str, dict[str, float]] | None,
    tf_pa_scores: dict[str, float] | None,
) -> list[str]:
    """Render TECHNICAL_ANALYSIS as a per-TF table for pattern vars plus
    standard rows for the L2/tape vars that are not per-TF."""
    out: list[str] = []
    g_score = g.score
    g_label = _quality_color(g_score, f"{g.name}")
    g_pct = _quality_color(g_score, f"{g_score:.0%}")
    out.append(f"  {g_label:<30s} {g_pct}")

    tfqs = tf_pa_var_qualities or {}
    present_tfs = [tf for tf in _TA_TF_ORDER if tf in tfqs]

    if present_tfs:
        out.append("")
        name_w  = 16          # inner width of name column
        cell_w  = 7           # inner width of each TF cell

        def _v(s: str) -> str:  # vertical bar, dim
            return _dim(_BX_V)

        def _border(left: str, mid: str, right: str) -> str:
            row = left + (_BX_H * (name_w + 2))
            for _ in present_tfs:
                row += mid + (_BX_H * cell_w)
            row += right
            return "  " + _dim(row)

        def _cell_text(text: str, color_fn=None) -> str:
            # Pad raw text to cell_w, then apply color so width is preserved.
            padded = f"{text:^{cell_w}}"
            return color_fn(padded) if color_fn else padded

        def _cell_q(q: float | None) -> str:
            if q is None:
                return _dim(f"{'—':^{cell_w}}")
            pct_txt = f"{q:.0%}"
            return _quality_color(q, f"{pct_txt:^{cell_w}}")

        def _row(name_text: str, cells: list[str], name_color=None) -> str:
            name_padded = f" {name_text:<{name_w}} "
            if name_color:
                name_padded = name_color(name_padded)
            body = _v(_BX_V) + name_padded
            for c in cells:
                body += _v(_BX_V) + c
            body += _v(_BX_V)
            return "  " + body

        # Top border
        out.append(_border(_BX_TL, _BX_T_DOWN, _BX_TR))
        # Header row
        header_cells = [_dim(f"{tf:^{cell_w}}") for tf in present_tfs]
        out.append(_row("", header_cells))
        # Separator
        out.append(_border(_BX_T_RIGHT, _BX_CROSS, _BX_T_LEFT))
        # Per-var rows
        for vname in _TA_PER_TF_VARS:
            label = _TA_VAR_LABELS.get(vname, vname)
            cells = [_cell_q(tfqs.get(tf, {}).get(vname)) for tf in present_tfs]
            out.append(_row(label, cells))
        # Combined row (group score per TF mini-group)
        if tf_pa_scores:
            out.append(_border(_BX_T_RIGHT, _BX_CROSS, _BX_T_LEFT))
            combined_cells = [_cell_q(tf_pa_scores.get(tf)) for tf in present_tfs]
            out.append(_row("combined", combined_cells, name_color=_dim))
        # Bottom border
        out.append(_border(_BX_BL, _BX_T_UP, _BX_BR))

    # Remaining (non-per-TF) vars rendered as normal rows
    non_tf_rows: list[str] = []
    for v in g.variables:
        if present_tfs and v.name in _TA_PER_TF_VARS:
            continue
        if v.value is None:
            val_str = _dim("—")
            bar = _dim("░" * 5)
            q_str = _dim("—")
        else:
            val_str = _quality_color(v.quality, _fmt_num(v.value))
            q = v.quality
            filled = round(q * 5)
            bar = _quality_color(q, "█" * filled + "░" * (5 - filled))
            q_str = _quality_color(q, f"{q:.0%}")
        sign = _compare_sign(v)
        thresh = _dim(_threshold_str(v))
        opt = _dim(" opt") if v.optional and v.value is None else ""
        fail_mark = _r("✗ ") if v.value is not None and v.quality < 0.25 else "  "
        name = f"  {fail_mark}{v.name}"
        non_tf_rows.append(
            f"  {name:<24s} {val_str:>10s} {sign} {thresh:<22s} {bar} {q_str}{opt}"
        )
    if non_tf_rows:
        if present_tfs:
            out.append("")
        out.extend(non_tf_rows)
    return out


def format_live_weights(
    entry,
    price_action_tf_scores: dict[str, float] | None = None,
    scorer_snapshot=None,
    found_ts: float | None = None,
) -> str:
    """Render a full weights block for a ticker's current scoring state.

    Returns a multi-line ANSI string showing every variable with:
      - current raw value (colored by quality)
      - </>/≈ indicator vs threshold
      - threshold description
      - quality bar
    """
    ev = entry.evaluation
    if ev is None:
        return f"  {_dim('No evaluation data')}"

    eval_score = entry.eval_confidence
    live_score = entry.live_confidence
    static_names = set(StaticInputs.__dataclass_fields__.keys())
    live_names = set(LiveInputs.__dataclass_fields__)
    both_names = static_names & live_names

    def _render_section(section: str) -> list[str]:
        lines_local: list[str] = []
        section_has_vars = False
        for g in ev.groups:
            if section == "eval":
                vars_in_section = [
                    v for v in g.variables
                    if v.name in static_names or v.name in both_names
                ]
            else:
                vars_in_section = [
                    v for v in g.variables
                    if v.name in live_names and v.name not in both_names
                ]

            if not vars_in_section:
                continue
            section_has_vars = True

            # Special-case TECHNICAL_ANALYSIS in the live section: render as
            # a variables × timeframes table for the per-TF pattern vars.
            if section == "live" and g.name == "TECHNICAL_ANALYSIS":
                tfqs = getattr(scorer_snapshot, "tf_pa_var_qualities", None) if scorer_snapshot else None
                tfs  = getattr(scorer_snapshot, "tf_pa_scores", None) if scorer_snapshot else price_action_tf_scores
                lines_local.extend(_render_ta_table(g, tfqs, tfs))
                continue

            w_sum = 0.0
            wq_sum = 0.0
            for v in vars_in_section:
                if v.value is None:
                    continue
                q = v.quality
                w_sum += v.weight
                wq_sum += v.weight * q
            g_score = (wq_sum / w_sum) if w_sum > 0.0 else None

            if g_score is not None:
                g_label = _quality_color(g_score, f"{g.name}")
                g_pct = _quality_color(g_score, f"{g_score:.0%}")
            else:
                g_label = _dim(f"{g.name}")
                g_pct = _dim("—")
            lines_local.append(f"  {g_label:<30s} {g_pct}")

            for v in vars_in_section:
                if v.value is None:
                    val_str = _dim("—")
                    bar = _dim("░░░░░")
                    q = 0.0
                    q_str = _dim("—")
                else:
                    val_str = _quality_color(v.quality, _fmt_num(v.value))
                    q = v.quality
                    filled = round(q * 5)
                    bar = _quality_color(q, "█" * filled + "░" * (5 - filled))
                    q_str = _quality_color(q, f'{q:.0%}')

                sign = _compare_sign(v)
                thresh = _dim(_threshold_str(v))

                opt = _dim(" opt") if v.optional and v.value is None else ""
                fail_mark = _r("✗ ") if _hard_cutoff_failed(v) else "  "
                name = f"  {fail_mark}{v.name}"

                lines_local.append(
                    f"  {name:<24s} {val_str:>10s} {sign} {thresh:<22s} {bar} {q_str}{opt}"
                )

        if not section_has_vars:
            lines_local.append(f"  {_dim('(no variables in this section)')}")
        return lines_local

    lines: list[str] = []
    _live_str = (
        _quality_color(live_score / 100, f'{live_score:.1f}%')
        if entry.has_live_data
        else _dim('—')
    )
    lines.append(
        f"  {_b(entry.symbol)}  {_w('Eval Confidence:')} "
        f"{_quality_color(eval_score / 100, f'{eval_score:.1f}%')}  "
        f"{_w('Live Confidence:')} {_live_str}"
    )
    # T&S tape (live, from scorer snapshot)
    if scorer_snapshot is not None:
        bp = scorer_snapshot.buying_pressure
        sp = scorer_snapshot.selling_pressure
        speed = scorer_snapshot.trade_speed
        accel = scorer_snapshot.pressure_acceleration
        accel_ready = scorer_snapshot.pressure_acceleration_ready
        total = bp + sp

        if total > 0:
            buy_pct = int(round(bp / total * 100))
            imbal_color = _g if buy_pct > 55 else (_r if buy_pct < 45 else _dim)
            accel_color = _g if accel > 0.05 else (_r if accel < -0.05 else _dim)
            accel_str = accel_color(f'{accel:+.2f}') if accel_ready else _dim('warming')

            lines.append(
                f"  {_w('T&S:')} {total} prints ({speed:.0f}/s)  "
                f"Accel {accel_str}  "
                f"Imbal {imbal_color(f'{buy_pct}%↑')}"
            )
    lines.append("")
    lines.append(f"  {'─' * 62}")

    lines.append(f"  {_dim('EVAL WEIGHTS (Stage-2/static inputs)')}")
    lines.append("")
    lines.extend(_render_section("eval"))
    lines.append("")
    lines.append(f"  {'─' * 62}")

    lines.append(f"  {_dim('LIVE WEIGHTS (DynamicScorer updates)')}")
    # Start-of-move / found timestamps
    _ts_parts: list[str] = []
    if scorer_snapshot is not None and getattr(scorer_snapshot, "move_start_ts", None) is not None:
        from config import MarketClock
        _local_ts = scorer_snapshot.move_start_ts.astimezone(MarketClock.local_tz())
        _ts_parts.append(f"Start of move: {_c(_local_ts.strftime('%I:%M:%S %p'))}")
    if found_ts is not None:
        from config import MarketClock
        _found_dt = datetime.fromtimestamp(found_ts, tz=timezone.utc).astimezone(MarketClock.local_tz())
        _now_local = datetime.now(MarketClock.local_tz())
        if _found_dt.date() == _now_local.date():
            _found_str = _found_dt.strftime('%I:%M:%S %p')
        else:
            _found_str = _found_dt.strftime('%b %d %I:%M:%S %p')
        _ts_parts.append(f"Found: {_c(_found_str)}")
    if _ts_parts:
        lines.append(f"  {'    '.join(_ts_parts)}")
    lines.append("")
    lines.extend(_render_section("live"))

    # Setup pattern (if available)
    from config import SETUP_PATTERN_GROUP
    setup_grp = next((g for g in ev.groups if g.name == "SETUP_PATTERN"), None)
    if setup_grp is None:
        # Scorer maintains the setup pattern group on the entry itself.
        setup_grp = getattr(entry, "setup_pattern", None)
    if setup_grp is None and scorer_snapshot is not None:
        # Last resort: build an empty group so the section renders.
        try:
            setup_groups = build_groups([SETUP_PATTERN_GROUP])
            setup_grp = setup_groups[0] if setup_groups else None
        except Exception:
            pass

    if setup_grp is not None:
        # Render based on real detector state (set by DynamicScorer):
        #   pending  → "<TF> Pullback   75% — Resolves at $X.XX"
        #   resolved → "<TF> Pullback   75% — Resolved"   (~10s badge)
        #   failed   → "<TF> Pullback   ..% — Failed"     (~10s badge)
        #   idle     → skip section entirely
        _TF_PRETTY = {"10sec": "10 Sec", "1min": "1 Min",
                      "2min": "2 Min", "5min": "5 Min"}
        _BADGE_TTL_SECS = 10.0
        name = getattr(entry, "setup_pattern_name", "") or ""
        tf = getattr(entry, "setup_pattern_timeframe", "") or ""
        level = getattr(entry, "setup_price_level", None)
        resolved_at = getattr(entry, "setup_resolved_at", None)
        failed_at = getattr(entry, "setup_failed_at", None)
        _now = time.monotonic()
        resolved_recent = (
            resolved_at is not None and (_now - resolved_at) < _BADGE_TTL_SECS
        )
        failed_recent = (
            failed_at is not None and (_now - failed_at) < _BADGE_TTL_SECS
        )

        # No active setup → show a minimal "no active setup" line and skip details
        if not (name or resolved_recent or failed_recent):
            lines.append(f"  {'─' * 62}")
            lines.append(f"  {_dim('SETUP PATTERN')}  {_dim('—')} {_dim('no active setup')}")
            setup_grp = None

    if setup_grp is not None:
        lines.append(f"  {'─' * 62}")
        lines.append(f"  {_dim('SETUP PATTERN')}")
        setup_score = setup_grp.score
        setup_pct = _quality_color(setup_score, f"{setup_score:.0%}")

        pretty_tf = _TF_PRETTY.get(tf, tf) if tf else ""
        heading = (f"{pretty_tf} Pullback".strip()) or "Pullback"

        if name == "pullback_detected":
            tail = f"Forming — trigger ${level:.2f}" if level else "Forming…"
            lines.append(f"  {heading:<30s} {_dim('—')}  {_dim('—')} {_y(tail)}")
        elif name == "pullback_pending":
            tail = f"Resolves at ${level:.2f}" if level else "Resolves…"
            lines.append(f"  {heading:<30s} {_dim('—')}  {_dim('—')} {_dim(tail)}")
        elif resolved_recent:
            lines.append(f"  {heading:<30s} {setup_pct}  {_dim('—')} {_g('Resolved')}")
        elif failed_recent:
            lines.append(f"  {heading:<30s} {setup_pct}  {_dim('—')} {_r('Failed')}")
        else:
            setup_label = _quality_color(setup_score, f"{setup_grp.name}")
            lines.append(f"  {setup_label:<30s} {setup_pct}")
        for v in setup_grp.variables:
            if v.value is None:
                val_str = _dim("—")
                bar = _dim("░░░░░")
                q = 0.0
                q_str = _dim("—")
            else:
                val_str = _quality_color(v.quality, _fmt_num(v.value))
                q = v.quality
                filled = round(q * 5)
                bar = _quality_color(q, "█" * filled + "░" * (5 - filled))
                q_str = _quality_color(q, f'{q:.0%}')

            sign = _compare_sign(v)
            thresh = _dim(_threshold_str(v))
            name = f"    {v.name}"
            lines.append(
                f"  {name:<24s} {val_str:>10s} {sign} {thresh:<22s} {bar} {q_str}"
            )

    # ── CRITERIA GATES (ATTENTION + EXECUTION) ───────────────────────
    cgs = getattr(entry, "criteria_groups", None) or []
    if cgs:
        lines.append(f"  {'─' * 62}")
        lines.append(f"  {_dim('CRITERIA GATES (boolean pass/fail)')}")
        for cg in cgs:
            all_pass = cg.required_passing
            g_label = _g(cg.name) if all_pass else _r(cg.name)
            g_status = _g("PASS") if all_pass else _r("FAIL")
            pass_count = sum(1 for c in cg.criteria if c.passing)
            total_count = len(cg.criteria)
            count_str = _dim(f"{pass_count}/{total_count} pass") if total_count else _dim("—")
            lines.append(f"  {g_label:<30s} {g_status}  {count_str}")
            for c in cg.criteria:
                mark = _g("✓") if c.passing else _r("✗")
                val_str = _fmt_num(c.value) if c.value is not None else "—"
                desc_text = c.description or ""
                opt_str = _dim(" (advisory)") if not c.required else ""
                desc = _dim(desc_text) if desc_text else ""
                lines.append(
                    f"    {mark} {c.name:<28s} {val_str:>8s}  {desc}{opt_str}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scanner hit summary
# ---------------------------------------------------------------------------

def watch_status_line(evaluator: StockEvaluator, faded: set[str] | None = None,
                      scorers: dict | None = None) -> str:
    """One-line status for prompt_toolkit bottom toolbar."""
    from config import (MOVING_DIST_HOD_ATR_MAX, MOVING_TAPE_PRESSURE_MIN,
                        HOD_PROXIMITY_PCT_MAX)

    _faded = faded or set()
    _scorers = scorers or {}

    focused = [e for e in evaluator.watchlist if e.symbol not in _faded and e.entry_allowed]

    # "moving" = active tape + near HOD (requires scorer snapshot)
    moving: list[tuple] = []  # (entry, snapshot)
    for e in evaluator.watchlist:
        if e.symbol in _faded or e.entry_allowed:
            continue
        scorer = _scorers.get(e.symbol)
        if scorer is None:
            continue
        try:
            snap_fn = getattr(scorer, "try_snapshot", scorer.snapshot)
            snap = snap_fn()
        except Exception:
            continue
        if snap is None:
            continue
        # Near HOD: within N ATRs OR within HOD_PROXIMITY_PCT_MAX of HOD.
        hod = e.daily_high or 0.0
        hod_dist_dollars = max(hod - snap.last_price, 0.0)
        near_atr = snap.dist_from_hod_atrs <= MOVING_DIST_HOD_ATR_MAX
        near_pct = hod > 0 and hod_dist_dollars <= HOD_PROXIMITY_PCT_MAX * hod
        if ((near_atr or near_pct)
                and snap.buying_pressure + snap.selling_pressure >= MOVING_TAPE_PRESSURE_MIN):
            moving.append((e, snap))

    parts: list[str] = []

    if focused:
        tickers = " ".join(
            f"{_b(e.symbol)} {_g(f'E{e.eval_confidence:.0f}/L{e.live_confidence:.0f}' if e.has_live_data else f'E{e.eval_confidence:.0f}/L--')}"
            for e in sorted(focused, key=lambda x: x.watchlist_sort_score, reverse=True)
        )
        parts.append(f"{_g('▶')} {_dim('FOCUSED:')} {tickers}")

    if moving:
        tickers = " ".join(
            f"{e.symbol} {_y(f'L{e.live_confidence:.0f}')} {_dim(f'(E{e.eval_confidence:.0f})')}"
            for e, _ in sorted(moving, key=lambda x: x[0].watchlist_sort_score, reverse=True)
        )
        parts.append(f"{_y('◉')} {_w('MOVING:')} {tickers}")

    if not parts:
        return f"  {_dim('No active tickers')}"
    return "  " + " │ ".join(parts)


def _print_scanner_hit(sym: str, cache: TickerCache, source: str = "momo", price: float = 0.0, change_pct: float = 0.0, chat_msgs: list[str] | None = None) -> None:
    """Print a recurring-hit summary line for a scanner symbol.

    If *sym* is in the cache, appends float and news freshness info in the
    style of the main evaluation summary.  Otherwise shows live price data.
    """
    source = _clean_source(source)
    cached = cache.get(sym)
    _ts = _dim(MarketClock.now_et().astimezone().strftime("%H:%M:%S"))
    if cached is None:
        extra = f"${price:.2f}  {change_pct:+.1f}%" if price else ""
        lines = [
            f"\n  {_ts}  {_c('⟐')}  {_b(sym)}  {_dim(f'({source})')}  "
            f"{_dim(extra)}"
        ]
        if chat_msgs:
            for msg in chat_msgs[:3]:
                lines.append(f"        {_dim('•')} {_dim(msg)}")
        safe_print("\n".join(lines))
        return
    entry = cached[0]
    float_ok = entry.float_shares_m <= FLOAT_MAX_M
    if not float_ok:
        float_mark = _r('✗')
    elif entry.float_shares_m > 10.0:
        float_mark = _y('✓')
    else:
        float_mark = _g('✓')
    cached_news = cache.get_news(sym) if not entry.news_articles else None
    news_articles = entry.news_articles
    news_count = entry.news_count_last_hour
    news_sentiment = getattr(entry, 'avg_sentiment', None)
    if cached_news:
        news_articles, news_count, news_sentiment, _ = cached_news

    if not news_articles:
        news_mark = _r('✗')
        news_str = "no news"
    else:
        art = news_articles[0]
        pub = getattr(art, 'published_at', None)
        news_is_breaking = entry.news_is_breaking
        news_is_fresh = entry.news_is_fresh
        # Recompute freshness for this render without mutating cached state.
        if pub:
            _sc, _bc, _ = news_freshness_cutoffs()
            news_is_breaking = pub >= _bc
            news_is_fresh = pub >= _sc
            # Check if news exceeds max age threshold
            age_hours = int((datetime.now(tz=timezone.utc) - pub).total_seconds()) // 3600
            age_days = age_hours // 24
            _sent = news_sentiment
            if age_days >= NEWS_OLD_DAYS_MAX:
                news_mark = _dim('—')
                news_str = _dim(f"stale ({age_days}d old)")
            elif news_is_breaking:
                news_mark = _g('✓')
                ago = f" ({_time_ago(pub)})" if pub else ""
                headline = getattr(art, 'title', '')
                hl_text = headline if headline else f"{news_count} article(s)"
                news_str = f"{_b('breaking')}{ago}: {_color_headline(hl_text, _sent)}"
            elif news_is_fresh:
                news_mark = _g('✓')
                ago = f" ({_time_ago(pub)})" if pub else ""
                headline = getattr(art, 'title', '')
                hl_text = headline if headline else f"{news_count} article(s)"
                news_str = f"fresh{ago}: {_color_headline(hl_text, _sent)}"
            else:
                news_mark = _y('✗')
                headline = getattr(art, 'title', '')
                hl_text = headline if headline else f"{news_count} article(s)"
                if pub:
                    h = int((datetime.now(tz=timezone.utc) - pub).total_seconds()) // 3600
                    if h < 1:
                        old_ago = "< 1 hour ago"
                    elif h < 24:
                        old_ago = f"{h} hour{'s' if h != 1 else ''} ago"
                    else:
                        d = h // 24
                        old_ago = f"{d} day{'s' if d != 1 else ''} ago"
                    news_str = f"{_dim('old')} {_dim(f'({old_ago})')}: {_dim(hl_text)}"
                else:
                    news_str = f"{_dim('old')}: {_dim(hl_text)}" if headline else _dim("old")
        else:
            news_mark = _y('✗')
            news_str = "no news"
    _price_val = price if price else (entry.price if hasattr(entry, 'price') and entry.price else None)
    _price_tag = f"  price: ${_price_val:.2f}" if _price_val else ""
    if entry.float_shares_m > 0:
        _fs = f"{entry.float_shares_m:.2f}M" if entry.float_shares_m >= 1.0 else f"{entry.float_shares_m * 1_000:.0f}K"
    else:
        _fs = "—"
    lines = [
        f"\n  {_ts}  {_c('⟐')}  {_b(sym)}  {_dim(f'({source})')}"
        f"{_price_tag}  "
        f"{float_mark} float: {_fs}  "
        f"{news_mark} {news_str}"
    ]
    if chat_msgs:
        for msg in chat_msgs[:3]:
            lines.append(f"        {_dim('•')} {_dim(msg)}")
    safe_print("\n".join(lines))


# ---------------------------------------------------------------------------
# IBKR top-gainers scanner — live view of the ranked list
# ---------------------------------------------------------------------------


def _scan_age(secs: float) -> str:
    """Human-readable age of a scan snapshot, e.g. '3s ago' / '2m ago'."""
    if secs < 1:
        return "just now"
    if secs < 60:
        return f"{secs:.0f}s ago"
    if secs < 3600:
        return f"{secs / 60:.0f}m ago"
    # Hours matter: a snapshot restored from the previous session can be
    # shown overnight, and '14h ago' reads as stale where '840m ago' hides it.
    return f"{secs / 3600:.1f}h ago"


def format_top_gainers(
    rows: list,
    *,
    age_secs: float | None = None,
    cache: TickerCache | None = None,
    scorers: dict | None = None,
    watchlist_symbols=None,
    paused: bool = False,
    interval: int | None = None,
    rejected=None,
    limit: int | None = None,
) -> str:
    """Format the latest IBKR top-gainers scan as an ANSI string.

    Renders the ranked list exactly as the scanner is watching it, annotated
    with what the algo has done with each name so far.  Enrichment is
    read-only and comes purely from memory the REPL already holds — the
    scan rows themselves carry only symbol/rank/exchange, and issuing a
    quote fetch from the render path would open a second IB connection
    against the Gateway's client cap.

    *rejected* names are dropped from the view entirely. These failed a
    structural gate (price band, float, static confidence) that no amount
    of further price action can clear, so the scan keeps ranking them but
    the algo will never act on them — showing them costs rows that the
    still-live names need. Soft failures (rvol, which improves as volume
    builds) are deliberately not in this set and stay visible.

    Filtering happens before *rows* is truncated by the caller's limit, so
    the view still fills to the requested count rather than showing gaps.

    *age_secs* is how long ago the snapshot landed; ``None`` means the
    scanner has not produced one yet.
    """
    _scorers = scorers or {}
    _watching = watchlist_symbols or set()
    _rejected = rejected or set()
    _scanned = len(rows)
    if _rejected:
        rows = [r for r in rows if r.symbol not in _rejected]
    _n_hidden = _scanned - len(rows)
    if limit is not None and limit > 0:
        rows = rows[:limit]

    head = f"  {_b('IBKR Top Gainers')}"
    meta: list[str] = []
    # Label extended-hours mode so a post-close reader knows the ranking is
    # AH change, not the day's frozen RTH leaderboard. Within one poll of a
    # session boundary the label can lead the snapshot — the age shows that.
    _scan_code, _ = scanner_ibkr_scan_spec()
    if "AFTER_HOURS" in _scan_code:
        meta.append(_c("after-hours"))
    if rows:
        meta.append(f"{len(rows)}")
    # Never let a filtered view read as a quiet scan — say what was dropped.
    if _n_hidden:
        meta.append(f"{_n_hidden} rejected hidden")
    if age_secs is not None:
        meta.append(_scan_age(age_secs))
    if interval:
        meta.append(f"{interval}s poll")
    if paused:
        meta.append(_y("paused"))
    if meta:
        head += f"  {_dim('·')}  {_dim('  ·  '.join(meta))}"
    lines = [head, f"  {'─' * 52}"]

    if age_secs is None:
        lines.append(f"  {_dim('waiting for first scan...')}")
        return "\n".join(lines)
    if not rows:
        lines.append(_dim(
            f"  all {_n_hidden} scanned name{'s' if _n_hidden != 1 else ''} hard-rejected"
            if _n_hidden else
            "  no symbols currently clearing the scan filters"
        ))
        return "\n".join(lines)

    for i, row in enumerate(rows, 1):
        sym = row.symbol
        entry = None
        if cache is not None:
            cached = cache.get(sym)
            if cached:
                entry = cached[0]

        # Live tape price beats the evaluation snapshot.
        live_price = None
        scorer = _scorers.get(sym)
        if scorer is not None:
            # Prefer the non-blocking snapshot: this renders on the prompt
            # event loop, and the scorer thread holds that lock on every tick.
            # Resolved lazily — a `getattr(x, "try_snapshot", x.snapshot)`
            # default is evaluated eagerly and raises on scorers with only one.
            snap_fn = getattr(scorer, "try_snapshot", None) or getattr(scorer, "snapshot", None)
            try:
                snap = snap_fn() if snap_fn is not None else None
                if snap is not None and getattr(snap, "last_price", 0) > 0:
                    live_price = float(snap.last_price)
            except Exception:
                pass

        eval_price = float(getattr(entry, "price", 0.0) or 0.0) if entry is not None else 0.0
        eval_chg = float(getattr(entry, "change_pct", 0.0) or 0.0) if entry is not None else 0.0
        price = live_price or eval_price or None
        # entry.price and entry.change_pct come from the same quote, so the
        # previous close they imply is exact — re-basing a live tape price on
        # it gives a live %, rather than pairing a fresh price with the stale
        # % from evaluation time.
        chg = eval_chg
        if live_price and eval_price > 0 and eval_chg > -100.0:
            prev_close = eval_price / (1.0 + eval_chg / 100.0)
            if prev_close > 0:
                chg = (live_price - prev_close) / prev_close * 100.0

        price_s = f"${price:.2f}" if price else "—"
        if chg:
            chg_plain = f"{chg:+.1f}%"
            chg_s = " " * max(0, 7 - len(chg_plain)) + (_g if chg > 0 else _r)(chg_plain)
        else:
            chg_s = " " * 6 + _dim("—")

        if sym in _watching:
            tag = _c("watching")
        elif scorer is not None:
            tag = _g("live")
        elif entry is not None:
            tag = _dim("cached")
        else:
            tag = ""

        lines.append(
            f"  {_dim(f'{i:>2}.')} {_b(f'{sym:<6}')} {price_s:>8}  {chg_s}  {tag}".rstrip()
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live T&S tape  (`<TICKER> tape`) and order book / OFI  (`<TICKER> ofi`)
# ---------------------------------------------------------------------------
# Both render from a view object the scorer assembled under its lock — these
# functions never touch the scorer or the stream, so they cannot block the
# prompt event loop or trigger a fetch.

_FLOW_RULE_WIDTH = 56


def _flow_head(title: str, meta: list[str]) -> list[str]:
    """Shared header + rule for the tape / book blocks.

    Each chip is dimmed on its own rather than dimming the joined string:
    the colour helpers close with a full reset, so an already-coloured chip
    (the L2/L1 source tag) would otherwise strip the dim off everything
    following it.
    """
    head = f"  {_b(title)}"
    if meta:
        chips = [m if "\033" in m else _dim(m) for m in meta]
        head += f"  {_dim('·')}  " + _dim("  ·  ").join(chips)
    return [head, f"  {'─' * _FLOW_RULE_WIDTH}"]


def _fmt_clock(ts: Optional[datetime]) -> str:
    """Local wall-clock with milliseconds — tape prints land sub-second."""
    if ts is None:
        return "--:--:--.---"
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    local = ts.astimezone()
    return f"{local:%H:%M:%S}.{local.microsecond // 1000:03d}"


def _fmt_age(secs: Optional[float]) -> str:
    """Compact age for a feed timestamp."""
    if secs is None:
        return "—"
    if secs < 1:
        return f"{secs * 1000:.0f}ms"
    if secs < 60:
        return f"{secs:.1f}s"
    return f"{secs / 60:.0f}m"


def _gate(label: str, value: str, ok: bool) -> str:
    """One entry-gate chip: green ✓ when the gate passes, red ✗ when it blocks."""
    mark = _g("✓") if ok else _r("✗")
    return f"{_dim(label)} {value} {mark}"


def _size_bar(size: int, peak: int, width: int = 8) -> str:
    """Proportional depth bar, scaled to the largest level on display."""
    if peak <= 0 or size <= 0:
        return ""
    filled = max(1, int(round(size / peak * width)))
    return "▮" * min(filled, width)


def format_tape(view, *, limit: int) -> str:
    """Format the live T&S tape for `view.symbol` as an ANSI string.

    Newest print first, so the row a trader is watching stays pinned to the
    top of the block instead of scrolling.  Prints are green when the buyer
    lifted the offer and red when the seller hit the bid — the same
    classification every tape var below is computed from, taken straight
    from the tracker rather than re-derived here.

    Only directional prints are shown: ``on_timesale`` discards mid/neutral
    ones, so they exist in no window the algo reads.
    """
    if view is None:
        return f"  {_dim('tape — refreshing...')}"

    g = view.gates
    meta = [f"{len(view.prints)}/{view.retained} in {view.window_secs}s"]
    if view.prints_seen:
        meta.append(f"{view.prints_seen:,} seen")
    if view.last_price > 0:
        meta.append(f"${view.last_price:.2f}")
    if view.bid > 0 and view.ask > 0:
        meta.append(f"{view.bid:.2f}×{view.ask:.2f}")
    lines = _flow_head(f"{view.symbol} Tape", meta)

    if not view.prints:
        lines.append(f"  {_dim('no directional prints in the window — tape is idle or not streaming')}")
        return "\n".join(lines)

    # limit <= 0 means "everything streamed" — every print in the rolling
    # window, however many that is.
    shown = view.prints if limit <= 0 else view.prints[:limit]
    for t in shown:
        buy = t.side == "buy"
        paint = _g if buy else _r
        lines.append(
            f"   {_dim(_fmt_clock(t.ts))}  {paint('B' if buy else 'S')}  "
            f"{paint(f'{t.price:>9.4f}')}  {paint(f'{t.size:>7,}')}"
        )
    # Fixed-height block for explicit counts: the rolling window grows and
    # trims print by print, and a block whose height changes every frame
    # reads as flicker in the toolbar — pad the missing rows instead. In
    # follow-the-window mode the height tracks the window by design.
    if limit > 0:
        lines.extend([""] * (limit - len(shown)))

    lines.append(f"  {'─' * _FLOW_RULE_WIDTH}")

    verdict = _g("tape confirms") if g.all_pass else _dim("tape blocks entry")
    lines.append(
        "   " + "  ".join([
            _gate("speed", f"{g.speed:.1f}/s", g.speed_ok),
            _gate("act", f"{g.activity}", g.activity_ok),
            _gate("ratio", f"{g.buy_ratio:.2f}", g.ratio_ok),
            _gate("flow", f"{g.vol_imbalance:+.2f}", g.vol_flow_ok),
            _gate("accel z", f"{g.accel_z:+.2f}" if g.accel_ready else "—", g.accel_ok),
        ]) + f"   {verdict}"
    )
    lines.append(
        f"   {_dim('speed')}    ema {view.tape_speed_ema:.1f}/s   "
        f"boxcar {view.trade_speed:.1f}/s   base {view.baseline_speed:.1f}/s   "
        f"ratio {view.speed_ratio:.1f}x"
    )
    lines.append(
        f"   {_dim('ratio')}    ema {view.buy_ratio_ema:.2f}   "
        f"boxcar {view.buy_ratio:.2f}   "
        f"{_g(f'{view.buying_pressure}↑')}/{_r(f'{view.selling_pressure}↓')}   "
        f"density {view.density_imbalance:+.2f}"
    )
    _imb = view.trade_imbalance
    lines.append(
        f"   {_dim('volume')}   {_g(f'{view.buy_volume:,}')} buy   "
        f"{_r(f'{view.sell_volume:,}')} sell   "
        f"imb {(_g if _imb > 0 else (_r if _imb < 0 else _dim))(f'{_imb:+.2f}')}   "
        f"avg print {view.avg_print_size:,.0f}"
    )
    # accel_raw/accel_z follow TAPE_ACCEL_ESTIMATOR; n_eff and phi are
    # diagnostics of the EMA kernel specifically, so they're only meaningful
    # (and only shown) when that estimator is the one driving the gate.
    _accel_s = f"{view.accel_raw:+.2f}" if view.accel_ready else "warming"
    _accel_line = (
        f"   {_dim('accel')}    {_accel_s}   z {view.accel_z:+.2f}   "
        f"{_dim(TAPE_ACCEL_ESTIMATOR)}"
    )
    if TAPE_ACCEL_ESTIMATOR == "ema":
        _accel_line += f"   n_eff {view.accel_n_eff:.0f}   phi {view.dispersion_phi:.2f}"
    lines.append(_accel_line)
    if view.hidden_seller > 0:
        lines.append(f"   {_dim('hidden')}   seller {_y(f'{view.hidden_seller:.2f}')}")
    return "\n".join(lines)


def format_order_book(view, *, limit: int) -> str:
    """Format the live order book + OFI reads for `view.symbol`.

    Renders the L2 ladder the OFI tracker last diffed (bids green on the
    left, asks red on the right, best price innermost) — the exact book
    state the reported OFI was computed from, not a re-fetch.  Depth lines
    are a scarce IBKR resource, so when no L2 feed is attached this falls
    back to the L1 NBBO and says so; OFI degrades the same way, from
    level-weighted deep OFI to L1 OFI.
    """
    if view is None:
        return f"  {_dim('book — refreshing...')}"

    now = view.last_event_ts
    book_age = None
    if view.last_book_ts is not None and now is not None:
        book_age = max(0.0, (now - view.last_book_ts).total_seconds())

    deep = view.depth_active and bool(view.bids or view.asks)
    if deep:
        source = _g("L2 deep")
    elif view.last_book_ts is not None:
        source = _y("L2 stale")
    else:
        source = _dim("L1 only")

    meta = [source]
    if book_age is not None:
        meta.append(_fmt_age(book_age))
    if view.last_price > 0:
        meta.append(f"${view.last_price:.2f}")
    if view.l1_bid > 0 and view.l1_ask > 0:
        spread = view.l1_ask - view.l1_bid
        mid = (view.l1_ask + view.l1_bid) / 2.0
        meta.append(f"spread ${spread:.3f} ({spread / mid * 100:.2f}%)" if mid > 0 else f"spread ${spread:.3f}")
    lines = _flow_head(f"{view.symbol} Order Book", meta)

    if deep:
        # limit <= 0 means "everything streamed" — the full ladder IBKR
        # delivers on the depth line (IBKR_L2_ROWS per side, pre-aggregation).
        bids = list(view.bids) if limit <= 0 else list(view.bids[:limit])
        asks = list(view.asks) if limit <= 0 else list(view.asks[:limit])
        peak = max([s for _, s in bids + asks] or [0])
        header = f"{'bid':>9} {'size':>8}     {'ask':<9} {'size':<8}"
        lines.append(f"   {' ' * 8} {_dim(header)}")
        for i in range(max(len(bids), len(asks))):
            if i < len(bids):
                bp, bs = bids[i]
                # Pad the bar before colouring — ANSI escapes have no width,
                # so rjust() on a coloured string mis-aligns the column.
                left = (
                    f"{_dim(_size_bar(bs, peak).rjust(8))} "
                    f"{_g(f'{bp:>9.4f}')} {_g(f'{bs:>8,}')}"
                )
            else:
                left = " " * 27
            if i < len(asks):
                ap, asz = asks[i]
                right = f"{_r(f'{ap:<9.4f}')} {_r(f'{asz:<8,}')} {_dim(_size_bar(asz, peak))}"
            else:
                right = ""
            lines.append(f"   {left}  {_dim('│')}  {right}".rstrip())
    else:
        if view.l1_bid > 0 or view.l1_ask > 0:
            lines.append(
                f"   {_g(f'{view.l1_bid:>9.4f}')} {_g(f'{view.l1_bid_size:>8,}')}"
                f"  {_dim('│')}  {_r(f'{view.l1_ask:<9.4f}')} {_r(f'{view.l1_ask_size:<8,}')}"
            )
        else:
            lines.append(f"  {_dim('no quote yet')}")
        if view.last_book_ts is None:
            lines.append(
                f"  {_dim(f'L2 depth not attached (IBKR grants {IBKR_L2_MAX_DEPTH_SUBS} lines, pin-priority) — OFI is L1')}"
            )

    lines.append(f"  {'─' * _FLOW_RULE_WIDTH}")

    _ofi = view.ofi_rolling
    _paint = _g if _ofi > 0 else (_r if _ofi < 0 else _dim)
    lines.append(
        f"   {_dim('OFI')}      {_paint(f'{_ofi:+,} sh')}   {_paint(f'{view.ofi_rate:+,.0f}/s')}   "
        f"gate {view.ofi_norm_gate:+.2f}   "
        f"{_dim(f'norm {view.ofi_normalized:+.2f}')}   "
        f"{_dim(f'{view.ofi_samples} samples / {view.ofi_window_secs}s')}"
    )
    _lean = view.book_lean
    lines.append(
        f"   {_dim('lean')}     {(_g if _lean > 0 else (_r if _lean < 0 else _dim))(f'{_lean:+.2f}')} ema   "
        f"{view.book_lean_raw:+.2f} raw   chop {view.ofi_chop:.2f}   "
        f"intensity {view.ofi_intensity:,.0f}"
    )
    lines.append(
        f"   {_dim('depth')}    avg {view.book_depth_avg:,.0f} sh   "
        f"{_dim(f'{view.quote_count:,} quotes')}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Levels printer — pretty terminal output for resistance / support map
# ---------------------------------------------------------------------------


_LEVEL_TYPE_TAG = {
    LevelType.STATIC:    "static",
    LevelType.STRUCTURE: "structure",
    LevelType.DYNAMIC:   "dynamic",
    LevelType.TRENDLINE: "trendline",
}


def _strength_bar(strength: float, width: int = 5) -> str:
    """Render a 0..1 strength as a ▮▮▮▯▯ bar."""
    s = max(0.0, min(1.0, float(strength)))
    filled = int(round(s * width))
    return "▮" * filled + _dim("▯" * (width - filled))


def format_levels(sym: str, entry, rmap, *, current_price: float | None = None) -> str:
    """
    Format the resistance/support level map for `sym` as an ANSI string.

    Levels at the same rounded price are merged in display (so a topping-tail
    sitting at the daily high shows once with both labels). Levels above the
    current price are shown as resistance, levels below as support. Each row
    shows distance ($ and %), source label(s), type, and strength.
    """
    if rmap is None:
        return f"  {_y('⚠')}  {sym}: no resistance map available"

    price = current_price
    if price is None or price <= 0:
        # Best-effort live price via scorer snapshot, else fall back to entry.price
        try:
            scorer = getattr(entry, "scorer", None)
            if scorer is not None:
                snap = scorer.snapshot()
                if snap and getattr(snap, "last_price", 0) > 0:
                    price = snap.last_price
        except Exception:
            pass
    if price is None or price <= 0:
        price = float(getattr(entry, "price", 0.0) or 0.0)

    def _show_level(lv) -> bool:
        # Static dollar rails are useful internally, but too noisy for `-l`.
        return lv.level_type != LevelType.STATIC

    levels = [l for l in rmap.all_levels(active_only=True) if _show_level(l)]
    broken = [
        l for l in rmap.all_levels(active_only=False)
        if not l.active and _show_level(l)
    ]

    if not levels and not broken:
        return f"  {_y('⚠')}  {sym}: no levels in map"

    # Merge same-price levels for display
    def _merge(rows):
        bucket: dict[float, dict] = {}
        for lv in rows:
            key = round(float(lv.price), 2)
            entry_d = bucket.setdefault(key, {
                "price": key,
                "sources": [],
                "types": set(),
                "strength": 0.0,
                "formed_at": None,
            })
            if lv.source and lv.source not in entry_d["sources"]:
                entry_d["sources"].append(lv.source)
            entry_d["types"].add(_LEVEL_TYPE_TAG.get(lv.level_type, "?"))
            if lv.strength > entry_d["strength"]:
                entry_d["strength"] = float(lv.strength)
            fa = getattr(lv, "formed_at", None)
            if fa is not None and (entry_d["formed_at"] is None or fa > entry_d["formed_at"]):
                entry_d["formed_at"] = fa
        return list(bucket.values())

    above = sorted([r for r in _merge(levels) if r["price"] >  price], key=lambda r: r["price"])
    below = sorted([r for r in _merge(levels) if r["price"] <= price], key=lambda r: -r["price"])

    def _fmt_row(r) -> str:
        delta = r["price"] - price
        pct = (delta / price * 100.0) if price > 0 else 0.0
        sign = "+" if delta >= 0 else "−"
        dist = f"{sign}${abs(delta):>5.2f}  {sign}{abs(pct):>5.2f}%"
        srcs = " / ".join(r["sources"]) if r["sources"] else "—"
        types = ",".join(sorted(r["types"]))
        bar = _strength_bar(r["strength"])
        price_s = f"${r['price']:>7.2f}"
        strength_s = f"{r['strength']:.2f}"
        types_s = f"[{types}]"
        return (
            f"    {_b(price_s)}  {_dim(dist):<28}  "
            f"{bar} {_dim(strength_s)}  "
            f"{srcs}  {_dim(types_s)}"
        )

    lines: list[str] = []
    price_s = f"${price:.2f}" if price > 0 else "—"
    counts_s = f"({len(levels)} active, {len(broken)} broken)"
    lines.append(f"\n  {_c('⟐')}  {_b(sym)} {_dim('levels')}  "
                 f"{_dim('current:')} {_b(price_s)}  "
                 f"{_dim(counts_s)}")

    if above:
        lines.append(f"\n  {_r('▲ resistance')}")
        for r in above:
            lines.append(_fmt_row(r))
    if below:
        lines.append(f"\n  {_g('▼ support')}")
        for r in below:
            lines.append(_fmt_row(r))

    if broken:
        lines.append(f"\n  {_dim('✕ broken / inactive')}")
        for lv in sorted(broken, key=lambda l: -float(l.price))[:8]:
            delta = float(lv.price) - price
            pct = (delta / price * 100.0) if price > 0 else 0.0
            sign = "+" if delta >= 0 else "−"
            dist = f"{sign}${abs(delta):>5.2f}  {sign}{abs(pct):>5.2f}%"
            price_s2 = f"${float(lv.price):>7.2f}"
            srcs = lv.source or "—"
            types = _LEVEL_TYPE_TAG.get(lv.level_type, "?")
            types_s = f"[{types}]"
            lines.append(f"    {_dim(price_s2)}  {_dim(dist):<28}  {_dim(srcs)}  {_dim(types_s)}")

    return "\n".join(lines)


def print_levels(sym: str, entry, rmap, *, current_price: float | None = None) -> None:
    """Pretty-print the resistance/support level map (calls format_levels)."""
    safe_print(format_levels(sym, entry, rmap, current_price=current_price))


# ---------------------------------------------------------------------------
# Live candle-detail display (TICKER live)
# ---------------------------------------------------------------------------

_LIVE_TF_ORDER = ("10s", "1m", "2m", "5m", "1d")
_LIVE_MAX_CANDLES = 12  # max candles to show per TF


def print_live_detail(
    sym: str,
    scorer,
    time_range: tuple[datetime, datetime] | None = None,
    time_range_label: str = "",
    max_rows: int | None = _LIVE_MAX_CANDLES,
    sort_by_severity: bool = True,
) -> None:
    """Print setup outcome detail for the TECHNICAL_ANALYSIS score.

    Shows resolved setups per timeframe, their quality, extension achieved,
    and whether they succeeded or failed.
    """
    outcomes = getattr(scorer, "_completed_outcomes", [])
    active_legs = getattr(scorer, "_active_legs", {})

    if not outcomes and not active_legs:
        safe_print(f"  {_y('⚠')}  {sym}: no setup outcomes recorded yet")
        return

    lines: list[str] = []
    title_suffix = f" {_dim(f'({time_range_label})')}" if time_range_label else ""
    lines.append(f"\n  {_c('⟐')}  {_b(sym)} {_dim('setup outcome detail')}{title_suffix}")

    for tf in _TA_TF_ORDER:
        tf_outcomes = [o for o in outcomes if o.timeframe == tf]
        active = active_legs.get(tf)
        if not tf_outcomes and active is None:
            continue

        lines.append(f"\n  {_b(tf.upper())} {'─' * 50}")

        if active is not None:
            ext_pct = active.extension_pct * 100
            q_color = _quality_color(min(1.0, ext_pct / 10.0), f"{ext_pct:.1f}%")
            lines.append(
                f"    {_c('ACTIVE')} quality={active.quality:.0%} "
                f"ext={q_color} bars={active.candle_count}"
            )

        for o in reversed(tf_outcomes[-max_rows:] if max_rows else tf_outcomes):
            mark = _g("✓") if o.success else _r("✗")
            ext_pct = o.extension_pct * 100
            lines.append(
                f"    {mark} quality={o.quality:.0%} "
                f"ext={ext_pct:.1f}% {'SUCCESS' if o.success else 'FAIL'}"
            )

    safe_print("\n".join(lines))


def _compute_atr(candles, period: int) -> float:
    """ATR from a list of OHLCVCandle (mirrors confidence_scorer._atr_from_candles)."""
    if len(candles) < 2:
        return 0.0
    tr_values: list[float] = []
    prev_close = float(candles[0].close)
    for c in candles:
        high, low = float(c.high), float(c.low)
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
        prev_close = float(c.close)
    lookback = min(max(1, period), len(tr_values))
    return sum(tr_values[-lookback:]) / lookback


def _fmt_ts(ts, tf: str) -> str:
    """Format candle timestamp compactly based on timeframe, in local timezone."""
    import pandas as pd
    from config import MarketClock
    local_tz = MarketClock.local_tz()
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        local = ts.to_pydatetime().astimezone(local_tz)
    else:
        from datetime import timezone as _tz
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_tz.utc)
        local = ts.astimezone(local_tz)
    if tf in ("10s",):
        return local.strftime("%H:%M:%S")
    if tf in ("1m", "2m"):
        return local.strftime("%H:%M")
    if tf in ("5m",):
        return local.strftime("%H:%M")
    # daily
    return local.strftime("%m/%d")


def _quality_bar(q: float) -> str:
    """Compact 5-char quality bar."""
    filled = round(q * 5)
    bar = "█" * filled + "░" * (5 - filled)
    return _quality_color(q, bar)


def _candle_header() -> str:
    """Column header for per-candle rows."""
    return (
        f"      {_dim('time'):>12}  "
        f"{_dim('O'):>7} {_dim('H'):>7} {_dim('L'):>7} {_dim('C'):>7}  "
        f"{_dim('raw'):>6}  {_dim('quality')}"
    )


def _candle_row(c, raw: float, q: float, tf: str) -> str:
    """Format one candle's quality detail."""
    ts_str = _fmt_ts(c.ts, tf)
    bar = _quality_bar(q)
    raw_s = _quality_color(q, f"{raw:.3f}")
    q_s = _quality_color(q, f"{q:.0%}")
    return (
        f"      {ts_str:>12}  "
        f"{c.open:>7.2f} {c.high:>7.2f} {c.low:>7.2f} {c.close:>7.2f}  "
        f"{raw_s:>6}  {bar} {q_s}"
    )


def _leg_header() -> str:
    """Column header for leg-smoothness rows."""
    return (
        f"      {_dim('from'):>12}  {_dim('to'):>12}  "
        f"{_dim('raw'):>7}  {_dim('quality')}"
    )


def _leg_row(window: list, raw: float, q: float, tf: str) -> str:
    """Format one leg-smoothness window."""
    ts_from = _fmt_ts(window[0].ts, tf)
    ts_to = _fmt_ts(window[-1].ts, tf)
    bar = _quality_bar(q)
    raw_s = _quality_color(q, f"{raw:+.3f}")
    q_s = _quality_color(q, f"{q:.0%}")
    return (
        f"      {ts_from:>12}  {ts_to:>12}  "
        f"{raw_s:>7}  {bar} {q_s}"
    )
