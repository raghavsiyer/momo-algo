#!/usr/bin/env python3
"""
yt_shadow_report.py

Offline evaluation of the ModernBERT shadow-mode news detector.

Reads the JSONL written by :class:`market_data.chat_logger.ChatShadowLogger`
(``logs/yt_chat/chat_YYYYMMDD.jsonl`` by default) and reports:

  1. Recall vs. the hard-coded Clementine matcher — what fraction of known
     news posts the classifier also flags.  Target ≥95% before going active.
  2. Every ModernBERT-only detection, for manual grading: these are either
     the wins (news from authors the matcher can't see) or the false
     positives that would leak into the pipeline.
  3. Score histogram and a threshold sweep (0.70-0.95) so the operating
     point can be retuned from data instead of guessed.
  4. Ticker-resolution method breakdown, including the news-with-no-ticker
     rate — those detections can never be traded.
  5. Per-author detection counts and mean batch latency.

Usage:
    python yt_shadow_report.py                    # every file in logs/yt_chat/
    python yt_shadow_report.py logs/yt_chat/chat_20260803.jsonl
    python yt_shadow_report.py --threshold 0.80 --max-examples 40
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_DEFAULT_DIR = Path(__file__).resolve().parent / "logs" / "yt_chat"
_THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def load_rows(paths: list[Path]) -> list[dict]:
    """Read every JSONL file, skipping malformed lines."""
    rows: list[dict] = []
    for path in paths:
        bad = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1
        if bad:
            print(f"  ! {path.name}: skipped {bad} malformed line(s)")
    return rows


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:5.1f}%" if den else "    —"


def report(rows: list[dict], threshold: float, max_examples: int) -> None:
    total = len(rows)
    scored = [r for r in rows if r.get("mb_score") is not None]
    clem = [r for r in rows if r.get("clementine_hit")]

    _rule("Corpus")
    print(f"  messages logged      : {total}")
    print(f"  classified           : {len(scored)}  ({_pct(len(scored), total)} of all)")
    print(f"  skipped (short/first-poll): {total - len(scored)}")
    print(f"  Clementine hits      : {len(clem)}")
    authors = Counter(r.get("author", "") for r in rows)
    print(f"  distinct authors     : {len(authors)}")

    if not scored:
        print("\nNo classified rows — nothing to evaluate yet.")
        return

    # --- 1. Recall vs. the Clementine matcher ---------------------------
    _rule(f"Recall vs. Clementine matcher  (threshold {threshold:.2f})")
    clem_scored = [r for r in clem if r.get("mb_score") is not None]
    hits = [r for r in clem_scored if r["mb_score"] >= threshold]
    missed = [r for r in clem_scored if r["mb_score"] < threshold]
    print(f"  Clementine posts classified : {len(clem_scored)} "
          f"({len(clem) - len(clem_scored)} skipped by the min-words guard)")
    print(f"  also flagged by ModernBERT  : {len(hits)}  ({_pct(len(hits), len(clem_scored))})"
          f"   target ≥95%")
    if missed:
        print(f"\n  Missed ({len(missed)}) — these are the recall failures:")
        for r in sorted(missed, key=lambda r: r["mb_score"], reverse=True)[:max_examples]:
            print(f"    {r['mb_score']:.3f}  [{r.get('clementine_ticker') or '?':<5}] "
                  f"{r.get('text', '')[:100]}")

    # --- 2. ModernBERT-only detections ----------------------------------
    only = [r for r in scored
            if r["mb_score"] >= threshold and not r.get("clementine_hit")]
    _rule(f"ModernBERT-only detections  ({len(only)}) — grade these by hand")
    for r in sorted(only, key=lambda r: r["mb_score"], reverse=True)[:max_examples]:
        method = r.get("mb_ticker_method") or "-"
        print(f"  {r['mb_score']:.3f}  {r.get('mb_ticker') or '----':<5} "
              f"[{method:<10}] {r.get('author', '')[:18]:<18} {r.get('text', '')[:80]}")
    if len(only) > max_examples:
        print(f"  … {len(only) - max_examples} more (raise --max-examples to see them)")

    # --- 3. Score distribution + threshold sweep ------------------------
    _rule("Score histogram (all classified messages)")
    buckets: Counter = Counter()
    for r in scored:
        buckets[min(int(r["mb_score"] * 10), 9)] += 1
    peak = max(buckets.values()) if buckets else 1
    for b in range(10):
        n = buckets.get(b, 0)
        bar = "█" * int(40 * n / peak) if n else ""
        print(f"  {b / 10:.1f}-{(b + 1) / 10:.1f}  {n:6d}  {bar}")

    _rule("Threshold sweep")
    print("  thresh   flagged   clem-recall   clem-hits   mb-only")
    for th in _THRESHOLDS:
        flagged = [r for r in scored if r["mb_score"] >= th]
        c_hits = sum(1 for r in flagged if r.get("clementine_hit"))
        mb_only = len(flagged) - c_hits
        recall = _pct(c_hits, len(clem_scored))
        print(f"   {th:.2f}   {len(flagged):7d}   {recall:>11}   {c_hits:9d}   {mb_only:7d}")
    print("\n  clem-recall = precision proxy floor (known-good posts caught);")
    print("  mb-only     = candidates needing manual grading (wins + false positives).")

    # --- 4. Ticker resolution -------------------------------------------
    _rule("Ticker resolution (rows flagged at the logged threshold)")
    flagged_live = [r for r in rows if r.get("mb_is_news")]
    methods = Counter(r.get("mb_ticker_method") or "none" for r in flagged_live)
    for method, n in methods.most_common():
        print(f"  {method:<12} {n:6d}   {_pct(n, len(flagged_live))}")
    no_ticker = methods.get("none", 0)
    print(f"\n  news with no resolvable ticker: {no_ticker} "
          f"({_pct(no_ticker, len(flagged_live))}) — never queueable")

    # --- 5. Authors and latency -----------------------------------------
    _rule("Detections by author (top 20)")
    by_author: Counter = Counter()
    author_msgs: defaultdict = defaultdict(int)
    for r in rows:
        author_msgs[r.get("author", "")] += 1
        if r.get("mb_score") is not None and r["mb_score"] >= threshold:
            by_author[r.get("author", "")] += 1
    for author, n in by_author.most_common(20):
        print(f"  {author[:28]:<28} {n:5d} / {author_msgs[author]:5d} msgs "
              f"({_pct(n, author_msgs[author])})")

    _rule("Latency")
    batch = [r["batch_ms"] for r in rows if r.get("batch_ms")]
    if batch:
        batch_sorted = sorted(batch)
        print(f"  mean batch : {sum(batch) / len(batch):8.0f} ms  over {len(batch)} rows")
        print(f"  median     : {batch_sorted[len(batch) // 2]:8.0f} ms")
        print(f"  max        : {batch_sorted[-1]:8.0f} ms")
    else:
        print("  no batch timings logged")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path,
                    help="JSONL files (default: every chat_*.jsonl in logs/yt_chat/)")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="operating point for the headline numbers (default 0.85)")
    ap.add_argument("--max-examples", type=int, default=25,
                    help="rows to print per example dump (default 25)")
    args = ap.parse_args()

    # Windows consoles default to cp1252 — the histogram bars and the chat
    # text itself (emoji, CJK) would otherwise blow up mid-report.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError, ValueError):
        pass

    paths = args.files or sorted(_DEFAULT_DIR.glob("chat_*.jsonl"))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        print(f"No shadow logs found in {_DEFAULT_DIR} — run a session with "
              f"SCANNER_YT_MODERNBERT_MODE=shadow first.")
        return 1

    print(f"Reading {len(paths)} file(s): {', '.join(p.name for p in paths)}")
    rows = load_rows(paths)
    if not rows:
        print("No records.")
        return 1
    report(rows, args.threshold, args.max_examples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
