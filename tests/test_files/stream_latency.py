#!/usr/bin/env python3
"""
stream_latency.py — measure IBKR stream cadence: is the data real-time?

Answers, per symbol, for both the tick-by-tick tape and L2 depth:
  - how often data actually arrives, and the shortest gap between arrivals
  - whether the cadence is a fixed timer or event-driven (stdev/mean)
  - whether prints were held before delivery (exchange-second straddling)
  - whether all symbols share one flush clock

Timestamps are taken inside the ``ib_insync`` Wrapper callbacks, i.e. the
moment a message is decoded off the socket. They are therefore PRE-dispatch:
they do not include the latency ``IBKRSharedStream._loop`` adds on top
(``waitOnUpdate(0.1)`` plus a 0.05s floor).

Bursts are grouped by a >2ms arrival gap, NOT by socket-read boundary — the
Gateway issues several back-to-back socket writes inside a single flush, so
read boundaries split one flush into pieces and understate the true interval.

Baseline measured 2026-08-03 (RTH, five symbols, RITR/DFNS/MU/MSFT/AMZN):
    TAPE   fixed ~520ms flush timer, ~2/sec, stdev/mean 0.19-0.28,
           one Gateway-wide clock (100% burst coincidence MU vs MSFT).
    DEPTH  event-driven, ~106ms floor (p5), stdev/mean 0.68-0.75,
           independent of the tape clock (1-3% coincidence).
Both feeds are BATCHED, not conflated — intermediate states survive.

Run against a live session; outside market hours it reports too few messages.
Uses its own client id (IBKR_PROBE_CLIENT_ID, default 94) so it does not
disturb the algo's connection pool. Needs a free tick-by-tick and depth slot
per symbol, so keep the symbol count at or below IBKR_MAX_TICK_SUBS.

Usage:
    python tests/test_files/stream_latency.py MU MSFT AMZN 90
    make latency MU MSFT AMZN
"""

import os
import statistics
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

# Imported for its updateMktDepthL2 bounds-check patch, so the probe exercises
# the same depth handler the live app runs.
import market_data.ibkr_stream  # noqa: F401,E402
from ib_insync import IB, Stock  # noqa: E402
from ib_insync.wrapper import Wrapper  # noqa: E402

HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = int(os.getenv("IBKR_PORT", "4001"))
CLIENT_ID = int(os.getenv("IBKR_PROBE_CLIENT_ID", "94"))

# sym -> arrival times;  tape also carries the exchange timestamp
depth: dict[str, list] = defaultdict(list)
tape: dict[str, list] = defaultdict(list)
reqmap: dict[int, str] = {}

_orig_depth = Wrapper.updateMktDepthL2
_orig_tbt = Wrapper.tickByTickAllLast


def _dep(self, reqId, position, mm, operation, side, price, size,
         isSmartDepth=False):
    depth[reqmap.get(reqId, reqId)].append(time.perf_counter())
    return _orig_depth(self, reqId, position, mm, operation, side, price,
                       size, isSmartDepth)


def _tbt(self, reqId, tickType, time_, price, size, attrib, exchange,
         specialConditions):
    tape[reqmap.get(reqId, reqId)].append((time.perf_counter(), time_))
    return _orig_tbt(self, reqId, tickType, time_, price, size, attrib,
                     exchange, specialConditions)


Wrapper.updateMktDepthL2 = _dep
Wrapper.tickByTickAllLast = _tbt

BUCKETS = [(0, 1), (1, 10), (10, 25), (25, 50), (50, 75), (75, 100),
           (100, 150), (150, 250), (250, 400), (400, 600), (600, 1000),
           (1000, float("inf"))]


def _pct(vals: list, p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round((p / 100.0) * (len(s) - 1)))))]


def _arrivals(times: list, thresh_ms: float = 2.0) -> list:
    """Collapse messages into distinct wire arrivals."""
    out = [times[0]]
    for i in range(1, len(times)):
        if (times[i] - times[i - 1]) * 1000.0 > thresh_ms:
            out.append(times[i])
    return out


def _analyse(label: str, times: list, span: float, note: str = ""):
    n = len(times)
    print(f"\n  {label}: n={n} ({n / span:.1f}/s){note}")
    if n < 10:
        print("    too few messages — is the market open?")
        return None
    arr = _arrivals(times)
    if len(arr) < 3:
        print(f"    only {len(arr)} distinct arrivals")
        return None
    iv = [(arr[i] - arr[i - 1]) * 1000.0 for i in range(1, len(arr))]
    sd = statistics.pstdev(iv) / statistics.mean(iv)
    print(f"    distinct arrivals: {len(arr)} ({len(arr) / span:.2f}/s), "
          f"{n / len(arr):.1f} msgs each")
    print(f"    shortest gap between arrivals: {min(iv):.2f} ms")
    print(f"    interval ms: p5={_pct(iv, 5):.1f} p25={_pct(iv, 25):.1f} "
          f"p50={_pct(iv, 50):.1f} p75={_pct(iv, 75):.1f} "
          f"p90={_pct(iv, 90):.1f}")
    print(f"    stdev/mean = {sd:.2f}  "
          f"({'FIXED TIMER' if sd < 0.45 else 'event-driven'})")
    hist = "    "
    for lo, hi in BUCKETS:
        c = sum(1 for x in iv if lo <= x < hi)
        if c:
            tag = f"{lo}-{hi:.0f}" if hi != float("inf") else f"{lo}+"
            hist += f"[{tag}ms {100 * c / len(iv):.0f}%] "
    print(hist)
    return arr


def _straddle_note(events: list) -> str:
    """Share of multi-print flushes carrying more than one exchange second.

    IBKR tick-by-tick timestamps are whole seconds. Prints delivered together
    but stamped in different seconds cannot have been simultaneous, so they
    were held — the signature of batched delivery.
    """
    if len(events) < 10:
        return ""
    groups, cur = [], [events[0]]
    for i in range(1, len(events)):
        if (events[i][0] - events[i - 1][0]) * 1000.0 > 2.0:
            groups.append(cur)
            cur = []
        cur.append(events[i])
    groups.append(cur)
    multi = [g for g in groups if len(g) > 1]
    if not multi:
        return ""
    strad = [g for g in multi if len({x[1] for x in g}) > 1]
    return (f"   multi-print flushes straddling >1 exchange second: "
            f"{100 * len(strad) / len(multi):.0f}%")


def _coincidence(a: list, b: list, tol: float = 0.005) -> tuple[int, int]:
    hit, j = 0, 0
    for t in a:
        while j < len(b) and b[j] < t - tol:
            j += 1
        if j < len(b) and abs(b[j] - t) <= tol:
            hit += 1
    return hit, len(a)


def main() -> None:
    args = sys.argv[1:]
    duration = 90.0
    if args and args[-1].replace(".", "").isdigit():
        duration = float(args.pop())
    symbols = [s.upper() for s in args] or ["MU", "MSFT", "AMZN"]

    ib = IB()
    errors: list = []
    ib.errorEvent += lambda rid, code, msg, c: errors.append((code, msg))
    # readonly=True skips the open/completed-order requests ib_insync fires on
    # connect, which a read-only Gateway rejects with Error 321. Market data
    # is unaffected either way.
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    for sym in symbols:
        contract = Stock(sym, "SMART", "USD")
        ib.qualifyContracts(contract)
        ib.reqMktDepth(contract, numRows=5, isSmartDepth=True)
        ib.reqTickByTickData(contract, "AllLast")
        for rid, tkr in ib.wrapper.reqId2Ticker.items():
            if tkr.contract.symbol == sym:
                reqmap[rid] = sym
        print(f"  subscribed {sym} conId={contract.conId}")

    print(f"sampling {duration:.0f}s ...")
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        ib.sleep(0.002)
    span = time.perf_counter() - start
    ib.disconnect()

    print(f"\n{'#' * 72}\n# IBKR stream cadence — {span:.1f}s — "
          f"{', '.join(symbols)}\n{'#' * 72}")
    # Connection/farm status codes are informational, not failures here.
    noise = {2103, 2104, 2105, 2106, 2107, 2119, 2152, 2158, 321, 2100}
    real = [e for e in errors if e[0] not in noise]
    if real:
        print("errors:", Counter(real).most_common(8))

    tape_arrivals: dict[str, list] = {}
    for sym in symbols:
        print(f"\n=== {sym} " + "=" * max(0, 66 - len(sym)))
        _analyse("L2 DEPTH", depth[sym], span)
        arr = _analyse("TAPE", [t for t, _ in tape[sym]], span,
                       _straddle_note(tape[sym]))
        if arr:
            tape_arrivals[sym] = arr

    if len(tape_arrivals) >= 2:
        print("\n=== SHARED-CLOCK TEST (tape flushes within 5ms) " + "=" * 23)
        print("  high % = one Gateway-wide flush timer serving every "
              "subscription")
        base = symbols[0]
        for other in symbols[1:]:
            if base in tape_arrivals and other in tape_arrivals:
                hit, tot = _coincidence(tape_arrivals[base],
                                        tape_arrivals[other])
                print(f"  {base} vs {other}: {hit}/{tot} "
                      f"({100 * hit / tot:.1f}%)")

    print("\n=== L2 vs TAPE CLOCK " + "=" * 50)
    print("  low % = depth is delivered independently of the tape flush")
    for sym in symbols:
        if depth[sym] and sym in tape_arrivals:
            hit, tot = _coincidence(_arrivals(depth[sym]), tape_arrivals[sym])
            print(f"  {sym}: {hit}/{tot} L2 arrivals land on a tape flush "
                  f"({100 * hit / tot:.1f}%)")


if __name__ == "__main__":
    main()
