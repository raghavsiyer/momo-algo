#!/usr/bin/env python3
"""
hist_pacing.py — does IBKR's "60 historical requests / 10 minutes" limit
apply to every bar size, or only to sub-minute bars?

IBKR documents three separate pacing rules
(https://interactivebrokers.github.io/tws-api/historical_limitations.html):

    R1  identical historical request within 15 seconds
    R2  six or more requests for the same Contract/Exchange/TickType
        within two seconds
    R3  more than 60 requests within any ten minute period

...and separately claims "Historical Data Limitations for barSize = '1 mins'
and greater have been lifted."  This probe tests whether that lift really
covers R3.  (Verdict, 2026-08-04: none of the three rules is enforced at
all — the Gateway shapes sub-30s-bar and tick traffic by delaying responses
instead; see the pacing comment in ``market_data/ibkr_rest.py``.  The
client-side gate that modelled the 60/10min budget was removed on the back
of these runs; this probe stays for re-verification after Gateway upgrades.)

Three arms, each in a fresh 10-minute window:

    A  "1 min"  x N                 is a 1-min request itself ever stalled?
    B  "30 secs" x N                POSITIVE CONTROL — can this probe detect
                                    a knee at all?  A "no knee" in A means
                                    nothing unless B produces one at ~61.
    C  "1 min" x 60, "30 secs" x 6  do 1-min requests CONSUME the budget?

Arm C is the one that matters and the reason A alone is not enough: a
request can be counted by the meter without ever being stalled itself.  If
1-min bars are counted-but-exempt, arm A looks clean while the budget is
quietly drained and the algo's real 10-sec/tick calls pay for it.  After 60
1-min calls, arm C's 30-sec tail stalls immediately if they were counted,
and sails through if they were not.

Every arm runs over TWO INDEPENDENT TRANSPORTS (--client both):

    insync  ib_insync's IB.reqHistoricalData, the library the algo uses
    raw     ibapi EClient.reqHistoricalData — bare TWS API sockets, no
            ib_insync anywhere in the path, requests written straight onto
            the wire and EWrapper.error/historicalDataEnd read straight off

If the two transports disagree, the knee is a client-library artifact and
not a Gateway rule.  Agreement is what licenses the conclusion.

Controls, so a knee can only come from R3:

    * every request is unique          -> endDateTime walks back one minute
                                          per call, and each run uses its own
                                          trading day, so R1 can never fire
                                          and no response can be served from
                                          a cache that might not be metered
    * 8 symbols round-robin + a 0.35s  -> a given contract is re-requested at
      floor between calls                 most once per ~2.8s, so R2 (6 in 2s)
                                          can never fire
    * 2 bars per request               -> payload is identical in size across
                                          arms, so the undocumented "soft"
                                          load-balancer throttle on large
                                          datasets cannot masquerade as a
                                          pacing knee
    * whatToShow=TRADES                -> cost 1/request; BID_ASK counts twice
    * never IBKRClient                 -> raw connections only, so no client
                                          layer of ours can skip or reorder
                                          calls and we measure the Gateway,
                                          not our own assumptions
    * recent sessions                  -> sub-minute bars are only available
                                          for the last six months

PRECONDITION: nothing else may be issuing historical requests. The budget is
metered Gateway-wide per account, not per client id, so a running main.py
poisons the count. A knee well before call 60 is reported as contamination
rather than as a result.

Runtime is roughly 10 min per run plus a 630s cooldown between runs (the
window must drain fully, or the arms are not independent). The full matrix
(3 arms x 2 transports) is ~70 min; --client insync --arms A,B,C is ~30 min.
--cooldown 0 is a plumbing smoke test only, never a result.

Usage:
    python tests/test_files/hist_pacing.py
    python tests/test_files/hist_pacing.py --client raw --arms B,C
    python tests/test_files/hist_pacing.py --calls 10 --cooldown 0   # smoke
    make pacing
"""

import argparse
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = int(os.getenv("IBKR_PORT", "4001"))
# Distinct from stream_latency.py's 94 so both probes can share a Gateway.
CLIENT_ID = int(os.getenv("IBKR_PACING_CLIENT_ID", "95"))

# Rotated round-robin so R2 (6 requests per contract per 2s) cannot fire.
# Liquid names, all SMART/USD, all with continuous RTH prints. --syms N
# widens the rotation: with --gap G, the same contract recurs every N*G
# seconds — at 16 x 1.0s that clears even the LOOSEST reading of the
# identical-request rule (same contract regardless of window, 15s).
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "META",
           "GOOGL", "AMD", "NFLX", "JPM", "BAC", "XOM", "DIA", "IWM"]
N_SYMS = 8              # default rotation width; --syms overrides

BARS_PER_CALL = 2       # keep every response the same trivial size
TICKS_PER_CALL = 50     # small on purpose: isolates request COUNT from the
                        # known payload throttle on 1000-tick responses
STALL_SECS = 25.0       # a stalled request is abandoned here, not after 600s
MIN_GAP_SECS = 0.35     # 8 symbols x 0.35s => same contract at most 1/2.8s
SLOW_SECS = 1.0         # above this a call is "slow" and gets a trace line
MAX_CONSEC_STALLS = 4   # the knee is proven; stop burning wall-clock
ANCHOR_HOUR_UTC = 17    # 13:00 ET — mid-RTH on every run's session

# Informational connection/farm messages, not failures.
NOISE_CODES = {321, 399, 2100, 2103, 2104, 2105, 2106, 2107, 2119, 2150,
               2158, 2174}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bar_seconds(bar_size: str) -> int:
    """Seconds per bar for an IBKR barSizeSetting string."""
    parts = str(bar_size).strip().lower().split()
    n, unit = int(parts[0]), parts[-1]
    if unit.startswith("sec"):
        return n
    if unit.startswith("min"):
        return n * 60
    if unit.startswith("hour"):
        return n * 3600
    return n * 86400


def _duration_for(bar_size: str) -> str:
    """durationStr yielding BARS_PER_CALL bars, floored at IBKR's minimum.

    The Gateway rejects second-durations below 30 S with Error 321
    ("requested duration is invalid"), so tiny bars get a 60 S floor —
    a few extra bars in the payload, still trivially small.
    """
    secs = max(60, _bar_seconds(bar_size) * BARS_PER_CALL)
    return f"{secs // 86400} D" if secs >= 86400 else f"{secs} S"


def _recent_weekdays(n: int, start: date) -> list:
    """The n most recent weekdays strictly before *start*, newest first."""
    out, d = [], start
    while len(out) < n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def _fmt_end(dt: datetime) -> str:
    return dt.strftime("%Y%m%d %H:%M:%S UTC")


def _classify(errs: list) -> tuple:
    """Split one call's IBKR errors into (pacing, no-data, other) strings."""
    paced, nodata, other = [], [], []
    for code, msg in errs:
        low = str(msg).lower()
        if "pacing" in low or "max number of requests" in low:
            paced.append(f"{code}:{msg}")
        elif "no data" in low or "query returned no data" in low:
            nodata.append(f"{code}:{msg}")
        elif code not in NOISE_CODES:
            other.append(f"{code}:{msg}")
    return paced, nodata, other


# ---------------------------------------------------------------------------
# transports — same experiment, two independent client stacks
# ---------------------------------------------------------------------------

class InsyncTransport:
    """ib_insync, i.e. exactly the path market_data/ibkr_rest.py takes."""

    name = "insync"

    def __init__(self):
        self._ib = None
        self._sink: list = []

    def connect(self, client_id: int) -> None:
        from ib_insync import IB
        self._ib = IB()
        self._ib.errorEvent += lambda rid, code, msg, c: self._sink.append(
            (code, msg))
        # readonly=True skips the open-order requests ib_insync fires on
        # connect, which a read-only Gateway rejects with Error 321.
        self._ib.connect(HOST, PORT, clientId=client_id, timeout=15,
                         readonly=True)

    def qualify(self) -> dict:
        from ib_insync import Stock
        out = {}
        for sym in SYMBOLS:
            c = Stock(sym, "SMART", "USD")
            self._ib.qualifyContracts(c)
            out[sym] = c
        return out

    def hist(self, contract, end, duration, bar_size) -> tuple:
        mark = len(self._sink)
        try:
            bars = self._ib.reqHistoricalData(
                contract, endDateTime=_fmt_end(end), durationStr=duration,
                barSizeSetting=bar_size, whatToShow="TRADES", useRTH=False,
                formatDate=2, timeout=STALL_SECS)
            exc = None
        except Exception as e:                                # noqa: BLE001
            bars, exc = [], (0, f"{type(e).__name__}: {e}")
        self._ib.sleep(0.05)                                  # drain events
        errs = list(self._sink[mark:]) + ([exc] if exc else [])
        return len(bars), errs

    def ticks(self, contract, end, what_to_show) -> tuple:
        """One reqHistoricalTicks call: last TICKS_PER_CALL before *end*.

        endDateTime is tz-aware UTC on purpose — a naive datetime is read
        as exchange-local and silently returns 0 ticks (measured 2026-08-03),
        which would make a throttled burst look fast because it did no work.
        """
        mark = len(self._sink)
        try:
            # ib_insync has no per-call timeout here; RequestTimeout guards
            # it instead, restored after so bar arms keep their own value.
            prev = self._ib.RequestTimeout
            self._ib.RequestTimeout = STALL_SECS
            try:
                ticks = self._ib.reqHistoricalTicks(
                    contract, startDateTime="", endDateTime=_fmt_end(end),
                    numberOfTicks=TICKS_PER_CALL, whatToShow=what_to_show,
                    useRth=False)
            finally:
                self._ib.RequestTimeout = prev
            exc = None
        except Exception as e:                                # noqa: BLE001
            ticks, exc = [], (0, f"{type(e).__name__}: {e}")
        self._ib.sleep(0.05)
        errs = list(self._sink[mark:]) + ([exc] if exc else [])
        return len(ticks), errs

    def sleep(self, secs: float) -> None:
        if secs > 0:
            self._ib.sleep(secs)

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()


class RawTransport:
    """Bare ibapi EClient/EWrapper — no ib_insync in the path at all.

    Requests go straight onto the TWS socket and completion is read from
    ``historicalDataEnd``/``error`` callbacks, so a knee observed here is
    the Gateway's behaviour and cannot be anything the wrapper library did.
    Errors are attributed by reqId rather than by time window, which is
    strictly tighter than what the ib_insync side can do.
    """

    name = "raw"

    def __init__(self):
        self._app = None
        self._thread = None
        self._next_id = 1000

    def connect(self, client_id: int) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        probe = self

        class _App(EWrapper, EClient):
            def __init__(self):
                EWrapper.__init__(self)
                EClient.__init__(self, self)
                self.ready = threading.Event()
                self.bars: dict = {}
                self.errs: dict = {}
                self.done: dict = {}
                self.details: dict = {}

            def nextValidId(self, orderId):
                probe._next_id = max(probe._next_id, orderId + 1)
                self.ready.set()

            def historicalData(self, reqId, bar):
                self.bars.setdefault(reqId, []).append(bar)

            def historicalDataEnd(self, reqId, start, end):
                ev = self.done.get(reqId)
                if ev:
                    ev.set()

            def contractDetails(self, reqId, contractDetails):
                self.details[reqId] = contractDetails.contract

            def contractDetailsEnd(self, reqId):
                ev = self.done.get(reqId)
                if ev:
                    ev.set()

            # ibapi 9.81 passes (reqId, code, msg); 10.x appends
            # advancedOrderRejectJson. Accept both.
            def error(self, reqId, errorCode, errorString, *rest):
                self.errs.setdefault(reqId, []).append((errorCode,
                                                        errorString))
                # A real (non-informational) error terminates the request.
                if errorCode not in NOISE_CODES:
                    ev = self.done.get(reqId)
                    if ev:
                        ev.set()

        self._app = _App()
        self._app.connect(HOST, PORT, client_id)
        self._thread = threading.Thread(target=self._app.run, daemon=True)
        self._thread.start()
        if not self._app.ready.wait(15):
            raise RuntimeError("raw ibapi: no nextValidId within 15s")

    def _rid(self) -> int:
        self._next_id += 1
        return self._next_id

    def qualify(self) -> dict:
        """Resolve conIds via reqContractDetails, outside the timed loop."""
        from ibapi.contract import Contract
        out = {}
        for sym in SYMBOLS:
            c = Contract()
            c.symbol, c.secType, c.exchange, c.currency = \
                sym, "STK", "SMART", "USD"
            rid = self._rid()
            ev = threading.Event()
            self._app.done[rid] = ev
            self._app.reqContractDetails(rid, c)
            if not ev.wait(15):
                raise RuntimeError(f"raw ibapi: contract details for {sym} "
                                   f"timed out")
            resolved = self._app.details.get(rid)
            if resolved is None:
                raise RuntimeError(f"raw ibapi: {sym} did not resolve "
                                   f"({self._app.errs.get(rid)})")
            # Pin the conId so SMART routing cannot re-resolve mid-run.
            resolved.exchange = "SMART"
            out[sym] = resolved
        return out

    def hist(self, contract, end, duration, bar_size) -> tuple:
        rid = self._rid()
        ev = threading.Event()
        self._app.done[rid] = ev
        self._app.reqHistoricalData(
            rid, contract, _fmt_end(end), duration, bar_size,
            "TRADES", 0, 2, False, [])
        if not ev.wait(STALL_SECS):
            self._app.cancelHistoricalData(rid)
        errs = list(self._app.errs.get(rid, []))
        # reqId -1 carries system-wide messages (incl. some pacing notices).
        errs += list(self._app.errs.pop(-1, []))
        return len(self._app.bars.get(rid, [])), errs

    def sleep(self, secs: float) -> None:
        if secs > 0:
            time.sleep(secs)

    def disconnect(self) -> None:
        if self._app is not None:
            self._app.disconnect()


TRANSPORTS = {"insync": InsyncTransport, "raw": RawTransport}


# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    key: str
    question: str
    segments: list           # [(bar_size, count), ...]
    anchor: datetime = None
    rows: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(c for _, c in self.segments)


def _run_arm(tr, arm: Arm, contracts: dict) -> None:
    """Fire the arm's calls back to back, recording latency and errors."""
    print(f"\n{'=' * 72}\n"
          f"[{tr.name}] ARM {arm.key} — {arm.question}\n"
          f"  {' then '.join(f'{c} x {b!r}' for b, c in arm.segments)}"
          f"   session {arm.anchor:%Y-%m-%d}, {arm.total} calls\n"
          f"{'=' * 72}", flush=True)

    calls = [(seg, bar) for seg, (bar, cnt) in enumerate(arm.segments)
             for bar in [bar] * cnt]
    consec = 0
    for i, (seg, bar_size) in enumerate(calls):
        sym = SYMBOLS[i % N_SYMS]
        end = arm.anchor - timedelta(minutes=i)
        t0 = time.monotonic()
        if bar_size.startswith("TICKS:"):
            n_bars, errs = tr.ticks(contracts[sym], end,
                                    bar_size.split(":", 1)[1])
        else:
            n_bars, errs = tr.hist(contracts[sym], end,
                                   _duration_for(bar_size), bar_size)
        dt = time.monotonic() - t0
        paced, nodata, other = _classify(errs)
        stalled = dt >= STALL_SECS * 0.9 or bool(paced)
        arm.rows.append({
            "i": i, "seg": seg, "bar": bar_size, "sym": sym, "dt": dt,
            "n": n_bars, "paced": paced, "nodata": nodata, "other": other,
            "stalled": stalled,
        })

        consec = consec + 1 if stalled else 0
        if stalled or dt > SLOW_SECS or other:
            tag = "PACED" if paced else ("STALL" if stalled else "slow ")
            detail = (paced + other or [""])[0]
            print(f"  call {i + 1:>3} {sym:<5} {bar_size:<8} {dt:6.2f}s  "
                  f"{tag}  {detail[:78]}", flush=True)
        if consec >= MAX_CONSEC_STALLS:
            print(f"  -> {consec} consecutive stalls at call {i + 1}; "
                  f"knee established, stopping arm", flush=True)
            break

        tr.sleep(MIN_GAP_SECS - (time.monotonic() - t0))


def _strip(arm: Arm) -> None:
    """One character per call: . fast  : 1-5s  ! 5s+  X stalled  P paced."""
    def ch(r):
        if r["paced"]:
            return "P"
        if r["stalled"]:
            return "X"
        return "!" if r["dt"] >= 5 else (":" if r["dt"] >= SLOW_SECS else ".")

    marks = [ch(r) for r in arm.rows]
    print("\n  per-call latency strip "
          "(. <1s   : 1-5s   ! 5s+   X stalled   P pacing error)")
    for base in range(0, len(marks), 30):
        print(f"   {base + 1:>4} {''.join(marks[base:base + 30])}")


def _seg_stats(arm: Arm, seg: int) -> dict:
    rows = [r for r in arm.rows if r["seg"] == seg]
    lat = [r["dt"] for r in rows]
    return {
        "rows": rows, "n": len(rows),
        "med": statistics.median(lat) if lat else float("nan"),
        "max": max(lat) if lat else float("nan"),
        "stalls": sum(1 for r in rows if r["stalled"]),
        "paced": sum(1 for r in rows if r["paced"]),
        "empty": sum(1 for r in rows if r["n"] == 0),
    }


def _report(arm: Arm, limit: int) -> dict:
    """Print the arm's numbers and return the facts the verdict needs."""
    _strip(arm)
    first_stall = next((r["i"] + 1 for r in arm.rows if r["stalled"]), None)
    for seg, (bar_size, _) in enumerate(arm.segments):
        s = _seg_stats(arm, seg)
        if not s["n"]:
            continue
        print(f"\n  segment {seg + 1}: {bar_size!r} x {s['n']}")
        print(f"    latency  median {s['med']:.2f}s   max {s['max']:.2f}s")
        print(f"    stalled  {s['stalls']}/{s['n']}   "
              f"pacing errors {s['paced']}   empty responses {s['empty']}")
        if s["empty"] == s["n"]:
            print("    !! every response was empty — wrong session date? "
                  "pass --date")

    # Within a single-segment arm the knee should land just past the limit.
    if len(arm.segments) == 1:
        head = [r["dt"] for r in arm.rows if r["i"] < limit]
        tail = [r["dt"] for r in arm.rows if r["i"] >= limit]
        if head and tail:
            print(f"\n    calls 1-{limit}:  median "
                  f"{statistics.median(head):.2f}s")
            print(f"    calls {limit + 1}+:  median "
                  f"{statistics.median(tail):.2f}s")
    print(f"\n    first stall: "
          f"{f'call {first_stall}' if first_stall else 'none'}", flush=True)
    return {"first_stall": first_stall, "arm": arm}


def _verdict_one(tr_name: str, res: dict, limit: int, bar: str,
                 control_bar: str) -> bool:
    """Print one transport's read of the evidence; return True if clean."""
    print(f"\n  --- {tr_name} " + "-" * (66 - len(tr_name)))
    a, b, c = res.get("A"), res.get("B"), res.get("C")
    control = None

    if b:
        fs = b["first_stall"]
        if fs and fs > limit:
            control = True
            print(f"    CONTROL   arm B {control_bar!r}: knee at call {fs} — "
                  f"the {limit}/10min meter is real and detectable here.")
        elif fs:
            control = False
            print(f"    CONTROL   arm B: stalled early at call {fs}, before "
                  f"the {limit}-call budget was spent — something else was "
                  f"consuming it. NOT clean.")
        else:
            control = False
            print(f"    CONTROL   arm B {control_bar!r}: no knee in "
                  f"{b['arm'].total} calls — the meter never fired at all, "
                  f"so a clean arm A proves nothing.")

    if a:
        fs = a["first_stall"]
        if fs is None:
            print(f"    ENFORCE   arm A {bar!r}: {len(a['arm'].rows)} calls, "
                  f"no stall — {bar!r} requests are never themselves paced.")
        elif fs > limit:
            print(f"    ENFORCE   arm A {bar!r}: knee at call {fs} — "
                  f"{bar!r} IS metered.")
        else:
            print(f"    ENFORCE   arm A: stalled at call {fs}, before the "
                  f"budget could be spent — contaminated.")

    tail_clean = None
    if c:
        tail = _seg_stats(c["arm"], 1)
        if not tail["n"]:
            print("    ACCOUNT   arm C: tail never ran.")
        elif tail["stalls"]:
            tail_clean = False
            print(f"    ACCOUNT   arm C: {control_bar!r} tail stalled "
                  f"{tail['stalls']}/{tail['n']} right after {limit} {bar!r} "
                  f"calls — {bar!r} DOES consume the budget.")
        else:
            tail_clean = True
            print(f"    ACCOUNT   arm C: {control_bar!r} tail ran clean "
                  f"(median {tail['med']:.2f}s) after {limit} {bar!r} calls "
                  f"— {bar!r} does NOT consume the budget.")

    return bool(control) and a is not None and a["first_stall"] is None \
        and tail_clean is True


def _verdict(res: dict, limit: int, bar: str, control_bar: str) -> None:
    print(f"\n\n{'#' * 72}\n# VERDICT\n{'#' * 72}")
    outcomes = {}
    for tr_name, arms in res.items():
        outcomes[tr_name] = _verdict_one(tr_name, arms, limit, bar,
                                         control_bar)

    print(f"\n  --- conclusion " + "-" * 57)
    if len(outcomes) > 1:
        vals = set(outcomes.values())
        if len(vals) > 1:
            print("    TRANSPORTS DISAGREE: "
                  + ", ".join(f"{k}={'exempt' if v else 'metered/unclean'}"
                              for k, v in outcomes.items())
                  + "\n    A rule enforced by the Gateway cannot depend on "
                    "the client library.\n    Treat this as inconclusive and "
                    "rerun — most likely one run was contaminated.")
            return
        print("    both transports agree (ib_insync and bare ibapi).")

    if all(outcomes.values()) and outcomes:
        print(f"    => CONFIRMED: the {limit}/10min limit applies only to "
              f"sub-minute bars.\n"
              f"       {bar!r} requests are neither stalled nor counted, "
              f"while {control_bar!r}\n"
              f"       knees at ~{limit + 1}.")
    else:
        print(f"    => NOT CONFIRMED. Either {bar!r} is subject to the "
              f"limit, or the\n       positive control failed. Re-read the "
              f"per-arm lines above; if a knee\n       landed before call "
              f"{limit}, stop main.py and every other IBKR client\n"
              f"       and rerun — the budget is metered per account, not "
              f"per client id.")


def main() -> None:
    global MIN_GAP_SECS, N_SYMS
    p = argparse.ArgumentParser(
        description="Test whether IBKR's 60-per-10-min historical limit "
                    "applies to bar sizes above 30 seconds.")
    p.add_argument("--client", default="both",
                   choices=["insync", "raw", "both"],
                   help="transport(s) to run the arms over (default both)")
    p.add_argument("--arms", default="A,B,C", help="subset of A,B,C")
    p.add_argument("--calls", type=int, default=75,
                   help="calls per single-segment arm (default 75)")
    p.add_argument("--limit", type=int, default=60,
                   help="the budget under test (default 60)")
    p.add_argument("--bar", default="1 min",
                   help="bar size under test in arms A and C")
    p.add_argument("--control-bar", default="30 secs",
                   help="metered bar size for the positive control")
    p.add_argument("--cooldown", type=float, default=630.0,
                   help="seconds between runs; must exceed the 600s window")
    p.add_argument("--predrain", type=float, default=0.0,
                   help="wait this long BEFORE the first run, so a preceding "
                        "smoke test or app shutdown cannot leave slots "
                        "charged against arm 1")
    p.add_argument("--date", default=None,
                   help="newest session to sample, YYYY-MM-DD")
    p.add_argument("--gap", type=float, default=MIN_GAP_SECS,
                   help="minimum seconds between calls (default 0.35)")
    p.add_argument("--syms", type=int, default=N_SYMS,
                   help=f"rotation width, max {len(SYMBOLS)} (default 8); "
                        f"same contract recurs every syms*gap seconds")
    args = p.parse_args()

    MIN_GAP_SECS = args.gap
    N_SYMS = min(args.syms, len(SYMBOLS))

    if args.calls > 180:
        print(f"--calls {args.calls} would walk endDateTime out of RTH; "
              f"clamping to 180")
        args.calls = 180

    keys = [x.strip().upper() for x in args.arms.split(",")]
    names = ["insync", "raw"] if args.client == "both" else [args.client]
    runs = [(t, k) for t in names for k in keys if k in "ABCT"]

    start = (datetime.strptime(args.date, "%Y-%m-%d").date() + timedelta(days=1)
             if args.date else datetime.now(timezone.utc).date())
    # One distinct trading session per run, so no request is ever repeated
    # across runs and nothing can be answered from a cache.
    days = _recent_weekdays(len(runs), start)

    print(f"\n{'#' * 72}\n"
          f"# IBKR historical pacing probe — {HOST}:{PORT} client {CLIENT_ID}\n"
          f"# question: does the {args.limit}/10min limit apply to "
          f"{args.bar!r}?\n"
          f"# transports: {', '.join(names)}    arms: {','.join(keys)}\n"
          f"# est. runtime: ~{(len(runs) - 1) * args.cooldown / 60 + 3 * len(runs):.0f} min\n"
          f"#\n"
          f"# PRECONDITION: main.py and every other historical consumer must\n"
          f"# be stopped — the budget is metered per account, not per client.\n"
          f"{'#' * 72}", flush=True)

    if args.predrain > 0:
        print(f"\n  pre-draining {args.predrain:.0f}s so the window is empty "
              f"before arm 1...", flush=True)
        time.sleep(args.predrain)

    res: dict = {}
    for n, (tr_name, key) in enumerate(runs):
        anchor = datetime.combine(days[n], datetime.min.time(),
                                  timezone.utc).replace(hour=ANCHOR_HOUR_UTC)
        arm = {
            "A": Arm("A", f"is a {args.bar!r} request itself ever stalled?",
                     [(args.bar, args.calls)], anchor),
            "B": Arm("B", f"POSITIVE CONTROL — {args.control_bar!r} must knee",
                     [(args.control_bar, args.calls)], anchor),
            "C": Arm("C", f"do {args.bar!r} requests consume the budget?",
                     [(args.bar, args.limit), (args.control_bar, 6)], anchor),
            # 50-tick payloads: any knee here is pacing, not transfer time.
            "T": Arm("T", "are historical TICK requests count- or "
                          "burst-metered? (TRADES then BID_ASK)",
                     [("TICKS:TRADES", args.calls),
                      ("TICKS:BID_ASK", 25)], anchor),
        }[key]

        if n:
            print(f"\n  cooling down {args.cooldown:.0f}s so the 10-minute "
                  f"window drains before [{tr_name}] arm {key}...", flush=True)
            time.sleep(args.cooldown)

        # A fresh connection (and client id) per run, so no per-connection
        # state can carry over between arms.
        tr = TRANSPORTS[tr_name]()
        tr.connect(CLIENT_ID + n)
        try:
            _run_arm(tr, arm, tr.qualify())
        finally:
            tr.disconnect()
            time.sleep(1.0)
        res.setdefault(tr_name, {})[key] = _report(arm, args.limit)

    _verdict(res, args.limit, args.bar, args.control_bar)


if __name__ == "__main__":
    main()
