#!/usr/bin/env python3
"""
tick_subs_probe.py — measure the account's true tick-by-tick subscription cap.

Subscribes ``reqTickByTickData("AllLast")`` to one symbol at a time until the
Gateway answers Error 10190 ("Max number of tick-by-tick requests has been
reached"), which fires at REQUEST time — so this works outside market hours.
Then cancels one line and retries, to confirm a cancel frees its slot
immediately.

Context: IBKR_MAX_TICK_SUBS=5 is an unverified belief; IBKR's docs tie the
tick-by-tick quota to the same formula as depth (=3 on this account). The
live 10190 handler self-corrects downward either way, but the real number
decides how many tape lines the stream can actually hold.

Usage:
    python tests/test_files/tick_subs_probe.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from ib_insync import IB, Stock  # noqa: E402

HOST = os.getenv("IBKR_HOST", "127.0.0.1")
PORT = int(os.getenv("IBKR_PORT", "4001"))
CLIENT_ID = int(os.getenv("IBKR_PROBE_CLIENT_ID", "96"))

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "NVDA", "TSLA", "META",
           "GOOGL", "AMD", "NFLX", "JPM"]
SETTLE_SECS = 2.0            # wait per subscription for a 10190 to arrive
NOISE = {2103, 2104, 2105, 2106, 2107, 2119, 2158, 321}


def main() -> None:
    ib = IB()
    errors: list = []       # (reqId, code, msg)
    ib.errorEvent += lambda rid, code, msg, c: errors.append((rid, code, msg))
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15, readonly=True)

    print(f"tick-by-tick cap probe — {HOST}:{PORT} client {CLIENT_ID}")
    granted: list = []      # (symbol, ticker)
    cap = None
    try:
        for sym in SYMBOLS:
            contract = Stock(sym, "SMART", "USD")
            ib.qualifyContracts(contract)
            mark = len(errors)
            ticker = ib.reqTickByTickData(contract, "AllLast", 0, False)
            ib.sleep(SETTLE_SECS)
            new = [(c, m) for _, c, m in errors[mark:] if c not in NOISE]
            hit_cap = any(c == 10190 or "tick-by-tick" in m.lower()
                          for c, m in new)
            if hit_cap:
                cap = len(granted)
                print(f"  {sym:<6} REFUSED  -> cap = {cap} concurrent lines")
                for c, m in new:
                    print(f"          error {c}: {m[:70]}")
                break
            granted.append((sym, contract))
            note = "; ".join(f"{c}:{m[:40]}" for c, m in new)
            print(f"  {sym:<6} granted  ({len(granted)} held)"
                  + (f"   [{note}]" if note else ""))
        else:
            print(f"  no refusal after {len(granted)} lines — cap is above "
                  f"the symbol list; extend SYMBOLS to push further")

        if cap is not None and granted:
            # Does cancelling free a slot immediately?
            free_sym, free_contract = granted[0]
            ib.cancelTickByTickData(free_contract, "AllLast")
            ib.sleep(1.0)
            retry = Stock(SYMBOLS[cap + 1] if cap + 1 < len(SYMBOLS)
                          else "IWM", "SMART", "USD")
            ib.qualifyContracts(retry)
            mark = len(errors)
            ib.reqTickByTickData(retry, "AllLast", 0, False)
            ib.sleep(SETTLE_SECS)
            refused = any(c == 10190 for _, c, m in errors[mark:])
            print(f"  cancelled {free_sym}, retried {retry.symbol}: "
                  + ("still refused — slot frees late/never"
                     if refused else "granted — cancel frees the slot "
                                     "immediately"))
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
