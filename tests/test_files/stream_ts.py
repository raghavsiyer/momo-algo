#!/usr/bin/env python3
"""
stream_ts.py — Live Time & Sales stream from Tradier (WebSocket).

Usage:
    python tests/test_files/stream_ts.py AAPL
    make stream AAPL
"""

import sys
import os
import signal
import threading
import uuid

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from market_data.tradier_stream import get_shared_stream, LiveTimesale, LiveQuote


def main():
    if len(sys.argv) < 2:
        print("Usage: python stream_ts.py <SYMBOL>")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    sub_id = str(uuid.uuid4())
    stop = threading.Event()

    print(f"Streaming T&S for {symbol} via WebSocket — press Ctrl+C to stop\n")
    print(f"{'Time':<12} {'Price':>8} {'Size':>7} {'Bid':>8} {'Ask':>8} {'Spread':>7} {'Side':<5} {'Flag'}")
    print("-" * 78)

    stream = get_shared_stream()

    def on_ts(event: LiveTimesale):
        if event.cancel or event.correction:
            return
        # Classify trade side
        if event.last >= event.ask:
            side = "\033[32mBUY\033[0m"   # green
        elif event.last <= event.bid:
            side = "\033[91mSELL\033[0m"  # red
        else:
            side = "\033[33mMID\033[0m"   # yellow

        spread = event.ask - event.bid
        ts_str = event.ts.astimezone().strftime("%H:%M:%S.%f")[:-3]

        print(
            f"{ts_str:<12} "
            f"${event.last:>7.4f} "
            f"{event.size:>7,} "
            f"${event.bid:>7.4f} "
            f"${event.ask:>7.4f} "
            f"${spread:>6.4f} "
            f"{side:<14} "
            f"{event.flag}"
        )

    def on_quote(event: LiveQuote):
        pass  # quotes ignored for this display

    def handle_sigint(sig, frame):
        print("\n\nStopping stream...")
        stream.unsubscribe(symbol, sub_id)
        stop.set()

    signal.signal(signal.SIGINT, handle_sigint)

    stream.subscribe(symbol, on_timesale=on_ts, on_quote=on_quote, sub_id=sub_id)

    # Block until Ctrl+C
    stop.wait()


if __name__ == "__main__":
    main()
