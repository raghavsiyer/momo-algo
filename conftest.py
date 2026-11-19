# conftest.py — project root marker for pytest.
#
# Tests must never inherit live-trading feature flags from the developer's
# .env: with IBKR_ENABLED=true, any code path that reaches
# provider.get_shared_stream() un-patched constructs a REAL IBKRSharedStream
# (its __init__ auto-starts the stream thread) and may even try to connect
# to a live IB gateway mid-suite. Overriding here wins because load_dotenv()
# never overrides pre-existing os.environ entries, and pytest imports
# conftest before any test module imports config.
import os

os.environ["IBKR_ENABLED"] = "false"
# Same rationale: with the dev .env's TRADIER_STREAM_ENABLED=true, any test
# that reaches DynamicScorer.start()/get_shared_stream() un-patched opens a
# REAL Tradier WebSocket session (observed as 401 warnings during shutdown).
# Tests that need stream-enabled behavior must patch stream_enabled/flags
# explicitly.
os.environ["TRADIER_STREAM_ENABLED"] = "false"
# Same again for the merged tape: a real ~/.scalpfeed/tastytrade.json on the
# dev machine would otherwise let a test that runs the stream thread open a
# live tastytrade DXLink socket. Tests drive the tape with fakes instead.
os.environ["TASTYTRADE_TAPE_ENABLED"] = "false"
