"""
market_data/provider.py

Provider abstraction layer — returns the correct market data client
based on config flags.  Defaults to Tradier; IBKR activates when
``IBKR_ENABLED=true`` in the environment.

Both providers return the same dataclass types (``StockQuote``,
``OHLCVCandle``, ``LiveTimesale``, ``LiveQuote``) so the rest of the
pipeline is data-source-agnostic.

Usage:
    from market_data.provider import get_historical_client, get_shared_stream

    hist   = get_historical_client()   # IBKRHistoricalClient | HistoricalClient
    stream = get_shared_stream()       # IBKRSharedStream     | SharedStreamClient
"""

import logging
import threading

from config import IBKR_ENABLED, TRADIER_ENABLED, TRADIER_STREAM_ENABLED

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# REST / Historical
# ---------------------------------------------------------------------------

_hist_client = None
_hist_client_lock = threading.Lock()
_hist_client_ibkr: bool | None = None  # tracks which provider the singleton was created for


def get_historical_client():
    """Return the process-wide historical-data client (lazy-initialised singleton).

    Returns ``IBKRHistoricalClient`` when ``IBKR_ENABLED`` is true,
    otherwise falls back to the Tradier ``HistoricalClient``.
    """
    global _hist_client, _hist_client_ibkr
    if _hist_client is not None and _hist_client_ibkr == IBKR_ENABLED:
        return _hist_client
    with _hist_client_lock:
        if _hist_client is not None and _hist_client_ibkr == IBKR_ENABLED:
            return _hist_client
        if IBKR_ENABLED:
            from market_data.ibkr_rest import IBKRHistoricalClient
            _hist_client = IBKRHistoricalClient()
        else:
            from market_data.tradier_rest import HistoricalClient
            _hist_client = HistoricalClient()
        _hist_client_ibkr = IBKR_ENABLED
    return _hist_client


# ---------------------------------------------------------------------------
# Streaming / Shared stream
# ---------------------------------------------------------------------------

_ibkr_stream = None
_ibkr_stream_lock = threading.Lock()


def get_shared_stream():
    """Return the process-wide shared streaming client.

    Uses ``IBKRSharedStream`` when ``IBKR_ENABLED`` is true, otherwise
    delegates to the Tradier ``SharedStreamClient`` singleton.
    """
    global _ibkr_stream
    if IBKR_ENABLED:
        with _ibkr_stream_lock:
            if _ibkr_stream is None:
                from market_data.ibkr_stream import IBKRSharedStream
                _ibkr_stream = IBKRSharedStream()
        return _ibkr_stream
    from market_data.tradier_stream import get_shared_stream as _tradier_stream
    return _tradier_stream()


def release_thread_connection() -> None:
    """Release the calling thread's provider connection, if it holds one.

    Call from a worker thread about to exit.  No-op for Tradier (stateless
    HTTP); for IBKR it closes the thread-local socket so the Gateway frees
    the connection slot instead of holding it until process exit.
    """
    if not IBKR_ENABLED:
        return
    try:
        from market_data.ibkr_rest import release_ib
        release_ib()
    except Exception as exc:
        logger.debug("release_thread_connection failed: %s", exc)


def shutdown_market_data() -> None:
    """Stop the shared stream and close all provider connections.

    Call once at session shutdown.  The Gateway caps concurrent API
    connections at 32 and only frees a slot when the socket closes, so
    leaving them open means the next session starts with a reduced pool
    until the Gateway times the stale ones out.
    """
    global _ibkr_stream
    with _ibkr_stream_lock:
        stream = _ibkr_stream
        _ibkr_stream = None
    if stream is not None:
        try:
            stream.stop()
        except Exception as exc:
            logger.debug("shutdown_market_data: stream stop failed: %s", exc)
    if IBKR_ENABLED:
        try:
            from market_data.ibkr_rest import disconnect_all
            disconnect_all()
        except Exception as exc:
            logger.debug("shutdown_market_data: IBKR disconnect failed: %s", exc)


def stream_enabled() -> bool:
    """Return ``True`` if any live streaming provider is configured."""
    if IBKR_ENABLED:
        return True
    return TRADIER_ENABLED and TRADIER_STREAM_ENABLED
