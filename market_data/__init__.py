import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def resilient_session(
    retries: int = 3,
    backoff: float = 0.3,
    status_forcelist: tuple[int, ...] = (502, 503, 504),
    pool_maxsize: int = 32,
) -> requests.Session:
    """Create a ``requests.Session`` with transport-level retry.

    Retries on connection errors (``ConnectionResetError``, DNS failures,
    etc.) and on the HTTP status codes in *status_forcelist*.
    Back-off doubles each attempt: 0.3s → 0.6s → 1.2s by default.

    *pool_maxsize* sets the max keep-alive connections per host.
    The default (32) accommodates multiple scorer threads running
    parallel tick-window fetches through a shared singleton client.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=status_forcelist,
        allowed_methods={"GET", "POST", "PUT", "DELETE"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=pool_maxsize)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


import logging as _logging
import os as _os
import re as _re

class _RedactAPIKeyFilter(_logging.Filter):
    """Strip API keys from urllib3 retry/redirect log messages."""
    _PATTERN = _re.compile(r"(apikey=)[^&\s]+", _re.IGNORECASE)

    def filter(self, record: _logging.LogRecord) -> bool:
        if record.args:
            record.args = tuple(
                self._PATTERN.sub(r"\1[REDACTED]", str(a)) if isinstance(a, str) else a
                for a in record.args
            )
        record.msg = self._PATTERN.sub(r"\1[REDACTED]", str(record.msg))
        return True

_logging.getLogger("urllib3.util.retry").addFilter(_RedactAPIKeyFilter())
_logging.getLogger("urllib3.connectionpool").addFilter(_RedactAPIKeyFilter())


class _IBKRNoiseFilter(_logging.Filter):
    """Drop ib_insync status chatter that repeats per connection or per poll.

    Every one of these is the Gateway reporting that nothing is wrong, and the
    app opens a connection per worker thread and scans every few seconds — so
    unfiltered they run to thousands of lines an hour and bury the errors that
    matter.

    Matched on message text rather than error code on purpose: 162 is shared
    between the benign scanner teardown and genuine historical-data failures
    (including pacing violations), so filtering by code would hide real
    problems.
    """

    # Pure status; nothing actionable ever appears in these.
    _DROP = (
        "scanner subscription cancelled",  # 162 — one-shot reqScannerData teardown
        "farm connection is OK",           # 2104/2106/2158 — per-connect handshake
        # 300 — echo of cancelling a tick sub IBKR already refused (the
        # refusal leaves a registration server-side that never existed, so
        # the defensive cancel always "fails"). Self-inflicted, never news.
        "Can't find EId with tickerId",
        # 10190/309 — quota refusals. IBKRSharedStream._on_ib_error always
        # translates these into one actionable WARNING with cap/backoff
        # context, so the raw copy is a duplicate.
        "Max number of tick-by-tick requests has been reached",
        "of market depth requests has been reached",
    )
    # Real but invariant: the permission set does not vary by symbol, so the
    # tenth copy tells you nothing the first did not.
    _ONCE = (
        "Need additional market data permissions",
    )

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: _logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(marker in msg for marker in self._DROP):
            return False
        for marker in self._ONCE:
            if marker in msg:
                if marker in self._seen:
                    return False
                self._seen.add(marker)
        return True


# Connect/disconnect narration ("Connecting to...", "Logged on to server
# version", "API connection ready", "Synchronization complete", "Disconnecting
# from...") is INFO on these two loggers — ~8 lines every time any thread opens
# a connection. Errors still come through, since only INFO is suppressed.
# Set IBKR_LOG_VERBOSE=true to restore it when debugging connection problems.
if _os.getenv("IBKR_LOG_VERBOSE", "false").lower() != "true":
    _logging.getLogger("ib_insync.client").setLevel(_logging.WARNING)
    _logging.getLogger("ib_insync.ib").setLevel(_logging.WARNING)

_logging.getLogger("ib_insync.wrapper").addFilter(_IBKRNoiseFilter())
