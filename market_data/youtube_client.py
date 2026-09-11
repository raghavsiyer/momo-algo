"""
market_data/youtube_client.py

YouTube Live Chat scraper — extracts ticker mentions and news headlines
from a live stream's chat feed using YouTube's internal InnerTube API.

No API key required. No browser. ~1 second latency per poll.

Usage:
    from market_data.youtube_client import YouTubeLiveChatClient

    client = YouTubeLiveChatClient("https://www.youtube.com/watch?v=8IY27unUzmg")
    # One-shot: grab all current chat messages
    messages = client.poll()

    # Extract tickers mentioned in chat
    tickers = client.extract_tickers(messages)
"""

import logging
import re
import json

import orjson
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from config import HTTP_TIMEOUT, MarketClock

# Per-day cache of resolved live-stream URLs.  The same broadcast runs
# for the full ET trading day, so once we resolve it we can skip the
# scrape until midnight ET.
_STREAM_CACHE_PATH = Path(__file__).resolve().parent.parent / "logs" / "youtube_stream_cache.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex for ticker extraction: $AAPL or standalone 3-4 letter uppercase words
# that look like tickers (preceded by whitespace/start, not part of a sentence).
# Only 3-4 letter symbols are matched from chat: 1-2 letters (F, GM, T) are
# indistinguishable from ordinary words and dollar amounts, and the traded
# universe here never uses 5-letter symbols.
# ---------------------------------------------------------------------------
_CASHTAG_RE = re.compile(r"\$([A-Z]{3,4})\b")
_BARE_TICKER_RE = re.compile(r"(?<!\w)([A-Z]{3,4})(?![A-Za-z])")
# Lowercase-tolerant cashtag ("$rklb") and the "TICKER (note) - headline" prefix
# form, both accepted at the *start* of a news post.
_LOWER_CASHTAG_RE = re.compile(r"\$([A-Za-z]{3,4})\b")
_PREFIX_TICKER_RE = re.compile(r"([A-Za-z]{3,4})\b\s*(?:\([^)]*\)\s*)?-\s*\S")

# Common English words that look like tickers — filter these out
_STOPWORDS = frozenset({
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "LET", "MAY",
    "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "GET", "HIM",
    "HIT", "HOT", "LOW", "MAN", "RUN", "TOO", "USE", "BIG", "BUY", "END",
    "FAR", "FEW", "GOT", "HAD", "HAS", "SAY", "SHE", "TRY", "WHY", "YES",
    "YET", "ADD", "AGO", "AID", "AIM", "AIR", "ARM", "BAD", "BAG", "BED",
    "BIT", "BOX", "BUS", "CUT", "DAY", "DOG", "EAR", "EAT", "ERA", "EYE",
    "FAN", "FIT", "FLY", "GAS", "GUN", "GUY", "HAT", "ICE", "ILL", "JOB",
    "KEY", "LAW", "LAY", "LED", "LIE", "LOG", "LOT", "MAP", "MET", "MIX",
    "NET", "NOR", "ODD", "PAY", "PEN", "PER", "PIT", "PUT", "RAW", "RED",
    "RID", "ROW", "SAD", "SAT", "SET", "SIT", "SIX", "SKI", "SUM", "TEN",
    "TIE", "TIP", "TOP", "TWO", "VAN", "WAR", "WET", "WON", "WON", "YET",
    "ALSO", "AREA", "BACK", "BEEN", "BEST", "BODY", "BOOK", "BOTH", "CALL",
    "CAME", "COME", "COULD", "EACH", "EVEN", "FACE", "FACT", "FEEL", "FIND",
    "FORM", "FROM", "FULL", "GAVE", "GIVE", "GOOD", "HAVE", "HEAD", "HELP",
    "HERE", "HIGH", "HOME", "HOPE", "IDEA", "INTO", "JUST", "KEEP", "KIND",
    "KNEW", "KNOW", "LAND", "LAST", "LEFT", "LESS", "LIFE", "LIKE", "LINE",
    "LIST", "LIVE", "LONG", "LOOK", "LORD", "LOSE", "LOVE", "MADE", "MAKE",
    "MANY", "MEAN", "MIND", "MORE", "MOST", "MUCH", "MUST", "NAME", "NEED",
    "NEWS", "NEXT", "NONE", "ONCE", "ONLY", "OPEN", "OVER", "PART", "PAST",
    "PLAN", "PLAY", "PULL", "PUSH", "PUTS", "RATE", "READ", "REAL", "REST",
    "ROAD", "ROLE", "ROOM", "RULE", "RUNS", "SAID", "SAME", "SEEN", "SELL",
    "SEND", "SHOW", "SIDE", "SIGN", "SING", "SIZE", "SKIN", "SOME", "SOON",
    "SORT", "STAY", "STEP", "STOP", "SUCH", "SURE", "TAKE", "TALK", "TELL",
    "TEND", "TERM", "TEST", "TEXT", "THAN", "THAT", "THEM", "THEN", "THEY",
    "THIS", "THUS", "TIME", "TOLD", "TOOK", "TURN", "TYPE", "UNIT", "UPON",
    "USED", "VERY", "VOTE", "WAIT", "WALK", "WALL", "WANT", "WEEK", "WELL",
    "WENT", "WERE", "WHAT", "WHEN", "WILL", "WITH", "WORD", "WORK", "YEAR",
    "YOUR", "ZERO", "BULL", "BEAR", "HOLD", "LONG", "PUMP", "DUMP", "MOON",
    "HODL", "YOLO", "FOMO", "HUGE", "LETS", "NICE", "HOLY", "LMAO", "YEAH",
    "CHAT", "GUYS", "HAHA", "WHAT", "OMFG", "JUST", "LIKE", "IMHO", "TRUE",
    "FAKE", "EASY", "HARD", "GOES", "GONE", "HALF", "HAND", "HANG", "RISK",
    "RISE", "DROP", "DOWN", "GAIN", "LOSS", "MOVE", "SHIP", "CASH", "FUND",
    "BANK", "WORDS", "STOCK", "SHARE", "STILL", "ABOUT", "AFTER", "BEING",
    "COULD", "EVERY", "FIRST", "GREAT", "HEDGE", "LARGE", "LIMIT", "ORDER",
    "OTHER", "POINT", "PRICE", "RIGHT", "SHALL", "SHORT", "SINCE", "SMALL",
    "START", "THEIR", "THERE", "THESE", "THING", "THINK", "THREE", "TODAY",
    "TOTAL", "TRADE", "UNDER", "UNTIL", "VALUE", "WHERE", "WHICH", "WHILE",
    "WORLD", "WOULD", "BREAK", "CLOSE", "COULD", "GREEN", "NEVER", "RALLY",
    "WATCH", "WHOLE", "TICKE", "MONEY", "CHECK", "STICK", "GONNA", "LOOKS",
    "REALLY", "SUPER",
})


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NEWS_AUTHOR = "@Clementine\u30fc"   # Only this user posts structured news headlines

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """Single YouTube live chat message."""
    message_id: str
    author: str
    text: str
    timestamp: datetime


@dataclass
class ChatNewsHeadline:
    """A structured news headline extracted from a chat message."""
    ticker: str
    headline: str
    author: str
    timestamp: datetime
    bypass_noise_filter: bool = False


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

def extract_ticker_from_text(text: str) -> tuple[Optional[str], str, str]:
    """Pull a ticker out of a chat message.

    Returns ``(ticker, method, remainder)`` where *method* is one of:

    ``"cashtag"``
        Message opens with ``$AAPL`` / ``$rklb``.  *remainder* is everything
        after the cashtag token.
    ``"prefix"``
        Message opens with ``AAPL - ...`` or ``AAPL (weak open) - ...``.
        *remainder* is everything after the ticker word.
    ``"bare"``
        No leading form matched, but a cashtag or a bare uppercase ticker
        appears somewhere in the text.  *remainder* is the full text — the
        ticker is not a structured prefix, so nothing can be stripped.
    ``"none"``
        Nothing ticker-like found; *ticker* is ``None``.

    Stopwords are filtered in every branch, so ``$TEST`` / ``TEST - ...``
    never resolve to a ticker.  Only the two leading forms (``cashtag`` and
    ``prefix``) carry a usable *remainder*; callers that need the strict
    ``TICKER - headline`` shape should accept those two methods only.
    """
    text = (text or "").strip()
    if not text:
        return None, "none", ""

    # Leading cashtag: "$AAPL - ..." or lowercase "$rklb - ..."
    m = _CASHTAG_RE.match(text)
    if not m:
        m = _LOWER_CASHTAG_RE.match(text)
    if m:
        ticker = m.group(1).upper()
        if ticker not in _STOPWORDS:
            return ticker, "cashtag", text[m.end():]
    else:
        # Leading bare ticker (upper or lower).  The dash delimiter is
        # required so random opening words aren't treated as tickers.
        m = _PREFIX_TICKER_RE.match(text)
        if m and m.group(1).upper() not in _STOPWORDS:
            return m.group(1).upper(), "prefix", text[len(m.group(1)):]

    # Fallback: a ticker mentioned anywhere in the message body.
    for regex in (_CASHTAG_RE, _BARE_TICKER_RE):
        for hit in regex.finditer(text):
            word = hit.group(1)
            if word not in _STOPWORDS:
                return word, "bare", text

    return None, "none", text


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class YouTubeLiveChatClient:
    """
    YouTube Live Chat client using the internal InnerTube API.
    No API key needed — just a live stream URL.
    """

    def __init__(self, stream_url: str) -> None:
        self._url = stream_url
        self._video_id = self._extract_video_id(stream_url)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })
        # Bypass YouTube consent wall (SOCS = accepted, YES+ = legacy fallback)
        self._session.cookies.set("SOCS", "CAISNQgDEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjQwNTI4LjA4X3AxGgJlbiADGgYIgOqxtwY", domain=".youtube.com")
        self._session.cookies.set("CONSENT", "YES+cb.20210328-17-p0.en+FX+999", domain=".youtube.com")
        self._session.cookies.set("PREF", "tz=America.New_York", domain=".youtube.com")
        self._api_key: Optional[str] = None
        self._client_version: Optional[str] = None
        self._visitor_data: Optional[str] = None
        self._continuation: Optional[str] = None
        self._seen: dict[str, None] = {}    # insertion-ordered dedup (bounded)
        self._SEEN_MAX = 100_000             # trim threshold
        self._SEEN_KEEP = 50_000             # keep most recent N after trim
        self._initialized = False
        self._timeout_ms: int = 0  # YouTube-recommended poll interval from last response

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_video_id(url: str) -> str:
        """Extract video ID from a YouTube URL."""
        # Handle youtu.be/ID and youtube.com/watch?v=ID formats
        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if match:
            return match.group(1)
        raise ValueError(f"Could not extract video ID from: {url}")

    def _init_chat(self) -> tuple[bool, list[ChatMessage]]:
        """
        Fetch the live chat iframe page, extract the InnerTube API key,
        client version, and initial continuation token.
        Returns (success, initial_messages).
        """
        # Clear stale YouTube cookies from previous failed attempts so each
        # init starts fresh.  Re-set the consent cookies that bypass the wall.
        self._session.cookies.clear(domain=".youtube.com")
        self._session.cookies.set("SOCS", "CAISNQgDEitib3FfaWRlbnRpdHlmcm9udGVuZHVpc2VydmVyXzIwMjQwNTI4LjA4X3AxGgJlbiADGgYIgOqxtwY", domain=".youtube.com")
        self._session.cookies.set("CONSENT", "YES+cb.20210328-17-p0.en+FX+999", domain=".youtube.com")

        url = f"https://www.youtube.com/live_chat?v={self._video_id}&is_popout=1"
        try:
            resp = self._session.get(url, timeout=HTTP_TIMEOUT)
            if not resp.ok:
                logger.warning(f"YouTube live chat page returned {resp.status_code}")
                return False, []
            html = resp.text
        except requests.exceptions.RequestException as exc:
            logger.warning(f"YouTube live chat page fetch failed: {exc}")
            return False, []

        # Extract API key
        api_match = re.search(r'"INNERTUBE_API_KEY":"(.*?)"', html)
        if not api_match:
            logger.warning("Could not find INNERTUBE_API_KEY in page")
            return False, []
        self._api_key = api_match.group(1)

        # Extract client version (use the live version, not hardcoded)
        ver_match = re.search(r'"clientVersion":"([\d.]+)"', html)
        self._client_version = ver_match.group(1) if ver_match else "2.20260408.00.00"

        # Extract visitor data
        vis_match = re.search(r'"visitorData":"(.*?)"', html)
        self._visitor_data = vis_match.group(1) if vis_match else None

        # Extract ytInitialData — chat page uses window["ytInitialData"] format
        data_match = re.search(
            r'window\["ytInitialData"\]\s*=\s*({.*?});\s*</script>',
            html, re.DOTALL,
        )
        if not data_match:
            # Fallback to bare assignment
            data_match = re.search(r"ytInitialData\s*=\s*({.*?});", html, re.DOTALL)
        if not data_match:
            logger.warning("Could not find ytInitialData in live chat page")
            return False, []

        try:
            yt_data = orjson.loads(data_match.group(1))
        except (orjson.JSONDecodeError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to parse ytInitialData: {exc}")
            return False, []

        # Navigate to liveChatRenderer (chat page structure)
        contents = yt_data.get("contents", {})
        lc = contents.get("liveChatRenderer")
        if lc is None:
            contents_keys = list(contents.keys()) if isinstance(contents, dict) else "N/A"
            logger.warning(
                f"Could not find liveChatRenderer — stream may not be live "
                f"(contents keys: {contents_keys})"
            )
            return False, []

        # Extract continuation token
        for cont in lc.get("continuations", []):
            for key in ("invalidationContinuationData", "timedContinuationData",
                        "reloadContinuationData"):
                if key in cont:
                    self._continuation = cont[key].get("continuation")
                    break
            if self._continuation:
                break

        if not self._continuation:
            logger.warning("No continuation token found — stream may not be live")
            return False, []

        # Parse initial messages from the page
        initial_msgs = self._parse_actions(lc.get("actions", []))

        self._initialized = True
        print(flush=True)
        logger.info(
            f"YouTube live chat initialized for video {self._video_id} "
            f"({len(initial_msgs)} initial messages)"
        )
        return True, initial_msgs

    # ------------------------------------------------------------------
    # Chat polling
    # ------------------------------------------------------------------

    def _parse_actions(self, actions: list) -> list[ChatMessage]:
        """Parse chat actions into ChatMessage objects, deduplicating."""
        messages: list[ChatMessage] = []
        for action in actions:
            item = action.get("addChatItemAction", {}).get("item", {})
            renderer = item.get("liveChatTextMessageRenderer")
            if not renderer:
                continue

            msg_id = renderer.get("id", "")
            if msg_id in self._seen:
                continue
            self._seen[msg_id] = None
            # Trim oldest entries when the set grows too large
            if len(self._seen) > self._SEEN_MAX:
                keys = list(self._seen)
                for k in keys[:len(keys) - self._SEEN_KEEP]:
                    del self._seen[k]

            text = "".join(
                run.get("text", "")
                for run in renderer.get("message", {}).get("runs", [])
            )
            author = renderer.get("authorName", {}).get("simpleText", "")

            # Parse timestamp
            ts_usec = renderer.get("timestampUsec", "0")
            try:
                ts = datetime.fromtimestamp(int(ts_usec) / 1_000_000, tz=timezone.utc)
            except (ValueError, OSError):
                ts = datetime.now(tz=timezone.utc)

            messages.append(ChatMessage(
                message_id=msg_id,
                author=author,
                text=text,
                timestamp=ts,
            ))
        return messages

    def poll(self) -> list[ChatMessage]:
        """
        Poll for new chat messages. On first call, initialises the
        connection and returns initial messages. Subsequent calls return
        only *new* messages (deduped internally). Returns [] on failure.
        """
        if not self._initialized:
            ok, initial = self._init_chat()
            if not ok:
                return []
            return initial

        url = (
            f"https://www.youtube.com/youtubei/v1/live_chat/get_live_chat"
            f"?key={self._api_key}"
        )
        payload = {
            "context": {
                "client": {
                    "clientName": "WEB",
                    "clientVersion": self._client_version,
                }
            },
            "continuation": self._continuation,
        }
        if self._visitor_data:
            payload["context"]["client"]["visitorData"] = self._visitor_data

        try:
            resp = self._session.post(url, json=payload, timeout=HTTP_TIMEOUT)
            if not resp.ok:
                logger.warning(f"YouTube live chat API returned {resp.status_code}")
                return []
            data = resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning(f"YouTube live chat poll failed: {exc}")
            return []
        except json.JSONDecodeError:
            logger.warning("YouTube live chat returned invalid JSON")
            return []

        chat = data.get("continuationContents", {}).get("liveChatContinuation", {})

        # Update continuation token and poll timing hint for next poll
        for cont in chat.get("continuations", []):
            if "invalidationContinuationData" in cont:
                self._continuation = cont["invalidationContinuationData"]["continuation"]
                self._timeout_ms = cont["invalidationContinuationData"].get("timeoutMs", 0)
                break
            if "timedContinuationData" in cont:
                self._continuation = cont["timedContinuationData"]["continuation"]
                self._timeout_ms = cont["timedContinuationData"].get("timeoutMs", 0)
                break

        return self._parse_actions(chat.get("actions", []))

    # ------------------------------------------------------------------
    # Ticker extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_tickers(messages: list[ChatMessage]) -> dict[str, int]:
        """
        Scan chat messages for ticker mentions. Returns a dict of
        {ticker: mention_count} sorted by count descending.

        Recognises:
          - Cashtags: $AAPL, $TSLA
          - Bare uppercase words (3-4 chars) that aren't common English
        """
        counts: dict[str, int] = {}
        for msg in messages:
            found: set[str] = set()

            # Cashtags (high confidence) — still filter obvious English/noise
            # words so $TEST, $THE, etc. don't leak through as fake tickers.
            for m in _CASHTAG_RE.finditer(msg.text):
                word = m.group(1)
                if word not in _STOPWORDS:
                    found.add(word)

            # Bare uppercase words (filter stopwords)
            for m in _BARE_TICKER_RE.finditer(msg.text):
                word = m.group(1)
                if word not in _STOPWORDS:
                    found.add(word)

            for ticker in found:
                counts[ticker] = counts.get(ticker, 0) + 1

        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # News headline extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_news_headlines(
        messages: list[ChatMessage],
        author: str = NEWS_AUTHOR,
    ) -> list[ChatNewsHeadline]:
        """Extract structured news headlines from the designated news author.

        Expected format from *author*:  ``TICKER - headline text``
        or ``$TICKER - headline text``.  Uses the same ticker detection
        regex as :meth:`extract_tickers`.  Messages from other users or
        messages that don't match the format are silently skipped.

        Returns a list of :class:`ChatNewsHeadline`.
        """
        headlines: list[ChatNewsHeadline] = []
        for msg in messages:
            if msg.author != author:
                continue

            text = msg.text.strip()
            if not text:
                continue

            # Leading "$AAPL ..." / "$rklb ..." / "AAPL (note) - ..." only —
            # a ticker merely mentioned mid-sentence ("bare") is not a
            # structured news post.
            ticker, method, rest = extract_ticker_from_text(text)
            if ticker is None or method not in ("cashtag", "prefix"):
                continue

            # Expect " - " delimiter between ticker and headline,
            # optionally preceded by a parenthetical annotation:
            # "AAPL - ..." or "AAPL (weak open) - ..."
            rest = rest.lstrip()
            if rest.startswith("("):
                # Skip over the parenthetical: "(text) - ..."
                close = rest.find(")")
                if close == -1:
                    continue
                rest = rest[close + 1:].lstrip()

            if not rest.startswith("-"):
                continue

            headline = rest[1:].strip()
            if not headline:
                continue

            headlines.append(ChatNewsHeadline(
                ticker=ticker,
                headline=headline,
                author=msg.author,
                timestamp=msg.timestamp,
                bypass_noise_filter=True,
            ))
        return headlines

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def video_id(self) -> str:
        return self._video_id

    @property
    def poll_timeout_ms(self) -> int:
        """Milliseconds YouTube recommends waiting before the next poll.
        0 means no hint received yet — fall back to the configured interval."""
        return self._timeout_ms

    @property
    def is_connected(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Live stream auto-detection
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_live_stream(channel_handle: str) -> Optional[str]:
        """Resolve a YouTube channel handle to its current live stream URL.

        Cached per (handle, ET trading date): the same broadcast runs all day,
        so once we resolve it we reuse the URL until midnight ET.

        Resolution strategy (first hit wins, then "prefer non-silent" tiebreak):
          1. Scrape ``/@handle/featured`` and walk ``channelFeaturedContentRenderer``
             items (new ``lockupViewModel`` schema introduced 2026).  Surfaces
             *all* concurrent broadcasts.
          2. Fall back to the legacy InnerTube ``/streams`` browse with
             ``thumbnailOverlayTimeStatusRenderer.style == "LIVE"`` walking.

        Returns ``https://www.youtube.com/watch?v=<id>`` or None.
        """
        handle = channel_handle.lstrip("@")
        today_et = MarketClock.now_et().date().isoformat()

        cached = _read_stream_cache().get(handle.lower())
        if cached and cached.get("date") == today_et and cached.get("url"):
            logger.info("[youtube] Using cached live stream for @%s → %s", handle, cached["url"])
            return cached["url"]

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })
        session.cookies.set("CONSENT", "YES+1", domain=".youtube.com")
        session.cookies.set("SOCS", "CAI", domain=".youtube.com")

        streams = _discover_live_from_featured(session, handle)
        if not streams:
            streams = _discover_live_from_innertube_streams(session, handle)

        if not streams:
            logger.warning("[youtube] No live stream found for @%s", handle)
            return None

        # Prefer the broadcast whose title doesn't advertise a silent feed.
        chat_streams = [(v, t) for v, t in streams if "silent" not in t.lower()]
        chosen_id, chosen_title = (chat_streams[0] if chat_streams else streams[0])
        video_url = f"https://www.youtube.com/watch?v={chosen_id}"

        _write_stream_cache_entry(handle.lower(), today_et, video_url)
        logger.info("[youtube] Resolved @%s → %s  (%s)", handle, video_url, chosen_title[:60])
        return video_url


# ---------------------------------------------------------------------------
# Live-stream resolver helpers (module level — no shared session state)
# ---------------------------------------------------------------------------

def _read_stream_cache() -> dict:
    try:
        with _STREAM_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_stream_cache_entry(handle_key: str, date_iso: str, url: str) -> None:
    cache = _read_stream_cache()
    cache[handle_key] = {"date": date_iso, "url": url}
    try:
        _STREAM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _STREAM_CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except OSError as exc:
        logger.warning("[youtube] Failed to write stream cache: %s", exc)


def _discover_live_from_featured(session: requests.Session, handle: str) -> list[tuple[str, str]]:
    """Parse the channel's featured-tab 'N live streams' shelf.

    Handles the 2026 ``lockupViewModel`` schema where each item carries a
    ``THUMBNAIL_OVERLAY_BADGE_STYLE_LIVE`` badge.  Falls through silently
    if the page format changes again.
    """
    try:
        resp = session.get(f"https://www.youtube.com/@{handle}/featured", timeout=HTTP_TIMEOUT)
        if not resp.ok:
            return []
        m = re.search(r'var ytInitialData\s*=\s*({.+?});</script>', resp.text)
        if not m:
            return []
        data = json.loads(m.group(1))
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return []

    shelves: list[dict] = []
    def _find_shelves(o: object) -> None:
        if isinstance(o, dict):
            if "channelFeaturedContentRenderer" in o:
                shelves.append(o["channelFeaturedContentRenderer"])
            for v in o.values():
                _find_shelves(v)
        elif isinstance(o, list):
            for it in o:
                _find_shelves(it)
    _find_shelves(data)

    results: list[tuple[str, str]] = []
    for shelf in shelves:
        for item in shelf.get("items", []):
            lvm = item.get("lockupViewModel")
            if lvm:
                blob = json.dumps(lvm)
                if "THUMBNAIL_OVERLAY_BADGE_STYLE_LIVE" not in blob:
                    continue
                vid = lvm.get("contentId")
                if not vid:
                    bm = re.search(r'"animationActivationTargetId":"([\w-]{11})"', blob)
                    vid = bm.group(1) if bm else None
                title = ""
                meta = lvm.get("metadata", {}).get("lockupMetadataViewModel", {})
                t = meta.get("title")
                if isinstance(t, dict):
                    title = t.get("content", "") or ""
                if vid:
                    results.append((vid, title))
                continue
            # Legacy videoRenderer shape (old shelf layout)
            vr = item.get("videoRenderer") or {}
            vid = vr.get("videoId")
            if not vid:
                continue
            overlays = vr.get("thumbnailOverlays", [])
            is_live = any(
                ov.get("thumbnailOverlayTimeStatusRenderer", {}).get("style") == "LIVE"
                for ov in overlays
            ) or "LIVE_NOW" in str(vr.get("badges", []))
            if not is_live:
                continue
            runs = vr.get("title", {}).get("runs", [])
            results.append((vid, runs[0].get("text", "") if runs else ""))

    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for vid, title in results:
        if vid in seen:
            continue
        seen.add(vid)
        uniq.append((vid, title))
    return uniq


def _discover_live_from_innertube_streams(session: requests.Session, handle: str) -> list[tuple[str, str]]:
    """Legacy fallback: InnerTube browse on the channel's /streams tab."""
    try:
        page = session.get(f"https://www.youtube.com/@{handle}", timeout=HTTP_TIMEOUT)
        if not page.ok:
            return []
    except requests.exceptions.RequestException:
        return []

    cid_match = re.search(r'"browseId"\s*:\s*"(UC[A-Za-z0-9_-]+)"', page.text)
    if not cid_match:
        return []
    channel_id = cid_match.group(1)

    payload = {
        "context": {"client": {"clientName": "WEB", "clientVersion": "2.20260601.00.00"}},
        "browseId": channel_id,
        "params": "EgZzdHJlYW1z",  # streams tab
    }
    try:
        resp = session.post("https://www.youtube.com/youtubei/v1/browse", json=payload, timeout=HTTP_TIMEOUT)
        if not resp.ok:
            return []
        data = resp.json()
    except (requests.exceptions.RequestException, json.JSONDecodeError):
        return []

    results: list[tuple[str, str]] = []
    def _walk(o: object, depth: int = 0) -> None:
        if depth > 20:
            return
        if isinstance(o, dict):
            vid = o.get("videoId")
            if vid:
                live = False
                for ov in o.get("thumbnailOverlays", []):
                    if ov.get("thumbnailOverlayTimeStatusRenderer", {}).get("style") == "LIVE":
                        live = True
                        break
                if not live and "LIVE_NOW" in str(o.get("badges", [])):
                    live = True
                if live:
                    runs = o.get("title", {}).get("runs", [])
                    results.append((vid, runs[0].get("text", "") if runs else ""))
            for v in o.values():
                _walk(v, depth + 1)
        elif isinstance(o, list):
            for it in o:
                _walk(it, depth + 1)
    _walk(data)

    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for vid, title in results:
        if vid in seen:
            continue
        seen.add(vid)
        uniq.append((vid, title))
    return uniq
