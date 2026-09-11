"""
market_data/chat_logger.py

Append-only JSONL log of YouTube live-chat messages and their ModernBERT
shadow-mode verdicts.

Chat isn't persisted anywhere else, so this file is both:
  * the comparison dataset for ``yt_shadow_report.py`` (zero-shot detections
    vs. the hard-coded Clementine matcher), and
  * a labelled corpus for a later optional fine-tune — Clementine hits are
    weak positives.

One JSON object per raw chat message, one file per ET date:

    logs/yt_chat/chat_YYYYMMDD.jsonl

Writes are open-append-close per batch under a small lock: the volume is a
handful of messages every few seconds, and never holding the handle keeps
the file safe to tail/copy mid-session.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from config import MarketClock

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "logs" / "yt_chat"


class ChatShadowLogger:
    """Appends chat/shadow records to a per-ET-day JSONL file."""

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self.log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_DIR
        self._lock = threading.Lock()

    def path_for_today(self) -> Path:
        """Current day's log file (ET date, matching the trading session)."""
        return self.log_dir / f"chat_{MarketClock.now_et().strftime('%Y%m%d')}.jsonl"

    def log_records(self, records: list[dict]) -> None:
        """Append *records* as JSONL.  Never raises — logging is best-effort."""
        if not records:
            return
        path = self.path_for_today()
        try:
            with self._lock:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    for rec in records:
                        fh.write(json.dumps(rec, ensure_ascii=False, default=str))
                        fh.write("\n")
        except (OSError, TypeError, ValueError) as exc:
            logger.debug("[chat-log] Write failed (%s): %s", path, exc)
