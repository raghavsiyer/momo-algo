"""
market_data/modernbert_client.py

Local zero-shot news detection for YouTube live-chat messages.

Mirrors :mod:`market_data.finbert_client` in shape (lazy load under a lock,
local-first snapshot download, device auto-detect, background preload) but
answers a different question: *is this chat message a news headline?*

The checkpoint is ``MoritzLaurer/ModernBERT-base-zeroshot-v2.0`` — an NLI
model usable through ``pipeline("zero-shot-classification")``, so no
training or labelled data is needed.  Two competing hypotheses are scored
per message and softmaxed against each other:

    "This live chat message is {a news headline reporting a factual event
     about a company}."
    "This live chat message is {a personal opinion, reaction, or chit-chat}."

The news label's share of that softmax is the score; anything at or above
the configured threshold is flagged as news.

The contrast label costs a second forward pass (~250 ms/msg on CPU vs.
~150 ms for one), and it is worth every millisecond: measured on
hand-written headlines, a *single* hypothesis scored raw entailment gives
real headlines anywhere from 0.06 to 0.94 — only 2 of 8 clear the 0.85
threshold — because raw NLI entailment is not calibrated across sentences.
Softmaxing against an explicit "chit-chat" alternative turns it into the
relative question the task actually asks, and separates cleanly: news
0.90-0.99, chatter 0.01-0.42.  Set the contrast label to "" to fall back
to the single-hypothesis form.

Raw scores are logged by the shadow pipeline so the threshold and both
label wordings can be re-tuned offline without touching the model.

Usage:
    from market_data.modernbert_client import ModernBERTNewsClassifier
    clf = ModernBERTNewsClassifier()
    clf.preload()                       # optional, background
    verdicts = clf.classify_batch(["MSFT - beats on earnings", "lol"])
    # verdicts: list[NewsVerdict | None], same length/order as the input
"""

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

from config import (
    SCANNER_YT_MODERNBERT_CONTRAST_LABEL,
    SCANNER_YT_MODERNBERT_HYPOTHESIS,
    SCANNER_YT_MODERNBERT_LABEL,
    SCANNER_YT_MODERNBERT_MIN_WORDS,
    SCANNER_YT_MODERNBERT_MODEL,
    SCANNER_YT_MODERNBERT_THRESHOLD,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = SCANNER_YT_MODERNBERT_MODEL   # ~150M params, ~600 MB fp32

_MAX_BATCH = 64          # texts per forward pass
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def alpha_word_count(text: str) -> int:
    """Number of alphabetic words (2+ letters) in *text*.

    Used as the cheap pre-filter: "lol", "$AAPL", "🚀🚀" carry no headline
    and shouldn't cost a forward pass.
    """
    if not text:
        return 0
    return len(_ALPHA_WORD_RE.findall(text))


@dataclass
class NewsVerdict:
    """One classification result."""
    is_news: bool
    score: float                # entailment probability for the news hypothesis
    latency_ms: float = 0.0     # wall time of the batch this verdict came from


class ModernBERTNewsClassifier:
    """Local zero-shot news detector.

    The model is loaded lazily on first classification, or eagerly in the
    background via :meth:`preload`.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        threshold: Optional[float] = None,
        hypothesis: Optional[str] = None,
        label: Optional[str] = None,
        contrast_label: Optional[str] = None,
        min_words: Optional[int] = None,
    ) -> None:
        self.model = model or _DEFAULT_MODEL
        self.threshold = (
            SCANNER_YT_MODERNBERT_THRESHOLD if threshold is None else float(threshold)
        )
        self.hypothesis = hypothesis or SCANNER_YT_MODERNBERT_HYPOTHESIS
        self.label = label or SCANNER_YT_MODERNBERT_LABEL
        self.contrast_label = (
            SCANNER_YT_MODERNBERT_CONTRAST_LABEL if contrast_label is None
            else contrast_label
        )
        self.min_words = (
            SCANNER_YT_MODERNBERT_MIN_WORDS if min_words is None else int(min_words)
        )
        self._pipe = None
        self._lock = threading.Lock()

    @property
    def labels(self) -> list[str]:
        """Candidate labels, news first."""
        return [self.label, self.contrast_label] if self.contrast_label else [self.label]

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def preload(self) -> None:
        """Start loading the model in a background thread."""
        threading.Thread(
            target=self._ensure_model, daemon=True, name="modernbert-preload",
        ).start()

    def _ensure_model(self) -> None:
        with self._lock:
            if self._pipe is not None:
                return
            import torch
            from huggingface_hub import snapshot_download
            from transformers import logging as tf_logging
            from transformers import pipeline

            logger.info("Loading ModernBERT news classifier…")
            try:
                path = snapshot_download(self.model, local_files_only=True)
            except FileNotFoundError:
                logger.info("Model not cached — downloading (~600 MB)…")
                path = snapshot_download(self.model)

            # Auto-detect best available device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"          # Apple Silicon GPU
            else:
                device = "cpu"
            logger.info("ModernBERT device: %s", device)

            tf_logging.set_verbosity_error()
            self._pipe = pipeline(
                "zero-shot-classification", model=path, tokenizer=path, device=device,
            )
            tf_logging.set_verbosity_warning()

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_batch(self, texts: list[str]) -> list[Optional[NewsVerdict]]:
        """Classify several messages at once.

        Returns a list the same length and order as *texts*.  Entries are
        ``None`` when the text was skipped by the min-words guard or when
        the model call failed — callers must treat ``None`` as "unknown",
        never as "not news".
        """
        if not texts:
            return []

        verdicts: list[Optional[NewsVerdict]] = [None] * len(texts)
        # Positions worth a forward pass
        idxs = [
            i for i, t in enumerate(texts)
            if t and alpha_word_count(t) >= self.min_words
        ]
        if not idxs:
            return verdicts

        labels = self.labels
        for start in range(0, len(idxs), _MAX_BATCH):
            chunk_idxs = idxs[start:start + _MAX_BATCH]
            chunk = [texts[i] for i in chunk_idxs]
            t0 = time.perf_counter()
            try:
                self._ensure_model()
                import torch
                with torch.inference_mode():
                    raw = self._pipe(
                        chunk,
                        candidate_labels=labels,
                        hypothesis_template=self.hypothesis,
                        # Softmax the news label against the contrast label;
                        # with no contrast label, fall back to raw entailment.
                        multi_label=len(labels) == 1,
                    )
            except Exception as exc:
                logger.warning("ModernBERT classification error: %s", exc)
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # A single-sequence call can come back as a bare dict.
            if isinstance(raw, dict):
                raw = [raw]
            if not isinstance(raw, list) or len(raw) != len(chunk):
                logger.warning(
                    "ModernBERT returned %s results for %d inputs — dropping batch",
                    type(raw).__name__ if not isinstance(raw, list) else len(raw),
                    len(chunk),
                )
                continue

            for pos, result in zip(chunk_idxs, raw):
                try:
                    score = float(self._score_of(result))
                except Exception:
                    continue
                verdicts[pos] = NewsVerdict(
                    is_news=score >= self.threshold,
                    score=round(score, 4),
                    latency_ms=round(elapsed_ms, 1),
                )

        return verdicts

    def classify(self, text: str) -> Optional[NewsVerdict]:
        """Classify a single message.  ``None`` on skip/failure."""
        return self.classify_batch([text])[0] if text else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_of(self, result: dict) -> float:
        """Pull the news label's probability out of one pipeline result.

        The pipeline returns labels sorted by score, so match by name —
        ``scores[0]`` would be the contrast label whenever chatter wins.
        """
        labels = result.get("labels") or []
        scores = result.get("scores") or []
        for lbl, sc in zip(labels, scores):
            if lbl == self.label:
                return sc
        return scores[0]
