"""
market_data/finbert_client.py

Local FinBERT sentiment scoring for news headlines.

Local sentiment scorer — no API key needed.
Uses ProsusAI/finbert (downloaded on first call, ~420 MB).

The model outputs softmax probabilities for [positive, negative, neutral].
We convert to a single float:  score = P(positive) - P(negative)
giving a range of -1.0 (strongly bearish) to +1.0 (strongly bullish).

Usage:
    from market_data.finbert_client import FinBERTClient
    client = FinBERTClient()
    score = client.score_headline("AIXI", headline="...", summary="...")
    # score: float -1.0 to 1.0, or None on failure
"""

import logging

logger = logging.getLogger(__name__)

_MODEL = "ProsusAI/finbert"


class FinBERTClient:
    """
    Local FinBERT sentiment scorer.
    Model is loaded lazily on first call, or eagerly via preload().
    """

    def __init__(self) -> None:
        self._pipe = None
        self._lock = __import__("threading").Lock()

    def preload(self) -> None:
        """Start loading the model in a background thread."""
        import threading
        threading.Thread(target=self._ensure_model, daemon=True).start()

    def _ensure_model(self) -> None:
        with self._lock:
            if self._pipe is not None:
                return
            import torch
            from huggingface_hub import snapshot_download
            from transformers import logging as tf_logging
            from transformers import pipeline

            logger.info("Loading FinBERT model…")
            try:
                path = snapshot_download(_MODEL, local_files_only=True)
            except FileNotFoundError:
                logger.info("Model not cached — downloading (~420 MB)…")
                path = snapshot_download(_MODEL)

            # Auto-detect best available device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"          # Apple Silicon GPU
            else:
                device = "cpu"
            logger.info("FinBERT device: %s", device)

            tf_logging.set_verbosity_error()
            self._pipe = pipeline(
                "sentiment-analysis", model=path, tokenizer=path, device=device,
            )
            tf_logging.set_verbosity_warning()

    def score_headline(
        self,
        symbol: str,
        headline: str,
        summary: str = "",
    ) -> float | None:
        """
        Score a single headline using FinBERT.
        Returns a float in [-1.0, 1.0], or None if scoring fails.

        ``symbol`` is accepted for interface compatibility but not used
        by the model.
        """
        text = f"{headline} {summary}".strip() if summary else headline
        if not text:
            return None

        try:
            self._ensure_model()
            import torch
            with torch.inference_mode():
                results = self._pipe(text, top_k=3)

            probs = {r["label"]: r["score"] for r in results}
            pos = probs.get("positive", 0.0)
            neg = probs.get("negative", 0.0)
            score = round(pos - neg, 4)
            return max(-1.0, min(1.0, score))

        except Exception as exc:
            logger.warning(f"FinBERT scoring error: {exc}")
            return None

    def score_headlines_batch(
        self,
        texts: list[str],
    ) -> list[float | None]:
        """Score multiple headlines in one forward pass (3-5x faster than individual calls).

        *texts* is a list of headline strings.  Returns a list of the same
        length with float scores in [-1.0, 1.0] (or ``None`` for failures).
        """
        if not texts:
            return []

        try:
            self._ensure_model()
            import torch
            with torch.inference_mode():
                batch_results = self._pipe(texts, top_k=3, batch_size=len(texts))
        except Exception as exc:
            logger.warning(f"FinBERT batch scoring error: {exc}")
            return [None] * len(texts)

        scores: list[float | None] = []
        for results in batch_results:
            try:
                probs = {r["label"]: r["score"] for r in results}
                pos = probs.get("positive", 0.0)
                neg = probs.get("negative", 0.0)
                scores.append(max(-1.0, min(1.0, round(pos - neg, 4))))
            except Exception:
                scores.append(None)
        return scores
