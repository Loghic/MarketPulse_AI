"""
sentiment.py – Pluggable sentiment scorers for news headlines.

Three scoring backends:
    "vader"    — NLTK VADER (rule-based, context-aware, general-purpose)
    "finbert"  — ProsusAI/finbert (transformer, fine-tuned on financial text)
    "naive"    — Keyword matching (zero-dependency baseline)

All scorers expose the same interface:

    scorer = get_scorer("vader")
    score = scorer.score_one("Apple beats earnings")      # → float in [-1, 1]
    scores = scorer.score_many(["...", "..."])             # → list[float]

`get_scorer()` returns a working scorer with graceful fallback: requesting
"finbert" without `transformers` installed falls back to "vader"; requesting
"vader" without `nltk` data falls back to "naive".

Scorers are intentionally stateless aside from lazy model loading, so they
can be cached and reused across many headlines.
"""

from __future__ import annotations

from typing import Literal, Protocol

SentimentMethod = Literal["vader", "finbert", "naive"]


# ----------------------------------------------------------------------
# Vocabulary for the naive scorer (kept for baseline comparison)
# ----------------------------------------------------------------------

_POS_WORDS = {
    "up",
    "bull",
    "buy",
    "growth",
    "profit",
    "surge",
    "positive",
    "win",
    "high",
    "boost",
    "top",
    "gain",
    "rally",
    "strong",
    "beat",
    "record",
    "upgrade",
    "outperform",
}
_NEG_WORDS = {
    "down",
    "bear",
    "sell",
    "loss",
    "drop",
    "negative",
    "fall",
    "risk",
    "debt",
    "crash",
    "short",
    "decline",
    "plunge",
    "weak",
    "miss",
    "warning",
    "downgrade",
    "underperform",
}


class SentimentScorer(Protocol):
    """Interface every scorer implements."""

    name: str

    def score_one(self, text: str) -> float: ...

    def score_many(self, texts: list[str]) -> list[float]: ...


# ----------------------------------------------------------------------
# Naive keyword scorer
# ----------------------------------------------------------------------


class NaiveScorer:
    """Sum of positive minus negative keyword hits, clipped to [-1, 1]."""

    name = "naive"

    def score_one(self, text: str) -> float:
        if not text:
            return 0.0
        words = text.lower().split()
        score = 0
        for w in words:
            if w in _POS_WORDS:
                score += 1
            elif w in _NEG_WORDS:
                score -= 1
        return max(-1.0, min(1.0, float(score)))

    def score_many(self, texts: list[str]) -> list[float]:
        return [self.score_one(t) for t in texts]


# ----------------------------------------------------------------------
# VADER scorer
# ----------------------------------------------------------------------


class VADERScorer:
    """NLTK VADER compound score, range [-1, 1]."""

    name = "vader"

    def __init__(self) -> None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self._sia = SentimentIntensityAnalyzer()

    def score_one(self, text: str) -> float:
        if not text:
            return 0.0
        return float(self._sia.polarity_scores(text)["compound"])

    def score_many(self, texts: list[str]) -> list[float]:
        return [self.score_one(t) for t in texts]


# ----------------------------------------------------------------------
# FinBERT scorer
# ----------------------------------------------------------------------


class FinBERTScorer:
    """
    ProsusAI/finbert — BERT fine-tuned for financial news sentiment.

    The model returns three logits: [negative, neutral, positive].
    We collapse to a single score by taking `p_positive - p_negative`
    (neutral mass cancels), giving a value in [-1, 1] that matches the
    convention of VADER's compound score.

    Loading the model is lazy (first scoring call) and slow (~400 MB download
    + ~2 s init). For backtests, instantiate once and reuse across headlines.
    """

    name = "finbert"
    MODEL_ID = "ProsusAI/finbert"

    def __init__(self, device: str | None = None, batch_size: int = 16) -> None:
        # Defer the heavy import — keeps import-time cost zero for non-FinBERT users
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)
        self._model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(device)
        self._batch_size = batch_size

        # FinBERT label order in the published checkpoint: ["positive","negative","neutral"]
        # Read from config so we never get this wrong.
        id2label = self._model.config.id2label
        self._idx_positive = next(i for i, lbl in id2label.items() if lbl.lower() == "positive")
        self._idx_negative = next(i for i, lbl in id2label.items() if lbl.lower() == "negative")

    def score_one(self, text: str) -> float:
        if not text:
            return 0.0
        return self.score_many([text])[0]

    def score_many(self, texts: list[str]) -> list[float]:
        if not texts:
            return []

        torch = self._torch
        out: list[float] = []

        with torch.no_grad():
            for start in range(0, len(texts), self._batch_size):
                batch = texts[start : start + self._batch_size]
                enc = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(self._device)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for row in probs:
                    out.append(float(row[self._idx_positive] - row[self._idx_negative]))
        return out


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


_SCORER_CACHE: dict[str, SentimentScorer] = {}


def get_scorer(method: str = "vader") -> SentimentScorer:
    """
    Return a sentiment scorer instance for the requested method, with
    graceful fallback.

    Cached per-process so the heavy FinBERT model is loaded only once.

    Fallback chain:
        "finbert" → VADER (if transformers/torch unavailable)
        "vader"   → NaiveScorer (if nltk unavailable)
        "naive"   → never fails
    """
    if method in _SCORER_CACHE:
        return _SCORER_CACHE[method]

    if method == "finbert":
        try:
            scorer: SentimentScorer = FinBERTScorer()
        except Exception as e:
            print(f"WARNING: FinBERT unavailable ({e}). Falling back to VADER.")
            scorer = get_scorer("vader")
    elif method == "vader":
        try:
            scorer = VADERScorer()
        except Exception as e:
            print(f"WARNING: VADER unavailable ({e}). Falling back to naive.")
            scorer = NaiveScorer()
    else:
        scorer = NaiveScorer()

    _SCORER_CACHE[method] = scorer
    return scorer


def clear_scorer_cache() -> None:
    """Drop cached scorer instances. Useful for tests."""
    _SCORER_CACHE.clear()
