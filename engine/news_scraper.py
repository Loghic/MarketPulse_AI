"""
news_scraper.py – Fetch news headlines and compute sentiment scores via yfinance.

Supports two scoring methods:
    "vader" — NLTK VADER (context-aware, handles negation, intensifiers, caps)
    "naive" — Simple keyword matching (original baseline, no dependencies)

If VADER fails to initialize (missing lexicon, no network), it falls back
to naive scoring automatically.

Compatible with yfinance >= 1.3.0 (new XHR news endpoint).
"""

import yfinance as yf
from typing import List, Tuple, Literal


class NewsScraper:
    def __init__(self):
        self.sia = None
        self._init_vader()

        # Legacy naive keyword sets (kept for baseline comparison)
        self.pos_words = {
            "up", "bull", "buy", "growth", "profit", "surge", "positive",
            "win", "high", "boost", "top", "gain", "rally", "strong",
            "beat", "record", "upgrade", "outperform",
        }
        self.neg_words = {
            "down", "bear", "sell", "loss", "drop", "negative", "fall",
            "risk", "debt", "crash", "short", "decline", "plunge", "weak",
            "miss", "warning", "downgrade", "underperform",
        }

    def _init_vader(self):
        """Try to initialize VADER. Falls back to naive if unavailable."""
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)

            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"WARNING: VADER unavailable ({e}), using naive sentiment scoring.")
            self.sia = None

    @property
    def vader_available(self) -> bool:
        return self.sia is not None

    @staticmethod
    def _extract_title(item: dict) -> str:
        """
        Extract headline from a yfinance news item.

        yfinance 1.3.0+ returns: {'content': {'title': '...'}, ...}
        Older versions: {'title': '...'} or {'headline': '...'}
        """
        content = item.get("content", {})
        if isinstance(content, dict):
            title = content.get("title")
            if title:
                return title
        return item.get("title") or item.get("headline") or item.get("summary") or ""

    def _score_naive(self, titles: List[str]) -> float:
        """Simple keyword-matching scoring. Returns score in [-1, 1]."""
        if not titles:
            return 0.0

        score = 0
        for title in titles:
            words = title.lower().split()
            for word in words:
                if word in self.pos_words:
                    score += 1
                if word in self.neg_words:
                    score -= 1

        return max(-1.0, min(1.0, score / max(len(titles), 1)))

    def _score_vader(self, titles: List[str]) -> float:
        """
        VADER sentiment scoring.

        Uses the compound score which combines positive, negative, and neutral
        into a single value in [-1, 1]. Averaged across all headlines.
        """
        if not titles or self.sia is None:
            return 0.0

        total_compound = 0.0
        for title in titles:
            scores = self.sia.polarity_scores(title)
            total_compound += scores["compound"]

        return total_compound / len(titles)

    def get_sentiment(
        self,
        ticker: str,
        method: Literal["vader", "naive"] = "vader",
    ) -> Tuple[float, List[str]]:
        """
        Fetch recent news and return a sentiment score with headlines.

        Args:
            ticker: Asset symbol (e.g. 'AAPL', 'BTC-USD').
            method: Scoring strategy — 'vader' (default) or 'naive'.
                    Falls back to 'naive' if VADER is unavailable.

        Returns:
            (score, headlines) — score in [-1, 1], up to 10 headlines.
        """
        try:
            stock = yf.Ticker(ticker)
            raw_news = stock.news

            if not raw_news:
                return 0.0, []

            # Extract titles (call _extract_title once per item)
            titles = []
            for item in raw_news[:10]:
                title = self._extract_title(item)
                if title:
                    titles.append(title)

            if not titles:
                return 0.0, []

            # Fall back to naive if VADER requested but not available
            if method == "vader" and self.vader_available:
                score = self._score_vader(titles)
            else:
                score = self._score_naive(titles)

            return score, titles

        except Exception as e:
            print(f"WARNING: News fetch failed for {ticker}: {e}")
            return 0.0, []
