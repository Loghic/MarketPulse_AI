"""
news_scraper.py – Fetch news headlines and compute simple sentiment scores via yfinance.
Compatible with yfinance >= 1.3.0 (new XHR news endpoint).
"""

import yfinance as yf
from typing import List, Tuple


class NewsScraper:
    def __init__(self):
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

    @staticmethod
    def _extract_title(item: dict) -> str:
        """
        Extract the headline from a news item.

        yfinance 1.3.0 returns items from the XHR stream:
            {'content': {'title': '...', ...}, ...}
        Older versions returned:
            {'title': '...'} or {'headline': '...'}
        """
        # yfinance 1.3.0+ (XHR stream format)
        content = item.get("content", {})
        if isinstance(content, dict):
            title = content.get("title")
            if title:
                return title

        # Fallback for older versions
        return item.get("title") or item.get("headline") or item.get("summary") or ""

    def _score_titles(self, titles: List[str]) -> float:
        """Simple keyword-matching sentiment scorer."""
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

        # Normalize to the [-1.0, 1.0] range
        return max(-1.0, min(1.0, score / max(len(titles), 1)))

    def get_sentiment(self, ticker: str) -> Tuple[float, List[str]]:
        """
        Fetch recent news and return a sentiment score with headlines.

        Returns:
            (score, headlines) – score in [-1, 1], up to 10 headlines.
        """
        try:
            stock = yf.Ticker(ticker)
            raw_news = stock.news

            if not raw_news:
                return 0.0, []

            titles = []
            for item in raw_news[:10]:
                title = self._extract_title(item)
                if title:
                    titles.append(title)

            if not titles:
                return 0.0, []

            score = self._score_titles(titles)
            return score, titles

        except Exception as e:
            print(f"WARNING: News fetch failed for {ticker}: {e}")
            return 0.0, []
