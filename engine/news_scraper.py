"""
news_scraper.py – News + sentiment orchestrator.

This module ties together two concerns that used to live here directly:

    1. WHERE headlines come from  →  engine/news_sources.py (Yahoo, GDELT, ...)
    2. HOW headlines are scored   →  engine/sentiment.py    (VADER, FinBERT, naive)

``NewsScraper`` is the high-level facade the rest of the codebase talks to.
It exposes three core methods:

    get_sentiment(ticker, method="vader")
        — Backward-compatible "current sentiment" call: fetch fresh headlines
          from the configured provider, score them, return (score, headlines).

    fetch_and_score(ticker, lookback_days, method, source)
        — Returns ``list[ScoredNewsItem]`` with each item carrying its real
          publication date and individual sentiment score. This is the
          method the data-refresh path now uses so historical news can be
          stored in the DB with their actual dates.

    weighted_score(news_items, asof_date, half_life_days)
        — Pure function. Given a list of ScoredNewsItem (with sentiment_score
          set) and a reference date, return a single weighted average
          sentiment score using exponential time-decay. Used inside the
          backtester to compute a per-day sentiment from stored history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from engine.news_sources import NewsItem, get_provider
from engine.sentiment import get_scorer

# Public type aliases. Both are deliberately *string* (not Literal) for the
# function signatures: CLI args (argparse) hand us plain ``str`` values, and
# the scorer factory falls back gracefully on unknown methods. The Literal
# is kept around as ``ScoringMethod`` for documentation / IDE help where the
# caller does have a literal in hand.
SentimentMethod = str  # widened from Literal for runtime CLI strings
ScoringMethod = Literal["vader", "finbert", "naive"]


# ----------------------------------------------------------------------
# Scored item dataclass
# ----------------------------------------------------------------------


@dataclass
class ScoredNewsItem:
    """A NewsItem that has been scored by a specific sentiment method."""

    ticker: str
    published_at: str
    headline: str
    source: str
    url: str
    sentiment_score: float
    method: str


class NewsScraper:
    """
    Facade around news sources + sentiment scorers.

    The scraper is parameterised by:
      * ``default_method`` — which sentiment scorer to use by default
      * ``default_source`` — which news provider to use by default

    Both can be overridden per-call. Scorers and providers are cached so
    instantiating multiple ``NewsScraper`` objects is cheap.
    """

    def __init__(
        self,
        default_method: SentimentMethod = "vader",
        default_source: str | list[str] = "yahoo",
    ) -> None:
        self.default_method = default_method
        self.default_source = default_source

    # ------------------------------------------------------------------
    # Capability flags (used by ``api.py`` and old tests)
    # ------------------------------------------------------------------

    @property
    def vader_available(self) -> bool:
        return get_scorer("vader").name == "vader"

    @property
    def finbert_available(self) -> bool:
        try:
            return get_scorer("finbert").name == "finbert"
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sentiment(
        self,
        ticker: str,
        method: SentimentMethod | None = None,
        source: str | list[str] | None = None,
        lookback_days: int = 7,
    ) -> tuple[float, list[str]]:
        """
        Fetch recent news and return (average sentiment, headlines).

        Backward-compatible signature: existing callers can keep passing
        ``method="vader"`` and ignore the new ``source`` and
        ``lookback_days`` arguments.
        """
        items = self.fetch_and_score(
            ticker,
            lookback_days=lookback_days,
            method=method or self.default_method,
            source=source or self.default_source,
        )
        if not items:
            return 0.0, []
        avg = sum(i.sentiment_score for i in items) / len(items)
        return float(avg), [i.headline for i in items]

    def fetch_and_score(
        self,
        ticker: str,
        lookback_days: int = 7,
        method: SentimentMethod | None = None,
        source: str | list[str] | None = None,
    ) -> list[ScoredNewsItem]:
        """
        Fetch headlines from the requested source and score them.

        Returns a list of ``ScoredNewsItem`` — each carries its real
        publication date plus an individual sentiment score. Empty list
        if the source yields nothing.
        """
        provider = get_provider(source or self.default_source)
        scorer = get_scorer(method or self.default_method)

        try:
            raw_items: list[NewsItem] = provider.fetch(ticker, lookback_days=lookback_days)
        except Exception as e:
            print(f"WARNING: news fetch failed for {ticker} ({provider.name}): {e}")
            return []

        if not raw_items:
            return []

        scores = scorer.score_many([i.headline for i in raw_items])
        scored = [
            ScoredNewsItem(
                ticker=item.ticker,
                published_at=item.published_at,
                headline=item.headline,
                source=item.source,
                url=item.url,
                sentiment_score=float(score),
                method=scorer.name,
            )
            for item, score in zip(raw_items, scores)
        ]
        return scored

    # ------------------------------------------------------------------
    # Look-ahead-safe weighted aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def weighted_score(
        items: list[ScoredNewsItem] | pd.DataFrame,
        asof_date: str,
        half_life_days: float = 0.0,
    ) -> float:
        """
        Return a single weighted-average sentiment score.

        Args:
            items: Either a list of ``ScoredNewsItem`` or a DataFrame with
                ``sentiment_score`` and either ``published_at`` or
                ``effective_date`` columns.
            asof_date: Reference date ("YYYY-MM-DD"). Each item's age is
                measured against this date. Items dated on or after
                ``asof_date`` are dropped (look-ahead protection).
            half_life_days: Exponential decay half-life. 0 (or negative)
                disables decay and gives a plain average — every headline
                counts equally regardless of age.

        Returns:
            Float in [-1, 1]. 0.0 if no items qualify.
        """
        if isinstance(items, pd.DataFrame):
            df = items.copy()
            if df.empty:
                return 0.0
            if "effective_date" in df.columns:
                date_col = "effective_date"
            elif "published_at" in df.columns:
                date_col = "published_at"
            else:
                date_col = "date"
            scores = df["sentiment_score"].astype(float).tolist()
            dates = df[date_col].fillna(df.get("date", "")).astype(str).tolist()
        else:
            if not items:
                return 0.0
            scores = [it.sentiment_score for it in items]
            dates = [it.published_at for it in items]

        asof = pd.to_datetime(asof_date)
        weighted_sum = 0.0
        weight_total = 0.0

        for score, d in zip(scores, dates):
            if not d:
                continue
            try:
                dt = pd.to_datetime(d)
            except Exception:
                continue
            age_days = (asof - dt).days
            if age_days < 0:
                # Look-ahead — drop. Shouldn't happen if caller used
                # ``get_news_before``, but cheap to double-check.
                continue

            if half_life_days and half_life_days > 0:
                weight = math.pow(0.5, age_days / half_life_days)
            else:
                weight = 1.0

            weighted_sum += score * weight
            weight_total += weight

        if weight_total <= 0:
            return 0.0
        return float(weighted_sum / weight_total)
