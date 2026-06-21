"""
test_news_pipeline.py – Tests for the refactored news + sentiment pipeline.

Covers:
  * Pluggable sentiment scorers (VADER, FinBERT mocked, naive).
  * Look-ahead-safe DB queries (only news strictly < asof_date).
  * Exponential decay weighting (recent > old).
  * Configurable lookback window.
  * Backtester per-day sentiment provider integration.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from engine.db_manager import DatabaseManager
from engine.news_scraper import NewsScraper, ScoredNewsItem
from engine.news_sources import GDELTNewsProvider, YahooNewsProvider, get_provider
from engine.sentiment import NaiveScorer, get_scorer

# ----------------------------------------------------------------------
# Sentiment scorers
# ----------------------------------------------------------------------


class TestSentimentScorers:
    def test_naive_directionality(self):
        s = NaiveScorer()
        assert s.score_one("Strong profit growth beats estimates") > 0
        assert s.score_one("Crash sell-off plunge debt warning") < 0
        assert s.score_one("Random unrelated sentence here") == 0

    def test_naive_score_range(self):
        s = NaiveScorer()
        for text in ["x", "growth growth growth growth growth", "crash crash crash crash crash"]:
            score = s.score_one(text)
            assert -1.0 <= score <= 1.0

    def test_factory_caches(self):
        # Same method should return the cached scorer instance
        from engine.sentiment import clear_scorer_cache

        clear_scorer_cache()
        a = get_scorer("naive")
        b = get_scorer("naive")
        assert a is b

    def test_finbert_fallback(self):
        """Asking for FinBERT without transformers installed falls back gracefully."""
        from engine import sentiment

        # Patch the FinBERTScorer constructor to simulate a failed import
        with patch.object(sentiment, "FinBERTScorer", side_effect=ImportError("no transformers")):
            sentiment.clear_scorer_cache()
            scorer = sentiment.get_scorer("finbert")
            assert scorer.name in ("vader", "naive")  # whichever VADER falls back to


# ----------------------------------------------------------------------
# DB look-ahead-safe query
# ----------------------------------------------------------------------


class TestDBLookAhead:
    def _seed(self, db_path):
        """Insert known news items at known dates for one ticker."""
        db = DatabaseManager(db_name=db_path.name)
        rows = []
        # Five headlines, one per day, oldest first
        for offset, hl, score in [
            (-10, "Old positive news", 0.6),
            (-5, "Mid-period negative news", -0.4),
            (-2, "Recent positive news", 0.8),
            (-1, "Yesterday slightly bad", -0.2),
            (0, "TODAY headline — must NOT leak", 0.9),
        ]:
            pub = (datetime(2026, 5, 21) + timedelta(days=offset)).strftime("%Y-%m-%d")
            rows.append(
                {
                    "ticker": "TEST",
                    "date": pub,
                    "headline": hl,
                    "sentiment_score": score,
                    "published_at": pub,
                    "source": "yahoo",
                    "method": "vader",
                }
            )
        db.save_news("TEST", pd.DataFrame(rows))
        return db

    def test_strict_before_asof(self, tmp_path, monkeypatch):
        """get_news_before returns ONLY rows with effective_date < asof_date."""
        monkeypatch.chdir(tmp_path)
        db = self._seed(tmp_path / "test_strict.db")

        df = db.get_news_before("TEST", asof_date="2026-05-21")
        # 4 of 5 rows are strictly older than 2026-05-21
        assert len(df) == 4
        assert "TODAY headline — must NOT leak" not in df["headline"].tolist()
        # All effective dates must be strictly older
        for d in df["effective_date"]:
            assert d < "2026-05-21"

    def test_lookback_window_drops_old(self, tmp_path, monkeypatch):
        """A lookback window of 3 days only keeps very recent rows."""
        monkeypatch.chdir(tmp_path)
        db = self._seed(tmp_path / "test_lookback.db")

        df = db.get_news_before("TEST", asof_date="2026-05-21", lookback_days=3)
        # Only the -2 and -1 entries qualify (>= asof - 3 days, < asof)
        assert len(df) == 2
        assert "Old positive news" not in df["headline"].tolist()
        assert "Mid-period negative news" not in df["headline"].tolist()
        assert "Recent positive news" in df["headline"].tolist()
        assert "Yesterday slightly bad" in df["headline"].tolist()

    def test_method_filter(self, tmp_path, monkeypatch):
        """method='finbert' returns only finbert rows (NULL methods also pass for backward compat)."""
        monkeypatch.chdir(tmp_path)
        db = DatabaseManager(db_name="test_method.db")

        rows = pd.DataFrame(
            [
                {
                    "ticker": "TEST",
                    "date": "2026-05-15",
                    "headline": "vader-scored",
                    "sentiment_score": 0.5,
                    "published_at": "2026-05-15",
                    "source": "yahoo",
                    "method": "vader",
                },
                {
                    "ticker": "TEST",
                    "date": "2026-05-16",
                    "headline": "finbert-scored",
                    "sentiment_score": 0.7,
                    "published_at": "2026-05-16",
                    "source": "gdelt",
                    "method": "finbert",
                },
            ]
        )
        db.save_news("TEST", rows)

        finbert = db.get_news_before("TEST", "2026-05-21", method="finbert")
        # Only the finbert-scored row should be returned (NULL methods would pass too)
        assert "finbert-scored" in finbert["headline"].tolist()
        assert "vader-scored" not in finbert["headline"].tolist()


# ----------------------------------------------------------------------
# Weighted score (time decay)
# ----------------------------------------------------------------------


class TestWeightedScore:
    def _items(self):
        """Three headlines with known scores at known ages from 2026-05-21."""
        return [
            ScoredNewsItem(
                ticker="T",
                published_at="2026-05-11",  # 10 days old
                headline="old",
                source="x",
                url="",
                sentiment_score=-1.0,
                method="vader",
            ),
            ScoredNewsItem(
                ticker="T",
                published_at="2026-05-18",  # 3 days old
                headline="mid",
                source="x",
                url="",
                sentiment_score=0.0,
                method="vader",
            ),
            ScoredNewsItem(
                ticker="T",
                published_at="2026-05-20",  # 1 day old
                headline="new",
                source="x",
                url="",
                sentiment_score=1.0,
                method="vader",
            ),
        ]

    def test_no_decay_is_average(self):
        items = self._items()
        score = NewsScraper.weighted_score(items, asof_date="2026-05-21", half_life_days=0)
        # With half_life=0 (no decay), this is the plain mean: (-1 + 0 + 1) / 3 = 0
        assert abs(score) < 1e-9

    def test_decay_shifts_toward_recent(self):
        """A half-life of 1 day weights the newest item ~512× more than the 10-day-old one."""
        items = self._items()
        score = NewsScraper.weighted_score(items, asof_date="2026-05-21", half_life_days=1.0)
        # The new (+1.0) headline dominates → score should be very close to +1
        assert score > 0.45

    def test_drops_future_news(self):
        """Items with published_at on/after asof are dropped (no look-ahead)."""
        items = self._items() + [
            ScoredNewsItem(
                ticker="T",
                published_at="2026-05-22",  # FUTURE
                headline="future",
                source="x",
                url="",
                sentiment_score=-1.0,
                method="vader",
            )
        ]
        score = NewsScraper.weighted_score(items, asof_date="2026-05-21", half_life_days=0)
        # Future row should be ignored — average of the 3 valid rows
        assert abs(score) < 1e-9

    def test_empty_returns_zero(self):
        assert NewsScraper.weighted_score([], asof_date="2026-05-21") == 0.0


# ----------------------------------------------------------------------
# News sources (just smoke tests — no network)
# ----------------------------------------------------------------------


class TestNewsSources:
    def test_get_provider_factory(self):
        p_yahoo = get_provider("yahoo")
        assert isinstance(p_yahoo, YahooNewsProvider)
        p_gdelt = get_provider("gdelt")
        assert isinstance(p_gdelt, GDELTNewsProvider)

    def test_multi_provider_dedup(self):
        """If two providers yield the same headline same date, we keep one."""
        provider = get_provider(["yahoo", "gdelt"])
        # Smoke: just confirm the multi-provider type + that fetch doesn't crash
        # with offline providers. Empty result is acceptable here.
        results = provider.fetch("AAPL", lookback_days=1)
        assert isinstance(results, list)

    def test_gdelt_query_builder(self):
        """Crypto symbols should strip the -USD suffix when building the query."""
        gdelt = GDELTNewsProvider()
        assert "Bitcoin" in gdelt._build_query("BTC-USD")
        assert "Apple" in gdelt._build_query("AAPL")


# ----------------------------------------------------------------------
# Backtester sentiment_provider integration
# ----------------------------------------------------------------------


class TestBacktesterSentimentProvider:
    def test_provider_called_per_day(self, full_df):
        """The provider gets called once per day with the prediction date."""
        from engine.backtester import Backtester

        seen_dates: list[str] = []

        def provider(date_str: str) -> float:
            seen_dates.append(date_str)
            return 0.0  # neutral

        from engine.knn_model import KNNModel

        bt = Backtester(n_days=5)
        bt.run(
            model=KNNModel(k=5, features=["returns"]),
            model_name="k-NN",
            df=full_df,
            ticker="TEST",
            sentiment_provider=provider,
        )
        # 5 backtest days → 5 provider calls
        assert len(seen_dates) == 5
        # Dates should be unique and in chronological order
        assert seen_dates == sorted(seen_dates)
        assert len(set(seen_dates)) == 5

    def test_provider_failure_is_handled(self, full_df):
        """If the sentiment_provider raises, backtest still runs (treats as 0.0)."""
        from engine.backtester import Backtester
        from engine.knn_model import KNNModel

        def bad_provider(date_str: str) -> float:
            raise RuntimeError("simulated network failure")

        bt = Backtester(n_days=3)
        result = bt.run(
            model=KNNModel(k=5, features=["returns"]),
            model_name="k-NN",
            df=full_df,
            sentiment_provider=bad_provider,
        )
        assert result.test_days == 3  # no crash

    def test_constant_sentiment_still_works(self, full_df):
        """Backward compat: sentiment_score=X still applies the constant."""
        from engine.backtester import Backtester
        from engine.knn_model import KNNModel

        bt = Backtester(n_days=3)
        result = bt.run(
            model=KNNModel(k=5, features=["returns"]),
            model_name="k-NN",
            df=full_df,
            sentiment_score=0.0,  # zero so result is deterministic w.r.t. price-only
        )
        assert result.test_days == 3


# ----------------------------------------------------------------------
# run_single_backtest – in-memory news cache + resilience
# ----------------------------------------------------------------------


class TestRunSingleBacktestNewsCache:
    """
    The 2026 walk-forward path pre-fetches news ONCE per ticker and
    filters in memory. Before this change every backtest day issued a
    fresh ``get_news_before`` query — thousands of SQLite connections per
    ``run_all.py`` job, which exhausted file descriptors on macOS.

    These tests assert two things:

      1. The walk-forward loop hits the DB at most a few times per
         (ticker, period) — not once per backtest day.
      2. If the DB raises on the news lookup, the backtest still runs
         (price-only) instead of crashing.
    """

    def test_db_get_news_called_once_per_ticker(self, api, full_df, monkeypatch):
        """``run_single_backtest`` should bulk-load news, not query per day."""
        from engine import backtest_helpers
        from engine.backtester import Backtester

        # Make today's news cache empty so _process_news_with_db does a fetch.
        monkeypatch.setattr(
            api, "_process_news_with_db", lambda *a, **kw: (0.0, ["headline-from-yahoo"])
        )

        call_log: list[tuple] = []
        real_get_news = api.db.get_news

        def counting_get_news(ticker, date=None):
            call_log.append((ticker, date))
            return real_get_news(ticker, date=date)

        monkeypatch.setattr(api.db, "get_news", counting_get_news)

        # Patch get_news_before to detect any per-day DB fall-through.
        before_calls: list[tuple] = []
        real_get_news_before = api.db.get_news_before

        def counting_get_news_before(*args, **kwargs):
            before_calls.append((args, kwargs))
            return real_get_news_before(*args, **kwargs)

        monkeypatch.setattr(api.db, "get_news_before", counting_get_news_before)

        bt = Backtester(n_days=5)
        results = backtest_helpers.run_single_backtest(
            api,
            bt,
            ticker="TEST",
            df=full_df,
            period="1y",
            n_days=5,
            full=False,
        )

        assert results, "expected at least one backtest result"
        # At most a handful of DB hits — the pre-fetch + maybe a few cache
        # checks. Crucially NOT 5 × N_news_variants per period.
        assert len(call_log) <= 4, f"too many get_news calls: {len(call_log)}"
        # Per-day path must not be used during the walk-forward loop.
        assert len(before_calls) == 0, (
            f"get_news_before should not be called during the in-memory path, "
            f"got {len(before_calls)} calls"
        )

    def test_db_failure_falls_back_to_no_news(self, api, full_df, monkeypatch):
        """If the news lookup raises mid-run, the backtest still completes."""
        import sqlite3

        from engine import backtest_helpers
        from engine.backtester import Backtester

        def boom(*_args, **_kwargs):
            raise sqlite3.OperationalError("unable to open database file")

        # Both the cache check and the bulk fetch raise — simulating the
        # macOS Spotlight / FD-exhaustion failure mode the user hit.
        monkeypatch.setattr(api.db, "get_news", boom)

        bt = Backtester(n_days=3)
        results = backtest_helpers.run_single_backtest(
            api,
            bt,
            ticker="TEST",
            df=full_df,
            period="1y",
            n_days=3,
            full=False,
        )
        # The backtest must not crash — that's the regression. The price-only
        # variants must be present; the "+ News" variants may also appear but
        # they fall through to a 0.0 sentiment for every day (no data).
        assert results, "backtest should still produce results"
        price_only = [r for r in results if "+ News" not in r.model_name]
        assert price_only, "expected at least the price-only model variants"
        # Every result completed cleanly. Most variants trade all 3 days; the
        # News-Informed baseline deliberately sits out (FLAT) when there is no
        # news — with the DB down it sees 0.0 sentiment every day, so it trades
        # nothing and reports 0 days. That's correct, not a crash. All other
        # variants must have walked the full 3-day window.
        for r in results:
            if "News Informed" in r.model_name:
                assert r.test_days == 0
            else:
                assert r.test_days == 3
