"""test_api.py – StockAppAPI, benchmarks, export, sentiment."""

import csv

import pytest

from interface.api import PredictionConfig


class TestAPI:
    """StockAppAPI facade tests."""

    def test_get_data(self, api):
        df = api.get_data("TEST", period="max")
        assert len(df) > 0
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns

    def test_prediction_all_models(self, api):
        for model in ["knn", "knn_enhanced", "linreg", "linreg_enhanced"]:
            cfg = PredictionConfig(ticker="TEST", period="1y", model_type=model)
            result = api.get_prediction(cfg)
            assert result.prediction in ("UP", "DOWN")
            assert result.data_points > 0
            assert result.last_price > 0

    def test_prediction_with_news(self, api):
        cfg = PredictionConfig(ticker="TEST", period="1y", model_type="knn", include_news=True)
        result = api.get_prediction(cfg)
        assert result.prediction in ("UP", "DOWN")
        assert result.sentiment in ("POSITIVE", "NEGATIVE", "NEUTRAL")

    def test_invalid_model_raises(self, api):
        cfg = PredictionConfig(ticker="TEST", model_type="invalid")
        with pytest.raises(ValueError, match="Unknown"):
            api.get_prediction(cfg)

    def test_refresh_tickers(self, api):
        """refresh_tickers should not crash."""
        api.refresh_tickers(["TEST1", "TEST2"])
        # Data should be in DB now
        df = api.get_data("TEST1", period="max")
        assert len(df) > 0


class TestBenchmarks:
    """Benchmark comparison logic."""

    def test_stock_benchmarks(self):
        from config import get_benchmarks

        bench = get_benchmarks("AAPL")
        assert "SPY" in bench
        assert "QQQ" in bench

    def test_crypto_benchmarks(self):
        from config import get_benchmarks

        bench = get_benchmarks("ETH-USD")
        assert "BTC-USD" in bench

    def test_btc_no_self_benchmark(self):
        """BTC shouldn't benchmark against itself."""
        from config import get_benchmarks

        bench = get_benchmarks("BTC-USD")
        assert "BTC-USD" not in bench
        assert len(bench) == 0

    def test_compute_benchmarks(self, api, full_df):
        from engine.backtest_helpers import compute_benchmarks
        from engine.backtester import Backtester

        # Benchmark data must be in DB first (simulates refresh step)
        api.get_data("SPY", period="max")
        api.get_data("QQQ", period="max")

        bt = Backtester(n_days=10)
        result = bt.run(api.knn, "k-NN", full_df, ticker="AAPL-FAKE")
        bench = compute_benchmarks(api, "AAPL-FAKE", result.days)
        assert "SPY" in bench
        assert "QQQ" in bench
        assert isinstance(bench["SPY"], float)


class TestExport:
    """CSV export with mixed benchmark columns."""

    def test_mixed_columns_csv(self, tmp_path):
        """Rows with different benchmark columns → CSV should handle it."""
        from engine.backtest_helpers import export_rows

        rows = [
            {"ticker": "AAPL", "return": 0.05, "bench_SPY": 0.03, "bench_QQQ": 0.04},
            {"ticker": "BTC-USD", "return": 0.10},  # no bench columns
            {"ticker": "ETH-USD", "return": 0.08, "bench_BTC-USD": 0.06},
        ]
        output = str(tmp_path / "test.csv")
        export_rows(rows, output)

        # Read back and verify
        with open(output) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            assert "bench_SPY" in headers
            assert "bench_QQQ" in headers
            assert "bench_BTC-USD" in headers

    def test_json_export(self, tmp_path):
        import json

        from engine.backtest_helpers import export_rows

        rows = [{"ticker": "TEST", "return": 0.05}]
        output = str(tmp_path / "test.json")
        export_rows(rows, output)

        with open(output) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["ticker"] == "TEST"


class TestSentiment:
    """News scraper and sentiment scoring."""

    def test_naive_scoring(self):
        from engine.sentiment import NaiveScorer

        scorer = NaiveScorer()
        scores = scorer.score_many(["Great profit and growth", "Market crash feared"])
        assert all(isinstance(s, float) for s in scores)
        assert all(-1.0 <= s <= 1.0 for s in scores)
        # First headline has 2 positive words → positive score
        assert scores[0] > 0
        # Second has 1 negative word → negative score
        assert scores[1] < 0

    def test_empty_headlines(self):
        from engine.sentiment import NaiveScorer

        scorer = NaiveScorer()
        assert scorer.score_one("") == 0.0
        assert scorer.score_many([]) == []

    def test_sentiment_integration(self, api):
        score, headlines = api._process_news_with_db("TEST")
        assert isinstance(score, float)
        assert isinstance(headlines, list)


class TestHelpers:
    """Backtest helper utilities."""

    def test_direction_accuracy(self):
        from engine.backtest_helpers import BacktestResult, direction_accuracy
        from engine.backtester import DayResult

        days = [
            DayResult("d1", "UP", "UP", 0.7, True, 100, 101, 101, 0.01, 0.01, False),
            DayResult("d2", "UP", "DOWN", 0.6, False, 100, 99, 99, -0.01, -0.01, False),
            DayResult("d3", "DOWN", "DOWN", 0.8, True, 100, 99, 99, 0.01, 0.01, False),
        ]
        r = BacktestResult(
            model_name="test", ticker="T", test_days=3, correct=2, accuracy=0.67, days=days
        )
        up_c, up_t, dn_c, dn_t = direction_accuracy(r)
        assert up_c == 1  # 1 correct UP
        assert up_t == 2  # 2 UP predictions
        assert dn_c == 1  # 1 correct DOWN
        assert dn_t == 1  # 1 DOWN prediction

    def test_period_to_start_date(self):
        from datetime import date

        from engine.utils import period_to_start_date

        d = period_to_start_date("max")
        assert d == date(1900, 1, 1)

        d = period_to_start_date("1y")
        assert d.year >= 2024  # should be ~1 year ago
