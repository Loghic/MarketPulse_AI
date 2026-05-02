"""test_web_api.py – FastAPI backend endpoint tests.

Tests all routes with mocked StockAppAPI (no network, no real DB).
"""

from datetime import datetime
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_prices(days=400, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    n = len(dates)
    prices = 150.0 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
    return pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n)),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.015, n))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.015, n))),
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 50_000_000, n),
        },
        index=dates,
    ).rename_axis("Date")


def _make_news():
    return [
        {"content": {"title": "Stock rally continues with strong growth"}},
        {"content": {"title": "Market risk warning: debt levels rising"}},
    ]


@pytest.fixture(scope="module")
def client():
    """Create TestClient with mocked yfinance."""
    with patch("engine.data_downloader.yf") as mock_dl, patch("engine.news_scraper.yf") as mock_ns:
        mock_dl.Ticker.return_value.history.return_value = _make_prices()
        type(mock_ns.Ticker.return_value).news = PropertyMock(return_value=_make_news())

        # Reset shared API instance
        from web.backend.routes import data as data_module

        data_module._api = None

        from web.backend.app import app

        with TestClient(app) as c:
            yield c


@pytest.fixture(scope="module")
def refreshed_client(client):
    """Client with data pre-loaded via refresh."""
    client.post("/api/data/refresh", json={"tickers": ["TEST"]})
    return client


# ------------------------------------------------------------------
# Health & Root
# ------------------------------------------------------------------


class TestRoot:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "MarketPulse AI"
        assert "endpoints" in data

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


# ------------------------------------------------------------------
# Data endpoints
# ------------------------------------------------------------------


class TestData:
    def test_list_tickers(self, client):
        r = client.get("/api/data/tickers")
        assert r.status_code == 200
        tickers = r.json()
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert "ticker" in tickers[0]
        assert "asset_type" in tickers[0]

    def test_refresh(self, client):
        r = client.post("/api/data/refresh", json={"tickers": ["TEST"]})
        assert r.status_code == 200
        results = r.json()
        assert isinstance(results, list)
        assert len(results) > 0
        # At least TEST should be refreshed
        test_result = next((x for x in results if x["ticker"] == "TEST"), None)
        assert test_result is not None
        assert test_result["rows"] > 0

    def test_refresh_all(self, client):
        r = client.post("/api/data/refresh", json={"tickers": []})
        assert r.status_code == 200
        results = r.json()
        assert len(results) > 0

    def test_get_ticker_data(self, refreshed_client):
        r = refreshed_client.get("/api/data/ticker/TEST?period=1y&limit=100")
        assert r.status_code == 200
        data = r.json()
        assert data["ticker"] == "TEST"
        assert data["rows"] > 0
        assert len(data["data"]) > 0
        row = data["data"][0]
        assert "date" in row
        assert "open" in row
        assert "close" in row
        assert "volume" in row

    def test_get_ticker_data_periods(self, refreshed_client):
        """Different periods should return different row counts."""
        r1 = refreshed_client.get("/api/data/ticker/TEST?period=1y")
        r2 = refreshed_client.get("/api/data/ticker/TEST?period=max")
        assert r1.status_code == 200
        assert r2.status_code == 200
        # max should have >= 1y rows
        assert r2.json()["rows"] >= r1.json()["rows"]

    def test_get_ticker_not_found(self, client):
        r = client.get("/api/data/ticker/NONEXISTENT")
        assert r.status_code == 404

    def test_ticker_data_sorted_descending(self, refreshed_client):
        r = refreshed_client.get("/api/data/ticker/TEST?period=1y&limit=10")
        data = r.json()["data"]
        dates = [row["date"] for row in data]
        assert dates == sorted(dates, reverse=True)


# ------------------------------------------------------------------
# Predict endpoints
# ------------------------------------------------------------------


class TestPredict:
    def test_predict_single(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [{"model": "k-NN", "period": "1y", "news": False}],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "predictions" in data
        assert "consensus" in data
        preds = data["predictions"]
        assert len(preds) >= 1
        p = preds[0]
        assert p["ticker"] == "TEST"
        assert p["prediction"] in ("UP", "DOWN", "N/A")

    def test_predict_multiple_models(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [
                    {"model": "k-NN", "period": "1y", "news": False},
                    {"model": "k-NN (TW)", "period": "1y", "news": False},
                ],
            },
        )
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) >= 1
        labels = {p["model"] for p in preds}
        assert any("k-NN" in label for label in labels)

    def test_predict_all_models(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [
                    {"model": m, "period": "1y", "news": False}
                    for m in ["k-NN", "k-NN (TW)", "LinReg", "LSTM"]
                ],
            },
        )
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) >= 2

    def test_predict_with_news(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [{"model": "k-NN", "period": "1y", "news": True}],
            },
        )
        assert r.status_code == 200

    def test_consensus_in_response(self, refreshed_client):
        r = refreshed_client.post(
            "/api/predict/run",
            json={
                "ticker": "TEST",
                "items": [
                    {"model": "k-NN", "period": "1y", "news": False},
                    {"model": "k-NN (TW)", "period": "1y", "news": False},
                ],
            },
        )
        assert r.status_code == 200
        cons = r.json()["consensus"]
        assert cons["direction"] in ("UP", "DOWN", "SPLIT")
        assert cons["up"] + cons["down"] == cons["total"]
        assert 0 <= cons["agreement"] <= 1


# ------------------------------------------------------------------
# Backtest endpoints
# ------------------------------------------------------------------


class TestBacktest:
    def test_backtest_basic(self, refreshed_client):
        r = refreshed_client.post(
            "/api/backtest",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "period": "1y",
                "fee_pct": 0.03,
                "refresh_data": False,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        results = data["results"]
        assert len(results) > 0

        # Check first result structure
        result = results[0]
        assert result["ticker"] == "TEST"
        assert "accuracy" in result
        assert "total_return" in result
        assert "profit_factor" in result
        assert "max_drawdown" in result
        assert "sharpe_ratio" in result
        assert 0 <= result["accuracy"] <= 1

    def test_backtest_has_daily_data(self, refreshed_client):
        r = refreshed_client.post(
            "/api/backtest",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "period": "1y",
                "refresh_data": False,
            },
        )
        results = r.json()["results"]
        # At least one result should have day-by-day data
        result = results[0]
        assert len(result["days"]) == 5
        day = result["days"][0]
        assert "date" in day
        assert "predicted" in day
        assert "actual" in day
        assert "trade_pnl_net" in day

    def test_backtest_with_stop_loss(self, refreshed_client):
        r = refreshed_client.post(
            "/api/backtest",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "period": "1y",
                "stop_loss_pct": 2.0,
                "refresh_data": False,
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        # With SL, should have both baseline and SL variants
        names = [r["model"] for r in results]
        has_sl = any("SL" in n for n in names)
        has_baseline = any("SL" not in n for n in names)
        assert has_sl
        assert has_baseline

    def test_backtest_with_fees(self, refreshed_client):
        r_no_fee = refreshed_client.post(
            "/api/backtest",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "period": "1y",
                "fee_pct": 0.0,
                "refresh_data": False,
            },
        )
        r_fee = refreshed_client.post(
            "/api/backtest",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "period": "1y",
                "fee_pct": 0.5,
                "refresh_data": False,
            },
        )
        # Same model, first result — fee version should have lower return
        no_fee_ret = r_no_fee.json()["results"][0]["total_return"]
        fee_ret = r_fee.json()["results"][0]["total_return"]
        assert fee_ret < no_fee_ret

    def test_backtest_best_models(self, refreshed_client):
        r = refreshed_client.post(
            "/api/backtest",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "period": "1y",
                "refresh_data": False,
            },
        )
        data = r.json()
        assert data["best_by_return"] is not None
        assert data["best_by_sharpe"] is not None
        assert "model" in data["best_by_return"]

    def test_backtest_compare_periods(self, refreshed_client):
        r = refreshed_client.post(
            "/api/backtest",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "compare_periods": True,
                "refresh_data": False,
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        # Should have results from multiple periods
        periods = set(r["period"] for r in results)
        assert len(periods) >= 2  # at least 1y and 2y should work


# ------------------------------------------------------------------
# Training endpoints
# ------------------------------------------------------------------


class TestTraining:
    def test_list_models_empty(self, client):
        r = client.get("/api/train/models")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_training_status_not_found(self, client):
        r = client.get("/api/train/status/nonexistent_key")
        assert r.status_code == 200
        assert r.json()["status"] == "not_found"


# ------------------------------------------------------------------
# Settings endpoints
# ------------------------------------------------------------------


class TestSettings:
    def test_get_defaults(self, client):
        r = client.get("/api/settings")
        assert r.status_code == 200
        s = r.json()
        assert "default_period" in s
        assert "knn_k" in s
        assert "default_fee_pct" in s
        assert s["knn_k"] == 5

    def test_patch_settings(self, client):
        r = client.patch("/api/settings", json={"knn_k": 8})
        assert r.status_code == 200
        assert r.json()["knn_k"] == 8

        # Verify persisted
        r2 = client.get("/api/settings")
        assert r2.json()["knn_k"] == 8

        # Reset
        client.patch("/api/settings", json={"knn_k": 5})

    def test_put_settings(self, client):
        r = client.get("/api/settings")
        original = r.json()

        # Full replace
        modified = {**original, "default_fee_pct": 0.1}
        r2 = client.put("/api/settings", json=modified)
        assert r2.status_code == 200
        assert r2.json()["default_fee_pct"] == 0.1

        # Restore
        client.put("/api/settings", json=original)


# ------------------------------------------------------------------
# Analysis endpoints
# ------------------------------------------------------------------


class TestAnalysis:
    def test_news_comparison(self, refreshed_client):
        r = refreshed_client.post(
            "/api/analysis/news-comparison",
            json={
                "tickers": ["TEST"],
                "days": 5,
                "period": "1y",
                "fee_pct": 0.03,
            },
        )
        assert r.status_code == 200
        results = r.json()
        assert isinstance(results, list)
        if len(results) > 0:
            row = results[0]
            assert "ticker" in row
            assert "model" in row
            assert "return_no_news" in row
            assert "return_with_news" in row
            assert "diff" in row
            assert "sharpe_no_news" in row
            assert "accuracy_no_news" in row
