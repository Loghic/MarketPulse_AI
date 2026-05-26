"""
conftest.py – Shared fixtures for all tests.

All tests use mock data — no network access needed.
"""

from datetime import datetime
from unittest.mock import PropertyMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_prices(days=400, seed=42, base=150.0, daily_return=0.0005, volatility=0.02):
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    n = len(dates)
    returns = np.random.normal(daily_return, volatility, n)
    prices = base * np.cumprod(1 + returns)

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
    """Fake news in yfinance format."""
    return [
        {"content": {"title": "Stock rally continues with strong growth"}},
        {"content": {"title": "Analysts upgrade outlook, profit expected"}},
        {"content": {"title": "Market risk warning: debt levels rising"}},
        {"content": {"title": "Company not expected to meet targets"}},
    ]


@pytest.fixture
def fake_prices():
    """400 days of synthetic price data."""
    return _make_prices(400)


@pytest.fixture
def small_prices():
    """50 days — fast tests."""
    return _make_prices(50)


@pytest.fixture
def tiny_prices():
    """10 days — edge case tests."""
    return _make_prices(10)


@pytest.fixture
def mock_yfinance():
    """Patch yfinance globally — returns 400 days of data."""
    prices = _make_prices(400)
    news = _make_news()

    # News fetching moved to engine.news_sources in the 2026 refactor.
    # We patch both download + news-source modules so old and new code
    # paths see the same fake data.
    with (
        patch("engine.data_downloader.yf") as mock_dl,
        patch("engine.news_sources.yf") as mock_ns,
    ):
        mock_dl.Ticker.return_value.history.return_value = prices
        type(mock_ns.Ticker.return_value).news = PropertyMock(return_value=news)
        yield mock_dl, mock_ns


@pytest.fixture
def api(mock_yfinance, tmp_path, monkeypatch):
    """StockAppAPI with mocked data layer **and an isolated DB**.

    ``DatabaseManager`` builds its path from ``Path("data") / db_name``,
    which is relative to CWD. Without the ``monkeypatch.chdir`` below,
    every test that touched ``api.get_data("SPY")`` etc. would persist
    the mocked synthetic fixture into the real ``data/market_data.db``
    under real ticker names — see scripts/clean_test_contamination.py
    for the cleanup of that historical mess.
    """
    monkeypatch.chdir(tmp_path)
    from interface.api import StockAppAPI

    return StockAppAPI()


@pytest.fixture
def full_df(api):
    """Full DataFrame from mocked API."""
    return api.get_data("TEST", period="max")
