"""
test_pipeline.py – Verify the full pipeline works with mock data (no network needed).
"""

import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch


def make_fake_prices(days: int = 400) -> pd.DataFrame:
    """Generate synthetic price data resembling a real stock."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    n = len(dates)
    base = 150.0
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base * np.cumprod(1 + returns)

    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "Close": prices,
        "Volume": np.random.randint(1_000_000, 50_000_000, n),
    }, index=dates)
    df.index.name = "Date"
    return df


def make_fake_news() -> list:
    """Return fake news items in the yfinance 1.3.0 XHR format."""
    return [
        {"content": {"title": "Stock rally continues with strong growth"}},
        {"content": {"title": "Analysts upgrade outlook, profit expected"}},
        {"content": {"title": "Market risk warning: debt levels rising"}},
    ]


# Patch yfinance so nothing hits the network
with patch("engine.data_downloader.yf") as mock_yf, \
     patch("engine.news_scraper.yf") as mock_news_yf:

    mock_yf.Ticker.return_value.history.return_value = make_fake_prices()
    type(mock_news_yf.Ticker.return_value).news = property(lambda self: make_fake_news())

    # Import after patching so modules pick up the mocked yfinance
    from interface.api import StockAppAPI, PredictionConfig
    from engine.backtester import Backtester

    api = StockAppAPI()

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 1: Data download + DB save/load")
    print("=" * 60)
    df = api.get_data("TEST", period="1y")
    print(f"  Rows loaded: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    assert len(df) > 0, "No data loaded!"
    assert "close" in df.columns, "Missing 'close' column!"
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 2: k-NN prediction (standard)")
    print("=" * 60)
    cfg = PredictionConfig(ticker="TEST", period="1y", model_type="knn",
                           use_time_weights=False, include_news=False)
    result = api.get_prediction(cfg)
    print(f"  Prediction: {result.prediction}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Last price: {result.last_price:.2f}")
    print(f"  Data points: {result.data_points}")
    assert result.prediction in ("UP", "DOWN"), f"Bad prediction: {result.prediction}"
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 3: k-NN prediction (time-weighted)")
    print("=" * 60)
    cfg_tw = PredictionConfig(ticker="TEST", period="1y", model_type="knn",
                              use_time_weights=True, include_news=False)
    result_tw = api.get_prediction(cfg_tw)
    print(f"  Prediction: {result_tw.prediction}")
    print(f"  Confidence: {result_tw.confidence}")
    assert result_tw.prediction in ("UP", "DOWN")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 4: LinReg prediction (standard + time-weighted)")
    print("=" * 60)
    for tw in [False, True]:
        label = "Time-Weighted" if tw else "Standard"
        cfg_lr = PredictionConfig(ticker="TEST", period="1y", model_type="linreg",
                                  use_time_weights=tw, include_news=False)
        res_lr = api.get_prediction(cfg_lr)
        print(f"  {label}: {res_lr.prediction} {res_lr.confidence}")
        assert res_lr.prediction in ("UP", "DOWN")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 5: News sentiment integration")
    print("=" * 60)
    for model in ["knn", "linreg"]:
        cfg_news = PredictionConfig(ticker="TEST", period="1mo", model_type=model,
                                    use_time_weights=True, include_news=True)
        res_news = api.get_prediction(cfg_news)
        print(f"  {model}: {res_news.prediction} {res_news.confidence} "
              f"sentiment={res_news.sentiment} ({res_news.sentiment_score})")
        assert res_news.prediction in ("UP", "DOWN")
    assert len(res_news.headlines) > 0, "No headlines!"
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 6: Full report simulation (all models × all periods)")
    print("=" * 60)
    for p in ["1mo", "1y", "2y", "max"]:
        for model in ["knn", "linreg"]:
            cfg_std = PredictionConfig(ticker="TEST", period=p, model_type=model,
                                       use_time_weights=False, include_news=False)
            r = api.get_prediction(cfg_std)
            print(f"  {p:<6} {model:<8} Standard:      {r.prediction:<6} {r.confidence:<8} ({r.data_points} pts)")

            cfg_wgh = PredictionConfig(ticker="TEST", period=p, model_type=model,
                                       use_time_weights=True, include_news=False)
            r2 = api.get_prediction(cfg_wgh)
            print(f"  {'':<6} {'':<8} Time-Weighted: {r2.prediction:<6} {r2.confidence:<8} ({r2.data_points} pts)")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 7: Walk-forward backtest")
    print("=" * 60)
    full_df = api.get_data("TEST", period="max")
    backtester = Backtester(n_days=5)
    for model, name in [(api.knn, "k-NN"), (api.linreg, "LinReg")]:
        bt = backtester.run(model=model, model_name=name, df=full_df, ticker="TEST")
        print(f"  {name}: {bt.correct}/{bt.test_days} correct ({bt.accuracy:.0%})")
        assert bt.test_days == 5, f"Expected 5 test days, got {bt.test_days}"
        assert 0.0 <= bt.accuracy <= 1.0
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" ALL TESTS PASSED ✓")
    print("=" * 60)
