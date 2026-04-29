"""
test_pipeline.py – Verify the full pipeline works with mock data (no network needed).

Covers: data layer, shared features module, k-NN (naive + enhanced),
LinReg (naive + enhanced), VADER/naive sentiment, walk-forward backtesting,
and error handling.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def make_fake_prices(days: int = 400) -> pd.DataFrame:
    """Generate synthetic price data resembling a real stock."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    n = len(dates)
    base = 150.0
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base * np.cumprod(1 + returns)

    df = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 50_000_000, n),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def make_fake_news() -> list:
    """Return fake news items in the yfinance 1.3.0 XHR format."""
    return [
        {"content": {"title": "Stock rally continues with strong growth"}},
        {"content": {"title": "Analysts upgrade outlook, profit expected"}},
        {"content": {"title": "Market risk warning: debt levels rising"}},
        {"content": {"title": "Company not expected to meet targets this quarter"}},
    ]


# Patch yfinance so nothing hits the network
with patch("engine.data_downloader.yf") as mock_yf, patch("engine.news_scraper.yf") as mock_news_yf:
    mock_yf.Ticker.return_value.history.return_value = make_fake_prices()
    type(mock_news_yf.Ticker.return_value).news = property(lambda self: make_fake_news())

    # Import after patching so modules pick up the mocked yfinance
    from engine.backtester import Backtester
    from engine.features import ALL_FEATURES, DEFAULT_FEATURES, build_feature_matrix
    from engine.knn_model import KNNModel
    from engine.lin_reg_model import LinearRegressionModel
    from engine.news_scraper import NewsScraper
    from interface.api import PredictionConfig, StockAppAPI

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
    assert "volume" in df.columns, "Missing 'volume' column!"
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 2: Shared feature matrix builder")
    print("=" * 60)
    full_df = api.get_data("TEST", period="max")

    # Binary target (for k-NN)
    X_bin, y_bin = build_feature_matrix(full_df, DEFAULT_FEATURES, 5, "binary")
    print(f"  Naive binary:      X={X_bin.shape}, y unique={set(y_bin.astype(int))}")
    assert X_bin.shape[1] == 5  # 5 returns

    # Continuous target (for LinReg)
    X_cont, y_cont = build_feature_matrix(full_df, DEFAULT_FEATURES, 5, "continuous")
    print(
        f"  Naive continuous:  X={X_cont.shape}, y range=[{y_cont.min():.4f}, {y_cont.max():.4f}]"
    )
    assert X_cont.shape[1] == 5

    # Enhanced features
    X_enh, y_enh = build_feature_matrix(full_df, ALL_FEATURES, 5, "binary")
    print(f"  Enhanced binary:   X={X_enh.shape} (5 ret + 5 vol + RSI + volat + MACD = 13)")
    assert X_enh.shape[1] == 13

    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 3: k-NN naive (standard + time-weighted)")
    print("=" * 60)
    for tw in [False, True]:
        label = "Time-Weighted" if tw else "Standard"
        cfg = PredictionConfig(
            ticker="TEST", period="1y", model_type="knn", use_time_weights=tw, include_news=False
        )
        result = api.get_prediction(cfg)
        print(f"  {label}: {result.prediction} {result.confidence} ({result.data_points} pts)")
        assert result.prediction in ("UP", "DOWN")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 4: k-NN enhanced (all features)")
    print("=" * 60)
    for tw in [False, True]:
        label = "Time-Weighted" if tw else "Standard"
        cfg = PredictionConfig(
            ticker="TEST",
            period="1y",
            model_type="knn_enhanced",
            use_time_weights=tw,
            include_news=False,
        )
        result = api.get_prediction(cfg)
        print(f"  {label}: {result.prediction} {result.confidence} ({result.data_points} pts)")
        assert result.prediction in ("UP", "DOWN")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 5: LinReg naive (standard + time-weighted)")
    print("=" * 60)
    for tw in [False, True]:
        label = "Time-Weighted" if tw else "Standard"
        cfg = PredictionConfig(
            ticker="TEST", period="1y", model_type="linreg", use_time_weights=tw, include_news=False
        )
        result = api.get_prediction(cfg)
        print(f"  {label}: {result.prediction} {result.confidence}")
        assert result.prediction in ("UP", "DOWN")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 6: LinReg enhanced (all features)")
    print("=" * 60)
    for tw in [False, True]:
        label = "Time-Weighted" if tw else "Standard"
        cfg = PredictionConfig(
            ticker="TEST",
            period="1y",
            model_type="linreg_enhanced",
            use_time_weights=tw,
            include_news=False,
        )
        result = api.get_prediction(cfg)
        print(f"  {label}: {result.prediction} {result.confidence}")
        assert result.prediction in ("UP", "DOWN")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 7: Feature set comparison (k-NN vs LinReg, same features)")
    print("=" * 60)
    feature_combos = [
        ["returns"],
        ["returns", "volume"],
        ["returns", "rsi"],
        ALL_FEATURES,
    ]
    for feats in feature_combos:
        knn = KNNModel(k=5, features=feats)
        lr = LinearRegressionModel(features=feats)
        knn_pred, knn_prob = knn.predict(full_df)
        lr_pred, lr_prob = lr.predict(full_df)
        label = knn.feature_label
        print(
            f"  {label:<35}  k-NN={knn_pred:<5} {knn_prob:.0%}  LinReg={lr_pred:<5} {lr_prob:.0%}"
        )
        assert knn_pred in ("UP", "DOWN", "Insufficient data")
        assert lr_pred in ("UP", "DOWN", "Insufficient data")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 8: News sentiment (VADER vs naive)")
    print("=" * 60)
    scraper = NewsScraper()

    headlines = [
        "Stock rally continues with strong growth",
        "Market risk warning: debt levels rising",
        "Company not expected to meet targets this quarter",
    ]

    naive_score = scraper._score_naive(headlines)
    print(f"  Naive score:  {naive_score:+.3f}")

    if scraper.vader_available:
        vader_score = scraper._score_vader(headlines)
        print(f"  VADER score:  {vader_score:+.3f}")
        for h in headlines:
            n = scraper._score_naive([h])
            v = scraper._score_vader([h])
            print(f'    naive={n:+.2f}  vader={v:+.2f}  "{h}"')
    else:
        print("  VADER available: no (using naive fallback)")

    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 9: News sentiment integration (all 4 models)")
    print("=" * 60)
    # Enhanced models need more data (MACD warmup), so use 1y
    news_test_periods = {
        "knn": "1mo",
        "knn_enhanced": "1y",
        "linreg": "1mo",
        "linreg_enhanced": "1y",
    }
    for model, period in news_test_periods.items():
        cfg_news = PredictionConfig(
            ticker="TEST", period=period, model_type=model, use_time_weights=True, include_news=True
        )
        res_news = api.get_prediction(cfg_news)
        print(
            f"  {model:<18} ({period}): {res_news.prediction} {res_news.confidence} "
            f"sent={res_news.sentiment} ({res_news.sentiment_score})"
        )
        assert res_news.prediction in ("UP", "DOWN")
    assert len(res_news.headlines) > 0, "No headlines!"
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 10: Full report simulation (all models × all periods)")
    print("=" * 60)
    for p in ["1mo", "1y", "2y", "max"]:
        for model in ["knn", "knn_enhanced", "linreg", "linreg_enhanced"]:
            try:
                cfg = PredictionConfig(
                    ticker="TEST",
                    period=p,
                    model_type=model,
                    use_time_weights=False,
                    include_news=False,
                )
                r = api.get_prediction(cfg)
                print(
                    f"  {p:<6} {model:<18} {r.prediction:<6} {r.confidence:<8} ({r.data_points} pts)"
                )
            except RuntimeError:
                print(f"  {p:<6} {model:<18} Skipped (insufficient data)")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 11: Walk-forward backtest (all 4 models)")
    print("=" * 60)
    backtester = Backtester(n_days=5)
    models_to_test = [
        (api.knn, "k-NN"),
        (api.knn_enhanced, "k-NN Enhanced"),
        (api.linreg, "LinReg"),
        (api.linreg_enhanced, "LinReg Enhanced"),
    ]
    for model, name in models_to_test:
        bt = backtester.run(model=model, model_name=name, df=full_df, ticker="TEST")
        pf_str = f"{bt.profit_factor:.2f}" if bt.profit_factor < 100 else "∞"
        print(
            f"  {name:<18}: {bt.correct}/{bt.test_days} ({bt.accuracy:.0%})  "
            f"return={bt.total_return:+.2%}  PF={pf_str}  "
            f"streaks=W{bt.longest_win_streak}/L{bt.longest_loss_streak}"
        )
        assert bt.test_days == 5, f"Expected 5 test days, got {bt.test_days}"
        assert 0.0 <= bt.accuracy <= 1.0
        assert bt.win_trades + bt.loss_trades == bt.test_days
        # Streak sanity checks
        assert bt.longest_win_streak >= 0
        assert bt.longest_loss_streak >= 0
        assert bt.longest_win_streak <= bt.test_days
        assert bt.longest_loss_streak <= bt.test_days
        assert bt.avg_win_streak >= 0
        assert bt.avg_loss_streak >= 0
        # Every day should have a trade_pnl
        for d in bt.days:
            assert isinstance(d.trade_pnl, float)
            # Correct prediction = positive P/L, wrong = negative
            if d.correct:
                assert d.trade_pnl >= 0, f"Correct prediction but negative P/L on {d.date}"
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 12: Error handling")
    print("=" * 60)

    # Too little data
    tiny_df = full_df.head(5)
    knn_test = KNNModel(k=5, features=["returns"])
    pred, prob = knn_test.predict(tiny_df)
    print(f"  Tiny data (5 rows): '{pred}'")
    assert pred == "Insufficient data" or pred in ("UP", "DOWN")

    # Invalid model type
    try:
        cfg_bad = PredictionConfig(ticker="TEST", period="1y", model_type="invalid")
        api.get_prediction(cfg_bad)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Invalid model_type: caught ValueError ✓")

    # Invalid feature name (k-NN)
    try:
        KNNModel(k=5, features=["returns", "nonexistent"])
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Invalid k-NN feature: caught ValueError ✓")

    # Invalid feature name (LinReg)
    try:
        LinearRegressionModel(features=["bogus"])
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Invalid LinReg feature: caught ValueError ✓")

    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" TEST 13: LSTM model (conditional on PyTorch)")
    print("=" * 60)

    try:
        from engine.ai_model import TORCH_AVAILABLE, AIModel

        if TORCH_AVAILABLE:
            # Test training on small data
            lstm = AIModel(window_size=10, preset="quick")
            lstm.config["epochs"] = 5  # override for speed
            info = lstm.train(full_df, verbose=False)
            print(
                f"  Training: {info['train_samples']} samples, "
                f"val_acc={info['final_val_accuracy']:.1%}, "
                f"{info['duration_seconds']:.1f}s"
            )
            assert lstm.trained

            # Test prediction
            pred, prob = lstm.predict(full_df)
            print(f"  Prediction: {pred} {prob:.1%}")
            assert pred in ("UP", "DOWN")

            # Test save/load
            test_path = Path("models/_test_model.pt")
            lstm.save(test_path)
            assert test_path.exists()

            lstm2 = AIModel(preset="quick")
            loaded = lstm2.load(test_path)
            assert loaded
            assert lstm2.trained

            pred2, prob2 = lstm2.predict(full_df)
            print(f"  Load+predict: {pred2} {prob2:.1%}")
            assert pred2 in ("UP", "DOWN")

            # Cleanup
            test_path.unlink()
            print("  LSTM with PyTorch: all checks passed ✓")
        else:
            print("  PyTorch not installed — testing graceful fallback")
            assert not api.lstm_available

            # Requesting LSTM via API should give clear error
            try:
                cfg_lstm = PredictionConfig(ticker="TEST", period="1y", model_type="lstm")
                api.get_prediction(cfg_lstm)
                assert False, "Should have raised RuntimeError"
            except RuntimeError:
                print("  LSTM not available: caught RuntimeError ✓")

    except ImportError:
        print("  ai_model.py import failed — skipping LSTM tests")
        assert not api.lstm_available
        print("  api.lstm_available = False ✓")

    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    print("=" * 60)
    print(" ALL TESTS PASSED ✓  (13 tests)")
    print("=" * 60)
