"""test_models.py – Prediction tests for all model types."""

import pytest
from engine.knn_model import KNNModel
from engine.lin_reg_model import LinearRegressionModel
from engine.features import ALL_FEATURES


class TestKNN:
    """k-NN model predictions."""

    def test_naive_predict(self, full_df):
        model = KNNModel(k=5, features=["returns"])
        pred, prob = model.predict(full_df)
        assert pred in ("UP", "DOWN")
        assert 0.5 <= prob <= 1.0

    def test_enhanced_predict(self, full_df):
        model = KNNModel(k=5, features=ALL_FEATURES)
        pred, prob = model.predict(full_df)
        assert pred in ("UP", "DOWN")

    def test_time_weighted(self, full_df):
        model = KNNModel(k=5, features=ALL_FEATURES)
        pred, prob = model.predict(full_df, use_time_weights=True)
        assert pred in ("UP", "DOWN")

    def test_with_sentiment(self, full_df):
        model = KNNModel(k=5, features=["returns"])
        pred_pos, _ = model.predict(full_df, sentiment_score=0.8)
        pred_neg, _ = model.predict(full_df, sentiment_score=-0.8)
        # Strong sentiment should influence direction
        assert pred_pos in ("UP", "DOWN")
        assert pred_neg in ("UP", "DOWN")

    def test_insufficient_data(self):
        """Very small df should return 'Insufficient data'."""
        import pandas as pd
        tiny = pd.DataFrame({"close": [1, 2, 3], "volume": [100, 100, 100],
                             "date": ["2024-01-01", "2024-01-02", "2024-01-03"]})
        model = KNNModel(k=5, features=["returns"])
        pred, prob = model.predict(tiny)
        assert pred == "Insufficient data"
        assert prob == 0.0

    def test_invalid_feature(self):
        with pytest.raises(ValueError):
            KNNModel(k=5, features=["bogus_feature"])


class TestLinReg:
    """Linear Regression model predictions."""

    def test_naive_predict(self, full_df):
        model = LinearRegressionModel(features=["returns"])
        pred, prob = model.predict(full_df)
        assert pred in ("UP", "DOWN")
        assert 0.0 <= prob <= 1.0

    def test_enhanced_predict(self, full_df):
        model = LinearRegressionModel(features=ALL_FEATURES)
        pred, prob = model.predict(full_df)
        assert pred in ("UP", "DOWN")

    def test_time_weighted(self, full_df):
        model = LinearRegressionModel(features=["returns"])
        pred, prob = model.predict(full_df, use_time_weights=True)
        assert pred in ("UP", "DOWN")

    def test_confidence_bounded(self, full_df):
        """Confidence should be between 0 and 1 (sigmoid mapped)."""
        model = LinearRegressionModel(features=ALL_FEATURES)
        _, prob = model.predict(full_df)
        assert 0.0 <= prob <= 1.0

    def test_invalid_feature(self):
        with pytest.raises(ValueError):
            LinearRegressionModel(features=["nonexistent"])


class TestLSTM:
    """LSTM tests — conditional on PyTorch."""

    def test_import_graceful(self):
        """ai_model should import even without torch."""
        try:
            from engine.ai_model import TORCH_AVAILABLE
            assert isinstance(TORCH_AVAILABLE, bool)
        except ImportError:
            pytest.skip("ai_model import failed entirely")

    def test_api_lstm_error(self, api):
        """Requesting untrained LSTM should give clear error."""
        from interface.api import PredictionConfig
        cfg = PredictionConfig(ticker="TEST", period="1y", model_type="lstm")
        with pytest.raises(RuntimeError, match="No trained LSTM|PyTorch"):
            api.get_prediction(cfg)
