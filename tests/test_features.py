"""test_features.py – Feature matrix and indicator tests."""

import numpy as np
import pytest

from engine.features import (
    ALL_FEATURES,
    DEFAULT_FEATURES,
    build_feature_matrix,
    min_rows_needed,
    validate_features,
)


class TestFeatureMatrix:
    """Feature matrix shape and content."""

    def test_naive_binary_shape(self, full_df):
        X, y = build_feature_matrix(full_df, DEFAULT_FEATURES, 5, "binary")
        assert X.shape[1] == 5  # 5 return windows
        assert set(y.astype(int)) <= {0, 1}

    def test_naive_continuous_shape(self, full_df):
        X, y = build_feature_matrix(full_df, DEFAULT_FEATURES, 5, "continuous")
        assert X.shape[1] == 5
        assert y.dtype in (np.float32, np.float64)

    def test_enhanced_has_13_features(self, full_df):
        X, y = build_feature_matrix(full_df, ALL_FEATURES, 5, "binary")
        assert X.shape[1] == 13  # 5 ret + 5 vol + RSI + volat + MACD

    def test_min_rows_needed(self):
        assert min_rows_needed(DEFAULT_FEATURES, 5) < min_rows_needed(ALL_FEATURES, 5)
        assert min_rows_needed(ALL_FEATURES, 5) >= 35  # MACD needs 26+

    def test_invalid_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            validate_features(["returns", "nonexistent"])

    def test_no_nan_in_matrix(self, full_df):
        X, y = build_feature_matrix(full_df, ALL_FEATURES, 5, "binary")
        assert not np.isnan(X).any(), "NaN in feature matrix"
        assert not np.isnan(y).any(), "NaN in labels"

    def test_fewer_rows_than_window(self, tiny_prices):
        """With 10 rows and window 5+MACD warmup, should return None or tiny."""
        from unittest.mock import patch

        with patch("engine.data_downloader.yf") as m:
            m.Ticker.return_value.history.return_value = tiny_prices
            from interface.api import StockAppAPI

            api = StockAppAPI()
            df = api.get_data("TINY", period="max")
            result = build_feature_matrix(df, ALL_FEATURES, 5, "binary")
            # Returns (X, y) tuple or (None, None) for insufficient data
            if result is None:
                pass  # OK
            elif result[0] is None:
                pass  # OK — returns (None, None)
            else:
                assert len(result[0]) < 5
