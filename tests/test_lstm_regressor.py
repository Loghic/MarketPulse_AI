"""test_lstm_regressor.py — LSTM regression point-forecaster.

The torch-free paths (sequence building, weight-path naming, graceful skip when
torch/weights are absent) run everywhere. The training + forward-pass tests
``importorskip("torch")`` so the suite stays green without the optional dep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.lstm_regressor import (
    REG_TRAINING_PRESETS,
    LSTMRegressorForecaster,
    build_sequences,
    regressor_path,
)


def _df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="B").astype(str)
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({"date": dates, "close": closes, "volume": vol})


class TestTorchFreePaths:
    def test_regressor_path_distinct_from_classifier(self):
        # Must not collide with {ticker}_{period}_{preset}.pt classifiers.
        p = regressor_path("AAPL")
        assert p.name == "AAPL_reg.pt"

    def test_build_sequences_shape(self):
        b = build_sequences(_df(200), ["returns"], window_size=20, horizon=1)
        assert b is not None
        x, y, cols = b
        assert x.ndim == 3 and x.shape[1] == 20 and x.shape[2] == 1
        assert x.shape[0] == y.shape[0]
        assert cols == ["_ret"]

    def test_build_sequences_delta_target(self):
        # Target is close[t+h] - close[t]; verify on a known ramp.
        closes = list(range(100, 160))  # +1 each step
        df = pd.DataFrame({"close": closes})
        b = build_sequences(df, ["returns"], window_size=5, horizon=1)
        assert b is not None
        _, y, _ = b
        # Every one-step delta on a +1 ramp is +1.
        assert np.allclose(y, 1.0)

    def test_build_sequences_too_short(self):
        assert build_sequences(_df(5), ["returns"], 20, 1) is None

    def test_multifeature_sequence_width(self):
        b = build_sequences(_df(200), ["returns", "volume"], window_size=15, horizon=1)
        assert b is not None
        x, _, cols = b
        assert x.shape[2] == 2
        assert cols == ["_ret", "_vol_chg"]

    def test_forecast_skips_without_weights(self):
        # No models/AAPL_reg.pt in the test env (and/or no torch) -> None, no raise.
        m = LSTMRegressorForecaster("NOPE_TICKER")
        assert m.forecast(_df(200)) is None

    def test_presets_defined(self):
        # The three training-effort tiers exist with the expected capacity knobs.
        assert set(REG_TRAINING_PRESETS) == {"quick", "standard", "cluster"}
        for cfg in REG_TRAINING_PRESETS.values():
            assert {"hidden_size", "num_layers", "dropout", "epochs"} <= set(cfg)
        # Capacity should increase across tiers.
        assert (
            REG_TRAINING_PRESETS["quick"]["hidden_size"]
            < REG_TRAINING_PRESETS["standard"]["hidden_size"]
            < REG_TRAINING_PRESETS["cluster"]["hidden_size"]
        )


class TestWithTorch:
    def test_train_and_forecast_roundtrip(self, tmp_path):
        pytest.importorskip("torch")
        import torch

        from engine.lstm_regressor import train_regressor

        df = _df(400, seed=1)
        # Use the quick preset but override epochs to keep the test fast.
        ckpt = train_regressor(df, preset="quick", window_size=10, horizon=1, epochs=3)
        assert ckpt is not None
        assert ckpt["input_size"] == 1
        assert "y_mean" in ckpt and "y_std" in ckpt
        assert ckpt["preset"] == "quick"
        # Quick preset capacity is recorded in the config.
        assert ckpt["config"]["hidden_size"] == REG_TRAINING_PRESETS["quick"]["hidden_size"]

        # Save where the per-ticker forecaster will look, then forecast.
        path = regressor_path("TST", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

        m = LSTMRegressorForecaster("TST", models_dir=tmp_path)
        fr = m.forecast(df)
        assert fr is not None
        assert np.isfinite(fr.point)
        # The forecast is a price level near the last close (a sane Δ off it),
        # not a raw Δ — sanity-bound it to the recent price range.
        assert df["close"].min() * 0.5 < fr.point < df["close"].max() * 1.5
