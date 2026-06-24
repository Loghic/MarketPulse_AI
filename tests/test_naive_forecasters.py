"""test_naive_forecasters.py — Regression baselines (RW / drift / seasonal)."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.naive_forecasters import (
    RandomWalkDriftForecaster,
    RandomWalkForecaster,
    SeasonalNaiveForecaster,
    default_naive_forecasters,
)


def _df(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="B").astype(str)
    return pd.DataFrame({"date": dates, "close": closes})


class TestRandomWalk:
    def test_point_equals_last_close(self):
        fr = RandomWalkForecaster()._raw_forecast(_df([100, 101, 102]))
        assert fr.point == 102.0
        assert fr.last_close == 102.0

    def test_horizon_still_no_change(self):
        fr = RandomWalkForecaster()._raw_forecast(_df([100, 101, 102]), horizon=5)
        assert fr.point == 102.0

    def test_single_row_ok(self):
        fr = RandomWalkForecaster()._raw_forecast(_df([100]))
        assert fr is not None and fr.point == 100.0


class TestRandomWalkDrift:
    def test_adds_mean_change(self):
        # diffs of [100,102,104,106] -> mean 2; point = 106 + 1*2 = 108.
        fr = RandomWalkDriftForecaster()._raw_forecast(_df([100, 102, 104, 106]))
        assert fr.point == pytest.approx(108.0)

    def test_horizon_scales_drift(self):
        fr = RandomWalkDriftForecaster()._raw_forecast(_df([100, 102, 104, 106]), horizon=3)
        assert fr.point == pytest.approx(106 + 3 * 2.0)

    def test_too_short_returns_none(self):
        assert RandomWalkDriftForecaster()._raw_forecast(_df([100])) is None


class TestSeasonalNaive:
    def test_repeats_value_m_back(self):
        # m=5, h=1 -> back=5 -> closes[-5].
        closes = [10, 11, 12, 13, 14, 15, 18]
        fr = SeasonalNaiveForecaster(m=5)._raw_forecast(_df(closes))
        assert fr.point == float(closes[-5])

    def test_falls_back_to_rw_when_short(self):
        # Not enough history for a full season -> random walk (last close).
        fr = SeasonalNaiveForecaster(m=20)._raw_forecast(_df([100, 101, 102]))
        assert fr.point == 102.0

    def test_invalid_m_raises(self):
        with pytest.raises(ValueError):
            SeasonalNaiveForecaster(m=0)


class TestFactory:
    def test_default_set(self):
        pairs = default_naive_forecasters(season=5)
        labels = [label for _, label in pairs]
        assert "Random Walk" in labels
        assert "Random Walk + Drift" in labels
        assert any("Seasonal Naive" in label for label in labels)

    def test_predict_contract(self):
        # All forecasters satisfy the (str, float) predict() contract via the base.
        df = _df([100, 101, 102, 103, 104, 105, 106])
        for model, _ in default_naive_forecasters():
            direction, conf = model.predict(df)
            assert direction in ("UP", "DOWN")
            assert 0.0 <= conf <= 1.0
