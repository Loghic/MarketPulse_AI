"""test_forecast_backtester.py — Walk-forward point-forecast harness.

Covers the leakage guarantee (the model never sees the realised target), the
random-walk U2==1 invariant end-to-end, and short-series handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.forecast_backtester import ForecastBacktester
from engine.naive_forecasters import RandomWalkForecaster


def _df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="B").astype(str)
    return pd.DataFrame({"date": dates, "close": closes})


class TestWalkForward:
    def test_random_walk_u2_is_one(self):
        bt = ForecastBacktester(n_days=50, horizon=1, min_train=40)
        run = bt.run(RandomWalkForecaster(), _df(200), "T")
        assert run is not None and run.metrics is not None
        assert run.metrics.theil_u2 == pytest.approx(1.0)
        assert run.metrics.n == 50

    def test_too_short_returns_none(self):
        bt = ForecastBacktester(n_days=50, horizon=1, min_train=60)
        assert bt.run(RandomWalkForecaster(), _df(50), "T") is None

    def test_horizon_shrinks_scoreable(self):
        # With horizon h, the last h steps can't be scored (no future truth).
        bt1 = ForecastBacktester(n_days=200, horizon=1, min_train=40)
        bt5 = ForecastBacktester(n_days=200, horizon=5, min_train=40)
        n1 = bt1.run(RandomWalkForecaster(), _df(150), "T").metrics.n
        n5 = bt5.run(RandomWalkForecaster(), _df(150), "T").metrics.n
        assert n5 == n1 - 4  # lose exactly (h-1) extra steps at the tail

    def test_no_leakage(self):
        """The model must only ever receive rows up to t — never the target t+h."""
        df = _df(150)
        h = 1
        seen_lengths: list[int] = []

        class _Spy(RandomWalkForecaster):
            def forecast(self, frame, horizon=1):  # type: ignore[override]
                seen_lengths.append(len(frame))
                return super().forecast(frame, horizon)

        bt = ForecastBacktester(n_days=40, horizon=h, min_train=60)
        run = bt.run(_Spy(), df, "T")
        # Every window length must be <= (its t)+1, and strictly less than the
        # full frame (the last h rows are never visible as inputs).
        assert run is not None
        assert max(seen_lengths) <= len(df) - h
        # The realised value for the last scored step is at index len-1; the
        # input window that produced it ended at len-1-h, so never included it.
        assert max(seen_lengths) < len(df)

    def test_max_train_caps_window(self):
        """The model must never see more than max_train rows, but U2 still holds."""
        df = _df(2000)
        seen: list[int] = []

        class _Spy(RandomWalkForecaster):
            def forecast(self, frame, horizon=1):  # type: ignore[override]
                seen.append(len(frame))
                return super().forecast(frame, horizon)

        bt = ForecastBacktester(n_days=50, horizon=1, min_train=60, max_train=300)
        run = bt.run(_Spy(), df, "T")
        assert max(seen) <= 300
        # The random-walk invariant survives the cap.
        assert run.metrics.theil_u2 == pytest.approx(1.0)

    def test_max_train_none_is_full_window(self):
        df = _df(800)
        seen: list[int] = []

        class _Spy(RandomWalkForecaster):
            def forecast(self, frame, horizon=1):  # type: ignore[override]
                seen.append(len(frame))
                return super().forecast(frame, horizon)

        bt = ForecastBacktester(n_days=50, horizon=1, min_train=60, max_train=None)
        bt.run(_Spy(), df, "T")
        # Uncapped: the last window is the full series minus the unscoreable tail.
        assert max(seen) > 700

    def test_max_train_below_min_train_raises(self):
        with pytest.raises(ValueError):
            ForecastBacktester(n_days=50, min_train=60, max_train=30)

    def test_bad_target_raises(self):
        with pytest.raises(ValueError):
            ForecastBacktester(n_days=50, target="returns")  # only level/log-return


class TestLogReturnTarget:
    def _lognormal_df(self, n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        dates = pd.date_range("2024-01-01", periods=n, freq="B").astype(str)
        return pd.DataFrame({"date": dates, "close": closes})

    def test_random_walk_u2_one_in_return_space(self):
        # RW predicts P_t → implied return log(P_t/P_t)=0, which IS the zero-
        # return naive reference, so U2 must be exactly 1.0 in return space too.
        bt = ForecastBacktester(n_days=80, horizon=1, min_train=60, target="log-return")
        run = bt.run(RandomWalkForecaster(), self._lognormal_df(200), "T")
        assert run.metrics.theil_u2 == pytest.approx(1.0)

    def test_level_and_return_modes_both_run(self):
        df = self._lognormal_df(200)
        lvl = ForecastBacktester(n_days=80, target="level", min_train=60).run(
            RandomWalkForecaster(), df, "T"
        )
        ret = ForecastBacktester(n_days=80, target="log-return", min_train=60).run(
            RandomWalkForecaster(), df, "T"
        )
        # Same step count; RW is U2=1.0 in both spaces.
        assert lvl.metrics.n == ret.metrics.n
        assert lvl.metrics.theil_u2 == pytest.approx(1.0)
        assert ret.metrics.theil_u2 == pytest.approx(1.0)
        # Return-space RMSE is on log-returns, so much smaller than level RMSE.
        assert ret.metrics.rmse < lvl.metrics.rmse

    def test_dates_align_to_target(self):
        df = _df(120)
        bt = ForecastBacktester(n_days=30, horizon=1, min_train=60)
        run = bt.run(RandomWalkForecaster(), df, "T")
        # Each step's recorded date is the *target* date (t+h), and y_true is
        # that row's close.
        date_to_close = dict(zip(df["date"], df["close"], strict=True))
        for s in run.steps:
            assert date_to_close[s.date] == pytest.approx(s.y_true)
