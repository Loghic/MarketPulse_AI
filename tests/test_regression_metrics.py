"""test_regression_metrics.py — Point-forecast metrics.

Pins the pure regression metrics against hand-computed values and locks in the
two invariants the whole regression track relies on:

* the random-walk forecast scores **Theil U2 == 1.0** by construction, and
* a perfect forecast scores zero error / U2 == 0.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.regression_metrics import (
    compute_all,
    mae,
    mape,
    mase,
    rmse,
    rmsse,
    smape,
    theil_u2,
)


class TestAbsolute:
    def test_rmse_mae_handcomputed(self):
        yt = [10.0, 12.0, 14.0]
        yp = [11.0, 11.0, 17.0]  # errors: -1, +1, -3
        assert mae(yt, yp) == pytest.approx((1 + 1 + 3) / 3)
        assert rmse(yt, yp) == pytest.approx(math.sqrt((1 + 1 + 9) / 3))

    def test_mape(self):
        yt = [100.0, 200.0]
        yp = [110.0, 180.0]  # 10% and 10%
        assert mape(yt, yp) == pytest.approx(0.10)

    def test_smape_bounded(self):
        # smape lies in [0, 2]; identical -> 0.
        assert smape([100, 200], [100, 200]) == pytest.approx(0.0)
        v = smape([100, 200], [120, 160])
        assert 0.0 <= v <= 2.0

    def test_perfect_forecast_zero_error(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert rmse(y, y) == 0.0
        assert mae(y, y) == 0.0
        assert mape(y, y) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rmse([1, 2, 3], [1, 2])

    def test_nonfinite_rows_dropped(self):
        yt = [1.0, 2.0, np.nan, 4.0]
        yp = [1.0, 2.0, 3.0, np.inf]
        # Only the first two rows are finite on both sides -> perfect.
        assert mae(yt, yp) == pytest.approx(0.0)


class TestScaleFree:
    def test_random_walk_is_u2_reference(self):
        """The naive forecast scored against itself must be exactly 1.0."""
        yt = np.array([10.0, 11.0, 9.0, 12.0])
        y_naive = np.array([9.5, 10.5, 10.0, 10.5])
        assert theil_u2(yt, y_naive, y_naive) == pytest.approx(1.0)

    def test_u2_below_one_when_better_than_naive(self):
        yt = np.array([10.0, 11.0, 12.0])
        y_naive = np.array([9.0, 10.0, 11.0])  # always off by ~1
        y_good = np.array([10.1, 10.9, 12.0])  # much closer
        assert theil_u2(yt, y_good, y_naive) < 1.0

    def test_u2_above_one_when_worse(self):
        yt = np.array([10.0, 11.0, 12.0])
        y_naive = np.array([10.0, 11.0, 12.0])  # perfect naive (rare, but defines ratio)
        y_bad = np.array([8.0, 13.0, 9.0])
        # naive rmse is 0 here -> U2 undefined -> NaN (guard, not inf).
        assert math.isnan(theil_u2(yt, y_bad, y_naive))

    def test_mase_handcomputed(self):
        # in-sample one-step diffs: |1|,|1|,|1| -> naive MAE = 1.
        insample = [10.0, 11.0, 12.0, 13.0]
        yt = [14.0, 15.0]
        yp = [14.5, 14.5]  # abs errors 0.5, 0.5 -> test MAE 0.5
        assert mase(yt, yp, insample) == pytest.approx(0.5)

    def test_rmsse_handcomputed(self):
        insample = [10.0, 11.0, 12.0, 13.0]  # naive MSE = 1
        yt = [14.0, 16.0]
        yp = [14.0, 14.0]  # sq errors 0, 4 -> test MSE 2 -> rmsse sqrt(2)
        assert rmsse(yt, yp, insample) == pytest.approx(math.sqrt(2.0))

    def test_degenerate_insample_returns_nan(self):
        const = [5.0, 5.0, 5.0, 5.0]  # naive MAE = 0 -> MASE undefined
        assert math.isnan(mase([5, 6], [5, 5], const))


class TestComputeAll:
    def test_bundle_random_walk_self(self):
        rng = np.random.default_rng(0)
        train = 100 + np.cumsum(rng.normal(0, 1, 150))
        full = np.concatenate([train, train[-1] + np.cumsum(rng.normal(0, 1, 30))])
        yt = full[len(train) :]
        y_naive = full[len(train) - 1 : len(full) - 1]  # RW = previous value
        m = compute_all(yt, y_naive, insample=train, y_naive=y_naive, season=1)
        assert m.n == 30
        assert m.theil_u2 == pytest.approx(1.0)  # RW vs RW
        assert m.mase == pytest.approx(1.0, abs=0.25)  # ~1 by construction

    def test_bundle_perfect(self):
        train = np.arange(1.0, 101.0)
        yt = np.arange(101.0, 121.0)
        y_naive = np.arange(100.0, 120.0)
        m = compute_all(yt, yt, insample=train, y_naive=y_naive, season=1)
        assert m.rmse == 0.0
        assert m.theil_u2 == 0.0
        assert m.mase == 0.0
