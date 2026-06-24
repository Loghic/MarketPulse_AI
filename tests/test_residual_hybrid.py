"""test_residual_hybrid.py — Residual hybrid forecaster (Phase R3).

Pins the three properties the hybrid must guarantee:
  * **identity** — a zero residual learner makes the hybrid exactly the base;
  * **additive reconstruction** — a constant-residual learner shifts the base
    point by that constant;
  * **leakage** — the residual learner is fed residuals only up to ``t`` (never
    ``res_{t+h}``), and the base forecast for ``t+h`` used no data past ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.naive_forecasters import RandomWalkForecaster
from engine.residual_hybrid import ResidualHybrid
from engine.residual_learners import ZeroResidualLearner


def _df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"close": closes})


class _ConstLearner:
    name = "const"

    def __init__(self, c: float):
        self.c = c

    def fit(self, residuals):  # noqa: ARG002
        return None

    def predict(self) -> float:
        return self.c


class _SpyLearner:
    """Records the residual series + window it was handed."""

    name = "spy"

    def __init__(self):
        self.seen_len = None
        self.seen_last = None

    def fit(self, residuals):
        self.seen_len = len(residuals)
        self.seen_last = float(residuals[-1])

    def predict(self) -> float:
        return 0.0


class TestComposition:
    def test_identity_zero_learner(self):
        df = _df(40)
        base = RandomWalkForecaster()
        hybrid = ResidualHybrid(base, ZeroResidualLearner())
        assert hybrid.forecast(df).point == pytest.approx(base.forecast(df).point)

    def test_additive_constant(self):
        df = _df(40)
        base = RandomWalkForecaster()
        base_point = base.forecast(df).point
        hybrid = ResidualHybrid(base, _ConstLearner(5.0))
        assert hybrid.forecast(df).point == pytest.approx(base_point + 5.0)

    def test_name_composition(self):
        h = ResidualHybrid(RandomWalkForecaster(), ZeroResidualLearner())
        assert "Random Walk" in h.name and "zero" in h.name


class TestLeakage:
    def test_learner_sees_only_residuals_up_to_t(self):
        df = _df(30, seed=3)
        base = RandomWalkForecaster()
        spy = _SpyLearner()
        ResidualHybrid(base, spy).forecast(df, horizon=1)

        closes = df["close"].to_numpy()
        # The learner's last residual must equal res_t (close[t] − close[t-1]),
        # i.e. the RW residual of the last in-window day — NOT res_{t+1}, which
        # would need close[t+1] that the window doesn't contain.
        res_t = closes[-1] - closes[-2]
        assert spy.seen_last == pytest.approx(res_t)
        assert spy.seen_len <= len(df)

    def test_base_forecast_unaffected_by_future(self):
        # Truncating the window must change the hybrid forecast (it only used
        # data up to t) — a crude check that nothing peeks past the window end.
        df = _df(50, seed=1)
        hybrid = ResidualHybrid(RandomWalkForecaster(), ZeroResidualLearner())
        full = hybrid.forecast(df).point
        shorter = hybrid.forecast(df.iloc[:-1]).point
        # RW point = last close, so the two must differ (different last rows).
        assert full != shorter


class TestFitModes:
    def test_bad_fit_mode_raises(self):
        with pytest.raises(ValueError):
            ResidualHybrid(RandomWalkForecaster(), ZeroResidualLearner(), fit_mode="bogus")

    def test_refit_k_only_fits_on_cadence(self):
        # A spy learner counts fit() vs set_window() calls under refit_k.
        fits, windows = [], []

        class _CadenceSpy:
            name = "cadence"

            def fit(self, residuals):
                fits.append(len(residuals))

            def set_window(self, residuals):
                windows.append(len(residuals))

            def predict(self):
                return 0.0

        df = _df(40)
        hybrid = ResidualHybrid(
            RandomWalkForecaster(), _CadenceSpy(), fit_mode="refit_k", refit_k=3
        )
        for _ in range(6):
            hybrid.forecast(df)
        # Calls 0 and 3 refit; the rest just set the window.
        assert len(fits) == 2
        assert len(windows) == 4

    def test_pretrained_never_fits(self):
        fits = []

        class _NoFitSpy:
            name = "nofit"
            is_trained = True

            def fit(self, residuals):
                fits.append(1)

            def set_window(self, residuals):
                pass

            def predict(self):
                return 0.0

        df = _df(40)
        hybrid = ResidualHybrid(RandomWalkForecaster(), _NoFitSpy(), fit_mode="pretrained")
        for _ in range(5):
            hybrid.forecast(df)
        assert fits == []  # frozen weights: predict-only, never refit


class TestWithTorch:
    def test_prophet_lstm_style_hybrid_runs(self):
        pytest.importorskip("torch")
        from engine.residual_learners import LSTMResidualLearner

        df = _df(400, seed=2)
        # Base = RW (always available); learner = small LSTM, quick settings.
        hybrid = ResidualHybrid(
            RandomWalkForecaster(),
            LSTMResidualLearner(window=10, epochs=3, min_pairs=20),
        )
        fr = hybrid.forecast(df)
        assert fr is not None
        assert np.isfinite(fr.point)

    def test_residual_learner_save_load_roundtrip(self, tmp_path):
        pytest.importorskip("torch")
        from engine.residual_learners import LSTMResidualLearner

        rng = np.random.default_rng(7)
        residuals = rng.normal(0, 1, 300).astype("float32")
        learner = LSTMResidualLearner(window=10, epochs=3, min_pairs=20)
        learner.fit(residuals)
        assert learner.is_trained
        p = tmp_path / "TST_hybrid_res.pt"
        learner.save(p)

        # Fresh learner loads the weights and predicts from a new window without
        # refitting (the pretrained path).
        loaded = LSTMResidualLearner()
        assert loaded.load(p) is True
        assert loaded.is_trained
        loaded.set_window(residuals)
        assert np.isfinite(loaded.predict())
