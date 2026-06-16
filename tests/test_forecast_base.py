"""
test_forecast_base.py – Offline tests for the forecasting adapter.

Pure numpy: no Prophet/Chronos/torch, no model downloads, no fixtures. Verifies
the value -> direction/confidence derivation, sentiment shifting, the
insufficient-data sentinel and the never-raise guarantee the backtester relies
on. Wire into the main pytest suite in Phase 3.
"""

import numpy as np
import pytest

from engine.forecast_base import ForecastModel, ForecastResult


class _Fake(ForecastModel):
    """Returns a canned ForecastResult so we can test the base in isolation."""

    name = "Fake"

    def __init__(self, fr):
        self._fr = fr

    def _raw_forecast(self, df, horizon=1):
        return self._fr


def _res(**kw):
    base = {"last_close": 100.0, "point": 100.0}
    base.update(kw)
    return ForecastResult(**base)


def test_quantiles_above_close_predict_up_with_high_confidence():
    m = _Fake(_res(point=105, quantiles={0.1: 103, 0.5: 105, 0.9: 107}))
    direction, conf = m.predict(None)
    assert direction == "UP"
    assert conf >= 0.85


def test_quantiles_straddling_close_are_near_coin_flip():
    m = _Fake(_res(point=100, quantiles={0.1: 97, 0.5: 100, 0.9: 103}))
    assert abs(m.forecast(None).prob_up - 0.5) < 0.05


def test_close_above_median_predicts_down():
    m = _Fake(_res(point=99, quantiles={0.1: 95, 0.5: 98, 0.9: 101}))
    fr = m.forecast(None)
    assert fr.direction == "DOWN"
    assert 0.5 < fr.confidence < 1.0


def test_monte_carlo_samples_give_empirical_prob_up():
    samples = np.array([99, 100, 101, 102, 103, 104, 98, 105, 106, 97.0])  # 6/10 > 100
    fr = _Fake(_res(point=101, samples=samples)).forecast(None)
    assert abs(fr.prob_up - 0.6) < 1e-9


def test_point_only_fallback_is_directional_but_modest():
    fr = _Fake(_res(point=102)).forecast(None)  # +2% move, no distribution
    assert fr.direction == "UP"
    assert 0.65 < fr.confidence < 0.72


def test_preset_prob_up_is_respected():
    m = _Fake(_res(point=104, prob_up=0.91, quantiles={0.1: 100, 0.5: 104, 0.9: 108}))
    assert abs(m.forecast(None).confidence - 0.91) < 1e-9


def test_sentiment_raises_confidence_and_can_flip():
    m = _Fake(_res(point=100.5, quantiles={0.1: 99, 0.5: 100.5, 0.9: 102}))
    d0, c0 = m.predict(None, sentiment_score=0.0)
    _, c_pos = m.predict(None, sentiment_score=1.0)
    d_neg, _ = m.predict(None, sentiment_score=-1.0)
    assert d0 == "UP"
    assert c_pos >= c0
    assert d_neg == "DOWN"


def test_insufficient_data_and_errors_never_raise():
    class _Empty(ForecastModel):
        name = "E"

        def _raw_forecast(self, df, horizon=1):
            return None

    class _Boom(ForecastModel):
        name = "B"

        def _raw_forecast(self, df, horizon=1):
            raise ValueError("kaboom")

    assert _Empty().predict(None) == ("Insufficient data", 0.0)
    assert _Boom().predict(None) == ("Insufficient data", 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
