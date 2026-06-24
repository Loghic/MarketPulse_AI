"""test_residual_diagnostics.py — Residual-structure diagnostics (R6).

Locks in the property the paper hinges on: white noise reads as *unstructured*
(Ljung–Box non-significant), genuine autocorrelation reads as *structured*.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.residual_diagnostics import (
    _chi2_sf,
    acf,
    diagnose,
    ljung_box,
    runs_test,
    structure_vs_gain,
    variance_ratio,
)


class TestChi2SF:
    def test_known_critical_values(self):
        # χ² 95th percentiles: 3.841 (dof 1), 18.307 (dof 10) → sf ≈ 0.05.
        assert _chi2_sf(3.841, 1) == pytest.approx(0.05, abs=2e-3)
        assert _chi2_sf(18.307, 10) == pytest.approx(0.05, abs=2e-3)

    def test_zero_and_monotone(self):
        assert _chi2_sf(0.0, 5) == 1.0
        assert _chi2_sf(50.0, 5) < _chi2_sf(5.0, 5)


class TestACF:
    def test_lag0_is_one(self):
        rng = np.random.default_rng(0)
        assert acf(rng.normal(0, 1, 200), nlags=5)[0] == pytest.approx(1.0)

    def test_white_noise_small_acf1(self):
        rng = np.random.default_rng(1)
        assert abs(acf(rng.normal(0, 1, 1000), nlags=5)[1]) < 0.1

    def test_positive_ar1_positive_acf1(self):
        rng = np.random.default_rng(2)
        x = np.zeros(1000)
        for i in range(1, 1000):
            x[i] = 0.7 * x[i - 1] + rng.normal(0, 1)
        assert acf(x, nlags=5)[1] > 0.5


class TestLjungBox:
    def test_white_noise_not_significant(self):
        rng = np.random.default_rng(3)
        lb = ljung_box(rng.normal(0, 1, 500), lags=10)
        assert lb.p_value > 0.05

    def test_ar1_significant(self):
        rng = np.random.default_rng(4)
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = 0.6 * x[i - 1] + rng.normal(0, 1)
        lb = ljung_box(x, lags=10)
        assert lb.p_value < 0.01

    def test_too_short_nan(self):
        lb = ljung_box([1.0, 2.0, 3.0], lags=10)
        assert math.isnan(lb.stat)


class TestRunsAndVR:
    def test_runs_white_noise(self):
        rng = np.random.default_rng(5)
        assert runs_test(rng.normal(0, 1, 500)).p_value > 0.05

    def test_variance_ratio_random_walk_near_one(self):
        rng = np.random.default_rng(6)
        rw = np.cumsum(rng.normal(0, 1, 1000))  # diffs are white → VR ≈ 1
        assert variance_ratio(rw, q=2) == pytest.approx(1.0, abs=0.2)


class TestDiagnose:
    def test_white_noise_unstructured(self):
        rng = np.random.default_rng(7)
        d = diagnose(rng.normal(0, 1, 500), lags=10)
        assert d.structured is False
        assert abs(d.acf1) < 0.15

    def test_ar1_structured(self):
        rng = np.random.default_rng(8)
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = 0.7 * x[i - 1] + rng.normal(0, 1)
        d = diagnose(x, lags=10)
        assert d.structured is True
        assert d.acf1 > 0.5


class TestStructureVsGain:
    def test_cross_tab_pairs_structure_and_gain(self):
        rng = np.random.default_rng(9)
        wn = rng.normal(0, 1, 500)
        ar = np.zeros(500)
        for i in range(1, 500):
            ar[i] = 0.7 * ar[i - 1] + rng.normal(0, 1)
        rows = structure_vs_gain(
            [
                ("WN", wn, 1.05, 1.04),  # unstructured, ~no gain
                ("AR", ar, 1.40, 1.05),  # structured, big gain
            ]
        )
        by = {r.ticker: r for r in rows}
        assert by["WN"].structured is False
        assert by["AR"].structured is True
        assert by["AR"].gain > by["WN"].gain
        assert by["AR"].gain == pytest.approx(0.35)
