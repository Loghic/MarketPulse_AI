"""test_forecast_significance.py — Diebold–Mariano + Wilcoxon forecast tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.forecast_significance import (
    _student_t_sf,
    compare_to_reference,
    dm_test,
    wilcoxon_loss_test,
)


class TestStudentT:
    def test_known_values(self):
        # Survival fn at 0 is 0.5; at t=2,dof=10 ≈ 0.0367 (textbook).
        assert _student_t_sf(0.0, 10) == pytest.approx(0.5, abs=1e-6)
        assert _student_t_sf(2.0, 10) == pytest.approx(0.0367, abs=2e-3)

    def test_large_dof_approaches_normal(self):
        # With many dof, t ≈ normal: P(T>1.96) ≈ 0.025.
        assert _student_t_sf(1.96, 100000) == pytest.approx(0.025, abs=2e-3)


class TestDM:
    def _errs(self, seed=0):
        rng = np.random.default_rng(seed)
        e1 = rng.normal(0, 0.5, 200)
        e2 = rng.normal(0, 2.0, 200)
        return e1, e2

    def test_better_model_negative_significant(self):
        e1, e2 = self._errs()
        r = dm_test(e1, e2)  # e1 smaller errors
        assert r.stat < 0  # model_1 better
        assert r.p_value < 0.05
        assert r.mean_diff < 0

    def test_symmetry(self):
        e1, e2 = self._errs()
        assert dm_test(e1, e2).stat == pytest.approx(-dm_test(e2, e1).stat)

    def test_identical_forecasts_zero_stat(self):
        e = self._errs()[0]
        r = dm_test(e, e)
        assert r.stat == 0.0
        assert r.p_value == 1.0

    def test_near_identical_not_significant(self):
        rng = np.random.default_rng(3)
        base = rng.normal(0, 1.0, 120)
        r = dm_test(base + rng.normal(0, 0.01, 120), base)
        assert r.p_value > 0.05  # the 0.998-vs-1.000 case: not real

    def test_horizon_changes_stat(self):
        # h>1 uses (h-1) Newey-West lags + the HLN small-sample correction, so a
        # multi-step DM differs from the 1-step one on the same data (we don't
        # assert a direction — the NW variance can move either way with the data,
        # only that the horizon actually feeds through).
        e1, e2 = self._errs()
        assert dm_test(e1, e2, horizon=1).stat != dm_test(e1, e2, horizon=5).stat
        # Both still agree on the sign (model_1 is better regardless of horizon).
        assert dm_test(e1, e2, horizon=1).stat < 0
        assert dm_test(e1, e2, horizon=5).stat < 0

    def test_absolute_loss_option(self):
        e1, e2 = self._errs()
        r = dm_test(e1, e2, loss="absolute")
        assert r.loss == "absolute"
        assert r.stat < 0

    def test_bad_loss_raises(self):
        with pytest.raises(ValueError):
            dm_test([0.1, 0.2], [0.3, 0.4], loss="huber")

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            dm_test([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_too_few_points_nan(self):
        r = dm_test([0.1], [0.2])
        assert math.isnan(r.stat)


class TestWilcoxon:
    def test_clear_difference_small_p(self):
        rng = np.random.default_rng(4)
        e1 = rng.normal(0, 0.4, 100)
        e2 = rng.normal(0, 2.0, 100)
        assert wilcoxon_loss_test(e1, e2).p_value < 0.05

    def test_identical_returns_p_one(self):
        e = np.array([0.1, -0.2, 0.3, 0.4, -0.1])
        # All differences zero → dropped → n=0 → p=1.0 by convention.
        assert wilcoxon_loss_test(e, e).p_value == 1.0


class TestGridFDR:
    def test_only_genuine_winner_flagged(self):
        rng = np.random.default_rng(5)
        ref = rng.normal(0, 1.0, 150)
        cases = [
            ("better", rng.normal(0, 0.4, 150), ref),
            ("tied", ref + rng.normal(0, 0.01, 150), ref),
            ("worse", rng.normal(0, 3.0, 150), ref),
        ]
        rows = {r.label: r for r in compare_to_reference(cases)}
        assert rows["better"].dm_significant is True
        assert rows["tied"].dm_significant is False
        # "worse" differs significantly but loses — must NOT be flagged a winner.
        assert rows["worse"].dm_significant is False
        assert rows["worse"].mean_diff > 0

    def test_empty_grid(self):
        assert compare_to_reference([]) == []
