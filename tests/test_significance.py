"""test_significance.py — Statistical-significance tests.

Hand-computed expectations for the binomial test, Wilson CI, bootstrap CI,
permutation test, and Benjamini-Hochberg FDR. Pure functions, fast.
"""

from __future__ import annotations

import math

from engine.significance import (
    benjamini_hochberg,
    binomial_test_two_sided,
    bootstrap_ci,
    permutation_test_accuracy,
    significance_for_days,
    wilson_interval,
)

# ----------------------------------------------------------------------
# Binomial test
# ----------------------------------------------------------------------


class TestBinomial:
    def test_exactly_chance_is_one(self):
        # 5/10 at p=0.5 is the most likely outcome → two-sided p = 1.0
        assert binomial_test_two_sided(5, 10) == 1.0

    def test_all_heads_small_n(self):
        # 10/10 at p=0.5: two-sided p = 2 * 0.5^10 = 1/512
        assert math.isclose(binomial_test_two_sided(10, 10), 2 / 1024, rel_tol=1e-9)

    def test_zero_trials(self):
        assert binomial_test_two_sided(0, 0) == 1.0

    def test_symmetry_under_pointfive(self):
        assert math.isclose(
            binomial_test_two_sided(8, 10), binomial_test_two_sided(2, 10), rel_tol=1e-9
        )

    def test_in_unit_interval(self):
        for k in range(0, 21):
            p = binomial_test_two_sided(k, 20)
            assert 0.0 <= p <= 1.0


# ----------------------------------------------------------------------
# Wilson interval
# ----------------------------------------------------------------------


class TestWilson:
    def test_bounds_within_unit_interval(self):
        ci = wilson_interval(50, 50, confidence=0.95)
        assert ci.lo >= 0.0
        assert ci.hi <= 1.0
        assert ci.point == 1.0

    def test_half_is_centered(self):
        ci = wilson_interval(50, 100, confidence=0.95)
        assert math.isclose(ci.point, 0.5, abs_tol=1e-9)
        # symmetric around 0.5
        assert math.isclose(0.5 - ci.lo, ci.hi - 0.5, abs_tol=1e-6)

    def test_known_value(self):
        # 27/40 = 0.675 ; Wilson 95% CI ≈ [0.519, 0.802]
        ci = wilson_interval(27, 40, confidence=0.95)
        assert math.isclose(ci.point, 0.675, abs_tol=1e-6)
        assert math.isclose(ci.lo, 0.519, abs_tol=0.005)
        assert math.isclose(ci.hi, 0.802, abs_tol=0.005)

    def test_zero_trials_is_full_range(self):
        ci = wilson_interval(0, 0)
        assert (ci.lo, ci.hi) == (0.0, 1.0)

    def test_wider_for_higher_confidence(self):
        c95 = wilson_interval(30, 50, 0.95)
        c99 = wilson_interval(30, 50, 0.99)
        assert (c99.hi - c99.lo) > (c95.hi - c95.lo)


# ----------------------------------------------------------------------
# Bootstrap CI
# ----------------------------------------------------------------------


class TestBootstrap:
    def test_constant_series_has_zero_width(self):
        ci = bootstrap_ci([0.01] * 30, "sum", n_boot=500, seed=1)
        # resampling a constant always gives the same sum
        assert math.isclose(ci.lo, ci.hi, abs_tol=1e-9)
        assert math.isclose(ci.point, 0.3, abs_tol=1e-9)

    def test_point_is_actual_statistic(self):
        pnls = [0.01, -0.02, 0.03, -0.01]
        ci = bootstrap_ci(pnls, "sum", n_boot=500, seed=1)
        assert math.isclose(ci.point, sum(pnls), abs_tol=1e-9)

    def test_ci_brackets_point(self):
        pnls = [0.02, -0.01, 0.015, -0.03, 0.04, -0.005] * 5
        ci = bootstrap_ci(pnls, "sum", n_boot=1000, seed=7)
        assert ci.lo <= ci.point <= ci.hi

    def test_reproducible_with_seed(self):
        pnls = [0.01, -0.02, 0.03, -0.01, 0.05]
        a = bootstrap_ci(pnls, "mean", n_boot=500, seed=99)
        b = bootstrap_ci(pnls, "mean", n_boot=500, seed=99)
        assert (a.lo, a.hi) == (b.lo, b.hi)

    def test_empty(self):
        ci = bootstrap_ci([], "sum")
        assert (ci.point, ci.lo, ci.hi) == (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# Permutation test
# ----------------------------------------------------------------------


class TestPermutation:
    def test_perfect_prediction_is_significant(self):
        pred = ["UP", "DOWN"] * 20
        actual = ["UP", "DOWN"] * 20
        # observed accuracy 1.0; very few/no shuffles match → small p
        p = permutation_test_accuracy(pred, actual, n_perm=500, seed=3)
        assert p < 0.1

    def test_anti_correlated_is_not_significant(self):
        pred = ["UP"] * 20 + ["DOWN"] * 20
        actual = ["DOWN"] * 20 + ["UP"] * 20  # always wrong
        p = permutation_test_accuracy(pred, actual, n_perm=500, seed=3)
        assert p > 0.5

    def test_empty(self):
        assert permutation_test_accuracy([], []) == 1.0

    def test_p_in_unit_interval(self):
        pred = ["UP", "DOWN", "UP", "UP", "DOWN"] * 4
        actual = ["UP", "UP", "DOWN", "UP", "DOWN"] * 4
        p = permutation_test_accuracy(pred, actual, n_perm=300, seed=11)
        assert 0.0 < p <= 1.0


# ----------------------------------------------------------------------
# Benjamini-Hochberg
# ----------------------------------------------------------------------


class TestBenjaminiHochberg:
    def test_all_tiny_p_rejected(self):
        assert benjamini_hochberg([0.001, 0.002, 0.003], alpha=0.05) == [True, True, True]

    def test_all_large_p_not_rejected(self):
        assert benjamini_hochberg([0.4, 0.6, 0.9], alpha=0.05) == [False, False, False]

    def test_classic_example(self):
        # Sorted p: 0.005, 0.01, 0.04, 0.5; m=4, alpha=0.05
        # thresholds k/m*alpha: 0.0125, 0.025, 0.0375, 0.05
        # 0.005<=0.0125 ✓, 0.01<=0.025 ✓, 0.04>0.0375 ✗ → but BH rejects up
        #   to the LARGEST k passing: k=2 here. So first two rejected.
        ps = [0.5, 0.04, 0.01, 0.005]
        out = benjamini_hochberg(ps, alpha=0.05)
        # the two smallest (0.005, 0.01) are rejected; aligned to input order
        assert out == [False, False, True, True]

    def test_order_preserved(self):
        ps = [0.001, 0.9, 0.002]
        out = benjamini_hochberg(ps, alpha=0.05)
        assert out[0] is True and out[2] is True and out[1] is False

    def test_empty(self):
        assert benjamini_hochberg([]) == []


# ----------------------------------------------------------------------
# Full bundle
# ----------------------------------------------------------------------


class TestSignificanceForDays:
    def test_bundle_fields(self):
        pred = ["UP", "UP", "DOWN", "UP"]
        actual = ["UP", "DOWN", "DOWN", "UP"]  # 3/4 correct
        pnls = [0.01, -0.02, 0.03, 0.01]
        rep = significance_for_days(pred, actual, pnls, n_boot=300, n_perm=300)
        assert rep.n == 4
        assert rep.correct == 3
        assert math.isclose(rep.accuracy, 0.75, abs_tol=1e-9)
        assert 0.0 <= rep.binomial_p <= 1.0
        assert rep.wilson.lo <= 0.75 <= rep.wilson.hi
        assert math.isclose(rep.return_ci.point, sum(pnls), abs_tol=1e-9)

    def test_empty_days(self):
        rep = significance_for_days([], [], [])
        assert rep.n == 0
        assert rep.binomial_p == 1.0
