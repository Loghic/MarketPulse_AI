"""test_calibration.py — Confidence calibration & gating metrics.

Pins the pure-function calibration metrics against hand-computed values and
verifies the in-engine confidence gate (``Backtester(min_confidence=θ)``)
matches the post-hoc ``gating_metrics`` and sits out the right days.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine.backtester import Backtester, DayResult
from engine.calibration import (
    brier_score,
    expected_calibration_error,
    gating_metrics,
    gating_sweep,
    pairs_from_days,
    reliability_bins,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _day(confidence: float, correct: bool, pnl_net: float = 0.0, traded: bool = True) -> DayResult:
    """Minimal DayResult carrying just the fields calibration reads."""
    return DayResult(
        date="2026-01-01",
        predicted="UP",
        actual="UP" if correct else "DOWN",
        confidence=confidence,
        correct=correct,
        close_before=100.0,
        close_actual=101.0 if correct else 99.0,
        exit_price=101.0 if correct else 99.0,
        trade_pnl=pnl_net,
        trade_pnl_net=pnl_net,
        stopped_out=False,
        traded=traded,
    )


# ----------------------------------------------------------------------
# Reliability bins
# ----------------------------------------------------------------------


class TestReliabilityBins:
    def test_bin_edges_and_counts(self):
        # confidences across the [0.5, 1.0] range, 5 bins of width 0.1
        pairs = [(0.52, True), (0.58, False), (0.63, True), (0.99, True)]
        bins = reliability_bins(pairs, n_bins=5)
        assert len(bins) == 5
        assert (bins[0].lo, bins[0].hi) == (0.5, 0.6)
        # two predictions in first bin, one correct → accuracy 0.5
        assert bins[0].count == 2
        assert bins[0].accuracy == 0.5
        # bin 1 [0.6,0.7): one correct
        assert bins[1].count == 1
        assert bins[1].accuracy == 1.0
        # top value 0.99 folds into the last bin [0.9,1.0]
        assert bins[4].count == 1

    def test_top_edge_folds_into_last_bin(self):
        bins = reliability_bins([(1.0, True)], n_bins=5)
        assert bins[-1].count == 1

    def test_empty_bins_have_zero_count(self):
        bins = reliability_bins([(0.55, True)], n_bins=5)
        assert sum(b.count for b in bins) == 1
        assert all(b.count == 0 for b in bins[1:])

    def test_gap_is_confidence_minus_accuracy(self):
        # one bin, conf 0.8, but wrong → accuracy 0, gap = 0.8
        bins = reliability_bins([(0.8, False)], n_bins=1, lo=0.5, hi=1.0)
        assert bins[0].count == 1
        assert math.isclose(bins[0].gap, 0.8, abs_tol=1e-9)


# ----------------------------------------------------------------------
# Brier score
# ----------------------------------------------------------------------


class TestBrier:
    def test_perfect_confident_correct_is_zero(self):
        assert brier_score([(1.0, True), (1.0, True)]) == 0.0

    def test_confident_but_wrong_is_one(self):
        assert brier_score([(1.0, False)]) == 1.0

    def test_half_confidence_is_quarter(self):
        # (0.5 - 1)^2 = 0.25 ; (0.5 - 0)^2 = 0.25
        assert brier_score([(0.5, True), (0.5, False)]) == 0.25

    def test_empty(self):
        assert brier_score([]) == 0.0

    def test_matches_manual(self):
        # (0.7-1)^2=0.09, (0.6-0)^2=0.36 → mean 0.225
        assert math.isclose(brier_score([(0.7, True), (0.6, False)]), 0.225, abs_tol=1e-6)


# ----------------------------------------------------------------------
# ECE
# ----------------------------------------------------------------------


class TestECE:
    def test_perfectly_calibrated_single_bin(self):
        # 10 preds at conf 0.6, exactly 6 correct → bin accuracy 0.6 = conf
        pairs = [(0.6, True)] * 6 + [(0.6, False)] * 4
        assert expected_calibration_error(pairs, n_bins=5) == 0.0

    def test_miscalibrated(self):
        # all conf 0.9 but only half correct → gap 0.4, single populated bin
        pairs = [(0.9, True)] * 5 + [(0.9, False)] * 5
        assert math.isclose(expected_calibration_error(pairs, n_bins=5), 0.4, abs_tol=1e-6)

    def test_empty(self):
        assert expected_calibration_error([]) == 0.0


# ----------------------------------------------------------------------
# Gating metrics
# ----------------------------------------------------------------------


class TestGatingMetrics:
    def test_threshold_zero_trades_everything(self):
        days = [_day(0.5, True, 0.01), _day(0.9, False, -0.02)]
        g = gating_metrics(days, threshold=0.0, fee_pct=0.05)
        assert g.traded == 2
        assert g.coverage == 1.0
        assert g.fees_saved == 0.0
        assert math.isclose(g.gated_return, -0.01, abs_tol=1e-9)
        assert math.isclose(g.ungated_return, -0.01, abs_tol=1e-9)

    def test_gate_sits_out_low_confidence(self):
        days = [
            _day(0.50, False, -0.03),  # below θ → sat out
            _day(0.70, True, 0.02),  # traded, correct
            _day(0.80, False, -0.01),  # traded, wrong
        ]
        g = gating_metrics(days, threshold=0.65, fee_pct=0.05)
        assert g.traded == 2
        assert g.total == 3
        assert math.isclose(g.coverage, 2 / 3, abs_tol=1e-6)
        # traded accuracy: 1 of 2 correct
        assert g.traded_accuracy == 0.5
        # gated return = only the two traded days
        assert math.isclose(g.gated_return, 0.01, abs_tol=1e-9)
        # one day sat out: round-trip fee saved = 2 * 0.05 / 100 = 0.001
        assert math.isclose(g.fees_saved, 0.001, abs_tol=1e-9)

    def test_sweep_is_monotone_in_coverage(self):
        days = [_day(0.5 + 0.05 * i, i % 2 == 0, 0.0) for i in range(10)]
        sweep = gating_sweep(days, [0.0, 0.6, 0.7, 0.8], fee_pct=0.05)
        covs = [g.coverage for g in sweep]
        # higher threshold never increases coverage
        assert covs == sorted(covs, reverse=True)


# ----------------------------------------------------------------------
# In-engine gate matches post-hoc metrics
# ----------------------------------------------------------------------


class _StepConfModel:
    """Toy model: predicts UP always, with a confidence that cycles by window
    length so that some evaluation days fall below any given gate.

    Confidence cycles through {0.55, 0.68, 0.81, 0.94} keyed on the number of
    rows in the training window — deterministic and reproducible, and it
    guarantees a mix above and below θ=0.7.
    """

    _LEVELS = [0.55, 0.68, 0.81, 0.94]

    def predict(self, df, use_time_weights=False, sentiment_score=0.0):
        conf = self._LEVELS[len(df) % len(self._LEVELS)]
        return "UP", conf


def _ramp_df(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    closes = np.linspace(100.0, 130.0, n)
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1_000_000.0),
        }
    )


class TestEngineGate:
    def test_gate_excludes_low_conf_days_from_accuracy(self):
        df = _ramp_df()
        model = _StepConfModel()

        ungated = Backtester(n_days=10, fee_pct=0.05, min_confidence=0.0).run(
            model, "toy", df, ticker="X"
        )
        gated = Backtester(n_days=10, fee_pct=0.05, min_confidence=0.7).run(
            model, "toy", df, ticker="X"
        )

        # All days recorded in both
        assert len(ungated.days) == len(gated.days)
        # Gated run sits out at least one low-confidence day
        assert gated.sat_out_count >= 1
        assert gated.test_days == sum(1 for d in gated.days if d.traded)
        assert gated.test_days < ungated.test_days
        # Coverage consistent
        total_seen = gated.test_days + gated.sat_out_count
        assert math.isclose(gated.coverage, gated.test_days / total_seen, abs_tol=1e-6)

    def test_engine_return_matches_post_hoc_gating(self):
        df = _ramp_df()
        model = _StepConfModel()
        # Run ungated, then apply the post-hoc gate to its days.
        ungated = Backtester(n_days=10, fee_pct=0.05, min_confidence=0.0).run(
            model, "toy", df, ticker="X"
        )
        gated = Backtester(n_days=10, fee_pct=0.05, min_confidence=0.7).run(
            model, "toy", df, ticker="X"
        )
        post = gating_metrics(ungated.days, threshold=0.7, fee_pct=0.05)
        # Engine total_return (traded days) equals post-hoc gated_return.
        assert math.isclose(gated.total_return, post.gated_return, abs_tol=1e-6)
        assert gated.test_days == post.traded

    def test_sat_out_days_have_zero_pnl(self):
        df = _ramp_df()
        gated = Backtester(n_days=10, fee_pct=0.05, min_confidence=0.7).run(
            _StepConfModel(), "toy", df, ticker="X"
        )
        for d in gated.days:
            if not d.traded:
                assert d.trade_pnl_net == 0.0
                assert d.trade_pnl == 0.0
                assert not d.stopped_out


def test_pairs_from_days_roundtrip():
    days = [_day(0.6, True), _day(0.8, False)]
    assert pairs_from_days(days) == [(0.6, True), (0.8, False)]
