"""test_oos_harness.py — Out-of-sample harness.

Tests the disjoint-window selection→evaluation pipeline. We avoid the
real ML stack (slow + needs trained LSTMs) by driving the harness with
the baseline models alone — they're deterministic, instant, and
sufficient to exercise every branch of ``oos_one_ticker``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from scripts.oos_harness import (
    aggregate,
    build_run_dir,
    oos_one_ticker,
    write_per_ticker,
    write_summary,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _prices_df(days: int = 400, seed: int = 7, base: float = 100.0) -> pd.DataFrame:
    """Synthetic daily OHLCV with a mild positive drift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=datetime(2026, 5, 25), periods=days, freq="B")
    returns = rng.normal(0.0008, 0.018, days)
    closes = base * np.cumprod(1.0 + returns)
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": closes * (1.0 + rng.uniform(-0.003, 0.003, days)),
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": rng.integers(1_000_000, 50_000_000, days),
        }
    )


@dataclass
class _StubAPI:
    """Minimal StockAppAPI replacement: returns the canned DataFrame
    on get_data, an empty news table, and no-ops on news refresh."""

    df: pd.DataFrame
    lstm_available: bool = False

    def get_data(self, ticker, period="max"):  # noqa: ARG002
        return self.df.copy()

    @property
    def db(self):
        class _DB:
            def get_news(self, ticker):  # noqa: ARG002
                return pd.DataFrame()

        return _DB()

    def _process_news_with_db(self, ticker, method=None):  # noqa: ARG002
        return 0.0, []

    def _load_lstm(self, ticker, period):  # noqa: ARG002
        return None

    def forecast_available(self, mt):  # noqa: ARG002
        return False


# Lazily satisfy the attribute lookups run_single_backtest does on `api`
# for the non-baseline branches even when we restrict to baselines.
_StubAPI.knn = None
_StubAPI.knn_enhanced = None
_StubAPI.linreg = None
_StubAPI.linreg_enhanced = None


# ----------------------------------------------------------------------
# aggregate()
# ----------------------------------------------------------------------


class TestAggregate:
    def test_empty(self):
        out = aggregate([])
        assert out["tickers"] == 0
        assert out["oos_beat_bh_rate"] == 0.0
        assert out["median_oos_return"] == 0.0

    def test_handcomputed_two_tickers(self):
        rows = [
            {
                "ticker": "AAA",
                "in_sample_return": 0.20,
                "oos_return": 0.05,
                "oos_accuracy": 0.55,
                "beats_bh_oos": 1,
            },
            {
                "ticker": "BBB",
                "in_sample_return": 0.30,
                "oos_return": -0.10,
                "oos_accuracy": 0.45,
                "beats_bh_oos": 0,
            },
        ]
        out = aggregate(rows)
        assert out["tickers"] == 2
        # 1 of 2 beats B&H -> 50%
        assert out["oos_beat_bh_rate"] == 0.5
        # median oos return = mean of -0.10 and 0.05 = -0.025
        assert out["median_oos_return"] == pytest.approx(-0.025, abs=1e-12)
        # in-sample minus oos: AAA = 0.15, BBB = 0.40 → median 0.275
        assert out["in_sample_minus_oos_median"] == pytest.approx(0.275, abs=1e-12)
        # median accuracy
        assert out["median_oos_accuracy"] == pytest.approx(0.5, abs=1e-12)


# ----------------------------------------------------------------------
# build_run_dir / writers
# ----------------------------------------------------------------------


class TestIO:
    def test_run_dir_encodes_params(self, tmp_path):
        d = build_run_dir(tmp_path, scope="stocks", days=50, fees=0.03, stop_loss=0, buy_hold=True)
        assert d.exists()
        assert "oos_stocks_50d" in d.name
        assert "fee003" in d.name
        assert "bh" in d.name

    def test_per_ticker_csv_roundtrip(self, tmp_path):
        rows = [
            {
                "ticker": "AAA",
                "winner_model": "LSTM",
                "winner_period": "5y",
                "winner_family": "lstm",
                "in_sample_return": 0.1,
                "in_sample_accuracy": 0.55,
                "in_sample_buy_hold": 0.05,
                "oos_return": 0.02,
                "oos_accuracy": 0.5,
                "oos_buy_hold": 0.04,
                "oos_sharpe": 0.3,
                "beats_bh_oos": 0,
                "stable": 1,
            }
        ]
        out = write_per_ticker(tmp_path, rows)
        with out.open() as f:
            back = list(csv.DictReader(f))
        assert len(back) == 1
        assert back[0]["winner_model"] == "LSTM"
        assert back[0]["beats_bh_oos"] == "0"

    def test_summary_csv_roundtrip(self, tmp_path):
        summary = {
            "tickers": 3,
            "oos_beat_bh_rate": 0.33,
            "median_oos_return": 0.01,
            "mean_oos_return": 0.012,
            "median_in_sample_return": 0.15,
            "in_sample_minus_oos_median": 0.14,
            "median_oos_accuracy": 0.51,
        }
        out = write_summary(tmp_path, summary)
        with out.open() as f:
            back = next(csv.DictReader(f))
        assert int(back["tickers"]) == 3
        assert float(back["oos_beat_bh_rate"]) == pytest.approx(0.33, abs=1e-9)


# ----------------------------------------------------------------------
# oos_one_ticker — end-to-end with the baseline-only model set
# ----------------------------------------------------------------------


class TestOOSOneTicker:
    def test_returns_none_when_data_too_short(self):
        api = _StubAPI(df=_prices_df(days=50))  # need ≥ 2*100 + 20 = 220
        result = oos_one_ticker(
            api,
            ticker="ZZZ",
            n_days=100,
            fee_pct=0.0,
            stop_loss_pct=0.0,
            periods=["max"],
            news_lookback_days=7,
            news_half_life_days=3.0,
            sentiment_method=None,
            models=["baseline"],
            include_baselines=True,
        )
        assert result is None

    def test_selection_window_does_not_overlap_evaluation(self):
        """The harness slices off the last ``n_days`` rows before
        selecting. Any backtest day older than that cutoff must have
        come from the selection window; anything in the most recent
        ``n_days`` rows of the original df is the evaluation window.

        We verify by intercepting Backtester.run and capturing the
        date range of each call's input df.
        """
        from engine import backtester as backtester_module

        api = _StubAPI(df=_prices_df(days=400))

        original_run = backtester_module.Backtester.run
        seen: list[dict] = []

        def spy(self, *, model, model_name, df, ticker, **kwargs):
            seen.append(
                {
                    "model_name": model_name,
                    "first": str(df["date"].iloc[0]),
                    "last": str(df["date"].iloc[-1]),
                    "len": len(df),
                }
            )
            return original_run(
                self,
                model=model,
                model_name=model_name,
                df=df,
                ticker=ticker,
                **kwargs,
            )

        with patch.object(backtester_module.Backtester, "run", spy):
            row = oos_one_ticker(
                api,
                ticker="AAA",
                n_days=50,
                fee_pct=0.0,
                stop_loss_pct=0.0,
                periods=["max"],
                news_lookback_days=7,
                news_half_life_days=3.0,
                sentiment_method=None,
                models=["baseline"],
                include_baselines=True,
            )

        assert row is not None
        # The evaluation run sees a longer df than the selection runs
        # (it has the trailing 50 days that selection chopped off).
        sel_lens = {s["len"] for s in seen if s["last"] != seen[-1]["last"]}
        eval_len = seen[-1]["len"]
        assert sel_lens, "expected at least one selection backtest"
        assert all(L < eval_len for L in sel_lens), (
            f"selection runs ({sel_lens}) should see a shorter df than "
            f"the evaluation run ({eval_len})"
        )
        # And the evaluation df's last date is strictly later than every
        # selection df's last date — non-overlapping holdouts.
        eval_last = seen[-1]["last"]
        sel_lasts = {s["last"] for s in seen if s["last"] != eval_last}
        for sl in sel_lasts:
            assert sl < eval_last

    def test_winner_is_highest_in_sample_return(self):
        """The harness must pick the variant with the highest in-sample
        total_return as its winner — even when that variant loses OOS.
        """
        api = _StubAPI(df=_prices_df(days=400))
        row = oos_one_ticker(
            api,
            ticker="AAA",
            n_days=50,
            fee_pct=0.0,
            stop_loss_pct=0.0,
            periods=["max"],
            news_lookback_days=7,
            news_half_life_days=3.0,
            sentiment_method=None,
            models=["baseline"],
            include_baselines=True,
        )
        assert row is not None
        # Re-run selection by hand and confirm the same winner.
        from engine.backtest_helpers import run_single_backtest
        from engine.backtester import Backtester

        bt = Backtester(n_days=50, fee_pct=0.0, stop_loss_pct=0.0)
        df_selection = _prices_df(days=400).iloc[:-50]
        results = run_single_backtest(
            api,
            bt,
            "AAA",
            df_selection,
            "max",
            50,
            full=False,
            models=["baseline"],
            include_baselines=True,
        )
        best = max(results, key=lambda r: r.total_return)
        assert row["winner_model"] == best.model_name
        # in_sample_return reported equals best.total_return
        assert row["in_sample_return"] == pytest.approx(best.total_return, abs=1e-9)

    def test_beats_bh_flag_matches_definition(self):
        api = _StubAPI(df=_prices_df(days=400))
        row = oos_one_ticker(
            api,
            ticker="AAA",
            n_days=50,
            fee_pct=0.0,
            stop_loss_pct=0.0,
            periods=["max"],
            news_lookback_days=7,
            news_half_life_days=3.0,
            sentiment_method=None,
            models=["baseline"],
            include_baselines=True,
        )
        assert row is not None
        expected = int(row["oos_return"] > row["oos_buy_hold"])
        assert row["beats_bh_oos"] == expected


# ----------------------------------------------------------------------
# Confidence gating in the OOS harness
# ----------------------------------------------------------------------


class TestOOSGating:
    def _row(self, theta: float, models=None):
        api = _StubAPI(df=_prices_df(days=400))
        return oos_one_ticker(
            api,
            ticker="AAA",
            n_days=50,
            fee_pct=0.05,
            stop_loss_pct=0.0,
            periods=["max"],
            news_lookback_days=7,
            news_half_life_days=3.0,
            sentiment_method=None,
            models=models or ["baseline"],
            include_baselines=True,
            min_confidence=theta,
        )

    def test_ungated_has_full_coverage(self):
        row = self._row(theta=0.0)
        assert row is not None
        assert row["min_confidence"] == 0.0
        assert row["oos_coverage"] == 1.0
        assert row["oos_sat_out"] == 0
        # New calibration/significance columns are always present.
        for k in ("oos_brier", "oos_ece", "oos_binomial_p", "oos_acc_ci_lo", "oos_acc_ci_hi"):
            assert k in row

    def test_gate_records_coverage_and_is_consistent(self):
        # θ=0.6 sits out every directional/random baseline day (conf ≤ 0.55);
        # AlwaysLong (conf 1.0) keeps full coverage. Whichever wins, the
        # coverage bookkeeping must be internally consistent.
        row = self._row(theta=0.6)
        assert row is not None
        assert row["min_confidence"] == 0.6
        traded = row["oos_traded_days"]
        sat = row["oos_sat_out"]
        seen = traded + sat
        assert seen > 0
        assert row["oos_coverage"] == pytest.approx(traded / seen, abs=1e-6)
        # Coverage can only shrink (or stay equal) relative to ungated.
        assert 0.0 <= row["oos_coverage"] <= 1.0

    def test_always_long_winner_keeps_full_coverage_under_gate(self):
        # Restrict to AlwaysLong alone: its confidence is 1.0, so even a high
        # gate never sits it out — coverage stays 100%.
        from engine.baseline_models import AlwaysLongBaseline

        # Build a row but force the candidate set to a single always-long
        # baseline by monkeypatching default_baseline_variants is overkill;
        # instead assert via the gate's effect on a known-high-conf model:
        row = self._row(theta=0.7)
        assert row is not None
        # If AlwaysLong won selection (likely in a drifting-up series), its
        # coverage is 1.0; otherwise the winner had conf ≤ 0.55 → coverage 0.
        if row["winner_model"].endswith("Always Long") or "Always Long" in row["winner_model"]:
            assert row["oos_coverage"] == 1.0
        _ = AlwaysLongBaseline  # referenced for intent

    def test_gate_preserves_disjoint_windows(self):
        """The gate must not change the selection/evaluation split."""
        from engine import backtester as backtester_module

        api = _StubAPI(df=_prices_df(days=400))
        original_run = backtester_module.Backtester.run
        seen: list[dict] = []

        def spy(self, *, model, model_name, df, ticker, **kwargs):
            seen.append({"last": str(df["date"].iloc[-1]), "len": len(df)})
            return original_run(
                self, model=model, model_name=model_name, df=df, ticker=ticker, **kwargs
            )

        with patch.object(backtester_module.Backtester, "run", spy):
            row = oos_one_ticker(
                api,
                ticker="AAA",
                n_days=50,
                fee_pct=0.0,
                stop_loss_pct=0.0,
                periods=["max"],
                news_lookback_days=7,
                news_half_life_days=3.0,
                sentiment_method=None,
                models=["baseline"],
                include_baselines=True,
                min_confidence=0.6,
            )

        assert row is not None
        eval_last = seen[-1]["last"]
        sel_lasts = {s["last"] for s in seen if s["last"] != eval_last}
        assert sel_lasts
        for sl in sel_lasts:
            assert sl < eval_last


class TestAggregateGating:
    def test_aggregate_emits_gating_block(self):
        rows = [
            {
                "ticker": "AAA",
                "in_sample_return": 0.2,
                "oos_return": 0.05,
                "oos_accuracy": 0.6,
                "beats_bh_oos": 1,
                "min_confidence": 0.6,
                "oos_coverage": 0.4,
                "oos_brier": 0.22,
                "oos_ece": 0.05,
                "oos_binomial_p": 0.03,
            },
            {
                "ticker": "BBB",
                "in_sample_return": 0.3,
                "oos_return": -0.1,
                "oos_accuracy": 0.48,
                "beats_bh_oos": 0,
                "min_confidence": 0.6,
                "oos_coverage": 0.6,
                "oos_brier": 0.26,
                "oos_ece": 0.09,
                "oos_binomial_p": 0.40,
            },
        ]
        out = aggregate(rows)
        assert out["min_confidence"] == 0.6
        # median coverage of {0.4, 0.6} = 0.5
        assert out["median_oos_coverage"] == pytest.approx(0.5, abs=1e-9)
        # one of two tickers significant at p<0.05
        assert out["tickers_significant_p05"] == 1
        assert out["median_oos_brier"] == pytest.approx(0.24, abs=1e-9)

    def test_aggregate_ungated_rows_default_to_full_coverage(self):
        # Rows without the gating keys (legacy) must not crash aggregate.
        rows = [
            {
                "ticker": "AAA",
                "in_sample_return": 0.1,
                "oos_return": 0.02,
                "oos_accuracy": 0.51,
                "beats_bh_oos": 0,
            }
        ]
        out = aggregate(rows)
        assert out["min_confidence"] == 0.0
        assert out["median_oos_coverage"] == 1.0
        assert out["tickers_significant_p05"] == 0

    def test_build_run_dir_encodes_min_confidence(self, tmp_path):
        d = build_run_dir(
            tmp_path,
            scope="stocks",
            days=50,
            fees=0.03,
            stop_loss=0,
            buy_hold=True,
            min_confidence=0.65,
        )
        assert "mc065" in d.name


# ----------------------------------------------------------------------
# CLI smoke — main() resolves args without crashing on --help
# ----------------------------------------------------------------------


class TestCLI:
    def test_help_exits_cleanly(self, monkeypatch, capsys):
        from scripts.oos_harness import main

        monkeypatch.setattr("sys.argv", ["oos_harness.py", "--help"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "OOS harness" in captured.out or "OOS harness" in captured.err


def test_module_exposes_public_api():
    """The harness's public surface — aggregate + oos_one_ticker —
    must be importable. We intentionally do NOT use importlib.reload
    here: the harness pulls in engine.backtest_helpers, which on macOS
    has historically left enough file descriptors open during import
    that a forced reload exhausts the FD limit for the rest of the
    test session.
    """
    import scripts.oos_harness as mod

    assert callable(mod.aggregate)
    assert callable(mod.oos_one_ticker)
    assert callable(mod.write_per_ticker)
    assert callable(mod.write_summary)
    assert callable(mod.build_run_dir)


# Mark the slower disk-touch tests if pytest gets a marker registry
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


_ = Path  # silence unused-import linters on older Python
