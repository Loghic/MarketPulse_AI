"""test_backtester.py – Walk-forward backtester, fees, stop-loss, risk metrics."""

import pytest

from engine.backtester import Backtester, DayResult


class TestBasicBacktest:
    """Core backtesting functionality."""

    def test_correct_day_count(self, api, full_df):
        bt = Backtester(n_days=5)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert result.test_days == 5
        assert len(result.days) == 5

    def test_accuracy_bounded(self, api, full_df):
        bt = Backtester(n_days=10)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert 0.0 <= result.accuracy <= 1.0
        assert result.correct <= result.test_days

    def test_win_loss_sum(self, api, full_df):
        bt = Backtester(n_days=10)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert result.win_trades + result.loss_trades == result.test_days

    def test_correct_prediction_positive_pnl(self, api, full_df):
        """Correct direction prediction → positive raw P/L."""
        bt = Backtester(n_days=20, fee_pct=0.0)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        for d in result.days:
            if d.correct:
                assert d.trade_pnl >= 0, f"Correct but negative P/L on {d.date}"

    def test_insufficient_data_raises(self, api):
        import pandas as pd

        tiny = pd.DataFrame({"close": range(10), "date": range(10)})
        bt = Backtester(n_days=5)
        with pytest.raises(ValueError, match="Not enough data"):
            bt.run(api.knn, "k-NN", tiny, ticker="TEST")


class TestFees:
    """Trading fee application."""

    def test_fees_reduce_return(self, api, full_df):
        """Same model with fees should have lower return than without."""
        bt_no_fee = Backtester(n_days=20, fee_pct=0.0)
        bt_fee = Backtester(n_days=20, fee_pct=0.1)
        r_no = bt_no_fee.run(api.knn, "k-NN", full_df, ticker="TEST")
        r_fee = bt_fee.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert r_fee.total_return < r_no.total_return

    def test_fee_math(self):
        """Round-trip fee = 2 × per-side fee."""
        raw_pnl = 0.05  # 5% gross return
        fee_pct = 0.1  # 0.1% per side
        net = Backtester._apply_fees(raw_pnl, fee_pct)
        expected = 0.05 - 2 * 0.001  # 5% - 0.2% = 4.8%
        assert abs(net - expected) < 1e-10

    def test_zero_fee_no_change(self):
        raw_pnl = 0.03
        net = Backtester._apply_fees(raw_pnl, 0.0)
        assert net == raw_pnl

    def test_fee_per_day(self, api, full_df):
        """Each day's net P/L should be exactly raw P/L minus round-trip fee."""
        fee = 0.05
        bt = Backtester(n_days=10, fee_pct=fee)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        rt_fee = 2 * fee / 100.0
        for d in result.days:
            expected_net = d.trade_pnl - rt_fee
            assert abs(d.trade_pnl_net - expected_net) < 1e-10


class TestStopLoss:
    """Stop-loss trigger logic."""

    def test_long_stop_loss(self):
        """Long: SL triggers if Low ≤ entry × (1 - SL%)."""
        bt = Backtester(stop_loss_pct=2.0)
        # Entry 100, SL at 98. Day low = 97 → should trigger
        sl = bt._check_stop_loss("UP", 100.0, day_high=103.0, day_low=97.0)
        assert sl is not None
        assert abs(sl - 98.0) < 0.01

    def test_short_stop_loss(self):
        """Short: SL triggers if High ≥ entry × (1 + SL%)."""
        bt = Backtester(stop_loss_pct=2.0)
        sl = bt._check_stop_loss("DOWN", 100.0, day_high=103.0, day_low=97.0)
        assert sl is not None
        assert abs(sl - 102.0) < 0.01

    def test_no_trigger(self):
        """Price stays within SL range → no trigger."""
        bt = Backtester(stop_loss_pct=5.0)
        sl = bt._check_stop_loss("UP", 100.0, day_high=103.0, day_low=97.0)
        assert sl is None  # 97 > 95 (5% SL), no trigger

    def test_disabled_sl(self):
        """SL = 0 → never triggers."""
        bt = Backtester(stop_loss_pct=0.0)
        sl = bt._check_stop_loss("UP", 100.0, day_high=200.0, day_low=50.0)
        assert sl is None

    def test_sl_in_backtest(self, api, full_df):
        """With SL enabled, some days should be stopped out."""
        bt = Backtester(n_days=50, stop_loss_pct=1.0)  # tight SL
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert result.stopped_out_count >= 0
        stopped_days = [d for d in result.days if d.stopped_out]
        assert len(stopped_days) == result.stopped_out_count
        # Stopped out days should use SL price, not close
        for d in stopped_days:
            assert d.exit_price != d.close_actual

    def test_sl_limits_loss(self, api, full_df):
        """SL should cap individual trade losses."""
        bt_sl = Backtester(n_days=50, stop_loss_pct=2.0)
        result = bt_sl.run(api.knn, "k-NN", full_df, ticker="TEST")
        for d in result.days:
            if d.stopped_out:
                # Loss should be approximately SL% (plus fees if any)
                assert d.trade_pnl >= -0.025  # 2% SL + small margin


class TestRiskMetrics:
    """Max drawdown, Sharpe, Sortino."""

    def test_max_drawdown_negative_or_zero(self, api, full_df):
        bt = Backtester(n_days=20)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert result.max_drawdown <= 0.0

    def test_drawdown_all_wins(self):
        """All positive P/L → drawdown should be 0."""
        pnls = [0.01, 0.02, 0.01, 0.03]
        dd = Backtester._compute_max_drawdown(pnls)
        assert dd == 0.0

    def test_drawdown_known_sequence(self):
        """Known sequence: 0% → +5% → +3% → +8%. DD at step 2."""
        pnls = [0.05, -0.02, 0.05]
        dd = Backtester._compute_max_drawdown(pnls)
        assert dd < 0.0
        assert dd > -0.05  # can't be worse than losing 5%

    def test_sharpe_needs_3_samples(self):
        assert Backtester._compute_sharpe([0.01]) == 0.0
        assert Backtester._compute_sharpe([0.01, 0.02]) == 0.0
        # 3+ samples should return a number
        result = Backtester._compute_sharpe([0.01, 0.02, 0.01])
        assert result != 0.0

    def test_sortino_ignores_upside(self):
        """All positive returns → Sortino should be very high (no downside)."""
        result = Backtester._compute_sortino([0.01, 0.02, 0.03, 0.01])
        assert result == 999.0  # no downside

    def test_sortino_needs_downside(self):
        """With negative returns, Sortino should be finite."""
        result = Backtester._compute_sortino([0.01, -0.02, 0.03, -0.01, 0.02])
        assert result != 999.0
        assert isinstance(result, float)

    def test_buy_hold_return(self, api, full_df):
        bt = Backtester(n_days=10)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        # B&H should be (last close - first entry) / first entry
        entry = result.days[0].close_before
        exit_ = result.days[-1].close_actual
        expected = (exit_ - entry) / entry
        assert abs(result.buy_hold_return - expected) < 1e-6

    def test_buy_hold_max_drawdown(self, api, full_df):
        bt = Backtester(n_days=20)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert result.buy_hold_max_drawdown <= 0.0


class TestStreaks:
    """Win/loss streak calculations."""

    def test_all_wins(self):
        days = [
            DayResult("d", "UP", "UP", 0.6, True, 100, 101, 101, 0.01, 0.01, False)
            for _ in range(5)
        ]
        s = Backtester._compute_streaks(days)
        assert s["longest_win_streak"] == 5
        assert s["longest_loss_streak"] == 0

    def test_alternating(self):
        days = []
        for i in range(6):
            win = i % 2 == 0
            pnl = 0.01 if win else -0.01
            days.append(
                DayResult(
                    "d", "UP", "UP" if win else "DOWN", 0.6, win, 100, 101, 101, pnl, pnl, False
                )
            )
        s = Backtester._compute_streaks(days)
        assert s["longest_win_streak"] == 1
        assert s["longest_loss_streak"] == 1

    def test_empty(self):
        s = Backtester._compute_streaks([])
        assert s["longest_win_streak"] == 0


class TestYearlyPerformance:
    """Rolling yearly breakdown."""

    def test_single_year_no_breakdown(self, api, full_df):
        """Short backtest within one year → no yearly breakdown."""
        bt = Backtester(n_days=5)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        # 5 days likely in same year → no yearly breakdown
        # (unless crossing Jan 1, but unlikely with 5 days)
        assert len(result.yearly_performance) <= 1

    def test_multi_year_has_breakdown(self, api, full_df):
        """Long backtest spanning years should have yearly breakdown."""
        bt = Backtester(n_days=300)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        if result.test_days > 250:
            # Should span 2+ years
            assert len(result.yearly_performance) >= 2
            for yp in result.yearly_performance:
                assert yp.trades > 0
                assert 0.0 <= yp.accuracy <= 1.0


class TestProfitMetrics:
    """Profit factor, gross profit/loss."""

    def test_profit_factor_positive(self, api, full_df):
        bt = Backtester(n_days=20)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert result.profit_factor >= 0.0

    def test_gross_profit_loss_sum(self, api, full_df):
        bt = Backtester(n_days=20, fee_pct=0.0)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        # total_return ≈ gross_profit - gross_loss (within rounding)
        expected = result.gross_profit - result.gross_loss
        assert abs(result.total_return - expected) < 1e-6

    def test_best_worst_day(self, api, full_df):
        bt = Backtester(n_days=20)
        result = bt.run(api.knn, "k-NN", full_df, ticker="TEST")
        assert result.best_day >= result.worst_day
        pnls = [d.trade_pnl_net for d in result.days]
        assert abs(result.best_day - max(pnls)) < 1e-8
        assert abs(result.worst_day - min(pnls)) < 1e-8


# ----------------------------------------------------------------------
# Turnover fees + hold-days (strategy experiments 2.1)
# ----------------------------------------------------------------------

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _ramp_df(n: int = 60, seed: int = 3) -> pd.DataFrame:
    """Deterministic OHLCV with mild noise for self-contained engine tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    closes = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
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


class _AlwaysUp:
    """Predicts UP every day at fixed confidence — position never changes."""

    def predict(self, df, use_time_weights=False, sentiment_score=0.0):
        return "UP", 0.8


class _Alternating:
    """Flips UP/DOWN by window length — position changes every day."""

    def predict(self, df, use_time_weights=False, sentiment_score=0.0):
        return ("UP" if len(df) % 2 == 0 else "DOWN"), 0.8


class TestTurnoverFees:
    def test_charge_every_day_is_default(self):
        df = _ramp_df()
        r = Backtester(n_days=20, fee_pct=0.05).run(_AlwaysUp(), "AU", df, ticker="X")
        # Default: a full round-trip fee on every traded day.
        assert r.fees_paid == pytest.approx(20 * 2 * 0.05 / 100, abs=1e-9)
        # AlwaysUp opens once and holds → a single position change.
        assert r.turnover_count == 1

    def test_turnover_fees_charge_only_on_change(self):
        df = _ramp_df()
        base = Backtester(n_days=20, fee_pct=0.05).run(_AlwaysUp(), "AU", df, ticker="X")
        turn = Backtester(n_days=20, fee_pct=0.05, turnover_fees=True).run(
            _AlwaysUp(), "AU", df, ticker="X"
        )
        # AlwaysUp: one open → one round-trip fee total under turnover_fees.
        assert turn.fees_paid == pytest.approx(1 * 2 * 0.05 / 100, abs=1e-9)
        # Fewer fees ⇒ higher net return; raw P&L (pre-fee) identical.
        assert turn.total_return > base.total_return
        assert sum(d.trade_pnl for d in turn.days) == pytest.approx(
            sum(d.trade_pnl for d in base.days), abs=1e-9
        )

    def test_daily_flip_pays_same_as_charge_every_day(self):
        df = _ramp_df()
        base = Backtester(n_days=20, fee_pct=0.05).run(_Alternating(), "ALT", df, ticker="X")
        turn = Backtester(n_days=20, fee_pct=0.05, turnover_fees=True).run(
            _Alternating(), "ALT", df, ticker="X"
        )
        # A position that flips every day changes every day → identical fees.
        assert turn.fees_paid == pytest.approx(base.fees_paid, abs=1e-9)
        assert turn.turnover_count == turn.test_days


class TestHoldDays:
    def test_hold_days_reduces_turnover(self):
        df = _ramp_df()
        hold5 = Backtester(n_days=20, fee_pct=0.05, turnover_fees=True, hold_days=5).run(
            _Alternating(), "ALT", df, ticker="X"
        )
        # Over 20 days with a 5-day hold, at most ~4 position opens.
        assert hold5.turnover_count <= 5
        assert hold5.hold_days == 5

    def test_position_held_through_signal_flips(self):
        df = _ramp_df()
        r = Backtester(n_days=12, fee_pct=0.0, hold_days=4).run(
            _Alternating(), "ALT", df, ticker="X"
        )
        # The held position should persist across consecutive days within a
        # hold window even though _Alternating flips its prediction daily.
        positions = [d.position for d in r.days]
        # At least one run of >=2 identical consecutive positions exists.
        run_len = 1
        max_run = 1
        for a, b in zip(positions, positions[1:], strict=False):
            run_len = run_len + 1 if a == b else 1
            max_run = max(max_run, run_len)
        assert max_run >= 2

    def test_accuracy_tracks_predictions_not_position(self):
        df = _ramp_df()
        # Accuracy must reflect the model's predictions (skill), independent of
        # how long positions are held.
        daily = Backtester(n_days=20, fee_pct=0.0).run(_Alternating(), "ALT", df, ticker="X")
        held = Backtester(n_days=20, fee_pct=0.0, hold_days=5).run(
            _Alternating(), "ALT", df, ticker="X"
        )
        assert daily.accuracy == held.accuracy


class TestTurnoverDefaultsUnchanged:
    def test_default_matches_legacy_behaviour(self):
        df = _ramp_df()
        r = Backtester(n_days=15, fee_pct=0.05).run(_AlwaysUp(), "AU", df, ticker="X")
        # With defaults (no turnover_fees, hold_days=1) every traded day is a
        # full round-trip, exactly as before this feature.
        assert not r.turnover_fees
        assert r.hold_days == 1
        rt = 2 * 0.05 / 100
        for d in r.days:
            assert d.trade_pnl_net == pytest.approx(d.trade_pnl - rt, abs=1e-12)


# ----------------------------------------------------------------------
# Position mode — compound a held run into one trade, fee once
# ----------------------------------------------------------------------


def _step_df(closes: list[float]) -> pd.DataFrame:
    """OHLCV from an explicit close path (open=high=low=close, no SL noise)."""
    n = len(closes)
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    c = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"date": dates, "open": c, "high": c, "low": c, "close": c, "volume": np.full(n, 1e6)}
    )


class TestPositionMode:
    def test_compound_hold_books_one_trade_one_fee(self):
        # 25 flat warm-up rows + the demo: hold UP across 100→110→90→95→97.
        # AllUp holds one position across the whole 5-day eval window.
        closes = [100.0] * 25 + [100, 110, 90, 95, 97]
        df = _step_df(closes)
        r = Backtester(n_days=5, fee_pct=0.05, position_mode=True).run(
            _AlwaysUp(), "AU", df, ticker="X"
        )
        # Entry = close_before of the first eval day; exit = last day's close.
        entry = r.days[0].close_before
        exit_ = r.days[-1].close_actual
        expected_raw = (exit_ - entry) / entry  # long, so unsigned
        rt = 2 * 0.05 / 100
        # The whole run lands on the last day; interior days are zeroed.
        assert r.days[-1].trade_pnl == pytest.approx(expected_raw, abs=1e-9)
        assert r.days[-1].trade_pnl_net == pytest.approx(expected_raw - rt, abs=1e-9)
        for d in r.days[:-1]:
            assert d.trade_pnl == 0.0 and d.trade_pnl_net == 0.0
        # Exactly ONE round-trip fee for the whole held run.
        assert r.fees_paid == pytest.approx(rt, abs=1e-9)
        # Total return = the compounded entry→exit move minus one fee.
        assert r.total_return == pytest.approx(expected_raw - rt, abs=1e-9)

    def test_position_mode_vs_daily_differ_on_holds(self):
        closes = [100.0] * 25 + [100, 110, 90, 95, 97]
        df = _step_df(closes)
        daily = Backtester(n_days=5, fee_pct=0.05).run(_AlwaysUp(), "AU", df, ticker="X")
        pos = Backtester(n_days=5, fee_pct=0.05, position_mode=True).run(
            _AlwaysUp(), "AU", df, ticker="X"
        )
        # Daily mode pays 5 round trips; position mode pays 1.
        assert daily.fees_paid == pytest.approx(5 * 2 * 0.05 / 100, abs=1e-9)
        assert pos.fees_paid == pytest.approx(1 * 2 * 0.05 / 100, abs=1e-9)
        assert pos.position_mode is True
        assert daily.position_mode is False

    def test_flip_breaks_the_run(self):
        # _Alternating flips direction every day → every day is its own run,
        # so position mode pays a fee per day, like daily mode.
        df = _ramp_df(n=60)
        pos = Backtester(n_days=20, fee_pct=0.05, position_mode=True).run(
            _Alternating(), "ALT", df, ticker="X"
        )
        # One run per day → one fee per traded day.
        rt = 2 * 0.05 / 100
        assert pos.fees_paid == pytest.approx(pos.test_days * rt, abs=1e-9)

    def test_accuracy_unchanged_by_position_mode(self):
        df = _ramp_df(n=60)
        daily = Backtester(n_days=20, fee_pct=0.0).run(_Alternating(), "ALT", df, ticker="X")
        pos = Backtester(n_days=20, fee_pct=0.0, position_mode=True).run(
            _Alternating(), "ALT", df, ticker="X"
        )
        # Position mode only changes P&L accounting, never the predictions.
        assert daily.accuracy == pos.accuracy


# ----------------------------------------------------------------------
# FLAT (no-trade) prediction path
# ----------------------------------------------------------------------


class _FlatEveryOther:
    """Predicts UP, but returns FLAT (sit out) on every other window length —
    a deterministic stand-in for a model that abstains some days."""

    def predict(self, df, use_time_weights=False, sentiment_score=0.0):
        return ("FLAT" if len(df) % 2 == 0 else "UP"), 0.8


class TestFlatPrediction:
    def test_flat_days_are_untraded(self):
        df = _ramp_df(n=60)
        r = Backtester(n_days=10, fee_pct=0.05).run(_FlatEveryOther(), "F", df, ticker="X")
        flat = [d for d in r.days if d.predicted == "FLAT"]
        assert flat, "expected some FLAT days"
        for d in flat:
            assert d.traded is False
            assert d.trade_pnl_net == 0.0
            assert d.trade_pnl == 0.0
        # All recorded (not skipped): traded + flat == every eval day.
        assert len(r.days) == 10

    def test_flat_excluded_from_accuracy_and_counts(self):
        df = _ramp_df(n=60)
        r = Backtester(n_days=10, fee_pct=0.05).run(_FlatEveryOther(), "F", df, ticker="X")
        traded = [d for d in r.days if d.traded]
        # test_days counts only traded (directional) days.
        assert r.test_days == len(traded)
        assert r.test_days < len(r.days)
        # Coverage reflects the sit-out days.
        assert r.sat_out_count == len(r.days) - r.test_days
        assert r.coverage == pytest.approx(r.test_days / len(r.days), abs=1e-6)
        # No FLAT day pays a fee.
        assert r.fees_paid == pytest.approx(r.test_days * 2 * 0.05 / 100, abs=1e-9)

    def test_garbage_prediction_still_skipped(self):
        # A non-UP/DOWN/FLAT/HOLD prediction is dropped entirely (not recorded).
        class _Garbage:
            def predict(self, df, use_time_weights=False, sentiment_score=0.0):
                return "SIDEWAYS", 0.8

        df = _ramp_df(n=60)
        r = Backtester(n_days=10).run(_Garbage(), "G", df, ticker="X")
        assert len(r.days) == 0
        assert r.test_days == 0


# ----------------------------------------------------------------------
# HOLD (buy-and-hold) prediction path — always a single trade, one fee
# ----------------------------------------------------------------------


class _HoldEveryDay:
    def predict(self, df, use_time_weights=False, sentiment_score=0.0):
        return "HOLD", 1.0


class TestHoldMode:
    def test_hold_is_single_buyhold_one_fee_flag_independent(self):
        df = _ramp_df(n=60)
        # No position-mode flag: HOLD must STILL collapse to one buy-hold trade.
        r = Backtester(n_days=20, fee_pct=0.05).run(_HoldEveryDay(), "H", df, ticker="X")
        rt = 2 * 0.05 / 100
        entry = r.days[0].close_before
        exit_ = r.days[-1].close_actual
        expected = (exit_ - entry) / entry  # long buy-hold
        # Exactly one round-trip fee (not 20), booked once.
        assert r.fees_paid == pytest.approx(rt, abs=1e-9)
        assert r.total_return == pytest.approx(expected - rt, abs=1e-6)
        # Matches the B&H benchmark column minus the single entry fee.
        assert r.total_return == pytest.approx(r.buy_hold_return - rt, abs=1e-6)

    def test_hold_excluded_from_accuracy(self):
        df = _ramp_df(n=60)
        r = Backtester(n_days=20, fee_pct=0.05).run(_HoldEveryDay(), "H", df, ticker="X")
        # HOLD makes no UP/DOWN call → accuracy is 0/0 → reported 0.0, correct 0.
        assert r.accuracy == 0.0
        assert r.correct == 0

    def test_hold_ignores_stop_loss(self):
        # A tight stop must not chop a pure buy-hold — it rides through.
        df = _ramp_df(n=60, seed=9)
        r = Backtester(n_days=20, fee_pct=0.0, stop_loss_pct=1.0).run(
            _HoldEveryDay(), "H", df, ticker="X"
        )
        assert r.stopped_out_count == 0


# ----------------------------------------------------------------------
# Stop-loss sweep via run_single_backtest (strategy experiments 2.2)
# ----------------------------------------------------------------------


class _StubAPIBaselines:
    """Minimal StockAppAPI for run_single_backtest's baseline-only path."""

    lstm_available = False
    knn = knn_enhanced = linreg = linreg_enhanced = None

    def __init__(self, df):
        self._df = df

    def get_data(self, ticker, period="max"):  # noqa: ARG002
        return self._df.copy()

    @property
    def db(self):
        class _DB:
            def get_news(self, ticker):  # noqa: ARG002
                return pd.DataFrame()

        return _DB()

    def _process_news_with_db(self, ticker, method=None):  # noqa: ARG002
        return 0.0, []

    def forecast_available(self, mt):  # noqa: ARG002
        return False


class TestStopLossSweep:
    def _run(self, sl_levels):
        from engine.backtest_helpers import run_single_backtest

        df = _ramp_df(n=120)
        api = _StubAPIBaselines(df)
        bt = Backtester(n_days=20, fee_pct=0.05)
        return run_single_backtest(
            api,
            bt,
            "X",
            df,
            "max",
            20,
            full=False,
            models=["baseline"],
            include_baselines=True,
            sl_levels=sl_levels,
        )

    def test_sweep_runs_each_level_per_model(self):
        results = self._run([0, 5, 10])
        # Each baseline model appears once per level. The 0 level keeps the
        # bare name; non-zero levels get an SL suffix.
        names = [r.model_name for r in results]
        base_names = {n for n in names if "SL" not in n}
        assert base_names, "expected baseline (no-SL) rows"
        for bn in base_names:
            assert f"{bn} SL5%" in names
            assert f"{bn} SL10%" in names
        # Each level carries the right stop_loss_pct.
        by_level = {}
        for r in results:
            by_level.setdefault(r.stop_loss_pct, 0)
            by_level[r.stop_loss_pct] += 1
        assert set(by_level) == {0.0, 5.0, 10.0}

    def test_sweep_dedupes_and_sorts(self):
        results = self._run([10, 5, 5, 0])
        levels = sorted({r.stop_loss_pct for r in results})
        assert levels == [0.0, 5.0, 10.0]

    def test_levels_without_zero_have_no_baseline_row(self):
        results = self._run([5, 10])
        assert all(r.stop_loss_pct in (5.0, 10.0) for r in results)
        assert all("SL" in r.model_name for r in results)
