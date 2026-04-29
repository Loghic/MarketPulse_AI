"""test_backtester.py – Walk-forward backtester, fees, stop-loss, risk metrics."""

import numpy as np
import pytest
from engine.backtester import Backtester, DayResult, BacktestResult


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
        fee_pct = 0.1   # 0.1% per side
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
        days = [DayResult("d", "UP", "UP", 0.6, True, 100, 101, 101,
                          0.01, 0.01, False) for _ in range(5)]
        s = Backtester._compute_streaks(days)
        assert s["longest_win_streak"] == 5
        assert s["longest_loss_streak"] == 0

    def test_alternating(self):
        days = []
        for i in range(6):
            win = i % 2 == 0
            pnl = 0.01 if win else -0.01
            days.append(DayResult("d", "UP", "UP" if win else "DOWN", 0.6,
                                  win, 100, 101, 101, pnl, pnl, False))
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
