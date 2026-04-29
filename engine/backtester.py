"""
backtester.py – Walk-forward backtest engine.

Metrics: accuracy, P/L (net of fees), profit factor, streaks,
max drawdown, Sharpe ratio, Sortino ratio, buy-and-hold benchmark,
and yearly rolling performance breakdown.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import DEFAULT_STOP_LOSS_PCT, DEFAULT_TRADING_FEE_PCT


@dataclass
class DayResult:
    """Result of a single day's prediction vs reality."""

    date: str
    predicted: str
    actual: str
    confidence: float
    correct: bool
    close_before: float
    close_actual: float
    exit_price: float
    trade_pnl: float
    trade_pnl_net: float
    stopped_out: bool


@dataclass
class YearlyPerformance:
    """Performance metrics for a single calendar year."""

    year: str
    trades: int
    correct: int
    accuracy: float
    total_return: float
    profit_factor: float
    max_drawdown: float
    win_trades: int
    loss_trades: int


@dataclass
class BacktestResult:
    """Aggregated backtest results."""

    model_name: str
    ticker: str
    test_days: int
    correct: int
    accuracy: float
    fee_pct: float = 0.0
    stop_loss_pct: float = 0.0
    # Profit metrics (net of fees)
    total_return: float = 0.0
    profit_factor: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    win_trades: int = 0
    loss_trades: int = 0
    # Stop-loss stats
    stopped_out_count: int = 0
    # Risk metrics
    max_drawdown: float = 0.0  # maximum peak-to-trough decline
    sharpe_ratio: float = 0.0  # annualized risk-adjusted return
    sortino_ratio: float = 0.0  # like Sharpe but only penalizes downside
    # Streak metrics
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    # Buy-and-hold benchmark
    buy_hold_return: float = 0.0
    buy_hold_max_drawdown: float = 0.0
    # Rolling performance
    yearly_performance: list[YearlyPerformance] = field(default_factory=list)
    # Day-by-day results
    days: list[DayResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        pf_str = f"{self.profit_factor:.2f}" if self.profit_factor < 100 else "∞"
        sl_str = (
            f"  SL: {self.stopped_out_count}/{self.test_days}" if self.stop_loss_pct > 0 else ""
        )
        lines = [
            f"  Model: {self.model_name}",
            f"  Accuracy: {self.correct}/{self.test_days} ({self.accuracy:.1%})",
            f"  Return: {self.total_return:+.4%}  |  PF: {pf_str}  |  "
            f"DD: {self.max_drawdown:+.4%}  |  "
            f"Sharpe: {self.sharpe_ratio:.2f}  |  Sortino: {self.sortino_ratio:.2f}",
            f"  B&H: {self.buy_hold_return:+.4%}  |  "
            f"Streaks: W{self.longest_win_streak}/L{self.longest_loss_streak}"
            f"{sl_str}",
            "",
        ]
        for d in self.days:
            mark = "✓" if d.correct else "✗"
            sl = " SL" if d.stopped_out else ""
            lines.append(
                f"    {d.date}  pred={d.predicted:<5} actual={d.actual:<5} "
                f"conf={d.confidence:.1%}  pnl={d.trade_pnl_net:+.4%}  {mark}{sl}"
            )
        return "\n".join(lines)


class Backtester:
    """Walk-forward backtester with fees, stop-loss, and risk metrics."""

    def __init__(
        self,
        n_days: int = 5,
        fee_pct: float = DEFAULT_TRADING_FEE_PCT,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    ):
        self.n_days = n_days
        self.fee_pct = fee_pct
        self.stop_loss_pct = stop_loss_pct

    # ------------------------------------------------------------------
    # Trade P/L
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_trade_pnl(predicted: str, entry_price: float, exit_price: float) -> float:
        ret = (exit_price - entry_price) / entry_price
        return ret if predicted == "UP" else -ret

    @staticmethod
    def _apply_fees(raw_pnl: float, fee_pct: float) -> float:
        return raw_pnl - 2 * fee_pct / 100.0

    def _check_stop_loss(
        self, predicted: str, entry_price: float, day_high: float, day_low: float
    ) -> float | None:
        if self.stop_loss_pct <= 0:
            return None
        sl_frac = self.stop_loss_pct / 100.0
        if predicted == "UP":
            sl_price = entry_price * (1 - sl_frac)
            if day_low <= sl_price:
                return sl_price
        else:
            sl_price = entry_price * (1 + sl_frac)
            if day_high >= sl_price:
                return sl_price
        return None

    # ------------------------------------------------------------------
    # Risk metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_max_drawdown(pnls: list[float]) -> float:
        """
        Maximum peak-to-trough decline of the cumulative equity curve.

        E.g. equity goes 0% → +5% → +3% → +8%
        Drawdown at step 3 = (3% - 5%) / (1 + 5%) = -1.9%
        Max DD is the worst such decline.
        """
        if not pnls:
            return 0.0

        cumulative = np.cumsum(pnls)
        equity = 1.0 + cumulative  # equity curve starting at 1.0

        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak  # always ≤ 0

        return round(float(np.min(drawdown)), 8) if len(drawdown) > 0 else 0.0

    @staticmethod
    def _compute_sharpe(pnls: list[float], risk_free_daily: float = 0.0) -> float:
        """
        Annualized Sharpe ratio.

        Sharpe = (mean_daily_return - risk_free) / std_daily_return × √252

        Risk-free rate default 0 (common for short backtests).
        Returns 0.0 if not enough data or zero variance.
        """
        if len(pnls) < 3:
            return 0.0
        arr = np.array(pnls)
        excess = arr - risk_free_daily
        std = np.std(excess, ddof=1)
        if std < 1e-12:
            return 0.0
        return round(float(np.mean(excess) / std * np.sqrt(252)), 4)

    @staticmethod
    def _compute_sortino(pnls: list[float], risk_free_daily: float = 0.0) -> float:
        """
        Annualized Sortino ratio.

        Like Sharpe but uses only downside deviation (std of negative returns).
        Better for strategies with asymmetric returns.
        """
        if len(pnls) < 3:
            return 0.0
        arr = np.array(pnls)
        excess = arr - risk_free_daily
        downside = excess[excess < 0]
        if len(downside) < 2:
            return 999.0 if np.mean(excess) > 0 else 0.0
        down_std = np.std(downside, ddof=1)
        if down_std < 1e-12:
            return 0.0
        return round(float(np.mean(excess) / down_std * np.sqrt(252)), 4)

    @staticmethod
    def _compute_buy_hold_drawdown(day_results: list[DayResult]) -> float:
        """Max drawdown of buy-and-hold over the test period."""
        if not day_results:
            return 0.0
        entry = day_results[0].close_before
        if entry == 0:
            return 0.0
        prices = [entry] + [d.close_actual for d in day_results]
        equity = np.array(prices) / entry
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        return round(float(np.min(dd)), 8)

    # ------------------------------------------------------------------
    # Yearly rolling performance
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_yearly_performance(day_results: list[DayResult]) -> list[YearlyPerformance]:
        """Break down results by calendar year."""
        if not day_results:
            return []

        # Group by year
        by_year: dict[str, list[DayResult]] = {}
        for d in day_results:
            year = d.date[:4]  # "2026-04-15" → "2026"
            by_year.setdefault(year, []).append(d)

        # Only produce yearly breakdown if data spans multiple years
        if len(by_year) <= 1:
            return []

        yearly = []
        for year in sorted(by_year.keys()):
            days = by_year[year]
            pnls = [d.trade_pnl_net for d in days]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            gross_p = sum(wins)
            gross_l = abs(sum(losses))

            if gross_l == 0:
                pf = 999.0 if gross_p > 0 else 0.0
            else:
                pf = gross_p / gross_l

            # Max drawdown for this year
            cumulative = np.cumsum(pnls)
            equity = 1.0 + cumulative
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

            correct = sum(1 for d in days if d.correct)

            yearly.append(
                YearlyPerformance(
                    year=year,
                    trades=len(days),
                    correct=correct,
                    accuracy=round(correct / len(days), 4) if days else 0.0,
                    total_return=round(sum(pnls), 8),
                    profit_factor=round(pf, 4),
                    max_drawdown=round(max_dd, 8),
                    win_trades=len(wins),
                    loss_trades=len(losses),
                )
            )

        return yearly

    # ------------------------------------------------------------------
    # Streaks + profit metrics (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_streaks(day_results: list[DayResult]) -> dict:
        if not day_results:
            return {
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
                "avg_win_streak": 0.0,
                "avg_loss_streak": 0.0,
            }
        win_streaks: list[int] = []
        loss_streaks: list[int] = []
        current_streak = 0
        current_type = None
        for d in day_results:
            is_win = d.trade_pnl_net > 0
            if current_type is None:
                current_type = is_win
                current_streak = 1
            elif is_win == current_type:
                current_streak += 1
            else:
                (win_streaks if current_type else loss_streaks).append(current_streak)
                current_type = is_win
                current_streak = 1
        if current_type is not None:
            (win_streaks if current_type else loss_streaks).append(current_streak)
        return {
            "longest_win_streak": max(win_streaks) if win_streaks else 0,
            "longest_loss_streak": max(loss_streaks) if loss_streaks else 0,
            "avg_win_streak": round(float(np.mean(win_streaks)), 1) if win_streaks else 0.0,
            "avg_loss_streak": round(float(np.mean(loss_streaks)), 1) if loss_streaks else 0.0,
        }

    @staticmethod
    def _compute_profit_metrics(day_results: list[DayResult]) -> dict:
        if not day_results:
            return {}
        pnls = [d.trade_pnl_net for d in day_results]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        if gross_loss == 0:
            profit_factor = 999.0 if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss
        return {
            "total_return": round(sum(pnls), 8),
            "profit_factor": round(profit_factor, 4),
            "gross_profit": round(gross_profit, 8),
            "gross_loss": round(gross_loss, 8),
            "avg_win": round(float(np.mean(wins)), 8) if wins else 0.0,
            "avg_loss": round(float(np.mean(losses)), 8) if losses else 0.0,
            "best_day": round(max(pnls), 8) if pnls else 0.0,
            "worst_day": round(min(pnls), 8) if pnls else 0.0,
            "win_trades": len(wins),
            "loss_trades": len(losses),
        }

    @staticmethod
    def _compute_buy_hold(day_results: list[DayResult]) -> float:
        if not day_results:
            return 0.0
        entry = day_results[0].close_before
        exit_ = day_results[-1].close_actual
        return (exit_ - entry) / entry if entry != 0 else 0.0

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        model: Any,
        model_name: str,
        df: pd.DataFrame,
        ticker: str = "",
        use_time_weights: bool = False,
        sentiment_score: float = 0.0,
    ) -> BacktestResult:
        """Run walk-forward backtest with all metrics."""
        if len(df) < self.n_days + 20:
            raise ValueError(f"Not enough data: need {self.n_days + 20} rows, got {len(df)}")

        has_ohlc = "high" in df.columns and "low" in df.columns
        day_results = []

        for i in range(self.n_days, 0, -1):
            train_df = df.iloc[:-i].copy()
            eval_idx = len(df) - i

            entry_price = float(train_df["close"].iloc[-1])
            close_actual = float(df["close"].iloc[eval_idx])
            actual_direction = "UP" if close_actual > entry_price else "DOWN"

            predicted, confidence = model.predict(
                train_df,
                use_time_weights=use_time_weights,
                sentiment_score=sentiment_score,
            )

            if predicted not in ("UP", "DOWN"):
                continue

            stopped_out = False
            exit_price = close_actual

            if self.stop_loss_pct > 0 and has_ohlc:
                day_high = float(df["high"].iloc[eval_idx])
                day_low = float(df["low"].iloc[eval_idx])
                sl_exit = self._check_stop_loss(predicted, entry_price, day_high, day_low)
                if sl_exit is not None:
                    exit_price = sl_exit
                    stopped_out = True

            raw_pnl = self._compute_trade_pnl(predicted, entry_price, exit_price)
            net_pnl = self._apply_fees(raw_pnl, self.fee_pct)

            day_results.append(
                DayResult(
                    date=str(df["date"].iloc[eval_idx]),
                    predicted=predicted,
                    actual=actual_direction,
                    confidence=confidence,
                    correct=(predicted == actual_direction),
                    close_before=entry_price,
                    close_actual=close_actual,
                    exit_price=exit_price,
                    trade_pnl=raw_pnl,
                    trade_pnl_net=net_pnl,
                    stopped_out=stopped_out,
                )
            )

        correct_count = sum(1 for d in day_results if d.correct)
        stopped_count = sum(1 for d in day_results if d.stopped_out)
        total = len(day_results)

        metrics = self._compute_profit_metrics(day_results)
        streaks = self._compute_streaks(day_results)
        buy_hold = round(self._compute_buy_hold(day_results), 8)
        buy_hold_dd = self._compute_buy_hold_drawdown(day_results)

        # Risk metrics from net P/L series
        pnls = [d.trade_pnl_net for d in day_results]
        max_dd = self._compute_max_drawdown(pnls)
        sharpe = self._compute_sharpe(pnls)
        sortino = self._compute_sortino(pnls)

        # Yearly breakdown
        yearly = self._compute_yearly_performance(day_results)

        return BacktestResult(
            model_name=model_name,
            ticker=ticker,
            test_days=total,
            correct=correct_count,
            accuracy=round(correct_count / total, 4) if total > 0 else 0.0,
            fee_pct=self.fee_pct,
            stop_loss_pct=self.stop_loss_pct,
            buy_hold_return=buy_hold,
            buy_hold_max_drawdown=buy_hold_dd,
            stopped_out_count=stopped_count,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            yearly_performance=yearly,
            days=day_results,
            **metrics,
            **streaks,
        )
