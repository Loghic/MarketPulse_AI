"""
backtester.py – Walk-forward backtest engine.

Evaluates model accuracy by hiding the last N days of data, then
predicting each day one at a time and comparing against the actual outcome.

Tracks simulated trading P/L with configurable trading fees.
Includes buy-and-hold benchmark for comparison.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from config import DEFAULT_TRADING_FEE_PCT


@dataclass
class DayResult:
    """Result of a single day's prediction vs reality."""
    date: str
    predicted: str          # "UP" or "DOWN"
    actual: str             # "UP" or "DOWN"
    confidence: float
    correct: bool
    close_before: float
    close_actual: float
    trade_pnl: float        # raw return (before fees)
    trade_pnl_net: float    # net return (after fees)


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    model_name: str
    ticker: str
    test_days: int
    correct: int
    accuracy: float
    fee_pct: float = 0.0
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
    # Streak metrics
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    # Buy-and-hold benchmark
    buy_hold_return: float = 0.0
    # Day-by-day results
    days: List[DayResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        pf_str = f"{self.profit_factor:.2f}" if self.profit_factor < 100 else "∞"
        lines = [
            f"  Model: {self.model_name}",
            f"  Accuracy: {self.correct}/{self.test_days} ({self.accuracy:.1%})",
            f"  Total return: {self.total_return:+.4%}  |  "
            f"Profit factor: {pf_str}  |  "
            f"Buy&Hold: {self.buy_hold_return:+.4%}  |  "
            f"Streaks: W{self.longest_win_streak}/L{self.longest_loss_streak}",
            "",
        ]
        for d in self.days:
            mark = "✓" if d.correct else "✗"
            lines.append(
                f"    {d.date}  pred={d.predicted:<5} actual={d.actual:<5} "
                f"conf={d.confidence:.1%}  pnl={d.trade_pnl_net:+.4%}  {mark}"
            )
        return "\n".join(lines)


class Backtester:
    """Walk-forward backtester with trading fee support."""

    def __init__(self, n_days: int = 5, fee_pct: float = DEFAULT_TRADING_FEE_PCT):
        self.n_days = n_days
        self.fee_pct = fee_pct

    @staticmethod
    def _compute_trade_pnl(predicted: str, close_before: float,
                           close_actual: float) -> float:
        """Raw return from following the prediction (before fees)."""
        actual_return = (close_actual - close_before) / close_before
        return actual_return if predicted == "UP" else -actual_return

    @staticmethod
    def _apply_fees(raw_pnl: float, fee_pct: float) -> float:
        """
        Deduct round-trip trading fees from a trade's P/L.
        fee_pct is per side, so round-trip = 2 × fee_pct.
        """
        round_trip_fee = 2 * fee_pct / 100.0
        return raw_pnl - round_trip_fee

    @staticmethod
    def _compute_buy_hold(day_results: List[DayResult]) -> float:
        """
        Compute buy-and-hold return over the test period.
        Buy at close on the day before the first test day,
        sell at close on the last test day.
        """
        if not day_results:
            return 0.0
        entry_price = day_results[0].close_before
        exit_price = day_results[-1].close_actual
        if entry_price == 0:
            return 0.0
        return (exit_price - entry_price) / entry_price

    @staticmethod
    def _compute_streaks(day_results: List[DayResult]) -> dict:
        """Compute win/loss streak statistics from daily net P/L."""
        if not day_results:
            return {
                "longest_win_streak": 0, "longest_loss_streak": 0,
                "avg_win_streak": 0.0, "avg_loss_streak": 0.0,
            }

        win_streaks, loss_streaks = [], []
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
    def _compute_profit_metrics(day_results: List[DayResult]) -> dict:
        """Compute aggregate profit metrics from net daily P/L."""
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

    def run(self, model, model_name: str, df: pd.DataFrame,
            ticker: str = "", use_time_weights: bool = False,
            sentiment_score: float = 0.0) -> BacktestResult:
        """
        Run walk-forward backtest on a single model.

        Each test day: train on data before that day, predict, compare,
        compute P/L net of fees.
        """
        if len(df) < self.n_days + 20:
            raise ValueError(
                f"Not enough data: need {self.n_days + 20} rows, got {len(df)}"
            )

        day_results = []

        for i in range(self.n_days, 0, -1):
            train_df = df.iloc[:-i].copy()
            eval_idx = len(df) - i

            close_before = float(train_df["close"].iloc[-1])
            close_actual = float(df["close"].iloc[eval_idx])
            actual_direction = "UP" if close_actual > close_before else "DOWN"

            predicted, confidence = model.predict(
                train_df, use_time_weights=use_time_weights,
                sentiment_score=sentiment_score,
            )

            if predicted not in ("UP", "DOWN"):
                continue

            raw_pnl = self._compute_trade_pnl(predicted, close_before, close_actual)
            net_pnl = self._apply_fees(raw_pnl, self.fee_pct)

            day_results.append(DayResult(
                date=str(df["date"].iloc[eval_idx]),
                predicted=predicted,
                actual=actual_direction,
                confidence=confidence,
                correct=(predicted == actual_direction),
                close_before=close_before,
                close_actual=close_actual,
                trade_pnl=raw_pnl,
                trade_pnl_net=net_pnl,
            ))

        correct_count = sum(1 for d in day_results if d.correct)
        total = len(day_results)
        metrics = self._compute_profit_metrics(day_results)
        streaks = self._compute_streaks(day_results)
        buy_hold = round(self._compute_buy_hold(day_results), 8)

        return BacktestResult(
            model_name=model_name,
            ticker=ticker,
            test_days=total,
            correct=correct_count,
            accuracy=round(correct_count / total, 4) if total > 0 else 0.0,
            fee_pct=self.fee_pct,
            buy_hold_return=buy_hold,
            days=day_results,
            **metrics,
            **streaks,
        )
