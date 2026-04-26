"""
backtester.py – Walk-forward backtest engine.

Evaluates model accuracy by hiding the last N days of data, then
predicting each day one at a time and comparing against the actual outcome.

Tracks simulated trading P/L: if the model says UP we go Long,
if it says DOWN we go Short. This answers the question "would following
this model actually make money?"

Works with any model that exposes the standard .predict(df, ...) interface
(KNNModel, LinearRegressionModel, and any future model).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class DayResult:
    """Result of a single day's prediction vs reality."""
    date: str
    predicted: str          # "UP" or "DOWN"
    actual: str             # "UP" or "DOWN"
    confidence: float
    correct: bool
    close_before: float     # closing price the model saw as "last"
    close_actual: float     # actual closing price of the predicted day
    trade_pnl: float        # simulated return: + if trade was profitable, - if not


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    model_name: str
    ticker: str
    test_days: int
    correct: int
    accuracy: float
    # Profit metrics
    total_return: float = 0.0       # sum of all trade P/L (as percentage)
    profit_factor: float = 0.0      # gross_profit / gross_loss
    gross_profit: float = 0.0       # sum of winning trades
    gross_loss: float = 0.0         # sum of losing trades (positive number)
    avg_win: float = 0.0            # average winning trade
    avg_loss: float = 0.0           # average losing trade
    best_day: float = 0.0           # largest single gain
    worst_day: float = 0.0          # largest single loss
    win_trades: int = 0
    loss_trades: int = 0
    # Streak metrics
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    avg_win_streak: float = 0.0
    avg_loss_streak: float = 0.0
    days: List[DayResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        pf_str = f"{self.profit_factor:.2f}" if self.profit_factor < 100 else "∞"
        lines = [
            f"  Model: {self.model_name}",
            f"  Accuracy: {self.correct}/{self.test_days} ({self.accuracy:.1%})",
            f"  Total return: {self.total_return:+.2%}  |  "
            f"Profit factor: {pf_str}  |  "
            f"Streaks: W{self.longest_win_streak}/L{self.longest_loss_streak}",
            "",
        ]
        for d in self.days:
            mark = "✓" if d.correct else "✗"
            lines.append(
                f"    {d.date}  pred={d.predicted:<5} actual={d.actual:<5} "
                f"conf={d.confidence:.1%}  pnl={d.trade_pnl:+.2%}  {mark}"
            )
        return "\n".join(lines)


class Backtester:
    """
    Walk-forward backtester.

    For each of the last `n_days` trading days:
        1. Slice data up to (but not including) that day.
        2. Ask the model to predict direction.
        3. Compare with the actual price movement.
        4. Compute simulated trade P/L.
    """

    def __init__(self, n_days: int = 5):
        self.n_days = n_days

    @staticmethod
    def _compute_trade_pnl(predicted: str, close_before: float, close_actual: float) -> float:
        """
        Compute the return from following the prediction.

        If predicted UP:   we go Long  → pnl = (actual - before) / before
        If predicted DOWN: we go Short → pnl = (before - actual) / before

        A correct prediction always produces positive P/L.
        """
        actual_return = (close_actual - close_before) / close_before
        if predicted == "UP":
            return actual_return
        else:  # DOWN → short
            return -actual_return

    @staticmethod
    def _compute_streaks(day_results: List[DayResult]) -> dict:
        """
        Compute win/loss streak statistics from daily results.

        A streak is a consecutive run of wins or losses.
        E.g. [W, W, W, L, L, W, W] → win streaks: [3, 2], loss streaks: [2]
        """
        if not day_results:
            return {
                "longest_win_streak": 0,
                "longest_loss_streak": 0,
                "avg_win_streak": 0.0,
                "avg_loss_streak": 0.0,
            }

        win_streaks = []
        loss_streaks = []
        current_streak = 0
        current_type = None  # True = win, False = loss

        for d in day_results:
            is_win = d.trade_pnl > 0

            if current_type is None:
                # First day
                current_type = is_win
                current_streak = 1
            elif is_win == current_type:
                # Same streak continues
                current_streak += 1
            else:
                # Streak broken — save the old one
                if current_type:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
                current_type = is_win
                current_streak = 1

        # Don't forget the last streak
        if current_type is not None:
            if current_type:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)

        return {
            "longest_win_streak": max(win_streaks) if win_streaks else 0,
            "longest_loss_streak": max(loss_streaks) if loss_streaks else 0,
            "avg_win_streak": float(np.mean(win_streaks)) if win_streaks else 0.0,
            "avg_loss_streak": float(np.mean(loss_streaks)) if loss_streaks else 0.0,
        }

    @staticmethod
    def _compute_profit_metrics(day_results: List[DayResult]) -> dict:
        """Compute aggregate profit metrics from daily results."""
        if not day_results:
            return {}

        pnls = [d.trade_pnl for d in day_results]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        # Profit factor: gross_profit / gross_loss
        if gross_loss == 0:
            profit_factor = 999.0 if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss

        return {
            "total_return": sum(pnls),
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "avg_win": np.mean(wins) if wins else 0.0,
            "avg_loss": np.mean(losses) if losses else 0.0,
            "best_day": max(pnls) if pnls else 0.0,
            "worst_day": min(pnls) if pnls else 0.0,
            "win_trades": len(wins),
            "loss_trades": len(losses),
        }

    def run(
        self,
        model,
        model_name: str,
        df: pd.DataFrame,
        ticker: str = "",
        use_time_weights: bool = False,
        sentiment_score: float = 0.0,
    ) -> BacktestResult:
        """
        Run the backtest on a single model.

        Args:
            model:           Any object with .predict(df, use_time_weights, sentiment_score)
            model_name:      Human-readable label for the report.
            df:              Full price DataFrame (must have 'close' and 'date' columns).
            ticker:          Ticker symbol (for labeling only).
            use_time_weights: Passed through to model.predict().
            sentiment_score:  Passed through to model.predict().

        Returns:
            BacktestResult with per-day breakdown, accuracy, profit, and streak metrics.
        """
        if len(df) < self.n_days + 20:
            raise ValueError(
                f"Not enough data for backtest: need at least {self.n_days + 20} rows, "
                f"got {len(df)}"
            )

        day_results = []

        for i in range(self.n_days, 0, -1):
            train_df = df.iloc[:-i].copy()
            eval_idx = len(df) - i

            close_before = float(train_df["close"].iloc[-1])
            close_actual = float(df["close"].iloc[eval_idx])
            actual_direction = "UP" if close_actual > close_before else "DOWN"

            predicted, confidence = model.predict(
                train_df,
                use_time_weights=use_time_weights,
                sentiment_score=sentiment_score,
            )

            if predicted not in ("UP", "DOWN"):
                continue

            trade_pnl = self._compute_trade_pnl(predicted, close_before, close_actual)

            day_results.append(DayResult(
                date=str(df["date"].iloc[eval_idx]),
                predicted=predicted,
                actual=actual_direction,
                confidence=confidence,
                correct=(predicted == actual_direction),
                close_before=close_before,
                close_actual=close_actual,
                trade_pnl=trade_pnl,
            ))

        correct_count = sum(1 for d in day_results if d.correct)
        total = len(day_results)
        metrics = self._compute_profit_metrics(day_results)
        streaks = self._compute_streaks(day_results)

        return BacktestResult(
            model_name=model_name,
            ticker=ticker,
            test_days=total,
            correct=correct_count,
            accuracy=correct_count / total if total > 0 else 0.0,
            days=day_results,
            **metrics,
            **streaks,
        )
