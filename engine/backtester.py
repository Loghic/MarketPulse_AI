"""
backtester.py – Walk-forward backtest engine.

Evaluates model accuracy by hiding the last N days of data, then
predicting each day one at a time and comparing against the actual outcome.

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


@dataclass
class BacktestResult:
    """Aggregated backtest results."""
    model_name: str
    ticker: str
    test_days: int
    correct: int
    accuracy: float
    days: List[DayResult] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"  Model: {self.model_name}",
            f"  Accuracy: {self.correct}/{self.test_days} "
            f"({self.accuracy:.1%})",
            "",
        ]
        for d in self.days:
            mark = "✓" if d.correct else "✗"
            lines.append(
                f"    {d.date}  pred={d.predicted:<5} actual={d.actual:<5} "
                f"conf={d.confidence:.1%}  {mark}"
            )
        return "\n".join(lines)


class Backtester:
    """
    Walk-forward backtester.

    For each of the last `n_days` trading days:
        1. Slice data up to (but not including) that day.
        2. Ask the model to predict direction.
        3. Compare with the actual price movement.
    """

    def __init__(self, n_days: int = 5):
        self.n_days = n_days

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
            BacktestResult with per-day breakdown and overall accuracy.
        """
        if len(df) < self.n_days + 20:
            raise ValueError(
                f"Not enough data for backtest: need at least {self.n_days + 20} rows, "
                f"got {len(df)}"
            )

        day_results = []

        for i in range(self.n_days, 0, -1):
            # Split: train on everything except the last `i` rows,
            # evaluate against the row right after the split point
            train_df = df.iloc[:-i].copy()
            eval_idx = len(df) - i  # index of the day we're predicting

            # Actual direction: did price go up from the last training day?
            close_before = float(train_df["close"].iloc[-1])
            close_actual = float(df["close"].iloc[eval_idx])
            actual_direction = "UP" if close_actual > close_before else "DOWN"

            # Model prediction (uses only training data)
            predicted, confidence = model.predict(
                train_df,
                use_time_weights=use_time_weights,
                sentiment_score=sentiment_score,
            )

            # Skip error/insufficient data results
            if predicted not in ("UP", "DOWN"):
                continue

            day_results.append(DayResult(
                date=str(df["date"].iloc[eval_idx]),
                predicted=predicted,
                actual=actual_direction,
                confidence=confidence,
                correct=(predicted == actual_direction),
                close_before=close_before,
                close_actual=close_actual,
            ))

        correct_count = sum(1 for d in day_results if d.correct)
        total = len(day_results)

        return BacktestResult(
            model_name=model_name,
            ticker=ticker,
            test_days=total,
            correct=correct_count,
            accuracy=correct_count / total if total > 0 else 0.0,
            days=day_results,
        )
