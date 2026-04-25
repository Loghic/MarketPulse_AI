"""
lin_reg_model.py – Linear Regression model for next-day price direction prediction.

Unlike k-NN (which classifies UP/DOWN directly), Linear Regression predicts
the next day's return as a continuous value. The sign determines direction,
and the magnitude is mapped to a confidence score via a sigmoid function.

Sentiment integration uses the same two-stage approach as k-NN:
    1. LinReg predicts direction and base probability from price-return patterns.
    2. Sentiment score adjusts the probability post-hoc.
"""

from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Tuple


# How much sentiment can move the base probability (same scale as k-NN)
SENTIMENT_WEIGHT = 0.20


class LinearRegressionModel:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def prepare_features(self, df) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Build feature matrix from a sliding window of daily returns.

        Each sample is a vector of `window_size` consecutive returns.
        Target is the next day's return (continuous, not binary).

        Returns:
            (X, y) numpy arrays, or (None, None) if data is insufficient.
        """
        if len(df) < self.window_size + 2:
            return None, None

        df = df.copy()
        df["return"] = df["close"].pct_change()
        df["next_return"] = df["return"].shift(-1)

        # Drop rows with NaN (first from pct_change, last from shift)
        df = df.dropna(subset=["return", "next_return"])

        X, y = [], []
        for i in range(len(df) - self.window_size):
            window = df["return"].iloc[i : i + self.window_size].values
            if np.isnan(window).any():
                continue
            target = df["next_return"].iloc[i + self.window_size - 1]
            if np.isnan(target):
                continue
            X.append(window)
            y.append(target)

        if len(X) == 0:
            return None, None

        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Confidence mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _return_to_confidence(predicted_return: float) -> float:
        """
        Map a predicted return to a confidence score in [0.5, 1.0].

        Uses a sigmoid-like function: small predicted returns → ~50% confidence,
        large predicted returns → approaching 100% confidence.

        The scaling factor (100) is tuned so that a ±1% predicted return
        maps to roughly 73% confidence, and ±2% to roughly 88%.
        """
        raw = 1.0 / (1.0 + np.exp(-abs(predicted_return) * 100))
        # Scale from sigmoid range [0.5, 1.0] to our confidence range
        return float(np.clip(raw, 0.50, 0.99))

    # ------------------------------------------------------------------
    # Sentiment adjustment (shared logic with k-NN)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_sentiment(
        prediction: int,
        prob: float,
        sentiment_score: float,
        weight: float = SENTIMENT_WEIGHT,
    ) -> Tuple[int, float]:
        """
        Shift base probability using the current sentiment score.

        Same two-stage approach as k-NN: positive sentiment pushes toward UP,
        negative pushes toward DOWN. Strong enough sentiment can flip the call.
        """
        if sentiment_score == 0.0:
            return prediction, prob

        # Convert to "probability of UP"
        prob_up = prob if prediction == 1 else (1.0 - prob)

        # Apply sentiment shift
        prob_up_adjusted = prob_up + sentiment_score * weight
        prob_up_adjusted = np.clip(prob_up_adjusted, 0.01, 0.99)

        if prob_up_adjusted >= 0.5:
            return 1, float(prob_up_adjusted)
        else:
            return 0, float(1.0 - prob_up_adjusted)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        df,
        use_time_weights: bool = False,
        sentiment_score: float = 0.0,
    ) -> Tuple[str, float]:
        """
        Train a Linear Regression on the provided data and predict next-day direction.

        Args:
            df:               DataFrame with at least a 'close' column.
            use_time_weights: If True, weight recent samples more heavily
                              using sample_weight (LinReg supports this natively).
            sentiment_score:  News sentiment in [-1, 1]. Applied as a
                              post-hoc adjustment to the base probability.

        Returns:
            (direction, confidence) — e.g. ("UP", 0.73)
        """
        X, y = self.prepare_features(df)

        if X is None or len(X) < self.window_size:
            return "Insufficient data", 0.0

        model = LinearRegression()

        if use_time_weights:
            # Linear weights: oldest sample = 0.1, newest = 1.0
            # Unlike k-NN, sklearn LinearRegression supports sample_weight natively
            weights = np.linspace(0.1, 1.0, len(y))
            model.fit(X, y, sample_weight=weights)
        else:
            model.fit(X, y)

        # Predict using the most recent window
        returns = df["close"].pct_change().dropna().values
        if len(returns) < self.window_size:
            return "Data error", 0.0

        last_window = returns[-self.window_size:].reshape(1, -1)
        if np.isnan(last_window).any():
            return "Data error", 0.0

        predicted_return = model.predict(last_window)[0]

        # Direction from sign, confidence from magnitude
        raw_prediction = 1 if predicted_return > 0 else 0
        raw_prob = self._return_to_confidence(predicted_return)

        # Sentiment adjustment
        final_pred, final_prob = self._apply_sentiment(
            raw_prediction, raw_prob, sentiment_score
        )

        return ("UP" if final_pred == 1 else "DOWN"), final_prob
