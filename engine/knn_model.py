"""
knn_model.py – k-Nearest Neighbors model for next-day price direction prediction.

Sentiment integration strategy:
    Historical per-day sentiment scores are rarely available for the full training
    window, so instead of adding sentiment as a training feature (which would create
    a dimension mismatch), we use a two-stage approach:

    1. k-NN predicts direction and base probability from price-return patterns.
    2. Sentiment score adjusts the probability post-hoc — positive news boosts
       an UP prediction (or weakens DOWN), negative news does the opposite.

    This keeps the model trainable on price data alone while still letting
    current news shift the final output.
"""

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from typing import Tuple


# How much sentiment can move the base probability (0.0–1.0).
# At 0.20 a perfect sentiment score (+1 or -1) shifts probability by ±20pp.
SENTIMENT_WEIGHT = 0.20


class KNNModel:
    def __init__(self, k: int = 5, window_size: int = 5):
        self.k = k
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def prepare_features(self, df) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """
        Build feature matrix from a sliding window of daily returns.

        Each sample is a vector of `window_size` consecutive returns.
        Target is binary: 1 if next day's close > today's close, else 0.

        Returns:
            (X, y) numpy arrays, or (None, None) if data is insufficient.
        """
        if len(df) < self.window_size + 2:
            return None, None

        df = df.copy()
        df["return"] = df["close"].pct_change()
        df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)

        # Drop rows with NaN (first from pct_change, last from shift)
        df = df.dropna(subset=["return", "target"])

        X, y = [], []
        for i in range(len(df) - self.window_size):
            window = df["return"].iloc[i : i + self.window_size].values
            if np.isnan(window).any():
                continue
            X.append(window)
            y.append(df["target"].iloc[i + self.window_size - 1])

        if len(X) == 0:
            return None, None

        return np.array(X), np.array(y)

    # ------------------------------------------------------------------
    # Sentiment adjustment
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_sentiment(
        prediction: int,
        prob: float,
        sentiment_score: float,
        weight: float = SENTIMENT_WEIGHT,
    ) -> Tuple[int, float]:
        """
        Shift k-NN probability using the current sentiment score.

        The adjustment is directional:
            - Positive sentiment pushes probability toward UP  (class 1).
            - Negative sentiment pushes probability toward DOWN (class 0).

        If the shift is strong enough it can flip the predicted class.

        Args:
            prediction:      Raw k-NN class (1=UP, 0=DOWN).
            prob:            Raw k-NN max-class probability.
            sentiment_score: News sentiment in [-1, 1].
            weight:          Max adjustment magnitude.

        Returns:
            (adjusted_prediction, adjusted_probability)
        """
        if sentiment_score == 0.0:
            return prediction, prob

        # Convert to "probability of UP": if prediction is UP prob_up = prob,
        # if prediction is DOWN prob_up = 1 - prob.
        prob_up = prob if prediction == 1 else (1.0 - prob)

        # Sentiment > 0 increases prob_up, sentiment < 0 decreases it
        prob_up_adjusted = prob_up + sentiment_score * weight
        prob_up_adjusted = np.clip(prob_up_adjusted, 0.01, 0.99)

        # Derive final prediction from adjusted probability
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
        Train k-NN on the provided data and predict next-day direction.

        Args:
            df:               DataFrame with at least a 'close' column.
            use_time_weights: If True, use 'distance' weighting and trim
                              training data to the more recent half.
            sentiment_score:  News sentiment in [-1, 1]. Applied as a
                              post-hoc adjustment to the base probability.

        Returns:
            (direction, confidence) — e.g. ("UP", 0.73)
        """
        X, y = self.prepare_features(df)

        if X is None or len(X) < self.k:
            return "Insufficient data", 0.0

        # --- Train ---
        weights = "distance" if use_time_weights else "uniform"
        model = KNeighborsClassifier(n_neighbors=self.k, weights=weights)

        if use_time_weights and len(X) > self.k * 3:
            # Time weighting: keep only the more recent half of samples.
            # (scikit-learn k-NN does not support sample_weight in fit)
            cutoff = len(X) // 2
            X_train, y_train = X[cutoff:], y[cutoff:]
        else:
            X_train, y_train = X, y

        model.fit(X_train, y_train)

        # --- Predict ---
        returns = df["close"].pct_change().dropna().values
        if len(returns) < self.window_size:
            return "Data error", 0.0

        last_window = returns[-self.window_size:].reshape(1, -1)
        if np.isnan(last_window).any():
            return "Data error", 0.0

        raw_prediction = model.predict(last_window)[0]
        raw_prob = float(np.max(model.predict_proba(last_window)))

        # --- Sentiment adjustment ---
        final_pred, final_prob = self._apply_sentiment(
            raw_prediction, raw_prob, sentiment_score
        )

        return ("UP" if final_pred == 1 else "DOWN"), final_prob
