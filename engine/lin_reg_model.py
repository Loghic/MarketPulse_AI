"""
lin_reg_model.py – Linear Regression model for next-day price direction prediction.

Unlike k-NN (which classifies UP/DOWN directly), Linear Regression predicts
the next day's return as a continuous value. The sign determines direction,
and the magnitude is mapped to a confidence score via a sigmoid function.

Supports the same configurable feature sets as k-NN (returns, volume, RSI,
volatility, MACD) via the shared engine.features module.

Sentiment integration uses the same two-stage post-hoc approach.
"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, List

from engine.features import (
    DEFAULT_FEATURES, ALL_FEATURES, validate_features, feature_label,
    build_feature_matrix, compute_feature_columns, build_feature_vector,
)


SENTIMENT_WEIGHT = 0.20


class LinearRegressionModel:
    def __init__(
        self,
        window_size: int = 5,
        features: List[str] | None = None,
    ):
        self.window_size = window_size
        self.features = features or DEFAULT_FEATURES
        validate_features(self.features)

    @property
    def feature_label(self) -> str:
        return feature_label(self.features)

    # ------------------------------------------------------------------
    # Confidence mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _return_to_confidence(predicted_return: float) -> float:
        """
        Map a predicted return to a confidence score in [0.5, 1.0].

        Uses a sigmoid: small predicted returns → ~50%, large → ~100%.
        Scaling factor 100: ±1% return ≈ 73% confidence, ±2% ≈ 88%.
        """
        raw = 1.0 / (1.0 + np.exp(-abs(predicted_return) * 100))
        return float(np.clip(raw, 0.50, 0.99))

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
        """Shift probability using sentiment. Can flip the prediction."""
        if sentiment_score == 0.0:
            return prediction, prob

        prob_up = prob if prediction == 1 else (1.0 - prob)
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
        Train Linear Regression and predict next-day direction.

        Args:
            df:               DataFrame with 'close' column (and 'volume' if used).
            use_time_weights: If True, weight recent samples more heavily
                              using sample_weight (LinReg supports this natively).
            sentiment_score:  News sentiment in [-1, 1], applied post-hoc.

        Returns:
            (direction, confidence) — e.g. ("UP", 0.73)
        """
        X, y = build_feature_matrix(
            df, self.features, self.window_size, target_type="continuous"
        )

        if X is None or len(X) < self.window_size:
            return "Insufficient data", 0.0

        # --- Scale + train ---
        scaler = StandardScaler()
        model = LinearRegression()

        if use_time_weights:
            weights = np.linspace(0.1, 1.0, len(y))
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y, sample_weight=weights)
        else:
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)

        # --- Build prediction vector from last window ---
        df_feat = compute_feature_columns(df, self.features, self.window_size)
        last_vec = build_feature_vector(
            df_feat, len(df_feat) - self.window_size, self.features, self.window_size
        )

        if last_vec is None:
            return "Data error", 0.0

        last_vec_scaled = scaler.transform(last_vec.reshape(1, -1))
        predicted_return = model.predict(last_vec_scaled)[0]

        # Direction from sign, confidence from magnitude
        raw_prediction = 1 if predicted_return > 0 else 0
        raw_prob = self._return_to_confidence(predicted_return)

        # Sentiment adjustment
        final_pred, final_prob = self._apply_sentiment(
            raw_prediction, raw_prob, sentiment_score
        )

        return ("UP" if final_pred == 1 else "DOWN"), final_prob
