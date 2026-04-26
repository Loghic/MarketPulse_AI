"""
knn_model.py – k-Nearest Neighbors model for next-day price direction prediction.

Feature engineering is handled by the shared engine.features module.
Sentiment adjustment is applied post-hoc to the base probability.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, List

from engine.features import (
    DEFAULT_FEATURES, ALL_FEATURES, validate_features, feature_label,
    build_feature_matrix, compute_feature_columns, build_feature_vector,
)


SENTIMENT_WEIGHT = 0.20


class KNNModel:
    def __init__(
        self,
        k: int = 5,
        window_size: int = 5,
        features: List[str] | None = None,
    ):
        self.k = k
        self.window_size = window_size
        self.features = features or DEFAULT_FEATURES
        validate_features(self.features)

    @property
    def feature_label(self) -> str:
        return feature_label(self.features)

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
        Train k-NN on the provided data and predict next-day direction.

        Args:
            df:               DataFrame with 'close' column (and 'volume' if used).
            use_time_weights: If True, use 'distance' weighting and trim training data.
            sentiment_score:  News sentiment in [-1, 1], applied post-hoc.

        Returns:
            (direction, confidence) — e.g. ("UP", 0.73)
        """
        X, y = build_feature_matrix(
            df, self.features, self.window_size, target_type="binary"
        )

        if X is None or len(X) < self.k:
            return "Insufficient data", 0.0

        # --- Scale + train ---
        scaler = StandardScaler()
        weights = "distance" if use_time_weights else "uniform"
        model = KNeighborsClassifier(n_neighbors=self.k, weights=weights)

        if use_time_weights and len(X) > self.k * 3:
            cutoff = len(X) // 2
            X_train, y_train = X[cutoff:], y[cutoff:]
        else:
            X_train, y_train = X, y

        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)

        # --- Build prediction vector from last window ---
        df_feat = compute_feature_columns(df, self.features, self.window_size)
        last_vec = build_feature_vector(
            df_feat, len(df_feat) - self.window_size, self.features, self.window_size
        )

        if last_vec is None:
            return "Data error", 0.0

        last_vec_scaled = scaler.transform(last_vec.reshape(1, -1))

        raw_prediction = model.predict(last_vec_scaled)[0]
        raw_prob = float(np.max(model.predict_proba(last_vec_scaled)))

        # --- Sentiment adjustment ---
        final_pred, final_prob = self._apply_sentiment(
            raw_prediction, raw_prob, sentiment_score
        )

        return ("UP" if final_pred == 1 else "DOWN"), final_prob
