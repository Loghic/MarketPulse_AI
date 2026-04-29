"""
knn_model.py – k-Nearest Neighbors model for next-day price direction prediction.

Feature engineering is handled by the shared engine.features module.
Sentiment adjustment is applied post-hoc to the base probability.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from engine.features import (
    DEFAULT_FEATURES,
    build_feature_matrix,
    build_feature_vector,
    compute_feature_columns,
    feature_label,
    validate_features,
)

SENTIMENT_WEIGHT = 0.20


class KNNModel:
    def __init__(
        self,
        k: int = 5,
        window_size: int = 5,
        features: list[str] | None = None,
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
    ) -> tuple[int, float]:
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
    ) -> tuple[str, float]:
        """
        Train k-NN on the provided data and predict next-day direction.

        Args:
            df:               DataFrame with 'close' column (and 'volume' if used).
            use_time_weights: If True, use 'distance' weighting and trim training data.
            sentiment_score:  News sentiment in [-1, 1], applied post-hoc.

        Returns:
            (direction, confidence) — e.g. ("UP", 0.73)
        """
        X, y = build_feature_matrix(df, self.features, self.window_size, target_type="binary")

        if X is None or y is None or len(X) < self.k:
            return "Insufficient data", 0.0

        # --- Scale on full dataset ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # --- Build prediction vector from last window ---
        df_feat = compute_feature_columns(df, self.features, self.window_size)
        last_vec = build_feature_vector(
            df_feat, len(df_feat) - self.window_size, self.features, self.window_size
        )

        if last_vec is None:
            return "Data error", 0.0

        last_vec_scaled = scaler.transform(last_vec.reshape(1, -1))

        if use_time_weights and len(X) > self.k * 3:
            # Exponential time decay × inverse distance: older samples get less weight.
            # λ=3 means the most recent sample is ~e^3 ≈ 20× more important than
            # the oldest. Combined with distance so similar patterns still matter.
            model = KNeighborsClassifier(n_neighbors=self.k, weights="uniform")
            model.fit(X_scaled, y)

            neigh_dist, neigh_idx = model.kneighbors(last_vec_scaled)
            positions = neigh_idx[0]
            n = len(X)

            time_w = np.exp(positions / n * 3.0)
            dist_w = 1.0 / (neigh_dist[0] + 1e-10)
            combined = time_w * dist_w

            labels = y[positions]
            up_w = combined[labels == 1].sum()
            down_w = combined[labels == 0].sum()
            total_w = up_w + down_w

            prob_up = up_w / total_w if total_w > 0 else 0.5
            raw_prediction = 1 if prob_up >= 0.5 else 0
            raw_prob = float(prob_up if raw_prediction == 1 else 1.0 - prob_up)
        else:
            model = KNeighborsClassifier(n_neighbors=self.k, weights="uniform")
            model.fit(X_scaled, y)
            raw_prediction = model.predict(last_vec_scaled)[0]
            raw_prob = float(np.max(model.predict_proba(last_vec_scaled)))

        # --- Sentiment adjustment ---
        final_pred, final_prob = self._apply_sentiment(raw_prediction, raw_prob, sentiment_score)

        return ("UP" if final_pred == 1 else "DOWN"), final_prob
