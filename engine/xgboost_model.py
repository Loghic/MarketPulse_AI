"""
xgboost_model.py – Gradient-boosted-tree point-forecaster (optional xgboost).

An ``XGBRegressor`` trained on the tabular feature matrix from ``features.py``
to predict the next (or h-ahead) **close price level**. It subclasses
``ForecastModel`` so it plugs into the forecast harness like every other
forecaster and exposes ``.point`` for the regression metrics.

Why predict a level (not Δ)? Per the evaluation contract (plan R0.1) the
regression track forecasts the **price level**. Trees can't extrapolate beyond
the training range, so a pure level target would cap forecasts at the last seen
price in a trending series. To avoid that we train on the **one-step change**
(``Δ = close[t+h] - close[t]``) and add it back to the last close:
``P̂_{t+h} = P_t + Δ̂``. This keeps the model honest on trends while still
reporting a level — the harness and metrics only ever see the level.

Leakage: every training pair uses a feature window ending at ``t`` to predict
``t+h``; the most-recent prediction uses the window ending at the last in-context
day. The model is refit by the harness on each window, never seeing future data.

Optional dependency: ``xgboost`` (``[forecast]`` extra). If absent, constructing
raises a clear RuntimeError and the harness skips it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.features import (
    DEFAULT_FEATURES,
    build_feature_vector,
    compute_feature_columns,
    min_rows_needed,
    validate_features,
)
from engine.forecast_base import ForecastModel, ForecastResult
from engine.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

try:
    from xgboost import XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

# Need enough rows to form a feature window AND a handful of training pairs.
_MIN_TRAIN_PAIRS = 40


class XGBoostForecaster(ForecastModel):
    """XGBoost regressor forecasting the next close-price level (via Δ target)."""

    name = "XGBoost"

    def __init__(
        self,
        *,
        features: list[str] | None = None,
        window_size: int = 10,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        if not _XGBOOST_AVAILABLE:
            raise RuntimeError(
                "xgboost is not installed. Install with:\n"
                "  uv pip install -e '.[forecast]'\n"
                "Or: uv pip install xgboost"
            )
        self.features = list(features) if features else list(DEFAULT_FEATURES)
        validate_features(self.features)
        self.window_size = window_size
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=1,  # deterministic + harness already parallel at ticker level
        )

    def _build_training_set(
        self, df: pd.DataFrame, horizon: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """(X, y) where X = feature window ending at t, y = close[t+h] - close[t]."""
        feat_df = compute_feature_columns(df, self.features, self.window_size)
        closes = np.asarray(feat_df["close"], dtype=float)
        n = len(feat_df)
        x_rows: list[np.ndarray] = []
        y_rows: list[float] = []
        # A window starts at idx and ends at idx+window_size-1 == t; target is t+h.
        for idx in range(0, n - self.window_size - horizon + 1):
            t = idx + self.window_size - 1
            vec = build_feature_vector(feat_df, idx, self.features, self.window_size)
            if vec is None:
                continue
            target = closes[t + horizon] - closes[t]
            if not np.isfinite(target):
                continue
            x_rows.append(vec)
            y_rows.append(float(target))
        if len(x_rows) < _MIN_TRAIN_PAIRS:
            return None
        return np.asarray(x_rows), np.asarray(y_rows)

    def _latest_vector(self, df: pd.DataFrame) -> np.ndarray | None:
        """Feature window ending at the last in-context day (the prediction input)."""
        feat_df = compute_feature_columns(df, self.features, self.window_size)
        last_start = len(feat_df) - self.window_size
        if last_start < 0:
            return None
        return build_feature_vector(feat_df, last_start, self.features, self.window_size)

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        if "close" not in df.columns:
            return None
        if len(df) < min_rows_needed(self.features, self.window_size) + horizon:
            return None

        built = self._build_training_set(df, horizon)
        if built is None:
            return None
        x, y = built

        x_last = self._latest_vector(df)
        if x_last is None:
            return None

        try:
            model = XGBRegressor(**self._params)
            model.fit(x, y)
            delta = float(model.predict(x_last.reshape(1, -1))[0])
        except Exception as e:  # noqa: BLE001 — a bad day must not kill a run
            log.debug("XGBoost fit/predict failed (%s).", e)
            return None

        last_close = float(np.asarray(df["close"], dtype=float).ravel()[-1])
        point = last_close + delta
        if not np.isfinite(point):
            return None
        return ForecastResult(last_close=last_close, point=point, horizon=horizon)
