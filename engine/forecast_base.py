"""
forecast_base.py – Shared base for *forecasting* models (Prophet, Chronos-2,
TiRex, Kronos).

Unlike k-NN / LinReg / LSTM, which directly classify next-day direction, these
models forecast a future *value* (a point estimate plus, where available, a
predictive distribution). This module adapts that value forecast to the same
``predict(df, use_time_weights, sentiment_score) -> (str, float)`` interface the
rest of the app expects, while *also* exposing the raw forecast via
``forecast(df) -> ForecastResult`` so the predicted value can be used directly
later (charts, regression targets, multi-step horizons, …).

Direction + confidence are derived from the predictive distribution so they line
up with how the existing models report confidence (the winning-class
probability):

    prob_up = P(forecast_next > last_close)
    direction  = "UP" if prob_up >= 0.5 else "DOWN"
    confidence = prob_up if UP else (1 - prob_up)

``prob_up`` comes from Monte-Carlo samples if present, else from the forecast
quantiles, else (last resort) a bounded function of the point estimate. Sentiment
is applied post-hoc with the same weight and logic as k-NN / LinReg / LSTM.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # keep the base importable with only numpy installed
    import pandas as pd

# Same weight used by k-NN, LinReg and LSTM, so "+ News" variants behave
# consistently across every model family.
SENTIMENT_WEIGHT = 0.20

# Fallback only: when a model gives a bare point estimate with no distribution,
# map the relative move to a bounded pseudo-probability via tanh. A move equal
# to this fraction (~5%) maps to ~0.88 confidence; a 2% move stays a modest
# ~0.69. Deliberately under-confident — none of the four models hit this path
# (Prophet sets prob_up; Chronos/TiRex give quantiles; Kronos gives samples).
_POINT_ONLY_SCALE = 0.05


@dataclass
class ForecastResult:
    """
    Raw forecast output. ``point`` is the headline predicted next-step value
    (median / mean close). Distributional info is optional but, when present,
    drives the direction probability.

    This is the object to reach for when you want the *value* rather than the
    UP/DOWN label.
    """

    last_close: float
    point: float
    horizon: int = 1
    # {quantile_level: forecast_value}, e.g. {0.1: 101.2, 0.5: 103.0, 0.9: 105.4}
    quantiles: dict[float, float] | None = None
    # Monte-Carlo sample values for the forecast step (1D array). Used by Kronos.
    samples: np.ndarray | None = None
    # Filled in by ForecastModel._finalize(). NaN means "derive it".
    prob_up: float = float("nan")
    direction: str = ""
    confidence: float = 0.0
    # Free-form room for model-specific extras (full path, raw frame, …).
    extra: dict = field(default_factory=dict)


class ForecastModel(ABC):
    """
    Base class adapting a value forecast to the app's predict() contract.

    Subclasses implement ``_raw_forecast`` and set ``name``. They may either set
    ``prob_up`` directly on the ForecastResult (e.g. Prophet, from its
    interval) or leave it NaN and provide ``quantiles`` / ``samples`` for the
    base class to derive it from.
    """

    name: str = "forecast"
    sentiment_weight: float = SENTIMENT_WEIGHT

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    @abstractmethod
    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        """
        Produce a ForecastResult from a price DataFrame (needs at least a
        'close' column; some models also use 'date'). Return None if there is
        not enough usable data — the caller turns that into an
        "Insufficient data" sentinel so the backtester skips the day.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Direction probability
    # ------------------------------------------------------------------

    @staticmethod
    def _prob_up_from_samples(last_close: float, samples: np.ndarray) -> float:
        arr = np.asarray(samples, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.5
        return float(np.mean(arr > last_close))

    @staticmethod
    def _prob_up_from_quantiles(last_close: float, quantiles: dict[float, float]) -> float:
        """
        Estimate P(value > last_close) by inverting the quantile function.

        The quantiles map levels -> forecast values (monotonically increasing in
        the level). Interpolating last_close back through (values -> levels)
        gives an estimate of the CDF at last_close; 1 - CDF is P(up). np.interp
        clamps outside the provided levels, which conveniently caps confidence
        at the outermost quantile (e.g. 0.9) rather than ever claiming 100%.
        """
        levels = sorted(quantiles)
        vals = [float(quantiles[lvl]) for lvl in levels]
        # Guard against non-monotone values (numerical noise) by sorting.
        order = np.argsort(vals)
        vals_sorted = np.asarray(vals)[order]
        levels_sorted = np.asarray(levels)[order]
        if len(vals_sorted) == 1:
            return 1.0 if last_close < vals_sorted[0] else 0.0
        cdf_at = float(np.interp(last_close, vals_sorted, levels_sorted))
        return 1.0 - cdf_at

    def _derive_prob_up(self, fr: ForecastResult) -> float:
        if fr.samples is not None and np.asarray(fr.samples).size > 0:
            return self._prob_up_from_samples(fr.last_close, fr.samples)
        if fr.quantiles:
            return self._prob_up_from_quantiles(fr.last_close, fr.quantiles)
        # Point-only fallback: bounded, direction-correct pseudo-probability.
        denom = abs(fr.last_close) * _POINT_ONLY_SCALE
        if denom <= 0:
            return 0.5
        return 0.5 + 0.5 * math.tanh((fr.point - fr.last_close) / denom)

    def _finalize(self, fr: ForecastResult) -> ForecastResult:
        prob_up = fr.prob_up
        if not np.isfinite(prob_up):
            prob_up = self._derive_prob_up(fr)
        prob_up = float(np.clip(prob_up, 0.0, 1.0))

        # Tie-break a coin-flip on the point estimate so we never emit a
        # direction that contradicts the headline value.
        if abs(prob_up - 0.5) < 1e-9:
            prob_up = 0.5 + (1e-6 if fr.point >= fr.last_close else -1e-6)

        fr.prob_up = prob_up
        if prob_up >= 0.5:
            fr.direction, fr.confidence = "UP", prob_up
        else:
            fr.direction, fr.confidence = "DOWN", 1.0 - prob_up
        return fr

    # ------------------------------------------------------------------
    # Sentiment (identical to k-NN / LinReg / LSTM)
    # ------------------------------------------------------------------

    @classmethod
    def _apply_sentiment(
        cls,
        prediction: int,
        prob: float,
        sentiment_score: float,
        weight: float | None = None,
    ) -> tuple[int, float]:
        """Shift the up-probability using sentiment. Can flip the prediction."""
        if sentiment_score == 0.0:
            return prediction, prob
        w = cls.sentiment_weight if weight is None else weight
        prob_up = prob if prediction == 1 else (1.0 - prob)
        prob_up_adjusted = float(np.clip(prob_up + sentiment_score * w, 0.01, 0.99))
        if prob_up_adjusted >= 0.5:
            return 1, prob_up_adjusted
        return 0, 1.0 - prob_up_adjusted

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        """
        Return the raw value forecast (with direction/confidence filled in), or
        None if there isn't enough data / the model errored. Never raises — the
        backtester relies on that.
        """
        try:
            fr = self._raw_forecast(df, horizon=horizon)
        except Exception:  # noqa: BLE001 - a bad day must not kill a long backtest
            return None
        if fr is None:
            return None
        return self._finalize(fr)

    def predict(
        self,
        df: pd.DataFrame,
        use_time_weights: bool = False,
        sentiment_score: float = 0.0,
    ) -> tuple[str, float]:
        """
        Next-day direction, same contract as every other model.

        Args:
            df:               Price DataFrame ('close' required, 'date' used by
                              some models).
            use_time_weights: Ignored (these models handle recency internally),
                              accepted for interface compatibility like LSTM.
            sentiment_score:  News sentiment in [-1, 1], applied post-hoc.

        Returns ("UP" | "DOWN", confidence) on success, or a sentinel
        ("Insufficient data" | "Data error", 0.0) that the backtester skips.
        """
        fr = self.forecast(df)
        if fr is None:
            return "Insufficient data", 0.0
        raw_pred = 1 if fr.direction == "UP" else 0
        final_pred, final_prob = self._apply_sentiment(raw_pred, fr.confidence, sentiment_score)
        return ("UP" if final_pred == 1 else "DOWN"), final_prob
