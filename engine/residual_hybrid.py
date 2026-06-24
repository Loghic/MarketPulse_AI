"""
residual_hybrid.py – Residual hybrid forecaster (the paper's central artifact).

    P̂_{t+1} = P̂^base_{t+1} + r̂es_{t+1}
    res_t    = close_t − fitted^base_t          (base's in-sample residuals)
    r̂es      = residual_learner trained on residuals up to t

The idea (Zhang 2003): a base model (Prophet, ARIMA, …) captures the smooth
trend/seasonality, and a second learner mops up the structure left in the base's
residuals. If those residuals are white noise, the learner predicts ~0 and the
hybrid reduces to the base — which is exactly the "*when* does residual learning
help" question the paper asks.

Composability (plan R3): ``ResidualHybrid(base, residual_learner)`` takes **any**
``ForecastModel`` as the base and **any** residual learner with a
``fit(residuals)`` / ``predict() -> float`` API. So Prophet+LSTM, ARIMA+LSTM,
Prophet+XGB, … are all just different constructor arguments.

Leakage (plan R0.2), enforced here and unit-tested:
  * the base's point forecast for ``t+h`` is a genuine OOS forecast
    (``base.forecast`` only sees the window ending at ``t``);
  * the residual learner is fit on ``res`` up to and including ``t`` and predicts
    the *next* residual — it never sees ``res_{t+h}``.

It subclasses ``ForecastModel`` and is stateless per call (fits the residual
learner inside ``_raw_forecast``), so it slots into the forecast harness exactly
like every other forecaster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from engine.forecast_base import ForecastModel, ForecastResult
from engine.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)


class ResidualLearner(Protocol):
    """Anything that learns next-residual from a residual series."""

    def fit(self, residuals: np.ndarray) -> None: ...
    def predict(self) -> float: ...


# Fit cadence for the residual learner across the walk-forward:
#   "per_step"   — refit every forecast call (most adaptive, slowest; default for
#                  correctness when not otherwise configured).
#   "refit_k"    — refit every K calls, reuse frozen weights in between (the
#                  learner must expose set_window()).
#   "pretrained" — never refit; weights were loaded once (pretrained on pre-eval
#                  residuals). Just set_window() + predict each call (fastest).
_FIT_MODES = ("per_step", "refit_k", "pretrained")


class ResidualHybrid(ForecastModel):
    """``base`` forecast + learned residual correction.

    ``fit_mode`` controls how often the residual learner is retrained across the
    walk-forward (see ``_FIT_MODES``). ``pretrained``/``refit_k`` need a learner
    that exposes ``set_window(residuals)`` (the shipped ``LSTMResidualLearner``
    does); a learner without it silently falls back to per-step behaviour.

    Leakage note: in every mode the learner only ever sees residuals from the
    window passed to ``forecast`` (which ends at ``t``). ``pretrained`` weights
    must have been trained on pre-eval residuals (the train script enforces
    that); reusing them to *predict* on later windows is not leakage — the same
    discipline as the saved LSTM-reg.
    """

    def __init__(
        self,
        base: ForecastModel,
        residual_learner: ResidualLearner,
        name: str | None = None,
        *,
        fit_mode: str = "per_step",
        refit_k: int = 21,
    ):
        if fit_mode not in _FIT_MODES:
            raise ValueError(f"fit_mode must be one of {_FIT_MODES}, got {fit_mode!r}")
        self.base = base
        self.residual_learner = residual_learner
        self.fit_mode = fit_mode
        self.refit_k = max(1, refit_k)
        self._call_count = 0
        learner_tag = getattr(residual_learner, "name", residual_learner.__class__.__name__)
        # e.g. "Prophet + LSTM-res"
        self.name = name or f"{base.name} + {learner_tag}-res"

    def _update_learner(self, residuals: np.ndarray) -> None:
        """Fit / refit / window-set the learner per the fit_mode."""
        set_window = getattr(self.residual_learner, "set_window", None)
        if self.fit_mode == "pretrained":
            # Frozen weights (loaded at construction). Just point them at the
            # latest window; if the learner can't, fall back to a one-off fit.
            if set_window is not None and getattr(self.residual_learner, "is_trained", True):
                set_window(residuals)
            else:
                self.residual_learner.fit(residuals)
            return
        if self.fit_mode == "refit_k" and set_window is not None:
            if self._call_count % self.refit_k == 0:
                self.residual_learner.fit(residuals)
            else:
                set_window(residuals)
            return
        # per_step (or refit_k without set_window support) → always refit.
        self.residual_learner.fit(residuals)

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        if "close" not in df.columns:
            return None

        # 1) Genuine OOS base forecast for t+h.
        base_fr = self.base.forecast(df, horizon=horizon)
        if base_fr is None or not np.isfinite(base_fr.point):
            return None
        p_base = float(base_fr.point)
        last_close = float(np.asarray(df["close"], dtype=float).ravel()[-1])

        # 2) In-sample residuals res_t = close_t - fitted^base_t, up to t only.
        fitted = np.asarray(self.base.fit_in_sample(df), dtype=float).ravel()
        closes = np.asarray(df["close"], dtype=float).ravel()
        if fitted.shape != closes.shape:
            # Misaligned fit → skip the residual step, return the base forecast.
            return ForecastResult(last_close=last_close, point=p_base, horizon=horizon)
        residuals = closes - fitted
        residuals = residuals[np.isfinite(residuals)]

        # 3) Update the learner (fit / refit / window) and predict r̂es_{t+h}.
        #    A failed/short fit predicts 0.0 → hybrid == base (safe fallback).
        try:
            self._update_learner(residuals)
            r_hat = float(self.residual_learner.predict())
        except Exception as e:  # noqa: BLE001 — never let the learner crash a run
            log.debug("%s: residual learner failed (%s); using base only.", self.name, e)
            r_hat = 0.0
        finally:
            self._call_count += 1
        if not np.isfinite(r_hat):
            r_hat = 0.0

        point = p_base + r_hat
        if not np.isfinite(point):
            return None
        return ForecastResult(last_close=last_close, point=point, horizon=horizon)
