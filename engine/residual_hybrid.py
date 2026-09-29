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
from engine.residual_diagnostics import ljung_box

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)


class ResidualLearner(Protocol):
    """Anything that learns next-residual from a residual series.

    ``exog`` is optional on both methods — a learner that ignores it (like
    ``ZeroResidualLearner``) still satisfies the protocol.
    """

    def fit(self, residuals: np.ndarray, exog: np.ndarray | None = None) -> None: ...
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
        adaptive_lambda: bool = False,
        lambda_window: int = 20,
        lambda_alpha: float = 0.05,
        macro_df=None,
    ):
        if fit_mode not in _FIT_MODES:
            raise ValueError(f"fit_mode must be one of {_FIT_MODES}, got {fit_mode!r}")
        self.base = base
        self.residual_learner = residual_learner
        self.fit_mode = fit_mode
        self.refit_k = max(1, refit_k)
        self.adaptive_lambda = adaptive_lambda
        self.lambda_window = lambda_window
        self.lambda_alpha = lambda_alpha
        self._call_count = 0
        # Optional exogenous panel for the RESIDUAL LEARNER (distinct from any
        # macro the `base` model itself may already consume) — same contract as
        # elsewhere (macro_data.align_macro / sentiment_data): already lag-1
        # aligned, row d holds info known at d-1, indexed by date string.
        if macro_df is not None and not macro_df.empty:
            self._macro = macro_df.copy()
            self._macro.index = self._macro.index.astype(str)
        else:
            self._macro = None
        learner_tag = getattr(residual_learner, "name", residual_learner.__class__.__name__)
        # e.g. "Prophet + LSTM-res"
        tag = f"{base.name} + {learner_tag}-res"
        self.name = name or (f"{tag} (adaptive-λ)" if adaptive_lambda else tag)

    def _lambda_t(self, residuals: np.ndarray) -> float:
        """Shrinkage weight in [0, 1] from rolling residual structure.

        λ = clip(1 - p_value/α, 0, 1): the Ljung-Box p-value on the most recent
        ``lambda_window`` residuals, rescaled against the same significance
        threshold ``diagnose()`` uses to call a residual "structured". p=0
        (strongly autocorrelated) → λ=1 (full correction); p>=α (white noise,
        fail to reject H0) → λ=0 (hybrid reduces to the base). Falls back to
        λ=1 (fixed-weight behaviour) when there isn't enough history to test.
        """
        window = residuals[-self.lambda_window :]
        lags = min(10, max(1, window.size - 2))
        if window.size <= lags + 1:
            return 1.0
        p = ljung_box(window, lags=lags).p_value
        if not np.isfinite(p):
            return 1.0
        return float(np.clip(1.0 - p / self.lambda_alpha, 0.0, 1.0))

    def _macro_vec(self, dates: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
        """Aligned macro rows for the (mask-filtered) residual dates, or None."""
        if self._macro is None:
            return None
        d = np.asarray(dates)[mask]
        rows = self._macro.reindex(d)
        if rows.isna().any().any():
            return None  # a gap in macro over this window → skip exog, stay univariate
        return rows.to_numpy(dtype=float)

    def _fit(self, residuals: np.ndarray, exog: np.ndarray | None) -> None:
        # Only pass exog when present, so learners with the old single-arg
        # ``fit(residuals)`` signature (this codebase's tests, any future
        # simple learner) keep working unchanged on the no-macro path.
        if exog is not None:
            self.residual_learner.fit(residuals, exog)
        else:
            self.residual_learner.fit(residuals)

    def _set_window(self, set_window, residuals: np.ndarray, last_exog: np.ndarray | None) -> None:
        if last_exog is not None:
            set_window(residuals, last_exog)
        else:
            set_window(residuals)

    def _update_learner(
        self, residuals: np.ndarray, exog: np.ndarray | None, last_exog: np.ndarray | None
    ) -> None:
        """Fit / refit / window-set the learner per the fit_mode."""
        set_window = getattr(self.residual_learner, "set_window", None)
        if self.fit_mode == "pretrained":
            # Frozen weights (loaded at construction). Just point them at the
            # latest window; if the learner can't, fall back to a one-off fit.
            if set_window is not None and getattr(self.residual_learner, "is_trained", True):
                self._set_window(set_window, residuals, last_exog)
            else:
                self._fit(residuals, exog)
            return
        if self.fit_mode == "refit_k" and set_window is not None:
            if self._call_count % self.refit_k == 0:
                self._fit(residuals, exog)
            else:
                self._set_window(set_window, residuals, last_exog)
            return
        # per_step (or refit_k without set_window support) → always refit.
        self._fit(residuals, exog)

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
        residuals_raw = closes - fitted
        finite_mask = np.isfinite(residuals_raw)
        residuals = residuals_raw[finite_mask]

        # 2b) Optional exog (macro) for the residual learner, aligned to the
        # same filtered positions/dates as `residuals` — see _macro_vec.
        exog = None
        last_exog = None
        if self._macro is not None and "date" in df.columns:
            dates = df["date"].astype(str).to_numpy()
            exog = self._macro_vec(dates, finite_mask)
            if exog is not None and len(exog):
                last_exog = exog[-1]

        # 3) Update the learner (fit / refit / window) and predict r̂es_{t+h}.
        #    A failed/short fit predicts 0.0 → hybrid == base (safe fallback).
        try:
            self._update_learner(residuals, exog, last_exog)
            r_hat = float(self.residual_learner.predict())
        except Exception as e:  # noqa: BLE001 — never let the learner crash a run
            log.debug("%s: residual learner failed (%s); using base only.", self.name, e)
            r_hat = 0.0
        finally:
            self._call_count += 1
        if not np.isfinite(r_hat):
            r_hat = 0.0
        if self.adaptive_lambda:
            r_hat *= self._lambda_t(residuals)

        point = p_base + r_hat
        if not np.isfinite(point):
            return None
        # Inherit the base's prediction interval as-is (not re-centered on the
        # corrected point) — an approximation, but the only interval available
        # without a dedicated uncertainty model for the residual step; good
        # enough for confidence gating (plan.md prerequisite #5).
        return ForecastResult(
            last_close=last_close,
            point=point,
            horizon=horizon,
            quantiles=base_fr.quantiles,
            extra=base_fr.extra,
        )
