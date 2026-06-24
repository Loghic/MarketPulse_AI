"""
arima_model.py – ARIMA point-forecaster (optional statsmodels dependency).

A classical Box–Jenkins ARIMA(p,d,q) fit on the close-price series, forecasting
``horizon`` steps ahead. It subclasses ``ForecastModel`` so it plugs into the
forecast harness exactly like the naive forecasters and Prophet, and exposes a
predictive interval that the base class turns into a direction/confidence (used
only by the directional adapter; the regression harness reads ``.point``).

Order selection: a fixed ``order`` (default ``(1,1,1)``) is used unless
``auto=True`` and ``pmdarima`` is installed, in which case ``auto_arima`` picks
the order. Per the evaluation contract (plan R0.5 / R2.2), any order search must
happen on the *selection* window only — the harness controls that by fitting on
the in-context series at each refit step; this model never peeks past ``t``.

Optional dependency: ``statsmodels`` (and optionally ``pmdarima`` for auto
order). Gated behind the ``[forecast]`` extra. If absent, constructing the
model raises a clear RuntimeError and the harness skips it — it never crashes a
run.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from engine.forecast_base import ForecastModel, ForecastResult
from engine.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

try:
    from statsmodels.tsa.arima.model import ARIMA as _SM_ARIMA

    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

try:
    from pmdarima import auto_arima as _auto_arima

    _PMDARIMA_AVAILABLE = True
except ImportError:
    _PMDARIMA_AVAILABLE = False

# Need a minimum of history for a stable fit.
MIN_ROWS = 30

# z for an ~80% central interval, matching Prophet's default reporting so
# direction/confidence are comparable across forecasting models.
_Z80 = 1.2815515594


class ARIMAForecaster(ForecastModel):
    """ARIMA(p,d,q) one-/h-step close-price forecaster."""

    name = "ARIMA"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        *,
        auto: bool = False,
        max_context: int | None = 750,
    ) -> None:
        if not _STATSMODELS_AVAILABLE:
            raise RuntimeError(
                "statsmodels is not installed (required for ARIMA). Install with:\n"
                "  uv pip install -e '.[forecast]'\n"
                "Or: uv pip install statsmodels"
            )
        self.order = order
        self.auto = auto
        # Cap the fit context: ARIMA on a very long series is slow and the
        # tail dominates a (1,1,1) anyway. None = use everything.
        self.max_context = max_context
        if auto and not _PMDARIMA_AVAILABLE:
            log.warning("ARIMA(auto=True) but pmdarima not installed; using fixed order %s.", order)
            self.auto = False

    def _select_order(self, y: np.ndarray) -> tuple[int, int, int]:
        if not self.auto:
            return self.order
        try:
            model = _auto_arima(
                y,
                seasonal=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
            )
            return tuple(model.order)  # type: ignore[return-value]
        except Exception as e:  # noqa: BLE001 — fall back, never crash a run
            log.debug("auto_arima failed (%s); using fixed order %s.", e, self.order)
            return self.order

    def fit_in_sample(self, df):
        """ARIMA's in-sample one-step fitted close series (``fittedvalues``).

        Falls back to the random-walk default if statsmodels is unavailable, the
        series is too short, or the fit errors. Aligned to ``df`` rows.
        """
        n = len(df)
        if not _STATSMODELS_AVAILABLE or "close" not in df.columns or n < MIN_ROWS:
            return super().fit_in_sample(df)
        y = np.asarray(df["close"], dtype=float).ravel()
        if not np.isfinite(y).all():
            return super().fit_in_sample(df)
        ctx = y if self.max_context is None else y[-self.max_context :]
        order = self._select_order(ctx)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted_ctx = np.asarray(_SM_ARIMA(ctx, order=order).fit().fittedvalues, dtype=float)
        except Exception:  # noqa: BLE001
            return super().fit_in_sample(df)
        if fitted_ctx.shape[0] != ctx.shape[0] or not np.isfinite(fitted_ctx).all():
            return super().fit_in_sample(df)
        # If we capped the context, pad the front with the RW default so the
        # returned array still aligns to all df rows.
        if fitted_ctx.shape[0] == n:
            return fitted_ctx
        out = super().fit_in_sample(df)
        out[-fitted_ctx.shape[0] :] = fitted_ctx
        return out

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        if "close" not in df.columns or len(df) < MIN_ROWS:
            return None
        y = np.asarray(df["close"], dtype=float).ravel()
        y = y[np.isfinite(y)]
        if y.size < MIN_ROWS:
            return None
        if self.max_context is not None and y.size > self.max_context:
            y = y[-self.max_context :]

        last_close = float(y[-1])
        order = self._select_order(y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence / non-stationarity chatter
            try:
                fit = _SM_ARIMA(y, order=order).fit()
                fc = fit.get_forecast(steps=horizon)
                point = float(np.asarray(fc.predicted_mean).ravel()[-1])
                se = float(np.asarray(fc.se_mean).ravel()[-1])
            except Exception as e:  # noqa: BLE001
                log.debug("ARIMA fit/forecast failed (%s).", e)
                return None

        if not np.isfinite(point):
            return None

        # Build an ~80% interval from the forecast SE so the base class can
        # derive a direction probability (regression harness ignores this).
        quantiles: dict[float, float] | None = None
        if np.isfinite(se) and se > 0:
            quantiles = {
                0.1: point - _Z80 * se,
                0.5: point,
                0.9: point + _Z80 * se,
            }
        return ForecastResult(
            last_close=last_close,
            point=point,
            horizon=horizon,
            quantiles=quantiles,
            extra={"order": order},
        )
