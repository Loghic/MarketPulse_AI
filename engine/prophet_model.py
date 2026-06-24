"""
prophet_model.py – Meta's Prophet as a next-day direction model.

Prophet is a classical additive model (trend + optional seasonality). Unlike the
zero-shot foundation models, it fits on every call — but a fit on a single price
series is fast. We forecast one step ahead, then turn the point estimate +
uncertainty interval into an UP/DOWN probability via a normal approximation:

    sigma   ≈ (yhat_upper - yhat_lower) / (2 · z)         z from interval_width
    prob_up  = P( N(yhat, sigma) > last_close )

No pre-training and nothing to download — ``pip install prophet`` is enough.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from statistics import NormalDist

import numpy as np
import pandas as pd

from engine.forecast_base import ForecastModel, ForecastResult
from engine.logger import get_logger

log = get_logger(__name__)

try:
    from prophet import Prophet

    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False

# First line of defense at import time.
for _noisy in ("prophet", "cmdstanpy"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


@contextmanager
def _quiet_stan():
    """
    Fully silence the prophet / cmdstanpy loggers for the duration of a fit.

    cmdstanpy re-sets its logger to INFO on every fit (emitting the
    "Chain [1] start/done processing" lines), so a one-time setLevel doesn't
    stick — we disable the loggers outright around each fit and restore after.
    """
    loggers = [logging.getLogger(n) for n in ("cmdstanpy", "prophet")]
    prev = [(lg.level, lg.disabled) for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.CRITICAL)
        lg.disabled = True
    try:
        yield
    finally:
        for lg, (level, disabled) in zip(loggers, prev):
            lg.setLevel(level)
            lg.disabled = disabled


# Minimum rows before a fit is worthwhile. The backtester already guarantees
# >= 20 training rows, so this mostly guards tiny custom calls.
MIN_ROWS = 20


class ProphetModel(ForecastModel):
    """Prophet next-day forecaster. Fits a fresh model on each window."""

    name = "Prophet"

    def __init__(
        self,
        interval_width: float = 0.80,
        weekly_seasonality: bool | str = False,
        yearly_seasonality: bool | str = "auto",
        daily_seasonality: bool | str = False,
        growth: str = "linear",
        macro_df=None,
    ):
        if not _PROPHET_AVAILABLE:
            raise RuntimeError(
                "Prophet is not installed. Install with: uv pip install prophet\n"
                "Or: uv pip install -e '.[forecast]'"
            )
        self.interval_width = interval_width
        self._kwargs = dict(
            interval_width=interval_width,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=daily_seasonality,
            growth=growth,
        )
        # z-score for the requested central interval, e.g. 0.80 -> ~1.2816.
        self._z = NormalDist().inv_cdf((1.0 + interval_width) / 2.0)
        # Optional macro regressors (R3.3). MUST be lag-1-aligned (align_macro):
        # the row for date d holds info known at d-1, so attaching macro[d] to
        # Prophet's forecast for d is leakage-safe. Indexed by date string.
        if macro_df is not None and not macro_df.empty:
            self._macro = macro_df.copy()
            self._macro.index = self._macro.index.astype(str)
            self._macro_cols = list(macro_df.columns)
            self.name = "Prophet + macro"
        else:
            self._macro = None
            self._macro_cols = []

    def fit_in_sample(self, df):
        """Prophet's in-sample fitted close series (yhat on the training dates).

        Falls back to the random-walk default if Prophet is unavailable, the
        series is too short, or the fit errors — so the residual hybrid never
        breaks. Aligned to ``df`` rows (length == len(df)).
        """
        n = len(df)
        if not _PROPHET_AVAILABLE or n < MIN_ROWS or "close" not in df.columns:
            return super().fit_in_sample(df)
        try:
            hist = pd.DataFrame(
                {
                    "ds": pd.to_datetime(df["date"]) if "date" in df.columns else range(n),
                    "y": pd.to_numeric(df["close"], errors="coerce"),
                }
            )
            fit_hist = hist.dropna()
            if len(fit_hist) < MIN_ROWS:
                return super().fit_in_sample(df)
            model = Prophet(**self._kwargs)
            with _quiet_stan():
                model.fit(fit_hist)
                # Predict on the original (full) ds so the result aligns row-for-row.
                yhat = model.predict(hist[["ds"]])["yhat"].to_numpy(dtype=float)
            if yhat.shape[0] != n or not np.isfinite(yhat).all():
                return super().fit_in_sample(df)
            return yhat
        except Exception:  # noqa: BLE001 — fall back to RW residuals, never crash
            return super().fit_in_sample(df)

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        if len(df) < MIN_ROWS or "close" not in df.columns:
            return None

        hist = pd.DataFrame(
            {
                "ds": pd.to_datetime(df["date"]) if "date" in df.columns else range(len(df)),
                "y": pd.to_numeric(df["close"], errors="coerce"),
            }
        ).dropna()
        if len(hist) < MIN_ROWS:
            return None

        last_close = float(hist["y"].iloc[-1])

        # Optional macro regressors (lag-1 aligned). Attach the macro row for
        # each training date; for the forecast row carry forward the last
        # in-window macro values (= macro at t, known at t-1 → leakage-safe).
        use_macro = self._macro is not None
        macro_cols: list[str] = []
        last_macro: dict[str, float] = {}
        if use_macro:
            key = hist["ds"].dt.strftime("%Y-%m-%d")
            ok = True
            for col in self._macro_cols:
                vals = key.map(self._macro[col]).to_numpy(dtype=float)
                if not np.isfinite(vals).all():
                    ok = False  # a gap in macro over the training window → skip macro
                    break
                hist[col] = vals
                last_macro[col] = float(vals[-1])
            macro_cols = self._macro_cols if ok else []
            use_macro = ok

        model = Prophet(**self._kwargs)
        for col in macro_cols:
            model.add_regressor(col)
        with _quiet_stan():
            model.fit(hist)
            future = model.make_future_dataframe(periods=horizon, freq="D")
            for col in macro_cols:
                # Map known training-date macro onto the future frame; the new
                # (forecast) rows get the carried-forward last value.
                key_f = future["ds"].dt.strftime("%Y-%m-%d")
                future[col] = key_f.map(self._macro[col]).astype(float)
                future[col] = future[col].fillna(last_macro[col])
            fc = model.predict(future).iloc[-1]

        yhat = float(fc["yhat"])
        lo = float(fc["yhat_lower"])
        hi = float(fc["yhat_upper"])

        sigma = max((hi - lo) / (2.0 * self._z), 1e-9)
        prob_up = 1.0 - NormalDist(mu=yhat, sigma=sigma).cdf(last_close)

        q_lo = (1.0 - self.interval_width) / 2.0
        q_hi = (1.0 + self.interval_width) / 2.0
        return ForecastResult(
            last_close=last_close,
            point=yhat,
            horizon=horizon,
            quantiles={q_lo: lo, 0.5: yhat, q_hi: hi},
            prob_up=prob_up,
            extra={"yhat_lower": lo, "yhat_upper": hi, "sigma": sigma},
        )
