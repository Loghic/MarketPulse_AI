"""
regression_metrics.py – Point-forecast accuracy metrics.

The question this module answers
--------------------------------

The *directional* track (``calibration.py`` / ``significance.py``) scores
UP/DOWN calls. The *regression* track scores a predicted **price level**
against the realised one, so it needs a completely different metric set.

The trap this module is built to avoid
---------------------------------------

On a persistent price level, the no-change **random-walk** forecast
(``P̂_{t+1} = P_t``) already lands very close most days, so its RMSE / MAE /
MAPE are tiny. That makes **absolute** error metrics flatter *every* model —
a model can post an impressive-looking RMSE while being no better than "assume
tomorrow equals today". Absolute numbers are therefore nearly uninformative on
financial levels.

The headline metrics here are the **scale-free, skill-relative** ones, all
measured against the random walk on the *same* window:

* **MASE** – mean absolute error divided by the in-sample one-step naive MAE.
  MASE < 1 ⇔ beats the in-sample naive forecast.
* **RMSSE** – the squared-error analogue (the M5 competition metric).
* **Theil's U2** – ``RMSE(model) / RMSE(random-walk)`` on the test window.
  **U2 < 1 ⇔ beats the random walk.** This is the one to report.

The absolute metrics (``rmse``, ``mae``, ``mape``, ``smape``) are still
computed — they're familiar and fine for *relative* comparison between models
on one series — but a paper that reports them without a skill-relative metric
is not interpretable. Report U2 / MASE first.

Everything here is **pure** numpy/stdlib: arrays in, numbers/dataclasses out.
No I/O, no globals. Console rendering and persistence live in the harness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

# A tiny floor to keep ratio metrics finite when a denominator is ~0 (a dead /
# constant series). We return NaN rather than inf in those degenerate cases so
# downstream aggregation (median across tickers) isn't poisoned by an inf.
_EPS = 1e-12


def _clean_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to 1-D float arrays and drop any row where either side is non-finite."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"length mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


# ----------------------------------------------------------------------
# Absolute error metrics
# ----------------------------------------------------------------------


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error, as a fraction (0.05 = 5%).

    Rows where ``y_true`` is ~0 are dropped — percentage error is undefined
    there and would blow up. On price levels this effectively never triggers.
    """
    yt, yp = _clean_pair(y_true, y_pred)
    nz = np.abs(yt) > _EPS
    if not np.any(nz):
        return float("nan")
    return float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])))


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric MAPE in [0, 2] as a fraction. Bounded, so robust to small ``y_true``.

    Uses the common ``|e| / ((|y| + |ŷ|)/2)`` form; rows where both are ~0 are
    dropped (the forecast is trivially perfect there and 0/0 is undefined).
    """
    yt, yp = _clean_pair(y_true, y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    nz = denom > _EPS
    if not np.any(nz):
        return float("nan")
    return float(np.mean(np.abs(yt[nz] - yp[nz]) / denom[nz]))


# ----------------------------------------------------------------------
# Scale-free, skill-relative metrics (the headline numbers)
# ----------------------------------------------------------------------


def _naive_insample_mae(insample: ArrayLike, season: int = 1) -> float:
    """In-sample MAE of the season-step naive forecast — the MASE denominator.

    For ``season=1`` this is the mean absolute one-step change of the training
    series (``mean |y_t - y_{t-1}|``), i.e. the error a random-walk forecaster
    would have made *in sample*. This is the canonical MASE scaling (Hyndman &
    Koehler 2006) and is what makes MASE comparable across series of different
    price scales.
    """
    arr = np.asarray(insample, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size <= season:
        return float("nan")
    return float(np.mean(np.abs(arr[season:] - arr[:-season])))


def _naive_insample_mse(insample: ArrayLike, season: int = 1) -> float:
    """In-sample MSE of the season-step naive forecast — the RMSSE denominator."""
    arr = np.asarray(insample, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size <= season:
        return float("nan")
    return float(np.mean((arr[season:] - arr[:-season]) ** 2))


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    insample: ArrayLike,
    season: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    ``MASE = MAE(model on test) / MAE(naive on the in-sample train series)``.
    < 1 means the model beats the in-sample naive forecast. ``insample`` is the
    *training* target series (not the test window); ``season`` is the naive
    step (1 = random walk, 5 = weekly for daily bars).
    """
    test_mae = mae(y_true, y_pred)
    denom = _naive_insample_mae(insample, season=season)
    if not np.isfinite(test_mae) or not np.isfinite(denom) or denom < _EPS:
        return float("nan")
    return test_mae / denom


def rmsse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    insample: ArrayLike,
    season: int = 1,
) -> float:
    """Root Mean Squared Scaled Error (the M5 metric). < 1 beats in-sample naive."""
    yt, yp = _clean_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    test_mse = float(np.mean((yt - yp) ** 2))
    denom = _naive_insample_mse(insample, season=season)
    if not np.isfinite(denom) or denom < _EPS:
        return float("nan")
    return float(np.sqrt(test_mse / denom))


def theil_u2(y_true: ArrayLike, y_pred: ArrayLike, y_naive: ArrayLike) -> float:
    """Theil's U2 = RMSE(model) / RMSE(naive), both on the *test* window.

    ``y_naive`` is the random-walk (or other reference) forecast aligned to the
    same test timestamps as ``y_pred``. **U2 < 1 ⇔ the model beats the naive
    reference.** By construction the naive forecaster scores exactly 1.0.
    """
    model_rmse = rmse(y_true, y_pred)
    naive_rmse = rmse(y_true, y_naive)
    if not np.isfinite(model_rmse) or not np.isfinite(naive_rmse) or naive_rmse < _EPS:
        return float("nan")
    return model_rmse / naive_rmse


# ----------------------------------------------------------------------
# One-shot bundle
# ----------------------------------------------------------------------


@dataclass
class ForecastMetrics:
    """All point-forecast metrics for one (model, ticker, horizon) series.

    ``n`` is the count of scored steps. The scale-free trio (mase / rmsse /
    theil_u2) is the headline; the absolute four are for within-series
    comparison only.
    """

    n: int
    rmse: float
    mae: float
    mape: float
    smape: float
    mase: float
    rmsse: float
    theil_u2: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_all(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    insample: ArrayLike,
    y_naive: ArrayLike,
    season: int = 1,
) -> ForecastMetrics:
    """Compute every metric in one pass.

    Args:
        y_true:   realised test values.
        y_pred:   model's forecasts, aligned to ``y_true``.
        insample: the training target series (MASE/RMSSE scaling denominator).
        y_naive:  the naive (random-walk) forecast on the *test* window, for U2.
        season:   naive step for the scaled metrics (1 = random walk).
    """
    yt, _ = _clean_pair(y_true, y_pred)
    return ForecastMetrics(
        n=int(yt.size),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        mape=mape(y_true, y_pred),
        smape=smape(y_true, y_pred),
        mase=mase(y_true, y_pred, insample, season=season),
        rmsse=rmsse(y_true, y_pred, insample, season=season),
        theil_u2=theil_u2(y_true, y_pred, y_naive),
    )
