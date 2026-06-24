"""
naive_forecasters.py – Trivial point-forecasters (the regression baselines).

These are to the *regression* track what ``baseline_models.py`` is to the
directional one: deliberately dumb forecasters that a real model must beat.
The most important is **RandomWalk** — it is the reference for Theil's U2 and
(in spirit) MASE, so "beating the random walk" is the bar every forecaster has
to clear before any result is interesting.

Each subclasses ``ForecastModel`` and implements ``_raw_forecast``, so it
plugs into the forecast harness exactly like Prophet / Chronos / Kronos and
gets the value→direction adapter for free. They need only a ``close`` column
(``SeasonalNaive`` also benefits from regular spacing but doesn't require
dates).

Forecasters here, distinct from the trading ``PreviousDay`` baseline:

* **RandomWalk** — ``P̂_{t+1} = P_t``. The no-change forecast; the U2/MASE
  reference. (Conceptually the same rule as the directional Previous-Day
  baseline, but it lives here as an explicit *value* forecaster so the
  regression harness can use it as the reference series.)
* **RandomWalkDrift** — ``P̂_{t+1} = P_t + μ`` where μ is the mean one-step
  change over the context. Captures a constant trend.
* **SeasonalNaive** — ``P̂_{t+1} = P_{t+1-m}`` (the value m steps back). With
  ``m`` set to a weekly period this is the seasonal reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engine.forecast_base import ForecastModel, ForecastResult

if TYPE_CHECKING:
    import pandas as pd


def _closes(df: pd.DataFrame) -> np.ndarray:
    """Finite close prices as a 1-D float array (chronological order assumed)."""
    arr = np.asarray(df["close"], dtype=float).ravel()
    return arr[np.isfinite(arr)]


class RandomWalkForecaster(ForecastModel):
    """``P̂_{t+1} = P_t``. The reference forecaster for U2 / MASE.

    A symmetric forecast (point == last_close) carries no directional view, so
    we hand the base class a tiny epsilon distribution; the harness reads
    ``.point`` for regression and ignores the direction for these.
    """

    name = "Random Walk"

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        closes = _closes(df)
        if closes.size < 1:
            return None
        last = float(closes[-1])
        # No-change forecast for any horizon.
        return ForecastResult(last_close=last, point=last, horizon=horizon)


class RandomWalkDriftForecaster(ForecastModel):
    """``P̂_{t+h} = P_t + h·μ``, μ = mean one-step change over the context."""

    name = "Random Walk + Drift"

    def __init__(self, context: int | None = None) -> None:
        # None = use the whole series; otherwise the most-recent `context` bars.
        self.context = context

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        closes = _closes(df)
        if closes.size < 2:
            return None
        ctx = closes if self.context is None else closes[-self.context :]
        if ctx.size < 2:
            ctx = closes
        drift = float(np.mean(np.diff(ctx)))
        last = float(closes[-1])
        return ForecastResult(last_close=last, point=last + horizon * drift, horizon=horizon)


class SeasonalNaiveForecaster(ForecastModel):
    """``P̂_{t+1} = P_{t+1-m}`` — the value one season (m steps) back.

    Falls back to a plain random walk when there isn't a full season of
    history yet, so it never errors on short windows.
    """

    def __init__(self, m: int = 5) -> None:
        if m < 1:
            raise ValueError("season m must be >= 1")
        self.m = m
        self.name = f"Seasonal Naive ({m})"

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        closes = _closes(df)
        if closes.size < 1:
            return None
        last = float(closes[-1])
        # The forecast for step t+horizon repeats the value m steps before it.
        back = self.m - horizon + 1
        if back >= 1 and closes.size >= back:
            point = float(closes[-back])
        else:
            point = last  # not enough history → random walk
        return ForecastResult(last_close=last, point=point, horizon=horizon)


def default_naive_forecasters(season: int = 5) -> list[tuple[ForecastModel, str]]:
    """The standard regression-baseline set as (model, label) pairs.

    ``season`` defaults to 5 (a trading week on daily bars).
    """
    return [
        (RandomWalkForecaster(), "Random Walk"),
        (RandomWalkDriftForecaster(), "Random Walk + Drift"),
        (SeasonalNaiveForecaster(m=season), f"Seasonal Naive ({season})"),
    ]
