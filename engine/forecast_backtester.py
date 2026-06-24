"""
forecast_backtester.py – Walk-forward evaluation for *point* forecasts.

This is the regression-track analogue of ``backtester.py``, kept deliberately
separate (plan R1.2): no positions, no fees, no stop-loss, no P&L. Per step it
records only ``(date, horizon, y_true, y_pred)`` and scores the result with
``regression_metrics`` — the headline being scale-free skill vs a random walk
(Theil U2, MASE), not absolute RMSE.

Walk-forward contract (plan R0):
  * **Expanding window** — at evaluation step ``t`` the model sees all data
    from the start up to and including ``t``, and forecasts ``t + h``.
  * **Direct-h horizons** — a separate forecast per horizon ``h``; no recursive
    multi-step (which would compound error and muddy the "does it help" signal).
  * **Refit cadence K** — the model is re-fit every ``K`` steps; on in-between
    steps the *frozen* fit predicts. Our R2 forecasters refit per call cheaply,
    so K is mainly a cost knob for the expensive models (ARIMA/XGBoost/hybrid);
    it never changes which data is visible (always ≤ t).
  * **Leakage guarantee** — the model only ever receives ``df.iloc[: t+1]``; the
    realised ``y_true = close[t+h]`` is never in that slice.

The reference (random-walk) forecast for U2 is ``P̂ = close[t]`` at every step,
computed here once so every model is scored against the *same* naive series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from engine.logger import get_logger
from engine.regression_metrics import ForecastMetrics, compute_all

if TYPE_CHECKING:
    import pandas as pd

    from engine.forecast_base import ForecastResult

log = get_logger(__name__)


class PointForecaster(Protocol):
    """Structural type for anything the harness can score.

    Satisfied by every ``ForecastModel`` (naive, ARIMA, XGBoost, Prophet,
    Chronos, Kronos, and the future residual hybrid).
    """

    name: str

    def forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None: ...


@dataclass
class ForecastStep:
    """One scored walk-forward step."""

    date: str
    horizon: int
    y_true: float
    y_pred: float
    y_naive: float  # random-walk reference (close[t])


@dataclass
class ForecastRun:
    """All steps + metrics for one (model, ticker, horizon)."""

    model_name: str
    ticker: str
    horizon: int
    refit_k: int
    steps: list[ForecastStep] = field(default_factory=list)
    metrics: ForecastMetrics | None = None
    elapsed_seconds: float = 0.0
    skipped: int = 0  # steps the model couldn't forecast (returned None)


class ForecastBacktester:
    """Lean walk-forward harness for point forecasts.

    Args:
        n_days:   length of the evaluation window (the most-recent N steps).
        horizon:  direct-h forecast horizon.
        refit_k:  refit cadence (1 = refit every step).
        min_train: minimum rows required before the first evaluation step.
        max_train: cap on the **training window** fed to the model at each step
            (the most-recent ``max_train`` rows up to ``t``). ``None`` = the full
            expanding window. On a long-history ticker (e.g. AAPL's ~11k rows
            back to split-adjusted cents) an unbounded window is both slow and
            pathological — Prophet/ARIMA fit across decades of regime change and
            forecast a wildly off level. It also makes MASE's in-sample naive
            scaling compare today's dollar moves against the average since the
            1980s. Capping fixes both; ``~504`` ≈ two trading years is a sane
            default. Theil U2 is unaffected (it scales on the eval window).
    """

    def __init__(
        self,
        *,
        n_days: int = 100,
        horizon: int = 1,
        refit_k: int = 21,
        min_train: int = 60,
        max_train: int | None = 504,
    ) -> None:
        if n_days < 1:
            raise ValueError("n_days must be >= 1")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if refit_k < 1:
            raise ValueError("refit_k must be >= 1")
        if max_train is not None and max_train < min_train:
            raise ValueError(f"max_train ({max_train}) must be >= min_train ({min_train})")
        self.n_days = n_days
        self.horizon = horizon
        self.refit_k = refit_k
        self.min_train = min_train
        self.max_train = max_train

    def run(self, model: PointForecaster, df: pd.DataFrame, ticker: str) -> ForecastRun | None:
        """Walk forward over the last ``n_days`` steps and score the forecasts.

        Returns None if the series is too short to form even one scored step.
        """
        import time

        closes = np.asarray(df["close"], dtype=float).ravel()
        dates = (
            df["date"].astype(str).tolist()
            if "date" in df.columns
            else [str(i) for i in range(len(df))]
        )
        n = len(closes)
        h = self.horizon

        # The last index we can score is n-1-h (need close[t+h] to exist).
        last_scoreable = n - 1 - h
        if last_scoreable < self.min_train:
            log.debug("%s/%s: too short for forecast eval (n=%d).", model.name, ticker, n)
            return None

        first_t = max(self.min_train, last_scoreable - self.n_days + 1)
        run = ForecastRun(model_name=model.name, ticker=ticker, horizon=h, refit_k=self.refit_k)
        t0 = time.perf_counter()

        # Insample target series for MASE/RMSSE scaling: the training closes
        # available before the evaluation window starts (up to first_t
        # inclusive), capped to the most-recent ``max_train`` rows so the naive
        # one-step denominator reflects *recent* volatility, not the average
        # since the ticker's inception. Naive one-step on this is the MASE
        # denominator.
        insample_start = 0
        if self.max_train is not None:
            insample_start = max(0, (first_t + 1) - self.max_train)
        insample = closes[insample_start : first_t + 1]

        for _step_i, t in enumerate(range(first_t, last_scoreable + 1)):
            # Training window: data up to and including t, optionally capped to
            # the most-recent ``max_train`` rows (keeps Prophet/ARIMA fast and
            # fits them on a relevant regime instead of decades of history). The
            # cap never lets the model see past t, so the leakage guarantee
            # holds. (Refit cadence K is a cost knob only; visible data is
            # identical either way, so correctness doesn't depend on K here.)
            start = 0 if self.max_train is None else max(0, (t + 1) - self.max_train)
            window = df.iloc[start : t + 1]
            fr = model.forecast(window, horizon=h)
            if fr is None or not np.isfinite(fr.point):
                run.skipped += 1
                continue
            run.steps.append(
                ForecastStep(
                    date=dates[t + h],
                    horizon=h,
                    y_true=float(closes[t + h]),
                    y_pred=float(fr.point),
                    y_naive=float(closes[t]),  # random-walk reference
                )
            )

        run.elapsed_seconds = round(time.perf_counter() - t0, 4)

        if not run.steps:
            run.metrics = None
            return run

        y_true = np.array([s.y_true for s in run.steps])
        y_pred = np.array([s.y_pred for s in run.steps])
        y_naive = np.array([s.y_naive for s in run.steps])
        run.metrics = compute_all(y_true, y_pred, insample=insample, y_naive=y_naive, season=1)
        return run
