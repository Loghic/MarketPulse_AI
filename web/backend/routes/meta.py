"""
routes/meta.py – Config-driven options for the frontend.

Single source of truth so the React pickers never hardcode the model list,
ticker set, asset classes, or benchmark options. Everything derives from
``config`` (the asset registry + model-family labels) and the live
``StockAppAPI`` availability flags, so adding a model or asset class in config
surfaces it in the UI automatically — gated by whether its optional dependency
is actually installed.
"""

from __future__ import annotations

from fastapi import APIRouter

from config import (
    ALL_BENCHMARKS,
    ALL_PERIODS,
    ASSET_CLASSES,
    CONFIDENCE_SWEEP,
    DEFAULT_BACKTEST_DAYS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_PERIOD,
    DEFAULT_SENTIMENT_METHOD,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TRADING_FEE_PCT,
    MODEL_FAMILY_LABELS,
    SL_SWEEP,
)
from web.backend.routes.data import get_api
from web.backend.schemas import AssetClassInfo, MetaResponse, ModelFamily

router = APIRouter(prefix="/api/meta", tags=["meta"])

# Which families are usable in the per-ticker Predict flow (not just backtests).
# k-NN / LinReg / LSTM expose a same-day prediction; the forecasting models are
# wired into backtests only, and the baselines are reference floors, not signals.
_PREDICT_FAMILIES = {"knn", "linreg", "lstm"}
# Families that depend on an optional extra / external clone.
_FORECAST_FAMILIES = {"prophet", "chronos", "kronos"}


@router.get("", response_model=MetaResponse)
def get_meta() -> MetaResponse:
    """Return every option the frontend needs to build its pickers."""
    api = get_api()

    families: list[ModelFamily] = []
    for key, label in MODEL_FAMILY_LABELS.items():
        if key == "lstm":
            available = bool(api.lstm_available)
            note = "" if available else "PyTorch not installed"
            predict = available
        elif key in _FORECAST_FAMILIES:
            available = bool(api.forecast_available(key))
            note = "" if available else "optional forecast dependency not installed"
            predict = False  # backtest-only
        elif key == "baseline":
            available = True
            note = "reference baselines (backtest-only)"
            predict = False
        else:  # knn, linreg — always available
            available = True
            note = ""
            predict = key in _PREDICT_FAMILIES
        families.append(
            ModelFamily(key=key, label=label, available=available, predict=predict, note=note)
        )

    asset_classes = [
        AssetClassInfo(
            key=ac.key,
            label=ac.label,
            cli_flag=ac.cli_flag,
            tickers=list(ac.tickers),
            benchmarks=list(ac.benchmarks),
        )
        for ac in ASSET_CLASSES
    ]

    return MetaResponse(
        model_families=families,
        asset_classes=asset_classes,
        benchmarks=list(ALL_BENCHMARKS),
        periods=list(ALL_PERIODS),
        sentiment_methods=["vader", "finbert", "naive"],
        sl_sweep=list(SL_SWEEP),
        confidence_sweep=list(CONFIDENCE_SWEEP),
        defaults={
            "fee_pct": DEFAULT_TRADING_FEE_PCT,
            "stop_loss_pct": DEFAULT_STOP_LOSS_PCT,
            "min_confidence": DEFAULT_MIN_CONFIDENCE,
            "backtest_days": DEFAULT_BACKTEST_DAYS,
            "period": DEFAULT_PERIOD,
            "sentiment_method": DEFAULT_SENTIMENT_METHOD,
        },
    )
