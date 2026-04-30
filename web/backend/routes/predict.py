"""
routes/predict.py – Prediction endpoints.
"""

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from config import ALL_TICKERS
from interface.api import PredictionConfig
from web.backend.routes.data import get_api
from web.backend.schemas import PredictionRow, PredictRequest, PredictResponse

router = APIRouter(prefix="/api/predict", tags=["predictions"])

CACHE_DIR = Path("predictions")


def _cache_key(ticker: str, model: str, period: str) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{today}_{ticker}_{model}_{period}"


def _load_cached(ticker: str, model: str, period: str) -> PredictionRow | None:
    """Load prediction from today's cache if exists."""
    key = _cache_key(ticker, model, period)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return PredictionRow(**data)
    return None


def _save_cache(row: PredictionRow):
    """Cache a prediction result."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(row.ticker, row.model, row.period)
    path = CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(row.model_dump(), indent=2))


ALL_MODELS = ["knn", "knn_enhanced", "linreg", "linreg_enhanced"]
TW_MODELS = {
    "knn_tw": ("knn", True, False),
    "knn_enhanced_tw": ("knn_enhanced", True, False),
    "linreg_tw": ("linreg", True, False),
    "linreg_enhanced_tw": ("linreg_enhanced", True, False),
}


@router.post("", response_model=PredictResponse)
def run_predictions(req: PredictRequest):
    """Run predictions for specified tickers and models."""
    api = get_api()

    tickers = [t.upper() for t in req.tickers] if req.tickers else ALL_TICKERS
    if req.refresh_data:
        api.refresh_tickers(tickers, verbose=False)

    # Determine which models to run
    if "all" in req.models:
        models_to_run = ALL_MODELS
    else:
        models_to_run = req.models

    predictions = []
    cached_count = 0

    for ticker in tickers:
        for model in models_to_run:
            # Check cache first
            cached = _load_cached(ticker, model, req.period)
            if cached and req.refresh_data is False:
                predictions.append(cached)
                cached_count += 1
                continue

            # Run prediction
            for tw in [False, True]:
                try:
                    cfg = PredictionConfig(
                        ticker=ticker,
                        period=req.period,
                        model_type=model,
                        use_time_weights=tw,
                        include_news=req.include_news,
                    )
                    result = api.get_prediction(cfg)

                    model_label = f"{model}"
                    if tw:
                        model_label += " TW"
                    if req.include_news and result.sentiment_score != 0:
                        model_label += " + News"

                    row = PredictionRow(
                        ticker=ticker,
                        model=model_label,
                        period=req.period,
                        prediction=result.prediction,
                        confidence=float(result.confidence.rstrip("%")) / 100,
                        last_price=result.last_price,
                        sentiment=result.sentiment,
                        sentiment_score=result.sentiment_score,
                        headlines=result.headlines,
                        timestamp=result.timestamp,
                    )
                    predictions.append(row)
                    _save_cache(row)

                except Exception:
                    continue

    return PredictResponse(
        predictions=predictions,
        cached=cached_count > 0,
    )


@router.get("/consensus/{ticker}")
def get_consensus(ticker: str, period: str = "1y"):
    """Get consensus across all models for a ticker."""
    api = get_api()
    ticker = ticker.upper()

    up_count, down_count = 0, 0
    model_votes = []

    for model in ALL_MODELS:
        for tw in [False, True]:
            try:
                cfg = PredictionConfig(
                    ticker=ticker,
                    period=period,
                    model_type=model,
                    use_time_weights=tw,
                    include_news=False,
                )
                result = api.get_prediction(cfg)
                label = f"{model}{' TW' if tw else ''}"
                model_votes.append(
                    {
                        "model": label,
                        "prediction": result.prediction,
                        "confidence": float(result.confidence.rstrip("%")) / 100,
                    }
                )
                if result.prediction == "UP":
                    up_count += 1
                else:
                    down_count += 1
            except Exception:
                continue

    total = up_count + down_count
    return {
        "ticker": ticker,
        "period": period,
        "consensus": "UP" if up_count > down_count else "DOWN",
        "up_votes": up_count,
        "down_votes": down_count,
        "total": total,
        "agreement": max(up_count, down_count) / total if total > 0 else 0,
        "votes": model_votes,
    }
