"""
routes/predict.py – Unified prediction endpoint.

POST /api/predict/run — accepts list of {model, period, news} items.
Results include both predictions table and consensus summary.
Cache: predictions/{ticker}/{date}.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from config import ALL_PERIODS
from interface.api import PredictionConfig, StockAppAPI
from web.backend.routes.data import get_api

router = APIRouter(prefix="/api/predict", tags=["predictions"])
log = logging.getLogger("marketpulse.web")

CACHE_DIR = Path("predictions")

# Always-available classifiers (no optional dependency).
_BASE_VARIANTS = [
    {"type": "knn", "tw": False, "label": "k-NN"},
    {"type": "knn", "tw": True, "label": "k-NN (TW)"},
    {"type": "knn_enhanced", "tw": False, "label": "k-NN Enhanced"},
    {"type": "knn_enhanced", "tw": True, "label": "k-NN Enhanced (TW)"},
    {"type": "linreg", "tw": False, "label": "LinReg"},
    {"type": "linreg", "tw": True, "label": "LinReg (TW)"},
    {"type": "linreg_enhanced", "tw": False, "label": "LinReg Enhanced"},
    {"type": "linreg_enhanced", "tw": True, "label": "LinReg Enhanced (TW)"},
]
# Dependency-gated variants: only offered when the model is actually loadable
# (LSTM needs torch; the forecasting models need the [forecast] extra / Kronos
# clone). _available_variants() filters these against the live API flags.
_LSTM_VARIANT = {"type": "lstm", "tw": False, "label": "LSTM"}
_FORECAST_VARIANTS = [
    {"type": "prophet", "tw": False, "label": "Prophet"},
    {"type": "chronos", "tw": False, "label": "Chronos-2"},
    {"type": "kronos", "tw": False, "label": "Kronos"},
]
# Full catalogue (for label lookup); availability is decided per-request.
MODEL_VARIANTS = [*_BASE_VARIANTS, _LSTM_VARIANT, *_FORECAST_VARIANTS]
VARIANT_BY_LABEL = {v["label"]: v for v in MODEL_VARIANTS}


def _available_variants(api: StockAppAPI) -> list[dict]:
    """The variants the running install can actually serve, in display order."""
    variants = list(_BASE_VARIANTS)
    if api.lstm_available:
        variants.append(_LSTM_VARIANT)
    variants.extend(v for v in _FORECAST_VARIANTS if api.forecast_available(str(v["type"])))
    return variants


def _cache_path(ticker: str, date: str) -> Path:
    return CACHE_DIR / ticker / f"{date}.json"


def _load_cache(ticker: str, date: str) -> list[dict] | None:
    p = _cache_path(ticker, date)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _save_cache(ticker: str, date: str, rows: list[dict]) -> None:
    p = _cache_path(ticker, date)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2))


def _next_trading_day() -> str:
    d = datetime.now() + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _run_one(api: StockAppAPI, ticker: str, period: str, variant: dict, news: bool) -> dict | None:
    try:
        cfg = PredictionConfig(
            ticker=ticker,
            period=period,
            model_type=variant["type"],
            use_time_weights=variant["tw"],
            include_news=news,
        )
        r = api.get_prediction(cfg)
        label = variant["label"]
        if news and r.sentiment_score != 0:
            label += " + News"
        return {
            "ticker": ticker,
            "model": label,
            "period": period,
            "prediction": r.prediction,
            "confidence": float(r.confidence.rstrip("%")) / 100,
            "last_price": r.last_price,
            "sentiment": r.sentiment,
            "sentiment_score": r.sentiment_score,
            "headlines": r.headlines,
            "timestamp": r.timestamp,
        }
    except Exception as e:
        log.debug(f"  {ticker}/{variant['label']}/{period}: {e}")
        return None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("/info")
def predict_info():
    api = get_api()
    return {
        "variants": [v["label"] for v in _available_variants(api)],
        "periods": ALL_PERIODS,
        "next_trading_day": _next_trading_day(),
    }


class RunItem(BaseModel):
    model: str  # variant label
    period: str  # e.g. "1y"
    news: bool = False


class RunRequest(BaseModel):
    ticker: str
    items: list[RunItem]
    refresh_data: bool = False


@router.post("/run")
def run_predictions(req: RunRequest):
    """
    Unified prediction endpoint.
    Accepts specific model+period+news combos.
    Returns predictions + consensus.
    """
    api = get_api()
    ticker = req.ticker.upper()
    today = datetime.now().strftime("%Y-%m-%d")

    if req.refresh_data:
        api.refresh_tickers([ticker], verbose=False)

    log.info(f"Running {len(req.items)} prediction(s) for {ticker}")

    results: list[dict] = []
    for item in req.items:
        variant = VARIANT_BY_LABEL.get(item.model)
        if not variant:
            results.append(
                {
                    "ticker": ticker,
                    "model": item.model,
                    "period": item.period,
                    "prediction": "N/A",
                    "confidence": 0,
                    "last_price": 0,
                    "sentiment": "",
                    "sentiment_score": 0,
                    "headlines": [],
                    "timestamp": today,
                    "error": f"Unknown: {item.model}",
                }
            )
            continue

        row = _run_one(api, ticker, item.period, variant, news=item.news)
        if row:
            results.append(row)
        else:
            results.append(
                {
                    "ticker": ticker,
                    "model": f"{variant['label']} (failed)",
                    "period": item.period,
                    "prediction": "N/A",
                    "confidence": 0,
                    "last_price": 0,
                    "sentiment": "",
                    "sentiment_score": 0,
                    "headlines": [],
                    "timestamp": today,
                }
            )

    # Consensus from valid results
    valid = [r for r in results if r["prediction"] in ("UP", "DOWN")]
    up = sum(1 for r in valid if r["prediction"] == "UP")
    down = len(valid) - up
    total = up + down

    # Cache
    existing = _load_cache(ticker, today) or []
    keys = {f"{r.get('model')}|{r.get('period')}" for r in existing}
    merged = list(existing)
    for r in results:
        k = f"{r['model']}|{r['period']}"
        if k not in keys:
            merged.append(r)
            keys.add(k)
    _save_cache(ticker, today, merged)

    return {
        "predictions": results,
        "consensus": {
            "direction": "UP" if up > down else ("DOWN" if down > up else "SPLIT"),
            "up": up,
            "down": down,
            "total": total,
            "agreement": max(up, down) / total if total > 0 else 0,
        },
    }


@router.get("/cached")
def list_cached():
    if not CACHE_DIR.exists():
        return []
    out = []
    for td in sorted(CACHE_DIR.iterdir()):
        if not td.is_dir():
            continue
        for f in sorted(td.glob("*.json"), reverse=True):
            data = json.loads(f.read_text())
            out.append({"ticker": td.name, "date": f.stem, "count": len(data)})
    return out


@router.get("/cached/{ticker}")
def cached_for_ticker(ticker: str):
    """
    Latest cached prediction set for one ticker.

    Returns the most recent (by file date) cache entry so the frontend can
    re-render a previous prediction on tab switch without re-running every
    model. Includes:
        * ``date`` – the YYYY-MM-DD bucket the cache was written under
        * ``cached_at`` – ISO timestamp of when the JSON file was last
          modified (used to show "cached X minutes ago" on the UI)
        * ``predictions`` – the same row shape as POST /run returns
        * ``consensus`` – recomputed from the cached rows

    Empty payload (``predictions: []``) when no cache exists yet — the
    frontend should hide its "cached" badge in that case.
    """
    ticker = ticker.upper()
    td = CACHE_DIR / ticker
    if not td.exists():
        return {"ticker": ticker, "predictions": [], "consensus": None}

    files = sorted(td.glob("*.json"), reverse=True)
    if not files:
        return {"ticker": ticker, "predictions": [], "consensus": None}

    latest = files[0]
    try:
        rows = json.loads(latest.read_text())
    except Exception:
        return {"ticker": ticker, "predictions": [], "consensus": None}

    valid = [r for r in rows if r.get("prediction") in ("UP", "DOWN")]
    up = sum(1 for r in valid if r["prediction"] == "UP")
    down = len(valid) - up
    total = up + down

    cached_at = datetime.fromtimestamp(latest.stat().st_mtime).isoformat(timespec="seconds")

    return {
        "ticker": ticker,
        "date": latest.stem,
        "cached_at": cached_at,
        "predictions": rows,
        "consensus": {
            "direction": "UP" if up > down else ("DOWN" if down > up else "SPLIT"),
            "up": up,
            "down": down,
            "total": total,
            "agreement": max(up, down) / total if total > 0 else 0,
        },
    }


@router.post("/historical")
def predict_historical(ticker: str, date: str, period: str = "1y"):
    api = get_api()
    ticker = ticker.upper()

    cached = _load_cache(ticker, date)
    if cached:
        filtered = [r for r in cached if r.get("period") == period]
        if filtered:
            return {"predictions": filtered, "cached": True, "date": date}

    df = api.db.get_prices(ticker)
    if df.empty:
        return {"predictions": [], "date": date, "error": "No data"}

    df["_date"] = pd.to_datetime(df["date"]).dt.date
    cutoff = pd.to_datetime(date).date()
    df_hist = df[df["_date"] <= cutoff].drop(columns=["_date"])
    if len(df_hist) < 30:
        return {"predictions": [], "date": date, "error": "Not enough data"}

    results: list[dict] = []
    for v in MODEL_VARIANTS:
        try:
            model = api._get_model(str(v["type"]), ticker, period)
            pred, conf = model.predict(df_hist, use_time_weights=v["tw"], sentiment_score=0.0)
            results.append(
                {
                    "ticker": ticker,
                    "model": v["label"],
                    "period": period,
                    "prediction": pred,
                    "confidence": conf,
                    "date": date,
                }
            )
        except Exception:
            continue

    if results:
        _save_cache(ticker, date, results)
    return {"predictions": results, "cached": False, "date": date}
