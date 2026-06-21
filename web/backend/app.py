"""
app.py – FastAPI application for MarketPulse AI.

Run with:
    uv run uvicorn web.backend.app:app --reload --port 8000

API docs at: http://localhost:8000/docs
"""

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.routes import (
    analysis,
    backtest,
    data,
    docs,
    meta,
    oos,
    predict,
    settings,
    train,
)

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)


app = FastAPI(
    title="MarketPulse AI",
    description=(
        "Multi-asset prediction engine: k-NN, LinReg, LSTM, Prophet, "
        "Chronos-2, Kronos + naive baselines, with VADER/FinBERT sentiment, "
        "walk-forward backtests and an out-of-sample harness."
    ),
    version="0.1.0",
)

# CORS for React dev server (localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(meta.router)
app.include_router(data.router)
app.include_router(predict.router)
app.include_router(backtest.router)
app.include_router(oos.router)
app.include_router(train.router)
app.include_router(settings.router)
app.include_router(analysis.router)
app.include_router(docs.router)


@app.get("/")
def root():
    return {
        "name": "MarketPulse AI",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "meta": "/api/meta",
            "tickers": "/api/data/tickers",
            "predict": "/api/predict",
            "backtest": "/api/backtest",
            "oos": "/api/oos",
            "train": "/api/train/models",
            "settings": "/api/settings",
            "analysis": "/api/analysis/news-comparison",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}
