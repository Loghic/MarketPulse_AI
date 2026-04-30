"""
app.py – FastAPI application for MarketPulse AI.

Run with:
    uv run uvicorn web.backend.app:app --reload --port 8000

API docs at: http://localhost:8000/docs
"""

import sys

sys.path.insert(0, ".")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web.backend.routes import analysis, backtest, data, predict, settings, train

app = FastAPI(
    title="MarketPulse AI",
    description="Stock/crypto prediction engine with k-NN, LinReg, LSTM + VADER sentiment",
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
app.include_router(data.router)
app.include_router(predict.router)
app.include_router(backtest.router)
app.include_router(train.router)
app.include_router(settings.router)
app.include_router(analysis.router)


@app.get("/")
def root():
    return {
        "name": "MarketPulse AI",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "tickers": "/api/data/tickers",
            "predict": "/api/predict",
            "backtest": "/api/backtest",
            "train": "/api/train/models",
            "settings": "/api/settings",
            "analysis": "/api/analysis/news-comparison",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}
