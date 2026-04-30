"""
routes/data.py – Ticker data and refresh endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

import sys
sys.path.insert(0, ".")

from interface.api import StockAppAPI
from config import STOCKS, CRYPTO, ALL_TICKERS, STOCK_BENCHMARKS, CRYPTO_BENCHMARKS
from web.backend.schemas import (
    TickerInfo, TickerDataResponse, OHLCVRow,
    RefreshRequest, RefreshStatus,
)

router = APIRouter(prefix="/api/data", tags=["data"])

# Shared API instance (created once, reused)
_api: Optional[StockAppAPI] = None


def get_api() -> StockAppAPI:
    global _api
    if _api is None:
        _api = StockAppAPI()
    return _api


@router.get("/tickers", response_model=list[TickerInfo])
def list_tickers():
    """List all configured tickers with metadata."""
    api = get_api()
    result = []
    for ticker in ALL_TICKERS:
        asset_type = "crypto" if "-USD" in ticker else "stock"
        df = api.db.get_prices(ticker)
        last_date = str(df["date"].iloc[-1]) if not df.empty else None
        result.append(TickerInfo(
            ticker=ticker,
            asset_type=asset_type,
            rows=len(df),
            last_date=last_date,
        ))
    return result


@router.get("/ticker/{ticker}", response_model=TickerDataResponse)
def get_ticker_data(ticker: str, period: str = "1y", limit: int = 500):
    """Get OHLCV data for a single ticker."""
    api = get_api()
    ticker = ticker.upper()

    df = api.db.get_prices(ticker)
    if df.empty:
        raise HTTPException(404, f"No data for {ticker}. Run refresh first.")

    # Apply period filter
    from engine.utils import period_to_start_date
    start = period_to_start_date(period)
    import pandas as pd
    df["_date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["_date"] >= start].drop(columns=["_date"])

    # Sort descending, limit rows
    df = df.sort_values("date", ascending=False).head(limit)

    rows = [
        OHLCVRow(
            date=str(r["date"]),
            open=round(float(r["open"]), 4),
            high=round(float(r["high"]), 4),
            low=round(float(r["low"]), 4),
            close=round(float(r["close"]), 4),
            volume=int(r["volume"]),
        )
        for _, r in df.iterrows()
    ]

    return TickerDataResponse(ticker=ticker, rows=len(rows), data=rows)


@router.post("/refresh", response_model=list[RefreshStatus])
def refresh_data(req: RefreshRequest):
    """Download latest prices and news for specified tickers."""
    api = get_api()
    tickers = [t.upper() for t in req.tickers] if req.tickers else ALL_TICKERS

    # Include benchmarks
    all_to_refresh = list(set(tickers + STOCK_BENCHMARKS + CRYPTO_BENCHMARKS))

    results = []
    for ticker in all_to_refresh:
        try:
            df = api.get_data(ticker, period="max")
            score, headlines = api._process_news_with_db(ticker)
            last_date = str(df["date"].iloc[-1]) if not df.empty else "n/a"
            results.append(RefreshStatus(
                ticker=ticker,
                rows=len(df),
                last_date=last_date,
                news_count=len(headlines),
            ))
        except Exception as e:
            results.append(RefreshStatus(
                ticker=ticker, rows=0, last_date="error", news_count=0,
            ))

    return results
