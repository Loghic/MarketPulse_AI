"""
routes/data.py – Ticker data and refresh endpoints.
"""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from config import ALL_TICKERS, DEFAULT_SENTIMENT_METHOD
from interface.api import StockAppAPI
from web.backend.schemas import (
    OHLCVRow,
    RefreshNewsRequest,
    RefreshNewsStatus,
    RefreshRequest,
    RefreshStatus,
    TickerDataResponse,
    TickerInfo,
)

router = APIRouter(prefix="/api/data", tags=["data"])
log = logging.getLogger("marketpulse.web")

# Shared API instance (created once, reused)
_api: StockAppAPI | None = None


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
        result.append(
            TickerInfo(
                ticker=ticker,
                asset_type=asset_type,
                rows=len(df),
                last_date=last_date,
            )
        )
    return result


@router.get("/ticker/{ticker}", response_model=TickerDataResponse)
def get_ticker_data(ticker: str, period: str = "1y", limit: int = 0):
    """Get OHLCV data for a single ticker. limit=0 means no limit."""
    api = get_api()
    ticker = ticker.upper()

    df = api.db.get_prices(ticker)
    if df.empty:
        raise HTTPException(404, f"No data for {ticker}. Run refresh first.")

    # Apply period filter (skip for "max")
    if period != "max":
        from engine.utils import period_to_start_date

        start = period_to_start_date(period)
        df["_date"] = pd.to_datetime(df["date"]).dt.date
        df = df[df["_date"] >= start].drop(columns=["_date"])

    # Sort descending
    df = df.sort_values("date", ascending=False)

    # Apply limit only if > 0
    if limit > 0:
        df = df.head(limit)

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

    log.info(f"Refresh requested for {len(tickers)} tickers: {tickers}")

    results = []
    for ticker in tickers:
        try:
            log.info(f"Refreshing {ticker}...")
            df = api.get_data(ticker, period="max")
            news_count = 0
            try:
                _score, headlines = api._process_news_with_db(ticker)
                news_count = len(headlines)
            except Exception:
                pass  # News is optional

            last_date = str(df["date"].iloc[-1]) if not df.empty else "n/a"
            results.append(
                RefreshStatus(
                    ticker=ticker,
                    rows=len(df),
                    last_date=last_date,
                    news_count=news_count,
                )
            )
            log.info(f"  {ticker}: {len(df)} rows, last={last_date}")
        except Exception as e:
            log.error(f"  {ticker}: FAILED - {e}")
            results.append(
                RefreshStatus(
                    ticker=ticker,
                    rows=0,
                    last_date="error",
                    news_count=0,
                )
            )

    log.info(f"Refresh complete: {len(results)} tickers processed")
    return results


@router.post("/refresh-news", response_model=list[RefreshNewsStatus])
def refresh_news(req: RefreshNewsRequest):
    """
    Bulk-fetch news for the given tickers with explicit source + scorer +
    history-days knobs — equivalent to::

        uv run python refresh.py --tickers <T...> \\
            --news-source <yahoo|gdelt|...> \\
            --news-history-days <N> \\
            --sentiment-method <vader|finbert|naive> \\
            --force-news

    Returns one row per ticker with the number of headlines pulled and the
    mean sentiment. Use this from the web UI to populate the DB with deep
    historical news before running ``+ News`` backtests.
    """
    api = get_api()
    tickers = [t.upper() for t in req.tickers] if req.tickers else ALL_TICKERS
    method = req.sentiment_method or DEFAULT_SENTIMENT_METHOD
    sources = req.news_source or ["yahoo"]

    log.info(
        f"News refresh: {len(tickers)} tickers · sources={sources} · "
        f"method={method} · history={req.news_history_days}d"
    )

    results: list[RefreshNewsStatus] = []
    for ticker in tickers:
        try:
            score, headlines = api._process_news_with_db(
                ticker,
                method=method,
                source=sources,
                lookback_days=req.news_history_days,
                force_refresh=req.force_news,
            )
            results.append(
                RefreshNewsStatus(
                    ticker=ticker,
                    headlines_pulled=len(headlines),
                    mean_sentiment=round(float(score), 4) if headlines else 0.0,
                    method=method,
                    sources=sources,
                )
            )
            log.info(f"  {ticker}: {len(headlines)} headlines, mean sentiment {score:+.3f}")
        except Exception as e:
            log.error(f"  {ticker}: news refresh failed — {e}")
            results.append(
                RefreshNewsStatus(
                    ticker=ticker,
                    headlines_pulled=0,
                    mean_sentiment=0.0,
                    method=method,
                    sources=sources,
                    error=str(e),
                )
            )
    return results
