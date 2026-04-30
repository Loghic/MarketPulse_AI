"""
routes/analysis.py – News vs No-News model comparison for academic paper.
"""

from fastapi import APIRouter

from engine.backtester import Backtester
from engine.backtest_helpers import filter_by_period, compute_benchmarks
from config import ALL_TICKERS, ALL_PERIODS
from web.backend.schemas import AnalysisRequest, NewsComparisonRow
from web.backend.routes.data import get_api

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# Models that support news (time-weighted variants)
NEWS_MODELS = [
    ("knn", "k-NN TW"),
    ("knn_enhanced", "k-NN Enh. TW"),
    ("linreg", "LinReg TW"),
    ("linreg_enhanced", "LinReg Enh. TW"),
]


@router.post("/news-comparison", response_model=list[NewsComparisonRow])
def compare_news_impact(req: AnalysisRequest):
    """
    Run each model twice (with and without news sentiment)
    and return paired comparison for statistical analysis.
    """
    api = get_api()
    tickers = [t.upper() for t in req.tickers] if req.tickers else ALL_TICKERS

    backtester = Backtester(n_days=req.days, fee_pct=req.fee_pct)
    results = []

    for ticker in tickers:
        df = api.get_data(ticker, period="max")
        if df.empty:
            continue

        filtered = filter_by_period(df, req.period)
        if len(filtered) < req.days + 20:
            continue

        # Get sentiment
        sentiment_score, headlines = api._process_news_with_db(ticker)
        has_news = len(headlines) > 0

        for model_type, label in NEWS_MODELS:
            model = api._get_model(model_type, ticker, req.period)

            # Run WITHOUT news
            r_no_news = backtester.run(
                model=model, model_name=f"{label} (no news)",
                df=filtered, ticker=ticker,
                use_time_weights=True, sentiment_score=0.0,
            )

            # Run WITH news
            r_with_news = backtester.run(
                model=model, model_name=f"{label} + News",
                df=filtered, ticker=ticker,
                use_time_weights=True,
                sentiment_score=sentiment_score if has_news else 0.0,
            )

            diff = r_with_news.total_return - r_no_news.total_return

            results.append(NewsComparisonRow(
                ticker=ticker,
                model=label,
                period=req.period,
                return_no_news=round(r_no_news.total_return, 8),
                return_with_news=round(r_with_news.total_return, 8),
                diff=round(diff, 8),
                sharpe_no_news=r_no_news.sharpe_ratio,
                sharpe_with_news=r_with_news.sharpe_ratio,
                accuracy_no_news=r_no_news.accuracy,
                accuracy_with_news=r_with_news.accuracy,
            ))

    return results
