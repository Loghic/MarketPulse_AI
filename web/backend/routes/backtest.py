"""
routes/backtest.py – Backtesting endpoints.
"""

from fastapi import APIRouter

from config import ALL_PERIODS, ALL_TICKERS, CRYPTO_BENCHMARKS, STOCK_BENCHMARKS
from engine.backtest_helpers import (
    compute_benchmarks,
    run_single_backtest,
)
from engine.backtester import Backtester
from web.backend.routes.data import get_api
from web.backend.schemas import (
    BacktestDayRow,
    BacktestModelResult,
    BacktestRequest,
    BacktestResponse,
)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


def _result_to_schema(
    r, ticker: str, period: str, benchmarks: dict | None = None
) -> BacktestModelResult:
    """Convert engine BacktestResult to API schema."""
    days = [
        BacktestDayRow(
            date=d.date,
            predicted=d.predicted,
            actual=d.actual,
            correct=d.correct,
            confidence=d.confidence,
            trade_pnl=round(d.trade_pnl, 8),
            trade_pnl_net=round(d.trade_pnl_net, 8),
            stopped_out=d.stopped_out,
        )
        for d in r.days
    ]

    return BacktestModelResult(
        model=r.model_name,
        ticker=ticker,
        period=period,
        accuracy=r.accuracy,
        total_return=r.total_return,
        buy_hold_return=r.buy_hold_return,
        profit_factor=r.profit_factor,
        max_drawdown=r.max_drawdown,
        sharpe_ratio=r.sharpe_ratio,
        sortino_ratio=r.sortino_ratio,
        buy_hold_max_drawdown=r.buy_hold_max_drawdown,
        fee_pct=r.fee_pct,
        stop_loss_pct=r.stop_loss_pct,
        stopped_out_count=r.stopped_out_count,
        win_trades=r.win_trades,
        loss_trades=r.loss_trades,
        avg_win=r.avg_win,
        avg_loss=r.avg_loss,
        best_day=r.best_day,
        worst_day=r.worst_day,
        longest_win_streak=r.longest_win_streak,
        longest_loss_streak=r.longest_loss_streak,
        benchmarks=benchmarks or {},
        days=days,
    )


@router.post("", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest):
    """Run walk-forward backtest."""
    api = get_api()
    tickers = [t.upper() for t in req.tickers] if req.tickers else ALL_TICKERS

    if req.refresh_data:
        all_to_refresh = list(set(tickers + STOCK_BENCHMARKS + CRYPTO_BENCHMARKS))
        api.refresh_tickers(all_to_refresh, verbose=False)

    backtester = Backtester(
        n_days=req.days,
        fee_pct=req.fee_pct,
        stop_loss_pct=req.stop_loss_pct,
    )

    # Resolve which periods to run.
    # Priority: explicit ``periods`` list > ``compare_periods`` flag > single ``period``.
    if req.periods:
        periods = req.periods
    elif req.compare_periods:
        periods = ALL_PERIODS
    else:
        periods = [req.period]

    # Forward the news / sentiment knobs to run_single_backtest when given.
    news_kwargs: dict = {}
    if req.sentiment_method is not None:
        news_kwargs["sentiment_method"] = req.sentiment_method
    if req.news_lookback_days is not None:
        news_kwargs["news_lookback_days"] = req.news_lookback_days
    if req.news_half_life_days is not None:
        news_kwargs["news_half_life_days"] = req.news_half_life_days

    all_results = []

    for ticker in tickers:
        df = api.get_data(ticker, period="max")
        if df.empty:
            continue

        for period in periods:
            results = run_single_backtest(
                api, backtester, ticker, df, period, req.days, full=False, **news_kwargs
            )
            if not results:
                continue

            bench = compute_benchmarks(api, ticker, results[0].days) if req.buy_hold else None

            for r in results:
                all_results.append(_result_to_schema(r, ticker, period, bench))

    # Find best models
    best_return = max(all_results, key=lambda r: r.total_return) if all_results else None
    best_sharpe = max(all_results, key=lambda r: r.sharpe_ratio) if all_results else None

    return BacktestResponse(
        results=all_results,
        best_by_return=best_return,
        best_by_sharpe=best_sharpe,
    )
