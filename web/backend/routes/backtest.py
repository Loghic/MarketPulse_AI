"""
routes/backtest.py – Backtesting endpoints.

A single ``POST /api/backtest`` runs the walk-forward sweep across every
selected (ticker × period) and returns one ``BacktestResponse``. The same
run also persists two artifacts on disk so the user can come back to it
later without re-running:

  * ``backtests/{run_id}.json`` – the full response, used by the Backtest
    tab to redisplay the latest run on mount.
  * ``results/{scope}_{days}d_…_{YYYYMMDD-HHMMSS}/`` – per-ticker CSV +
    ``_summary.csv``, same layout as ``run_all.py``. Lets
    ``scripts/news_impact.py`` and the Analysis tab pick the run up.

While a backtest is running the route updates a module-level
``_progress`` dict that ``GET /api/backtest/progress`` exposes. The
frontend polls it every ~500 ms to drive a progress bar (current ticker
N/M, current period, elapsed seconds). FastAPI runs sync routes in a
threadpool, so a long-running backtest doesn't block other requests
from reading the progress endpoint.
"""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

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

CACHE_DIR = Path("backtests")
RESULTS_DIR = Path("results")

# ----------------------------------------------------------------------
# Progress tracker (shared across requests)
# ----------------------------------------------------------------------

_progress: dict[str, Any] = {"running": False}
_progress_lock = threading.Lock()


def _set_progress(**fields: Any) -> None:
    with _progress_lock:
        _progress.update(fields)


def _reset_progress() -> None:
    with _progress_lock:
        _progress.clear()
        _progress["running"] = False


# ----------------------------------------------------------------------
# Schema conversion
# ----------------------------------------------------------------------


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
            sentiment_score=round(d.sentiment_score, 4),
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


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------


def _scope_label(tickers: list[str]) -> str:
    """Mirror run_all.py's directory naming convention."""
    from config import CRYPTO, STOCKS

    stock_set, crypto_set = set(STOCKS), set(CRYPTO)
    if set(tickers) == set(STOCKS + CRYPTO):
        return "all"
    if set(tickers) == stock_set:
        return "stocks"
    if set(tickers) == crypto_set:
        return "crypto"
    return "custom"


def _results_dir_name(
    scope: str, days: int, fee_pct: float, sl_pct: float, bh: bool, ts: str
) -> str:
    """``results/{scope}_{days}d[_fee{N}][_sl{N}][_bh]_{timestamp}/``"""
    parts = [scope, f"{days}d"]
    if fee_pct > 0:
        parts.append(f"fee{fee_pct * 100:03.0f}")
    if sl_pct > 0:
        parts.append(f"sl{sl_pct:g}")
    if bh:
        parts.append("bh")
    parts.append(ts)
    return "_".join(parts)


def _row_to_csv_dict(r: BacktestModelResult) -> dict:
    """Flatten a BacktestModelResult for CSV export — matches run_all.py columns."""
    # Per-day sentiment summary — non-zero only for "+ News" model variants.
    # ``mean_sentiment`` answers "what was the average mood of the news the
    # model saw across the holdout window"; ``sentiment_active_days`` answers
    # "on how many days did news actually influence the call" (for the
    # paper / poster narrative around news-vs-no-news).
    sentiments = [d.sentiment_score for d in r.days]
    nonzero = [v for v in sentiments if v != 0]
    mean_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
    sentiment_active_days = len(nonzero)

    row = {
        "ticker": r.ticker,
        "period": r.period,
        "model": r.model,
        "accuracy": round(r.accuracy, 4),
        "total_return": round(r.total_return, 8),
        "buy_hold_return": round(r.buy_hold_return, 8),
        "profit_factor": round(r.profit_factor, 4),
        "max_drawdown": round(r.max_drawdown, 8),
        "sharpe_ratio": round(r.sharpe_ratio, 4),
        "sortino_ratio": round(r.sortino_ratio, 4),
        "buy_hold_max_drawdown": round(r.buy_hold_max_drawdown, 8),
        "fee_pct": r.fee_pct,
        "stop_loss_pct": r.stop_loss_pct,
        "stopped_out": r.stopped_out_count,
        "win_trades": r.win_trades,
        "loss_trades": r.loss_trades,
        "avg_win": round(r.avg_win, 8),
        "avg_loss": round(r.avg_loss, 8),
        "best_day": round(r.best_day, 8),
        "worst_day": round(r.worst_day, 8),
        "longest_win_streak": r.longest_win_streak,
        "longest_loss_streak": r.longest_loss_streak,
        "mean_sentiment": round(mean_sentiment, 4),
        "sentiment_active_days": sentiment_active_days,
    }
    for bench, bret in (r.benchmarks or {}).items():
        row[f"bench_{bench}"] = bret
    return row


def _write_run_csv(results: list[BacktestModelResult], dest: Path) -> None:
    """Write one CSV per ticker + a _summary.csv (best per ticker)."""
    dest.mkdir(parents=True, exist_ok=True)
    by_ticker: dict[str, list[BacktestModelResult]] = {}
    for r in results:
        by_ticker.setdefault(r.ticker, []).append(r)

    # Per-ticker CSVs
    for ticker, rows in by_ticker.items():
        dicts = [_row_to_csv_dict(r) for r in rows]
        if not dicts:
            continue
        fieldnames: dict[str, None] = {}
        for d in dicts:
            for k in d:
                fieldnames[k] = None
        with open(dest / f"{ticker}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            w.writeheader()
            w.writerows(dicts)

    # Summary: best by total_return per ticker (one row per ticker)
    summary_rows: list[dict] = []
    for rows in by_ticker.values():
        if not rows:
            continue
        best = max(rows, key=lambda r: (r.total_return, r.accuracy))
        summary_rows.append(_row_to_csv_dict(best))
    if summary_rows:
        fieldnames = {}
        for d in summary_rows:
            for k in d:
                fieldnames[k] = None
        with open(dest / "_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            w.writeheader()
            w.writerows(summary_rows)


def _persist_run(
    response: BacktestResponse,
    request: BacktestRequest,
    tickers: list[str],
    results_dir_name: str,
    run_id: str,
) -> None:
    """Save JSON cache + CSV export side-by-side."""
    # JSON cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{run_id}.json"
    payload = {
        "run_id": run_id,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "results_dir": results_dir_name,
        "request": request.model_dump(),
        "tickers": tickers,
        "response": response.model_dump(),
    }
    cache_path.write_text(json.dumps(payload, default=str))

    # CSV export
    if response.results:
        _write_run_csv(response.results, RESULTS_DIR / results_dir_name)


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


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

    # ------------------------------------------------------------------
    # Progress tracker — populated as we iterate. Frontend polls
    # GET /api/backtest/progress (~500ms) for live updates.
    # ------------------------------------------------------------------
    started = datetime.now()
    _set_progress(
        running=True,
        stage="starting",
        ticker=None,
        ticker_idx=0,
        ticker_total=len(tickers),
        period=None,
        period_idx=0,
        period_total=len(periods),
        started_at=started.isoformat(timespec="seconds"),
        completed_units=0,
        total_units=len(tickers) * len(periods),
        results_so_far=0,
    )

    all_results: list[BacktestModelResult] = []
    try:
        for ti, ticker in enumerate(tickers, start=1):
            _set_progress(
                stage="ticker",
                ticker=ticker,
                ticker_idx=ti,
                period=None,
                period_idx=0,
            )

            df = api.get_data(ticker, period="max")
            if df.empty:
                _set_progress(completed_units=(ti - 1) * len(periods) + len(periods))
                continue

            for pi, period in enumerate(periods, start=1):
                _set_progress(
                    stage="running",
                    ticker=ticker,
                    ticker_idx=ti,
                    period=period,
                    period_idx=pi,
                )
                results = run_single_backtest(
                    api, backtester, ticker, df, period, req.days, full=False, **news_kwargs
                )
                if not results:
                    _set_progress(completed_units=(ti - 1) * len(periods) + pi)
                    continue

                bench = compute_benchmarks(api, ticker, results[0].days) if req.buy_hold else None

                for r in results:
                    all_results.append(_result_to_schema(r, ticker, period, bench))

                _set_progress(
                    completed_units=(ti - 1) * len(periods) + pi,
                    results_so_far=len(all_results),
                )

        # Find best models
        best_return = max(all_results, key=lambda r: r.total_return) if all_results else None
        best_sharpe = max(all_results, key=lambda r: r.sharpe_ratio) if all_results else None

        response = BacktestResponse(
            results=all_results,
            best_by_return=best_return,
            best_by_sharpe=best_sharpe,
        )

        # ------------------------------------------------------------------
        # Persist to disk: JSON cache for tab redisplay + CSV for Analysis.
        # Timestamped directory name lets the user keep multiple runs with
        # the same fee/SL settings without overwriting.
        # ------------------------------------------------------------------
        ts = started.strftime("%Y%m%d-%H%M%S")
        scope = _scope_label(tickers)
        results_dir_name = _results_dir_name(
            scope, req.days, req.fee_pct, req.stop_loss_pct, req.buy_hold, ts
        )
        run_id = f"{ts}_{scope}_{req.days}d"
        try:
            _persist_run(response, req, tickers, results_dir_name, run_id)
        except Exception as e:
            # Persistence failure shouldn't fail the request — the user
            # still gets the response, just no cache/CSV.
            _set_progress(persist_error=str(e))

        _set_progress(
            stage="complete",
            run_id=run_id,
            results_dir=results_dir_name,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        return response
    except Exception as e:
        _set_progress(stage="error", error=str(e))
        raise
    finally:
        # Mark progress as no-longer-running but keep the last-known state
        # so the frontend can read the final "complete" snapshot.
        with _progress_lock:
            _progress["running"] = False


@router.get("/progress")
def get_progress():
    """
    Current backtest progress (or last-known state if a run just finished).

    Polled by the frontend while a /backtest request is in flight to drive
    the progress bar. Returns ``{"running": false}`` between runs.
    """
    with _progress_lock:
        return dict(_progress)


# ----------------------------------------------------------------------
# Run cache: list + load
# ----------------------------------------------------------------------


@router.get("/runs")
def list_runs():
    """List every cached backtest run on disk, newest first."""
    if not CACHE_DIR.exists():
        return []
    out = []
    for f in sorted(CACHE_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        out.append(
            {
                "run_id": f.stem,
                "saved_at": data.get("saved_at"),
                "results_dir": data.get("results_dir"),
                "tickers": data.get("tickers", []),
                "request": data.get("request", {}),
                "result_count": len(data.get("response", {}).get("results", [])),
            }
        )
    return out


@router.get("/runs/{run_id}")
def load_run(run_id: str):
    """Return one cached backtest run."""
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    p = CACHE_DIR / f"{run_id}.json"
    if not p.is_file():
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
