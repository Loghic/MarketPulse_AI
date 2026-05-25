"""
routes/analysis.py – News vs No-News model comparison + results-tree browser.

The browser endpoints (results-dirs, result-csv) intentionally just expose
the raw CSVs produced by ``run_all.py``. The frontend does the pairing /
delta computation in JavaScript — same logic as ``scripts/news_impact.py``
but kept client-side so we don't lock the schema in two places.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from config import ALL_TICKERS
from engine.backtest_helpers import filter_by_period
from engine.backtester import Backtester
from web.backend.routes.data import get_api
from web.backend.schemas import AnalysisRequest, NewsComparisonRow

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

RESULTS_DIR = Path("results")

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
                model=model,
                model_name=f"{label} (no news)",
                df=filtered,
                ticker=ticker,
                use_time_weights=True,
                sentiment_score=0.0,
            )

            # Run WITH news
            r_with_news = backtester.run(
                model=model,
                model_name=f"{label} + News",
                df=filtered,
                ticker=ticker,
                use_time_weights=True,
                sentiment_score=sentiment_score if has_news else 0.0,
            )

            diff = r_with_news.total_return - r_no_news.total_return

            results.append(
                NewsComparisonRow(
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
                )
            )

    return results


# ----------------------------------------------------------------------
# Browse the ``results/`` tree (output of ``run_all.py``)
# ----------------------------------------------------------------------


def _safe_dir(dir_name: str) -> Path:
    """Validate a results subdirectory name to prevent path traversal."""
    if "/" in dir_name or "\\" in dir_name or ".." in dir_name or dir_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid directory name")
    p = RESULTS_DIR / dir_name
    if not p.is_dir():
        raise HTTPException(status_code=404, detail=f"Directory not found: {dir_name}")
    return p


@router.get("/results-dirs")
def list_results_dirs():
    """
    Enumerate ``results/{scope}_{days}d…/`` directories with each subdir's
    available CSVs (so the Analysis tab can populate a picker).

    Each entry has:
      * ``name`` – subdir name as produced by ``run_all.py``
      * ``modified`` – ISO timestamp of the most recent CSV inside
      * ``ticker_csvs`` – per-ticker CSV filenames (without extension)
      * ``has_summary`` – whether ``_summary.csv`` was written
      * ``has_news_impact`` – whether ``_news_vs_no_news_*.csv`` were written
        by ``scripts/news_impact.py``
    """
    if not RESULTS_DIR.exists():
        return []

    out = []
    for sub in sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir()):
        csvs = sorted(sub.glob("*.csv"))
        if not csvs:
            continue
        ticker_csvs = [f.stem for f in csvs if not f.name.startswith("_")]
        has_summary = (sub / "_summary.csv").exists()
        has_news_impact = any(
            f.name.startswith("_news_vs_no_news_") and f.name != "_news_vs_no_news_overall.csv"
            for f in csvs
        )
        # Most recent mtime across the CSVs is a reasonable proxy for "run on"
        latest_mtime = max(f.stat().st_mtime for f in csvs)
        out.append(
            {
                "name": sub.name,
                "modified": datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds"),
                "csv_count": len(csvs),
                "ticker_csvs": ticker_csvs,
                "has_summary": has_summary,
                "has_news_impact": has_news_impact,
            }
        )
    return out


@router.get("/result-csv")
def read_result_csv(dir: str, file: str):
    """
    Return a single CSV from a results/{dir}/ subdirectory as a list of rows
    (each row a dict of {column: string}). The frontend coerces numeric
    columns itself.

    Two query params:
      * ``dir`` – the subdirectory under ``results/`` (e.g. ``stocks_50d_fee003_bh``)
      * ``file`` – the CSV name with or without the ``.csv`` extension
        (e.g. ``AAPL`` or ``_summary``)
    """
    base = _safe_dir(dir)
    # Allow either "AAPL" or "AAPL.csv". Also accept the leading underscore
    # used for derived files (``_summary``, ``_news_vs_no_news_AAPL``).
    fname = file if file.endswith(".csv") else f"{file}.csv"
    if "/" in fname or "\\" in fname or ".." in fname:
        raise HTTPException(status_code=400, detail="Invalid file name")
    target = base / fname
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {dir}/{fname}")

    with open(target, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return {"dir": dir, "file": fname, "rows": rows, "row_count": len(rows)}
