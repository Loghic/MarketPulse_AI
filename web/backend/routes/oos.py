"""
routes/oos.py – Out-of-sample model-selection harness over HTTP.

Wraps ``scripts.oos_harness`` (the disciplined select-on-window-N →
evaluate-on-disjoint-N+1 pipeline) so the frontend can launch OOS runs, poll
live progress, and reload persisted runs — mirroring the Backtest route's
shape. The harness writes its usual ``results/oos_<scope>_…/`` CSV tree, and we
also cache the full JSON response under ``oos_runs/{run_id}.json`` for instant
tab redisplay and for the OOS-comparison tab to diff against.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from cli_helpers import scope_label_from_tickers  # see note below
from config import ALL_TICKERS, CRYPTO_BENCHMARKS, STOCK_BENCHMARKS
from scripts.oos_harness import (
    aggregate,
    build_run_dir,
    oos_one_ticker,
    write_per_ticker,
    write_summary,
)
from web.backend.routes.data import get_api
from web.backend.schemas import OOSRequest, OOSResponse, OOSSummary, OOSTickerRow

router = APIRouter(prefix="/api/oos", tags=["oos"])

RESULTS_DIR = Path("results")
CACHE_DIR = Path("oos_runs")

# ----------------------------------------------------------------------
# Progress tracker (shared across requests, same pattern as backtest)
# ----------------------------------------------------------------------

_progress: dict[str, Any] = {"running": False}
_progress_lock = threading.Lock()


def _set_progress(**fields: Any) -> None:
    with _progress_lock:
        _progress.update(fields)


def _scope_label(tickers: list[str]) -> str:
    """Best-effort scope label for the run-dir / id (custom when mixed)."""
    return scope_label_from_tickers(tickers)


@router.post("", response_model=OOSResponse)
def run_oos(req: OOSRequest) -> OOSResponse:
    """Run the OOS harness for the requested tickers and persist the result."""
    api = get_api()
    tickers = [t.upper() for t in req.tickers] if req.tickers else list(ALL_TICKERS)

    if req.refresh_data:
        all_to_refresh = list(set(tickers + STOCK_BENCHMARKS + CRYPTO_BENCHMARKS))
        api.refresh_tickers(all_to_refresh, verbose=False)

    periods = req.periods or None  # None → oos_one_ticker uses all periods

    started = datetime.now()
    _set_progress(
        running=True,
        stage="starting",
        ticker=None,
        ticker_idx=0,
        ticker_total=len(tickers),
        started_at=started.isoformat(timespec="seconds"),
        completed_units=0,
        total_units=len(tickers),
        rows_so_far=0,
    )

    rows: list[dict] = []
    try:
        from config import ALL_PERIODS

        period_list = periods or list(ALL_PERIODS)
        for ti, ticker in enumerate(tickers, start=1):
            _set_progress(stage="running", ticker=ticker, ticker_idx=ti)
            try:
                row = oos_one_ticker(
                    api,
                    ticker,
                    n_days=req.days,
                    fee_pct=req.fee_pct,
                    stop_loss_pct=req.stop_loss_pct,
                    periods=period_list,
                    news_lookback_days=req.news_lookback_days or 7,
                    news_half_life_days=req.news_half_life_days or 0.0,
                    sentiment_method=req.sentiment_method,
                    models=req.models,
                    include_baselines=req.include_baselines,
                    min_confidence=req.min_confidence,
                    turnover_fees=req.turnover_fees,
                    hold_days=req.hold_days,
                )
            except Exception:  # noqa: BLE001 — one bad ticker shouldn't kill the run
                row = None
            if row is not None:
                rows.append(row)
            _set_progress(completed_units=ti, rows_so_far=len(rows))

        summary = aggregate(rows)

        # Persist the usual CSV tree (per-ticker + summary) so the
        # OOS-comparison tab + Analysis browser can read it.
        scope = _scope_label(tickers)
        run_dir = build_run_dir(
            RESULTS_DIR,
            scope,
            req.days,
            req.fee_pct,
            req.stop_loss_pct,
            req.buy_hold,
            min_confidence=req.min_confidence,
        )
        write_per_ticker(run_dir, rows)
        write_summary(run_dir, summary)

        run_id = run_dir.name
        # Pydantic validates/coerces each dict against the schema; unknown keys
        # are ignored and defaults fill any the harness didn't emit. Using
        # model_validate keeps mypy happy (no **dict[str, Any|None] splat).
        response = OOSResponse(
            rows=[OOSTickerRow.model_validate(r) for r in rows],
            summary=OOSSummary.model_validate(summary),
            results_dir=run_dir.name,
            run_id=run_id,
        )

        # JSON cache for instant tab redisplay.
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            (CACHE_DIR / f"{run_id}.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "saved_at": started.isoformat(timespec="seconds"),
                        "results_dir": run_dir.name,
                        "request": req.model_dump(),
                        "tickers": tickers,
                        "response": response.model_dump(),
                    },
                    indent=2,
                )
            )
        except Exception as e:  # noqa: BLE001
            _set_progress(persist_error=str(e))

        _set_progress(
            stage="complete",
            run_id=run_id,
            results_dir=run_dir.name,
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )
        return response
    except Exception as e:
        _set_progress(stage="error", error=str(e))
        raise
    finally:
        with _progress_lock:
            _progress["running"] = False


@router.get("/progress")
def get_progress() -> dict:
    """Live OOS progress (or last-known state between runs)."""
    with _progress_lock:
        return dict(_progress)


@router.get("/runs")
def list_runs() -> list[dict]:
    """List cached OOS runs, newest first."""
    if not CACHE_DIR.exists():
        return []
    out = []
    for f in sorted(CACHE_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        out.append(
            {
                "run_id": f.stem,
                "saved_at": data.get("saved_at"),
                "results_dir": data.get("results_dir"),
                "tickers": data.get("tickers", []),
                "request": data.get("request", {}),
                "row_count": len(data.get("response", {}).get("rows", [])),
            }
        )
    return out


@router.get("/runs/{run_id}")
def load_run(run_id: str) -> dict:
    """Load one cached OOS run by id."""
    p = CACHE_DIR / f"{run_id}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"OOS run '{run_id}' not found")
    return json.loads(p.read_text())
