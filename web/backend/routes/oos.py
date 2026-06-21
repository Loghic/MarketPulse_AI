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

import csv
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


# ----------------------------------------------------------------------
# CSV-run discovery
# ----------------------------------------------------------------------
#
# Runs launched from the CLI (`scripts/oos_harness.py`) only write the
# `results/oos_<scope>_…/` CSV tree — they never produce an `oos_runs/*.json`
# cache. To make *every* past run browsable + comparable in the web UI, we also
# discover those CSV directories and reconstruct an OOSResponse from them. The
# directory name is used as the run_id (same convention as the JSON cache), so a
# CSV dir is skipped whenever a JSON cache of the same id already exists — no
# duplicates.


def _csv_run_dirs() -> list[Path]:
    """All `results/oos_*/` dirs that carry a per-ticker CSV, newest first."""
    if not RESULTS_DIR.exists():
        return []
    dirs = [
        d for d in RESULTS_DIR.glob("oos_*") if d.is_dir() and (d / "_oos_per_ticker.csv").exists()
    ]
    return sorted(dirs, key=lambda d: d.name, reverse=True)


def _cached_run_ids() -> set[str]:
    """run_ids that already have a JSON cache entry (these win over CSVs)."""
    if not CACHE_DIR.exists():
        return set()
    return {f.stem for f in CACHE_DIR.glob("*.json")}


def _clean_row(raw: dict[str, str]) -> dict[str, str]:
    """Drop empty cells so schema defaults apply (CSV writes "" not null)."""
    return {k: v for k, v in raw.items() if v != ""}


def _read_csv_run(run_dir: Path) -> dict[str, Any]:
    """Reconstruct the cached-run JSON shape from a CSV run directory.

    Mirrors what `run_oos` persists, minus the request echo (CLI runs don't
    record their request). Rows / summary are validated through the same
    pydantic models, so the response shape is identical to a cached run.
    """
    per_ticker = run_dir / "_oos_per_ticker.csv"
    summary_csv = run_dir / "_oos_summary.csv"

    with per_ticker.open(newline="") as f:
        raw_rows = list(csv.DictReader(f))
    rows = [OOSTickerRow.model_validate(_clean_row(r)) for r in raw_rows]

    if summary_csv.exists():
        with summary_csv.open(newline="") as f:
            summary_rows = list(csv.DictReader(f))
        summary_dict = _clean_row(summary_rows[0]) if summary_rows else {}
    else:
        summary_dict = {}
    # An empty/absent summary still validates: only `tickers` etc. are required,
    # so fall back to a recomputed aggregate when the CSV had no summary row.
    if summary_dict:
        summary = OOSSummary.model_validate(summary_dict)
    else:
        summary = OOSSummary.model_validate(aggregate([r.model_dump() for r in rows]))

    response = OOSResponse(
        rows=rows, summary=summary, results_dir=run_dir.name, run_id=run_dir.name
    )
    tickers = [r.ticker for r in rows]
    try:
        saved_at = datetime.fromtimestamp(per_ticker.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        saved_at = None
    return {
        "run_id": run_dir.name,
        "saved_at": saved_at,
        "results_dir": run_dir.name,
        "request": {},  # CLI runs don't echo their request
        "tickers": tickers,
        "source": "csv",
        "response": response.model_dump(),
    }


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
                    position_mode=req.position_mode,
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
    """List OOS runs (JSON cache + discovered CSV runs), newest first.

    Web-launched runs come from the `oos_runs/*.json` cache; CLI runs are
    discovered from the `results/oos_*/` CSV tree. A CSV run is skipped when a
    JSON cache of the same run_id already exists, so the two never duplicate.
    """
    out: list[dict] = []
    cached_ids: set[str] = set()

    if CACHE_DIR.exists():
        for f in sorted(CACHE_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text())
            except Exception:  # noqa: BLE001
                continue
            cached_ids.add(f.stem)
            out.append(
                {
                    "run_id": f.stem,
                    "saved_at": data.get("saved_at"),
                    "results_dir": data.get("results_dir"),
                    "tickers": data.get("tickers", []),
                    "request": data.get("request", {}),
                    "row_count": len(data.get("response", {}).get("rows", [])),
                    "source": data.get("source", "web"),
                }
            )

    for d in _csv_run_dirs():
        if d.name in cached_ids:
            continue  # JSON cache wins — no duplicate
        pt_csv = d / "_oos_per_ticker.csv"
        try:
            with pt_csv.open(newline="") as fh:
                tickers = [r.get("ticker", "") for r in csv.DictReader(fh)]
            row_count = len(tickers)
            saved_at = datetime.fromtimestamp(pt_csv.stat().st_mtime).isoformat(timespec="seconds")
        except Exception:  # noqa: BLE001 — a malformed dir shouldn't kill the list
            continue
        out.append(
            {
                "run_id": d.name,
                "saved_at": saved_at,
                "results_dir": d.name,
                "tickers": tickers,
                "request": {},
                "row_count": row_count,
                "source": "csv",
            }
        )

    # Newest first across both sources (run-dir names are timestamp-suffixed).
    out.sort(key=lambda r: r.get("saved_at") or "", reverse=True)
    return out


@router.get("/runs/{run_id}")
def load_run(run_id: str) -> dict:
    """Load one OOS run by id — JSON cache first, then a CSV run directory."""
    p = CACHE_DIR / f"{run_id}.json"
    if p.exists():
        return json.loads(p.read_text())

    run_dir = RESULTS_DIR / run_id
    if run_dir.is_dir() and (run_dir / "_oos_per_ticker.csv").exists():
        try:
            return _read_csv_run(run_dir)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=500,
                detail=f"OOS run '{run_id}' CSV could not be parsed: {e}",
            ) from e

    raise HTTPException(status_code=404, detail=f"OOS run '{run_id}' not found")
