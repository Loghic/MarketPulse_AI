"""
forecast_harness.py – Walk-forward point-forecast evaluation (regression track).

The regression-track analogue of ``oos_harness.py``. Instead of trading P&L it
scores how well each model predicts the next (or h-ahead) **close price level**,
ranked by scale-free skill vs a random walk (Theil U2 / MASE) — the metrics that
don't flatter every model the way raw RMSE does on a price level.

Models scored:
  * Always: Random Walk, Random Walk + Drift, Seasonal Naive (the regression
    baselines). Random Walk is the U2 reference (scores 1.0 by construction).
  * When their optional library is installed: ARIMA (statsmodels), XGBoost
    (xgboost), and the existing forecasting models (Prophet, Chronos-2, Kronos).

Output: a tidy per-step CSV per ticker plus a per-(model,ticker,horizon) summary
under ``results/fc_<scope>_h<h>_<ts>/``, and a console table ranked by MASE/U2.

Example:
    uv run python scripts/forecast_harness.py --stocks --days 100 --horizon 1 \
        --no-refresh
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

# Make the repo root importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli_helpers import (  # noqa: E402
    add_scope_args,
    resolve_scope,
    scope_label,
)
from config import ALL_TICKERS  # noqa: E402
from engine.forecast_backtester import ForecastBacktester, ForecastRun  # noqa: E402
from engine.logger import get_logger, progress_bar  # noqa: E402
from engine.naive_forecasters import default_naive_forecasters  # noqa: E402
from interface.api import StockAppAPI  # noqa: E402

log = get_logger("forecast_harness")


# ----------------------------------------------------------------------
# Model registry (naive always; optional models when their lib is present)
# ----------------------------------------------------------------------


def build_forecasters(season: int = 5) -> list:
    """All available point-forecasters as (model, label) pairs.

    Optional models are appended only when their library imports — exactly the
    graceful-skip behaviour the directional forecasting models already use.
    """
    models: list = [(m, label) for m, label in default_naive_forecasters(season=season)]

    try:
        from engine.arima_model import _STATSMODELS_AVAILABLE, ARIMAForecaster

        if _STATSMODELS_AVAILABLE:
            models.append((ARIMAForecaster(), "ARIMA"))
    except Exception as e:  # noqa: BLE001
        log.debug("ARIMA unavailable: %s", e)

    try:
        from engine.xgboost_model import _XGBOOST_AVAILABLE, XGBoostForecaster

        if _XGBOOST_AVAILABLE:
            models.append((XGBoostForecaster(), "XGBoost"))
    except Exception as e:  # noqa: BLE001
        log.debug("XGBoost unavailable: %s", e)

    # The existing forecasting models already expose forecast(df, horizon).
    try:
        from engine.prophet_model import _PROPHET_AVAILABLE, ProphetModel

        if _PROPHET_AVAILABLE:
            models.append((ProphetModel(), "Prophet"))
    except Exception as e:  # noqa: BLE001
        log.debug("Prophet unavailable: %s", e)

    return models


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def build_run_dir(root: Path, scope: str, days: int, horizon: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"fc_{scope}_{days}d_h{horizon}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_per_ticker_steps(run_dir: Path, ticker: str, runs: list[ForecastRun]) -> None:
    """Tidy long CSV: one row per (model, step)."""
    out = run_dir / f"{ticker}.csv"
    cols = ["model", "ticker", "horizon", "date", "y_true", "y_pred", "y_naive"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for run in runs:
            for s in run.steps:
                w.writerow(
                    {
                        "model": run.model_name,
                        "ticker": run.ticker,
                        "horizon": s.horizon,
                        "date": s.date,
                        "y_true": s.y_true,
                        "y_pred": s.y_pred,
                        "y_naive": s.y_naive,
                    }
                )


def summary_rows(all_runs: list[ForecastRun]) -> list[dict]:
    rows: list[dict] = []
    for run in all_runs:
        if run.metrics is None:
            continue
        m = run.metrics
        rows.append(
            {
                "model": run.model_name,
                "ticker": run.ticker,
                "horizon": run.horizon,
                "n": m.n,
                "skipped": run.skipped,
                "rmse": round(m.rmse, 6),
                "mae": round(m.mae, 6),
                "mape": round(m.mape, 6),
                "smape": round(m.smape, 6),
                "mase": round(m.mase, 6),
                "rmsse": round(m.rmsse, 6),
                "theil_u2": round(m.theil_u2, 6),
                "elapsed_seconds": run.elapsed_seconds,
            }
        )
    return rows


def write_summary(run_dir: Path, rows: list[dict]) -> Path:
    out = run_dir / "_fc_summary.csv"
    cols = [
        "model",
        "ticker",
        "horizon",
        "n",
        "skipped",
        "rmse",
        "mae",
        "mape",
        "smape",
        "mase",
        "rmsse",
        "theil_u2",
        "elapsed_seconds",
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


# ----------------------------------------------------------------------
# Console
# ----------------------------------------------------------------------


def print_results(rows: list[dict], horizon: int) -> str:
    lines: list[str] = []
    add = lines.append
    sep = "=" * 92
    add(sep)
    add(f" FORECAST HARNESS — point-forecast skill vs random walk (horizon h={horizon})")
    add(" U2 < 1 ⇔ beats the random walk · MASE < 1 ⇔ beats in-sample naive")
    add(sep)
    if not rows:
        add(" No models produced scored forecasts.")
        text = "\n".join(lines)
        print(text)
        return text

    add(
        f"  {'MODEL':<22} {'TICKER':<10} {'N':>4} "
        f"{'RMSE':>10} {'MAE':>10} {'MAPE':>8} {'MASE':>8} {'U2':>8}"
    )
    add("  " + "-" * 88)

    # Sort by U2 ascending (best skill first), NaNs last.
    def _u2(r: dict) -> float:
        v = r["theil_u2"]
        return float("inf") if v != v else v  # NaN check

    for r in sorted(rows, key=lambda r: (r["ticker"], _u2(r))):
        add(
            f"  {r['model'][:22]:<22} {r['ticker']:<10} {r['n']:>4} "
            f"{r['rmse']:>10.4f} {r['mae']:>10.4f} {r['mape']:>8.4f} "
            f"{r['mase']:>8.3f} {r['theil_u2']:>8.3f}"
        )

    # Aggregate: median U2 / MASE per model across tickers.
    add("  " + "-" * 88)
    add(" AGGREGATE (median across tickers, per model)")
    add("  " + "-" * 88)
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    def _median(vals: list[float]) -> float:
        clean = [v for v in vals if v == v]  # drop NaN
        return statistics.median(clean) if clean else float("nan")

    add(f"  {'MODEL':<22} {'tickers':>8} {'med MASE':>10} {'med U2':>10}")
    for model, mrows in sorted(
        by_model.items(), key=lambda kv: _median([r["theil_u2"] for r in kv[1]])
    ):
        med_mase = _median([r["mase"] for r in mrows])
        med_u2 = _median([r["theil_u2"] for r in mrows])
        add(f"  {model[:22]:<22} {len(mrows):>8} {med_mase:>10.3f} {med_u2:>10.3f}")
    add("  → A model is only interesting if median U2 < 1.0 (beats the random walk).")
    add(sep)

    text = "\n".join(lines)
    print(text)
    return text


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast harness: walk-forward point-forecast evaluation, ranked "
            "by skill vs a random walk (Theil U2 / MASE). Regression-track "
            "companion to oos_harness.py."
        )
    )
    add_scope_args(parser)
    parser.add_argument("--days", type=int, default=100, help="Evaluation window length (steps).")
    parser.add_argument("--horizon", type=int, default=1, help="Direct-h forecast horizon.")
    parser.add_argument(
        "--refit-k",
        type=int,
        default=21,
        help="Refit cadence in steps (cost knob; default ~monthly).",
    )
    parser.add_argument(
        "--min-train",
        type=int,
        default=60,
        help="Minimum training rows before the first scored step.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=504,
        help=(
            "Cap the training window to the most-recent N rows (default 504 ≈ 2 "
            "trading years). Keeps ARIMA/Prophet fast and fitted on a relevant "
            "regime instead of decades of history, and makes MASE scale against "
            "recent volatility. Use 0 for the full expanding window."
        ),
    )
    parser.add_argument(
        "--no-refresh", action="store_true", help="Skip the data download (use cached prices)."
    )
    parser.add_argument(
        "--no-lstm",
        action="store_true",
        help=(
            "Skip the per-ticker LSTM regressor even if torch + weights exist. "
            "Train weights first with scripts/train_lstm_regressor.py."
        ),
    )
    parser.add_argument("--dir", type=str, default="results", help="Root output directory.")
    args = parser.parse_args()

    tickers = resolve_scope(args, default=ALL_TICKERS)
    scope = scope_label(args)
    run_dir = build_run_dir(Path(args.dir), scope, args.days, args.horizon)

    forecasters = build_forecasters()
    labels = ", ".join(label for _, label in forecasters)
    if not args.no_lstm:
        labels += ", LSTM-reg (per-ticker, when weights exist)"

    print("=" * 92)
    print(
        f" FORECAST HARNESS — {len(tickers)} tickers × {len(forecasters)} models "
        f"× {args.days}d eval, horizon h={args.horizon}"
    )
    print(f" Models: {labels}")
    print(f" Output: {run_dir.resolve()}/")
    print("=" * 92 + "\n")

    api = StockAppAPI()
    if not args.no_refresh:
        api.refresh_tickers(list(tickers), verbose=False)

    # 0 means "no cap" (full expanding window); otherwise the row cap.
    max_train = None if args.max_train == 0 else args.max_train
    bt = ForecastBacktester(
        n_days=args.days,
        horizon=args.horizon,
        refit_k=args.refit_k,
        min_train=args.min_train,
        max_train=max_train,
    )

    all_runs: list[ForecastRun] = []
    for ticker in progress_bar(tickers, desc="Forecast harness"):
        df = api.get_data(ticker, period="max")
        if df is None or df.empty:
            log.info("%s: no data, skipping.", ticker)
            continue
        # The LSTM regressor is per-ticker (loads models/{ticker}_reg.pt); the
        # rest are shared instances. The forecaster skips itself if torch is
        # missing or no weights exist for this ticker, so this is safe to always
        # add when not explicitly disabled.
        ticker_models = list(forecasters)
        if not args.no_lstm:
            from engine.lstm_regressor import LSTMRegressorForecaster

            ticker_models.append((LSTMRegressorForecaster(ticker), "LSTM-reg"))

        ticker_runs: list[ForecastRun] = []
        for model, _label in ticker_models:
            try:
                run = bt.run(model, df, ticker)
            except Exception as e:  # noqa: BLE001 — one bad model shouldn't kill the run
                log.warning("%s/%s: run failed (%s).", model.name, ticker, e)
                continue
            if run is not None:
                ticker_runs.append(run)
        if ticker_runs:
            write_per_ticker_steps(run_dir, ticker, ticker_runs)
            all_runs.extend(ticker_runs)

    rows = summary_rows(all_runs)
    write_summary(run_dir, rows)
    text = print_results(rows, args.horizon)
    (run_dir / "_fc_console.txt").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
