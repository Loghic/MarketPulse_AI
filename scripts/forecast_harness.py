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


# ----------------------------------------------------------------------
# Model selection: keys, named groups, and the registry
# ----------------------------------------------------------------------
#
# --models accepts a comma/space-separated list of *keys* and/or *group* names.
#   keys:   rw, rwdrift, seasonal, arima, xgboost, prophet, chronos, kronos
#           (lstm-reg + the hybrid are added per-ticker in main, gated by their
#            own flags — not part of this shared list).
#   groups: paper       — the study + its benchmarks (default)
#           benchmarks  — the naive/classical reference set
#           foundation  — the zero-shot foundation models (Chronos-2, Kronos)
#           all         — every available shared model
# A model that isn't installed is silently skipped (logged), as before.

_GROUPS: dict[str, list[str]] = {
    "benchmarks": ["rw", "rwdrift", "seasonal", "arima", "xgboost"],
    # "paper" is the Prophet/LSTM study (Prophet here; LSTM-reg + hybrid join
    # per-ticker in main) plus the benchmarks it's measured against.
    "paper": ["rw", "rwdrift", "seasonal", "arima", "xgboost", "prophet"],
    "foundation": ["chronos", "kronos"],
    "all": ["rw", "rwdrift", "seasonal", "arima", "xgboost", "prophet", "chronos", "kronos"],
}
_ALL_KEYS: list[str] = _GROUPS["all"]


def resolve_model_keys(spec: str | None) -> list[str]:
    """Expand a --models spec (keys and/or group names) into an ordered key list.

    ``None`` → the default 'paper' group. Unknown tokens raise ValueError. Order
    follows ``_ALL_KEYS`` and de-dupes, so the table layout is stable regardless
    of how the user listed things.
    """
    if spec is None or not spec.strip():
        tokens = ["paper"]
    else:
        tokens = [t for t in spec.replace(",", " ").split() if t]
    wanted: set[str] = set()
    for tok in tokens:
        low = tok.lower()
        if low in _GROUPS:
            wanted.update(_GROUPS[low])
        elif low in _ALL_KEYS:
            wanted.add(low)
        else:
            raise ValueError(
                f"unknown --models token {tok!r}. Keys: {_ALL_KEYS}. Groups: {sorted(_GROUPS)}."
            )
    return [k for k in _ALL_KEYS if k in wanted]


def build_forecasters(keys: list[str], season: int = 5) -> list:
    """Instantiate the selected shared forecasters as (model, label) pairs.

    Only the requested keys are built, and only when their optional library
    imports (graceful skip, logged) — so e.g. asking for chronos without it
    installed yields nothing rather than an error.
    """
    models: list = []
    # Naive forecasters come as a fixed (model, label) list; index by key.
    naive_by_key = {}
    for m, label in default_naive_forecasters(season=season):
        if label == "Random Walk":
            naive_by_key["rw"] = (m, label)
        elif label == "Random Walk + Drift":
            naive_by_key["rwdrift"] = (m, label)
        elif label.startswith("Seasonal Naive"):
            naive_by_key["seasonal"] = (m, label)

    for key in keys:
        if key in naive_by_key:
            models.append(naive_by_key[key])
        elif key == "arima":
            try:
                from engine.arima_model import _STATSMODELS_AVAILABLE, ARIMAForecaster

                if _STATSMODELS_AVAILABLE:
                    models.append((ARIMAForecaster(), "ARIMA"))
            except Exception as e:  # noqa: BLE001
                log.debug("ARIMA unavailable: %s", e)
        elif key == "xgboost":
            try:
                from engine.xgboost_model import _XGBOOST_AVAILABLE, XGBoostForecaster

                if _XGBOOST_AVAILABLE:
                    models.append((XGBoostForecaster(), "XGBoost"))
            except Exception as e:  # noqa: BLE001
                log.debug("XGBoost unavailable: %s", e)
        elif key == "prophet":
            try:
                from engine.prophet_model import _PROPHET_AVAILABLE, ProphetModel

                if _PROPHET_AVAILABLE:
                    models.append((ProphetModel(), "Prophet"))
            except Exception as e:  # noqa: BLE001
                log.debug("Prophet unavailable: %s", e)
        elif key == "chronos":
            try:
                from engine.chronos_model import _CHRONOS_AVAILABLE, Chronos2Model

                if _CHRONOS_AVAILABLE:
                    models.append((Chronos2Model(), "Chronos-2"))
            except Exception as e:  # noqa: BLE001
                log.debug("Chronos-2 unavailable: %s", e)
        elif key == "kronos":
            try:
                from engine.kronos_model import _KRONOS_AVAILABLE, KronosModel

                if _KRONOS_AVAILABLE:
                    models.append((KronosModel(), "Kronos"))
            except Exception as e:  # noqa: BLE001
                log.debug("Kronos unavailable: %s", e)
    # NB: lstm-reg + the residual hybrid are per-ticker (own flags) and added in
    # main, so they're intentionally not built here.
    return models


def _build_macro_prophet(df, macro_panel):
    """Prophet with lag-1-aligned macro regressors for this ticker, or None.

    Macro is aligned onto the ticker's dates and added via Prophet
    add_regressor; the forecast-date regressor value is carried forward from the
    last in-window macro (= macro at t, known at t-1) — leakage-safe.
    """
    try:
        from engine.macro_data import align_macro
        from engine.prophet_model import _PROPHET_AVAILABLE, ProphetModel

        if not _PROPHET_AVAILABLE:
            return None
        dates = df["date"].astype(str) if "date" in df.columns else range(len(df))
        aligned = align_macro(list(dates), macro_panel, lag=1)
        return ProphetModel(macro_df=aligned)
    except Exception as e:  # noqa: BLE001
        log.debug("macro Prophet unavailable: %s", e)
        return None


def _build_macro_xgb(df, macro_panel):
    """XGBoost with lag-1-aligned macro features for this ticker, or None.

    The macro panel is aligned onto the ticker's own dates (forward-filled +
    lagged 1 day by align_macro), so appending macro[t] to a window ending at t
    is leakage-safe. Returns None if xgboost is missing.
    """
    try:
        from engine.macro_data import align_macro
        from engine.xgboost_model import _XGBOOST_AVAILABLE, XGBoostForecaster

        if not _XGBOOST_AVAILABLE:
            return None
        dates = df["date"].astype(str) if "date" in df.columns else range(len(df))
        aligned = align_macro(list(dates), macro_panel, lag=1)
        return XGBoostForecaster(macro_df=aligned)
    except Exception as e:  # noqa: BLE001
        log.debug("macro XGBoost unavailable: %s", e)
        return None


def _build_hybrid(ticker: str, args):
    """Construct the per-ticker Prophet + LSTM-res hybrid, or None if unavailable.

    In ``pretrained`` mode the residual learner loads
    ``models/{ticker}_hybrid_res.pt`` (train it with
    scripts/train_hybrid_residual.py); if those weights are missing the hybrid
    still runs but its learner predicts 0 → it falls back to the Prophet base.
    """
    try:
        from engine.prophet_model import _PROPHET_AVAILABLE, ProphetModel
        from engine.residual_hybrid import ResidualHybrid
        from engine.residual_learners import (
            _TORCH_AVAILABLE,
            LSTMResidualLearner,
            hybrid_residual_path,
        )
    except Exception as e:  # noqa: BLE001
        log.debug("Hybrid imports unavailable: %s", e)
        return None

    if not _PROPHET_AVAILABLE or not _TORCH_AVAILABLE:
        log.warning("--hybrid needs Prophet + torch; skipping the hybrid.")
        return None

    learner = LSTMResidualLearner()
    if args.hybrid_fit == "pretrained":
        loaded = learner.load(hybrid_residual_path(ticker, "models"))
        if not loaded:
            # Per-ticker detail at debug; the run-level warning (in main, before
            # the loop) is the loud one so it isn't lost in per-ticker noise.
            log.debug("%s: no pretrained hybrid weights; hybrid tracks the base.", ticker)
    return ResidualHybrid(
        ProphetModel(),
        learner,
        fit_mode=args.hybrid_fit,
        refit_k=args.hybrid_refit_k,
    )


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
        "--models",
        type=str,
        default="paper",
        help=(
            "Which forecasters to run: comma/space-separated keys "
            "(rw, rwdrift, seasonal, arima, xgboost, prophet, chronos, kronos) "
            "and/or groups (paper [default] = the Prophet/LSTM study + benchmarks; "
            "benchmarks; foundation = Chronos-2 + Kronos; all). Chronos/Kronos run "
            "only via 'all' or by naming them. (LSTM-reg / hybrid have their own "
            "flags: --no-lstm, --hybrid.)"
        ),
    )
    parser.add_argument(
        "--target",
        choices=["level", "log-return"],
        default="level",
        help=(
            "Scoring space. 'level' scores the predicted price; 'log-return' scores "
            "the implied return r=log(P̂/P_t) vs a zero-return (efficient-market) "
            "benchmark. Models are unchanged; only the metric space differs."
        ),
    )
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
        "--macro",
        action="store_true",
        help=(
            "Add an 'XGBoost + macro' variant alongside plain XGBoost — the "
            "price-only vs +macro ablation. Fetches VIX/DXY/Gold/SP500/DGS1 once, "
            "lag-1 aligned per ticker (leakage-safe). Needs xgboost; cached macro "
            "is reused with --no-refresh."
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
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help=(
            "Add the residual hybrid (Prophet base + LSTM residual learner). "
            "Off by default — it's the slowest model. Needs Prophet + torch."
        ),
    )
    parser.add_argument(
        "--hybrid-fit",
        choices=["pretrained", "refit_k", "per_step"],
        default="pretrained",
        help=(
            "Hybrid residual-learner fit cadence. 'pretrained' loads frozen "
            "weights from scripts/train_hybrid_residual.py (fastest); 'refit_k' "
            "retrains every --hybrid-refit-k steps; 'per_step' retrains every "
            "step (slowest, most adaptive)."
        ),
    )
    parser.add_argument(
        "--hybrid-refit-k", type=int, default=21, help="Refit cadence for --hybrid-fit refit_k."
    )
    parser.add_argument("--dir", type=str, default="results", help="Root output directory.")
    args = parser.parse_args()

    tickers = resolve_scope(args, default=ALL_TICKERS)
    scope = scope_label(args)
    run_dir = build_run_dir(Path(args.dir), scope, args.days, args.horizon)

    try:
        model_keys = resolve_model_keys(args.models)
    except ValueError as e:
        parser.error(str(e))
    forecasters = build_forecasters(model_keys)
    labels = ", ".join(label for _, label in forecasters)
    if not args.no_lstm:
        labels += ", LSTM-reg (per-ticker, when weights exist)"
    if args.hybrid:
        labels += f", Prophet+LSTM-res (hybrid, {args.hybrid_fit})"
    if args.macro:
        # Claim a macro variant only when its key was selected AND its lib is
        # importable (--macro implies macro for the chosen macro-capable models).
        if "xgboost" in model_keys:
            try:
                from engine.xgboost_model import _XGBOOST_AVAILABLE
            except Exception:  # noqa: BLE001
                _XGBOOST_AVAILABLE = False
            if _XGBOOST_AVAILABLE:
                labels += ", XGBoost + macro"
            else:
                log.warning("--macro: xgboost not installed; XGBoost + macro skipped.")
        if "prophet" in model_keys:
            try:
                from engine.prophet_model import _PROPHET_AVAILABLE
            except Exception:  # noqa: BLE001
                _PROPHET_AVAILABLE = False
            if _PROPHET_AVAILABLE:
                labels += ", Prophet + macro"
            else:
                log.warning("--macro: prophet not installed; Prophet + macro skipped.")

    print("=" * 92)
    print(
        f" FORECAST HARNESS — {len(tickers)} tickers × {len(forecasters)} models "
        f"× {args.days}d eval, horizon h={args.horizon}, target={args.target}"
    )
    print(f" Models: {labels}")
    print(f" Output: {run_dir.resolve()}/")
    print("=" * 92 + "\n")

    api = StockAppAPI()
    if not args.no_refresh:
        api.refresh_tickers(list(tickers), verbose=False)

    # Up-front, loud warning if the LSTM regressor is enabled but its per-ticker
    # weights are missing — otherwise it just produces no rows and is silently
    # absent from the table.
    if not args.no_lstm:
        from engine.lstm_regressor import regressor_path

        missing_reg = [t for t in tickers if not regressor_path(t, "models").exists()]
        if missing_reg:
            shown = ", ".join(missing_reg[:6]) + ("…" if len(missing_reg) > 6 else "")
            log.warning(
                "LSTM-reg: no weights for %d/%d ticker(s) [%s] — LSTM-reg will be "
                "absent for them. Train: scripts/train_lstm_regressor.py with the "
                "SAME --days %d --horizon %d; or pass --no-lstm to silence this.",
                len(missing_reg),
                len(tickers),
                shown,
                args.days,
                args.horizon,
            )

    # Up-front, loud warning if --hybrid --hybrid-fit pretrained but the
    # per-ticker residual weights are missing — otherwise the hybrid silently
    # degenerates to the Prophet base (predicts 0 residual) and "Prophet +
    # LSTM-res" rows would just equal Prophet, which is easy to misread.
    if args.hybrid and args.hybrid_fit == "pretrained":
        from engine.residual_learners import hybrid_residual_path

        missing = [t for t in tickers if not hybrid_residual_path(t, "models").exists()]
        if missing:
            shown = ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "")
            log.warning(
                "--hybrid --hybrid-fit pretrained: no residual weights for %d/%d "
                "ticker(s) [%s] — the hybrid will fall back to the Prophet base for "
                "them (its 'Prophet + LSTM-res' rows would just equal Prophet). "
                "Train first: scripts/train_hybrid_residual.py with the SAME "
                "--days %d --horizon %d; or use --hybrid-fit per_step (trains on "
                "the fly, no pretraining).",
                len(missing),
                len(tickers),
                shown,
                args.days,
                args.horizon,
            )

    # Macro panel (fetched/cached once, aligned per ticker in the loop).
    macro_panel = None
    if args.macro:
        from engine.macro_data import MacroCache, fetch_macro

        cache = MacroCache()
        macro_panel = cache.load() if args.no_refresh else None
        if macro_panel is None or macro_panel.empty:
            macro_panel = fetch_macro()
            cache.save(macro_panel)
        if macro_panel is None or macro_panel.empty:
            log.warning("--macro: no macro series available; the macro variant will be skipped.")
            macro_panel = None

    # 0 means "no cap" (full expanding window); otherwise the row cap.
    max_train = None if args.max_train == 0 else args.max_train
    bt = ForecastBacktester(
        n_days=args.days,
        horizon=args.horizon,
        refit_k=args.refit_k,
        min_train=args.min_train,
        max_train=max_train,
        target=args.target,
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

        if args.hybrid:
            hyb = _build_hybrid(ticker, args)
            if hyb is not None:
                ticker_models.append((hyb, "Prophet + LSTM-res"))

        # Macro variants are added only for macro-capable models that were
        # actually selected via --models (so --macro "implies macro" for the
        # chosen models, not for ones the user didn't ask for).
        if macro_panel is not None:
            if "xgboost" in model_keys:
                mac = _build_macro_xgb(df, macro_panel)
                if mac is not None:
                    ticker_models.append((mac, "XGBoost + macro"))
            if "prophet" in model_keys:
                mac_p = _build_macro_prophet(df, macro_panel)
                if mac_p is not None:
                    ticker_models.append((mac_p, "Prophet + macro"))

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
