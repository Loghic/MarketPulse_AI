"""
oos_harness.py – Out-of-sample model-selection harness (Plan §1.1).

The problem we're guarding against
----------------------------------

``run_all.py`` reports the **best** model+period per ticker. That is
selection-inflated: across ~9 model variants × 5 periods × 11 tickers
we are picking the maximum of ~500 backtests. Even random predictors
will produce winners that look good in-sample. Worse, those winners
are unstable — only 1 of 7 stable across the 40d/100d runs we ran.

This harness implements the disciplined alternative the plan calls
for:

    1. Pick the best model+period on a **selection window** (the
       second-to-last ``--days`` block of each ticker's history).
    2. Evaluate **only** that winning configuration on the next,
       strictly disjoint **evaluation window** (the most recent
       ``--days`` block).
    3. Report the OOS beat-buy-and-hold rate and median OOS return —
       the honest version of the run_all.py summary.

The two windows never overlap, so the OOS score cannot be inflated
by selecting on the same days we evaluate on.

Usage
-----

    # All stocks, 50-day holdouts, FinBERT news, with baselines as
    # candidates so a baseline beating real models is detectable:
    uv run python scripts/oos_harness.py --stocks --days 50 \\
        --fees 0.03 --buy-hold --no-refresh --sentiment-method finbert

    # Just one ticker, single period, no baselines:
    uv run python scripts/oos_harness.py --tickers NVDA \\
        --periods 5y --days 100 --no-baselines

Output is written under ``results/oos_<scope>_<days>d_..._<timestamp>/``
with three files:

    _oos_per_ticker.csv   — one row per ticker (winner + OOS metrics)
    _oos_summary.csv      — one-line aggregate stats
    _oos_console.txt      — text copy of the printed summary table
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

from config import (  # noqa: E402
    ALL_PERIODS,
    ALL_TICKERS,
    CRYPTO,
    CRYPTO_BENCHMARKS,
    DEFAULT_NEWS_HALF_LIFE_DAYS,
    DEFAULT_NEWS_LOOKBACK_DAYS,
    DEFAULT_SENTIMENT_METHOD,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TRADING_FEE_PCT,
    MODEL_FAMILIES,
    STOCK_BENCHMARKS,
    STOCKS,
)
from engine.backtest_helpers import _family_key, run_single_backtest  # noqa: E402
from engine.backtester import Backtester, BacktestResult  # noqa: E402
from engine.logger import get_logger, progress_bar  # noqa: E402
from interface.api import StockAppAPI  # noqa: E402

log = get_logger("oos_harness")

# Minimum extra rows the underlying Backtester requires on top of n_days.
# Mirrors the guard in backtest_helpers.run_single_backtest.
MIN_OVERHEAD = 20


# ----------------------------------------------------------------------
# Core: select-on-window-A → evaluate-on-window-B for one ticker
# ----------------------------------------------------------------------


def oos_one_ticker(
    api: StockAppAPI,
    ticker: str,
    *,
    n_days: int,
    fee_pct: float,
    stop_loss_pct: float,
    periods: list[str],
    news_lookback_days: int,
    news_half_life_days: float,
    sentiment_method: str | None,
    models: list[str] | None,
    include_baselines: bool,
) -> dict | None:
    """Run the OOS pipeline for one ticker.

    Returns ``None`` if the ticker does not have enough history to fit
    two disjoint ``n_days``-long holdout windows.
    """
    df = api.get_data(ticker, period="max")
    if df.empty:
        log.info(f"{ticker}: no data, skipping.")
        return None

    needed = 2 * n_days + MIN_OVERHEAD
    if len(df) < needed:
        log.info(
            f"{ticker}: only {len(df)} rows, need {needed} for two disjoint {n_days}-day windows."
        )
        return None

    # Selection window = trim the last n_days off; the resulting df's
    # last n_days are exactly the days we will select on.
    df_selection = df.iloc[:-n_days].copy()
    df_evaluation = df

    backtester = Backtester(n_days=n_days, fee_pct=fee_pct, stop_loss_pct=stop_loss_pct)

    # ------------------------------------------------------------------
    # 1) Selection — run every candidate, pick the highest in-sample
    #    total_return across (model, period).
    # ------------------------------------------------------------------
    selection: list[tuple[BacktestResult, str]] = []
    for period in periods:
        try:
            results = run_single_backtest(
                api,
                backtester,
                ticker,
                df_selection,
                period,
                n_days,
                full=False,
                news_lookback_days=news_lookback_days,
                news_half_life_days=news_half_life_days,
                sentiment_method=sentiment_method,
                models=models,
                include_baselines=include_baselines,
            )
        except Exception as e:  # noqa: BLE001
            log.warning(f"{ticker} period={period}: selection raised ({e}); skipping.")
            continue
        for r in results:
            selection.append((r, period))

    if not selection:
        log.info(f"{ticker}: no models produced selection results.")
        return None

    winner_result, winner_period = max(selection, key=lambda pair: pair[0].total_return)
    winner_name = winner_result.model_name
    winner_family = _family_key(winner_name)
    in_sample_return = winner_result.total_return
    in_sample_accuracy = winner_result.accuracy
    in_sample_bh = winner_result.buy_hold_return

    # ------------------------------------------------------------------
    # 2) Evaluation — re-run JUST that winning variant on the disjoint
    #    evaluation window. We filter by family for cheapness, then
    #    find the exact variant by name.
    # ------------------------------------------------------------------
    try:
        eval_results = run_single_backtest(
            api,
            backtester,
            ticker,
            df_evaluation,
            winner_period,
            n_days,
            full=False,
            news_lookback_days=news_lookback_days,
            news_half_life_days=news_half_life_days,
            sentiment_method=sentiment_method,
            models=[winner_family],
            include_baselines=include_baselines,
        )
    except Exception as e:  # noqa: BLE001
        log.warning(f"{ticker}: evaluation raised ({e}); skipping.")
        return None

    eval_match = next((r for r in eval_results if r.model_name == winner_name), None)
    if eval_match is None:
        log.warning(
            f"{ticker}: selection winner {winner_name!r} did not reappear "
            f"on the evaluation window — skipping."
        )
        return None

    return {
        "ticker": ticker,
        "winner_model": winner_name,
        "winner_period": winner_period,
        "winner_family": winner_family,
        "in_sample_return": in_sample_return,
        "in_sample_accuracy": in_sample_accuracy,
        "in_sample_buy_hold": in_sample_bh,
        "oos_return": eval_match.total_return,
        "oos_accuracy": eval_match.accuracy,
        "oos_buy_hold": eval_match.buy_hold_return,
        "oos_sharpe": eval_match.sharpe_ratio,
        "beats_bh_oos": int(eval_match.total_return > eval_match.buy_hold_return),
        "stable": int(eval_match.total_return > 0),
    }


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


def _safe_median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def aggregate(rows: list[dict]) -> dict:
    """Reduce per-ticker rows into the headline OOS summary."""
    if not rows:
        return {
            "tickers": 0,
            "oos_beat_bh_rate": 0.0,
            "median_oos_return": 0.0,
            "mean_oos_return": 0.0,
            "median_in_sample_return": 0.0,
            "in_sample_minus_oos_median": 0.0,
            "median_oos_accuracy": 0.0,
        }
    oos_returns = [r["oos_return"] for r in rows]
    in_sample_returns = [r["in_sample_return"] for r in rows]
    return {
        "tickers": len(rows),
        "oos_beat_bh_rate": sum(r["beats_bh_oos"] for r in rows) / len(rows),
        "median_oos_return": _safe_median(oos_returns),
        "mean_oos_return": sum(oos_returns) / len(oos_returns),
        "median_in_sample_return": _safe_median(in_sample_returns),
        # How much of the in-sample edge disappears OOS? A large positive
        # number is the selection-inflation tax.
        "in_sample_minus_oos_median": _safe_median(
            [r["in_sample_return"] - r["oos_return"] for r in rows]
        ),
        "median_oos_accuracy": _safe_median([r["oos_accuracy"] for r in rows]),
    }


# ----------------------------------------------------------------------
# I/O — output directory and CSVs
# ----------------------------------------------------------------------


def build_run_dir(
    root: Path, scope: str, days: int, fees: float, stop_loss: float, buy_hold: bool
) -> Path:
    parts = ["oos", scope, f"{days}d"]
    if fees > 0:
        parts.append(f"fee{fees * 100:03.0f}")
    if stop_loss > 0:
        parts.append(f"sl{stop_loss:g}")
    if buy_hold:
        parts.append("bh")
    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_dir = root / "_".join(parts)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_per_ticker(run_dir: Path, rows: list[dict]) -> Path:
    out = run_dir / "_oos_per_ticker.csv"
    if not rows:
        out.write_text("")
        return out
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def write_summary(run_dir: Path, summary: dict) -> Path:
    out = run_dir / "_oos_summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return out


# ----------------------------------------------------------------------
# Console output
# ----------------------------------------------------------------------


def print_results(rows: list[dict], summary: dict) -> str:
    """Return the printed banner so we can also persist it to disk."""
    lines: list[str] = []
    add = lines.append
    sep = "=" * 90

    add(sep)
    add(" OUT-OF-SAMPLE HARNESS — selection-window winner → next-disjoint-window check")
    add(sep)
    if not rows:
        add(" No tickers produced both selection and evaluation results.")
        text = "\n".join(lines)
        print(text)
        return text

    add(
        f"  {'TICKER':<10} {'WINNER':<32} {'PERIOD':<5} "
        f"{'IN-SAMPLE':<12} {'OOS RET':<11} {'OOS B&H':<11} {'BEAT B&H?'}"
    )
    add("  " + "-" * 86)
    for r in rows:
        marker = "✓" if r["beats_bh_oos"] else "✗"
        add(
            f"  {r['ticker']:<10} {r['winner_model'][:32]:<32} "
            f"{r['winner_period']:<5} "
            f"{r['in_sample_return']:<+12.4%} "
            f"{r['oos_return']:<+11.4%} "
            f"{r['oos_buy_hold']:<+11.4%} {marker}"
        )

    add("  " + "-" * 86)
    add(" AGGREGATE")
    add("  " + "-" * 86)
    add(f"  Tickers evaluated         : {summary['tickers']}")
    add(f"  OOS beat-B&H rate         : {summary['oos_beat_bh_rate']:.1%}")
    add(f"  Median OOS return         : {summary['median_oos_return']:+.4%}")
    add(f"  Mean OOS return           : {summary['mean_oos_return']:+.4%}")
    add(f"  Median in-sample return   : {summary['median_in_sample_return']:+.4%}")
    add(
        f"  Selection-inflation gap   : "
        f"{summary['in_sample_minus_oos_median']:+.4%}  (in-sample − OOS, median)"
    )
    add(f"  Median OOS accuracy       : {summary['median_oos_accuracy']:.4f}")
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
            "Phase-1.1 OOS harness: pick the best model+period on a "
            "selection window and re-evaluate it on the next disjoint "
            "window. Reports the honest beat-B&H rate."
        )
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stocks", action="store_true")
    group.add_argument("--crypto", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--days", type=int, default=50)
    parser.add_argument(
        "--fees",
        type=float,
        default=DEFAULT_TRADING_FEE_PCT,
        help=f"Trading fee %% per side (default: {DEFAULT_TRADING_FEE_PCT}).",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=DEFAULT_STOP_LOSS_PCT,
        help="Stop-loss %% (0 = disabled).",
    )
    parser.add_argument(
        "--buy-hold",
        action="store_true",
        help="Compute buy-and-hold benchmarks for the report header.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip data download; use only cached DB data.",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=ALL_PERIODS,
        default=list(ALL_PERIODS),
        help="Lookback periods to consider for selection (default: all).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_FAMILIES,
        default=None,
        help="Restrict candidate families (default: all).",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip the naive baselines. Default: baselines included.",
    )
    parser.add_argument(
        "--sentiment-method",
        choices=["vader", "finbert", "naive"],
        default=DEFAULT_SENTIMENT_METHOD,
    )
    parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=DEFAULT_NEWS_LOOKBACK_DAYS,
    )
    parser.add_argument(
        "--news-half-life-days",
        type=float,
        default=DEFAULT_NEWS_HALF_LIFE_DAYS,
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Root output directory (default: results/).",
    )
    args = parser.parse_args()

    # Resolve the ticker scope.
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        scope = "custom"
    elif args.stocks:
        tickers, scope = STOCKS, "stocks"
    elif args.crypto:
        tickers, scope = CRYPTO, "crypto"
    elif args.all:
        tickers, scope = ALL_TICKERS, "all"
    else:
        tickers, scope = ALL_TICKERS, "all"

    run_dir = build_run_dir(
        Path(args.dir), scope, args.days, args.fees, args.stop_loss, args.buy_hold
    )

    print(f"{'=' * 90}")
    print(
        f" OOS HARNESS — {len(tickers)} tickers × {len(args.periods)} periods "
        f"× {args.days}d selection + {args.days}d evaluation"
    )
    print(f" Output: {run_dir.resolve()}/")
    print(f"{'=' * 90}\n")

    api = StockAppAPI(sentiment_method=args.sentiment_method)
    if not args.no_refresh:
        all_to_refresh = list(set(tickers + STOCK_BENCHMARKS + CRYPTO_BENCHMARKS))
        api.refresh_tickers(
            all_to_refresh,
            news_lookback_days=args.news_lookback_days,
            sentiment_method=args.sentiment_method,
        )

    rows: list[dict] = []
    for ticker in progress_bar(tickers, desc="OOS harness"):
        row = oos_one_ticker(
            api,
            ticker,
            n_days=args.days,
            fee_pct=args.fees,
            stop_loss_pct=args.stop_loss,
            periods=args.periods,
            news_lookback_days=args.news_lookback_days,
            news_half_life_days=args.news_half_life_days,
            sentiment_method=args.sentiment_method,
            models=args.models,
            include_baselines=not args.no_baselines,
        )
        if row is not None:
            rows.append(row)

    summary = aggregate(rows)
    text = print_results(rows, summary)
    write_per_ticker(run_dir, rows)
    write_summary(run_dir, summary)
    (run_dir / "_oos_console.txt").write_text(text + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
