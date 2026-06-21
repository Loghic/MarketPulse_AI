"""
oos_harness.py – Out-of-sample model-selection harness.

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

from cli_helpers import (  # noqa: E402
    add_common_run_args,
    add_model_filter_args,
    add_news_args,
    add_scope_args,
    add_strategy_args,
    resolve_scope,
    scope_label,
)
from config import (  # noqa: E402
    ALL_TICKERS,
    CRYPTO_BENCHMARKS,
    STOCK_BENCHMARKS,
)
from engine.backtest_helpers import (  # noqa: E402
    _family_key,
    compute_benchmarks,
    run_single_backtest,
)
from engine.backtester import Backtester, BacktestResult  # noqa: E402
from engine.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    pairs_from_days,
)
from engine.logger import get_logger, progress_bar  # noqa: E402
from engine.significance import significance_for_days  # noqa: E402
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
    min_confidence: float = 0.0,
    turnover_fees: bool = False,
    hold_days: int = 1,
    position_mode: bool = False,
) -> dict | None:
    """Run the OOS pipeline for one ticker.

    Returns ``None`` if the ticker does not have enough history to fit
    two disjoint ``n_days``-long holdout windows.

    ``min_confidence`` applies the **same** confidence gate
    to both the selection and the evaluation window — so the question it
    answers is the honest one: "if I commit to gating at θ, does the OOS
    edge survive?" We deliberately do *not* sweep θ here; letting the
    harness pick the best-looking θ on the evaluation window would
    reintroduce exactly the selection inflation the harness exists to
    eliminate. To compare thresholds, run the harness once per θ.
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

    backtester = Backtester(
        n_days=n_days,
        fee_pct=fee_pct,
        stop_loss_pct=stop_loss_pct,
        min_confidence=min_confidence,
        turnover_fees=turnover_fees,
        hold_days=hold_days,
        position_mode=position_mode,
    )

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

    # ------------------------------------------------------------------
    # 3) Calibration + significance on the OOS evaluation window only.
    #    These are descriptive: they don't change which config we picked,
    #    they tell us whether the OOS result is calibrated / real.
    # ------------------------------------------------------------------
    eval_days = eval_match.days
    pairs = pairs_from_days(eval_days)
    brier = brier_score(pairs)
    ece = expected_calibration_error(pairs)

    # Significance over real directional traded days only (a HOLD buy-hold or a
    # FLAT sit-out makes no UP/DOWN call to test).
    dir_days = [d for d in eval_days if d.traded and d.predicted in ("UP", "DOWN")]
    sig = significance_for_days(
        [d.predicted for d in dir_days],
        [d.actual for d in dir_days],
        [d.trade_pnl_net for d in dir_days],
    )

    # Market-benchmark return (SPY/QQQ/BTC…) over the OOS eval window — lets us
    # compare the winner to its index, not just to holding the ticker itself.
    try:
        bench_map = compute_benchmarks(api, ticker, eval_days)
    except Exception:  # noqa: BLE001
        bench_map = {}
    # Representative single number = best (max) benchmark return, or 0 if none.
    oos_benchmark = max(bench_map.values()) if bench_map else 0.0

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
        # Market benchmark (best of the ticker's index set) over the OOS window,
        # and whether the winner beat it. 0.0 when no benchmark data.
        "oos_benchmark": round(oos_benchmark, 8),
        "beats_benchmark_oos": int(eval_match.total_return > oos_benchmark) if bench_map else 0,
        # Confidence gating + calibration — only meaningful
        # columns when --min-confidence > 0, but always emitted so the CSV
        # schema is stable.
        "min_confidence": min_confidence,
        "oos_coverage": eval_match.coverage,
        "oos_traded_days": eval_match.test_days,
        "oos_sat_out": eval_match.sat_out_count,
        "oos_brier": brier,
        "oos_ece": ece,
        # Significance on the OOS traded days.
        "oos_binomial_p": sig.binomial_p,
        "oos_acc_ci_lo": sig.wilson.lo,
        "oos_acc_ci_hi": sig.wilson.hi,
    }


# ----------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------


def _safe_median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def aggregate(rows: list[dict]) -> dict:
    """Reduce per-ticker rows into the headline OOS summary.

    When a confidence gate is active (``min_confidence > 0`` on any row),
    a second block of gating-aware aggregates is appended: median OOS
    coverage, the median OOS *traded-day* accuracy (the headline "does
    gating isolate a predictive subset?" number), median Brier/ECE, and
    the count of tickers whose OOS accuracy is significant at p < 0.05.
    """
    if not rows:
        return {
            "tickers": 0,
            "oos_beat_bh_rate": 0.0,
            "median_oos_return": 0.0,
            "mean_oos_return": 0.0,
            "median_in_sample_return": 0.0,
            "in_sample_minus_oos_median": 0.0,
            "median_oos_accuracy": 0.0,
            "oos_beat_benchmark_rate": 0.0,
            "min_confidence": 0.0,
            "median_oos_coverage": 1.0,
            "median_oos_brier": 0.0,
            "median_oos_ece": 0.0,
            "tickers_significant_p05": 0,
        }
    oos_returns = [r["oos_return"] for r in rows]
    in_sample_returns = [r["in_sample_return"] for r in rows]
    # Backward-compatible: older rows (pre-gating) lack the new keys.
    theta = max((r.get("min_confidence", 0.0) for r in rows), default=0.0)
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
        # Beat the market index (not the ticker's own B&H), over rows that have
        # a benchmark; 0 when none do.
        "oos_beat_benchmark_rate": (
            sum(r.get("beats_benchmark_oos", 0) for r in rows)
            / sum(1 for r in rows if r.get("oos_benchmark", 0.0))
            if any(r.get("oos_benchmark", 0.0) for r in rows)
            else 0.0
        ),
        # --- Gating / calibration block ---
        "min_confidence": theta,
        "median_oos_coverage": _safe_median([r.get("oos_coverage", 1.0) for r in rows]),
        "median_oos_brier": _safe_median([r.get("oos_brier", 0.0) for r in rows]),
        "median_oos_ece": _safe_median([r.get("oos_ece", 0.0) for r in rows]),
        "tickers_significant_p05": sum(1 for r in rows if r.get("oos_binomial_p", 1.0) < 0.05),
    }


# ----------------------------------------------------------------------
# I/O — output directory and CSVs
# ----------------------------------------------------------------------


def build_run_dir(
    root: Path,
    scope: str,
    days: int,
    fees: float,
    stop_loss: float,
    buy_hold: bool,
    min_confidence: float = 0.0,
) -> Path:
    parts = ["oos", scope, f"{days}d"]
    if fees > 0:
        parts.append(f"fee{fees * 100:03.0f}")
    if stop_loss > 0:
        parts.append(f"sl{stop_loss:g}")
    if min_confidence > 0:
        parts.append(f"mc{min_confidence * 100:03.0f}")
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

    gated = summary.get("min_confidence", 0.0) > 0
    has_bench = any(r.get("oos_benchmark", 0.0) for r in rows)

    header = (
        f"  {'TICKER':<10} {'WINNER':<32} {'PERIOD':<5} "
        f"{'IN-SAMPLE':<12} {'OOS RET':<11} {'OOS B&H':<11} {'BEAT B&H?'}"
    )
    if has_bench:
        header += f"  {'OOS BENCH':<11} {'BEAT BM?'}"
    if gated:
        header += f"  {'COVERAGE':<13} {'OOS ACC':<8}"
    add(header)
    width = 86 + (20 if has_bench else 0) + (24 if gated else 0)
    add("  " + "-" * width)
    for r in rows:
        marker = "✓" if r["beats_bh_oos"] else "✗"
        line = (
            f"  {r['ticker']:<10} {r['winner_model'][:32]:<32} "
            f"{r['winner_period']:<5} "
            f"{r['in_sample_return']:<+12.4%} "
            f"{r['oos_return']:<+11.4%} "
            f"{r['oos_buy_hold']:<+11.4%} {marker}"
        )
        if has_bench:
            bm = "✓" if r.get("beats_benchmark_oos") else "✗"
            line += f"  {r.get('oos_benchmark', 0.0):<+11.4%} {bm}"
        if gated:
            cov = f"{r.get('oos_traded_days', 0)}/{r.get('oos_traded_days', 0) + r.get('oos_sat_out', 0)} ({r.get('oos_coverage', 1.0):.0%})"
            line += f"  {cov:<13} {r.get('oos_accuracy', 0.0):<8.1%}"
        add(line)

    add("  " + "-" * width)
    add(" AGGREGATE")
    add("  " + "-" * 86)
    add(f"  Tickers evaluated         : {summary['tickers']}")
    add(f"  OOS beat-B&H rate         : {summary['oos_beat_bh_rate']:.1%}")
    if has_bench:
        add(
            f"  OOS beat-benchmark rate   : {summary.get('oos_beat_benchmark_rate', 0.0):.1%}"
            "  (vs the market index, not the ticker)"
        )
    add(f"  Median OOS return         : {summary['median_oos_return']:+.4%}")
    add(f"  Mean OOS return           : {summary['mean_oos_return']:+.4%}")
    add(f"  Median in-sample return   : {summary['median_in_sample_return']:+.4%}")
    add(
        f"  Selection-inflation gap   : "
        f"{summary['in_sample_minus_oos_median']:+.4%}  (in-sample − OOS, median)"
    )
    add(f"  Median OOS accuracy       : {summary['median_oos_accuracy']:.4f}")
    if gated:
        add("  " + "-" * 86)
        add(f"  Confidence gate θ         : {summary['min_confidence']:.2f}")
        add(
            f"  Median OOS coverage       : {summary['median_oos_coverage']:.1%}"
            "  (share of days actually traded)"
        )
        add(
            f"  Median OOS Brier / ECE    : "
            f"{summary['median_oos_brier']:.4f} / {summary['median_oos_ece']:.4f}"
            "  (lower = better calibrated)"
        )
        add(
            f"  Tickers signif. (p<0.05)  : "
            f"{summary['tickers_significant_p05']}/{summary['tickers']}"
            "  (OOS accuracy ≠ 0.5, binomial)"
        )
        add(
            "  → Gating helps only if traded-day OOS accuracy clears 0.5 "
            "AND OOS return improves vs an ungated run."
        )
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
            "OOS harness: pick the best model+period on a "
            "selection window and re-evaluate it on the next disjoint "
            "window. Reports the honest beat-B&H rate."
        )
    )
    add_scope_args(parser)
    add_common_run_args(parser, days_default=50)
    add_model_filter_args(parser)
    # OOS does NOT sweep stop-loss (that would re-inflate selection), so the
    # shared strategy group is requested in single-SL mode. --min-confidence
    # gets the OOS-specific both-windows / not-swept wording.
    add_strategy_args(
        parser,
        sl_sweep=False,
        min_confidence_help=(
            "Confidence gate: sit out days below THETA confidence "
            "(0..1) on BOTH the selection and evaluation windows. Answers "
            "'does the OOS edge survive if I commit to gating at θ?'. "
            "θ is NOT swept here (that would re-inflate selection) — run once "
            "per θ to compare. 0 = trade every day (default)."
        ),
    )
    add_news_args(parser)
    parser.add_argument(
        "--dir",
        type=str,
        default="results",
        help="Root output directory (default: results/).",
    )
    args = parser.parse_args()

    # Resolve the ticker scope via the shared selectors (so commodities /
    # indices / fx combine just like the other CLIs). Default: every ticker.
    tickers = resolve_scope(args, default=ALL_TICKERS)
    scope = scope_label(args)

    run_dir = build_run_dir(
        Path(args.dir),
        scope,
        args.days,
        args.fees,
        args.stop_loss,
        args.buy_hold,
        min_confidence=args.min_confidence,
    )

    print(f"{'=' * 90}")
    print(
        f" OOS HARNESS — {len(tickers)} tickers × {len(args.periods)} periods "
        f"× {args.days}d selection + {args.days}d evaluation"
    )
    if args.min_confidence > 0:
        print(
            f" Confidence gate θ={args.min_confidence:.2f} applied to BOTH windows "
            f"(fixed, not swept)"
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
            min_confidence=args.min_confidence,
            turnover_fees=args.turnover_fees,
            hold_days=args.hold_days,
            position_mode=args.position_mode,
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
