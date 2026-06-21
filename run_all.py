"""
run_all.py – Batch backtest runner.

Runs --compare-periods for each ticker and saves results to organized
subdirectories under results/.

Output structure:
    results/
    ├── stocks_50d_fee003_bh/
    │   ├── AAPL.csv
    │   ├── MSFT.csv
    │   ├── ...
    │   └── _summary.csv
    ├── crypto_50d_fee015_sl2/
    │   ├── BTC-USD.csv
    │   └── _summary.csv
    └── all_20d/
        ├── AAPL.csv
        ├── BTC-USD.csv
        └── _summary.csv

Usage:
    uv run python run_all.py --days 20
    uv run python run_all.py --stocks --days 50 --fees 0.03 --buy-hold
    uv run python run_all.py --crypto --days 50 --fees 0.15 --stop-loss 3
    uv run python run_all.py --all --days 50 --fees 0.1 --stop-loss 2 --buy-hold
    uv run python run_all.py --tickers AAPL NVDA --days 20 --buy-hold
    uv run python run_all.py --stocks --days 100 --periods 1y 2y 5y --models knn lstm chronos --buy-hold
"""

import argparse
import csv
from pathlib import Path

from cli_helpers import add_scope_args, resolve_scope, scope_label
from config import (
    ALL_PERIODS,
    ALL_TICKERS,
    CRYPTO_BENCHMARKS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_NEWS_HALF_LIFE_DAYS,
    DEFAULT_NEWS_LOOKBACK_DAYS,
    DEFAULT_SENTIMENT_METHOD,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TRADING_FEE_PCT,
    MODEL_FAMILIES,
    STOCK_BENCHMARKS,
)
from engine.backtest_helpers import (
    compute_benchmarks,
    filter_by_period,
    pf_str,
    result_to_summary_row,
    run_single_backtest,
)
from engine.backtester import Backtester
from engine.logger import get_logger, progress_bar
from interface.api import StockAppAPI

log = get_logger("run_all")


RESULTS_DIR = Path("results")


def _family(name: str) -> str:
    """Map a model variant name to its family (for the time rollup)."""
    for f in ("k-NN", "LinReg", "LSTM", "Prophet", "Chronos-2", "Kronos"):
        if name.startswith(f):
            return f
    return "Other"


def build_dir_name(
    scope: str,
    n_days: int,
    fee_pct: float,
    stop_loss_pct: float,
    buy_hold: bool,
    sentiment_method: str | None = None,
    min_confidence: float = 0.0,
) -> str:
    """
    Build subdirectory name from run parameters.

    Examples:
        stocks_50d_fee003_bh
        crypto_20d_fee015_sl3
        all_50d_fee010_sl2_bh
        custom_20d
        stocks_100d_fee003_bh_finbert
        stocks_100d_fee003_mc060
    """
    parts = [scope, f"{n_days}d"]
    if fee_pct > 0:
        parts.append(f"fee{fee_pct * 100:03.0f}")
    if stop_loss_pct > 0:
        parts.append(f"sl{stop_loss_pct:g}")
    if min_confidence > 0:
        parts.append(f"mc{min_confidence * 100:03.0f}")
    if buy_hold:
        parts.append("bh")
    # Only tag the dir when the user explicitly overrode the default scorer,
    # so legacy result directories keep their old names.
    if sentiment_method and sentiment_method != DEFAULT_SENTIMENT_METHOD:
        parts.append(sentiment_method)
    return "_".join(parts)


def run_ticker_comparison(
    api,
    ticker,
    n_days,
    fee_pct,
    stop_loss_pct,
    buy_hold,
    run_dir,
    *,
    periods: list[str] | None = None,
    model_time: dict | None = None,
    news_lookback_days: int = DEFAULT_NEWS_LOOKBACK_DAYS,
    news_half_life_days: float = DEFAULT_NEWS_HALF_LIFE_DAYS,
    sentiment_method: str | None = None,
    models=None,
    include_baselines: bool = True,
    min_confidence: float = 0.0,
):
    """Run compare-periods for one ticker, save CSV, return best combo."""

    periods = periods or list(ALL_PERIODS)
    df = api.get_data(ticker, period="max")
    if df.empty:
        print(f"  Skipping {ticker}: no data")
        return None

    backtester = Backtester(
        n_days=n_days,
        fee_pct=fee_pct,
        stop_loss_pct=stop_loss_pct,
        min_confidence=min_confidence,
    )
    all_rows = []
    all_combos = []
    bench = None

    for period in periods:
        results = run_single_backtest(
            api,
            backtester,
            ticker,
            df,
            period,
            n_days,
            full=False,
            news_lookback_days=news_lookback_days,
            news_half_life_days=news_half_life_days,
            sentiment_method=sentiment_method,
            models=models,
            include_baselines=include_baselines,
        )
        if not results:
            filtered = filter_by_period(df, period)
            print(f"  {ticker} period={period}: skipped ({len(filtered)} rows)")
            continue

        # Compute benchmarks once (same test dates across periods)
        if bench is None and buy_hold:
            bench = compute_benchmarks(api, ticker, results[0].days)

        for r in results:
            if model_time is not None:
                model_time[r.model_name] = model_time.get(r.model_name, 0.0) + getattr(
                    r, "elapsed_seconds", 0.0
                )
            all_rows.append(result_to_summary_row(r, ticker, period, benchmarks=bench))
            combo = {
                "ticker": ticker,
                "period": period,
                "model": r.model_name,
                "accuracy": r.accuracy,
                "total_return": r.total_return,
                "buy_hold_return": r.buy_hold_return,
                "profit_factor": r.profit_factor,
                "max_drawdown": r.max_drawdown,
                "sharpe_ratio": r.sharpe_ratio,
                "sortino_ratio": r.sortino_ratio,
                "buy_hold_max_drawdown": r.buy_hold_max_drawdown,
                "fee_pct": r.fee_pct,
                "stop_loss_pct": r.stop_loss_pct,
                "stopped_out": r.stopped_out_count,
                "correct": r.correct,
                "total": r.test_days,
                "win_trades": r.win_trades,
                "loss_trades": r.loss_trades,
                "longest_win_streak": r.longest_win_streak,
                "longest_loss_streak": r.longest_loss_streak,
                "avg_win_streak": r.avg_win_streak,
                "avg_loss_streak": r.avg_loss_streak,
            }
            if bench:
                for bname, bret in bench.items():
                    combo[f"bench_{bname}"] = bret
            all_combos.append(combo)

    if not all_rows:
        return None

    # Save individual ticker CSV
    ticker_file = run_dir / f"{ticker}.csv"
    if all_rows:
        all_keys = {}
        for row in all_rows:
            for k in row:
                all_keys[k] = None
        with open(ticker_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_keys.keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    # Find best combo by return
    best = max(all_combos, key=lambda c: (c["total_return"], c["accuracy"]))
    pf = best["profit_factor"]

    line = (
        f"  {ticker:<10} → {best['model']:<25} {best['period']:<6} "
        f"return={best['total_return']:+.4%}  PF={pf_str(pf)}  "
        f"DD={best['max_drawdown']:+.2%}  Sharpe={best['sharpe_ratio']:.2f}  "
        f"W{best['longest_win_streak']}/L{best['longest_loss_streak']}"
    )
    if buy_hold:
        bh = best["buy_hold_return"]
        beats = "✓" if best["total_return"] > bh else "✗"
        line += f"  B&H={bh:+.4%} {beats}"
    if bench:
        for bname, bret in bench.items():
            marker = "✓" if best["total_return"] > bret else "✗"
            line += f"  {bname}={bret:+.2%}{marker}"
    if best.get("stopped_out", 0) > 0:
        line += f"  SL={best['stopped_out']}×"
    print(line)

    return best


def main():
    parser = argparse.ArgumentParser(
        description="MarketPulse AI – Batch backtest (one CSV per ticker)"
    )
    add_scope_args(parser)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument(
        "--periods",
        nargs="+",
        choices=ALL_PERIODS,
        default=list(ALL_PERIODS),
        help=f"Periods to test (default: all {ALL_PERIODS}). e.g. --periods 1y 2y 5y to skip max.",
    )
    parser.add_argument(
        "--fees",
        type=float,
        default=DEFAULT_TRADING_FEE_PCT,
        help=f"Fee %% per side (default: {DEFAULT_TRADING_FEE_PCT})",
    )
    parser.add_argument(
        "--stop-loss", type=float, default=DEFAULT_STOP_LOSS_PCT, help="Stop-loss %% (0=disabled)"
    )
    parser.add_argument("--buy-hold", action="store_true", help="Include buy-and-hold comparison")
    parser.add_argument(
        "--no-refresh", action="store_true", help="Skip data download, use only cached data from DB"
    )
    parser.add_argument(
        "--dir", type=str, default="results", help="Root output directory (default: results/)"
    )
    parser.add_argument(
        "--sentiment-method",
        choices=["vader", "finbert", "naive"],
        default=DEFAULT_SENTIMENT_METHOD,
        help=(
            "Sentiment scorer for the '+ News' variants. Only news rows scored "
            f"with this method (or NULL legacy rows) are used. Default: {DEFAULT_SENTIMENT_METHOD}."
        ),
    )
    parser.add_argument(
        "--news-source",
        nargs="+",
        choices=["yahoo", "gdelt"],
        default=None,
        help=(
            "News provider(s) used by the upfront refresh step (when --no-refresh "
            "is NOT set). Defaults to config.DEFAULT_NEWS_SOURCES."
        ),
    )

    parser.add_argument(
        "--news-history-days",
        type=int,
        default=DEFAULT_NEWS_LOOKBACK_DAYS,
        help=(
            "How many days of news history to PULL during the upfront refresh "
            "(bulk fetch). Yahoo caps at ~7 days; GDELT honours larger values "
            "up to 250 articles per call. Distinct from --news-lookback-days, "
            "which is the per-day window applied inside the backtest. "
            "Ignored with --no-refresh. Default: %(default)s."
        ),
    )

    parser.add_argument(
        "--force-news",
        action="store_true",
        help=(
            "Bypass the 'already fetched today' cache during the upfront refresh "
            "and re-pull news from the provider. Use when adding a second source, "
            "or re-scoring the same headlines with a different --sentiment-method, "
            "on the same day. Not needed for --news-history-days > 7 (that already "
            "bypasses the cache). Ignored with --no-refresh."
        ),
    )

    parser.add_argument(
        "--news-lookback-days",
        type=int,
        default=DEFAULT_NEWS_LOOKBACK_DAYS,
        help=(
            "Per-day backtest window: only news published in the N days before "
            "each prediction date contributes to sentiment "
            "(default: %(default)s, 0 = unbounded)."
        ),
    )
    parser.add_argument(
        "--news-half-life-days",
        type=float,
        default=DEFAULT_NEWS_HALF_LIFE_DAYS,
        help=(
            "Exponential decay half-life for per-day sentiment weighting "
            "(default: %(default)s, 0 = no decay)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_FAMILIES,
        default=None,
        help="Only run these model families (default: all). e.g. --models knn lstm chronos",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help=(
            "Skip the naive baselines (AlwaysLong, PreviousDay, 5/20-Day "
            "Momentum, Random). Default: baselines included."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        metavar="THETA",
        help=(
            "Confidence gate: sit out days below THETA confidence "
            "(0..1). Sat-out days are flat and excluded from accuracy. "
            "0 = trade every day (default)."
        ),
    )
    args = parser.parse_args()

    tickers = resolve_scope(args, default=ALL_TICKERS)
    scope = scope_label(args)

    # Build output directory
    dir_name = build_dir_name(
        scope,
        args.days,
        args.fees,
        args.stop_loss,
        args.buy_hold,
        sentiment_method=args.sentiment_method,
        min_confidence=args.min_confidence,
    )
    run_dir = Path(args.dir) / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fee_info = f", fees={args.fees}% per side" if args.fees > 0 else ""
    sl_info = f", SL={args.stop_loss}%" if args.stop_loss > 0 else ""
    bh_info = ", vs buy-and-hold" if args.buy_hold else ""

    print(f"{'=' * 80}")
    print(
        f" BATCH BACKTEST: {len(tickers)} tickers × {len(args.periods)} periods "
        f"× {args.days} days{fee_info}{sl_info}{bh_info}"
    )
    print(f" Output: {run_dir.resolve()}/")
    print(f"{'=' * 80}\n")

    api = StockAppAPI(
        sentiment_method=args.sentiment_method,
        news_sources=args.news_source,
    )

    if not args.no_refresh:
        all_to_refresh = list(set(tickers + STOCK_BENCHMARKS + CRYPTO_BENCHMARKS))
        api.refresh_tickers(
            all_to_refresh,
            news_lookback_days=args.news_history_days,
            sentiment_method=args.sentiment_method,
            news_sources=args.news_source,
            force_news_refresh=args.force_news,
        )

    best_per_ticker = []
    model_time: dict[str, float] = {}

    for ticker in progress_bar(tickers, desc="Batch backtest"):
        best = run_ticker_comparison(
            api,
            ticker,
            args.days,
            args.fees,
            args.stop_loss,
            args.buy_hold,
            run_dir,
            periods=args.periods,
            model_time=model_time,
            news_lookback_days=args.news_lookback_days,
            news_half_life_days=args.news_half_life_days,
            sentiment_method=args.sentiment_method,
            models=args.models,
            include_baselines=not args.no_baselines,
            min_confidence=args.min_confidence,
        )
        if best:
            best_per_ticker.append(best)

    # --- Summary ---
    if best_per_ticker:
        summary_file = run_dir / "_summary.csv"
        # Collect all possible keys (benchmark columns vary per ticker)
        all_keys = {}
        for b in best_per_ticker:
            for k in b:
                all_keys[k] = None
        fieldnames = list(all_keys.keys())

        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(best_per_ticker)

        print(f"\n{'=' * 80}")
        print(" SUMMARY: Best model+period per ticker")
        print(f"{'=' * 80}")

        header = (
            f"\n  {'TICKER':<10} {'MODEL':<25} {'PERIOD':<8} "
            f"{'RETURN':<12} {'PF':<8} {'MAX DD':<10} {'SHARPE':<8}"
        )
        if args.stop_loss > 0:
            header += f" {'SL':<5}"
        if args.buy_hold:
            header += f" {'B&H':<12} {'BEAT?'}"
        print(header)
        divider_len = 84 + (7 if args.stop_loss > 0 else 0) + (18 if args.buy_hold else 0)
        print(f"  {'-' * divider_len}")

        for b in best_per_ticker:
            line = (
                f"  {b['ticker']:<10} {b['model']:<25} {b['period']:<8} "
                f"{b['total_return']:<+12.4%} "
                f"{pf_str(b['profit_factor']):<8} "
                f"{b['max_drawdown']:<+10.4%} "
                f"{b['sharpe_ratio']:<8.2f}"
            )
            if args.stop_loss > 0:
                line += f" {b.get('stopped_out', 0):<5}"
            if args.buy_hold:
                bh = b["buy_hold_return"]
                beats = "✓" if b["total_return"] > bh else "✗"
                line += f" {bh:<+12.4%} {beats}"
            print(line)

        overall = max(best_per_ticker, key=lambda b: b["total_return"])
        print(
            f"\n  ★ Best return:  {overall['ticker']} + {overall['model']} + "
            f"{overall['period']} → {overall['total_return']:+.4%} "
            f"(PF {pf_str(overall['profit_factor'])}, "
            f"DD {overall['max_drawdown']:+.4%})"
        )

        best_sharpe = max(best_per_ticker, key=lambda b: b["sharpe_ratio"])
        print(
            f"  ★ Best Sharpe:  {best_sharpe['ticker']} + {best_sharpe['model']} + "
            f"{best_sharpe['period']} → Sharpe {best_sharpe['sharpe_ratio']:.2f} "
            f"(return {best_sharpe['total_return']:+.4%})"
        )

        if args.buy_hold:
            beating_bh = sum(1 for b in best_per_ticker if b["total_return"] > b["buy_hold_return"])
            print(f"  Models beating Buy&Hold: {beating_bh}/{len(best_per_ticker)}")

        print(f"\n  Results: {run_dir}/")
        print(f"  Summary: {summary_file}")

        # --- Time-by-family rollup (compute time vs. how often a family won) ---
    if model_time:
        fam_time: dict[str, float] = {}
        for name, secs in model_time.items():
            fam_time[_family(name)] = fam_time.get(_family(name), 0.0) + secs
        fam_wins: dict[str, int] = {}
        for b in best_per_ticker:
            fam_wins[_family(b["model"])] = fam_wins.get(_family(b["model"]), 0) + 1
        total_t = sum(fam_time.values())

        print(f"\n{'=' * 80}")
        print(" TIME BY MODEL FAMILY  (compute time across all tickers × periods)")
        print(f"{'=' * 80}")
        print(f"  {'FAMILY':<12} {'TIME':>10} {'SHARE':>8} {'WINS':>6}")
        print(f"  {'-' * 40}")
        for fam in sorted(fam_time, key=lambda k: fam_time[k], reverse=True):
            secs = fam_time[fam]
            share = secs / total_t if total_t else 0.0
            print(f"  {fam:<12} {secs:>9.1f}s {share:>7.0%} {fam_wins.get(fam, 0):>6}")
        print(f"  {'-' * 40}")
        print(f"  {'TOTAL':<12} {total_t:>9.1f}s")
        print(
            "\n  WINS = times this family had the best return for a ticker "
            f"(of {len(best_per_ticker)})."
        )
        print("  High SHARE + 0 WINS → a candidate to drop for speed.")

    print(f"\n{'*' * 80}")


if __name__ == "__main__":
    main()
