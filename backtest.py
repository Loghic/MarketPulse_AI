"""
backtest.py – CLI for running walk-forward backtests.

Usage:
    uv run python backtest.py
    uv run python backtest.py --days 20 --fees 0.1
    uv run python backtest.py --full --period 1y --buy-hold
    uv run python backtest.py --compare-periods --output results.csv
    uv run python backtest.py --stocks --compare-periods --buy-hold --fees 0.05
    uv run python backtest.py --tickers NVDA --compare-periods --periods 1y 2y --days 30 --timing
    uv run python backtest.py --tickers NVDA --compare-periods --periods 1y 2y --models knn lstm chronos --timing
"""

import argparse

from cli_helpers import (
    add_common_run_args,
    add_model_filter_args,
    add_scope_args,
    add_strategy_args,
    resolve_scope,
    resolve_sl_levels,
)
from config import (
    ALL_PERIODS,
    ALL_TICKERS,
    CRYPTO_BENCHMARKS,
    DEFAULT_BACKTEST_DAYS,
    DEFAULT_PERIOD,
    STOCK_BENCHMARKS,
)
from engine.backtest_helpers import (
    compute_benchmarks,
    export_rows,
    filter_by_period,
    pf_str,
    print_confidence_calibration,
    print_confidence_sweep,
    print_consensus,
    print_direction_accuracy,
    print_next_day_forecast,
    print_profit_analysis,
    print_significance,
    print_summary_table,
    print_timing_table,
    result_to_daily_rows,
    result_to_summary_row,
    run_single_backtest,
)
from engine.backtester import Backtester
from engine.logger import get_logger, progress_bar
from interface.api import StockAppAPI

log = get_logger("backtest")


# ------------------------------------------------------------------
# Single-period backtest mode
# ------------------------------------------------------------------


def run_backtest(
    tickers,
    n_days,
    full=False,
    period="max",
    output=None,
    fee_pct=0.0,
    stop_loss_pct=0.0,
    buy_hold=False,
    no_refresh=False,
    timing=False,
    models=None,
    include_baselines: bool = True,
    min_confidence: float = 0.0,
    confidence_sweep: bool = False,
    significance: bool = False,
    turnover_fees: bool = False,
    hold_days: int = 1,
    position_mode: bool = False,
    sl_levels=None,
):
    """Standard single-period backtest mode."""
    api = StockAppAPI()
    backtester = Backtester(
        n_days=n_days,
        fee_pct=fee_pct,
        stop_loss_pct=stop_loss_pct,
        min_confidence=min_confidence,
        turnover_fees=turnover_fees,
        hold_days=hold_days,
        position_mode=position_mode,
    )
    all_export_rows = []

    if not no_refresh:
        # Also refresh benchmark tickers (SPY, QQQ, BTC-USD)
        all_to_refresh = list(set(tickers + STOCK_BENCHMARKS + CRYPTO_BENCHMARKS))
        api.refresh_tickers(all_to_refresh)

    for ticker in progress_bar(tickers, desc="Backtesting"):
        sl_str = f", SL={stop_loss_pct}%" if stop_loss_pct > 0 else ""
        print(f"\n{'=' * 70}")
        print(
            f" BACKTEST: {ticker} (last {n_days} days, period={period}"
            f"{f', fees={fee_pct}%' if fee_pct > 0 else ''}{sl_str})"
        )
        print(f"{'=' * 70}")

        df = api.get_data(ticker, period="max")
        if df.empty:
            print(f"  Skipping {ticker}: no data")
            continue

        filtered = filter_by_period(df, period)
        if len(filtered) < n_days + 20:
            print(f"  Skipping {ticker}: not enough data ({len(filtered)} rows)")
            continue

        print(
            f"  Data: {len(filtered)} rows "
            f"({filtered['date'].iloc[0]} → {filtered['date'].iloc[-1]})"
        )

        sentiment_score, headlines = api._process_news_with_db(ticker)
        has_news = len(headlines) > 0
        if has_news:
            sl = (
                "POSITIVE"
                if sentiment_score > 0.15
                else "NEGATIVE"
                if sentiment_score < -0.15
                else "NEUTRAL"
            )
            print(f"  Sentiment: {sl} ({sentiment_score:+.2f})")

        results = run_single_backtest(
            api,
            backtester,
            ticker,
            df,
            period,
            n_days,
            full,
            models=models,
            include_baselines=include_baselines,
            sl_levels=sl_levels,
        )
        if not results:
            continue

        # Compute benchmark returns (SPY/QQQ for stocks, BTC for crypto)
        bench = compute_benchmarks(api, ticker, results[0].days) if buy_hold else None

        print()
        print_summary_table(results, show_buy_hold=buy_hold, benchmarks=bench)
        if timing:
            print_timing_table(results)

        if has_news:
            print(f"\n  News ({sentiment_score:+.2f}):")
            for h in headlines[:3]:
                print(f"    > {h}")

        if full or confidence_sweep or significance:
            print_consensus(results, n_days)
            print_direction_accuracy(results)
            print_confidence_calibration(results)
            if confidence_sweep:
                print_confidence_sweep(results, fee_pct)
            if significance:
                print_significance(results)
            print_profit_analysis(results, show_buy_hold=buy_hold, benchmarks=bench)
            print_next_day_forecast(results)
        else:
            best = max(results, key=lambda r: (r.total_return, r.accuracy))
            print(f"\n  Day-by-day ({best.model_name}):")
            print(best.summary().split("\n", 3)[-1])
            print("\n  (use --full for all details)")

        if output:
            for r in results:
                if full:
                    all_export_rows.extend(result_to_daily_rows(r, ticker, period))
                else:
                    all_export_rows.append(
                        result_to_summary_row(r, ticker, period, benchmarks=bench)
                    )

        print(f"\n{'*' * 70}")

    if output and all_export_rows:
        export_rows(all_export_rows, output)


# ------------------------------------------------------------------
# Cross-period comparison mode
# ------------------------------------------------------------------


def run_compare_periods(
    tickers,
    n_days,
    output=None,
    fee_pct=0.0,
    stop_loss_pct=0.0,
    buy_hold=False,
    no_refresh=False,
    timing=False,
    periods=None,
    models=None,
    include_baselines: bool = True,
    min_confidence: float = 0.0,
    turnover_fees: bool = False,
    hold_days: int = 1,
    position_mode: bool = False,
    sl_levels=None,
):
    """Run backtest across all periods, find optimal model+period."""
    periods = periods or list(ALL_PERIODS)
    api = StockAppAPI()
    backtester = Backtester(
        n_days=n_days,
        fee_pct=fee_pct,
        stop_loss_pct=stop_loss_pct,
        min_confidence=min_confidence,
        turnover_fees=turnover_fees,
        hold_days=hold_days,
        position_mode=position_mode,
    )
    all_export_rows = []

    if not no_refresh:
        all_to_refresh = list(set(tickers + STOCK_BENCHMARKS + CRYPTO_BENCHMARKS))
        api.refresh_tickers(all_to_refresh)

    for ticker in progress_bar(tickers, desc="Comparing periods"):
        sl_str = f", SL={stop_loss_pct}%" if stop_loss_pct > 0 else ""
        print(f"\n{'=' * 80}")
        print(
            f" PERIOD COMPARISON: {ticker} (holdout={n_days} days"
            f"{f', fees={fee_pct}%' if fee_pct > 0 else ''}{sl_str})"
        )
        print(f"{'=' * 80}")

        df = api.get_data(ticker, period="max")
        if df.empty:
            print(f"  Skipping {ticker}: no data")
            continue

        print(f"  Total data: {len(df)} rows ({df['date'].iloc[0]} → {df['date'].iloc[-1]})")

        # Run all periods
        period_results = {}
        for period in periods:
            results = run_single_backtest(
                api,
                backtester,
                ticker,
                df,
                period,
                n_days,
                full=False,
                models=models,
                include_baselines=include_baselines,
                sl_levels=sl_levels,
            )
            if results:
                period_results[period] = results
            else:
                filtered = filter_by_period(df, period)
                print(
                    f"  Skipping period={period}: not enough data "
                    f"({len(filtered)} rows, need {n_days + 20})"
                )

        if not period_results:
            print("  No periods had enough data.")
            continue

        # Compute benchmark returns (same test dates across all periods)
        first_results = next(iter(period_results.values()))
        bench = compute_benchmarks(api, ticker, first_results[0].days) if buy_hold else None

        # Collect model names
        model_names = []
        for results in period_results.values():
            for r in results:
                if r.model_name not in model_names:
                    model_names.append(r.model_name)

        # --- Accuracy matrix ---
        print(f"\n  Accuracy by period × model (holdout={n_days} days):\n")
        header = f"  {'MODEL':<25} |"
        for p in periods:
            if p in period_results:
                header += f" {p:^8} |"
        print(header)
        print(f"  {'-' * (28 + sum(11 for p in periods if p in period_results))}")

        best_per_model = {}
        all_combos = []

        for model_name in model_names:
            row = f"  {model_name:<25} |"
            best_acc = -1
            best_period = ""

            for period in periods:
                if period not in period_results:
                    continue
                match = [r for r in period_results[period] if r.model_name == model_name]
                if match:
                    r = match[0]
                    row += f" {r.accuracy:>6.0%}   |"
                    all_combos.append(
                        (
                            model_name,
                            period,
                            r.accuracy,
                            r.correct,
                            r.test_days,
                            r.total_return,
                            r.profit_factor,
                            r.longest_win_streak,
                            r.longest_loss_streak,
                            r.avg_win_streak,
                            r.avg_loss_streak,
                            r.buy_hold_return,
                            r.max_drawdown,
                            r.sharpe_ratio,
                            r.sortino_ratio,
                            r.buy_hold_max_drawdown,
                        )
                    )
                    all_export_rows.append(
                        result_to_summary_row(r, ticker, period, benchmarks=bench)
                    )
                    if r.accuracy > best_acc:
                        best_acc = r.accuracy
                        best_period = period
                else:
                    row += f" {'n/a':^8} |"

            row += f"  ◄ best: {best_period}"
            print(row)
            best_per_model[model_name] = (best_period, best_acc)

        # --- Best per model ---
        print(f"\n  {'BEST PERIOD PER MODEL':=^60}")
        for mn, (bp, ba) in best_per_model.items():
            print(f"  {mn:<25} → {bp:<6} ({ba:.0%})")

        period_votes = {}
        for _, (bp, _) in best_per_model.items():
            period_votes[bp] = period_votes.get(bp, 0) + 1
        overall_best = max(period_votes, key=period_votes.get)
        print(
            f"\n  Most popular period: {overall_best} "
            f"({period_votes[overall_best]}/{len(best_per_model)} models peak here)"
        )

        # --- Top combinations ---
        print(f"\n  {'TOP COMBINATIONS (model + period)':=^70}")
        all_combos.sort(key=lambda x: (x[5], x[2]), reverse=True)

        top_ret = all_combos[0][5]
        top_list = [c for c in all_combos if c[5] == top_ret]

        if len(top_list) == 1:
            c = top_list[0]
            print(
                f"\n  ★ BEST: {c[0]} + {c[1]} → return {c[5]:+.4%}, "
                f"accuracy {c[2]:.0%} ({c[3]}/{c[4]}), "
                f"PF {pf_str(c[6])}"
            )
        else:
            print(f"\n  ★ TIED AT {top_ret:+.4%} return:")
            for c in top_list:
                print(f"    • {c[0]} + {c[1]} (acc {c[2]:.0%}, PF {pf_str(c[6])})")

        # Top 5
        unique_rets = len(set(round(c[5], 8) for c in all_combos))
        if unique_rets > 1:
            cols = (
                f"  {'#':<4} {'MODEL':<25} {'PERIOD':<8} {'RETURN':<12} "
                f"{'PF':<8} {'MAX DD':<10} {'SHARPE':<8}"
            )
            if buy_hold:
                cols += f" {'B&H':<12}"
            print("\n  Top 5 by return:")
            print(cols)
            print(f"  {'-' * (80 + (14 if buy_hold else 0))}")
            shown = set()
            rank = 0
            for c in all_combos:
                key = (c[0], c[1])
                if key in shown:
                    continue
                shown.add(key)
                rank += 1
                marker = " ★" if c[5] == top_ret else ""
                line = (
                    f"  {rank:<4} {c[0]:<25} {c[1]:<8} "
                    f"{c[5]:<+12.4%} {pf_str(c[6]):<8} "
                    f"{c[12]:<+10.4%} {c[13]:<8.2f}"
                    f"{marker}"
                )
                if buy_hold:
                    line += f" {c[11]:<+12.4%}"
                print(line)
                if rank >= 5:
                    break

        # --- Streak analysis ---
        print(f"\n  {'STREAK ANALYSIS':=^70}")
        print(
            f"  {'MODEL + PERIOD':<35} | {'MAX W':<7} | {'MAX L':<7} | {'AVG W':<7} | {'AVG L':<7}"
        )
        print(f"  {'-' * 70}")

        def streak_score(c):
            aw = c[9] if c[9] > 0 else 0.1
            al = c[10] if c[10] > 0 else 0.1
            return aw / al

        combos_by_streak = sorted(all_combos, key=streak_score, reverse=True)
        shown = set()
        count = 0
        for c in combos_by_streak:
            key = (c[0], c[1])
            if key in shown:
                continue
            shown.add(key)
            label = f"{c[0]} + {c[1]}"
            print(f"  {label:<35} | {c[7]:<7} | {c[8]:<7} | {c[9]:<7.1f} | {c[10]:<7.1f}")
            count += 1
            if count >= 5:
                break

        bws = max(all_combos, key=lambda c: c[7])
        wls = max(all_combos, key=lambda c: c[8])
        print(f"\n  Best win streak:    {bws[0]} + {bws[1]} ({bws[7]} wins)")
        print(f"  Worst loss streak:  {wls[0]} + {wls[1]} ({wls[8]} losses)")

        # Buy-and-hold comparison
        if buy_hold and all_combos:
            bh = all_combos[0][11]
            bh_dd = all_combos[0][15]
            beat = sum(1 for c in all_combos if c[5] > bh)
            total = len(all_combos)
            print(f"\n  {'BUY & HOLD COMPARISON':=^70}")
            print(f"  Buy & Hold return:    {bh:+.4%}  (max DD: {bh_dd:+.4%})")
            print(f"  Models beating B&H:   {beat}/{total} ({beat / total:.0%})")

            if bench:
                best_return = max(c[5] for c in all_combos)
                for bname, bret in bench.items():
                    b_beat = sum(1 for c in all_combos if c[5] > bret)
                    marker = "✓" if best_return > bret else "✗"
                    print(
                        f"  {bname:<10} return:     {bret:+.4%}  "
                        f"| Models beating: {b_beat}/{total} {marker}"
                    )

        # Risk-adjusted ranking (by Sharpe)
        if len(all_combos) > 1:
            by_sharpe = sorted(all_combos, key=lambda c: c[13], reverse=True)
            print(f"\n  {'RISK-ADJUSTED RANKING (by Sharpe)':=^70}")
            print(
                f"  {'#':<4} {'MODEL':<25} {'PERIOD':<8} {'SHARPE':<8} "
                f"{'SORTINO':<8} {'MAX DD':<10} {'RETURN':<12}"
            )
            print(f"  {'-' * 80}")
            shown = set()
            rank = 0
            for c in by_sharpe:
                key = (c[0], c[1])
                if key in shown:
                    continue
                shown.add(key)
                rank += 1
                sortino_s = f"{c[14]:.2f}" if c[14] < 100 else "∞"
                print(
                    f"  {rank:<4} {c[0]:<25} {c[1]:<8} {c[13]:<8.2f} "
                    f"{sortino_s:<8} {c[12]:<+10.4%} {c[5]:<+12.4%}"
                )
                if rank >= 5:
                    break

        if timing:
            all_period_results = [r for results in period_results.values() for r in results]
            print_timing_table(all_period_results)

        print(f"\n{'*' * 80}")

    if output and all_export_rows:
        export_rows(all_export_rows, output)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main():
    # in main(): replace the scope block (keep --days and everything after)
    parser = argparse.ArgumentParser(description="MarketPulse AI – Backtest")
    add_scope_args(parser)
    add_common_run_args(parser, days_default=DEFAULT_BACKTEST_DAYS)
    add_model_filter_args(parser)
    add_strategy_args(parser)

    # backtest.py-only flags
    parser.add_argument("--period", default=DEFAULT_PERIOD, choices=ALL_PERIODS)
    parser.add_argument("--full", action="store_true", help="Detailed output")
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print a per-model compute-time breakdown (slowest first) after the summary.",
    )
    parser.add_argument(
        "--compare-periods", action="store_true", help="Run all periods, show comparison matrix"
    )
    parser.add_argument("--output", type=str, default=None, help="Export to CSV or JSON")
    parser.add_argument(
        "--confidence-sweep",
        action="store_true",
        help=(
            "Print a θ-sweep table (coverage / traded-day accuracy / return / "
            "fees saved) over CONFIDENCE_SWEEP, computed post-hoc from a single "
            "ungated run. Implies --full detail for the calibration section."
        ),
    )
    parser.add_argument(
        "--significance",
        action="store_true",
        help=(
            "Print statistical-significance tests: binomial p-value "
            "+ Wilson CI on accuracy, bootstrap CI on return, with "
            "Benjamini-Hochberg FDR across the models shown."
        ),
    )
    args = parser.parse_args()

    tickers = resolve_scope(args, default=ALL_TICKERS[:3])

    # --stop-loss / --sl-sweep → (sl_levels, legacy_sl). A single --stop-loss
    # value keeps the legacy single-run path (run_single_backtest pairs the
    # no-SL baseline with the SL run); see cli_helpers.resolve_sl_levels.
    sl_levels, legacy_sl = resolve_sl_levels(args)

    if args.compare_periods:
        run_compare_periods(
            tickers,
            args.days,
            args.output,
            args.fees,
            legacy_sl,
            args.buy_hold,
            args.no_refresh,
            args.timing,
            periods=args.periods,
            models=args.models,
            include_baselines=not args.no_baselines,
            min_confidence=args.min_confidence,
            turnover_fees=args.turnover_fees,
            hold_days=args.hold_days,
            position_mode=args.position_mode,
            sl_levels=sl_levels,
        )
    else:
        run_backtest(
            tickers,
            args.days,
            args.full,
            args.period,
            args.output,
            args.fees,
            legacy_sl,
            args.buy_hold,
            args.no_refresh,
            args.timing,
            models=args.models,
            include_baselines=not args.no_baselines,
            min_confidence=args.min_confidence,
            confidence_sweep=args.confidence_sweep,
            significance=args.significance,
            turnover_fees=args.turnover_fees,
            hold_days=args.hold_days,
            position_mode=args.position_mode,
            sl_levels=sl_levels,
        )


if __name__ == "__main__":
    main()
