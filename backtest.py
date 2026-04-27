"""
backtest.py – CLI for running walk-forward backtests.

Usage:
    uv run python backtest.py
    uv run python backtest.py --days 20 --fees 0.1
    uv run python backtest.py --full --period 1y --buy-hold
    uv run python backtest.py --compare-periods --output results.csv
    uv run python backtest.py --stocks --compare-periods --buy-hold --fees 0.05
"""

import argparse
from typing import Optional

from interface.api import StockAppAPI
from engine.backtester import Backtester
from engine.backtest_helpers import (
    filter_by_period, run_single_backtest, result_to_summary_row,
    result_to_daily_rows, export_rows, pf_str,
    print_summary_table, print_consensus, print_direction_accuracy,
    print_confidence_calibration, print_profit_analysis,
    print_next_day_forecast,
)
from config import (
    ALL_TICKERS, STOCKS, CRYPTO, ALL_PERIODS, DEFAULT_PERIOD,
    DEFAULT_BACKTEST_DAYS, DEFAULT_TRADING_FEE_PCT,
)


# ------------------------------------------------------------------
# Single-period backtest mode
# ------------------------------------------------------------------

def run_backtest(tickers, n_days, full=False, period="max",
                 output=None, fee_pct=0.0, buy_hold=False):
    """Standard single-period backtest mode."""
    api = StockAppAPI()
    backtester = Backtester(n_days=n_days, fee_pct=fee_pct)
    all_export_rows = []

    for ticker in tickers:
        print(f"\n{'=' * 70}")
        print(f" BACKTEST: {ticker} (last {n_days} days, period={period}"
              f"{f', fees={fee_pct}%' if fee_pct > 0 else ''})")
        print(f"{'=' * 70}")

        df = api.get_data(ticker, period="max")
        if df.empty:
            print(f"  Skipping {ticker}: no data")
            continue

        filtered = filter_by_period(df, period)
        if len(filtered) < n_days + 20:
            print(f"  Skipping {ticker}: not enough data ({len(filtered)} rows)")
            continue

        print(f"  Data: {len(filtered)} rows "
              f"({filtered['date'].iloc[0]} → {filtered['date'].iloc[-1]})")

        sentiment_score, headlines = api._process_news_with_db(ticker)
        has_news = len(headlines) > 0
        if has_news:
            sl = ("POSITIVE" if sentiment_score > 0.15
                  else "NEGATIVE" if sentiment_score < -0.15 else "NEUTRAL")
            print(f"  Sentiment: {sl} ({sentiment_score:+.2f})")

        results = run_single_backtest(
            api, backtester, ticker, df, period, n_days, full
        )
        if not results:
            continue

        print()
        print_summary_table(results, show_buy_hold=buy_hold)

        if has_news:
            print(f"\n  News ({sentiment_score:+.2f}):")
            for h in headlines[:3]:
                print(f"    > {h}")

        if full:
            print_consensus(results, n_days)
            print_direction_accuracy(results)
            print_confidence_calibration(results)
            print_profit_analysis(results, show_buy_hold=buy_hold)
            print_next_day_forecast(results)
        else:
            best = max(results, key=lambda r: (r.total_return, r.accuracy))
            print(f"\n  Day-by-day ({best.model_name}):")
            print(best.summary().split("\n", 3)[-1])
            print(f"\n  (use --full for all details)")

        if output:
            for r in results:
                if full:
                    all_export_rows.extend(result_to_daily_rows(r, ticker, period))
                else:
                    all_export_rows.append(result_to_summary_row(r, ticker, period))

        print(f"\n{'*' * 70}")

    if output and all_export_rows:
        export_rows(all_export_rows, output)


# ------------------------------------------------------------------
# Cross-period comparison mode
# ------------------------------------------------------------------

def run_compare_periods(tickers, n_days, output=None, fee_pct=0.0,
                        buy_hold=False):
    """Run backtest across all periods, find optimal model+period."""
    api = StockAppAPI()
    backtester = Backtester(n_days=n_days, fee_pct=fee_pct)
    all_export_rows = []

    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(f" PERIOD COMPARISON: {ticker} (holdout={n_days} days"
              f"{f', fees={fee_pct}%' if fee_pct > 0 else ''})")
        print(f"{'=' * 80}")

        df = api.get_data(ticker, period="max")
        if df.empty:
            print(f"  Skipping {ticker}: no data")
            continue

        print(f"  Total data: {len(df)} rows "
              f"({df['date'].iloc[0]} → {df['date'].iloc[-1]})")

        # Run all periods
        period_results = {}
        for period in ALL_PERIODS:
            results = run_single_backtest(
                api, backtester, ticker, df, period, n_days, full=False
            )
            if results:
                period_results[period] = results
            else:
                filtered = filter_by_period(df, period)
                print(f"  Skipping period={period}: not enough data "
                      f"({len(filtered)} rows, need {n_days + 20})")

        if not period_results:
            print(f"  No periods had enough data.")
            continue

        # Collect model names
        model_names = []
        for results in period_results.values():
            for r in results:
                if r.model_name not in model_names:
                    model_names.append(r.model_name)

        # --- Accuracy matrix ---
        print(f"\n  Accuracy by period × model (holdout={n_days} days):\n")
        header = f"  {'MODEL':<25} |"
        for p in ALL_PERIODS:
            if p in period_results:
                header += f" {p:^8} |"
        print(header)
        print(f"  {'-' * (28 + sum(11 for p in ALL_PERIODS if p in period_results))}")

        best_per_model = {}
        all_combos = []

        for model_name in model_names:
            row = f"  {model_name:<25} |"
            best_acc = -1
            best_period = ""

            for period in ALL_PERIODS:
                if period not in period_results:
                    continue
                match = [r for r in period_results[period]
                         if r.model_name == model_name]
                if match:
                    r = match[0]
                    row += f" {r.accuracy:>6.0%}   |"
                    all_combos.append((
                        model_name, period, r.accuracy, r.correct,
                        r.test_days, r.total_return, r.profit_factor,
                        r.longest_win_streak, r.longest_loss_streak,
                        r.avg_win_streak, r.avg_loss_streak,
                        r.buy_hold_return,
                    ))
                    all_export_rows.append(
                        result_to_summary_row(r, ticker, period)
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
        for mn, (bp, ba) in best_per_model.items():
            period_votes[bp] = period_votes.get(bp, 0) + 1
        overall_best = max(period_votes, key=period_votes.get)
        print(f"\n  Most popular period: {overall_best} "
              f"({period_votes[overall_best]}/{len(best_per_model)} models peak here)")

        # --- Top combinations ---
        print(f"\n  {'TOP COMBINATIONS (model + period)':=^70}")
        all_combos.sort(key=lambda x: (x[5], x[2]), reverse=True)

        top_ret = all_combos[0][5]
        top_list = [c for c in all_combos if c[5] == top_ret]

        if len(top_list) == 1:
            c = top_list[0]
            print(f"\n  ★ BEST: {c[0]} + {c[1]} → return {c[5]:+.4%}, "
                  f"accuracy {c[2]:.0%} ({c[3]}/{c[4]}), "
                  f"PF {pf_str(c[6])}")
        else:
            print(f"\n  ★ TIED AT {top_ret:+.4%} return:")
            for c in top_list:
                print(f"    • {c[0]} + {c[1]} (acc {c[2]:.0%}, "
                      f"PF {pf_str(c[6])})")

        # Top 5
        unique_rets = len(set(round(c[5], 8) for c in all_combos))
        if unique_rets > 1:
            cols = f"  {'#':<4} {'MODEL':<25} {'PERIOD':<8} {'RETURN':<12} {'ACC.':<8} {'P.FACTOR':<10}"
            if buy_hold:
                cols += f" {'B&H':<12}"
            print(f"\n  Top 5 by return:")
            print(cols)
            print(f"  {'-' * (72 + (14 if buy_hold else 0))}")
            shown = set()
            rank = 0
            for c in all_combos:
                key = (c[0], c[1])
                if key in shown:
                    continue
                shown.add(key)
                rank += 1
                marker = " ★" if c[5] == top_ret else ""
                line = (f"  {rank:<4} {c[0]:<25} {c[1]:<8} "
                        f"{c[5]:<+12.4%} {c[2]:<8.0%} {pf_str(c[6]):<10}"
                        f"{marker}")
                if buy_hold:
                    line += f" {c[11]:<+12.4%}"
                print(line)
                if rank >= 5:
                    break

        # --- Streak analysis ---
        print(f"\n  {'STREAK ANALYSIS':=^70}")
        print(f"  {'MODEL + PERIOD':<35} | {'MAX W':<7} | {'MAX L':<7} | "
              f"{'AVG W':<7} | {'AVG L':<7}")
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
            print(f"  {label:<35} | {c[7]:<7} | {c[8]:<7} | "
                  f"{c[9]:<7.1f} | {c[10]:<7.1f}")
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
            beat = sum(1 for c in all_combos if c[5] > bh)
            total = len(all_combos)
            print(f"\n  {'BUY & HOLD COMPARISON':=^70}")
            print(f"  Buy & Hold return:    {bh:+.4%}")
            print(f"  Models beating B&H:   {beat}/{total} "
                  f"({beat/total:.0%})")

        print(f"\n{'*' * 80}")

    if output and all_export_rows:
        export_rows(all_export_rows, output)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MarketPulse AI – Backtest")
    parser.add_argument("--tickers", nargs="+", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stocks", action="store_true",
                       help=f"Only stocks: {STOCKS}")
    group.add_argument("--crypto", action="store_true",
                       help=f"Only crypto: {CRYPTO}")
    parser.add_argument("--all", action="store_true",
                        help="All tickers")
    parser.add_argument("--days", type=int, default=DEFAULT_BACKTEST_DAYS)
    parser.add_argument("--period", default=DEFAULT_PERIOD,
                        choices=ALL_PERIODS)
    parser.add_argument("--full", action="store_true",
                        help="Detailed output")
    parser.add_argument("--compare-periods", action="store_true",
                        help="Run all periods, show comparison matrix")
    parser.add_argument("--output", type=str, default=None,
                        help="Export to CSV or JSON")
    parser.add_argument("--fees", type=float, default=DEFAULT_TRADING_FEE_PCT,
                        help=f"Trading fee %% per side (default: {DEFAULT_TRADING_FEE_PCT})")
    parser.add_argument("--buy-hold", action="store_true",
                        help="Compare with buy-and-hold benchmark")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.stocks:
        tickers = STOCKS
    elif args.crypto:
        tickers = CRYPTO
    elif args.all:
        tickers = ALL_TICKERS
    else:
        tickers = ALL_TICKERS[:3] if len(ALL_TICKERS) >= 3 else ALL_TICKERS

    if args.compare_periods:
        run_compare_periods(tickers, args.days, args.output,
                            args.fees, args.buy_hold)
    else:
        run_backtest(tickers, args.days, args.full, args.period,
                     args.output, args.fees, args.buy_hold)


if __name__ == "__main__":
    main()
