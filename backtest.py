"""
backtest.py – CLI script for running walk-forward backtests.

Trains each model on all data except the last N days, then predicts
each held-out day one at a time and reports accuracy.

Usage:
    uv run python backtest.py
    uv run python backtest.py --days 10
    uv run python backtest.py --tickers AAPL MSFT
    uv run python backtest.py --full                          # detailed stats
    uv run python backtest.py --full --period 1y              # train on last year only
    uv run python backtest.py --compare-periods               # run all periods, show comparison
    uv run python backtest.py --compare-periods --output results.csv
    uv run python backtest.py --compare-periods --days 10 --tickers AAPL BTC-USD
"""

import argparse
import csv
import json
from datetime import datetime, timedelta, date
from typing import Optional

import pandas as pd

from interface.api import StockAppAPI
from engine.backtester import Backtester, BacktestResult


DEFAULT_TICKERS = ["BTC-USD", "AAPL", "MSFT"]
DEFAULT_DAYS = 5
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]
VALID_PERIODS = ALL_PERIODS


def period_to_start_date(period: str) -> date:
    """Convert a period string to the earliest date to include."""
    today = datetime.now().date()
    mapping = {
        "1mo": today - timedelta(days=30),
        "1y":  today - timedelta(days=365),
        "2y":  today - timedelta(days=730),
        "5y":  today - timedelta(days=1825),
        "max": date(1900, 1, 1),
    }
    return mapping.get(period, today - timedelta(days=365))


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Trim a price DataFrame to only include rows within the given period."""
    if period == "max":
        return df

    start = period_to_start_date(period)
    df = df.copy()
    df["_date"] = pd.to_datetime(df["date"]).dt.date
    filtered = df[df["_date"] >= start].drop(columns=["_date"])
    return filtered


def direction_accuracy(result: BacktestResult):
    """Return (up_correct, up_total, down_correct, down_total)."""
    up_c = sum(1 for d in result.days if d.predicted == "UP" and d.correct)
    up_t = sum(1 for d in result.days if d.predicted == "UP")
    dn_c = sum(1 for d in result.days if d.predicted == "DOWN" and d.correct)
    dn_t = sum(1 for d in result.days if d.predicted == "DOWN")
    return up_c, up_t, dn_c, dn_t


# ------------------------------------------------------------------
# Display helpers (single-period mode)
# ------------------------------------------------------------------

def print_consensus(all_results, n_days):
    """Day-by-day consensus: what did each model predict vs reality?"""
    if not all_results or not all_results[0].days:
        return

    dates = [d.date for d in all_results[0].days]

    print(f"\n  {'DAY-BY-DAY CONSENSUS':=^66}")
    header = f"  {'DATE':<12} |"
    for r in all_results:
        short = r.model_name.replace("Time-Weighted", "TW").replace(" + News", "+N")
        header += f" {short:<8} |"
    header += f" {'ACTUAL':<8} | {'AGREE'}"
    print(header)
    print(f"  {'-' * (14 + len(all_results) * 11 + 20)}")

    for i, date_str in enumerate(dates):
        row = f"  {date_str:<12} |"
        predictions = []
        actual = None

        for r in all_results:
            if i < len(r.days):
                d = r.days[i]
                mark = "✓" if d.correct else "✗"
                row += f" {d.predicted:<5}{mark}  |"
                predictions.append(d.predicted)
                actual = d.actual

        if actual:
            up_count = predictions.count("UP")
            down_count = predictions.count("DOWN")
            total = len(predictions)
            majority = "UP" if up_count >= down_count else "DOWN"
            agree_pct = max(up_count, down_count) / total if total > 0 else 0
            consensus_correct = "✓" if majority == actual else "✗"

            row += f" {actual:<8} | {agree_pct:.0%} {consensus_correct}"
            if agree_pct == 1.0:
                row += " ◄ unanimous"

        print(row)


def print_direction_accuracy(all_results):
    """Per-direction accuracy breakdown."""
    print(f"\n  {'DIRECTION ACCURACY':=^66}")
    print(f"  {'MODEL':<25} | {'UP acc.':<12} | {'DOWN acc.':<12}")
    print(f"  {'-' * 55}")

    for r in all_results:
        up_c, up_t, dn_c, dn_t = direction_accuracy(r)
        up_str = f"{up_c}/{up_t} ({up_c/up_t:.0%})" if up_t > 0 else "n/a"
        dn_str = f"{dn_c}/{dn_t} ({dn_c/dn_t:.0%})" if dn_t > 0 else "n/a"
        print(f"  {r.model_name:<25} | {up_str:<12} | {dn_str:<12}")


def print_confidence_calibration(all_results):
    """Are high-confidence predictions actually more accurate?"""
    print(f"\n  {'CONFIDENCE CALIBRATION':=^66}")
    print(f"  {'MODEL':<25} | {'High (>65%)':<16} | {'Low (≤65%)':<16}")
    print(f"  {'-' * 62}")

    for r in all_results:
        high = [d for d in r.days if d.confidence > 0.65]
        low = [d for d in r.days if d.confidence <= 0.65]

        high_acc = sum(1 for d in high if d.correct) / len(high) if high else 0
        low_acc = sum(1 for d in low if d.correct) / len(low) if low else 0

        high_str = f"{high_acc:.0%} ({len(high)} pred)" if high else "n/a"
        low_str = f"{low_acc:.0%} ({len(low)} pred)" if low else "n/a"

        print(f"  {r.model_name:<25} | {high_str:<16} | {low_str:<16}")


def print_next_day_forecast(all_results):
    """What would each model predict for the most recent holdout day?"""
    if not all_results or not all_results[0].days:
        return

    print(f"\n  {'NEXT-DAY SIGNAL (most recent holdout day)':=^66}")

    last_day = all_results[0].days[-1]
    print(f"  Date: {last_day.date}")
    print(f"  Price before: {last_day.close_before:.2f} → Actual: {last_day.close_actual:.2f}\n")

    up_votes = 0
    down_votes = 0

    for r in all_results:
        d = r.days[-1]
        mark = "✓" if d.correct else "✗"
        print(f"  {r.model_name:<25}  {d.predicted:<6} (conf: {d.confidence:.1%})  {mark}")
        if d.predicted == "UP":
            up_votes += 1
        else:
            down_votes += 1

    total = up_votes + down_votes
    if total > 0:
        signal = "UP" if up_votes > down_votes else "DOWN"
        strength = max(up_votes, down_votes) / total
        print(f"\n  Consensus: {signal} ({strength:.0%} of models agree)")


# ------------------------------------------------------------------
# Single-period backtest
# ------------------------------------------------------------------

def run_single_backtest(api, backtester, ticker, df, period, n_days, full):
    """Run backtest for one ticker × one period. Returns list of BacktestResult."""
    filtered = filter_by_period(df, period)

    if len(filtered) < n_days + 20:
        return []

    # Fetch current news sentiment
    sentiment_score, headlines = api._process_news_with_db(ticker)
    has_news = len(headlines) > 0

    # Model variants
    variants = [
        (api.knn,    "k-NN",                  False, 0.0),
        (api.knn,    "k-NN Time-Weighted",     True,  0.0),
        (api.linreg, "LinReg",                 False, 0.0),
        (api.linreg, "LinReg Time-Weighted",   True,  0.0),
    ]

    if has_news:
        variants.extend([
            (api.knn,    "k-NN TW + News",     True,  sentiment_score),
            (api.linreg, "LinReg TW + News",   True,  sentiment_score),
        ])

    results = []
    for model, name, tw, sent in variants:
        result = backtester.run(
            model=model,
            model_name=name,
            df=filtered,
            ticker=ticker,
            use_time_weights=tw,
            sentiment_score=sent,
        )
        results.append(result)

    return results


def run_backtest(tickers: list[str], n_days: int, full: bool = False, period: str = "max"):
    """Standard single-period backtest mode."""
    api = StockAppAPI()
    backtester = Backtester(n_days=n_days)

    for ticker in tickers:
        print(f"\n{'=' * 70}")
        print(f" BACKTEST: {ticker} (last {n_days} trading days, period={period})")
        print(f"{'=' * 70}")

        df = api.get_data(ticker, period="max")
        if df.empty:
            print(f"  Skipping {ticker}: no data available")
            continue

        filtered = filter_by_period(df, period)
        if len(filtered) < n_days + 20:
            print(f"  Skipping {ticker}: not enough data ({len(filtered)} rows)")
            continue

        print(f"  Training data: {len(filtered)} rows "
              f"({filtered['date'].iloc[0]} → {filtered['date'].iloc[-1]})")

        sentiment_score, headlines = api._process_news_with_db(ticker)
        has_news = len(headlines) > 0

        if has_news:
            sentiment_label = "POSITIVE" if sentiment_score > 0.15 else (
                "NEGATIVE" if sentiment_score < -0.15 else "NEUTRAL"
            )
            print(f"  News sentiment: {sentiment_label} ({sentiment_score:+.2f})")

        results = run_single_backtest(api, backtester, ticker, df, period, n_days, full)
        if not results:
            continue

        # Summary table
        print(f"\n  {'MODEL':<25} | {'ACCURACY':<12} | {'CORRECT':<10}")
        print(f"  {'-' * 55}")
        for r in results:
            print(f"  {r.model_name:<25} | {r.accuracy:<12.1%} | {r.correct}/{r.test_days}")

        if has_news:
            print(f"\n  News headlines used for sentiment ({sentiment_score:+.2f}):")
            for h in headlines[:3]:
                print(f"    > {h}")

        if full:
            print_consensus(results, n_days)
            print_direction_accuracy(results)
            print_confidence_calibration(results)
            print_next_day_forecast(results)
        else:
            best = max(results, key=lambda r: (r.accuracy, r.correct))
            print(f"\n  Day-by-day ({best.model_name}):")
            print(best.summary().split("\n", 3)[-1])
            print(f"\n  (use --full for consensus view, direction stats, "
                  f"and confidence calibration)")

        print(f"\n{'*' * 70}")


# ------------------------------------------------------------------
# Cross-period comparison mode
# ------------------------------------------------------------------

def run_compare_periods(tickers: list[str], n_days: int, output: Optional[str] = None):
    """Run backtest across all periods for each ticker, find the optimal period."""
    api = StockAppAPI()
    backtester = Backtester(n_days=n_days)

    # Collect all rows for CSV/JSON export
    all_rows = []

    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(f" PERIOD COMPARISON: {ticker} (holdout={n_days} days)")
        print(f"{'=' * 80}")

        df = api.get_data(ticker, period="max")
        if df.empty:
            print(f"  Skipping {ticker}: no data available")
            continue

        total_rows = len(df)
        print(f"  Total data: {total_rows} rows "
              f"({df['date'].iloc[0]} → {df['date'].iloc[-1]})")

        # Run all periods
        period_results = {}  # period -> list of BacktestResult

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
            print(f"  No periods had enough data for backtesting.")
            continue

        # Collect all unique model names
        model_names = []
        for results in period_results.values():
            for r in results:
                if r.model_name not in model_names:
                    model_names.append(r.model_name)

        # --- Accuracy matrix ---
        print(f"\n  Accuracy by period × model (holdout={n_days} days):\n")

        header = f"  {'MODEL':<25} |"
        for period in ALL_PERIODS:
            if period in period_results:
                header += f" {period:^8} |"
        print(header)
        print(f"  {'-' * (28 + sum(11 for p in ALL_PERIODS if p in period_results))}")

        # Track best period per model + all combinations for ranking
        best_per_model = {}
        all_combos = []  # (model_name, period, accuracy, correct, total)

        for model_name in model_names:
            row = f"  {model_name:<25} |"
            best_acc = -1
            best_period = ""

            for period in ALL_PERIODS:
                if period not in period_results:
                    continue

                match = [r for r in period_results[period] if r.model_name == model_name]
                if match:
                    r = match[0]
                    acc = r.accuracy
                    row += f" {acc:>6.0%}   |"

                    up_c, up_t, dn_c, dn_t = direction_accuracy(r)

                    all_combos.append((model_name, period, acc, r.correct, r.test_days))

                    # Collect for export
                    all_rows.append({
                        "ticker": ticker,
                        "period": period,
                        "model": model_name,
                        "accuracy": round(acc, 4),
                        "correct": r.correct,
                        "total": r.test_days,
                        "up_accuracy": round(up_c / up_t, 4) if up_t > 0 else None,
                        "up_predictions": up_t,
                        "down_accuracy": round(dn_c / dn_t, 4) if dn_t > 0 else None,
                        "down_predictions": dn_t,
                    })

                    if acc > best_acc:
                        best_acc = acc
                        best_period = period
                else:
                    row += f" {'n/a':^8} |"

            row += f"  ◄ best: {best_period}"
            print(row)
            best_per_model[model_name] = (best_period, best_acc)

        # --- Best per model ---
        print(f"\n  {'BEST PERIOD PER MODEL':=^60}")
        for model_name, (bp, ba) in best_per_model.items():
            print(f"  {model_name:<25} → {bp:<6} ({ba:.0%})")

        # Find overall best period (most models peak there)
        period_votes = {}
        for model_name, (bp, ba) in best_per_model.items():
            period_votes[bp] = period_votes.get(bp, 0) + 1
        overall_best = max(period_votes, key=period_votes.get)
        print(f"\n  Most popular period: {overall_best} "
              f"({period_votes[overall_best]}/{len(best_per_model)} models peak here)")

        # --- Overall top combinations ---
        print(f"\n  {'TOP COMBINATIONS (model + period)':=^60}")

        # Sort by accuracy desc, then by correct count desc (tiebreaker)
        all_combos.sort(key=lambda x: (x[2], x[3]), reverse=True)

        top_acc = all_combos[0][2]
        top_combos = [c for c in all_combos if c[2] == top_acc]

        if len(top_combos) == 1:
            name, period, acc, correct, total = top_combos[0]
            print(f"\n  ★ BEST: {name} + {period} → {acc:.0%} ({correct}/{total})")
        else:
            print(f"\n  ★ TIED AT {top_acc:.0%}:")
            for name, period, acc, correct, total in top_combos:
                print(f"    • {name} + {period} ({correct}/{total})")

        # Show top 5 overall (skip if all tied at the same score)
        unique_scores = len(set(c[2] for c in all_combos))
        if unique_scores > 1:
            print(f"\n  Top 5 overall:")
            print(f"  {'#':<4} {'MODEL':<25} {'PERIOD':<8} {'ACCURACY':<10} {'CORRECT':<10}")
            print(f"  {'-' * 60}")
            shown = set()
            rank = 0
            for name, period, acc, correct, total in all_combos:
                key = (name, period)
                if key in shown:
                    continue
                shown.add(key)
                rank += 1
                marker = " ★" if acc == top_acc else ""
                print(f"  {rank:<4} {name:<25} {period:<8} {acc:<10.0%} {correct}/{total}{marker}")
                if rank >= 5:
                    break

        print(f"\n{'*' * 80}")

    # --- Export ---
    if output and all_rows:
        if output.endswith(".json"):
            with open(output, "w") as f:
                json.dump(all_rows, f, indent=2)
            print(f"\nResults exported to {output}")
        else:
            if not output.endswith(".csv"):
                output += ".csv"
            with open(output, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)
            print(f"\nResults exported to {output}")


def main():
    parser = argparse.ArgumentParser(description="MarketPulse AI – Backtest")
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help=f"Tickers to backtest (default: {DEFAULT_TICKERS})"
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"Number of days to hold out for testing (default: {DEFAULT_DAYS})"
    )
    parser.add_argument(
        "--period", default="max", choices=VALID_PERIODS,
        help="How much historical data to train on (default: max)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show detailed day-by-day consensus, direction accuracy, "
             "and confidence calibration"
    )
    parser.add_argument(
        "--compare-periods", action="store_true",
        help="Run backtest across all periods (1mo, 1y, 2y, 5y, max) "
             "and show which period works best for each ticker/model"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Export results to CSV or JSON file (e.g. results.csv, results.json). "
             "Only works with --compare-periods"
    )
    args = parser.parse_args()

    if args.compare_periods:
        run_compare_periods(args.tickers, args.days, args.output)
    else:
        run_backtest(args.tickers, args.days, args.full, args.period)


if __name__ == "__main__":
    main()
