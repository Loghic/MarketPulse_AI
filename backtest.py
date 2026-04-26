"""
backtest.py – CLI script for running walk-forward backtests.

Trains each model on all data except the last N days, then predicts
each held-out day one at a time and reports accuracy.

--output works in ALL modes:
    basic:            one row per model (summary: accuracy, return, PF, streaks)
    --full:           one row per DAY per model (day-by-day predictions + P/L)
    --compare-periods: one row per model × period (accuracy, return, PF, streaks)

Usage:
    uv run python backtest.py
    uv run python backtest.py --days 10
    uv run python backtest.py --tickers AAPL MSFT
    uv run python backtest.py --full --output full_results.csv
    uv run python backtest.py --full --period 1y --output details.csv
    uv run python backtest.py --compare-periods --output comparison.csv
    uv run python backtest.py --compare-periods --output comparison.json
"""

import argparse
import csv
import json
from datetime import datetime, timedelta, date
from typing import Optional, List

import pandas as pd

from interface.api import StockAppAPI
from engine.backtester import Backtester, BacktestResult
from config import ALL_TICKERS, STOCKS, CRYPTO, ALL_PERIODS, DEFAULT_PERIOD, DEFAULT_BACKTEST_DAYS


VALID_PERIODS = ALL_PERIODS


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


def export_rows(rows: list, output: str):
    """Export a list of dicts to CSV or JSON."""
    if not rows:
        print("  No data to export.")
        return

    if output.endswith(".json"):
        with open(output, "w") as f:
            json.dump(rows, f, indent=2)
    else:
        if not output.endswith(".csv"):
            output += ".csv"
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nResults exported to {output}")


def result_to_summary_row(r: BacktestResult, ticker: str, period: str) -> dict:
    """Convert a BacktestResult to a summary export row."""
    up_c, up_t, dn_c, dn_t = direction_accuracy(r)
    return {
        "ticker": ticker,
        "period": period,
        "model": r.model_name,
        "accuracy": round(r.accuracy, 4),
        "correct": r.correct,
        "total": r.test_days,
        "total_return": round(r.total_return, 6),
        "profit_factor": round(r.profit_factor, 4),
        "win_trades": r.win_trades,
        "loss_trades": r.loss_trades,
        "avg_win": round(r.avg_win, 6) if r.win_trades > 0 else None,
        "avg_loss": round(r.avg_loss, 6) if r.loss_trades > 0 else None,
        "best_day": round(r.best_day, 6),
        "worst_day": round(r.worst_day, 6),
        "longest_win_streak": r.longest_win_streak,
        "longest_loss_streak": r.longest_loss_streak,
        "avg_win_streak": round(r.avg_win_streak, 2),
        "avg_loss_streak": round(r.avg_loss_streak, 2),
        "up_accuracy": round(up_c / up_t, 4) if up_t > 0 else None,
        "up_predictions": up_t,
        "down_accuracy": round(dn_c / dn_t, 4) if dn_t > 0 else None,
        "down_predictions": dn_t,
    }


def result_to_daily_rows(r: BacktestResult, ticker: str, period: str) -> List[dict]:
    """Convert a BacktestResult to per-day export rows (for --full)."""
    rows = []
    for d in r.days:
        rows.append({
            "ticker": ticker,
            "period": period,
            "model": r.model_name,
            "date": d.date,
            "predicted": d.predicted,
            "actual": d.actual,
            "correct": d.correct,
            "confidence": round(d.confidence, 4),
            "trade_pnl": round(d.trade_pnl, 6),
            "close_before": round(d.close_before, 4),
            "close_actual": round(d.close_actual, 4),
        })
    return rows


# ------------------------------------------------------------------
# Display helpers
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


def print_profit_analysis(all_results):
    """Detailed profit metrics + streaks."""
    print(f"\n  {'PROFIT ANALYSIS':=^66}")
    print(f"  {'MODEL':<25} | {'RETURN':<10} | {'P.FACTOR':<10} | {'AVG WIN':<10} | {'AVG LOSS':<10}")
    print(f"  {'-' * 70}")

    for r in all_results:
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "∞"
        avg_w = f"{r.avg_win:+.2%}" if r.win_trades > 0 else "n/a"
        avg_l = f"{r.avg_loss:+.2%}" if r.loss_trades > 0 else "n/a"
        print(f"  {r.model_name:<25} | {r.total_return:<+10.2%} | {pf_str:<10} | {avg_w:<10} | {avg_l:<10}")

    # Streaks table
    print(f"\n  {'STREAKS':=^66}")
    print(f"  {'MODEL':<25} | {'MAX WIN':<10} | {'MAX LOSS':<10} | {'AVG WIN':<10} | {'AVG LOSS':<10}")
    print(f"  {'-' * 70}")

    for r in all_results:
        print(f"  {r.model_name:<25} | {r.longest_win_streak:<10} | {r.longest_loss_streak:<10} "
              f"| {r.avg_win_streak:<10.1f} | {r.avg_loss_streak:<10.1f}")

    if all_results:
        best_model = max(all_results, key=lambda r: r.total_return)
        worst_model = min(all_results, key=lambda r: r.total_return)
        print(f"\n  Most profitable:      {best_model.model_name} ({best_model.total_return:+.2%})")
        print(f"  Least profitable:     {worst_model.model_name} ({worst_model.total_return:+.2%})")

        best_pf = max(all_results, key=lambda r: r.profit_factor)
        pf_str = f"{best_pf.profit_factor:.2f}" if best_pf.profit_factor < 100 else "∞"
        print(f"  Best profit factor:   {best_pf.model_name} ({pf_str})")

        best_streak = max(all_results, key=lambda r: r.longest_win_streak)
        worst_streak = max(all_results, key=lambda r: r.longest_loss_streak)
        print(f"  Longest win streak:   {best_streak.model_name} ({best_streak.longest_win_streak} days)")
        print(f"  Longest loss streak:  {worst_streak.model_name} ({worst_streak.longest_loss_streak} days)")


# ------------------------------------------------------------------
# Single-period backtest
# ------------------------------------------------------------------

def run_single_backtest(api, backtester, ticker, df, period, n_days, full):
    """Run backtest for one ticker × one period. Returns list of BacktestResult."""
    filtered = filter_by_period(df, period)

    if len(filtered) < n_days + 20:
        return []

    sentiment_score, headlines = api._process_news_with_db(ticker)
    has_news = len(headlines) > 0

    variants = [
        (api.knn,              "k-NN",                  False, 0.0),
        (api.knn,              "k-NN Time-Weighted",     True,  0.0),
        (api.knn_enhanced,     "k-NN Enhanced",          False, 0.0),
        (api.knn_enhanced,     "k-NN Enh. TW",           True,  0.0),
        (api.linreg,           "LinReg",                 False, 0.0),
        (api.linreg,           "LinReg Time-Weighted",   True,  0.0),
        (api.linreg_enhanced,  "LinReg Enhanced",        False, 0.0),
        (api.linreg_enhanced,  "LinReg Enh. TW",         True,  0.0),
    ]

    if has_news:
        variants.extend([
            (api.knn,              "k-NN TW + News",       True,  sentiment_score),
            (api.knn_enhanced,     "k-NN Enh. TW + News",  True,  sentiment_score),
            (api.linreg,           "LinReg TW + News",     True,  sentiment_score),
            (api.linreg_enhanced,  "LinReg Enh. TW + News", True,  sentiment_score),
        ])

    # Add LSTM if a trained model exists for this ticker+period
    if api.lstm_available:
        lstm_model = api._load_lstm(ticker, period)
        if lstm_model:
            variants.append((lstm_model, "LSTM", False, 0.0))
            if has_news:
                variants.append((lstm_model, "LSTM + News", False, sentiment_score))
        else:
            # Log once per ticker+period (not per variant)
            print(f"  NOTE: No trained LSTM model for {ticker} (period={period}). "
                  f"Train with: uv run python train.py --ticker {ticker} --period {period}")

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


def run_backtest(tickers: list[str], n_days: int, full: bool = False,
                 period: str = "max", output: Optional[str] = None):
    """Standard single-period backtest mode."""
    api = StockAppAPI()
    backtester = Backtester(n_days=n_days)
    all_export_rows = []

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
        print(f"\n  {'MODEL':<25} | {'ACCURACY':<10} | {'RETURN':<10} | {'P.FACTOR':<10} | {'W/L':<8}")
        print(f"  {'-' * 73}")
        for r in results:
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 100 else "∞"
            print(f"  {r.model_name:<25} | {r.accuracy:<10.1%} | {r.total_return:<+10.2%} "
                  f"| {pf_str:<10} | {r.win_trades}/{r.loss_trades}")

        if has_news:
            print(f"\n  News headlines used for sentiment ({sentiment_score:+.2f}):")
            for h in headlines[:3]:
                print(f"    > {h}")

        if full:
            print_consensus(results, n_days)
            print_direction_accuracy(results)
            print_confidence_calibration(results)
            print_profit_analysis(results)
            print_next_day_forecast(results)
        else:
            best = max(results, key=lambda r: (r.accuracy, r.correct))
            print(f"\n  Day-by-day ({best.model_name}):")
            print(best.summary().split("\n", 3)[-1])
            print(f"\n  (use --full for consensus view, direction stats, "
                  f"and confidence calibration)")

        # Collect export rows
        if output:
            for r in results:
                if full:
                    # --full: export day-by-day rows
                    all_export_rows.extend(result_to_daily_rows(r, ticker, period))
                else:
                    # basic: export summary rows
                    all_export_rows.append(result_to_summary_row(r, ticker, period))

        print(f"\n{'*' * 70}")

    # Export
    if output and all_export_rows:
        export_rows(all_export_rows, output)


# ------------------------------------------------------------------
# Cross-period comparison mode
# ------------------------------------------------------------------

def run_compare_periods(tickers: list[str], n_days: int, output: Optional[str] = None):
    """Run backtest across all periods for each ticker, find the optimal period."""
    api = StockAppAPI()
    backtester = Backtester(n_days=n_days)
    all_export_rows = []

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

        best_per_model = {}
        all_combos = []

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

                    all_combos.append((
                        model_name, period, acc, r.correct, r.test_days,
                        r.total_return, r.profit_factor,
                        r.longest_win_streak, r.longest_loss_streak,
                        r.avg_win_streak, r.avg_loss_streak,
                    ))

                    # Collect for export
                    all_export_rows.append(result_to_summary_row(r, ticker, period))

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

        period_votes = {}
        for model_name, (bp, ba) in best_per_model.items():
            period_votes[bp] = period_votes.get(bp, 0) + 1
        overall_best = max(period_votes, key=period_votes.get)
        print(f"\n  Most popular period: {overall_best} "
              f"({period_votes[overall_best]}/{len(best_per_model)} models peak here)")

        # --- Overall top combinations ---
        print(f"\n  {'TOP COMBINATIONS (model + period)':=^70}")

        all_combos.sort(key=lambda x: (x[5], x[2]), reverse=True)

        top_return = all_combos[0][5]
        top_combos = [c for c in all_combos if c[5] == top_return]

        if len(top_combos) == 1:
            c = top_combos[0]
            pf_str = f"{c[6]:.2f}" if c[6] < 100 else "∞"
            print(f"\n  ★ BEST: {c[0]} + {c[1]} → return {c[5]:+.2%}, "
                  f"accuracy {c[2]:.0%} ({c[3]}/{c[4]}), PF {pf_str}")
        else:
            print(f"\n  ★ TIED AT {top_return:+.2%} return:")
            for c in top_combos:
                pf_str = f"{c[6]:.2f}" if c[6] < 100 else "∞"
                print(f"    • {c[0]} + {c[1]} (acc {c[2]:.0%}, PF {pf_str})")

        # Top 5 by return
        unique_returns = len(set(round(c[5], 6) for c in all_combos))
        if unique_returns > 1:
            print(f"\n  Top 5 by return:")
            print(f"  {'#':<4} {'MODEL':<25} {'PERIOD':<8} {'RETURN':<10} {'ACC.':<8} {'P.FACTOR':<10}")
            print(f"  {'-' * 70}")
            shown = set()
            rank = 0
            for c in all_combos:
                key = (c[0], c[1])
                if key in shown:
                    continue
                shown.add(key)
                rank += 1
                pf_str = f"{c[6]:.2f}" if c[6] < 100 else "∞"
                marker = " ★" if c[5] == top_return else ""
                print(f"  {rank:<4} {c[0]:<25} {c[1]:<8} {c[5]:<+10.2%} {c[2]:<8.0%} {pf_str:<10}{marker}")
                if rank >= 5:
                    break

        # --- Streak summary ---
        print(f"\n  {'STREAK ANALYSIS':=^70}")
        print(f"  {'MODEL + PERIOD':<35} | {'MAX W':<7} | {'MAX L':<7} | {'AVG W':<7} | {'AVG L':<7}")
        print(f"  {'-' * 70}")

        def streak_score(c):
            avg_w = c[9] if c[9] > 0 else 0.1
            avg_l = c[10] if c[10] > 0 else 0.1
            return avg_w / avg_l

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

        worst_loss_streak = max(all_combos, key=lambda c: c[8])
        best_win_streak = max(all_combos, key=lambda c: c[7])
        print(f"\n  Best win streak:    {best_win_streak[0]} + {best_win_streak[1]} "
              f"({best_win_streak[7]} consecutive wins)")
        print(f"  Worst loss streak:  {worst_loss_streak[0]} + {worst_loss_streak[1]} "
              f"({worst_loss_streak[8]} consecutive losses)")

        print(f"\n{'*' * 80}")

    # Export
    if output and all_export_rows:
        export_rows(all_export_rows, output)


def main():
    parser = argparse.ArgumentParser(description="MarketPulse AI – Backtest")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Specific tickers to backtest (overrides --stocks/--crypto)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--stocks", action="store_true",
        help=f"Backtest only stocks: {STOCKS}"
    )
    group.add_argument(
        "--crypto", action="store_true",
        help=f"Backtest only crypto: {CRYPTO}"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Backtest all tickers (stocks + crypto)"
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_BACKTEST_DAYS,
        help=f"Number of days to hold out for testing (default: {DEFAULT_BACKTEST_DAYS})"
    )
    parser.add_argument(
        "--period", default=DEFAULT_PERIOD, choices=VALID_PERIODS,
        help=f"How much historical data to train on (default: {DEFAULT_PERIOD})"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Show detailed day-by-day consensus, direction accuracy, "
             "and confidence calibration"
    )
    parser.add_argument(
        "--compare-periods", action="store_true",
        help="Run backtest across all periods and show which period works best"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Export results to CSV or JSON. Content depends on mode: "
             "basic → summary per model, --full → day-by-day per model, "
             "--compare-periods → summary per model × period"
    )
    args = parser.parse_args()

    # Determine tickers
    if args.tickers:
        tickers = args.tickers
    elif args.stocks:
        tickers = STOCKS
    elif args.crypto:
        tickers = CRYPTO
    elif args.all:
        tickers = ALL_TICKERS
    else:
        tickers = ALL_TICKERS[:3] if len(ALL_TICKERS) >= 3 else ALL_TICKERS

    if args.compare_periods:
        run_compare_periods(tickers, args.days, args.output)
    else:
        run_backtest(tickers, args.days, args.full, args.period, args.output)


if __name__ == "__main__":
    main()
