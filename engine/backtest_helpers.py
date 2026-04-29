"""
backtest_helpers.py – Shared helpers for backtest.py and run_all.py.

Contains: period filtering, direction accuracy, export row builders,
and display/print functions.
"""

import csv
import json
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from engine.backtester import BacktestResult
from engine.logger import get_logger
from engine.utils import period_to_start_date
from config import ALL_PERIODS, get_benchmarks

log = get_logger("helpers")


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------


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


def compute_benchmarks(api, ticker: str, day_results) -> Dict[str, float]:
    """
    Compute buy-and-hold return for benchmark indices over the same
    period as the backtest results.

    For stocks: compares against SPY and QQQ.
    For crypto: compares against BTC-USD (if the ticker isn't BTC itself).

    Returns dict like {"SPY": 0.035, "QQQ": 0.042} (as decimals).
    """
    if not day_results:
        return {}

    benchmarks = get_benchmarks(ticker)
    if not benchmarks:
        return {}

    # Date range from backtest results
    start_date = day_results[0].date   # first test day
    end_date = day_results[-1].date    # last test day

    results = {}
    for bench in benchmarks:
        try:
            df = api.get_data(bench, period="max")
            if df.empty:
                continue

            # Filter to the test period
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            period_df = df[mask]

            if len(period_df) < 2:
                continue

            entry = float(period_df["close"].iloc[0])
            exit_ = float(period_df["close"].iloc[-1])
            if entry > 0:
                results[bench] = round((exit_ - entry) / entry, 8)
        except Exception:
            continue

    return results


# ------------------------------------------------------------------
# Export row builders
# ------------------------------------------------------------------

def result_to_summary_row(r: BacktestResult, ticker: str, period: str,
                          benchmarks: Dict[str, float] | None = None) -> dict:
    """Convert a BacktestResult to a summary export row."""
    up_c, up_t, dn_c, dn_t = direction_accuracy(r)
    row = {
        "ticker": ticker,
        "period": period,
        "model": r.model_name,
        "accuracy": round(r.accuracy, 4),
        "correct": r.correct,
        "total": r.test_days,
        "total_return": round(r.total_return, 8),
        "buy_hold_return": round(r.buy_hold_return, 8),
        "profit_factor": round(r.profit_factor, 4),
        "max_drawdown": round(r.max_drawdown, 8),
        "buy_hold_max_drawdown": round(r.buy_hold_max_drawdown, 8),
        "sharpe_ratio": round(r.sharpe_ratio, 4),
        "sortino_ratio": round(r.sortino_ratio, 4),
        "fee_pct": r.fee_pct,
        "stop_loss_pct": r.stop_loss_pct,
        "stopped_out": r.stopped_out_count,
        "win_trades": r.win_trades,
        "loss_trades": r.loss_trades,
        "avg_win": round(r.avg_win, 8),
        "avg_loss": round(r.avg_loss, 8),
        "best_day": round(r.best_day, 8),
        "worst_day": round(r.worst_day, 8),
        "longest_win_streak": r.longest_win_streak,
        "longest_loss_streak": r.longest_loss_streak,
        "avg_win_streak": round(r.avg_win_streak, 1),
        "avg_loss_streak": round(r.avg_loss_streak, 1),
        "up_accuracy": round(up_c / up_t, 4) if up_t > 0 else None,
        "up_predictions": up_t,
        "down_accuracy": round(dn_c / dn_t, 4) if dn_t > 0 else None,
        "down_predictions": dn_t,
    }
    # Add benchmark returns as separate columns
    if benchmarks:
        for bench, ret in benchmarks.items():
            row[f"bench_{bench}"] = ret
    return row


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
            "trade_pnl": round(d.trade_pnl, 8),
            "trade_pnl_net": round(d.trade_pnl_net, 8),
            "fee_pct": r.fee_pct,
            "stopped_out": d.stopped_out,
            "exit_price": round(d.exit_price, 4),
            "close_before": round(d.close_before, 4),
            "close_actual": round(d.close_actual, 4),
        })
    return rows


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
            all_keys = {}
            for row in rows:
                for k in row.keys():
                    all_keys[k] = None
            writer = csv.DictWriter(f, fieldnames=list(all_keys.keys()),
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nResults exported to {output}")


# ------------------------------------------------------------------
# Run models for a single ticker × period
# ------------------------------------------------------------------

def run_single_backtest(api, backtester, ticker, df, period, n_days, full):
    """
    Run backtest for one ticker × one period. Returns list of BacktestResult.

    If backtester has stop-loss enabled, each model runs twice:
    once without SL (baseline) and once with SL — so you can compare.
    """
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

    # Add LSTM if available
    if api.lstm_available:
        lstm_model = api._load_lstm(ticker, period)
        if lstm_model:
            variants.append((lstm_model, "LSTM", False, 0.0))
            if has_news:
                variants.append((lstm_model, "LSTM + News", False, sentiment_score))
        else:
            log.info(f"No trained LSTM for {ticker} (period={period}). "
                     f"Train: uv run python train.py --ticker {ticker} --period {period}")

    # If stop-loss is enabled, also create a no-SL backtester for comparison
    has_sl = backtester.stop_loss_pct > 0
    if has_sl:
        from engine.backtester import Backtester as BT
        backtester_no_sl = BT(
            n_days=backtester.n_days,
            fee_pct=backtester.fee_pct,
            stop_loss_pct=0.0,
        )

    results = []
    for model, name, tw, sent in variants:
        # Run without SL first (baseline)
        if has_sl:
            result_no_sl = backtester_no_sl.run(
                model=model, model_name=name, df=filtered,
                ticker=ticker, use_time_weights=tw, sentiment_score=sent,
            )
            results.append(result_no_sl)

        # Run with SL (or the only run if SL is disabled)
        result = backtester.run(
            model=model,
            model_name=f"{name} SL{backtester.stop_loss_pct:g}%" if has_sl else name,
            df=filtered,
            ticker=ticker, use_time_weights=tw, sentiment_score=sent,
        )
        results.append(result)

    return results


# ------------------------------------------------------------------
# Display helpers
# ------------------------------------------------------------------

def pf_str(pf: float) -> str:
    """Format profit factor for display."""
    return f"{pf:.2f}" if pf < 100 else "∞"


def print_summary_table(results: List[BacktestResult], show_buy_hold: bool = False,
                        benchmarks: Dict[str, float] | None = None):
    """Print the summary table."""
    has_sl = any(r.stop_loss_pct > 0 for r in results)
    header = (f"  {'MODEL':<25} | {'ACC.':<8} | {'RETURN':<10} | "
              f"{'P.FACTOR':<10} | {'MAX DD':<10} | {'SHARPE':<8}")
    if has_sl:
        header += f" | {'SL':<4}"
    if show_buy_hold:
        header += f" | {'B&H':<10}"
    if benchmarks:
        for bench in benchmarks:
            header += f" | {bench:<10}"
    print(header)
    divider = 83 + (7 if has_sl else 0) + (13 if show_buy_hold else 0)
    divider += sum(13 for _ in (benchmarks or {}))
    print(f"  {'-' * divider}")

    for r in results:
        line = (f"  {r.model_name:<25} | {r.accuracy:<8.1%} | "
                f"{r.total_return:<+10.4%} | {pf_str(r.profit_factor):<10} | "
                f"{r.max_drawdown:<+10.4%} | {r.sharpe_ratio:<8.2f}")
        if has_sl:
            line += f" | {r.stopped_out_count:<4}"
        if show_buy_hold:
            line += f" | {r.buy_hold_return:<+10.4%}"
        if benchmarks:
            for bench, ret in benchmarks.items():
                line += f" | {ret:<+10.4%}"
        print(line)


def print_consensus(all_results, n_days):
    """Day-by-day consensus."""
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
        up_s = f"{up_c}/{up_t} ({up_c/up_t:.0%})" if up_t > 0 else "n/a"
        dn_s = f"{dn_c}/{dn_t} ({dn_c/dn_t:.0%})" if dn_t > 0 else "n/a"
        print(f"  {r.model_name:<25} | {up_s:<12} | {dn_s:<12}")


def print_confidence_calibration(all_results):
    """Are high-confidence predictions actually more accurate?"""
    print(f"\n  {'CONFIDENCE CALIBRATION':=^66}")
    print(f"  {'MODEL':<25} | {'High (>65%)':<16} | {'Low (≤65%)':<16}")
    print(f"  {'-' * 62}")
    for r in all_results:
        high = [d for d in r.days if d.confidence > 0.65]
        low = [d for d in r.days if d.confidence <= 0.65]
        ha = sum(1 for d in high if d.correct) / len(high) if high else 0
        la = sum(1 for d in low if d.correct) / len(low) if low else 0
        hs = f"{ha:.0%} ({len(high)} pred)" if high else "n/a"
        ls = f"{la:.0%} ({len(low)} pred)" if low else "n/a"
        print(f"  {r.model_name:<25} | {hs:<16} | {ls:<16}")


def print_profit_analysis(all_results, show_buy_hold: bool = False,
                          benchmarks: Dict[str, float] | None = None):
    """Detailed profit metrics + streaks."""
    print(f"\n  {'PROFIT ANALYSIS':=^66}")
    header = (f"  {'MODEL':<25} | {'RETURN':<10} | {'P.FACTOR':<10} | "
              f"{'AVG WIN':<10} | {'AVG LOSS':<10}")
    if show_buy_hold:
        header += f" | {'B&H':<10}"
    print(header)
    print(f"  {'-' * (70 + (13 if show_buy_hold else 0))}")

    for r in all_results:
        avg_w = f"{r.avg_win:+.4%}" if r.win_trades > 0 else "n/a"
        avg_l = f"{r.avg_loss:+.4%}" if r.loss_trades > 0 else "n/a"
        line = (f"  {r.model_name:<25} | {r.total_return:<+10.4%} | "
                f"{pf_str(r.profit_factor):<10} | {avg_w:<10} | {avg_l:<10}")
        if show_buy_hold:
            line += f" | {r.buy_hold_return:<+10.4%}"
        print(line)

    # Streaks
    print(f"\n  {'STREAKS':=^66}")
    print(f"  {'MODEL':<25} | {'MAX WIN':<10} | {'MAX LOSS':<10} | "
          f"{'AVG WIN':<10} | {'AVG LOSS':<10}")
    print(f"  {'-' * 70}")
    for r in all_results:
        print(f"  {r.model_name:<25} | {r.longest_win_streak:<10} | "
              f"{r.longest_loss_streak:<10} | {r.avg_win_streak:<10.1f} | "
              f"{r.avg_loss_streak:<10.1f}")

    # Risk metrics
    print(f"\n  {'RISK METRICS':=^66}")
    print(f"  {'MODEL':<25} | {'MAX DD':<10} | {'SHARPE':<10} | {'SORTINO':<10} | {'W/L':<8}")
    print(f"  {'-' * 70}")
    for r in all_results:
        sortino_s = f"{r.sortino_ratio:.2f}" if r.sortino_ratio < 100 else "∞"
        print(f"  {r.model_name:<25} | {r.max_drawdown:<+10.4%} | "
              f"{r.sharpe_ratio:<10.2f} | {sortino_s:<10} | "
              f"{r.win_trades}/{r.loss_trades}")

    if all_results:
        best = max(all_results, key=lambda r: r.total_return)
        worst = min(all_results, key=lambda r: r.total_return)
        print(f"\n  Most profitable:      {best.model_name} ({best.total_return:+.4%})")
        print(f"  Least profitable:     {worst.model_name} ({worst.total_return:+.4%})")
        bp = max(all_results, key=lambda r: r.profit_factor)
        print(f"  Best profit factor:   {bp.model_name} ({pf_str(bp.profit_factor)})")
        best_sharpe = max(all_results, key=lambda r: r.sharpe_ratio)
        print(f"  Best Sharpe ratio:    {best_sharpe.model_name} ({best_sharpe.sharpe_ratio:.2f})")
        least_dd = max(all_results, key=lambda r: r.max_drawdown)  # closest to 0
        print(f"  Smallest max DD:      {least_dd.model_name} ({least_dd.max_drawdown:+.4%})")
        bws = max(all_results, key=lambda r: r.longest_win_streak)
        wls = max(all_results, key=lambda r: r.longest_loss_streak)
        print(f"  Longest win streak:   {bws.model_name} ({bws.longest_win_streak} days)")
        print(f"  Longest loss streak:  {wls.model_name} ({wls.longest_loss_streak} days)")
        if show_buy_hold and all_results[0].buy_hold_return != 0:
            bh = all_results[0].buy_hold_return
            bh_dd = all_results[0].buy_hold_max_drawdown
            beat = sum(1 for r in all_results if r.total_return > bh)
            print(f"\n  Buy & Hold return:    {bh:+.4%}  (max DD: {bh_dd:+.4%})")
            print(f"  Models beating B&H:   {beat}/{len(all_results)}")

        if benchmarks:
            print(f"\n  {'BENCHMARK COMPARISON':=^66}")
            best_return = max(r.total_return for r in all_results)
            for bench, ret in benchmarks.items():
                beat = sum(1 for r in all_results if r.total_return > ret)
                marker = "✓" if best_return > ret else "✗"
                print(f"  {bench:<10} return: {ret:+.4%}  "
                      f"| Models beating: {beat}/{len(all_results)} {marker}")

        if any(r.stop_loss_pct > 0 for r in all_results):
            total_stopped = sum(r.stopped_out_count for r in all_results)
            total_trades = sum(r.test_days for r in all_results)
            print(f"\n  Stop-loss triggers:   {total_stopped}/{total_trades} trades "
                  f"({total_stopped/total_trades:.0%})")

    # Yearly performance (only shown if data spans multiple years)
    has_yearly = any(r.yearly_performance for r in all_results)
    if has_yearly:
        print_yearly_performance(all_results)


def print_yearly_performance(all_results):
    """Show performance breakdown by calendar year for the best model."""
    # Pick the model with most yearly data
    best = max(all_results, key=lambda r: len(r.yearly_performance))
    if not best.yearly_performance:
        return

    print(f"\n  {'YEARLY PERFORMANCE':=^66}")
    print(f"  Model: {best.model_name}\n")
    print(f"  {'YEAR':<8} | {'TRADES':<8} | {'ACC.':<8} | {'RETURN':<10} | "
          f"{'PF':<8} | {'MAX DD':<10}")
    print(f"  {'-' * 60}")

    for yp in best.yearly_performance:
        pf_s = f"{yp.profit_factor:.2f}" if yp.profit_factor < 100 else "∞"
        print(f"  {yp.year:<8} | {yp.trades:<8} | {yp.accuracy:<8.0%} | "
              f"{yp.total_return:<+10.4%} | {pf_s:<8} | {yp.max_drawdown:<+10.4%}")

    # Also show top 3 models' yearly summary if multiple have yearly data
    models_with_yearly = [r for r in all_results if r.yearly_performance]
    if len(models_with_yearly) > 1:
        years = sorted(set(yp.year for r in models_with_yearly for yp in r.yearly_performance))
        if len(years) >= 2:
            print(f"\n  {'YEARLY RETURN BY MODEL':=^66}")
            header = f"  {'MODEL':<25} |"
            for y in years:
                header += f" {y:^8} |"
            print(header)
            print(f"  {'-' * (28 + len(years) * 11)}")

            for r in models_with_yearly[:8]:  # limit to top 8 for readability
                row = f"  {r.model_name:<25} |"
                yp_map = {yp.year: yp for yp in r.yearly_performance}
                for y in years:
                    if y in yp_map:
                        row += f" {yp_map[y].total_return:>+7.2%} |"
                    else:
                        row += f" {'n/a':^8} |"
                print(row)


def print_next_day_forecast(all_results):
    """What would each model predict for the most recent holdout day?"""
    if not all_results or not all_results[0].days:
        return
    print(f"\n  {'NEXT-DAY SIGNAL (most recent holdout day)':=^66}")
    last_day = all_results[0].days[-1]
    print(f"  Date: {last_day.date}")
    print(f"  Price: {last_day.close_before:.2f} → {last_day.close_actual:.2f}\n")
    up_votes = down_votes = 0
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
