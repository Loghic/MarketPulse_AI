"""
run_all.py – Batch backtest runner.

Runs --compare-periods for each ticker and saves individual CSV reports
to the results/ directory. Filename includes days and flags used.

Output:
    results/
    ├── AAPL_20d.csv                        # basic
    ├── AAPL_20d_bh.csv                     # with buy-and-hold
    ├── AAPL_20d_fee005.csv                 # with 0.05% fee
    ├── AAPL_20d_fee010_bh.csv              # both
    └── _summary_20d_fee005_bh_20260427.csv # summary of all tickers

Usage:
    uv run python run_all.py --days 20
    uv run python run_all.py --days 50 --fees 0.1 --buy-hold
    uv run python run_all.py --stocks --days 20
    uv run python run_all.py --tickers AAPL NVDA --days 20 --buy-hold
"""

import argparse
import csv
from pathlib import Path
from datetime import datetime

from interface.api import StockAppAPI
from engine.backtester import Backtester
from engine.backtest_helpers import (
    filter_by_period, run_single_backtest, result_to_summary_row, pf_str,
)
from config import (
    ALL_TICKERS, STOCKS, CRYPTO, ALL_PERIODS, DEFAULT_TRADING_FEE_PCT,
)


RESULTS_DIR = Path("results")


def build_suffix(n_days: int, fee_pct: float, buy_hold: bool) -> str:
    """Build filename suffix from options, e.g. '20d_fee010_bh'."""
    parts = [f"{n_days}d"]
    if fee_pct > 0:
        # 0.05 → "fee005", 0.1 → "fee010"
        parts.append(f"fee{fee_pct * 100:03.0f}")
    if buy_hold:
        parts.append("bh")
    return "_".join(parts)


def run_ticker_comparison(api, ticker, n_days, fee_pct, buy_hold, results_dir, suffix):
    """Run compare-periods for one ticker, save CSV, return best combo."""

    df = api.get_data(ticker, period="max")
    if df.empty:
        print(f"  Skipping {ticker}: no data")
        return None

    backtester = Backtester(n_days=n_days, fee_pct=fee_pct)
    all_rows = []
    all_combos = []

    for period in ALL_PERIODS:
        results = run_single_backtest(
            api, backtester, ticker, df, period, n_days, full=False
        )
        if not results:
            filtered = filter_by_period(df, period)
            print(f"  {ticker} period={period}: skipped ({len(filtered)} rows)")
            continue

        for r in results:
            all_rows.append(result_to_summary_row(r, ticker, period))
            all_combos.append({
                "ticker": ticker,
                "period": period,
                "model": r.model_name,
                "accuracy": r.accuracy,
                "total_return": r.total_return,
                "buy_hold_return": r.buy_hold_return,
                "profit_factor": r.profit_factor,
                "fee_pct": r.fee_pct,
                "correct": r.correct,
                "total": r.test_days,
                "win_trades": r.win_trades,
                "loss_trades": r.loss_trades,
                "longest_win_streak": r.longest_win_streak,
                "longest_loss_streak": r.longest_loss_streak,
                "avg_win_streak": r.avg_win_streak,
                "avg_loss_streak": r.avg_loss_streak,
            })

    if not all_rows:
        return None

    # Save individual ticker CSV
    ticker_file = results_dir / f"{ticker}_{suffix}.csv"
    with open(ticker_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Find best combo by return
    best = max(all_combos, key=lambda c: (c["total_return"], c["accuracy"]))
    pf = best["profit_factor"]

    line = (f"  {ticker:<10} → {best['model']:<25} {best['period']:<6} "
            f"return={best['total_return']:+.4%}  acc={best['accuracy']:.0%}  "
            f"PF={pf_str(pf)}  "
            f"W{best['longest_win_streak']}/L{best['longest_loss_streak']}")
    if buy_hold:
        bh = best["buy_hold_return"]
        beats = "✓" if best["total_return"] > bh else "✗"
        line += f"  B&H={bh:+.4%} {beats}"
    line += f"  → {ticker_file.name}"
    print(line)

    return best


def main():
    parser = argparse.ArgumentParser(
        description="MarketPulse AI – Batch backtest (one CSV per ticker)"
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stocks", action="store_true")
    group.add_argument("--crypto", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--fees", type=float, default=DEFAULT_TRADING_FEE_PCT,
                        help=f"Fee %% per side (default: {DEFAULT_TRADING_FEE_PCT})")
    parser.add_argument("--buy-hold", action="store_true",
                        help="Include buy-and-hold comparison")
    parser.add_argument("--dir", type=str, default="results")
    args = parser.parse_args()

    # Tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.stocks:
        tickers = STOCKS
    elif args.crypto:
        tickers = CRYPTO
    elif args.all:
        tickers = ALL_TICKERS
    else:
        tickers = ALL_TICKERS

    results_dir = Path(args.dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = build_suffix(args.days, args.fees, args.buy_hold)
    timestamp = datetime.now().strftime("%Y%m%d")

    fee_info = f", fees={args.fees}% per side" if args.fees > 0 else ""
    bh_info = ", vs buy-and-hold" if args.buy_hold else ""

    print(f"{'=' * 80}")
    print(f" BATCH BACKTEST: {len(tickers)} tickers × {len(ALL_PERIODS)} periods "
          f"× {args.days} days{fee_info}{bh_info}")
    print(f" Output: {results_dir.resolve()}/")
    print(f"{'=' * 80}\n")

    api = StockAppAPI()
    best_per_ticker = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...")
        best = run_ticker_comparison(
            api, ticker, args.days, args.fees, args.buy_hold,
            results_dir, suffix,
        )
        if best:
            best_per_ticker.append(best)

    # --- Summary ---
    if best_per_ticker:
        summary_file = results_dir / f"_summary_{suffix}_{timestamp}.csv"
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=best_per_ticker[0].keys())
            writer.writeheader()
            writer.writerows(best_per_ticker)

        print(f"\n{'=' * 80}")
        print(f" SUMMARY: Best model+period per ticker")
        print(f"{'=' * 80}")

        header = (f"\n  {'TICKER':<10} {'MODEL':<25} {'PERIOD':<8} "
                  f"{'RETURN':<12} {'ACC.':<8} {'PF':<8} {'STREAKS':<10}")
        if args.buy_hold:
            header += f" {'B&H':<12} {'BEAT?'}"
        print(header)
        print(f"  {'-' * (84 + (18 if args.buy_hold else 0))}")

        for b in best_per_ticker:
            line = (f"  {b['ticker']:<10} {b['model']:<25} {b['period']:<8} "
                    f"{b['total_return']:<+12.4%} {b['accuracy']:<8.0%} "
                    f"{pf_str(b['profit_factor']):<8} "
                    f"W{b['longest_win_streak']}/L{b['longest_loss_streak']:<7}")
            if args.buy_hold:
                bh = b["buy_hold_return"]
                beats = "✓" if b["total_return"] > bh else "✗"
                line += f" {bh:<+12.4%} {beats}"
            print(line)

        overall = max(best_per_ticker, key=lambda b: b["total_return"])
        print(f"\n  ★ Overall best: {overall['ticker']} + {overall['model']} + "
              f"{overall['period']} → {overall['total_return']:+.4%} "
              f"(PF {pf_str(overall['profit_factor'])})")

        if args.buy_hold:
            beating_bh = sum(1 for b in best_per_ticker
                             if b["total_return"] > b["buy_hold_return"])
            print(f"  Models beating Buy&Hold: {beating_bh}/{len(best_per_ticker)}")

        print(f"\n  Individual CSVs: {results_dir}/{{TICKER}}_{suffix}.csv")
        print(f"  Summary CSV:     {summary_file}")

    print(f"\n{'*' * 80}")


if __name__ == "__main__":
    main()
