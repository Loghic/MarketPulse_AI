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
"""

import argparse
import csv
from pathlib import Path
from datetime import datetime

from interface.api import StockAppAPI
from engine.backtester import Backtester
from engine.backtest_helpers import (
    filter_by_period, run_single_backtest, result_to_summary_row,
    compute_benchmarks, pf_str,
)
from config import (
    ALL_TICKERS, STOCKS, CRYPTO, ALL_PERIODS,
    DEFAULT_TRADING_FEE_PCT, DEFAULT_STOP_LOSS_PCT,
    STOCK_BENCHMARKS, CRYPTO_BENCHMARKS,
)


RESULTS_DIR = Path("results")


def build_dir_name(scope: str, n_days: int, fee_pct: float,
                   stop_loss_pct: float, buy_hold: bool) -> str:
    """
    Build subdirectory name from run parameters.

    Examples:
        stocks_50d_fee003_bh
        crypto_20d_fee015_sl3
        all_50d_fee010_sl2_bh
        custom_20d
    """
    parts = [scope, f"{n_days}d"]
    if fee_pct > 0:
        parts.append(f"fee{fee_pct * 100:03.0f}")
    if stop_loss_pct > 0:
        parts.append(f"sl{stop_loss_pct:g}")
    if buy_hold:
        parts.append("bh")
    return "_".join(parts)


def run_ticker_comparison(api, ticker, n_days, fee_pct, stop_loss_pct,
                          buy_hold, run_dir):
    """Run compare-periods for one ticker, save CSV, return best combo."""

    df = api.get_data(ticker, period="max")
    if df.empty:
        print(f"  Skipping {ticker}: no data")
        return None

    backtester = Backtester(n_days=n_days, fee_pct=fee_pct,
                            stop_loss_pct=stop_loss_pct)
    all_rows = []
    all_combos = []
    bench = None

    for period in ALL_PERIODS:
        results = run_single_backtest(
            api, backtester, ticker, df, period, n_days, full=False
        )
        if not results:
            filtered = filter_by_period(df, period)
            print(f"  {ticker} period={period}: skipped ({len(filtered)} rows)")
            continue

        # Compute benchmarks once (same test dates across periods)
        if bench is None and buy_hold:
            bench = compute_benchmarks(api, ticker, results[0].days)

        for r in results:
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
    with open(ticker_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Find best combo by return
    best = max(all_combos, key=lambda c: (c["total_return"], c["accuracy"]))
    pf = best["profit_factor"]

    line = (f"  {ticker:<10} → {best['model']:<25} {best['period']:<6} "
            f"return={best['total_return']:+.4%}  PF={pf_str(pf)}  "
            f"DD={best['max_drawdown']:+.2%}  Sharpe={best['sharpe_ratio']:.2f}  "
            f"W{best['longest_win_streak']}/L{best['longest_loss_streak']}")
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
    parser.add_argument("--tickers", nargs="+", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stocks", action="store_true")
    group.add_argument("--crypto", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--fees", type=float, default=DEFAULT_TRADING_FEE_PCT,
                        help=f"Fee %% per side (default: {DEFAULT_TRADING_FEE_PCT})")
    parser.add_argument("--stop-loss", type=float, default=DEFAULT_STOP_LOSS_PCT,
                        help="Stop-loss %% (0=disabled)")
    parser.add_argument("--buy-hold", action="store_true",
                        help="Include buy-and-hold comparison")
    parser.add_argument("--no-refresh", action="store_true",
                        help="Skip data download, use only cached data from DB")
    parser.add_argument("--dir", type=str, default="results",
                        help="Root output directory (default: results/)")
    args = parser.parse_args()

    # Determine tickers and scope label
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        scope = "custom"
    elif args.stocks:
        tickers = STOCKS
        scope = "stocks"
    elif args.crypto:
        tickers = CRYPTO
        scope = "crypto"
    elif args.all:
        tickers = ALL_TICKERS
        scope = "all"
    else:
        tickers = ALL_TICKERS
        scope = "all"

    # Build output directory
    dir_name = build_dir_name(
        scope, args.days, args.fees, args.stop_loss, args.buy_hold
    )
    run_dir = Path(args.dir) / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    fee_info = f", fees={args.fees}% per side" if args.fees > 0 else ""
    sl_info = f", SL={args.stop_loss}%" if args.stop_loss > 0 else ""
    bh_info = ", vs buy-and-hold" if args.buy_hold else ""

    print(f"{'=' * 80}")
    print(f" BATCH BACKTEST: {len(tickers)} tickers × {len(ALL_PERIODS)} periods "
          f"× {args.days} days{fee_info}{sl_info}{bh_info}")
    print(f" Output: {run_dir.resolve()}/")
    print(f"{'=' * 80}\n")

    api = StockAppAPI()

    if not args.no_refresh:
        all_to_refresh = list(set(tickers + STOCK_BENCHMARKS + CRYPTO_BENCHMARKS))
        api.refresh_tickers(all_to_refresh)

    best_per_ticker = []

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...")
        best = run_ticker_comparison(
            api, ticker, args.days, args.fees, args.stop_loss,
            args.buy_hold, run_dir,
        )
        if best:
            best_per_ticker.append(best)

    # --- Summary ---
    if best_per_ticker:
        summary_file = run_dir / "_summary.csv"
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=best_per_ticker[0].keys())
            writer.writeheader()
            writer.writerows(best_per_ticker)

        print(f"\n{'=' * 80}")
        print(f" SUMMARY: Best model+period per ticker")
        print(f"{'=' * 80}")

        header = (f"\n  {'TICKER':<10} {'MODEL':<25} {'PERIOD':<8} "
                  f"{'RETURN':<12} {'PF':<8} {'MAX DD':<10} {'SHARPE':<8}")
        if args.stop_loss > 0:
            header += f" {'SL':<5}"
        if args.buy_hold:
            header += f" {'B&H':<12} {'BEAT?'}"
        print(header)
        divider_len = 84 + (7 if args.stop_loss > 0 else 0) + (18 if args.buy_hold else 0)
        print(f"  {'-' * divider_len}")

        for b in best_per_ticker:
            sortino_s = f"{b['sortino_ratio']:.2f}" if b['sortino_ratio'] < 100 else "∞"
            line = (f"  {b['ticker']:<10} {b['model']:<25} {b['period']:<8} "
                    f"{b['total_return']:<+12.4%} "
                    f"{pf_str(b['profit_factor']):<8} "
                    f"{b['max_drawdown']:<+10.4%} "
                    f"{b['sharpe_ratio']:<8.2f}")
            if args.stop_loss > 0:
                line += f" {b.get('stopped_out', 0):<5}"
            if args.buy_hold:
                bh = b["buy_hold_return"]
                beats = "✓" if b["total_return"] > bh else "✗"
                line += f" {bh:<+12.4%} {beats}"
            print(line)

        overall = max(best_per_ticker, key=lambda b: b["total_return"])
        print(f"\n  ★ Best return:  {overall['ticker']} + {overall['model']} + "
              f"{overall['period']} → {overall['total_return']:+.4%} "
              f"(PF {pf_str(overall['profit_factor'])}, "
              f"DD {overall['max_drawdown']:+.4%})")

        best_sharpe = max(best_per_ticker, key=lambda b: b["sharpe_ratio"])
        print(f"  ★ Best Sharpe:  {best_sharpe['ticker']} + {best_sharpe['model']} + "
              f"{best_sharpe['period']} → Sharpe {best_sharpe['sharpe_ratio']:.2f} "
              f"(return {best_sharpe['total_return']:+.4%})")

        if args.buy_hold:
            beating_bh = sum(1 for b in best_per_ticker
                             if b["total_return"] > b["buy_hold_return"])
            print(f"  Models beating Buy&Hold: {beating_bh}/{len(best_per_ticker)}")

        print(f"\n  Results: {run_dir}/")
        print(f"  Summary: {summary_file}")

    print(f"\n{'*' * 80}")


if __name__ == "__main__":
    main()
