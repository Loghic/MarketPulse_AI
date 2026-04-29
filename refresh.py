"""
refresh.py – Download latest prices and news for all tickers.

Pre-fetches data into SQLite so that main.py, backtest.py, and run_all.py
don't have to wait for downloads. Same as running main.py without --no-refresh,
but without running any models.

Usage:
    uv run python refresh.py              # all tickers
    uv run python refresh.py --stocks     # only stocks
    uv run python refresh.py --crypto     # only crypto
    uv run python refresh.py --tickers AAPL NVDA BTC-USD
"""

import argparse
from datetime import datetime

from interface.api import StockAppAPI
from config import ALL_TICKERS, STOCKS, CRYPTO


def main():
    parser = argparse.ArgumentParser(
        description="MarketPulse AI – Refresh prices & news"
    )
    parser.add_argument("--tickers", nargs="+", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stocks", action="store_true")
    group.add_argument("--crypto", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.stocks:
        tickers = STOCKS
    elif args.crypto:
        tickers = CRYPTO
    else:
        tickers = ALL_TICKERS

    print(f"{'=' * 60}")
    print(f" DATA REFRESH: {len(tickers)} tickers")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}\n")

    api = StockAppAPI()
    api.refresh_tickers(tickers)

    print(f"{'=' * 60}")
    print(f" DONE — DB: data/market_data.db")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
