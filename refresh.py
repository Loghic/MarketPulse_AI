"""
refresh.py – Download latest prices and news for all tickers.

Usage:
    uv run python refresh.py              # all tickers
    uv run python refresh.py --stocks     # only stocks
    uv run python refresh.py --crypto     # only crypto
    uv run python refresh.py --tickers AAPL NVDA BTC-USD
"""

import argparse
import sys

from config import ALL_TICKERS, CRYPTO, STOCKS
from engine.logger import get_logger
from interface.api import StockAppAPI

sys.stdout.reconfigure(encoding="utf-8")


log = get_logger("refresh")


def main():
    parser = argparse.ArgumentParser(description="MarketPulse AI – Refresh prices & news")
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

    api = StockAppAPI()
    api.refresh_tickers(tickers)


if __name__ == "__main__":
    main()
