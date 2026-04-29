"""
refresh.py – Download latest prices and news for all tickers.

Pre-fetches data into SQLite so that main.py, backtest.py, and run_all.py
don't have to wait for downloads. Run this once at the start of your day.

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
    success = 0
    failed = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)

        # Prices
        try:
            df = api.get_data(ticker, period="max")
            rows = len(df)
            last = df["date"].iloc[-1] if not df.empty else "n/a"
        except Exception as e:
            print(f"PRICE ERROR: {e}")
            failed += 1
            continue

        # News
        try:
            score, headlines = api._process_news_with_db(ticker)
            news_count = len(headlines)
            if news_count > 0:
                label = "POS" if score > 0.15 else "NEG" if score < -0.15 else "NEU"
                news_str = f"{news_count} headlines ({label} {score:+.2f})"
            else:
                news_str = "no news"
        except Exception:
            news_str = "news error"

        print(f"{rows} rows (→ {last}), {news_str}")
        success += 1

    print(f"\n{'=' * 60}")
    print(f" DONE: {success} refreshed, {failed} failed")
    print(f" DB: data/market_data.db")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
