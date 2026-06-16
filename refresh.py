"""
refresh.py – Download latest prices and news for all tickers.

Usage:
    uv run python refresh.py                          # all tickers, defaults
    uv run python refresh.py --stocks                 # only stocks
    uv run python refresh.py --crypto                 # only crypto
    uv run python refresh.py --tickers AAPL NVDA BTC-USD

Historical news (one-off bulk fetch for the DB):
    # Pull 1 year of GDELT news for every stock, score with FinBERT
    uv run python refresh.py --stocks \\
        --news-source gdelt --news-history-days 365 \\
        --sentiment-method finbert --force-news

    # Combined Yahoo + GDELT, default scorer, ~3 months
    uv run python refresh.py --all \\
        --news-source yahoo gdelt --news-history-days 90 --force-news
"""

import argparse
import sys

from cli_helpers import add_scope_args, resolve_scope
from config import (
    ALL_TICKERS,
    DEFAULT_NEWS_LOOKBACK_DAYS,
    DEFAULT_SENTIMENT_METHOD,
)
from engine.logger import get_logger
from interface.api import StockAppAPI

sys.stdout.reconfigure(encoding="utf-8")


log = get_logger("refresh")


def main():
    parser = argparse.ArgumentParser(description="MarketPulse AI – Refresh prices & news")
    add_scope_args(parser)

    parser.add_argument(
        "--news-source",
        nargs="+",
        choices=["yahoo", "gdelt"],
        default=None,
        help=(
            "Which news provider(s) to use. One or more of {yahoo,gdelt}. "
            "Defaults to the DEFAULT_NEWS_SOURCES set in config.py."
        ),
    )
    parser.add_argument(
        "--sentiment-method",
        choices=["vader", "finbert", "naive"],
        default=DEFAULT_SENTIMENT_METHOD,
        help=f"Sentiment scorer for new headlines (default: {DEFAULT_SENTIMENT_METHOD}).",
    )
    parser.add_argument(
        "--news-history-days",
        type=int,
        default=DEFAULT_NEWS_LOOKBACK_DAYS,
        help=(
            "How many days of news to pull per ticker. "
            "Yahoo silently caps at ~3-7. GDELT honours larger values up to "
            "250 articles per call (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--force-news",
        action="store_true",
        help=(
            "Bypass the same-day cache. Use this when populating the DB "
            "with historical news for the first time."
        ),
    )

    args = parser.parse_args()

    tickers = resolve_scope(args, default=ALL_TICKERS)

    # When --news-source is given, pass it to both API construction (so the
    # scraper's default matches) and to refresh_tickers (per-call override).
    api = StockAppAPI(
        sentiment_method=args.sentiment_method,
        news_sources=args.news_source,
    )
    api.refresh_tickers(
        tickers,
        news_lookback_days=args.news_history_days,
        sentiment_method=args.sentiment_method,
        news_sources=args.news_source,
        force_news_refresh=args.force_news,
    )


if __name__ == "__main__":
    main()
