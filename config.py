"""
config.py – Centralized configuration for tickers, periods, and defaults.

Add new tickers here. Everything else (main.py, backtest.py) reads from this file.
"""

# ------------------------------------------------------------------
# Tickers by category
# ------------------------------------------------------------------

STOCKS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # NVIDIA
    "META",  # Meta (Facebook)
    "GOOGL",  # Alphabet (Google)
    "AMD",  # AMD
    "TSM",  # Taiwan Semiconductor (TSMC)
    "ASML",  # ASML
    "AVGO",  # Broadcom
    "TSLA",  # Tesla
    "INTC",  # Intel
]

CRYPTO = [
    "BTC-USD",  # Bitcoin
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana
]

# Combined list — used as default in main.py and backtest.py
ALL_TICKERS = STOCKS + CRYPTO

# ------------------------------------------------------------------
# Periods
# ------------------------------------------------------------------

ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]

# Default period for single-period backtests and predictions
DEFAULT_PERIOD = "max"

# ------------------------------------------------------------------
# Defaults for CLI
# ------------------------------------------------------------------

DEFAULT_BACKTEST_DAYS = 5

# ------------------------------------------------------------------
# Logging & display mode
# ------------------------------------------------------------------

# "cli" = verbose logging + progress bars (for terminal use)
# "gui" = minimal logging, WARNING+ only (for future web/desktop UI)
LOG_MODE = "cli"

# Log level override (DEBUG, INFO, WARNING, ERROR). None = auto from LOG_MODE.
LOG_LEVEL = None

# Stocks are compared against broad market indices
STOCK_BENCHMARKS = ["SPY", "QQQ"]  # S&P 500, Nasdaq 100

# Crypto is compared against Bitcoin
CRYPTO_BENCHMARKS = ["BTC-USD"]


# Returns the relevant benchmarks for a ticker
def get_benchmarks(ticker: str) -> list[str]:
    """Return benchmark tickers for comparison, excluding the ticker itself."""
    if "-USD" in ticker:
        return [b for b in CRYPTO_BENCHMARKS if b != ticker]
    return STOCK_BENCHMARKS


# Total cost per trade as a percentage of trade value.
# Covers: broker commission + bid-ask spread + slippage.
#
# Typical values:
#   Stocks (commission-free brokers like IBKR Lite, Robinhood):
#       0.01-0.05%  (spread + slippage only, no commission)
#   Stocks (traditional brokers):
#       0.05-0.15%  (commission + spread + slippage)
#   Crypto (exchanges like Binance, Coinbase):
#       0.05-0.20%  (maker/taker fee + spread)
#
# We buy AND sell each day → fee is applied TWICE per round-trip.
# Default 0.05% per side = 0.10% round-trip.
DEFAULT_TRADING_FEE_PCT = 0.05  # 0.05% per trade (buy or sell)

# Stop-loss as percentage. If the position moves against you by this much
# intraday, it's automatically closed at the stop-loss price.
# 0 = disabled (hold until end of day regardless).
# 2.0 = close if price drops 2% from entry (long) or rises 2% (short).
DEFAULT_STOP_LOSS_PCT = 0.0  # disabled by default

# ------------------------------------------------------------------
# News & sentiment defaults
# ------------------------------------------------------------------

# Which sentiment scorer to use by default.
#   "vader"   – fast, general-purpose, no GPU. Default.
#   "finbert" – ProsusAI/finbert. Slow first-load (~400 MB), best for
#               financial text. Requires `transformers` + `torch`.
#   "naive"   – keyword-matching baseline, zero deps.
DEFAULT_SENTIMENT_METHOD = "vader"

# Which news source(s) to use by default. Can be a single name or a list.
#   "yahoo"  – yfinance (limited history, no key)
#   "gdelt"  – GDELT 2.0 (years of history, no key, ~15 min indexing lag)
# Pass a list to combine them and dedupe by headline.
DEFAULT_NEWS_SOURCES: list[str] = ["yahoo"]

# How many days of news to look back when computing sentiment for a given
# prediction date. Only news strictly BEFORE the prediction date and
# within the window contribute. 0 disables the window (use everything older).
DEFAULT_NEWS_LOOKBACK_DAYS = 7

# Exponential half-life for time-decay weighting of news.
#   0.0 → all headlines weighted equally inside the lookback window
#   3.0 → a 3-day-old headline carries half the weight of a 0-day-old one
# Larger values mean a slower decay (older news still matters).
DEFAULT_NEWS_HALF_LIFE_DAYS = 3.0
