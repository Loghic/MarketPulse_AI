"""
config.py – Centralized configuration for tickers, periods, and defaults.

Add new tickers here. Everything else (main.py, backtest.py) reads from this file.
"""

# ------------------------------------------------------------------
# Tickers by category
# ------------------------------------------------------------------

STOCKS = [
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "NVDA",     # NVIDIA
    "META",     # Meta (Facebook)
    "GOOGL",    # Alphabet (Google)
    "AMD",      # AMD
    "TSM",      # Taiwan Semiconductor (TSMC)
    "ASML",     # ASML
    "AVGO",     # Broadcom
    "TSLA",     # Tesla
    "INTC",     # Intel
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
