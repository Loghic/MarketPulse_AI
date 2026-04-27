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

# ------------------------------------------------------------------
# Trading costs (applied per trade in backtesting)
# ------------------------------------------------------------------

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
