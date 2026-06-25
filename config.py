"""
config.py – Centralized configuration for tickers, periods, and defaults.

The asset universe is data-driven: one ``AssetClass`` entry per class, and the
flat names the rest of the codebase imports (``STOCKS``, ``CRYPTO``,
``ALL_TICKERS``, ``STOCK_BENCHMARKS`` …) plus ``get_benchmarks`` and
``TICKER_NAMES`` are *derived* from it. To add a ticker, append to a class's
``tickers``; to add a whole class, append an ``AssetClass``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ==================================================================
# Asset universe (single source of truth)
# ==================================================================


@dataclass(frozen=True)
class AssetClass:
    """One tradeable asset class.

    key        : short id, also the ``asset_type`` stored in the DB.
    label      : human-readable name (reports / GUI).
    cli_flag   : argparse flag stem, e.g. ``"stocks"`` -> ``--stocks``.
    tickers    : yfinance symbols traded in this class.
    benchmarks : buy-&-hold comparison set (the ticker itself is auto-excluded).
    news_names : ticker -> GDELT search query (the bare symbol matches few
                 articles; a company / phrase matches many). Anything omitted
                 falls back to the symbol minus ``-USD`` in news_sources.
    """

    key: str
    label: str
    cli_flag: str
    tickers: list[str]
    benchmarks: list[str]
    news_names: dict[str, str] = field(default_factory=dict)


# Equity ETFs proxy the index / commodity / FX classes so volume-based features
# (RSI/MACD/volatility + LSTM) keep working — index spot (^GSPC) and FX spot
# (EURUSD=X) return no volume from yfinance. VOO / QQQM (not SPY / QQQ) are used
# for the indices so they stay distinct from the stock benchmark set below.
ASSET_CLASSES: list[AssetClass] = [
    AssetClass(
        key="stock",
        label="Stocks",
        cli_flag="stocks",
        tickers=[
            "AAPL",
            "MSFT",
            "NVDA",
            "META",
            "GOOGL",
            "AMD",
            "TSM",
            "ASML",
            "AVGO",
            "TSLA",
            "INTC",
        ],
        benchmarks=["SPY", "QQQ"],
        news_names={
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "NVDA": "NVIDIA",
            "META": "Meta Platforms",
            "GOOGL": "Alphabet Google",
            "AMD": "AMD",
            "TSM": "TSMC Taiwan Semiconductor",
            "ASML": "ASML",
            "AVGO": "Broadcom",
            "TSLA": "Tesla",
            "INTC": "Intel",
        },
    ),
    AssetClass(
        key="crypto",
        label="Crypto",
        cli_flag="crypto",
        tickers=["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
        benchmarks=["BTC-USD"],
        news_names={
            "BTC-USD": "Bitcoin",
            "ETH-USD": "Ethereum",
            "SOL-USD": "Solana",
            "BNB-USD": "Binance Coin",
        },
    ),
    AssetClass(
        key="commodity",
        label="Commodities",
        cli_flag="commodities",
        tickers=["GLD"],  # SPDR Gold Shares (gold ETF proxy)
        benchmarks=["SPY"],
        news_names={"GLD": "gold price"},
    ),
    AssetClass(
        key="index",
        label="Indices",
        cli_flag="indices",
        tickers=["VOO", "QQQM"],  # S&P 500 (Vanguard) / Nasdaq-100 (Invesco) ETF proxies
        benchmarks=["SPY", "QQQ"],
        news_names={"VOO": "S&P 500", "QQQM": "Nasdaq 100"},
    ),
    AssetClass(
        key="fx",
        label="FX",
        cli_flag="fx",
        tickers=["FXE"],  # Invesco CurrencyShares Euro Trust (EUR/USD ETF proxy)
        benchmarks=["SPY"],
        news_names={"FXE": "euro dollar exchange rate"},
    ),
    # Less-efficient corners: small-caps, sector ETFs, and commodity-sensitive
    # names — where short-horizon predictability is *most* plausible. The point
    # is to test the no-edge thesis where it's hardest to hold, not mega-caps.
    # All are liquid with real volume so the volume-based features still work.
    AssetClass(
        key="smallcap",
        label="Small-cap & Sector",
        cli_flag="smallcap",
        tickers=[
            "IWM",  # Russell 2000 small-cap ETF
            "XLE",  # Energy sector ETF
            "XLF",  # Financials sector ETF
            "XLU",  # Utilities sector ETF
            "XOM",  # ExxonMobil (oil-sensitive single name)
            "FCX",  # Freeport-McMoRan (copper-sensitive single name)
        ],
        benchmarks=["SPY"],
        news_names={
            "IWM": "Russell 2000 small cap stocks",
            "XLE": "energy sector stocks oil",
            "XLF": "financial sector banks stocks",
            "XLU": "utilities sector stocks",
            "XOM": "ExxonMobil oil",
            "FCX": "Freeport-McMoRan copper",
        },
    ),
]

# --- Derived lookups (keep the old flat names working) ------------
_CLASS_BY_KEY: dict[str, AssetClass] = {ac.key: ac for ac in ASSET_CLASSES}

STOCKS: list[str] = _CLASS_BY_KEY["stock"].tickers
CRYPTO: list[str] = _CLASS_BY_KEY["crypto"].tickers
COMMODITIES: list[str] = _CLASS_BY_KEY["commodity"].tickers
INDICES: list[str] = _CLASS_BY_KEY["index"].tickers
FX: list[str] = _CLASS_BY_KEY["fx"].tickers

# Combined list — used as default in main.py and backtest.py (--all)
ALL_TICKERS: list[str] = [t for ac in ASSET_CLASSES for t in ac.tickers]

# Benchmarks: stocks vs SPY+QQQ, crypto vs BTC-USD, others vs SPY (self excluded).
STOCK_BENCHMARKS: list[str] = _CLASS_BY_KEY["stock"].benchmarks  # S&P 500, Nasdaq 100
CRYPTO_BENCHMARKS: list[str] = _CLASS_BY_KEY["crypto"].benchmarks
# Every benchmark symbol that must be downloaded even if not itself tradeable.
ALL_BENCHMARKS: list[str] = sorted({b for ac in ASSET_CLASSES for b in ac.benchmarks})

# ticker -> asset_type key (for db_manager / data_downloader tagging, if wired)
ASSET_TYPE: dict[str, str] = {t: ac.key for ac in ASSET_CLASSES for t in ac.tickers}

# ticker -> GDELT search query. Single source; engine/news_sources imports this.
TICKER_NAMES: dict[str, str] = {t: n for ac in ASSET_CLASSES for t, n in ac.news_names.items()}

# cli_flag -> asset-class key, for building the --stocks/--crypto/... selectors
SCOPE_FLAGS: dict[str, str] = {ac.cli_flag: ac.key for ac in ASSET_CLASSES}


def asset_type_of(ticker: str) -> str:
    """Asset-type key for a ticker (defaults to 'stock' for benchmark-only symbols like SPY)."""
    return ASSET_TYPE.get(ticker, "stock")


def get_benchmarks(ticker: str) -> list[str]:
    """Return benchmark tickers for comparison, excluding the ticker itself."""
    ac = _CLASS_BY_KEY.get(asset_type_of(ticker))
    return [b for b in (ac.benchmarks if ac else STOCK_BENCHMARKS) if b != ticker]


def tickers_for_scope(*keys: str) -> list[str]:
    """Tickers for one or more asset-class keys, e.g. tickers_for_scope('stock', 'index')."""
    wanted = set(keys)
    return [t for ac in ASSET_CLASSES if ac.key in wanted for t in ac.tickers]


# ==================================================================
# Periods
# ==================================================================

ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]

# Default period for single-period backtests and predictions
DEFAULT_PERIOD = "max"

# ------------------------------------------------------------------
# Defaults for CLI
# ------------------------------------------------------------------

DEFAULT_BACKTEST_DAYS = 5

# ==================================================================
# Logging & display mode
# ==================================================================

# "cli" = verbose logging + progress bars (for terminal use)
# "gui" = minimal logging, WARNING+ only (for future web/desktop UI)
LOG_MODE = "cli"

# Log level override (DEBUG, INFO, WARNING, ERROR). None = auto from LOG_MODE.
LOG_LEVEL = None

# ==================================================================
# Trading
# ==================================================================

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

# Stop-loss sweep — the default set of levels the backtest --sl-sweep flag
# iterates over (each model runs once per level; 0 = the no-SL baseline).
# Wide/off is usually best for daily holds (a 10%+ intraday trigger is rare
# for large-caps); most informative on volatile names.
SL_SWEEP = [0.0, 5.0, 10.0, 15.0]

# Confidence gating. Days whose model confidence is below this
# threshold are sat out (flat): 0 P&L, no fee, and excluded from the traded-day
# accuracy. 0.0 = disabled (trade every day). The sweep below is what the
# backtest --confidence-sweep flag iterates over.
DEFAULT_MIN_CONFIDENCE = 0.0
CONFIDENCE_SWEEP = [0.0, 0.55, 0.60, 0.65, 0.70]

# News-aware baselines: how strong the per-day sentiment must be (absolute
# value, sentiment ∈ [-1, 1]) before it overrides the baseline's price rule.
# 0.15 matches the POSITIVE/NEGATIVE cutoff used elsewhere (api.get_prediction).
BASELINE_NEWS_THRESHOLD = 0.15

# ==================================================================
# News & sentiment defaults
# ==================================================================

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

# ==================================================================
# Forecasting models (Prophet, Chronos-2, Kronos)
# ==================================================================
# (model_type, display label). backtest_helpers iterates this and silently
# skips any whose library isn't installed. Add TiRex here later.

# Kronos (financial K-line foundation model). NOT a pip package — clone it as a
# sibling of this repo (see docs/forecasting.md). Override location via KRONOS_PATH.
KRONOS_PATH = None  # None -> ../Kronos (sibling of repo root)
KRONOS_MODEL_ID = "NeoQuasar/Kronos-small"  # small=24.7M/ctx512; base=102M; mini=4.1M/ctx2048
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"  # use Kronos-Tokenizer-2k for mini
KRONOS_MAX_CONTEXT = 512  # 512 for small/base, 2048 for mini
KRONOS_SAMPLE_COUNT = 5  # internal averaging per predict() call
KRONOS_PROB_SAMPLES = 1  # >1 = empirical P(up) from N stochastic passes (slower)
KRONOS_T = 1.0  # sampling temperature
KRONOS_TOP_P = 0.9  # nucleus sampling

FORECAST_MODELS = [
    ("prophet", "Prophet"),
    ("chronos", "Chronos-2"),
    ("kronos", "Kronos"),
]

FORECAST_DEVICE: str | None = None  # None = auto (cuda if available else cpu)
CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_CONTEXT = 512  # most-recent closes used as context

# ==================================================================
# Model families (for the --models filter on backtest.py / run_all.py)
# ==================================================================
# Single source of truth: CLI key -> display-name prefix used in result rows.
# MODEL_FAMILIES is the list of valid --models choices; backtest_helpers derives
# the reverse (label -> key) map from this, so the names live in one place.
MODEL_FAMILY_LABELS: dict[str, str] = {
    "knn": "k-NN",
    "linreg": "LinReg",
    "lstm": "LSTM",
    "prophet": "Prophet",
    "chronos": "Chronos-2",
    "kronos": "Kronos",
    # Naive baselines — every row has the "Baseline" prefix.
    "baseline": "Baseline",
}
MODEL_FAMILIES = list(MODEL_FAMILY_LABELS)  # --models choices: ["knn", "linreg", ...]
