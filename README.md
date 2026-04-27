# MarketPulse AI

Stock prediction engine combining k-NN, Linear Regression, and LSTM neural networks with VADER sentiment analysis. Built as a modular system with a clean separation between data layer, model engine, and interface — ready to plug into a web or desktop UI.

> **Disclaimer:** This is an educational/research project. Predictions are not financial advice.

## Quick Start

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
git clone <repo-url>
cd marketpulse-ai

uv venv
uv pip install -e .
uv run python main.py
```

For LSTM model support (optional):
```bash
uv pip install torch
```

## Supported tickers

Configured in `config.py`. To add a new ticker, edit that file — nothing else changes.

**Stocks:** AAPL, MSFT, NVDA, META, GOOGL, AMD, TSM, ASML, AVGO, TSLA, INTC

**Crypto:** BTC-USD, ETH-USD, SOL-USD

All CLI scripts support `--stocks`, `--crypto`, `--all`, or `--tickers`:

```bash
uv run python main.py --stocks
uv run python main.py --crypto
uv run python main.py --all
uv run python main.py --tickers AAPL NVDA BTC-USD
```

## Models

**k-NN** (naive + enhanced) — classifies next-day direction from return patterns. Enhanced adds volume, RSI, volatility, MACD.

**Linear Regression** (naive + enhanced) — predicts next-day return, derives direction from sign. Confidence via sigmoid mapping.

**LSTM** — recurrent neural network for sequential patterns. Requires pre-training via `train.py`. Three presets: `quick` (~1-5 min), `standard` (~5-15 min), `cluster` (hours on GPU). Early stopping prevents overfitting.

**Sentiment** — all models predict from price first, then VADER sentiment shifts the probability post-hoc.

## LSTM Training

```bash
uv run python train.py --ticker AAPL --period 1y --preset quick
uv run python train.py --stocks --preset standard
uv run python train.py --all --periods 1y 2y max --preset cluster
uv run python train.py --list                             # show saved models
```

Models saved to `models/{ticker}_{period}_{preset}.pt`. Auto-loaded in predictions (cluster > standard > quick priority).

## Backtesting

Walk-forward testing with simulated trading P/L, configurable fees, and buy-and-hold benchmark.

```bash
# Basic
uv run python backtest.py --tickers AAPL --days 20

# With trading fees (0.1% per side) and buy-and-hold comparison
uv run python backtest.py --stocks --days 20 --fees 0.1 --buy-hold

# Detailed output
uv run python backtest.py --full --period 1y --buy-hold

# Cross-period comparison + export
uv run python backtest.py --compare-periods --output results.csv --buy-hold

# Batch runner: one CSV per ticker
uv run python run_all.py --days 50 --fees 0.05 --buy-hold
```

### Trading fees

Configurable via `--fees` (percentage per side, default from `config.py`). Covers commission + spread + slippage. Applied twice per trade (buy + sell = round-trip). A model with +5% gross return and 50 trades at 0.1% per side loses 10% to fees → net -5%.

### Buy-and-hold benchmark

`--buy-hold` compares each model's active trading return against simply buying on day 1 and holding. If buy-and-hold beats most models, active trading doesn't add value for that ticker.

### Output modes

| Flag | Terminal | CSV rows |
|---|---|---|
| (none) | Summary per model | 1 per model |
| `--full` | Summary + consensus + profit + streaks | 1 per day per model |
| `--compare-periods` | Period × model matrix + top 5 | 1 per model × period |

### Batch runner (`run_all.py`)

Runs `--compare-periods` for each ticker separately, saves individual CSVs with descriptive filenames:

```
results/
├── AAPL_50d_fee005_bh.csv              # AAPL, 50 days, 0.05% fee, with B&H
├── BTC-USD_50d_fee005_bh.csv
├── ...
└── _summary_50d_fee005_bh_20260427.csv # best model per ticker
```

## Project Structure

```
marketpulse-ai/
├── config.py                # ★ Tickers, periods, fees, defaults
├── main.py                  # CLI — prediction reports
├── backtest.py              # CLI — model evaluation
├── train.py                 # CLI — LSTM training
├── run_all.py               # CLI — batch backtest (one CSV per ticker)
├── test_pipeline.py         # 13 offline tests
├── pyproject.toml           # Dependencies & build config
├── Containerfile            # Podman/Docker build
├── AGENTS.md                # AI assistant context file
│
├── interface/
│   ├── __init__.py
│   └── api.py               # StockAppAPI facade
│
├── engine/
│   ├── __init__.py
│   ├── features.py          # Shared feature engineering
│   ├── knn_model.py         # k-NN (naive + enhanced)
│   ├── lin_reg_model.py     # LinReg (naive + enhanced)
│   ├── ai_model.py          # LSTM (train, save/load, predict, early stopping)
│   ├── backtester.py        # Walk-forward engine (P/L, fees, B&H, streaks)
│   ├── backtest_helpers.py  # Shared helpers (display, export, period filtering)
│   ├── utils.py             # Common helpers shared across engine and interface
│   ├── data_downloader.py   # Yahoo Finance data
│   ├── db_manager.py        # SQLite storage
│   └── news_scraper.py      # VADER/naive sentiment
│
├── models/                  # Saved LSTM weights (gitignored)
├── results/                 # Backtest CSV outputs (gitignored)
├── data/                    # SQLite database (auto-created)
│
└── docs/                    # In-depth documentation
    ├── README.md            # Index
    ├── knn.md, linear-regression.md, lstm.md
    ├── features.md, sentiment.md
    ├── backtesting.md       # Methodology, fees, B&H, streaks, metrics
    └── api.md               # Architecture, DB schema, model contract
```

## Configuration

`config.py` is the single source of truth:

```python
STOCKS = ["AAPL", "MSFT", "NVDA", ...]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
ALL_TICKERS = STOCKS + CRYPTO
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]
DEFAULT_TRADING_FEE_PCT = 0.05  # 0.05% per side
```

## Documentation

`docs/` has in-depth explanations of every component. `AGENTS.md` is a compact context file for AI assistants — upload it when working on the codebase in any AI chat.

## Testing

```bash
uv run python test_pipeline.py
```

13 tests covering all models, features, sentiment, backtesting, fees, and error handling.

## Roadmap

- [x] k-NN model — naive + enhanced
- [x] Linear Regression — naive + enhanced
- [x] LSTM neural network (presets, early stopping, save/load)
- [x] Shared feature engineering (RSI, MACD, volume, volatility)
- [x] VADER sentiment + naive fallback
- [x] Walk-forward backtesting (P/L, profit factor, streaks)
- [x] Trading fees + buy-and-hold benchmark
- [x] Batch runner (`run_all.py`)
- [x] Centralized config, CLI filtering, CSV/JSON export
- [x] Documentation (`docs/` + `AGENTS.md`)
- [ ] FinBERT sentiment (finance-specific transformer)
- [ ] Visualization layer (Plotly/Matplotlib)
- [ ] Web UI (Flask/FastAPI)

## Tech Stack

Python 3.12 · pandas · yfinance · scikit-learn · NLTK (VADER) · PyTorch (LSTM) · NumPy · SQLite · uv
