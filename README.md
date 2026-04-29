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
uv run python train.py --list
```

Models saved to `models/{ticker}_{period}_{preset}.pt`. Auto-loaded in predictions (cluster > standard > quick priority).

## Data Refresh

All scripts download fresh data automatically before running. To skip downloads (offline mode), use `--no-refresh`.

```bash
# Standalone refresh (download only, no models)
uv run python refresh.py
uv run python refresh.py --stocks

# Predictions download data automatically
uv run python main.py --stocks

# Offline mode (use cached data from DB)
uv run python main.py --stocks --no-refresh
uv run python backtest.py --stocks --days 50 --no-refresh
uv run python run_all.py --stocks --days 20 --no-refresh
```

### Daily workflow

```bash
# 1. Morning: predictions (auto-refreshes data)
uv run python main.py --stocks

# 2. Or: refresh first, then run multiple analyses offline
uv run python refresh.py
uv run python main.py --stocks --no-refresh
uv run python backtest.py --stocks --days 20 --fees 0.03 --no-refresh
uv run python run_all.py --stocks --days 50 --fees 0.03 --buy-hold --no-refresh
```

## Backtesting

Walk-forward testing with simulated trading P/L, configurable fees, stop-loss, buy-and-hold benchmark, and risk metrics (max drawdown, Sharpe ratio, Sortino ratio, yearly rolling performance).

```bash
# Basic
uv run python backtest.py --tickers AAPL --days 20

# With fees and buy-and-hold
uv run python backtest.py --stocks --days 20 --fees 0.03 --buy-hold

# With stop-loss (runs each model twice: with and without SL for comparison)
uv run python backtest.py --tickers AAPL --days 20 --fees 0.03 --stop-loss 2

# Offline (skip data download)
uv run python backtest.py --stocks --days 50 --no-refresh

# Full details
uv run python backtest.py --full --period 1y --buy-hold --stop-loss 2

# Cross-period comparison + export
uv run python backtest.py --compare-periods --output results.csv --buy-hold
```

### Stop-loss

`--stop-loss 2` means: if the position drops 2% intraday, exit immediately at the stop-loss price instead of holding until close. Uses actual High/Low data to check if the stop would have triggered.

When enabled, every model runs twice — once without SL (baseline) and once with SL — so you can directly compare:

```
k-NN Enhanced              +3.26%  (PF 1.38)
k-NN Enhanced SL2%         +6.35%  (PF 2.15)  ← SL cut 2 big losses
```

### Trading fees

`--fees 0.03` means 0.03% per side (buy + sell = 0.06% round-trip). Covers commission + spread + slippage. Default from `config.py`.

### Batch runner (`run_all.py`)

Runs `--compare-periods` for each ticker, saves organized results:

```bash
uv run python run_all.py --stocks --days 50 --fees 0.03 --buy-hold
uv run python run_all.py --crypto --days 50 --fees 0.15 --stop-loss 3
uv run python run_all.py --all --days 20
```

Output is organized into subdirectories:

```
results/
├── stocks_50d_fee003_bh/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── ...
│   └── _summary.csv
├── crypto_50d_fee015_sl3/
│   ├── BTC-USD.csv
│   └── _summary.csv
└── all_20d/
    ├── AAPL.csv
    └── _summary.csv
```

Directory name encodes run parameters (`scope_days_fees_sl_bh`). Different runs don't overwrite each other.

## Project Structure

```
marketpulse-ai/
├── config.py                # ★ Tickers, periods, fees, stop-loss, benchmarks, logging mode
├── main.py                  # CLI — prediction reports
├── backtest.py              # CLI — model evaluation
├── train.py                 # CLI — LSTM training
├── run_all.py               # CLI — batch backtest (organized subdirectories)
├── refresh.py               # CLI — download latest prices + news (no models)
├── test_pipeline.py         # Quick smoke test (13 tests, no extra deps)
├── pyproject.toml           # Dependencies & build config
├── Containerfile            # Podman/Docker build
├── AGENTS.md                # AI assistant context file
│
├── tests/                   # Comprehensive pytest suite (77 tests)
│   ├── conftest.py          # Shared fixtures (mock data, patched yfinance)
│   ├── test_features.py     # Feature matrix shape, NaN, edge cases
│   ├── test_models.py       # k-NN, LinReg, LSTM predict + errors
│   ├── test_backtester.py   # P/L, fees, stop-loss, DD, Sharpe, streaks, yearly
│   ├── test_api.py          # API facade, benchmarks, CSV export, sentiment
│   └── test_logger.py       # Logger modes, progress bar, config sanity
│
├── interface/
│   ├── __init__.py
│   └── api.py               # StockAppAPI facade (refresh, predict, data)
│
├── engine/
│   ├── __init__.py
│   ├── logger.py            # Centralized logging + progress bars (cli/gui modes)
│   ├── features.py          # Shared feature engineering
│   ├── knn_model.py         # k-NN (naive + enhanced)
│   ├── lin_reg_model.py     # LinReg (naive + enhanced)
│   ├── ai_model.py          # LSTM (train, save/load, predict, early stopping)
│   ├── backtester.py        # Walk-forward engine (P/L, fees, SL, DD, Sharpe, B&H, streaks)
│   ├── backtest_helpers.py  # Shared helpers (display, export, benchmarks, model variants)
│   ├── utils.py             # Common helpers shared across layers
│   ├── data_downloader.py   # Yahoo Finance data
│   ├── db_manager.py        # SQLite storage
│   └── news_scraper.py      # VADER/naive sentiment
│
├── models/                  # Saved LSTM weights (gitignored)
├── results/                 # Backtest CSV outputs (organized subdirectories)
├── data/                    # SQLite database (auto-created)
│
└── docs/                    # In-depth documentation
    ├── README.md            # Index
    ├── knn.md, linear-regression.md, lstm.md
    ├── features.md, sentiment.md
    ├── backtesting.md       # Methodology, fees, stop-loss, B&H, streaks
    └── api.md               # Architecture, DB schema, model contract
```

## Documentation

`docs/` has in-depth explanations of every component. `AGENTS.md` is a compact context file for AI assistants — upload it when working on the codebase in any AI chat.

## Testing

Two test suites — quick smoke test and comprehensive pytest:

```bash
# Quick smoke test (no extra dependencies, 13 tests)
uv run python test_pipeline.py

# Full pytest suite (77 tests, needs pytest)
uv run python -m pytest tests/ -v

# Run specific test file
uv run python -m pytest tests/test_backtester.py -v

# Run specific test class
uv run python -m pytest tests/test_backtester.py::TestFees -v
```

Test coverage: models (k-NN, LinReg, LSTM), feature engineering, backtester (P/L math, fees, stop-loss, risk metrics, streaks, yearly), benchmarks (SPY/QQQ/BTC), CSV export, sentiment, logger, config.

## Roadmap

- [x] k-NN model — naive + enhanced
- [x] Linear Regression — naive + enhanced
- [x] LSTM neural network (presets, early stopping, save/load)
- [x] Shared feature engineering (RSI, MACD, volume, volatility)
- [x] VADER sentiment + naive fallback
- [x] Walk-forward backtesting (P/L, profit factor, streaks)
- [x] Trading fees + stop-loss + buy-and-hold benchmark
- [x] Risk metrics (max drawdown, Sharpe, Sortino, yearly rolling performance)
- [x] Batch runner with organized output (`run_all.py`)
- [x] Centralized logging (cli/gui modes) + progress bars (tqdm)
- [x] Centralized config, CLI filtering, CSV/JSON export
- [x] Documentation (`docs/` + `AGENTS.md`)
- [x] Pytest suite (77 tests: models, backtester, benchmarks, export, logger)
- [ ] FinBERT sentiment (finance-specific transformer)
- [ ] Visualization layer (Plotly/Matplotlib)
- [ ] Web UI (Flask/FastAPI)

## Tech Stack

Python 3.12 · pandas · yfinance · scikit-learn · NLTK (VADER) · PyTorch (LSTM) · NumPy · tqdm · SQLite · uv
