# MarketPulse AI

[![Tests](https://github.com/Loghic/MarketPulse_AI/actions/workflows/tests.yml/badge.svg)](https://github.com/loghi-gh/mp_ai/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Loghic/MarketPulse_AI/graph/badge.svg)](https://codecov.io/gh/loghi-gh/mp_ai)

Stock prediction engine combining k-NN, Linear Regression, LSTM neural networks,
and time-series forecasting models (Prophet, Chronos-2, Kronos) with VADER/FinBERT
sentiment analysis. Built as a modular system with a clean separation between
data layer, model engine, and interface — ready to plug into a web or desktop UI.

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

For forecasting models:
```bash
uv pip install -e '.[forecast]'      # Prophet + Chronos-2
# Kronos (optional, OHLCV candlestick model — cloned, not pip-installed):
git clone https://github.com/shiyu-coder/Kronos.git ../Kronos
uv pip install -e '.[kronos]'
```

## Supported tickers

Configured in `config.py` as a data-driven asset registry (`ASSET_CLASSES`). To add a
ticker, add it to the relevant class's `tickers`; to add a whole asset class, append one
`AssetClass` entry — the CLI flags, benchmarks, asset-type tags, and GDELT news queries
all derive from it automatically.

**Stocks:** AAPL, MSFT, NVDA, META, GOOGL, AMD, TSM, ASML, AVGO, TSLA, INTC
**Crypto:** BTC-USD, ETH-USD, SOL-USD, BNB-USD
**Commodities:** GLD (gold ETF)
**Indices:** VOO (S&P 500), QQQM (Nasdaq-100)
**FX:** FXE (EUR/USD)

The index/commodity/FX classes use liquid **ETF proxies** so the volume-based features
and LSTM work exactly as for stocks — index/FX *spot* symbols (`^GSPC`, `EURUSD=X`) carry
no volume. `VOO`/`QQQM` are used (not `SPY`/`QQQ`) so the tradeable indices stay distinct
from the `SPY`/`QQQ` benchmark set.

All CLI scripts share the same scope selectors — the per-class flags **combine** (union),
and `--all` spans every class:

```bash
uv run python main.py --stocks
uv run python main.py --crypto --commodities          # classes combine (union)
uv run python main.py --indices --fx
uv run python main.py --all
uv run python main.py --tickers AAPL NVDA BNB-USD GLD
```

## Models

**k-NN** (naive + enhanced) — classifies next-day direction from return patterns. Enhanced adds volume, RSI, volatility, MACD.

**Linear Regression** (naive + enhanced) — predicts next-day return, derives direction from sign. Confidence via sigmoid mapping.

**LSTM** — recurrent neural network for sequential patterns. Requires pre-training via `train.py`. Three presets: `quick` (~1-5 min), `standard` (~5-15 min), `cluster` (hours on GPU). Early stopping prevents overfitting.

**Prophet** (forecasting) — Meta's additive trend/seasonality model. Fits per call (no pre-training, CPU-only). Direction derived from the forecast interval.

**Chronos-2** (forecasting) — Amazon's 120M-parameter zero-shot foundation model. No training; weights download on first use (~478 MB), then reused for every ticker. Direction derived from forecast quantiles.

**Kronos** (forecasting) — shiyu-coder's decoder-only foundation model for OHLCV candlesticks (MIT). Cloned as a sibling repo, not pip-installed; uses full open/high/low/close. Direction from sampled forecast paths. *Forecasting models are available in backtests and via the API; TiRex is parked. See [docs/forecasting.md](docs/forecasting.md).*

**Sentiment** — all models predict from price first, then VADER/FinBERT sentiment shifts the probability post-hoc.

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

## Web GUI

Browser-based dashboard with FastAPI backend and React frontend.

```bash
# Install dependencies
uv pip install -e ".[web]"
cd web/frontend && npm install && cd ../..

# Start both servers
chmod +x web/dev.sh
./web/dev.sh

# Or manually in two terminals:
# Terminal 1: uv run uvicorn web.backend.app:app --reload --port 8000
# Terminal 2: cd web/frontend && npm run dev
```

- **Frontend:** http://localhost:5173 (React dashboard)
- **Backend API:** http://localhost:8000 (FastAPI)
- **API docs:** http://localhost:8000/docs (auto-generated Swagger)

### Pages

| Tab | Status | Description |
|---|---|---|
| **Dashboard** | ✓ | Ticker selector, zoomable chart (line/candle, pan bar), stats cards, OHLCV table with Δ% sorting, custom period, export CSV |
| **Predict** | ✓ | Prediction builder (per-model period + news), quick presets, 9 model variants incl. LSTM, auto consensus, **per-ticker backend caching with timestamp + re-run**, historical predictions, optional price chart |
| **Backtest** | ✓ | Grouped ticker selector (All Stocks / All Crypto / individual), multi-period builder, global Fee/SL/B&H, results filtering, best-models with ties, optional price chart |
| **Training** | ✓ | LSTM model inventory with timestamps, active-model marker (preset priority cluster > standard > quick), one-click "Start training" with live status polling |
| **Analysis** | ✓ | Research tab: pick a `results/` directory, see best-models, news-vs-no-news win rates with leaderboards, **compare two runs side-by-side** (e.g. VADER vs FinBERT) |
| **Settings** | ✓ | Persistent k, fees, SL, LSTM preference with fallback, developer settings (collapsible) |

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/data/tickers` | List all tickers with metadata |
| GET | `/api/data/ticker/{ticker}` | OHLCV data (period filter, limit=0 for all) |
| POST | `/api/data/refresh` | Download latest prices + news |
| GET | `/api/predict/info` | Available models, periods, next trading day |
| POST | `/api/predict/run` | Unified prediction (per-model period + news) |
| GET | `/api/predict/cached` | List all cached prediction files (every ticker × date) |
| GET | `/api/predict/cached/{ticker}` | Latest cached prediction for one ticker + `cached_at` timestamp (powers the in-page cache badge) |
| POST | `/api/predict/historical` | Predict for any past date |
| POST | `/api/backtest` | Walk-forward backtest (multi-period via `periods: list[str]`, news/sentiment knobs) |
| GET | `/api/train/models` | List saved LSTM models with training timestamps |
| POST | `/api/train/start` | Start LSTM training (background) |
| GET | `/api/train/status/{key}` | Live training status (used for the Training tab spinner) |
| GET/PUT/PATCH | `/api/settings` | User settings (persistent JSON) |
| POST | `/api/analysis/news-comparison` | News vs No-News paired comparison |
| GET | `/api/analysis/results-dirs` | Enumerate `results/{scope}_…/` subdirectories with metadata |
| GET | `/api/analysis/result-csv?dir=…&file=…` | Return one CSV from a results subdir as JSON rows |

See [docs/web.md](docs/web.md) for full API documentation with request/response examples.

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

# Restrict periods / model families + per-model timing breakdown
uv run python backtest.py --tickers NVDA --compare-periods --periods 1y 2y --models knn linreg lstm chronos --timing
```

### Choosing models, periods, and timing

- `--models F [...]` — run only these families: `knn`, `linreg`, `lstm`, `prophet`, `chronos`, `kronos` (default: all). Same flag on `run_all.py`. Handy for dropping the slow forecasting models.
- `--periods P [...]` — restrict the period set in `--compare-periods` (and on `run_all.py`); skip the slow `max` window.
- `--timing` — print a slowest-first per-model compute-time table after the summary. `run_all.py` prints a time-by-model-family rollup automatically.

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

# Skip slow 'max', drop the heavy forecasting models, see a time-by-family rollup
uv run python run_all.py --stocks --days 100 --periods 1y 2y 5y --models knn linreg lstm chronos
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
├── .github/
│   └── workflows/
│       └── tests.yml            # CI: lint (ruff) + typecheck (mypy) + test (pytest+coverage)
├── .codecov.yml                 # Coverage thresholds and Codecov config
├── .pre-commit-config.yaml      # Git hooks: ruff + mypy before every commit
├── config.py                # ★ Asset registry (tickers/classes/benchmarks/news names), periods, fees, stop-loss, logging
├── cli_helpers.py           # Shared CLI scope flags + resolver (--stocks/--crypto/--commodities/--indices/--fx/--all/--tickers)
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
├── web/                         # Web GUI
│   ├── dev.sh                   # Start both servers
│   ├── backend/
│   │   ├── app.py               # FastAPI main (CORS, Swagger at /docs)
│   │   ├── schemas.py           # Pydantic request/response models
│   │   └── routes/
│   │       ├── data.py          # Tickers, OHLCV, refresh
│   │       ├── predict.py       # Unified prediction builder + caching + consensus
│   │       ├── backtest.py      # Walk-forward backtesting
│   │       ├── train.py         # LSTM training + model inventory
│   │       ├── settings.py      # Persistent user settings (JSON)
│   │       └── analysis.py      # News vs No-News comparison
│   └── frontend/
│       ├── package.json         # React 19 + Vite + TypeScript + Plotly
│       ├── vite.config.ts       # Dev proxy /api → localhost:8000
│       └── src/
│           ├── main.tsx         # Entry + router + layout
│           ├── lib/api.ts       # Typed API client
│           └── pages/           # Dashboard, Predict, Backtest, Training, Analysis, Settings
│
├── tests/                   # Comprehensive pytest suite (103 tests)
│   ├── conftest.py          # Shared fixtures (mock data, patched yfinance)
│   ├── test_features.py     # Feature matrix shape, NaN, edge cases
│   ├── test_models.py       # k-NN, LinReg, LSTM predict + errors
│   ├── test_backtester.py   # P/L, fees, stop-loss, DD, Sharpe, streaks, yearly
│   ├── test_api.py          # API facade, benchmarks, CSV export, sentiment
│   ├── test_logger.py       # Logger modes, progress bar, config sanity
│   └── test_web_api.py      # FastAPI endpoints (26 tests: data, predict, backtest, settings)
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
│   ├── forecast_base.py     # Shared base for forecasting models (ForecastResult + ForecastModel)
│   ├── prophet_model.py     # Prophet (fits per call, CPU)
│   ├── chronos_model.py     # Chronos-2 (zero-shot foundation model, loads once)
│   ├── kronos_model.py      # Kronos (OHLCV candlestick foundation model, sibling clone)
│   ├── backtester.py        # Walk-forward engine (P/L, fees, SL, DD, Sharpe, B&H, streaks, elapsed_seconds)
│   ├── backtest_helpers.py  # Shared helpers (display, export, benchmarks, model variants, timing)
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
    ├── run.md               # The runbook (install, every CLI flag, workflows, troubleshooting)
    ├── knn.md, linear-regression.md, lstm.md
    ├── features.md, sentiment.md
    ├── forecasting.md       # Prophet, Chronos-2, Kronos + the ForecastModel interface
    ├── backtesting.md       # Methodology, fees, stop-loss, B&H, streaks, timing
    └── api.md               # Architecture, DB schema, model contract
```

## Documentation

`docs/` has in-depth explanations of every component. `AGENTS.md` is a compact context file for AI assistants — upload it when working on the codebase in any AI chat.

## Testing

Two test suites — quick smoke test and comprehensive pytest:

```bash
# Quick smoke test (no extra dependencies, 13 tests)
uv run python test_pipeline.py

# Full pytest suite (103 tests, needs pytest)
uv run python -m pytest

# Run specific test file or class
uv run python -m pytest tests/test_backtester.py -v
uv run python -m pytest tests/test_backtester.py::TestFees -v
```

Test coverage: models (k-NN, LinReg, LSTM), feature engineering, backtester (P/L math, fees, stop-loss, risk metrics, streaks, yearly), benchmarks (SPY/QQQ/BTC), CSV export, sentiment, logger, config.

## CI / CD

Every push and PR to `main` triggers three parallel jobs via GitHub Actions:

| Job | Tool | What it checks | Blocking? |
|---|---|---|---|
| **lint** | Ruff | Unused imports, import order, deprecated syntax, common bugs, formatting | Yes |
| **typecheck** | Mypy | Type annotations, None safety, wrong argument types | Yes |
| **test** | Pytest | 103 tests + coverage upload to Codecov (Python 3.12 + 3.13 matrix) | Yes |

### Pre-commit hooks

Git hooks that run **before every commit** — catches issues locally before they reach CI:

```bash
# One-time setup
uv pip install -e ".[dev]"
uv run pre-commit install

# Now every git commit auto-runs:
#   1. ruff --fix     (auto-fixes imports, unused vars)
#   2. ruff format    (auto-formats code)
#   3. mypy           (type checking)
```

If ruff modifies files, the commit stops — just `git add -A` and commit again. If mypy fails, you need to fix the type error manually.

To skip hooks for emergency fixes: `git commit --no-verify -m "hotfix"`

### Static analysis locally

```bash
# Lint (must pass before push)
uv run ruff check .
uv run ruff format --check .

# Auto-fix lint issues
uv run ruff check --fix .
uv run ruff format .

# Type check
uv run mypy engine/ interface/
```

### Coverage

Coverage is uploaded to [Codecov](https://codecov.io) after each test run. Current coverage is shown in the badge at the top of this README. Core engine modules are at 90%+, overall ~59% (LSTM module pulls it down since PyTorch isn't in CI).

### Adding new code

Pre-commit hooks catch most issues automatically. For what they can't auto-fix:

- **Ruff**: imports must be sorted, no unused imports, use `list`/`dict` instead of `typing.List`/`typing.Dict`
- **Mypy**: add `if X is None` guards before using Optional values. Strict modules (`engine/backtester.py`, `engine/utils.py`) require full type annotations on all functions.
- **Tests**: add tests in `tests/` for new features. Run `uv run python -m pytest` before pushing.

## Roadmap

See [plan.md](plan.md) for the research roadmap and backlog.

## Tech Stack

**Engine:** Python 3.12 · pandas · yfinance · scikit-learn · NLTK (VADER) · transformers + PyTorch (FinBERT + LSTM, optional `ai` extra) · NumPy · tqdm · SQLite · Prophet · Chronos-2 (chronos-forecasting) · Kronos (sibling clone)

**News sources:** Yahoo Finance · GDELT 2.0 Doc API (free, no key, multi-year history)

**Web:** FastAPI · uvicorn · React 19 · TypeScript · Vite · TanStack Query · Plotly.js

**Dev:** pytest · ruff · mypy · pre-commit · GitHub Actions · Codecov · uv
