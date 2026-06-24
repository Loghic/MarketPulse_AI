# MarketPulse AI

[![Tests](https://github.com/Loghic/MarketPulse_AI/actions/workflows/tests.yml/badge.svg)](https://github.com/loghi-gh/mp_ai/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/Loghic/MarketPulse_AI/graph/badge.svg)](https://codecov.io/gh/loghi-gh/mp_ai)

Stock/crypto/commodity/index/FX prediction engine combining LSTM neural networks
and time-series forecasting models (Prophet, Chronos-2, Kronos) — plus simple
k-NN / LinReg references and a suite of naive + news-aware baselines — with
VADER/FinBERT sentiment. Walk-forward backtesting with realistic fees (turnover /
position-based), stop-loss sweeps, a confidence gate, and an **out-of-sample
harness** + significance tests so results are honest, not selection-inflated.
A separate **point-forecast track** scores predicted price *levels* against a
random-walk baseline (Theil U2 / MASE) with ARIMA / XGBoost / naive forecasters.
Modular: clean separation of data layer, model engine, CLI, and a React web UI.

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

Models fall in three tiers (the web UI groups them this way):

**Forecast / main models** — the ones to actually study:

**LSTM** — recurrent neural network for sequential patterns. Requires pre-training via `train.py`. Three presets: `quick` (~1-5 min), `standard` (~5-15 min), `cluster` (hours on GPU). Early stopping prevents overfitting.

**Prophet** (forecasting) — Meta's additive trend/seasonality model. Fits per call (no pre-training, CPU-only). Direction derived from the forecast interval.

**Chronos-2** (forecasting) — Amazon's 120M-parameter zero-shot foundation model. No training; weights download on first use (~478 MB), then reused for every ticker. Direction derived from forecast quantiles.

**Kronos** (forecasting) — shiyu-coder's decoder-only foundation model for OHLCV candlesticks (MIT). Cloned as a sibling repo, not pip-installed; uses full open/high/low/close. Direction from sampled forecast paths. *Forecasting models are available in backtests and via the API; TiRex is parked. See [docs/forecasting.md](docs/forecasting.md).*

**Educational / simple models** — kept as illustrative references, not the focus:

**k-NN** (naive + enhanced) — classifies next-day direction from return patterns. Enhanced adds volume, RSI, volatility, MACD.

**Linear Regression** (naive + enhanced) — predicts next-day return, derives direction from sign. Confidence via sigmoid mapping.

**Baselines** — trivial floors every real model must clear (not just buy-and-hold). Price-only: **Always-Long**, **Always-Short**, **Previous-Day**, **5/20-Day Momentum**, **Random**. News-aware (stateless — they *react* to sentiment but never *learn*): **News Previous-Day**, **News-Informed** (trades only on clear news, else sits out), **News Momentum**. A model is only interesting if it beats Previous-Day and Always-Long. See [docs/run/research.md](docs/run/research.md).

**Sentiment** — models predict from price first, then VADER/FinBERT sentiment shifts the probability post-hoc; the news-aware baselines use the same look-ahead-safe per-day score.

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
| **Dashboard** | ✓ | Ticker selector grouped by every asset class, zoomable chart (line/candle, pan bar), stats cards, OHLCV table with Δ% sorting, custom period, export CSV, news refresh |
| **Predict** | ✓ | Prediction builder (per-model period + news); model variants come from the backend (gated by availability — Prophet/Chronos/Kronos appear when installed); auto consensus, per-ticker caching, historical predictions, optional chart |
| **Backtest** | ✓ | Tickers grouped by all asset classes, model **family** picker (main vs simple tiers) + baselines toggle, Fee / SL / **SL-sweep** / **min-confidence** / **turnover** / **hold-days** / **position-mode** knobs, live progress, persisted-run picker, Coverage/Turnover columns |
| **OOS** | ✓ | Out-of-sample harness: select-on-one-window → evaluate-on-disjoint-window; aggregate (beat-B&H rate, selection-inflation gap, calibration) + per-ticker table, live progress, persisted runs |
| **OOS Compare** | ✓ | Diff two saved OOS runs side-by-side (aggregate + per-ticker), e.g. gate on vs off |
| **Training** | ✓ | LSTM model inventory with timestamps, active-model marker (preset priority cluster > standard > quick), one-click "Start training" with live status polling |
| **Analysis** | ✓ | Research tab: pick a `results/` directory, see best-models, news-vs-no-news win rates with leaderboards, **compare two runs side-by-side** (e.g. VADER vs FinBERT) |
| **Settings** | ✓ | Persistent k, fees, SL, LSTM preference with fallback, developer settings (collapsible) |
| **Help** | ✓ | Searchable in-app glossary — what every model, knob, and metric means (served from `web/docs/`); the Backtest/OOS tabs deep-link into it |

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/meta` | Config-driven options: model families (gated + tiers), asset classes, benchmarks, periods, SL/confidence sweeps, defaults — the frontend's single source of truth |
| GET | `/api/data/tickers` | List all tickers with metadata (registry asset class) |
| GET | `/api/data/ticker/{ticker}` | OHLCV data (period filter, limit=0 for all) |
| POST | `/api/data/refresh` | Download latest prices + news |
| GET | `/api/predict/info` | Available models (availability-gated), periods, next trading day |
| POST | `/api/predict/run` | Unified prediction (per-model period + news) |
| GET | `/api/predict/cached` | List all cached prediction files (every ticker × date) |
| GET | `/api/predict/cached/{ticker}` | Latest cached prediction for one ticker + `cached_at` timestamp (powers the in-page cache badge) |
| POST | `/api/predict/historical` | Predict for any past date |
| POST | `/api/backtest` | Walk-forward backtest (multi-period, news + model-filter + gate/turnover/hold/position/SL-sweep knobs); `/progress` + `/runs` for live progress and persisted-run reload |
| POST | `/api/oos` | Out-of-sample harness run (+ `/progress`, `/runs`, `/runs/{id}`) |
| GET | `/api/docs` · `/api/docs/{slug}` | End-user concept glossary served to the Help tab |
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

- `--models F [...]` — run only these families: `knn`, `linreg`, `lstm`, `prophet`, `chronos`, `kronos`, `baseline` (default: all). Same flag on `run_all.py`. Handy for dropping the slow forecasting models. `--no-baselines` drops the naive baselines.
- `--periods P [...]` — restrict the period set in `--compare-periods` (and on `run_all.py`); skip the slow `max` window.
- `--timing` — print a slowest-first per-model compute-time table after the summary. `run_all.py` prints a time-by-model-family rollup automatically.

### Measurement rigor

- `--min-confidence θ` — confidence gate: sit out days the model is less than θ sure about (excluded from accuracy; coverage reported). `--confidence-sweep` shows coverage/traded-accuracy/return + Brier/ECE across thresholds.
- `--significance` — binomial p + Wilson CI on accuracy, bootstrap CI on return, with Benjamini-Hochberg FDR — is the result distinguishable from a coin flip?
- For the **honest** read, use the out-of-sample harness rather than `--compare-periods`' best-of: `scripts/oos_harness.py` picks a winner on one window and scores it on a disjoint one. See [docs/run/research.md](docs/run/research.md).

### Point-forecast track (predicting values, not direction)

A separate evaluation path scores predicted **price levels** instead of UP/DOWN trades — no fees, positions, or P&L. The headline is **scale-free skill vs a random walk** (Theil U2 < 1 ⇔ beats RW; MASE), because on a price level absolute RMSE/MAPE flatter every model.

```bash
# Rank every available forecaster by skill vs a random walk
uv run python scripts/forecast_harness.py --stocks --days 100 --horizon 1 --no-refresh
```

Forecasters: **Random Walk** (the reference), **RW + Drift**, **Seasonal Naive**, plus **ARIMA** (`statsmodels`) and **XGBoost** (`xgboost`) from the `[forecast]` extra, an **LSTM regressor** (per-ticker, `torch`), and the existing Prophet/Chronos/Kronos point forecasts. Output lands in `results/fc_<scope>_<days>d_h<h>_<ts>/`.

The LSTM regressor (a separate Δ-predicting network — *not* the directional classifier) needs pre-trained per-ticker weights; train them leakage-safely first (same `--days`/`--horizon` you score with):

```bash
uv run python scripts/train_lstm_regressor.py --stocks --days 100 --horizon 1
```

A **residual hybrid** (`P̂ = base + learned residual`, e.g. Prophet + an LSTM residual learner) composes any base forecaster with any residual learner — the paper's central artifact; if the base's residuals are white noise the hybrid cleanly reduces to the base. It's opt-in (`--hybrid`, the slowest model) with a `--hybrid-fit` cadence (`pretrained` frozen weights / `refit_k` / `per_step`); pretrain the residual learner leakage-safely with `scripts/train_hybrid_residual.py`. **Diebold–Mariano** and **Wilcoxon** forecast-comparison tests (`engine/forecast_significance.py`, FDR-corrected) answer "is the difference vs the random walk statistically real?". See [docs/forecasting-regression.md](docs/forecasting-regression.md).

### Stop-loss (single or sweep)

`--stop-loss 2` means: if the position drops 2% intraday, exit immediately at the stop-loss price instead of holding until close (uses real High/Low). A single value runs each model twice (no-SL baseline + SL). Pass several (`--stop-loss 0 5 10 15`) or `--sl-sweep` to compare levels side by side.

### Fees & turnover realism

- `--fees 0.03` — 0.03% per side (buy + sell = 0.06% round-trip). Default from `config.py`.
- `--turnover-fees` — charge the round-trip fee only on days the position *changes*, not every day (the realistic "trade on signal changes" cost).
- `--hold-days N` — hold an opened position N days before re-reading the signal.
- `--position-mode` — hold one position across same-direction days and book its *compounded* entry→exit return as a single trade, paying one round-trip fee per held run (vs the default daily mark-to-market).

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
│   ├── naive_forecasters.py # Regression baselines: Random Walk / RW+Drift / Seasonal Naive (U2 reference)
│   ├── arima_model.py       # ARIMA point-forecaster (optional statsmodels)
│   ├── xgboost_model.py     # XGBoost point-forecaster on the feature matrix (optional xgboost)
│   ├── lstm_regressor.py    # LSTM point-forecaster (Δ-target, per-ticker weights; optional torch)
│   ├── residual_learners.py # Residual learners (Zero, per-call LSTM) for the hybrid
│   ├── residual_hybrid.py   # ResidualHybrid(base, learner): P̂ = base + learned residual
│   ├── forecast_significance.py # Diebold–Mariano + Wilcoxon (+ FDR) for forecasts
│   ├── regression_metrics.py# Point-forecast metrics: RMSE/MAE/MAPE/sMAPE + MASE/RMSSE/Theil U2
│   ├── forecast_backtester.py# Walk-forward point-forecast harness (no trading; leakage-guarded)
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
    ├── forecasting-regression.md # Point-forecast track: U2/MASE, RW/ARIMA/XGBoost, forecast harness
    ├── backtesting.md       # Methodology, fees, stop-loss, B&H, streaks, timing
    └── api.md               # Architecture, DB schema, model contract
```

## Documentation

`docs/` has in-depth explanations of every component (start at [docs/README.md](docs/README.md)); the runbook with every CLI flag + recipes is in [docs/run/](docs/run/) and the full web API in [docs/web.md](docs/web.md). End users get plain-language concept docs (what stop-loss / OOS / the baselines / each metric mean) in the app's **Help** tab, sourced from [web/docs/](web/docs/). `AGENTS.md` is a compact context file for AI assistants — upload it when working on the codebase in any AI chat.

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

Test coverage: models (k-NN, LinReg, LSTM), feature engineering, backtester (P/L, fees, stop-loss + sweep, turnover, hold-days, position mode, FLAT no-trade, risk metrics, streaks, yearly), baselines (naive + news-aware), confidence calibration + gating, statistical significance, the OOS harness, the point-forecast track (regression metrics + MASE/U2 invariants, naive forecasters, walk-forward forecast harness with leakage guarantee), shared CLI arg groups, news pipeline, web API (data / meta / predict / backtest / oos / docs / settings / analysis), CSV export, logger, config.

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

**Engine:** Python 3.12 · pandas · yfinance · scikit-learn · NLTK (VADER) · transformers + PyTorch (FinBERT + LSTM, optional `ai` extra) · NumPy · tqdm · SQLite · Prophet · Chronos-2 (chronos-forecasting) · Kronos (sibling clone) · statsmodels (ARIMA) · XGBoost · SciPy — forecasting stats (all optional `forecast` extra)

**News sources:** Yahoo Finance · GDELT 2.0 Doc API (free, no key, multi-year history)

**Web:** FastAPI · uvicorn · React 19 · TypeScript · Vite · TanStack Query · Plotly.js

**Dev:** pytest · ruff · mypy · pre-commit · GitHub Actions · Codecov · uv
