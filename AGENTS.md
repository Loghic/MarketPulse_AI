# AGENTS.md — AI Context for MarketPulse AI

Read this before making changes. Compact reference for AI assistants.

## What this does

Predicts next-day stock/crypto price direction (UP/DOWN) using k-NN, LinReg, and LSTM. Sentiment adjustment via VADER. Walk-forward backtesting with simulated P/L, fees, stop-loss, and buy-and-hold benchmark.

## Project layout

```
.github/workflows/tests.yml    → CI: lint (ruff) + typecheck (mypy) + test (pytest+codecov)
.codecov.yml                    → Coverage thresholds (60% target)
.pre-commit-config.yaml         → Git hooks: ruff auto-fix + format + mypy before every commit

config.py                   → ★ Tickers, periods, fees, stop-loss defaults. Edit to add assets.
main.py                     → CLI: prediction reports (--stocks/--crypto/--all/--tickers)
backtest.py                 → CLI: model evaluation (--full/--compare-periods/--output/--stop-loss)
train.py                    → CLI: LSTM training (--preset quick/standard/cluster)
run_all.py                  → CLI: batch runner (organized subdirectories in results/)
refresh.py                  → CLI: download latest prices + news (no models, just data)
test_pipeline.py            → 13 offline smoke tests (no pytest needed)

tests/                      → Comprehensive pytest suite (103 tests)
  conftest.py               → Fixtures: mock prices, news, patched yfinance, api
  test_features.py          → Feature matrix shape, NaN, edge cases
  test_models.py            → k-NN, LinReg, LSTM predict + errors
  test_backtester.py        → P/L math, fees, SL, DD, Sharpe, Sortino, streaks, yearly
  test_api.py               → API facade, benchmarks, CSV export, sentiment
  test_logger.py            → Logger modes, progress bar, config sanity
  test_web_api.py           → FastAPI endpoints: data, predict, backtest, train, settings, analysis

web/                        → Web GUI (FastAPI + React)
  dev.sh                    → Start both servers (backend + frontend)
  backend/
    app.py                  → FastAPI main (CORS, logging, Swagger at /docs)
    schemas.py              → Pydantic request/response models
    routes/
      data.py               → GET tickers, GET OHLCV (limit=0=all), POST refresh
      predict.py            → POST /run (unified per-model config), /cached, /historical
      backtest.py           → POST backtest (delegates to engine)
      train.py              → GET models inventory, POST start training
      settings.py           → GET/PUT/PATCH persistent user settings
      analysis.py           → POST news-comparison (for academic paper)
  frontend/
    package.json            → React 19 + Vite + TS + TanStack Query
    vite.config.ts          → Dev proxy /api → backend:8000
    src/
      main.tsx              → Entry + router + layout (6 tabs)
      app.css               → Dark theme, no spinbox arrows
      lib/api.ts            → Typed fetch client for all endpoints
      components/
        ui.tsx              → Panel, Btn, LoadingBox, pct(), usd()
        PriceChart.tsx      → Reusable zoomable SVG chart (line/candle, pan bar)
        DataTable.tsx       → Reusable sortable/filterable/paginated table
      pages/
        Dashboard.tsx       → ★ Complete: chart, stats, OHLCV table
        Predict.tsx         → ★ Complete: builder, consensus, caching, historical
        Settings.tsx        → ★ Complete: persistent prefs, dev section
        Backtest.tsx        → Stub (backend ready)
        Training.tsx        → Stub (backend ready)
        Analysis.tsx        → Stub (backend ready)

interface/
  api.py                    → StockAppAPI facade — single entry point

engine/
  features.py               → Shared indicators (RSI, MACD, volatility, volume)
  logger.py                 → Centralized logging + progress bars (tqdm/fallback)
  knn_model.py              → k-NN (naive + enhanced)
  lin_reg_model.py          → LinReg (naive + enhanced)
  ai_model.py               → LSTM (train/save/load/predict + early stopping)
  backtester.py             → Walk-forward engine (P/L, fees, SL, DD, Sharpe, Sortino, B&H, streaks, yearly)
  backtest_helpers.py       → Shared: period filter, display, export, model variant runner
  utils.py                  → Common helpers (period_to_start_date) shared across layers
  data_downloader.py        → yfinance fetching
  db_manager.py             → SQLite (prices + news)
  news_scraper.py           → VADER + naive fallback

models/                     → Saved LSTM weights ({ticker}_{period}_{preset}.pt)
results/                    → Organized: {scope}_{days}d[_fee{N}][_sl{N}][_bh]/{TICKER}.csv
data/                       → SQLite DB (auto-created)
docs/                       → In-depth docs for humans
```

## Config

```python
# Tickers
STOCKS = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMD", "TSM", "ASML", "AVGO", "TSLA", "INTC"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
ALL_TICKERS = STOCKS + CRYPTO
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]

# Logging: "cli" = INFO + progress bars, "gui" = WARNING only
LOG_MODE = "cli"
LOG_LEVEL = None  # override: "DEBUG", "INFO", "WARNING", "ERROR"

# Benchmarks: stocks vs SPY+QQQ, crypto vs BTC-USD (excluding self)
STOCK_BENCHMARKS = ["SPY", "QQQ"]
CRYPTO_BENCHMARKS = ["BTC-USD"]

# Trading
DEFAULT_TRADING_FEE_PCT = 0.05   # per side, round-trip = 2x
DEFAULT_STOP_LOSS_PCT = 0.0      # 0 = disabled
```

## Model variants

| model_type | Class | Features | Pre-training? |
|---|---|---|---|
| `"knn"` | KNNModel | returns only | No |
| `"knn_enhanced"` | KNNModel | all (RSI, MACD, vol, volat) | No |
| `"linreg"` | LinearRegressionModel | returns only | No |
| `"linreg_enhanced"` | LinearRegressionModel | all | No |
| `"lstm"` | AIModel | all (sequential) | Yes → `train.py` |

Shared interface: `model.predict(df, use_time_weights, sentiment_score) → (str, float)`

## Stop-loss behavior

When `--stop-loss X` is passed, **each model runs twice**: once without SL (baseline) and once with SL. The SL variant gets a name suffix like `k-NN SL2%`. This is implemented in `backtest_helpers.run_single_backtest()` — it creates a second Backtester with `stop_loss_pct=0` for the baseline runs.

Stop-loss uses intraday High/Low: Long exits if Low ≤ entry × (1 - SL%), Short exits if High ≥ entry × (1 + SL%).

Without `--stop-loss`, no duplication occurs.

## Key types

```python
@dataclass
class DayResult:
    date, predicted, actual, confidence, correct,
    close_before, close_actual, exit_price,     # exit_price = SL price or close
    trade_pnl, trade_pnl_net, stopped_out       # stopped_out: bool

@dataclass
class BacktestResult:
    model_name, ticker, test_days, correct, accuracy,
    fee_pct, stop_loss_pct, stopped_out_count,
    total_return, profit_factor, gross_profit, gross_loss,
    avg_win, avg_loss, best_day, worst_day, win_trades, loss_trades,
    max_drawdown, sharpe_ratio, sortino_ratio,
    longest_win_streak, longest_loss_streak, avg_win_streak, avg_loss_streak,
    buy_hold_return, buy_hold_max_drawdown,
    yearly_performance: List[YearlyPerformance],
    days: List[DayResult]
```

## CLI

```bash
# Predictions
uv run python main.py --stocks
uv run python main.py --tickers NVDA TSLA

# Training
uv run python train.py --all --periods 1y 2y max --preset standard
uv run python train.py --list

# Backtest
uv run python backtest.py --stocks --days 20 --fees 0.03 --buy-hold
uv run python backtest.py --tickers AAPL --days 20 --stop-loss 2 --fees 0.03
uv run python backtest.py --compare-periods --output results.csv

# Batch runner
uv run python run_all.py --stocks --days 50 --fees 0.03 --stop-loss 2 --buy-hold
# → results/stocks_50d_fee003_sl2_bh/{AAPL,MSFT,...,_summary}.csv

# Data refresh (no models, just download)
uv run python refresh.py
uv run python refresh.py --stocks

# All scripts auto-refresh by default. Skip with --no-refresh:
uv run python main.py --stocks --no-refresh
uv run python backtest.py --stocks --days 50 --no-refresh
uv run python run_all.py --stocks --days 20 --no-refresh
```

## Common tasks

**Add a ticker:** Edit `config.py` → `STOCKS` or `CRYPTO`. Done.

**Add a model:** Create `engine/new_model.py` with `.predict()` → register in `api._get_model()` → add to `backtest_helpers.run_single_backtest()` variants → add to `main.py` → test.

**Add a feature:** Add to `features.py` → update `ALL_FEATURES` + `min_rows_needed()`.

**Change defaults:** Edit `config.py` → `DEFAULT_TRADING_FEE_PCT`, `DEFAULT_STOP_LOSS_PCT`.

## Web GUI

FastAPI backend + React frontend. Backend wraps `StockAppAPI` — no business logic duplication.

```
React (localhost:5173)  →  Vite proxy /api  →  FastAPI (localhost:8000)  →  StockAppAPI  →  engine/
```

**Run:** `./web/dev.sh` or manually: `uvicorn web.backend.app:app --reload` + `cd web/frontend && npm run dev`

**Pages (status):**
- Dashboard ✓ — zoomable chart (line/candle, pan bar), stats, OHLCV table (Δ% sort, export CSV), custom period
- Predict ✓ — unified builder (per-model period+news), 9 variants incl. LSTM, quick presets, auto consensus, JSON caching, historical predictions
- Settings ✓ — persistent JSON, k/fee/SL sliders+text, LSTM preference with fallback, collapsible dev section
- Backtest, Training, Analysis — stubs (backend endpoints ready)

**Backend routes:**
- `routes/data.py` — ticker list, OHLCV (from DB, limit=0 = no limit), refresh
- `routes/predict.py` — `POST /run` (unified per-model config), `/cached`, `/historical`. Cache: `predictions/{ticker}/{date}.json`
- `routes/backtest.py` — delegates to `engine/backtester.py`
- `routes/train.py` — background LSTM training, model inventory from `models/`
- `routes/settings.py` — GET/PUT/PATCH `data/settings.json`
- `routes/analysis.py` — News vs No-News paired comparison

**Reusable components:** `PriceChart` (zoomable SVG), `DataTable` (generic sort/filter/paginate/export), `ui.tsx` (Panel, Btn, pct, usd)

**Predict endpoint:** `POST /api/predict/run` takes `{ticker, items: [{model, period, news}], refresh_data}`. Returns `{predictions, consensus}`.

**9 model variants:** k-NN, k-NN (TW), k-NN Enhanced, k-NN Enhanced (TW), LinReg, LinReg (TW), LinReg Enhanced, LinReg Enhanced (TW), LSTM. Each can independently have news on/off and its own period.

## CI / CD

### Pre-commit hooks (local)

Runs before every `git commit` — auto-fixes what it can, blocks on the rest:

```
git commit → ruff --fix → ruff format → mypy → commit (or fail)
```

Setup: `uv pip install -e ".[dev]" && uv run pre-commit install`

Config in `.pre-commit-config.yaml`. Uses `uv run mypy` so it works without activated venv (Neovim/fugitive compatible).

### GitHub Actions (remote)

`.github/workflows/tests.yml` runs on every push/PR to `main`:

```
lint       → ruff check + ruff format --check    (blocking)
typecheck  → mypy engine/ interface/              (blocking)
test       → pytest --cov + upload to Codecov     (Python 3.12 + 3.13 matrix)
```

Coverage uploaded to Codecov (`.codecov.yml` sets 60% target).

**Before pushing:** pre-commit hooks auto-run on `git commit`. Manual check: `ruff check --fix . && ruff format . && mypy engine/ interface/ web/backend/ && python -m pytest`

**Ruff rules:** unused imports, import sorting, `list`/`dict` instead of `typing.List`/`Dict`, bugbear, simplify. Tests exempt from annotation rules.

**Mypy:** strict on `engine/backtester.py` and `engine/utils.py` (require full annotations). Lenient on rest. `engine/ai_model.py` excluded (torch Optional patterns).

## Logging & progress

`engine/logger.py` provides centralized logging and progress bars. Two modes via `config.py → LOG_MODE`:

| Mode | Log level | Progress bars | Use case |
|---|---|---|---|
| `"cli"` | INFO | tqdm (with fallback) | Terminal / development |
| `"gui"` | WARNING | silent passthrough | Future web/desktop UI |

```python
from engine.logger import get_logger, progress_bar, epoch_progress

log = get_logger(__name__)
log.info("operational message")   # shown in cli, hidden in gui
log.warning("something wrong")    # shown in both modes

for ticker in progress_bar(tickers, desc="Predicting"):  # tqdm in cli, silent in gui
    ...

pbar = epoch_progress(100, desc="LSTM quick")  # manual update for training
for epoch in range(100):
    pbar.update(1)
    pbar.set_postfix_str(f"loss={loss:.4f}")
pbar.close()
```

Convention: `print()` for user-facing tables/reports. `log.*` for operational messages.

## Architecture notes

- `backtest.py` is a thin CLI wrapper (~240 lines). Logic lives in `backtester.py` and `backtest_helpers.py`.
- All scripts call `api.refresh_tickers(tickers)` upfront (prices + news → SQLite). `--no-refresh` skips this for offline use.
- `api.refresh_tickers()` is the single refresh method — used by refresh.py, main.py, backtest.py, run_all.py, and future GUI.
- `engine/utils.py` holds shared helpers used by both engine and interface layers.
- Backtester computes: accuracy, P/L, PF, max drawdown, Sharpe, Sortino, streaks, B&H + B&H DD, yearly breakdown.
- Sharpe = (mean daily return / std) × √252. Sortino = same but downside std only.
- Max drawdown = worst peak-to-trough decline of cumulative equity curve.
- Yearly performance only shown when data spans 2+ calendar years.
- Stop-loss: checked against intraday High/Low. Exit at SL price, not close.
- When SL is on, each model runs twice (baseline + SL) in `run_single_backtest()`.
- Buy-and-hold = (last close - first close) / first close.
- LSTM auto-loads best model (cluster > standard > quick). Normalizes via StandardScaler (saved with weights).
- k-NN time-weighting uses exponential decay × distance (not data trimming).
- Early stopping in LSTM: patience 10/20/50 for quick/standard/cluster.
- `run_all.py` creates subdirectories: `results/{scope}_{days}d[_fee][_sl][_bh]/`.
