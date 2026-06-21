# AGENTS.md — AI Context for MarketPulse AI

Read this before making changes. Compact reference for AI assistants.

## What this does

Predicts next-day stock/crypto price direction (UP/DOWN) using k-NN, LinReg, LSTM, and time-series forecasting models (Prophet, Chronos-2, Kronos). Sentiment adjustment via VADER or FinBERT. Walk-forward backtesting with simulated P/L, fees, stop-loss, and buy-and-hold benchmark.

## Project layout

```
.github/workflows/tests.yml    → CI: lint (ruff) + typecheck (mypy) + test (pytest+codecov)
.codecov.yml                    → Coverage thresholds (60% target)
.pre-commit-config.yaml         → Git hooks: ruff auto-fix + format + mypy before every commit

config.py                   → ★ Asset registry (ASSET_CLASSES) + periods, fees, stop-loss, forecasting config. Edit ASSET_CLASSES to add assets.
cli_helpers.py              → Shared CLI scope flags + resolver (--stocks/--crypto/--commodities/--indices/--fx/--all/--tickers), driven by ASSET_CLASSES
main.py                     → CLI: prediction reports (scope flags via cli_helpers)
backtest.py                 → CLI: model evaluation (--full/--compare-periods/--periods/--timing/--output/--stop-loss)
train.py                    → CLI: LSTM training (--preset quick/standard/cluster)
run_all.py                  → CLI: batch runner (--periods/--timing; organized subdirectories in results/)
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
  forecast_base.py          → Shared base for forecasting models (ForecastResult + ForecastModel)
  prophet_model.py          → Prophet (fits per call, CPU)
  chronos_model.py          → Chronos-2 (zero-shot foundation model, loads once)
  kronos_model.py           → Kronos (OHLCV candlestick foundation model; sibling clone via KRONOS_PATH)
  backtester.py             → Walk-forward engine (P/L, fees, SL, DD, Sharpe, Sortino, B&H, streaks, yearly, elapsed_seconds)
  backtest_helpers.py       → Shared: period filter, display, export, model variant runner, timing table
  utils.py                  → Common helpers (period_to_start_date) shared across layers
  data_downloader.py        → yfinance fetching
  db_manager.py             → SQLite (prices + news)
  news_scraper.py           → VADER + naive fallback

models/                     → Saved LSTM weights ({ticker}_{period}_{preset}.pt)
results/                    → Organized: {scope}_{days}d[_fee{N}][_sl{N}][_bh]/{TICKER}.csv
data/                       → SQLite DB (auto-created)
docs/                       → In-depth docs for humans (see docs/forecasting.md for the forecasting models)
```

## Config

```python
# Asset universe — data-driven registry. One frozen AssetClass per class;
# the rest (STOCKS/CRYPTO/COMMODITIES/INDICES/FX, ALL_TICKERS, STOCK_BENCHMARKS,
# CRYPTO_BENCHMARKS, ALL_BENCHMARKS, ASSET_TYPE, TICKER_NAMES, get_benchmarks(),
# tickers_for_scope(), SCOPE_FLAGS) all DERIVE from it.
@dataclass(frozen=True)
class AssetClass:
    key: str; label: str; cli_flag: str
    tickers: list[str]; benchmarks: list[str]
    news_names: dict[str, str] = field(default_factory=dict)  # ticker -> GDELT query

ASSET_CLASSES = [
    AssetClass("stock",     "Stocks",      "stocks",      [AAPL,MSFT,NVDA,META,GOOGL,AMD,TSM,ASML,AVGO,TSLA,INTC], ["SPY","QQQ"]),
    AssetClass("crypto",    "Crypto",      "crypto",      ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"],               ["BTC-USD"]),
    AssetClass("commodity", "Commodities", "commodities", ["GLD"],          ["SPY"]),   # gold ETF proxy
    AssetClass("index",     "Indices",     "indices",     ["VOO","QQQM"],   ["SPY","QQQ"]),  # S&P 500 / Nasdaq-100 ETF proxies
    AssetClass("fx",        "FX",          "fx",          ["FXE"],          ["SPY"]),   # EUR/USD ETF proxy
]
# get_benchmarks(ticker) keeps its NAME (backtest_helpers imports it) but is now
# registry-driven + self-excluding. news_names hold the GDELT query map exposed as
# config.TICKER_NAMES (engine/news_sources imports it — single source).
# ETF proxies (GLD/VOO/QQQM/FXE) carry volume so volume features + LSTM work unchanged;
# VOO/QQQM (not SPY/QQQ) keep tradeable indices off the benchmark set.
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]

# Logging: "cli" = INFO + progress bars, "gui" = WARNING only
LOG_MODE = "cli"
LOG_LEVEL = None  # override: "DEBUG", "INFO", "WARNING", "ERROR"

# Trading
DEFAULT_TRADING_FEE_PCT = 0.05   # per side, round-trip = 2x
DEFAULT_STOP_LOSS_PCT = 0.0      # 0 = disabled

# Forecasting models (backtests). Skipped if the lib/clone isn't present.
FORECAST_MODELS = [("prophet", "Prophet"), ("chronos", "Chronos-2"), ("kronos", "Kronos")]
FORECAST_DEVICE = None        # None = auto (cuda if available else cpu)
CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_CONTEXT = 512
# Kronos — external clone (not pip). See docs/forecasting.md.
KRONOS_PATH = None            # None -> ../Kronos (sibling of repo root)
KRONOS_MODEL_ID = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_MAX_CONTEXT = 512
KRONOS_SAMPLE_COUNT = 5        # internal averaging per predict() call
KRONOS_PROB_SAMPLES = 1        # >1 = empirical P(up) from N stochastic passes (slower)
KRONOS_T = 1.0
KRONOS_TOP_P = 0.9
```

## Model variants

| model_type | Class | Features | Pre-training? |
|---|---|---|---|
| `"knn"` | KNNModel | returns only | No |
| `"knn_enhanced"` | KNNModel | all (RSI, MACD, vol, volat) | No |
| `"linreg"` | LinearRegressionModel | returns only | No |
| `"linreg_enhanced"` | LinearRegressionModel | all | No |
| `"lstm"` | AIModel | all (sequential) | Yes → `train.py` |
| `"prophet"` | ProphetModel | close (univariate) | No (fits per call) |
| `"chronos"` | Chronos2Model | close (univariate) | No (zero-shot, downloads weights) |
| `"kronos"` | KronosModel | OHLCV | No (zero-shot, sibling clone + downloads weights) |
| *(baselines)* | AlwaysLong / PreviousDay / Momentum(n=5,20) / Random | close only | No (no parameters) |

> Forecasting models (Prophet, Chronos-2, Kronos) subclass engine/forecast_base.py:ForecastModel and also expose forecast(df) → ForecastResult for the raw predicted value. use_time_weights is ignored (like LSTM).
>
> Baselines (`engine/baseline_models.py`, family key `baseline`) share the same `predict()` contract but ignore `sentiment_score` — no "+ News" siblings. They exist as the floor real models must clear; included by default, skip with `--no-baselines`. See *Phase-1 measurement rigor* in *2026 changes*.

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
    elapsed_seconds,                             # wall time for this model's walk-forward run
    yearly_performance: List[YearlyPerformance],
    days: List[DayResult]
```

## CLI

```bash
# Predictions
uv run python main.py --stocks
uv run python main.py --commodities --fx          # per-class flags combine (union)
uv run python main.py --tickers NVDA TSLA GLD

# Training
uv run python train.py --all --periods 1y 2y max --preset standard
uv run python train.py --list

# Backtest
uv run python backtest.py --stocks --days 20 --fees 0.03 --buy-hold
uv run python backtest.py --tickers AAPL --days 20 --stop-loss 2 --fees 0.03
uv run python backtest.py --tickers NVDA --days 30 --full --timing            # per-model time breakdown
uv run python backtest.py --tickers NVDA --compare-periods --periods 1y 2y    # subset of periods (matrix)
uv run python backtest.py --compare-periods --output results.csv

# Batch runner
uv run python run_all.py --stocks --days 50 --fees 0.03 --stop-loss 2 --buy-hold
# → results/stocks_50d_fee003_sl2_bh/{AAPL,MSFT,...,_summary}.csv
uv run python run_all.py --stocks --days 100 --periods 1y 2y 5y --sentiment-method finbert
# --periods skips slow 'max'; a time-by-model-family rollup prints at the end

# Data refresh (no models, just download)
uv run python refresh.py
uv run python refresh.py --stocks

# All scripts auto-refresh by default. Skip with --no-refresh:
uv run python main.py --stocks --no-refresh
uv run python backtest.py --stocks --days 50 --no-refresh
uv run python run_all.py --stocks --days 20 --no-refresh
```

## Common tasks

**Add a ticker:** Edit `config.py` → add it to the relevant `AssetClass.tickers` in `ASSET_CLASSES` (and optionally a `news_names` entry for its GDELT query). **Add an asset class:** append one `AssetClass(key, label, cli_flag, tickers, benchmarks, news_names)` — its CLI flag (`--<cli_flag>`, combinable), benchmarks, `asset_type` tag, and news query all derive automatically; no per-script edits. Done.

**Add a model:** Create `engine/new_model.py` with `.predict()` → register in `api._get_model()` → add to `backtest_helpers.run_single_backtest()` variants → add to `main.py` → test.

**Add a forecasting model (Prophet/Chronos-style):** Create `engine/<name>_model.py` subclassing `ForecastModel` with `_raw_forecast()` → register in `api._get_model()` + `api._load_forecast_model()` (annotate `model: ForecastModel`) → add to `config.FORECAST_MODELS`. `backtest_helpers` picks it up automatically (no `backtester.py` / `backtest.py` changes). For a repo-based model with no PyPI package (like Kronos): clone it as a sibling and *append* its root to `sys.path` (append, never insert, so it can't shadow this project's `config.py`/`utils.py`), expose the location via a `config.*_PATH` override, and force a device (some predictors default to `cuda:0`).

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

**9 model variants:** k-NN, k-NN (TW), k-NN Enhanced, k-NN Enhanced (TW), LinReg, LinReg (TW), LinReg Enhanced, LinReg Enhanced (TW), LSTM. Each can independently have news on/off and its own period. (Forecasting models are backtest-only; not in the GUI builder yet.)

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

**Mypy:** strict on `engine/backtester.py` and `engine/utils.py` (require full annotations). Lenient on rest. `engine/ai_model.py`, `engine/chronos_model.py`, and `engine/kronos_model.py` excluded via `ignore_errors` (torch / external-import patterns); `prophet.*`, `chronos.*`, `model.*` in `ignore_missing_imports`.

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

- `backtest.py` is a thin CLI wrapper. Logic lives in `backtester.py` and `backtest_helpers.py`.
- All scripts call `api.refresh_tickers(tickers)` upfront (prices + news → SQLite). `--no-refresh` skips this for offline use.
- `api.refresh_tickers()` is the single refresh method — used by refresh.py, main.py, backtest.py, run_all.py, and future GUI.
- `engine/utils.py` holds shared helpers used by both engine and interface layers.
- Backtester computes: accuracy, P/L, PF, max drawdown, Sharpe, Sortino, streaks, B&H + B&H DD, yearly breakdown, and per-run `elapsed_seconds`.
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
- `--periods` (on `run_all.py` and `backtest.py --compare-periods`) restricts the period set; default is all of `ALL_PERIODS`. Skipping `max` is near-free — Prophet refits per day, and the foundation models truncate to a ~512-bar context, so 2y ≈ 5y ≈ max input.

## 2026 changes

### Asset registry & CLI scope

- `config.py` defines the asset universe as a data-driven `ASSET_CLASSES` list of frozen
  `AssetClass(key, label, cli_flag, tickers, benchmarks, news_names)`. Everything else
  derives: `STOCKS/CRYPTO/COMMODITIES/INDICES/FX`, `ALL_TICKERS`, `STOCK_BENCHMARKS`,
  `CRYPTO_BENCHMARKS`, `ALL_BENCHMARKS`, `ASSET_TYPE`, `TICKER_NAMES`, `get_benchmarks()`
  (kept name — `backtest_helpers` imports it; now registry-driven + self-excluding),
  `asset_type_of()`, `tickers_for_scope()`, `SCOPE_FLAGS`. `engine/news_sources.py` imports
  `TICKER_NAMES` from config (single source; the old local dict is gone).
- Classes: stock, crypto (now incl. `BNB-USD`), commodity (`GLD`), index (`VOO`, `QQQM`),
  fx (`FXE`). The non-stock/crypto classes use **ETF proxies** so volume features + LSTM work
  unchanged (index/FX spot symbols carry no volume); `VOO`/`QQQM` (not `SPY`/`QQQ`) keep
  tradeable indices off the benchmark set. The existing stock-vs-crypto `-USD` heuristic still
  classifies them correctly without asset_type changes.
- `cli_helpers.py` (repo root) is the single definition of the scope selectors. Every CLI calls
  `add_scope_args(parser)` to register `--stocks/--crypto/--commodities/--indices/--fx/--all/--tickers`
  and `resolve_scope(args, default=…)` to resolve them (precedence: tickers > all > class flags >
  default; explicit `--tickers` upper-cased; a class name passed to `--tickers` warns). Per-class
  flags **combine** (union), so `--commodities --fx` → GLD + FXE. `run_all.py` uses
  `scope_label(args)` for the output-dir name (combined classes join with `-`, e.g. `commodities-fx`).
  Defaults preserved: `main.py`/`backtest.py` fall back to the first 3 tickers, `run_all.py`/`refresh.py`
  to all.

### News pipeline (no look-ahead)

- `Backtester.run(sentiment_provider=...)` calls the provider once per backtest day. `backtest_helpers.run_single_backtest()` pre-fetches the ticker's full news history once via `api.db.get_news(ticker)` and filters it in memory — the closure honours `effective_date = COALESCE(published_at, date) < asof_date`, the lookback window, and the exponential half-life. Same semantics as `db.get_news_before()` but ~100× faster and no SQLite FD exhaustion on long batches. The constant `sentiment_score` argument is kept for backward compat only.
- `engine/sentiment.py` exposes `get_scorer("vader" | "finbert" | "naive")`, cached per-process. `engine/news_sources.py` exposes `get_provider("yahoo" | "gdelt" | [...])`, with the MultiProvider deduplicating by `(date, headline)`. The news table has `published_at` / `source` / `method` columns (auto-migrated on existing DBs).

### Data sanity guards

- `db_manager.save_prices()` drops rows with NULL or non-positive close before persisting; `engine/backtester.py` further drops any backtest day whose price ratio exceeds `MAX_PLAUSIBLE_DAILY_MOVE = 0.5` (±50%). Both log a warning. Without these, a single corrupt row (yfinance occasionally returns close=0 on partial bars) would yield total_return ≈ 385 (38,500%) for a 5-day run.
- `scripts/clean_prices.py` reports and optionally deletes any existing bad rows in `data/market_data.db`, plus flags suspicious adjacent-day moves above a threshold.

### Backtest persistence

- Every `POST /api/backtest` writes `backtests/{run_id}.json` (full response for tab-redisplay) plus per-ticker CSVs + `_summary.csv` to `results/{scope}_{days}d…_{YYYYMMDD-HHMMSS}/`. Timestamp in the dir means re-runs never overwrite.
- `GET /api/backtest/progress` exposes a module-level dict updated as the route iterates tickers/periods; FastAPI runs the sync route in a threadpool so the progress endpoint stays responsive.
- `GET /api/backtest/runs` and `GET /api/backtest/runs/{run_id}` list and load persisted runs.

### Web GUI

- All six tabs in `web/frontend/src/pages/` are wired in `main.tsx`. Predict has per-ticker backend caching via `GET /api/predict/cached/{ticker}`. Predict + Backtest have an opt-in chart toggle (default hidden). Analysis consumes the `results/` tree via `GET /api/analysis/results-dirs` + `GET /api/analysis/result-csv`.
- Backtest tab uses explicit `setInterval`-based polling for `/api/backtest/progress` (not TanStack Query's `refetchInterval` predicate — that was flaky) and a separate per-second `nowTick` so the "Elapsed: Ns" counter advances between polls.

### Test isolation

- `TestBacktest` in `tests/test_web_api.py` uses an autouse `_redirect_persistence` fixture that points `CACHE_DIR` + `RESULTS_DIR` at `tmp_path`. Without it, every backtest test would pollute the developer's real `backtests/` and `results/` (which the Analysis tab picker then reads). Add the same redirect to any new test that triggers `POST /api/backtest`.

### Helper scripts

- `scripts/news_impact.py` — post-processor for a `run_all.py` result tree. Pairs `+ News` rows with their no-news siblings, emits `_news_vs_no_news_{TICKER}.csv`, `_news_vs_no_news_summary.csv`, `_news_vs_no_news_overall.csv`. Pure-function helpers (`pair_rows`, `summarize_per_ticker_model`, `overall_stats`) are unit-tested in `tests/test_news_impact.py`. The same pairing logic is reimplemented in TypeScript in `web/frontend/src/pages/Analysis.tsx` so the browser doesn't need a round-trip.
- `scripts/oos_harness.py` — Phase-1.1 out-of-sample model-selection harness, see *Phase-1 measurement rigor* below.
- `scripts/clean_prices.py` — one-off DB cleanup, see *Data sanity guards* above.
- `scripts/clean_test_contamination.py` — one-off cleanup for the historical bug where test fixtures wrote the 400-day `_make_prices(seed=42)` series into the real `data/market_data.db` under real ticker names. Detect-by-fingerprint, dry-run by default, `--apply` to delete. The conftest now redirects to `tmp_path` so new test runs can't reproduce the leak.

### Forecasting models (Prophet, Chronos-2, Kronos)

- `engine/forecast_base.py` — `ForecastModel` adapts a value forecast to the `(direction, confidence)` contract, deriving `prob_up = P(forecast > last_close)` from MC samples → quantiles → point fallback. Exposes `forecast(df) → ForecastResult` so the predicted value is available (wiring it into `DayResult`/CSV is a later step). Sentiment applied post-hoc (weight 0.20), same as the other models. `predict()` never raises.
- `engine/prophet_model.py` (Meta Prophet — fits per call, CPU, direction from the prediction interval), `engine/chronos_model.py` (Amazon Chronos-2 — 120M zero-shot, loads once, CPU/GPU, direction from quantiles), and `engine/kronos_model.py` (shiyu-coder Kronos — decoder-only OHLCV candlestick foundation model). Prophet + Chronos-2 via the optional `[forecast]` extra; **Kronos is a sibling git clone, not a pip package** — imported by appending `../Kronos` to `sys.path` (override `config.KRONOS_PATH`), installed via the minimal `[kronos]` extra (einops + huggingface_hub; **not** the full `requirements.txt`, whose matplotlib 3.9.3 won't build on Python 3.14). All lazy-imported with `_PROPHET_AVAILABLE` / `_CHRONOS_AVAILABLE` / `_KRONOS_AVAILABLE`, skipped gracefully when absent.
- Kronos: point-estimate direction by default (`predict()` averages `KRONOS_SAMPLE_COUNT` stochastic paths internally); set `KRONOS_PROB_SAMPLES > 1` for empirical `P(up)` from N independent passes at ~N× cost. Candlestick-native, so it uses full OHLCV, not just close.
- Wired into **backtests only** via `config.FORECAST_MODELS` → `run_single_backtest()`. `main.py` report + web GUI not yet. TiRex parked (not on PyPI, macOS-experimental, non-standard NX-AI license).
- `print_summary_table` groups results by model family and ranks by return within group; `print_next_day_forecast` guards against models with zero valid days.

### Per-model timing & period selection

- `BacktestResult.elapsed_seconds` — `Backtester.run()` times each walk-forward run. `backtest.py --timing` prints a slowest-first per-model breakdown (`print_timing_table` in `backtest_helpers`) after the summary; `run_all.py` prints a time-by-model-family rollup (time / share / wins) at the end of a batch. Use these to drop a model that costs a lot and rarely wins.
- `--periods 1y 2y 5y` — on `run_all.py` (batch) and `backtest.py --compare-periods` (matrix), restricts the period set; default is all of `ALL_PERIODS`. Note `backtest.py` also keeps the singular `--period` for single-period mode. See *Architecture notes* for why skipping `max` is near-free.
- Empirical finding (100-day FinBERT stocks batch): direction accuracy ≈ coin flip (~0.49), only ~19% of combos beat their own buy & hold, headline returns are largely selection bias + bull market. Chronos-2 was the most useful of the new models; Prophet and Kronos the slowest and weakest on average. Treat results as research, not signal. (Details in `docs/forecasting.md`.)
- Confirmed across horizons (40/100/300-day batches): accuracy stays ~0.49–0.51 everywhere (no edge at any horizon), and the strategy's beat-buy-and-hold rate *decays* 27% → 19% → 2% as the horizon lengthens — daily fees compound (~4%/6%/30% drag) while B&H rides the bull. Even cherry-picking the best model per ticker, only ~2/11 beat B&H over 300 days. LSTM was the only family with a positive median return/Sharpe at 300d; k-NN consistently worst. The first two items on `plan.md` *Phase 1 — Measurement rigor* (naive baselines and the OOS selection harness) are now implemented — see *Phase-1 measurement rigor* below; confidence gating + statistical significance testing remain.

### Phase-1 measurement rigor (baselines + OOS harness)

Implements `plan.md` §1.1 (out-of-sample model-selection harness) and §1.2 (naive baselines). The headline result rate from `run_all.py`'s `_summary.csv` is selection-inflated; these two pieces give an honest read.

- **`engine/baseline_models.py`** (§1.2) — five trivial "predictors" sharing the same `predict(df, use_time_weights, sentiment_score) → (direction, confidence)` contract as the real models:
  - `AlwaysLongBaseline` (UP every day, conf 1.0)
  - `PreviousDayBaseline` (copy yesterday's realised direction)
  - `MomentumBaseline(n=5)` and `MomentumBaseline(n=20)` (UP iff `close[-1] > close[-1-n]`; n=20 is the plan's "Sign-Only Momentum")
  - `RandomBaseline(seed=42)` — deterministic coin flip seeded by `sha256(last_date_in_df, seed)`, so re-runs reproduce exactly.

  Wired into `backtest_helpers.run_single_backtest()` via `default_baseline_variants()`. Labels start with `"Baseline "` so the existing `--models` filter picks them up via `config.MODEL_FAMILY_LABELS["baseline"]`. **No "+ News" siblings** — baselines ignore `sentiment_score` by construction. Skip with `--no-baselines` on `backtest.py` / `run_all.py`. Pass bar for any real model: beat `Previous Day` and `Always Long`, not just B&H.

- **`scripts/oos_harness.py`** (§1.1) — disciplined alternative to `run_all.py`'s "best per ticker". For each ticker: trim the last `--days` rows → run every candidate on the selection holdout → pick the highest in-sample `total_return` → re-run **only** that exact `(model_name, period)` on the full df. The evaluation holdout is the last `--days` of the full df, strictly disjoint from selection. Reports OOS beat-B&H rate, median OOS return, and the **selection-inflation gap** (`median(in_sample − OOS)`) — the headline number for "how much of the in-sample edge is real".

  Output mirrors `run_all.py`'s convention: `results/oos_<scope>_<days>d_..._<ts>/_oos_per_ticker.csv` + `_oos_summary.csv` + `_oos_console.txt`. Shares all the news/sentiment flags with `backtest.py`. Needs at least `2 × --days + 20` rows per ticker; shorter tickers are skipped with a log line.

  Pairs naturally with baselines — `--models baseline` measures how often a coin flip "wins" OOS; mixed candidates (e.g. `--models lstm baseline`) detect when a baseline beats real models. The harness is the gating check for any Phase-2 experiment (turnover, SL sweep, LSTM tuning): if a config has no OOS edge, "tuning" it just selects on noise.

- **Tests:** `tests/test_baselines.py` (18 cases pinning each baseline's behaviour plus interface contract on the factory) and `tests/test_oos_harness.py` (11 cases — aggregate math against a hand-computed two-ticker example, CSV roundtrips, **disjoint-window guarantee** via a `Backtester.run` spy that captures every call's `df` date range, winner-is-highest-in-sample invariant, beats-B&H flag definition). End-to-end OOS tests drive the harness with the baseline-only model set so they run in milliseconds without a trained LSTM.

- **macOS FD pressure** — full pytest on macOS used to fail mid-run with `OSError: [Errno 24] Too many open files` (default `ulimit -n = 256` is exhausted by HuggingFace caches, SQLite, FastAPI TestClient sockets, tmp_path dirs). `tests/conftest.py` now bumps `RLIMIT_NOFILE` to `min(4096, hard)` at import time, wrapped in try/except so Linux / CI / sandboxed shells silently no-op. The OOS module-imports smoke test originally used `importlib.reload()` which re-executed the entire `engine.backtest_helpers` chain and was the FD-cascade trigger; it's a plain attribute check now.
