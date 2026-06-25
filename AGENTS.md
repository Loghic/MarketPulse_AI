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
cli_helpers.py              → Shared CLI arg groups: scope (add_scope_args/resolve_scope/scope_label, driven by ASSET_CLASSES) + add_strategy_args (fees/stop-loss/sl-sweep/turnover/hold/min-confidence/buy-hold) + resolve_sl_levels + add_model_filter_args + add_news_args + add_common_run_args. backtest.py/run_all.py/oos_harness.py all compose these.
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
  test_calibration.py       → §1.3 reliability bins, Brier, ECE, gating + in-engine gate
  test_significance.py      → §1.4 binomial, Wilson CI, bootstrap CI, permutation, BH-FDR
  test_logger.py            → Logger modes, progress bar, config sanity
  test_web_api.py           → FastAPI endpoints: data, predict, backtest, train, settings, analysis

web/                        → Web GUI (FastAPI + React)
  dev.sh                    → Start both servers (backend + frontend)
  backend/
    app.py                  → FastAPI main (CORS, logging, Swagger at /docs)
    schemas.py              → Pydantic request/response models
    routes/
      meta.py               → GET /api/meta — config-driven options (model families w/ availability, asset classes, benchmarks, periods, SL/confidence sweeps, defaults). Single source so the frontend never hardcodes lists.
      data.py               → GET tickers (asset_type via config.asset_type_of), GET OHLCV, POST refresh / refresh-news
      predict.py            → POST /run (unified per-model config), /cached, /historical, /info (availability-gated variants incl. Prophet/Chronos/Kronos)
      backtest.py           → POST backtest (+ models/include_baselines/min_confidence/turnover_fees/hold_days/sl_levels/sl_sweep), /progress, /runs
      oos.py                → POST /api/oos (harness run + persist), /progress, /runs (web cache + CLI CSV discovery), /runs/{id}
      train.py              → GET models inventory, POST start training
      settings.py           → GET/PUT/PATCH persistent user settings
      analysis.py           → POST news-comparison (for academic paper)
      docs.py               → GET /api/docs + /api/docs/{slug} — serves web/docs/*.md to the Help tab
  docs/                     → ★ END-USER concept glossary (markdown): getting-started, models, strategy, metrics, oos. Plain-language, no architecture/CLI. Rendered by the Help tab. (Separate from the repo-root docs/ dev tree.)
  frontend/
    package.json            → React 19 + Vite + TS + TanStack Query
    vite.config.ts          → Dev proxy /api → backend:8000
    src/
      main.tsx              → ★ Entry + router + layout (8 tabs). The REAL entry (App.tsx + components/layout.tsx are a dead duplicate). Imports use on-disk lowercase filenames (avoid TS1261 casing errors).
      app.css               → Dark theme, no spinbox arrows
      lib/api.ts            → Typed fetch client for all endpoints (incl. getMeta, oos*, backtest progress/runs)
      components/
        ui.tsx              → Panel, Btn, LoadingBox, pct(), usd()
        priceChart.tsx      → Reusable zoomable SVG chart (line/candle, pan bar)
        datatable.tsx       → Reusable sortable/filterable/paginated table
      pages/
        dashboard.tsx       → ★ chart, stats, OHLCV table; ticker dropdown + news scope grouped by all asset classes (meta)
        predict.tsx         → ★ builder; variants from /api/predict/info (gated); tickers grouped by all classes
        backtest.tsx        → ★ family picker (meta) + baselines, min-conf/turnover/hold/SL-sweep knobs, progress, persisted runs
        oos.tsx             → ★ OOS harness: config + live progress + aggregate + per-ticker table
        ooscompare.tsx      → ★ diff two saved OOS runs (aggregate + per-ticker)
        help.tsx            → ★ searchable Help/glossary; dep-free markdown renderer; deep-links via #docSlug/sectionSlug
        settings.tsx        → ★ persistent prefs, dev section
        training.tsx        → Stub (backend ready)
        analysis.tsx        → Complete: results browser, news-vs-no-news, compare runs

  Concept-help system: `web/docs/*.md` (end-user glossary) ← `GET /api/docs` ← `pages/help.tsx`. Feature tabs deep-link into it via `components/ui.tsx:HelpLink` (a "?" → `/help#<docSlug>/<sectionSlug>`, e.g. `strategy/stop-loss`); the heading slugs the renderer generates (GitHub-style) must match those link targets.

interface/
  api.py                    → StockAppAPI facade — single entry point

engine/
  features.py               → Shared indicators (RSI, MACD, volatility, volume)
  calibration.py            → Confidence calibration + gating metrics (reliability bins, Brier, ECE, gating_metrics/sweep) — Plan §1.3
  significance.py           → Statistical significance (binomial, Wilson CI, bootstrap CI, permutation, Benjamini-Hochberg FDR) — Plan §1.4
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
| *(baselines)* | AlwaysLong / HoldLong / AlwaysShort / PreviousDay / Momentum(n=5,20) / Random (price-only) + NewsPreviousDay / NewsInformed / NewsMomentum (news-aware) | close (+ sentiment for news-aware) | No (no parameters) |

> Forecasting models (Prophet, Chronos-2, Kronos) subclass engine/forecast_base.py:ForecastModel and also expose forecast(df) → ForecastResult for the raw predicted value. use_time_weights is ignored (like LSTM).
>
> Baselines (`engine/baseline_models.py`, family key `baseline`) share the same `predict()` contract. **Price-only** ones (AlwaysLong, HoldLong, AlwaysShort, PreviousDay, Momentum, Random) ignore `sentiment_score`. `HoldLong` returns the `"HOLD"` sentinel (buy once, hold to the end, single round-trip fee) — the "do nothing" floor, distinct from AlwaysLong's daily churn. **News-aware** ones *react* to today's look-ahead-safe sentiment with a fixed rule but never *learn* from past outcomes — staying valid stateless floors, not models: `NewsAwarePreviousDay` (yesterday's direction, flip when |sentiment| ≥ `config.BASELINE_NEWS_THRESHOLD`), `NewsAwareMomentum` (n-day momentum, same flip), and `NewsInformed` (trade the news sign on strong news, else return **`"FLAT"`** to sit the day out — distinct from NewsAwarePreviousDay, which always trades). `default_baseline_variants()` returns `(model, label, uses_news)`; `run_single_backtest` attaches the per-day sentiment provider to the news-aware ones (only when news exists). Included by default, skip with `--no-baselines`. A learning sentiment→outcome predictor would be a real model (e.g. a future sentiment-kNN), not a baseline.

Shared interface: `model.predict(df, use_time_weights, sentiment_score) → (str, float)`

## Stop-loss behavior + sweep

`run_single_backtest()` resolves a set of stop-loss **levels** via its `sl_levels` arg. A single `--stop-loss X` (X>0) keeps the legacy behaviour: each model runs **twice** (no-SL baseline + SL), the SL variant suffixed `k-NN SL2%`. `--stop-loss 0 5 10 15` or `--sl-sweep` (uses `config.SL_SWEEP`) runs each model once per level; `0` is the no-SL baseline. Levels are de-duped + sorted; each per-level `Backtester` inherits fee / gate / turnover / hold settings from the caller's instance so only the SL knob varies.

Stop-loss uses intraday High/Low: Long exits if Low ≤ entry × (1 - SL%), Short exits if High ≥ entry × (1 + SL%). A stop-out flattens the position (the next traded day re-opens and pays a turnover fee). Without `--stop-loss`/`--sl-sweep`, a single run (no duplication).

## Turnover / fee realism + hold-days + position mode

`Backtester(turnover_fees=True)` charges the round-trip fee only on days `position_changed` (open / flip), not every day — the realistic "trade only on signal changes" cost. `hold_days=N` holds an opened position N days before re-reading the signal: the model still `predict()`s every day (so **accuracy = model skill, unchanged**), but P&L uses the held `position`, which can differ from `predicted` inside a hold window. Both default off (`turnover_fees=False`, `hold_days=1`) → byte-for-byte the old charge-every-day behaviour. `BacktestResult` gains `turnover_count` (number of position changes) and `fees_paid` (actual fee drag = Σ raw−net over traded days). CLI: `--turnover-fees` / `--hold-days N` on `backtest.py`, `run_all.py`, `oos_harness.py`; `run_all` tags the dir `to` / `holdN`.

`Backtester(position_mode=True)` switches from daily mark-to-market to **position-based** P&L: the post-loop pass `_apply_position_mode()` collapses each maximal run of consecutive same-direction *traded* days (broken by a flip, a sat-out gate day, or a stop-out) into one trade — booking the **compounded** `exit/entry−1` (sign-flipped for shorts) on the run's last day, zeroing interior days, and charging **one** round-trip fee per run. This is the "buy at 100 → hold → cash out at 90 = −10% + 1 round trip" model the daily mode doesn't do. Default off → unchanged. CLI `--position-mode` everywhere; `run_all`/route dir tag `pos`. Covered by `tests/test_backtester.py::TestPositionMode`.

**Model tiers (UI):** `/api/meta` tags each family `tier` ∈ {`educational` (k-NN, LinReg — simple/illustrative), `forecast` (LSTM, Prophet, Chronos-2, Kronos — main), `baseline`} and a `slow` flag (Prophet/Chronos/Kronos slow; LSTM fast). The Backtest/OOS pickers render the main families first and the educational pair in a de-emphasised "Simple" row; `--models` keys are unchanged.

## Key types

```python
@dataclass
class DayResult:
    date, predicted, actual, confidence, correct,   # predicted ∈ {UP, DOWN, FLAT, HOLD}; FLAT = deliberate no-trade, HOLD = buy-and-hold
    close_before, close_actual, exit_price,     # exit_price = SL price or close
    trade_pnl, trade_pnl_net, stopped_out,      # stopped_out: bool
    traded,                                      # False when sat out by the confidence gate OR a FLAT signal
    position, position_changed                   # held direction + whether it changed (turnover/hold-days)

@dataclass
class BacktestResult:
    model_name, ticker, test_days, correct, accuracy,
    fee_pct, stop_loss_pct, stopped_out_count,
    min_confidence, sat_out_count, coverage,     # confidence gate; accuracy/return describe TRADED days only
    turnover_fees, hold_days, turnover_count, fees_paid,  # turnover / fee realism (2.1)
    position_mode,                               # compound same-direction holds into one trade (one round-trip fee/run)
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
uv run python backtest.py --tickers NVDA --days 100 --confidence-sweep         # θ-sweep: coverage/traded-acc/return/fees-saved
uv run python backtest.py --tickers NVDA --days 100 --significance             # binomial p + Wilson CI + bootstrap CI + FDR
uv run python backtest.py --stocks --days 100 --min-confidence 0.65            # gate: sit out days below 65% confidence
uv run python backtest.py --tickers NVDA --days 100 --sl-sweep --buy-hold      # stop-loss sweep {0,5,10,15}
uv run python backtest.py --tickers NVDA --days 100 --turnover-fees --hold-days 5  # fee only on signal changes, hold 5d
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
- Dashboard ✓ — chart, stats, OHLCV table; ticker dropdown + news-refresh scope grouped by all asset classes (via `/api/meta`)
- Predict ✓ — unified builder, variants from `/api/predict/info` (availability-gated, incl. Prophet/Chronos/Kronos), tickers grouped by all classes
- Backtest ✓ — family picker (meta) + baselines toggle, min-confidence / turnover-fees / hold-days / SL-sweep knobs, live progress, persisted-run picker, Coverage/Turnover result columns
- OOS ✓ — harness config + live progress + aggregate (beat-B&H, selection-inflation gap, gate-aware Brier/ECE) + per-ticker table
- OOS Compare ✓ — diff two saved OOS runs (aggregate side-by-side + per-ticker return diff)
- Settings ✓ — persistent JSON (fixed `lstm_preferred_preset`), dev section
- Analysis ✓ — results browser, news-vs-no-news, compare runs
- Help ✓ — searchable concept glossary (markdown from `/api/docs`); feature tabs "?" deep-link into it
- Training — stub (backend endpoints ready)

**Backend routes:**
- `routes/meta.py` — `GET /api/meta`: config-driven model families (availability-gated via `api.lstm_available`/`forecast_available`), asset classes, benchmarks, periods, sweeps, defaults
- `routes/data.py` — ticker list (asset_type from `config.asset_type_of`), OHLCV, refresh, refresh-news
- `routes/predict.py` — `POST /run`, `/cached`, `/historical`, `/info` (gated variants). Cache: `predictions/{ticker}/{date}.json`
- `routes/backtest.py` — delegates to `engine/backtester.py`; plumbs models/include_baselines/min_confidence/turnover_fees/hold_days/sl_levels/sl_sweep; `/progress` + `/runs`
- `routes/oos.py` — wraps `scripts.oos_harness`; `POST /api/oos` (run + persist CSV tree + JSON cache under `oos_runs/`), `/progress`, `/runs`, `/runs/{id}`. **`/runs` lists both sources:** the web JSON cache *and* CLI-produced `results/oos_*/` dirs (auto-discovered via `_csv_run_dirs()`/`_read_csv_run()`, reconstructed through `OOSTickerRow`/`OOSSummary.model_validate`, with `source: "web"|"csv"`). A CSV dir whose name matches an existing `oos_runs/{id}.json` is skipped (JSON wins — no duplicate); `run_id` = the run-dir name in both. `/runs/{id}` falls back to the CSV tree when no JSON cache exists. Empty cells fall back to schema defaults (old runs lacking the benchmark columns parse fine). OOS + OOS-Compare tabs both read this list, so every past run (CLI or web) is browsable/comparable. Covered by `tests/test_web_api.py::TestOOSRunDiscovery`.
- `routes/train.py` — background LSTM training, model inventory from `models/`
- `routes/settings.py` — GET/PUT/PATCH `data/settings.json`
- `routes/analysis.py` — News vs No-News paired comparison + results-dir browser

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
- **FLAT signal**: `model.predict()` may return `"FLAT"` to sit a day out regardless of the confidence gate — recorded as an untraded day (0 P&L, no fee, excluded from accuracy / coverage / calibration), and it breaks a held run in position mode. Non-`UP/DOWN/FLAT/HOLD` predictions are dropped (malformed). Used by the `NewsInformed` baseline (flat on weak news). Calibration (`pairs_from_days`) and `gating_metrics` count only real `UP/DOWN` calls.
- **HOLD signal**: `model.predict()` may return `"HOLD"` to mean buy-and-hold — `is_hold` days are always "traded" (gate-independent) and book a long per-day mark-to-market, but are **fee-exempt per day** and **stop-loss-exempt**; the post-loop collapse pass (always run when any HOLD day exists, even with `position_mode=False`) compounds the whole run into a single trade with **one** round-trip fee. Net effect = B&H return minus one round-trip fee. Excluded from accuracy (not a directional call). Used by the `HoldLong` baseline. See `tests/test_backtester.py::TestHoldMode`.
- **OOS benchmark column**: `scripts/oos_harness.py` compares each ticker's OOS winner against a fixed market benchmark (`compute_benchmarks`) over the eval window — surfaced as `oos_benchmark` / `beats_benchmark_oos` per row and `oos_beat_benchmark_rate` in the aggregate (web: `OOSTickerRow`/`OOSSummary` fields, "OOS Bench"/"Beat BM?" columns). A surfaced comparison, not a baseline (one benchmark spans all tickers).
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
- **Shared non-scope arg groups** (same file): `add_strategy_args(parser, *, sl_sweep=True, min_confidence_help=None)`
  (fees / stop-loss / sl-sweep / turnover-fees / hold-days / min-confidence / buy-hold) +
  `resolve_sl_levels(args) → (sl_levels, legacy_sl)`; `add_model_filter_args` (--models / --no-baselines);
  `add_news_args` (--sentiment-method / --news-lookback-days / --news-half-life-days); `add_common_run_args(parser, *, days_default, with_periods=True)`
  (--days / --periods / --no-refresh). `backtest.py`, `run_all.py`, `oos_harness.py` all compose these, so a flag's
  spelling/default/help lives in one place. `oos_harness.py` calls `add_strategy_args(sl_sweep=False)` (single-valued
  `--stop-loss`, no `--sl-sweep` — the harness must not sweep SL) with a custom `min_confidence_help`; run_all keeps
  its bulk-news-only flags (`--news-source`/`--news-history-days`/`--force-news`) locally. The OOS harness now uses the
  full combinable scope set (gained commodities/indices/fx by adopting `add_scope_args`). Covered by `tests/test_cli_helpers.py`.

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

### Regression / point-forecast track (R-phase) — separate from trading

A parallel evaluation path that scores predicted **price levels**, not UP/DOWN
trades. It is kept deliberately separate from `backtester.py` (no positions /
fees / SL) per `plan.md` (forecasting plan, R1.2). See that file for the full
R0–R8 roadmap (residual Prophet+LSTM hybrid, DM/Wilcoxon, residual diagnostics).
Implemented so far (R0–R2 foundation):
- `engine/regression_metrics.py` — **pure** (numpy/stdlib). Absolute: `rmse`,
  `mae`, `mape`, `smape`. **Scale-free skill (the headline):** `mase`, `rmsse`
  (in-sample naive scaling), `theil_u2` = RMSE(model)/RMSE(random-walk). **U2 <
  1 ⇔ beats RW**; the RW forecaster scores exactly 1.0. `compute_all()` bundles
  them into `ForecastMetrics`. Rationale: on a price level the no-change RW
  already gets a tiny RMSE/MAPE, so absolute errors flatter everything — report
  U2/MASE first. Degenerate denominators return NaN (never inf), so medians
  across tickers stay clean.
- `engine/naive_forecasters.py` — the regression baselines (analogue of
  `baseline_models.py`): `RandomWalkForecaster` (`P̂=P_t`, the U2 reference),
  `RandomWalkDriftForecaster` (+ mean change), `SeasonalNaiveForecaster(m)`. All
  subclass `ForecastModel` via `_raw_forecast`, so they plug into the harness
  and get the value→direction adapter free. `default_naive_forecasters(season)`.
- `engine/arima_model.py` (`ARIMAForecaster`, optional **statsmodels**;
  `auto=True` uses pmdarima if present) and `engine/xgboost_model.py`
  (`XGBoostForecaster`, optional **xgboost**) — both subclass `ForecastModel`,
  gated by `_STATSMODELS_AVAILABLE` / `_XGBOOST_AVAILABLE`, raise a clear
  RuntimeError on construction when absent so the harness skips them. XGBoost
  trains on the **Δ target** (`close[t+h]−close[t]`) and adds it back to the last
  close, so trees aren't capped by the training range on a trend (target is a
  level either way — plan R0.1). Reuses `features.py`.
- `engine/forecast_backtester.py` — lean walk-forward loop (`ForecastBacktester`).
  Expanding window, **direct-h** horizons, refit-cadence `K` (cost knob only —
  visible data is always ≤ t regardless of K), per step records `(date,
  horizon, y_true, y_pred, y_naive=close[t])`. **Leakage guarantee:** the model
  only ever receives `df.iloc[:t+1]`; the realised `close[t+h]` is never in that
  slice (unit-tested with a spy). Scores via `compute_all` against the shared RW
  reference. Returns `ForecastRun` (steps + `ForecastMetrics`).
- `engine/lstm_regressor.py` (`LSTMRegressorForecaster`, optional **torch**) — a
  **new** LSTM with a *linear* head predicting the next-close Δ (added back to
  last close). **Distinct from `ai_model.py`**, which is a UP/DOWN classifier
  whose sigmoid output has no magnitude info — so the saved `{ticker}_{period}_
  {preset}.pt` classifiers are *not* reusable here. LSTM-reg is **per-ticker**:
  it lazily loads pre-trained weights from `models/{ticker}_reg.pt` (`_reg`
  suffix avoids collision) and skips the ticker if torch/weights are absent.
  `train_regressor(preset=…)` builds + fits the net (chronological val split,
  early stop, feature + Δ standardisation saved in the checkpoint). Presets
  `quick/standard/cluster` (`REG_TRAINING_PRESETS`, mirroring the classifier's
  tiers) set capacity+budget; explicit kwargs override. **No discrete period
  files** — the training window is the `--max-train` knob (no per-ticker period
  *selection*, by design, to avoid regression-track selection inflation). Will
  double as the Phase-R3 residual learner. Trained via `scripts/train_lstm_regressor.py`, which
  **trims the last `--days+--horizon` rows before fitting** so the harness eval
  window is unseen (loaded weights are OOS-valid only when trained with the same
  `--days/--horizon` used to score).
- **Residual hybrid (R3, the paper's central artifact):**
  `engine/residual_hybrid.py` `ResidualHybrid(base, residual_learner)` is a
  `ForecastModel`: `forecast(df,h)` = base OOS point + `residual_learner` trained
  on the base's **in-sample residuals** (`res_t = close_t − fitted_t`). Composable
  — any `ForecastModel` base × any `fit(residuals)/predict()->float` learner
  (Prophet+LSTM, ARIMA+LSTM, …). **Leakage rule (R0.2), unit-tested:** learner
  sees residuals up to `t` only, base forecast for `t+h` uses no data past `t`; a
  zero/short learner makes hybrid ≡ base (safe fallback). Needs the new
  `ForecastModel.fit_in_sample(df) -> np.ndarray` hook (default = random-walk
  fitted = shift-by-1; **overridden** in Prophet via in-sample `predict` and
  ARIMA via `fittedvalues`, both falling back to the RW default on error).
  Residual learners live in `engine/residual_learners.py`: `ZeroResidualLearner`
  (identity/fallback) and `LSTMResidualLearner` (small univariate LSTM,
  early-stopped; predicts 0 without torch; has `save`/`load`/`set_window`/
  `is_trained` for the frozen path). `hybrid_residual_path(ticker)` =
  `models/{ticker}_hybrid_res.pt`.
  **Fit cadence** (`ResidualHybrid(fit_mode=…)`): `per_step` (refit every call),
  `refit_k` (refit every K calls, `set_window` between), `pretrained` (frozen
  weights, predict-only — fastest). Pretrained weights come from
  `scripts/train_hybrid_residual.py` (trims the last `--days+--horizon` rows
  before fitting the base + residuals, like the LSTM-reg trainer;
  `--preset {quick,standard,cluster}` reuses `REG_TRAINING_PRESETS`, default
  standard). **Opt-in** in
  the harness via `--hybrid` (off by default — slowest model), with
  `--hybrid-fit {pretrained,refit_k,per_step}` (default pretrained) +
  `--hybrid-refit-k`; built per-ticker so pretrained mode loads its weights.
- **`--target {level,log-return}` (R0.1, score-only):** `ForecastBacktester(target=…)`
  — in log-return mode it converts each step's level forecast to `r̂=log(P̂/close[t])`
  (close[t] = the recorded `y_naive`) and scores vs a **zero-return** reference
  (RW still → U2 1.0); MASE scales on in-sample log-returns. Models untouched; U2
  ranking identical to level (it divides out persistence) — the value is the
  reviewer-expected "beat predict-no-move?" framing. Native return-target training
  is a TODO (plan R0.1). New **`smallcap`** asset class (`--smallcap`: IWM, XLE/
  XLF/XLU, XOM, FCX) — the less-efficient-corners test for the no-edge thesis.
- `scripts/forecast_harness.py` — CLI entrypoint (regression analogue of
  `oos_harness.py`). **`--models`** selector: `resolve_model_keys(spec)` (pure)
  expands keys (`rw/rwdrift/seasonal/arima/xgboost/prophet/chronos/kronos`) and
  groups (`paper` default = study + benchmarks; `benchmarks`; `foundation` =
  Chronos-2+Kronos; `all`) into an ordered key list; `build_forecasters(keys)`
  instantiates only those that are available (gracefully skips missing libs).
  Chronos-2/Kronos run only via `all` or by name (slow). Per-ticker LSTM-reg
  (`--no-lstm` to drop) + the Prophet+LSTM hybrid (`--hybrid`) are added in the
  loop. `--macro` adds a `+macro` variant for *selected* macro-capable models
  (xgboost/prophet) only. Persists tidy per-step CSV +
  `_fc_summary.csv` under `results/fc_<scope>_<days>d_h<h>_<ts>/`; console table
  ranked by U2/MASE. `--days/--horizon/--refit-k/--min-train` + scope flags.
- New optional deps (statsmodels, xgboost, scipy, pmdarima) added to the
  `[forecast]` extra. Tests: `tests/test_regression_metrics.py`,
  `test_naive_forecasters.py`, `test_forecast_backtester.py` (all pure, run
  without the optional libs).
- **Forecast-comparison stats (R5):** `engine/forecast_significance.py` — the
  regression analogue of `significance.py` (which is directional only). On
  per-step errors: `dm_test` = Diebold–Mariano on the loss differential
  `g(e₁)−g(e₂)` (squared/abs), **pure numpy** — Newey–West HAC var (`h−1` lags) +
  Harvey–Leybourne–Newbold correction + Student-t p (scipy `t` if present, else a
  numpy `betainc`/CF fallback). Sign: `stat<0` ⇒ model 1 better. `wilcoxon_loss_
  test` = signed-rank (scipy if present, else numpy normal-approx w/ tie
  correction). `compare_to_reference(cases)` runs DM+Wilcoxon for each
  model-vs-reference and applies `significance.benjamini_hochberg` FDR across the
  grid; a row is `dm_significant` only if it survives FDR **and** `mean_diff<0`
  (genuinely beats, not merely differs). scipy gated behind `[forecast]`; module
  works without it. Tests: `tests/test_forecast_significance.py` (DM symmetry
  `DM(a,b)=−DM(b,a)`, identical→0, near-identical not-sig, Student-t known
  values, grid/FDR winner-only).
- **Residual diagnostics (R6):** `engine/residual_diagnostics.py` — pure
  numpy/stdlib: `acf`, `ljung_box` (Q-test for white noise; χ² sf via scipy or a
  numpy upper-incomplete-gamma fallback), `runs_test`, `variance_ratio`, bundled
  by `diagnose(residuals) -> ResidualDiagnostics` (`.structured` = Ljung–Box
  rejects white noise at 5%). `structure_vs_gain(cases)` builds the paper's
  central cross-tab: base-residual autocorrelation (Ljung–Box / |ACF1|) vs hybrid
  gain `ΔU2 = U2_base − U2_hybrid`. Tests: `tests/test_residual_diagnostics.py`
  (white noise → not structured, AR(1) → structured + positive ACF1, χ² known
  critical values, VR≈1 for a random walk, cross-tab pairing).
- **Macro features (R4.1/R4.2):** `engine/macro_data.py` — `fetch_macro()` pulls
  VIX/DXY(UUP fallback)/Gold(GLD)/SP500 as **log-returns** (yfinance) + DGS1 as a
  **level** from FRED's public CSV (no key); each series fetched defensively
  (drop+log on failure), cached via `MacroCache` (`macro_series` table). The
  leakage-critical `align_macro(ticker_dates, macro_df, lag=1)` is **pure**:
  reindex onto the ticker calendar → forward-fill macro gaps → lag (value for `t`
  is known at `t−1`). Unit-tested no-lookahead. Tests:
  `tests/test_macro_data.py`.
- **Macro wired into XGBoost + Prophet (R4.4 / R3.3):** `XGBoostForecaster(macro_df=…)`
  appends lag-1 `macro[t]` to each feature window (width-consistent; missing-macro
  rows dropped; name → "XGBoost + macro"). `ProphetModel(macro_df=…)` adds each
  macro column via `add_regressor`, carrying the forecast-date regressor value
  forward from the last in-window macro (= `t`, known at `t−1`); falls back to
  univariate on a macro gap (name → "Prophet + macro"). Harness `--macro` fetches/
  caches the panel once and adds both variants per ticker (`_build_macro_xgb`,
  `_build_macro_prophet`). Tests: `tests/test_macro_xgboost.py`,
  `test_macro_prophet.py` (lib-gated).
- **Not yet (next R-phase sessions):** macro into the LSTM-reg (pretrained
  checkpoint) + hybrid residual learner; R4.3 pos/neg sentiment split;
  multi-horizon / asset-class / regime slices (R7); robustness/seed-sweeps (R8);
  wiring DM/Wilcoxon + the residual cross-tab into the harness output.

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

### Phase-1.3/1.4 measurement rigor (calibration + gating + significance)

Implements `plan.md` §1.3 (confidence calibration + gating) and §1.4 (statistical significance). Calibration says *whether* confidence means anything; gating *acts* on it; significance says *whether any reported edge is real*. All metric code is pure (numpy + stdlib only — **no scipy**, which is only a transitive dep here).

- **`engine/calibration.py`** (§1.3) — pure metrics from a `DayResult` list:
  - `reliability_bins()` — the data behind a reliability diagram (mean confidence vs observed accuracy per bucket; confidence lives in [0.5, 1.0] since it's the prob of the *chosen* direction).
  - `brier_score()` — MSE of confidence vs the 0/1 correctness target (0.25 = "always 0.5").
  - `expected_calibration_error()` — bin-weighted mean |confidence − accuracy| (0 = perfectly calibrated).
  - `gating_metrics()` / `gating_sweep()` — for a threshold θ: traded count, **coverage**, **traded-day accuracy**, gated return, and **fees saved**. Computed post-hoc from one ungated run so a single backtest sweeps many θ; matches the in-engine gate exactly.

- **In-engine gate** — `Backtester(min_confidence=θ)`. In the per-day loop, a day with `confidence < θ` is sat out: `traded=False`, P&L and fee zeroed, stop-loss unset. Aggregation (accuracy / return / streaks / Sharpe / yearly) is computed over **traded days only**; sat-out days stay in `.days` so their confidence still feeds calibration. New `BacktestResult` fields: `min_confidence`, `sat_out_count`, `coverage`. With θ=0 (default) every day trades and behaviour is byte-for-byte unchanged. Wired through `run_single_backtest` (the no-SL clone inherits θ) and exposed as `--min-confidence` on `backtest.py` + `run_all.py` (the run-dir name gets a `mc060` segment so gated batches don't overwrite ungated ones).

- **`engine/significance.py`** (§1.4) — `binomial_test_two_sided()` (exact, H0 p=0.5), `wilson_interval()` (closed-form, never escapes [0,1]), `bootstrap_ci()` (seeded percentile bootstrap on the daily P&L series; `sum`/`mean`/`sharpe`), `permutation_test_accuracy()` (shuffle predicted directions for the null), `benjamini_hochberg()` (FDR across a family of p-values). `significance_for_days()` bundles them for one model. The plan's anti-p-hacking rule is enforced two ways: tests run only on **traded** days, and `print_significance` applies BH-FDR **across the models in one report** rather than reading a single raw p-value.

- **Console** — `print_confidence_calibration` now also prints a Brier/ECE table. `--confidence-sweep` prints `print_confidence_sweep` (the θ-sweep over `config.CONFIDENCE_SWEEP`); `--significance` prints `print_significance`. Both imply the `--full` detail block. Config: `DEFAULT_MIN_CONFIDENCE = 0.0`, `CONFIDENCE_SWEEP = [0.0, 0.55, 0.60, 0.65, 0.70]`.

- **Tests:** `tests/test_calibration.py` (19 cases — bin edges/folding, Brier & ECE hand-computed values, gating math, plus an in-engine gate test proving sat-out days have zero P&L and that engine `total_return` equals the post-hoc `gating_metrics`) and `tests/test_significance.py` (26 cases — binomial symmetry/known values, Wilson 27/40≈[0.519,0.802], bootstrap brackets the point & is seed-reproducible, permutation extremes, the classic BH worked example). All pure-function/toy-model driven, run in <0.3s, no trained model or network needed.

- **Pass bars to read off the output:** §1.3 — traded-day accuracy materially > 0.5 *and* gated return improves vs θ=0. §1.4 — accuracy CI excludes 0.5 *and* the binomial p survives BH-FDR *and* the return bootstrap CI excludes 0. The baseline finding (no edge at any horizon) predicts these mostly fail; that's the point — the harness now makes "no edge" a measured conclusion, not an impression.

- **OOS gating** — `scripts/oos_harness.py` accepts `--min-confidence θ`, threaded into the `Backtester` so the **same** gate applies to both the selection and evaluation windows (the honest "does committing to θ survive OOS?" question). **θ is fixed, never swept inside the harness** — sweeping it on the eval window would reintroduce the selection inflation the harness exists to kill; run once per θ and compare `_oos_summary.csv` files instead. `oos_one_ticker` now also returns OOS calibration (`oos_coverage`, `oos_traded_days`, `oos_sat_out`, `oos_brier`, `oos_ece`) and significance (`oos_binomial_p`, `oos_acc_ci_lo/hi`) on the evaluation window; `aggregate()` adds `median_oos_coverage`, `median_oos_brier`, `median_oos_ece`, `tickers_significant_p05`; `build_run_dir` tags the dir `mcNNN`; the console prints a gating block + per-ticker coverage column only when θ>0. Tests in `tests/test_oos_harness.py` (`TestOOSGating`, `TestAggregateGating`) cover coverage bookkeeping, the disjoint-window guarantee *under* the gate, the gating aggregate block, and legacy-row back-compat. See [docs/run/research.md](docs/run/research.md) "Confidence gating, out-of-sample".
