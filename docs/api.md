# API & Architecture

## Design philosophy

One class (`StockAppAPI`) exposes a simple interface while hiding the complexity of data fetching, caching, model selection, and sentiment analysis.

```
┌──────────────────┐     ┌──────────────────────────────┐
│  Any UI layer    │────▶│  StockAppAPI                  │
│  CLI / Web / App │     │    .get_prediction(config)    │
└──────────────────┘     │    .get_data(ticker, period)  │
                         └──────────────────────────────┘
                                      │
                    ┌─────────┬───────┴───────┬──────────┐
                    ▼         ▼               ▼          ▼
                 Models    NewsScraper    DBManager   DataDownloader
                 k-NN ×2   VADER/naive   SQLite      yfinance
                 LinReg ×2
                 LSTM (optional)
                 Prophet · Chronos-2 · Kronos (forecasting, optional)
```

## StockAppAPI

### refresh_tickers(tickers)

Pre-fetches prices (full history via yfinance) and news (VADER sentiment) for all given tickers into SQLite. Called automatically by all CLI scripts before running models. Use `--no-refresh` to skip.

```python
api = StockAppAPI()
api.refresh_tickers(["AAPL", "MSFT", "BTC-USD"])  # download all
# Now get_prediction() reads from DB — no network needed
```

Designed for GUI integration: one "Refresh" button calls this method, then all predictions are instant.

### get_prediction(config) → PredictionResult

1. Data fetch (DB → yfinance if stale)
2. Period filter
3. News fetch + VADER scoring (if `include_news=True`)
4. Model routing via `model_type` (including LSTM auto-load and forecasting-model load/cache)
5. Result packaging

### PredictionConfig

| Field | Type | Default | Options |
|---|---|---|---|
| `ticker` | str | required | Any yfinance symbol |
| `period` | str | "1y" | "1mo", "1y", "2y", "5y", "max" |
| `model_type` | str | "knn" | "knn", "knn_enhanced", "linreg", "linreg_enhanced", "lstm", "prophet", "chronos", "kronos" |
| `use_time_weights` | bool | False | Ignored for LSTM and forecasting models |
| `include_news` | bool | True | |

## Module structure

| Module | Responsibility |
|---|---|
| `backtester.py` | Core engine: walk-forward loop, P/L with fees, stop-loss (intraday H/L), max drawdown, Sharpe/Sortino, buy-and-hold + B&H DD, streaks, yearly breakdown, per-run `elapsed_seconds` |
| `backtest_helpers.py` | Shared: period filtering, direction accuracy, export builders, display functions, model variant runner (including SL side-by-side logic), `--models` family filter, timing table |
| `utils.py` | Common helpers shared across engine and interface (e.g. `period_to_start_date`) |
| `features.py` | Technical indicators + feature matrix building |
| `logger.py` | Centralized logging (cli/gui modes) + progress bars (tqdm with fallback) |
| `knn_model.py` / `lin_reg_model.py` / `ai_model.py` | Model implementations |
| `forecast_base.py` | Base for forecasting models: `ForecastModel` adapter + `ForecastResult` value object (direction from distribution, post-hoc sentiment) |
| `prophet_model.py` / `chronos_model.py` / `kronos_model.py` | Forecasting models — Prophet (fits per call, CPU), Chronos-2 (zero-shot, loaded once), Kronos (OHLCV candlestick foundation model, sibling clone). `[forecast]` extra covers Prophet + Chronos-2; `[kronos]` covers Kronos |
| `data_downloader.py` + `db_manager.py` | Data layer |
| `news_scraper.py` | Sentiment scoring |

`backtest.py` is a thin CLI wrapper (~240 lines). `run_all.py` creates organized subdirectories in `results/`. `refresh.py` calls `api.refresh_tickers()` without running models. All three (plus `main.py`) call `refresh_tickers()` upfront by default — `--no-refresh` skips the download for offline use.

## Backtester

The `Backtester` class accepts three parameters:

```python
Backtester(n_days=20, fee_pct=0.05, stop_loss_pct=2.0)
```

When `stop_loss_pct > 0`, the backtester checks each day's High/Low against the stop-loss threshold. If triggered, `exit_price` is the stop-loss price (not close), and `stopped_out=True` on that `DayResult`.

The side-by-side comparison (baseline vs SL) is handled in `backtest_helpers.run_single_backtest()`, which creates a second `Backtester(stop_loss_pct=0)` when SL is enabled.

### DayResult

| Field | Type | Description |
|---|---|---|
| `date` | str | Trading day |
| `predicted` / `actual` | str | "UP" or "DOWN" |
| `confidence` | float | Model confidence |
| `correct` | bool | Prediction matched actual |
| `close_before` | float | Entry price (previous close) |
| `close_actual` | float | End-of-day close |
| `exit_price` | float | Actual exit: close or stop-loss price |
| `trade_pnl` | float | Gross P/L |
| `trade_pnl_net` | float | Net P/L (after fees) |
| `stopped_out` | bool | Was stop-loss triggered? |

### BacktestResult

| Field | Type | Description |
|---|---|---|
| `accuracy` | float | Correct / total |
| `total_return` | float | Sum of net P/L |
| `profit_factor` | float | Gross profit / gross loss |
| `max_drawdown` | float | Worst peak-to-trough decline |
| `sharpe_ratio` | float | Annualized risk-adjusted return |
| `sortino_ratio` | float | Like Sharpe, downside only |
| `buy_hold_return` | float | Passive benchmark return |
| `buy_hold_max_drawdown` | float | Passive benchmark max DD |
| `fee_pct` | float | Fee per side used |
| `stop_loss_pct` | float | SL threshold (0 = disabled) |
| `stopped_out_count` | int | Days where SL triggered |
| `longest_win/loss_streak` | int | Max consecutive wins/losses |
| `avg_win/loss_streak` | float | Average streak length |
| `win/loss_trades` | int | Trade counts |
| `avg_win/avg_loss` | float | Average trade P/L |
| `yearly_performance` | List | Per-year breakdown (if multi-year) |
| `elapsed_seconds` | float | Wall-clock time for this model's walk-forward run (0.0 if untimed). Powers the `--timing` table and the `run_all.py` time-by-family rollup |

## Database schema

### stock_prices
```sql
CREATE TABLE stock_prices (
    ticker TEXT, asset_type TEXT, date TEXT,
    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
    PRIMARY KEY (ticker, date)
);
```

### news_sentiment
```sql
CREATE TABLE news_sentiment (
    ticker          TEXT,
    date            TEXT,    -- bucket date: when the row was fetched
    headline        TEXT,
    sentiment_score REAL,
    published_at    TEXT,    -- real publication date (added 2026)
    source          TEXT,    -- "yahoo" | "gdelt" | ...  (added 2026)
    method          TEXT,    -- "vader" | "finbert" | "naive"  (added 2026)
    PRIMARY KEY (ticker, date, headline, method)
);
```

The `published_at` / `source` / `method` columns were added in the 2026 news
refactor and auto-migrate on existing DBs (pre-2026 rows have NULL there).
Backtests compute `effective_date = COALESCE(published_at, date)` so per-day
sentiment uses only news published *before* each prediction day (look-ahead
safe). `method` is part of the key so the same headline can be scored by both
VADER and FinBERT and stored side-by-side.

## Model interface contract

```python
def predict(self, df, use_time_weights=False, sentiment_score=0.0) -> Tuple[str, float]:
    """Returns ("UP"/"DOWN", confidence) or ("Insufficient data"/"Model not trained", 0.0)"""
```

### Forecasting models (Prophet, Chronos-2, Kronos)

Forecasting models predict a future *value* rather than classifying direction.
They subclass `engine/forecast_base.py:ForecastModel`, which adapts the value
forecast to the same `predict()` contract above — deriving direction from the
predictive distribution (`prob_up = P(forecast > last_close)`) and applying
sentiment post-hoc (weight 0.20). They also expose the raw forecast:

```python
def forecast(self, df, horizon=1) -> ForecastResult | None:
    """Raw value forecast. None if insufficient data / model errored (never raises)."""
```

`ForecastResult` carries `last_close`, `point` (predicted next value), optional
`quantiles` / `samples`, and the derived `prob_up` / `direction` / `confidence`.
`predict()` is the thin UP/DOWN adapter on top of `forecast()`; `use_time_weights`
is ignored (like LSTM). Prophet and Chronos-2 forecast from the close series;
Kronos consumes the full OHLCV window and draws sampled forecast paths, so its
`prob_up` is the fraction of sampled paths that close above `last_close`. See
[forecasting.md](forecasting.md).

## Code quality

Static analysis is enforced at two levels:

**Pre-commit hooks** (local, every commit): ruff auto-fixes imports and formatting, mypy checks types. Setup: `uv run pre-commit install`. Config in `.pre-commit-config.yaml`.

**CI pipeline** (remote, every push): same checks plus pytest with coverage. Config in `.github/workflows/tests.yml`.

| Tool | Config in | What it enforces |
|---|---|---|
| **Ruff** | `pyproject.toml [tool.ruff]` | Import ordering, unused imports, modern syntax (`list` not `List`), bugbear |
| **Mypy** | `pyproject.toml [tool.mypy]` | Type safety. Strict on `backtester.py` + `utils.py`. `ai_model.py` + `chronos_model.py` + `kronos_model.py` excluded (torch / external sibling import); `model.*` under `ignore_missing_imports`. |
| **Pytest** | `pyproject.toml [tool.pytest]` | 103 tests, coverage to Codecov |

To add a new strict module, move it from the lenient override to the strict override in `pyproject.toml` and add type annotations to all functions.

## Web GUI architecture

See [web.md](web.md) for full documentation (pages, API endpoints, components, caching).

```
React Frontend  ────▸  FastAPI Backend  ────▸  StockAppAPI (unchanged)
localhost:5173        localhost:8000           engine/ layer
Vite proxies /api     /api/* routes
```

**Key decisions:**
- One `StockAppAPI` instance shared across all routes (`routes/data.py:get_api()`)
- Unified `POST /api/predict/run` with per-model config (model + period + news per item)
- Prediction caching: `predictions/{ticker}/{date}.json`
- Consensus auto-computed from results (no separate consensus call needed)
- Reusable `PriceChart` and `DataTable` components across pages
- Settings: `data/settings.json` via GET/PUT/PATCH

**Completed pages:** Dashboard, Predict, Settings. **Stubs:** Backtest, Training, Analysis (backend endpoints ready).

## Adding new endpoints

The backend is modular — add a new route file and register it in `app.py`:

```python
# web/backend/routes/new_feature.py
from fastapi import APIRouter
from web.backend.routes.data import get_api

router = APIRouter(prefix="/api/new-feature", tags=["new-feature"])

@router.get("/{ticker}")
def my_endpoint(ticker: str, period: str = "1y"):
    api = get_api()
    df = api.get_data(ticker, period=period)
    # ... your logic ...
    return {"ticker": ticker, "result": "..."}
```

```python
# web/backend/app.py — add one line:
from web.backend.routes import new_feature
app.include_router(new_feature.router)
```
