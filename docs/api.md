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
4. Model routing via `model_type` (including LSTM auto-load)
5. Result packaging

### PredictionConfig

| Field | Type | Default | Options |
|---|---|---|---|
| `ticker` | str | required | Any yfinance symbol |
| `period` | str | "1y" | "1mo", "1y", "2y", "5y", "max" |
| `model_type` | str | "knn" | "knn", "knn_enhanced", "linreg", "linreg_enhanced", "lstm" |
| `use_time_weights` | bool | False | Ignored for LSTM |
| `include_news` | bool | True | |

## Module structure

| Module | Responsibility |
|---|---|
| `backtester.py` | Core engine: walk-forward loop, P/L with fees, stop-loss (intraday H/L), max drawdown, Sharpe/Sortino, buy-and-hold + B&H DD, streaks, yearly breakdown |
| `backtest_helpers.py` | Shared: period filtering, direction accuracy, export builders, display functions, model variant runner (including SL side-by-side logic) |
| `utils.py` | Common helpers shared across engine and interface (e.g. `period_to_start_date`) |
| `features.py` | Technical indicators + feature matrix building |
| `logger.py` | Centralized logging (cli/gui modes) + progress bars (tqdm with fallback) |
| `knn_model.py` / `lin_reg_model.py` / `ai_model.py` | Model implementations |
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
    ticker TEXT, date TEXT, headline TEXT, sentiment_score REAL,
    PRIMARY KEY (ticker, date, headline)
);
```

## Model interface contract

```python
def predict(self, df, use_time_weights=False, sentiment_score=0.0) -> Tuple[str, float]:
    """Returns ("UP"/"DOWN", confidence) or ("Insufficient data"/"Model not trained", 0.0)"""
```

## Code quality

Static analysis is enforced at two levels:

**Pre-commit hooks** (local, every commit): ruff auto-fixes imports and formatting, mypy checks types. Setup: `uv run pre-commit install`. Config in `.pre-commit-config.yaml`.

**CI pipeline** (remote, every push): same checks plus pytest with coverage. Config in `.github/workflows/tests.yml`.

| Tool | Config in | What it enforces |
|---|---|---|
| **Ruff** | `pyproject.toml [tool.ruff]` | Import ordering, unused imports, modern syntax (`list` not `List`), bugbear |
| **Mypy** | `pyproject.toml [tool.mypy]` | Type safety. Strict on `backtester.py` + `utils.py`. `ai_model.py` excluded (torch). |
| **Pytest** | `pyproject.toml [tool.pytest]` | 103 tests, coverage to Codecov |

To add a new strict module, move it from the lenient override to the strict override in `pyproject.toml` and add type annotations to all functions.

## Web GUI architecture

The web layer (`web/`) wraps `StockAppAPI` without duplicating business logic:

```
┌──────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  React Frontend      │────▶│  FastAPI Backend  │────▶│  StockAppAPI     │
│  localhost:5173      │     │  localhost:8000   │     │  (unchanged)     │
│  Vite proxies /api   │     │  /api/* routes    │     │  engine/ layer   │
└──────────────────────┘     └──────────────────┘     └──────────────────┘
```

**Backend** (`web/backend/`):
- `app.py` — FastAPI app with CORS for React dev server, mounts all route modules
- `schemas.py` — Pydantic models matching engine dataclasses (with validation)
- `routes/data.py` — ticker CRUD, OHLCV data from DB, refresh trigger
- `routes/predict.py` — predictions with file-based caching, consensus endpoint
- `routes/backtest.py` — delegates to `Backtester` + `run_single_backtest()`
- `routes/train.py` — LSTM training via `BackgroundTasks`, model file inventory
- `routes/settings.py` — user settings persisted to `data/settings.json`
- `routes/analysis.py` — News vs No-News paired comparison for academic paper

**Frontend** (`web/frontend/`):
- Single `main.tsx` with React Router (6 routes) + TanStack Query for data fetching
- `lib/api.ts` — typed fetch wrappers matching every backend endpoint
- Pages: Dashboard (functional), Predict/Backtest/Training/Analysis/Settings (stubs)

**Shared API instance:** `routes/data.py:get_api()` creates one `StockAppAPI` reused across all routes. No separate initialization — same instance as CLI would use.

**Running:** `./web/dev.sh` starts both servers. API docs auto-generated at `/docs` (Swagger).

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
