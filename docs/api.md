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
| `knn_model.py` / `lin_reg_model.py` / `ai_model.py` | Model implementations |
| `data_downloader.py` + `db_manager.py` | Data layer |
| `news_scraper.py` | Sentiment scoring |

`backtest.py` is a thin CLI wrapper (~240 lines). `run_all.py` creates organized subdirectories in `results/`.

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

## Adding a web UI

```python
from flask import Flask, jsonify, request
from interface.api import StockAppAPI, PredictionConfig

app = Flask(__name__)
api = StockAppAPI()

@app.route("/predict")
def predict():
    config = PredictionConfig(
        ticker=request.args.get("ticker", "AAPL"),
        period=request.args.get("period", "1y"),
        model_type=request.args.get("model", "knn"),
    )
    result = api.get_prediction(config)
    return jsonify({
        "prediction": result.prediction,
        "confidence": result.confidence,
        "price": result.last_price,
    })
```
