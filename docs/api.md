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

### Initialization

```python
api = StockAppAPI()
```

Creates all models:
- `api.knn` / `api.knn_enhanced` — KNNModel (returns only / all features)
- `api.linreg` / `api.linreg_enhanced` — LinearRegressionModel
- LSTM models loaded on-demand from `models/` directory (cached per ticker+period)

### get_prediction(config) → PredictionResult

The main entry point:
1. Data fetch (DB → yfinance if stale)
2. Period filter
3. News fetch + VADER scoring (if `include_news=True`)
4. Model routing via `model_type` (including LSTM auto-load)
5. Result packaging

### LSTM auto-loading

When `model_type="lstm"`, the API searches `models/` for saved weights:
1. Tries `{ticker}_{period}_cluster.pt`
2. Falls back to `_standard.pt`
3. Falls back to `_quick.pt`

Returns clear error with training command if no model exists.

## PredictionConfig

| Field | Type | Default | Options |
|---|---|---|---|
| `ticker` | str | required | Any yfinance symbol |
| `period` | str | "1y" | "1mo", "1y", "2y", "5y", "max" |
| `model_type` | str | "knn" | "knn", "knn_enhanced", "linreg", "linreg_enhanced", "lstm" |
| `use_time_weights` | bool | False | Ignored for LSTM |
| `include_news` | bool | True | |

## PredictionResult

| Field | Type | Example |
|---|---|---|
| `prediction` | str | "UP" |
| `confidence` | str | "73.5%" |
| `last_price` | float | 198.85 |
| `sentiment` | str | "POSITIVE" |
| `sentiment_score` | float | 0.42 |
| `headlines` | List[str] | [...] |
| `data_points` | int | 261 |

## Module structure

The engine is split into focused modules:

| Module | Responsibility |
|---|---|
| `backtester.py` | Core engine: walk-forward loop, P/L with fees, buy-and-hold, streaks |
| `backtest_helpers.py` | Shared utilities: period filtering, direction accuracy, export builders, display functions |
| `features.py` | Technical indicators + feature matrix building |
| `knn_model.py` / `lin_reg_model.py` / `ai_model.py` | Model implementations |
| `data_downloader.py` + `db_manager.py` | Data layer |
| `news_scraper.py` | Sentiment scoring |

`backtest.py` (CLI) is a thin wrapper (~240 lines) that imports from `backtest_helpers.py` and `backtester.py`. Same for `run_all.py`.

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
