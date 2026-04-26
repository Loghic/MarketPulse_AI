# API & Architecture

## Design philosophy

The system is built around the **Facade pattern**: one class (`StockAppAPI`) that exposes a simple interface while hiding the complexity of data fetching, caching, model selection, and sentiment analysis.

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
```

This means: to build a web UI, you import `StockAppAPI`, call `get_prediction()`, and display the result. No need to know about k-NN internals, DB schema, or yfinance API quirks.

## StockAppAPI

### Initialization

```python
api = StockAppAPI()
```

Creates instances of all models and supporting services:
- `api.knn` — KNNModel with returns only
- `api.knn_enhanced` — KNNModel with all features
- `api.linreg` — LinearRegressionModel with returns only
- `api.linreg_enhanced` — LinearRegressionModel with all features
- `api.news_scraper` — NewsScraper (VADER + naive fallback)
- `api.db` — DatabaseManager (SQLite)

### get_prediction(config) → PredictionResult

The main entry point. Takes a `PredictionConfig`, returns a `PredictionResult`.

```python
config = PredictionConfig(
    ticker="AAPL",
    period="1y",
    model_type="knn",            # "knn", "knn_enhanced", "linreg", "linreg_enhanced"
    use_time_weights=True,
    include_news=True,
)
result = api.get_prediction(config)
```

What happens internally:
1. **Data fetch** — checks DB for cached prices, downloads from yfinance if missing or stale
2. **Period filter** — trims data to the requested period (1mo, 1y, etc.)
3. **News fetch** (if `include_news=True`) — checks DB cache, downloads from yfinance if needed, scores with VADER
4. **Model predict** — routes to the correct model via `model_type`, passes sentiment score
5. **Result packaging** — wraps everything in a `PredictionResult` dataclass

### get_data(ticker, period) → DataFrame

Fetches price data with automatic caching and staleness detection:
- First call: downloads from yfinance, saves to SQLite
- Subsequent calls: returns from SQLite (fast)
- If data is stale (>1 day for crypto, >1 business day for stocks): refreshes automatically

Returns a DataFrame with columns: `ticker, asset_type, date, open, high, low, close, volume`.

### Data staleness logic

```
Crypto (BTC-USD, ETH-USD, ...):
    → Update if last data point is >1 day old (trades 24/7)

Stocks (AAPL, MSFT, ...):
    → Update if it's a weekday and data is >1 day old
    → OR if data is >3 days old (covers weekends)
```

## PredictionConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `ticker` | str | required | Asset symbol ("AAPL", "BTC-USD") |
| `period` | str | "1y" | Training data window: "1mo", "1y", "2y", "5y", "max" |
| `model_type` | str | "knn" | Model to use: "knn", "knn_enhanced", "linreg", "linreg_enhanced" |
| `use_time_weights` | bool | False | Prioritize recent data in training |
| `include_news` | bool | True | Fetch and apply news sentiment |

## PredictionResult

| Field | Type | Example | Description |
|---|---|---|---|
| `ticker` | str | "AAPL" | Asset symbol |
| `prediction` | str | "UP" | Direction: "UP" or "DOWN" |
| `confidence` | str | "73.5%" | Model confidence as formatted string |
| `last_price` | float | 198.85 | Most recent closing price in the data |
| `analysis_period` | str | "1y" | Period used for training |
| `model_type` | str | "knn" | Model that produced this result |
| `sentiment` | str | "POSITIVE" | "POSITIVE", "NEUTRAL", or "NEGATIVE" |
| `sentiment_score` | float | 0.42 | Raw score in [-1, 1] |
| `headlines` | List[str] | [...] | Up to 5 news headlines |
| `data_points` | int | 261 | Number of rows used for training |
| `timestamp` | str | "2026-04-25 14:30" | When the prediction was made |

## Database schema

Two tables in `data/market_data.db`:

### stock_prices
```sql
CREATE TABLE stock_prices (
    ticker TEXT,
    asset_type TEXT,        -- "stock" or "crypto"
    date TEXT,              -- "2026-04-25"
    open REAL, high REAL, low REAL, close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, date)
);
```

Upsert via `INSERT OR REPLACE`. This means re-downloading data for the same date overwrites the old values (handles corrections/adjustments).

### news_sentiment
```sql
CREATE TABLE news_sentiment (
    ticker TEXT,
    date TEXT,
    headline TEXT,
    sentiment_score REAL,
    PRIMARY KEY (ticker, date, headline)
);
```

Insert via `INSERT OR IGNORE`. Same headline on the same date for the same ticker is not re-inserted. The score is the same for all headlines in a batch (it's the average VADER compound score).

## Model interface contract

Every model must implement:

```python
def predict(
    self,
    df: pd.DataFrame,
    use_time_weights: bool = False,
    sentiment_score: float = 0.0,
) -> Tuple[str, float]:
    """
    Returns: ("UP" or "DOWN", confidence_between_0_and_1)
    May also return: ("Insufficient data", 0.0) or ("Data error", 0.0)
    """
```

This is the only requirement. The model can use any algorithm internally. The API and backtester only call this method.

## Adding a web UI

The API is designed for this. A minimal Flask example:

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
        "sentiment": result.sentiment,
    })
```

No changes to the engine layer required.
