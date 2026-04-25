# MarketPulse AI

Stock prediction engine combining k-NN and Linear Regression with news sentiment analysis. Built as a modular system with a clean separation between data layer, model engine, and interface — ready to plug into a web or desktop UI.

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

## Running with Podman

[Podman](https://podman.io/) is a Docker-compatible container engine that runs without root privileges — useful for university clusters and shared servers where Docker is not available.

### First-time setup (macOS)

```bash
# Install Podman
brew install podman

# Create and start the Linux VM (Podman needs Linux under the hood)
podman machine init
podman machine start
```

On Linux, Podman runs natively — just install it via your package manager (`apt install podman`, `dnf install podman`, etc.) and skip the machine commands.

### Build and run

```bash
cd marketpulse-ai

# Build the image (downloads Python, installs deps, copies code)
podman build -t marketpulse .

# Run it
podman run --rm -v ./data:/app/data:z marketpulse
```

What the flags do:
- `--rm` — automatically remove the container after it finishes (keeps things clean)
- `-v ./data:/app/data:z` — mount the local `data/` folder into the container so the SQLite database persists between runs
- `:z` — SELinux relabel (harmless on macOS, may be needed on Linux)

### Useful commands

```bash
podman images       # list built images
podman ps           # list running containers
podman ps -a        # include stopped containers
podman rmi marketpulse   # delete the image to force a clean rebuild
```

### Running on a university cluster

If the cluster uses **Singularity/Apptainer** instead of Podman:

```bash
# On your machine: export the image
podman save marketpulse -o marketpulse.tar

# On the cluster: convert and run
singularity build marketpulse.sif docker-archive://marketpulse.tar
singularity run --bind ./data:/app/data marketpulse.sif
```

## What It Does

`main.py` generates a strategic report for each configured ticker (default: BTC-USD, AAPL, MSFT). For every ticker it runs predictions across multiple time horizons (1mo, 1y, 2y, max) using two models in several modes:

- **k-NN** — classifies next-day direction (UP/DOWN) from sliding-window return patterns
- **Linear Regression** — predicts next-day return as a continuous value, derives direction from the sign and confidence from the magnitude
- **Time-Weighted** variants — prioritize recent data (k-NN trims the training set, LinReg uses native `sample_weight`)
- **+ News** variants — sentiment score from recent headlines adjusts the base probability post-hoc

Example output:

```
==========================================================================================
 STRATEGIC REPORT: AAPL
==========================================================================================
PERIOD     | MODEL                  | PRED.    | CONF.      | SAMPLES
------------------------------------------------------------------------------------------
1mo        | k-NN                   | DOWN     | 60.0%      | 22
           | k-NN Time-Weighted     | DOWN     | 67.5%      | 22
           | LinReg                 | DOWN     | 71.9%      | 22
           | LinReg Time-Weighted   | DOWN     | 73.6%      | 22
           | k-NN TW + News         | UP       | 52.5%      | 22
           | LinReg TW + News       | DOWN     | 53.6%      | 22
------------------------------------------------------------------------------------------
1y         | k-NN                   | UP       | 80.0%      | 261
           | k-NN Time-Weighted     | UP       | 58.6%      | 261
           | LinReg                 | UP       | 53.4%      | 261
           | LinReg Time-Weighted   | UP       | 51.5%      | 261
------------------------------------------------------------------------------------------
  Current Market Price: 198.85 USD
  Market Sentiment:    POSITIVE (Score: 0.6)
  Latest News:
    > Apple Reports Record Q2 Earnings, Revenue Up 12%
    > Analysts Upgrade AAPL Price Target
******************************************************************************************
```

## Backtesting

`backtest.py` evaluates model accuracy using walk-forward testing: it hides the last N trading days, then predicts each one at a time using only the data available *before* that day — no look-ahead bias.

If news headlines are available for the ticker, news-informed model variants (`k-NN TW + News`, `LinReg TW + News`) are automatically included in the comparison.

```bash
uv run python backtest.py                            # default: BTC-USD, AAPL, MSFT, 5 days, all history
uv run python backtest.py --days 10                  # hold out 10 days instead of 5
uv run python backtest.py --tickers AAPL --days 20   # single ticker, 20 days
uv run python backtest.py --period 1y                # train on last year only (not full history)
uv run python backtest.py --full                     # detailed consensus + stats
uv run python backtest.py --full --period 1y --days 10 --tickers AAPL
```

### Basic output

```
======================================================================
 BACKTEST: AAPL (last 10 trading days, period=1y)
======================================================================
  Training data: 261 rows (2025-04-25 → 2026-04-24)
  News sentiment: POSITIVE (+0.60)

  MODEL                     | ACCURACY     | CORRECT
  -------------------------------------------------------
  k-NN                      | 80.0%        | 8/10
  k-NN Time-Weighted        | 70.0%        | 7/10
  LinReg                    | 50.0%        | 5/10
  LinReg Time-Weighted      | 50.0%        | 5/10
  k-NN TW + News            | 70.0%        | 7/10
  LinReg TW + News          | 40.0%        | 4/10
```

### Full output (`--full`)

The `--full` flag adds four extra sections:

**Day-by-day consensus** — what each model predicted vs reality for every test day. Unanimous days (all models agree) are marked — these are the strongest signals for day trading.

```
  DATE         | k-NN     | k-NN TW  | LinReg   | LinReg TW | ACTUAL   | AGREE
  --------------------------------------------------------------------------
  2026-04-17   | UP   ✓  | UP   ✓  | UP   ✓  | UP   ✓  | UP       | 100% ✓ ◄ unanimous
  2026-04-22   | UP   ✗  | UP   ✗  | UP   ✗  | UP   ✗  | DOWN     | 100% ✗ ◄ unanimous
```

**Direction accuracy** — is the model better at predicting UP or DOWN? If a model is 80% accurate on DOWN but 40% on UP, you'd only trust it for short signals.

```
  MODEL                     | UP acc.      | DOWN acc.
  -------------------------------------------------------
  k-NN Time-Weighted        | 4/6 (67%)    | 4/4 (100%)
```

**Confidence calibration** — are high-confidence predictions actually more accurate? If not, the confidence score is meaningless.

```
  MODEL                     | High (>65%)      | Low (≤65%)
  --------------------------------------------------------------
  k-NN Time-Weighted        | 83% (6 pred)     | 75% (4 pred)
```

**Next-day signal** — what each model would have predicted for the most recent holdout day (closest to "live" usage), with a consensus vote across all models.

```
  k-NN                       DOWN   (conf: 80.0%)  ✓
  k-NN Time-Weighted         DOWN   (conf: 62.6%)  ✓
  LinReg                     DOWN   (conf: 53.1%)  ✓
  LinReg Time-Weighted       DOWN   (conf: 56.9%)  ✓

  Consensus: DOWN (100% of models agree)
```

### The `--period` flag

By default the backtest trains on the full price history (`max`). The `--period` flag limits training data to a recent window: `1mo`, `1y`, `2y`, `5y`, or `max`.

This lets you test whether shorter training windows improve accuracy — old price patterns (e.g. from the 90s) may not be relevant for tomorrow's prediction.

### Cross-period comparison (`--compare-periods`)

Instead of testing one period at a time, `--compare-periods` runs the backtest across all periods (1mo, 1y, 2y, 5y, max) and shows a side-by-side accuracy matrix. This reveals which training window works best for each ticker — for example, BTC-USD might peak at 1y while MSFT does better with 5y.

```bash
uv run python backtest.py --compare-periods
uv run python backtest.py --compare-periods --days 10 --tickers AAPL BTC-USD
uv run python backtest.py --compare-periods --output results.csv
uv run python backtest.py --compare-periods --output results.json
```

Example output:

```
================================================================================
 PERIOD COMPARISON: AAPL (holdout=5 days)
================================================================================
  Total data: 11234 rows (1980-12-12 → 2026-04-24)

  Accuracy by period × model (holdout=5 days):

  MODEL                     |    1mo   |    1y    |    2y    |    5y    |   max    |
  ----------------------------------------------------------------------------------
  k-NN                      |    60%   |    80%   |    60%   |    60%   |    60%   |  ◄ best: 1y
  k-NN Time-Weighted        |    60%   |    60%   |    40%   |    60%   |    60%   |  ◄ best: 1mo
  LinReg                    |    40%   |    80%   |    60%   |   100%   |   100%   |  ◄ best: 5y
  LinReg Time-Weighted      |    60%   |    60%   |    80%   |   100%   |   100%   |  ◄ best: 5y
  k-NN TW + News            |    60%   |    60%   |    60%   |    40%   |    40%   |  ◄ best: 1mo
  LinReg TW + News          |    80%   |    60%   |    60%   |    60%   |    60%   |  ◄ best: 1mo

  ====================RECOMMENDED PERIODS=====================
  k-NN                      → 1y     (80%)
  k-NN Time-Weighted        → 1mo    (60%)
  LinReg                    → 5y     (100%)
  LinReg Time-Weighted      → 5y     (100%)
  k-NN TW + News            → 1mo    (60%)
  LinReg TW + News          → 1mo    (80%)

  Overall best period for AAPL: 1mo (3/6 models peak here)

********************************************************************************
```

The `--output` flag exports results to CSV or JSON (auto-detected from extension). Each row contains: ticker, period, model, accuracy, correct/total, up/down accuracy and prediction counts — ready for analysis in pandas, Excel, or any data tool.

## Project Structure

```
marketpulse-ai/
├── main.py                  # CLI entry point — prediction reports
├── backtest.py              # CLI entry point — model evaluation
├── pyproject.toml           # Dependencies & build config (uv/pip)
├── Containerfile            # Podman/Docker build
├── .containerignore         # Excludes .venv, .db etc. from Podman build context
├── test_pipeline.py         # Integration test (runs with mock data, no network)
│
├── interface/
│   ├── __init__.py
│   └── api.py               # StockAppAPI facade — single entry point for all UI layers
│
├── engine/
│   ├── __init__.py
│   ├── data_downloader.py   # Yahoo Finance historical data via yfinance
│   ├── db_manager.py        # SQLite storage (prices + news sentiment)
│   ├── knn_model.py         # k-Nearest Neighbors classifier
│   ├── lin_reg_model.py     # Linear Regression model
│   ├── backtester.py        # Walk-forward backtest engine
│   └── news_scraper.py      # News fetching + keyword sentiment scoring
│
└── data/
    └── market_data.db       # SQLite database (auto-created on first run)
```

### Architecture

```
┌──────────────────┐     ┌──────────────────────────────────────────┐
│  CLI             │     │  StockAppAPI  (interface/api.py)         │
│  main.py         │────▶│  - get_prediction(config) → result       │
│  backtest.py     │     │  - get_data(ticker, period) → DataFrame  │
└──────────────────┘     └──────┬───────┬──────────┬───────────────┘
                                │       │          │
                  ┌─────────────▼──┐ ┌──▼────┐ ┌───▼──────────┐
                  │ Models         │ │ News  │ │ DB Manager   │
                  │  KNNModel      │ │Scraper│ │ (SQLite)     │
                  │  LinRegModel   │ └───────┘ └──────┬───────┘
                  └─────────────┬──┘                  │
                                │              ┌──────▼───────┐
                  ┌─────────────▼──┐           │ data/        │
                  │ Backtester     │           │ market_data  │
                  │ (walk-forward) │           │ .db          │
                  └────────────────┘           └──────────────┘
```

`StockAppAPI` acts as a facade. All model calls, data fetching, caching, and sentiment analysis go through it — making it straightforward to swap the CLI for a Flask/FastAPI backend or a desktop GUI without touching the engine layer.

All models share the same `.predict(df, use_time_weights, sentiment_score)` interface, so the backtester and API work with any model without special-casing.

## Configuration

Predictions are configured via `PredictionConfig`:

```python
from interface.api import StockAppAPI, PredictionConfig

api = StockAppAPI()

config = PredictionConfig(
    ticker="AAPL",          # Stock or crypto ticker
    period="1y",            # 1mo, 1y, 2y, 5y, max
    model_type="knn",       # "knn" or "linreg"
    use_time_weights=True,  # Prioritize recent patterns
    include_news=True,      # Fetch & score news sentiment
)

result = api.get_prediction(config)
print(result.prediction)   # "UP" or "DOWN"
print(result.confidence)   # "80.0%"
print(result.sentiment)    # "POSITIVE" / "NEUTRAL" / "NEGATIVE"
```

## Testing

Runs entirely offline with mock data — no Yahoo Finance access needed:

```bash
uv run python test_pipeline.py
```

## Roadmap

- [x] Data layer (yfinance download + SQLite caching)
- [x] k-NN model (standard + time-weighted + news-adjusted)
- [x] Linear Regression model (standard + time-weighted + news-adjusted)
- [x] News sentiment (keyword scoring via yfinance news)
- [x] API facade (`StockAppAPI`)
- [x] CLI interface
- [x] Walk-forward backtesting (`--full`, `--period`, `--compare-periods`, `--output`)
- [ ] Neural network model — LSTM/Transformer (`engine/ai_model.py`)
- [ ] NLP sentiment upgrade (VADER or BERT instead of keyword matching)
- [ ] Visualization layer (Plotly/Matplotlib)
- [ ] Web UI (Flask/FastAPI)

## Tech Stack

Python 3.12 · pandas · yfinance · scikit-learn · NumPy · SQLite · uv
