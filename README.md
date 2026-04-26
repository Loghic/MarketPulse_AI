# MarketPulse AI

Stock prediction engine combining k-NN, Linear Regression, and LSTM neural networks with VADER sentiment analysis. Built as a modular system with a clean separation between data layer, model engine, and interface — ready to plug into a web or desktop UI.

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
# or install everything: uv pip install -e '.[ai]'
```

## Supported tickers

Tickers are configured in `config.py`:

**Stocks:** AAPL, MSFT, NVDA, META, GOOGL, AMD, TSM, ASML, AVGO, TSLA, INTC

**Crypto:** BTC-USD, ETH-USD, SOL-USD

Both `main.py` and `backtest.py` support `--stocks`, `--crypto`, `--all`, or `--tickers` flags:

```bash
uv run python main.py --stocks              # only stocks
uv run python main.py --crypto              # only crypto
uv run python main.py --all                 # everything
uv run python main.py --tickers AAPL NVDA   # specific picks
uv run python main.py                       # default: first 3 tickers
```

To add a new ticker, edit `config.py` — no other files need to change.

## Running with Podman

[Podman](https://podman.io/) is a Docker-compatible container engine that runs without root privileges — useful for university clusters and shared servers where Docker is not available.

### First-time setup (macOS)

```bash
brew install podman
podman machine init
podman machine start
```

On Linux, Podman runs natively — just install via your package manager and skip the machine commands.

### Build and run

```bash
podman build -t marketpulse .
podman run --rm -v ./data:/app/data:z marketpulse
```

### Running on a university cluster (Singularity/Apptainer)

```bash
podman save marketpulse -o marketpulse.tar
singularity build marketpulse.sif docker-archive://marketpulse.tar
singularity run --bind ./data:/app/data marketpulse.sif
```

## Models

### k-NN (Naive + Enhanced)

Classifies next-day direction (UP/DOWN) from a sliding window of daily returns. Enhanced mode adds volume, RSI, volatility, and MACD — features are shared with LinReg via `engine/features.py`. All features are auto-scaled via `StandardScaler`.

### Linear Regression (Naive + Enhanced)

Predicts next-day return as a continuous value, derives direction from the sign and confidence via sigmoid mapping. Enhanced mode uses the same feature set as k-NN Enhanced. Supports native `sample_weight` for time-weighting.

### LSTM Neural Network

Recurrent neural network that processes price data as a time series — sees the **order** of features, not just a flat vector. Requires pre-training (via `train.py`), then saved weights are loaded for instant predictions.

Three training presets:

| Preset | Time (CPU) | Use case |
|---|---|---|
| `quick` | ~2-5 min | Testing, experiments |
| `standard` | ~15-30 min | Real use |
| `cluster` | hours (GPU) | Best accuracy, research |

```bash
# Train
uv run python train.py --ticker AAPL --period 1y --preset quick
uv run python train.py --stocks --preset standard
uv run python train.py --all --periods 1y 2y max --preset cluster

# List saved models
uv run python train.py --list

# Predict (auto-loads saved model)
uv run python main.py --tickers AAPL
```

Saved models go to `models/` with naming: `{ticker}_{period}_{preset}.pt`. Auto-loaded in order cluster → standard → quick (best available). See [docs/lstm.md](docs/lstm.md) for details.

### Sentiment adjustment

All models predict from price patterns first, then sentiment shifts the probability post-hoc. VADER is the default scorer (handles negation, intensifiers, caps), with naive keyword fallback.

## Backtesting

Walk-forward testing: hides the last N days, predicts each one using only prior data — no look-ahead bias. Tracks simulated trading P/L, profit factor, win/loss streaks. LSTM variants are included automatically when a trained model exists.

```bash
uv run python backtest.py --stocks --days 20
uv run python backtest.py --crypto --full
uv run python backtest.py --all --compare-periods --output results.csv
uv run python backtest.py --tickers NVDA TSLA --compare-periods --days 50
```

### Output modes

| Flag | Terminal output | CSV content |
|---|---|---|
| (none) | Summary table per model | One row per model (accuracy, return, PF, streaks) |
| `--full` | Summary + consensus + profit analysis + streaks | One row per day per model |
| `--compare-periods` | Period × model matrix + top 5 + streaks | One row per model × period |

`--output` works in all modes. CSV for non-programmers, JSON for further analysis.

## Project Structure

```
marketpulse-ai/
├── config.py                # ★ Tickers, periods, defaults — edit this to add assets
├── main.py                  # CLI — prediction reports
├── backtest.py              # CLI — model evaluation
├── train.py                 # CLI — LSTM model training
├── test_pipeline.py         # 13 offline tests
├── pyproject.toml           # Dependencies & build config
├── Containerfile            # Podman/Docker build
├── .containerignore         # Build context excludes
├── AGENTS.md                # AI assistant context file
│
├── interface/
│   ├── __init__.py
│   └── api.py               # StockAppAPI facade
│
├── engine/
│   ├── __init__.py
│   ├── features.py          # Shared feature engineering (RSI, MACD, volatility, volume)
│   ├── knn_model.py         # k-NN classifier (naive + enhanced)
│   ├── lin_reg_model.py     # Linear Regression (naive + enhanced)
│   ├── ai_model.py          # LSTM neural network (train, save, load, predict)
│   ├── backtester.py        # Walk-forward backtest engine
│   ├── data_downloader.py   # Yahoo Finance data via yfinance
│   ├── db_manager.py        # SQLite storage
│   └── news_scraper.py      # VADER/naive sentiment scoring
│
├── models/                  # Saved LSTM weights (auto-created, gitignored)
│   └── AAPL_1y_quick.pt
│
├── data/
│   └── market_data.db       # SQLite database (auto-created)
│
└── docs/                    # In-depth documentation
    ├── README.md            # Documentation index
    ├── knn.md               # k-NN deep dive
    ├── linear-regression.md # LinReg deep dive
    ├── lstm.md              # LSTM: training, presets, cluster deployment
    ├── features.md          # Technical indicators
    ├── sentiment.md         # Sentiment analysis
    ├── backtesting.md       # Backtesting methodology + metrics
    └── api.md               # API, architecture, DB schema
```

## Configuration

All tickers and periods live in `config.py`:

```python
STOCKS = ["AAPL", "MSFT", "NVDA", ...]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
ALL_TICKERS = STOCKS + CRYPTO
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]
```

Predictions are requested via `PredictionConfig`:

```python
from interface.api import StockAppAPI, PredictionConfig

api = StockAppAPI()
config = PredictionConfig(
    ticker="NVDA",
    period="1y",
    model_type="knn_enhanced",   # "knn", "knn_enhanced", "linreg", "linreg_enhanced", "lstm"
    use_time_weights=True,
    include_news=True,
)
result = api.get_prediction(config)
```

## Documentation

`docs/` has in-depth explanations of every component — models, features, sentiment, backtesting, API. `AGENTS.md` is a compact context file for AI assistants (Claude, GPT, Gemini) — upload it when working on the codebase.

## Testing

```bash
uv run python test_pipeline.py
```

13 tests covering all models (including LSTM when PyTorch is available), features, sentiment, backtesting, and error handling. Runs offline with mock data.

## Roadmap

- [x] Data layer (yfinance + SQLite caching)
- [x] k-NN model — naive + enhanced (RSI, MACD, volume, volatility)
- [x] Linear Regression — naive + enhanced
- [x] LSTM neural network (training presets, save/load, cluster-ready)
- [x] Shared feature engineering (`engine/features.py`)
- [x] News sentiment — VADER + naive fallback
- [x] API facade (`StockAppAPI`)
- [x] CLI with `--stocks` / `--crypto` / `--all` filtering
- [x] Walk-forward backtesting (P/L, profit factor, streaks, CSV/JSON export)
- [x] Centralized config (`config.py`)
- [x] In-depth documentation (`docs/`)
- [ ] FinBERT sentiment (finance-specific transformer)
- [ ] Visualization layer (Plotly/Matplotlib)
- [ ] Web UI (Flask/FastAPI)

## Tech Stack

Python 3.12 · pandas · yfinance · scikit-learn · NLTK (VADER) · PyTorch (LSTM) · NumPy · SQLite · uv
