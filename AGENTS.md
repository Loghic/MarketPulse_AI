# AGENTS.md — AI Context for MarketPulse AI

This file helps AI assistants quickly understand the codebase. Read this before making changes.

## What this project does

Predicts next-day stock/crypto price direction (UP/DOWN) using k-NN, Linear Regression, and LSTM neural networks, optionally adjusted by news sentiment. CLI-only for now, designed for easy extension to web/desktop.

## Project layout

```
config.py                   → ★ Tickers, periods, defaults. Edit this to add assets.
main.py                     → CLI: prediction reports
backtest.py                 → CLI: model evaluation with walk-forward testing
train.py                    → CLI: LSTM model training (quick/standard/cluster presets)
test_pipeline.py            → 13 offline tests (mock data, no network)
pyproject.toml              → Dependencies, build config (uv + setuptools)
Containerfile               → Podman/Docker build
.containerignore            → Excludes .venv, .db, __pycache__ from build context

interface/
  api.py                    → StockAppAPI facade — THE single entry point for all UI layers

engine/
  features.py               → Shared feature engineering (RSI, MACD, volatility, volume)
  knn_model.py              → k-NN classifier (naive or enhanced features)
  lin_reg_model.py           → Linear Regression (naive or enhanced features)
  ai_model.py               → LSTM neural network (train, save/load, predict)
  backtester.py             → Walk-forward backtest engine (P/L, PF, streaks)
  data_downloader.py        → yfinance data fetching
  db_manager.py             → SQLite storage (prices + news sentiment)
  news_scraper.py           → News headlines + VADER/naive sentiment scoring

models/                     → Saved LSTM weights ({ticker}_{period}_{preset}.pt)
data/                       → SQLite DB (auto-created, gitignored)
docs/                       → In-depth documentation for humans
```

## Config system

`config.py` is the single source of truth for tickers and periods:

```python
STOCKS = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMD", "TSM", "ASML", "AVGO", "TSLA", "INTC"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
ALL_TICKERS = STOCKS + CRYPTO
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]
DEFAULT_PERIOD = "max"
DEFAULT_BACKTEST_DAYS = 5
```

All CLI scripts import from config.py and support `--stocks`, `--crypto`, `--all`, `--tickers` flags.

## Key architecture decisions

**Facade pattern.** All logic goes through `StockAppAPI` in `interface/api.py`.

**Shared model interface.** Every model exposes:
```python
def predict(self, df, use_time_weights=False, sentiment_score=0.0) -> Tuple[str, float]
```
Returns `("UP", 0.73)`. To add a new model: implement this interface, register in `api._get_model()`.

**Shared feature engineering.** `engine/features.py` — all indicators used by k-NN, LinReg, and LSTM.

**Two-stage sentiment.** Models predict from price patterns first, then `_apply_sentiment()` shifts probability post-hoc.

**LSTM is train-once, predict-many.** Unlike k-NN/LinReg which train on every call, LSTM is pre-trained via `train.py` and weights saved to `models/`. API auto-loads the best available model (cluster > standard > quick).

## Model variants

| model_type | Class | Features | Pre-training? |
|---|---|---|---|
| `"knn"` | KNNModel | returns only | No |
| `"knn_enhanced"` | KNNModel | all (RSI, MACD, vol, volat) | No |
| `"linreg"` | LinearRegressionModel | returns only | No |
| `"linreg_enhanced"` | LinearRegressionModel | all | No |
| `"lstm"` | AIModel | all (sequential) | Yes → `train.py` |

k-NN/LinReg each have 3 modes: standard, time-weighted, + news.
LSTM has 2 modes: standard, + news (time awareness is built-in).

## LSTM training presets

```python
TRAINING_PRESETS = {
    "quick":    {"hidden_size": 32,  "num_layers": 1, "epochs": 50,   ...},
    "standard": {"hidden_size": 64,  "num_layers": 2, "epochs": 200,  ...},
    "cluster":  {"hidden_size": 128, "num_layers": 3, "epochs": 1000, ...},
}
```

Models saved to: `models/{ticker}_{period}_{preset}.pt`

## Important types

```python
@dataclass
class PredictionConfig:
    ticker: str
    period: str = "1y"
    model_type: str = "knn"    # "knn", "knn_enhanced", "linreg", "linreg_enhanced", "lstm"
    use_time_weights: bool = False
    include_news: bool = True

@dataclass
class BacktestResult:
    model_name, ticker, test_days, correct, accuracy,
    total_return, profit_factor, gross_profit, gross_loss,
    avg_win, avg_loss, best_day, worst_day, win_trades, loss_trades,
    longest_win_streak, longest_loss_streak, avg_win_streak, avg_loss_streak,
    days: List[DayResult]
```

## CLI usage

```bash
# Predictions
uv run python main.py --stocks
uv run python main.py --crypto
uv run python main.py --tickers NVDA TSLA

# Training (LSTM only)
uv run python train.py --ticker AAPL --period 1y --preset quick
uv run python train.py --stocks --preset standard
uv run python train.py --all --periods 1y 2y max --preset cluster
uv run python train.py --list

# Backtest
uv run python backtest.py --stocks --days 20
uv run python backtest.py --crypto --compare-periods --output results.csv
uv run python backtest.py --full --period 1y --output daily.csv

# Tests
uv run python test_pipeline.py
```

## Common tasks

**Add a new ticker:** Edit `config.py` → add to `STOCKS` or `CRYPTO`. Done.

**Add a new model:** Create `engine/new_model.py` with `.predict(df, use_time_weights, sentiment_score)` → add to `api._get_model()` → add variants to `backtest.py` → add to `main.py` → add tests.

**Add a new feature:** Add to `features.py` → update `ALL_FEATURES` → update `min_rows_needed()` if warmup needed.

**Train LSTM for a new ticker:** `uv run python train.py --ticker XYZ --preset quick`

## Known limitations

- LSTM requires PyTorch (optional dependency, graceful fallback if missing)
- LSTM must be trained before use — no cold-start prediction
- One LSTM model per ticker × period × preset (not transferable)
- Sentiment is today's score applied uniformly to all backtest days
- Enhanced models skip periods with <35 rows (MACD warmup)
- k-NN time-weighting is approximated by trimming training data
