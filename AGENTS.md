# AGENTS.md — AI Context for MarketPulse AI

Read this before making changes. Compact reference for AI assistants.

## What this does

Predicts next-day stock/crypto price direction (UP/DOWN) using k-NN, LinReg, and LSTM. Sentiment adjustment via VADER. Walk-forward backtesting with simulated P/L, fees, and buy-and-hold benchmark.

## Project layout

```
config.py                   → ★ Tickers, periods, fees, defaults. Edit to add assets.
main.py                     → CLI: prediction reports (--stocks/--crypto/--all/--tickers)
backtest.py                 → CLI: model evaluation (--full/--compare-periods/--output)
train.py                    → CLI: LSTM training (--preset quick/standard/cluster)
run_all.py                  → CLI: batch runner (one CSV per ticker in results/)
test_pipeline.py            → 13 offline tests

interface/
  api.py                    → StockAppAPI facade — single entry point

engine/
  features.py               → Shared indicators (RSI, MACD, volatility, volume)
  knn_model.py              → k-NN (naive + enhanced)
  lin_reg_model.py          → LinReg (naive + enhanced)
  ai_model.py               → LSTM (train/save/load/predict + early stopping)
  backtester.py             → Walk-forward engine (P/L, fees, buy-and-hold, streaks)
  backtest_helpers.py       → Shared: period filter, display, export row builders
  data_downloader.py        → yfinance fetching
  db_manager.py             → SQLite (prices + news)
  news_scraper.py           → VADER + naive fallback

models/                     → Saved LSTM weights ({ticker}_{period}_{preset}.pt)
results/                    → Batch CSV outputs ({TICKER}_{DAYS}d[_fee{N}][_bh].csv)
data/                       → SQLite DB (auto-created)
docs/                       → In-depth docs for humans
```

## Config

```python
STOCKS = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMD", "TSM", "ASML", "AVGO", "TSLA", "INTC"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
ALL_TICKERS = STOCKS + CRYPTO
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]
DEFAULT_TRADING_FEE_PCT = 0.05  # per side, round-trip = 2×
```

## Model variants

| model_type | Class | Features | Pre-training? |
|---|---|---|---|
| `"knn"` | KNNModel | returns only | No |
| `"knn_enhanced"` | KNNModel | all (RSI, MACD, vol, volat) | No |
| `"linreg"` | LinearRegressionModel | returns only | No |
| `"linreg_enhanced"` | LinearRegressionModel | all | No |
| `"lstm"` | AIModel | all (sequential) | Yes → `train.py` |

Shared interface: `model.predict(df, use_time_weights, sentiment_score) → (str, float)`

## Key types

```python
@dataclass
class BacktestResult:
    model_name, ticker, test_days, correct, accuracy, fee_pct,
    total_return, profit_factor, gross_profit, gross_loss,
    avg_win, avg_loss, best_day, worst_day, win_trades, loss_trades,
    longest_win_streak, longest_loss_streak, avg_win_streak, avg_loss_streak,
    buy_hold_return, days: List[DayResult]

@dataclass
class DayResult:
    date, predicted, actual, confidence, correct,
    close_before, close_actual, trade_pnl, trade_pnl_net
```

## CLI

```bash
# Predictions
uv run python main.py --stocks
uv run python main.py --tickers NVDA TSLA

# Training
uv run python train.py --all --periods 1y 2y max --preset standard
uv run python train.py --list

# Backtest
uv run python backtest.py --stocks --days 20 --fees 0.1 --buy-hold
uv run python backtest.py --compare-periods --output results.csv
uv run python backtest.py --full --period 1y --buy-hold

# Batch runner
uv run python run_all.py --days 50 --fees 0.05 --buy-hold
```

## Common tasks

**Add a ticker:** Edit `config.py` → `STOCKS` or `CRYPTO`. Done.

**Add a model:** Create `engine/new_model.py` with `.predict()` → register in `api._get_model()` → add to `backtest_helpers.run_single_backtest()` variants → add to `main.py` → test.

**Add a feature:** Add to `features.py` → update `ALL_FEATURES` + `min_rows_needed()`.

**Change default fees:** Edit `config.py` → `DEFAULT_TRADING_FEE_PCT`.

## Architecture notes

- `backtest.py` is a thin CLI wrapper (~240 lines). All logic is in `engine/backtester.py` and `engine/backtest_helpers.py`.
- Fees are applied per trade as round-trip (2 × fee_pct). Deducted from raw P/L → net P/L.
- Buy-and-hold = (last close - first close) / first close over the test period.
- LSTM auto-loads best available model (cluster > standard > quick).
- Early stopping in LSTM: patience 10/20/50 for quick/standard/cluster.
- All numeric outputs rounded to 8 decimal places max, streaks to 1.
