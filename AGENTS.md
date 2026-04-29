# AGENTS.md — AI Context for MarketPulse AI

Read this before making changes. Compact reference for AI assistants.

## What this does

Predicts next-day stock/crypto price direction (UP/DOWN) using k-NN, LinReg, and LSTM. Sentiment adjustment via VADER. Walk-forward backtesting with simulated P/L, fees, stop-loss, and buy-and-hold benchmark.

## Project layout

```
config.py                   → ★ Tickers, periods, fees, stop-loss defaults. Edit to add assets.
main.py                     → CLI: prediction reports (--stocks/--crypto/--all/--tickers)
backtest.py                 → CLI: model evaluation (--full/--compare-periods/--output/--stop-loss)
train.py                    → CLI: LSTM training (--preset quick/standard/cluster)
run_all.py                  → CLI: batch runner (organized subdirectories in results/)
refresh.py                  → CLI: download latest prices + news (no models, just data)
test_pipeline.py            → 13 offline tests

interface/
  api.py                    → StockAppAPI facade — single entry point

engine/
  features.py               → Shared indicators (RSI, MACD, volatility, volume)
  knn_model.py              → k-NN (naive + enhanced)
  lin_reg_model.py          → LinReg (naive + enhanced)
  ai_model.py               → LSTM (train/save/load/predict + early stopping)
  backtester.py             → Walk-forward engine (P/L, fees, SL, DD, Sharpe, Sortino, B&H, streaks, yearly)
  backtest_helpers.py       → Shared: period filter, display, export, model variant runner
  utils.py                  → Common helpers (period_to_start_date) shared across layers
  data_downloader.py        → yfinance fetching
  db_manager.py             → SQLite (prices + news)
  news_scraper.py           → VADER + naive fallback

models/                     → Saved LSTM weights ({ticker}_{period}_{preset}.pt)
results/                    → Organized: {scope}_{days}d[_fee{N}][_sl{N}][_bh]/{TICKER}.csv
data/                       → SQLite DB (auto-created)
docs/                       → In-depth docs for humans
```

## Config

```python
STOCKS = ["AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMD", "TSM", "ASML", "AVGO", "TSLA", "INTC"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"]
ALL_TICKERS = STOCKS + CRYPTO
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]
DEFAULT_TRADING_FEE_PCT = 0.05   # per side, round-trip = 2×
DEFAULT_STOP_LOSS_PCT = 0.0      # 0 = disabled
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

## Stop-loss behavior

When `--stop-loss X` is passed, **each model runs twice**: once without SL (baseline) and once with SL. The SL variant gets a name suffix like `k-NN SL2%`. This is implemented in `backtest_helpers.run_single_backtest()` — it creates a second Backtester with `stop_loss_pct=0` for the baseline runs.

Stop-loss uses intraday High/Low: Long exits if Low ≤ entry × (1 - SL%), Short exits if High ≥ entry × (1 + SL%).

Without `--stop-loss`, no duplication occurs.

## Key types

```python
@dataclass
class DayResult:
    date, predicted, actual, confidence, correct,
    close_before, close_actual, exit_price,     # exit_price = SL price or close
    trade_pnl, trade_pnl_net, stopped_out       # stopped_out: bool

@dataclass
class BacktestResult:
    model_name, ticker, test_days, correct, accuracy,
    fee_pct, stop_loss_pct, stopped_out_count,
    total_return, profit_factor, gross_profit, gross_loss,
    avg_win, avg_loss, best_day, worst_day, win_trades, loss_trades,
    max_drawdown, sharpe_ratio, sortino_ratio,
    longest_win_streak, longest_loss_streak, avg_win_streak, avg_loss_streak,
    buy_hold_return, buy_hold_max_drawdown,
    yearly_performance: List[YearlyPerformance],
    days: List[DayResult]
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
uv run python backtest.py --stocks --days 20 --fees 0.03 --buy-hold
uv run python backtest.py --tickers AAPL --days 20 --stop-loss 2 --fees 0.03
uv run python backtest.py --compare-periods --output results.csv

# Batch runner
uv run python run_all.py --stocks --days 50 --fees 0.03 --stop-loss 2 --buy-hold
# → results/stocks_50d_fee003_sl2_bh/{AAPL,MSFT,...,_summary}.csv

# Data refresh (no models, just download)
uv run python refresh.py
uv run python refresh.py --stocks

# All scripts auto-refresh by default. Skip with --no-refresh:
uv run python main.py --stocks --no-refresh
uv run python backtest.py --stocks --days 50 --no-refresh
uv run python run_all.py --stocks --days 20 --no-refresh
```

## Common tasks

**Add a ticker:** Edit `config.py` → `STOCKS` or `CRYPTO`. Done.

**Add a model:** Create `engine/new_model.py` with `.predict()` → register in `api._get_model()` → add to `backtest_helpers.run_single_backtest()` variants → add to `main.py` → test.

**Add a feature:** Add to `features.py` → update `ALL_FEATURES` + `min_rows_needed()`.

**Change defaults:** Edit `config.py` → `DEFAULT_TRADING_FEE_PCT`, `DEFAULT_STOP_LOSS_PCT`.

## Architecture notes

- `backtest.py` is a thin CLI wrapper (~240 lines). Logic lives in `backtester.py` and `backtest_helpers.py`.
- All scripts call `api.refresh_tickers(tickers)` upfront (prices + news → SQLite). `--no-refresh` skips this for offline use.
- `api.refresh_tickers()` is the single refresh method — used by refresh.py, main.py, backtest.py, run_all.py, and future GUI.
- `engine/utils.py` holds shared helpers used by both engine and interface layers.
- Backtester computes: accuracy, P/L, PF, max drawdown, Sharpe, Sortino, streaks, B&H + B&H DD, yearly breakdown.
- Sharpe = (mean daily return / std) × √252. Sortino = same but downside std only.
- Max drawdown = worst peak-to-trough decline of cumulative equity curve.
- Yearly performance only shown when data spans 2+ calendar years.
- Stop-loss: checked against intraday High/Low. Exit at SL price, not close.
- When SL is on, each model runs twice (baseline + SL) in `run_single_backtest()`.
- Buy-and-hold = (last close - first close) / first close.
- LSTM auto-loads best model (cluster > standard > quick). Normalizes via StandardScaler (saved with weights).
- k-NN time-weighting uses exponential decay × distance (not data trimming).
- Early stopping in LSTM: patience 10/20/50 for quick/standard/cluster.
- `run_all.py` creates subdirectories: `results/{scope}_{days}d[_fee][_sl][_bh]/`.
