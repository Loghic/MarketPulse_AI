# Backtesting

## What is walk-forward testing?

Walk-forward testing simulates real-world trading by training the model only on data that would have been available at each prediction point. No future data leaks into the training set.

For `--days 5`:

```
Day 1:  Train on [day_1 ... day_N-5]  →  Predict day_N-4  →  Compare with reality
Day 2:  Train on [day_1 ... day_N-4]  →  Predict day_N-3  →  Compare with reality
...
Day 5:  Train on [day_1 ... day_N-1]  →  Predict day_N    →  Compare with reality
```

Each step, the model is retrained from scratch. Walk-forward is the gold standard for time-series evaluation because it respects temporal order.

## Trading simulation

Each day, the model's prediction becomes a trade:
- **UP** → Long (buy at open, sell at close). P/L = `(close_actual - close_before) / close_before`
- **DOWN** → Short (sell at open, buy at close). P/L = `(close_before - close_actual) / close_before`

### Trading fees

Configurable via `--fees` (default from `config.py`, 0.05% per side). Each trade has two legs (entry + exit), so the round-trip cost is `2 × fee_pct`:

```
Raw P/L:   +1.50%
Fee:       -0.10%  (0.05% × 2 sides)
Net P/L:   +1.40%
```

Fees matter more than you'd think. With 50 trades at 0.1% per side, you lose 10% to fees alone. A model needs to generate >10% gross return just to break even.

Typical fee ranges:
- Stocks (commission-free brokers): 0.01-0.05% per side (spread + slippage only)
- Stocks (traditional brokers): 0.05-0.15% per side
- Crypto (Binance, Coinbase): 0.05-0.20% per side

### Buy-and-hold benchmark

`--buy-hold` computes what you'd earn by simply buying on day 1 of the test period and holding through the last day. No trades, no fees (except entry/exit once).

This is the baseline any active strategy must beat. If buy-and-hold returns +8% over 20 days and your best model returns +6% after fees, active trading is destroying value — you'd be better off doing nothing.

## Metrics

### Accuracy

Percentage of correct direction predictions. With a 50% baseline (coin flip), anything consistently above 55% is meaningful. Professional quant funds aim for 52-55% and profit through position sizing.

### Profit metrics

**Total return** — sum of net daily P/L. Positive = the model made money.

**Profit factor** — `gross_profits / gross_losses`. The single most important number:
- PF < 1.0 → losing money
- PF 1.0-1.5 → marginally profitable
- PF 1.5-2.0 → solid
- PF > 2.0 → strong
- PF = ∞ → no losing trades (suspicious on small samples)

**Average win / average loss** — size of wins vs losses. If avg_win = +1.5% and avg_loss = -0.5%, you need only 25% accuracy to break even.

### Streak metrics

Streaks measure consecutive wins or losses. E.g. `[W, W, W, L, L, W, W]` → win streaks [3, 2], loss streak [2].

**Longest win/loss streak** — extremes. A max loss streak of 6 means six losing trades in a row at some point.

**Average win/loss streak** — typical run length. You want avg_win_streak > avg_loss_streak. If avg_loss = 3.0 and avg_win = 1.5, the model tends to lose in long runs and recover slowly.

### Direction accuracy

Breaks down accuracy by predicted direction. If a model is 80% on DOWN but 40% on UP, only trust its short signals.

### Confidence calibration

Splits predictions into high-confidence (>65%) and low-confidence (≤65%). If high-confidence predictions aren't more accurate, the confidence score is noise.

### Consensus

When running multiple models, consensus shows agreement per day. Unanimous days (100%) are the strongest signals.

## CLI flags

### `--days N`

Number of test days. 5 = quick check (unreliable), 20 = solid, 50 = most reliable.

### `--period`

Training data window: `1mo`, `1y`, `2y`, `5y`, `max`. Shorter = more recent patterns. Use `--compare-periods` to find the optimal period.

### `--fees FLOAT`

Fee percentage per side. Default: `DEFAULT_TRADING_FEE_PCT` from config.py (0.05%). Set to 0 for gross returns.

### `--buy-hold`

Add buy-and-hold return to all outputs. Shows how many models beat passive investing.

### `--full`

Detailed output: day-by-day consensus, direction accuracy, confidence calibration, profit analysis with streaks, next-day signal.

### `--compare-periods`

Runs all periods, shows accuracy matrix, top 5 by return, streak analysis. The most useful mode for finding the best model + period for a specific ticker.

### `--output FILE`

Export to CSV or JSON. Content depends on mode:

| Mode | CSV rows |
|---|---|
| Basic | 1 per model (22 columns: accuracy, return, PF, fees, B&H, streaks, direction) |
| `--full` | 1 per day per model (13 columns: date, predicted, actual, pnl_net, fee) |
| `--compare-periods` | 1 per model × period (same 22 columns as basic) |

## Batch runner (`run_all.py`)

Runs `--compare-periods` for each ticker separately, saves individual CSVs with descriptive filenames:

```bash
uv run python run_all.py --days 50 --fees 0.05 --buy-hold
```

Output:
```
results/
├── AAPL_50d_fee005_bh.csv
├── BTC-USD_50d_fee005_bh.csv
├── ...
└── _summary_50d_fee005_bh_20260427.csv
```

Filename encodes the parameters so results don't overwrite each other when re-run with different settings.

The summary CSV has one row per ticker with the best model+period combination.

## Interpreting results

### Accuracy vs profit

A model with 55% accuracy and PF 2.0 is better than 70% accuracy with PF 0.8. The 55% model wins more than it loses in dollar terms.

### Models vs buy-and-hold

If no model beats buy-and-hold after fees, active trading doesn't work for that ticker/period. This is common — markets are efficient for a reason. The value of backtesting is figuring out **where** active models add value, not assuming they always do.

### Small sample bias

With 5 days, one lucky prediction swings accuracy by 20%. Use 20+ days. Profit factor ∞ on 5 days means nothing — it's 5 correct guesses.
