# Backtesting

## What is walk-forward testing?

Walk-forward testing simulates real-world trading by training the model only on data that would have been available at each prediction point. No future data leaks into the training set.

For `--days 5`, the model is retrained 5 times, each with one more day of data. Walk-forward is the gold standard for time-series evaluation.

## Trading simulation

Each day, the model's prediction becomes a trade:
- **UP** → Long (buy at open, sell at close). P/L = `(exit - entry) / entry`
- **DOWN** → Short (sell at open, buy at close). P/L = `(entry - exit) / entry`

The exit price is normally the end-of-day close, unless a stop-loss is triggered during the day.

### Trading fees

`--fees 0.03` means 0.03% per side. Round-trip = 2 × 0.03% = 0.06% per trade. With 50 trades at 0.1% per side, you lose 10% to fees alone.

Typical values:
- US stocks (commission-free brokers): `--fees 0.03` (spread + slippage)
- EU stocks: `--fees 0.07`
- Crypto: `--fees 0.15`

### Stop-loss

`--stop-loss 2` means: if the position moves 2% against you during the day, exit immediately at the stop-loss price. Uses actual intraday High/Low data:

- **Long:** if day's Low ≤ entry × (1 - 2%) → stopped out
- **Short:** if day's High ≥ entry × (1 + 2%) → stopped out

When stop-loss is enabled, **every model runs twice** — once without SL (baseline) and once with SL. This gives you a direct side-by-side comparison:

```
k-NN Enhanced              +3.26%  (PF 1.38)      ← baseline
k-NN Enhanced SL2%         +6.35%  (PF 2.15)      ← stop-loss cut big losses

k-NN Enh. TW              -4.96%  (PF 0.62)      ← baseline (losing)
k-NN Enh. TW SL2%         -0.87%  (PF 0.90)      ← SL limited the damage
```

The SL column shows how many trades were stopped out. If a model has 6/10 stopped out, the stop-loss is too tight — the position gets cut before the market can move in the right direction.

Typical stop-loss values: **1.5%** conservative, **2%** standard, **3-5%** for crypto.

Without `--stop-loss`, no duplication — each model runs once as before.

### Buy-and-hold benchmark

`--buy-hold` computes what you'd earn by buying on day 1 and holding. If buy-and-hold beats most models after fees, active trading doesn't add value for that ticker.

## Metrics

### Accuracy

Percentage of correct direction predictions. 50% = coin flip. Anything consistently above 55% is meaningful.

### Profit metrics

**Total return** — sum of net daily P/L.

**Profit factor** — `gross_profits / gross_losses`. PF < 1.0 = losing money. PF 1.5-2.0 = solid. PF > 2.0 = strong.

**Average win / average loss** — if avg_win = +1.5% and avg_loss = -0.5%, you need only 25% accuracy to break even.

### Streak metrics

**Longest win/loss streak** — extremes. Max loss streak of 6 = six losing trades in a row.

**Average win/loss streak** — you want avg_win > avg_loss. If avg_loss = 3.0 and avg_win = 1.5, the model loses in long runs and recovers slowly.

### Risk metrics

**Max Drawdown (DD)** — maximum peak-to-trough decline of the cumulative equity curve. If equity goes +5% → +3% → +8%, the drawdown at step 2 is -1.9%. Max DD is the worst such decline. A model with +20% return but -15% max DD means at some point you were down 15% from your peak.

**Sharpe Ratio** — risk-adjusted return, annualized: `(mean daily return / std daily return) × √252`.

| Sharpe | Meaning |
|---|---|
| < 0 | Losing money |
| 0-1 | Positive but risky |
| 1-2 | Good risk-adjusted return |
| > 2 | Excellent (rare for daily trading) |

Higher Sharpe = more return per unit of risk. +10% return with Sharpe 1.5 is better than +15% with Sharpe 0.8.

**Sortino Ratio** — like Sharpe but only penalizes downside volatility. If a model has big wins and small losses, Sortino > Sharpe (good — volatility is mostly on the upside).

**Buy & Hold Max DD** — passive benchmark's max drawdown. If model DD is -5% and B&H DD is -12%, the model protected you from a big drop.

The `--compare-periods` mode includes a **Risk-Adjusted Ranking** section sorted by Sharpe ratio — the best risk-adjusted models, not just the highest return.

### Yearly rolling performance

When test data spans multiple years, `--full` shows performance by calendar year:

```
  YEAR     | TRADES | ACC. | RETURN     | PF     | MAX DD
  2023     | 30     | 63%  | +12.50%    | 1.82   | -3.20%
  2024     | 20     | 45%  | -5.30%     | 0.71   | -8.10%
```

A model great in 2023 (AI rally) but terrible in 2024 may be overfitting to bull conditions. Stable yearly performance is more trustworthy than a high aggregate return.

### Direction accuracy, confidence calibration, consensus

Direction accuracy breaks down by UP vs DOWN. Confidence calibration checks if high-confidence predictions are actually better. Consensus shows model agreement per day — unanimous days are strongest signals.

## CLI flags

| Flag | Description |
|---|---|
| `--days N` | Holdout days (5=quick, 20=solid, 50=reliable) |
| `--period` | Training window: 1mo, 1y, 2y, 5y, max |
| `--fees FLOAT` | Fee % per side (default from config.py) |
| `--stop-loss FLOAT` | Stop-loss % (0=disabled). Runs each model twice for comparison |
| `--buy-hold` | Add buy-and-hold return to output |
| `--no-refresh` | Skip data download, use cached DB only (offline mode) |
| `--full` | Detailed: consensus, direction accuracy, profit analysis, streaks |
| `--compare-periods` | All periods, accuracy matrix, top 5, streak analysis |
| `--output FILE` | Export CSV or JSON |

### `--output` content by mode

| Mode | CSV rows |
|---|---|
| Basic | 1 per model (accuracy, return, PF, fees, SL, B&H, streaks) |
| `--full` | 1 per day per model (date, predicted, actual, pnl, stopped_out) |
| `--compare-periods` | 1 per model × period |

With `--stop-loss`, row count doubles (baseline + SL variant for each model).

## Data refresh

All scripts (`main.py`, `backtest.py`, `run_all.py`) automatically download fresh prices and news before running. This ensures you always work with the latest data.

To skip downloads and use only cached data from the database:

```bash
uv run python backtest.py --stocks --days 50 --no-refresh
uv run python run_all.py --all --days 20 --no-refresh
```

Useful when: running multiple analyses in a row (refresh once, then `--no-refresh` for the rest), working offline, or when data was already fetched by `refresh.py`.

The refresh logic lives in `api.refresh_tickers()` — a single method shared by all scripts, ready for GUI integration.

## Batch runner (`run_all.py`)

Runs `--compare-periods` for each ticker separately, saves to organized subdirectories:

```bash
uv run python run_all.py --stocks --days 50 --fees 0.03 --stop-loss 2 --buy-hold
```

```
results/
├── stocks_50d_fee003_sl2_bh/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── ...
│   └── _summary.csv          ← best model per ticker
├── crypto_50d_fee015_sl3/
│   └── ...
└── all_20d/
    └── ...
```

Directory name encodes: scope (`stocks`/`crypto`/`all`/`custom`) + days + fees + stop-loss + buy-hold. Different parameter combinations create different directories — no overwriting.

## Interpreting results

### Accuracy vs profit

55% accuracy with PF 2.0 beats 70% accuracy with PF 0.8. The first wins more in dollar terms.

### Stop-loss: when it helps

SL helps most when a model has good accuracy but occasional large losses. It caps the downside. SL hurts when the market is volatile and dips below the stop before recovering — you get stopped out and miss the recovery.

### Models vs buy-and-hold

If no model beats buy-and-hold after fees, active trading doesn't work for that ticker. Common in strong uptrends — buy-and-hold captures the full rally, while models that occasionally short miss parts of it.

### Small sample bias

With 5 days, one prediction swings accuracy by 20%. Use 20+ days. PF ∞ on 5 days = 5 lucky guesses.
