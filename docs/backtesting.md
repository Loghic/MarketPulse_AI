# Backtesting

## What is walk-forward testing?

Walk-forward testing simulates real-world trading by training the model only on data that would have been available at each prediction point. No future data leaks into the training set.

For `--days 5`, the model is retrained 5 times, each with one more day of data. Walk-forward is the gold standard for time-series evaluation.

## Trading simulation

Each day, the model's prediction becomes a trade:
- **UP** → Long (buy at open, sell at close). P/L = `(exit - entry) / entry`
- **DOWN** → Short (sell at open, buy at close). P/L = `(entry - exit) / entry`

The exit price is normally the end-of-day close, unless a stop-loss is triggered during the day. A prediction of **FLAT** (only the News-Informed baseline currently does this) sits the day out: no trade, no fee, excluded from accuracy.

By default this is **daily mark-to-market** — each day's close-to-close move is booked as its own trade. `--position-mode` instead holds one position across consecutive same-direction days and books the *compounded* entry→exit return as a single trade. The confidence gate (`--min-confidence θ`) sits out low-confidence days entirely.

### Trading fees

`--fees 0.03` means 0.03% per side. Round-trip = 2 × 0.03% = 0.06% per trade. With 50 trades at 0.1% per side, you lose 10% to fees alone — which is why a daily-flip strategy usually trails buy-and-hold.

Two knobs make the cost realistic:
- `--turnover-fees` — charge the round-trip fee only on days the position *changes* (open / flip), not every day. Same-direction days are held free.
- `--hold-days N` — hold an opened position N days before re-reading the signal.

(`--position-mode` is the strongest form: one fee per held run.)

Typical fee values:
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


## Forecasting models (Prophet, Chronos-2, Kronos)

Three forecasting models join the backtest when the relevant extras are
installed. Unlike the classifiers, they predict a *value* and derive direction
from it — see [forecasting.md](forecasting.md) for the mechanics.

- **Prophet** — fits on every walk-forward day (CPU), so it's the slowest model
  on large `--days` runs. Direction comes from its prediction interval.
  Needs the `forecast` extra.
- **Chronos-2** — a zero-shot foundation model loaded once and reused for every
  ticker; the first run downloads ~478 MB of weights. Needs the `forecast` extra.
- **Kronos** — a decoder-only foundation model for OHLCV candlesticks. Cloned as
  a sibling repo, not pip-installed (`[kronos]` extra + `git clone`); direction
  is the share of sampled forecast paths that close above the last close. It is
  the heaviest model per walk-forward day.

They appear as `Prophet` / `Chronos-2` / `Kronos` plus `+ News` variants,
flowing through the same fee / stop-loss duplication as every other model. A
very confident model (Prophet often sits near 99%) can show identical base and
`+ News` rows — the sentiment nudge isn't large enough to flip the call. That's
expected.

The summary table groups results by model family (k-NN / LinReg / LSTM /
Prophet / Chronos-2 / Kronos) and ranks by return within each group, with a `★`
on the best return and a one-line winner in the footer.

### What we found (100-day stock run)

Across a 100-day FinBERT backtest over every stock and period, mean direction
accuracy sat around 0.49 — essentially a coin flip — and fewer than one run in
five beat its own buy-and-hold benchmark. Headline returns were dominated by a
bull market plus selection bias, not edge. Among the classifiers LSTM was the
strongest family. Among the three forecasting models, Chronos-2 was both the
cheapest and the best (it won outright on several tickers and posted the highest
beat-buy-and-hold rate), while Prophet and Kronos were the weakest *and* the
slowest. The practical takeaway baked into the flags below: drop the slow `max`
period and the Prophet/Kronos families when you want a fast, representative
sweep — `--periods 1y 2y 5y --models knn linreg lstm chronos` cuts a forecasting
run roughly in half with little loss of signal. Full numbers live in
[forecasting.md](forecasting.md).

## CLI flags

| Flag | Description |
|---|---|
| `--days N` | Holdout days (5=quick, 20=solid, 50=reliable) |
| `--period` | Training window for single-period mode: 1mo, 1y, 2y, 5y, max |
| `--periods P [...]` | Restrict which periods `--compare-periods` runs (subset of `ALL_PERIODS`). Skip the slow `max`. Same flag on `run_all.py` |
| `--models F [...]` | Only run these model families: `knn`, `linreg`, `lstm`, `prophet`, `chronos`, `kronos`, `baseline` (default: all). Same flag on `run_all.py` |
| `--no-baselines` | Skip the naive + news-aware baselines |
| `--fees FLOAT` | Fee % per side (default from config.py) |
| `--turnover-fees` | Charge the round-trip fee only on position changes, not every day |
| `--hold-days N` | Hold an opened position N days before re-reading the signal |
| `--position-mode` | Compound consecutive same-direction days into one trade (one fee per held run) |
| `--stop-loss FLOAT [...]` | Stop-loss % (0=disabled). Single value runs each model twice; pass several to sweep |
| `--sl-sweep` | Sweep the default stop-loss set (`config.SL_SWEEP`) |
| `--min-confidence θ` | Confidence gate: sit out days below θ confidence (excluded from accuracy) |
| `--confidence-sweep` | θ-sweep table (coverage / traded accuracy / return / fees) + Brier/ECE |
| `--significance` | Binomial p + Wilson CI + bootstrap CI + Benjamini-Hochberg FDR |
| `--buy-hold` | Add buy-and-hold return to output |
| `--no-refresh` | Skip data download, use cached DB only (offline mode) |
| `--full` | Detailed: consensus, direction accuracy, profit analysis, streaks |
| `--compare-periods` | All periods (or the `--periods` subset), accuracy matrix, top 5, streak analysis |
| `--timing` | Print a slowest-first per-model compute-time table after the summary |
| `--output FILE` | Export CSV or JSON |

### `--output` content by mode

| Mode | CSV rows |
|---|---|
| Basic | 1 per model (accuracy, return, PF, fees, SL, B&H, streaks) |
| `--full` | 1 per day per model (date, predicted, actual, pnl, stopped_out) |
| `--compare-periods` | 1 per model × period |

With `--stop-loss`, row count doubles (baseline + SL variant for each model).

### Timing breakdown

`--timing` adds a per-model wall-clock table (slowest first) after the normal
summary, sourced from each `BacktestResult.elapsed_seconds`. When a run spans
multiple periods it also shows a per-run average. This is how the forecasting
overhead was measured: on a representative run the forecasting models accounted
for roughly two-thirds of total wall time, with Kronos and Prophet by far the
slowest and Chronos-2 cheap by comparison. `run_all.py` prints a
**time-by-model-family** rollup (time, share, wins) at the end of the batch
automatically — no flag needed.

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

# Skip slow 'max', drop the heavy forecasting models
uv run python run_all.py --stocks --days 100 --periods 1y 2y 5y --models knn linreg lstm chronos
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

Directory name encodes: scope (`stocks`/`crypto`/`all`/`custom`) + days + fees + stop-loss + buy-hold. Different parameter combinations create different directories — no overwriting. (The `--periods` / `--models` subset isn't encoded in the directory name, so name your output deliberately if you run several subsets.)

## Interpreting results

### Accuracy vs profit

55% accuracy with PF 2.0 beats 70% accuracy with PF 0.8. The first wins more in dollar terms.

### Stop-loss: when it helps

SL helps most when a model has good accuracy but occasional large losses. It caps the downside. SL hurts when the market is volatile and dips below the stop before recovering — you get stopped out and miss the recovery.

### Models vs buy-and-hold

If no model beats buy-and-hold after fees, active trading doesn't work for that ticker. Common in strong uptrends — buy-and-hold captures the full rally, while models that occasionally short miss parts of it. In our own sweeps this was the norm rather than the exception: most (model, ticker, period) combinations failed to beat their own buy-and-hold, so treat any single headline return with suspicion until it survives a multi-period, multi-ticker check.

### Small sample bias

With 5 days, one prediction swings accuracy by 20%. Use 20+ days. PF ∞ on 5 days = 5 lucky guesses.

## Verification

The backtester has 30 dedicated tests in `tests/test_backtester.py`:

```bash
uv run python -m pytest tests/test_backtester.py -v
```

Tests cover: fee math (round-trip = 2x per-side), stop-loss triggers on High/Low, max drawdown from known sequences, Sharpe/Sortino edge cases (all wins, few samples), streak calculation, yearly breakdown, B&H return correctness, and gross profit/loss sum.
