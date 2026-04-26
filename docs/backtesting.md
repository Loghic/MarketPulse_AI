# Backtesting

## What is walk-forward testing?

Walk-forward testing simulates real-world trading by training the model only on data that would have been available at each prediction point. No future data leaks into the training set.

For `--days 5`:

```
Day 1:  Train on [day_1 ... day_N-5]  →  Predict day_N-4  →  Compare with reality
Day 2:  Train on [day_1 ... day_N-4]  →  Predict day_N-3  →  Compare with reality
Day 3:  Train on [day_1 ... day_N-3]  →  Predict day_N-2  →  Compare with reality
Day 4:  Train on [day_1 ... day_N-2]  →  Predict day_N-1  →  Compare with reality
Day 5:  Train on [day_1 ... day_N-1]  →  Predict day_N    →  Compare with reality
```

Each step, the model is retrained from scratch with one more day of data. This is expensive (5 full train cycles for 5 test days) but realistic.

## Why not train/test split?

A simple 80/20 split trains once and tests on the last 20% of data. This is faster but problematic:
- The model never sees recent data during training
- It doesn't simulate the real workflow where you retrain daily
- Results are sensitive to where you split

Walk-forward testing is the gold standard for time-series evaluation because it respects the temporal order of data.

## Metrics

### Accuracy

```
accuracy = correct_predictions / total_predictions
```

For `--days 20`: accuracy of 60% means 12/20 days predicted correctly. With a 50% baseline (coin flip), anything consistently above 55% is meaningful.

### Direction accuracy

Breaks down accuracy by predicted direction:

```
UP accuracy:   correct_UP_calls / total_UP_calls
DOWN accuracy: correct_DOWN_calls / total_DOWN_calls
```

Critical for day trading. If a model is 80% accurate on DOWN but 40% on UP, you'd only trade its DOWN signals.

### Confidence calibration

Splits predictions into high-confidence (>65%) and low-confidence (≤65%), then compares accuracy in each bucket. If high-confidence predictions aren't more accurate than low-confidence ones, the confidence score is noise and shouldn't influence your trading decisions.

### Consensus

When running multiple models, consensus shows what percentage of models agree on each day. Unanimous days (100% agreement) are the strongest signals. From backtesting, unanimous predictions are correct more often than split-vote days — but they're not infallible.

### Profit metrics

Accuracy alone doesn't tell you if a model makes money. A model with 60% accuracy that wins big and loses small is more profitable than one with 80% accuracy that wins small and occasionally loses huge. The backtest tracks simulated trades:

- If the model predicts **UP** → simulated Long (buy). P/L = `(close_actual - close_before) / close_before`
- If the model predicts **DOWN** → simulated Short (sell). P/L = `(close_before - close_actual) / close_before`

From this we compute:

**Total return** — sum of all daily P/L. Positive = the model made money over the test period.

**Profit factor** — `gross_profits / gross_losses`. This is the single most important number:
- PF < 1.0 → losing money (losses outweigh wins)
- PF = 1.0 → breakeven
- PF 1.0–1.5 → marginally profitable
- PF 1.5–2.0 → solid strategy
- PF > 2.0 → strong strategy
- PF = ∞ → no losing trades (too good to be true on small samples)

**Average win / average loss** — how big are wins vs losses? If avg_win = +1.5% and avg_loss = -0.5%, the model needs only 25% accuracy to break even.

**W/L** — win/loss count. Together with profit factor, tells you the risk profile: many small wins + few big losses (trend-following) vs few big wins + many small losses (mean-reversion).

### Streak metrics

Streaks measure consecutive wins or losses. E.g. the sequence `[W, W, W, L, L, W, W]` has win streaks of [3, 2] and a loss streak of [2].

**Longest win streak** — maximum consecutive correct predictions. A model with a long win streak is "hot" during certain market conditions. Useful for identifying when the model is in sync with the market regime.

**Longest loss streak** — maximum consecutive wrong predictions. Critical for risk management: if you're trading with real money, you need to survive the worst drawdown. A model with max loss streak of 6 means you'd see 6 losing trades in a row at some point.

**Average win/loss streak** — typical length of a winning or losing run. If avg_win_streak = 1.5 and avg_loss_streak = 3.0, the model tends to get one or two right, then enters extended losing periods — a red flag even if overall accuracy looks decent.

How to use streaks:
- Compare avg_win_streak vs avg_loss_streak. You want avg_win > avg_loss.
- Long max loss streak + high accuracy = the model goes through "regimes" where it works and doesn't. Consider combining with a regime detection mechanism.
- If all models have similar max loss streaks but different avg loss streaks, prefer the one with the lower average — it recovers faster.

## CLI flags

### `--days N`

Number of test days. More days = more reliable accuracy estimate, but also more computation. Recommended values:
- 5: quick check, unreliable (one miss = 20% swing)
- 20: solid for comparison
- 50: most reliable, takes longer to run

### `--period`

How much historical data to train on. Options: `1mo`, `1y`, `2y`, `5y`, `max`.

Shorter periods train only on recent data (more relevant patterns, fewer samples). Longer periods have more samples but include old patterns that may not apply anymore. The optimal period depends on the asset — use `--compare-periods` to find it.

### `--full`

Adds four extra sections to the output:
1. **Day-by-day consensus** — full prediction table for every model × every day
2. **Direction accuracy** — UP vs DOWN accuracy per model
3. **Confidence calibration** — are confident predictions actually better?
4. **Next-day signal** — what each model predicted for the most recent day

### `--compare-periods`

Runs the backtest across all periods (1mo, 1y, 2y, 5y, max) and produces:
- Accuracy matrix: period × model
- Best period per model
- Overall best model + period combination (with ties)
- Top 5 leaderboard

This is the most useful mode for finding the optimal configuration for a specific ticker.

### `--output FILE`

Export results to CSV or JSON (detected from extension). Works in ALL modes — the content matches what you see in the terminal:

| Mode | What each row contains |
|---|---|
| Basic (no flags) | One row per model: accuracy, return, PF, streaks, direction accuracy |
| `--full` | One row per day per model: date, predicted, actual, confidence, trade P/L, prices |
| `--compare-periods` | One row per model × period: accuracy, return, PF, streaks, direction accuracy |

```bash
# Summary per model
uv run python backtest.py --output summary.csv

# Day-by-day detail (for charting or manual review)
uv run python backtest.py --full --output daily.csv

# Cross-period comparison (for finding optimal model + period)
uv run python backtest.py --compare-periods --output comparison.csv
uv run python backtest.py --compare-periods --output comparison.json
```

The full-mode CSV is the most useful for sharing with non-programmers — open it in Excel, filter by model, and sort by trade_pnl to see which days made or lost money.

## Interpreting results

### What's a good accuracy?

On daily stock prediction:
- **50%** — coin flip (random baseline)
- **52-55%** — could be noise or slight edge
- **55-60%** — meaningful if consistent across many test days
- **60%+** — strong signal, verify it's not overfitting

For context: professional quant funds aim for 52-55% accuracy on high-frequency trades and make money through volume and position sizing, not raw accuracy.

### Accuracy vs profit

A model with 55% accuracy but profit factor 2.0 is better than one with 70% accuracy and PF 0.8. Why? The 55% model wins slightly more than half the time, but its wins are twice as large as its losses. The 70% model wins more often, but when it loses, it loses big.

Always check both accuracy AND profit factor. The `--compare-periods` top 5 leaderboard is sorted by total return, not accuracy, for exactly this reason.

### Common pitfalls

- **Small sample bias.** With `--days 5`, one lucky/unlucky day swings accuracy by 20%. Use `--days 20+` for reliable results.
- **Period sensitivity.** A model that scores 80% on `1y` and 40% on `2y` is probably catching a temporary pattern, not a real edge.
- **News contamination.** `+ News` variants use today's sentiment for all backtest days. Their accuracy is less trustworthy than non-news variants.
- **Overfitting to the test set.** If you keep tweaking model parameters to improve backtest results, you're overfitting to the test period. The real test is out-of-sample performance (future days you haven't seen yet).
