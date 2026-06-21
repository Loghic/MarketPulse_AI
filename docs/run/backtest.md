# Backtests — `backtest.py`, `train.py`, `run_all.py`

Three scripts. `backtest.py` is the interactive workhorse, `train.py`
builds LSTM weights, `run_all.py` is the batch wrapper that hands one
subdirectory per scope back to you.

## `backtest.py` — walk-forward backtests

### Your usual command, unchanged

```bash
uv run python backtest.py --tickers AAPL --days 20 --fees 0.03 --stop-loss 2
```

This still refreshes data + news, then runs each model variant for 20
walk-forward days. The only thing that's changed under the hood: each
"+ News" variant now uses look-ahead-safe per-day sentiment from the
DB.

### Forecasting models (Prophet, Chronos-2, Kronos)

If the `forecast` extra is installed, `Prophet` and `Chronos-2`
variants (plus `+ News`) appear automatically in every backtest — no
flags needed. Add the Kronos sibling clone + `[kronos]` extra (see
[setup.md](setup.md#kronos-sibling-clone-not-pip)) and a `Kronos`
variant joins them. All three are skipped silently when their
dependencies aren't present.

```bash
uv pip install -e ".[forecast]"
uv run python backtest.py --tickers NVDA --days 20 --full
```

The summary table now groups models by family and ranks by return
within each group (a `★` marks the best return). Notes: Prophet refits
on every walk-forward day (one of the slowest models on large
`--days`); Chronos-2 loads once and downloads ~478 MB on first run;
Kronos consumes the full OHLCV window, draws sampled forecast paths,
and is the heaviest per day. See [docs/forecasting.md](../forecasting.md).

**Run controls.** Because the forecasting models can dominate runtime,
use `--models` to include only the families you want and `--periods`
to skip slow windows, and add `--timing` to see exactly where the time
goes:

```bash
# Classifiers + Chronos-2 only, skip 'max', with a timing breakdown
uv run python backtest.py --tickers NVDA --days 20 --compare-periods \
    --periods 1y 2y 5y --models knn linreg lstm chronos --timing
```

In a representative sweep the forecasting models were ~two-thirds of
total wall-clock time, with Kronos and Prophet the slowest and
Chronos-2 cheap — and Chronos-2 was also the strongest of the three on
accuracy and beat-buy-and-hold rate, so `--models … chronos`
(dropping prophet/kronos) is the usual fast default. Full numbers in
[docs/forecasting.md](../forecasting.md).

### Naive baselines

Five trivial "predictors" run alongside the real models by default
(family key `baseline` for `--models` filtering): `Always Long`,
`Previous Day`, `5-Day Momentum`, `20-Day Momentum`, `Random`. They
exist so the real models have to clear a real bar, not just B&H — if
an LSTM doesn't beat Always-Long in a bull market, that's the
finding. Pass `--no-baselines` to skip them, or `--models baseline
lstm` to run only LSTM and the baselines side by side. The
out-of-sample harness ([research.md](research.md#oos-harness)) uses
this family explicitly.

### Confidence gating (`--min-confidence`, `--confidence-sweep`)

Every model emits a per-day **confidence** (the probability of the
direction it chose, always ≥ 0.5). Confidence gating sits out the
low-confidence days instead of trading them — the idea being that a
model might only be predictive when it's sure.

`--min-confidence θ` runs the backtest with a live gate: any day whose
confidence is below `θ` is **flat** (0 P&L, no fee) and is **excluded
from accuracy**. The reported accuracy/return then describe *traded*
days only, and the summary adds a coverage line (`Coverage: 12/30
(40%, θ=0.65)`).

```bash
# Trade only days the model is ≥65% sure about
uv run python backtest.py --stocks --days 100 --min-confidence 0.65 --buy-hold
```

You don't have to pick `θ` blind. `--confidence-sweep` runs **one**
ungated backtest and reports the gate's effect at every threshold in
`config.CONFIDENCE_SWEEP` (`0.0, 0.55, 0.60, 0.65, 0.70`) — coverage,
**traded-day accuracy**, gated return, and fees saved — plus a Brier
score and Expected Calibration Error (ECE) per model so you can see
whether confidence means anything at all:

```bash
uv run python backtest.py --tickers NVDA --days 100 --confidence-sweep
```

```
=================== CONFIDENCE GATING SWEEP ====================
  MODEL                  | θ      | COVERAGE       | TRADED ACC  | RETURN      | FEES SAVED
  --------------------------------------------------------------------------------------
  LSTM                   | 0.00   | 100/100 (100%) | 51.0%       |    -2.1500% |    +0.0000%
                         | 0.55   | 71/100 (71%)   | 52.1%       |    -1.3100% |    +0.0580%
                         | 0.60   | 44/100 (44%)   | 54.5%       |    -0.4200% |    +0.1120%
                         | 0.65   | 18/100 (18%)   | 50.0%       |    -0.3900% |    +0.1640%
                         | 0.70   |  4/100 (4%)    | 50.0%       |    -0.0900% |    +0.1920%
  --------------------------------------------------------------------------------------
```

**How to read it.** The gate adds edge only if **traded-day accuracy
is materially > 0.5 *and* the return improves vs θ=0**. A flat
accuracy column (and a Brier/ECE that doesn't improve with confidence)
means confidence is uncalibrated — gating just shrinks exposure
without finding a predictive subset. The baseline finding (no edge at
any horizon) predicts gating mostly fails; the point is to *measure*
that, not assume it.

`--min-confidence` also works on `run_all.py` (the batch wrapper). The
run-dir name then carries an `mcNNN` segment (e.g.
`stocks_100d_fee003_mc065`) so gated and ungated batches don't
overwrite each other.

### Statistical significance (`--significance`)

A 54% accuracy over 50 days is consistent with a coin flip.
`--significance` puts error bars on every model so you don't read
meaning into noise:

```bash
uv run python backtest.py --tickers NVDA --days 100 --significance
```

It prints, per model: a two-sided **binomial p-value** (H0: accuracy =
0.5), a **Wilson confidence interval** on accuracy, a **bootstrap CI**
on total return (resampled from the daily P&L, so fat tails are
respected), and a **permutation p-value** vs shuffled directions. The
final `FDR✓` column applies a **Benjamini-Hochberg** false-discovery
correction *across the models in the report* — the plan's antidote to
p-hacking a big grid. Read it as: a model is only interesting if its
accuracy CI excludes 0.5, its binomial p survives FDR, **and** its
return CI excludes 0.

Tests run on **traded** days only, so `--significance` composes with
`--min-confidence` (the CI describes the gated subset). Both
`--confidence-sweep` and `--significance` imply the `--full` detail
block.

### Common recipes

All examples below print a summary table to the terminal. Add
`--output PATH.csv` (or `.json`) to also dump structured rows you can
inspect or chart later. Each row carries the ticker, period, model
name, all the metrics, and benchmark returns if `--buy-hold` is set.

#### One-and-done recipes (refresh + backtest + CSV in a single command)

The cleanest "I want to look at this later" pattern is to combine a
deep news refresh with the backtest in one go, then write everything
to CSV.

```bash
# Single ticker, 20-day holdout, FinBERT + 6 months of GDELT, write to CSV
uv run python backtest.py --tickers AAPL --days 20 --fees 0.03 \
    --news-source gdelt --news-history-days 180 \
    --sentiment-method finbert --buy-hold \
    --output results/aapl_finbert.csv

# All stocks, same settings
uv run python backtest.py --stocks --days 20 --fees 0.03 \
    --news-source gdelt --news-history-days 180 \
    --sentiment-method finbert --buy-hold \
    --output results/stocks_finbert.csv

# All crypto, same settings (crypto fees usually higher → 0.10-0.15%)
uv run python backtest.py --crypto --days 20 --fees 0.15 \
    --news-source gdelt --news-history-days 180 \
    --sentiment-method finbert --buy-hold \
    --output results/crypto_finbert.csv

# Both stocks AND crypto in one run (uses 0.05% fees — adjust to taste)
uv run python backtest.py --all --days 20 --fees 0.05 \
    --news-source yahoo gdelt --news-history-days 180 \
    --sentiment-method finbert --buy-hold \
    --output results/all_finbert.csv

# Mixed explicit list (a few stocks plus BTC and ETH)
uv run python backtest.py --tickers AAPL NVDA TSLA BTC-USD ETH-USD \
    --days 20 --fees 0.05 \
    --news-source gdelt --news-history-days 180 \
    --sentiment-method finbert --buy-hold \
    --output results/mixed_finbert.csv
```

After the run, each CSV has one row per `(ticker × model × period)`
with columns like `accuracy`, `total_return`, `profit_factor`,
`max_drawdown`, `sharpe_ratio`, `sortino_ratio`, `buy_hold_return`,
plus `bench_SPY` / `bench_QQQ` / `bench_BTC-USD` if `--buy-hold` was
set.

#### Find the best model + period for a ticker (`--compare-periods`)

The "one-and-done" recipes above use a single period (`max` by
default). When you want to find which model performs best on a given
ticker — and which lookback period that model likes — use
`--compare-periods`. It runs each model on every period in
`ALL_PERIODS` (`1mo`, `1y`, `2y`, `5y`, `max`) and prints a
model × period accuracy + return matrix, the best period per model, a
top-5 leaderboard, a streak analysis and a risk-adjusted ranking.
Narrow the period set with `--periods` (e.g. `--periods 1y 2y 5y`)
when you want to skip the slow `max` window.

Each row of the CSV is keyed `(ticker, model, period)` — sort by
`total_return`, `accuracy`, or `sharpe_ratio` in a spreadsheet to
surface the winner. **Be aware:** picking the best of ~500 backtests
is selection-inflated. Run the OOS harness
([research.md](research.md#oos-harness)) afterwards to see how much of
that "winner" survives on data it didn't get to select on.

```bash
# Single ticker — find best model+period for AAPL, FinBERT-aware
uv run python backtest.py --tickers AAPL --days 20 --fees 0.03 \
    --compare-periods \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --buy-hold \
    --output results/aapl_best_period.csv

# Same for BTC-USD (note higher crypto fees)
uv run python backtest.py --tickers BTC-USD --days 20 --fees 0.15 \
    --compare-periods \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --buy-hold \
    --output results/btc_best_period.csv

# All stocks — sweep every (ticker, model, period) into one CSV
uv run python backtest.py --stocks --days 20 --fees 0.05 \
    --compare-periods \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --buy-hold \
    --output results/stocks_best_period.csv

# All crypto
uv run python backtest.py --crypto --days 20 --fees 0.15 \
    --compare-periods \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --buy-hold \
    --output results/crypto_best_period.csv

# Everything (stocks + crypto) in one go
uv run python backtest.py --all --days 20 --fees 0.05 \
    --compare-periods \
    --news-source yahoo gdelt --news-history-days 365 \
    --sentiment-method finbert --buy-hold \
    --output results/all_best_period.csv
```

If the DB is already populated (you ran a `refresh.py` earlier), add
`--no-refresh` to skip the upfront network step — `--compare-periods`
re-runs five backtests per ticker so the iteration win is noticeable.

Quick pandas snippet to surface the winner per ticker from one of
these CSVs:

```python
import pandas as pd
df = pd.read_csv("results/all_best_period.csv")
# Best (model, period) per ticker by total return
best = df.sort_values("total_return", ascending=False).groupby("ticker").head(1)
print(best[["ticker", "model", "period", "accuracy", "total_return",
            "sharpe_ratio", "buy_hold_return"]])
```

Or — equivalently from the terminal — `run_all.py` does this
batch-style (see [run_all.py](#run_allpy--batch-backtest) below).

#### Other useful patterns

```bash
# 1. Multiple tickers + buy-and-hold benchmark (no CSV)
uv run python backtest.py --stocks --days 20 --fees 0.05 --buy-hold

# 2. Detailed output (per-day consensus, direction accuracy, profit analysis)
uv run python backtest.py --tickers AAPL --days 20 --full --buy-hold

# 3. With stop-loss (each model runs twice — without and with SL — for comparison)
uv run python backtest.py --tickers AAPL --days 20 --fees 0.03 --stop-loss 2 \
    --output results/aapl_sl2.csv

# 4. Cross-period comparison (runs all of 1mo, 1y, 2y, 5y, max)
uv run python backtest.py --tickers AAPL --days 20 --compare-periods --buy-hold \
    --output results/aapl_periods.csv

# 4b. Same, but a faster subset of periods and only some model families
uv run python backtest.py --tickers AAPL --days 20 --compare-periods --buy-hold \
    --periods 1y 2y 5y --models knn linreg lstm chronos --timing \
    --output results/aapl_periods_fast.csv

# 5. Tune the lookback window + decay
uv run python backtest.py --stocks --days 20 \
    --news-lookback 14 --news-half-life 7 \
    --output results/stocks_l14_h7.csv

# 6. Offline (DB-only, skip network refresh) — fast iteration once the DB is populated
uv run python backtest.py --stocks --days 20 --no-refresh --buy-hold \
    --sentiment-method finbert \
    --output results/stocks_offline.csv

# 7. Compare VADER vs FinBERT — same DB, different scorer per run
uv run python backtest.py --stocks --days 20 --no-refresh --buy-hold \
    --sentiment-method vader   --output results/stocks_vader.csv

uv run python backtest.py --stocks --days 20 --no-refresh --buy-hold \
    --sentiment-method finbert --output results/stocks_finbert.csv
```

The `results/` directory is auto-created. CSV vs JSON is decided by
the file extension on `--output` — both formats expose the same
columns.

### Stop-loss (single level or sweep)

`--stop-loss 2` means: if the position drops 2% intraday, exit
immediately at the stop-loss price instead of holding until close.
Uses real High/Low data. With a single value, every model runs twice —
once without SL (baseline) and once with SL — for a side-by-side
comparison.

`--stop-loss` also accepts **several values** to sweep, and `--sl-sweep`
runs a predefined set (`config.SL_SWEEP`, default `0 5 10 15`). Each
model runs once per level; `0` is the no-SL baseline. The `SL{n}%`
suffix on the model name tells the levels apart.

```bash
# Explicit sweep
uv run python backtest.py --tickers NVDA --days 100 --stop-loss 0 5 10 15 --buy-hold
# Same, via the config default set
uv run python backtest.py --tickers NVDA --days 100 --sl-sweep --buy-hold
```

Stop-loss is a risk knob, not an edge: wide/off is usually best for
daily holds (a 10%+ intraday trigger is rare for large-caps), and it's
most informative on volatile names (TSLA, NVDA, crypto). It won't
create an edge that isn't there.

### Trading fees & turnover (`--turnover-fees`, `--hold-days`)

`--fees 0.03` means 0.03% per side (buy + sell = 0.06% round-trip).
Default comes from `config.DEFAULT_TRADING_FEE_PCT`.

By default the backtester charges a full round-trip fee **every traded
day** — i.e. it assumes you close and re-open daily. Over long horizons
that fee churn dominates the result (it's the bulk of the ~30% drag
over 300 days). Two knobs make the cost realistic:

* **`--turnover-fees`** — charge the round-trip fee only on days the
  position *changes* (opens or flips). Same-direction days carry the
  position fee-free, modelling "trade only on signal changes". Raw P&L
  is unchanged; only the fee attribution differs.
* **`--hold-days N`** — once a position is opened, hold it `N` days
  before re-reading the signal (default 1 = re-evaluate daily). The
  model still predicts each day (so **accuracy reflects model skill,
  unchanged**), but the *position* that drives P&L persists through the
  hold window. Most meaningful together with `--turnover-fees`, which
  then skips fees on the held days.

```bash
# Realistic fees: only pay when the signal actually changes
uv run python backtest.py --tickers NVDA --days 100 --turnover-fees --buy-hold

# Hold 5 trading days per signal, fee only on entries
uv run python backtest.py --tickers NVDA --days 100 --turnover-fees --hold-days 5 --buy-hold
```

The summary line then reports turnover: `Turnover: 7/100 (fees
+0.7000%, hold=5d)` — the number of position changes, the fee actually
paid, and the hold window. CSV exports gain `turnover_fees`,
`hold_days`, `turnover_count`, and `fees_paid` columns.

## `train.py` — LSTM training

Requires the `ai` extra installed.

```bash
uv run python train.py --ticker AAPL --period 1y --preset quick
uv run python train.py --stocks --preset standard
uv run python train.py --all --periods 1y 2y max --preset cluster
uv run python train.py --list           # list saved models
```

Presets:

| Preset | Duration | Use |
|---|---|---|
| `quick`    | ~1-5 min       | smoke / dev iteration |
| `standard` | ~5-15 min      | normal use |
| `cluster`  | hours on GPU   | best quality |

Models saved to `models/{ticker}_{period}_{preset}.pt`. The API
auto-loads the best available preset (`cluster > standard > quick`).

## `run_all.py` — batch backtest

Runs `--compare-periods` for each ticker and writes organized
subdirectories under `results/`. Supports the same news / sentiment
flags as `backtest.py` (`--sentiment-method`, `--news-source`,
`--news-lookback`, `--news-half-life`, `--news-history-days`,
`--force-news`) plus `--periods` (restrict the period sweep),
`--models` (restrict model families), `--min-confidence θ` (confidence
gate — see [above](#confidence-gating---min-confidence---confidence-sweep)),
and the turnover / stop-loss knobs (`--turnover-fees`, `--hold-days N`,
multi-value `--stop-loss`, `--sl-sweep`). It also prints a
time-by-model-family rollup at the end of the batch automatically.

```bash
uv run python run_all.py --stocks --days 50 --fees 0.03 --buy-hold
uv run python run_all.py --crypto --days 50 --fees 0.15 --stop-loss 3
uv run python run_all.py --all --days 20

# With FinBERT + 1 year of GDELT history
uv run python run_all.py --stocks --days 20 --fees 0.05 --buy-hold \
    --sentiment-method finbert --news-source gdelt --news-history-days 365

# Fast sweep — skip 'max', drop the heavy forecasting models
uv run python run_all.py --stocks --days 100 --fees 0.05 --buy-hold \
    --periods 1y 2y 5y --models knn linreg lstm chronos

# Stop-loss sweep + realistic turnover fees
uv run python run_all.py --stocks --days 100 --fees 0.05 --buy-hold \
    --sl-sweep --turnover-fees --hold-days 5
```

Output layout:

```
results/
├── stocks_50d_fee003_bh/
│   ├── AAPL.csv
│   ├── MSFT.csv
│   ├── ...
│   └── _summary.csv
├── crypto_50d_fee015_sl3/
│   ├── BTC-USD.csv
│   └── _summary.csv
└── all_20d/
    └── ...
```

Directory name encodes run parameters
(`scope_days_fees_sl_mc_to_hold_bh` — e.g. `stocks_100d_fee003_mc065_bh`
with `--min-confidence 0.65`, or `stocks_100d_fee003_to_hold5` with
`--turnover-fees --hold-days 5`; `to` flags turnover fees, `holdN` the
hold window, and a swept stop-loss tags the max level `slN`), so
different runs don't overwrite each other. (The `--periods` /
`--models` subset is *not* encoded in the directory name, so pick a
distinct scope or move the output if you keep several subset runs side
by side.)

The per-ticker CSVs contain `(model × period)` rows. `_summary.csv`
is a single row per ticker — the model+period combination that
produced the highest return. That summary file is the fastest way to
answer *"what's the best model for ticker X?"* across every ticker in
one shot — but treat the winners as in-sample best, not OOS-honest,
and feed the same setup through [the OOS harness](research.md#oos-harness)
for a non-inflated number.
