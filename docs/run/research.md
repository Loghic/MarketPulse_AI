# Research scripts

Two scripts that turn raw backtest output into honest, paper-ready
numbers:

* [`scripts/news_impact.py`](#news-impact) — quantifies news vs
  no-news on a `run_all.py` result tree.
* [`scripts/oos_harness.py`](#oos-harness) — out-of-sample model-
  selection harness (Plan §1.1). Pick a winner on one window,
  evaluate on the next disjoint one. The honest version of "best per
  ticker".

## News impact

Once `run_all.py` has written its per-ticker CSVs, this script pairs
the `+ News` variants with their price-only siblings and quantifies
the effect for every (ticker, model_family, period) triple. Designed
to feed straight into a paper or poster.

```bash
uv run python scripts/news_impact.py results/stocks_50d_fee003_bh
uv run python scripts/news_impact.py results/crypto_50d_fee015_bh
# Process several directories in one go
uv run python scripts/news_impact.py \
    results/stocks_50d_fee003_bh \
    results/crypto_50d_fee015_bh
```

Three files land in each run directory (prefixed with `_` so they
sort together and don't get reprocessed):

| File | Shape | Use |
|---|---|---|
| `_news_vs_no_news_{TICKER}.csv` | one row per (model_family, period) | per-ticker deep-dive: base / news / delta for accuracy, return, profit factor, max DD, Sharpe, Sortino, plus boolean `*_news_wins` flags |
| `_news_vs_no_news_summary.csv`  | one row per (ticker, model_family) | aggregated across periods — `news_wins_return`, `news_wins_accuracy`, `news_wins_sharpe` counts, plus median and mean deltas, plus `best_period_for_news` / `worst_period_for_news` |
| `_news_vs_no_news_overall.csv`  | one row total | the abstract-paragraph numbers: total pairs, return-wins, accuracy-wins, Sharpe-wins, plus median / mean deltas |

The console digest looks like this (real output from the smoke test):

```
==============================================================================
 News-vs-no-news report — results/stocks_demo
==============================================================================

  Pairs compared:  6
  News beats no-news on:
      Accuracy → 4/6  (67%)
      Return   → 4/6  (67%)
      Sharpe   → 4/6  (67%)

  Median return delta:    +0.9250%
  Mean return delta:      +0.6050%
  Median accuracy delta:  +0.0250

  Top 5 (ticker, model, period) where news helped most:
      AAPL       k-NN Enh. TW           1y      Δreturn=+1.6900%  Δacc=+0.0600
      ...
  Bottom 5 (news hurt most):
      ...
```

### Model pairings used

The script pairs these `+ News` variants with their no-news siblings:

| Price-only baseline | News-aware sibling |
|---|---|
| `k-NN Time-Weighted` | `k-NN TW + News` |
| `k-NN Enh. TW` | `k-NN Enh. TW + News` |
| `LinReg Time-Weighted` | `LinReg TW + News` |
| `LinReg Enh. TW` | `LinReg Enh. TW + News` |
| `LSTM` | `LSTM + News` |

Variants without a `+ News` sibling (plain `k-NN`, `k-NN Enhanced`,
`LinReg`, `LinReg Enhanced`, all baselines) are skipped — baselines
don't use sentiment at all by design. If you've added a new sibling,
update `PAIRS` at the top of `scripts/news_impact.py`.

### Suggested workflow for the paper

```bash
# 1. (One-off) Populate the DB with historical news
uv run python refresh.py --all \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news

# 2. Backtest
uv run python run_all.py --stocks --days 50 --fees 0.03 --buy-hold \
    --no-refresh --sentiment-method finbert
uv run python run_all.py --crypto --days 50 --fees 0.15 --buy-hold \
    --no-refresh --sentiment-method finbert

# 3. Quantify the news effect
uv run python scripts/news_impact.py \
    results/stocks_50d_fee003_bh \
    results/crypto_50d_fee015_bh
```

After step 3 you have, per asset class:
- a one-line **overall** CSV for the abstract,
- a per-ticker × model **summary** CSV for the results table,
- per-ticker drill-down CSVs you can spot-check or chart.

## OOS harness

`scripts/oos_harness.py` is the disciplined alternative to
`run_all.py`'s selection-inflated "best per ticker" summary. Plan §1.1
in the project's [`plan.md`](../../plan.md) lays out why: across ~9
model variants × 5 periods × 11 tickers, `run_all.py` is picking the
max of ~500 backtests every time. Even random predictors will produce
"winners" that look good. The harness fixes that by separating the
window we **select on** from the window we **evaluate on**.

### What it does, per ticker

1. **Selection window** — trim the last `--days` rows off the price
   df; the resulting df's last `--days` are the selection holdout.
2. Run every candidate model × period on that window and pick the
   highest in-sample `total_return`.
3. **Evaluation window** — re-run **only** that exact `(model_name,
   period)` configuration on the full df. Its last `--days` are the
   evaluation holdout — strictly disjoint from selection.
4. Record OOS return, OOS accuracy, beat-B&H flag, and the selection-
   inflation gap (`in_sample − OOS`, median across tickers).

### Usage

```bash
ulimit -n 4096   # macOS prtection againts FD cascade (see reference.md)

# ───── Smoke test (~30s) — baselines only, two tickers ─────
uv run python scripts/oos_harness.py \
    --tickers AAPL BTC-USD --days 50 \
    --models baseline --no-refresh

# ───── Honest stocks sweep (~3-5 min) ─────
# Every model, every period, 50-day disjoint windows, FinBERT news.
uv run python scripts/oos_harness.py \
    --stocks --days 50 --fees 0.03 --buy-hold \
    --no-refresh --sentiment-method finbert

# ───── Crypto (~2 min) — note higher fees ─────
uv run python scripts/oos_harness.py \
    --crypto --days 50 --fees 0.15 --buy-hold \
    --no-refresh --sentiment-method finbert

# ───── "Can a baseline beat the LSTM?" (~3 min) ─────
# Restrict candidates to LSTM + baselines.
uv run python scripts/oos_harness.py \
    --stocks --days 100 \
    --models lstm baseline \
    --no-refresh

# ───── 100-day windows for the paper (~10-15 min) ─────
uv run python scripts/oos_harness.py \
    --stocks --days 100 --fees 0.03 --buy-hold \
    --no-refresh --sentiment-method finbert \
    --periods 1y 2y 5y max
```

The harness shares all of `backtest.py`'s news/sentiment flags
(`--sentiment-method`, `--news-source`, `--news-history-days`,
`--news-lookback-days`, `--news-half-life-days`, `--force-news`) plus
the same scope / period / model selectors documented in the [README
common flags table](README.md#common-flags-across-scripts). It needs
at least `2 × --days + 20` rows of price history per ticker; shorter
tickers are skipped with a log line.

### Output

```
results/oos_<scope>_<days>d_<fee>_<sl>_<bh>_<ts>/
├── _oos_per_ticker.csv     — one row per ticker
├── _oos_summary.csv        — single-row aggregate
└── _oos_console.txt        — copy of the printed banner
```

`_oos_per_ticker.csv` columns:

| Column | What it means |
|---|---|
| `ticker` | symbol |
| `winner_model` | exact variant name picked on the selection window (e.g. `LSTM + News`, `LinReg Time-Weighted`, `Baseline 20-Day Momentum`) |
| `winner_period` | training period the winner used |
| `winner_family` | family key (`lstm`, `linreg`, `baseline`, …) |
| `in_sample_return` | the inflation: how good the winner looked when we picked it |
| `in_sample_accuracy` | accuracy on the selection window |
| `in_sample_buy_hold` | B&H on the selection window for context |
| `oos_return` | the honest number: what the winner did on data it didn't get to select on |
| `oos_accuracy` | OOS accuracy |
| `oos_buy_hold` | B&H on the evaluation window |
| `oos_sharpe` | OOS Sharpe |
| `beats_bh_oos` | 1 if `oos_return > oos_buy_hold`, else 0 |
| `stable` | 1 if `oos_return > 0`, else 0 |

`_oos_summary.csv` is one row with the four headline numbers:

| Column | What it tells you |
|---|---|
| `tickers` | how many tickers produced a complete OOS evaluation |
| `oos_beat_bh_rate` | the honest beat-B&H rate. Plan baseline was 27% / 19% / 2% across 40d / 100d / 300d in-sample runs — this is how much survives OOS |
| `median_oos_return` | typical OOS return per winner |
| `mean_oos_return` | OOS mean (catches the fat tails) |
| `median_in_sample_return` | typical in-sample return per winner — gives context for the gap |
| `in_sample_minus_oos_median` | **selection-inflation gap.** Median(in_sample − OOS) across tickers. Big number = big inflation. |
| `median_oos_accuracy` | OOS direction-call accuracy |

### How to read the numbers

* If `oos_beat_bh_rate ≲ 25%`, the project's
  [falsification criterion](../../plan.md#falsification-criteria)
  about no directional edge is supported.
* If `in_sample_minus_oos_median` is big (≳ 5 percentage points), the
  "best per ticker" picks from `run_all.py` are mostly overfit, not
  edge.
* Look at the `winner_family` column. If `baseline` dominates (e.g.
  Always-Long is the winner for 8/11 stocks), the real models aren't
  beating "hold long in a bull market".

### Pairing with `news_impact.py`

`news_impact.py` answers *"does news help, when news matters?"* and
`oos_harness.py` answers *"does any of this survive OOS?"*. They
operate on different result trees and don't share files. A typical
end-to-end research run does both:

```bash
# 1. Populate the DB (one-off, see refresh.md)
uv run python refresh.py --all \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news

# 2. Headline batch — populates results/stocks_50d_fee003_bh/
uv run python run_all.py --stocks --days 50 --fees 0.03 --buy-hold \
    --no-refresh --sentiment-method finbert

# 3a. News impact on that tree
uv run python scripts/news_impact.py results/stocks_50d_fee003_bh

# 3b. OOS honesty check on the same setup
uv run python scripts/oos_harness.py \
    --stocks --days 50 --fees 0.03 --buy-hold \
    --no-refresh --sentiment-method finbert
```

The OOS harness is the gating check before any Phase-2 experiment in
[`plan.md`](../../plan.md) — turnover, stop-loss, LSTM tuning — is
worth running. If a config has no OOS edge, "tuning" it just selects
on noise.
