# Running MarketPulse AI

Everything you need to install, configure, and run MarketPulse AI — in one
place. The top-level `README.md` is intentionally short and points here.

If something is missing, the most likely place to look next is one of the
focused docs:

* [docs/api.md](api.md) — architecture, DB schema, model contract
* [docs/backtesting.md](backtesting.md) — walk-forward methodology, all metrics
* [docs/forecasting.md](forecasting.md) — Prophet, Chronos-2, Kronos + the ForecastModel interface
* [docs/sentiment.md](sentiment.md) — VADER vs FinBERT vs naive
* [docs/news_sources.md](news_sources.md) — Yahoo, GDELT, adding new providers
* [docs/features.md](features.md), [docs/k-NN.md](k-NN.md),
  [docs/linear_regression.md](linear_regression.md), [docs/lstm.md](lstm.md) — per-component
* [docs/web.md](web.md) — full Web GUI / REST API docs

---

## 1. Prerequisites

* **Python 3.12+** (the project pins this in `pyproject.toml`)
* **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python
  package manager. All commands below use `uv run`, which auto-creates and
  activates the venv as needed.
* **macOS / Linux / WSL.** Bare Windows works but isn't CI-tested.
* For the Web GUI you also need **Node.js 18+** (only if you plan to run the
  React frontend).

Clone and `cd` into the project:

```bash
git clone <repo-url>
cd marketpulse-ai
uv venv
```

---

## 2. Install

The project has one required dependency set and several optional extras. Install
only what you need.

### Core (always required)

```bash
uv pip install -e .
```

This installs `pandas`, `yfinance`, `scikit-learn`, `numpy`, `nltk`, `tqdm`.
After this, k-NN and Linear Regression models plus VADER + naive sentiment all
work, along with Yahoo + GDELT news sources.

### LSTM + FinBERT (the `ai` extra)

```bash
uv pip install -e ".[ai]"
```

Adds `torch>=2.0` and `transformers>=4.40`. Required if you want to:

* train and run the LSTM model (`train.py`, the `lstm` model variant)
* use FinBERT as a sentiment scorer (`--sentiment-method finbert`)

This is a large download (~1 GB including PyTorch wheels). On Linux without a
GPU, install the CPU-only PyTorch first if you want to save space:

```bash
uv pip install "torch==2.*+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[ai]"
```

The first FinBERT call downloads the ProsusAI/finbert model (~400 MB) and
caches it under `~/.cache/huggingface/`. Subsequent runs are instant.

### Forecasting models (the `forecast` extra)

```bash
uv pip install -e ".[forecast]"
```

Adds `prophet` and `chronos-forecasting`. Required for the Prophet and
Chronos-2 forecasting models in backtests. Prophet needs no download; Chronos-2
downloads its weights (~478 MB) from the Hugging Face Hub on first use and
caches them under `~/.cache/huggingface/`. CPU works out of the box.

### Kronos (sibling clone, not pip)

Kronos isn't on PyPI — clone it next to the repo and install the minimal extra:

```bash
git clone https://github.com/shiyu-coder/Kronos.git ../Kronos
uv pip install -e ".[kronos]"        # einops + huggingface_hub (torch already present)
```

Do **not** run `pip install -r ../Kronos/requirements.txt` — on Python 3.14 its
pinned matplotlib 3.9.3 has no wheel and the bundled freetype source build
fails. The adapter doesn't need matplotlib. The clone is expected as a sibling
of this repo (`../Kronos`); override the location with `KRONOS_PATH` if it lives
elsewhere. The first backtest downloads the Kronos-small weights from the Hub
and forces CPU on Macs (the upstream default targets `cuda:0`).

### Web GUI (the `web` extra)

```bash
uv pip install -e ".[web]"
cd web/frontend && npm install && cd ../..
```

Adds `fastapi`, `uvicorn`. The `npm install` step pulls React, Vite, TanStack
Query and Plotly.

### Plotting helpers (the `viz` extra)

```bash
uv pip install -e ".[viz]"
```

Adds `plotly` and `matplotlib`. Optional — only needed if you want to render
charts from a Python script outside the Web GUI.

### Dev tooling (the `dev` extra)

```bash
uv pip install -e ".[dev]"
uv run pre-commit install
```

Adds `pytest`, `pytest-cov`, `ruff`, `mypy`, `httpx`, `pre-commit`. Required to
run the test suite and the pre-commit hooks (ruff + mypy on every commit).

### Everything at once

```bash
uv pip install -e ".[ai,web,viz,dev,forecast]"
git clone https://github.com/shiyu-coder/Kronos.git ../Kronos   # Kronos is a separate clone
uv pip install -e ".[kronos]"
cd web/frontend && npm install && cd ../..
uv run pre-commit install
```

---

## 3. First-time setup (one-off)

A few network-dependent assets get pulled lazily on first use. If you're on a
network with strict egress rules, do these explicitly so they fail loudly
rather than during a backtest run.

### VADER lexicon (~1 MB)

The first VADER call auto-downloads NLTK's `vader_lexicon`. Force it now:

```bash
uv run python -c "import nltk; nltk.download('vader_lexicon')"
```

### FinBERT model (~400 MB, only if you installed the `ai` extra)

```bash
uv run python -c "from engine.sentiment import FinBERTScorer; FinBERTScorer()"
```

### Chronos-2 weights (~478 MB, only if you installed the `forecast` extra)

```bash
uv run python -c "from engine.chronos_model import Chronos2Model; Chronos2Model()._load()"
```

### Kronos weights (only if you cloned Kronos + installed the `kronos` extra)

```bash
uv run python -c "from engine.kronos_model import KronosModel; KronosModel()._load()"
```

This pulls the Kronos-small tokenizer + model from the Hugging Face Hub. If it
can't find the repo, check that `../Kronos` exists (or `KRONOS_PATH` points at
the clone).

### GDELT reachability check (optional)

GDELT's free Doc API is over `https://api.gdeltproject.org`. If your network
blocks it the provider returns an empty list silently. Verify reachability:

```bash
curl -sI "https://api.gdeltproject.org/api/v2/doc/doc?query=Apple&format=JSON&mode=ArtList&maxrecords=1" | head -3
```

You should see a `200 OK`.

---

## 4. Tickers

Defined in `config.py` as a data-driven asset registry (`ASSET_CLASSES`). Add a
ticker by adding it to the relevant class's `tickers`; add a whole asset class by
appending one `AssetClass` entry — flags, benchmarks, asset-type tags and GDELT
news queries all derive from it. No other code changes needed.

**Stocks:** AAPL, MSFT, NVDA, META, GOOGL, AMD, TSM, ASML, AVGO, TSLA, INTC
**Crypto:** BTC-USD, ETH-USD, SOL-USD, BNB-USD
**Commodities:** GLD (gold ETF)
**Indices:** VOO (S&P 500), QQQM (Nasdaq-100)
**FX:** FXE (EUR/USD)

The non-stock/crypto classes use ETF proxies (volume-bearing, so features + LSTM
behave as for stocks). `VOO`/`QQQM` are deliberately distinct from the `SPY`/`QQQ`
benchmark set.

Every CLI script accepts the same scope flags, and the per-class flags **combine**:

```bash
--stocks                  # only stocks
--crypto                  # only crypto
--commodities             # only commodities (GLD)
--indices                 # only indices (VOO, QQQM)
--fx                      # only FX (FXE)
--commodities --fx        # combine classes (union: GLD + FXE)
--all                     # every class
--tickers AAPL NVDA GLD   # explicit list
```
---

## 5. The five CLI scripts

| Script | What it does |
|---|---|
| `main.py`     | Next-day prediction report for each model variant |
| `refresh.py`  | Pre-fetch prices + news into the DB (no models) |
| `backtest.py` | Walk-forward backtest with metrics and CSV/JSON export |
| `train.py`    | Train LSTM models for future use |
| `run_all.py`  | Batch wrapper that runs `--compare-periods` per ticker into subdirectories |
| `scripts/news_impact.py` | Post-processes a `run_all.py` result tree to quantify news-vs-no-news per ticker, model, and overall |
| `scripts/clean_prices.py` | Reports/removes rows with NULL or non-positive close in `data/market_data.db`, plus flags suspicious adjacent-day moves. Run this if you ever see absurd `total_return` numbers (>200% in days). |

All five auto-refresh data on startup unless you pass `--no-refresh`. They all
share the news/sentiment flags introduced in the 2026 refactor.

### Common flags across scripts

| Flag | Default | Purpose |
|---|---|---|
| `--no-refresh` | off | Skip data download (offline mode) |
| `--sentiment-method {vader,finbert,naive}` | `vader` | Which scorer to use |
| `--news-source {yahoo,gdelt} [...]` | `yahoo` | Provider(s); list = combined+dedup |
| `--news-history-days N` | — | Pull N days of news during refresh (bulk fetch) |
| `--force-news` | off | Bypass "already fetched today" cache |

`backtest.py` adds two more:

| Flag | Default | Purpose |
|---|---|---|
| `--news-lookback N` | 7 | Per-day window: only count news from last N days |
| `--news-half-life H` | 3 | Exponential decay half-life (0 = uniform weighting) |

`backtest.py` and `run_all.py` also accept these run-shaping flags (handy now
that the forecasting models make a full sweep slow):

| Flag | Default | Purpose |
|---|---|---|
| `--periods P [...]` | all | Restrict the period set in `--compare-periods` (subset of `ALL_PERIODS`). Skip the slow `max` window. Same flag on `run_all.py`. |
| `--models F [...]` | all | Only run these model families: `knn`, `linreg`, `lstm`, `prophet`, `chronos`, `kronos`. Maps SL / `+ News` / time-weighted variants to the right family. Same flag on `run_all.py`. |
| `--timing` *(backtest.py)* | off | Print a slowest-first per-model compute-time table after the summary. `run_all.py` prints a time-by-model-family rollup automatically. |

```bash
# Fast representative sweep — drop slow 'max' and the heavy Prophet/Kronos models,
# and see where the time goes
uv run python backtest.py --tickers NVDA --compare-periods \
    --periods 1y 2y 5y --models knn linreg lstm chronos --timing
```

---

## 6. `main.py` — predictions

```bash
# Default: a curated subset of tickers, all models, current prices
uv run python main.py

# Pick scope
uv run python main.py --stocks
uv run python main.py --crypto
uv run python main.py --tickers AAPL NVDA BTC-USD

# Pick scorer / news source
uv run python main.py --stocks --sentiment-method finbert
```

Prints a per-ticker table with each model's UP/DOWN call, confidence, and the
sentiment-adjusted version. No DB writes other than the news/price caches.

---

## 7. `refresh.py` — populate the DB

This is the script you use when you want to **pre-load historical news** (for
the article-style backtests) without running any models.

### Daily refresh (same as before — no flags needed)

```bash
uv run python refresh.py              # all tickers, Yahoo, VADER
uv run python refresh.py --stocks
uv run python refresh.py --crypto
uv run python refresh.py --tickers AAPL NVDA BTC-USD
```

This pulls prices for each ticker, fetches a handful of recent headlines from
Yahoo, scores them with VADER, and stores each row with its real
`published_at` date.

### Historical bulk fetch (new in 2026)

The shape is always:

```bash
uv run python refresh.py <SCOPE> \
    --news-source <SOURCE...> \
    --news-history-days <N> \
    --sentiment-method <SCORER> \
    --force-news
```

`<SCOPE>` is one of `--stocks`, `--crypto`, `--all`, or `--tickers AAPL NVDA …`.

#### Single ticker — 6 months of GDELT, scored with FinBERT

```bash
uv run python refresh.py --tickers AAPL \
    --news-source gdelt --news-history-days 180 \
    --sentiment-method finbert --force-news
```

#### All stocks — 1 year of GDELT, FinBERT

```bash
uv run python refresh.py --stocks \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news
```

#### All crypto — 1 year of GDELT, FinBERT

```bash
uv run python refresh.py --crypto \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news
```

Crypto tickers (`BTC-USD`, `ETH-USD`, `SOL-USD`) are handled the same way as
stocks. The `-USD` suffix is automatically stripped when building the GDELT
search query, so `BTC-USD` becomes `"Bitcoin"`, `ETH-USD` → `"Ethereum"`,
`SOL-USD` → `"Solana"` (the GDELT query map lives in `config.py`'s asset registry as TICKER_NAMER, re-exported by engine/news_sources)

#### Stocks + crypto in one shot

```bash
# All 19 tickers (every asset class), Yahoo + GDELT combined and deduped
uv run python refresh.py --all \
    --news-source yahoo gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news
```

#### Explicit ticker list (mix of stocks and crypto)

```bash
uv run python refresh.py --tickers AAPL NVDA TSLA BTC-USD ETH-USD \
    --news-source gdelt --news-history-days 180 \
    --sentiment-method finbert --force-news
```

#### Score the same news a second time with a different scorer

VADER and FinBERT rows can live side-by-side in the DB (distinguished by the
`method` column). Re-run with `--sentiment-method vader` and `--force-news`
to add a second set of scores, which lets you A/B both scorers on the same
historical headlines:

```bash
uv run python refresh.py --all \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method vader --force-news
```

* `--news-history-days 365` tells the provider to pull a year of headlines.
  Yahoo silently caps at ~3-7 days; GDELT honours larger values up to 250
  articles per call.
* `--force-news` bypasses the same-day cache, which would otherwise
  short-circuit a re-fetch.

Verify what landed:

```bash
uv run python -c "
from engine.db_manager import DatabaseManager
db = DatabaseManager()
df = db.get_news('AAPL')
print(df[['published_at','source','method','sentiment_score','headline']].head(10))
print(f'Total rows for AAPL: {len(df)}')
print(f'Date range: {df[\"published_at\"].min()} → {df[\"published_at\"].max()}')
"
```

---

## 8. `backtest.py` — walk-forward backtests

### Your usual command, unchanged

```bash
uv run python backtest.py --tickers AAPL --days 20 --fees 0.03 --stop-loss 2
```

This still refreshes data + news, then runs each model variant for 20 walk-
forward days. The only thing that's changed under the hood: each "+ News"
variant now uses look-ahead-safe per-day sentiment from the DB.

### Forecasting models (Prophet, Chronos-2, Kronos)

If the `forecast` extra is installed, `Prophet` and `Chronos-2` variants (plus
`+ News`) appear automatically in every backtest — no flags needed. Add the
Kronos sibling clone + `[kronos]` extra (see §2) and a `Kronos` variant joins
them. All three are skipped silently when their dependencies aren't present.

```bash
uv pip install -e ".[forecast]"
uv run python backtest.py --tickers NVDA --days 20 --full
```

The summary table now groups models by family and ranks by return within each
group (a `★` marks the best return). Notes: Prophet refits on every
walk-forward day (one of the slowest models on large `--days`); Chronos-2 loads
once and downloads ~478 MB on first run; Kronos consumes the full OHLCV window,
draws sampled forecast paths, and is the heaviest per day. See
[docs/forecasting.md](forecasting.md).

**Run controls.** Because the forecasting models can dominate runtime, use
`--models` to include only the families you want and `--periods` to skip slow
windows, and add `--timing` to see exactly where the time goes:

```bash
# Classifiers + Chronos-2 only, skip 'max', with a timing breakdown
uv run python backtest.py --tickers NVDA --days 20 --compare-periods \
    --periods 1y 2y 5y --models knn linreg lstm chronos --timing
```

In a representative sweep the forecasting models were ~two-thirds of total
wall-clock time, with Kronos and Prophet the slowest and Chronos-2 cheap — and
Chronos-2 was also the strongest of the three on accuracy and beat-buy-and-hold
rate, so `--models … chronos` (dropping prophet/kronos) is the usual fast
default. Full numbers in [docs/forecasting.md](forecasting.md).

### Common recipes

All examples below print a summary table to the terminal. Add `--output
PATH.csv` (or `.json`) to also dump structured rows you can inspect or chart
later. Each row carries the ticker, period, model name, all the metrics, and
benchmark returns if `--buy-hold` is set.

#### One-and-done recipes (refresh + backtest + CSV in a single command)

The cleanest "I want to look at this later" pattern is to combine a deep
news refresh with the backtest in one go, then write everything to CSV.

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

After the run, each CSV has one row per `(ticker × model × period)` with
columns like `accuracy`, `total_return`, `profit_factor`, `max_drawdown`,
`sharpe_ratio`, `sortino_ratio`, `buy_hold_return`, plus `bench_SPY` /
`bench_QQQ` / `bench_BTC-USD` if `--buy-hold` was set.

#### Find the best model + period for a ticker (`--compare-periods`)

The "one-and-done" recipes above use a single period (`max` by default).
When you want to find which model performs best on a given ticker — and
which lookback period that model likes — use `--compare-periods`. It runs
each model on every period in `ALL_PERIODS` (`1mo`, `1y`, `2y`, `5y`, `max`)
and prints a model × period accuracy + return matrix, the best period per
model, a top-5 leaderboard, a streak analysis and a risk-adjusted ranking.
Narrow the period set with `--periods` (e.g. `--periods 1y 2y 5y`) when you
want to skip the slow `max` window.

Each row of the CSV is keyed `(ticker, model, period)` — sort by
`total_return`, `accuracy`, or `sharpe_ratio` in a spreadsheet to surface
the winner.

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

Quick pandas snippet to surface the winner per ticker from one of these
CSVs:

```python
import pandas as pd
df = pd.read_csv("results/all_best_period.csv")
# Best (model, period) per ticker by total return
best = df.sort_values("total_return", ascending=False).groupby("ticker").head(1)
print(best[["ticker", "model", "period", "accuracy", "total_return",
            "sharpe_ratio", "buy_hold_return"]])
```

Or — equivalently from the terminal — `run_all.py` does this batch-style and
writes one CSV per ticker plus a `_summary.csv`. It accepts the same news /
sentiment flags as `backtest.py`:

```bash
# Simple — defaults from config.py
uv run python run_all.py --stocks --days 20 --fees 0.05 --buy-hold
uv run python run_all.py --crypto --days 20 --fees 0.15 --buy-hold
uv run python run_all.py --all    --days 20 --fees 0.05 --buy-hold

# With FinBERT + deep GDELT history in one shot
uv run python run_all.py --stocks --days 20 --fees 0.05 --buy-hold \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert

uv run python run_all.py --crypto --days 20 --fees 0.15 --buy-hold \
    --news-source gdelt --news-history-days 365 \
    --sentiment-method finbert

uv run python run_all.py --all --days 20 --fees 0.05 --buy-hold \
    --news-source yahoo gdelt --news-history-days 365 \
    --sentiment-method finbert

# Output trees:
#   results/stocks_20d_fee005_bh/{AAPL,MSFT,…,_summary}.csv
#   results/crypto_20d_fee015_bh/{BTC-USD,ETH-USD,SOL-USD,_summary}.csv
#   results/all_20d_fee005_bh/{AAPL,…,BTC-USD,…,_summary}.csv
```

The per-ticker CSVs contain `(model × period)` rows. `_summary.csv` is a
single row per ticker — the model+period combination that produced the
highest return. That summary file is the fastest way to answer
*"what's the best model for ticker X?"* across all 11 stocks + 3 crypto in
one shot.

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

The `results/` directory is auto-created. CSV vs JSON is decided by the file
extension on `--output` — both formats expose the same columns.

### Stop-loss

`--stop-loss 2` means: if the position drops 2% intraday, exit immediately at
the stop-loss price instead of holding until close. Uses real High/Low data.

When enabled, every model runs twice — once without SL (baseline) and once
with SL — so you see a direct side-by-side comparison.

### Trading fees

`--fees 0.03` means 0.03% per side (buy + sell = 0.06% round-trip). Default
comes from `config.DEFAULT_TRADING_FEE_PCT`.

---

## 9. `train.py` — LSTM training

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

Models saved to `models/{ticker}_{period}_{preset}.pt`. The API auto-loads
the best available preset (`cluster > standard > quick`).

---

## 10. `run_all.py` — batch backtest

Runs `--compare-periods` for each ticker and writes organized subdirectories
under `results/`. Supports the same news / sentiment flags as `backtest.py`
(`--sentiment-method`, `--news-source`, `--news-lookback`, `--news-half-life`,
`--news-history-days`, `--force-news`) plus `--periods` (restrict the period
sweep) and `--models` (restrict model families). It also prints a
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

Directory name encodes run parameters (`scope_days_fees_sl_bh`), so different
runs don't overwrite each other. (The `--periods` / `--models` subset is *not*
encoded in the directory name, so pick a distinct scope or move the output if
you keep several subset runs side by side.)

---

## 10b. `scripts/news_impact.py` — news vs no-news analysis

Once `run_all.py` has written its per-ticker CSVs, this script pairs the
`+ News` variants with their price-only siblings and quantifies the effect
for every (ticker, model_family, period) triple. Designed to feed straight
into a paper or poster.

```bash
uv run python scripts/news_impact.py results/stocks_50d_fee003_bh
uv run python scripts/news_impact.py results/crypto_50d_fee015_bh
# Process several directories in one go
uv run python scripts/news_impact.py \
    results/stocks_50d_fee003_bh \
    results/crypto_50d_fee015_bh
```

Three files land in each run directory (prefixed with `_` so they sort
together and don't get reprocessed):

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
`LinReg`, `LinReg Enhanced`) are skipped. If you've added a new sibling,
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

## 11. Web GUI

```bash
# One-time install (after `uv pip install -e ".[web]"`)
cd web/frontend && npm install && cd ../..
chmod +x web/dev.sh

# Start both servers
./web/dev.sh

# Or manually in two terminals:
# Terminal 1: uv run uvicorn web.backend.app:app --reload --port 8000
# Terminal 2: cd web/frontend && npm run dev
```

* Frontend: <http://localhost:5173>
* Backend API: <http://localhost:8000>
* Swagger docs: <http://localhost:8000/docs>

See [docs/web.md](web.md) for the full API surface and per-page documentation.

---

## 12. Suggested workflows

### Daily (just want a prediction)

```bash
uv run python main.py --stocks
```

That auto-refreshes data and prints today's prediction table. Done.

### Daily (want to backtest yesterday)

```bash
uv run python backtest.py --stocks --days 1 --buy-hold
```

### Researcher mode (deep historical analysis for the article)

```bash
# ----------------------------------------------------------------------
# 1. One-off: populate DB with 1 year of GDELT for EVERYTHING (stocks + crypto).
#    Score the same headlines with both FinBERT and VADER so the comparison
#    can be done offline afterwards.
# ----------------------------------------------------------------------
uv run python refresh.py --all \
    --news-source yahoo gdelt --news-history-days 365 \
    --sentiment-method finbert --force-news

uv run python refresh.py --all \
    --news-source yahoo gdelt --news-history-days 365 \
    --sentiment-method vader --force-news

# ----------------------------------------------------------------------
# 2. Headline runs — stocks and crypto each get their own CSV.
#    `--no-refresh` keeps everything offline (the DB now has all the news).
# ----------------------------------------------------------------------
uv run python backtest.py --stocks --days 20 --buy-hold --no-refresh \
    --sentiment-method finbert --output results/stocks_finbert_20d.csv

uv run python backtest.py --crypto --days 20 --fees 0.15 --buy-hold --no-refresh \
    --sentiment-method finbert --output results/crypto_finbert_20d.csv

uv run python backtest.py --stocks --days 20 --buy-hold --no-refresh \
    --sentiment-method vader --output results/stocks_vader_20d.csv

uv run python backtest.py --crypto --days 20 --fees 0.15 --buy-hold --no-refresh \
    --sentiment-method vader --output results/crypto_vader_20d.csv

# ----------------------------------------------------------------------
# 3. Sensitivity sweep — same FinBERT scorer, different lookback windows.
#    One CSV per setting so you can chart "accuracy vs window length".
# ----------------------------------------------------------------------
for L in 3 7 30; do
    uv run python backtest.py --stocks --days 20 --buy-hold --no-refresh \
        --sentiment-method finbert --news-lookback $L \
        --output results/stocks_finbert_l${L}.csv
    uv run python backtest.py --crypto --days 20 --fees 0.15 --buy-hold --no-refresh \
        --sentiment-method finbert --news-lookback $L \
        --output results/crypto_finbert_l${L}.csv
done

# ----------------------------------------------------------------------
# 4. Single-ticker spotlight (useful when you want a clean focused chart
#    of one asset for the poster — e.g. the asset with the biggest news lift)
# ----------------------------------------------------------------------
uv run python backtest.py --tickers AAPL --days 30 --full --buy-hold --no-refresh \
    --sentiment-method finbert --output results/aapl_spotlight.csv

uv run python backtest.py --tickers BTC-USD --days 30 --full --fees 0.15 --buy-hold --no-refresh \
    --sentiment-method finbert --output results/btc_spotlight.csv
```

Each CSV is self-describing — the `ticker`, `period`, `model`, `accuracy`,
`total_return`, `profit_factor`, `max_drawdown`, `sharpe_ratio`,
`buy_hold_return` and `bench_*` columns are enough to slice the results
later in pandas or a spreadsheet.

---

## 13. Configuration (`config.py`)

The bits you'll most likely tweak:

```python
# Asset registry — one AssetClass per class. STOCKS/CRYPTO/COMMODITIES/INDICES/FX,
# ALL_TICKERS, STOCK_BENCHMARKS/CRYPTO_BENCHMARKS/ALL_BENCHMARKS, TICKER_NAMES and
# get_benchmarks() all DERIVE from this list.
ASSET_CLASSES = [
    AssetClass("stock",     "Stocks",      "stocks",      [...11 stocks...],                          ["SPY", "QQQ"]),
    AssetClass("crypto",    "Crypto",      "crypto",      ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"], ["BTC-USD"]),
    AssetClass("commodity", "Commodities", "commodities", ["GLD"],          ["SPY"]),       # gold ETF proxy
    AssetClass("index",     "Indices",     "indices",     ["VOO", "QQQM"],  ["SPY", "QQQ"]),# S&P 500 / Nasdaq-100 ETF proxies
    AssetClass("fx",        "FX",          "fx",          ["FXE"],          ["SPY"]),       # EUR/USD ETF proxy
]

# Periods + defaults
ALL_PERIODS = ["1mo", "1y", "2y", "5y", "max"]
DEFAULT_PERIOD = "max"
DEFAULT_BACKTEST_DAYS = 5

# Trading
DEFAULT_TRADING_FEE_PCT = 0.05  # per side
DEFAULT_STOP_LOSS_PCT   = 0.0   # 0 = disabled

# News + sentiment
DEFAULT_SENTIMENT_METHOD    = "vader"    # "vader" | "finbert" | "naive"
DEFAULT_NEWS_SOURCES        = ["yahoo"]  # ["yahoo", "gdelt"] for combined
DEFAULT_NEWS_LOOKBACK_DAYS  = 7          # 0 = unbounded
DEFAULT_NEWS_HALF_LIFE_DAYS = 3.0        # 0 = no decay

# Forecasting models (Prophet, Chronos-2, Kronos) — backtests
FORECAST_MODELS = [
    ("prophet", "Prophet"),
    ("chronos", "Chronos-2"),
    ("kronos",  "Kronos"),
]
FORECAST_DEVICE = None        # None = auto (cuda if available else cpu)
CHRONOS_MODEL_ID = "amazon/chronos-2"
CHRONOS_CONTEXT = 512         # most-recent closes used as context

# Kronos (sibling clone, OHLCV candlestick model)
KRONOS_PATH         = "../Kronos"          # clone location (override if not a sibling)
KRONOS_MODEL_ID     = "NeoQuasar/Kronos-small"
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
KRONOS_MAX_CONTEXT  = 512     # candlesticks of context
KRONOS_SAMPLE_COUNT = 5       # forecast paths sampled to estimate prob_up
KRONOS_PROB_SAMPLES = 1
KRONOS_T            = 1.0     # sampling temperature
KRONOS_TOP_P        = 0.9     # nucleus sampling cutoff

# Model-family labels — the single source the `--models` filter uses
# (key → display name). MODEL_FAMILIES is derived from it.
MODEL_FAMILY_LABELS = {
    "knn":     "k-NN",
    "linreg":  "LinReg",
    "lstm":    "LSTM",
    "prophet": "Prophet",
    "chronos": "Chronos-2",
    "kronos":  "Kronos",
}
MODEL_FAMILIES = list(MODEL_FAMILY_LABELS)
```

Every CLI flag has a `config.py` default. Changing the default avoids having
to pass the flag every time.

---

## 14. Testing

```bash
# Quick smoke test, no extra deps (13 tests)
uv run python test_pipeline.py

# Full pytest suite (~120 tests, needs the `dev` extra)
uv run python -m pytest

# A specific module
uv run python -m pytest tests/test_news_pipeline.py -v
```

### Static checks (also run by pre-commit)

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy engine/ interface/ web/backend/
```

`ai_model.py`, `chronos_model.py` and `kronos_model.py` are excluded from
strict mypy (torch / external sibling import). Running `mypy engine interface`
alone is fine; the `web.backend.*` "unused section" note is benign.

---

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `WARNING: FinBERT unavailable (No module named 'transformers'). Falling back to VADER.` | `ai` extra not installed | `uv pip install -e ".[ai]"` |
| `WARNING: VADER unavailable (...). Falling back to naive.` | NLTK lexicon not downloaded | `uv run python -c "import nltk; nltk.download('vader_lexicon')"` |
| `+ News` variants show 0 sentiment for every historical day | Yahoo headlines only cover ~7 days; the rest of your backtest sees an empty DB lookup | Bulk-fetch with GDELT: `uv run python refresh.py --all --news-source gdelt --news-history-days 365 --force-news` |
| GDELT returns nothing | Network blocks `api.gdeltproject.org`, or you searched an obscure ticker not in `TICKER_NAMES` | Verify with the curl command in §3; extend the class's `new_names` in `config.py`'s `ASSET_CLASSES`. |
| `sqlite3.OperationalError: disk I/O error` | Usually a stale `data/` mount or read-only volume | Remove `data/market_data.db` and let it rebuild |
| `sqlite3.OperationalError: unable to open database file` mid-batch | Transient — macOS Spotlight indexing the WAL/SHM files, Time Machine snapshot, or FD pressure | Since 2026 the walk-forward loop pre-fetches news once per ticker (no thousands of per-day DB calls) and `run_all.py` wraps each ticker in try/except, so a single failure no longer kills the batch. If you still see it, re-run with `--no-refresh` once the DB is populated and check `Activity Monitor → Disk` for whatever's hammering `data/`. |
| `total_return` in the hundreds (e.g. 385.58 for 5 days) | A single bad row in `data/market_data.db` with close = 0 or NULL → trade_pnl = `(exit - 0) / 0` is clipped to a huge value | Run `uv run python scripts/clean_prices.py` to report bad rows, then `--apply` to delete them. New writes are filtered automatically (`db_manager.save_prices`) and the backtester itself drops days with > 50% single-day moves since 2026. Re-run `refresh.py` after cleanup. |
| Test runs show up in the Analysis tab picker (e.g. `custom_5d_…` dirs you never created) | `tests/test_web_api.py::TestBacktest` was missing the autouse `_redirect_persistence` fixture before 2026 | Fixed in 2026: the autouse fixture points `CACHE_DIR` + `RESULTS_DIR` at `tmp_path` for every backtest test. Manually clean any leftover test artifacts with `rm -rf results/custom_* backtests/*custom*`. If you add a new backtest-triggering test elsewhere, copy that fixture. |
| Pre-commit hook hangs on `mypy`, files appear "deleted" after CTRL+C | Pre-commit stashes unstaged changes before running hooks; CTRL+C between stash and restore leaves the changes locked in the stash | `git stash list` → you'll see a `pre-commit autosquash` stash. `git stash apply stash@{0}` recovers the files. To prevent: either let mypy finish (~20-60s), kill `mypy` itself (not `git`) in another terminal, or drop the mypy hook from `.pre-commit-config.yaml` and rely on CI. |
| `RuntimeError: LSTM requires PyTorch` | `ai` extra not installed | `uv pip install -e ".[ai]"` (and re-train with `train.py`) |
| `RuntimeError: No trained LSTM for X (period=Y)` | LSTM weights file missing | `uv run python train.py --ticker X --period Y --preset standard` |
| `--news-history-days 365` is slow | GDELT call + scoring 250 headlines each ticker | One-off; you only need to do it once. Add `--no-refresh` to subsequent backtests. |
| Changed `--sentiment-method` and scores didn't change | Cached scores in the DB were stored under the old method; new scores append. Pass `--force-news` to re-fetch and re-score | `... --sentiment-method finbert --force-news` |
| `cannot import name 'UTC' from 'datetime'` | You're on Python 3.10 | Project requires 3.12+. `uv venv --python 3.12 && uv pip install -e .` |
| `Prophet` / `Chronos-2` rows missing from backtest output | `forecast` extra not installed | `uv pip install -e ".[forecast]"` |
| `Kronos` rows missing from backtest output | Kronos clone or `[kronos]` extra missing (it's skipped silently) | `git clone https://github.com/shiyu-coder/Kronos.git ../Kronos` then `uv pip install -e ".[kronos]"`. If the clone isn't a sibling, set `KRONOS_PATH` in `config.py`. |
| `pip install -r ../Kronos/requirements.txt` fails building matplotlib 3.9.3 (`unknown type name 'Byte'` in freetype) | Kronos pins matplotlib 3.9.3, which has no Python 3.14 wheel and the bundled freetype source build errors out | Don't install Kronos's requirements file — the adapter doesn't need matplotlib. Use the `[kronos]` extra instead (`uv pip install -e ".[kronos]"`), which pulls only `einops` + `huggingface_hub`. |
| Kronos tries to use `cuda:0` / crashes on a Mac | Upstream Kronos defaults to a CUDA device | The adapter forces CPU when CUDA isn't available; if you're on an older copy, pull the current `engine/kronos_model.py`. |
| Chronos-2 shows `0/0` with no predictions | Old `chronos_model.py` calling `predict_quantiles(context=…)` | Update to the fixed adapter (positional `inputs` list); needs `chronos-forecasting>=2.0` |
| `cmdstanpy - INFO - Chain [1] start/done processing` floods the output | Prophet's Stan backend logging | Fixed — silenced around each fit. If still seen, your `prophet_model.py` predates the `_quiet_stan()` fix |
| `torch_dtype is deprecated! Use dtype instead!` | Old `chronos_model.py` passing `torch_dtype=` | Fixed — CPU passes no dtype, GPU uses `dtype=` |
| `Prophet` and `Prophet + News` rows are identical | Prophet's confidence is too high for the ±(sentiment × 0.20) nudge to flip the call | Expected, not a bug — see docs/forecasting.md |
| A forecasting run takes forever | Kronos + Prophet dominate per-day compute | Drop them with `--models knn linreg lstm chronos`, trim periods with `--periods 1y 2y 5y`, and add `--timing` to confirm where the time goes |

---

## 16. Where the data lives

```
data/market_data.db          ← SQLite (prices + news, auto-created)
models/{ticker}_{period}_{preset}.pt    ← saved LSTM weights (gitignored)
predictions/{ticker}/{date}.json        ← cached predict outputs
backtests/{ticker}/{date}_{days}d.json  ← cached backtest outputs
results/{scope}_{days}d_…/              ← run_all.py output trees
```

Delete any of these at any time — they'll regenerate on the next run.
