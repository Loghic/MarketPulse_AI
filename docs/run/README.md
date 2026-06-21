# Running MarketPulse AI

Everything you need to install, configure, and run MarketPulse AI. The
top-level `README.md` is intentionally short and points here; this is
the in-depth runbook, split into focused files so each topic stays
readable.

## Quickstart (3 minutes, you already cloned the repo)

```bash
uv venv
uv pip install -e ".[ai,web,viz,dev,forecast]"   # the works
uv run python -c "import nltk; nltk.download('vader_lexicon')"
uv run python main.py --stocks                   # today's predictions
```

If a flag or warning looks unfamiliar, jump to the right page below.

## Pages

| Page | Read when you want to… |
|---|---|
| [setup.md](setup.md) | install, populate first-run model weights, see the ticker list |
| [predict.md](predict.md) | get a next-day prediction (no backtesting) |
| [refresh.md](refresh.md) | populate the DB with historical prices + news |
| [backtest.md](backtest.md) | run a walk-forward backtest, sweep periods, train an LSTM |
| [research.md](research.md) | quantify news-vs-no-news, run the out-of-sample harness |
| [web-gui.md](web-gui.md) | launch the FastAPI + React frontend |
| [workflows.md](workflows.md) | copy-paste end-to-end recipes (daily / paper / spotlight) |
| [reference.md](reference.md) | `config.py` knobs, testing, troubleshooting, storage layout |

## The seven CLI scripts

| Script | What it does | Detail page |
|---|---|---|
| `main.py`     | Next-day prediction report for each model variant | [predict.md](predict.md) |
| `refresh.py`  | Pre-fetch prices + news into the DB (no models)   | [refresh.md](refresh.md) |
| `backtest.py` | Walk-forward backtest with metrics and CSV/JSON export | [backtest.md](backtest.md) |
| `train.py`    | Train LSTM models for future use                  | [backtest.md](backtest.md) |
| `run_all.py`  | Batch wrapper that runs `--compare-periods` per ticker into subdirectories | [backtest.md](backtest.md) |
| `scripts/news_impact.py` | Post-processes a `run_all.py` result tree to quantify news-vs-no-news per ticker, model, and overall | [research.md](research.md) |
| `scripts/oos_harness.py` | Out-of-sample model-selection harness: pick the winner on one window, evaluate on the next disjoint one | [research.md](research.md) |

All scripts auto-refresh data on startup unless you pass `--no-refresh`.
They share the news/sentiment flags introduced in the 2026 refactor.

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

`backtest.py` and `run_all.py` also accept these run-shaping flags
(handy now that the forecasting models make a full sweep slow):

| Flag | Default | Purpose |
|---|---|---|
| `--periods P [...]` | all | Restrict the period set in `--compare-periods` (subset of `ALL_PERIODS`). Skip the slow `max` window. Same flag on `run_all.py`. |
| `--models F [...]` | all | Only run these model families: `knn`, `linreg`, `lstm`, `prophet`, `chronos`, `kronos`, `baseline`. Maps SL / `+ News` / time-weighted variants to the right family. Same flag on `run_all.py`. |
| `--no-baselines` | off | Skip the naive baselines (AlwaysLong, PreviousDay, 5/20-Day Momentum, Random). Baselines included by default. |
| `--min-confidence θ` | 0 | **Confidence gate.** Sit out days below θ confidence (flat, no fee, excluded from accuracy). Same flag on `run_all.py` (adds an `mcNNN` segment to the output dir) and on `oos_harness.py`. |
| `--confidence-sweep` *(backtest.py)* | off | Print a θ-sweep table (coverage / traded-day accuracy / return / fees saved) + Brier/ECE, from a single ungated run. Implies `--full`. |
| `--significance` *(backtest.py)* | off | Print binomial p + Wilson CI on accuracy, bootstrap CI on return, permutation p, with Benjamini-Hochberg FDR. Implies `--full`. |
| `--timing` *(backtest.py)* | off | Print a slowest-first per-model compute-time table after the summary. `run_all.py` prints a time-by-model-family rollup automatically. |

```bash
# Fast representative sweep — drop slow 'max' and the heavy
# Prophet/Kronos models, and see where the time goes.
uv run python backtest.py --tickers NVDA --compare-periods \
    --periods 1y 2y 5y --models knn linreg lstm chronos --timing
```

## Where to look next when something is missing

* [docs/api.md](../api.md) — architecture, DB schema, model contract
* [docs/backtesting.md](../backtesting.md) — walk-forward methodology, all metrics
* [docs/forecasting.md](../forecasting.md) — Prophet, Chronos-2, Kronos + the ForecastModel interface
* [docs/sentiment.md](../sentiment.md) — VADER vs FinBERT vs naive
* [docs/news_sources.md](../news_sources.md) — Yahoo, GDELT, adding new providers
* [docs/features.md](../features.md), [docs/k-NN.md](../k-NN.md), [docs/linear_regression.md](../linear_regression.md), [docs/lstm.md](../lstm.md) — per-component
* [docs/web.md](../web.md) — full Web GUI / REST API docs
