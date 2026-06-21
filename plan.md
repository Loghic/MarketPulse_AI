# MarketPulse AI — Plan

Working plan and research roadmap. Captured here so the top-level `README.md`
can stay short (link to this file).

## Where we are (baseline finding)

No directional edge at any horizon — accuracy is ~0.49–0.51 for 40 / 100 / 300-day
holdouts. The daily long/short strategy trails buy-and-hold, and the gap *widens*
with horizon (beat-B&H rate 27% → 19% → 2%) because daily round-trips compound fees
(~4% / 6% / 30% drag) while B&H compounds the bull trend. Even cherry-picking the best
model per ticker, only ~2/11 beat B&H over 300 days. LSTM is the only family with a
positive median return / Sharpe at 300d; k-NN is consistently worst.

**Organizing principle:** before chasing an edge, build the *measurement rigor* to
detect one. Phase 1 makes every later result trustworthy; skipping it means every
experiment in Phase 2 just produces selection-inflated, statistically-meaningless
numbers (the same trap as the cherry-picked `_summary.csv`).

The benchmark to beat is not just buy-and-hold — it's the **naive baselines** below.

---

## Phase 1 — Measurement rigor (do first)

### 1.1 Out-of-sample model-selection harness  [foundational] ✅ DONE
- Pick the best model+period on window N; evaluate it *only* on the next disjoint
  window N+1. Report OOS beat-B&H rate + median return.
- Motivation: best-per-ticker is selection-inflated, and winners are unstable
  (1 of 7 stable across the 40d/100d runs). Every result below is judged OOS,
  never on in-sample best-per-ticker.

### 1.2 Naive-strategy benchmark suite ✅ DONE
- Add trivial "models" so the ML must clear a real bar, not just B&H:
  - **Always Long** — predict UP every day (define as hold-long, no daily churn, so
    it isn't just paying fees; this is the "is the model better than blind optimism?" test).
  - **Previous-Day Direction** — predict today = yesterday's realized direction.
  - **5-Day Momentum** — UP if `close[t] > close[t-5]`, else DOWN.
  - **Random Direction** — seeded coin flip (the floor + the null for significance).
  - **Sign-Only Momentum** - predict UP if cumulative return over last N days > 0. (eg. N = 20, 50 etc.)
- Implementation: each is a `predict()` that ignores the trained model; slot into the
  backtester as variants so they get identical metrics and are directly comparable.
  Consider a `--baselines` flag (or always include them).
- Pass bar for any real model: **beat Previous-Day-Direction and Always-Long**, not
  just B&H.

### 1.3 Confidence calibration + gating  [highest-value lever]  ✅ DONE
> Implemented: `engine/calibration.py` (reliability bins, Brier, ECE, gating
> metrics), `Backtester(min_confidence=θ)` in-engine gate, `--min-confidence`
> on `backtest.py`/`run_all.py`, `--confidence-sweep` printer, and
> `--min-confidence` on `scripts/oos_harness.py` (same θ on both windows — the
> out-of-sample "does gating survive?" test). See *Phase-1.3/1.4* in AGENTS.md.
- First read the existing confidence-calibration output: are high-confidence days
actually more accurate? If calibration is flat, gating only shrinks exposure.
- Add `--min-confidence θ`: days below θ are flat (0 P&L, no fee, excluded from
accuracy). Sweep θ ∈ {0.55, 0.60, 0.65, 0.70}.
- Track: accuracy on traded days, coverage (% days traded), return, fees saved.
- Pass bar: traded-day accuracy materially > 0.5 AND return improves vs θ=0.
- Implementation note: the gate lives in `engine/backtester.py` (per-day loop) plus a
flag in `backtest.py` / `run_all.py`.
- Reliability diagram
- Brier score
- Expected Calibration Error (ECE)

### 1.4 Statistical significance testing  ✅ DONE
> Implemented: `engine/significance.py` (exact binomial test, Wilson CI,
> bootstrap CI on returns, permutation test, Benjamini-Hochberg FDR),
> `--significance` printer on `backtest.py`. Tests only *traded* days and
> applies FDR across the models in one report (no grid p-hacking).
- Currently reported: accuracy, returns, Sharpe, Sortino. Add:
  - **Binomial test** on directional accuracy (H0: p = 0.5) → p-value.
  - **Wilson score CI** on accuracy (honest interval, not ±point estimate).
  - **Bootstrap CI** on returns / Sharpe (returns are fat-tailed — resample daily P&L).
  - **Permutation test** vs Random Direction (shuffle predicted directions for the null).
- **Critical caveat:** across hundreds of model×period×ticker combos, raw p-values are
  selection-inflated. Either correct for multiple comparisons (Benjamini-Hochberg FDR)
  or formally test only the single OOS-selected config from 1.1. Don't p-hack the grid.


---

## Phase 2 — Strategy experiments (once Phase 1 can trust the numbers)

### 2.1 Turnover / fee realism
- Daily round-trips dominate cost (~30% over 300d). Test trading only on signal
  *changes* (hold through same-direction days) or longer holding periods. Report
  turnover-adjusted returns.

### 2.2 Stop-loss sweep
- `--stop-loss ∈ {0, 5, 10, 15}` on 100d + 300d. Expect wide/off best for daily holds
  (a 10%+ intraday trigger is rare for large-caps); most informative on volatile names
  (TSLA, NVDA, crypto). Won't fix a missing edge — it's a risk knob.

### 2.3 LSTM focus
- Only family with positive median return / Sharpe. Tune lookback / features / epochs;
  try LSTM-only confidence gating. Spend tuning effort here, not on k-NN.

### 2.4 k-NN k-sweep  [low priority]
- k ∈ {3, 5, 7, 9, 11}. Only meaningful under the OOS harness (1.1); high overfitting
  risk and k-NN is the worst family. Don't expect much.

### 2.5 Reframe the success metric
- Beating B&H absolute return in a bull market via daily flips is very hard. Also track
  risk-adjusted (Sharpe / Sortino) and drawdown vs B&H, and define a clear, falsifiable
  success bar up front.

## Falsification criteria

The project will consider daily-direction prediction unsuporrted if:

- OOS accuracy remains statistically indistinguishable from 50%.
- confidence gating fails to produce a profitable subset.
- No model consistently beats the naive baselines.
- No strategy beats buy-and-hold on a risk-adjusted basis after fees.

If all four hold after Phase 2, future work shifts toward:
- regression forecasting
- portfolio construction
- volatility/risk prediction
ratther than next-day direction prediction.

---

## Phase 3 — Engineering / infrastructure (supporting, lower priority than research)

### 3.1 Experiment tracking & reproducibility
- Biggest infra gap. Keep it proportionate — **not** Weights & Biases (cloud/collab
  overkill for a solo project). Options, cheapest first:
  - A custom SQLite `experiments` table: one row per run with `run_id`, `git_sha`,
    `timestamp`, `config_json`, scope/days/fee, and the headline metrics. Sits next to
    the existing `results/` CSVs (the dir-naming already encodes half of this).
  - Local MLflow (`mlflow.log_params/metrics`) if a browsable UI is wanted.
- Reproducibility: pin seeds (numpy / torch / random), and record git SHA (+ dirty flag),
  the resolved config, and the data date-range with every run.

### 3.2 Model registry
- Collapse the manual 4-step add (file → `api._get_model` → `backtest_helpers` variants
  → `main.py`) into a decorator:
  ```python
  @register_model("chronos")
  class Chronos2Model: ...
  ```
  populating a central `MODEL_REGISTRY` that `api` and `backtest_helpers` read.
- Unify with the existing `config.FORECAST_MODELS` + `ForecastModel` pattern rather than
  adding a parallel mechanism. Best done the next time a model is added. Nice DX,
  low research payoff.

---

## Phase 4 — Bigger features (after an edge is demonstrated)

### 4.1 Multi-asset portfolio backtesting
- Current backtests are single-ticker × period. Portfolio-level allocation, rebalancing,
  and portfolio risk are valuable — but premature until individual signals beat the naive
  baselines (1.2). Allocating across coin-flip signals isn't meaningful. Revisit once
  Phase 1 shows something real.


---

## Ideas & research backlog (unscheduled)

Captured for later — not yet implemented.

### Richer signals (features.py / ALL_FEATURES)
- **Volatility momentum** — momentum measured on volatility / vol-adjusted momentum.
- **Cross-asset features** — feed correlated assets in as inputs (e.g. gold↔USD,
  BTC↔BNB, index↔constituent), not just the ticker's own history. (Now that commodities/
  indices/FX exist in the registry, the cross-asset inputs are available locally.)

### Reddit sentiment (new provider behind `get_provider()`)
- Add Reddit alongside yahoo/gdelt, scored with the existing VADER/FinBERT pipeline.
- **Pushshift caveat:** the open Pushshift API is gone — Reddit restricted it to approved
  moderators in 2023–24. For live posts use the official Reddit API via PRAW (free app
  key); for historical backfill use Arctic Shift (the Pushshift successor — monthly Reddit
  dumps via Academic Torrents).

### Specific-price (regression) prediction + intraday
- Predict an actual price level, not just UP/DOWN. The forecasting models already compute
  a point value (`ForecastResult.point`) that we currently collapse to a direction —
  surface it and score with regression metrics (MAE / RMSE / MAPE) instead of accuracy.
- Explore minute / intraday bars. **Caveat:** yfinance only serves ~7 days of 1-minute
  data (~60 days for coarser intraday intervals), so minute-level backtests have a short
  lookback — a dedicated intraday data source is needed for real history.
- Investigate whether news improves specific-price prediction. News can stay **daily**
  even when prices are intraday — daily sentiment as a slow-moving feature is fine.

---

## Priority summary

| Priority | Items |
|---|---|
| **Done** | 1.1 OOS harness · 1.2 naive benchmarks · 1.3 confidence calibration + gating · 1.4 significance testing |
| **Next** | 2.1 turnover/fees · 2.2 SL sweep · 2.3 LSTM focus · 2.5 reframe metric |
| **Later** | 2.4 k-NN sweep · 3.1 experiment tracking · 3.2 model registry |
| **After an edge** | 4.1 multi-asset portfolio · backlog (signals, Reddit, regression/intraday) |

## Roadmap

- [x] k-NN model — naive + enhanced (+ time-weighted variants)
- [x] Linear Regression — naive + enhanced (+ time-weighted variants)
- [x] LSTM neural network (presets, early stopping, save/load)
- [x] Shared feature engineering (RSI, MACD, volume, volatility)
- [x] VADER sentiment + naive fallback
- [x] **FinBERT sentiment (ProsusAI/finbert, finance-specific transformer)**
- [x] **Pluggable news sources — Yahoo Finance + GDELT 2.0 (free, historical, no key)**
- [x] **Look-ahead-safe per-day news in backtests (no future-news leakage)**
- [x] **Exponential time-decay weighting + configurable news lookback window**
- [x] Walk-forward backtesting (P/L, profit factor, streaks)
- [x] Trading fees + stop-loss + buy-and-hold benchmark
- [x] Risk metrics (max drawdown, Sharpe, Sortino, yearly rolling performance)
- [x] **Data sanity guards (drop close ≤ 0; skip > 50% single-day moves)**
- [x] Batch runner with organized output (`run_all.py`)
- [x] **`scripts/news_impact.py` — quantifies news-vs-no-news across a run_all tree**
- [x] **`scripts/clean_prices.py` — one-off DB cleanup tool**
- [x] Centralized logging (cli/gui modes) + progress bars (tqdm)
- [x] Centralized config, CLI filtering, CSV/JSON export
- [x] **Data-driven asset registry — commodities/indices/FX via ETF proxies (GLD, VOO, QQQM, FXE) + BNB; combinable `--commodities`/`--indices`/`--fx` scope flags via shared `cli_helpers.py`**
- [x] Documentation (`docs/` + `AGENTS.md` + `docs/run.md` runbook)
- [x] Pytest suite (~140 tests: models, backtester, news pipeline, web API, persistence)
- [x] CI pipeline (GitHub Actions: ruff + mypy + pytest, Codecov coverage)
- [x] Pre-commit hooks (ruff auto-fix + format + mypy on every commit)
- [x] Web GUI scaffold (FastAPI + React + TypeScript, 6 pages, typed API client)
- [x] Web GUI: Dashboard (zoomable chart, OHLCV table, stats, custom period, export CSV)
- [x] Web GUI: Predict (per-ticker backend cache with timestamp, optional chart, 9 model variants, consensus, historical)
- [x] **Web GUI: Backtest (grouped ticker picker, multi-period, news/SL/B&H knobs, live progress bar, persisted runs + cached redisplay)**
- [x] **Web GUI: Training (LSTM model inventory with timestamps + active-preset marker, one-click training)**
- [x] **Web GUI: Analysis (results-dir browser, news-vs-no-news leaderboards, cross-run comparison)**
- [x] Web GUI: Settings (persistent JSON, k-NN k, fees, SL, LSTM preference, developer section)
- [x] **Time-series forecasting models — Prophet, Chronos-2, Kronos, wired into backtests**
- [x] **Per-model timing (`--timing`, time-by-family rollup) + configurable `--periods` + `--models` family filter**
- [ ] DRY model-family names: `run_all.py:_family()` and `backtest_helpers._model_family()` still hardcode the family prefixes (incl. the legacy "Chronos" alias, "TiRex", "Other"). Fold them onto `config.MODEL_FAMILY_LABELS` — the single source the `--models` filter already uses.
- [ ] TiRex forecasting model (parked — macOS-experimental, non-standard license)
- [ ] Forecasting models in `main.py` report + web GUI Predict/Backtest tabs
- [ ] Authentication (API key for public deploy)
- [ ] Multi-asset portfolio backtests (current is single-ticker × period)
