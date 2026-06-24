# Regression / point-forecast track

A second evaluation path that scores **predicted price levels**, not UP/DOWN
trades. It is deliberately separate from the trading backtester — no positions,
fees, stop-loss, or P&L — because the prediction target and the metrics are
different. The directional track (`backtester.py`, the OOS harness) answers
"does a daily long/short strategy beat buy-and-hold?"; this track answers "how
close is the predicted next price to the real one, and is that better than
assuming tomorrow equals today?"

This is the foundation (Phases R0–R2) of the residual-hybrid research track. The
full roadmap — the Prophet+LSTM residual hybrid, macro features, Diebold–Mariano
/ Wilcoxon comparison tests, and residual diagnostics — lives in the forecasting
plan; this page documents what is implemented today.

## The metric trap this track is built around

On a persistent price level, the no-change **random-walk** forecast
(`P̂_{t+1} = P_t`) is already very close most days, so its RMSE / MAE / MAPE are
tiny. That makes *absolute* error metrics flatter every model — a model can post
an impressive-looking RMSE while being no better than "assume no change".
Absolute numbers are therefore nearly uninformative on financial levels.

The headline metrics are the **scale-free, skill-relative** ones, all measured
against the random walk on the same window:

- **Theil's U2** = `RMSE(model) / RMSE(random-walk)`. **U2 < 1 ⇔ beats the
  random walk.** By construction the random-walk forecaster scores exactly 1.0.
  This is the number to report first.
- **MASE** = test MAE ÷ the in-sample one-step naive MAE. < 1 beats the
  in-sample naive forecast. Comparable across tickers of different price scales.
- **RMSSE** — the squared-error analogue (the M5-competition metric).

The absolute four (`rmse`, `mae`, `mape`, `smape`) are still computed — fine for
comparing models *on one series* — but a result reported without U2 or MASE is
not interpretable. Degenerate denominators (a dead/constant series) return `NaN`
rather than `inf`, so medians across tickers stay clean.

## Forecasters

Every forecaster subclasses the existing `engine/forecast_base.py:ForecastModel`
(it implements `_raw_forecast` and gets the value→direction adapter for free), so
the same objects work in both tracks — the regression harness reads
`ForecastResult.point`, the directional one reads the derived UP/DOWN.

**Regression baselines** (`engine/naive_forecasters.py`) — the analogue of the
directional `baseline_models.py`:

- **Random Walk** — `P̂_{t+1} = P_t`. The reference for U2 / MASE.
- **Random Walk + Drift** — `P̂_{t+h} = P_t + h·μ`, μ = mean one-step change.
- **Seasonal Naive (m)** — the value `m` steps back (weekly = 5 on daily bars).

**Real forecasters** (optional dependencies, skipped gracefully when absent):

- **ARIMA** (`engine/arima_model.py`, needs `statsmodels`; `auto=True` uses
  `pmdarima` for order selection on the selection window only).
- **XGBoost** (`engine/xgboost_model.py`, needs `xgboost`). Trains on the
  tabular feature matrix from `features.py`. It predicts the **one-step change**
  (`close[t+h] − close[t]`) and adds it back to the last close, so the trees
  aren't capped by the training price range on a trending series — the reported
  value is still a level.
- **LSTM-reg** (`engine/lstm_regressor.py`, needs `torch`). A *new* network with
  a **linear** head predicting the next-close Δ — distinct from the directional
  LSTM in `ai_model.py`, whose sigmoid classifier head carries no price-magnitude
  information and so can't be scored here. It is **per-ticker**: it loads
  pre-trained weights from `models/{ticker}_reg.pt` (the `_reg` suffix keeps it
  separate from the `{ticker}_{period}_{preset}.pt` classifiers). Train them
  first — see below — or it skips that ticker. This is also the residual learner
  the Phase-R3 hybrid will reuse.
- **Prophet / Chronos-2 / Kronos** — the existing forecasting models, reused
  here for their point forecast.

Install the optional libraries with the `[forecast]` extra:

```bash
uv pip install -e '.[forecast]'   # prophet, chronos, statsmodels, xgboost, scipy, pmdarima
```

If a library isn't installed, that forecaster simply doesn't appear in the run —
nothing crashes.

## Walk-forward contract

`engine/forecast_backtester.py:ForecastBacktester` runs the evaluation:

- **Expanding window** — at evaluation step `t` the model sees all data up to and
  including `t`, and forecasts `t + h`.
- **Direct-h horizons** — a separate forecast per horizon `h`; no recursive
  multi-step (which compounds error and muddies the "does it help" signal).
- **Training-window cap (`--max-train`, default ≈ 504 ≈ 2 years)** — bounds how
  far back the expanding window reaches. On a long-history ticker (AAPL has
  ~11k rows back to split-adjusted cents) an unbounded window is both slow and
  pathological: Prophet/ARIMA fit across decades of regime change and forecast a
  wildly off level. The cap also makes MASE's in-sample naive scaling reflect
  *recent* volatility rather than the average daily move since the 1980s
  (uncapped, MASE on AAPL reads ~9 — today's ~$3 moves vs a ~$0.37 lifetime
  average; capped it's ≈ 1). **Theil U2 is unaffected** either way (it scales
  model and reference on the same eval window). Use `--max-train 0` for the full
  expanding window.
- **Refit cadence `K`** — a cost knob for expensive models; it never changes
  which data is visible (always ≤ `t`).
- **Leakage guarantee** — the model only ever receives a window ending at `t`
  (the most-recent `max_train` rows up to `t`); the realised `close[t+h]` is
  never in that slice. Unit-tested with a spy that records every window length
  the model is handed.

The random-walk reference (`close[t]`) is recorded at each step so every model is
scored against the *same* naive series.

## Running it

```bash
# Score every available forecaster on the stocks, 100-step eval window, h=1
uv run python scripts/forecast_harness.py --stocks --days 100 --horizon 1 --no-refresh

# A different horizon, weekly
uv run python scripts/forecast_harness.py --crypto --days 100 --horizon 5
```

Flags: the usual scope selectors (`--stocks`/`--crypto`/`--commodities`/
`--indices`/`--fx`/`--all`/`--tickers`), `--days` (eval window), `--horizon`
(direct-h), `--refit-k`, `--min-train`, `--no-refresh`.

Output goes to `results/fc_<scope>_<days>d_h<h>_<timestamp>/`:

```
results/fc_stocks_100d_h1_20260624-101500/
├── AAPL.csv          # tidy per-step: model, ticker, horizon, date, y_true, y_pred, y_naive
├── MSFT.csv
├── _fc_summary.csv   # per (model, ticker, horizon): n, rmse, mae, mape, smape, mase, rmsse, theil_u2
└── _fc_console.txt   # the printed table
```

The console table is ranked by U2 (best skill first) and ends with a per-model
median across tickers. **A model is only interesting if its median U2 < 1.0** —
i.e. it beats the random walk.

### Training the LSTM regressor (leakage-safe)

The LSTM-reg forecaster needs pre-trained per-ticker weights. Train them with
`scripts/train_lstm_regressor.py`, using the **same** `--days`/`--horizon` you'll
pass to the harness:

```bash
uv run python scripts/train_lstm_regressor.py --stocks --days 100 --horizon 1 --preset standard
uv run python scripts/forecast_harness.py     --stocks --days 100 --horizon 1 --no-refresh
```

`--preset {quick,standard,cluster}` sets the training-effort tier (capacity +
budget), mirroring the directional LSTM's presets: `quick` for iterating,
`cluster` for the final paper run. The **training window** is the `--max-train`
knob (rows before the eval window; `0` = all history, `504` ≈ 2 years) — there's
no per-ticker period *selection* here, deliberately, so the regression track
doesn't reintroduce the selection inflation the directional OOS harness exists
to expose. Report a `--max-train` sensitivity row if you want robustness.

The training script trims the **last `--days + --horizon` rows** off each ticker
before fitting, so the harness's evaluation window is never seen during training
— the loaded weights are genuinely out-of-sample. If you train with a *different*
`--days` than you score with, that guarantee breaks: the eval window can overlap
the training data. Weights land in `models/{ticker}_reg.pt`; the harness picks
them up automatically (disable with `--no-lstm`). Without torch, or for tickers
with no weights, LSTM-reg simply doesn't appear in the run.

## What's next (not yet implemented)

- **Residual hybrid** — `P̂ = P̂_base + r̂es`, the paper's central artifact
  (`ResidualHybrid(base, residual_learner)` so Prophet+LSTM / Prophet+Chronos /
  ARIMA+XGBoost are all free). Needs a `fit_in_sample` hook on the base model and
  a residual-LSTM regressor.
- **Forecast-comparison statistics** — Diebold–Mariano and Wilcoxon signed-rank
  on paired per-step losses, FDR-corrected across the model × ticker × horizon
  grid (the regression analogue of `engine/significance.py`).
- **Macro / exogenous features** and **residual diagnostics** (Ljung–Box etc.).

See the forecasting plan for the full R0–R8 roadmap.
