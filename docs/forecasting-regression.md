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
  first — see below — or it skips that ticker (the harness prints a loud up-front
  warning naming the tickers with no weights, so a missing model is never
  silent). This is also the residual learner the Phase-R3 hybrid reuses.

> **New asset class / first run:** pretrained weights are per ticker, so a new
> class (e.g. `--smallcap`) has none yet. The harness warns up front and the
> models fall back (hybrid → Prophet base; LSTM-reg → absent). Either train the
> weights first (`train_lstm_regressor.py` / `train_hybrid_residual.py` with the
> **same** `--days`/`--horizon`) or use `--hybrid-fit per_step` (no pretraining).
> Drop `--no-refresh` on the first run of a new class so its prices download.

- **Prophet** — the trend/seasonality base of the study (univariate, or
  multivariate with `--macro`).
- **Chronos-2 / Kronos** — Amazon's zero-shot foundation forecaster and the
  OHLCV candlestick foundation model, reused here for their point forecast.
  These are the slow `foundation` group — run them via `--models all` or
  `--models chronos,kronos`, not in the default `paper` set.
- **Residual hybrid** (`engine/residual_hybrid.py`, needs Prophet + torch) — the
  paper's central artifact: `P̂ = P̂^base + r̂es`. A base model (Prophet by
  default) captures trend/seasonality, and a residual learner (an LSTM fit on the
  base's in-sample residuals) predicts what the base missed. If the residuals are
  white noise the learner predicts ~0 and the hybrid reduces to the base — which
  is exactly the "*when* does residual learning help" question. It's fully
  composable: `ResidualHybrid(base, residual_learner)` takes any `ForecastModel`
  base and any `fit/predict` learner, so Prophet+LSTM, ARIMA+LSTM, Prophet+XGB …
  are just different arguments.

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
`--indices`/`--fx`/`--smallcap`/`--all`/`--tickers`), `--days` (eval window),
`--horizon` (direct-h), `--models`, `--target {level,log-return}`, `--refit-k`,
`--min-train`, `--no-refresh`.

**`--models`** chooses which shared forecasters run — comma/space-separated keys
(`rw`, `rwdrift`, `seasonal`, `arima`, `xgboost`, `prophet`, `chronos`, `kronos`)
and/or named groups:

- **`paper`** (default) — the Prophet/LSTM study plus its benchmarks (RW, RW+
  Drift, Seasonal Naive, ARIMA, XGBoost, Prophet). The LSTM regressor and the
  Prophet+LSTM hybrid join via their own flags (`--no-lstm` to drop the former,
  `--hybrid` to add the latter).
- **`benchmarks`** — the naive/classical reference set only.
- **`foundation`** — Chronos-2 + Kronos (slow, zero-shot; not in `paper`).
- **`all`** — every available shared model.

`--macro` implies a `+ macro` variant for the *selected* macro-capable models
(XGBoost, Prophet) — so `--models paper --macro` adds XGBoost+macro and
Prophet+macro, but `--models rw --macro` adds nothing. A model whose library
isn't installed is skipped (logged), never an error.

The **`--smallcap`** class (IWM, XLE/XLF/XLU sector ETFs, XOM, FCX) is the
"less-efficient corners" test: if even mega-caps show no edge, the honest check
is whether small-caps / sector / commodity-sensitive names do. Same null is a
stronger result; a DM-significant U2 < 1 there is an interesting, publishable
exception.

The **`--target log-return`** mode scores the implied return `r̂ = log(P̂/P_t)`
against a **zero-return** (efficient-market) benchmark instead of the price
level. This is score-only — models still forecast a level; only the metric space
changes. It's the framing reviewers expect (does it beat "predict no move?"),
and it removes the level-persistence that makes a price-level random walk look
unbeatable. Note: the random walk maps to "predict 0 return", so it still scores
**U2 = 1.0** — and because U2 already divides out level persistence, the *ranking*
is identical to level mode. The value is interpretive clarity, not a different
conclusion. (A *native* return-forecasting variant — models trained on `r`
directly — is planned for a true level-vs-return comparison; see plan.md.)

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

## The residual hybrid (Prophet + LSTM)

The hybrid is the paper's central artifact:

```
P̂_{t+1} = P̂^base_{t+1} + r̂es_{t+1}
res_t    = close_t − fitted^base_t        (the base's in-sample residuals)
r̂es      = a learner trained on residuals up to t
```

`ResidualHybrid(base, residual_learner)` (`engine/residual_hybrid.py`) is a
`ForecastModel`, so it runs in the harness like any other forecaster. Per step it
takes the base's genuine OOS forecast for `t+h`, computes the base's in-sample
residuals via `base.fit_in_sample(df)` (Prophet's in-sample `predict`, ARIMA's
`fittedvalues`, or the random-walk default for anything else), trains the
residual learner on those residuals **up to `t` only**, and adds the predicted
next residual. The leakage rule is enforced and unit-tested: the learner never
sees `res_{t+h}`, and the base forecast for `t+h` used no data past `t`.

Composability is the point — swap the base and you get Prophet+LSTM,
ARIMA+LSTM, Prophet+XGB, … for free. If torch is missing or the residuals are
too short, the learner predicts 0 and the hybrid cleanly reduces to the base
model. That reduction is itself the experiment: **residual learning only helps
when the base's residuals carry structure** — on a near-random-walk daily series
they're close to white noise, so expect the hybrid's U2 to sit right on the
base's.

**Enabling it + the fit cadence.** The hybrid is the slowest model, so it's
**off by default** — pass `--hybrid`. How often the residual learner retrains
across the walk-forward is set by `--hybrid-fit`:

- `pretrained` (default) — load **frozen** weights trained once on pre-eval
  residuals; predict-only each step. Fastest. Train them first:
  ```bash
  uv run python scripts/train_hybrid_residual.py --stocks --days 100 --horizon 1 --preset standard
  uv run python scripts/forecast_harness.py --stocks --days 100 --horizon 1 \
      --hybrid --hybrid-fit pretrained --no-refresh
  ```
  Like the LSTM-reg trainer, this trims the last `--days + --horizon` rows before
  fitting, so the eval window is unseen — use the **same** `--days`/`--horizon`
  to train and score. `--preset {quick,standard,cluster}` sets the residual
  learner's effort tier (the same tiers as the LSTM regressor; default
  `standard`). Weights go to `models/{ticker}_hybrid_res.pt`. If they're missing,
  the hybrid still runs but its learner predicts 0 (≈ base).
- `refit_k` — retrain every `--hybrid-refit-k` steps, reuse frozen weights in
  between (partial speedup, somewhat adaptive).
- `per_step` — retrain every step (slowest, most adaptive; the original
  behaviour).

All three are leakage-safe: the learner only ever sees residuals from the window
ending at `t`, and pretrained weights came from pre-eval residuals only.

## Is the difference real? (Diebold–Mariano + Wilcoxon)

A U2 of 0.998 vs 1.000 looks like a win but is almost always noise. `engine/
forecast_significance.py` makes that judgment formal, on the per-step errors:

- **Diebold–Mariano** (`dm_test`) — the standard forecast-comparison test, on
  the loss differential `d_t = g(e₁) − g(e₂)` (squared or absolute loss). Pure
  numpy: Newey–West HAC variance (`h−1` lags for an `h`-step forecast) + the
  Harvey–Leybourne–Newbold small-sample correction, compared to a Student-`t`.
  Sign convention: **`stat < 0` ⇒ model 1 beats model 2.** So test the hybrid by
  passing `(hybrid_errors, rw_errors)` — a significant *negative* stat is a real
  win.
- **Wilcoxon signed-rank** (`wilcoxon_loss_test`) — non-parametric companion on
  the paired losses (uses `scipy.stats.wilcoxon` when installed, else a numpy
  normal approximation).
- **Grid + FDR** (`compare_to_reference`) — compares many models against a
  reference and applies Benjamini–Hochberg FDR across the whole grid (reusing
  `significance.benjamini_hochberg`), so a single cell can't be cherry-picked off
  a 200-cell table. A row is flagged a winner only if it survives FDR *and* has a
  negative mean loss differential.

The expected result on daily levels: no model is DM-significant vs the random
walk, and the hybrid is not DM-significant vs its base — which turns "they all
look like 1.000" into the defensible claim "indistinguishable from a random
walk".

## Why the hybrid can't beat the random walk (residual diagnostics)

`engine/residual_diagnostics.py` answers the paper's title question directly:
does the base model's residual contain *learnable structure*? The hybrid can
only beat the base if it does — if the residual is white noise, there's nothing
for the residual learner to grab and the hybrid reduces to the base.

On a residual series `res_t = y_true_t − base_pred_t` it computes:

- **ACF** — sample autocorrelation; `acf1` (lag 1) is a quick structure scalar.
- **Ljung–Box Q-test** (`ljung_box`) — the headline: jointly tests "white noise
  up to lag h". p < 0.05 ⇒ the residual is autocorrelated (structured); p large
  ⇒ indistinguishable from white noise. `diagnose(...).structured` is this flag.
- **Runs test** and **Lo–MacKinlay variance ratio** — companion randomness /
  mean-reversion checks.

All pure numpy (the χ² p-value uses scipy when present, else a numpy
upper-incomplete-gamma fallback).

`structure_vs_gain(cases)` builds the paper's **central cross-tab**: per ticker,
the base residual's autocorrelation strength (Ljung–Box stat / |ACF1|) against
the hybrid's skill gain `ΔU2 = U2_base − U2_hybrid`. The thesis: gain appears
**iff** the residual is structured. On daily equity levels the expected result is
the clean negative — residuals are white noise (Ljung–Box non-significant), so
`ΔU2 ≈ 0` and the hybrid sits on the base, which sits on the random walk.

## Macro / exogenous features

`engine/macro_data.py` adds market-wide inputs alongside a ticker's own history:
VIX, the dollar index (DXY, UUP fallback), Gold, and the S&P 500 as **log-
returns** (via yfinance), plus the 1-Year Treasury yield (DGS1) as a **level**
from FRED's public CSV endpoint (no API key). Each series is fetched defensively
— a failure drops that column and logs, never crashing — and cached to SQLite
(`MacroCache`, `macro_series` table).

The correctness-critical piece is `align_macro(ticker_dates, macro_df, lag=1)`:
it reindexes the macro panel onto the ticker's trading calendar, **forward-fills**
the macro's own holidays/gaps, then **lags by one day** so the value attached to
date `t` is the one known at `t − 1`. With the default `lag=1`, the feature row
that predicts `t+1` uses only macro information available at `t` — never the
future (the R0.2 leakage rule, unit-tested with a no-lookahead spy). The first
row is NaN (no prior macro yet).

Macro is wired into **XGBoost** and **Prophet**: pass `--macro` to the harness
and it adds **`XGBoost + macro`** and **`Prophet + macro`** variants beside their
plain counterparts — the price-only vs +macro ablation. Per ticker the macro
panel is aligned (`align_macro`, lag-1) onto that ticker's dates.

- **XGBoost** (`XGBoostForecaster(macro_df=…)`): each feature window ending at
  `t` gets `macro[t]` appended; missing-macro rows are dropped from training.
- **Prophet** (`ProphetModel(macro_df=…)`, R3.3 multivariate Prophet): each macro
  column is added via `add_regressor`; the regressor value for the forecast date
  is **carried forward from the last in-window macro** (= macro at `t`, known at
  `t−1`). If macro has a gap over the training window, Prophet falls back to
  univariate for that call rather than erroring.

Both are leakage-safe because the panel is lag-1 aligned.

```bash
uv run python scripts/forecast_harness.py --stocks --days 100 --macro --no-refresh
```

Expected, consistent with everything else: macro shouldn't pull XGBoost's U2
below 1 on daily levels — but the ablation has to be *run* for the paper to say
"macro doesn't help" rather than assume it.

## What's next (not yet implemented)

- **Macro into the LSTM regressor + hybrid residual learner** (rest of R4.4) —
  XGBoost and Prophet consume macro; the LSTM-reg (pretrained checkpoint) and the
  hybrid's residual learner don't yet.
- **Sentiment pos/neg split** (R4.3) — two separate sentiment signals as inputs.
- **Multi-horizon / asset-class / regime slices** (R7) and **robustness** (R8 —
  seed sweeps, lookback/refit sensitivity, level-vs-log-return).
- Wiring the DM/Wilcoxon comparison and the residual cross-tab into the harness
  output (both currently consume the per-step CSVs the harness already writes).

See the forecasting plan for the full R0–R8 roadmap.
