# MarketPulse AI — Forecasting Plan (Residual-Hybrid track)

Companion to `plan.md`. `plan.md` is the **directional** (UP/DOWN, trading-P&L)
research track. **This file is the point-forecast / regression track** that the
paper *"When Does Residual Learning Improve Financial Time-Series Forecasting:
Evidence from Prophet–LSTM Hybrid Models"* actually needs.

The two tracks share the data layer, the asset registry, the walk-forward
discipline, the OOS harness mindset, and the multiple-comparison hygiene — but
they have **different prediction targets and different metrics**, so they get
separate engines and separate evaluation paths. Do **not** bolt regression onto
the trading backtester; keep them parallel.

---

## North star

Produce, score, and statistically compare the artifact in the paper:

```
P̂_{t+1} = P̂_{t+1}^{base} + r̂es_{t+1}
res_t   = P_t − P̂_t^{base}            (base = Prophet for now)
r̂es     = residual learner (LSTM-regressor for now)
```

The paper's title is a **conditional** ("*When* does it improve"). So the
deliverable is not "hybrid wins" — it's a **map of the regimes/assets/horizons
where the residual step adds real, out-of-sample, statistically-significant skill
over (a) the base model alone and (b) a random walk**, plus the residual
diagnostics that explain *why*.

### Extensibility (explicit design requirement)

`base` and `residual learner` must be **composable**, because the paper roadmap is
Prophet+LSTM now, then Prophet+Chronos, Prophet+Kronos, etc. Build **one**
`ResidualHybrid(base, residual_learner)` (Phase R3) so the combination matrix is
free:

| base ↓ / residual learner → | LSTM-reg | XGBoost | (future) |
|---|---|---|---|
| **Prophet** | paper v1 | ablation | |
| **Chronos-2** | later | later | |
| **Kronos** | later | later | |
| **ARIMA** | classic Zhang-2003 hybrid (good sanity baseline) | | |

Any model exposing a point forecast (`ForecastResult.point`, already in
`engine/forecast_base.py`) can be a `base`; any regressor can be the residual
learner.

---

## Phase R0 — Evaluation contract (do first; cheap, prevents rework)

Decide and write down, before any code, because each choice silently changes every
later number:

- **R0.1 Target space.** Forecast the **price level** (matches the paper and
  Prophet's trend/seasonality decomposition). Add **log-return space** as a
  robustness variant in R8, *not* as the primary — but commit to level now.
  - **DONE (score-only):** `forecast_harness.py --target log-return` scores the
    *implied* return `r̂=log(P̂/P_t)` vs a zero-return benchmark, without changing
    any model (level forecasts converted at scoring time). U2 ranking is
    identical to level mode (U2 divides out level persistence); the value is the
    reviewer-expected "beat predict-no-move?" framing.
  - **TODO (native):** train the ML models (XGBoost, LSTM-reg) on `r` *directly*
    so the optimisation objective is return-space, then compare native-return vs
    score-only-converted. Only the native path can back a paper sentence that
    says "the models forecast returns"; the comparison itself (does training in
    return space change anything?) is a worthwhile robustness result.
- **R0.2 Leakage rule for residuals** (the single most important correctness
  decision):
  - *Training* the residual learner: use the base model's **in-sample fitted**
    residuals on the training window (standard, Zhang 2003).
  - *Prediction* at test time: the base forecast `P̂^{base}_{t+1}` must be a genuine
    out-of-sample forecast, and the residual learner may use **only residuals up to
    `t`**. No residual at `t+1` is ever visible at prediction time.
  - Any exogenous regressor fed to Prophet for `t+1` must be known at `t` (lag-1) —
    Prophet's `add_regressor` does **not** forecast regressors, so future values
    would be leakage. See R4.
- **R0.3 Walk-forward + refit cadence.** Prophet already refits per day in the
  directional engine, but a full hybrid (Prophet refit + LSTM retrain) per day is
  infeasible. Use **expanding window, refit every `K` trading days** (start `K=21`,
  ~monthly), predicting the in-between steps from the frozen fit. Expose `K` as a
  config knob and sweep it in R8. Log `elapsed_seconds` per refit (engine already
  supports this).
- **R0.4 Multi-horizon convention.** Use **direct-h** (a separate model/forecast per
  horizon) for `h ∈ {1, 5, 10, 20}`. Prophet forecasts h-ahead natively; the residual
  learner gets a per-horizon target. Avoid recursive multi-step (error compounding
  muddies the "does residual help" signal).
- **R0.5 Split discipline.** Reuse the OOS philosophy from `plan.md` §1.1: any
  hyperparameter / model selection happens on a window strictly disjoint from the
  reported evaluation window.

---

## Phase R1 — Regression evaluation path  [foundational]

A point-forecast scoring path parallel to the trading backtester.

- **R1.1 `engine/regression_metrics.py`** — pure (numpy + stdlib), mirroring the
  style of `engine/calibration.py`:
  - Absolute: `rmse`, `mae`, `mape`, `smape`.
  - **Scale-free skill (the anti-level-trap core — these are the headline metrics):**
    - **MASE** = `mean(|e_t|) / MAE_naive_insample`, naive = one-step random walk.
    - **RMSSE** = `sqrt(mean(e_t²) / MSE_naive_insample)` (the M5 metric).
    - **Theil's U2** = `RMSE(model) / RMSE(random-walk)`. **U2 < 1 ⇔ beats RW.**
  - Rationale to put in the paper: on a persistent price level, the no-change RW
    forecast already gets a tiny RMSE/MAPE, so **absolute** errors flatter every
    model and are nearly uninformative. Report skill *relative to RW* or the results
    are not interpretable. **This is the biggest methodological risk in the draft —
    make it a first-class result, not a footnote.**
- **R1.2 `engine/forecast_backtester.py`** — a lean walk-forward loop that, per step,
  records `(date, horizon, y_true, y_pred, model_name, ticker)` and nothing trade-
  related. Do **not** reuse `backtester.py` (it's saturated with position/fee/SL
  logic irrelevant here). Honour R0.3 refit cadence and R0.4 horizons.
- **R1.3 Persist per-step predictions.** Surface `ForecastResult.point` end-to-end
  (the README/AGENTS already flag this as "a later step") into a tidy CSV:
  `results/fc_<scope>_h<h>_<ts>/{TICKER}.csv` with columns above + per-model summary.
- **R1.4 Console + summary table** grouped by model family, ranked by **MASE / U2**
  (not RMSE).
- **Tests:** `tests/test_regression_metrics.py` — hand-computed MASE/RMSSE/U2 on a
  toy series; RW must score U2 = 1.0 and MASE ≈ 1.0 by construction.
- **Pass bar:** every model reports MASE + Theil U2 against RW on the same OOS window.

---

## Phase R2 — Regression benchmarks (the bar the hybrid must clear)

The paper names Random Walk, ARIMA, Prophet, LSTM, XGBoost. Prophet/LSTM exist but
as direction emitters; the rest don't exist at all.

- **R2.1 `engine/naive_forecasters.py`** — `RandomWalk` (`P̂_{t+1}=P_t`),
  `RandomWalkDrift` (+ mean historical change), `SeasonalNaive` (weekly/`m`-step).
  These are the regression analogue of `baseline_models.py`. RW is the *reference*
  for U2/MASE, so it must exist as an explicit forecaster, distinct from the trading
  `PreviousDay` baseline.
- **R2.2 `engine/arima_model.py`** — ARIMA via `statsmodels` (or `pmdarima.auto_arima`
  for order selection on the selection window only). Refit on R0.3 cadence.
- **R2.3 `engine/xgboost_model.py`** — `XGBRegressor` on the tabular feature matrix
  (the paper lists XGBoost as *both* a feature consumer and a benchmark). Targets the
  level (or Δ — decide in R0.1). Reuses `features.py` + R4 macro features.
- **R2.4** All four subclass the regression forecaster contract (point-forecast +
  `fit/predict`), so `forecast_backtester` picks them up like `FORECAST_MODELS`
  picks up Prophet/Chronos/Kronos today.
- **Pass bar for *any* model, hybrid included:** Theil U2 < 1 (beats RW) **and**
  DM-significant vs RW (R5). The paper's interesting comparison is hybrid vs
  **base-alone** and vs **plain LSTM / plain XGBoost**, not vs RW (which is the floor).

---

## Phase R3 — The residual hybrid (paper's central artifact)

- **R3.1 `engine/lstm_regressor.py`** (or a regression head on `ai_model.py`) — LSTM
  that predicts a continuous target `r̂es_{t+1}` from a window of past residuals
  (+ optionally features/macro). Today's `ai_model.py` is a classifier; do **not**
  overload it — a clean regressor is less risky. Reuse the StandardScaler + early-
  stopping plumbing.
- **R3.2 `engine/residual_hybrid.py`** — the composition:
  ```python
  class ResidualHybrid:
      def __init__(self, base: ForecastModel, residual_learner): ...
      def fit(self, df):
          base_fit   = self.base.fit_in_sample(df)        # fitted P̂^base on train
          residuals  = df.close - base_fit                 # R0.2 training residuals
          self.residual_learner.fit(residuals, exog=...)   # learns structure base missed
      def forecast(self, df, h) -> ForecastResult:
          p_base = self.base.forecast(df, h).point          # genuine OOS base forecast
          r_hat  = self.residual_learner.predict(...)        # residuals up to t only
          return ForecastResult(point=p_base + r_hat)
  ```
  This is what makes Prophet+Kronos / Prophet+Chronos free later — swap `base`.
- **R3.3 Multivariate Prophet** — extend `prophet_model.py` with `add_regressor` for
  OHLC-in-level + macro (R4) + technicals + the two sentiment signals (R0.2 lag rule
  applies). Keep univariate Prophet as a separate registered model so the
  multivariate uplift is measurable.
- **R3.4 Residual-construction utility** with the R0.2 leakage rule unit-tested
  (a `forecast_backtester` spy that asserts the residual learner never sees
  `res_{t+1}` and the base forecast for `t+1` used no data past `t`).
- **Tests:** `tests/test_residual_hybrid.py` — identity check (zero residual learner →
  hybrid ≡ base), additive-reconstruction check, and the disjoint/leakage guarantee.
- **Pass bar:** hybrid Theil U2 < base-alone Theil U2, **DM-significant**, OOS.

---

## Phase R4 — Macro & exogenous features

Currently `features.py` is per-ticker only; the registry holds only tradeable
tickers (GLD/VOO/QQQM/FXE). The paper wants macro exogenous inputs.

- **R4.1 `engine/macro_data.py`** — fetch VIX (`^VIX`), DXY (`DX-Y.NYB`, fallback ETF
  `UUP`), Gold (`GLD`/`GC=F`), SP500 (`^GSPC`/`SPY`) in **log-returns**, and DGS1
  (1-Year Treasury, **FRED** — not yfinance) as a level/yield. Cache to SQLite next to
  prices.
- **R4.2 Calendar alignment** — reindex onto each ticker's trading calendar,
  **forward-fill** gaps (FRED is business-day with holidays/missing), then **lag by 1
  day** so only information available at `t` predicts `t+1` (R0.2). Unit-test the
  no-lookahead alignment.
- **R4.3 Sentiment pos/neg split** — surface positive and negative scores separately
  (paper specifies two signals), not just one signed score; the news pipeline already
  produces per-day scores.
- **R4.4 Wiring** — macro/sentiment columns flow into the LSTM/XGBoost feature matrix
  *and* into Prophet as regressors (R3.3).
- **Pass bar:** an **ablation** (price-only vs +tech vs +macro vs +sentiment) reported
  honestly. "Macro doesn't help" is a fine, publishable result if that's what the data
  says.

---

## Phase R5 — Forecast-comparison statistics

`engine/significance.py` is entirely directional (binomial / Wilson / permutation).
Forecast accuracy needs different tests.

- **R5.1 Diebold–Mariano** — on the loss differential `d_t = g(e¹_t) − g(e²_t)`,
  `g ∈ {squared, abs}`. Implementable in pure numpy: DM stat = `d̄ / sqrt(HAC-var(d̄))`
  with Newey–West (`h−1` lags for h-step). Apply the **Harvey–Leybourne–Newbold**
  small-sample correction and compare to `t_{T−1}`. This is *the* standard test for
  the paper's tables.
- **R5.2 Wilcoxon signed-rank** — on paired per-step losses (non-parametric companion
  to DM). **Dependency decision:** either add `scipy` (`scipy.stats.wilcoxon`) as a
  real dep now that you're doing forecasting stats, or implement the normal
  approximation with tie correction in numpy to keep the "no-scipy" property. DM is
  trivially numpy; Wilcoxon is the only thing pulling toward scipy. Recommend: numpy DM
  + scipy Wilcoxon, and gate scipy behind the `[forecast]` extra.
- **R5.3 Multiple comparisons** — reuse the existing Benjamini–Hochberg FDR across the
  model × ticker × horizon grid (the same anti-p-hacking rule as `plan.md` §1.4). Don't
  read a single raw DM p-value off a 200-cell grid.
- **Tests:** `tests/test_forecast_significance.py` — DM symmetry (`DM(a,b) = −DM(b,a)`),
  DM≈0 for identical forecasts, a known worked example, HLN correction sign.
- **Pass bar:** hybrid vs each benchmark via DM **and** Wilcoxon, FDR-corrected, on the
  OOS window only.

---

## Phase R6 — Residual diagnostics (the empirical heart of "*When*")

This is what literally answers the title — does the Prophet residual contain learnable
structure, and does its presence predict where the hybrid wins?

- **R6.1 `engine/residual_diagnostics.py`** — ACF/PACF of base residuals, **Ljung–Box**
  Q-test (residual autocorrelation), variance-ratio / runs test. Per ticker, per regime
  (R7), per horizon.
- **R6.2 The key cross-tab** — plot/tabulate **residual autocorrelation strength** (e.g.
  Ljung–Box statistic, or |ACF(1)|) against **hybrid skill gain** (`ΔU2 = U2_base −
  U2_hybrid`). The paper's thesis: residual learning helps **iff** the base model's
  residual is structured (autocorrelated), and adds nothing when residuals are white
  noise (efficient/random-walk-like series).
- **Pass bar:** a monotone-ish relationship (more residual structure → more hybrid gain),
  or a clean negative result ("residuals are white noise everywhere → hybrid ≈ base").
  Either way it's the paper's central figure.

---

## Phase R7 — Multi-horizon, regime, asset-class analysis (maps to Results §5.x)

Reuse the existing scope/registry machinery; these are evaluation slices, not new
engines.

- **R7.1 Horizons** `h ∈ {1,5,10,20}` (R0.4 direct-h). Expect skill to decay with `h`.
- **R7.2 Asset class** — Currencies (FXE), Indices (VOO/QQQM), Crypto, Commodities
  (GLD). Hypothesis: more hybrid gain on less-efficient classes (crypto/commodities).
- **R7.3 Regime split** — bull / bear / high-vol / low-vol (VIX terciles or realized-vol
  quantiles). Hypothesis: residual structure (hence hybrid gain) concentrates in
  high-vol / trending regimes.
- **Pass bar:** results reported per slice, with the R6 residual-structure overlay so
  the "when" is *explained*, not just tabulated.

---

## Immediate action — prerequisites, then the evaluation-window (`--days`) sweep

**Trimmed asset universe (Loghi's call).** Every run below uses an explicit
`--tickers` list instead of `--all`, so `config.py`'s asset registry stays
untouched (no blast radius on the directional/trading track or the web GUI,
which both read the full registry) — this is scoped to the paper reruns only:

```
--tickers AAPL MSFT NVDA GOOGL META TSLA BTC-USD ETH-USD GLD SLV VOO QQQM FXE FXY
```

14 tickers: 6 stocks (dropped AMD/TSM/ASML/AVGO/INTC), 2 crypto (BTC/ETH,
dropped SOL/BNB), 2 commodities (GLD **+ SLV, new — silver, not yet in the news
registry**), 2 indices (VOO/QQQM, unchanged), 2 FX (FXE **+ FXY, new — Japanese
Yen**). Smallcap/sector class dropped entirely from these runs (config.py keeps
it; it's just not passed via `--tickers`). SLV/FXY aren't in `config.py`'s
`news_names` map, so their sentiment fetch will fall back to the bare ticker
symbol as the news search query — fine, just less precise than a mapped name;
not worth a config.py edit for two tickers.

**Data refresh first (prices are ~3 months stale).** Before any of the runs
below, refresh prices for this exact ticker list:

```
uv run python refresh.py --tickers AAPL MSFT NVDA GOOGL META TSLA BTC-USD ETH-USD GLD SLV VOO QQQM FXE FXY
```

Then drop `--no-refresh` (i.e. let the harness/training scripts refresh again,
or keep `--no-refresh` once the cache is confirmed current — cheaper) for the
first run in the sequence below.

Six prerequisites must land **before** any further sweep or rerun, because
several invalidate the existing `--days 100` results too (not just future
runs) — the 100-day baseline itself needs to be regenerated once these land,
not just the 200/400/800 sweep. **Status: all six are implemented and
smoke-tested (1 ticker, `--days 5`–`10`, `--max-train 504`) as of this
session — none have been run at real paper scale yet.** `uv run pytest
tests/ -q` (349 tests, excluding the pre-existing unrelated
`test_web_api.py` fastapi gap) and `ruff`/`mypy` are clean.

1. **Retrain-per-window (leakage fix). — process, not code; already
   supported.** `--hybrid-fit pretrained` (and plain `LSTM-reg`) load weights
   from `models/{ticker}_reg.pt` / `models/{ticker}_hybrid_res.pt`, trained by
   `train_lstm_regressor.py` / `train_hybrid_residual.py` with a **fixed**
   `--days`/`--horizon` that trims only that many trailing rows before
   fitting. Scoring at a *larger* `--days` than the weights were trained for
   re-exposes rows the network already trained on — a real leakage bug
   (caught when the `--days 200` sweep run was stopped; see session notes).
   Nothing to build: both training scripts already trim correctly. The fix is
   **discipline** — retrain with the matching `--days D --horizon 1` before
   scoring every window in the sweep below; step 0 makes this explicit.
2. **Macro into the hybrid. — done.** `scripts/forecast_harness.py:_build_hybrid()`
   now takes `df`/`macro_panel` and, when passed, builds a macro-aware Prophet
   base (`ProphetModel(macro_df=...)`, same panel `_build_macro_prophet` uses)
   **and** passes the same lag-1-aligned panel into `ResidualHybrid(macro_df=...)`
   for the LSTM residual learner. `engine/residual_learners.py:LSTMResidualLearner`
   gained an optional exog channel (`_ResNet`'s linear head widens from
   `hidden_size` to `hidden_size+exog_dim`; `exog_dim=0` reproduces the old
   architecture exactly, so existing univariate pretrained weights still
   load). `train_hybrid_residual.py --macro` trains a **separate**
   `{ticker}_hybrid_res_macro.pt` (different architecture, can't share a
   file with the univariate weights). New variant:
   `Prophet + LSTM-res (hybrid) + macro`, added alongside the plain hybrid,
   not replacing it. Smoke-tested on AAPL (`--days 5`/`10`, `--max-train
   504`): produces a genuinely different prediction from the plain hybrid
   (not a silent no-op fallback). **Gotcha hit during testing, now fixed:**
   `train_hybrid_residual.py`'s `--max-train` used to default to `0`
   (uncapped) — unlike the harness's own 504-row default — so training on
   AAPL's full ~46-year history hit dates before the macro series (VIX/DXY/
   etc.) even starts, and the alignment check correctly (not a bug) skipped
   the macro variant, silently. **Fixed:** `--max-train` now defaults to `504`
   automatically whenever `--macro` is passed (with an explicit `log.warning`
   explaining why), and the script warns loudly — both up front if you
   override to `--max-train 0` anyway, and in the final summary if any/all
   tickers' macro variant ends up skipped (`trained_macro == 0` or
   `skipped_macro > 0`) — so a doomed-to-skip run can no longer pass silently.
   No manual `--max-train 504` reminder needed in the sweep steps below
   anymore; left as a no-op if passed explicitly.
3. **Per-asset regime labels (fixes the bull/bear gap). — done.**
   `scripts/paper_aggregate.py:per_asset_trend()` fetches each ticker's own
   10y history via yfinance and computes its own 50/200-day MA cross
   (falling back to a 20-day rolling-return-sign heuristic if a ticker has
   under 200 rows of history), independent of the shared SPY-wide label
   `regime_labels()` still produces. `regime_table()` now merges both and
   tags each row's `source` column (`spy-wide` vs `per-asset`) so both are
   reported, not just one replacing the other. Smoke-tested directly
   (`per_asset_trend(["TSLA","AAPL"])`, no harness run needed): **confirmed
   TSLA's own trend is currently `bear` for its entire recent stretch through
   today**, while AAPL's is `bull` — exactly the idiosyncratic-bear signal
   the SPY-wide label was masking. TSLA's overall history splits ~1293
   bull / ~1220 bear days (10y), a real, usable regime split.
4. **Adaptive residual weight λ_t. — done.** `engine/residual_hybrid.py`
   generalizes `P̂ = P̂^Prophet + reŝ` to `P̂ = P̂^Prophet + λ_t·reŝ` behind a new
   `adaptive_lambda: bool = False` constructor flag (default off — the
   existing fixed-λ=1 hybrid is untouched). `λ_t = clip(1 - p/α, 0, 1)`, `p` =
   Ljung–Box p-value on the most recent `lambda_window` (default 20)
   residuals, `α` = 0.05 (the same significance threshold
   `residual_diagnostics.diagnose()` already uses to call a residual
   "structured") — p≈0 (structured) → λ≈1 (full correction); p≥α (white
   noise) → λ=0 (hybrid reduces to base). New variant:
   `Prophet + LSTM-res (hybrid, adaptive-λ)`, added alongside the fixed-λ
   hybrid. Smoke-tested on AAPL: at `--days 5` it collapsed to λ≈0 for every
   step (identical output to plain Prophet — plausible, not a bug: a
   20-point rolling window with `lags=10` is a low-power test, easy to land
   p≥0.05 even when the full-series test rejects strongly); at `--days 10` it
   diverged from both the plain hybrid and plain Prophet, confirming the
   mechanism isn't stuck at a constant. **Watch this in the real sweep** — if
   `lambda_window=20` proves too noisy at paper scale (100+ day windows),
   widening it is a one-line change.
5. **Confidence gating (Prophet / ARIMA / hybrid only). — done.** Prophet
   already emitted an interval via `ForecastResult.extra["yhat_lower"/"yhat_upper"]`;
   ARIMA reports one via `ForecastResult.quantiles` (`{0.1, 0.5, 0.9}`)
   instead — `engine/forecast_backtester.py`'s walk-forward loop now checks
   both carriers and persists the resulting `interval_width` (`hi - lo`) per
   step (`None` for XGBoost/LSTM-reg — never faked). `ResidualHybrid` now
   also copies the base's `quantiles`/`extra` onto its own `ForecastResult`,
   so the hybrid genuinely inherits Prophet's interval (it didn't before this
   fix — the "inherits" claim in the original prerequisite note was
   aspirational until now). New CSV column `interval_width` in the per-step
   output (`scripts/forecast_harness.py:write_per_ticker_steps`).
   `scripts/paper_aggregate.py:confidence_gating_table()` computes coverage/
   Theil-U2-on-acted-on-subset at `{0.25, 0.5, 0.75, 1.0}` thresholds for
   whichever models have an interval. Smoke-tested end-to-end (AAPL, 10 days,
   `--hybrid`): ARIMA/Prophet/hybrid rows populated, XGBoost/RW rows blank as
   expected; `paper_aggregate.py` ran clean and produced a sane 16-row table.
6. **Dual sentiment scorer comparison (VADER vs FinBERT) — done, not in the
   original prerequisite list, added per Loghi's request.** The earlier
   sentiment-ablation work hardcoded VADER. `engine/sentiment_data.py` /
   `scripts/forecast_harness.py:_fetch_sentiment_panel()` now takes a
   `method` param and filters cached news by the DB's `method` column before
   building the panel (live top-up also scores with the requested method);
   `--sentiment` now builds **two** variants per selected model —
   `XGBoost + sentiment (vader)` / `(finbert)` and the Prophet equivalents —
   via a new `--sentiment-methods` flag (default: both). Smoke-tested on
   AAPL (`--days 5`): both variants ran and produced different numbers;
   confirmed in the news DB that real `vader` (62 rows) and `finbert` (20
   rows) scores were independently computed, not one relabeled as the other.
   FinBERT genuinely works in this environment (downloaded `ProsusAI/finbert`
   via `transformers`, no fallback triggered).

**Then the adaptive `--days` doubling procedure** (stop as soon as the metric
stops moving), rerun with both fixes in place, **starting with `--days 50`,
followed by `--days 100`** (per Loghi's direction — the 100-day leg supersedes
the earlier pre-fix 100-day runs and, together with 50, becomes the baseline
pair the 200/400/800 sweep diffs against):

**Steps -1 and 0 — done (2026-09-30).** Both ran on the full 14-ticker
universe, `--preset standard`, all variants (macro, hybrid+macro, adaptive-λ,
dual vader/finbert sentiment) confirmed present in the output — the
macro-hybrid skip fix worked at scale (14/14 trained, 0 skipped, both
windows). Runs: `results/fc_custom_50d_h1_20260930-074808/`,
`results/fc_custom_100d_h1_20260930-082911/`. Headline: still a clean
negative — 0/266 DM cells significant after FDR at `--days 100` (identical
conclusion to the pre-fix 25-ticker run). Median hybrid $U_2$: 1.225 (d50) →
1.188 (d100); macro-hybrid direction flips between windows (1.117 at d50,
1.230 at d100) — both differences read as noise on 14 tickers, not a real
macro effect either way. **New finding, not yet explained:** the
adaptive-$\lambda_t$ hybrid variant underperforms the fixed-weight hybrid
substantially and consistently at both windows (U2 1.849 / 1.916 vs.\ 1.225 /
1.188) — the Ljung-Box-driven shrinkage is hurting, not helping; flagged in
`paper.tex` §Forecast Accuracy / §Hybrid Model, not root-caused here, a
candidate follow-up. `paper.tex`'s Forecast Accuracy and Hybrid Model
sections were updated with the new numbers; **stopped here per instruction —
did not proceed to 200/400/800.** Also corrected a paper.tex inaccuracy while
in that section: the LSTM residual learner's actual input is a window of
past residuals (+ optional macro), not the full standalone-LSTM feature set
as an earlier draft claimed — see §Hybrid Model.
Items 7 (graphs) and 8 (promising-ticker case study) were not attempted this
pass — text+tables only, per priority.

-1. Retrain weights for `--days 50 --horizon 1` — **both**
    `train_lstm_regressor.py --days 50 --horizon 1` and
    `train_hybrid_residual.py --days 50 --horizon 1 --macro` (`--max-train`
    auto-defaults to 504 whenever `--macro` is set, per prerequisite #2's fix).
    Then run the core config (`--tickers <the 14 above> --horizon 1 --macro
    --hybrid --hybrid-fit pretrained --sentiment`, which now automatically
    includes the hybrid+macro and adaptive-λ variants alongside the plain
    hybrid, and both `(vader)`/`(finbert)` sentiment variants) at `--days 50`.
    This is the shortest window in the sweep — a real data point, not just a
    smoke test, but expect noisier per-ticker U2 than the longer windows given
    fewer eval steps.
0. Retrain weights for `--days 100 --horizon 1` (both scripts, same flags as
   above), then run the core config at `--days 100`. This is the new
   post-fix 100-day baseline — compare it against both (a) the existing
   pre-fix `--days 100` run, to sanity-check the fixes changed what was
   expected (hybrid+macro should differ from the old macro-less hybrid;
   leakage fix shouldn't change 100 itself much, since 100 was the window the
   old weights were already trained for — a large shift here would be a red
   flag worth stopping on), and (b) the new `--days 50` run from step -1, as
   the first leg of the actual convergence check.
1. Retrain weights for `--days 200`, run it, diff against the `--days 100`
   baseline from step 0 (and, informally, against the `50`→`100` delta from
   step -1 — is the metric still moving by a similar amount, or slowing down?).
   Then retrain + run `--days 400`, diffing `400` against `200`.
2. If `200` differs meaningfully from `100` but `400` ≈ `200` → **stop**, `200`-ish
   is enough; optionally fill in `300` to locate the elbow more precisely.
3. If `400` still differs meaningfully from `200` → the metric hasn't converged yet;
   go to `800` and repeat the same comparison (retrain weights for `800` too).
4. "Meaningfully different" = model rankings reorder, or median U2 shifts by more
   than the run-to-run DM-noise floor already characterized in the h=1 core run —
   don't chase third-decimal wobble.
5. Report the convergence point (or lack of one) as a new Robustness Checks
   paragraph; if it never converges within a reasonable ceiling (~800–1000, limited
   by shortest-history tickers), say so plainly rather than picking a window that
   happens to flatter the result.
6. **Model Confidence Set (MCS).** Post-hoc only — doesn't need a rerun, compute
   it from whichever run's saved per-step predictions are current at the time
   (Hansen–Lunde–Nason procedure, or a simpler elimination approach if a full MCS
   implementation is overkill for pure numpy). Complements the existing FDR grid
   in `engine/forecast_significance.py`: instead of "is model X significantly
   different from RW," report the *set* of models statistically indistinguishable
   from the best at each ticker/horizon — a cleaner summary than a 200-cell p-value
   table. Add wherever `compare_to_reference` results are already reported (the
   Forecast Accuracy / Statistical Testing sections of `paper.tex`).
7. **Graphs.** Entirely producible from already-saved CSVs (`results/fc_*`,
   `results/robust_*`, the residual-diagnostics and sentiment-ablation outputs) —
   no pipeline changes needed, run this after the reruns above land so the
   numbers match the final tables (matplotlib, already a transitive dep via the
   forecast extras). Minimum set, into `docs/paper/figures/`, referenced from the
   matching `paper.tex` subsection:
   - **U2 by model** — box/strip plot across all 14 tickers, reference line at
     U2=1. The headline Forecast Accuracy figure.
   - **Residual structure vs. hybrid gain** — scatter of Ljung–Box stat (or
     |ACF1|) vs. ΔU2 per ticker, from `structure_vs_gain`. This *is* the paper's
     central "when does it help" figure per `docs/forecasting-regression.md`.
   - **Actual vs. predicted overlay** — 2–3 representative tickers (e.g. TSLA for
     the bear stretch, AAPL for bull): true close vs. RW vs. Prophet vs. hybrid
     over the eval window.
   - **U2 vs. horizon** — line plot per model across h∈{1,5,10,20}.
   - **Regime bar chart** — median U2 by vol tercile (and the new per-asset
     bull/bear split), grouped by model.
   - **Robustness lookback plot** — U2 vs `--max-train` (252/504/full, log-scale
     y) — sells the "uncapping is catastrophic" finding visually.
   - **Sentiment ablation** — ΔU2 with vs. without news per ticker, centered near
     zero — sells the null result visually.
8. **Flag a promising ticker (case study).** Post-hoc only — every window's
   `_fc_summary.csv` is already ticker × model × horizon, so no new run is
   needed once the `--days` sweep (100/200/400/800, whichever actually run) is
   done. Any asset class is fair game (crypto/FX/commodity/stock/index — don't
   pre-restrict to stocks). Selection criteria, applied across **every window
   the sweep actually ran**, not just one:
   - Hybrid Theil U2 < 1 (beats RW) in that window, ideally in **most/all**
     windows run, not a single-window fluke.
   - DM-significant vs. RW **and** vs. Prophet-alone (FDR-corrected), not just a
     numerically lower U2 — the paper's own pass bar (`plan.md`'s Phase R2/R3
     falsification criteria already say the same: a real result needs both).
   - Prefer a ticker whose gain also shows up in the residual-structure cross-tab
     (item 7's second figure) — a high Ljung-Box/ACF1 paired with a real ΔU2 is
     the mechanistic story, not just a coincidence.
   - If nothing clears this bar (plausible, given the current headline result is
     a clean negative) — report the **closest call** honestly (e.g. "ticker X
     came closest, U2=0.97 in 2/4 windows, but didn't survive FDR") rather than
     manufacturing a win. A one-off "lucky ticker" call-out otherwise reads as
     cherry-picking, which the paper's own statistical-testing section exists to
     guard against.
   - Add as a short "Case Study" paragraph (with one of the actual-vs-predicted
     overlay plots from item 7 for that specific ticker) in Results — a nice
     complement to the aggregate tables, not a replacement for the honest
     overall median-U2 conclusion.

---

## Phase R8 — Robustness (maps to Robustness §7)

- Lookback windows (LSTM sequence length), refit cadence `K` (R0.3), hyperparameters
  (LSTM units/depth, XGBoost depth/lr, ARIMA order policy), asset subsets, and the
  **level-space vs log-return-space** variant from R0.1.
- Seed sweep (numpy/torch) — report mean±std of headline metrics, not a single lucky run.
- **Reproducibility statement** for the paper: pinned seeds, public git SHA, resolved
  config, data date-range, compute cost (`elapsed_seconds` already logged). This is
  `plan.md` §3.1 — share it with the directional track.

---

## Paper ↔ code map

| Paper section | Provided by |
|---|---|
| §3.2 Data Preprocessing | R4 (macro), R0.2 (lags), existing `features.py` |
| §3.3 Prophet Model | R3.3 multivariate Prophet |
| §3.4 Residual Construction | R0.2 rule + R3.4 utility |
| §3.5 LSTM Residual Predictor | R3.1 `lstm_regressor` |
| §3.6 Hybrid Model | R3.2 `ResidualHybrid` |
| §3.7 Benchmark Models (RW/ARIMA/Prophet/LSTM/XGBoost) | R2 + R3.1/R3.3 |
| §4.1 Walk-Forward Validation | R0.3 + R1.2 |
| §4.2 Evaluation Metrics (RMSE/MAE/MAPE **+ MASE/U2**) | R1.1 |
| §4.3 Statistical Testing (DM, Wilcoxon) | R5 |
| §5.1 Forecast Accuracy | R1 |
| §5.2 Residual Predictability | R6 |
| §5.3 Asset-Class Analysis | R7.2 |
| §5.4 Regime Analysis | R7.3 |
| §7 Robustness Checks | R8 |

---

## Falsification criteria (this track)

Residual learning is considered **unsupported** for daily financial price forecasting if,
out-of-sample and FDR-corrected:

1. The hybrid's Theil U2 ≥ 1 (does not beat a random walk), **or**
2. The hybrid does not beat **base-alone** by a DM-significant margin, **or**
3. Any apparent gain is in-sample only (vanishes under the R0.3/R0.5 disjoint windows), **or**
4. Base residuals are white noise across all assets/regimes (Ljung–Box non-significant
   everywhere), so there is no structure for the residual learner to exploit.

A clean negative or **strongly conditional** result ("helps only when residual
autocorrelation is present, i.e. crypto/commodities in high-vol regimes") is the
publishable finding and is more credible than a fragile positive.

---

## Priority summary

| Priority | Items |
|---|---|
| **First (unblockers)** | R0 contract · R1 regression metrics + forecast backtester · R2.1 RandomWalk |
| **Core paper artifact** | R3 residual hybrid · R3.3 multivariate Prophet · R5 DM/Wilcoxon |
| **Benchmarks** | R2.2 ARIMA · R2.3 XGBoost |
| **Empirics / "when"** | R6 residual diagnostics · R4 macro + ablation |
| **Slices** | R7 horizon/regime/asset-class |
| **Robustness** | R8 |
| **Later (paper v2)** | Prophet+Chronos, Prophet+Kronos via the R3.2 composition |

## Build order checklist

- [x] R0 — write the evaluation contract (target space, leakage rule, refit cadence, horizons)
- [x] R1.1 — `regression_metrics.py` (RMSE/MAE/MAPE/sMAPE + MASE/RMSSE/Theil U2) + tests
- [x] R2.1 — `naive_forecasters.py` (RandomWalk reference)
- [x] R1.2/R1.3 — `forecast_backtester.py` + persist per-step predictions
- [x] R3.1 — `lstm_regressor.py` (regression head)
- [x] R3.2/R3.4 — `residual_hybrid.py` + leakage-safe residual construction + tests
- [x] R3.3 — multivariate Prophet (`add_regressor`)
- [x] R5.1/R5.2 — Diebold–Mariano + Wilcoxon + FDR + tests
- [x] R2.2 — ARIMA benchmark
- [x] R2.3 — XGBoost benchmark
- [x] R4 — `macro_data.py` (VIX/DXY/Gold/SP500/DGS1) + lag-safe alignment.
- [x] R4.3/R4.4 — `sentiment_data.py` (per-ticker daily pos/neg panel, leakage-safe by
      construction) wired into XGBoost/Prophet via `--sentiment` (mirrors `--macro`);
      ablation run and reported (`docs/paper/paper.tex` §News/Sentiment Ablation): no
      DM-significant effect after FDR correction (2/100 cells survive, both a tiny
      Prophet/ETH-USD improvement). **Caveat:** GDELT/Yahoo have no point-in-time
      historical backfill, so real news coverage only spans ~7 of the eval window's
      ~21 weeks for the best-covered tickers — a preliminary negative, not definitive
      (see plan.md's own R8 robustness note in the paper).
- [x] R6 — `residual_diagnostics.py` (ACF/PACF/Ljung–Box) + the structure-vs-gain cross-tab — run for the paper
      (`docs/paper/paper.tex` §Residual Predictability): 25/25 tickers show structured (Ljung–Box) Prophet
      residuals, mean ΔU2 = 1.80.
- [x] R7 — horizon/regime/asset-class slicing — run via `scripts/forecast_harness.py` (h∈{1,5,10,20}, all 6
      asset classes) + `scripts/paper_aggregate.py` (bull/bear + VIX-tercile regime split). Bull/bear was
      uninformative this run (eval window fell entirely in a bull regime); vol-tercile split worked.
- [~] R8 — robustness sweeps: **partial**. Ran lookback (`--max-train` 252/504/0) and refit-cadence
      (`--refit-k` 5/21) on the `--stocks` subset only — not the full asset-subset × hyperparameter grid, and
      no seed sweep. See `docs/paper/paper.tex` §Robustness Checks for exactly what ran.

---

# Track B — Engineering & non-paper backlog (carried over)

These are the still-open items from the original `plan.md` and roadmap that are **not**
tied to the residual-hybrid paper, kept here so they don't get dropped while the paper is
the focus. They run in parallel and at lower priority than the Phase-R unblockers — **except**
the cross-track DX items (B1), which are worth doing early because they cut friction for the
many new models R2/R3 add.

Legend: ⤳ = also accelerates the paper track (do once, benefit both).

## B1 — Cross-track DX & reproducibility (highest non-paper value)

- **⤳ Model registry** (was `plan.md` §3.2) — collapse the manual 4-step add
  (`file → api._get_model → backtest_helpers variants → main.py`) into a
  `@register_model("name")` decorator populating a central `MODEL_REGISTRY` that `api`
  and `backtest_helpers` read. Unify with the existing `config.FORECAST_MODELS` +
  `ForecastModel` pattern rather than adding a parallel mechanism. **Do this early** —
  R2/R3 register RandomWalk, ARIMA, XGBoost, LSTM-regressor, and the hybrid; the
  decorator pays for itself immediately.
- **⤳ Experiment tracking & reproducibility** (was `plan.md` §3.1) — a SQLite
  `experiments` table: one row per run with `run_id`, `git_sha` (+ dirty flag),
  `timestamp`, `config_json`, scope/days/fee, data date-range, and headline metrics.
  Pin seeds (numpy / torch / random). **This is the same deliverable as R8's
  reproducibility statement** — build it once and both tracks use it. Keep it
  proportionate (no Weights & Biases; local MLflow only if a browsable UI is wanted).
- **DRY model-family names** (roadmap open) — `run_all.py:_family()` and
  `backtest_helpers._model_family()` still hardcode family prefixes (incl. the legacy
  "Chronos" alias, "TiRex", "Other"). Fold onto `config.MODEL_FAMILY_LABELS`, the single
  source the `--models` filter already uses. Cheap; do alongside the registry.

## B2 — Directional-track research (independent of paper)

- **LSTM focus** (was `plan.md` §2.3) — the only family with positive median
  return/Sharpe at 300d. Tune lookback / features / epochs; try LSTM-only confidence
  gating. Spend tuning effort here, not on k-NN. *(Note: the directional LSTM is separate
  from the paper's R3.1 LSTM-regressor — different target, but feature/lookback findings
  may transfer.)*
- **Reframe the success metric** (was `plan.md` §2.5) — beating B&H absolute return in a
  bull market via daily flips is very hard. Track risk-adjusted (Sharpe/Sortino) and
  drawdown vs B&H; define a falsifiable bar up front.
- **k-NN k-sweep** (was `plan.md` §2.4, low priority) — `k ∈ {3,5,7,9,11}`, only under the
  OOS harness. High overfit risk, worst family — don't expect much.

## B3 — Product / GUI / deploy (roadmap open)

- **⤳ Forecasting models in `main.py` report + web GUI Predict/Backtest tabs** — currently
  backtest-only. Surfacing them (and eventually the hybrid + its point forecast) in the
  Predict tab and `main.py` makes the paper's models demoable, not just scriptable.
- **TiRex forecasting model** — parked (not on PyPI, macOS-experimental, non-standard
  NX-AI license). Revisit only if a clean install path appears.
- **Authentication** — API key for any public deploy of the web GUI.

## B4 — Features & data backlog (unscheduled)

- **⤳ Cross-asset features** — feed correlated assets as inputs (gold↔USD, BTC↔BNB,
  index↔constituent), not just the ticker's own history. **Overlaps R4** — the macro
  ingestion (VIX/DXY/Gold/SP500/DGS1) is the first cross-asset feed; generalise the same
  alignment/lag plumbing rather than building a second path.
- **Volatility momentum** — momentum measured on volatility / vol-adjusted momentum
  (`features.py` / `ALL_FEATURES`).
- **Reddit sentiment** — new provider behind `get_provider()` alongside yahoo/gdelt,
  scored with the existing VADER/FinBERT pipeline. Pushshift is gone — use the official
  Reddit API via PRAW for live posts, Arctic Shift for historical backfill.
- **Intraday bars** — minute/intraday exploration. Caveat: yfinance serves only ~7 days of
  1-minute (~60 days coarser intraday), so a dedicated intraday source is needed for real
  history. *(The "specific-price regression" half of this old backlog item has graduated
  into the Phase-R forecasting track — only intraday remains backlog.)*

## B5 — After an edge / after the paper

- **Multi-asset portfolio backtesting** (was `plan.md` §4.1) — current backtests are
  single-ticker × period. Portfolio allocation, rebalancing, and portfolio risk are
  valuable but premature until individual signals beat the naive baselines / a hybrid
  beats RW. Allocating across coin-flip signals isn't meaningful.

## Track B priority

| Priority | Items |
|---|---|
| **Early (unblocks paper too)** | B1 model registry · B1 experiment tracking/seeds · B1 DRY family names |
| **Parallel** | B2 LSTM focus · B2 reframe metric · B3 forecasting models in GUI |
| **Later** | B2 k-NN sweep · B3 auth · B4 cross-asset/vol-momentum/Reddit |
| **After an edge** | B5 multi-asset portfolio · B4 intraday |
