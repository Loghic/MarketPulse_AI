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

- [ ] R0 — write the evaluation contract (target space, leakage rule, refit cadence, horizons)
- [ ] R1.1 — `regression_metrics.py` (RMSE/MAE/MAPE/sMAPE + MASE/RMSSE/Theil U2) + tests
- [ ] R2.1 — `naive_forecasters.py` (RandomWalk reference)
- [ ] R1.2/R1.3 — `forecast_backtester.py` + persist per-step predictions
- [ ] R3.1 — `lstm_regressor.py` (regression head)
- [ ] R3.2/R3.4 — `residual_hybrid.py` + leakage-safe residual construction + tests
- [ ] R3.3 — multivariate Prophet (`add_regressor`)
- [ ] R5.1/R5.2 — Diebold–Mariano + Wilcoxon + FDR + tests
- [ ] R2.2 — ARIMA benchmark
- [ ] R2.3 — XGBoost benchmark
- [ ] R4 — `macro_data.py` (VIX/DXY/Gold/SP500/DGS1) + lag-safe alignment + pos/neg sentiment
- [ ] R6 — `residual_diagnostics.py` (ACF/PACF/Ljung–Box) + the structure-vs-gain cross-tab
- [ ] R7 — horizon/regime/asset-class slicing
- [ ] R8 — robustness sweeps + reproducibility statement

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
