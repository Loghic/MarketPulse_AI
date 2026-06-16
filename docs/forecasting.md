# Forecasting Models

Prophet, Chronos-2 and Kronos are **forecasting** models: unlike k-NN, Linear
Regression and LSTM — which classify next-day direction directly — these predict
a future *value* (a point estimate plus, where available, a predictive
distribution). A shared adapter turns that value forecast into the same
`(direction, confidence)` contract the rest of the app expects, while keeping the
raw value accessible for later use (charts, regression targets, multi-step
horizons).

> **Status:** Prophet, Chronos-2 and Kronos are available in **backtests** and
> via the `StockAppAPI`. `main.py`'s report and the web GUI don't list them yet.
> **TiRex** is parked (see the note at the end); it would plug into the same base
> class if revisited.

## Install

### Prophet + Chronos-2 — the `forecast` extra

```bash
uv pip install -e '.[forecast]'      # prophet + chronos-forecasting
```

Prophet needs no download. Chronos-2 downloads its weights (~478 MB) from the
Hugging Face Hub on first use, then caches them under `~/.cache/huggingface/`;
the loaded pipeline is reused for every ticker. CPU works out of the box.

### Kronos — sibling clone + the `kronos` extra

Kronos is a research repo, **not a pip package**, so it's cloned next to this
project and imported by path:

```bash
cd ..                                  # the parent dir, next to mp_ai
git clone https://github.com/shiyu-coder/Kronos.git
cd mp_ai
uv pip install -e '.[kronos]'          # einops + huggingface_hub (torch already present)
```

That gives you `sc/mp_ai/` and `sc/Kronos/` side by side; the adapter appends
`../Kronos` to `sys.path` automatically (override with `config.KRONOS_PATH` or
the `KRONOS_PATH` env var). The first backtest downloads the Kronos-small weights
from Hugging Face.

> **Do not run `pip install -r Kronos/requirements.txt`.** On Python 3.14 it
> resolves `matplotlib==3.9.3`, which has no 3.14 wheel, so it builds from source
> and the bundled freetype compile fails (`unknown type name 'Byte'`). matplotlib
> is only used by Kronos's plotting/example scripts — the adapter needs only
> `torch` (already installed), `pandas`/`numpy`, `einops` and `huggingface_hub`,
> which is exactly what the `[kronos]` extra installs. Kronos must live in the
> **same venv** as the rest of the app, since the adapter imports it in-process.

## The shared interface

All three subclass `ForecastModel` (`engine/forecast_base.py`) and expose two
methods:

| Method | Returns | Use |
|---|---|---|
| `predict(df, use_time_weights, sentiment_score)` | `(str, float)` | Standard contract — `"UP"`/`"DOWN"` + confidence. Used by the backtester. |
| `forecast(df, horizon=1)` | `ForecastResult` \| `None` | The raw value forecast — reach for this when you want the predicted price rather than the label. |

`use_time_weights` is accepted but ignored (these models handle recency
internally), exactly like LSTM. `predict()` is wrapped so it never raises — a
single bad day returns an `"Insufficient data"` sentinel that the backtester
skips, rather than aborting the run.

### `ForecastResult`

```python
@dataclass
class ForecastResult:
    last_close: float                     # most recent close
    point: float                          # predicted next value (median/mean)
    horizon: int = 1
    quantiles: dict[float, float] | None  # {level: value}, e.g. {0.1: 101.2, 0.5: 103.0}
    samples: np.ndarray | None            # Monte-Carlo sample values (Kronos)
    prob_up: float                        # P(next > last_close)
    direction: str                        # "UP" / "DOWN"
    confidence: float                     # winning-class probability
    extra: dict                           # model-specific extras
```

### Direction and confidence

Direction is derived from the predictive distribution so it matches how the
existing models report confidence (the winning-class probability):

```
prob_up    = P(forecast_next > last_close)
direction  = "UP" if prob_up >= 0.5 else "DOWN"
confidence = prob_up if UP else (1 - prob_up)
```

`prob_up` is computed from, in order of preference: Monte-Carlo **samples**
(empirical fraction above `last_close` — this is the Kronos path when
`KRONOS_PROB_SAMPLES > 1`), then forecast **quantiles** (invert the quantile
function — interpolating `last_close` through value→level gives the CDF, and
`1 − CDF` is `prob_up`), then a bounded function of the bare **point** estimate
(a deliberately under-confident fallback). Interpolating through the quantiles
also caps confidence at the outermost level (e.g. 0.9), so these models never
claim 100%.

### Sentiment

Sentiment is applied post-hoc with the same logic and weight (0.20) as k-NN /
LinReg / LSTM: it shifts `prob_up` and can flip the direction. The raw value
from `forecast()` is sentiment-independent; only `predict()` applies the shift.
This is why a very confident model (Prophet often near 99%) can show identical
`Model` and `Model + News` rows — the nudge isn't large enough to flip it.

## Prophet

Meta's classical additive model (trend + optional seasonality). It **fits on
every call** — fast on a single price series — so there is no pre-training and
nothing to download.

- **Input:** univariate close series.
- **Hardware:** CPU only.
- **Direction:** a normal approximation on the prediction interval —
  `sigma ≈ (yhat_upper − yhat_lower) / (2·z)` (z from `interval_width`), then
  `prob_up = P(N(yhat, sigma) > last_close)`.
- **License:** MIT.

Configurable via `ProphetModel.__init__` (`interval_width`, `weekly/yearly/
daily_seasonality`, `growth`). Defaults are trend-only with `yearly="auto"` to
keep per-day fits fast. The cmdstanpy "Chain [1] …" log spam is silenced around
each fit by a `_quiet_stan()` context manager.

## Chronos-2

Amazon's 120M-parameter, encoder-only **zero-shot** time-series foundation model.
No training: it forecasts from the recent context directly.

- **Input:** univariate close series (last `CHRONOS_CONTEXT` closes).
- **Hardware:** CPU or GPU. On CPU it loads in float32; on GPU it requests
  bfloat16 via the modern `dtype=` kwarg.
- **Direction:** from the 1-step quantile forecast
  (`Chronos2Pipeline.predict_quantiles`, called with a positional list of series).
- **License:** Apache-2.0.

Configurable in `config.py`: `CHRONOS_MODEL_ID`, `CHRONOS_CONTEXT`,
`FORECAST_DEVICE`.

## Kronos

shiyu-coder's decoder-only **zero-shot** foundation model purpose-built for
financial **candlesticks** (OHLCV) — the only model here that uses your full
open/high/low/close, not just the close.

- **Input:** the recent OHLCV window (last `KRONOS_MAX_CONTEXT` bars).
- **Hardware:** CPU or GPU. `KronosPredictor` defaults to `cuda:0`, so the
  adapter forces a sane device (CPU on a Mac) via `FORECAST_DEVICE`.
- **Direction:** by default from the point estimate — `predict()` runs
  `KRONOS_SAMPLE_COUNT` stochastic paths and **averages them internally** into a
  single forecast path, from which the next close gives direction (with modest
  point-based confidence). Set `KRONOS_PROB_SAMPLES > 1` to instead run N
  independent stochastic passes and take the empirical `P(up)` from the spread of
  predicted closes — a real distribution, at ~N× the per-day cost.
- **License:** MIT.

Configurable in `config.py`:

```python
KRONOS_PATH = None                          # None -> ../Kronos (sibling of repo root)
KRONOS_MODEL_ID = "NeoQuasar/Kronos-small"  # small=24.7M/ctx512; base=102M; mini=4.1M/ctx2048
KRONOS_TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"  # use Kronos-Tokenizer-2k for mini
KRONOS_MAX_CONTEXT = 512                     # 512 for small/base, 2048 for mini
KRONOS_SAMPLE_COUNT = 5                      # internal averaging per predict() call
KRONOS_PROB_SAMPLES = 1                      # >1 = empirical P(up) from N stochastic passes (slower)
KRONOS_T = 1.0                               # sampling temperature
KRONOS_TOP_P = 0.9                           # nucleus sampling
```

## Use in backtests

Forecasting models are listed in `config.py → FORECAST_MODELS`:

```python
FORECAST_MODELS = [
    ("prophet", "Prophet"),
    ("chronos", "Chronos-2"),
    ("kronos", "Kronos"),
]
```

`backtest_helpers.run_single_backtest()` iterates this list, skips any model
whose library/clone isn't present, and adds one plain variant plus a `+ News`
variant (when news exists) for each. They flow through the same fee / stop-loss
duplication and `Backtester.run()` loop as every other model. The summary table
groups results by model family and ranks by return within each group.

```bash
uv run python backtest.py --tickers NVDA --days 20 --full
```

## Performance & cost

These models are far slower than k-NN/LinReg, so two tools exist to see and
control where the time goes:

- **Per-model timing.** `Backtester.run()` records `elapsed_seconds` on every
  `BacktestResult`. Add `--timing` to `backtest.py` for a slowest-first
  per-model breakdown after the summary; `run_all.py` prints a **time-by-model-
  family** rollup (time, share, and how often each family won) at the end of a
  batch. Use these to drop a model that costs a lot and rarely wins.
- **Choosing periods.** `run_all.py --periods 1y 2y 5y` skips the (default)
  `max` window. This matters because Prophet **refits on every walk-forward day**
  (the slowest model on large `--days`), and the foundation models **truncate to
  their context length** (~512 bars ≈ 2 trading years): feeding them `5y` or
  `max` gives essentially identical input, so `max` is wasted compute. A 100-day
  batch with `max` included can take hours; dropping it is close to free.

### What we actually found

On a 100-day, FinBERT, stocks-only batch (`--fees 0.03`, periods 1y/2y/5y),
treat the results as research, not signal:

- Direction accuracy is essentially a coin flip (mean ≈ 0.49 across all combos;
  only ~10% clear 55%). Headline returns are mostly **selection bias** (best of
  ~60 combos per ticker) amplified by a bull market.
- Only ~19% of combos beat their own buy & hold — B&H is a hard baseline.
- Among the new models, **Chronos-2** was the most useful (highest beat-B&H
  rate, positive mean Sharpe); **Prophet** and **Kronos** were the slowest and
  weakest on average. LSTM had the best averages of the lot.
- FinBERT news had a negligible effect on accuracy.

## Adding a forecasting model

This is the path TiRex would follow (and the path Kronos took for the
external-clone case):

1. Create `engine/<name>_model.py`, subclass `ForecastModel`, implement
   `_raw_forecast(df, horizon) → ForecastResult | None`. Set `quantiles` /
   `samples` (or `prob_up` directly) so the base can derive direction.
   Lazy-import the heavy dependency and expose a `_<NAME>_AVAILABLE` flag.
2. Register it in `interface/api.py`: an availability flag, a branch in
   `_load_forecast_model()` (annotated `model: ForecastModel`), and a branch in
   `_get_model()`.
3. Add `("<type>", "<Label>")` to `config.py → FORECAST_MODELS`.
4. **Repo-based models (no PyPI package), like Kronos:** clone as a sibling and
   *append* its root to `sys.path` (append, not insert, so it can't shadow this
   project's `config.py`/`utils.py`); expose the location via a `config.*_PATH`
   override; force a device, since some predictors default to `cuda:0`.

No changes to `backtester.py` or `backtest.py` are needed — the variant loop
picks it up automatically.

## TiRex (parked)

NX-AI's 35M-parameter xLSTM zero-shot model. Not currently integrated: it isn't
on PyPI (git clone + `pip install -e .`), macOS support is experimental and its
custom CUDA kernels don't apply on CPU/Mac, and it ships under NX-AI's
non-standard Community License. Revisit only if those constraints change; the
`ForecastModel` base would accept it with a quantile-based `_raw_forecast`.
