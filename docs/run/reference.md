# Reference

`config.py` knobs, the test commands, troubleshooting matrix, and the
on-disk layout. Keep this open in a side tab when something goes
sideways.

## Configuration (`config.py`)

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
# (key → display name). MODEL_FAMILIES is derived from it. Phase-1.2
# adds "baseline" for the naive baselines so --models baseline works.
MODEL_FAMILY_LABELS = {
    "knn":      "k-NN",
    "linreg":   "LinReg",
    "lstm":     "LSTM",
    "prophet":  "Prophet",
    "chronos":  "Chronos-2",
    "kronos":   "Kronos",
    "baseline": "Baseline",
}
MODEL_FAMILIES = list(MODEL_FAMILY_LABELS)
```

Every CLI flag has a `config.py` default. Changing the default avoids
having to pass the flag every time.

## Testing

```bash
# Quick smoke test, no extra deps (13 tests)
uv run python test_pipeline.py

# Full pytest suite (~140 tests, needs the `dev` extra)
uv run python -m pytest

# A specific module
uv run python -m pytest tests/test_news_pipeline.py -v
uv run python -m pytest tests/test_baselines.py -v       # Phase-1.2
uv run python -m pytest tests/test_oos_harness.py -v     # Phase-1.1
```

### Static checks (also run by pre-commit)

```bash
uv run ruff check --fix .
uv run ruff format .
uv run mypy engine/ interface/ web/backend/
```

`ai_model.py`, `chronos_model.py` and `kronos_model.py` are excluded
from strict mypy (torch / external sibling import). Running `mypy
engine interface` alone is fine; the `web.backend.*` "unused section"
note is benign.

### Tests don't share state with your real DB

The `tests/conftest.py` `api` fixture chdir's to `tmp_path` before
constructing `StockAppAPI`, and the `test_web_api.py` `client` fixture
does the same with `tmp_path_factory`. Without those, the mocked
`yf.Ticker(*).history()` would have written the synthetic 400-day
fixture into `data/market_data.db` under every real ticker name
(reproduced as a regression test in `scripts/clean_test_contamination.py`).

### macOS FD pressure

The conftest also bumps `RLIMIT_NOFILE` to `min(4096, hard)` because
macOS defaults to `ulimit -n = 256` and the suite (HuggingFace caches,
SQLite, FastAPI TestClient sockets, tmp_path dirs) collectively
breaches that. Linux / CI / sandboxed shells silently no-op.

If you still see `OSError: [Errno 24] Too many open files` in your
shell after running the suite, you've inherited a low ulimit from the
parent process — re-run with `ulimit -n 4096` before pytest.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `WARNING: FinBERT unavailable (No module named 'transformers'). Falling back to VADER.` | `ai` extra not installed | `uv pip install -e ".[ai]"` |
| `WARNING: VADER unavailable (...). Falling back to naive.` | NLTK lexicon not downloaded | `uv run python -c "import nltk; nltk.download('vader_lexicon')"` |
| `+ News` variants show 0 sentiment for every historical day | Yahoo headlines only cover ~7 days; the rest of your backtest sees an empty DB lookup | Bulk-fetch with GDELT: `uv run python refresh.py --all --news-source gdelt --news-history-days 365 --force-news` |
| GDELT returns nothing | Network blocks `api.gdeltproject.org`, or you searched an obscure ticker not in `TICKER_NAMES` | Verify with the curl command in [setup.md](setup.md#gdelt-reachability-check-optional); extend the class's `news_names` in `config.py`'s `ASSET_CLASSES`. |
| `sqlite3.OperationalError: disk I/O error` | Usually a stale `data/` mount or read-only volume | Remove `data/market_data.db` and let it rebuild |
| `sqlite3.OperationalError: unable to open database file` mid-batch | Transient — macOS Spotlight indexing the WAL/SHM files, Time Machine snapshot, or FD pressure | Since 2026 the walk-forward loop pre-fetches news once per ticker (no thousands of per-day DB calls) and `run_all.py` wraps each ticker in try/except, so a single failure no longer kills the batch. If you still see it, re-run with `--no-refresh` once the DB is populated and check `Activity Monitor → Disk` for whatever's hammering `data/`. |
| `total_return` in the hundreds (e.g. 385.58 for 5 days) | A single bad row in `data/market_data.db` with close = 0 or NULL → trade_pnl = `(exit - 0) / 0` is clipped to a huge value | Run `uv run python scripts/clean_prices.py` to report bad rows, then `--apply` to delete them. New writes are filtered automatically (`db_manager.save_prices`) and the backtester itself drops days with > 50% single-day moves since 2026. Re-run `refresh.py` after cleanup. |
| Test runs show up in the Analysis tab picker (e.g. `custom_5d_…` dirs you never created) | `tests/test_web_api.py::TestBacktest` was missing the autouse `_redirect_persistence` fixture before 2026 | Fixed in 2026: the autouse fixture points `CACHE_DIR` + `RESULTS_DIR` at `tmp_path` for every backtest test. Manually clean any leftover test artifacts with `rm -rf results/custom_* backtests/*custom*`. If you add a new backtest-triggering test elsewhere, copy that fixture. |
| Synthetic 400-day prices showing up under every real ticker in `data/market_data.db` | Old test runs (pre-2026) where the `api` fixture didn't chdir to tmp_path | Run `uv run python scripts/clean_test_contamination.py` for a dry-run, then `--apply` to delete. Fix is permanent: the conftest fixture now uses `monkeypatch.chdir(tmp_path)`. |
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
| `Prophet` and `Prophet + News` rows are identical | Prophet's confidence is too high for the ±(sentiment × 0.20) nudge to flip the call | Expected, not a bug — see [docs/forecasting.md](../forecasting.md) |
| A forecasting run takes forever | Kronos + Prophet dominate per-day compute | Drop them with `--models knn linreg lstm chronos`, trim periods with `--periods 1y 2y 5y`, and add `--timing` to confirm where the time goes |
| `OSError: [Errno 24] Too many open files` on macOS during pytest | Default `ulimit -n = 256` exhausted by HuggingFace + SQLite + tmp_path dirs | conftest now bumps it to 4096 automatically; if you still hit it, `ulimit -n 4096` before pytest. |
| OOS harness `Ticker: only N rows, need 2*days+20 rows` log line | The ticker has too little history for two disjoint windows | Either shorten `--days` or drop that ticker from the run. The harness skips it cleanly. |

## Where the data lives

```
data/market_data.db                       ← SQLite (prices + news, auto-created)
models/{ticker}_{period}_{preset}.pt      ← saved LSTM weights (gitignored)
predictions/{ticker}/{date}.json          ← cached predict outputs
backtests/{ticker}/{date}_{days}d.json    ← cached backtest outputs
results/{scope}_{days}d_…/                ← run_all.py output trees
results/oos_{scope}_{days}d_…_{ts}/       ← oos_harness.py output trees
```

Delete any of these at any time — they'll regenerate on the next run.
