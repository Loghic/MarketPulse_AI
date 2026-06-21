# Setup

Prerequisites, install steps, first-run downloads, and the ticker
registry. Once this page is done, jump to whichever workflow fits:
[predict](predict.md), [refresh](refresh.md), [backtest](backtest.md),
or [research](research.md).

## 1. Prerequisites

* **Python 3.12+** (the project pins this in `pyproject.toml`)
* **[uv](https://docs.astral.sh/uv/getting-started/installation/)** —
  fast Python package manager. All commands below use `uv run`, which
  auto-creates and activates the venv as needed.
* **macOS / Linux / WSL.** Bare Windows works but isn't CI-tested.
* For the Web GUI you also need **Node.js 18+** (only if you plan to
  run the React frontend).

Clone and `cd` into the project:

```bash
git clone <repo-url>
cd marketpulse-ai
uv venv
```

## 2. Install

The project has one required dependency set and several optional
extras. Install only what you need.

### Core (always required)

```bash
uv pip install -e .
```

This installs `pandas`, `yfinance`, `scikit-learn`, `numpy`, `nltk`,
`tqdm`. After this, k-NN and Linear Regression models plus VADER +
naive sentiment all work, along with Yahoo + GDELT news sources, and
the naive baseline classes.

### LSTM + FinBERT (the `ai` extra)

```bash
uv pip install -e ".[ai]"
```

Adds `torch>=2.0` and `transformers>=4.40`. Required if you want to:

* train and run the LSTM model (`train.py`, the `lstm` model variant)
* use FinBERT as a sentiment scorer (`--sentiment-method finbert`)

This is a large download (~1 GB including PyTorch wheels). On Linux
without a GPU, install the CPU-only PyTorch first if you want to save
space:

```bash
uv pip install "torch==2.*+cpu" --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[ai]"
```

The first FinBERT call downloads the ProsusAI/finbert model (~400 MB)
and caches it under `~/.cache/huggingface/`. Subsequent runs are
instant.

### Forecasting models (the `forecast` extra)

```bash
uv pip install -e ".[forecast]"
```

Adds `prophet` and `chronos-forecasting`. Required for the Prophet and
Chronos-2 forecasting models in backtests. Prophet needs no download;
Chronos-2 downloads its weights (~478 MB) from the Hugging Face Hub on
first use and caches them under `~/.cache/huggingface/`. CPU works out
of the box.

### Kronos (sibling clone, not pip)

Kronos isn't on PyPI — clone it next to the repo and install the
minimal extra:

```bash
git clone https://github.com/shiyu-coder/Kronos.git ../Kronos
uv pip install -e ".[kronos]"        # einops + huggingface_hub (torch already present)
```

Do **not** run `pip install -r ../Kronos/requirements.txt` — on Python
3.14 its pinned matplotlib 3.9.3 has no wheel and the bundled freetype
source build fails. The adapter doesn't need matplotlib. The clone is
expected as a sibling of this repo (`../Kronos`); override the location
with `KRONOS_PATH` if it lives elsewhere. The first backtest downloads
the Kronos-small weights from the Hub and forces CPU on Macs (the
upstream default targets `cuda:0`).

### Web GUI (the `web` extra)

```bash
uv pip install -e ".[web]"
cd web/frontend && npm install && cd ../..
```

Adds `fastapi`, `uvicorn`. The `npm install` step pulls React, Vite,
TanStack Query and Plotly.

### Plotting helpers (the `viz` extra)

```bash
uv pip install -e ".[viz]"
```

Adds `plotly` and `matplotlib`. Optional — only needed if you want to
render charts from a Python script outside the Web GUI.

### Dev tooling (the `dev` extra)

```bash
uv pip install -e ".[dev]"
uv run pre-commit install
```

Adds `pytest`, `pytest-cov`, `ruff`, `mypy`, `httpx`, `pre-commit`.
Required to run the test suite and the pre-commit hooks (ruff + mypy
on every commit).

### Everything at once

```bash
uv pip install -e ".[ai,web,viz,dev,forecast]"
git clone https://github.com/shiyu-coder/Kronos.git ../Kronos   # Kronos is a separate clone
uv pip install -e ".[kronos]"
cd web/frontend && npm install && cd ../..
uv run pre-commit install
```

## 3. First-time setup (one-off)

A few network-dependent assets get pulled lazily on first use. If
you're on a network with strict egress rules, do these explicitly so
they fail loudly rather than during a backtest run.

### VADER lexicon (~1 MB)

The first VADER call auto-downloads NLTK's `vader_lexicon`. Force it
now:

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

This pulls the Kronos-small tokenizer + model from the Hugging Face
Hub. If it can't find the repo, check that `../Kronos` exists (or
`KRONOS_PATH` points at the clone).

### GDELT reachability check (optional)

GDELT's free Doc API is over `https://api.gdeltproject.org`. If your
network blocks it the provider returns an empty list silently. Verify
reachability:

```bash
curl -sI "https://api.gdeltproject.org/api/v2/doc/doc?query=Apple&format=JSON&mode=ArtList&maxrecords=1" | head -3
```

You should see a `200 OK`.

## 4. Tickers

Defined in `config.py` as a data-driven asset registry
(`ASSET_CLASSES`). Add a ticker by adding it to the relevant class's
`tickers`; add a whole asset class by appending one `AssetClass`
entry — flags, benchmarks, asset-type tags and GDELT news queries all
derive from it. No other code changes needed.

**Stocks:** AAPL, MSFT, NVDA, META, GOOGL, AMD, TSM, ASML, AVGO, TSLA, INTC
**Crypto:** BTC-USD, ETH-USD, SOL-USD, BNB-USD
**Commodities:** GLD (gold ETF)
**Indices:** VOO (S&P 500), QQQM (Nasdaq-100)
**FX:** FXE (EUR/USD)

The non-stock/crypto classes use ETF proxies (volume-bearing, so
features + LSTM behave as for stocks). `VOO`/`QQQM` are deliberately
distinct from the `SPY`/`QQQ` benchmark set.

Every CLI script accepts the same scope flags, and the per-class
flags **combine**:

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
