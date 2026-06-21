# Web GUI

Browser-based dashboard for MarketPulse AI. FastAPI backend + React frontend.

## Quick start

```bash
# Install
uv pip install -e ".[web]"
cd web/frontend && npm install && cd ../..

# Run (starts both servers)
chmod +x web/dev.sh
./web/dev.sh
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- Swagger docs: http://localhost:8000/docs

## Architecture

```
React (localhost:5173)  ──Vite proxy /api──▸  FastAPI (localhost:8000)  ──▸  StockAppAPI  ──▸  engine/
```

The backend wraps `StockAppAPI` without duplicating business logic. One shared instance (`routes/data.py:get_api()`) is reused across all routes.

## Pages

### Dashboard

Ticker overview with interactive chart and OHLCV data.

- **Ticker selector** — dropdown grouped by every asset class from `/api/meta` (Stocks / Crypto / Commodities / Indices / FX)
- **Period tabs** — 1MO, 1Y, 2Y, 5Y, MAX, Custom (date pickers)
- **Update buttons** — refresh single ticker or all tickers
- **News refresh panel** — scope chips driven by the asset registry (this ticker / per-class / all)
- **Stats cards** — Last Close, Day Change (±%), Period High/Low, Avg Volume (20d), Period Return
- **Price chart** (SVG):
  - Line / Candlestick toggle
  - Scroll to zoom, drag to select range
  - Navigation bar at bottom (drag to pan, click to jump)
  - Hover crosshair with OHLCV tooltip
  - Volume bars in background
  - Auto-reset on ticker/period change
- **OHLCV table**:
  - Sortable by any column (sorts entire dataset, not just current page)
  - Δ% column (day-over-day change, sortable)
  - Filter by date
  - Pagination (First / Prev / Next / Last)
  - Export CSV

### Predict

Unified prediction builder with per-model configuration.

**Prediction Builder:**

```
Quick add: [All] [All+News] [k-NN family] [k-NN+News] [LinReg family] [LinReg+News] [LSTM] [LSTM+News]
Period:    [1MO] [1Y] [2Y] [5Y] [MAX]

Models to run (3):
┌─────────────────────┬──────┬────────┬───┐
│ k-NN Enhanced       │  1Y  │ ☐ News │ ✕ │
│ LinReg Enhanced (TW)│  2Y  │ ☑ News │ ✕ │
│ LSTM                │ MAX  │ ☐ News │ ✕ │
└─────────────────────┴──────┴────────┴───┘
  [+ Add model]                [Run (3 models)]
```

- **Per-model configuration** — each row has its own model, period, and news toggle
- **Model variants from `/api/predict/info`** — availability-gated, so Prophet / Chronos-2 / Kronos appear when their dependency is installed (no hardcoded list)
- **News per model** — not global; choose which models use sentiment
- **Ticker dropdown** — grouped by every asset class
- **Chart** — optional, independent period from model period
- **Consensus** — auto-computed from run results (UP/DOWN/SPLIT + agreement %)
- **Results table** — sortable, filterable, export CSV
- **Prediction caching** — results saved to `predictions/{ticker}/{date}.json`
- **Historical prediction** — predict as if today were any past date

### Backtest

Walk-forward backtest builder, fully wired to the engine.

- **Tickers** — grouped by every asset class from `/api/meta`
- **Models** — family picker from `/api/meta` (k-NN, LinReg, LSTM, Prophet, Chronos-2, Kronos), unavailable ones shown "(n/a)"; separate **Baselines** toggle
- **Strategy knobs** — fee %, single SL % or **SL sweep** (`config.SL_SWEEP`), **min-confidence θ** gate, **turnover fees**, **hold days**, B&H benchmark
- **News** — scorer + lookback/half-life when enabled
- **Live progress bar** + **persisted-run picker** (reload past runs without re-running)
- **Summary** (best model per metric) + **results table** with conditional Coverage / Turnover columns; CSV export

### OOS (out-of-sample harness)

Runs the select-on-window-N → evaluate-on-disjoint-N+1 pipeline. Same
ticker/model/strategy/news pickers as Backtest (SL single-valued, not swept).
Shows the **aggregate** read (OOS beat-B&H rate, median OOS return,
selection-inflation gap, gate-aware Brier/ECE/coverage) and a **per-ticker**
table (winner, in-sample vs OOS return, beat-B&H, coverage, OOS p-value).
Live progress + persisted runs.

### OOS Compare

Pick two saved OOS runs (A vs B) and diff them: aggregate metrics side-by-side
with a "better" verdict per row, plus a per-ticker OOS-return diff. Built for
"did adding the gate / turnover fees / a different model set change the honest
beat-B&H picture?" — run the harness once per setting, then compare.

### Training (stub)

LSTM training with progress tracking. Planned: preset selector, model inventory table.

### Analysis (stub)

News vs No-News model comparison for academic paper. Planned: paired metrics, statistical tests, LaTeX export.

### Settings

Persistent user preferences saved to `data/settings.json`.

- **Global Defaults** — period, trading fee (slider + text input), stop-loss, backtest days
- **k-NN Model** — k parameter for naive and enhanced (1–50)
- **LSTM** — preferred model preset (cluster > standard > quick, with fallback)
- **Developer Settings** (collapsible) — log mode (GUI quiet / CLI verbose), JSON config preview
- **Save / Reset defaults** buttons with ✓ saved indicator
- Number inputs: free text entry, validates on blur (clamps to min/max, red flash on error)

## API Endpoints

### Meta (config-driven options)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/meta` | Single source of truth for the frontend pickers: model families (availability-gated), asset classes (with tickers + benchmarks), benchmarks, periods, sentiment methods, `sl_sweep`, `confidence_sweep`, defaults |

The frontend reads `/api/meta` instead of hardcoding model/ticker lists, so a new model or asset class in `config.py` surfaces in the UI automatically — gated by whether its optional dependency is installed (LSTM→torch, Prophet/Chronos/Kronos→`[forecast]`/clone; baselines are backtest-only).

### Data

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/data/tickers` | All tickers with row count, last date, asset type (registry class: stock / crypto / commodity / index / fx) |
| GET | `/api/data/ticker/{ticker}?period=1y&limit=0` | OHLCV data. limit=0 = no limit |
| POST | `/api/data/refresh` | Download latest prices + news. Body: `{"tickers": ["AAPL"]}` (empty = all) |
| POST | `/api/data/refresh-news` | Bulk news pull (scope by ticker list; scorer / source / history-days knobs) |

### Predictions

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/predict/info` | Available models, periods, next trading day |
| POST | `/api/predict/run` | Run predictions with per-model config (see below) |
| GET | `/api/predict/cached` | List cached prediction files |
| POST | `/api/predict/historical?ticker=AAPL&date=2025-01-15&period=1y` | Predict for past date |

**POST /api/predict/run** body:

```json
{
  "ticker": "AAPL",
  "items": [
    {"model": "k-NN Enhanced", "period": "1y", "news": false},
    {"model": "LinReg Enhanced (TW)", "period": "2y", "news": true},
    {"model": "LSTM", "period": "max", "news": false}
  ],
  "refresh_data": false
}
```

Response includes `predictions` array and `consensus` summary.

### Backtest

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/backtest` | Walk-forward backtest (body below) |
| GET | `/api/backtest/progress` | Live progress (polled while a run is in flight) |
| GET | `/api/backtest/runs` | List persisted runs (newest first) |
| GET | `/api/backtest/runs/{run_id}` | Load one persisted run |

**POST /api/backtest** body fields: `tickers`, `days`, `period` / `periods`,
`fee_pct`, `stop_loss_pct`, `buy_hold`, `refresh_data`, news knobs
(`sentiment_method` / `news_lookback_days` / `news_half_life_days`), the
model filter (`models` family keys + `include_baselines`), the confidence
gate (`min_confidence`), turnover realism (`turnover_fees`, `hold_days`),
and the stop-loss sweep (`sl_levels` list, or `sl_sweep: true` for the
`config.SL_SWEEP` set). Each result row carries `coverage`,
`turnover_count`, `fees_paid` alongside the usual metrics.

### Out-of-sample harness

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/oos` | Run the harness: select winner on window N, evaluate on disjoint N+1. Same gate/turnover/news knobs as backtest; SL is single-valued (not swept). Persists a CSV tree + JSON cache. |
| GET | `/api/oos/progress` | Live progress |
| GET | `/api/oos/runs` | List persisted OOS runs |
| GET | `/api/oos/runs/{run_id}` | Load one OOS run (rows + aggregate summary) |

Response: `rows` (per-ticker winner + OOS metrics: return, accuracy, beat-B&H,
coverage, Brier/ECE, binomial p) and `summary` (OOS beat-B&H rate, median OOS
return, **selection-inflation gap**, and gate-aware aggregates when
`min_confidence > 0`).

### Training

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/train/models` | List saved LSTM model files |
| POST | `/api/train/start` | Start background training |
| GET | `/api/train/status/{key}` | Check training progress |

### Settings

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/settings` | Current settings |
| PUT | `/api/settings` | Full replace |
| PATCH | `/api/settings` | Partial update |

### Docs (Help tab)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/docs` | List end-user concept docs (`web/docs/*.md`), ordered, with titles |
| GET | `/api/docs/{slug}` | One doc's raw markdown |

The **Help** tab renders these with a dependency-free markdown renderer and a
search box. Feature tabs deep-link into specific sections via "?" icons
(`/help#<docSlug>/<sectionSlug>`). Concept content lives in `web/docs/`, edited
as plain markdown — separate from this developer `docs/` tree.

### Analysis

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/analysis/news-comparison` | News vs No-News paired comparison |

## File structure

```
web/
├── dev.sh                          # Start both servers
├── backend/
│   ├── app.py                      # FastAPI (CORS, logging, Swagger at /docs)
│   ├── schemas.py                  # Pydantic request/response models
│   └── routes/
│       ├── data.py                 # Tickers, OHLCV (from DB), refresh
│       ├── predict.py              # Unified prediction builder + caching
│       ├── backtest.py             # Walk-forward backtesting
│       ├── train.py                # LSTM training + model inventory
│       ├── settings.py             # Persistent JSON settings
│       └── analysis.py             # News vs No-News comparison
└── frontend/
    ├── package.json                # React 19, Vite, TypeScript, TanStack Query
    ├── vite.config.ts              # Proxy /api → localhost:8000
    └── src/
        ├── main.tsx                # Router + layout (6 tabs)
        ├── app.css                 # Dark theme, no spinbox arrows
        ├── lib/api.ts              # Typed fetch client
        ├── components/
        │   ├── ui.tsx              # Panel, Btn, LoadingBox, pct(), usd()
        │   ├── PriceChart.tsx      # Zoomable SVG chart (line/candle, pan bar)
        │   └── DataTable.tsx       # Generic sortable/filterable/paginated table
        └── pages/
            ├── Dashboard.tsx       # ★ Complete
            ├── Predict.tsx         # ★ Complete
            ├── Backtest.tsx        # Stub
            ├── Training.tsx        # Stub
            ├── Analysis.tsx        # Stub
            └── Settings.tsx        # ★ Complete
```

## Reusable components

### PriceChart

```tsx
import PriceChart from "../components/PriceChart";
<PriceChart rows={ohlcvRows} height={350} />
```

Accepts array of `{date, open, high, low, close, volume}`. Renders SVG with zoom, pan, line/candle toggle.

### DataTable

```tsx
import DataTable from "../components/DataTable";
<DataTable
  rows={data}
  columns={[
    { key: "date", label: "Date" },
    { key: "close", label: "Close", align: "right", fmt: v => `$${v}` },
    { key: "delta", label: "Δ%", sortValue: row => row.delta ?? 0, color: v => v >= 0 ? green : red },
  ]}
  defaultSort="date"
  perPage={30}
  exportFilename="export.csv"
/>
```

Generic table with full-dataset sorting, computed columns, filter, pagination, CSV export.

## Prediction caching

Predictions are cached to JSON files:

```
predictions/
├── AAPL/
│   ├── 2026-04-30.json     # All predictions for AAPL today
│   └── 2026-04-29.json
├── BTC-USD/
│   └── 2026-04-30.json
└── ...
```

Each file contains all model results run for that ticker on that date. New results merge with existing cache (no duplicates). Historical predictions are also cached.

## Adding a new page

1. Create `web/frontend/src/pages/NewPage.tsx`
2. Import in `main.tsx`, add route: `<Route path="/newpage" element={<NewPage />} />`
3. Add tab to `tabs` array in `Layout`

## Adding a new API endpoint

1. Create or edit `web/backend/routes/your_route.py`
2. Register in `web/backend/app.py`: `app.include_router(your_route.router)`
3. Add typed fetch wrapper in `web/frontend/src/lib/api.ts`
