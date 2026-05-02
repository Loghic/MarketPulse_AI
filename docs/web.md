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

- **Ticker selector** — dropdown grouped by Stocks / Crypto
- **Period tabs** — 1MO, 1Y, 2Y, 5Y, MAX, Custom (date pickers)
- **Update buttons** — refresh single ticker or all tickers
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
- **Quick presets** — one-click to fill builder (All, All+News, k-NN family, LinReg family, LSTM)
- **9 model variants**: k-NN, k-NN (TW), k-NN Enhanced, k-NN Enhanced (TW), LinReg, LinReg (TW), LinReg Enhanced, LinReg Enhanced (TW), LSTM
- **News per model** — not global; choose which models use sentiment
- **Chart** — optional (☑ Show chart), independent period from model period
- **Consensus** — auto-computed from run results (UP/DOWN/SPLIT + agreement %)
- **Results table** — sortable by confidence, filterable, export CSV
- **News headlines** — displayed when predictions use sentiment
- **Prediction caching** — results saved to `predictions/{ticker}/{date}.json`
- **Historical prediction** — predict as if today were any past date
- **Prediction target** — shows next trading day (skips weekends for stocks)

### Backtest (stub)

Walk-forward backtest configurator. Planned: results table, best models, equity curve chart.

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

### Data

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/data/tickers` | All tickers with row count, last date, asset type |
| GET | `/api/data/ticker/{ticker}?period=1y&limit=0` | OHLCV data. limit=0 = no limit |
| POST | `/api/data/refresh` | Download latest prices + news. Body: `{"tickers": ["AAPL"]}` |

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
| POST | `/api/backtest` | Walk-forward backtest. Body: tickers, days, period, fees, SL |

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
