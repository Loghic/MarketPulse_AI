/**
 * api.ts – Typed API client for FastAPI backend.
 *
 * All endpoints return typed responses.
 * Base URL proxied by Vite in dev (/api → localhost:8000/api).
 */

const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

// ------------------------------------------------------------------
// Types (matching backend schemas)
// ------------------------------------------------------------------

export interface TickerInfo {
  ticker: string;
  asset_type: string;
  rows: number;
  last_date: string | null;
}

export interface OHLCVRow {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface TickerDataResponse {
  ticker: string;
  rows: number;
  data: OHLCVRow[];
}

export interface RefreshStatus {
  ticker: string;
  rows: number;
  last_date: string;
  news_count: number;
}

export interface PredictionRow {
  ticker: string;
  model: string;
  period: string;
  prediction: string;
  confidence: number;
  last_price: number;
  sentiment: string;
  sentiment_score: number;
  headlines: string[];
  timestamp: string;
}

export interface BacktestDayRow {
  date: string;
  predicted: string;
  actual: string;
  correct: boolean;
  confidence: number;
  trade_pnl: number;
  trade_pnl_net: number;
  stopped_out: boolean;
}

export interface BacktestModelResult {
  model: string;
  ticker: string;
  period: string;
  accuracy: number;
  total_return: number;
  buy_hold_return: number;
  profit_factor: number;
  max_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  buy_hold_max_drawdown: number;
  fee_pct: number;
  stop_loss_pct: number;
  stopped_out_count: number;
  win_trades: number;
  loss_trades: number;
  avg_win: number;
  avg_loss: number;
  best_day: number;
  worst_day: number;
  longest_win_streak: number;
  longest_loss_streak: number;
  benchmarks: Record<string, number>;
  days: BacktestDayRow[];
}

export interface ModelInventoryItem {
  ticker: string;
  period: string;
  preset: string;
  filename: string;
  size_kb: number;
  modified: string;
}

export interface UserSettings {
  default_period: string;
  default_fee_pct: number;
  default_stop_loss_pct: number;
  default_backtest_days: number;
  knn_k: number;
  knn_enhanced_k: number;
  lstm_preferred_preset: string;
  lstm_fallback: boolean;
  log_mode: string;
}

export interface NewsComparisonRow {
  ticker: string;
  model: string;
  period: string;
  return_no_news: number;
  return_with_news: number;
  diff: number;
  sharpe_no_news: number;
  sharpe_with_news: number;
  accuracy_no_news: number;
  accuracy_with_news: number;
}

// ------------------------------------------------------------------
// API functions
// ------------------------------------------------------------------

export const api = {
  // Data
  getTickers: () => request<TickerInfo[]>("/data/tickers"),

  getTickerData: (ticker: string, period = "1y", limit = 0) =>
    request<TickerDataResponse>(`/data/ticker/${ticker}?period=${period}&limit=${limit}`),

  refresh: (tickers: string[] = []) =>
    request<RefreshStatus[]>("/data/refresh", {
      method: "POST",
      body: JSON.stringify({ tickers }),
    }),

  // Predictions
  predict: (params: {
    tickers?: string[];
    models?: string[];
    period?: string;
    include_news?: boolean;
    refresh_data?: boolean;
  }) =>
    request<{ predictions: PredictionRow[]; cached: boolean }>("/predict", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  getConsensus: (ticker: string, period = "1y") =>
    request<{
      ticker: string;
      consensus: string;
      up_votes: number;
      down_votes: number;
      agreement: number;
      votes: { model: string; prediction: string; confidence: number }[];
    }>(`/predict/consensus/${ticker}?period=${period}`),

  listCached: () => request<{ ticker: string; date: string; count: number; models: string[] }[]>("/predict/cached"),

  /** Latest cached prediction for one ticker (or empty payload if none). */
  cachedForTicker: (ticker: string) =>
    request<{
      ticker: string;
      date?: string;
      cached_at?: string;
      predictions: PredictionRow[];
      consensus: { direction: string; up: number; down: number; total: number; agreement: number } | null;
    }>(`/predict/cached/${ticker}`),

  // Backtest
  backtest: (params: {
    tickers?: string[];
    days?: number;
    period?: string;
    /** Multi-period selection — overrides `period` + `compare_periods` if set. */
    periods?: string[];
    fee_pct?: number;
    stop_loss_pct?: number;
    compare_periods?: boolean;
    buy_hold?: boolean;
    refresh_data?: boolean;
    sentiment_method?: string;
    news_lookback_days?: number;
    news_half_life_days?: number;
  }) =>
    request<{
      results: BacktestModelResult[];
      best_by_return: BacktestModelResult | null;
      best_by_sharpe: BacktestModelResult | null;
    }>("/backtest", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  // Training
  getModels: () => request<ModelInventoryItem[]>("/train/models"),

  startTraining: (params: { ticker: string; period: string; preset: string }) =>
    request<{ status: string; message: string }>("/train/start", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  // Settings
  getSettings: () => request<UserSettings>("/settings"),
  updateSettings: (settings: UserSettings) =>
    request<UserSettings>("/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    }),
  patchSettings: (updates: Partial<UserSettings>) =>
    request<UserSettings>("/settings", {
      method: "PATCH",
      body: JSON.stringify(updates),
    }),

  // Analysis
  newsComparison: (params: {
    tickers?: string[];
    days?: number;
    period?: string;
    fee_pct?: number;
  }) =>
    request<NewsComparisonRow[]>("/analysis/news-comparison", {
      method: "POST",
      body: JSON.stringify(params),
    }),

  /** Enumerate the `results/{scope}_{days}d…/` subdirectories. */
  listResultsDirs: () =>
    request<{
      name: string;
      modified: string;
      csv_count: number;
      ticker_csvs: string[];
      has_summary: boolean;
      has_news_impact: boolean;
    }[]>("/analysis/results-dirs"),

  /** Read one CSV from a results subdir as a list of string-valued rows. */
  readResultCsv: (dir: string, file: string) =>
    request<{
      dir: string;
      file: string;
      rows: Record<string, string>[];
      row_count: number;
    }>(`/analysis/result-csv?dir=${encodeURIComponent(dir)}&file=${encodeURIComponent(file)}`),

  // Training
  trainingStatus: (key: string) =>
    request<{
      ticker: string;
      period: string;
      preset: string;
      status: string;
      epoch: number;
      total_epochs: number;
      val_loss: number;
      val_accuracy: number;
      message: string;
    }>(`/train/status/${key}`),
};
