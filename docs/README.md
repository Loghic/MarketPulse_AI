# Documentation

Technical documentation for MarketPulse AI. Start here if you're new to the codebase or returning after a break.

## Getting started

- **[run.md](run.md)** — **The runbook.** Install, optional extras (`ai` /
  `web` / `viz` / `dev` / `forecast` / `kronos`), first-time setup (VADER,
  FinBERT, GDELT), every CLI flag with examples, suggested workflows,
  troubleshooting. If you're not sure where to look, look here first.

## Models

- **[LSTM Neural Network](lstm.md)** — Recurrent network for sequential patterns. Training presets, early stopping, input normalization, save/load, cluster deployment.
- **[Forecasting Models](forecasting.md)** — Prophet, Chronos-2 and Kronos. The `ForecastModel` / `ForecastResult` interface, deriving direction from a value forecast, post-hoc sentiment, per-model timing (`--timing`) and period selection (`--periods`), backtest integration, and how to add a model.
- **[k-NN](k-NN.md)** / **[Linear Regression](linear_regression.md)** — the simple/educational references. Naive vs enhanced, time-weighting, sigmoid confidence.
- **Baselines** — naive (Always-Long/Short, Previous-Day, Momentum, Random) + stateless news-aware ones (News Previous-Day, News-Informed, News Momentum); defined in `engine/baseline_models.py`, used as the floor real models must clear. See [run/research.md](run/research.md).

## Engine

- **[Technical Indicators](features.md)** — Shared features: RSI, MACD, volatility, volume. Scaling. How to add new features.
- **[Sentiment Analysis](sentiment.md)** — Pluggable scorers (VADER, FinBERT, naive), two-stage integration, look-ahead-safe DB lookup, time-decay weighting, configurable lookback window.
- **[News Sources](news_sources.md)** — Yahoo Finance, GDELT (free, historical), combining providers, adding new ones.

## Evaluation

- **[Backtesting](backtesting.md)** — Walk-forward methodology, trading fees (incl. turnover / position-based), stop-loss + sweep, confidence gate, buy-and-hold benchmark, profit factor, max drawdown, Sharpe / Sortino, streaks, yearly performance, news sentiment, data sanity guards, per-model timing, CSV export.
- **[Research workflow](run/research.md)** — The out-of-sample harness (selection-inflation gap, beat-B&H rate, calibration/significance), news-vs-no-news impact, confidence-sweep + significance flags. The honest-results tooling.
- **[Regression / point-forecast track](forecasting-regression.md)** — The *value*-prediction path (separate from trading): scale-free skill vs a random walk (Theil U2 / MASE), regression baselines (Random Walk / Drift / Seasonal Naive), ARIMA + XGBoost forecasters, the walk-forward forecast harness (`scripts/forecast_harness.py`), and what's next (Prophet+LSTM residual hybrid, DM/Wilcoxon tests).

## Architecture

- **[API & Architecture](api.md)** — `StockAppAPI` facade, module structure, `Backtester` (fees + stop-loss + B&H), `DayResult` / `BacktestResult` fields, database schema, model contract.
- **[AGENTS.md](../AGENTS.md)** — Compact context file for AI assistants — includes 2026 architecture notes (news pipeline, sanity guards, persistence, test isolation, forecasting models, timing).

## Web GUI

- **[Web GUI](web.md)** — Architecture, all pages (Dashboard, Predict, Backtest, OOS, OOS Compare, Training, Analysis, Settings, Help), the config-driven `/api/meta` + OOS + docs endpoints, reusable components, caching, file structure. End-user concept glossary lives in [web/docs/](../web/docs/) and renders in the Help tab.
FastAPI backend wraps `StockAppAPI` with REST endpoints. React frontend provides a browser dashboard with live progress polling during backtests and on-disk persistence so closing a tab doesn't lose results. Swagger docs at `http://localhost:8000/docs`. Run: `./web/dev.sh` (or see [run.md](run.md)).

## Testing & CI

See the *Testing* and *Static checks* sections of [run.md](run.md).
- `test_pipeline.py` — Quick 13-test smoke test (no pytest needed)
- `tests/` — ~140-test pytest suite covering models, features, backtester (fees, SL, DD, Sharpe, streaks, data-sanity guards), benchmarks, news pipeline (look-ahead safety, time-decay, FinBERT fallback), web API endpoints (data, predict, backtest persistence + progress, train, settings, analysis browser), export, logger
- Pre-commit hooks (`.pre-commit-config.yaml`): ruff auto-fix + format + mypy on every commit. **Note:** mypy on every commit is slow (~20-60s); if you ever CTRL+C the hook mid-run, recover lost unstaged work with `git stash list` → `git stash apply`. See the troubleshooting table in [run.md](run.md).
- GitHub Actions: three parallel jobs — **lint** (ruff), **typecheck** (mypy), **test** (pytest + Codecov upload) on Python 3.12 + 3.13 matrix
- Coverage badge and test results visible on [Codecov](https://codecov.io)
