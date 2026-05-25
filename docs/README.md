# Documentation

Technical documentation for MarketPulse AI. Start here if you're new to the codebase or returning after a break.

## Getting started

- **[run.md](run.md)** — **The runbook.** Install, optional extras (`ai` /
  `web` / `viz` / `dev`), first-time setup (VADER, FinBERT, GDELT), every
  CLI flag with examples, suggested workflows, troubleshooting. If you're
  not sure where to look, look here first.

## Models

- **[k-NN](k-NN.md)** — Nearest Neighbors classifier. Naive vs enhanced, exponential time-weighting, parameters.
- **[Linear Regression](linear_regression.md)** — Return prediction with sigmoid confidence. Naive vs enhanced, comparison with k-NN.
- **[LSTM Neural Network](lstm.md)** — Recurrent network for sequential patterns. Training presets, early stopping, input normalization, save/load, cluster deployment.

## Engine

- **[Technical Indicators](features.md)** — Shared features: RSI, MACD, volatility, volume. Scaling. How to add new features.
- **[Sentiment Analysis](sentiment.md)** — Pluggable scorers (VADER, FinBERT, naive), two-stage integration, look-ahead-safe DB lookup, time-decay weighting, configurable lookback window.
- **[News Sources](news_sources.md)** — Yahoo Finance, GDELT (free, historical), combining providers, adding new ones.

## Evaluation

- **[Backtesting](backtesting.md)** — Walk-forward methodology, trading fees, stop-loss, buy-and-hold benchmark, profit factor, max drawdown, Sharpe / Sortino ratios, streaks, yearly rolling performance, news sentiment in backtests, data sanity guards, CSV export, interpreting results.

## Architecture

- **[API & Architecture](api.md)** — `StockAppAPI` facade, module structure, `Backtester` (fees + stop-loss + B&H), `DayResult` / `BacktestResult` fields, database schema, model contract.
- **[AGENTS.md](../AGENTS.md)** — Compact context file for AI assistants — includes 2026 architecture notes (news pipeline, sanity guards, persistence, test isolation).

## Web GUI

- **[Web GUI](web.md)** — Architecture, all six pages (Dashboard, Predict, Backtest, Training, Analysis, Settings), API endpoints with examples, reusable components, prediction + backtest caching, file structure.

FastAPI backend wraps `StockAppAPI` with REST endpoints. React frontend provides a browser dashboard with live progress polling during backtests and on-disk persistence so closing a tab doesn't lose results. Swagger docs at `http://localhost:8000/docs`. Run: `./web/dev.sh` (or see [run.md](run.md)).

## Testing & CI

See the *Testing* and *Static checks* sections of [run.md](run.md).

- `test_pipeline.py` — Quick 13-test smoke test (no pytest needed)
- `tests/` — ~140-test pytest suite covering models, features, backtester (fees, SL, DD, Sharpe, streaks, data-sanity guards), benchmarks, news pipeline (look-ahead safety, time-decay, FinBERT fallback), web API endpoints (data, predict, backtest persistence + progress, train, settings, analysis browser), export, logger
- Pre-commit hooks (`.pre-commit-config.yaml`): ruff auto-fix + format + mypy on every commit. **Note:** mypy on every commit is slow (~20-60s); if you ever CTRL+C the hook mid-run, recover lost unstaged work with `git stash list` → `git stash apply`. See the troubleshooting table in [run.md](run.md).
- GitHub Actions: three parallel jobs — **lint** (ruff), **typecheck** (mypy), **test** (pytest + Codecov upload) on Python 3.12 + 3.13 matrix
- Coverage badge and test results visible on [Codecov](https://codecov.io)
