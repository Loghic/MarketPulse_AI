# Documentation

Technical documentation for MarketPulse AI. Start here if you're new to the codebase or returning after a break.

## Models

- **[k-NN](knn.md)** — Nearest Neighbors classifier. Naive vs enhanced, exponential time-weighting, parameters.
- **[Linear Regression](linear-regression.md)** — Return prediction with sigmoid confidence. Naive vs enhanced, comparison with k-NN.
- **[LSTM Neural Network](lstm.md)** — Recurrent network for sequential patterns. Training presets, early stopping, input normalization, save/load, cluster deployment.

## Engine

- **[Technical Indicators](features.md)** — Shared features: RSI, MACD, volatility, volume. Scaling. How to add new features.
- **[Sentiment Analysis](sentiment.md)** — VADER vs naive scoring, two-stage integration, limitations.

## Evaluation

- **[Backtesting](backtesting.md)** — Walk-forward methodology, trading fees, stop-loss, buy-and-hold benchmark, profit factor, max drawdown, Sharpe/Sortino ratios, streaks, yearly rolling performance, data refresh, batch runner, CSV export, interpreting results.

## Architecture

- **[API & Architecture](api.md)** — StockAppAPI facade, module structure, Backtester (fees + stop-loss + B&H), DayResult/BacktestResult fields, database schema, model contract, web UI example.
- **[AGENTS.md](../AGENTS.md)** — Compact context file for AI assistants.

## Web GUI

- **[Web GUI](web.md)** — Full documentation: architecture, pages (Dashboard, Predict, Settings), API endpoints with examples, reusable components, prediction caching, file structure.

FastAPI backend wraps `StockAppAPI` with REST endpoints. React frontend provides a browser dashboard. Swagger docs at `http://localhost:8000/docs`. Run: `./web/dev.sh`.

## Testing

- `test_pipeline.py` — Quick 13-test smoke test (no pytest needed)
- `tests/` — 103-test pytest suite: models, features, backtester (fees, SL, DD, Sharpe, streaks), benchmarks, web API endpoints (data, predict, backtest, settings, analysis), export, logger

## CI / CD

- `.pre-commit-config.yaml` — Git hooks: **ruff** auto-fix + format + **mypy** type check before every commit. Setup: `uv run pre-commit install`
- `.github/workflows/tests.yml` — Three parallel jobs: **lint** (ruff), **typecheck** (mypy), **test** (pytest + coverage upload). Runs on Python 3.12 + 3.13 matrix.
- `.codecov.yml` — Coverage thresholds (60% target, 50% patch).
- Coverage badge and test results visible on [Codecov](https://codecov.io).
