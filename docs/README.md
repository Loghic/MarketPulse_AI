# Documentation

Technical documentation for MarketPulse AI. Start here if you're new to the codebase or returning after a break.

## Models

- **[k-NN](knn.md)** — Nearest Neighbors classifier. Naive vs enhanced, time-weighting, parameters.
- **[Linear Regression](linear-regression.md)** — Return prediction with sigmoid confidence. Naive vs enhanced, comparison with k-NN.
- **[LSTM Neural Network](lstm.md)** — Recurrent network for sequential patterns. Training presets, early stopping, save/load, cluster deployment.

## Engine

- **[Technical Indicators](features.md)** — Shared features: RSI, MACD, volatility, volume. How to add new features.
- **[Sentiment Analysis](sentiment.md)** — VADER vs naive scoring, two-stage integration, limitations.

## Evaluation

- **[Backtesting](backtesting.md)** — Walk-forward methodology, trading fees, buy-and-hold benchmark, profit factor, streaks, all CLI flags, batch runner (`run_all.py`), CSV export, interpreting results.

## Architecture

- **[API & Architecture](api.md)** — StockAppAPI facade, module structure, database schema, model contract, LSTM auto-loading, web UI example.
- **[AGENTS.md](../AGENTS.md)** — Compact context file for AI assistants.
