# Documentation

Technical documentation for MarketPulse AI. Start here if you're new to the codebase or returning after a break.

## Models

- **[k-NN](knn.md)** — k-Nearest Neighbors classifier. Covers naive vs enhanced mode, time-weighting, sentiment adjustment, parameters, and when it works well vs when it struggles.
- **[Linear Regression](linear-regression.md)** — Continuous return prediction with sigmoid confidence mapping. Covers naive vs enhanced, time-weighting via sample_weight, and comparison with k-NN.
- **[LSTM Neural Network](lstm.md)** — Recurrent neural network for sequential pattern recognition. Covers training presets (quick/standard/cluster), model saving/loading, cluster deployment, and when to use LSTM vs simpler models.

## Engine

- **[Technical Indicators](features.md)** — Shared feature engineering: returns, volume, RSI, MACD, volatility. How each indicator works, the feature vector structure, scaling, warmup requirements, and how to add new features.
- **[Sentiment Analysis](sentiment.md)** — VADER vs naive scoring, the two-stage integration approach, the math behind probability adjustment, caching, and limitations.

## Evaluation

- **[Backtesting](backtesting.md)** — Walk-forward methodology, all CLI flags (`--days`, `--period`, `--full`, `--compare-periods`, `--output`), metrics (accuracy, direction accuracy, confidence calibration, consensus, profit factor, streaks), and how to interpret results.

## Architecture

- **[API & Architecture](api.md)** — StockAppAPI facade, data flow, PredictionConfig/PredictionResult types, database schema, model interface contract, and how to add a web UI.
- **[AGENTS.md](../AGENTS.md)** — Quick-reference context file for AI assistants (compact, not for deep reading).
