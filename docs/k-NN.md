# k-Nearest Neighbors (k-NN)

## How it works

k-NN is a classification algorithm. Given a new data point, it finds the `k` most similar points in the training data and takes a majority vote — if 3 out of 5 neighbors went UP, the prediction is UP.

In our case:
- **Data point** = a window of 5 consecutive daily returns (e.g. `[-0.01, 0.02, -0.005, 0.01, 0.003]`)
- **Similarity** = Euclidean distance between return windows
- **Target** = did the price go UP or DOWN the next day?

The intuition: if the last 5 days looked similar to some pattern in the past, what happened next time this pattern appeared?

## Naive mode (`features=["returns"]`)

Uses only daily price returns. Each sample is a vector of length 5 (window_size):

```
Sample:  [ret_day1, ret_day2, ret_day3, ret_day4, ret_day5]
Target:  1 (UP) or 0 (DOWN)
```

This is the baseline. Simple but surprisingly competitive — in backtesting it often matches or beats more complex variants.

## Enhanced mode (`features=ALL_FEATURES`)

Adds volume change, RSI, volatility, and MACD to the feature vector:

```
Sample:  [ret_1..5, vol_chg_1..5, rsi, volatility, macd_histogram]
         ─────────  ───────────── ───  ──────────  ──────────────
         5 values   5 values      1    1           1    = 13 total
```

### Why these features?

- **Volume change** — a price move on high volume is more significant than on low volume. If AAPL drops 2% on 3x average volume, that's a stronger signal than the same drop on normal volume.
- **RSI (Relative Strength Index)** — measures if a stock is overbought (RSI > 0.7) or oversold (RSI < 0.3). Mean-reversion traders buy oversold and sell overbought.
- **Volatility** — rolling standard deviation of returns. High volatility = uncertain market, patterns may be less reliable.
- **MACD histogram** — momentum indicator. Positive = bullish momentum, negative = bearish. Crossovers from negative to positive are classic buy signals.

### Feature scaling

Enhanced mode uses `StandardScaler` before training. This is critical because returns are tiny numbers (~0.01), volume changes can be large (~2.0), and RSI is between 0 and 1. Without scaling, k-NN's distance metric would be dominated by whatever feature has the largest numbers.

### Warmup requirements

MACD needs ~34 data points before producing valid values (26-period EMA + 9-period signal line). RSI needs 14. This means enhanced k-NN cannot work with very short periods like 1mo (~22 trading days). It returns "Insufficient data" instead of silently producing garbage.

## Time-weighting

Standard k-NN treats all training samples equally — a pattern from 2020 has the same weight as one from yesterday. Time-weighted mode addresses this by:

1. Keeping only the more recent half of training data (dropping older patterns)
2. Using `weights='distance'` in scikit-learn, which gives closer neighbors more vote weight

This is an approximation. Ideally we'd use `sample_weight` in `.fit()`, but scikit-learn's `KNeighborsClassifier` doesn't support it. The trimming approach is a practical workaround.

## Sentiment adjustment

Sentiment doesn't go into the k-NN training. Instead, it's a post-hoc adjustment:

```
1. k-NN predicts:  UP with 60% confidence
2. News sentiment: +0.5 (positive)
3. Adjustment:     60% + (0.5 × 0.20) = 70% → still UP, but more confident
```

The weight (0.20) means perfect sentiment (±1.0) can shift probability by ±20 percentage points. This can flip a prediction: if k-NN says UP at 55% and sentiment is strongly negative (-1.0), the adjusted probability becomes 35% → flips to DOWN.

Why post-hoc? Because we don't have historical daily sentiment scores for the training data. We only have today's news. Feeding today's sentiment into the training features would create a data leak.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `k` | 5 | Number of neighbors. Higher = smoother predictions, lower = more reactive |
| `window_size` | 5 | Days in the return window. Higher = looks at longer patterns |
| `features` | `["returns"]` | Which features to include in the vector |

## When it works well

- Assets with repeating price patterns (mean-reversion behavior)
- Shorter training periods (1-2 years) where patterns are still relevant
- When combined with time-weighting on volatile assets

## When it struggles

- Trending markets where past patterns don't repeat
- Very short data (< 50 rows after warmup)
- When `k` is too high relative to the training set size
