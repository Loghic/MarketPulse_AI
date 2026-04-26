# Linear Regression

## How it works

Linear Regression predicts a continuous value — in our case, tomorrow's return (e.g. +0.8%). Unlike k-NN which directly classifies UP/DOWN, LinReg predicts "how much" and we derive direction from the sign:

```
Predicted return = +0.012  →  UP   (positive sign)
Predicted return = -0.005  →  DOWN (negative sign)
```

The model fits a linear equation: `next_return = w1×ret1 + w2×ret2 + ... + wN×retN + bias`, where `ret1..retN` are the last N daily returns. The weights `w1..wN` are learned from training data via least squares.

## Naive vs Enhanced

Same feature sets as k-NN:

| Mode | Features | Vector size |
|---|---|---|
| Naive (`"linreg"`) | 5 daily returns | 5 |
| Enhanced (`"linreg_enhanced"`) | returns + volume + RSI + volatility + MACD | 13 |

Enhanced mode also uses `StandardScaler` — important because linear regression coefficients are directly affected by feature magnitudes.

## Confidence mapping

k-NN gives probability directly (from neighbor vote proportions). LinReg gives a predicted return, which we need to map to a confidence score. We use a sigmoid function:

```
confidence = 1 / (1 + e^(-|predicted_return| × 100))
```

This maps:
- ±0.1% predicted return → ~52% confidence (barely above coin flip)
- ±1.0% predicted return → ~73% confidence
- ±2.0% predicted return → ~88% confidence
- ±5.0% predicted return → ~99% confidence

The scaling factor (100) was tuned so that typical daily stock returns produce reasonable confidence values. Crypto returns are larger, so LinReg tends to produce higher confidence on crypto — which doesn't necessarily mean it's more accurate.

## Time-weighting

Unlike k-NN (which has to approximate time-weighting by trimming data), LinReg supports `sample_weight` natively in scikit-learn's `.fit()`. We use linear weights:

```python
weights = np.linspace(0.1, 1.0, len(training_data))
# Oldest sample: weight 0.1
# Newest sample: weight 1.0
model.fit(X, y, sample_weight=weights)
```

This means the model still sees old data, but recent patterns have 10× more influence on the fitted coefficients. This is a genuine advantage over k-NN's cruder approach.

## Sentiment adjustment

Same post-hoc mechanism as k-NN (see `knn.md`). The predicted return determines base direction and confidence, then sentiment shifts the probability.

## When to use LinReg vs k-NN

From backtesting results:

**LinReg tends to be better for:**
- Crypto (BTC-USD) — larger daily returns mean the sign prediction is more reliable and the sigmoid produces meaningful confidence differentiation
- Longer training periods (5y, max) — linear relationships are more stable over time

**k-NN tends to be better for:**
- Stocks (AAPL, MSFT) — smaller daily returns where pattern matching outperforms linear extrapolation
- Shorter training periods (1-2y) — captures recent regime changes

**Neither consistently beats the other.** The backtest `--compare-periods` mode exists precisely to find which model works best for which ticker and period.

## Limitations

- Assumes a linear relationship between past returns and future returns. Markets are nonlinear — this is the fundamental limitation.
- The sigmoid confidence mapping is somewhat arbitrary. A predicted return of +0.01% and +2% both say "UP" but with very different confidence. Whether that confidence is well-calibrated depends on the data.
- No regularization (we use plain `LinearRegression`, not Ridge or Lasso). For 13 features this is fine, but if we added many more features, overfitting could become an issue.
