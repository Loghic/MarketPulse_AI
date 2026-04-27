# Technical Indicators (features.py)

All indicators are computed in `engine/features.py` and shared between k-NN and LinReg. Features are configurable — you choose which to include via a list of strings.

## Available features

### Returns (always included)

Daily percentage change in closing price:

```
return_t = (close_t - close_{t-1}) / close_{t-1}
```

A return of +0.02 means the stock went up 2% that day. Each sample includes a window of `window_size` (default 5) consecutive returns.

### Volume change

Percentage change in trading volume, clipped to [-5, +5] to prevent extreme spikes from dominating:

```
vol_change_t = (volume_t - volume_{t-1}) / volume_{t-1}
```

Why it matters: a 3% price drop on 5× normal volume is a very different signal than the same drop on low volume. High volume suggests institutional activity and conviction behind the move.

Like returns, volume changes are included as a full window (5 values per sample).

### RSI (Relative Strength Index)

Classic momentum oscillator. Measures the ratio of average gains to average losses over the last 14 periods:

```
RS = average_gain_14 / average_loss_14
RSI = RS / (1 + RS)    ← normalized to [0, 1] (standard RSI uses [0, 100])
```

Interpretation:
- RSI > 0.7 → overbought (price may reverse down)
- RSI < 0.3 → oversold (price may reverse up)
- RSI ≈ 0.5 → neutral

We normalize to [0, 1] instead of the traditional [0, 100] for better compatibility with StandardScaler and other features.

RSI is included as a single value (the RSI at the end of the window), not as a full window. This keeps the feature vector compact.

**Warmup:** 14 data points before the first valid RSI value.

### Volatility

Rolling standard deviation of returns over the window:

```
volatility = std(returns[t-4 : t+1])    ← for window_size=5
```

High volatility means the price has been swinging a lot recently. Models can learn that patterns during high-volatility periods behave differently than during calm markets.

Included as a single value per sample.

### MACD Histogram

Moving Average Convergence Divergence — a trend-following momentum indicator:

```
MACD line   = EMA(close, 12) - EMA(close, 26)
Signal line = EMA(MACD_line, 9)
Histogram   = MACD_line - Signal_line
```

Where EMA is the Exponential Moving Average with the standard 12/26/9 parameters.

Interpretation:
- Histogram > 0 → bullish momentum (short-term trend above long-term)
- Histogram < 0 → bearish momentum
- Histogram crossing zero → trend reversal signal

Included as a single value per sample.

**Warmup:** ~34 data points (26 for EMA26 + 9 for the signal line). This is the reason enhanced models can't work with 1-month periods (~22 trading days).

## Feature vector structure

For `window_size=5`:

| Feature set | Vector contents | Length |
|---|---|---|
| `["returns"]` | `[ret_1, ret_2, ret_3, ret_4, ret_5]` | 5 |
| `["returns", "volume"]` | `[ret_1..5, vol_1..5]` | 10 |
| `["returns", "rsi"]` | `[ret_1..5, rsi]` | 6 |
| `ALL_FEATURES` | `[ret_1..5, vol_1..5, rsi, volatility, macd]` | 13 |

## Adding a new feature

1. Add computation to `compute_feature_columns()` — add a new column prefixed with `_`
2. Add extraction to `build_feature_vector()` — decide if it's a window (like returns) or a single value (like RSI)
3. Add the name string to `ALL_FEATURES`
4. If it has a warmup period, update `min_rows_needed()`
5. Run `test_pipeline.py` to verify

## Scaling

When features have different magnitudes (returns ≈ 0.01, volume changes ≈ 1.0, RSI ≈ 0.5), distance-based models like k-NN would be dominated by the largest features. All three models (k-NN, LinReg, LSTM) apply `StandardScaler` (zero mean, unit variance) before training. For k-NN and LinReg this is done per prediction run. For LSTM the scaler is fitted once on training data and saved with the model weights, so it can be reapplied consistently during prediction.
