# LSTM Neural Network

## How it works

LSTM (Long Short-Term Memory) is a recurrent neural network for sequential data. Unlike k-NN/LinReg which see a flat feature vector, LSTM reads through each day in order and maintains internal "memory."

```
Input sequence (20 days × 5 features)  →  LSTM layers  →  FC layers  →  Sigmoid → P(UP)
```

Each timestep has: daily return, volume change, RSI, volatility, MACD histogram (shared with k-NN/LinReg via `features.py`).

The key advantage: LSTM sees the **order** — it can learn "RSI rising for 5 days then dropping" as a pattern, which flat vectors can't express.

## Input normalization

Like k-NN and LinReg, LSTM uses `StandardScaler` to normalize features before training. The scaler is fitted only on training data (no data leakage to validation), then applied to validation data and to input sequences during prediction. The fitted scaler is saved alongside model weights, so loaded models produce consistent predictions.

## Training presets

| Preset | Hidden | Layers | Max epochs | Patience | Approx. time |
|---|---|---|---|---|---|
| `quick` | 32 | 1 | 50 | 10 | ~1-5 min (CPU) |
| `standard` | 64 | 2 | 200 | 20 | ~5-15 min (CPU) |
| `cluster` | 128 | 3 | 1000 | 50 | hours (GPU) |

"Patience" = early stopping parameter (see below).

## Early stopping

Without early stopping, LSTM overfits heavily. Typical pattern:

```
Epoch  10: train_loss=0.68  val_loss=0.70  ← model is learning
Epoch  20: train_loss=0.65  val_loss=0.70  ← val plateaus
Epoch  50: train_loss=0.40  val_loss=1.20  ← memorizing training data
Epoch 200: train_loss=0.22  val_loss=2.00  ← completely overfit
```

Early stopping monitors `val_loss`. If it doesn't improve for `patience` epochs, training stops and restores the best model weights:

```
Epoch  20: val_loss=0.6949  (best so far)
Epoch  25: val_loss=0.7010  (no improvement: 5/20)
Epoch  40: val_loss=0.7200  (no improvement: 20/20 → STOP)
→ Restored model from epoch 20 (val_loss=0.6949)
```

This saves time (standard stops at ~30-40 epochs instead of 200) and produces a better model.

## Workflow

```bash
# 1. Train (once per ticker × period)
uv run python train.py --ticker AAPL --period 1y --preset quick

# 2. Verify
uv run python train.py --list

# 3. Use (auto-loads saved model)
uv run python main.py --tickers AAPL
uv run python backtest.py --tickers AAPL --days 20
```

Train multiple at once:
```bash
uv run python train.py --stocks --preset standard
uv run python train.py --all --periods 1y 2y max --preset cluster
```

## Saved models

Saved to `models/{ticker}_{period}_{preset}.pt`. Each file contains: weights, config, training metrics (accuracy, epochs completed, early stop info, duration).

Auto-loaded by priority: cluster > standard > quick. If no model exists for a ticker+period, backtest logs a NOTE with the exact training command.

## Cluster deployment

```bash
# With Podman (GPU passthrough)
podman run --rm --gpus all -v ./data:/app/data:z -v ./models:/app/models:z \
    marketpulse python train.py --all --preset cluster

# With Singularity (--nv for NVIDIA)
singularity run --nv --bind ./data:/app/data,./models:/app/models \
    marketpulse.sif python train.py --all --preset cluster
```

PyTorch auto-detects GPU. The `cluster` preset also uses learning rate scheduling (ReduceLROnPlateau).

## When to use LSTM vs k-NN/LinReg

**LSTM better:** lots of data (1000+ rows), sequential patterns, you can afford one-time training.

**k-NN/LinReg better:** limited data, instant results needed, frequent ticker/period experimentation.

**The backtest tells you.** Train, then run `--compare-periods`. If LSTM doesn't beat simpler models, the extra complexity isn't worth it — which is common for daily stock prediction.

## Limitations

- Requires PyTorch (`uv pip install torch`)
- Must be trained before use (no cold-start)
- One model per ticker × period × preset (not transferable)
- CPU training slow for `standard`/`cluster`
- Daily price data is inherently noisy — even with early stopping, LSTM often converges to ~50% accuracy (coin flip)
