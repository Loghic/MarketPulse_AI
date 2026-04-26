# LSTM Neural Network

## How it works

LSTM (Long Short-Term Memory) is a recurrent neural network designed for sequential data. Unlike k-NN and LinReg which see a flat feature vector, LSTM processes data as a time series — it reads through each day in order and maintains an internal "memory" of the sequence.

The architecture:

```
Input sequence (20 days × 5 features)
        ↓
    LSTM layers (1-3 layers depending on preset)
        ↓
    Last hidden state
        ↓
    Fully connected layers
        ↓
    Sigmoid → probability of UP ∈ [0, 1]
```

Each timestep has these features (same as k-NN/LinReg enhanced):
- Daily return
- Volume change
- RSI
- Volatility
- MACD histogram

The key difference from k-NN/LinReg: LSTM sees the **order** of the features. It can learn patterns like "RSI was rising for 5 days then suddenly dropped" — something a flat feature vector can't express.

## Training vs prediction

Unlike k-NN and LinReg (which train from scratch on every prediction), LSTM is trained **once** and the weights are saved to disk. This is necessary because:
- Training takes minutes to hours (vs milliseconds for k-NN)
- The trained model is reusable across many predictions
- GPU training on a cluster produces a model that runs on CPU for inference

### Workflow

```bash
# 1. Train (once)
uv run python train.py --ticker AAPL --period 1y --preset quick

# 2. Predict (reuses saved model, instant)
uv run python main.py --tickers AAPL
uv run python backtest.py --tickers AAPL --days 20
```

## Training presets

| Preset | Hidden size | Layers | Epochs | Approx. time | Use case |
|---|---|---|---|---|---|
| `quick` | 32 | 1 | 50 | ~2-5 min (CPU) | Testing, quick experiments |
| `standard` | 64 | 2 | 200 | ~15-30 min (CPU) | Real use, decent accuracy |
| `cluster` | 128 | 3 | 1000 | hours (GPU) | Best accuracy, research |

The `cluster` preset also uses learning rate scheduling (ReduceLROnPlateau) which automatically lowers the learning rate when validation loss stops improving.

### Training on a cluster

```bash
# Build the container
podman build -t marketpulse .

# Run training inside the container (with GPU if available)
podman run --rm -v ./data:/app/data:z -v ./models:/app/models:z \
    marketpulse python train.py --all --preset cluster

# Models are saved to the mounted models/ directory
```

For Singularity/Apptainer:
```bash
singularity run --nv --bind ./data:/app/data,./models:/app/models \
    marketpulse.sif python train.py --all --preset cluster
```

The `--nv` flag passes through NVIDIA GPUs. PyTorch automatically detects and uses them.

## Saved models

Models are saved to `models/` with the naming convention:

```
models/{ticker}_{period}_{preset}.pt
```

Examples:
```
models/AAPL_1y_quick.pt
models/BTC-USD_max_standard.pt
models/NVDA_2y_cluster.pt
```

Each `.pt` file contains:
- Model weights (state dict)
- Training config (hidden size, layers, epochs, etc.)
- Training metrics (final accuracy, best validation loss, duration)
- Metadata (features used, window size, timestamp)

List all saved models:
```bash
uv run python train.py --list
```

### Auto-loading

When you request `model_type="lstm"` in a prediction, the API automatically searches for the best available saved model:
1. Tries `cluster` preset first (highest quality)
2. Falls back to `standard`
3. Falls back to `quick`

If no model exists, it returns a clear error with the exact training command needed.

## Window size

LSTM uses `window_size=20` by default (vs 5 for k-NN/LinReg). Longer windows let the LSTM see more history per prediction, which helps it learn longer-term patterns. The tradeoff is more data needed for training.

## Sentiment adjustment

Same post-hoc approach as k-NN and LinReg. The LSTM predicts probability of UP, then sentiment shifts it. `use_time_weights` is ignored for LSTM since time awareness is built into the architecture.

## When to use LSTM vs k-NN/LinReg

**LSTM is better when:**
- You have lots of training data (1000+ rows)
- Patterns are sequential (momentum, mean-reversion over multiple days)
- You can afford to train once and reuse

**k-NN/LinReg are better when:**
- Data is limited (< 200 rows)
- You want instant results without pre-training
- You're experimenting with different tickers/periods frequently

**The backtest will tell you.** After training, run `--compare-periods` and check if LSTM actually beats the simpler models for your specific ticker.

## Limitations

- Requires PyTorch (`uv pip install torch` or `uv pip install -e '.[ai]'`)
- Must be trained before use (no cold-start prediction)
- One trained model per ticker × period × preset (not transferable between tickers)
- CPU training is slow for `standard` and `cluster` presets
- Risk of overfitting on small datasets — monitor validation loss during training
- No early stopping in `quick` preset (to keep it simple)
