"""
lstm_regressor.py – LSTM *regression* point-forecaster (optional torch).

A separate model from ``ai_model.py`` (which is a UP/DOWN **classifier** — its
sigmoid head carries no price-magnitude information, so it can't be scored by
the regression harness). This network has a **linear** head and predicts a
continuous target: the one-step change ``Δ = close[t+h] − close[t]``, which is
added back to the last close to produce a price level (same Δ-target trick as
``xgboost_model.py``, so trees/nets aren't capped by the training price range on
a trend). The reported value is always a level — the harness/metrics never see
the Δ.

It is also the natural **residual learner** for the Phase-R3 hybrid: a residual
series is just another continuous target this regressor can fit.

Training is leakage-safe by construction *if* the weights were trained only on
data before the evaluation window (see ``scripts/train_lstm_regressor.py``,
which trims the eval window before fitting). The harness loads the saved
``models/{ticker}_reg.pt`` per ticker; if no weights exist for a ticker (or
torch isn't installed), the forecaster skips it gracefully — exactly like the
other optional forecasters.

Optional dependency: ``torch`` (the ``[ai]`` extra).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from engine.features import (
    DEFAULT_FEATURES,
    compute_feature_columns,
    min_rows_needed,
    validate_features,
)
from engine.forecast_base import ForecastModel, ForecastResult
from engine.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

log = get_logger(__name__)

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

_DEFAULT_WINDOW = 20
_MIN_TRAIN_PAIRS = 60
_MODELS_DIR = Path("models")

# Training-effort tiers, mirroring ai_model.TRAINING_PRESETS so the regressor
# has the same quick/standard/cluster knob as the directional LSTM. Only the
# training cost/capacity differs — the model + Δ-target are identical.
REG_TRAINING_PRESETS: dict[str, dict] = {
    "quick": {
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "epochs": 50,
        "batch_size": 32,
        "lr": 0.001,
        "patience": 8,
        "description": "~1-5 min on CPU. Small model, good for iterating.",
    },
    "standard": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 200,
        "batch_size": 64,
        "lr": 0.001,
        "patience": 20,
        "description": "~5-15 min on CPU. Medium model, good for real runs.",
    },
    "cluster": {
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "epochs": 1000,
        "batch_size": 128,
        "lr": 0.0005,
        "patience": 50,
        "description": "Hours on GPU. Large model, best accuracy.",
    },
}


def regressor_path(ticker: str, models_dir: Path | str = _MODELS_DIR) -> Path:
    """Weights path for a ticker's regressor — distinct from the classifiers.

    Classifiers are ``{ticker}_{period}_{preset}.pt``; the regressor uses the
    ``_reg`` suffix so the two never collide.
    """
    return Path(models_dir) / f"{ticker}_reg.pt"


if _TORCH_AVAILABLE:

    class _LSTMRegNetwork(nn.Module):
        """LSTM → FC → linear scalar (no sigmoid — a regression head)."""

        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden).squeeze(-1)


# ----------------------------------------------------------------------
# Sequence building (shared by training and inference)
# ----------------------------------------------------------------------


def _step_columns(features: list[str], df_feat: pd.DataFrame) -> list[str]:
    cols = ["_ret"]
    if "volume" in features and "_vol_chg" in df_feat.columns:
        cols.append("_vol_chg")
    if "rsi" in features and "_rsi" in df_feat.columns:
        cols.append("_rsi")
    if "volatility" in features and "_volat" in df_feat.columns:
        cols.append("_volat")
    if "macd" in features and "_macd" in df_feat.columns:
        cols.append("_macd")
    return cols


def build_sequences(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """(X, y, step_cols): X = (n, window, n_feat) sequences, y = Δ close[t+h]−close[t]."""
    if len(df) < min_rows_needed(features, window_size) + horizon:
        return None
    df_feat = compute_feature_columns(df, features, window_size)
    step_cols = _step_columns(features, df_feat)
    feat = df_feat[step_cols].to_numpy(dtype=np.float32)
    closes = df_feat["close"].to_numpy(dtype=np.float32)
    n = len(df_feat)

    x_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for idx in range(0, n - window_size - horizon + 1):
        t = idx + window_size - 1
        seq = feat[idx : idx + window_size]
        if not np.isfinite(seq).all():
            continue
        target = closes[t + horizon] - closes[t]
        if not np.isfinite(target):
            continue
        x_rows.append(seq)
        y_rows.append(float(target))
    if not x_rows:
        return None
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32), step_cols


# ----------------------------------------------------------------------
# The forecaster
# ----------------------------------------------------------------------


class LSTMRegressorForecaster(ForecastModel):
    """Per-ticker LSTM regressor. Loads saved weights lazily on first forecast.

    Construct with the ticker so it knows which ``models/{ticker}_reg.pt`` to
    load. If torch is missing or the weights don't exist, ``_raw_forecast``
    returns None and the harness skips this model for that ticker.
    """

    name = "LSTM-reg"

    def __init__(self, ticker: str, models_dir: Path | str = _MODELS_DIR) -> None:
        self.ticker = ticker
        self.models_dir = Path(models_dir)
        self._loaded = False
        self._ok = False
        self._network = None
        self._features: list[str] = list(DEFAULT_FEATURES)
        self._window = _DEFAULT_WINDOW
        self._horizon = 1
        self._y_mean = 0.0
        self._y_std = 1.0
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._ok
        self._loaded = True
        if not _TORCH_AVAILABLE:
            return False
        path = regressor_path(self.ticker, self.models_dir)
        if not path.exists():
            log.debug("No regressor weights for %s at %s.", self.ticker, path)
            return False
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            self._features = ckpt["features"]
            self._window = ckpt["window_size"]
            self._horizon = ckpt.get("horizon", 1)
            self._y_mean = ckpt["y_mean"]
            self._y_std = ckpt["y_std"] or 1.0
            self._x_mean = np.asarray(ckpt["x_mean"], dtype=np.float32)
            self._x_std = np.asarray(ckpt["x_std"], dtype=np.float32)
            cfg = ckpt["config"]
            self._network = _LSTMRegNetwork(
                input_size=ckpt["input_size"],
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
            )
            self._network.load_state_dict(ckpt["model_state"])
            self._network.eval()
            self._ok = True
        except Exception as e:  # noqa: BLE001 — a bad checkpoint must not crash a run
            log.warning("Failed to load regressor for %s: %s", self.ticker, e)
            self._ok = False
        return self._ok

    def _latest_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """The most-recent window's per-step features (the prediction input)."""
        df_feat = compute_feature_columns(df, self._features, self._window)
        step_cols = _step_columns(self._features, df_feat)
        feat = df_feat[step_cols].to_numpy(dtype=np.float32)
        if len(feat) < self._window:
            return None
        seq = feat[-self._window :]
        if not np.isfinite(seq).all():
            return None
        return seq

    def _raw_forecast(self, df: pd.DataFrame, horizon: int = 1) -> ForecastResult | None:
        if not self._ensure_loaded():
            return None
        if "close" not in df.columns:
            return None
        seq = self._latest_sequence(df)
        if seq is None or self._x_mean is None or self._x_std is None:
            return None

        # Standardise with the *training* stats, then predict the scaled Δ.
        x = (seq - self._x_mean) / self._x_std
        try:
            with torch.no_grad():  # type: ignore[union-attr]
                tens = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                scaled_delta = float(self._network(tens).item())  # type: ignore[union-attr]
        except Exception as e:  # noqa: BLE001
            log.debug("LSTM-reg forward failed for %s: %s", self.ticker, e)
            return None

        delta = scaled_delta * self._y_std + self._y_mean
        last_close = float(np.asarray(df["close"], dtype=float).ravel()[-1])
        point = last_close + delta
        if not np.isfinite(point):
            return None
        return ForecastResult(last_close=last_close, point=point, horizon=horizon)


# ----------------------------------------------------------------------
# Training helper (used by scripts/train_lstm_regressor.py)
# ----------------------------------------------------------------------


def train_regressor(
    df: pd.DataFrame,
    *,
    preset: str = "standard",
    features: list[str] | None = None,
    window_size: int = _DEFAULT_WINDOW,
    horizon: int = 1,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    val_frac: float = 0.2,
    patience: int | None = None,
    seed: int = 42,
) -> dict | None:
    """Train an LSTM regressor on ``df`` and return a saveable checkpoint dict.

    ``preset`` (quick/standard/cluster) sets the training-effort tier — capacity
    (hidden_size, num_layers, dropout) and budget (epochs, lr, batch_size,
    patience). Any of those passed explicitly overrides the preset. The caller
    is responsible for passing a ``df`` that **excludes the evaluation window**
    — this function fits on whatever it's given. Returns None if torch is
    missing or there isn't enough data.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed (needed to train the LSTM regressor).")
    if preset not in REG_TRAINING_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(REG_TRAINING_PRESETS)}")
    cfg = REG_TRAINING_PRESETS[preset]
    # Preset supplies defaults; explicit kwargs (non-None) override.
    hidden_size = cfg["hidden_size"] if hidden_size is None else hidden_size
    num_layers = cfg["num_layers"] if num_layers is None else num_layers
    dropout = cfg["dropout"] if dropout is None else dropout
    epochs = cfg["epochs"] if epochs is None else epochs
    lr = cfg["lr"] if lr is None else lr
    batch_size = cfg["batch_size"] if batch_size is None else batch_size
    patience = cfg["patience"] if patience is None else patience

    feats = list(features) if features else list(DEFAULT_FEATURES)
    validate_features(feats)

    built = build_sequences(df, feats, window_size, horizon)
    if built is None or len(built[0]) < _MIN_TRAIN_PAIRS:
        return None
    x, y, step_cols = built

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardise features and target on the training set only.
    x_mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    x_std = x.reshape(-1, x.shape[-1]).std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean = float(y.mean())
    y_std = float(y.std()) or 1.0
    xn = (x - x_mean) / x_std
    yn = (y - y_mean) / y_std

    # Chronological train/val split (no shuffling — it's a time series).
    n = len(xn)
    n_val = max(1, int(n * val_frac))
    n_tr = n - n_val
    x_tr = torch.tensor(xn[:n_tr])
    y_tr = torch.tensor(yn[:n_tr])
    x_val = torch.tensor(xn[n_tr:])
    y_val = torch.tensor(yn[n_tr:])

    net = _LSTMRegNetwork(
        input_size=x.shape[-1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    for _epoch in range(epochs):
        net.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            out = net(x_tr[idx])
            loss = loss_fn(out, y_tr[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(net(x_val), y_val).item()) if n_val > 0 else 0.0
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    return {
        "model_state": net.state_dict(),
        "config": {"hidden_size": hidden_size, "num_layers": num_layers, "dropout": dropout},
        "preset": preset,
        "features": feats,
        "step_cols": step_cols,
        "window_size": window_size,
        "horizon": horizon,
        "input_size": int(x.shape[-1]),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean,
        "y_std": y_std,
        "val_loss": best_val,
        "n_train": int(n_tr),
    }
