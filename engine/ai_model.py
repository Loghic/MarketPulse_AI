"""
ai_model.py – LSTM neural network for next-day price direction prediction.

Unlike k-NN/LinReg which train from scratch on every prediction, the LSTM
is trained once (via train.py) and the weights are saved to disk. Prediction
loads the saved model — no retraining needed.

Training presets:
    "quick"    — ~2-5 min on CPU. Small model, few epochs. Good for testing.
    "standard" — ~15-30 min on CPU. Medium model. Good for real use.
    "cluster"  — hours on GPU. Large model, many epochs. Best accuracy.

Saved models go to: models/{ticker}_{period}_{preset}.pt
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from engine.features import (
    ALL_FEATURES,
    compute_feature_columns,
    feature_label,
    min_rows_needed,
    validate_features,
)
from engine.logger import epoch_progress, get_logger

log = get_logger(__name__)

# Try to import torch — graceful fallback if not installed
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ------------------------------------------------------------------
# Training presets
# ------------------------------------------------------------------

TRAINING_PRESETS = {
    "quick": {
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "early_stopping_patience": 10,
        "description": "~1-5 min on CPU. Small model, good for testing.",
    },
    "standard": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "epochs": 200,
        "batch_size": 64,
        "learning_rate": 0.001,
        "early_stopping_patience": 20,
        "description": "~5-15 min on CPU. Medium model, good for real use.",
    },
    "cluster": {
        "hidden_size": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "epochs": 1000,
        "batch_size": 128,
        "learning_rate": 0.0005,
        "early_stopping_patience": 50,
        "description": "Hours on GPU. Large model, best accuracy.",
    },
}

# Default directory for saved models
MODELS_DIR = Path("models")

# Sentiment weight (same as k-NN and LinReg)
SENTIMENT_WEIGHT = 0.20


# ------------------------------------------------------------------
# LSTM network definition
# ------------------------------------------------------------------

if TORCH_AVAILABLE:

    class _LSTMNetwork(nn.Module):
        """Simple LSTM → FC → Sigmoid for binary classification."""

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
                nn.Sigmoid(),
            )

        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            # Take only the last timestep's output
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden).squeeze(-1)


# ------------------------------------------------------------------
# Public model class
# ------------------------------------------------------------------


class AIModel:
    """
    LSTM-based model for next-day price direction prediction.

    Shares the same .predict() interface as KNNModel and LinearRegressionModel,
    but requires pre-training via .train() or loading a saved model.

    Usage:
        model = AIModel(features=ALL_FEATURES, preset="quick")
        model.train(df)                          # train on price data
        model.save("models/AAPL_1y_quick.pt")    # save weights
        model.load("models/AAPL_1y_quick.pt")    # load later
        direction, confidence = model.predict(df) # predict
    """

    def __init__(
        self,
        window_size: int = 20,
        features: list[str] | None = None,
        preset: str = "quick",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Install with: uv pip install torch\n"
                "Or: uv pip install -e '.[ai]'"
            )

        self.window_size = window_size
        self.features = features or ALL_FEATURES
        validate_features(self.features)
        self.preset = preset

        if preset not in TRAINING_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Available: {list(TRAINING_PRESETS.keys())}"
            )

        self.config = TRAINING_PRESETS[preset]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None
        self.trained = False
        self.training_info = {}

        # Number of features per timestep
        self._features_per_step = self._count_features_per_step()

    def _count_features_per_step(self) -> int:
        """Count how many feature values each timestep produces."""
        count = 0
        if "returns" in self.features:
            count += 1
        if "volume" in self.features:
            count += 1
        if "rsi" in self.features:
            count += 1
        if "volatility" in self.features:
            count += 1
        if "macd" in self.features:
            count += 1
        return max(count, 1)

    @property
    def feature_label(self) -> str:
        return feature_label(self.features)

    # ------------------------------------------------------------------
    # Data preparation (sequential format for LSTM)
    # ------------------------------------------------------------------

    def _prepare_sequences(self, df) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Build sequential data for LSTM training.

        Unlike k-NN/LinReg which use flat feature vectors, LSTM needs
        (batch, sequence_length, features_per_timestep) format.

        Returns:
            (X, y) where X is (n_samples, window_size, n_features) and
            y is (n_samples,) binary target.
        """
        min_n = min_rows_needed(self.features, self.window_size)
        if len(df) < min_n:
            return None, None

        df_feat = compute_feature_columns(df, self.features, self.window_size)

        # Build per-timestep feature columns
        step_cols = ["_ret"]
        if "volume" in self.features and "_vol_chg" in df_feat.columns:
            step_cols.append("_vol_chg")
        if "rsi" in self.features and "_rsi" in df_feat.columns:
            step_cols.append("_rsi")
        if "volatility" in self.features and "_volat" in df_feat.columns:
            step_cols.append("_volat")
        if "macd" in self.features and "_macd" in df_feat.columns:
            step_cols.append("_macd")

        # Target: 1 if next day's close > today's close
        df_feat["_target"] = np.where(df_feat["close"].shift(-1) > df_feat["close"], 1, 0)

        # Drop NaN rows
        df_feat = df_feat.dropna(subset=step_cols + ["_target"])

        X, y = [], []
        values = df_feat[step_cols].values
        targets = df_feat["_target"].values

        for i in range(len(df_feat) - self.window_size):
            window = values[i : i + self.window_size]
            if np.isnan(window).any():
                continue
            X.append(window)
            y.append(targets[i + self.window_size - 1])

        if len(X) == 0:
            return None, None

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df, verbose: bool = True) -> dict:
        """
        Train the LSTM on price data.

        Args:
            df:      DataFrame with 'close' (and optionally 'volume') column.
            verbose: Print training progress.

        Returns:
            dict with training metrics (loss history, final accuracy, duration).
        """
        X, y = self._prepare_sequences(df)
        if X is None:
            raise RuntimeError("Not enough data for training")

        # Train/validation split (80/20, chronological)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if verbose:
            log.info(
                f"Training LSTM ({self.preset}): {len(X_train)} train, "
                f"{len(X_val)} val, shape=({self.window_size}, {X.shape[2]}), "
                f"device={self.device}"
            )

        # Create network
        self.network = _LSTMNetwork(
            input_size=X.shape[2],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

        # DataLoaders
        train_ds = TensorDataset(
            torch.tensor(X_train).to(self.device),
            torch.tensor(y_train).to(self.device),
        )
        train_loader = DataLoader(train_ds, batch_size=self.config["batch_size"], shuffle=True)

        X_val_t = torch.tensor(X_val).to(self.device)
        y_val_t = torch.tensor(y_val).to(self.device)

        # Training loop
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["learning_rate"])
        criterion = nn.BCELoss()

        # Learning rate scheduler for cluster preset
        scheduler = None
        if self.preset == "cluster":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=50, factor=0.5
            )

        loss_history = []
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        patience = self.config["early_stopping_patience"]
        stopped_early = False
        start_time = datetime.now()

        pbar = (
            epoch_progress(self.config["epochs"], desc=f"LSTM {self.preset}") if verbose else None
        )

        for epoch in range(self.config["epochs"]):
            # Train
            self.network.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = self.network(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)

            # Validate
            self.network.eval()
            with torch.no_grad():
                val_output = self.network(X_val_t)
                val_loss = criterion(val_output, y_val_t).item()
                val_preds = (val_output > 0.5).float()
                val_acc = (val_preds == y_val_t).float().mean().item()

            loss_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }
            )

            # Save best model + early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.network.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if scheduler:
                scheduler.step(val_loss)

            # Progress bar update
            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"loss={avg_train_loss:.4f} val={val_loss:.4f} "
                    f"acc={val_acc:.0%} wait={epochs_without_improvement}/{patience}"
                )

            # Early stopping
            if epochs_without_improvement >= patience:
                stopped_early = True
                if pbar:
                    pbar.close()
                log.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} epochs, "
                    f"best val_loss={best_val_loss:.4f})"
                )
                break
        else:
            # Loop completed without break
            if pbar:
                pbar.close()

        # Restore best model
        if best_state:
            self.network.load_state_dict(best_state)
            self.network.to(self.device)

        duration = (datetime.now() - start_time).total_seconds()
        self.trained = True

        # Final validation accuracy
        self.network.eval()
        with torch.no_grad():
            final_output = self.network(X_val_t)
            final_preds = (final_output > 0.5).float()
            final_acc = (final_preds == y_val_t).float().mean().item()

        self.training_info = {
            "preset": self.preset,
            "features": self.features,
            "window_size": self.window_size,
            "input_size": int(X.shape[2]),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "final_val_accuracy": final_acc,
            "best_val_loss": best_val_loss,
            "epochs_completed": len(loss_history),
            "epochs_max": self.config["epochs"],
            "stopped_early": stopped_early,
            "duration_seconds": round(duration, 1),
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
        }

        if verbose:
            stop_info = (
                f"early stop at epoch {len(loss_history)}"
                if stopped_early
                else f"all {self.config['epochs']} epochs"
            )
            log.info(
                f"Training complete: {duration:.1f}s ({stop_info}), "
                f"val_accuracy={final_acc:.1%}, "
                f"best_val_loss={best_val_loss:.4f}"
            )

        return self.training_info

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    @staticmethod
    def model_path(ticker: str, period: str, preset: str, models_dir: Path = MODELS_DIR) -> Path:
        """Generate standard model file path."""
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir / f"{ticker}_{period}_{preset}.pt"

    def save(self, path: str | Path):
        """Save trained model weights + metadata to disk."""
        if not self.trained or self.network is None:
            raise RuntimeError("Cannot save: model is not trained")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state": self.network.state_dict(),
                "config": self.config,
                "training_info": self.training_info,
                "features": self.features,
                "window_size": self.window_size,
                "preset": self.preset,
            },
            path,
        )

        log.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> bool:
        """
        Load a trained model from disk.

        Returns True if loaded successfully, False if file doesn't exist.
        Raises RuntimeError if file exists but is incompatible.
        """
        path = Path(path)
        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore config
        self.features = checkpoint["features"]
        self.window_size = checkpoint["window_size"]
        self.preset = checkpoint["preset"]
        self.config = checkpoint["config"]
        self.training_info = checkpoint.get("training_info", {})
        self._features_per_step = self._count_features_per_step()

        # Rebuild and load network
        input_size = self.training_info.get("input_size", self._features_per_step)
        self.network = _LSTMNetwork(
            input_size=input_size,
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.network.load_state_dict(checkpoint["model_state"])
        self.network.eval()
        self.trained = True

        return True

    # ------------------------------------------------------------------
    # Sentiment adjustment (same as k-NN and LinReg)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_sentiment(
        prediction: int,
        prob: float,
        sentiment_score: float,
        weight: float = SENTIMENT_WEIGHT,
    ) -> tuple[int, float]:
        """Shift probability using sentiment. Can flip the prediction."""
        if sentiment_score == 0.0:
            return prediction, prob

        prob_up = prob if prediction == 1 else (1.0 - prob)
        prob_up_adjusted = np.clip(prob_up + sentiment_score * weight, 0.01, 0.99)

        if prob_up_adjusted >= 0.5:
            return 1, float(prob_up_adjusted)
        else:
            return 0, float(1.0 - prob_up_adjusted)

    # ------------------------------------------------------------------
    # Prediction (same interface as k-NN and LinReg)
    # ------------------------------------------------------------------

    def predict(
        self,
        df,
        use_time_weights: bool = False,
        sentiment_score: float = 0.0,
    ) -> tuple[str, float]:
        """
        Predict next-day direction using the trained LSTM.

        Args:
            df:               DataFrame with 'close' column (and 'volume' if used).
            use_time_weights: Ignored for LSTM (time awareness is built into the architecture).
            sentiment_score:  News sentiment in [-1, 1], applied post-hoc.

        Returns:
            (direction, confidence) — e.g. ("UP", 0.73)
        """
        if not self.trained or self.network is None:
            return "Model not trained", 0.0

        # Build the last sequence
        min_n = min_rows_needed(self.features, self.window_size)
        if len(df) < min_n:
            return "Insufficient data", 0.0

        df_feat = compute_feature_columns(df, self.features, self.window_size)

        step_cols = ["_ret"]
        if "volume" in self.features and "_vol_chg" in df_feat.columns:
            step_cols.append("_vol_chg")
        if "rsi" in self.features and "_rsi" in df_feat.columns:
            step_cols.append("_rsi")
        if "volatility" in self.features and "_volat" in df_feat.columns:
            step_cols.append("_volat")
        if "macd" in self.features and "_macd" in df_feat.columns:
            step_cols.append("_macd")

        df_feat = df_feat.dropna(subset=step_cols)

        if len(df_feat) < self.window_size:
            return "Insufficient data", 0.0

        # Take the last window_size rows as the input sequence
        last_seq = df_feat[step_cols].iloc[-self.window_size :].values
        if np.isnan(last_seq).any():
            return "Data error", 0.0

        # Predict
        self.network.eval()
        with torch.no_grad():
            x = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            prob_up = self.network(x).item()

        # Convert to prediction
        if prob_up >= 0.5:
            raw_pred, raw_prob = 1, prob_up
        else:
            raw_pred, raw_prob = 0, 1.0 - prob_up

        # Sentiment adjustment
        final_pred, final_prob = self._apply_sentiment(raw_pred, raw_prob, sentiment_score)

        return ("UP" if final_pred == 1 else "DOWN"), final_prob
