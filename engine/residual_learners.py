"""
residual_learners.py – Residual learners for the Phase-R3 hybrid.

A *residual learner* takes the base model's in-sample residual series
(``res_t = close_t − fitted_t``) and predicts the **next** residual
``r̂es_{t+1}``. The hybrid adds that to the base's out-of-sample point forecast:
``P̂ = P̂^base + r̂es``. The learner therefore only ever sees residuals up to
``t`` (the R0.2 leakage rule) — the hybrid is responsible for never handing it
``res_{t+1}``.

The contract is intentionally tiny so any model can play the role:

    learner.fit(residuals: 1-D array)      # residuals up to and including t
    learner.predict() -> float             # r̂es_{t+1}

Two learners ship here:

* ``ZeroResidualLearner`` — always predicts 0. Makes the hybrid *exactly* the
  base model; used as the identity check in tests and as a safe fallback.
* ``LSTMResidualLearner`` — a small LSTM fit **per call** on the residual series
  (univariate: a window of past residuals → next residual). Reuses the network
  + early-stopping plumbing from ``lstm_regressor``. If torch is missing or the
  series is too short, ``fit`` is a no-op and ``predict`` returns 0.0, so the
  hybrid gracefully degenerates to the base model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.logger import get_logger

log = get_logger(__name__)

try:
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def hybrid_residual_path(ticker: str, models_dir="models"):
    """Weights path for a ticker's pretrained hybrid residual learner.

    ``{ticker}_hybrid_res.pt`` — distinct from the LSTM-reg (`_reg.pt`) and the
    directional classifiers (`{ticker}_{period}_{preset}.pt`).
    """
    from pathlib import Path

    return Path(models_dir) / f"{ticker}_hybrid_res.pt"


class ZeroResidualLearner:
    """Predicts a zero residual → hybrid ≡ base. The identity/fallback learner."""

    name = "zero"

    def fit(self, residuals: np.ndarray) -> None:  # noqa: ARG002
        return None

    def predict(self) -> float:
        return 0.0


if _TORCH_AVAILABLE:

    class _ResNet(nn.Module):
        """Univariate LSTM → linear scalar, for next-residual prediction."""

        def __init__(self, hidden_size: int, num_layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMResidualLearner:
    """Small LSTM fit per call on a 1-D residual series.

    Builds windows ``(res[i:i+W] → res[i+W])`` from the residual history,
    standardises, trains with early stopping, and predicts the next residual
    from the most-recent window. Degenerates to predicting 0.0 (i.e. the hybrid
    falls back to the base model) when torch is absent or there isn't enough
    residual history to train.
    """

    name = "lstm"

    def __init__(
        self,
        *,
        window: int = 20,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 8,
        min_pairs: int = 40,
        seed: int = 42,
    ) -> None:
        self.window = window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.min_pairs = min_pairs
        self.seed = seed

        self._net: Any = None  # _ResNet when trained (torch only)
        self._mean = 0.0
        self._std = 1.0
        self._last_window: np.ndarray | None = None  # standardised, for predict()

    def fit(self, residuals: np.ndarray) -> None:
        self._net = None  # reset; a failed/short fit ⇒ predict() returns 0
        if not _TORCH_AVAILABLE:
            return
        res = np.asarray(residuals, dtype=np.float32).ravel()
        res = res[np.isfinite(res)]
        if res.size < self.window + self.min_pairs:
            return

        # Windowed supervised set: past `window` residuals → next residual.
        x_list, y_list = [], []
        for i in range(res.size - self.window):
            x_list.append(res[i : i + self.window])
            y_list.append(res[i + self.window])
        x = np.asarray(x_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)

        self._mean = float(x.mean())
        self._std = float(x.std()) or 1.0
        xn = (x - self._mean) / self._std
        yn = (y - self._mean) / self._std

        try:
            torch.manual_seed(self.seed)
            n = len(xn)
            n_val = max(1, int(n * 0.2))
            n_tr = n - n_val
            xt = torch.tensor(xn).unsqueeze(-1)  # (n, window, 1)
            yt = torch.tensor(yn)
            x_tr, y_tr = xt[:n_tr], yt[:n_tr]
            x_val, y_val = xt[n_tr:], yt[n_tr:]

            net = _ResNet(self.hidden_size, self.num_layers, self.dropout)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            best_val, best_state, bad = float("inf"), None, 0
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(n_tr)
                for j in range(0, n_tr, self.batch_size):
                    idx = perm[j : j + self.batch_size]
                    opt.zero_grad()
                    loss = loss_fn(net(x_tr[idx]), y_tr[idx])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(net(x_val), y_val).item())
                if vl < best_val - 1e-6:
                    best_val, bad = vl, 0
                    best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self._net = net
            self._last_window = ((res[-self.window :] - self._mean) / self._std).astype(np.float32)
        except Exception as e:  # noqa: BLE001 — fall back to base (predict 0)
            log.debug("residual LSTM fit failed (%s); hybrid falls back to base.", e)
            self._net = None

    def set_window(self, residuals: np.ndarray) -> None:
        """Set the prediction input from a residual series *without refitting*.

        Used in the frozen/pretrained mode: the learner keeps its trained weights
        and scaler, but predicts from the most-recent ``window`` residuals of a
        fresh series. No-op if not enough residuals or the learner isn't trained.
        """
        if self._net is None:
            return
        res = np.asarray(residuals, dtype=np.float32).ravel()
        res = res[np.isfinite(res)]
        if res.size < self.window:
            self._last_window = None
            return
        self._last_window = ((res[-self.window :] - self._mean) / self._std).astype(np.float32)

    def predict(self) -> float:
        if self._net is None or self._last_window is None or not _TORCH_AVAILABLE:
            return 0.0
        try:
            with torch.no_grad():
                t = torch.tensor(self._last_window, dtype=torch.float32).reshape(1, -1, 1)
                scaled = float(self._net(t).item())
            return scaled * self._std + self._mean
        except Exception:  # noqa: BLE001
            return 0.0

    @property
    def is_trained(self) -> bool:
        return self._net is not None

    def save(self, path) -> None:
        """Persist trained weights + scaler + hyperparams to ``path``."""
        if self._net is None:
            raise RuntimeError("Cannot save: residual learner is not trained.")
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self._net.state_dict(),
                "window": self.window,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "mean": self._mean,
                "std": self._std,
            },
            p,
        )

    def load(self, path) -> bool:
        """Load weights + scaler. Returns False if the file is missing; safe
        (leaves the learner untrained) on any error."""
        from pathlib import Path

        p = Path(path)
        if not p.exists() or not _TORCH_AVAILABLE:
            return False
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            self.window = ck["window"]
            self.hidden_size = ck["hidden_size"]
            self.num_layers = ck["num_layers"]
            self.dropout = ck["dropout"]
            self._mean = ck["mean"]
            self._std = ck["std"] or 1.0
            net = _ResNet(self.hidden_size, self.num_layers, self.dropout)
            net.load_state_dict(ck["model_state"])
            net.eval()
            self._net = net
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load residual learner from %s: %s", p, e)
            self._net = None
            return False
