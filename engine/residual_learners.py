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


def hybrid_residual_path(ticker: str, models_dir="models", macro: bool = False):
    """Weights path for a ticker's pretrained hybrid residual learner.

    ``{ticker}_hybrid_res.pt`` — distinct from the LSTM-reg (`_reg.pt`) and the
    directional classifiers (`{ticker}_{period}_{preset}.pt`). ``macro=True``
    uses a separate ``_hybrid_res_macro.pt`` file — the macro-exog architecture
    (wider input to the linear head) isn't compatible with the univariate
    weights, so the two must never share a path.
    """
    from pathlib import Path

    suffix = "_hybrid_res_macro.pt" if macro else "_hybrid_res.pt"
    return Path(models_dir) / f"{ticker}{suffix}"


class ZeroResidualLearner:
    """Predicts a zero residual → hybrid ≡ base. The identity/fallback learner."""

    name = "zero"

    def fit(self, residuals: np.ndarray, exog: np.ndarray | None = None) -> None:  # noqa: ARG002
        return None

    def predict(self) -> float:
        return 0.0


if _TORCH_AVAILABLE:

    class _ResNet(nn.Module):
        """Univariate LSTM (+ optional static exog vector) → linear scalar.

        ``exog_dim=0`` (default) reproduces the original univariate
        architecture exactly, so old pretrained weight files still load. With
        ``exog_dim>0``, the exog vector (e.g. lag-1-aligned macro at the last
        in-window date) is concatenated onto the LSTM's final hidden state
        before the linear head — a static feature, not a per-timestep one.
        """

        def __init__(self, hidden_size: int, num_layers: int, dropout: float, exog_dim: int = 0):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.exog_dim = exog_dim
            self.fc = nn.Linear(hidden_size + exog_dim, 1)

        def forward(self, x, exog=None):
            out, _ = self.lstm(x)
            h = out[:, -1, :]
            if self.exog_dim > 0 and exog is not None:
                h = torch.cat([h, exog], dim=-1)
            return self.fc(h).squeeze(-1)


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
        self._exog_dim = 0
        self._exog_mean: np.ndarray | None = None
        self._exog_std: np.ndarray | None = None
        self._last_exog: np.ndarray | None = None  # standardised, for predict()
        self.trained_horizon: int | None = (
            None  # set on save()/load(); None = unknown (old checkpoint)
        )

    def fit(self, residuals: np.ndarray, exog: np.ndarray | None = None) -> None:
        """``exog`` (optional): shape (len(residuals), k), position-aligned with
        ``residuals`` — e.g. lag-1-aligned macro at each residual's date. Row
        ``i+window-1`` (the last in-window date, i.e. known at t) is used to
        predict the residual at ``i+window`` (t+1), matching the macro
        contract used elsewhere (Prophet/XGBoost `macro_df`).
        """
        self._net = None  # reset; a failed/short fit ⇒ predict() returns 0
        if not _TORCH_AVAILABLE:
            return
        res = np.asarray(residuals, dtype=np.float32).ravel()
        mask = np.isfinite(res)
        ex = None
        if exog is not None:
            ex = np.asarray(exog, dtype=np.float32)
            if ex.ndim == 1:
                ex = ex.reshape(-1, 1)
            if ex.shape[0] != res.shape[0]:
                ex = None  # misaligned caller input → ignore exog, stay univariate
            else:
                mask &= np.isfinite(ex).all(axis=1)
        res = res[mask]
        if ex is not None:
            ex = ex[mask]
        if res.size < self.window + self.min_pairs:
            return

        # Windowed supervised set: past `window` residuals → next residual.
        x_list, y_list, exog_list = [], [], []
        for i in range(res.size - self.window):
            x_list.append(res[i : i + self.window])
            y_list.append(res[i + self.window])
            if ex is not None:
                exog_list.append(ex[i + self.window - 1])
        x = np.asarray(x_list, dtype=np.float32)
        y = np.asarray(y_list, dtype=np.float32)

        self._mean = float(x.mean())
        self._std = float(x.std()) or 1.0
        xn = (x - self._mean) / self._std
        yn = (y - self._mean) / self._std

        exn = None
        if ex is not None:
            exog_arr = np.asarray(exog_list, dtype=np.float32)
            self._exog_dim = exog_arr.shape[1]
            self._exog_mean = exog_arr.mean(axis=0)
            self._exog_std = np.where(exog_arr.std(axis=0) == 0, 1.0, exog_arr.std(axis=0))
            exn = (exog_arr - self._exog_mean) / self._exog_std
        else:
            self._exog_dim = 0
            self._exog_mean = None
            self._exog_std = None

        try:
            torch.manual_seed(self.seed)
            n = len(xn)
            n_val = max(1, int(n * 0.2))
            n_tr = n - n_val
            xt = torch.tensor(xn).unsqueeze(-1)  # (n, window, 1)
            yt = torch.tensor(yn)
            ext = torch.tensor(exn) if exn is not None else None
            x_tr, y_tr = xt[:n_tr], yt[:n_tr]
            x_val, y_val = xt[n_tr:], yt[n_tr:]
            ex_tr = ext[:n_tr] if ext is not None else None
            ex_val = ext[n_tr:] if ext is not None else None

            net = _ResNet(self.hidden_size, self.num_layers, self.dropout, exog_dim=self._exog_dim)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            loss_fn = nn.MSELoss()
            best_val, best_state, bad = float("inf"), None, 0
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(n_tr)
                for j in range(0, n_tr, self.batch_size):
                    idx = perm[j : j + self.batch_size]
                    opt.zero_grad()
                    exog_batch = ex_tr[idx] if ex_tr is not None else None
                    loss = loss_fn(net(x_tr[idx], exog_batch), y_tr[idx])
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vl = float(loss_fn(net(x_val, ex_val), y_val).item())
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
            self._last_exog = exn[-1] if exn is not None and len(exn) else None
        except Exception as e:  # noqa: BLE001 — fall back to base (predict 0)
            log.debug("residual LSTM fit failed (%s); hybrid falls back to base.", e)
            self._net = None

    def set_window(self, residuals: np.ndarray, exog: np.ndarray | None = None) -> None:
        """Set the prediction input from a residual series *without refitting*.

        Used in the frozen/pretrained mode: the learner keeps its trained weights
        and scaler, but predicts from the most-recent ``window`` residuals of a
        fresh series. No-op if not enough residuals or the learner isn't trained.
        ``exog`` (optional): the single most-recent exog row (e.g. macro at the
        latest date) — ignored if the learner wasn't trained with exog.
        """
        if self._net is None:
            return
        res = np.asarray(residuals, dtype=np.float32).ravel()
        res = res[np.isfinite(res)]
        if res.size < self.window:
            self._last_window = None
            return
        self._last_window = ((res[-self.window :] - self._mean) / self._std).astype(np.float32)
        if self._exog_dim > 0 and exog is not None and self._exog_mean is not None:
            ex = np.asarray(exog, dtype=np.float32).ravel()
            if ex.size == self._exog_dim and np.isfinite(ex).all():
                self._last_exog = ((ex - self._exog_mean) / self._exog_std).astype(np.float32)
            else:
                self._last_exog = None
        else:
            self._last_exog = None

    def predict(self) -> float:
        if self._net is None or self._last_window is None or not _TORCH_AVAILABLE:
            return 0.0
        if self._exog_dim > 0 and self._last_exog is None:
            return 0.0  # trained with exog but none available now → don't guess
        try:
            with torch.no_grad():
                t = torch.tensor(self._last_window, dtype=torch.float32).reshape(1, -1, 1)
                exog_t = (
                    torch.tensor(self._last_exog, dtype=torch.float32).reshape(1, -1)
                    if self._last_exog is not None
                    else None
                )
                scaled = float(self._net(t, exog_t).item())
            return scaled * self._std + self._mean
        except Exception:  # noqa: BLE001
            return 0.0

    @property
    def is_trained(self) -> bool:
        return self._net is not None

    def save(self, path, horizon: int | None = None) -> None:
        """Persist trained weights + scaler + hyperparams to ``path``.

        ``horizon`` records which ``--horizon`` this checkpoint was trained
        for, so ``load()``/the caller can catch a stale pretrained-weight
        mismatch instead of silently scoring the wrong horizon.
        """
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
                "exog_dim": self._exog_dim,
                "exog_mean": self._exog_mean,
                "exog_std": self._exog_std,
                "horizon": horizon,
            },
            p,
        )

    def load(self, path) -> bool:
        """Load weights + scaler. Returns False if the file is missing; safe
        (leaves the learner untrained) on any error.

        Sets ``self.trained_horizon`` from the checkpoint (``None`` for an
        older checkpoint saved before this field existed — callers should
        treat that as "unknown, don't assume it matches").
        """
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
            self._exog_dim = ck.get("exog_dim", 0)
            self._exog_mean = ck.get("exog_mean")
            self._exog_std = ck.get("exog_std")
            self.trained_horizon = ck.get("horizon")
            net = _ResNet(self.hidden_size, self.num_layers, self.dropout, exog_dim=self._exog_dim)
            net.load_state_dict(ck["model_state"])
            net.eval()
            self._net = net
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load residual learner from %s: %s", p, e)
            self._net = None
            return False
